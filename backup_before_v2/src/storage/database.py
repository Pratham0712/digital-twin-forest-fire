"""
database.py - persistent storage layer (SQLite).

Why SQLite and not Postgres/MySQL: this is a single-instance academic
deployment (one Streamlit process, one dashboard), so a server-based DB adds
operational overhead (a service to run, credentials to manage, a host to
deploy) with no real benefit here. SQLite gives real, queryable, persistent
tables - snapshots, alerts, users - backed by an actual file on disk, which
is what "we need an actual database, not just CSVs" (cross-question 11)
is asking for. Swapping this for Postgres later is a small change: every
function below only uses ANSI-ish SQL, no SQLite-only syntax.

Three tables:
  - snapshots   : one row per DigitalTwin.refresh() call (a time-series log
                  of the twin's state, per region)
  - alerts_log  : one row per zone alert generated in a snapshot, so every
                  alert the system ever raised is queryable later (audit
                  trail for "did we detect this fire, and when")
  - users       : login credentials + role, for the auth module
"""
import hashlib
import logging
import secrets
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "digital_twin.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    region_name     TEXT NOT NULL,
    timestamp_utc   TEXT NOT NULL,
    offline_mode    INTEGER NOT NULL,
    total_zones     INTEGER NOT NULL,
    total_alerts    INTEGER NOT NULL,
    extreme_count   INTEGER NOT NULL DEFAULT 0,
    high_count      INTEGER NOT NULL DEFAULT 0,
    max_risk_score  REAL NOT NULL,
    mean_risk_score REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_snapshots_region_time
    ON snapshots(region_name, timestamp_utc);

CREATE TABLE IF NOT EXISTS alerts_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id     INTEGER NOT NULL REFERENCES snapshots(id),
    region_name     TEXT NOT NULL,
    timestamp_utc   TEXT NOT NULL,
    zone_id         TEXT NOT NULL,
    latitude        REAL NOT NULL,
    longitude       REAL NOT NULL,
    risk_score      REAL NOT NULL,
    severity        TEXT NOT NULL,
    reason          TEXT,
    notified        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_alerts_zone_region
    ON alerts_log(zone_id, region_name);
CREATE INDEX IF NOT EXISTS idx_alerts_severity
    ON alerts_log(severity);

CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT NOT NULL UNIQUE,
    password_hash   TEXT NOT NULL,
    salt            TEXT NOT NULL,
    role            TEXT NOT NULL CHECK(role IN ('admin', 'viewer')),
    created_utc     TEXT NOT NULL
);
"""


@contextmanager
def get_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.executescript(SCHEMA)
    logger.info("Database initialised at %s", DB_PATH)


# ── snapshots / alerts ──────────────────────────────────────────────────────

def save_snapshot(region_name: str, summary: dict, alerts: Iterable, offline: bool) -> int:
    """Persists one refresh cycle: a snapshots row + one alerts_log row per
    zone alert. Returns the new snapshot id (used by callers that want to
    reference it, e.g. tests)."""
    sev = summary.get("severity_breakdown", {})
    from datetime import datetime, timezone
    ts = summary.get("timestamp") or datetime.now(timezone.utc).isoformat()

    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO snapshots
               (region_name, timestamp_utc, offline_mode, total_zones, total_alerts,
                extreme_count, high_count, max_risk_score, mean_risk_score)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (region_name, ts, int(offline), summary.get("total_zones", 0),
             summary.get("total_alerts", 0), sev.get("EXTREME", 0), sev.get("HIGH", 0),
             summary.get("max_risk_score", 0.0), summary.get("mean_risk_score", 0.0)),
        )
        snapshot_id = cur.lastrowid

        rows = [
            (snapshot_id, region_name, ts, a.zone_id, a.latitude, a.longitude,
             a.risk_score, a.severity, a.reason)
            for a in alerts
        ]
        if rows:
            conn.executemany(
                """INSERT INTO alerts_log
                   (snapshot_id, region_name, timestamp_utc, zone_id, latitude,
                    longitude, risk_score, severity, reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
    logger.info("Saved snapshot #%d for %s (%d alerts)", snapshot_id, region_name, len(rows))
    return snapshot_id


def get_recent_snapshots(region_name: Optional[str] = None, limit: int = 50):
    with get_connection() as conn:
        if region_name:
            cur = conn.execute(
                "SELECT * FROM snapshots WHERE region_name = ? ORDER BY id DESC LIMIT ?",
                (region_name, limit),
            )
        else:
            cur = conn.execute("SELECT * FROM snapshots ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur.fetchall()]


def get_recent_alerts(region_name: Optional[str] = None, limit: int = 200):
    with get_connection() as conn:
        if region_name:
            cur = conn.execute(
                "SELECT * FROM alerts_log WHERE region_name = ? ORDER BY id DESC LIMIT ?",
                (region_name, limit),
            )
        else:
            cur = conn.execute("SELECT * FROM alerts_log ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur.fetchall()]


def get_previously_extreme_zone_ids(region_name: str) -> set:
    """Looks at the second-most-recent snapshot for this region (i.e. the
    state BEFORE the refresh that just happened) and returns the set of
    zone_ids that were EXTREME in it. The caller compares this against the
    current EXTREME zone set to find genuinely NEW detections, so a
    notification only fires once per new fire, not on every refresh while
    a zone stays extreme."""
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT id FROM snapshots WHERE region_name = ? ORDER BY id DESC LIMIT 2",
            (region_name,),
        )
        snap_ids = [r["id"] for r in cur.fetchall()]
        if len(snap_ids) < 2:
            return set()
        prev_snapshot_id = snap_ids[1]
        cur = conn.execute(
            "SELECT zone_id FROM alerts_log WHERE snapshot_id = ? AND severity = 'EXTREME'",
            (prev_snapshot_id,),
        )
        return {r["zone_id"] for r in cur.fetchall()}


def mark_alerts_notified(alert_row_ids: Iterable[int]):
    ids = list(alert_row_ids)
    if not ids:
        return
    with get_connection() as conn:
        conn.executemany("UPDATE alerts_log SET notified = 1 WHERE id = ?", [(i,) for i in ids])


# ── users / auth ────────────────────────────────────────────────────────────

_PBKDF2_ITERATIONS = 200_000


def _hash_password(password: str, salt: Optional[str] = None) -> tuple:
    if salt is None:
        salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), _PBKDF2_ITERATIONS
    ).hex()
    return digest, salt


def create_user(username: str, password: str, role: str = "viewer") -> bool:
    """Returns True on success, False if the username already exists."""
    if role not in ("admin", "viewer"):
        raise ValueError("role must be 'admin' or 'viewer'")
    password_hash, salt = _hash_password(password)
    from datetime import datetime, timezone
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash, salt, role, created_utc) "
                "VALUES (?, ?, ?, ?, ?)",
                (username, password_hash, salt, role, datetime.now(timezone.utc).isoformat()),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def verify_user(username: str, password: str) -> Optional[str]:
    """Returns the user's role ('admin'/'viewer') if credentials are valid,
    else None. Uses secrets.compare_digest to avoid timing side-channels."""
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT password_hash, salt, role FROM users WHERE username = ?", (username,)
        )
        row = cur.fetchone()
    if row is None:
        return None
    computed, _ = _hash_password(password, row["salt"])
    if secrets.compare_digest(computed, row["password_hash"]):
        return row["role"]
    return None


def list_users():
    with get_connection() as conn:
        cur = conn.execute("SELECT id, username, role, created_utc FROM users ORDER BY id")
        return [dict(r) for r in cur.fetchall()]


def delete_user(username: str) -> bool:
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM users WHERE username = ?", (username,))
        return cur.rowcount > 0


def set_user_role(username: str, role: str) -> bool:
    if role not in ("admin", "viewer"):
        raise ValueError("role must be 'admin' or 'viewer'")
    with get_connection() as conn:
        cur = conn.execute("UPDATE users SET role = ? WHERE username = ?", (role, username))
        return cur.rowcount > 0


def ensure_default_admin():
    """Creates a default admin/changeme123 account ONLY if the users table
    is completely empty (fresh install), so the dashboard is never locked
    out on first run. Loudly warns to change it - configurable via env vars
    so a real deployment never has to keep the default."""
    import os
    with get_connection() as conn:
        cur = conn.execute("SELECT COUNT(*) AS n FROM users")
        n = cur.fetchone()["n"]
    if n == 0:
        username = os.getenv("DEFAULT_ADMIN_USERNAME", "admin")
        password = os.getenv("DEFAULT_ADMIN_PASSWORD", "changeme123")
        create_user(username, password, role="admin")
        logger.warning(
            "No users existed - created default admin account '%s'. "
            "CHANGE THIS PASSWORD before any real deployment (set "
            "DEFAULT_ADMIN_USERNAME / DEFAULT_ADMIN_PASSWORD env vars, or "
            "change it from the Admin page once logged in).", username,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
    ensure_default_admin()
    with get_connection() as conn:
        n_users = conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
        n_snaps = conn.execute("SELECT COUNT(*) AS n FROM snapshots").fetchone()["n"]
    print(f"Database ready at {DB_PATH}")
    print(f"Users: {n_users}, Snapshots: {n_snaps}")
