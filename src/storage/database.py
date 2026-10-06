"""
database.py - persistent storage layer (SQLAlchemy Core: MySQL or SQLite).

Backend is chosen by the DATABASE_URL environment variable:

    DATABASE_URL=mysql+pymysql://USER:PASSWORD@HOST:PORT/DBNAME   -> MySQL
    (unset)                                                        -> SQLite file
                                                                      data/digital_twin.db

Hosted MySQL providers usually require TLS. Set DATABASE_SSL_CA to the
provider's CA certificate path (verified TLS), or DATABASE_SSL=true for
encrypted-but-unverified TLS when no CA file is provided.

Every query goes through SQLAlchemy Core with bound parameters, so the same
code runs unchanged on both backends and is not open to SQL injection. The
SQLite fallback keeps local development and the test-suite dependency-free.

Tables:
  - snapshots    : one row per DigitalTwin refresh (time series of twin state)
  - alerts_log   : one row per zone alert in a snapshot (audit trail)
  - users        : login credentials (PBKDF2-SHA256, per-user salt) + role
  - activity_log : every meaningful action - refreshes, alerts, emails,
                   sign-ins, admin changes - shown live on the Activity page
"""
import hashlib
import logging
import time
import threading
import os
import secrets
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from sqlalchemy import (
    Column, Float, ForeignKey, Index, Integer, MetaData, String, Table, Text,
    create_engine, delete, event, func, insert, select, update,
)
from sqlalchemy.engine import Engine
from sqlalchemy import exc as exc_mod
from sqlalchemy.exc import IntegrityError, InterfaceError, OperationalError

# Read DATABASE_URL / DATABASE_SSL* from the project's .env even when this module
# is used on its own (e.g. scripts/migrate_sqlite_to_mysql.py). Real environment
# variables still win over the file.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "digital_twin.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

metadata = MetaData()

snapshots = Table(
    "snapshots", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("region_name", String(120), nullable=False),
    Column("timestamp_utc", String(40), nullable=False),
    Column("offline_mode", Integer, nullable=False),
    Column("total_zones", Integer, nullable=False),
    Column("total_alerts", Integer, nullable=False),
    Column("extreme_count", Integer, nullable=False, server_default="0"),
    Column("high_count", Integer, nullable=False, server_default="0"),
    Column("max_risk_score", Float, nullable=False),
    Column("mean_risk_score", Float, nullable=False),
    Index("idx_snapshots_region_time", "region_name", "timestamp_utc"),
)

alerts_log = Table(
    "alerts_log", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("snapshot_id", Integer, ForeignKey("snapshots.id"), nullable=False),
    Column("region_name", String(120), nullable=False),
    Column("timestamp_utc", String(40), nullable=False),
    Column("zone_id", String(40), nullable=False),
    Column("latitude", Float, nullable=False),
    Column("longitude", Float, nullable=False),
    Column("risk_score", Float, nullable=False),
    Column("severity", String(16), nullable=False),
    Column("reason", Text),
    Column("notified", Integer, nullable=False, server_default="0"),
    Index("idx_alerts_zone_region", "zone_id", "region_name"),
    Index("idx_alerts_severity", "severity"),
)

users = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", String(80), nullable=False, unique=True),
    Column("password_hash", String(128), nullable=False),
    Column("salt", String(64), nullable=False),
    Column("role", String(16), nullable=False),
    Column("created_utc", String(40), nullable=False),
)

activity_log = Table(
    "activity_log", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp_utc", String(40), nullable=False),
    Column("event_type", String(40), nullable=False),
    Column("actor", String(80), nullable=False, server_default="system"),
    Column("region_name", String(120)),
    Column("detail", Text),
    Index("idx_activity_time", "timestamp_utc"),
    Index("idx_activity_type", "event_type"),
)

_ENGINES: dict = {}

# Set when the configured remote database cannot be reached: the app then keeps
# working on the local SQLite file instead of crashing, and says so on screen.
_FALLBACK: dict = {"reason": None}


def _configured_url() -> str:
    return os.getenv("DATABASE_URL", "").strip() or f"sqlite:///{DB_PATH}"


def database_url() -> str:
    if _FALLBACK["reason"]:
        return f"sqlite:///{DB_PATH}"
    return _configured_url()


def fallback_reason() -> Optional[str]:
    """Why the app is on the local fallback database, or None when it is not."""
    return _FALLBACK["reason"]


def reset_fallback():
    """Forget a previous failure so the next init_db() tries the remote database again."""
    _FALLBACK["reason"] = None


def backend_name() -> str:
    """Human-readable backend label for the dashboard."""
    if _FALLBACK["reason"]:
        return "SQLite (local fallback)"
    url = database_url()
    if url.startswith("mysql"):
        return "MySQL"
    if url.startswith("postgres"):
        return "PostgreSQL"
    return "SQLite"


_IDLE_PING_AFTER_S = 30.0


def _install_idle_ping(engine: Engine) -> None:
    """pool_pre_ping costs one extra round trip on EVERY checkout, which on a
    remote MySQL is ~100-300 ms per query. Ping only connections that sat idle
    long enough for the host to have dropped them."""
    @event.listens_for(engine, "checkout")
    def _ping_if_idle(dbapi_conn, record, proxy):
        now = time.monotonic()
        last = record.info.get("last_used")
        record.info["last_used"] = now
        if last is not None and now - last > _IDLE_PING_AFTER_S:
            try:
                dbapi_conn.ping(reconnect=False)
            except Exception as exc:
                raise exc_mod.DisconnectionError(str(exc)) from exc


def get_engine() -> Engine:
    url = database_url()
    if url not in _ENGINES:
        if url.startswith("sqlite"):
            engine = create_engine(url, future=True)

            @event.listens_for(engine, "connect")
            def _fk_on(dbapi_conn, _):
                dbapi_conn.execute("PRAGMA foreign_keys = ON;")
        else:
            connect_args = {"connect_timeout": 8} if url.startswith("mysql") else {}
            ca = os.getenv("DATABASE_SSL_CA", "").strip()
            if ca:
                connect_args["ssl"] = {"ca": ca}
            elif os.getenv("DATABASE_SSL", "").strip().lower() in ("1", "true", "yes", "required"):
                connect_args["ssl"] = {"verify_mode": "none"}
            # Hosted MySQL closes idle connections; pre-ping + recycle keeps
            # the pool healthy across the dashboard's quiet periods.
            engine = create_engine(url, future=True, pool_recycle=280,
                                   pool_size=5, max_overflow=5, connect_args=connect_args)
            _install_idle_ping(engine)
        _ENGINES[url] = engine
    return _ENGINES[url]


@contextmanager
def get_connection():
    """Transactional connection: commits on success, rolls back on error."""
    with get_engine().begin() as conn:
        yield conn


_INITIALISED: set = set()


def init_db():
    """Creates missing tables. Runs once per process and database: the check
    costs several round trips on a remote MySQL, and callers invoke this on
    every refresh."""
    url = database_url()
    if url in _INITIALISED:
        return
    try:
        metadata.create_all(get_engine(), checkfirst=True)
    except (OperationalError, InterfaceError) as exc:
        if url.startswith("sqlite"):
            raise
        # Remote database unreachable (no internet, DNS failure, service powered
        # off, bad credentials...): carry on with the local SQLite file.
        first_line = str(getattr(exc, "orig", exc)).splitlines()[0][:200]
        _FALLBACK["reason"] = first_line
        logger.warning("Database %s unreachable (%s) - using local SQLite instead", url.split("@")[-1], first_line)
        url = database_url()
        metadata.create_all(get_engine(), checkfirst=True)
    _INITIALISED.add(url)
    logger.info("Database initialised (%s)", backend_name())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rows(result) -> list:
    return [dict(r._mapping) for r in result]


# ── activity log ────────────────────────────────────────────────────────────

def log_activity(event_type: str, detail: str = "", actor: str = "system",
                 region_name: Optional[str] = None) -> None:
    """Records one user/system action. Never raises - an activity-log write
    failing must not break the refresh or the page that triggered it."""
    values = dict(timestamp_utc=_now(), event_type=event_type, actor=actor or "system",
                  region_name=region_name, detail=detail)

    def _write():
        try:
            with get_connection() as conn:
                conn.execute(insert(activity_log).values(**values))
        except Exception as exc:  # pragma: no cover - logged, deliberately swallowed
            logger.warning("activity_log write failed: %s", exc)

    # Against a remote MySQL the insert is a full network round trip the page
    # does not need to wait for; the local SQLite file is instant, and tests
    # read the log straight back, so it stays synchronous there.
    if database_url().startswith("sqlite"):
        _write()
    else:
        threading.Thread(target=_write, daemon=True).start()


def get_activity(limit: int = 200, event_types: Optional[Iterable[str]] = None,
                 since_id: Optional[int] = None) -> list:
    q = select(activity_log).order_by(activity_log.c.id.desc()).limit(limit)
    if event_types:
        q = q.where(activity_log.c.event_type.in_(list(event_types)))
    if since_id is not None:
        q = q.where(activity_log.c.id > since_id)
    with get_connection() as conn:
        return _rows(conn.execute(q))


def get_activity_event_types() -> list:
    with get_connection() as conn:
        return [r[0] for r in conn.execute(
            select(activity_log.c.event_type).distinct().order_by(activity_log.c.event_type))]


def get_table_counts() -> dict:
    with get_connection() as conn:
        return {t.name: conn.execute(select(func.count()).select_from(t)).scalar_one()
                for t in (snapshots, alerts_log, users, activity_log)}


# ── snapshots / alerts ──────────────────────────────────────────────────────

def save_snapshot(region_name: str, summary: dict, alerts: Iterable, offline: bool) -> int:
    """Persists one refresh cycle: a snapshots row + one alerts_log row per
    zone alert. Returns the new snapshot id."""
    sev = summary.get("severity_breakdown", {})
    ts = summary.get("timestamp") or _now()
    alerts = list(alerts)

    with get_connection() as conn:
        res = conn.execute(insert(snapshots).values(
            region_name=region_name, timestamp_utc=ts, offline_mode=int(offline),
            total_zones=int(summary.get("total_zones", 0)),
            total_alerts=int(summary.get("total_alerts", 0)),
            extreme_count=int(sev.get("EXTREME", 0)), high_count=int(sev.get("HIGH", 0)),
            max_risk_score=float(summary.get("max_risk_score", 0.0)),
            mean_risk_score=float(summary.get("mean_risk_score", 0.0)),
        ))
        snapshot_id = int(res.inserted_primary_key[0])
        if alerts:
            conn.execute(insert(alerts_log), [
                dict(snapshot_id=snapshot_id, region_name=region_name, timestamp_utc=ts,
                     zone_id=a.zone_id, latitude=float(a.latitude), longitude=float(a.longitude),
                     risk_score=float(a.risk_score), severity=a.severity, reason=a.reason)
                for a in alerts
            ])
    logger.info("Saved snapshot #%d for %s (%d alerts)", snapshot_id, region_name, len(alerts))
    return snapshot_id


def get_recent_snapshots(region_name: Optional[str] = None, limit: int = 50):
    q = select(snapshots).order_by(snapshots.c.id.desc()).limit(limit)
    if region_name:
        q = q.where(snapshots.c.region_name == region_name)
    with get_connection() as conn:
        return _rows(conn.execute(q))


def get_recent_alerts(region_name: Optional[str] = None, limit: int = 200):
    q = select(alerts_log).order_by(alerts_log.c.id.desc()).limit(limit)
    if region_name:
        q = q.where(alerts_log.c.region_name == region_name)
    with get_connection() as conn:
        return _rows(conn.execute(q))


def get_previously_extreme_zone_ids(region_name: str) -> set:
    """Zone_ids that were EXTREME in the second-most-recent snapshot for this
    region (the state BEFORE the refresh that just happened), so a
    notification fires once per new fire, not on every refresh."""
    with get_connection() as conn:
        snap_ids = [r[0] for r in conn.execute(
            select(snapshots.c.id).where(snapshots.c.region_name == region_name)
            .order_by(snapshots.c.id.desc()).limit(2))]
        if len(snap_ids) < 2:
            return set()
        return {r[0] for r in conn.execute(
            select(alerts_log.c.zone_id).where(
                (alerts_log.c.snapshot_id == snap_ids[1]) & (alerts_log.c.severity == "EXTREME")))}


def mark_alerts_notified(alert_row_ids: Iterable[int]):
    ids = [int(i) for i in alert_row_ids]
    if not ids:
        return
    with get_connection() as conn:
        conn.execute(update(alerts_log).where(alerts_log.c.id.in_(ids)).values(notified=1))


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
    try:
        with get_connection() as conn:
            conn.execute(insert(users).values(
                username=username, password_hash=password_hash, salt=salt,
                role=role, created_utc=_now()))
        return True
    except IntegrityError:
        return False


def verify_user(username: str, password: str) -> Optional[str]:
    """Role ('admin'/'viewer') if the credentials are valid, else None.
    Constant-time comparison avoids timing side-channels."""
    with get_connection() as conn:
        row = conn.execute(select(users.c.password_hash, users.c.salt, users.c.role)
                           .where(users.c.username == username)).first()
    if row is None:
        return None
    computed, _ = _hash_password(password, row.salt)
    return row.role if secrets.compare_digest(computed, row.password_hash) else None


def list_users():
    with get_connection() as conn:
        return _rows(conn.execute(select(users.c.id, users.c.username, users.c.role,
                                         users.c.created_utc).order_by(users.c.id)))


def delete_user(username: str) -> bool:
    with get_connection() as conn:
        return conn.execute(delete(users).where(users.c.username == username)).rowcount > 0


def set_user_role(username: str, role: str) -> bool:
    if role not in ("admin", "viewer"):
        raise ValueError("role must be 'admin' or 'viewer'")
    with get_connection() as conn:
        return conn.execute(update(users).where(users.c.username == username)
                            .values(role=role)).rowcount > 0


def ensure_default_admin():
    """Creates a default admin account ONLY if the users table is empty
    (fresh install), so the dashboard is never locked out on first run.
    Configurable via DEFAULT_ADMIN_USERNAME / DEFAULT_ADMIN_PASSWORD."""
    with get_connection() as conn:
        n = conn.execute(select(func.count()).select_from(users)).scalar_one()
    if n == 0:
        username = os.getenv("DEFAULT_ADMIN_USERNAME", "admin")
        password = os.getenv("DEFAULT_ADMIN_PASSWORD", "changeme123")
        create_user(username, password, role="admin")
        logger.warning(
            "No users existed - created default admin account '%s'. Change this "
            "password before any real deployment (DEFAULT_ADMIN_USERNAME / "
            "DEFAULT_ADMIN_PASSWORD env vars).", username,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
    ensure_default_admin()
    print(f"Database ready ({backend_name()})")
    for table, n in get_table_counts().items():
        print(f"  {table:<13} {n} rows")
