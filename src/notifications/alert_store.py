"""
alert_store.py - storage for the opt-in emergency-alert workflow (same database
as the rest of the app: MySQL via DATABASE_URL or the local SQLite file).

Tables (created on first use):
  alert_recipients   registered forest-department / monitoring personnel
  alert_verifications one-time verification codes (hashed, 15-minute expiry)
  emergency_alerts   one row per explicit, confirmed send action (audit)
  alert_deliveries   one row per recipient x channel attempt, with provider status
  inapp_alerts       authenticated in-app notifications + acknowledgements
  simulation_reports reproducible snapshot of each completed simulation

Personal data (email / mobile) is only read by admin pages and the send
function; it is never written to the activity log or to reports.
"""
from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional

from sqlalchemy import Column, Index, Integer, String, Table, Text, insert, select, update

from src.storage import database as db

T_RECIP = Table(
    "alert_recipients", db.metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(120), nullable=False),
    Column("organisation", String(160)),
    Column("designation", String(120)),
    Column("email", String(200)),
    Column("phone", String(32)),                          # E.164, e.g. +919812345678
    Column("prefs", Text),                                # JSON {"email": bool, "sms": bool}
    Column("categories", Text),                           # JSON list of permitted alert categories
    Column("email_verified", Integer, nullable=False, server_default="0"),
    Column("phone_verified", Integer, nullable=False, server_default="0"),
    Column("enabled", Integer, nullable=False, server_default="1"),
    Column("created_utc", String(40), nullable=False),
    Column("updated_utc", String(40), nullable=False),
    Column("created_by", String(80)),
    extend_existing=True,
)
T_VERIFY = Table(
    "alert_verifications", db.metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("recipient_id", Integer, nullable=False),
    Column("channel", String(8), nullable=False),
    Column("code_hash", String(128), nullable=False),
    Column("expires_utc", String(40), nullable=False),
    Column("used", Integer, nullable=False, server_default="0"),
    extend_existing=True,
)
# Hardened verification codes (the older alert_verifications table is kept for history but no longer written):
# bound to the exact address / number, single use, short expiry, limited attempts, resend rate limit.
T_VCODE = Table(
    "recipient_verification_codes", db.metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("recipient_id", Integer, nullable=False),
    Column("channel", String(8), nullable=False),
    Column("target_hash", String(64), nullable=False),    # sha256 of the address / number the code was sent to
    Column("code_hash", String(64), nullable=False),      # sha256(salt | recipient | channel | code)
    Column("salt", String(32), nullable=False),
    Column("created_utc", String(40), nullable=False),
    Column("expires_utc", String(40), nullable=False),
    Column("attempts", Integer, nullable=False, server_default="0"),
    Column("state", String(12), nullable=False, server_default="active"),   # active|used|expired|locked|superseded|failed
)

T_ALERTS = Table(
    "emergency_alerts", db.metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("alert_id", String(40), nullable=False, unique=True),
    Column("created_utc", String(40), nullable=False),
    Column("sender", String(80), nullable=False),
    Column("report_id", String(40)),
    Column("alert_type", String(40), nullable=False),     # SIMULATION ONLY | OBSERVATION-BASED
    Column("category", String(40)),
    Column("region", String(160)),
    Column("subject", String(300)),
    Column("body_sha256", String(64)),
    Column("summary", Text),                              # JSON status summary
    extend_existing=True,
)
T_DELIV = Table(
    "alert_deliveries", db.metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("alert_id", String(40), nullable=False),
    Column("report_id", String(40)),
    Column("recipient_id", Integer, nullable=False),
    Column("channel", String(8), nullable=False),
    Column("provider", String(40), nullable=False),
    Column("status", String(24), nullable=False),         # queued|accepted|delivered|failed|unknown|mock|not_configured|duplicate
    Column("provider_message_id", String(120)),
    Column("error", Text),
    Column("timestamp_utc", String(40), nullable=False),
    extend_existing=True,
)
T_INAPP = Table(
    "inapp_alerts", db.metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("alert_id", String(40), nullable=False),
    Column("created_utc", String(40), nullable=False),
    Column("title", String(300), nullable=False),
    Column("body", Text),
    Column("alert_type", String(40)),
    Column("acked_by", Text),                             # JSON list of usernames
    extend_existing=True,
)
T_REPORTS = Table(
    "simulation_reports", db.metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("report_id", String(40), nullable=False, unique=True),
    Column("created_utc", String(40), nullable=False),
    Column("actor", String(80)),
    Column("snapshot", Text, nullable=False),
    extend_existing=True,
)
_TABLES = [T_RECIP, T_VERIFY, T_VCODE, T_ALERTS, T_DELIV, T_INAPP, T_REPORTS]
_READY: set = set()


# Delivery rows are looked up by alert id (alert history, delivery status). create_all() does not add indexes to
# tables that already exist, so the index is created separately - non-destructive, IF NOT EXISTS semantics.
IX_DELIV_ALERT = Index("ix_alert_deliveries_alert_id", T_DELIV.c.alert_id)


def ensure_tables():
    db.init_db()
    url = db.database_url()
    if url not in _READY:
        db.metadata.create_all(db.get_engine(), tables=_TABLES, checkfirst=True)
        try:
            IX_DELIV_ALERT.create(db.get_engine(), checkfirst=True)
        except Exception as exc:                          # an index is an optimisation, never a reason to fail
            import logging
            logging.getLogger(__name__).warning("could not create delivery index: %s", type(exc).__name__)
        _READY.add(url)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── recipients ──────────────────────────────────────────────────────────────

def _norm_phone(phone) -> Optional[str]:
    """Stored in international format; an invalid number is rejected, never stored half-valid."""
    if not (phone or "").strip():
        return None
    from src.notifications.config import normalise_phone
    n = normalise_phone(phone)
    if n is None:
        raise ValueError("invalid mobile number - use a 10-digit Indian mobile number or +<country code><number>")
    return n


def add_recipient(name: str, organisation: str = "", designation: str = "", email: str = "", phone: str = "",
                  prefs: Optional[dict] = None, categories: Optional[List[str]] = None, actor: str = "admin") -> int:
    ensure_tables()
    now = _now()
    with db.get_connection() as c:
        r = c.execute(insert(T_RECIP).values(
            name=name.strip(), organisation=organisation.strip(), designation=designation.strip(),
            email=(email or "").strip().lower() or None, phone=_norm_phone(phone),
            prefs=json.dumps(prefs or {"email": True, "sms": False}),
            categories=json.dumps(categories or ["SIMULATION ONLY", "OBSERVATION-BASED"]),
            created_utc=now, updated_utc=now, created_by=actor))
        rid = int(r.inserted_primary_key[0])
    db.log_activity("alert_recipient", f"Recipient #{rid} registered", actor=actor)
    return rid


def update_recipient(rid: int, actor: str = "admin", **fields) -> None:
    ensure_tables()
    vals = {}
    for k, v in fields.items():
        if k in ("prefs", "categories"):
            vals[k] = json.dumps(v)
        elif k == "email":
            new = (v or "").strip().lower() or None
            vals["email"] = new
            vals["email_verified"] = 0                    # a changed address must be verified again
        elif k == "phone":
            vals["phone"] = _norm_phone(v)
            vals["phone_verified"] = 0
        elif k in ("name", "organisation", "designation", "enabled", "email_verified"):
            vals[k] = v
    if "email" in fields and fields.get("email_verified"):
        vals["email_verified"] = 1
    vals["updated_utc"] = _now()
    with db.get_connection() as c:
        c.execute(update(T_RECIP).where(T_RECIP.c.id == int(rid)).values(**vals))
    db.log_activity("alert_recipient", f"Recipient #{rid} updated ({', '.join(sorted(fields))})", actor=actor)


def list_recipients(enabled_only: bool = False) -> List[dict]:
    ensure_tables()
    q = select(T_RECIP).order_by(T_RECIP.c.name)
    if enabled_only:
        q = q.where(T_RECIP.c.enabled == 1)
    with db.get_connection() as c:
        rows = [dict(r._mapping) for r in c.execute(q)]
    for r in rows:
        r["prefs"] = json.loads(r.get("prefs") or "{}")
        r["categories"] = json.loads(r.get("categories") or "[]")
    return rows


def get_recipient(rid: int) -> Optional[dict]:
    return next((r for r in list_recipients() if r["id"] == int(rid)), None)


def _h(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


VERIFY_TTL_MIN = 10            # a code is valid for 10 minutes
VERIFY_MAX_ATTEMPTS = 5        # wrong entries before the code is locked
RESEND_MIN_GAP_S = 60          # at most one new code per recipient and channel per minute
RESEND_MAX_PER_HOUR = 5        # and at most five per hour


def _code_hash(salt: str, rid: int, channel: str, code: str) -> str:
    return hashlib.sha256(f"{salt}|{int(rid)}|{channel}|{code}".encode()).hexdigest()


def _target_hash(target: str) -> str:
    return hashlib.sha256((target or "").strip().lower().encode()).hexdigest()


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def verification_wait(rid: int, channel: str, now: Optional[datetime] = None) -> int:
    """Seconds until a new code may be requested (0 = allowed)."""
    ensure_tables()
    now = now or datetime.now(timezone.utc)
    with db.get_connection() as c:
        rows = [r.created_utc for r in c.execute(
            select(T_VCODE.c.created_utc).where((T_VCODE.c.recipient_id == int(rid)) & (T_VCODE.c.channel == channel)
                                                & (T_VCODE.c.created_utc >= _iso(now - timedelta(hours=1)))))]
    if not rows:
        return 0
    last = max(datetime.fromisoformat(x) for x in rows)
    wait = RESEND_MIN_GAP_S - (now - last).total_seconds()
    if len(rows) >= RESEND_MAX_PER_HOUR:
        oldest = min(datetime.fromisoformat(x) for x in rows)
        wait = max(wait, 3600 - (now - oldest).total_seconds())
    return max(0, int(wait + 0.999))


VERIFY_CHANNELS = ("email",)    # SMS verification was removed: mobile numbers are never marked verified


def issue_verification(rid: int, channel: str, target: str, now: Optional[datetime] = None) -> str:
    """New single-use 6-digit code for exactly this e-mail address. Earlier unused codes are superseded.
    The caller checks verification_wait() first and sends the code; the plain code is never stored."""
    if channel not in VERIFY_CHANNELS:
        raise ValueError("only e-mail addresses can be verified")
    ensure_tables()
    now = now or datetime.now(timezone.utc)
    code = f"{secrets.randbelow(10 ** 6):06d}"
    salt = secrets.token_hex(8)
    with db.get_connection() as c:
        c.execute(update(T_VCODE).where((T_VCODE.c.recipient_id == int(rid)) & (T_VCODE.c.channel == channel)
                                        & (T_VCODE.c.state == "active")).values(state="superseded"))
        c.execute(insert(T_VCODE).values(recipient_id=int(rid), channel=channel, target_hash=_target_hash(target),
                                         code_hash=_code_hash(salt, rid, channel, code), salt=salt,
                                         created_utc=_iso(now),
                                         expires_utc=_iso(now + timedelta(minutes=VERIFY_TTL_MIN))))
    return code


def cancel_verification(rid: int, channel: str) -> None:
    """The code could not be sent: invalidate it (it still counts towards the resend limit)."""
    with db.get_connection() as c:
        c.execute(update(T_VCODE).where((T_VCODE.c.recipient_id == int(rid)) & (T_VCODE.c.channel == channel)
                                        & (T_VCODE.c.state == "active")).values(state="failed"))


def check_verification(rid: int, channel: str, code: str, actor: str = "admin",
                       now: Optional[datetime] = None) -> tuple:
    """(ok, reason). Marks the e-mail verified only for a correct, unexpired, unused code that was sent to the
    recipient's CURRENT address. Any other channel is refused (SMS verification was removed)."""
    if channel not in VERIFY_CHANNELS:
        return False, "Only e-mail addresses can be verified."
    ensure_tables()
    now = now or datetime.now(timezone.utc)
    code = (code or "").strip()
    r = get_recipient(rid)
    if r is None:
        return False, "Recipient not found."
    target = r.get("email")
    with db.get_connection() as c:
        row = c.execute(select(T_VCODE).where((T_VCODE.c.recipient_id == int(rid)) & (T_VCODE.c.channel == channel))
                        .order_by(T_VCODE.c.id.desc())).first()
        if row is None or row.state in ("superseded", "failed"):
            return False, "No active code - send a new verification code first."
        if row.state == "used":
            return False, "This code was already used - send a new verification code."
        if row.state == "locked":
            return False, "Too many incorrect attempts - send a new verification code."
        if row.state == "expired" or row.expires_utc < _iso(now):
            c.execute(update(T_VCODE).where(T_VCODE.c.id == row.id).values(state="expired"))
            return False, f"The code has expired (valid {VERIFY_TTL_MIN} minutes) - send a new one."
        if not target or row.target_hash != _target_hash(target):
            c.execute(update(T_VCODE).where(T_VCODE.c.id == row.id).values(state="superseded"))
            return False, "The address / number changed after the code was sent - send a new code."
        if not (code.isdigit() and len(code) == 6 and
                secrets.compare_digest(row.code_hash, _code_hash(row.salt, rid, channel, code))):
            attempts = row.attempts + 1
            locked = attempts >= VERIFY_MAX_ATTEMPTS
            c.execute(update(T_VCODE).where(T_VCODE.c.id == row.id)
                      .values(attempts=attempts, state="locked" if locked else "active"))
            result = (False, ("Too many incorrect attempts - send a new verification code." if locked else
                              f"Incorrect code ({VERIFY_MAX_ATTEMPTS - attempts} attempt(s) left)."))
            note = f"Recipient #{rid} {channel} verification: incorrect code" + (" (locked)" if locked else "")
        else:
            c.execute(update(T_VCODE).where(T_VCODE.c.id == row.id).values(state="used"))
            c.execute(update(T_RECIP).where(T_RECIP.c.id == int(rid)).values(email_verified=1, updated_utc=_now()))
            result, note = (True, "Verified."), f"Recipient #{rid} {channel} verified"
    db.log_activity("alert_recipient", note, actor=actor)     # after the transaction (SQLite: no self-lock)
    return result


def create_verification(rid: int, channel: str) -> str:
    """Compatibility wrapper: issue a code for the recipient's current address / number."""
    r = get_recipient(rid) or {}
    return issue_verification(rid, channel, r.get("email"))


def confirm_verification(rid: int, channel: str, code: str, actor: str = "admin") -> bool:
    return check_verification(rid, channel, code, actor=actor)[0]


# ── alerts, deliveries, in-app ──────────────────────────────────────────────

def record_alert(alert_id: str, sender: str, report_id: str, alert_type: str, category: str, region: str,
                 subject: str, body: str, summary: dict) -> None:
    ensure_tables()
    with db.get_connection() as c:
        c.execute(insert(T_ALERTS).values(alert_id=alert_id, created_utc=_now(), sender=sender, report_id=report_id,
                                          alert_type=alert_type, category=category, region=region, subject=subject,
                                          body_sha256=hashlib.sha256(body.encode()).hexdigest(),
                                          summary=json.dumps(summary)))


def update_alert_summary(alert_id: str, summary: dict) -> None:
    with db.get_connection() as c:
        c.execute(update(T_ALERTS).where(T_ALERTS.c.alert_id == alert_id).values(summary=json.dumps(summary)))


def record_delivery(alert_id: str, report_id: str, recipient_id: int, channel: str, provider: str, status: str,
                    message_id: Optional[str] = None, error: Optional[str] = None) -> None:
    ensure_tables()
    with db.get_connection() as c:
        c.execute(insert(T_DELIV).values(alert_id=alert_id, report_id=report_id, recipient_id=int(recipient_id),
                                         channel=channel, provider=provider, status=status,
                                         provider_message_id=message_id, error=(error or None) and str(error)[:500],
                                         timestamp_utc=_now()))


def recent_delivery(report_id: str, recipient_id: int, channel: str, within_minutes: int = 30) -> bool:
    """A non-failed send of the same simulation report to the same recipient and channel recently?"""
    ensure_tables()
    since = (datetime.now(timezone.utc) - timedelta(minutes=within_minutes)).isoformat(timespec="seconds")
    with db.get_connection() as c:
        rows = list(c.execute(select(T_DELIV.c.status).where(
            (T_DELIV.c.report_id == report_id) & (T_DELIV.c.recipient_id == int(recipient_id)) &
            (T_DELIV.c.channel == channel) & (T_DELIV.c.timestamp_utc >= since))))
    return any(r.status in ("queued", "accepted", "delivered", "mock", "unknown") for r in rows)


def alert_history(limit: int = 100) -> List[dict]:
    ensure_tables()
    with db.get_connection() as c:
        alerts = [dict(r._mapping) for r in c.execute(select(T_ALERTS).order_by(T_ALERTS.c.id.desc()).limit(limit))]
        ids = [a["alert_id"] for a in alerts]
        by_alert: dict = {i: [] for i in ids}
        if ids:                                           # one query for all deliveries (was one per alert)
            for r in c.execute(select(T_DELIV).where(T_DELIV.c.alert_id.in_(ids)).order_by(T_DELIV.c.id)):
                by_alert[r.alert_id].append(dict(r._mapping))
        for a in alerts:
            a["summary"] = json.loads(a.get("summary") or "{}")
            a["deliveries"] = by_alert.get(a["alert_id"], [])
    return alerts


def add_inapp(alert_id: str, title: str, body: str, alert_type: str) -> None:
    ensure_tables()
    with db.get_connection() as c:
        c.execute(insert(T_INAPP).values(alert_id=alert_id, created_utc=_now(), title=title, body=body,
                                         alert_type=alert_type, acked_by="[]"))


def unacked_inapp(username: str, limit: int = 5) -> List[dict]:
    ensure_tables()
    with db.get_connection() as c:
        rows = [dict(r._mapping) for r in c.execute(select(T_INAPP).order_by(T_INAPP.c.id.desc()).limit(50))]
    return [r for r in rows if username not in json.loads(r.get("acked_by") or "[]")][:limit]


def ack_inapp(row_id: int, username: str) -> None:
    ensure_tables()
    with db.get_connection() as c:
        r = c.execute(select(T_INAPP.c.acked_by).where(T_INAPP.c.id == int(row_id))).first()
        if r is None:
            return
        acked = json.loads(r.acked_by or "[]")
        if username not in acked:
            acked.append(username)
        c.execute(update(T_INAPP).where(T_INAPP.c.id == int(row_id)).values(acked_by=json.dumps(acked)))
    db.log_activity("alert_ack", f"In-app alert #{row_id} acknowledged", actor=username)


def save_report_snapshot(report_id: str, snapshot: dict, actor: str = "") -> None:
    ensure_tables()
    try:
        with db.get_connection() as c:
            c.execute(insert(T_REPORTS).values(report_id=report_id, created_utc=_now(), actor=actor,
                                               snapshot=json.dumps(snapshot, default=str)))
    except Exception:                                     # duplicate report ID: already stored
        pass


def load_report_snapshot(report_id: str) -> Optional[dict]:
    ensure_tables()
    with db.get_connection() as c:
        r = c.execute(select(T_REPORTS.c.snapshot).where(T_REPORTS.c.report_id == report_id)).first()
    return json.loads(r.snapshot) if r else None


def mask(value: Optional[str]) -> str:
    """Display form of an email / phone (never the full value in summaries)."""
    v = value or ""
    if "@" in v:
        user, dom = v.split("@", 1)
        return (user[:2] + "***@" + dom) if user else "***@" + dom
    return ("***" + v[-3:]) if len(v) > 3 else "***"


def recent_deliveries(limit: int = 50, channel: Optional[str] = None) -> List[dict]:
    """Latest delivery attempts (all events, incl. tests and verifications), newest first."""
    ensure_tables()
    q = select(T_DELIV).order_by(T_DELIV.c.id.desc()).limit(limit)
    if channel:
        q = select(T_DELIV).where(T_DELIV.c.channel == channel).order_by(T_DELIV.c.id.desc()).limit(limit)
    with db.get_connection() as c:
        return [dict(r._mapping) for r in c.execute(q)]


def update_delivery_status(delivery_id: int, status: str, error: Optional[str] = None) -> None:
    """Provider-confirmed status update (e.g. Twilio 'delivered' / 'undelivered')."""
    ensure_tables()
    vals = {"status": status}
    if error:
        vals["error"] = str(error)[:500]
    with db.get_connection() as c:
        c.execute(update(T_DELIV).where(T_DELIV.c.id == int(delivery_id)).values(**vals))
