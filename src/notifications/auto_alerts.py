"""
auto_alerts.py - AUTOMATIC fire-risk notifications after each twin refresh (dashboard or scheduler).

Decision for every zone in a refresh, in this order (each outcome is written to auto_alert_log):
  1. automatic alerts disabled in the Admin settings      -> "disabled"            (nothing evaluated further)
  2. demo / offline (synthetic) data or a What-If scenario -> "skipped_not_real"    (never notify about
     invented data or hypothetical conditions)
  3. the zone is not an eligible condition: only EXTREME zones (model risk index >= 60 %, the validated
     operating point where at least half of historical zone-days this high burned) are eligible
  4. no valid model confidence for the zone                -> "skipped_invalid"     (missing, NaN, outside 0..1)
  5. confidence below the configured minimum               -> "suppressed_threshold"
  6. same incident notified within the cooldown            -> "suppressed_cooldown" (unless escalation, below)
  7. otherwise                                             -> "sent" (one notification per refresh cycle,
                                                              listing all eligible zones)

"Confidence" is the model's own fire-occurrence probability for that 0.1-degree zone and day
(processed_grid["fire_probability"], the XGBoost output; calibrated on the held-out 2025 season, ECE 0.003).
It is a probability of a satellite-detected fire in the zone that day - a prediction, not a confirmed fire.

Incident identity: "<region>|<zone_id>". Cooldown (default 6 h) is persistent (auto_alert_state table), so it
survives restarts and is shared by the dashboard and the background scheduler. Escalation: inside the cooldown
an incident is notified again only if its confidence rose by at least the escalation step (default +0.10)
above the confidence of its last notification; otherwise the repeat is suppressed.

Concurrency: an incident is CLAIMED with a single conditional UPDATE (or an INSERT on its primary key) before
anything is sent, so two refresh jobs running at the same time cannot both notify it (the database decides);
an in-process lock serialises the dashboard's own background threads as well.

Channels (policy, default "sms_then_email"):
  sms_then_email  SMS to ALERT_SMS_RECIPIENTS; e-mail to ALERT_RECIPIENT_EMAILS only if no SMS was accepted
  both            SMS and e-mail
  sms_only / email_only
SMS and e-mail outcomes are stored separately (alert_deliveries + auto_alert_log); an e-mail fallback that
succeeds never turns a failed SMS into a success. Each notification is sent once - no automatic re-sends.
"""
from __future__ import annotations

import json
import math
import secrets
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from sqlalchemy import Column, Float, Integer, String, Table, Text, insert, select, update
from sqlalchemy.exc import IntegrityError

from src.notifications import alert_store as store
from src.notifications import config as cfg
from src.storage import database as db

POLICIES = ("sms_then_email", "both", "sms_only", "email_only")
DEFAULTS = {"enabled": False, "min_confidence": 0.35, "cooldown_minutes": 360, "escalation_step": 0.10,
            "policy": "sms_then_email", "manual_cooldown_minutes": 5}
SMS_OK = ("accepted", "queued", "sent", "delivered")
EMAIL_OK = ("accepted",)
MAX_LISTED = 5

T_SETTINGS = Table(
    "notification_settings", db.metadata,
    Column("key", String(60), primary_key=True),
    Column("value", Text, nullable=False),
    Column("updated_utc", String(40), nullable=False),
    Column("updated_by", String(80)),
    extend_existing=True,
)
T_STATE = Table(
    "auto_alert_state", db.metadata,
    Column("incident_key", String(200), primary_key=True),
    Column("last_sent_utc", String(40), nullable=False),
    Column("last_confidence", Float),
    Column("last_alert_id", String(40)),
    Column("claim_token", String(40)),
    extend_existing=True,
)
T_LOG = Table(
    "auto_alert_log", db.metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp_utc", String(40), nullable=False),
    Column("cycle_id", String(40), nullable=False),
    Column("incident_key", String(200)),
    Column("region", String(160)),
    Column("zone_id", String(40)),
    Column("latitude", Float),
    Column("longitude", Float),
    Column("severity", String(16)),
    Column("confidence", Float),
    Column("threshold", Float),
    Column("decision", String(32), nullable=False),
    Column("channels", String(40)),
    Column("sms_status", String(40)),
    Column("email_status", String(40)),
    Column("alert_id", String(40)),
    Column("note", Text),
    extend_existing=True,
)
T_MANUAL = Table(                                       # manual simulation SMS: per scenario x recipient guard
    "manual_sms_guard", db.metadata,
    Column("guard_key", String(120), primary_key=True),   # "<report_id>|<sha256(phone)[:16]>" - no raw number stored
    Column("last_sent_utc", String(40), nullable=False),
    Column("claim_token", String(40)),
    Column("last_alert_id", String(40)),
    extend_existing=True,
)
_READY: set = set()
_LOCK = threading.Lock()


def ensure_tables():
    store.ensure_tables()
    url = db.database_url()
    if url not in _READY:
        db.metadata.create_all(db.get_engine(), tables=[T_SETTINGS, T_STATE, T_LOG, T_MANUAL], checkfirst=True)
        _READY.add(url)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(t: datetime) -> str:
    return t.isoformat(timespec="seconds")


# ── settings (persistent, editable on the Admin page) ──────────────────────

def validate_settings(s: dict) -> List[str]:
    out = []
    mc = s.get("min_confidence")
    if not isinstance(mc, (int, float)) or isinstance(mc, bool) or not math.isfinite(mc) or not (0.01 <= mc <= 0.99):
        out.append("minimum confidence must be a probability between 0.01 and 0.99")
    cd = s.get("cooldown_minutes")
    if not isinstance(cd, (int, float)) or isinstance(cd, bool) or not (0 <= cd <= 10080):
        out.append("cooldown must be between 0 and 10080 minutes (7 days)")
    es = s.get("escalation_step")
    if not isinstance(es, (int, float)) or isinstance(es, bool) or not math.isfinite(es) or not (0.01 <= es <= 1.0):
        out.append("escalation step must be between 0.01 and 1.0")
    mm = s.get("manual_cooldown_minutes")
    if not isinstance(mm, (int, float)) or isinstance(mm, bool) or not (1 <= mm <= 1440):
        out.append("manual SMS cooldown must be between 1 and 1440 minutes")
    if s.get("policy") not in POLICIES:
        out.append(f"policy must be one of {', '.join(POLICIES)}")
    if not isinstance(s.get("enabled"), bool):
        out.append("enabled must be true or false")
    return out


def get_settings() -> dict:
    """Stored settings over the defaults. Invalid stored values fall back to the defaults (never crash)."""
    ensure_tables()
    s = dict(DEFAULTS)
    with db.get_connection() as c:
        rows = c.execute(select(T_SETTINGS.c.key, T_SETTINGS.c.value)).all()
    for k, v in rows:
        if k in DEFAULTS:
            try:
                s[k] = json.loads(v)
            except ValueError:
                pass
    if validate_settings(s):
        bad = dict(DEFAULTS)
        for k in DEFAULTS:
            trial = dict(bad, **{k: s[k]})
            if not validate_settings(trial):
                bad[k] = s[k]
        s = bad
    return s


def save_settings(new: dict, actor: str = "admin") -> List[str]:
    s = dict(get_settings())
    s.update({k: new[k] for k in DEFAULTS if k in new})
    for k in ("cooldown_minutes", "manual_cooldown_minutes"):
        if isinstance(s.get(k), float) and s[k].is_integer():
            s[k] = int(s[k])
    probs = validate_settings(s)
    if probs:
        return probs
    now = _iso(_now())
    with db.get_connection() as c:
        for k in DEFAULTS:
            if c.execute(update(T_SETTINGS).where(T_SETTINGS.c.key == k)
                         .values(value=json.dumps(s[k]), updated_utc=now, updated_by=actor)).rowcount == 0:
                c.execute(insert(T_SETTINGS).values(key=k, value=json.dumps(s[k]), updated_utc=now, updated_by=actor))
    db.log_activity("notify_settings", "Automatic alerts: " + ", ".join(f"{k}={s[k]}" for k in DEFAULTS), actor=actor)
    return []


# ── decision helpers ───────────────────────────────────────────────────────

def valid_confidence(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and 0.0 <= f <= 1.0 else None


def incident_key(region: str, zone_id: str) -> str:
    return f"{region}|{zone_id}"


def claim(key: str, confidence: float, cooldown_minutes: float, escalation_step: float, now: Optional[datetime] = None):
    """Atomically claim the right to notify `key`. Returns (claimed, reason, token, info).
    reason: "new" | "cooldown_expired" | "escalation" | "cooldown".
    info: {"last_sent_utc", "last_confidence", "next_routine_utc", "escalation_at"} for notes / the UI."""
    ensure_tables()
    now = now or _now()
    token = secrets.token_hex(8)
    cutoff = _iso(now - timedelta(minutes=cooldown_minutes))
    nxt = lambda iso: _iso(datetime.fromisoformat(iso) + timedelta(minutes=cooldown_minutes))
    with db.get_connection() as c:
        row = c.execute(select(T_STATE).where(T_STATE.c.incident_key == key)).mappings().first()
        if row is None:
            try:
                c.execute(insert(T_STATE).values(incident_key=key, last_sent_utc=_iso(now), last_confidence=confidence,
                                                 claim_token=token))
                return True, "new", token, {"last_sent_utc": None, "last_confidence": None,
                                            "next_routine_utc": nxt(_iso(now)), "escalation_at": confidence + escalation_step}
            except IntegrityError:                      # another job inserted it first
                return False, "cooldown", None, {"last_sent_utc": None, "last_confidence": None,
                                                 "next_routine_utc": nxt(_iso(now)), "escalation_at": None}
        info = {"last_sent_utc": row["last_sent_utc"], "last_confidence": row["last_confidence"],
                "next_routine_utc": nxt(row["last_sent_utc"]),
                "escalation_at": None if row["last_confidence"] is None else row["last_confidence"] + escalation_step}
        expired = row["last_sent_utc"] <= cutoff
        escalated = (row["last_confidence"] is not None and confidence >= row["last_confidence"] + escalation_step - 1e-9)
        if not (expired or escalated):
            return False, "cooldown", None, info
        # conditional update on the values we read: only one concurrent job can win it
        res = c.execute(update(T_STATE).where(T_STATE.c.incident_key == key)
                        .where(T_STATE.c.last_sent_utc == row["last_sent_utc"])
                        .values(last_sent_utc=_iso(now), last_confidence=confidence, claim_token=token))
        if res.rowcount != 1:
            return False, "cooldown", None, info
        info = dict(info, next_routine_utc=nxt(_iso(now)), escalation_at=confidence + escalation_step)
        return True, ("cooldown_expired" if expired else "escalation"), token, info


def _ist(iso: Optional[str]) -> str:
    if not iso or iso.startswith("1970"):
        return "-"
    from src.utils.timezone import format_ist
    return format_ist(iso)


def claim_note(reason: str, conf: float, info: dict, cooldown_minutes: float) -> str:
    """Plain explanation of an automatic decision for the history table."""
    nxt = f"next routine alert eligible after {_ist(info.get('next_routine_utc'))}"
    esc = info.get("escalation_at")
    esc_txt = f" or earlier if probability reaches >= {esc:.2f} (escalation)" if esc is not None and esc <= 1 else ""
    if reason == "new":
        return f"new incident - first alert for this zone; {nxt}{esc_txt}"
    if reason == "cooldown_expired":
        return (f"routine repeat: still EXTREME after the {cooldown_minutes:g}-min cooldown (last alert "
                f"{_ist(info.get('last_sent_utc'))}); {nxt}{esc_txt}")
    if reason == "escalation":
        return (f"escalation permitted inside the cooldown: probability {conf:.2f} >= last alerted "
                f"{(info.get('last_confidence') or 0):.2f} + step; {nxt}{esc_txt}")
    return (f"duplicate of an incident already alerted at {_ist(info.get('last_sent_utc'))}"
            f" (probability then {(info.get('last_confidence') or 0):.2f}); {nxt}{esc_txt}")


def incident_states(limit: int = 50) -> List[dict]:
    """Per-incident automatic-alert state: last alert, its probability, when the next routine alert is allowed."""
    ensure_tables()
    cd = get_settings()["cooldown_minutes"]
    with db.get_connection() as c:
        rows = [dict(r) for r in c.execute(select(T_STATE).order_by(T_STATE.c.last_sent_utc.desc()).limit(limit)).mappings()]
    for r in rows:
        r["next_routine_utc"] = (None if r["last_sent_utc"].startswith("1970") else
                                 _iso(datetime.fromisoformat(r["last_sent_utc"]) + timedelta(minutes=cd)))
    return rows


# ── manual simulation SMS guard (separate from the automatic incident state) ──

def manual_guard_key(scenario_id: str, phone: str) -> str:
    import hashlib
    return f"{scenario_id}|{hashlib.sha256(phone.strip().encode()).hexdigest()[:16]}"


def manual_status(scenario_id: str, phone: str, now: Optional[datetime] = None) -> dict:
    """{"eligible": bool, "next_eligible_utc", "seconds_left"} for one scenario x recipient."""
    ensure_tables()
    now = now or _now()
    mins = get_settings()["manual_cooldown_minutes"]
    with db.get_connection() as c:
        row = c.execute(select(T_MANUAL).where(T_MANUAL.c.guard_key == manual_guard_key(scenario_id, phone))).mappings().first()
    if row is None or row["last_sent_utc"].startswith("1970"):
        return {"eligible": True, "next_eligible_utc": None, "seconds_left": 0, "cooldown_minutes": mins}
    nxt = datetime.fromisoformat(row["last_sent_utc"]) + timedelta(minutes=mins)
    left = max(0, int((nxt - now).total_seconds() + 0.999))
    return {"eligible": left == 0, "next_eligible_utc": _iso(nxt), "seconds_left": left, "cooldown_minutes": mins,
            "last_alert_id": row["last_alert_id"]}


def manual_claim(scenario_id: str, phone: str, now: Optional[datetime] = None):
    """Atomically claim one manual send. Returns (claimed, token, status-dict)."""
    ensure_tables()
    now = now or _now()
    key = manual_guard_key(scenario_id, phone)
    st = manual_status(scenario_id, phone, now)
    if not st["eligible"]:
        return False, None, st
    token = secrets.token_hex(8)
    with db.get_connection() as c:
        row = c.execute(select(T_MANUAL).where(T_MANUAL.c.guard_key == key)).mappings().first()
        if row is None:
            try:
                c.execute(insert(T_MANUAL).values(guard_key=key, last_sent_utc=_iso(now), claim_token=token))
            except IntegrityError:
                return False, None, manual_status(scenario_id, phone, now)
        else:
            res = c.execute(update(T_MANUAL).where(T_MANUAL.c.guard_key == key)
                            .where(T_MANUAL.c.last_sent_utc == row["last_sent_utc"])
                            .values(last_sent_utc=_iso(now), claim_token=token, last_alert_id=None))
            if res.rowcount != 1:
                return False, None, manual_status(scenario_id, phone, now)
    return True, token, st


def manual_finish(scenario_id: str, phone: str, token: str, alert_id: str, release: bool):
    """Store the alert id of a manual send, or release the guard when nothing was sent (definite failure)."""
    vals = {"last_sent_utc": "1970-01-01T00:00:00+00:00"} if release else {"last_alert_id": alert_id}
    with db.get_connection() as c:
        c.execute(update(T_MANUAL).where(T_MANUAL.c.guard_key == manual_guard_key(scenario_id, phone))
                  .where(T_MANUAL.c.claim_token == token).values(**vals))


def _set_alert_id(key: str, token: str, alert_id: str):
    with db.get_connection() as c:
        c.execute(update(T_STATE).where(T_STATE.c.incident_key == key).where(T_STATE.c.claim_token == token)
                  .values(last_alert_id=alert_id))


def _release(key: str, token: str):
    """Undo a claim whose notification certainly failed on every channel (nothing was delivered)."""
    with db.get_connection() as c:
        c.execute(update(T_STATE).where(T_STATE.c.incident_key == key).where(T_STATE.c.claim_token == token)
                  .values(last_sent_utc="1970-01-01T00:00:00+00:00"))


def _log(rows: List[dict]):
    if rows:
        with db.get_connection() as c:
            c.execute(insert(T_LOG), rows)


def history(limit: int = 100) -> List[dict]:
    ensure_tables()
    with db.get_connection() as c:
        return [dict(r) for r in c.execute(select(T_LOG).order_by(T_LOG.c.id.desc()).limit(limit)).mappings()]


def describe_row(r: dict) -> str:
    """History label for one decision row (distinguishes new / routine repeat / escalation sends)."""
    note = (r.get("note") or "")
    if r.get("decision") == "sent":
        if note.startswith("escalation"):
            return "Automatic alert permitted: risk escalation condition met"
        if note.startswith("routine repeat"):
            return "Automatic alert sent: routine repeat after cooldown"
        if note.startswith("new incident"):
            return "Automatic alert sent: new incident"
    return describe_decision(r.get("decision"))


def describe_decision(decision: str) -> str:
    return {"sent": "Notification sent (see channel status)",
            "suppressed_threshold": "Suppressed by confidence threshold",
            "suppressed_cooldown": "Automatic alert suppressed by cooldown (duplicate of an alerted incident)",
            "skipped_invalid": "Skipped: no valid model confidence",
            "skipped_not_real": "Skipped: demo / What-If data, not real observations",
            "disabled": "Automatic alerts disabled", "no_channels": "Not sent: no SMS / e-mail recipient configured",
            "failed": "Failed on every channel (will retry next refresh)", "error": "Evaluation error",
            "not_eligible": "Not an alert condition"}.get(decision, decision)


def describe_channel(status: Optional[str], channel: str) -> str:
    if not status:
        return "-"
    if channel == "sms":
        return {"accepted": "Request accepted by gateway", "queued": "Handed to the phone (delivery status unknown)",
                "sent": "Sent to the network (delivery status unknown)", "delivered": "Delivery confirmed by provider",
                "unknown": "Delivery status unknown", "failed": "Failed", "not_configured": "Not configured",
                "mock": "Mock (nothing sent)", "skipped": "Not attempted (policy)"}.get(status, status)
    return {"accepted": "Email accepted by SMTP server", "failed": "Failed", "not_configured": "Not configured",
            "mock": "Mock (nothing sent)", "skipped": "Not attempted (SMS accepted / policy)"}.get(status, status)


# ── message ────────────────────────────────────────────────────────────────

def build_messages(region: str, items: List[dict], alert_id: str, data_label: str) -> Dict[str, str]:
    """Automatic alert texts. The ML Fire Risk Score is the value prediction_service stored for the zone
    (processed_grid.ml_risk_score, the same number the report prints); its category comes from risk_score.category.
    'EXTREME' is the dashboard alert level that made the zone eligible - labelled as such."""
    from src.ml_models.risk_score import category, format_score
    from src.notifications.emergency import ADVICE, SYSTEM_ID, fmt_prob, ist
    when = ist(_iso(_now()))
    top = sorted(items, key=lambda x: x["confidence"], reverse=True)
    first = top[0]

    def score_of(x):
        sc = format_score(x.get("ml_risk_score"))
        return f"{sc} ({category(float(x['ml_risk_score']))})" if sc else None

    s1 = score_of(first)
    sms_lines = ["🚨🌲 FOREST FIRE DIGITAL TWIN",
                 "⚠️ AUTOMATIC FIRE-RISK ALERT — MODEL PREDICTION, NOT A CONFIRMED FIRE",
                 f"📍 {region} · zone {first['zone_id']} ({first['lat']:.3f}, {first['lon']:.3f})",
                 f"📊 ML Fire Risk Score: {s1}" if s1 else "📊 ML Fire Risk Score: not available",
                 f"🔥 Fire probability: {fmt_prob(first['confidence'] * 100)}",
                 "⚡ Dashboard alert level: EXTREME"]
    if len(top) > 1:
        sms_lines.append(f"➕ {len(top) - 1} more zone(s) - see the dashboard")
    sms_lines += [f"🕒 {when}", f"🆔 Ref: {alert_id}"]
    lines = ["AUTOMATIC FIRE-RISK ALERT - model prediction, not a confirmed fire.", "",
             f"System: {SYSTEM_ID}", f"Alert ID: {alert_id}", f"Region: {region}", f"Evaluated: {when}",
             f"Data: {data_label}", "",
             f"{len(top)} zone(s) reached the EXTREME dashboard alert level with a model fire probability at or above "
             f"the configured minimum confidence:"]
    for x in top[:MAX_LISTED]:
        sx = score_of(x)
        lines.append(f"  - Zone {x['zone_id']} ({x['lat']:.4f} N, {x['lon']:.4f} E): "
                     + (f"ML Fire Risk Score {sx}, " if sx else "")
                     + f"fire-occurrence probability {fmt_prob(x['confidence'] * 100)} (threshold "
                       f"{x['threshold'] * 100:.0f}%) - {x['reason']}")
    if len(top) > MAX_LISTED:
        lines.append(f"  ... and {len(top) - MAX_LISTED} more zone(s) (see the dashboard).")
    lines += ["", "ML Fire Risk Score = how the model ranks the zone-day against historical fire-season days "
              "(a ranking, not a probability). Fire-occurrence probability = the model's probability of a "
              "satellite-detected fire in the ~11 km zone today. Both are predictions for situational awareness, "
              "not observations or an official forecast.", "",
              "Advisory:"] + [f"  - {a}" for a in ADVICE]
    return {"subject": f"[FIRE RISK ALERT] {len(top)} EXTREME zone(s) - {region} - {alert_id}",
            "body": "\n".join(lines), "sms": "\n".join(sms_lines)}


# ── main entry ─────────────────────────────────────────────────────────────

def process_refresh(twin, actor: str = "system", email_provider=None, sms_provider=None,
                    settings: Optional[dict] = None) -> dict:
    """Evaluate a completed refresh and notify if warranted. Never raises (failures are logged).
    Returns {"decision", "sent_zones", "sms_status", "email_status", "alert_id", "counts"}."""
    out = {"decision": "not_eligible", "sent_zones": 0, "sms_status": None, "email_status": None,
           "alert_id": None, "counts": {}}
    try:
        with _LOCK:
            return _process(twin, actor, email_provider, sms_provider, settings, out)
    except Exception as exc:                            # notifications must never break a refresh
        import logging
        logging.getLogger(__name__).warning("automatic alert evaluation failed: %s", exc)
        out["decision"] = "error"
        out["error"] = cfg.redact(str(exc))[:300]
        return out


def _process(twin, actor, email_provider, sms_provider, settings, out) -> dict:
    ensure_tables()
    s = settings or get_settings()
    snap = twin.current_snapshot
    if snap is None:
        return out
    region = twin.region.name
    cycle = "CYCLE-" + _now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2).upper()
    eligible = [a for a in snap.alerts if a.severity == "EXTREME"]
    if not eligible:
        return out
    grid = snap.processed_grid
    probs = {}
    if "fire_probability" in grid.columns and "zone_id" in grid.columns:
        probs = dict(zip(grid["zone_id"].astype(str), grid["fire_probability"]))
    scores = {}                                        # ML Fire Risk Score stored by prediction_service.predict()
    if "ml_risk_score" in grid.columns and "zone_id" in grid.columns:
        scores = dict(zip(grid["zone_id"].astype(str), grid["ml_risk_score"]))
    scenario = getattr(getattr(twin, "ingestion", None), "scenario", None)
    real = not twin.offline and not scenario
    data_label = "live data (NASA FIRMS / OpenWeatherMap)" if real else (
        "What-If scenario" if scenario else "demo / offline synthetic data")
    now_iso = _iso(_now())
    thr = float(s["min_confidence"])
    policy = s["policy"]
    sms_to = [p for p in cfg.sms_recipients() if cfg.valid_phone(p)]
    email_to = [e for e in cfg.smtp_settings()["recipients"] if cfg.valid_email(e)]
    want_sms = policy in ("sms_then_email", "both", "sms_only")
    want_email = policy in ("both", "email_only")
    has_channel = bool((want_sms and sms_to) or ((want_email or policy == "sms_then_email") and email_to))
    rows, to_send = [], []
    counts: Dict[str, int] = {}

    def row(a, conf, decision, note=""):
        counts[decision] = counts.get(decision, 0) + 1
        return {"timestamp_utc": now_iso, "cycle_id": cycle, "incident_key": incident_key(region, a.zone_id),
                "region": region, "zone_id": str(a.zone_id), "latitude": float(a.latitude),
                "longitude": float(a.longitude), "severity": a.severity, "confidence": conf, "threshold": thr,
                "decision": decision, "channels": None, "sms_status": None, "email_status": None, "alert_id": None,
                "note": note}

    for a in eligible:
        conf = valid_confidence(probs.get(str(a.zone_id)))
        if not s["enabled"]:
            rows.append(row(a, conf, "disabled"))
        elif not real:
            rows.append(row(a, conf, "skipped_not_real", data_label))
        elif conf is None:
            rows.append(row(a, None, "skipped_invalid", "model fire probability missing or not in 0..1"))
        elif conf < thr:
            rows.append(row(a, conf, "suppressed_threshold", f"{conf:.3f} < {thr:.3f}"))
        elif not has_channel:                          # checked before claiming: nothing is marked as notified
            rows.append(row(a, conf, "no_channels", "no valid ALERT_SMS_RECIPIENTS / ALERT_RECIPIENT_EMAILS for "
                                                    f"policy {policy}"))
        else:
            ok, why, token, info = claim(incident_key(region, a.zone_id), conf, s["cooldown_minutes"],
                                         s["escalation_step"])
            if not ok:
                rows.append(row(a, conf, "suppressed_cooldown", claim_note(why, conf, info, s["cooldown_minutes"])))
            else:
                r = row(a, conf, "sent", claim_note(why, conf, info, s["cooldown_minutes"]))
                rows.append(r)
                to_send.append((a, conf, token, r))
    out["counts"] = counts
    if not to_send:
        _log(rows)
        out["decision"] = max(counts, key=counts.get) if counts else "not_eligible"
        return out

    alert_id = "AUTO-" + _now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2).upper()
    items = [{"zone_id": a.zone_id, "lat": a.latitude, "lon": a.longitude, "confidence": c, "threshold": thr,
              "reason": a.reason, "ml_risk_score": scores.get(str(a.zone_id))} for a, c, _, _ in to_send]
    msg = build_messages(region, items, alert_id, data_label)
    store.record_alert(alert_id, actor, None, "AUTOMATIC", "EXTREME", region, msg["subject"], msg["body"],
                       {"state": "sending", "zones": len(items), "cycle": cycle})
    from src.notifications import providers as prov
    sms_statuses, email_statuses = [], []
    if want_sms and sms_to:
        sp = sms_provider or prov.sms_provider()
        for p in sms_to:
            res = sp.send(p, msg["sms"])
            store.record_delivery(alert_id, None, 0, "sms", res.provider, res.status, res.message_id, res.error)
            sms_statuses.append(res.status)
    sms_ok = any(x in SMS_OK for x in sms_statuses)
    if email_to and (want_email or (policy == "sms_then_email" and not sms_ok)):
        ep = email_provider or prov.EmailProvider()
        for e in email_to:                              # one attempt per address, never re-sent
            res = ep.send(e, msg["subject"], msg["body"])
            store.record_delivery(alert_id, None, 0, "email", res.provider, res.status, res.message_id, res.error)
            email_statuses.append(res.status)

    def summarise(sts, ok):
        if not sts:
            return None
        good = [x for x in sts if x in ok]
        return good[0] if len(good) == len(sts) else (f"{len(good)}/{len(sts)} ok" if good else sts[0])

    sms_sum = summarise(sms_statuses, SMS_OK) or ("skipped" if want_sms else None)
    email_sum = summarise(email_statuses, EMAIL_OK) or ("skipped" if email_to and policy != "sms_only" else None)
    chans = "+".join(c for c, st in (("sms", sms_statuses), ("email", email_statuses)) if st) or "-"
    any_ok = sms_ok or any(x in EMAIL_OK for x in email_statuses)
    uncertain = any(x == "unknown" for x in sms_statuses + email_statuses)
    for a, _, token, r in to_send:
        r.update(channels=chans, sms_status=sms_sum, email_status=email_sum, alert_id=alert_id)
        if any_ok or uncertain:
            _set_alert_id(r["incident_key"], token, alert_id)
        else:                                          # every channel failed for certain: allow a retry next refresh
            r["decision"], r["note"] = "failed", "all channels failed; incident left eligible for the next refresh"
            _release(r["incident_key"], token)
    _log(rows)
    store.update_alert_summary(alert_id, {"state": "done", "sms": sms_statuses, "email": email_statuses,
                                          "zones": len(items), "policy": policy})
    db.log_activity("auto_alert", f"{alert_id}: {len(items)} EXTREME zone(s) at >= {thr:.0%} confidence; "
                    f"SMS {sms_sum or '-'}, email {email_sum or '-'}", actor, region)
    out.update(decision="sent" if (any_ok or uncertain) else "failed", sent_zones=len(items), sms_status=sms_sum,
               email_status=email_sum, alert_id=alert_id)
    return out
