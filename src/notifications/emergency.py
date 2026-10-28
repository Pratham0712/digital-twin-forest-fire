"""
emergency.py - OPT-IN emergency alerts about a completed simulation.

Nothing here runs automatically: a simulation starting, completing or
predicting high risk never sends anything. `send_alert` refuses to run
without `confirmed=True`, which the UI only passes after the authorised
sender reviewed the message and recipients and pressed the final
confirmation button.

Alert type comes ONLY from the simulation snapshot:
  * WHAT-IF / hypothetical ignition -> "SIMULATION ONLY" (fixed wording below)
  * LIVE run started from valid NASA FIRMS detections -> "OBSERVATION-BASED"
    (satellite hotspots are observations, not ground-verified fires)
A high ML risk or missing data never produces a confirmed-fire message.
"""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from src.notifications import alert_store as store
from src.notifications import providers as _prov
from src.notifications.providers import EmailProvider, SendResult

SIM_ONLY = "SIMULATION ONLY — This is a hypothetical forest-fire scenario, not confirmation of an active fire."
OBS_NOTE = ("Based on NASA FIRMS satellite hotspot detections. Satellite hotspots are observations, not "
            "independently ground-verified fires.")
SYSTEM_ID = "Forest Fire Digital Twin - Bandipur / Karnataka"
ADVICE = ["Verify the situation through official forest-department channels before acting.",
          "Do not travel into the area based on this message alone.",
          "Follow instructions from the Karnataka Forest Department and local authorities."]


def alert_type_of(snapshot: dict) -> str:
    if snapshot.get("mode") == "LIVE" and (snapshot.get("firms") or {}).get("n_valid", 0) > 0:
        return "OBSERVATION-BASED"
    return "SIMULATION ONLY"


def ist(utc_iso: Optional[str]) -> str:
    if not utc_iso:
        return "not available"
    from src.utils.timezone import format_ist
    return format_ist(utc_iso)


# ── one shared, already-computed fact sheet for every e-mail / SMS about a simulation ────────
# All values are copied from the simulation snapshot - the same prediction block the simulation
# report prints (simulation_report._prediction_block / _risk_line). Nothing is re-predicted here.

def percent_from_fraction(p) -> Optional[float]:
    """A model probability in 0..1 as a percentage (0.0145 -> 1.45). None when missing / out of range."""
    try:
        v = float(p)
    except (TypeError, ValueError):
        return None
    return v * 100.0 if v == v and 0.0 <= v <= 1.0 else None


def percent_value(v) -> Optional[float]:
    """A value that is ALREADY a 0..100 percentage / score (never multiplied again)."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x and 0.0 <= x <= 100.0 else None


def fmt_prob(pct: Optional[float]) -> Optional[str]:
    """Same precision rule as the report (_pct): 0.18%, 4.09%, 38.7%."""
    if pct is None:
        return None
    return f"{pct:.2f}%" if pct < 10 else f"{pct:.1f}%"


def _num(v):
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def alert_facts(snapshot: dict, alert_id: str = "") -> dict:
    """The shared prediction payload of every e-mail (plain + HTML) and SMS about one simulation.

    Every value is read from THIS simulation's snapshot - the block the PDF / Word report is generated from - so a
    previous simulation can never leak in and nothing is re-predicted here:
      ml_fire_risk_score_pct  snapshot.prediction.risk_score: prediction_service.predict() -> RiskScore.score(p)
                              (models/risk_score_real.json), stored as processed_grid.ml_risk_score and copied by
                              simulation_report._add_risk_score. 0-100; a ranking, not a probability.
      risk_level              snapshot.prediction.risk_score_category = risk_score.category(score)
      fire_probability_pct    snapshot.prediction.probability (raw model output, 0..1) x 100"""
    from src.ml_models.risk_score import valid_score
    at = alert_type_of(snapshot)
    loc = snapshot.get("location") or {}
    pred, res = snapshot.get("prediction") or {}, snapshot.get("results") or {}
    w = snapshot.get("weather") or {}
    score = valid_score(pred.get("risk_score"))
    firms = snapshot.get("firms") or {}
    return {
        "alert_type": at, "hypothetical": at == "SIMULATION ONLY",
        "simulation_mode": "WHAT-IF" if at == "SIMULATION ONLY" else "REAL",
        "label": MANUAL_HYPO_LABEL if at == "SIMULATION ONLY" else MANUAL_OBS_LABEL,
        "alert_id": alert_id or None, "report_id": snapshot.get("report_id"),
        "location": loc.get("name") or None, "latitude": _num(loc.get("lat")), "longitude": _num(loc.get("lon")),
        "region": snapshot.get("region"), "when": ist(snapshot.get("generated_utc")),
        "ml_fire_risk_score_pct": score,
        "risk_level": (pred.get("risk_score_category") or None) if score is not None else None,
        "fire_probability_pct": percent_from_fraction(pred.get("probability")),
        "temperature_c": _num(w.get("temp_c")), "humidity_pct": _num(w.get("humidity_pct")),
        "wind_speed_ms": _num(w.get("wind_speed_ms")), "wind_from_deg": _num(w.get("wind_from_deg")),
        "rain": (pred.get("scenario_inputs") or {}).get("rain") or None, "weather_label": w.get("label"),
        "simulation_duration_min": _num(res.get("duration_min")), "duration_label": res.get("duration_label"),
        "burned_area_ha": _num(res.get("burned_ha")), "burning_ha": _num(res.get("burning_ha")),
        "fire_front_distance_m": _num(res.get("front_distance_m")), "spread_towards": res.get("spread_towards"),
        "firms_valid": int(firms.get("n_valid") or 0),
    }


def score_text(f: dict) -> Optional[str]:
    """'42% — LOW' - the report's ML Fire Risk Score and its category, printed identically everywhere."""
    from src.ml_models.risk_score import format_score
    s = format_score(f["ml_fire_risk_score_pct"])
    return None if s is None else f"{s} — {f['risk_level'] or 'uncategorised'}"


def score_lines(f: dict) -> List[tuple]:
    """(label, value) rows: ML Fire Risk Score first, the separate fire-occurrence probability below."""
    rows = []
    st = score_text(f)
    if st:
        rows.append(("ML Fire Risk Score", f"{st} (ranks above {f['ml_fire_risk_score_pct']:.0f}% of historical "
                                           "fire-season days; a relative ranking, not a probability)"))
    if f["fire_probability_pct"] is not None:
        rows.append(("Fire-occurrence probability", f"{fmt_prob(f['fire_probability_pct'])} (model estimate of a "
                                                    "satellite-detected fire in the ~11 km grid zone that day)"))
    return rows or [("Model risk", "not available")]


def weather_text(f: dict) -> str:
    bits = []
    if f["temperature_c"] is not None:
        bits.append(f"{f['temperature_c']:g} °C")
    if f["humidity_pct"] is not None:
        bits.append(f"{f['humidity_pct']:g} % RH")
    if f["wind_speed_ms"] is not None:
        bits.append(f"wind {f['wind_speed_ms']:g} m/s"
                    + (f" from {f['wind_from_deg']:.0f}°" if f["wind_from_deg"] is not None else ""))
    if f["rain"]:
        bits.append(f"rain {f['rain']}")
    return ", ".join(bits) or "not available"


def build_message(snapshot: dict, alert_id: str, sms: bool = False) -> Dict[str, str]:
    """Subject + plain body (+ HTML for email) from the exact simulation snapshot (no new data fetched)."""
    at = alert_type_of(snapshot)
    loc = snapshot.get("location") or {}
    risk = snapshot.get("risk") or {}
    res = snapshot.get("results") or {}
    when = ist(snapshot.get("generated_utc"))
    pred = snapshot.get("prediction") or {}
    risk_txt = (f"{risk.get('category')} (risk index {risk.get('score_pct')}% model-predicted"
                + (f", fire probability {pred['probability'] * 100:.2f}%" if pred.get("probability") is not None else "")
                + ")" if risk.get("score_pct") is not None else "not available")
    head = SIM_ONLY if at == "SIMULATION ONLY" else "OBSERVATION-BASED ALERT — " + OBS_NOTE
    icon = "🔥" if at == "SIMULATION ONLY" else "🛰️"
    subject = (f"{icon} [{at}] Forest fire {('scenario' if at == 'SIMULATION ONLY' else 'alert')} - "
               f"{loc.get('name', '-')} - {alert_id}")
    if sms:
        cat = risk.get("category") or "n/a"
        rp = f"{risk.get('score_pct')}%" if risk.get("score_pct") is not None else "n/a"
        body = (f"⚠ {at}: {loc.get('name', '-')} ({loc.get('lat', 0):.3f},{loc.get('lon', 0):.3f}) "
                + ("WHAT-IF scenario, NOT a confirmed fire. " if at == "SIMULATION ONLY" else
                   "Satellite hotspot(s), not ground-verified. ")
                + f"Risk {cat} {rp} (model). Sim {res.get('duration_label', '-')}: {res.get('burned_ha', 0):.1f} ha, "
                  f"front {res.get('front_distance_m', 0) or 0:.0f} m. {when}. Verify officially. Ref {alert_id}")
        return {"subject": subject, "body": body}
    lines = [head, "", f"System: {SYSTEM_ID}", f"Alert ID: {alert_id}", f"Report ID: {snapshot.get('report_id')}",
             f"Simulation time stamp: {when}", f"Alert type: {at}",
             f"Region: {loc.get('name', '-')} ({snapshot.get('region', '-')})",
             f"Coordinates: {loc.get('lat', 0):.5f} N, {loc.get('lon', 0):.5f} E",
             ]
    facts = alert_facts(snapshot, alert_id)
    lines += ["Model predictions (not observations):"] + [f"  - {k}: {v}" for k, v in score_lines(facts)] + [""]
    if at == "OBSERVATION-BASED":
        f = snapshot.get("firms") or {}
        lines += [f"Evidence: {f.get('n_valid', 0)} valid NASA FIRMS detection(s) in the simulation area "
                  f"(latest acquisition {ist(f.get('latest_acq_utc'))}, fetched {ist(f.get('fetched_utc'))})."]
    else:
        lines += ["Evidence: hypothetical ignition chosen by the user - no observed fire."]
    w = snapshot.get("weather") or {}
    lines += [f"Weather used ({w.get('label', '-')}): {weather_text(facts)}.", "",
              f"SIMULATED spread over {res.get('duration_label', '-')}: {res.get('burned_ha', 0):.2f} ha burned, "
              f"{res.get('burning_ha', 0):.2f} ha burning at the end, front distance "
              f"{res.get('front_distance_m', 0):.0f} m, spread towards {res.get('spread_towards', '-')}.",
              "The simulation supports situational awareness; it is not an official forecast.", "",
              "Advisory:"] + [f"  - {a}" for a in ADVICE]
    url = _dashboard_url()
    lines += ["", "Full report: available to authorised users in the Forest Fire Digital Twin dashboard "
              f"(Spread Simulation, report {snapshot.get('report_id')})." + (f" {url}" if url else "")]
    body = "\n".join(lines)
    return {"subject": subject, "body": body, "html": _html_email(snapshot, at, alert_id, when, facts, res, url)}


def _dashboard_url() -> str:
    from src.notifications import config as cfg
    u = cfg.get("DASHBOARD_URL").strip()
    return u if u.startswith(("http://", "https://")) else ""


RISK_COLOURS = {"LOW": "#2e7d32", "MODERATE": "#b7791f", "HIGH": "#d9480f", "VERY HIGH": "#c2410c", "EXTREME": "#b3261e"}


def _html_email(snapshot, at, alert_id, when, facts, res, url) -> str:
    from html import escape as e
    loc = snapshot.get("location") or {}
    cat = facts["risk_level"]
    col = RISK_COLOURS.get(cat, "#5b6673")
    head = (SIM_ONLY if at == "SIMULATION ONLY" else "OBSERVATION-BASED ALERT — " + OBS_NOTE)
    badge = f"ML FIRE RISK SCORE: {score_text(facts)}" if score_text(facts) else "ML FIRE RISK SCORE: NOT AVAILABLE"
    icons = {"ML Fire Risk Score": "📊", "Fire-occurrence probability": "🎯", "Model risk": "📊"}
    rows = ([("📍 Location", f"{loc.get('name', '-')} · {loc.get('lat', 0):.5f} N, {loc.get('lon', 0):.5f} E"),
             ("🕒 Simulation time", when)]
            + [(f"{icons.get(k, '📊')} {k}", v) for k, v in score_lines(facts)]
            + [("🌦️ Weather used", weather_text(facts)),
               ("🔥 Simulated spread", f"{res.get('burned_ha', 0):.2f} ha burned, {res.get('burning_ha', 0):.2f} ha burning, "
                                      f"front {res.get('front_distance_m', 0) or 0:.0f} m towards {res.get('spread_towards', '-')} "
                                      f"over {res.get('duration_label', '-')}"),
               ("📄 Report", f"{snapshot.get('report_id')} · alert {alert_id}")])
    trs = "".join(f"<tr><td style='padding:6px 10px;color:#5b6673;white-space:nowrap'>{e(k)}</td>"
                  f"<td style='padding:6px 10px;color:#13233a'>{e(v)}</td></tr>" for k, v in rows)
    btn = (f"<p><a href='{e(url)}' style='background:#e8710a;color:#fff;padding:9px 16px;border-radius:6px;"
           f"text-decoration:none;font-weight:600'>Open the dashboard</a></p>" if url else "")
    adv = "".join(f"<li>{e(a)}</li>" for a in ADVICE)
    return (f"<!doctype html><html><body style='margin:0;background:#eef2f6;font-family:Segoe UI,Arial,sans-serif'>"
            f"<table role='presentation' width='100%' cellpadding='0' cellspacing='0'><tr><td align='center' style='padding:18px'>"
            f"<table role='presentation' width='620' cellpadding='0' cellspacing='0' style='background:#fff;border-radius:10px;overflow:hidden'>"
            f"<tr><td style='background:#13233a;padding:16px 20px;border-bottom:4px solid #e8710a'>"
            f"<div style='color:#f2a33a;font-size:11px;font-weight:700;letter-spacing:.06em'>FOREST FIRE DIGITAL TWIN</div>"
            f"<div style='color:#fff;font-size:19px;font-weight:700'>⚠️ {e(at)}</div></td></tr>"
            f"<tr><td style='padding:14px 20px'><div style='border-left:4px solid {col};background:#f7f9fb;padding:10px 12px;"
            f"color:#13233a'><b>{e(head)}</b></div>"
            f"<p style='margin:12px 0 4px'><span style='background:{col};color:#fff;padding:3px 10px;border-radius:12px;"
            f"font-size:12px;font-weight:700'>{e(badge)}</span></p>"
            f"<table role='presentation' width='100%' style='border-collapse:collapse;font-size:13px'>{trs}</table>"
            f"<p style='font-size:13px;color:#13233a;margin:12px 0 4px'><b>Advisory</b></p><ul style='font-size:13px;color:#13233a'>{adv}</ul>"
            f"{btn}<p style='font-size:11px;color:#5b6673'>The simulation supports situational awareness; it is not an "
            f"official forecast. {e(SYSTEM_ID)}</p></td></tr></table></td></tr></table></body></html>")


def eligible_channels(recipient: dict, alert_type: str) -> List[str]:
    """Channels this recipient may receive: enabled and category permitted; e-mail only when VERIFIED and preferred;
    SMS when 'SMS alerts' is enabled and the stored number is a valid international number (mobile numbers are
    not verified - SMS verification was removed; phone_verified is ignored, never set)."""
    if not recipient.get("enabled") or alert_type not in (recipient.get("categories") or []):
        return []
    out = []
    prefs = recipient.get("prefs") or {}
    if recipient.get("email") and recipient.get("email_verified") and prefs.get("email", True):
        out.append("email")
    from src.notifications.config import valid_phone
    if recipient.get("phone") and valid_phone(recipient["phone"]) and prefs.get("sms", False):
        out.append("sms")
    return out


def new_alert_id() -> str:
    return "ALERT-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2).upper()


def send_alert(snapshot: dict, selections: Sequence[tuple], sender: str, sender_role: str, confirmed: bool,
               email_provider=None, sms_provider=None, category: Optional[str] = None) -> dict:
    """selections: [(recipient_id, channel), ...] chosen by the sender.
    Returns a truthful summary {alert_id, counts by status, results[]}."""
    if sender_role != "admin":
        raise PermissionError("Only authorised administrators can send emergency alerts.")
    if not confirmed:
        raise PermissionError("Emergency alert not sent: explicit final confirmation is required.")
    if not snapshot.get("report_id"):
        raise ValueError("No completed simulation snapshot.")
    at = alert_type_of(snapshot)
    alert_id = new_alert_id()
    email_p = email_provider or EmailProvider()
    sms_p = sms_provider or _prov.sms_provider()
    full = build_message(snapshot, alert_id)
    short = build_message(snapshot, alert_id, sms=True)
    loc = (snapshot.get("location") or {}).get("name", "-")
    store.record_alert(alert_id, sender, snapshot["report_id"], at, category or at,
                       loc, full["subject"], full["body"], {"state": "sending"})
    results = []
    recips = {r["id"]: r for r in store.list_recipients()}
    for rid, ch in selections:
        r = recips.get(int(rid))
        if r is None or ch not in eligible_channels(r, at):
            res = SendResult("failed", "-", error="recipient / channel not eligible (disabled, unverified or "
                                                  "category not permitted)")
        elif store.recent_delivery(snapshot["report_id"], r["id"], ch):
            res = SendResult("duplicate", "-", error="already sent for this simulation in the last 30 minutes")
        elif ch == "email":
            res = email_p.send(r["email"], full["subject"], full["body"], html=full.get("html"))
        else:
            res = sms_p.send(r["phone"], short["body"])
        store.record_delivery(alert_id, snapshot["report_id"], int(rid), ch, res.provider, res.status,
                              res.message_id, res.error)
        results.append({"recipient_id": int(rid), "name": (r or {}).get("name", "?"), "channel": ch,
                        "to": store.mask((r or {}).get("email") if ch == "email" else (r or {}).get("phone")),
                        "status": res.status, "provider": res.provider, "error": res.error})
    counts: Dict[str, int] = {}
    for x in results:
        counts[x["status"]] = counts.get(x["status"], 0) + 1
    summary = {"state": "done", "counts": counts, "n": len(results)}
    store.update_alert_summary(alert_id, summary)
    # in-app notification for authenticated dashboard users (receiver-side alarm)
    store.add_inapp(alert_id, full["subject"], short["body"], at)
    from src.storage import database as db
    db.log_activity("emergency_alert", f"{alert_id} ({at}) report {snapshot['report_id']}: "
                    + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), actor=sender)
    return {"alert_id": alert_id, "alert_type": at, "counts": counts, "results": results,
            "body_sha256": hashlib.sha256(full["body"].encode()).hexdigest()}


_TEST_GUARD: Dict[tuple, float] = {}
TEST_COOLDOWN_S = 30


def send_test(channel: str, to: str, sender: str, sender_role: str, email_provider=None, sms_provider=None) -> dict:
    """An explicitly requested TEST ALERT to one address / number (admin only). Labelled as a test
    everywhere; repeated clicks within TEST_COOLDOWN_S are not sent again."""
    import time
    from src.notifications import config as cfg
    if sender_role != "admin":
        raise PermissionError("Only authorised administrators can send test alerts.")
    to = (to or "").strip()
    if channel == "email" and not cfg.valid_email(to):
        return {"status": "failed", "error": "enter a valid email address", "alert_id": None}
    if channel == "sms" and not cfg.valid_phone(to):
        return {"status": "failed", "error": "enter the number in E.164 format, e.g. +919812345678", "alert_id": None}
    key = (channel, to)
    if time.time() - _TEST_GUARD.get(key, 0) < TEST_COOLDOWN_S:
        return {"status": "duplicate", "error": f"a test was sent to this {channel} less than {TEST_COOLDOWN_S} s ago",
                "alert_id": None}
    _TEST_GUARD[key] = time.time()
    alert_id = "TEST-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2).upper()
    when = ist(datetime.now(timezone.utc).isoformat())
    if channel == "email":
        subj = f"✅ [TEST ALERT] Forest Fire Digital Twin - email check {alert_id}"
        body = (f"TEST ALERT - this message only checks the email configuration. No fire, prediction or simulation is "
                f"involved.\n\nSystem: {SYSTEM_ID}\nTest ID: {alert_id}\nSent: {when}\nRequested by: {sender}")
        html = (f"<div style='font-family:Segoe UI,Arial,sans-serif;max-width:560px;border:1px solid #d5dbe3;"
                f"border-top:4px solid #2f6db3;padding:14px 18px'><div style='font-weight:700;color:#13233a'>✅ TEST ALERT</div>"
                f"<p style='color:#13233a'>This message only checks the email configuration of the Forest Fire Digital Twin. "
                f"No fire, prediction or simulation is involved.</p><p style='font-size:12px;color:#5b6673'>Test ID {alert_id} · "
                f"{when} · requested by {sender}</p></div>")
        res = (email_provider or EmailProvider()).send(to, subj, body, html=html)
    else:
        body = f"TEST ALERT (no fire): Forest Fire Digital Twin SMS check. {when}. Ref {alert_id}"
        res = (sms_provider or _prov.sms_provider()).send(to, body)
    store.record_alert(alert_id, sender, None, "TEST", "TEST", "-", f"Test {channel}", body if channel == "sms" else subj,
                       {"state": "done", "counts": {res.status: 1}, "n": 1})
    store.record_delivery(alert_id, "TEST", 0, channel, res.provider, res.status, res.message_id, res.error)
    from src.storage import database as db
    db.log_activity("test_alert", f"{alert_id} {channel} to {store.mask(to)}: {res.status}", actor=sender)
    return {"status": res.status, "error": res.error, "alert_id": alert_id, "message_id": res.message_id}


VERIFY_EMAIL = ("Forest Fire Digital Twin: your e-mail verification code is {code} (valid {ttl} minutes). "
                "This is not a fire alert. If you did not expect it, ignore this message.")


def verification_readiness(channel: str = "email") -> dict:
    """Can a verification e-mail be sent right now? (Same SMTP settings as the e-mail alerts.)"""
    from src.notifications import config as cfg
    if channel != "email":
        return {"ok": False, "reason": SMS_VERIFY_REMOVED}
    if cfg.mock_mode():
        return {"ok": False, "reason": "Live sending is disabled (ALERT_PROVIDER_MODE=mock): no code would reach the "
                                       "recipient. Set ALERT_PROVIDER_MODE=live in .env (or on this page) and restart."}
    probs = cfg.smtp_problems()
    return {"ok": not probs, "reason": ("E-mail (SMTP) not configured: " + "; ".join(probs)) if probs else ""}


SMS_VERIFY_REMOVED = "SMS verification is not available - only e-mail addresses are verified."


def send_verification(rid: int, channel: str, sender: str, sender_role: str, email_provider=None,
                      now=None) -> dict:
    """Send an e-mail verification code (admin only) through the configured SMTP provider. Returns {"status",
    "message", "wait_s"}; the code itself is never returned, logged or shown. Only e-mail is verified - any other
    channel is refused (SMS verification was removed)."""
    from src.notifications import config as cfg
    if sender_role != "admin":
        raise PermissionError("Only authorised administrators can verify recipients.")
    if channel != "email":
        raise ValueError(SMS_VERIFY_REMOVED)
    r = store.get_recipient(rid)
    target = (r or {}).get("email")
    if not target:
        return {"status": "failed", "message": "This recipient has no e-mail address."}
    if not cfg.valid_email(target):
        return {"status": "failed", "message": "The stored e-mail address is not valid."}
    ready = verification_readiness("email")
    if not ready["ok"]:
        return {"status": "not_configured", "message": ready["reason"]}
    wait = store.verification_wait(rid, "email", now=now)
    if wait:
        return {"status": "rate_limited", "wait_s": wait,
                "message": f"A code was sent recently - wait {wait} s before requesting another."}
    code = store.issue_verification(rid, "email", target, now=now)
    res = (email_provider or EmailProvider()).send(target, "Forest Fire Digital Twin - e-mail verification code",
                                                   VERIFY_EMAIL.format(code=code, ttl=store.VERIFY_TTL_MIN))
    if res.status in ("failed", "not_configured", "mock"):
        store.cancel_verification(rid, "email")          # nothing usable reached the recipient
    store.record_delivery("VERIFY", "-", int(rid), "email", res.provider, res.status, res.message_id,
                          cfg.redact(res.error) if res.error else None)
    from src.storage import database as db
    db.log_activity("alert_recipient", f"Recipient #{rid} email verification code: {res.status} "
                    f"(to {store.mask(target)})", actor=sender)
    if res.status == "accepted":
        msg = (f"Verification e-mail accepted by the mail server - delivery not confirmed. Enter the code below when "
               f"it arrives (valid {store.VERIFY_TTL_MIN} min).")
    else:
        msg = f"Verification e-mail NOT sent - {res.error or res.status}."
    return {"status": res.status, "message": cfg.redact(msg)}


def refresh_sms_statuses(sms_provider=None, limit: int = 50) -> int:
    """Asks the SMS provider for the current status of recent, not-final SMS deliveries and stores
    provider-confirmed changes (the only way a delivery becomes 'delivered'). Returns #updated."""
    p = sms_provider or _prov.sms_provider()
    n = 0
    for d in store.recent_deliveries(limit, channel="sms"):
        if d["status"] not in ("queued", "accepted", "sent", "unknown") or not d.get("provider_message_id"):
            continue
        r = p.status(d["provider_message_id"])
        if r.status not in ("mock", "not_configured") and r.status != d["status"]:
            store.update_delivery_status(d["id"], r.status, r.error)
            n += 1
    return n


def describe_status(status: str) -> str:
    return {"connected": "connected and authenticated (nothing sent)",
            "delivered": "Delivery confirmed by provider",
            "accepted": "accepted (email: by the SMTP server; SMS: by the gateway) - delivery not confirmed",
            "queued": "queued by the SMS provider / handed to the phone (delivery not yet confirmed)",
            "sent": "sent to the mobile network (handset delivery not confirmed)",
            "mock": "MOCK - nothing was sent (ALERT_PROVIDER_MODE=mock)",
            "not_configured": "NOT SENT - provider not configured",
            "duplicate": "NOT SENT - duplicate within 30 minutes",
            "failed": "FAILED", "unknown": "status unknown"}.get(status, status)


# ── manual simulation SMS ("📱 Send SMS Alert" after a completed simulation) ──────────────────
# Explicit, confirmed, REAL SMS about one simulation report to the configured ALERT_SMS_RECIPIENTS.
# Independent of the automatic-alert threshold / cooldown (it never reads or writes auto_alert_state);
# protected by its own per scenario x recipient cooldown (Admin setting, default 5 min).

MANUAL_HYPO_LABEL = "SIMULATION TEST ALERT - HYPOTHETICAL SCENARIO - NOT A CONFIRMED FIRE"
MANUAL_OBS_LABEL = "SIMULATION ALERT - SPREAD SIMULATED FROM SATELLITE HOTSPOTS - NOT GROUND-VERIFIED"
MANUAL_ALERT_TYPE = "MANUAL SIMULATION"


def manual_sms_text(snapshot: dict, alert_id: str = "") -> str:
    """Mobile-friendly SMS from alert_facts (the report's values). Missing fields are left out, never invented."""
    from src.ml_models.risk_score import format_score
    f = alert_facts(snapshot, alert_id)
    lines = ["🚨🌲 FOREST FIRE DIGITAL TWIN"]
    if f["hypothetical"]:
        lines.append("⚠️ SIMULATION ONLY — HYPOTHETICAL SCENARIO — NOT A CONFIRMED FIRE")
    else:
        lines.append("⚠️ SIMULATED SPREAD FROM SATELLITE HOTSPOTS — NOT GROUND-VERIFIED")
        if f["firms_valid"]:
            lines.append(f"🛰️ NASA FIRMS: {f['firms_valid']} hotspot(s) observed")
    lines.append(f"📍 {f['location'] or 'Location not available'}")
    sc = format_score(f["ml_fire_risk_score_pct"])
    lines.append(f"📊 ML Fire Risk Score: {sc} ({f['risk_level'] or 'uncategorised'})" if sc
                 else "📊 ML Fire Risk Score: not available")
    if f["fire_probability_pct"] is not None:
        lines.append(f"🔥 Fire probability: {fmt_prob(f['fire_probability_pct'])}")
    wx = []
    if f["temperature_c"] is not None:
        wx.append(f"Temp: {f['temperature_c']:.1f}°C")
    if f["humidity_pct"] is not None:
        wx.append(f"Humidity: {f['humidity_pct']:.0f}%")
    if wx:
        lines.append("🌡️ " + " | ".join(wx))
    if f["wind_speed_ms"] is not None:
        lines.append(f"💨 Wind: {f['wind_speed_ms']:.1f} m/s"
                     + (f", {f['wind_from_deg']:.0f}°" if f["wind_from_deg"] is not None else ""))
    if f["burned_area_ha"] is not None:
        lines.append(f"🔥 Simulated burned area: {f['burned_area_ha']:.2f} ha")
    if f["fire_front_distance_m"] is not None:
        lines.append(f"➡️ Fire front: {f['fire_front_distance_m']:.0f} m"
                     + (f" towards {f['spread_towards']}" if f["spread_towards"] else ""))
    lines.append(f"🆔 Report: {f['report_id'] or '-'}")
    return "\n".join(lines)


GSM7 = set("@£$¥èéùìòÇ\nØø\rÅåΔ_ΦΓΛΩΠΨΣΘΞÆæßÉ !\"#¤%&'()*+,-./0123456789:;<=>?¡ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÑÜ§"
           "¿abcdefghijklmnopqrstuvwxyzäöñüà^{}\\[~]|€")


def sms_segments(text: str) -> dict:
    """How the phone will split the SMS: GSM-7 (160 / 153 per part) or Unicode UCS-2 (70 / 67 per part, emojis
    count as 2 units). The phone joins the parts into one message; carriers may count each part separately."""
    gsm = all(ch in GSM7 for ch in text)
    units = len(text) if gsm else sum(2 if ord(ch) > 0xFFFF else 1 for ch in text)
    single, multi = (160, 153) if gsm else (70, 67)
    parts = 1 if units <= single else -(-units // multi)
    return {"encoding": "GSM-7" if gsm else "UCS-2 (Unicode)", "units": units, "segments": parts}


def manual_sms_recipients(snapshot: Optional[dict] = None) -> List[str]:
    """ALERT_SMS_RECIPIENTS plus registered recipients whose mobile is verified, SMS-enabled and permitted for this
    alert type. Any number of numbers; all are sent from the one paired gateway phone."""
    from src.notifications import config as cfg
    out = [p for p in cfg.sms_recipients() if cfg.valid_phone(p)]
    if snapshot is not None:
        at = alert_type_of(snapshot)
        try:
            for r in store.list_recipients(enabled_only=True):
                p = (r.get("phone") or "").strip()
                if "sms" in eligible_channels(r, at) and cfg.valid_phone(p) and p not in out:
                    out.append(p)
        except Exception:
            pass
    return out


def manual_sms_readiness(snapshot: Optional[dict] = None) -> dict:
    """Can a REAL manual SMS be sent right now? {"ok", "reason", "provider", "recipients" (valid E.164)}."""
    from src.notifications import config as cfg
    recips = manual_sms_recipients(snapshot)
    bad = [p for p in cfg.sms_recipients() if not cfg.valid_phone(p)]
    name = cfg.sms_provider_name()
    if cfg.mock_mode():
        return {"ok": False, "provider": name, "recipients": recips,
                "reason": "Live sending is disabled (ALERT_PROVIDER_MODE=mock): nothing would be sent. Switch the "
                          "provider mode to live on Admin → Notifications."}
    probs = cfg.textbee_problems() if name == "textbee" else cfg.sms_problems()
    if probs:
        return {"ok": False, "provider": name, "recipients": recips,
                "reason": f"SMS gateway not configured ({name}): " + "; ".join(probs)}
    if not recips:
        return {"ok": False, "provider": name, "recipients": [],
                "reason": "No SMS recipient configured: set ALERT_SMS_RECIPIENTS (international format, e.g. "
                          "+919812345678) or verify a recipient's mobile on Admin → Emergency alerts" + (f" - {len(bad)} entry(ies) are not valid E.164 numbers" if bad else "")}
    return {"ok": True, "provider": name, "recipients": recips, "reason": ""}


def send_manual_sim_sms(snapshot: dict, sender: str, sender_role: str, confirmed: bool, sms_provider=None,
                        now=None) -> dict:
    """Send the manual simulation SMS after explicit confirmation. Returns
    {"alert_id", "results": [{"to", "status", "error", "detail", "next_eligible_utc"}], "blocked": reason|None}."""
    from src.notifications import auto_alerts as aa
    if sender_role != "admin":
        raise PermissionError("Only authorised administrators can send SMS alerts.")
    if not confirmed:
        raise PermissionError("SMS not sent: explicit confirmation is required.")
    if not snapshot.get("report_id"):
        raise ValueError("No completed simulation snapshot.")
    ready = manual_sms_readiness(snapshot)
    if not ready["ok"]:
        return {"alert_id": None, "results": [], "blocked": ready["reason"]}
    alert_id = "MSIM-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2).upper()
    body = manual_sms_text(snapshot, alert_id)
    provider = sms_provider or _prov.sms_provider()
    results, statuses = [], []
    for phone in ready["recipients"]:
        ok, token, st = aa.manual_claim(snapshot["report_id"], phone, now=now)
        if not ok:
            results.append({"to": store.mask(phone), "status": "cooldown", "error": None, "detail": None,
                            "next_eligible_utc": st["next_eligible_utc"], "seconds_left": st["seconds_left"]})
            continue
        res = provider.send(phone, body)
        definite_fail = res.status in ("failed", "not_configured")
        aa.manual_finish(snapshot["report_id"], phone, token, alert_id, release=definite_fail)
        store.record_delivery(alert_id, snapshot["report_id"], 0, "sms", res.provider, res.status, res.message_id, res.error)
        nxt = None if definite_fail else aa.manual_status(snapshot["report_id"], phone, now=now)["next_eligible_utc"]
        results.append({"to": store.mask(phone), "status": res.status, "error": res.error, "detail": res.detail,
                        "next_eligible_utc": nxt, "message_id": res.message_id})
        statuses.append(res.status)
    if statuses:
        store.record_alert(alert_id, sender, snapshot["report_id"], MANUAL_ALERT_TYPE, alert_type_of(snapshot),
                           (snapshot.get("location") or {}).get("name", "-"), MANUAL_ALERT_TYPE + " SMS", body,
                           {"state": "done", "sms": statuses, "n": len(statuses)})
        from src.storage import database as db
        db.log_activity("manual_sim_sms", f"{alert_id} report {snapshot['report_id']} ({alert_type_of(snapshot)}): "
                        + ", ".join(statuses), actor=sender)
    return {"alert_id": alert_id if statuses else None, "results": results, "blocked": None, "body": body}


# ── manual simulation e-mail ("✉️ Email Alert" in the Send Emergency Alert menu) ─────────────
# Same rules as the manual SMS: admin only, explicit confirmation, simulation-only labelling, its own
# per scenario x recipient guard (manual cooldown) and never any effect on automatic-alert state.

def manual_email_recipients(snapshot: dict) -> List[str]:
    """Configured ALERT_RECIPIENT_EMAILS plus registered recipients whose e-mail is enabled, verified and
    permitted for this alert type (Admin → Emergency alerts). Valid addresses only, de-duplicated."""
    from src.notifications import config as cfg
    at = alert_type_of(snapshot)
    out: List[str] = []
    for e in cfg.smtp_settings()["recipients"]:
        if cfg.valid_email(e) and e.lower() not in {x.lower() for x in out}:
            out.append(e)
    try:
        for r in store.list_recipients(enabled_only=True):
            e = (r.get("email") or "").strip()
            if "email" in eligible_channels(r, at) and cfg.valid_email(e) and e.lower() not in {x.lower() for x in out}:
                out.append(e)
    except Exception:                                   # recipient table unavailable: configured list only
        pass
    return out


def manual_email_readiness(snapshot: dict) -> dict:
    """Can a REAL manual e-mail be sent right now? {"ok", "reason", "provider", "recipients"}."""
    from src.notifications import config as cfg
    recips = manual_email_recipients(snapshot)
    if cfg.mock_mode():
        return {"ok": False, "provider": "smtp", "recipients": recips,
                "reason": "Live sending is disabled (ALERT_PROVIDER_MODE=mock): nothing would be sent. Switch the "
                          "provider mode to live on Admin → Notifications."}
    probs = cfg.smtp_problems()
    if probs:
        return {"ok": False, "provider": "smtp", "recipients": recips,
                "reason": "E-mail (SMTP) not configured: " + "; ".join(probs)}
    if not recips:
        return {"ok": False, "provider": "smtp", "recipients": [],
                "reason": "No e-mail recipient configured: set ALERT_RECIPIENT_EMAILS or register and verify a "
                          "recipient on Admin → Emergency alerts."}
    return {"ok": True, "provider": "smtp", "recipients": recips, "reason": ""}


def manual_email_message(snapshot: dict, alert_id: str) -> Dict[str, str]:
    """The emergency e-mail (build_message) with the manual simulation label on subject, body and HTML."""
    at = alert_type_of(snapshot)
    label = MANUAL_HYPO_LABEL if at == "SIMULATION ONLY" else MANUAL_OBS_LABEL
    m = build_message(snapshot, alert_id)
    subject = f"[{label}] {(snapshot.get('location') or {}).get('name', '-')} - {alert_id}"
    body = label + "\n\n" + m["body"]
    banner = (f"<div style='background:#fff4e5;border:1px solid #e8710a;color:#7a3b00;padding:9px 12px;"
              f"font-weight:700;font-size:13px;margin-bottom:10px'>{label}</div>")
    page = m.get("html") or ""
    page = page.replace("<tr><td style='padding:14px 20px'>", "<tr><td style='padding:14px 20px'>" + banner, 1)
    return {"subject": subject, "body": body, "html": page}


def send_manual_sim_email(snapshot: dict, sender: str, sender_role: str, confirmed: bool, email_provider=None,
                          now=None) -> dict:
    """Send the manual simulation e-mail after explicit confirmation (same contract as send_manual_sim_sms)."""
    from src.notifications import auto_alerts as aa
    if sender_role != "admin":
        raise PermissionError("Only authorised administrators can send e-mail alerts.")
    if not confirmed:
        raise PermissionError("E-mail not sent: explicit confirmation is required.")
    if not snapshot.get("report_id"):
        raise ValueError("No completed simulation snapshot.")
    ready = manual_email_readiness(snapshot)
    if not ready["ok"]:
        return {"alert_id": None, "results": [], "blocked": ready["reason"]}
    alert_id = "MSIM-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2).upper()
    msg = manual_email_message(snapshot, alert_id)
    provider = email_provider or EmailProvider()
    results, statuses = [], []
    for addr in ready["recipients"]:
        guard = manual_email_guard(addr)          # never collides with an SMS recipient guard
        ok, token, st = aa.manual_claim(snapshot["report_id"], guard, now=now)
        if not ok:
            results.append({"to": store.mask(addr), "status": "cooldown", "error": None, "detail": None,
                            "next_eligible_utc": st["next_eligible_utc"], "seconds_left": st["seconds_left"]})
            continue
        res = provider.send(addr, msg["subject"], msg["body"], html=msg.get("html"))
        definite_fail = res.status in ("failed", "not_configured")
        aa.manual_finish(snapshot["report_id"], guard, token, alert_id, release=definite_fail)
        store.record_delivery(alert_id, snapshot["report_id"], 0, "email", res.provider, res.status, res.message_id,
                              res.error)
        results.append({"to": store.mask(addr), "status": res.status, "error": res.error, "detail": None,
                        "next_eligible_utc": None if definite_fail else
                        aa.manual_status(snapshot["report_id"], guard, now=now)["next_eligible_utc"],
                        "message_id": res.message_id})
        statuses.append(res.status)
    if statuses:
        store.record_alert(alert_id, sender, snapshot["report_id"], MANUAL_ALERT_TYPE, alert_type_of(snapshot),
                           (snapshot.get("location") or {}).get("name", "-"), msg["subject"], msg["body"],
                           {"state": "done", "email": statuses, "n": len(statuses)})
        from src.storage import database as db
        db.log_activity("manual_sim_email", f"{alert_id} report {snapshot['report_id']} ({alert_type_of(snapshot)}): "
                        + ", ".join(statuses), actor=sender)
    return {"alert_id": alert_id if statuses else None, "results": results, "blocked": None,
            "subject": msg["subject"]}


def mask_phone(phone: str) -> str:
    """'+919812345678' -> '******5678' (last four digits only)."""
    digits = "".join(ch for ch in str(phone or "") if ch.isdigit())
    return "******" + digits[-4:] if len(digits) >= 4 else "******"


def manual_email_guard(addr: str) -> str:
    return "email:" + addr.strip().lower()
