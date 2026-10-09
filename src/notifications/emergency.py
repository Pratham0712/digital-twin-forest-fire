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
from src.notifications.providers import EmailProvider, SendResult, SmsProvider

SIM_ONLY = "SIMULATION ONLY — This is a hypothetical forest-fire scenario, not confirmation of an active fire."
OBS_NOTE = ("Based on NASA FIRMS satellite hotspot detections. Satellite hotspots are observations, not "
            "independently ground-verified fires.")
SYSTEM_ID = "Forest Fire Digital Twin (BMSCE) - Bandipur / Karnataka"
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


def build_message(snapshot: dict, alert_id: str, sms: bool = False) -> Dict[str, str]:
    """Subject + body from the exact simulation snapshot (no new data fetched)."""
    at = alert_type_of(snapshot)
    loc = snapshot.get("location") or {}
    risk = snapshot.get("risk") or {}
    res = snapshot.get("results") or {}
    when = ist(snapshot.get("generated_utc"))
    risk_txt = (f"{risk.get('category')} ({risk.get('score_pct')}% model-predicted)" if risk.get("score_pct") is not None
                else "not available")
    head = SIM_ONLY if at == "SIMULATION ONLY" else "OBSERVATION-BASED ALERT — " + OBS_NOTE
    subject = f"[{at}] Forest fire {('scenario' if at == 'SIMULATION ONLY' else 'alert')} - {loc.get('name', '-')} - {alert_id}"
    if sms:
        body = (f"{at}: {loc.get('name', '-')} ({loc.get('lat', 0):.4f}, {loc.get('lon', 0):.4f}). "
                + ("Hypothetical scenario, NOT a confirmed fire. " if at == "SIMULATION ONLY" else
                   "NASA FIRMS hotspot(s), not ground-verified. ")
                + f"Risk {risk_txt}. Sim {res.get('duration_label', '-')}: {res.get('burned_ha', 0):.1f} ha burned, "
                  f"front {res.get('front_distance_m', 0):.0f} m. Verify via official channels. ID {alert_id}")
        return {"subject": subject, "body": body}
    lines = [head, "", f"System: {SYSTEM_ID}", f"Alert ID: {alert_id}", f"Report ID: {snapshot.get('report_id')}",
             f"Simulation time stamp: {when}", f"Alert type: {at}",
             f"Region: {loc.get('name', '-')} ({snapshot.get('region', '-')})",
             f"Coordinates: {loc.get('lat', 0):.5f} N, {loc.get('lon', 0):.5f} E",
             f"Risk category (ML prediction, not an observation): {risk_txt}", ""]
    if at == "OBSERVATION-BASED":
        f = snapshot.get("firms") or {}
        lines += [f"Evidence: {f.get('n_valid', 0)} valid NASA FIRMS detection(s) in the simulation area "
                  f"(latest acquisition {ist(f.get('latest_acq_utc'))}, fetched {ist(f.get('fetched_utc'))})."]
    else:
        lines += ["Evidence: hypothetical ignition chosen by the user - no observed fire."]
    w = snapshot.get("weather") or {}
    lines += [f"Weather used ({w.get('label', '-')}): {w.get('temp_c', '-')} °C, {w.get('humidity_pct', '-')} % RH, "
              f"wind {w.get('wind_speed_ms', '-')} m/s from {w.get('wind_from_deg', '-')}°.", "",
              f"SIMULATED spread over {res.get('duration_label', '-')}: {res.get('burned_ha', 0):.2f} ha burned, "
              f"{res.get('burning_ha', 0):.2f} ha burning at the end, front distance "
              f"{res.get('front_distance_m', 0):.0f} m, spread towards {res.get('spread_towards', '-')}.",
              "The simulation supports situational awareness; it is not an official forecast.", "",
              "Advisory:"] + [f"  - {a}" for a in ADVICE]
    lines += ["", "Full report: available to authorised users in the Forest Fire Digital Twin dashboard "
              f"(Spread Simulation, report {snapshot.get('report_id')})."]
    return {"subject": subject, "body": "\n".join(lines)}


def eligible_channels(recipient: dict, alert_type: str) -> List[str]:
    """Channels this recipient may receive: enabled, verified, preferred and category permitted."""
    if not recipient.get("enabled") or alert_type not in (recipient.get("categories") or []):
        return []
    out = []
    prefs = recipient.get("prefs") or {}
    if recipient.get("email") and recipient.get("email_verified") and prefs.get("email", True):
        out.append("email")
    if recipient.get("phone") and recipient.get("phone_verified") and prefs.get("sms", False):
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
    sms_p = sms_provider or SmsProvider()
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
            res = email_p.send(r["email"], full["subject"], full["body"])
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


def describe_status(status: str) -> str:
    return {"accepted": "accepted by the email server (inbox delivery not confirmed)",
            "queued": "queued by the SMS provider (delivery not yet confirmed)",
            "sent": "sent to the carrier (handset delivery not confirmed)",
            "mock": "MOCK - nothing was sent (ALERT_PROVIDER_MODE=mock)",
            "not_configured": "NOT SENT - provider not configured",
            "duplicate": "NOT SENT - duplicate within 30 minutes",
            "failed": "FAILED", "unknown": "status unknown"}.get(status, status)
