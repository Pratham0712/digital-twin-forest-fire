"""
alerts_ui.py - simulation report downloads, the opt-in emergency-alert review /
confirmation dialog, recipient management, alert history and the in-app
receiver alarm. All of it requires a signed-in user; sending and recipient
management require the admin role.
"""
from __future__ import annotations

import json
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components

from src.notifications import alert_store as store
from src.notifications import emergency as em
from src.notifications.providers import provider_status


def _user() -> str:
    return st.session_state.get("auth_user", "")


def _role() -> str:
    return st.session_state.get("auth_role", "viewer")


# ── after a completed simulation ────────────────────────────────────────────

def render_result_actions(snapshot: Optional[dict], kp: str):
    if not snapshot:
        return
    st.markdown(
        f"<div class='info-box'><b>Simulation complete</b> · report <code>{snapshot['report_id']}</code> · "
        f"{'LIVE (observed ignition, simulated spread)' if snapshot['mode'] == 'LIVE' else snapshot['mode'] + ' (hypothetical - SIMULATED)'}"
        f" · {snapshot['results']['burned_ha']:.1f} ha burned in {snapshot['results']['duration_label']}, front "
        f"{snapshot['results']['front_distance_m'] or 0:.0f} m towards the {snapshot['results']['spread_towards']}</div>",
        unsafe_allow_html=True)
    c1, c2, _ = st.columns([1.4, 1.2, 2])
    if c1.button("📄 Download / Print Simulation Report", key=f"{kp}_rep_btn", use_container_width=True):
        _report_dialog(snapshot)
    if c2.button("🚨 Send Emergency Alert", key=f"{kp}_alert_btn", use_container_width=True):
        _alert_dialog(snapshot)


@st.dialog("Simulation report", width="large")
def _report_dialog(snapshot: dict):
    from src.reports import simulation_report as rep
    st.caption(f"Report {snapshot['report_id']} - generated from the stored snapshot of this completed simulation "
               "(no data is fetched again). Labels: OBSERVED · PREDICTED · DERIVED · SCENARIO INPUT · SIMULATED.")
    c1, c2 = st.columns(2)
    try:
        pdf = rep.render_pdf(snapshot)
        c1.download_button("Download PDF", pdf, f"{snapshot['report_id']}.pdf", "application/pdf",
                           use_container_width=True, type="primary")
    except ImportError:
        c1.error("PDF needs the 'reportlab' package: pip install reportlab")
    try:
        docx = rep.render_docx(snapshot)
        c2.download_button("Download DOCX", docx, f"{snapshot['report_id']}.docx",
                           "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                           use_container_width=True)
    except ImportError:
        c2.error("DOCX needs the 'python-docx' package: pip install python-docx")
    st.markdown("**Print report** - preview below; use its *Print report* button (prints only the report).")
    components.html(rep.render_html(snapshot), height=520, scrolling=True)


@st.dialog("Send emergency alert", width="large")
def _alert_dialog(snapshot: dict):
    at = em.alert_type_of(snapshot)
    if _role() != "admin":
        st.error("Only authorised administrators can send emergency alerts. Ask an admin to review this simulation.")
        return
    ps = provider_status()
    loc, risk, res = snapshot["location"], snapshot["risk"], snapshot["results"]
    st.markdown(f"**Alert type:** `{at}`" + (" - hypothetical scenario, NOT a confirmed fire." if at == "SIMULATION ONLY"
                                             else " - NASA FIRMS hotspots (satellite observations, not ground-verified)."))
    st.markdown(f"**Region:** {loc['name']} ({snapshot.get('region') or '-'}) · {loc['lat']:.5f} N, {loc['lon']:.5f} E  \n"
                f"**Risk (ML prediction):** {risk.get('category') or '-'} {'' if risk.get('score_pct') is None else str(risk['score_pct']) + '%'}"
                f"  \n**Simulation:** {res['burned_ha']:.1f} ha burned in {res['duration_label']}, front "
                f"{res['front_distance_m'] or 0:.0f} m → {res['spread_towards']} · report {snapshot['report_id']} · "
                f"{em.ist(snapshot['generated_utc'])}")
    st.caption(f"Providers: mode **{ps['mode']}** · email: {ps['email']} · SMS: {ps['sms']}")
    alert_preview = em.build_message(snapshot, "ALERT-ID-ON-SEND")
    with st.expander("Message preview (email)"):
        st.text(alert_preview["subject"] + "\n\n" + alert_preview["body"])
    with st.expander("Message preview (SMS)"):
        st.text(em.build_message(snapshot, "ALERT-ID-ON-SEND", sms=True)["body"])
    recips = store.list_recipients(enabled_only=True)
    if not recips:
        st.warning("No enabled recipients are registered. Add and verify recipients on the Admin page.")
        return
    sel = []
    st.markdown("**Recipients** (only enabled, verified channels permitted for this alert type can be selected)")
    for r in recips:
        ch = em.eligible_channels(r, at)
        cols = st.columns([2.2, 1, 1])
        cols[0].markdown(f"{r['name']} · {r.get('designation') or ''} {r.get('organisation') or ''}".strip())
        for i, c in enumerate(("email", "sms")):
            target = store.mask(r.get("email") if c == "email" else r.get("phone"))
            if c in ch:
                if cols[1 + i].checkbox(f"{c.upper()} {target}", key=f"al_{r['id']}_{c}"):
                    sel.append((r["id"], c))
            else:
                cols[1 + i].caption(f"{c.upper()}: not available")
    st.divider()
    ok = st.checkbox("I have reviewed the message, the recipients and the alert type, and I confirm sending this "
                     "emergency alert now.", key="al_confirm")
    if st.button("Confirm and send", type="primary", disabled=not (ok and sel)):
        try:
            out = em.send_alert(snapshot, sel, _user(), _role(), confirmed=True)
        except PermissionError as exc:
            st.error(str(exc))
            return
        st.session_state["_last_alert_result"] = out
        st.success(f"Send attempt recorded as {out['alert_id']}.")
        for x in out["results"]:
            st.write(f"- {x['name']} · {x['channel']} {x['to']}: **{em.describe_status(x['status'])}**"
                     + (f" ({x['error']})" if x.get("error") else ""))
        st.caption("Delivery is only reported as confirmed when the provider confirms it; SMTP acceptance or an SMS "
                   "'queued' status is not proof of delivery.")


# ── Admin page: recipients and history ──────────────────────────────────────

def render_recipient_admin():
    if _role() != "admin":
        st.info("Recipient management is available to administrators only.")
        return
    ps = provider_status()
    st.caption(f"Providers: mode **{ps['mode']}** · email: {ps['email']} · SMS: {ps['sms']}. Credentials come from "
               ".env only (see .env.example); they are never shown here.")
    with st.form("add_recipient", clear_on_submit=True):
        st.markdown("**Register a recipient** (forest-department / monitoring personnel)")
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Name")
        org = c2.text_input("Organisation / department")
        des = c3.text_input("Designation")
        c4, c5 = st.columns(2)
        email = c4.text_input("Email")
        phone = c5.text_input("Mobile (with country code, e.g. +9198...)")
        c6, c7 = st.columns(2)
        pe = c6.checkbox("Email alerts", value=True)
        psms = c7.checkbox("SMS alerts", value=False)
        cats = st.multiselect("Permitted alert categories", ["SIMULATION ONLY", "OBSERVATION-BASED"],
                              default=["SIMULATION ONLY", "OBSERVATION-BASED"])
        if st.form_submit_button("Add recipient"):
            if not name.strip() or not (email.strip() or phone.strip()):
                st.error("Name and at least one of email / mobile are required.")
            elif phone.strip() and not (phone.strip().startswith("+") and phone.strip()[1:].isdigit()):
                st.error("Mobile number must be in international format, e.g. +919812345678.")
            elif email.strip() and "@" not in email:
                st.error("Enter a valid email address.")
            else:
                store.add_recipient(name, org, des, email, phone, {"email": pe, "sms": psms}, cats, actor=_user())
                st.success("Recipient added (unverified). Verify each channel below before it can receive alerts.")
    rs = store.list_recipients()
    if not rs:
        st.caption("No recipients registered yet.")
    from src.notifications.providers import EmailProvider, SmsProvider
    for r in rs:
        with st.expander(f"{'🟢' if r['enabled'] else '⚪'} {r['name']} · {r.get('designation') or ''} · "
                         f"{r.get('organisation') or ''}"):
            st.write(f"Email {store.mask(r.get('email'))} - {'verified' if r['email_verified'] else 'NOT verified'} · "
                     f"Mobile {store.mask(r.get('phone'))} - {'verified' if r['phone_verified'] else 'NOT verified'} · "
                     f"categories {', '.join(r['categories'])}")
            c1, c2, c3 = st.columns(3)
            if c1.button("Disable" if r["enabled"] else "Enable", key=f"rc_en_{r['id']}"):
                store.update_recipient(r["id"], actor=_user(), enabled=0 if r["enabled"] else 1)
                st.rerun()
            for col, ch in ((c2, "email"), (c3, "sms")):
                target = r.get("email") if ch == "email" else r.get("phone")
                if target and col.button(f"Send {ch} verification code", key=f"rc_v_{r['id']}_{ch}"):
                    code = store.create_verification(r["id"], ch)
                    msg = (f"Forest Fire Digital Twin: your verification code is {code} (valid 15 minutes).")
                    res = (EmailProvider().send(target, "Verification code", msg) if ch == "email"
                           else SmsProvider().send(target, msg))
                    store.record_delivery("VERIFY", "-", r["id"], ch, res.provider, res.status, res.message_id, res.error)
                    st.info(f"Verification {ch}: {em.describe_status(res.status)}" +
                            (" - mock mode: the code is shown here for testing only: " + code if res.status == "mock"
                             else ""))
            c4, c5 = st.columns([1, 1])
            ch = c4.selectbox("Channel", ["email", "sms"], key=f"rc_vc_{r['id']}")
            code = c5.text_input("Verification code", key=f"rc_code_{r['id']}")
            if st.button("Verify", key=f"rc_ok_{r['id']}"):
                st.success("Verified.") if store.confirm_verification(r["id"], ch, code, actor=_user()) else \
                    st.error("Invalid or expired code.")


def render_alert_history():
    if _role() != "admin":
        st.info("Alert history is available to administrators only.")
        return
    hist = store.alert_history(50)
    if not hist:
        st.caption("No emergency alerts have been sent.")
        return
    names = {r["id"]: r["name"] for r in store.list_recipients()}
    for a in hist:
        cnt = ", ".join(f"{k}: {v}" for k, v in (a["summary"].get("counts") or {}).items())
        with st.expander(f"{a['alert_id']} · {a['alert_type']} · {em.ist(a['created_utc'])} · by {a['sender']} · {cnt}"):
            st.caption(f"Report {a['report_id']} · {a['region']} · body SHA-256 {a['body_sha256'][:16]}…")
            for d in a["deliveries"]:
                st.write(f"- {names.get(d['recipient_id'], '#' + str(d['recipient_id']))} · {d['channel']} · "
                         f"{d['provider']} · **{em.describe_status(d['status'])}** · {em.ist(d['timestamp_utc'])}"
                         + (f" · {d['error']}" if d.get("error") else ""))


# ── receiver side: in-app alarm ─────────────────────────────────────────────

_ALARM_JS = """
<script>
(function(){
  const key='ffdt_alarm_seen', id=%s, mute=%s, test=%s;
  let seen=[];try{seen=JSON.parse(sessionStorage.getItem(key)||'[]')}catch(e){}
  if(mute||(!test&&seen.indexOf(id)>=0))return;            // each alert sounds once per browser session
  try{seen.push(id);sessionStorage.setItem(key,JSON.stringify(seen.slice(-50)))}catch(e){}
  try{const C=new (window.AudioContext||window.webkitAudioContext)();let t=C.currentTime;
    for(let k=0;k<3;k++){const o=C.createOscillator(),g=C.createGain();o.type='square';o.frequency.value=k%%2?660:880;
      g.gain.setValueAtTime(0.0001,t);g.gain.exponentialRampToValueAtTime(0.18,t+0.02);g.gain.exponentialRampToValueAtTime(0.0001,t+0.35);
      o.connect(g);g.connect(C.destination);o.start(t);o.stop(t+0.36);t+=0.45;}
  }catch(e){document.body.innerText='Audio blocked by the browser - click anywhere on the page once to allow sound.';}
})();
</script>"""


def play_alarm(alert_key: str, test: bool = False):
    mute = bool(st.session_state.get("alarm_muted", False))
    components.html(_ALARM_JS % (json.dumps(alert_key), "true" if mute else "false", "true" if test else "false"),
                    height=0)


def render_inapp_alerts():
    """Unacknowledged emergency alerts for the signed-in user (sidebar), with an
    audible alarm (once per alert, mutable) and an acknowledge button. Email /
    SMS cannot force a custom sound on a recipient's phone; this alarm works in
    the open dashboard (a future mobile app / installed PWA could add push)."""
    user = _user()
    if not user:
        return
    try:
        rows = store.unacked_inapp(user)
    except Exception:
        return
    with st.sidebar:
        with st.expander(f"Emergency alerts{' · ' + str(len(rows)) + ' new' if rows else ''}", expanded=bool(rows)):
            st.toggle("Mute alert sound", key="alarm_muted")
            if st.button("Test alarm (no message is sent)", key="alarm_test"):
                play_alarm("test", test=True)
            st.caption("In-app alarm only. Email / SMS cannot force a custom sound on a recipient's phone.")
        for r in rows:
            st.error(f"**{r['alert_type']}** · {r['alert_id']} · {em.ist(r['created_utc'])}\n\n{r['body']}")
            if st.button("Acknowledge", key=f"ack_{r['id']}"):
                store.ack_inapp(r["id"], user)
                st.rerun()
        if rows:
            play_alarm(rows[0]["alert_id"])
