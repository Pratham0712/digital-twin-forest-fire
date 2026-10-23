"""
alerts_ui.py - simulation report downloads, the opt-in emergency-alert review /
confirmation dialog, recipient management, alert history and the in-app
receiver alarm. All of it requires a signed-in user; sending and recipient
management require the admin role.
"""
from __future__ import annotations

import html
import json
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components

from src.notifications import alert_store as store
from src.notifications import emergency as em
from src.notifications import config as cfg
from src.notifications.providers import provider_status


def _user() -> str:
    return st.session_state.get("auth_user", "")


def _role() -> str:
    return st.session_state.get("auth_role", "viewer")


# ── after a completed simulation ────────────────────────────────────────────

EA_MENU_CSS = """
<style>
[data-testid="stPopoverBody"]:has(.ea-menu) { width: 330px !important; max-width: calc(100vw - 32px) !important;
  padding: 6px !important; background: #111722 !important; border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 12px !important; box-shadow: 0 16px 40px rgba(0,0,0,0.5) !important; }
[data-testid="stPopoverBody"]:has(.ea-menu) [data-testid="stVerticalBlock"] { gap: 2px !important; }
.ea-menu { font-size: 11px; font-weight: 700; letter-spacing: 0.08em; color: #8B97A8; padding: 6px 10px 8px;
  border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 4px; }
:is([class*="_ea_opt_"], [class*="_xp_opt_"]) button { width: 100% !important; display: flex !important; flex-direction: column !important;
  align-items: flex-start !important; justify-content: flex-start !important; text-align: left !important;
  min-height: 0 !important; padding: 9px 12px !important; border-radius: 8px !important;
  background: transparent !important; border: 1px solid transparent !important; box-shadow: none !important; }
:is([class*="_ea_opt_"], [class*="_xp_opt_"]) button > div, :is([class*="_ea_opt_"], [class*="_xp_opt_"]) button [data-testid="stMarkdownContainer"] { width: 100% !important;
  justify-content: flex-start !important; text-align: left !important; }
:is([class*="_ea_opt_"], [class*="_xp_opt_"]) button p { text-align: left !important; font-size: 14px !important; font-weight: 600 !important; color: #E8EDF3 !important; }
:is([class*="_ea_opt_"], [class*="_xp_opt_"]) button::after { display: block; padding-left: 1.75em; margin-top: 2px; font-size: 12px;
  font-weight: 400; line-height: 1.35; color: #A9B4C4; }
[class*="_ea_opt_email"] button::after { content: "Send the emergency alert to the configured e-mail recipient"; }
[class*="_ea_opt_sms"] button::after { content: "Send an SMS to the configured mobile recipient (textbee)"; }
[class*="_xp_opt_pdf"] button::after { content: "Simulation report as a PDF document"; }
[class*="_xp_opt_docx"] button::after { content: "Simulation report as an editable Word document"; }
:is([class*="_ea_opt_"], [class*="_xp_opt_"]) button:hover { background: rgba(255,122,24,0.10) !important; border-color: rgba(255,122,24,0.35) !important; }
:is([class*="_ea_opt_"], [class*="_xp_opt_"]) button:focus-visible { outline: 2px solid #FF7A18 !important; outline-offset: 1px; }
html[data-ff-theme="navy"] [data-testid="stPopoverBody"]:has(.ea-menu) { background: #081D33 !important;
  border-color: #163A5E !important; }
html[data-ff-theme="navy"] .ea-menu { color: #8298B3; border-bottom-color: #163A5E; }
html[data-ff-theme="navy"] [class*="_ea_menu"] button { background: #0878F9 !important; border-color: #0878F9 !important;
  color: #FFFFFF !important; }
html[data-ff-theme="navy"] [class*="_ea_menu"] button:hover { background: #0756C9 !important; border-color: #0756C9 !important; }
html[data-ff-theme="navy"] [data-testid="stPopoverBody"] :is([class*="_ea_opt_"], [class*="_xp_opt_"]) button { background: transparent !important;
  border-color: transparent !important; }
html[data-ff-theme="navy"] :is([class*="_ea_opt_"], [class*="_xp_opt_"]) button p { color: #F3F7FC !important; }
html[data-ff-theme="navy"] :is([class*="_ea_opt_"], [class*="_xp_opt_"]) button::after { color: #B6C7DC; }
html[data-ff-theme="navy"] [data-testid="stPopoverBody"] :is([class*="_ea_opt_"], [class*="_xp_opt_"]) button:hover { background: rgba(8,120,249,0.18) !important;
  border-color: #0878F9 !important; }
html[data-ff-theme="navy"] :is([class*="_ea_opt_"], [class*="_xp_opt_"]) button:focus-visible { outline-color: #00B8F0 !important; }
</style>
"""

EA_OPTIONS = (("email", "✉️ Email Alert"), ("sms", "📱 SMS to Mobile"))


def _choose_channel(channel: str, rid: str, menu_key: str):
    """Menu option callback: close the menu and open that channel's confirmation dialog (nothing is sent)."""
    st.session_state[menu_key] = False
    st.session_state["_ea_open"] = (channel, rid)


def render_result_actions(snapshot: Optional[dict], kp: str):
    if not snapshot:
        return
    note = st.session_state.pop("_ea_toast", None)
    st.markdown(
        f"<div class='info-box'><b>Simulation complete</b> · report <code>{snapshot['report_id']}</code> · "
        f"{'LIVE (observed ignition, simulated spread)' if snapshot['mode'] == 'LIVE' else snapshot['mode'] + ' (hypothetical - SIMULATED)'}"
        f" · {snapshot['results']['burned_ha']:.1f} ha burned in {snapshot['results']['duration_label']}, front "
        f"{snapshot['results']['front_distance_m'] or 0:.0f} m towards the {snapshot['results']['spread_towards']}</div>"
        "\n\n" + EA_MENU_CSS                        # menu styles ride along: no extra element, no blank gap
        + ("\n\n" + SUCCESS_TOAST_CSS if note and note[0] in SUCCESS_TEXT.values() else ""),
        unsafe_allow_html=True)
    rid = snapshot["report_id"]
    if st.session_state.get("_charts_prewarmed") != rid:   # once per finished simulation
        st.session_state["_charts_prewarmed"] = rid
        from src.reports import simulation_report as rep
        rep.prewarm_charts(snapshot)
    c1, c2, _ = st.columns([1.4, 1.4, 1.6])
    with c1:
        _export_menu(snapshot, kp)
    # one entry point; the delivery-method menu is drawn only while it is open (nothing is sent from the menu)
    menu_key = f"{kp}_ea_menu"
    with c2:
        menu = st.popover("🚨 Send Emergency Alert", type="primary", key=menu_key, on_change="rerun",
                          use_container_width=True)
    if menu.open:
        with menu:
            st.markdown("<div class='ea-menu' role='heading' aria-level='3'>DELIVERY METHOD</div>",
                        unsafe_allow_html=True)
            for ch, label in EA_OPTIONS:
                st.button(label, key=f"{kp}_ea_opt_{ch}", use_container_width=True,
                          on_click=_choose_channel, args=(ch, rid, menu_key))
    st.session_state.pop("_ea_done", None)
    want = st.session_state.get("_ea_open")
    if want and want[1] == rid:
        st.session_state.pop("_ea_open", None)          # open once; reruns inside the dialog keep it open
        (_email_dialog if want[0] == "email" else _manual_sms_dialog)(snapshot)
    if note:                                             # one short result; details live in Admin
        st.toast(note[0], icon=note[1]) if note[1] else st.toast(note[0])


# ── manual simulation SMS ────────────────────────────────────────────────────

def _clear_awaiting():
    """Dialog dismissed (✕ / Esc / outside click): nothing is sent."""
    st.session_state.pop("_ea_open", None)


def _readiness(snapshot: dict, channel: str) -> dict:
    return em.manual_email_readiness(snapshot) if channel == "email" else em.manual_sms_readiness(snapshot)


def _guards(ready: dict, channel: str) -> list:
    return [em.manual_email_guard(e) for e in ready["recipients"]] if channel == "email" else list(ready["recipients"])


SUCCESS_TEXT = {"sms": "✓ SMS sent successfully!", "email": "✓ Email sent successfully!"}
# green success toast (same green as the dashboard's success messages); added only on the run that shows it
SUCCESS_TOAST_CSS = ('<style>[data-testid="stToast"]{background:#15803d !important;color:#ffffff !important;'
                     'border:1px solid #22c55e !important;}[data-testid="stToast"] *{color:#ffffff !important;}</style>')


def result_toast(channel: str, out: dict):
    """(message, icon) for the short toast after a send attempt - truthful: acceptance is never delivery."""
    from src.utils.timezone import format_ist
    what = "Email" if channel == "email" else "SMS"
    if out.get("blocked"):
        return f"{what} not sent - {out['blocked']}", "⛔"
    res = out.get("results") or []
    if res and all(r["status"] == "cooldown" for r in res):
        nxt = min(r["next_eligible_utc"] for r in res)
        return f"{what} not sent - already sent for this simulation; next send allowed at {format_ist(nxt)}.", "⏳"
    sent = [r for r in res if r["status"] != "cooldown"]
    ok = [r for r in sent if r["status"] in ("accepted", "queued", "sent", "delivered")]
    if ok and len(ok) == len(sent):
        # same success condition as before (provider accepted every message); delivery details stay in
        # Admin -> Notifications, never claimed here
        return SUCCESS_TEXT[channel], None
    if ok:
        return f"{what} sent to {len(ok)} of {len(sent)} recipients - {len(sent) - len(ok)} failed (see Admin → Notifications).", "⚠️"
    bad = next((r for r in sent if r["status"] != "unknown"), None)
    if bad is None:
        return f"{what}: the provider did not answer - status unknown (see Admin → Notifications).", "⚠️"
    return f"{what} failed - {bad.get('error') or bad['status']}.", "⛔"


def _do_send(snapshot: dict, channel: str, send):
    """Confirm callback - the only place a manual e-mail / SMS is sent. A second (double) click only meets the
    per-recipient guard (cooldown rows, nothing resent) and does not replace the first result."""
    try:
        out = send()
    except PermissionError as exc:
        out = {"blocked": str(exc), "results": [], "alert_id": None}
    first = st.session_state.get("_ea_toast")
    if not (first and out.get("results") and all(r["status"] == "cooldown" for r in out["results"])):
        st.session_state["_ea_toast"] = result_toast(channel, out)
    st.session_state["_ea_done"] = True


def _cancel_send():
    """Cancel callback: closes the dialog; never sends."""
    st.session_state["_ea_done"] = True


def _close_if_done():
    if st.session_state.pop("_ea_done", False):
        st.rerun()                                      # full rerun closes the dialog and shows the toast


def _confirm_dialog(snapshot: dict, channel: str, question: str, show, send):
    """Minimal confirmation: question, destination, Confirm & Send / Cancel. Errors that stop a send are shown
    briefly here; no message preview (the full message is built only when sending)."""
    from src.notifications import auto_alerts as aa
    from src.utils.timezone import format_ist
    _close_if_done()
    if _role() != "admin":
        st.error("Only authorised administrators can send emergency alerts.")
        return
    ready = _readiness(snapshot, channel)
    st.write(question)
    who = ", ".join(show(r) for r in ready["recipients"]) or "no recipient configured"
    st.markdown(f"**To:** {html.escape(who)}")
    can_send = ready["ok"]
    if not ready["ok"]:
        st.error(ready["reason"])
    else:
        waits = [aa.manual_status(snapshot["report_id"], g) for g in _guards(ready, channel)]
        blocked = [w for w in waits if not w["eligible"]]
        if blocked and len(blocked) == len(waits):
            can_send = False
            st.warning("Already sent for this simulation - next send allowed at "
                       f"{format_ist(min(w['next_eligible_utc'] for w in blocked))}.")
        elif blocked:
            st.caption(f"{len(blocked)} recipient(s) are in the {waits[0]['cooldown_minutes']}-minute cooldown and "
                       "will be skipped.")
    c1, c2 = st.columns(2)
    c1.button("Confirm & Send", type="primary", disabled=not can_send, key=f"ea_{channel}_confirm",
              use_container_width=True, on_click=_do_send, args=(snapshot, channel, send))
    c2.button("Cancel", key=f"ea_{channel}_cancel", use_container_width=True, on_click=_cancel_send)


@st.dialog("Confirm Email Alert", on_dismiss=_clear_awaiting)
def _email_dialog(snapshot: dict):
    _confirm_dialog(snapshot, "email", "Send an emergency alert email for this simulation?", store.mask,
                    lambda: em.send_manual_sim_email(snapshot, _user(), _role(), confirmed=True))


@st.dialog("Confirm SMS Alert", on_dismiss=_clear_awaiting)
def _manual_sms_dialog(snapshot: dict):
    _confirm_dialog(snapshot, "sms", "Send an emergency SMS for this simulation?", em.mask_phone,
                    lambda: em.send_manual_sim_sms(snapshot, _user(), _role(), confirmed=True))


XP_OPTIONS = (("pdf", "PDF (.pdf)"), ("docx", "Word (.docx)"))


def _export_available(kind: str) -> bool:
    import importlib.util
    return importlib.util.find_spec("reportlab" if kind == "pdf" else "docx") is not None


def _export_menu(snapshot: dict, kp: str):
    """Export Report dropdown: exactly PDF and Word. Nothing is built when the menu opens; clicking a format builds
    only that file (in Streamlit's deferred-download thread, no page rerun) and the browser downloads it. A failed
    build shows "Failed to generate file for download" under the button - never a false success."""
    from src.reports import simulation_report as rep
    rid = snapshot["report_id"]
    with st.popover("📄 Export Report", key=f"{kp}_xp_menu", use_container_width=True):
        st.markdown("<div class='ea-menu' role='heading' aria-level='3'>REPORT FORMAT</div>", unsafe_allow_html=True)
        for kind, label in XP_OPTIONS:
            mime, ext = rep.EXPORT_FORMATS[kind]
            if not _export_available(kind):
                st.button(label, key=f"{kp}_xp_opt_{kind}", disabled=True, use_container_width=True,
                          help=f"{label} export needs {'reportlab' if kind == 'pdf' else 'python-docx'} installed")
                continue
            st.download_button(label, (lambda k=kind: rep.export_document(snapshot, k)), file_name=f"{rid}{ext}",
                               mime=mime, key=f"{kp}_xp_opt_{kind}", on_click="ignore", use_container_width=True)


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
        phone = c5.text_input("Mobile (10-digit Indian number or +<country code>...)")
        c6, c7 = st.columns(2)
        pe = c6.checkbox("Email alerts", value=True)
        psms = c7.checkbox("SMS alerts", value=False)
        cats = st.multiselect("Permitted alert categories", ["SIMULATION ONLY", "OBSERVATION-BASED"],
                              default=["SIMULATION ONLY", "OBSERVATION-BASED"])
        if st.form_submit_button("Add recipient"):
            if not name.strip() or not (email.strip() or phone.strip()):
                st.error("Name and at least one of email / mobile are required.")
            elif phone.strip() and cfg.normalise_phone(phone) is None:
                st.error("Enter a valid mobile number: 10-digit Indian mobile (e.g. 98123 45678) or "
                         "+<country code><number>.")
            elif email.strip() and "@" not in email:
                st.error("Enter a valid email address.")
            else:
                store.add_recipient(name, org, des, email, cfg.normalise_phone(phone) if phone.strip() else "",
                                    {"email": pe, "sms": psms}, cats, actor=_user())
                st.success("Recipient added (unverified). Verify each channel below before it can receive alerts.")
    rs = store.list_recipients()
    if not rs:
        st.caption("No recipients registered yet.")
    st.caption("E-mail addresses are confirmed with a one-time code (valid "
               f"{store.VERIFY_TTL_MIN} minutes, single use, {store.VERIFY_MAX_ATTEMPTS} attempts, one new code per "
               "minute). Mobile numbers are not verified: SMS alerts go to valid numbers with 'SMS alerts' enabled, "
               "sent from the phone paired with textbee.")
    for r in rs:
        with st.expander(f"{'🟢' if r['enabled'] else '⚪'} {r['name']} · {r.get('designation') or ''} · "
                         f"{r.get('organisation') or ''}"):
            email_state = (f"{store.mask(r['email'])} - {'verified' if r['email_verified'] else 'NOT verified'}"
                           if r.get("email") else "none")
            sms_state = (f"{store.mask(r['phone'])} (SMS alerts {'on' if (r.get('prefs') or {}).get('sms') else 'off'})"
                         if r.get("phone") else "none")
            st.write(f"Email {email_state} · Mobile {sms_state} · categories {', '.join(r['categories'])}")
            c1, c2, _ = st.columns(3)
            if c1.button("Disable" if r["enabled"] else "Enable", key=f"rc_en_{r['id']}"):
                store.update_recipient(r["id"], actor=_user(), enabled=0 if r["enabled"] else 1)
                st.rerun()
            if not r.get("email"):
                continue
            if c2.button("Send email verification code", key=f"rc_v_{r['id']}_email"):
                try:
                    out = em.send_verification(r["id"], "email", _user(), _role())
                except PermissionError as exc:
                    out = {"status": "failed", "message": str(exc)}
                st.session_state[f"_rc_msg_{r['id']}"] = (out["status"], out["message"])
            note = st.session_state.get(f"_rc_msg_{r['id']}")
            if note:
                (st.info if note[0] in ("accepted", "queued", "sent", "delivered", "unknown") else st.warning)(note[1])
            code = st.text_input("Email verification code", key=f"rc_code_{r['id']}", max_chars=6)
            if st.button("Verify", key=f"rc_ok_{r['id']}"):
                ok, why = store.check_verification(r["id"], "email", code, actor=_user())
                if ok:
                    st.session_state.pop(f"_rc_msg_{r['id']}", None)
                    st.success("E-mail verified.")
                    st.rerun()
                else:
                    st.error(why)


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


# ── Admin page: provider configuration (email / SMS) ────────────────────────

_BADGE = {"ok": ("#1f7a3a", "✅"), "warn": ("#9a6b00", "⚠️"), "bad": ("#b3261e", "⛔"), "info": ("#2f6db3", "ℹ️")}


def _badge(kind: str, text: str):
    col, ico = _BADGE[kind]
    st.markdown(f"<div style='border-left:4px solid {col};background:rgba(127,127,127,.08);padding:8px 12px;"
                f"border-radius:6px;margin:4px 0 10px'>{ico} {text}</div>", unsafe_allow_html=True)


def _result_badge(res, what: str, channel: Optional[str] = None):
    s = res.status if hasattr(res, "status") else res.get("status")
    err = res.error if hasattr(res, "error") else res.get("error")
    detail = res.detail if hasattr(res, "detail") else res.get("detail")
    label = em.describe_status(s)
    if channel in ("sms", "email") and s not in ("connected",):
        from src.notifications.auto_alerts import describe_channel
        label = describe_channel(s, channel)
    if s == "not_configured":
        label = "Not configured"
    elif s == "failed" and channel is None and "connection" in what.lower():
        label = "Connection failed"
    if s in ("connected", "accepted", "queued", "sent", "delivered"):
        _badge("ok", f"{what}: <b>{label}</b>" + (f" - {html.escape(str(detail))}" if detail else ""))
    elif s == "mock":
        _badge("info", f"{what}: <b>MOCK mode</b> - nothing was sent (ALERT_PROVIDER_MODE=mock).")
    elif s == "duplicate":
        _badge("warn", f"{what}: not sent again - {err}")
    else:
        _badge("bad", f"{what}: <b>{label}</b>" + (f" - {html.escape(str(err))}" if err else ""))


def render_provider_config():
    """Email (SMTP) and SMS gateway (textbee Android phone or Twilio) configuration, validation and test actions. Values typed here apply to
    the running app only (process memory) - they are never written to disk, the database, logs or reports.
    For a permanent setup use .env locally or Streamlit secrets when hosted (see .env.example)."""
    from src.notifications import config as cfg
    from src.notifications.providers import EmailProvider, SmsProvider
    if _role() != "admin":
        st.info("Notification configuration is available to administrators only.")
        return
    mode = "mock" if cfg.mock_mode() else "live"
    if mode == "mock":
        _badge("info", "Provider mode <b>MOCK</b>: every send is recorded as 'mock' and nothing leaves this machine. "
                       "Set ALERT_PROVIDER_MODE=live (or choose Live below) to send real messages.")
    m1, m2 = st.columns([1, 2])
    new_mode = m1.radio("Provider mode", ["mock", "live"], index=0 if mode == "mock" else 1, horizontal=True,
                        key="cfg_mode", help="mock = record only, live = really send")
    if new_mode != mode:
        cfg.set_runtime({"ALERT_PROVIDER_MODE": new_mode})
        st.rerun()
    m2.caption("Settings precedence: this page (current session) → environment / .env → Streamlit secrets. "
               "Secrets are masked and never displayed.")

    # ── email ────────────────────────────────────────────────────────────
    st.markdown("#### 📧 Email (SMTP)")
    s = cfg.smtp_settings()
    probs = cfg.smtp_problems(s)
    if probs:
        _badge("warn", "Email is <b>not configured</b>: " + "; ".join(probs) + ".")
    else:
        _badge("ok", f"Email settings complete ({s['host']}:{s['port']}, {s['security'].upper()}, sender "
                     f"{s['sender']}). Use <i>Test connection</i> to verify the login.")
    with st.form("smtp_cfg"):
        c1, c2, c3 = st.columns([2, 1, 1])
        host = c1.text_input("SMTP host", value=s["host"], placeholder="smtp.gmail.com")
        port = c2.number_input("Port", 1, 65535, value=s["port"] if 0 < s["port"] < 65536 else 587)
        sec = c3.selectbox("Security", cfg.SECURITY_MODES, index=cfg.SECURITY_MODES.index(s["security"])
                           if s["security"] in cfg.SECURITY_MODES else 0, help="STARTTLS on 587, SSL on 465")
        c4, c5 = st.columns(2)
        user = c4.text_input("SMTP username", value=s["user"], placeholder="alerts@yourdomain.org")
        pw = c5.text_input("Password / app password", type="password", value="",
                           placeholder="•••••• (set)" if s["password"] else "not set",
                           help="Leave empty to keep the current value. Gmail needs an App Password.")
        c6, c7 = st.columns(2)
        sender = c6.text_input("Sender address (From)", value=cfg.get("SMTP_FROM"), placeholder="defaults to the username")
        tmo = c7.number_input("Connection timeout (s)", 3, 120, value=int(s["timeout"]))
        url = st.text_input("Dashboard URL for alert links (optional)", value=cfg.get("DASHBOARD_URL"),
                            placeholder="https://your-dashboard.example")
        if st.form_submit_button("Save for this session and validate", type="primary"):
            bad = []
            if sender.strip() and not cfg.valid_email(sender):
                bad.append("sender address is not a valid email")
            if url.strip() and not url.strip().startswith(("http://", "https://")):
                bad.append("dashboard URL must start with http:// or https://")
            if bad:
                st.error("; ".join(bad))
            else:
                cfg.set_runtime({"SMTP_HOST": host, "SMTP_PORT": str(int(port)), "SMTP_SECURITY": sec,
                                 "SMTP_USER": user, "SMTP_PASSWORD": pw, "SMTP_FROM": sender,
                                 "SMTP_TIMEOUT": str(int(tmo)), "DASHBOARD_URL": url})
                st.rerun()
    t1, t2, t3 = st.columns([1, 1.6, 1])
    if t1.button("🔌 Test connection", key="smtp_test", use_container_width=True):
        with st.spinner("Connecting to the SMTP server…"):
            st.session_state["_smtp_test"] = EmailProvider().test_connection()
    to = t2.text_input("Test email recipient", key="smtp_test_to", label_visibility="collapsed",
                       placeholder="test recipient email")
    if t3.button("✉️ Send test email", key="smtp_send", use_container_width=True):
        with st.spinner("Sending test email…"):
            st.session_state["_smtp_send"] = em.send_test("email", to, _user(), _role())
    if st.session_state.get("_smtp_test") is not None:
        _result_badge(st.session_state["_smtp_test"], "SMTP connection test")
    if st.session_state.get("_smtp_send") is not None:
        _result_badge(st.session_state["_smtp_send"], "Test email", channel="email")

    # ── SMS ──────────────────────────────────────────────────────────────
    from src.notifications.providers import sms_provider
    prov = cfg.sms_provider_name()
    st.markdown("#### 📱 SMS gateway")
    pc1, pc2 = st.columns([1, 2])
    choice = pc1.radio("Provider", ["textbee", "twilio"], index=0 if prov == "textbee" else 1, horizontal=True,
                       key="cfg_sms_provider", format_func=lambda v: {"textbee": "Android phone (textbee.dev)",
                                                                       "twilio": "Twilio"}[v])
    if choice != prov:
        cfg.set_runtime({"SMS_PROVIDER": choice})
        st.rerun()
    pc2.caption("textbee: your own Android phone and SIM send the SMS through the textbee app (normal SIM SMS "
                "charges apply). Twilio: a paid cloud SMS account. Setup guide: docs/SMS_GATEWAY_SETUP.md.")
    if prov == "textbee":
        tb = cfg.textbee_settings()
        p2 = cfg.textbee_problems(tb)
        if p2:
            _badge("warn", "SMS gateway: <b>Not configured</b> - " + html.escape("; ".join(p2)) + ".")
        else:
            _badge("ok", "textbee settings complete (API key set" + (f", device {store.mask(tb['device_id'])}"
                   if tb["device_id"] else ", default device") + "). Use <i>Test connection</i> to check the key "
                   "and the phone.")
        with st.form("textbee_cfg"):
            c1, c2 = st.columns(2)
            key = c1.text_input("API key", type="password", value="", placeholder="•••••• (set)" if tb["api_key"]
                                else "not set", help="textbee dashboard → API keys. Leave empty to keep the current value.")
            dev = c2.text_input("Device ID (optional)", value=tb["device_id"],
                                help="textbee dashboard → Devices. Empty = your default / most recently active phone.")
            c3, c4 = st.columns([2, 1])
            base = c3.text_input("API base URL", value=tb["base_url"])
            tmo = c4.number_input("Timeout (s)", 1, 120, value=int(tb["timeout"]) if 1 <= tb["timeout"] <= 120 else 15)
            if st.form_submit_button("Save for this session and validate", type="primary"):
                trial = dict(tb, api_key=key or tb["api_key"], device_id=dev.strip(), base_url=base.strip().rstrip("/"),
                             timeout=float(tmo))
                probs = cfg.textbee_problems(trial)
                if probs:
                    st.error("Configuration invalid: " + "; ".join(probs))
                else:
                    cfg.set_runtime({"TEXTBEE_API_KEY": key, "TEXTBEE_DEVICE_ID": dev.strip(),
                                     "TEXTBEE_BASE_URL": base.strip().rstrip("/"), "TEXTBEE_TIMEOUT": str(int(tmo))})
                    st.rerun()
    else:
        s2 = cfg.sms_settings()
        p2 = cfg.sms_problems(s2)
        if p2:
            _badge("warn", "SMS gateway: <b>Not configured</b> - " + "; ".join(p2) + ".")
        else:
            _badge("ok", f"Twilio settings complete (sender {store.mask(s2['from_number'])}). Use <i>Test connection</i> "
                         "to verify the credentials.")
        with st.form("sms_cfg"):
            c1, c2, c3 = st.columns(3)
            sid = c1.text_input("Account SID", value=s2["sid"], placeholder="AC…")
            tok = c2.text_input("Auth token", type="password", value="",
                                placeholder="•••••• (set)" if s2["token"] else "not set",
                                help="Leave empty to keep the current value.")
            frm = c3.text_input("Sender number (E.164)", value=s2["from_number"], placeholder="+14155550100")
            if st.form_submit_button("Save for this session and validate", type="primary"):
                if frm.strip() and not cfg.valid_phone(frm):
                    st.error("Sender number must be in E.164 format, e.g. +14155550100.")
                else:
                    cfg.set_runtime({"TWILIO_ACCOUNT_SID": sid, "TWILIO_AUTH_TOKEN": tok, "TWILIO_FROM_NUMBER": frm})
                    st.rerun()
    t1, t2, t3, t4 = st.columns([1, 1.3, 1, 1.1])
    if t1.button("🔌 Test connection", key="sms_test", use_container_width=True,
                 help="Checks the credentials (and, for textbee, the registered phone). Sends nothing."):
        with st.spinner("Checking the SMS gateway…"):
            st.session_state["_sms_test"] = sms_provider().test_connection()
    to2 = t2.text_input("Test SMS number", key="sms_test_to", label_visibility="collapsed", placeholder="+9198…",
                        help="International E.164 format, e.g. +919812345678")
    if t3.button("💬 Send test SMS", key="sms_send", use_container_width=True,
                 help="Sends ONE real SMS marked TEST ALERT (no fire). It never starts the fire-alert workflow."):
        with st.spinner("Submitting test SMS…"):
            st.session_state["_sms_send"] = em.send_test("sms", to2, _user(), _role())
    if t4.button("🔄 Refresh SMS status", key="sms_refresh", use_container_width=True,
                 help="Ask the gateway for the current status of recent SMS (the only source of 'delivered')"):
        st.session_state["_sms_refresh"] = em.refresh_sms_statuses()
    if st.session_state.get("_sms_test") is not None:
        _result_badge(st.session_state["_sms_test"], "SMS gateway connection test")
    if st.session_state.get("_sms_send") is not None:
        _result_badge(st.session_state["_sms_send"], "Test SMS", channel="sms")
    if st.session_state.get("_sms_refresh") is not None:
        st.caption(f"{st.session_state['_sms_refresh']} SMS status update(s) received from the provider.")

    # ── delivery log ─────────────────────────────────────────────────────
    st.markdown("#### 🧾 Recent delivery attempts")
    rows = store.recent_deliveries(25)
    if not rows:
        st.caption("No delivery attempts recorded yet.")
    else:
        import pandas as pd
        from src.utils.timezone import format_ist
        recips = {r["id"]: r for r in store.list_recipients()}
        st.dataframe(pd.DataFrame([{
            "time (IST)": format_ist(d["timestamp_utc"]), "event": d["alert_id"], "channel": d["channel"],
            "to": store.mask((recips.get(d["recipient_id"]) or {}).get("email" if d["channel"] == "email" else "phone"))
            if d["recipient_id"] else ("configured SMS recipient" if str(d["alert_id"]).startswith("MSIM-") else
                                       "automatic-alert recipient" if str(d["alert_id"]).startswith("AUTO-") else
                                       "test address"), "provider": d["provider"],
            "status": d["status"], "detail": d.get("error") or ""} for d in rows]),
            use_container_width=True, hide_index=True)


def render_auto_alerts():
    """Automatic fire alerts: enable / threshold / cooldown / channel policy (stored in the database),
    automatic-alert recipients (this session; permanent via .env), current evaluation preview and history."""
    from src.notifications import auto_alerts as aa
    from src.notifications import config as cfg
    from src.utils.timezone import format_ist
    import pandas as pd
    if _role() != "admin":
        st.info("Automatic-alert settings are available to administrators only.")
        return
    s = aa.get_settings()
    st.markdown("#### 🚨 Automatic fire alerts")
    if s["enabled"]:
        _badge("ok", f"Automatic alerts are <b>ON</b>: EXTREME zones with a model fire probability ≥ "
                     f"<b>{s['min_confidence']:.0%}</b>, from live data only, at most once per incident every "
                     f"<b>{s['cooldown_minutes']} min</b> (re-sent earlier only if the probability rises by "
                     f"≥ {s['escalation_step']:.0%}). Channels: <b>{s['policy'].replace('_', ' ')}</b>.")
    else:
        _badge("info", "Automatic alerts are <b>OFF</b>: refreshes are evaluated and logged as 'disabled', nothing is "
                       "sent. Manual emergency alerts (Spread Simulation) are unaffected.")
    with st.form("auto_alert_cfg"):
        c1, c2, c3 = st.columns(3)
        en = c1.toggle("Enable automatic alerts", value=s["enabled"])
        mc = c2.number_input("Minimum model fire probability", 0.01, 0.99, value=float(s["min_confidence"]), step=0.01,
                             format="%.2f", help="The XGBoost fire-occurrence probability of the zone (0-1). 0.35 ≈ the "
                             "EXTREME operating point, where at least half of historical zone-days this high burned.")
        cd = c3.number_input("Cooldown per incident (minutes)", 0, 10080, value=int(s["cooldown_minutes"]), step=30)
        c4, c5, c6 = st.columns(3)
        es = c4.number_input("Escalation step (probability)", 0.01, 1.0, value=float(s["escalation_step"]), step=0.01,
                             format="%.2f", help="Inside the cooldown, notify again only if the probability rose by at "
                                                 "least this much since the last notification of that zone.")
        pol = c5.selectbox("Channels", aa.POLICIES, index=aa.POLICIES.index(s["policy"]),
                           format_func=lambda p: {"sms_then_email": "SMS, e-mail only if no SMS was accepted",
                                                  "both": "SMS and e-mail", "sms_only": "SMS only",
                                                  "email_only": "E-mail only"}[p])
        mcd = c6.number_input("Manual simulation alert cooldown (minutes)", 1, 1440, value=int(s["manual_cooldown_minutes"]),
                              help="Per simulation and recipient, for Send Emergency Alert → Email Alert / SMS to Mobile. Separate "
                                   "from the automatic incident cooldown; there is no override.")
        if st.form_submit_button("Save automatic-alert settings", type="primary"):
            probs = aa.save_settings({"enabled": bool(en), "min_confidence": float(mc), "cooldown_minutes": int(cd),
                                      "escalation_step": float(es), "policy": pol,
                                      "manual_cooldown_minutes": int(mcd)}, actor=_user())
            if probs:
                st.error("Configuration invalid: " + "; ".join(probs))
            else:
                st.rerun()

    sms_to, email_to = cfg.sms_recipients(), cfg.smtp_settings()["recipients"]
    bad_sms = [p for p in sms_to if not cfg.valid_phone(p)]
    st.caption("Automatic-alert recipients - SMS: " + (", ".join(store.mask(p) for p in sms_to) or "none")
               + " · e-mail: " + (", ".join(store.mask(e) for e in email_to) or "none")
               + ". Set permanently with ALERT_SMS_RECIPIENTS / ALERT_RECIPIENT_EMAILS in .env.")
    if bad_sms:
        _badge("warn", f"{len(bad_sms)} SMS recipient number(s) are not in E.164 format and will be skipped.")
    with st.expander("Change automatic-alert recipients for this session"):
        with st.form("auto_alert_recips"):
            sms_in = st.text_input("SMS recipients (E.164, comma separated)", value=",".join(sms_to),
                                   placeholder="+919812345678")
            email_in = st.text_input("E-mail recipients (comma separated)", value=",".join(email_to),
                                     placeholder="officer@forest.example")
            if st.form_submit_button("Save for this session"):
                ps = [p.strip() for p in sms_in.split(",") if p.strip()]
                es_ = [e.strip() for e in email_in.split(",") if e.strip()]
                bad = [store.mask(p) for p in ps if not cfg.valid_phone(p)] + [store.mask(e) for e in es_ if not cfg.valid_email(e)]
                if bad:
                    st.error("Not valid: " + ", ".join(bad) + " (numbers need the +country code, e.g. +9198…)")
                else:
                    cfg.set_runtime({"ALERT_SMS_RECIPIENTS": ",".join(ps), "ALERT_RECIPIENT_EMAILS": ",".join(es_)})
                    st.rerun()

    # current evaluation (preview only - nothing is sent from here)
    twin = st.session_state.get("twin")
    snap = getattr(twin, "current_snapshot", None)
    st.markdown("**Latest prediction vs. threshold** (preview of the next decision; nothing is sent from this page)")
    if snap is None:
        st.caption("No twin loaded in this session yet - open the Command Center once.")
    else:
        grid = snap.processed_grid
        probs = dict(zip(grid["zone_id"].astype(str), grid["fire_probability"])) if "fire_probability" in grid else {}
        scenario = getattr(getattr(twin, "ingestion", None), "scenario", None)
        real = not twin.offline and not scenario
        ext = [a for a in snap.alerts if a.severity == "EXTREME"]
        if not ext:
            st.caption(f"No EXTREME zone in the current {twin.region.name} prediction - no alert condition.")
        else:
            prev = []
            for a in sorted(ext, key=lambda a: -(aa.valid_confidence(probs.get(str(a.zone_id))) or 0))[:8]:
                c = aa.valid_confidence(probs.get(str(a.zone_id)))
                d = ("disabled" if not s["enabled"] else "skipped_not_real" if not real else
                     "skipped_invalid" if c is None else "suppressed_threshold" if c < s["min_confidence"] else
                     "eligible (cooldown checked at send time)")
                prev.append({"zone": a.zone_id, "location": f"{a.latitude:.3f}, {a.longitude:.3f}",
                             "fire probability": "-" if c is None else f"{c:.1%}",
                             "threshold": f"{s['min_confidence']:.0%}", "decision": aa.describe_decision(d)})
            st.dataframe(pd.DataFrame(prev), use_container_width=True, hide_index=True)
            if not real:
                st.caption("Current data: " + ("What-If scenario" if scenario else "demo / offline (synthetic)")
                           + " - automatic alerts are never sent for it.")

    st.markdown("**Automatic-alert history**")
    rows = aa.history(60)
    if not rows:
        st.caption("No automatic evaluation recorded yet (written after each refresh that has an EXTREME zone).")
    else:
        st.dataframe(pd.DataFrame([{
            "time (IST)": format_ist(r["timestamp_utc"]), "incident / location": f"{r['zone_id']} · "
            f"{(r['latitude'] or 0):.3f}, {(r['longitude'] or 0):.3f} · {r['region']}",
            "confidence": "-" if r["confidence"] is None else f"{r['confidence']:.1%}",
            "threshold": "-" if r["threshold"] is None else f"{r['threshold']:.0%}",
            "decision": aa.describe_row(r), "channels": r["channels"] or "-",
            "SMS": aa.describe_channel(r["sms_status"], "sms"), "e-mail": aa.describe_channel(r["email_status"], "email"),
            "alert": r["alert_id"] or "-", "note": r["note"] or ""} for r in rows]),
            use_container_width=True, hide_index=True)

    st.markdown("**Automatic incident cooldowns** (persistent; survives restarts)")
    states = aa.incident_states(30)
    if not states:
        st.caption("No incident has been alerted automatically yet.")
    else:
        st.dataframe(pd.DataFrame([{
            "incident (region | zone)": r["incident_key"],
            "last automatic alert": "- (failed, retry allowed)" if r["last_sent_utc"].startswith("1970")
            else format_ist(r["last_sent_utc"]),
            "probability then": "-" if r["last_confidence"] is None else f"{r['last_confidence']:.1%}",
            "next routine alert eligible": format_ist(r["next_routine_utc"]) if r["next_routine_utc"] else "now",
            "escalation if probability ≥": "-" if r["last_confidence"] is None
            else f"{min(1.0, r['last_confidence'] + s['escalation_step']):.0%}",
            "alert": r["last_alert_id"] or "-"} for r in states]), use_container_width=True, hide_index=True)

    st.markdown("**Manual simulation e-mail / SMS** (explicit *Send Emergency Alert*; logged separately from "
                "automatic alerts)")
    man = sorted([d for ch in ("sms", "email") for d in store.recent_deliveries(200, channel=ch)
                  if str(d["alert_id"]).startswith("MSIM-")], key=lambda d: d["timestamp_utc"], reverse=True)[:30]
    if not man:
        st.caption("No manual simulation e-mail or SMS sent yet.")
    else:
        st.dataframe(pd.DataFrame([{
            "time (IST)": format_ist(d["timestamp_utc"]), "alert": d["alert_id"], "simulation report": d["report_id"],
            "channel": d["channel"], "provider": d["provider"], "status": aa.describe_channel(d["status"], d["channel"]), "detail": d.get("error") or ""}
            for d in man]), use_container_width=True, hide_index=True)
