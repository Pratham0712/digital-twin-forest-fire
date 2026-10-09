"""
Admin - user management, stored snapshot/alert history, notification setup
check. Admin-only page: viewers are redirected back with a message.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

import pandas as pd
import streamlit as st

from src.dashboard.dashboard_common import set_page, build_sidebar
from src.auth.auth_gate import require_login, is_admin, render_user_badge_in_sidebar, current_role
from src.storage import database as db
from src.notifications.email_notifier import EmailNotifier

set_page("Admin")
require_login()

if not is_admin():
    st.warning(
        f"You're signed in as **{current_role()}**. This page is restricted to admin "
        "accounts - user management and the raw database view are not viewer-level "
        "actions. Everything else in the dashboard is fully open to you."
    )
    st.stop()

offline, scenario, region, force_refresh = build_sidebar()
render_user_badge_in_sidebar()

st.markdown("""
<div class="hero">
  <div class="eyebrow">Forest Fire Digital Twin</div>
  <h1>Administration</h1>
  <div class="sub">User accounts, stored history, and notification configuration for this deployment.</div>
</div>
""", unsafe_allow_html=True)

db.init_db()


# Every tab body runs on each interaction, so database reads are cached briefly
# (a hosted MySQL costs a network round trip per query).
@st.cache_data(ttl=20, show_spinner=False)
def _users():
    return db.list_users()


@st.cache_data(ttl=20, show_spinner=False)
def _counts():
    return db.get_table_counts()


@st.cache_data(ttl=20, show_spinner=False)
def _snapshots(limit: int):
    return db.get_recent_snapshots(limit=limit)


@st.cache_data(ttl=20, show_spinner=False)
def _alerts(limit: int):
    return db.get_recent_alerts(limit=limit)


def _ist_table(rows) -> pd.DataFrame:
    """Stored rows for display: every *_utc column shown in IST (the database keeps UTC)."""
    from src.utils.timezone import format_ist_series
    df = pd.DataFrame(rows)
    for c in [c for c in df.columns if c.endswith("_utc")]:
        df[c] = format_ist_series(df[c])
        df = df.rename(columns={c: c[:-4] + " (IST)"})
    return df


tab_users, tab_history, tab_notify, tab_emerg = st.tabs(["Users", "Stored history", "Notifications",
                                                         "Emergency alerts"])

with tab_emerg:
    from src.dashboard.ui.alerts_ui import render_alert_history, render_recipient_admin
    st.subheader("Emergency-alert recipients")
    st.caption("Opt-in only: nothing is sent automatically. Alerts are sent from a completed simulation (Spread "
               "Simulation → Send Emergency Alert) after a review and an explicit confirmation, only to enabled "
               "recipients on verified channels.")
    render_recipient_admin()
    st.subheader("Alert history and delivery status")
    render_alert_history()

with tab_users:
    st.subheader("Accounts")
    users = _users()
    if users:
        st.dataframe(_ist_table(users)[["username", "role", "created (IST)"]],
                      use_container_width=True, hide_index=True)
    else:
        st.info("No users yet.")

    st.markdown("---")
    col_add, col_manage = st.columns(2)

    with col_add:
        st.markdown("**Add a user**")
        with st.form("add_user_form", clear_on_submit=True):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            new_role = st.selectbox("Role", ["viewer", "admin"])
            add_submitted = st.form_submit_button("Create account")
        if add_submitted:
            if not new_username or not new_password:
                st.error("Username and password are both required.")
            elif db.create_user(new_username.strip(), new_password, new_role):
                db.log_activity("admin", f"Created {new_role} account '{new_username.strip()}'",
                                actor=st.session_state.get("auth_user", "admin"))
                _users.clear(); _counts.clear()
                st.success(f"Created {new_role} account '{new_username.strip()}'.")
                st.rerun()
            else:
                st.error("That username already exists.")

    with col_manage:
        st.markdown("**Remove a user**")
        other_usernames = [u["username"] for u in users
                            if u["username"] != st.session_state.get("auth_user")]
        if other_usernames:
            with st.form("remove_user_form"):
                target = st.selectbox("Account", other_usernames)
                remove_submitted = st.form_submit_button("Delete account", type="secondary")
            if remove_submitted:
                db.delete_user(target)
                _users.clear(); _counts.clear()
                db.log_activity("admin", f"Deleted account '{target}'",
                                actor=st.session_state.get("auth_user", "admin"))
                st.success(f"Deleted '{target}'.")
                st.rerun()
        else:
            st.caption("No other accounts to remove. (You can't delete your own account "
                       "while signed in - sign in as another admin first.)")

with tab_history:
    counts = _counts()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Database", db.backend_name())
    m2.metric("Snapshots", f"{counts['snapshots']:,}")
    m3.metric("Alert records", f"{counts['alerts_log']:,}")
    m4.metric("Activity events", f"{counts['activity_log']:,}")
    m5.metric("Users", f"{counts['users']:,}")

    st.subheader("Recent refresh snapshots")
    snaps = _snapshots(30)
    if snaps:
        st.dataframe(_ist_table(snaps), use_container_width=True, hide_index=True)
    else:
        st.info("No snapshots recorded yet - they're written automatically every time "
                "the dashboard (or the background scheduler) refreshes the twin.")

    st.subheader("Recent alerts")
    alerts = _alerts(100)
    if alerts:
        st.dataframe(_ist_table(alerts), use_container_width=True, hide_index=True)
    else:
        st.info("No alerts recorded yet.")

with tab_notify:
    st.subheader("Email notifications")
    notifier = EmailNotifier()
    if notifier.is_configured:
        st.success(
            f"Configured - alerts for newly-EXTREME zones are sent to "
            f"{len(notifier.recipients)} recipient(s) via {notifier.smtp_host}."
        )
    else:
        st.warning("Not configured yet.")
        with st.expander("Configure email alerts"):
            st.markdown(
                "Add the mail server (SMTP host and port), the sender login (SMTP user and "
                "app password) and the list of recipient emails to the app's settings "
                "(`.env` locally, Secrets on Streamlit Cloud).\n\n"
                "For Gmail, use an App Password (Google Account > Security > App passwords), "
                "not your normal password."
            )

    st.markdown("---")
    st.markdown("**Send a test alert now**")
    st.caption(
        "Sends a real alert email immediately, without waiting for an actual EXTREME "
        "zone to be detected. Does not affect any stored data."
    )
    if st.button("Send test alert email", disabled=not notifier.is_configured):
        from datetime import datetime, timezone
        from src.digital_twin.twin_state import ZoneAlert
        demo_alert = ZoneAlert(
            zone_id="DEMO-0001", latitude=13.35, longitude=75.20, risk_score=0.91,
            severity="EXTREME",
            reason="extreme FWI (78.4); low humidity (11%); high wind (11.2 m/s) "
                   "[TEST ALERT - triggered manually from the Admin page, not a real detection]",
            triggered_at=datetime.now(timezone.utc).isoformat(),
        )
        with st.spinner("Sending..."):
            sent = notifier.send_new_extreme_alert("Karnataka Western Ghats (TEST)", [demo_alert])
        db.log_activity("email_sent" if sent else "email_failed",
                        "Test alert email " + ("sent" if sent else "could not be sent"),
                        actor=st.session_state.get("auth_user", "admin"))
        if sent:
            st.success(f"Sent - check the inbox for {', '.join(notifier.recipients)}.")
        else:
            st.error("Send failed - check the SMTP settings above (host, port, and app password).")
    if not notifier.is_configured:
        st.caption("Configure SMTP above first - the button unlocks once all four variables are set.")

    st.subheader("Automatic data refresh")
    interval = __import__('config.config', fromlist=['SYSTEM']).SYSTEM.min_refresh_interval_minutes
    st.success(f"Data refreshes automatically every **{interval} minutes**.")

    with st.expander("Advanced options"):
        st.markdown(
            "Refreshes can also run as a separate background process (the scheduler module) "
            "instead of inside the dashboard."
        )
