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
    render_recipient_admin()
    st.subheader("Alert history and delivery status")
    render_alert_history()

with tab_users:
    st.subheader("Accounts")
    users = _users()
    if users:
        _udf = _ist_table(users)
        _udf["email"] = _udf["email"].fillna("—") if "email" in _udf else "—"
        st.dataframe(_udf[["username", "role", "email", "created (IST)"]],
                      use_container_width=True, hide_index=True)
    else:
        st.info("No users yet.")

    st.markdown("---")
    col_add, col_manage = st.columns(2)

    with col_add:
        st.markdown("**Add a user**")
        st.caption("Admin accounts are created here only. The sign-in details are emailed to the new user. "
                   "Leave the password blank to generate a strong one.")
        with st.form("add_user_form", clear_on_submit=True):
            new_username = st.text_input("Username")
            new_email = st.text_input("Email (the sign-in details are sent here)")
            new_role = st.selectbox("Role", ["admin", "viewer"])
            new_password = st.text_input("Password (optional)", type="password")
            add_submitted = st.form_submit_button("Create account and email the details")
        if add_submitted:
            import re
            import secrets
            uname, mail = new_username.strip(), new_email.strip()
            pw = new_password or secrets.token_urlsafe(9)
            if not re.match(r"^[A-Za-z0-9_.-]{3,30}$", uname):
                st.error("Username must be 3-30 characters: letters, digits, dot, underscore or hyphen.")
            elif not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", mail):
                st.error("Enter the new user's email address - the sign-in details are sent there.")
            elif len(pw) < 8:
                st.error("Password must be at least 8 characters (or leave it blank to generate one).")
            elif db.admin_create_user(st.session_state.get("auth_user", ""), uname, pw, new_role, email=mail):
                actor = st.session_state.get("auth_user", "admin")
                _users.clear(); _counts.clear()
                import os
                notifier_ = EmailNotifier()
                with st.spinner("Emailing the sign-in details..."):
                    sent = notifier_.send_credentials(mail, uname, pw, new_role, os.getenv("APP_URL", "").strip())
                db.log_activity("admin", f"Sign-in details for '{uname}' "
                                + (f"emailed to {mail}" if sent else "not emailed (send failed)"), actor=actor)
                if sent:
                    st.success(f"Created {new_role} account '{uname}' and emailed the sign-in details to {mail}.")
                else:
                    st.warning(f"Created {new_role} account '{uname}', but the email could not be sent "
                               "(SMTP not configured or the send failed). Share these details privately:")
                    st.code(f"Username: {uname}\nPassword: {pw}", language=None)
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
    from src.dashboard.ui.alerts_ui import render_auto_alerts, render_provider_config
    render_provider_config()
    st.markdown("---")
    render_auto_alerts()
