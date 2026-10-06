"""
auth_gate.py - role-based login gate for the Streamlit dashboard
(cross-question 14: "only certain privileged users - admin vs viewer -
should be able to access the system, not everyone").

Design:
  - Two roles: 'admin' and 'viewer'. Both can view every page (this is a
    monitoring dashboard - hiding the risk map from a viewer defeats the
    point of a public-safety tool). What's role-gated is anything that
    CHANGES system state: editing user accounts, and forcing a live
    (non-offline) refresh that spends real API quota.
  - Session-based: once logged in, st.session_state["auth_user"] /
    ["auth_role"] persist for the rest of that browser session (matches how
    every other page already keeps twin state in st.session_state).
  - Backed by src.storage.database (verify_user / create_user), so this is
    one shared user table across every page, not per-page state.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st

from src.storage import database as db


_READY_URLS: set = set()


def _ensure_db_ready():
    # Once per process and database (not per browser session): each of these
    # is several round trips on a hosted MySQL.
    url = db.database_url()
    if url not in _READY_URLS:
        db.init_db()
        db.ensure_default_admin()
        _READY_URLS.add(url)


def _render_db_status():
    """When the configured MySQL cannot be reached the app runs on the local
    SQLite file; say so, and let the user retry the remote database."""
    reason = db.fallback_reason()
    if not reason:
        return
    st.sidebar.warning(
        "The cloud database is not reachable right now, so this session is using a local "
        "database. Users and history saved now stay on this computer.")
    if st.sidebar.button("Retry cloud database", use_container_width=True, key="db_retry_btn"):
        db.reset_fallback()
        _READY_URLS.clear()
        st.rerun()


def is_logged_in() -> bool:
    return bool(st.session_state.get("auth_user"))


def current_role() -> str:
    return st.session_state.get("auth_role", "viewer")


def is_admin() -> bool:
    return current_role() == "admin"


def logout():
    user = st.session_state.get("auth_user")
    if user:
        db.log_activity("logout", "Signed out", actor=user)
    for k in ("auth_user", "auth_role"):
        st.session_state.pop(k, None)


def require_login():
    """Call this as the first thing on every page (right after set_page()).
    Renders a login form and st.stop()s the page until the user is
    authenticated - matches the pattern every page already uses for
    set_page()/build_sidebar() being mandatory first calls."""
    _ensure_db_ready()
    _render_db_status()

    if is_logged_in():
        return

    _, mid, _ = st.columns([1, 1.5, 1])
    with mid:
        st.markdown(
            "<div class='hero' style='margin-top:48px;'>"
            "<div class='eyebrow'>BMS College of Engineering</div>"
            "<h1>Forest Fire Digital Twin</h1>"
            "<div class='sub'>Sign in to open the command center.</div></div>",
            unsafe_allow_html=True,
        )
    with mid:
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", use_container_width=True)

        if submitted:
            role = db.verify_user(username.strip(), password)
            if role:
                st.session_state["auth_user"] = username.strip()
                st.session_state["auth_role"] = role
                db.log_activity("login", f"Signed in ({role})", actor=username.strip())
                st.rerun()
            else:
                db.log_activity("login_failed", "Incorrect username or password",
                                actor=username.strip()[:80] or "unknown")
                st.error("Incorrect username or password.")
    st.stop()


def render_user_badge_in_sidebar():
    """Small sidebar footer showing who's logged in + a logout button.
    Call after build_sidebar() on any page that wants it (app.py does)."""
    if not is_logged_in():
        return
    st.sidebar.markdown("---")
    role_label = "Admin" if is_admin() else "Viewer"
    st.sidebar.markdown(
        f"**{st.session_state['auth_user']}** &nbsp; "
        f"<span style='background:rgba(255,107,53,.14);color:#ff9a3c;border:1px solid rgba(255,107,53,.35);"
        f"padding:2px 10px;border-radius:999px;font-size:11px;font-weight:700;letter-spacing:.08em;"
        f"text-transform:uppercase;'>{role_label}</span>",
        unsafe_allow_html=True,
    )
    if st.sidebar.button("Sign out", use_container_width=True):
        logout()
        st.rerun()
