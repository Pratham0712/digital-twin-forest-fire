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


def _ensure_db_ready():
    # Cheap to call repeatedly - CREATE TABLE IF NOT EXISTS - but only do the
    # real work once per process via a session_state flag.
    if not st.session_state.get("_db_initialised"):
        db.init_db()
        db.ensure_default_admin()
        st.session_state["_db_initialised"] = True


def is_logged_in() -> bool:
    return bool(st.session_state.get("auth_user"))


def current_role() -> str:
    return st.session_state.get("auth_role", "viewer")


def is_admin() -> bool:
    return current_role() == "admin"


def logout():
    for k in ("auth_user", "auth_role"):
        st.session_state.pop(k, None)


def require_login():
    """Call this as the first thing on every page (right after set_page()).
    Renders a login form and st.stop()s the page until the user is
    authenticated - matches the pattern every page already uses for
    set_page()/build_sidebar() being mandatory first calls."""
    _ensure_db_ready()

    if is_logged_in():
        return

    st.markdown(
        "<div style='max-width:420px;margin:64px auto 0 auto;'>"
        "<h2 style='margin-bottom:4px;'>Sign in</h2>"
        "<p style='color:#8b949e;margin-top:0;'>"
        "Digital Twin Framework for Forest Fire Prediction &mdash; "
        "BMS College of Engineering</p></div>",
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([1, 1.4, 1])
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
                st.rerun()
            else:
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
        f"<span style='background:#30363d;padding:2px 8px;border-radius:10px;"
        f"font-size:12px;'>{role_label}</span>",
        unsafe_allow_html=True,
    )
    if st.sidebar.button("Sign out", use_container_width=True):
        logout()
        st.rerun()
