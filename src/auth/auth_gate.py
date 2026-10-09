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
import threading
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st

from src.storage import database as db


_READY_URLS: set = set()


_INIT_LOCK = threading.Lock()


def _ensure_db_ready():
    # Once per process and database (not per browser session): each of these
    # is several round trips on a hosted MySQL. The lock lets the background
    # warm-up and a foreground caller share one run instead of doubling it.
    with _INIT_LOCK:
        url = db.database_url()
        if url not in _READY_URLS:
            db.init_db()
            db.ensure_default_admin()
            _READY_URLS.add(url)


def _warm_up_in_background():
    """Database set-up and the default region's data load while the person is
    still typing their password, instead of before the login form appears."""
    if _READY_URLS or getattr(_warm_up_in_background, "started", False):
        return
    _warm_up_in_background.started = True

    def _run():
        try:
            _ensure_db_ready()
        except Exception:
            pass
        try:
            from src.dashboard.dashboard_common import prewarm_default_twin
            prewarm_default_twin()
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True, name="login-warmup").start()


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
    if is_logged_in():
        _ensure_db_ready()
        _render_db_status()
        return
    if db.database_url() in _READY_URLS:
        _render_db_status()
    else:
        _warm_up_in_background()          # form appears now; set-up runs behind it

    # Login screen: cinematic background + glass card (presentation only; the
    # credential check below is unchanged). There is no self-registration, so
    # "Create account" explains how accounts are issued instead of faking one.
    from src.dashboard.ui.login import BRAND_HTML, CREATE_ACCOUNT_HTML, FOOT_HTML, login_css
    st.markdown(login_css(), unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.25, 1])
    with mid:
        with st.container(key="login_card"):
            st.markdown(BRAND_HTML, unsafe_allow_html=True)
            if db.fallback_reason():
                st.caption("The cloud database is not reachable; this session uses the local database.")
            tab_in, tab_new = st.tabs(["Sign in", "Create account"])
            with tab_in:
                with st.form("login_form", clear_on_submit=False):
                    username = st.text_input("Username", placeholder="Enter your username")
                    password = st.text_input("Password", type="password", placeholder="Enter your password")
                    submitted = st.form_submit_button("Sign in", use_container_width=True)
                if submitted:
                    _ensure_db_ready()             # waits for the background set-up if still running
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
            with tab_new:
                st.markdown(CREATE_ACCOUNT_HTML, unsafe_allow_html=True)
            st.markdown(FOOT_HTML, unsafe_allow_html=True)
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
