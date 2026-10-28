"""
auth_gate.py - role-based login gate for the Streamlit dashboard.

Design:
  - Two roles: 'admin' and 'viewer'. Viewers can sign themselves up on the
    login screen; admin accounts are created only by an existing admin on the
    Admin page, which e-mails the new admin their sign-in details.
  - Admin-only pages (Admin, Activity Log) are hidden from viewers.
  - Sessions survive a page refresh: on sign-in a random token is stored in a
    browser cookie (only its SHA-256 is kept in the database), and a refreshed
    page restores the session from it.
  - Inactivity timeout: a session with no user interaction for
    SESSION_IDLE_SECONDS (10 min) is signed out and the login screen returns.
    Background auto-refreshes do not count as activity.
  - Backed by src.storage.database, so the user and session tables are shared
    by every page.
"""
import re
import sys
import threading
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
import streamlit.components.v1 as components

from src.storage import database as db

COOKIE = "ffdt_sid"
IDLE_SECONDS = db.SESSION_IDLE_SECONDS
_TOUCH_EVERY_S = 60                      # database last-seen update, at most once a minute
ADMIN_PAGES = ("Admin", "Activity_Log")

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


# ── browser cookie ──────────────────────────────────────────────────────────

def _cookie_js(value: str, max_age: int) -> str:
    # components.html runs in a same-origin iframe, so the cookie is written on the
    # app's own document (window.parent).
    return f"""<script>
(function(){{
  var sec = (window.parent.location.protocol === 'https:') ? '; Secure' : '';
  window.parent.document.cookie = '{COOKIE}={value}; path=/; max-age={max_age}; SameSite=Lax' + sec;
}})();
</script>"""


def _set_cookie(token: str):
    components.html(_cookie_js(token, 7 * 24 * 3600), height=0)


def _clear_cookie():
    components.html(_cookie_js("", 0), height=0)


def _cookie_token() -> str:
    try:
        return st.context.cookies.get(COOKIE, "") or ""
    except Exception:
        return ""


# ── session state ───────────────────────────────────────────────────────────

def is_logged_in() -> bool:
    return bool(st.session_state.get("auth_user"))


def current_role() -> str:
    return st.session_state.get("auth_role", "viewer")


def is_admin() -> bool:
    return current_role() == "admin"


def _start_session(username: str, role: str, theme: str = "dark", token: str = None):
    st.session_state["auth_user"] = username
    st.session_state["auth_role"] = role
    st.session_state["theme"] = theme if theme in ("dark", "light") else "dark"
    st.session_state["_last_active"] = time.time()
    st.session_state["_last_touch"] = time.time()
    if token is None:
        token = db.create_session(username, role, st.session_state["theme"])
    st.session_state["_sid"] = token
    st.session_state["_cookie_pending"] = True


def _end_session():
    db.delete_session(st.session_state.get("_sid", ""))
    for k in ("auth_user", "auth_role", "_sid", "_last_active", "_last_touch", "_cookie_pending"):
        st.session_state.pop(k, None)
    st.session_state["_clear_cookie"] = True


def logout():
    user = st.session_state.get("auth_user")
    if user:
        db.log_activity("logout", "Signed out", actor=user)
    _end_session()


def _expire():
    user = st.session_state.get("auth_user")
    if user:
        db.log_activity("logout", f"Session timed out after {IDLE_SECONDS // 60} min of inactivity", actor=user)
    _end_session()
    st.session_state["_session_expired"] = True


def _idle_seconds() -> float:
    return time.time() - float(st.session_state.get("_last_active", time.time()))


def _record_activity():
    """Called on every full page run. Background auto-refresh reruns set
    _auto_rerun and are not user activity."""
    if st.session_state.pop("_auto_rerun", False):
        return
    now = time.time()
    st.session_state["_last_active"] = now
    if now - float(st.session_state.get("_last_touch", 0)) >= _TOUCH_EVERY_S:
        st.session_state["_last_touch"] = now
        db.touch_session(st.session_state.get("_sid", ""))


def set_theme(theme: str):
    """Stores the chosen theme for this session (kept across page refreshes)."""
    st.session_state["theme"] = theme
    db.touch_session(st.session_state.get("_sid", ""), theme=theme)


def _try_restore() -> bool:
    """Restores a signed-in session after a page refresh from the browser cookie."""
    token = _cookie_token()
    if not token:
        return False
    try:
        _ensure_db_ready()
        sess = db.get_session(token)
    except Exception:
        return False
    if not sess:
        st.session_state["_clear_cookie"] = True
        return False
    _start_session(sess["username"], sess["role"], sess.get("theme") or "dark", token=token)
    st.session_state["_cookie_pending"] = False       # the browser already has this cookie
    db.touch_session(token)
    return True


@st.fragment(run_every=30)
def _session_watch():
    """Signs the session out after IDLE_SECONDS without interaction, even when
    the person is not clicking anything (the login screen then appears)."""
    if is_logged_in() and _idle_seconds() > IDLE_SECONDS:
        _expire()
        st.rerun(scope="app")


def _sync_light_flag(light: bool):
    """Marks the page with data-ff-sky="1" while the Light theme is on. The header "⋮" menu
    (components/theme_switch) then keeps its Navy colours off, so the two never mix; turning
    Light off restores the theme saved in that menu."""
    with st.container(key="ff_light_flag"):          # hidden by theme_switch CSS: takes no space
        _light_flag_script(light)


def _light_flag_script(light: bool):
    components.html(f"""<script>
(function(){{
  try {{
    var P = window.parent, r = P.document.documentElement;
    if ({'true' if light else 'false'}) r.setAttribute('data-ff-sky', '1'); else r.removeAttribute('data-ff-sky');
    if (P.__ffTheme) P.__ffTheme.apply(P.__ffTheme.get());
  }} catch (e) {{}}
}})();
</script>""", height=0)


def _hide_admin_pages_css():
    sel = ", ".join(f'[data-testid="stSidebarNav"] li:has(a[href*="{p}"])' for p in ADMIN_PAGES)
    st.markdown(f"<style>{sel} {{ display: none !important; }}</style>", unsafe_allow_html=True)


def require_login():
    """Call this as the first thing on every page (right after set_page()).
    Renders the login / sign-up screen and st.stop()s the page until the user
    is authenticated."""
    if st.session_state.pop("_clear_cookie", False):
        _clear_cookie()

    if is_logged_in() and _idle_seconds() > IDLE_SECONDS:
        _expire()
        st.rerun()

    if not is_logged_in():
        _try_restore()

    if is_logged_in():
        _ensure_db_ready()
        _record_activity()
        if st.session_state.pop("_cookie_pending", False):
            _set_cookie(st.session_state["_sid"])
        light = st.session_state.get("theme") == "light"
        if light:
            from src.dashboard.ui.theme import LIGHT_THEME_CSS
            st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)
        _sync_light_flag(light)
        if not is_admin():
            _hide_admin_pages_css()
        _session_watch()
        return

    if db.database_url() not in _READY_URLS:
        _warm_up_in_background()          # form appears now; set-up runs behind it
    _render_login_screen()
    st.stop()


# ── login / sign-up screen ──────────────────────────────────────────────────

_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,30}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _render_login_screen():
    from src.dashboard.ui.login import BRAND_HTML, FOOT_HTML, SIGNUP_NOTE_HTML, login_css
    st.markdown(login_css(), unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.25, 1])
    with mid:
        with st.container(key="login_card"):
            st.markdown(BRAND_HTML, unsafe_allow_html=True)
            if st.session_state.pop("_session_expired", False):
                st.info(f"Your session timed out after {IDLE_SECONDS // 60} minutes of inactivity. "
                        "Please sign in again.")
            if st.session_state.get("_signup_done"):
                st.success(f"Account '{st.session_state.pop('_signup_done')}' created. Sign in below.")
            tab_in, tab_new = st.tabs(["Sign in", "Create account"])
            with tab_in:
                _render_sign_in()
            with tab_new:
                _render_sign_up(SIGNUP_NOTE_HTML)
            st.markdown(FOOT_HTML, unsafe_allow_html=True)


def _render_sign_in():
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        st.markdown("<div class='lg-hint'>Press Enter or select Sign in</div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Sign in", use_container_width=True)
    if submitted:
        _ensure_db_ready()             # waits for the background set-up if still running
        role = db.verify_user(username.strip(), password)
        if role:
            _start_session(username.strip(), role)
            db.log_activity("login", f"Signed in ({role})", actor=username.strip())
            st.rerun()
        else:
            db.log_activity("login_failed", "Incorrect username or password",
                            actor=username.strip()[:80] or "unknown")
            st.error("Incorrect username or password.")


def _render_sign_up(note_html: str):
    st.markdown(note_html, unsafe_allow_html=True)
    with st.form("signup_form", clear_on_submit=False):
        username = st.text_input("Choose a username", placeholder="3-30 letters, digits, . _ -")
        email = st.text_input("Email", placeholder="you@example.com")
        password = st.text_input("Password", type="password", placeholder="At least 8 characters",
                                 key="signup_pw")
        confirm = st.text_input("Confirm password", type="password", placeholder="Repeat the password",
                                key="signup_pw2")
        submitted = st.form_submit_button("Create viewer account", use_container_width=True)
    if not submitted:
        return
    username, email = username.strip(), email.strip()
    if not _USERNAME_RE.match(username):
        st.error("Username must be 3-30 characters: letters, digits, dot, underscore or hyphen.")
    elif not _EMAIL_RE.match(email):
        st.error("Enter a valid email address.")
    elif len(password) < 8:
        st.error("Password must be at least 8 characters.")
    elif password != confirm:
        st.error("The two passwords do not match.")
    else:
        _ensure_db_ready()
        if db.create_user(username, password, role="viewer", email=email):
            db.log_activity("signup", "Viewer account created by self sign-up", actor=username)
            st.session_state["_signup_done"] = username
            st.rerun()
        else:
            st.error("That username is already taken.")


# ── sidebar badge ───────────────────────────────────────────────────────────

def render_user_badge_in_sidebar():
    """Sidebar footer showing who's logged in + a sign-out button."""
    if not is_logged_in():
        return
    st.sidebar.markdown("---")
    role_label = "Admin" if is_admin() else "Viewer"
    st.sidebar.markdown(
        f"**{st.session_state['auth_user']}** &nbsp; "
        f"<span class='role-pill'>{role_label}</span>",
        unsafe_allow_html=True,
    )
    if st.sidebar.button("Sign out", use_container_width=True, key="sign_out_btn"):
        logout()
        st.rerun()
