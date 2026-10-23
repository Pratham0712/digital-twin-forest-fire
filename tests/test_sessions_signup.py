"""Viewer sign-up, admin-only pages, session restore after a refresh,
10-minute inactivity timeout, activity-log user filter and terminal output,
credential e-mails, and the no-ignition guard for spread simulation."""
import time
from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest

DASH = Path(__file__).resolve().parents[1] / "src" / "dashboard"


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}")
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    from src.storage import database
    database.init_db()
    database.ensure_default_admin()
    return database


def _signed_in(page="app.py", role="admin", user="admin"):
    at = AppTest.from_file(str(DASH / page), default_timeout=240)
    at.session_state["auth_user"] = user
    at.session_state["auth_role"] = role
    at.session_state["_pref_offline_mode"] = True
    return at


# ── database: sessions, email column, activity filter ──────────────────────

def test_session_lifecycle_and_idle_expiry(db):
    token = db.create_session("admin", "admin", theme="light")
    s = db.get_session(token)
    assert s["username"] == "admin" and s["role"] == "admin" and s["theme"] == "light"
    assert db.get_session("not-a-token") is None
    assert db.get_session(token, max_idle_s=-1) is None              # idle too long -> expired and deleted
    assert db.get_session(token) is None
    token = db.create_session("admin", "admin")
    db.delete_session(token)
    assert db.get_session(token) is None


def test_session_follows_user_role_and_deletion(db):
    assert db.create_user("vic", "password123", "viewer", email="vic@example.com")
    token = db.create_session("vic", "viewer")
    db.set_user_role("vic", "admin")
    assert db.get_session(token)["role"] == "admin"
    db.delete_user("vic")
    assert db.get_session(token) is None


def test_users_store_email_and_old_databases_are_migrated(tmp_path, monkeypatch):
    import sqlite3
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username VARCHAR(80) UNIQUE NOT NULL, "
                "password_hash VARCHAR(128) NOT NULL, salt VARCHAR(64) NOT NULL, role VARCHAR(16) NOT NULL, "
                "created_utc VARCHAR(40) NOT NULL)")
    con.commit(); con.close()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{path}")
    from src.storage import database
    database.init_db()
    assert database.create_user("ann", "password123", "viewer", email="ann@example.com")
    assert [u["email"] for u in database.list_users() if u["username"] == "ann"] == ["ann@example.com"]


def test_activity_filter_by_user_and_terminal_output(db, capsys):
    db.log_activity("login", "Signed in (viewer)", actor="alice")
    db.log_activity("login", "Signed in (admin)", actor="admin")
    out = capsys.readouterr().out
    assert "[ACTIVITY]" in out and "alice" in out and "Signed in (viewer)" in out
    assert {r["actor"] for r in db.get_activity(actors=["alice"])} == {"alice"}
    assert {"alice", "admin"} <= set(db.get_activity_actors())


# ── login screen: viewer sign-up ────────────────────────────────────────────

def test_viewer_sign_up_creates_a_viewer(db):
    at = AppTest.from_file(str(DASH / "app.py"), default_timeout=240)
    at.run()
    assert [t.label for t in at.tabs] == ["Sign in", "Create account"]
    form = {t.label: t for t in at.text_input}
    form["Choose a username"].input("newviewer")
    form["Email"].input("new@example.com")
    at.text_input(key="signup_pw").input("goodpass123")
    at.text_input(key="signup_pw2").input("goodpass123")
    next(b for b in at.button if b.label == "Create viewer account").click().run()
    assert not at.exception
    users = {u["username"]: u for u in db.list_users()}
    assert users["newviewer"]["role"] == "viewer" and users["newviewer"]["email"] == "new@example.com"
    assert any("created" in s.value for s in at.success)


def test_sign_up_rejects_mismatched_passwords(db):
    at = AppTest.from_file(str(DASH / "app.py"), default_timeout=240)
    at.run()
    form = {t.label: t for t in at.text_input}
    form["Choose a username"].input("bob")
    form["Email"].input("bob@example.com")
    at.text_input(key="signup_pw").input("goodpass123")
    at.text_input(key="signup_pw2").input("different99")
    next(b for b in at.button if b.label == "Create viewer account").click().run()
    assert any("do not match" in e.value for e in at.error)
    assert "bob" not in {u["username"] for u in db.list_users()}


# ── sessions in the app ─────────────────────────────────────────────────────

def test_refresh_restores_session_from_cookie(db, monkeypatch):
    from src.auth import auth_gate
    token = db.create_session("admin", "admin", theme="light")
    monkeypatch.setattr(auth_gate, "_cookie_token", lambda: token)
    at = AppTest.from_file(str(DASH / "app.py"), default_timeout=240)
    at.session_state["_pref_offline_mode"] = True
    at.run()                                     # a fresh browser session (page refresh)
    assert not at.exception
    assert at.session_state["auth_user"] == "admin" and at.session_state["theme"] == "light"
    assert not [t for t in at.tabs if t.label == "Create account"]   # no login screen


def test_idle_session_is_signed_out(db):
    at = _signed_in()
    at.session_state["_last_active"] = time.time() - 11 * 60
    at.run()
    assert not at.exception
    assert "auth_user" not in at.session_state
    assert any("timed out" in i.value for i in at.info)


def test_light_theme_css_is_applied(db):
    at = _signed_in()
    at.session_state["theme"] = "light"
    at.run()
    md = "\n".join(m.value for m in at.markdown)
    assert "#DDEFFC" in md and [t.value for t in at.sidebar.toggle if t.label == "Light theme"] == [True]


# ── admin-only pages ────────────────────────────────────────────────────────

def test_activity_log_is_admin_only(db):
    at = _signed_in("pages/7_Activity_Log.py", role="viewer", user="vic")
    at.run()
    assert any("administrators only" in w.value for w in at.warning)
    at = _signed_in("pages/7_Activity_Log.py")
    at.run()
    assert not at.exception and [m.label for m in at.multiselect] == ["Event type", "User"]


def test_admin_pages_hidden_from_viewers(db):
    at = _signed_in(role="viewer", user="vic")
    at.run()
    md = "\n".join(m.value for m in at.markdown)
    assert 'li:has(a[href*="Activity_Log"])' in md and 'li:has(a[href*="Admin"])' in md


def test_admin_creates_admin_and_emails_credentials(db, monkeypatch):
    from src.notifications.email_notifier import EmailNotifier
    sent = {}
    monkeypatch.setattr(EmailNotifier, "send_credentials",
                        lambda self, to, u, p, r, url="": sent.update(to=to, user=u, pw=p, role=r) or True)
    at = _signed_in("pages/6_Admin.py")
    at.run()
    form = {t.label: t for t in at.text_input}
    form["Username"].input("admin2")
    form["Email (the sign-in details are sent here)"].input("admin2@example.com")
    next(b for b in at.button if b.label == "Create account and email the details").click().run()
    assert not at.exception
    assert sent["to"] == "admin2@example.com" and sent["role"] == "admin" and len(sent["pw"]) >= 8
    assert db.verify_user("admin2", sent["pw"]) == "admin"
    md = "\n".join(m.value for m in at.markdown)
    assert "Automatic data refresh" not in md
    assert not [m for m in at.get("metric") if "Database" in str(m.proto)]


# ── spread guard ────────────────────────────────────────────────────────────

def test_nothing_to_ignite_without_alerts_or_fires():
    import numpy as np
    import pandas as pd
    from types import SimpleNamespace
    from src.dashboard.geo_spread import nothing_to_ignite
    grid = pd.DataFrame({"active_fire_nearby": [False, False]})
    quiet = SimpleNamespace(current_snapshot=SimpleNamespace(alerts=[], processed_grid=grid,
                                                             risk_scores=np.array([0.1, 0.2])),
                            ingestion=SimpleNamespace(last_hotspots=pd.DataFrame()))
    assert nothing_to_ignite(quiet)
    high = SimpleNamespace(current_snapshot=SimpleNamespace(alerts=[SimpleNamespace(severity="HIGH")],
                                                            processed_grid=grid, risk_scores=np.array([0.9])),
                           ingestion=quiet.ingestion)
    assert not nothing_to_ignite(high)
    fire = SimpleNamespace(current_snapshot=quiet.current_snapshot,
                           ingestion=SimpleNamespace(last_hotspots=pd.DataFrame({"latitude": [13.0]})))
    assert not nothing_to_ignite(fire)


def test_unreachable_remote_database_is_retried(tmp_path, monkeypatch):
    """A remote database outage falls back to SQLite only temporarily: after
    FALLBACK_RETRY_SECONDS the configured remote database is tried again."""
    from src.storage import database
    monkeypatch.setenv("DATABASE_URL", "mysql+pymysql://u:p@127.0.0.1:1/x")
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "fallback.db")
    database.reset_fallback()
    database._INITIALISED.clear()
    database.init_db()                                   # connection refused -> local fallback
    assert database.fallback_reason() and database.database_url().startswith("sqlite")
    assert not database.is_persistent()
    monkeypatch.setattr(database, "FALLBACK_RETRY_SECONDS", 0)
    assert database.database_url().startswith("mysql")   # retried
    database.reset_fallback()
