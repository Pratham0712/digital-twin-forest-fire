"""Confidential login (no self-registration), e-mail-only recipient verification (SMS verification removed) and
SMS alerts that do not depend on verified phones. Offline: every provider is a fake; nothing is sent."""
from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.notifications.providers import SendResult
from tests.test_sms_auto_alerts import KEY, FakeHTTP, Resp, OK_SEND, _tb, _textbee_on, env  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


# ── login / accounts ────────────────────────────────────────────────────────

@pytest.fixture()
def users_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'u.db'}")
    from src.storage import database as db
    db.init_db()
    db.create_user("boss", "Adm1n-pass", role="admin")                 # existing accounts (test fixture)
    db.create_user("watcher", "View-pass1", role="viewer")
    return db


def test_no_unauthenticated_or_non_admin_account_creation(users_db):
    db = users_db
    for actor in ("", None, "nobody", "watcher"):
        with pytest.raises(PermissionError):
            db.admin_create_user(actor, "intruder", "pw123456", "admin")
    assert db.verify_user("intruder", "pw123456") is None
    assert {u["username"] for u in db.list_users()} == {"boss", "watcher"}


def test_admin_can_still_create_accounts_and_existing_ones_stay(users_db):
    db = users_db
    assert db.admin_create_user("boss", "newviewer", "Pass-1234", "viewer") is True
    assert db.admin_create_user("boss", "newviewer", "other", "viewer") is False        # duplicate refused
    assert db.verify_user("newviewer", "Pass-1234") == "viewer"
    assert db.verify_user("boss", "Adm1n-pass") == "admin" and db.verify_user("watcher", "View-pass1") == "viewer"
    db.ensure_default_admin()                                                           # users exist: no new admin
    assert {u["username"] for u in db.list_users()} == {"boss", "watcher", "newviewer"}


def test_login_page_has_no_registration_and_admin_uses_guarded_path():
    from src.auth import auth_gate
    src = inspect.getsource(auth_gate.require_login)
    assert "st.tabs(" not in src and "Create account" not in src and "create_user(" not in src
    login = (ROOT / "src/dashboard/ui/login.py").read_text(encoding="utf-8")
    assert "CREATE_ACCOUNT_HTML" not in login
    admin = (ROOT / "src/dashboard/pages/6_Admin.py").read_text(encoding="utf-8")
    assert "db.admin_create_user(" in admin and "db.create_user(" not in admin
    # every page goes through the login gate (no unauthenticated route)
    for page in [ROOT / "src/dashboard/app.py", *sorted((ROOT / "src/dashboard/pages").glob("*.py"))]:
        assert "require_login" in page.read_text(encoding="utf-8"), page.name


def test_admin_page_unauthenticated_shows_only_sign_in(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'p.db'}")
    st_testing = pytest.importorskip("streamlit.testing.v1")
    at = st_testing.AppTest.from_file(str(ROOT / "src/dashboard/pages/6_Admin.py"), default_timeout=120)
    at.query_params["register"] = "1"                                # a query parameter changes nothing
    at.run()
    assert not at.exception and not at.tabs
    assert [b.label for b in at.button] == ["Sign in"]
    assert "Accounts" not in " ".join(m.value for m in at.markdown) and not at.dataframe


# ── phone numbers ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,norm", [("9812345678", "+919812345678"), ("098123 45678", "+919812345678"),
                                      ("+91 98123-45678", "+919812345678"), ("919812345678", "+919812345678"),
                                      ("0091 9812345678", "+919812345678"), ("+447700900123", "+447700900123")])
def test_phone_normalisation(raw, norm):
    from src.notifications.config import normalise_phone
    assert normalise_phone(raw) == norm


@pytest.mark.parametrize("bad", ["12345", "5812345678", "+9112345", "+91 581234 5678", "98123x5678", "", None])
def test_invalid_phone_rejected(bad, env):                                # noqa: F811
    from src.notifications import alert_store as store
    from src.notifications.config import normalise_phone
    assert normalise_phone(bad) is None
    if bad:
        with pytest.raises(ValueError):
            store.add_recipient("X", phone=bad)


# ── verification (e-mail only) ─────────────────────────────────────────────

class Spy:
    def __init__(self, status="accepted", error=None):
        self.status, self.error, self.sent = status, error, []

    def send(self, to, *a, **k):
        self.sent.append((to, a))
        return SendResult(self.status, "fake", "m-1" if self.status == "accepted" else None, self.error)


def _code_of(spy):
    import re
    return re.search(r"\b(\d{6})\b", " ".join(map(str, spy.sent[-1][1]))).group(1)


def _smtp(cfg):
    cfg.set_runtime({"SMTP_HOST": "smtp.example.org", "SMTP_PORT": "587", "SMTP_USER": "bot@example.org",
                     "SMTP_PASSWORD": "app-pass-123", "SMTP_FROM": "bot@example.org"})


@pytest.fixture()
def rec(env):                                                             # noqa: F811
    from src.notifications import alert_store as store
    _textbee_on(env)
    _smtp(env)
    a = store.add_recipient("Officer A", email="a@forest.gov.in", phone="9812345678", prefs={"email": True, "sms": True})
    b = store.add_recipient("Officer B", email="b@forest.gov.in", phone="+91 99000 11122", prefs={"sms": True})
    return store, a, b


def test_email_verification_works_and_needs_the_code(rec):
    store, a, _ = rec
    from src.notifications import emergency as em
    ep = Spy()
    out = em.send_verification(a, "email", "admin", "admin", email_provider=ep)
    assert out["status"] == "accepted" and "delivery not confirmed" in out["message"]
    assert ep.sent[0][0] == "a@forest.gov.in" and "not a fire alert" in ep.sent[0][1][1]
    code = _code_of(ep)
    assert code not in out["message"] and store.get_recipient(a)["email_verified"] == 0   # accepted != verified
    assert store.check_verification(a, "email", code) == (True, "Verified.")
    r = store.get_recipient(a)
    assert r["email_verified"] == 1 and r["phone_verified"] == 0          # the phone is never marked verified
    assert store.check_verification(a, "email", code)[0] is False         # single use


def test_sms_verification_is_gone_everywhere(rec):
    store, a, _ = rec
    from src.notifications import emergency as em
    from src.dashboard.ui import alerts_ui
    with pytest.raises(ValueError):
        em.send_verification(a, "sms", "admin", "admin")
    with pytest.raises(ValueError):
        store.issue_verification(a, "sms", "+919812345678")
    with pytest.raises(ValueError):
        store.create_verification(a, "sms")
    assert store.check_verification(a, "sms", "123456") == (False, "Only e-mail addresses can be verified.")
    assert store.confirm_verification(a, "sms", "123456") is False
    assert em.verification_readiness("sms")["ok"] is False
    assert not hasattr(em, "verification_sms_status") and not hasattr(em, "VERIFY_SMS")
    src = inspect.getsource(alerts_ui.render_recipient_admin)
    for gone in ("Send sms verification code", "Check SMS status", '"sms")', "selectbox(\"Channel\"", "phone_verified"):
        assert gone not in src, gone
    assert store.get_recipient(a)["phone_verified"] == 0
    store.update_recipient(a, phone_verified=1)                           # no code path can set it any more
    assert store.get_recipient(a)["phone_verified"] == 0


def test_admin_page_shows_email_only_verification(tmp_path, monkeypatch, env):   # noqa: F811
    st_testing = pytest.importorskip("streamlit.testing.v1")
    from src.notifications import alert_store as store
    store.add_recipient("Officer A", email="a@forest.gov.in", phone="9812345678", prefs={"email": True, "sms": True})
    page = """
import streamlit as st
from src.dashboard.ui import alerts_ui
alerts_ui.render_recipient_admin()
"""
    at = st_testing.AppTest.from_string(page, default_timeout=60)
    at.session_state["auth_user"] = "boss"
    at.session_state["auth_role"] = "admin"
    at.run()
    assert not at.exception
    labels = [b.label for b in at.button]
    assert "Send email verification code" in labels and "Verify" in labels
    assert not any("sms" in l.lower() and "verif" in l.lower() for l in labels) and "Check SMS status" not in labels
    assert not at.selectbox                                               # no channel choice
    text = " ".join(m.value for m in at.markdown)
    assert "a***@forest.gov.in - NOT verified" in text and "***678 (SMS alerts on)" in text
    assert "Mobile ***678 - " not in text                                 # never implies a verified phone


def test_wrong_expired_locked_and_resend_limits(rec):
    store, a, _ = rec
    from src.notifications import emergency as em
    t0 = datetime.now(timezone.utc)
    ep = Spy()
    em.send_verification(a, "email", "admin", "admin", email_provider=ep, now=t0)
    code = _code_of(ep)
    wrong = "000000" if code != "000000" else "111111"
    ok, why = store.check_verification(a, "email", wrong, now=t0)
    assert not ok and "4 attempt(s) left" in why
    ok, why = store.check_verification(a, "email", code, now=t0 + timedelta(minutes=11))
    assert not ok and "expired" in why and store.get_recipient(a)["email_verified"] == 0
    again = em.send_verification(a, "email", "admin", "admin", email_provider=ep, now=t0 + timedelta(seconds=20))
    assert again["status"] == "rate_limited" and again["wait_s"] == 40
    em.send_verification(a, "email", "admin", "admin", email_provider=ep, now=t0 + timedelta(minutes=12))
    code2 = _code_of(ep)
    w2 = "000000" if code2 != "000000" else "111111"
    for _ in range(5):
        ok, why = store.check_verification(a, "email", w2, now=t0 + timedelta(minutes=12))
    assert "Too many incorrect attempts" in why
    assert store.check_verification(a, "email", code2, now=t0 + timedelta(minutes=12))[0] is False


def test_failed_or_unconfigured_email_is_not_reported_as_sent(rec, env):  # noqa: F811
    store, a, _ = rec
    from src.notifications import emergency as em
    out = em.send_verification(a, "email", "admin", "admin", email_provider=Spy("failed", error="SMTP refused"))
    assert out["status"] == "failed" and "NOT sent" in out["message"]
    assert store.check_verification(a, "email", "123456")[0] is False     # the failed code was invalidated
    env.set_runtime({"ALERT_PROVIDER_MODE": "mock"})
    out = em.send_verification(a, "email", "admin", "admin", email_provider=Spy())
    assert out["status"] == "not_configured" and out["message"].startswith("Live sending is disabled")
    with pytest.raises(PermissionError):
        em.send_verification(a, "email", "viewer", "viewer", email_provider=Spy())


def test_changed_address_needs_new_verification(rec):
    store, a, _ = rec
    from src.notifications import emergency as em
    ep = Spy()
    em.send_verification(a, "email", "admin", "admin", email_provider=ep)
    code = _code_of(ep)
    store.update_recipient(a, email="new@forest.gov.in")
    ok, why = store.check_verification(a, "email", code)
    assert not ok and "changed" in why


def test_sms_alerts_do_not_need_phone_verification(rec):
    store, a, b = rec
    from src.notifications import emergency as em
    from tests.test_sms_auto_alerts import _snapshot
    assert store.get_recipient(a)["phone_verified"] == 0 and store.get_recipient(b)["phone_verified"] == 0
    assert set(em.manual_sms_recipients(_snapshot())) == {"+919812345678", "+919900011122"}
    sms = Spy()
    out = em.send_manual_sim_sms(_snapshot(rid="R-NOVERIFY"), "admin", "admin", confirmed=True, sms_provider=sms)
    assert out["blocked"] is None and {t for t, _ in sms.sent} == {"+919812345678", "+919900011122"}
    # registered-recipient path (send_alert) also needs no verified phone; e-mail still needs a verified address
    r = store.get_recipient(a)
    assert em.eligible_channels(r, "SIMULATION ONLY") == ["sms"]
    store.update_recipient(a, email_verified=1)
    assert em.eligible_channels(store.get_recipient(a), "SIMULATION ONLY") == ["email", "sms"]
    store.update_recipient(b, prefs={"sms": False})
    assert em.eligible_channels(store.get_recipient(b), "SIMULATION ONLY") == []      # 'SMS alerts' off


def test_sms_alert_safety_controls_kept(rec, env):                        # noqa: F811
    store, a, _ = rec
    from src.notifications import emergency as em
    from tests.test_sms_auto_alerts import _snapshot
    sms = Spy()
    with pytest.raises(PermissionError):
        em.send_manual_sim_sms(_snapshot(), "admin", "admin", confirmed=False, sms_provider=sms)
    with pytest.raises(PermissionError):
        em.send_manual_sim_sms(_snapshot(), "viewer", "viewer", confirmed=True, sms_provider=sms)
    em.send_manual_sim_sms(_snapshot(rid="R-CD"), "admin", "admin", confirmed=True, sms_provider=sms)
    again = em.send_manual_sim_sms(_snapshot(rid="R-CD"), "admin", "admin", confirmed=True, sms_provider=sms)
    assert {r["status"] for r in again["results"]} == {"cooldown"}        # 5-minute manual cooldown
    env.set_runtime({"ALERT_PROVIDER_MODE": "mock"})
    assert em.send_manual_sim_sms(_snapshot(rid="R-M"), "admin", "admin", True, sms_provider=sms)["blocked"]
    from src.notifications import alert_store
    import sqlalchemy as sa
    from src.storage import database as db
    with db.get_connection() as c:                                        # a stored invalid number is skipped
        c.execute(sa.update(alert_store.T_RECIP).where(alert_store.T_RECIP.c.id == a).values(phone="12345"))
    assert "12345" not in em.manual_sms_recipients(_snapshot())


def test_no_secrets_or_codes_in_logs(rec):
    store, a, _ = rec
    from src.notifications import emergency as em
    from src.storage import database as db
    ep = Spy("failed", error="auth failed app-pass-123")
    em.send_verification(a, "email", "admin", "admin", email_provider=ep)
    code = _code_of(ep)
    logs = " ".join(str(r) for r in db.get_activity(100))
    deliv = " ".join(str(d) for d in store.recent_deliveries(10))
    assert "verification code: failed" in logs
    for blob in (logs, deliv):
        assert "app-pass-123" not in blob and f" {code}" not in blob and "a@forest.gov.in" not in blob
    import sqlalchemy as sa
    with db.get_connection() as c:
        rows = [dict(r._mapping) for r in c.execute(sa.select(store.T_VCODE))]
    assert rows and all(code not in str(r) for r in rows)
