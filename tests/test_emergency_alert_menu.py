"""Send Emergency Alert: one button, a delivery-method menu (E-mail / SMS) shown only when opened, and a
separate confirmation workflow per channel. AppTest runs a tiny page with a completed What-If snapshot;
every provider is a fake (no SMTP server, no textbee, no network)."""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from tests.test_sms_auto_alerts import FakeProv, _manual_ready, _snapshot, env  # noqa: F401  (fixture)

ROOT = Path(__file__).resolve().parents[1]
PAGE = '''
import streamlit as st
from src.dashboard.ui import alerts_ui
from tests.test_sms_auto_alerts import _snapshot
alerts_ui.render_result_actions(_snapshot(), "sim")
'''


def _smtp_on(cfg, to="ops@example.org"):
    cfg.set_runtime({"SMTP_HOST": "smtp.example.org", "SMTP_PORT": "587", "SMTP_USER": "bot@example.org",
                     "SMTP_PASSWORD": "app-pass-123", "SMTP_FROM": "bot@example.org", "ALERT_RECIPIENT_EMAILS": to})


@pytest.fixture()
def app(env, tmp_path, monkeypatch):                     # noqa: F811
    st_testing = pytest.importorskip("streamlit.testing.v1")
    sent = {"email": [], "sms": []}
    from src.notifications import emergency as em, providers as prov

    class Spy(FakeProv):
        def __init__(self, ch):
            super().__init__()
            self.sent = sent[ch]
    monkeypatch.setattr(em, "EmailProvider", lambda: Spy("email"))
    monkeypatch.setattr(prov, "sms_provider", lambda: Spy("sms"))
    _manual_ready(env)
    _smtp_on(env)
    at = st_testing.AppTest.from_string(PAGE, default_timeout=60)
    at.session_state["auth_user"] = "tester"
    at.session_state["auth_role"] = "admin"
    return at, sent


def _labels(at):
    return [b.label for b in at.button]


def _open_menu_and_pick(at, channel):
    at.session_state["sim_ea_menu"] = True               # what a click on the menu button does
    at.run()
    at.button(key=f"sim_ea_opt_{channel}").click().run()


def test_one_emergency_button_and_menu_hidden_by_default(app):
    at, sent = app
    at.run()
    assert not at.exception
    pops = {p.proto.popover.label: p for p in at.get("popover")}
    assert set(pops) == {"📄 Export Report", "🚨 Send Emergency Alert"}            # one menu each, no duplicates
    alert = pops["🚨 Send Emergency Alert"]
    assert alert.proto.popover.type == "primary" and alert.proto.popover.open is False
    labels = _labels(at)
    assert not any("SMS Alert" in l for l in labels)                      # the standalone SMS button is gone
    assert "✉️ Email Alert" not in labels and "📱 SMS to Mobile" not in labels   # menu closed: nothing drawn
    assert not sent["email"] and not sent["sms"]


def test_open_menu_shows_exactly_two_options(app):
    at, sent = app
    at.session_state["sim_ea_menu"] = True
    at.run()
    opts = [b.label for b in at.button if b.key.startswith("sim_ea_opt_")]
    assert opts == ["✉️ Email Alert", "📱 SMS to Mobile"]
    assert not sent["email"] and not sent["sms"]                          # opening the menu sends nothing


TITLES = {"email": "Confirm Email Alert", "sms": "Confirm SMS Alert"}
QUESTIONS = {"email": "Send an emergency alert email for this simulation?",
             "sms": "Send an emergency SMS for this simulation?"}


def _page_text(at):
    return " ".join([m.value for m in at.markdown] + [c.value for c in at.caption] + [e.value for e in at.info]
                    + [e.value for e in at.success] + [e.value for e in at.warning] + [e.value for e in at.error])


@pytest.mark.parametrize("channel", ["email", "sms"])
def test_option_opens_minimal_dialog_and_cancel_sends_nothing(app, channel):
    at, sent = app
    _open_menu_and_pick(at, channel)
    assert not at.exception
    assert at.session_state["sim_ea_menu"] is False                        # menu closes after a choice
    labels = _labels(at)
    assert labels.count("Confirm & Send") == 1 and "Cancel" in labels      # only the chosen channel's dialog
    assert at.button(key=f"ea_{channel}_confirm").label == "Confirm & Send"
    texts = _page_text(at)
    assert QUESTIONS[channel] in texts
    assert ("op***@example.org" if channel == "email" else "******5678") in texts   # masked destination
    for preview in ("SIMULATION TEST ALERT", "Subject", "Message preview", "HTML preview", "FOREST FIRE DIGITAL TWIN",
                    "MSIM-", "ML Risk Score", "Bandipur"):
        assert preview not in texts, preview                              # no message contents in the dialog
    assert not at.get("code") and not at.expander                         # no text box / expandable preview
    assert not sent["email"] and not sent["sms"]                          # opening sends nothing
    at.button(key=f"ea_{channel}_cancel").click().run()
    assert not sent["email"] and not sent["sms"]


@pytest.mark.parametrize("channel", ["email", "sms"])
def test_confirm_sends_only_the_selected_channel_with_full_content(app, channel):
    at, sent = app
    _open_menu_and_pick(at, channel)
    at.button(key=f"ea_{channel}_confirm").click().run()
    assert not at.exception
    other = "sms" if channel == "email" else "email"
    assert len(sent[channel]) == 1 and not sent[other]
    payload = " ".join(str(x) for x in sent[channel][0][1])
    assert ("SIMULATION TEST ALERT - HYPOTHETICAL SCENARIO - NOT A CONFIRMED FIRE" in payload if channel == "email"
            else "SIMULATION ONLY — HYPOTHETICAL SCENARIO — NOT A CONFIRMED FIRE" in payload)
    assert "Bandipur" in payload and "FFDT-SIM-1" in payload               # the real message is complete
    toast = " ".join(t.value for t in at.toast)
    assert toast == ("✓ Email sent successfully!" if channel == "email" else "✓ SMS sent successfully!")
    for tech in ("gateway", "accepted", "delivery", "textbee", "Textbee", "SMTP"):
        assert tech not in toast
    assert '[data-testid="stToast"]{background:#15803d' in " ".join(m.value for m in at.markdown)   # green
    assert "delivered" not in toast.lower().replace("delivery not", "")


def test_no_status_banners_on_the_simulation_page(app):
    at, sent = app
    at.run()
    _open_menu_and_pick(at, "sms")
    at.button(key="ea_sms_confirm").click().run()
    at.run()                                                              # a later rerun: still nothing
    text = _page_text(at)
    for gone in ("Automatic fire alerts", "skipped - hypothetical run", "Manual SMS", "Manual E-mail",
                 "Manual simulation", "Status last checked", "Check delivery status", "ready to send", "cooldown"):
        assert gone not in text, gone
    assert "🔄 Check delivery status" not in _labels(at) and not at.get("iframe")
    assert len(sent["sms"]) == 1                                          # sent and logged, just not shown here
    from src.notifications import alert_store as store
    assert store.recent_deliveries(5, channel="sms")[0]["alert_id"].startswith("MSIM-")


def test_cooldown_and_errors_shown_in_the_dialog(app, env):                # noqa: F811
    at, sent = app
    _open_menu_and_pick(at, "sms")
    at.button(key="ea_sms_confirm").click().run()
    _open_menu_and_pick(at, "sms")
    assert "Already sent for this simulation - next send allowed at" in _page_text(at)
    assert at.button(key="ea_sms_confirm").disabled and len(sent["sms"]) == 1
    env.set_runtime({"ALERT_PROVIDER_MODE": "mock"})
    at.button(key="ea_sms_cancel").click().run()
    _open_menu_and_pick(at, "email")
    assert "Live sending is disabled" in _page_text(at) and at.button(key="ea_email_confirm").disabled


def test_result_toast_is_truthful():
    from src.dashboard.ui import alerts_ui as ui
    assert ui.result_toast("sms", {"blocked": "Live sending is disabled"})[0] == "SMS not sent - Live sending is disabled"
    assert "failed - authentication failed" in ui.result_toast("sms", {"results": [
        {"status": "failed", "error": "authentication failed"}]})[0]
    assert "status unknown" in ui.result_toast("email", {"results": [{"status": "unknown", "error": "t"}]})[0]
    assert "next send allowed" in ui.result_toast("email", {"results": [
        {"status": "cooldown", "next_eligible_utc": "2026-10-17T13:00:00+00:00"}]})[0]
    assert "sent to 1 of 2 recipients - 1 failed" in ui.result_toast("sms", {"results": [
        {"status": "accepted"}, {"status": "failed"}]})[0]
    # success only when every message was accepted (unchanged condition); failures never say "successfully"
    assert ui.result_toast("sms", {"results": [{"status": "accepted"}]}) == ("✓ SMS sent successfully!", None)
    assert ui.result_toast("email", {"results": [{"status": "accepted"}]}) == ("✓ Email sent successfully!", None)
    for bad in ({"blocked": "x"}, {"results": [{"status": "failed", "error": "e"}]},
                {"results": [{"status": "unknown", "error": "t"}]}, {"results": [{"status": "not_configured", "error": "n"}]},
                {"results": [{"status": "mock"}]}):
        for ch in ("sms", "email"):
            assert "successfully" not in ui.result_toast(ch, bad)[0]


@pytest.mark.parametrize("channel", ["email", "sms"])
def test_failed_send_shows_error_not_success(app, channel, monkeypatch):
    at, sent = app
    from src.notifications import emergency as em, providers as prov
    from src.notifications.providers import SendResult

    class Boom:
        def send(self, to, *a, **k):
            sent[channel].append((to, a))
            return SendResult("failed", "fake", None, "authentication failed")
    monkeypatch.setattr(em, "EmailProvider", lambda: Boom())
    monkeypatch.setattr(prov, "sms_provider", lambda: Boom())
    _open_menu_and_pick(at, channel)
    at.button(key=f"ea_{channel}_confirm").click().run()
    toast = " ".join(t.value for t in at.toast)
    assert "failed - authentication failed" in toast and "successfully" not in toast
    assert "#15803d" not in " ".join(m.value for m in at.markdown)       # no green success styling


def test_admin_keeps_the_history(env, tmp_path):                         # noqa: F811
    import inspect
    from src.dashboard.ui import alerts_ui as ui
    src = inspect.getsource(ui.render_auto_alerts)
    assert "Manual simulation e-mail / SMS" in src and 'startswith("MSIM-")' in src
    assert "refresh_sms_statuses" in inspect.getsource(ui.render_provider_config)


def test_email_preview_and_message_keep_simulation_label(env):     # noqa: F811
    from src.notifications import emergency as em
    _smtp_on(env)
    m = em.manual_email_message(_snapshot(), "MSIM-1")
    lab = "SIMULATION TEST ALERT - HYPOTHETICAL SCENARIO - NOT A CONFIRMED FIRE"
    assert m["subject"].startswith(f"[{lab}]") and m["body"].startswith(lab) and lab in m["html"]
    assert "SIMULATION ONLY" in m["body"]                                 # the original disclaimer stays
    live = em.manual_email_message(_snapshot("LIVE"), "MSIM-2")
    assert live["subject"].startswith("[SIMULATION ALERT - SPREAD SIMULATED FROM SATELLITE HOTSPOTS")
    ep = FakeProv()
    out = em.send_manual_sim_email(_snapshot(), "admin", "admin", confirmed=True, email_provider=ep)
    to, (subject, body) = ep.sent[0][0], ep.sent[0][1][:2]
    assert to == "ops@example.org" and lab in subject and lab in body and out["alert_id"] in subject


def test_email_rules_confirmation_admin_readiness_and_cooldown(env):     # noqa: F811
    from datetime import datetime, timedelta, timezone
    from src.notifications import alert_store as store, auto_alerts as aa, emergency as em
    ep = FakeProv()
    with pytest.raises(PermissionError):
        em.send_manual_sim_email(_snapshot(), "admin", "admin", confirmed=False, email_provider=ep)
    with pytest.raises(PermissionError):
        em.send_manual_sim_email(_snapshot(), "viewer", "viewer", confirmed=True, email_provider=ep)
    env.set_runtime({"ALERT_PROVIDER_MODE": "mock"})
    assert em.send_manual_sim_email(_snapshot(), "admin", "admin", True, ep)["blocked"].startswith("Live sending is disabled")
    env.set_runtime({"ALERT_PROVIDER_MODE": "live"})
    assert em.send_manual_sim_email(_snapshot(), "admin", "admin", True, ep)["blocked"].startswith("E-mail (SMTP) not configured")
    _smtp_on(env, to="")
    assert em.send_manual_sim_email(_snapshot(), "admin", "admin", True, ep)["blocked"].startswith("No e-mail recipient")
    assert not ep.sent
    _smtp_on(env)
    t0 = datetime.now(timezone.utc)
    first = em.send_manual_sim_email(_snapshot(), "admin", "admin", True, ep, now=t0)
    assert first["results"][0]["status"] == "accepted" and len(ep.sent) == 1
    again = em.send_manual_sim_email(_snapshot(), "admin", "admin", True, ep, now=t0 + timedelta(minutes=4))
    assert again["results"][0]["status"] == "cooldown" and len(ep.sent) == 1     # double submit: nothing resent
    later = em.send_manual_sim_email(_snapshot(), "admin", "admin", True, ep, now=t0 + timedelta(minutes=5, seconds=1))
    assert later["results"][0]["status"] == "accepted" and len(ep.sent) == 2
    d = store.recent_deliveries(5, channel="email")[0]
    assert d["alert_id"] == later["alert_id"] and d["status"] == "accepted"
    assert not aa.incident_states()                                       # automatic state untouched
    # e-mail and SMS guards are independent: an e-mail cooldown never blocks the SMS
    _manual_ready(env)
    sms = FakeProv()
    assert em.send_manual_sim_sms(_snapshot(), "admin", "admin", True, sms, now=t0 + timedelta(minutes=5, seconds=2)
                                  )["results"][0]["status"] == "accepted"


def test_email_failure_is_reported_and_retry_allowed(env):     # noqa: F811
    from src.notifications import emergency as em
    _smtp_on(env)
    out = em.send_manual_sim_email(_snapshot(), "admin", "admin", True, FakeProv("failed"))
    assert out["results"][0]["status"] == "failed" and out["results"][0]["error"] == "boom"
    assert em.send_manual_sim_email(_snapshot(), "admin", "admin", True, FakeProv())["results"][0]["status"] == "accepted"


def test_menu_is_themed_and_accessible():
    from src.dashboard.ui import alerts_ui as ui
    css = ui.EA_MENU_CSS
    assert 'html[data-ff-theme="navy"] [data-testid="stPopoverBody"]:has(.ea-menu)' in css   # follows the theme
    assert ":focus-visible" in css and ":hover" in css and "max-width: calc(100vw - 32px)" in css
    assert "Send the emergency alert to the configured e-mail recipient" in css
    assert "Send an SMS to the configured mobile recipient" in css
    src = inspect.getsource(ui.render_result_actions)
    assert 'on_change="rerun"' in src and "if menu.open:" in src and "send_" not in src   # the menu never sends
    assert ui.EA_OPTIONS == (("email", "✉️ Email Alert"), ("sms", "📱 SMS to Mobile"))
