"""textbee.dev Android SMS gateway + automatic fire alerts (threshold, cooldown, fallback, history).

Every HTTP call goes to a fake session and every send to a fake provider: these tests never
contact textbee, Twilio or an SMTP server and never send a real SMS or e-mail. Each test uses its
own temporary SQLite database. The real end-to-end SMS test is the explicit "Send test SMS" action
on Admin -> Notifications (see docs/SMS_GATEWAY_SETUP.md)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from src.notifications.providers import SendResult

KEY = "tb_test_key_0123456789abcdef"
DEV = "64f1a2b3c4d5e6f708192a3b"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'a.db'}")
    from src.notifications import config as cfg
    from src.notifications import providers as prov
    cfg.clear_runtime()
    with cfg._LOCK:                                     # isolate from the machine's .env / secrets
        cfg._RUNTIME.update({k: "" for k in cfg.KEYS})
        cfg._RUNTIME["ALERT_PROVIDER_MODE"] = "live"
    prov._DEVICE_CACHE.clear()
    from src.notifications import emergency as em
    em._TEST_GUARD.clear()
    yield cfg
    cfg.clear_runtime()


def _textbee_on(cfg, device=DEV):
    cfg.set_runtime({"SMS_PROVIDER": "textbee", "TEXTBEE_API_KEY": KEY, "TEXTBEE_DEVICE_ID": device})


class Resp:
    def __init__(self, code, body=None):
        self.status_code, self._b = code, body

    def json(self):
        if self._b is None:
            raise ValueError("no json")
        return self._b


class FakeHTTP:
    """Records calls; answers from queues (a value or an exception)."""
    def __init__(self, post=(), get=()):
        self.posts, self.gets = list(post), list(get)
        self.post_calls, self.get_calls = [], []

    def post(self, url, json=None, headers=None, timeout=None):
        self.post_calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        r = self.posts.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def get(self, url, headers=None, timeout=None):
        self.get_calls.append({"url": url, "headers": headers})
        r = self.gets.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _tb(http):
    from src.notifications.providers import TextbeeSmsProvider
    return TextbeeSmsProvider(session=http, sleep=lambda s: None)


OK_SEND = Resp(200, {"data": {"success": True, "message": "queued", "smsBatchId": "b123", "recipientCount": 1,
                              "successCount": 1, "failureCount": 0}})


# ── 1. configuration ───────────────────────────────────────────────────────

def test_missing_and_invalid_gateway_configuration(env):
    cfg = env
    http = FakeHTTP()
    r = _tb(http).send("+919812345678", "x")
    assert r.status == "not_configured" and "TEXTBEE_API_KEY" in r.error and not http.post_calls
    cfg.set_runtime({"TEXTBEE_API_KEY": "short key", "TEXTBEE_BASE_URL": "http://insecure", "TEXTBEE_TIMEOUT": "999"})
    probs = cfg.textbee_problems()
    assert any("malformed" in p for p in probs) and any("https" in p for p in probs) and any("TIMEOUT" in p for p in probs)
    assert "short key" not in " ".join(probs)                         # the key itself is never echoed
    assert _tb(FakeHTTP()).test_connection().status == "not_configured"


def test_provider_selection(env):
    cfg = env
    from src.notifications import providers as prov
    assert cfg.sms_provider_name() == "textbee"                        # nothing configured -> textbee (not configured)
    cfg.set_runtime({"TWILIO_ACCOUNT_SID": "AC" + "0" * 32})
    assert cfg.sms_provider_name() == "twilio" and isinstance(prov.sms_provider(), prov.SmsProvider)
    cfg.set_runtime({"TEXTBEE_API_KEY": KEY})
    assert cfg.sms_provider_name() == "textbee" and isinstance(prov.sms_provider(), prov.TextbeeSmsProvider)
    cfg.set_runtime({"SMS_PROVIDER": "twilio"})
    assert cfg.sms_provider_name() == "twilio"


def test_invalid_recipient_is_rejected_locally(env):
    _textbee_on(env)
    http = FakeHTTP()
    r = _tb(http).send("9812345678", "x")
    assert r.status == "failed" and "E.164" in r.error and not http.post_calls


# ── 2 / 4. request accepted (documented API shape), delivery not claimed ──

def test_send_uses_documented_api_and_reports_accepted_not_delivered(env):
    _textbee_on(env)
    http = FakeHTTP(post=[OK_SEND])
    r = _tb(http).send("+919812345678", "TEST ALERT (no fire)")
    call = http.post_calls[0]
    assert call["url"] == "https://api.textbee.dev/api/v1/gateway/send-sms"
    assert call["headers"]["x-api-key"] == KEY
    assert call["json"] == {"recipients": ["+919812345678"], "message": "TEST ALERT (no fire)", "deviceId": DEV}
    assert r.status == "accepted" and r.status != "delivered" and r.message_id == f"{DEV}:b123"
    assert "not yet confirmed" in r.detail
    from src.notifications.auto_alerts import describe_channel
    assert describe_channel(r.status, "sms") == "Request accepted by gateway"


def test_delivery_status_only_from_provider_record(env):
    _textbee_on(env)
    p = _tb(FakeHTTP(get=[Resp(200, {"data": {"batch": {"status": "processing"},
                                              "messages": [{"status": "pending"}]}})]))
    assert p.status(f"{DEV}:b123").status == "accepted"
    p = _tb(FakeHTTP(get=[Resp(200, {"data": {"batch": {"status": "completed"}, "messages": [{"status": "sent"}]}})]))
    assert p.status(f"{DEV}:b123").status == "sent"
    http = FakeHTTP(get=[Resp(200, {"data": {"batch": {"status": "delivered"}, "messages": [{"status": "delivered"}]}})])
    r = _tb(http).status(f"{DEV}:b123")
    assert r.status == "delivered" and http.get_calls[0]["url"].endswith(f"/gateway/devices/{DEV}/sms-batch/b123")
    p = _tb(FakeHTTP(get=[Resp(200, {"data": {"messages": [{"status": "failed", "errorMessage": "no signal"}]}})]))
    r = p.status(f"{DEV}:b123")
    assert r.status == "failed" and "no signal" in r.error


def test_device_resolved_when_not_configured(env):
    _textbee_on(env, device="")
    devices = Resp(200, {"data": [{"_id": "old", "enabled": True, "lastHeartbeat": "2026-01-01T00:00:00Z"},
                                  {"_id": "dflt", "enabled": True, "isDefault": True},
                                  {"_id": "off", "enabled": False, "isDefault": True}]})
    http = FakeHTTP(post=[OK_SEND], get=[devices])
    r = _tb(http).send("+919812345678", "x")
    assert http.post_calls[0]["json"]["deviceId"] == "dflt" and r.message_id == "dflt:b123"


def test_connection_test_reports_phone_state(env):
    _textbee_on(env)
    fresh = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    stale = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
    ok = _tb(FakeHTTP(get=[Resp(200, {"data": [{"_id": DEV, "enabled": True, "name": "Lab phone",
                                                "lastHeartbeat": fresh}]})])).test_connection()
    assert ok.status == "connected" and "Lab phone" in ok.detail and "offline" not in ok.detail
    old = _tb(FakeHTTP(get=[Resp(200, {"data": [{"_id": DEV, "enabled": True, "lastHeartbeat": stale}]})])).test_connection()
    assert old.status == "connected" and "may be offline" in old.detail
    none = _tb(FakeHTTP(get=[Resp(200, {"data": []})])).test_connection()
    assert none.status == "failed" and "no Android device" in none.error
    dis = _tb(FakeHTTP(get=[Resp(200, {"data": [{"_id": DEV, "enabled": False}]})])).test_connection()
    assert dis.status == "failed" and "disabled" in dis.error
    bad = _tb(FakeHTTP(get=[Resp(401, {"message": "Unauthorized"})])).test_connection()
    assert bad.status == "failed" and "authentication failed" in bad.error


# ── 3. timeout, auth failure, rate limit, server error, safe retries ──────

def test_read_timeout_is_unknown_and_never_retried(env):
    _textbee_on(env)
    http = FakeHTTP(post=[requests.exceptions.ReadTimeout("slow"), OK_SEND])
    r = _tb(http).send("+919812345678", "x")
    assert r.status == "unknown" and "not retried" in r.error and len(http.post_calls) == 1


def test_connect_failure_retried_only_when_nothing_was_sent(env):
    _textbee_on(env)
    http = FakeHTTP(post=[requests.exceptions.ConnectTimeout("connect"), OK_SEND])
    assert _tb(http).send("+919812345678", "x").status == "accepted" and len(http.post_calls) == 2
    http = FakeHTTP(post=[requests.exceptions.ConnectTimeout("c")] * 3)
    r = _tb(http).send("+919812345678", "x")
    assert r.status == "failed" and "nothing was sent" in r.error and len(http.post_calls) == 3
    http = FakeHTTP(post=[requests.exceptions.ConnectionError("Connection reset by peer"), OK_SEND])
    r = _tb(http).send("+919812345678", "x")                          # ambiguous: may have reached the gateway
    assert r.status == "unknown" and len(http.post_calls) == 1


@pytest.mark.parametrize("code,body,expect", [
    (401, {"message": "Unauthorized"}, "authentication failed"),
    (429, {"message": "Daily limit reached"}, "plan limit reached"),
    (500, None, "gateway server error"),
    (400, {"message": ["recipients must be valid"]}, "rejected"),
])
def test_http_errors_are_failed_with_clear_reason(env, code, body, expect):
    _textbee_on(env)
    http = FakeHTTP(post=[Resp(code, body)])
    r = _tb(http).send("+919812345678", "x")
    assert r.status == "failed" and expect in r.error and len(http.post_calls) == 1
    assert KEY not in r.error


def test_gateway_reported_failure_in_body(env):
    _textbee_on(env)
    r = _tb(FakeHTTP(post=[Resp(200, {"data": {"success": False, "message": "device offline"}})])).send("+919812345678", "x")
    assert r.status == "failed" and "device offline" in r.error


def test_api_key_redacted_from_errors(env):
    _textbee_on(env)
    r = _tb(FakeHTTP(post=[Resp(403, {"message": f"key {KEY} not allowed"})])).send("+919812345678", "x")
    assert KEY not in r.error and "•••" in r.error


# ── automatic alerts: fixtures ─────────────────────────────────────────────

class FakeProv:
    def __init__(self, status="accepted"):
        self.status_, self.sent = status, []

    def send(self, to, *a, **k):
        self.sent.append((to, a))
        return SendResult(self.status_, "fake", "id-1" if self.status_ in ("accepted",) else None,
                          None if self.status_ == "accepted" else "boom")


def _twin(zones, offline=False, scenario=None, region="Karnataka Western Ghats"):
    alerts = [SimpleNamespace(zone_id=z, latitude=12.0 + i * 0.1, longitude=76.0, risk_score=0.8, severity=sev,
                              reason="test reason") for i, (z, p, sev) in enumerate(zones)]
    grid = pd.DataFrame({"zone_id": [z for z, _, _ in zones], "fire_probability": [p for _, p, _ in zones]})
    snap = SimpleNamespace(alerts=alerts, processed_grid=grid)
    return SimpleNamespace(current_snapshot=snap, region=SimpleNamespace(name=region), offline=offline,
                           ingestion=SimpleNamespace(scenario=scenario))


def _on(cfg, **kw):
    from src.notifications import auto_alerts as aa
    s = dict(aa.DEFAULTS, enabled=True)
    s.update(kw)
    assert aa.save_settings(s, actor="test") == []
    cfg.set_runtime({"ALERT_SMS_RECIPIENTS": "+919812345678", "ALERT_RECIPIENT_EMAILS": "officer@example.org"})
    return aa


def _decisions(aa):
    return [(r["zone_id"], r["decision"]) for r in aa.history(50)][::-1]


# ── 5 / 6 / 7. confidence threshold ────────────────────────────────────────

def test_below_threshold_suppressed_and_at_threshold_sent(env):
    aa = _on(env, min_confidence=0.35)
    sms, mail = FakeProv(), FakeProv()
    out = aa.process_refresh(_twin([("G-1", 0.34, "EXTREME"), ("G-2", 0.35, "EXTREME"), ("G-3", 0.9, "HIGH")]),
                             sms_provider=sms, email_provider=mail)
    d = dict(_decisions(aa))
    assert d == {"G-1": "suppressed_threshold", "G-2": "sent"}            # HIGH zone is not an alert condition
    assert out["decision"] == "sent" and out["sent_zones"] == 1 and len(sms.sent) == 1 and not mail.sent
    row = [r for r in aa.history() if r["zone_id"] == "G-2"][0]
    assert row["confidence"] == pytest.approx(0.35) and row["threshold"] == pytest.approx(0.35)
    assert row["sms_status"] == "accepted" and row["email_status"] == "skipped"


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -0.1, 1.5, "abc"])
def test_missing_or_invalid_confidence_skipped(env, bad):
    aa = _on(env)
    sms = FakeProv()
    aa.process_refresh(_twin([("G-9", bad, "EXTREME")]), sms_provider=sms, email_provider=FakeProv())
    assert _decisions(aa) == [("G-9", "skipped_invalid")] and not sms.sent


def test_settings_validation(env):
    from src.notifications import auto_alerts as aa
    assert aa.save_settings({"min_confidence": 0}) and aa.save_settings({"min_confidence": 1.2})
    assert aa.save_settings({"cooldown_minutes": -5}) and aa.save_settings({"policy": "pigeon"})
    assert aa.save_settings({"min_confidence": float("nan")})
    assert aa.get_settings() == aa.DEFAULTS                             # nothing invalid was stored
    assert aa.get_settings()["enabled"] is False                        # automatic alerts are off by default


def test_disabled_demo_and_whatif_never_notify(env):
    aa = _on(env)
    sms = FakeProv()
    aa.process_refresh(_twin([("D-1", 0.9, "EXTREME")], offline=True), sms_provider=sms, email_provider=FakeProv())
    aa.process_refresh(_twin([("W-1", 0.9, "EXTREME")], scenario={"temp_c": 45}), sms_provider=sms,
                       email_provider=FakeProv())
    aa.save_settings({"enabled": False})
    aa.process_refresh(_twin([("X-1", 0.9, "EXTREME")]), sms_provider=sms, email_provider=FakeProv())
    assert _decisions(aa) == [("D-1", "skipped_not_real"), ("W-1", "skipped_not_real"), ("X-1", "disabled")]
    assert not sms.sent


# ── 8 / 9 / 10. duplicates, cooldown expiry, new incidents, escalation ────

def test_duplicate_suppressed_new_incident_allowed(env):
    aa = _on(env, cooldown_minutes=360)
    sms = FakeProv()
    aa.process_refresh(_twin([("G-1", 0.5, "EXTREME")]), sms_provider=sms, email_provider=FakeProv())
    aa.process_refresh(_twin([("G-1", 0.52, "EXTREME")]), sms_provider=sms, email_provider=FakeProv())
    aa.process_refresh(_twin([("G-1", 0.5, "EXTREME"), ("G-7", 0.6, "EXTREME")]), sms_provider=sms,
                       email_provider=FakeProv())
    assert _decisions(aa) == [("G-1", "sent"), ("G-1", "suppressed_cooldown"), ("G-1", "suppressed_cooldown"),
                              ("G-7", "sent")]
    assert len(sms.sent) == 2                                          # one SMS per notifying refresh
    # the same zone in another region is a different incident
    aa.process_refresh(_twin([("G-1", 0.5, "EXTREME")], region="Uttarakhand"), sms_provider=sms,
                       email_provider=FakeProv())
    assert _decisions(aa)[-1] == ("G-1", "sent")


def test_cooldown_expiry_and_escalation(env):
    aa = _on(env, cooldown_minutes=60, escalation_step=0.10)
    t0 = datetime.now(timezone.utc)
    k = aa.incident_key("R", "Z")
    assert aa.claim(k, 0.40, 60, 0.10, now=t0)[:2] == (True, "new")
    assert aa.claim(k, 0.45, 60, 0.10, now=t0 + timedelta(minutes=30))[:2] == (False, "cooldown")
    assert aa.claim(k, 0.50, 60, 0.10, now=t0 + timedelta(minutes=31))[:2] == (True, "escalation")
    assert aa.claim(k, 0.55, 60, 0.10, now=t0 + timedelta(minutes=40))[:2] == (False, "cooldown")
    assert aa.claim(k, 0.30, 60, 0.10, now=t0 + timedelta(minutes=92))[:2] == (True, "cooldown_expired")


def test_concurrent_claims_only_one_wins(env):
    from concurrent.futures import ThreadPoolExecutor
    aa = _on(env)
    aa.ensure_tables()
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda _: aa.claim(aa.incident_key("R", "Z"), 0.5, 360, 0.1)[0], range(8)))
    assert res.count(True) == 1


def test_cooldown_persists_across_restart(env):
    aa = _on(env, cooldown_minutes=360)
    aa.process_refresh(_twin([("G-1", 0.5, "EXTREME")]), sms_provider=FakeProv(), email_provider=FakeProv())
    import importlib
    from src.notifications import auto_alerts
    aa2 = importlib.reload(auto_alerts)                                # a fresh process: no in-memory state
    aa2._READY.clear()
    sms = FakeProv()
    aa2.process_refresh(_twin([("G-1", 0.5, "EXTREME")]), sms_provider=sms, email_provider=FakeProv())
    assert _decisions(aa2)[-1] == ("G-1", "suppressed_cooldown") and not sms.sent


# ── 11 / 12. fallback ──────────────────────────────────────────────────────

def test_sms_failure_triggers_email_fallback_recorded_separately(env):
    aa = _on(env, policy="sms_then_email")
    sms, mail = FakeProv("failed"), FakeProv("accepted")
    out = aa.process_refresh(_twin([("G-1", 0.5, "EXTREME")]), sms_provider=sms, email_provider=mail)
    assert len(sms.sent) == 1 and len(mail.sent) == 1
    assert out["decision"] == "sent" and out["sms_status"] == "failed" and out["email_status"] == "accepted"
    row = aa.history()[0]
    assert row["sms_status"] == "failed" and row["email_status"] == "accepted" and row["channels"] == "sms+email"
    from src.notifications import alert_store as store
    ch = sorted((d["channel"], d["status"]) for d in store.recent_deliveries(10))
    assert ch == [("email", "accepted"), ("sms", "failed")]           # never one merged "success"


def test_sms_accepted_means_no_fallback_email_and_both_policy(env):
    aa = _on(env, policy="sms_then_email")
    mail = FakeProv()
    aa.process_refresh(_twin([("G-1", 0.5, "EXTREME")]), sms_provider=FakeProv(), email_provider=mail)
    assert not mail.sent
    aa.save_settings({"policy": "both"})
    mail2 = FakeProv()
    aa.process_refresh(_twin([("G-2", 0.5, "EXTREME")]), sms_provider=FakeProv(), email_provider=mail2)
    assert len(mail2.sent) == 1


def test_both_channels_fail_leaves_incident_retryable(env):
    aa = _on(env)
    out = aa.process_refresh(_twin([("G-1", 0.5, "EXTREME")]), sms_provider=FakeProv("failed"),
                             email_provider=FakeProv("failed"))
    assert out["decision"] == "failed" and aa.history()[0]["decision"] == "failed"
    sms = FakeProv()
    aa.process_refresh(_twin([("G-1", 0.5, "EXTREME")]), sms_provider=sms, email_provider=FakeProv())
    assert aa.history()[0]["decision"] == "sent" and len(sms.sent) == 1


def test_uncertain_sms_keeps_cooldown_no_duplicate(env):
    aa = _on(env, policy="sms_only")
    aa.process_refresh(_twin([("G-1", 0.5, "EXTREME")]), sms_provider=FakeProv("unknown"), email_provider=FakeProv())
    sms = FakeProv()
    aa.process_refresh(_twin([("G-1", 0.5, "EXTREME")]), sms_provider=sms, email_provider=FakeProv())
    assert aa.history()[0]["decision"] == "suppressed_cooldown" and not sms.sent


def test_no_recipients_does_not_claim(env):
    aa = _on(env)
    env.set_runtime({"ALERT_SMS_RECIPIENTS": "12345", "ALERT_RECIPIENT_EMAILS": "not-an-email"})
    aa.process_refresh(_twin([("G-1", 0.5, "EXTREME")]), sms_provider=FakeProv(), email_provider=FakeProv())
    assert aa.history()[0]["decision"] == "no_channels"
    env.set_runtime({"ALERT_SMS_RECIPIENTS": "+919812345678"})
    sms = FakeProv()
    aa.process_refresh(_twin([("G-1", 0.5, "EXTREME")]), sms_provider=sms, email_provider=FakeProv())
    assert aa.history()[0]["decision"] == "sent" and sms.sent


# ── 13. history / dashboard labels ─────────────────────────────────────────

def test_history_and_status_labels_are_truthful(env):
    from src.notifications import auto_alerts as aa
    sms = {s: aa.describe_channel(s, "sms") for s in ("accepted", "queued", "sent", "delivered", "unknown", "failed",
                                                       "not_configured")}
    assert sms["accepted"] == "Request accepted by gateway" and sms["delivered"] == "Delivery confirmed by provider"
    assert "unknown" in sms["unknown"].lower() and sms["not_configured"] == "Not configured"
    assert all("delivered" not in v.lower() and "confirmed" not in v.lower()
               for k, v in sms.items() if k != "delivered")
    assert aa.describe_channel("accepted", "email") == "Email accepted by SMTP server"
    assert aa.describe_decision("suppressed_threshold") == "Suppressed by confidence threshold"
    assert aa.describe_decision("suppressed_cooldown").startswith("Automatic alert suppressed by cooldown")
    m = aa.build_messages("R", [{"zone_id": "G-1", "lat": 12.0, "lon": 76.0, "confidence": 0.5, "threshold": 0.35,
                                 "reason": "r"}], "AUTO-1", "live data")
    assert "not a confirmed fire" in m["sms"].lower() and "not a confirmed fire" in m["body"] and len(m["sms"]) < 400


def test_errors_never_raise_out_of_a_refresh(env, monkeypatch):
    aa = _on(env)
    monkeypatch.setattr(aa, "claim", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down")))
    out = aa.process_refresh(_twin([("G-1", 0.5, "EXTREME")]), sms_provider=FakeProv(), email_provider=FakeProv())
    assert out["decision"] == "error" and "db down" in out["error"]


# ── 14. test notifications are isolated from the fire-alert workflow ──────

def test_test_sms_does_not_touch_fire_alert_workflow(env):
    from src.notifications import auto_alerts as aa
    from src.notifications import emergency as em
    _textbee_on(env)
    http = FakeHTTP(post=[OK_SEND])
    out = em.send_test("sms", "+919812345678", "admin", "admin", sms_provider=_tb(http))
    assert out["status"] == "accepted" and out["alert_id"].startswith("TEST-")
    assert "TEST ALERT (no fire)" in http.post_calls[0]["json"]["message"]
    aa.ensure_tables()
    assert aa.history() == []                                          # no automatic decision recorded
    with aa.db.get_connection() as c:
        assert c.execute(aa.select(aa.T_STATE)).all() == []            # no cooldown touched
    with pytest.raises(PermissionError):
        em.send_test("sms", "+919812345678", "viewer", "viewer", sms_provider=_tb(FakeHTTP()))


def test_persist_hook_uses_auto_alerts(env, monkeypatch):
    """The dashboard / scheduler hook hands every refresh to auto_alerts (and never to the old e-mail-only path)."""
    from src.digital_twin import twin_state
    from src.notifications import auto_alerts as aa
    seen = []
    monkeypatch.setattr(aa, "process_refresh", lambda twin, actor="system", **k: seen.append(actor) or
                        {"decision": "sent", "email_status": "accepted", "sms_status": "accepted"})
    monkeypatch.setattr(twin_state, "logger", twin_state.logger)
    import inspect
    src = inspect.getsource(twin_state.persist_snapshot_and_notify)
    assert "auto_alerts.process_refresh" in src and "send_new_extreme_alert" not in src


# ═════ Manual simulation SMS (explicit "Send SMS Alert") and resend policies ═════

@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch):
    """15. Any real HTTP request from these tests fails loudly - nothing can reach textbee / Twilio."""
    def boom(*a, **k):
        raise AssertionError("a test tried to make a real HTTP request")
    monkeypatch.setattr(requests.Session, "request", boom)


def _snapshot(mode="WHAT-IF", rid="FFDT-SIM-1"):
    return {"report_id": rid, "mode": mode, "firms": {"n_valid": 2 if mode == "LIVE" else 0},
            "location": {"name": "Bandipur Tiger Reserve", "lat": 11.668, "lon": 76.633},
            "risk": {"category": "LOW", "score_pct": 19}, "prediction": {"probability": 0.0018},
            "results": {"burned_ha": 68.3, "duration_label": "1 h", "front_distance_m": 1182, "spread_towards": "ENE"},
            "generated_utc": "2026-10-17T08:00:00+00:00", "region": "Karnataka Western Ghats"}


def _manual_ready(cfg):
    _textbee_on(cfg)
    cfg.set_runtime({"ALERT_SMS_RECIPIENTS": "+919812345678"})


def test_auto_alerts_skip_hypothetical_runs(env):
    """1. Automatic alerts: What-If / hypothetical data is always skipped, even when it qualifies."""
    aa = _on(env, min_confidence=0.01)
    sms = FakeProv()
    out = aa.process_refresh(_twin([("W-1", 0.99, "EXTREME")], scenario={"temp_c": 48, "humidity_pct": 5}),
                             sms_provider=sms, email_provider=FakeProv())
    assert out["decision"] == "skipped_not_real" and not sms.sent
    assert aa.history()[0]["note"] == "What-If scenario"


def test_confirmed_manual_sms_uses_real_textbee_provider_in_live_mode(env, monkeypatch):
    """2. Confirmed manual SMS during a hypothetical run goes through the configured textbee provider (live)."""
    from src.notifications import emergency as em
    from src.notifications import providers as prov
    _manual_ready(env)
    assert isinstance(prov.sms_provider(), prov.TextbeeSmsProvider)
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append((url, json, headers))
        return OK_SEND
    monkeypatch.setattr(prov.requests, "post", fake_post)               # the provider's own HTTP call, intercepted
    out = em.send_manual_sim_sms(_snapshot(), "admin", "admin", confirmed=True)
    assert out["blocked"] is None and out["results"][0]["status"] == "accepted"
    url, payload, headers = calls[0]
    assert url.endswith("/gateway/send-sms") and headers["x-api-key"] == KEY
    assert payload["recipients"] == ["+919812345678"] and payload["deviceId"] == DEV
    assert out["results"][0]["status"] != "delivered"                    # accepted is not delivery


def test_cancel_or_no_confirmation_sends_nothing(env):
    """3. Without the explicit confirmation nothing is sent; the dialog's Cancel only closes it."""
    import inspect
    from src.notifications import emergency as em
    from src.dashboard.ui import alerts_ui
    _manual_ready(env)
    sms = FakeProv()
    with pytest.raises(PermissionError):
        em.send_manual_sim_sms(_snapshot(), "admin", "admin", confirmed=False, sms_provider=sms)
    assert not sms.sent
    row = inspect.getsource(alerts_ui._confirm_dialog)
    assert "on_click=_do_send" in row and "on_click=_cancel_send" in row
    assert "send" not in inspect.getsource(alerts_ui._cancel_send).split('"""')[2]   # Cancel never sends
    assert "out = send()" in inspect.getsource(alerts_ui._do_send)
    dlg = inspect.getsource(getattr(alerts_ui._manual_sms_dialog, "__wrapped__", alerts_ui._manual_sms_dialog))
    assert '"Confirm & Send"' in row and "confirmed=True" in dlg
    with pytest.raises(PermissionError):                                # viewers cannot send
        em.send_manual_sim_sms(_snapshot(), "viewer", "viewer", confirmed=True, sms_provider=sms)


def test_manual_message_is_labelled_hypothetical(env):
    """4. The outgoing text says it is a simulation test alert about a hypothetical scenario."""
    from src.notifications import emergency as em
    _manual_ready(env)
    sms = FakeProv()
    out = em.send_manual_sim_sms(_snapshot(), "admin", "admin", confirmed=True, sms_provider=sms)
    text = sms.sent[0][1][0]
    assert text.splitlines()[1] == "⚠️ SIMULATION ONLY — HYPOTHETICAL SCENARIO — NOT A CONFIRMED FIRE"
    assert "Bandipur" in text and "FFDT-SIM-1" in text and out["alert_id"].startswith("MSIM-")
    assert em.sms_segments(text)["encoding"].startswith("UCS-2")         # emojis: Unicode SMS, documented
    live = em.manual_sms_text(_snapshot("LIVE"), "MSIM-1")
    assert "⚠️ SIMULATED SPREAD FROM SATELLITE HOTSPOTS — NOT GROUND-VERIFIED" in live


def test_manual_cooldown_suppresses_then_expires(env):
    """5 / 6. Same scenario and recipient: blocked inside 5 minutes, allowed again after."""
    from src.notifications import auto_alerts as aa
    from src.notifications import emergency as em
    _manual_ready(env)
    assert aa.get_settings()["manual_cooldown_minutes"] == 5
    sms = FakeProv()
    t0 = datetime.now(timezone.utc)
    first = em.send_manual_sim_sms(_snapshot(), "admin", "admin", confirmed=True, sms_provider=sms, now=t0)
    assert first["results"][0]["status"] == "accepted"
    again = em.send_manual_sim_sms(_snapshot(), "admin", "admin", confirmed=True, sms_provider=sms,
                                   now=t0 + timedelta(minutes=4, seconds=59))
    r = again["results"][0]
    assert r["status"] == "cooldown" and r["seconds_left"] == 1 and len(sms.sent) == 1 and again["alert_id"] is None
    assert r["next_eligible_utc"].startswith((t0 + timedelta(minutes=5)).isoformat(timespec="seconds")[:16])
    other = em.send_manual_sim_sms(_snapshot(rid="FFDT-SIM-2"), "admin", "admin", confirmed=True, sms_provider=sms,
                                   now=t0 + timedelta(minutes=1))                # another scenario is separate
    assert other["results"][0]["status"] == "accepted"
    later = em.send_manual_sim_sms(_snapshot(), "admin", "admin", confirmed=True, sms_provider=sms,
                                   now=t0 + timedelta(minutes=5))
    assert later["results"][0]["status"] == "accepted" and len(sms.sent) == 3
    assert aa.save_settings({"manual_cooldown_minutes": 0})              # no zero / bypass value accepted
    assert aa.save_settings({"manual_cooldown_minutes": 10}) == [] and aa.get_settings()["manual_cooldown_minutes"] == 10


def test_manual_sends_never_touch_automatic_cooldowns(env):
    """7. Manual simulation SMS neither creates nor changes automatic incident state, nor blocks a real alert."""
    from src.notifications import emergency as em
    aa = _on(env)
    _manual_ready(env)
    em.send_manual_sim_sms(_snapshot(), "admin", "admin", confirmed=True, sms_provider=FakeProv())
    assert aa.incident_states() == [] and aa.history() == []
    sms = FakeProv()
    aa.process_refresh(_twin([("G-1", 0.5, "EXTREME")]), sms_provider=sms, email_provider=FakeProv())
    before = aa.incident_states()
    em.send_manual_sim_sms(_snapshot(rid="FFDT-SIM-9"), "admin", "admin", confirmed=True, sms_provider=FakeProv())
    assert aa.incident_states() == before and aa.history()[0]["decision"] == "sent" and len(sms.sent) == 1
    from src.notifications import alert_store as store
    with store.db.get_connection() as c:
        types = {r[0] for r in c.execute(store.select(store.T_ALERTS.c.alert_type)).all()}
    assert types == {"MANUAL SIMULATION", "AUTOMATIC"}                  # logged as different kinds of alert


class Clock:
    def __init__(self):
        self.t = datetime(2026, 10, 17, 6, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.t


def test_same_zone_every_15_minutes_and_6_hour_eligibility(env, monkeypatch):
    """8 / 9. A zone that stays EXTREME on every 15-minute refresh gets one routine SMS per 6 h, not one per refresh."""
    aa = _on(env)
    assert aa.get_settings()["cooldown_minutes"] == 360 and aa.get_settings()["escalation_step"] == pytest.approx(0.10)
    clock = Clock()
    monkeypatch.setattr(aa, "_now", clock)
    sms = FakeProv()
    for i in range(25):                                                 # 0 .. 6 h in 15-minute refreshes
        aa.process_refresh(_twin([("G-1", 0.50, "EXTREME")]), sms_provider=sms, email_provider=FakeProv())
        clock.t += timedelta(minutes=15)
    assert len(sms.sent) == 2                                            # at t=0 and again at t=6 h
    rows = aa.history(50)[::-1]
    assert [r["decision"] for r in rows].count("sent") == 2 and rows[1]["decision"] == "suppressed_cooldown"
    assert aa.describe_row(rows[0]) == "Automatic alert sent: new incident"
    assert "next routine alert eligible after" in rows[1]["note"] and "17 Oct 2026, 05:30 PM" in rows[1]["note"]
    assert aa.describe_row(rows[24]) == "Automatic alert sent: routine repeat after cooldown"


def test_escalation_exception_inside_cooldown(env, monkeypatch):
    """10. Inside the cooldown, a +0.10 rise in probability is alerted (once), smaller rises are not."""
    aa = _on(env)
    clock = Clock()
    monkeypatch.setattr(aa, "_now", clock)
    sms = FakeProv()
    for p in (0.40, 0.49, 0.50, 0.55, 0.59, 0.61):
        aa.process_refresh(_twin([("G-1", p, "EXTREME")]), sms_provider=sms, email_provider=FakeProv())
        clock.t += timedelta(minutes=15)
    d = [(round(r["confidence"], 2), r["decision"]) for r in aa.history(20)[::-1]]
    assert d == [(0.40, "sent"), (0.49, "suppressed_cooldown"), (0.50, "sent"), (0.55, "suppressed_cooldown"),
                 (0.59, "suppressed_cooldown"), (0.61, "sent")]
    assert aa.describe_row(aa.history(20)[::-1][2]) == "Automatic alert permitted: risk escalation condition met"


def test_new_incident_while_other_zone_in_cooldown(env):
    """11. A different zone is a new incident even while another zone is cooling down."""
    aa = _on(env)
    sms = FakeProv()
    aa.process_refresh(_twin([("G-1", 0.5, "EXTREME")]), sms_provider=sms, email_provider=FakeProv())
    aa.process_refresh(_twin([("G-1", 0.5, "EXTREME"), ("G-2", 0.45, "EXTREME")]), sms_provider=sms,
                       email_provider=FakeProv())
    assert dict((r["zone_id"], r["decision"]) for r in aa.history(5)[:2]) == {"G-1": "suppressed_cooldown",
                                                                             "G-2": "sent"}


def test_restart_keeps_manual_and_automatic_cooldowns(env):
    """12. Both cooldowns live in the database, not in memory."""
    import importlib
    from src.notifications import auto_alerts, emergency
    _manual_ready(env)
    aa = _on(env)
    _manual_ready(env)
    emergency.send_manual_sim_sms(_snapshot(), "admin", "admin", confirmed=True, sms_provider=FakeProv())
    aa.process_refresh(_twin([("G-1", 0.5, "EXTREME")]), sms_provider=FakeProv(), email_provider=FakeProv())
    aa2 = importlib.reload(auto_alerts)
    aa2._READY.clear()
    em2 = importlib.reload(emergency)
    sms = FakeProv()
    out = em2.send_manual_sim_sms(_snapshot(), "admin", "admin", confirmed=True, sms_provider=sms)
    assert out["results"][0]["status"] == "cooldown" and not sms.sent
    aa2.process_refresh(_twin([("G-1", 0.5, "EXTREME")]), sms_provider=sms, email_provider=FakeProv())
    assert aa2.history()[0]["decision"] == "suppressed_cooldown" and not sms.sent


def test_report_download_never_notifies(env, monkeypatch):
    """13. Building / downloading the PDF and Word versions of a hypothetical report sends nothing."""
    import inspect
    from src.notifications import providers as prov
    from src.reports import simulation_report as rep
    from src.dashboard.ui import alerts_ui
    for cls in (prov.TextbeeSmsProvider, prov.SmsProvider, prov.EmailProvider):
        monkeypatch.setattr(cls, "send", lambda *a, **k: (_ for _ in ()).throw(AssertionError("notification sent")))
    from tests.test_duration_report_alerts import _snap
    _, snap = _snap(5)
    assert rep.render_pdf(snap)[:4] == b"%PDF" and rep.render_docx(snap)[:2] == b"PK" and "<html" in rep.render_html(snap)
    assert rep.export_document(snap, "pdf")[:4] == b"%PDF" and rep.export_document(snap, "docx")[:2] == b"PK"
    menu = inspect.getsource(alerts_ui._export_menu) + inspect.getsource(rep.export_document)
    assert "send_" not in menu and "sms_provider" not in menu and 'on_click="ignore"' in menu


@pytest.mark.parametrize("status,released", [("failed", True), ("not_configured", True), ("unknown", False),
                                             ("accepted", False)])
def test_manual_failures_and_uncertain_outcomes_recorded_honestly(env, status, released):
    """14. Definite failures are recorded and may be retried at once; uncertain outcomes keep the cooldown."""
    from src.notifications import alert_store as store
    from src.notifications import emergency as em
    _manual_ready(env)
    out = em.send_manual_sim_sms(_snapshot(), "admin", "admin", confirmed=True, sms_provider=FakeProv(status))
    assert out["results"][0]["status"] == status
    d = store.recent_deliveries(5, channel="sms")[0]
    assert d["status"] == status and d["alert_id"] == out["alert_id"]
    again = em.send_manual_sim_sms(_snapshot(), "admin", "admin", confirmed=True, sms_provider=FakeProv())
    assert (again["results"][0]["status"] == "accepted") is released


def test_manual_sms_blocked_with_clear_reason_when_not_ready(env):
    from src.notifications import emergency as em
    sms = FakeProv()
    env.set_runtime({"ALERT_PROVIDER_MODE": "mock"})
    out = em.send_manual_sim_sms(_snapshot(), "admin", "admin", confirmed=True, sms_provider=sms)
    assert out["blocked"].startswith("Live sending is disabled") and not sms.sent
    env.set_runtime({"ALERT_PROVIDER_MODE": "live"})
    out = em.send_manual_sim_sms(_snapshot(), "admin", "admin", confirmed=True, sms_provider=sms)
    assert out["blocked"].startswith("SMS gateway not configured (textbee)") and not sms.sent
    _textbee_on(env)
    out = em.send_manual_sim_sms(_snapshot(), "admin", "admin", confirmed=True, sms_provider=sms)
    assert out["blocked"].startswith("No SMS recipient configured") and not sms.sent
    _manual_ready(env)
    from src.dashboard.ui import alerts_ui                                # auth / timeout errors are shown verbatim
    tb = _tb(FakeHTTP(post=[Resp(401, {"message": "Unauthorized"})]))
    out = em.send_manual_sim_sms(_snapshot(rid="R-401"), "admin", "admin", confirmed=True, sms_provider=tb)
    assert out["results"][0]["status"] == "failed" and "authentication failed" in out["results"][0]["error"]
    tb = _tb(FakeHTTP(post=[requests.exceptions.ReadTimeout("t")]))
    out = em.send_manual_sim_sms(_snapshot(rid="R-TO"), "admin", "admin", confirmed=True, sms_provider=tb)
    assert out["results"][0]["status"] == "unknown" and "did not answer" in out["results"][0]["error"]
