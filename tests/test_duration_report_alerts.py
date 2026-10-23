"""Simulation clock (1 / 5 / 10 / 15 min), the redesigned report (PDF / DOCX / HTML) and the
email / SMS alert workflow. SMTP and Twilio are mocked: no real message is ever sent."""
import io
import re
import zipfile
import zlib

import numpy as np
import pytest

from src.simulation.local_spread import FOCUS_AREAS, FocusArea, run_local_spread

BP = FOCUS_AREAS["Bandipur Tiger Reserve"]
HOT = {"ffmc": 93.0, "bui": 60.0, "ndvi": 0.55, "risk_score": 0.2, "zone_id": "G-60", "fwi": 40.0,
       "fire_probability": 0.041, "model_version": "v3.0", "predicted_utc": "2026-10-16T10:00:00+00:00"}


def _run(dur, cond=HOT, ws=6.7, wd=81.0):
    f = FocusArea("t", BP.lat, BP.lon, 500, 500, 25)
    return run_local_spread(f, cond, ws, wd, placement="Upwind edge", n_ignition=3, duration_minutes=dur, seed=1)


# ── simulation clock ──────────────────────────────────────────────────────── #

@pytest.mark.parametrize("dur", [1, 5, 10, 15])
def test_selected_duration_is_the_simulated_time(dur):
    r = _run(dur)
    c = r.params["clock"]
    assert c["requested_s"] == dur * 60 and c["simulated_s"] == pytest.approx(dur * 60)      # exact, no overshoot
    t = [m["minutes"] for m in r.metrics]
    assert t[0] == 0 and t[-1] == pytest.approx(dur) and all(b > a for a, b in zip(t, t[1:]))
    assert max(t) <= dur + 1e-9 and c["n_frames"] == len(t) - 1 and c["status"].startswith("completed")
    assert c["frame_s"] * c["n_frames"] == pytest.approx(dur * 60)
    assert c["runtime_s"] >= 0 and c["started_utc"] and c["completed_utc"]                # runtime kept separate


def test_active_fire_progresses_frame_by_frame_and_is_not_a_repeated_state():
    r = _run(15)
    aff = [m["burned"] + m["burning"] for m in r.metrics]
    assert aff == sorted(aff) and aff[-1] > aff[0]                    # cumulative footprint grows
    assert len({m["front_distance_m"] for m in r.metrics}) > 3       # the front really moves
    assert r.params["clock"]["simulated_min"] == 15


def test_a_fire_that_goes_out_still_runs_the_clock_to_the_selected_duration():
    wet = {"ffmc": 70.0, "bui": 10.0, "ndvi": 0.55}
    r = _run(15, wet, ws=1.0)
    c = r.params["clock"]
    assert c["simulated_s"] == pytest.approx(900) and r.metrics[-1]["minutes"] == pytest.approx(15)
    if r.params.get("extinguished_at_min") is not None:
        assert c["fire_active_until_min"] < 15 and "fire activity ended" in c["status"]
        assert r.metrics[-1]["burning"] == 0                          # no invented activity afterwards


def test_reruns_reproduce_the_same_simulation():
    a, b = _run(5), _run(5)
    assert [m["burned"] for m in a.metrics] == [m["burned"] for m in b.metrics]


# ── report ────────────────────────────────────────────────────────────────── #

def _snap(dur=15):
    from src.reports.simulation_report import build_snapshot
    r = _run(dur)
    r.params["report_id"] = f"FFDT-T-{dur}"
    setup = {"location": {"name": BP.name, "lat": BP.lat, "lon": BP.lon}, "placement": "Upwind edge"}
    sc = {"temp_c": 23.2, "humidity_pct": 7, "wind_speed_ms": 6.7, "wind_from_deg": 81}
    return r, build_snapshot(r, HOT, setup, "WHAT-IF", sc, None)


def _pdf_text(pdf: bytes) -> str:
    """Decoded page-content streams (reportlab writes ASCII85 + Flate) - no extra dependency."""
    import base64
    out = []
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", pdf, re.S):
        data = m.group(1).strip()
        for dec in (lambda d: zlib.decompress(base64.a85decode(d[:-2] if d.endswith(b"~>") else d)),
                    zlib.decompress):
            try:
                out.append(dec(data).decode("latin-1"))
                break
            except Exception:
                continue
    return "".join(out)


def test_report_uses_the_actual_completed_duration_and_has_all_sections():
    from src.reports.simulation_report import sections
    r, s = _snap(5)
    assert s["clock"]["simulated_s"] == 300 and s["series"][-1]["minutes"] == 5
    titles = [x["title"] for x in sections(s)]
    for t in ("1. Executive summary", "2. Scenario and environmental conditions", "3. ML-based fire-risk assessment",
              "4. Fire Weather Index analysis", "5. Simulation method and configuration", "6. Simulation results",
              "7. Graphical analysis", "8. Data sources and data quality", "9. Findings and operational summary"):
        assert t in titles
    txt = str(sections(s))
    # no standalone "Model limitation" section / table (the ML section may state its scope in one sentence)
    secs = sections(s)
    assert not any("limitation" in x["title"].lower() for x in secs)
    assert not any(b[0] == "table" and any("limitation" in str(h).lower() for h in b[1]) for x in secs for b in x["blocks"])
    assert "5 min (300 s)" in txt and "SIMULATED" in txt and "PREDICTED" in txt
    assert "probability 4.10%" in txt and "v3.0" in txt                  # the dashboard's prediction, reused


def test_four_charts_come_from_the_recorded_series():
    from src.reports.simulation_report import chart_captions, charts
    _, s = _snap(15)
    ch = charts(s)
    assert set(ch) == {"area", "front", "rates", "indicators"} and all(v[:4] == b"\x89PNG" for v in ch.values())
    cap = chart_captions(s)
    assert f"{s['series'][-1]['burned_ha']:.2f} ha" in cap["area"] and "peaked" in cap["front"]
    assert charts(dict(s, series=s["series"][:1])) == {}                         # nothing from one value


def test_pdf_has_page_x_of_y_footer_and_no_limitation_section():
    from src.reports.simulation_report import render_pdf
    _, s = _snap(15)
    pdf = render_pdf(s)
    txt = _pdf_text(pdf)
    n = len(re.findall(rb"/Type\s*/Page[^s]", pdf))
    assert n >= 3 and f"Page 1 of {n}" in txt and f"Page {n} of {n}" in txt
    assert "(Model limitation) Tj" not in txt and "FFDT-T-15" in txt          # no standalone limitation table


def test_docx_has_the_same_content_and_page_fields():
    from src.reports.simulation_report import render_docx
    _, s = _snap(5)
    z = zipfile.ZipFile(io.BytesIO(render_docx(s)))
    doc = z.read("word/document.xml").decode()
    footer = "".join(z.read(n).decode() for n in z.namelist() if n.startswith("word/footer"))
    assert "NUMPAGES" in footer and "PAGE" in footer and "FFDT-T-5" in footer
    for t in ("Executive summary", "ML-based fire-risk assessment", "Graphical analysis", "SIMULATION ONLY"):
        assert t in doc
    assert "<w:t>Model limitation</w:t>" not in doc                          # no standalone limitation table
    assert len([n for n in z.namelist() if n.startswith("word/media/")]) == 4      # the four charts


def test_print_view_contains_only_the_report():
    from src.reports.simulation_report import render_html
    _, s = _snap(5)
    h = render_html(s, auto_print=True)
    assert "window.print()" in h and "@media print{.bar{display:none}" in h and "FFDT-T-5" in h


# ── email / SMS (mocked providers) ────────────────────────────────────────── #

@pytest.fixture()
def notify_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'n.db'}")
    for k in ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "SMTP_FROM", "SMTP_SECURITY",
              "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER", "ALERT_PROVIDER_MODE", "DASHBOARD_URL"):
        monkeypatch.delenv(k, raising=False)
    from src.notifications import config as cfg
    cfg.clear_runtime()
    # isolate from the machine's .env / secrets: every notification key starts blank, mode live
    with cfg._LOCK:
        cfg._RUNTIME.update({k: "" for k in cfg.KEYS})
        cfg._RUNTIME["ALERT_PROVIDER_MODE"] = "live"
    from src.notifications import emergency as em
    em._TEST_GUARD.clear()
    yield cfg
    cfg.clear_runtime()


def _smtp_on(cfg):
    cfg.set_runtime({"SMTP_HOST": "smtp.example.org", "SMTP_PORT": "587", "SMTP_USER": "alerts@example.org",
                     "SMTP_PASSWORD": "s3cretAppPass", "SMTP_SECURITY": "starttls"})


class FakeSMTP:
    sent, fail = [], None

    def __init__(self, host, port, timeout=None, context=None):
        self.host = host

    def starttls(self, context=None):
        pass

    def login(self, u, p):
        if FakeSMTP.fail:
            raise FakeSMTP.fail

    def noop(self):
        return (250, b"ok")

    def sendmail(self, frm, to, msg):
        FakeSMTP.sent.append(msg)
        return {}

    def quit(self):
        pass


def test_missing_and_invalid_configuration_is_reported_not_sent(notify_env):
    from src.notifications.providers import EmailProvider, SmsProvider
    e, s = EmailProvider(), SmsProvider()
    assert not e.configured and "SMTP host is not set" in e.missing()
    assert e.send("a@b.org", "x", "y").status == "not_configured"
    assert s.send("+919812345678", "x").status == "not_configured"
    notify_env.set_runtime({"TWILIO_ACCOUNT_SID": "AC123", "TWILIO_AUTH_TOKEN": "tok", "TWILIO_FROM_NUMBER": "12345"})
    probs = notify_env.sms_problems()
    assert any("32 hex" in p for p in probs) and any("E.164" in p for p in probs)
    assert notify_env.valid_email("x@forest.gov.in") and not notify_env.valid_email("x@")
    assert notify_env.valid_phone("+919812345678") and not notify_env.valid_phone("9812345678")


def test_email_success_builds_multipart_html_and_never_leaks_the_password(notify_env, monkeypatch):
    import smtplib
    from src.notifications import emergency as em
    from src.notifications.providers import EmailProvider
    _smtp_on(notify_env)
    FakeSMTP.sent, FakeSMTP.fail = [], None
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    _, s = _snap(5)
    msg = em.build_message(s, "ALERT-1")
    r = EmailProvider().send("officer@forest.gov.in", msg["subject"], msg["body"], html=msg["html"])
    assert r.status == "accepted" and r.ok
    raw = FakeSMTP.sent[-1]
    assert "multipart/alternative" in raw and "text/html" in raw and "text/plain" in raw
    assert "s3cretAppPass" not in raw
    assert "SIMULATION ONLY" in msg["body"] and "SIMULATION ONLY" in msg["html"]
    assert "hypothetical forest-fire scenario" in msg["html"]
    assert EmailProvider().test_connection().status == "connected"
    pub = notify_env.public_view()
    assert pub["SMTP_PASSWORD"] == "set" and "s3cretAppPass" not in str(pub)


def test_email_auth_failure_and_timeout_are_failures_with_redacted_reasons(notify_env, monkeypatch):
    import smtplib
    import socket
    from src.notifications.providers import EmailProvider
    _smtp_on(notify_env)
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    FakeSMTP.fail = smtplib.SMTPAuthenticationError(535, b"bad s3cretAppPass")
    r = EmailProvider().test_connection()
    assert r.status == "failed" and "authentication failed" in r.error and "s3cretAppPass" not in r.error
    FakeSMTP.fail = socket.timeout("timed out")
    assert EmailProvider().send("a@b.org", "s", "b").status == "failed"
    FakeSMTP.fail = None


class FakeResp:
    def __init__(self, code, js):
        self.status_code, self._js = code, js

    def json(self):
        return self._js


def _sms_on(cfg):
    cfg.set_runtime({"TWILIO_ACCOUNT_SID": "AC" + "a" * 32, "TWILIO_AUTH_TOKEN": "tok-secret-123",
                     "TWILIO_FROM_NUMBER": "+14155550100"})


def test_sms_statuses_are_the_providers_and_delivery_comes_only_from_a_status_check(notify_env, monkeypatch):
    import requests
    from src.notifications import alert_store as store
    from src.notifications import emergency as em
    from src.notifications.providers import SmsProvider
    _sms_on(notify_env)
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResp(201, {"status": "queued", "sid": "SM1"}))
    out = em.send_test("sms", "+919812345678", "admin", "admin")
    assert out["status"] == "queued" and out["message_id"] == "SM1"            # submitted, NOT delivered
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResp(200, {"status": "delivered"}))
    assert em.refresh_sms_statuses() == 1
    assert store.recent_deliveries(5, channel="sms")[0]["status"] == "delivered"
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResp(400, {"code": 21211, "message": "Invalid 'To'"}))
    r = SmsProvider().send("+919812345679", "x")
    assert r.status == "failed" and "21211" in r.error and "tok-secret-123" not in r.error


def test_test_alerts_validate_recipients_block_duplicates_and_need_admin(notify_env, monkeypatch):
    import smtplib
    from src.notifications import emergency as em
    _smtp_on(notify_env)
    FakeSMTP.sent, FakeSMTP.fail = [], None
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    with pytest.raises(PermissionError):
        em.send_test("email", "a@b.org", "viewer", "viewer")
    assert em.send_test("email", "not-an-email", "admin", "admin")["status"] == "failed"
    assert em.send_test("email", "a@b.org", "admin", "admin")["status"] == "accepted"
    assert em.send_test("email", "a@b.org", "admin", "admin")["status"] == "duplicate"   # rerun / double click
    import email
    m = email.message_from_string(FakeSMTP.sent[0])
    plain = next(p for p in m.walk() if p.get_content_type() == "text/plain").get_payload(decode=True).decode()
    assert len(FakeSMTP.sent) == 1 and "TEST ALERT" in plain and "No fire" in plain


def test_mock_mode_records_but_sends_nothing(notify_env, monkeypatch):
    import smtplib
    from src.notifications import emergency as em
    _smtp_on(notify_env)
    notify_env.set_runtime({"ALERT_PROVIDER_MODE": "mock"})
    FakeSMTP.sent = []
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    assert em.send_test("email", "a@b.org", "admin", "admin")["status"] == "mock" and FakeSMTP.sent == []


def test_sms_text_is_short_labelled_and_timestamped():
    from src.notifications import emergency as em
    _, s = _snap(5)
    body = em.build_message(s, "ALERT-9", sms=True)["body"]
    assert body.startswith("⚠ SIMULATION ONLY") and "NOT a confirmed fire" in body and "IST" in body
    assert "ALERT-9" in body and len(body) <= 320


def test_admin_notifications_panel_renders_status_without_secrets(notify_env, monkeypatch):
    pytest.importorskip("streamlit.testing.v1")
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    dash = Path(__file__).resolve().parents[1] / "src" / "dashboard"
    notify_env.set_runtime({"SMTP_HOST": "smtp.example.org", "SMTP_PORT": "587", "SMTP_USER": "alerts@example.org",
                            "SMTP_PASSWORD": "s3cretAppPass"})
    at = AppTest.from_file(str(dash / "pages" / "6_Admin.py"), default_timeout=120)
    at.session_state["auth_user"] = "admin"
    at.session_state["auth_role"] = "admin"
    at.session_state["_pref_offline_mode"] = True
    at.run()
    assert not at.exception
    md = "\n".join(m.value for m in at.markdown)
    assert "Email settings complete" in md and "SMS gateway: <b>Not configured</b>" in md
    assert "Automatic alerts are <b>OFF</b>" in md                    # automatic alerts: off by default
    assert "s3cretAppPass" not in md and all("s3cretAppPass" != t.value for t in at.text_input)
    assert {b.label for b in at.button} >= {"🔌 Test connection", "✉️ Send test email", "💬 Send test SMS"}
