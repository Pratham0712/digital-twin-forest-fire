"""One ML Fire Risk Score everywhere: the PDF / Word report, the HTML and plain-text e-mail and the SMS read the
same value from the same simulation snapshot (alert_facts), formatted by risk_score.format_score.
Controlled mock predictions; nothing is sent (no provider is called)."""
from __future__ import annotations

import inspect
import re

import pytest

from src.ml_models.risk_score import category, format_score
from src.notifications import emergency as em
from tests.test_duration_report_alerts import _snap


@pytest.fixture(scope="module")
def snap():
    _, s = _snap(5)              # built by the real simulation_report.build_snapshot pipeline
    return s


def _pred(s, score, prob, rid=None):
    out = dict(s, prediction=dict(s["prediction"], risk_score=score, risk_score_category=category(score),
                                  probability=prob))
    return dict(out, report_id=rid) if rid else out


def _report_line(s):
    from src.reports.simulation_report import _risk_line
    return _risk_line(s)


def test_report_email_and_sms_show_the_same_score(snap):
    s = _pred(snap, 42.3, 0.0141)
    rep = _report_line(s)
    assert "ML Fire Risk Score 42% (LOW)" in rep and "1.41%" in rep
    m = em.build_message(s, "ALERT-1")
    for part in (m["body"], m["html"]):                                  # both e-mail alternatives
        assert "42% — LOW" in part and "Fire-occurrence probability" in part and "1.41%" in part
    assert "ML FIRE RISK SCORE: 42% — LOW" in m["html"]                  # badge
    sms = em.manual_sms_text(s, "MSIM-1")
    assert "📊 ML Fire Risk Score: 42% (LOW)" in sms and "🔥 Fire probability: 1.41%" in sms
    from src.reports.simulation_report import _fmt_score
    assert _fmt_score(42.3) == format_score(42.3) == "42%"               # one formatter for all outputs


def test_score_follows_the_prediction_and_never_leaks(snap):
    a = _pred(snap, 12.0, 0.003, rid="FFDT-A")
    b = _pred(snap, 96.4, 0.31, rid="FFDT-B")
    sa, sb = em.manual_sms_text(a), em.manual_sms_text(b)
    assert "12% (LOW)" in sa and "96% (EXTREME)" in sb and "96%" not in sa and "12%" not in sb
    assert "FFDT-A" in sa and "FFDT-B" in sb
    ea, eb = em.build_message(a, "X")["body"], em.build_message(b, "Y")["body"]
    assert "12% — LOW" in ea and "96% — EXTREME" in eb
    assert em.alert_facts(a)["ml_fire_risk_score_pct"] == 12.0          # no module-level cache between calls


@pytest.mark.parametrize("score,level", [(10, "LOW"), (60, "MODERATE"), (80, "HIGH"), (92, "VERY HIGH"), (97, "EXTREME")])
def test_classification_is_the_existing_one(snap, score, level):
    f = em.alert_facts(_pred(snap, float(score), 0.05))
    assert f["risk_level"] == level == category(score)


def test_probability_stays_separate(snap):
    s = _pred(snap, 42.0, 0.0141)
    f = em.alert_facts(s)
    assert f["ml_fire_risk_score_pct"] == 42.0 and f["fire_probability_pct"] == pytest.approx(1.41)
    body = em.build_message(s, "A")["body"]
    assert "risk index" not in body.lower()                              # legacy index not shown as a score
    assert em.percent_from_fraction(68) is None and em.percent_value(68) == 68   # no double x100
    sms = em.manual_sms_text(s)
    assert sms.count("42%") == 1 and sms.count("1.41%") == 1


def test_sms_is_concise_and_labelled(snap):
    sms = em.manual_sms_text(_pred(snap, 42.0, 0.0141), "MSIM-1")
    lines = sms.splitlines()
    assert lines[0] == "🚨🌲 FOREST FIRE DIGITAL TWIN"
    assert lines[1] == "⚠️ SIMULATION ONLY — HYPOTHETICAL SCENARIO — NOT A CONFIRMED FIRE"
    assert lines[2] == f"📍 {snap['location']['name']}"
    assert sum("CONFIRMED FIRE" in l for l in lines) == 1                # the warning appears once
    assert lines[-1] == f"🆔 Report: {snap['report_id']}"
    assert any(l.startswith("🌡️ Temp:") and "| Humidity:" in l for l in lines)
    assert any(l.startswith("💨 Wind:") for l in lines) and any(l.startswith("➡️ Fire front:") for l in lines)
    assert em.sms_segments(sms)["segments"] <= 6


def test_missing_fields_do_not_crash_or_invent(snap):
    bare = dict(snap, prediction={}, risk={}, weather={}, results={}, location={})
    sms = em.manual_sms_text(bare)
    assert "ML Fire Risk Score: not available" in sms and "Fire probability" not in sms and "Temp" not in sms
    assert "Wind" not in sms and "burned area" not in sms and "Location not available" in sms
    m = em.build_message(bare, "R-1")
    assert "Model risk: not available" in m["body"] and "NOT AVAILABLE" in m["html"]
    assert em.manual_email_message(bare, "R-1")["subject"]


def test_alert_code_never_predicts_by_itself():
    for fn in (em.alert_facts, em.manual_sms_text, em.build_message, em._html_email, em.score_lines):
        code = inspect.getsource(fn).split('"""')[-1]                    # code after the docstring
        assert not re.search(r"predict\w*\(|RiskScore|\.score\(|\bml_model\b|xgboost|random", code), fn.__name__


def test_hypothetical_and_observed_wording(snap):
    mail = em.manual_email_message(snap, "R")
    assert "SIMULATION TEST ALERT - HYPOTHETICAL SCENARIO - NOT A CONFIRMED FIRE" in mail["html"]
    assert em.SIM_ONLY in mail["body"]
    obs = dict(snap, mode="LIVE", firms={"n_valid": 3, "latest_acq_utc": None, "fetched_utc": None})
    o = em.manual_sms_text(obs)
    assert "NOT GROUND-VERIFIED" in o and "NASA FIRMS: 3 hotspot(s) observed" in o and "CONFIRMED FIRE" not in o
    assert em.alert_facts(obs)["simulation_mode"] == "REAL" and em.alert_facts(snap)["simulation_mode"] == "WHAT-IF"


def test_email_design_kept(snap):
    h = em.build_message(snap, "ALERT-1")["html"]
    for keep in ("FOREST FIRE DIGITAL TWIN", "background:#13233a", "border-bottom:4px solid #e8710a", "📍 Location",
                 "🕒 Simulation time", "🌦️ Weather used", "🔥 Simulated spread", "📄 Report", "Advisory", em.SIM_ONLY):
        assert keep in h, keep


def test_automatic_alerts_use_the_stored_score():
    from src.notifications import auto_alerts as aa
    m = aa.build_messages("Karnataka", [{"zone_id": "G-1", "lat": 12.0, "lon": 76.0, "confidence": 0.5,
                                         "threshold": 0.35, "reason": "r", "ml_risk_score": 97.6}], "AUTO-1", "live")
    assert "📊 ML Fire Risk Score: 98% (EXTREME)" in m["sms"] and "🔥 Fire probability: 50.0%" in m["sms"]
    assert "ML Fire Risk Score 98% (EXTREME)" in m["body"] and "Dashboard alert level: EXTREME" in m["sms"]
    n = aa.build_messages("K", [{"zone_id": "G-2", "lat": 12.0, "lon": 76.0, "confidence": 0.4, "threshold": 0.35,
                                 "reason": "r"}], "AUTO-2", "live")
    assert "ML Fire Risk Score: not available" in n["sms"]               # older grids without the column


def test_auto_refresh_passes_the_grid_score(monkeypatch, tmp_path):
    import pandas as pd
    from tests.test_sms_auto_alerts import FakeProv, _on, _twin
    from src.notifications import config as cfg
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'a.db'}")
    cfg.clear_runtime()
    with cfg._LOCK:
        cfg._RUNTIME.update({k: "" for k in cfg.KEYS})
        cfg._RUNTIME["ALERT_PROVIDER_MODE"] = "live"
    try:
        aa = _on(cfg, min_confidence=0.1)
        tw = _twin([("Z-9", 0.6, "EXTREME")])
        tw.current_snapshot.processed_grid = pd.DataFrame({"zone_id": ["Z-9"], "fire_probability": [0.6],
                                                           "ml_risk_score": [99.2]})
        sms = FakeProv()
        aa.process_refresh(tw, sms_provider=sms, email_provider=FakeProv())
        assert sms.sent and "ML Fire Risk Score: 99% (EXTREME)" in sms.sent[0][1][0]
    finally:
        cfg.clear_runtime()
