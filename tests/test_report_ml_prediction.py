"""The exported report carries the actual model prediction of the simulated scenario: real pipeline (weather ->
FWI codes -> features -> XGBoost) for wet / moderate / dry What-If inputs, identical in PDF / DOCX / print."""
import io
import re
import zipfile

import pytest

from tests.test_duration_report_alerts import _pdf_text

LAT, LON = 11.6560, 76.5800
SCEN = {"wet": (22, 95, 1, 10), "moderate": (28, 55, 4, 0), "dry": (40, 10, 12, 0)}


def _case(name, placement="Upwind edge", seed=1):
    import logging
    logging.disable(logging.WARNING)
    from src.dashboard.live_modes import zone_risk
    from src.digital_twin.twin_state import DigitalTwin
    from src.ml_models.model_registry import load_xgb
    from src.reports.simulation_report import build_snapshot
    from src.simulation.local_spread import FOCUS_AREAS, FocusArea, run_local_spread
    t, h, w, r = SCEN[name]
    pw = {"temperature_c": t, "humidity_pct": h, "wind_speed_ms": w, "wind_deg": 81.0, "pressure_hpa": 1010.0,
          "precipitation_mm": r, "clouds_pct": 0, "weather_main": "x"}
    # a LIVE-style What-If: no satellite detections in the window (as in the user's runs)
    tw = DigitalTwin(ml_model=load_xgb("xgboost_real.json"), offline=True,
                     scenario={"point_weather": pw, "weather_mode": "scenario", "observed_hotspots": [], "firms_mode": "live"})
    tw.refresh()
    cond = zone_risk(tw, LAT, LON)
    b = FOCUS_AREAS["Bandipur Tiger Reserve"]
    res = run_local_spread(FocusArea("t", b.lat, b.lon, 500, 500, 25), cond, w, 81.0, duration_minutes=5, seed=seed,
                           placement=placement, n_ignition=3)
    res.params["report_id"] = f"FFDT-ML-{name}-{placement[:3]}"
    snap = build_snapshot(res, cond, {"location": {"name": b.name, "lat": b.lat, "lon": b.lon}, "placement": placement},
                          "WHAT-IF", {"temp_c": t, "humidity_pct": h, "wind_speed_ms": w, "wind_from_deg": 81}, None, tw)
    return tw, cond, snap


@pytest.fixture(scope="module")
def cases():
    return {k: _case(k) for k in SCEN}


def test_prediction_follows_the_scenario_and_is_the_model_output(cases):
    from src.ml_models.prediction_service import feature_schema, prepare_features
    p = {k: c[2]["prediction"]["probability"] for k, c in cases.items()}
    assert p["wet"] < p["moderate"] < p["dry"]                          # wetter -> lower, drier/hotter/windier -> higher
    for k, (tw, cond, snap) in cases.items():
        g = tw.current_snapshot.processed_grid
        row = g[g["zone_id"] == cond["zone_id"]]
        X, missing = prepare_features(row, feature_schema(tw.ml_model))
        assert not missing
        direct = float(tw.ml_model.model.predict_proba(X)[:, 1][0])      # recompute from the received features
        assert snap["prediction"]["probability"] == pytest.approx(direct, abs=1e-6)
        t, h, w, r = SCEN[k]
        assert float(row["wx_temperature_c"].iloc[0]) == t and float(row["wx_humidity_pct"].iloc[0]) == h
        assert float(row["wx_precipitation_mm"].iloc[0]) == r                # the scenario's inputs, not a baseline


def test_report_shows_probability_index_and_simulation_separately(cases):
    from src.reports.simulation_report import _pct, render_docx, render_html, render_pdf, sections
    for k, (_, _, snap) in cases.items():
        pct = _pct(snap["prediction"]["probability"])
        txt = str(sections(snap))
        score = f"{snap['prediction']['risk_score']:.0f}%"
        assert f"fire-occurrence probability {pct}" in txt and f"ML Fire Risk Score {score}" in txt
        assert "not interchangeable" in txt and "Observed fire rate" in txt
        assert "not tied to the 5 min simulation" in txt and "not a model input" in txt
        pdf = _pdf_text(render_pdf(snap))
        doc = zipfile.ZipFile(io.BytesIO(render_docx(snap))).read("word/document.xml").decode()
        html = render_html(snap)
        for out in (pdf, doc, html):
            assert pct in out and snap["report_id"] in out and f"ML Fire Risk Score {score}" in out                 # same value, same report in every format


def test_extreme_inputs_are_flagged_as_outside_the_training_range(cases):
    oor = {n for n, *_ in cases["dry"][2]["prediction"]["out_of_range"]}
    assert {"humidity", "wind speed", "FFMC"} <= oor
    assert "Inputs outside the model's training range" in str(__import__("src.reports.simulation_report",
                                                                          fromlist=["sections"]).sections(cases["dry"][2]))


def test_hypothetical_ignition_does_not_change_the_prediction():
    _, _, a = _case("moderate", "Upwind edge")
    _, _, b = _case("moderate", "Centre", seed=7)
    assert a["prediction"]["probability"] == pytest.approx(b["prediction"]["probability"])
    assert a["results"] != b["results"] or a["report_id"] != b["report_id"]


def test_risk_score_is_the_validated_percentile_of_the_model_probability(cases):
    from src.ml_models.risk_score import RiskScore, category
    rs = RiskScore.load()
    sc = {k: c[2]["prediction"]["risk_score"] for k, c in cases.items()}
    assert sc["wet"] < sc["moderate"] < sc["dry"]                       # dynamic, follows the scenario
    for k, (_, cond, snap) in cases.items():
        pr = snap["prediction"]
        assert pr["risk_score"] == pytest.approx(float(rs.score(pr["probability"])), abs=0.06)   # exact transform
        assert pr["risk_score"] == pytest.approx(cond["ml_risk_score"], abs=0.06)              # same as dashboard
        assert pr["risk_score_category"] == category(pr["risk_score"])
    val = rs.meta["validation"]
    rates = [v["observed_fire_rate"] for v in val]
    assert rates == sorted(rates) and rates[-1] > 10 * rates[0]           # categories validated on the held-out season
    assert [v["category"] for v in val] == ["LOW", "MODERATE", "HIGH", "VERY HIGH", "EXTREME"]


def test_risk_score_category_bands_and_monotonic_transform():
    import numpy as np
    from src.ml_models.risk_score import RiskScore, category
    rs = RiskScore.load()
    p = np.array([0.0, 0.002, 0.01, 0.05, 0.2, 0.6, 1.0])
    s = rs.score(p)
    assert np.all(np.diff(s) >= 0) and s[0] >= 0 and s[-1] <= 100
    assert category(49.9) == "LOW" and category(50) == "MODERATE" and category(90) == "VERY HIGH" and category(99) == "EXTREME"
