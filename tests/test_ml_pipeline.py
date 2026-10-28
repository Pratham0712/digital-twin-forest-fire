"""ML pipeline: shared prediction service, schema / input safety, risk categories,
model versioning, the promotion gate (an inferior candidate is rejected), and
that changing the prediction inputs changes every dependent dashboard value.
No network, no API quota: the twin runs offline on scenario weather."""
import json
import logging
import shutil

import numpy as np
import pandas as pd
import pytest

from config.config import MODELS_DIR
from src.ml_models import model_selection as ms
from src.ml_models import prediction_service as ps


# ── schema / input safety ─────────────────────────────────────────────────── #

def test_prepare_features_orders_columns_and_neutralises_bad_values():
    f = pd.DataFrame({"b": [1.0, np.inf], "a": [True, False], "z": [9, 9]})
    X, missing = ps.prepare_features(f, ["a", "b", "c"])
    assert list(X.columns) == ["a", "b", "c"]                       # model order, extra column dropped
    assert X["a"].tolist() == [1.0, 0.0] and np.isnan(X.loc[1, "b"])  # bool -> int, inf -> missing
    assert missing == ["c"] and X["c"].isna().all()                 # reported, never invented


def _serving_model():
    from src.ml_models.model_registry import load_xgb
    m = load_xgb(ms.SERVING_MODEL)
    assert m is not None
    return m


def test_loaded_model_uses_its_trained_operating_points_and_schema():
    meta = json.loads((MODELS_DIR / ms.SERVING_META).read_text())
    m = _serving_model()
    assert m.threshold_ == pytest.approx(meta["xgboost_decision_threshold"])
    assert m.alert_threshold_ == pytest.approx(meta["xgboost_alert_threshold"])
    assert ps.feature_schema(m) == meta["feature_columns"]          # training order == inference order


def test_prediction_ranges_and_categories_follow_the_documented_bands():
    from src.digital_twin.twin_state import AlertEngine
    from src.ml_models.risk_index import RiskIndex
    m = _serving_model()
    cols = ps.feature_schema(m)
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(50, 30, size=(200, len(cols))), columns=cols)
    frame.iloc[0, 0] = np.inf                                       # bad input must not crash
    ri = RiskIndex.load(MODELS_DIR, ms.SERVING_INDEX)
    res = ps.predict(m, frame, ri)
    assert ((res.fire_probability >= 0) & (res.fire_probability <= 1)).all()
    assert ((res.risk_index >= 0) & (res.risk_index <= 1)).all()
    order = np.argsort(res.fire_probability)
    assert np.all(np.diff(res.risk_index[order]) >= -1e-12)         # index is monotone in the probability
    assert [b for b in AlertEngine.SEVERITY_BANDS] == [tuple(b) for b in ps.SEVERITY_BANDS]
    eng = AlertEngine()
    assert res.categories == [eng.classify(v) for v in res.risk_index]
    assert ps.categorize(0.19) == "LOW" and ps.categorize(0.2) == "MODERATE" and ps.categorize(0.6) == "EXTREME"
    assert ps.categorize(float("nan")) == "UNKNOWN"
    assert res.model_version and res.predicted_utc and res.missing_features == []


def test_model_version_comes_from_the_registry(tmp_path):
    for f in (ms.SERVING_MODEL, ms.SERVING_META, ms.SERVING_INDEX):
        shutil.copy2(MODELS_DIR / f, tmp_path / f)
    reg = ms.ensure_baseline_registered(tmp_path)
    v = reg["active"]
    assert ps.model_version(ms.SERVING_MODEL, tmp_path) == v
    assert (tmp_path / ms.VERSIONS_DIR / v / ms.SERVING_MODEL).exists()       # rollback copy kept
    assert ms.ensure_baseline_registered(tmp_path)["versions"] == reg["versions"]   # idempotent


# ── promotion gate / retraining workflow ──────────────────────────────────── #

def _m(pr, roc, ll, rec):
    return {"pr_auc": pr, "roc_auc": roc, "log_loss": ll, "early_warning": {"recall": rec}}


def test_gate_rejects_inferior_or_unsafe_candidates_and_accepts_a_real_gain():
    base = _m(0.26, 0.87, 0.135, 0.83)
    assert not ms.gate(base, _m(0.25, 0.87, 0.135, 0.83))["passed"]          # worse ranking
    assert not ms.gate(base, _m(0.262, 0.87, 0.135, 0.83))["passed"]         # gain below the minimum
    assert not ms.gate(base, _m(0.30, 0.88, 0.130, 0.70))["passed"]          # misses more fires
    assert not ms.gate(base, _m(0.30, 0.88, 0.140, 0.83))["passed"]          # worse probabilities
    assert ms.gate(base, _m(0.30, 0.88, 0.130, 0.84))["passed"]


def _synthetic(seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(np.concatenate([rng.choice(pd.date_range(f"{y}-01-01", f"{y}-05-31"), 1500)
                                           for y in (2023, 2024, 2025)]))
    x1, x2 = rng.normal(size=len(dates)), rng.normal(size=len(dates))
    y = (rng.random(len(dates)) < 1 / (1 + np.exp(-(2.2 * x1 - 3.0)))).astype(int)
    return pd.DataFrame({"date": dates, "signal": x1, "noise": x2, ms.LABEL: y})


def test_retraining_rejects_an_inferior_candidate_and_keeps_the_serving_model(tmp_path):
    import xgboost as xgb
    df = _synthetic()
    train, _ = ms.split(df)
    serving = xgb.XGBClassifier(n_estimators=60, max_depth=2, random_state=0).fit(train[["signal", "noise"]],
                                                                                 train[ms.LABEL])
    serving.save_model(str(tmp_path / ms.SERVING_MODEL))
    (tmp_path / ms.SERVING_META).write_text(json.dumps({"feature_columns": ["signal", "noise"], "feature_set_version": 1,
                                                        "xgboost_decision_threshold": 0.05,
                                                        "xgboost_alert_threshold": 0.2}))
    ms.ensure_baseline_registered(tmp_path)
    sha_before = ms._sha(tmp_path / ms.SERVING_MODEL)
    bad = {"drops_the_signal": {"drop": ["signal"], "params": dict(n_estimators=40, max_depth=2, random_state=0)}}
    rep = ms.evaluate(df, bad, models_dir=tmp_path, out_dir=tmp_path, n_jobs=1)
    cand = rep["candidates"][0]
    assert cand["metrics"]["pr_auc"] < rep["baseline"]["metrics"]["pr_auc"]
    assert not cand["gate"]["passed"] and rep["decision"].startswith("KEEP")
    assert ms.promote(rep, tmp_path) is None
    assert ms._sha(tmp_path / ms.SERVING_MODEL) == sha_before                  # serving model untouched
    saved = json.loads((tmp_path / ms.REPORT_JSON).read_text())
    for k in ("roc_auc", "pr_auc", "brier", "log_loss", "calibration"):
        assert k in saved["baseline"]["metrics"]
    for k in ("precision", "recall", "f1", "macro_f1", "weighted_f1", "confusion_matrix", "per_class",
              "false_positive_rate", "false_negative_rate"):
        assert k in saved["baseline"]["metrics"]["early_warning"]
    assert (tmp_path / ms.REPORT_CSV).exists()


def test_promotion_of_a_better_candidate_and_rollback(tmp_path):
    import xgboost as xgb
    df = _synthetic(1)
    train, _ = ms.split(df)
    cols = ["noise", "signal"]
    # weak serving model: trained with the informative column blanked out
    weak = xgb.XGBClassifier(n_estimators=3, max_depth=1, random_state=0).fit(
        train[cols].assign(signal=0.0), train[ms.LABEL])
    weak.save_model(str(tmp_path / ms.SERVING_MODEL))
    (tmp_path / ms.SERVING_META).write_text(json.dumps({"feature_columns": cols, "feature_set_version": 1,
                                                        "xgboost_decision_threshold": 0.5,
                                                        "xgboost_alert_threshold": 0.6}))
    old = ms.ensure_baseline_registered(tmp_path)["active"]
    good = {"uses_signal": {"drop": [], "params": dict(n_estimators=60, max_depth=2, random_state=0)}}
    rep = ms.evaluate(df, good, models_dir=tmp_path, out_dir=tmp_path, n_jobs=1)
    assert rep["candidates"][0]["gate"]["passed"], rep["candidates"][0]["gate"]
    new = ms.promote(rep, tmp_path)
    assert new and ms.load_registry(tmp_path)["active"] == new
    assert ps.model_version(ms.SERVING_MODEL, tmp_path) == new
    assert json.loads((tmp_path / ms.SERVING_META).read_text())["model_version"] == new
    ms.activate(old, tmp_path)                                                 # rollback
    assert ps.model_version(ms.SERVING_MODEL, tmp_path) == old


# ── dynamic, consistent outputs through the real pipeline ─────────────────── #

def _pw(t, h, w, rain=0.0):
    return {"temperature_c": t, "humidity_pct": h, "wind_speed_ms": w, "wind_deg": 225.0, "pressure_hpa": 1013.0,
            "precipitation_mm": rain, "clouds_pct": 0, "weather_main": "Clear"}


def _twin(pw):
    from src.digital_twin.twin_state import DigitalTwin
    t = DigitalTwin(ml_model=_serving_model(), offline=True, scenario={"point_weather": pw, "weather_mode": "scenario"})
    t.refresh()
    return t


def test_changing_inputs_updates_the_prediction_and_every_dependent_value():
    from src.dashboard.live_modes import zone_risk
    lat, lon = 11.656, 76.58                                   # Bandipur preset
    out = {}
    for name, pw in (("cool_wet", _pw(22, 90, 1, 5)), ("hot_dry_windy", _pw(40, 12, 12))):
        tw = _twin(pw)
        snap, summ = tw.current_snapshot, tw.get_summary()
        z = zone_risk(tw, lat, lon)
        g = snap.processed_grid.reset_index(drop=True)
        i = int(g.index[g["zone_id"] == z["zone_id"]][0])
        # one prediction, everywhere: zone value == snapshot value == summary aggregates
        assert z["risk_score"] == pytest.approx(float(snap.risk_scores[i]))
        assert z["risk_score"] == pytest.approx(float(g.loc[i, "risk_index"]))
        assert z["severity"] == ps.categorize(z["risk_score"]) == tw.alert_engine.classify(z["risk_score"])
        assert summ["max_risk_score"] == pytest.approx(float(snap.risk_scores.max()))
        assert summ["mean_risk_score"] == pytest.approx(float(snap.risk_scores.mean()))
        assert summ["total_alerts"] == int((snap.risk_scores >= tw.alert_engine.threshold).sum())
        assert summ["model_version"] == snap.prediction["model_version"] and summ["predicted_utc"]
        assert snap.prediction["missing_features"] == []         # every model feature is supplied
        assert float(g.loc[i, "wx_temperature_c"]) == pw["temperature_c"]   # the inputs actually used
        out[name] = (z, summ, g.loc[i, "fire_probability"])
    cool, hot = out["cool_wet"], out["hot_dry_windy"]
    assert hot[0]["ffmc"] > cool[0]["ffmc"] and hot[0]["fwi"] > cool[0]["fwi"]
    assert hot[2] > cool[2]                                      # model probability responds
    assert hot[0]["risk_score"] > cool[0]["risk_score"]          # risk index responds
    assert hot[1]["mean_risk_score"] > cool[1]["mean_risk_score"]
    assert hot[1]["total_alerts"] > cool[1]["total_alerts"]
