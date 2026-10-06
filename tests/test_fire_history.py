"""
Tests for the v2 model features (src/ml_models/fire_history.py) and the
multi-state India builder. All synthetic and offline - no network, no real
data files needed.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.config import RegionConfig
from src.data_ingestion.ingestion_module import build_region_grid
from src.ml_models.fire_history import (
    FEATURE_COLUMNS_V2, FIRE_HISTORY_COLUMNS, HISTORY_DAYS, StaticSourceMask, ZoneClimatology,
    add_climatology_training, add_fire_history_training, confident_detections,
    live_fire_history_features,
)

SMALL = RegionConfig(name="Test box", min_lat=12.0, max_lat=12.8, min_lon=75.0, max_lon=75.8,
                     grid_resolution_deg=0.1, weather_grid_resolution_deg=0.4)


def _labelled_days(grid, n_days=40, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-03-01", periods=n_days)
    rows = []
    for d in dates:
        lab = (rng.random(len(grid)) < 0.08).astype(int)
        rows.append(pd.DataFrame({"zone_id": grid["zone_id"], "date": d,
                                  "latitude": grid["latitude"], "longitude": grid["longitude"],
                                  "fire_risk_label": lab}))
    return pd.concat(rows, ignore_index=True)


def test_no_ndvi_in_model_features():
    assert "ndvi" not in FEATURE_COLUMNS_V2
    assert len(FEATURE_COLUMNS_V2) == len(set(FEATURE_COLUMNS_V2))


def test_live_features_match_training_features_exactly():
    """Train/serve parity: the live path, fed the same detections as
    hotspots, must reproduce the training features for that day."""
    grid = build_region_grid(SMALL)
    df = add_fire_history_training(_labelled_days(grid), grid)
    for day in pd.to_datetime(["2024-03-05", "2024-03-20", "2024-04-09"]):
        past = df[(df.date < day) & (df.date >= day - pd.Timedelta(days=HISTORY_DAYS))
                  & (df.fire_risk_label == 1)]
        hotspots = pd.DataFrame({"latitude": past.latitude, "longitude": past.longitude,
                                 "acq_date": past.date.dt.strftime("%Y-%m-%d"), "confidence": "h"})
        live = live_fire_history_features(grid, hotspots, SMALL.grid_resolution_deg, today=day)
        train = df[df.date == day].set_index("zone_id")[FIRE_HISTORY_COLUMNS]
        diff = (live.set_index("zone_id").loc[train.index, FIRE_HISTORY_COLUMNS] - train).abs().max()
        assert diff.max() == 0, diff


def test_todays_detections_are_never_features():
    grid = build_region_grid(SMALL)
    today = pd.Timestamp("2024-03-10")
    hs = pd.DataFrame({"latitude": grid.latitude[:5], "longitude": grid.longitude[:5],
                       "acq_date": today.strftime("%Y-%m-%d"), "confidence": "h"})
    live = live_fire_history_features(grid, hs, SMALL.grid_resolution_deg, today=today)
    assert live[["fire_lag1", "fire_7d", "fire_10d", "nbr3_lag1"]].to_numpy().sum() == 0


def test_history_does_not_cross_season_gap():
    grid = build_region_grid(SMALL)
    a = _labelled_days(grid, 10)
    b = _labelled_days(grid, 10)
    b["date"] = b["date"] + pd.Timedelta(days=300)          # next season
    df = add_fire_history_training(pd.concat([a, b], ignore_index=True), grid)
    first_of_b = df[df.date == b.date.min()]
    assert first_of_b[["fire_lag1", "fire_7d", "fire_10d", "region_fires_lag1"]].to_numpy().sum() == 0


def test_climatology_is_leave_one_season_out():
    grid = build_region_grid(SMALL)
    s1 = _labelled_days(grid, 20, seed=1)
    s2 = _labelled_days(grid, 20, seed=2)
    s2["date"] = s2["date"] + pd.DateOffset(years=1)
    df = pd.concat([s1, s2], ignore_index=True)
    out, table = add_climatology_training(df, grid, train_mask=pd.Series(True, index=df.index))
    z = grid.zone_id.iloc[0]
    rate_s2 = s2[s2.zone_id == z].fire_risk_label.mean()
    got = out[(out.zone_id == z) & (out.date.dt.year == s1.date.dt.year.iloc[0])].zone_clim.iloc[0]
    assert got == pytest.approx(rate_s2)   # season-1 rows see only season-2's rate
    assert set(table.columns) >= {"zone_clim", "nbr_clim", "latitude", "longitude"}


def test_climatology_falls_back_to_mean_for_unknown_cells():
    table = pd.DataFrame({"zone_id": ["G-0", "G-1"], "latitude": [12.05, 12.15],
                          "longitude": [75.05, 75.05], "zone_clim": [0.1, 0.3], "nbr_clim": [0.2, 0.2]})
    grid = pd.DataFrame({"zone_id": ["A", "B"], "latitude": [12.05, 30.0], "longitude": [75.05, 80.0]})
    out = ZoneClimatology(table).transform(grid)
    assert out.zone_clim.tolist() == pytest.approx([0.1, 0.2])


def test_static_industrial_sources_are_removed():
    archive = pd.DataFrame({"latitude": [15.15, 13.0], "longitude": [76.65, 75.0], "type": [2, 0]})
    mask = StaticSourceMask.from_archive(archive)
    live = pd.DataFrame({"latitude": [15.151, 13.0, 14.0], "longitude": [76.651, 75.0, 76.0]})
    kept = mask.filter(live)
    assert len(kept) == 2 and 15.151 not in kept.latitude.values
    labelled = confident_detections(pd.DataFrame({"latitude": [1, 2, 3], "longitude": [1, 2, 3],
                                                  "type": [0, 2, 3], "confidence": ["h", "h", "n"]}))
    assert labelled.latitude.tolist() == [1]


def _load_india_builder():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_india_regions_dataset.py"
    spec = importlib.util.spec_from_file_location("india_builder", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_india_builder_trains_from_cached_files(tmp_path, monkeypatch):
    """End-to-end on tiny synthetic 'cached downloads' for one state: build
    the dataset, train the pooled model, write per-state metrics."""
    b = _load_india_builder()
    for name in ("raw", "proc", "models"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(b, "DATA_RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(b, "DATA_PROCESSED_DIR", tmp_path / "proc")
    monkeypatch.setattr(b, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(b, "DATASET_PATH", tmp_path / "proc" / "india.csv")
    states = {"Uttarakhand": b.india_states()["Uttarakhand"]}
    r = states["Uttarakhand"]
    rng = np.random.default_rng(0)

    for year in b.YEARS:
        days = pd.date_range(f"{year}-03-01", f"{year}-03-31")
        k = 400
        pd.DataFrame({"latitude": rng.uniform(r.min_lat, r.max_lat, k),
                      "longitude": rng.uniform(r.min_lon, r.max_lon, k),
                      "acq_date": rng.choice(days.strftime("%Y-%m-%d"), k),
                      "confidence": "h", "type": 0}) \
            .to_csv(tmp_path / "raw" / f"historical_fires_india_{year}.csv", index=False)
    wx = [dict(date=d.strftime("%Y-%m-%d"), temp=rng.uniform(15, 35), rhum=rng.uniform(20, 90),
               prcp=0.0, wspd=rng.uniform(2, 20), pres=1010.0, latitude=la, longitude=lo)
          for y in b.YEARS for d in pd.date_range(f"{y}-03-01", f"{y}-03-31")
          for la in (r.min_lat, r.max_lat) for lo in (r.min_lon, r.max_lon)]
    pd.DataFrame(wx).to_csv(tmp_path / "raw" / "historical_weather_uttarakhand.csv", index=False)

    data = b.build(states, b.YEARS, skip_fetch=False)
    assert set(data.region) == {"Uttarakhand"} and data.fire_risk_label.sum() > 0
    b.train(data, states)

    assert (tmp_path / "models" / "xgboost_india.json").exists()
    assert (tmp_path / "models" / "zone_climatology_india.csv").exists()
    metrics = pd.read_csv(tmp_path / "proc" / "india_model_by_state.csv")
    assert "Uttarakhand" in metrics.region.values


def test_alert_threshold_meets_accuracy_target_on_validation():
    from sklearn.metrics import accuracy_score
    from src.ml_models.model_trainer import _pick_accuracy_threshold
    rng = np.random.default_rng(3)
    y = (rng.random(20000) < 0.05).astype(int)
    p = np.clip(0.03 + 0.5 * y * rng.random(20000) + 0.1 * rng.random(20000), 0, 1)
    t_alert = _pick_accuracy_threshold(y, p, target=0.92)
    assert accuracy_score(y, p >= t_alert) >= 0.92
    # lowest such threshold: one step lower must miss the target
    assert accuracy_score(y, p >= t_alert - 0.005) < 0.92 or t_alert <= 0.01


def test_literature_protocol_helpers():
    """Balancing is exactly 1:1 (or the requested counts) and metrics use the given threshold."""
    from src.ml_models.literature_protocol import _balance, _metrics
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"fire_risk_label": [1] * 30 + [0] * 500, "x": np.arange(530)})
    b = _balance(df, rng)
    assert b["fire_risk_label"].value_counts().to_dict() == {1: 30, 0: 30}
    small = _balance(df, rng, n_pos=20, n_neg=10)
    assert small["fire_risk_label"].value_counts().to_dict() == {1: 20, 0: 10}
    y = pd.Series([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.9])
    assert _metrics(y, p, 0.5)["accuracy"] == 0.5 and _metrics(y, p, 0.35)["recall"] == 1.0
