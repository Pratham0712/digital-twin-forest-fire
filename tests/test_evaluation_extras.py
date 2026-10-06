"""Evaluation helpers: lift, persistence baseline, grid-comparable metrics,
the second hold-out season split, and the spread back-test."""
import numpy as np
import pandas as pd

from config.config import RegionConfig
from src.data_ingestion.ingestion_module import build_region_grid
from src.ml_models import evaluation_extras as ev
from src.ml_models import train_real
from src.simulation import spread_validation as sv


def test_lift_table_perfect_and_random_scores():
    y = np.array([1] * 10 + [0] * 90)
    perfect = ev.lift_table(y, y.astype(float), (0.1,)).iloc[0]
    assert perfect["recall"] == 1.0 and perfect["precision"] == 1.0 and perfect["lift_vs_random"] == 10.0
    # a constant score carries no information: precision equals the base rate, lift ~1
    flat = ev.lift_table(y, np.zeros(100), (0.5,)).iloc[0]
    assert abs(flat["lift_vs_random"] - flat["precision"] / 0.1) < 1e-9


def test_pr_curve_is_thinned():
    rng = np.random.default_rng(0)
    p = rng.random(5000)
    y = (rng.random(5000) < p * 0.2).astype(int)
    assert len(ev.pr_curve(y, p, max_points=100)) <= 100


def test_persistence_rule_uses_same_flag_count_for_model():
    df = pd.DataFrame({"fire_risk_label": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                       "fire_lag1": [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                       "fire_7d": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                       "nbr5_7d": [0] * 10})
    prob = np.array([0.9, 0.8, 0.1, 0, 0, 0, 0, 0, 0, 0], float)
    t = ev.persistence_table(df, prob)
    row = t[t["method"] == "Zone had a fire yesterday"].iloc[0]
    assert row["flagged_share"] == 0.2 and row["recall"] == 0.5 and row["precision"] == 0.5
    assert row["model_recall_same_share"] == 1.0          # model flags its top 2 -> both fires


def test_comparable_metrics_coarse_blocks_take_max():
    df = pd.DataFrame({
        "date": ["d1"] * 4, "latitude": [0.05, 0.15, 0.25, 0.75], "longitude": [0.05] * 4,
        "fire_risk_label": [0, 1, 0, 0]})
    prob = np.array([0.1, 0.9, 0.2, 0.3])
    out = ev.comparable_metrics(df, prob, block_deg=0.5)
    coarse = out[out["scoring_grid"].str.contains("blocks")].iloc[0]
    assert coarse["cell_days"] == 2 and coarse["fire_rate"] == 0.5     # first three cells merge into one block


def test_2026_is_kept_out_of_the_headline_test_season():
    dates = pd.to_datetime(["2024-02-01", "2025-02-01", "2026-02-01"])
    df = pd.DataFrame({"date": dates, "fire_risk_label": [0, 1, 0]})
    train, test = train_real.temporal_split(df)
    assert list(train["date"]) == [dates[0]] and list(test["date"]) == [dates[1]]
    assert list(train_real.second_holdout(df)["date"]) == [dates[2]]


def test_eligible_days_need_seeds_and_a_next_day():
    d = pd.date_range("2025-02-01", periods=4)
    df = pd.DataFrame({"date": np.repeat(d, 2), "zone_id": ["a", "b"] * 4,
                       "fire_risk_label": [1, 0, 0, 0, 1, 0, 1, 0]})
    days = sv.eligible_days(df)
    assert list(days) == [d[0], d[2]]                   # last day has no successor; day 2 has no seeds


def test_neighbours_excludes_seed_cells():
    m = np.zeros((3, 3), bool); m[1, 1] = True
    nb = sv._neighbours(m)
    assert nb.sum() == 8 and not nb[1, 1]


def test_spread_backtest_runs_and_random_baseline_is_exact():
    r = RegionConfig(name="T", min_lat=12.0, max_lat=12.9, min_lon=75.0, max_lon=75.9,
                     grid_resolution_deg=0.1, weather_grid_resolution_deg=0.4)
    g = build_region_grid(r)
    rng = np.random.default_rng(0)
    rows = []
    fire = np.zeros(len(g), bool); fire[[5, 6]] = True
    for d in pd.date_range("2025-02-01", periods=6):
        rows.append(pd.DataFrame({"date": d, "zone_id": g["zone_id"].values,
                                  "fire_risk_label": fire.astype(int),
                                  "ffmc": rng.uniform(70, 95, len(g)), "bui": 30.0, "wx_wind_speed_ms": 3.0}))
        fire = fire | (rng.random(len(g)) < 0.02)
    df = pd.concat(rows)
    res = sv.backtest(df, g, None, 0.1, sv.eligible_days(df), replicates=3, horizons_h=(2, 24))
    assert set(res["horizon_hours"]) == {2, 24} and res["method"].nunique() == 3
    assert (res["precision"].dropna().between(0, 1)).all()
    ca = res[res["method"].str.startswith("Cellular") & (res["horizon_hours"] == 24)].iloc[0]
    rnd = res[res["method"].str.startswith("Random") & (res["horizon_hours"] == 24)].iloc[0]
    assert ca["cells_predicted"] == rnd["cells_predicted"]          # same count by construction
