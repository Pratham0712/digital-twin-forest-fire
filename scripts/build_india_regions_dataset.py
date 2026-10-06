"""
Builds the real multi-state training set for India's curated fire-prone
states and (with --train) trains the pan-India model that serves every
region in src/regions.py except Karnataka.

Pipeline (run on a machine with internet + FIRMS_MAP_KEY in .env):
  1. NASA FIRMS VIIRS archive (VIIRS_SNPP_SP): one request per day for a
     single bounding box covering all the states - ~450 requests for three
     Jan-May seasons, cached per season in data/raw/historical_fires_india_<year>.csv.
  2. Meteostat station weather per state (no key needed), cached in
     data/raw/historical_weather_<state>.csv.
  3. Per state: the same row builder as Karnataka (build_dataset_for_region)
     on that state's live 0.5 deg grid -> data/processed/india_states_training_data.csv
  4. --train: the same v2 feature pipeline as train_real.py (fire history +
     leave-one-season-out climatology), ONE pooled XGBoost across all states,
     temporal holdout (train 2023-24, test 2025), metrics reported per state.
       -> models/xgboost_india.json, models/zone_climatology_india.csv,
          data/processed/india_model_by_state.csv

Every step is cached, so an interrupted run resumes where it stopped.

    python scripts/build_india_regions_dataset.py            # fetch + build
    python scripts/build_india_regions_dataset.py --train    # fetch + build + train
    python scripts/build_india_regions_dataset.py --train --skip-fetch
"""
import argparse
import json
import logging
import re
import sys
from datetime import date
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config.config import API, DATA_PROCESSED_DIR, DATA_RAW_DIR, MODELS_DIR, REGION
from src.regions import REGION_PRESETS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("india_builder")

YEARS = [2023, 2024, 2025]
SEASON = ((1, 1), (5, 31))          # Jan-May, same window as the Karnataka data
DATASET_PATH = DATA_PROCESSED_DIR / "india_states_training_data.csv"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def india_states():
    return {name: r for name, r in REGION_PRESETS.items() if r.name != REGION.name}


def union_bbox(regions) -> dict:
    return {
        "min_lat": min(r.min_lat for r in regions), "max_lat": max(r.max_lat for r in regions),
        "min_lon": min(r.min_lon for r in regions), "max_lon": max(r.max_lon for r in regions),
    }


# ── 1. fires ────────────────────────────────────────────────────────────────

def fetch_fires(states: dict, years) -> pd.DataFrame:
    from src.data_ingestion.historical_firms import HistoricalFIRMSClient
    frames = []
    box = union_bbox(states.values())
    client = None
    for year in years:
        path = DATA_RAW_DIR / f"historical_fires_india_{year}.csv"
        if path.exists():
            logger.info("Fires %d: cached (%s)", year, path.name)
            frames.append(pd.read_csv(path))
            continue
        if not API.firms_map_key:
            sys.exit("FIRMS_MAP_KEY is not set in .env - needed to download fire history.")
        client = client or HistoricalFIRMSClient(API.firms_map_key)
        (m0, d0), (m1, d1) = SEASON
        logger.info("Fires %d: downloading India box %s", year, box)
        df = client.fetch_range(box["min_lat"], box["min_lon"], box["max_lat"], box["max_lon"],
                                date(year, m0, d0), date(year, m1, d1), pause_seconds=1.0)
        if df.empty:
            logger.warning("Fires %d: no detections returned (archive may lag for recent years)", year)
            continue
        df["fire_season_year"] = year
        df.to_csv(path, index=False)
        frames.append(df)
    if not frames:
        sys.exit("No fire history available.")
    fires = pd.concat(frames, ignore_index=True)
    fires["acq_date"] = pd.to_datetime(fires["acq_date"]).dt.date
    return fires


# ── 2. weather ──────────────────────────────────────────────────────────────

def fetch_weather(name: str, region, years) -> pd.DataFrame:
    path = DATA_RAW_DIR / f"historical_weather_{slug(name)}.csv"
    cached = pd.read_csv(path) if path.exists() else pd.DataFrame()
    have = set(pd.to_datetime(cached["date"], format="mixed").dt.year) if len(cached) else set()
    missing = [y for y in years if y not in have]
    if not missing:
        logger.info("Weather %s: cached (%s)", name, path.name)
        df = cached
    else:
        from src.data_ingestion.historical_weather import fetch_region_historical_weather
        bounds = {"min_lat": region.min_lat, "max_lat": region.max_lat,
                  "min_lon": region.min_lon, "max_lon": region.max_lon}
        frames = [cached] if len(cached) else []
        (m0, d0), (m1, d1) = SEASON
        for year in missing:
            logger.info("Weather %s %d: downloading", name, year)
            w = fetch_region_historical_weather(bounds, date(year, m0, d0), date(year, m1, d1))
            if not w.empty:
                frames.append(w)
        if not frames:
            logger.warning("Weather %s: nothing retrieved - state skipped", name)
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(path, index=False)
    df["date"] = pd.to_datetime(df["date"], format="mixed").dt.date
    # Only the requested seasons: weather for a season with no fire archive
    # loaded would otherwise be labelled "no fire" everywhere.
    return df[pd.to_datetime(df["date"]).dt.year.isin(list(years))].reset_index(drop=True)


# ── 3. build ────────────────────────────────────────────────────────────────

def build(states: dict, years, skip_fetch: bool) -> pd.DataFrame:
    if skip_fetch and DATASET_PATH.exists():
        logger.info("Using existing %s", DATASET_PATH.name)
        return pd.read_csv(DATASET_PATH)

    from src.ml_models.build_real_dataset import build_dataset_for_region
    from src.ml_models.fire_history import StaticSourceMask
    fires = fetch_fires(states, years)
    StaticSourceMask.from_archive(fires).save(MODELS_DIR, "static_heat_sources_india.csv")
    parts = []
    for name, region in states.items():
        weather = fetch_weather(name, region, years)
        if weather.empty:
            continue
        margin = region.grid_resolution_deg
        f = fires[fires["latitude"].between(region.min_lat - margin, region.max_lat + margin)
                  & fires["longitude"].between(region.min_lon - margin, region.max_lon + margin)]
        df = build_dataset_for_region(region, f, weather)
        df.insert(0, "region", name)
        parts.append(df)
    if not parts:
        sys.exit("No state could be built.")
    data = pd.concat(parts, ignore_index=True)
    data.to_csv(DATASET_PATH, index=False)
    logger.info("Saved %d rows for %d states to %s", len(data), data["region"].nunique(), DATASET_PATH)
    return data


# ── 4. train ────────────────────────────────────────────────────────────────

def train(data: pd.DataFrame, states: dict):
    from src.ml_models.fire_history import FEATURE_COLUMNS_V2, FEATURE_COLUMNS_V3, ZoneClimatology
    from src.ml_models.model_registry import (
        INDIA_CLIMATOLOGY, INDIA_METRICS, INDIA_MODEL, INDIA_RISK_INDEX,
    )
    from src.ml_models.model_trainer import XGBoostModel, evaluate_predictions
    from src.ml_models.evaluation_extras import lift_table, persistence_table, summary
    from src.ml_models.train_real import (
        SECOND_TEST_START, TEST_SEASON_START, add_model_features, add_terrain_training,
    )

    featured, clim_tables = [], []
    for name, part in data.groupby("region", sort=False):
        df, clim = add_model_features(part.drop(columns=["region"]), states[name])
        df = add_terrain_training(df, states[name])     # elev_m / slope_pct when built for this state
        df.insert(0, "region", name)
        featured.append(df)
        clim_tables.append(clim.assign(region=name))
    # Terrain (v3 features) only if EVERY state has a terrain table
    # (python scripts/build_terrain.py --all); otherwise the v2 feature set.
    terrain_on = all("elev_m" in f.columns for f in featured)
    feature_cols = list(FEATURE_COLUMNS_V3 if terrain_on else FEATURE_COLUMNS_V2)
    logger.info("Pan-India feature set: %s (%d features)", "v3 with terrain" if terrain_on else "v2", len(feature_cols))
    if not terrain_on:
        featured = [f.drop(columns=["elev_m", "slope_pct"], errors="ignore") for f in featured]
    df = pd.concat(featured, ignore_index=True)

    train_df = df[df["date"] < TEST_SEASON_START]
    test_df = df[(df["date"] >= TEST_SEASON_START) & (df["date"] < SECOND_TEST_START)]
    test26_df = df[df["date"] >= SECOND_TEST_START]
    if test_df.empty or train_df["fire_risk_label"].nunique() < 2:
        sys.exit("Need both training seasons and the 2025 test season to train/evaluate.")
    logger.info("Pooled India model: train=%d rows, test=%d rows, %d states",
                len(train_df), len(test_df), df["region"].nunique())

    model = XGBoostModel().fit(train_df[feature_cols], train_df["fire_risk_label"], tune=True)
    model.model.save_model(str(MODELS_DIR / INDIA_MODEL))
    from src.ml_models.risk_index import RiskIndex
    RiskIndex.fit(train_df["fire_risk_label"], model.model.predict_proba(train_df[feature_cols])[:, 1],
                  model.threshold_, high_threshold=model.alert_threshold_).save(MODELS_DIR, INDIA_RISK_INDEX)
    ZoneClimatology(pd.concat(clim_tables, ignore_index=True)).save(MODELS_DIR, INDIA_CLIMATOLOGY)

    if terrain_on:
        from sklearn.metrics import roc_auc_score
        ref = XGBoostModel().fit(train_df[FEATURE_COLUMNS_V2], train_df["fire_risk_label"], tune=True)
        ref_auc = roc_auc_score(test_df["fire_risk_label"], ref.predict(test_df[FEATURE_COLUMNS_V2])[1])
        logger.info("Reference pooled AUC without terrain (v2): %.4f", ref_auc)

    def score(season_df, season):
        """Per-state (and pooled) metrics for one held-out season."""
        rows, lifts, persist = [], [], []
        groups = [("All states (pooled)", season_df)] + list(season_df.groupby("region", sort=False))
        for name, g in groups:
            y = g["fire_risk_label"].values
            if len(np.unique(y)) < 2:
                continue
            _, prob = model.predict(g[feature_cols])
            pred = (prob >= model.threshold_).astype(int)
            m = evaluate_predictions(y, pred, prob)
            a = evaluate_predictions(y, (prob >= model.alert_threshold_).astype(int), prob)
            rows.append({"region": name, "test_rows": len(g), "test_fire_days": int(y.sum()),
                         "positive_rate": float(y.mean()),
                         **{k: float(v) for k, v in m.items()},
                         "alert_accuracy": float(a["accuracy"]), "alert_recall": float(a["recall"]),
                         "alert_precision": float(a["precision"]),
                         # base-rate-normalised, comparable across states with different fire frequency
                         **{k: v for k, v in summary(y, prob, model.alert_threshold_).items()
                            if k in ("ap_over_base_rate", "recall_top10pct", "lift_top10pct",
                                     "alert_precision_lift")}})
            lifts.append(lift_table(y, prob).assign(region=name, season=season))
            persist.append(persistence_table(g, prob).assign(region=name, season=season))
        return pd.DataFrame(rows).round(4), pd.concat(lifts), pd.concat(persist)

    metrics, lift_df, persist_df = score(test_df, "2025")
    metrics.to_csv(DATA_PROCESSED_DIR / INDIA_METRICS, index=False)
    lift_df.to_csv(DATA_PROCESSED_DIR / "eval_india_lift.csv", index=False)
    persist_df.to_csv(DATA_PROCESSED_DIR / "eval_india_persistence.csv", index=False)

    if len(test26_df):
        m26, l26, p26 = score(test26_df, "2026")
        m26.to_csv(DATA_PROCESSED_DIR / "india_model_by_state_2026.csv", index=False)
        pd.concat([lift_df, l26]).to_csv(DATA_PROCESSED_DIR / "eval_india_lift.csv", index=False)
        pd.concat([persist_df, p26]).to_csv(DATA_PROCESSED_DIR / "eval_india_persistence.csv", index=False)
        print("\n=== Pan-India model, SECOND held-out season: 2026 ===")
        print(m26[["region", "test_fire_days", "auc_roc", "avg_precision", "recall",
                   "ap_over_base_rate", "lift_top10pct"]].to_string(index=False))

    with open(MODELS_DIR / "training_metadata_india.json", "w") as f:
        json.dump({
            "states": sorted(df["region"].unique().tolist()),
            "train_rows": int(len(train_df)), "test_rows": int(len(test_df)),
            "feature_columns": feature_cols, "terrain": terrain_on, "decision_threshold": float(model.threshold_),
            "alert_threshold": float(model.alert_threshold_),
            "split_method": "temporal - train 2023-2024, test 2025; one pooled model, metrics per state",
        }, f, indent=2)

    print("\n=== Pan-India model, held-out 2025 season ===")
    print(metrics[["region", "test_fire_days", "auc_roc", "avg_precision", "recall",
                   "ap_over_base_rate", "lift_top10pct", "alert_recall"]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="store_true", help="also train + evaluate the pan-India model")
    ap.add_argument("--skip-fetch", action="store_true", help="reuse the saved dataset if present")
    ap.add_argument("--states", nargs="*", help="subset of state names (default: all curated states)")
    ap.add_argument("--years", nargs="*", type=int, default=YEARS)
    args = ap.parse_args()

    states = india_states()
    if args.states:
        unknown = set(args.states) - set(states)
        if unknown:
            sys.exit(f"Unknown state(s): {sorted(unknown)}. Choose from: {sorted(states)}")
        states = {k: states[k] for k in args.states}

    data = build(states, args.years, args.skip_fetch)
    if args.train:
        train(data, states)


if __name__ == "__main__":
    main()
