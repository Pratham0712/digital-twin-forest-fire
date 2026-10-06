"""
train_real.py - trains RF/XGBoost/CNN-LSTM on the REAL, leakage-free dataset
built by build_real_dataset.py (real FIRMS fire occurrence = label, real
Meteostat weather = features). This replaces train.py's synthetic-data
training for any result that goes in the report or paper.

Uses a TEMPORAL split, not a random split: train on 2023+2024 fire seasons,
test on the fully held-out 2025 season. This is a stricter, more honest
evaluation than a random split - it tests whether the model generalizes to
an entirely unseen year, not just unseen rows shuffled from the same period
(random splits can leak information via nearby dates/zones ending up on both
sides of a random split, which is a subtler form of leakage than the FWI
issue but still inflates scores).

Feature set v2 (current): same-day weather/FWI PLUS satellite fire history
(previous 10 days of FIRMS detections in and around each zone) and per-zone
fire climatology from the training seasons. Built by the shared
src/ml_models/fire_history.py module that live inference also uses, so the
deployed model sees exactly the features it was trained on. The v1
weather-only XGBoost is retrained alongside as a baseline row in the
comparison table, so the improvement is measured, not asserted.

Run:
    python src/ml_models/train_real.py
"""
import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from config.config import MODELS_DIR, DATA_PROCESSED_DIR, REGION
from src.data_ingestion.ingestion_module import build_region_grid
from src.data_ingestion.terrain import TERRAIN_COLUMNS, TerrainTable
from src.ml_models.fire_history import (
    BASE_FEATURE_COLUMNS, FEATURE_COLUMNS_V2, FEATURE_COLUMNS_V3, ZoneClimatology,
    add_climatology_training, add_fire_history_training,
)
from src.ml_models.risk_index import RiskIndex, tier_performance
from src.ml_models.model_trainer import (
    RandomForestModel, XGBoostModel, CNNLSTMModel, evaluate_predictions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = FEATURE_COLUMNS_V2          # v2: default; main() switches to v3 when real terrain exists
V1_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS     # weather-only baseline, for comparison
TEST_SEASON_START = pd.Timestamp("2025-01-01")
# Second, untouched hold-out season. Never used for training, tuning or
# threshold selection: the 2025 season is the headline test, 2026 (when added
# with scripts/add_season.py) is an independent confirmation.
SECOND_TEST_START = pd.Timestamp("2026-01-01")


def add_model_features(df: pd.DataFrame, region=REGION):
    """All engineered features for one region's real dataset. Shared by this
    script (Karnataka) and scripts/build_india_regions_dataset.py (other
    states), so every model is trained on identically-built features.
    Returns (df, climatology_table)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    # Temporal features: fire risk is strongly seasonal
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df = df.sort_values(["zone_id", "date"])
    # fwi_lag1: yesterday's FWI per zone - captures "FWI rising" signal without
    # the train/inference mismatch of fwi_roll7 (which collapses to fwi at inference
    # time since there's only one snapshot). lag1 is honest: at inference we use
    # the previous snapshot's FWI stored in twin state.
    df["fwi_lag1"] = (
        df.groupby("zone_id")["fwi"]
        .transform(lambda s: s.shift(1).fillna(s))
    )

    # v2 features (see fire_history.py for the leakage rules each one obeys)
    grid = build_region_grid(region)
    df = add_fire_history_training(df, grid)
    df, clim_table = add_climatology_training(df, grid, train_mask=df["date"] < TEST_SEASON_START)
    df = df.sort_values(["zone_id", "date"]).reset_index(drop=True)
    return df, clim_table


def add_terrain_training(df: pd.DataFrame, region=REGION, table: TerrainTable = None) -> pd.DataFrame:
    """Adds elev_m / slope_pct from the cached terrain table (static per zone,
    so there is nothing time-dependent to leak). Returns df unchanged - without
    the columns - when no terrain table covers the whole grid."""
    table = table or TerrainTable.load(MODELS_DIR, region.name)
    grid = build_region_grid(region)
    if table.coverage(grid) < 1.0:
        return df
    return df.drop(columns=TERRAIN_COLUMNS, errors="ignore").merge(table.transform(grid), on="zone_id", how="left")


def load_real_dataset(return_climatology: bool = False):
    path = DATA_PROCESSED_DIR / "real_training_data.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run src/ml_models/build_real_dataset.py first."
        )
    df, clim_table = add_model_features(pd.read_csv(path), REGION)
    return (df, clim_table) if return_climatology else df


def save_risk_index(xgb_model, X_train, y_train, X_test, y_test,
                    filename: str = "risk_index.json", tier_csv: str = "alert_tier_performance.csv"):
    """Fits the dashboard's risk index on TRAINING predictions only, then
    reports how each alert tier performs on the held-out test season."""
    ri = RiskIndex.fit(y_train, xgb_model.model.predict_proba(X_train)[:, 1], xgb_model.threshold_,
                       high_threshold=getattr(xgb_model, "alert_threshold_", None))
    ri.save(MODELS_DIR, filename)
    tiers = pd.DataFrame(tier_performance(y_test, xgb_model.model.predict_proba(X_test)[:, 1], ri)).round(4)
    tiers.to_csv(DATA_PROCESSED_DIR / tier_csv, index=False)
    logger.info("Risk index anchors: %s\nAlert tiers on held-out season:\n%s", ri, tiers.to_string(index=False))
    return ri


def second_holdout(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of the second hold-out season (2026 onwards); empty if not added yet."""
    return df[df["date"] >= SECOND_TEST_START]


def temporal_split(df: pd.DataFrame):
    """Train = before 2025; test = the 2025 season only. Later seasons are kept
    out of both so adding 2026 data never changes the headline numbers."""
    train = df[df["date"] < TEST_SEASON_START]
    test = df[(df["date"] >= TEST_SEASON_START) & (df["date"] < SECOND_TEST_START)]
    logger.info(
        "Temporal split: train=%d rows (%s to %s), test=%d rows (%s to %s)",
        len(train), train["date"].min().date(), train["date"].max().date(),
        len(test), test["date"].min().date(), test["date"].max().date(),
    )
    logger.info("Train positive rate: %.2f%% | Test positive rate: %.2f%%",
                100 * train["fire_risk_label"].mean(), 100 * test["fire_risk_label"].mean())
    return train, test


def build_real_sequences(df: pd.DataFrame, n_days: int = 7, feature_cols=None):
    """
    Builds genuine trailing n-day sequences per zone from REAL consecutive
    daily rows (not the synthetic AR(1) walk used for offline/demo mode).
    Zones with gaps or fewer than n_days of history are dropped - real
    station data isn't perfectly gap-free, so this naturally excludes
    incomplete windows rather than fabricating them.
    """
    feature_cols = list(feature_cols or FEATURE_COLUMNS)
    df = df.sort_values(["zone_id", "date"])
    sequences, labels = [], []

    for zone_id, group in df.groupby("zone_id"):
        group = group.reset_index(drop=True)
        values = group[feature_cols].values
        y = group["fire_risk_label"].values
        for i in range(n_days - 1, len(group)):
            window = values[i - n_days + 1: i + 1]
            if len(window) == n_days:
                sequences.append(window)
                labels.append(y[i])

    X_seq = np.array(sequences)
    y_seq = np.array(labels)
    logger.info("Built %d real sequences of length %d from %d zones", len(X_seq), n_days, df["zone_id"].nunique())
    return X_seq, y_seq, feature_cols


def main(use_terrain: bool = True):
    logger.info("=== Loading real dataset ===")
    df, clim_table = load_real_dataset(return_climatology=True)
    feature_cols = list(FEATURE_COLUMNS_V2)
    if use_terrain:
        with_terrain = add_terrain_training(df)
        if "elev_m" in with_terrain.columns:
            df, feature_cols = with_terrain, list(FEATURE_COLUMNS_V3)
            logger.info("Real terrain found - training with v3 features (%d): + %s",
                        len(feature_cols), TERRAIN_COLUMNS)
        else:
            logger.warning("No terrain table covering the grid (run scripts/build_terrain.py) - "
                           "training with v2 features.")
    terrain_on = "elev_m" in feature_cols
    train_df, test_df = temporal_split(df)

    X_train, y_train = train_df[feature_cols], train_df["fire_risk_label"]
    X_test, y_test = test_df[feature_cols], test_df["fire_risk_label"]

    results = {}         # early-warning operating point (~80% recall target)
    alert_results = {}   # alert operating point (>= 92% validation accuracy)

    logger.info("=== Random Forest (real data, tuned) ===")
    rf = RandomForestModel().fit(X_train, y_train, tune=True)
    results["Random Forest"] = rf.evaluate(X_test, y_test)
    alert_results["Random Forest"] = rf.evaluate(X_test, y_test, threshold=rf.alert_threshold_)
    logger.info("RF: %s", results["Random Forest"])
    with open(MODELS_DIR / "random_forest_real.pkl", "wb") as f:
        pickle.dump(rf.model, f)

    logger.info("=== XGBoost (real data, tuned) ===")
    xgb_model = XGBoostModel().fit(X_train, y_train, tune=True)
    results["XGBoost"] = xgb_model.evaluate(X_test, y_test)
    alert_results["XGBoost"] = xgb_model.evaluate(X_test, y_test, threshold=xgb_model.alert_threshold_)
    logger.info("XGBoost: %s", results["XGBoost"])
    xgb_model.model.save_model(str(MODELS_DIR / "xgboost_real.json"))
    save_risk_index(xgb_model, X_train, y_train, X_test, y_test)

    logger.info("=== CNN+LSTM (real data, real 7-day sequences) ===")
    X_seq_train, y_seq_train, _ = build_real_sequences(train_df, feature_cols=feature_cols)
    X_seq_test, y_seq_test, _ = build_real_sequences(test_df, feature_cols=feature_cols)

    if len(X_seq_train) < 50 or len(X_seq_test) < 20:
        logger.warning(
            "Too few real sequences to train CNN+LSTM reliably (train=%d, test=%d) - "
            "skipping. This can happen if the real weather data has date gaps.",
            len(X_seq_train), len(X_seq_test),
        )
    else:
        cnn_lstm = CNNLSTMModel(n_days=X_seq_train.shape[1], n_features=X_seq_train.shape[2])
        cnn_lstm.fit(X_seq_train, y_seq_train, epochs=50, verbose=1)
        results["CNN+LSTM"] = cnn_lstm.evaluate(X_seq_test, y_seq_test)
        alert_results["CNN+LSTM"] = cnn_lstm.evaluate(X_seq_test, y_seq_test,
                                                      threshold=cnn_lstm.alert_threshold_)
        logger.info("CNN+LSTM: %s", results["CNN+LSTM"])
        cnn_lstm.model.save(str(MODELS_DIR / "cnn_lstm_real.keras"))
        with open(MODELS_DIR / "cnn_lstm_real_scaler.pkl", "wb") as f:
            pickle.dump(cnn_lstm.scaler, f)

    logger.info("=== Baseline: XGBoost on v1 weather-only features ===")
    v1 = XGBoostModel().fit(train_df[V1_FEATURE_COLUMNS], y_train, tune=True)
    results["XGBoost (v1, weather only)"] = v1.evaluate(test_df[V1_FEATURE_COLUMNS], y_test)
    alert_results["XGBoost (v1, weather only)"] = v1.evaluate(
        test_df[V1_FEATURE_COLUMNS], y_test, threshold=v1.alert_threshold_)
    logger.info("XGBoost v1 baseline: %s", results["XGBoost (v1, weather only)"])

    if terrain_on:
        logger.info("=== Reference: XGBoost on v2 features (no terrain) ===")
        v2 = XGBoostModel().fit(train_df[FEATURE_COLUMNS_V2], y_train, tune=True)
        results["XGBoost (v2, no terrain)"] = v2.evaluate(test_df[FEATURE_COLUMNS_V2], y_test)
        alert_results["XGBoost (v2, no terrain)"] = v2.evaluate(
            test_df[FEATURE_COLUMNS_V2], y_test, threshold=v2.alert_threshold_)
        logger.info("XGBoost v2 reference: %s", results["XGBoost (v2, no terrain)"])

    ZoneClimatology(clim_table).save(MODELS_DIR)

    # Second hold-out season (2026), if it has been added: same trained models,
    # same thresholds, never seen in training/tuning.
    test26 = second_holdout(df)
    if len(test26) and test26["fire_risk_label"].nunique() == 2:
        y26 = test26["fire_risk_label"]
        rows26, alert26 = {}, {}
        for name, m, cols in (("Random Forest", rf, feature_cols), ("XGBoost", xgb_model, feature_cols),
                              ("XGBoost (v1, weather only)", v1, V1_FEATURE_COLUMNS)):
            rows26[name] = m.evaluate(test26[cols], y26)
            alert26[name] = m.evaluate(test26[cols], y26, threshold=m.alert_threshold_)
        c26 = pd.DataFrame(rows26).T.round(4)
        c26.index.name = "Model"
        c26.to_csv(DATA_PROCESSED_DIR / "model_comparison_2026.csv")
        a26 = pd.DataFrame(alert26).T.round(4)
        a26.index.name = "Model"
        a26.to_csv(DATA_PROCESSED_DIR / "model_comparison_alert_2026.csv")
        logger.info("\n=== SECOND HOLD-OUT SEASON: 2026 (%d rows, %.2f%% fire days) ===\n%s",
                    len(test26), 100 * y26.mean(), c26.to_string())
    else:
        logger.info("No 2026 season in the dataset - run scripts/add_season.py --year 2026 to add a second hold-out.")

    comparison = pd.DataFrame(results).T.round(4)
    comparison.index.name = "Model"
    comparison.to_csv(DATA_PROCESSED_DIR / "model_comparison_real.csv")
    alert_cmp = pd.DataFrame(alert_results).T.round(4)
    alert_cmp.index.name = "Model"
    alert_cmp.to_csv(DATA_PROCESSED_DIR / "model_comparison_alert.csv")
    logger.info("\n=== ALERT OPERATING POINT (threshold for >=92%% validation accuracy) ===\n%s",
                alert_cmp.to_string())

    logger.info("\n=== REAL DATA MODEL COMPARISON (temporal holdout: test = 2025 season) ===\n%s",
                comparison.to_string())

    with open(MODELS_DIR / "training_metadata_real.json", "w") as f:
        json.dump({
            "train_rows": int(len(train_df)), "test_rows": int(len(test_df)),
            "train_date_range": [str(train_df["date"].min().date()), str(train_df["date"].max().date())],
            "test_date_range": [str(test_df["date"].min().date()), str(test_df["date"].max().date())],
            "train_positive_rate": float(train_df["fire_risk_label"].mean()),
            "test_positive_rate": float(test_df["fire_risk_label"].mean()),
            "feature_columns": feature_cols,
            "feature_set_version": 3 if terrain_on else 2,
            "xgboost_decision_threshold": float(xgb_model.threshold_),
            "xgboost_alert_threshold": float(xgb_model.alert_threshold_),
            "split_method": "temporal - train 2023-2024, test 2025 (held-out year); fwi_lag1 replaces fwi_roll7 to avoid train/inference mismatch; fire-history features use only days t-10..t-1; zone climatology is leave-one-season-out on train rows",
        }, f, indent=2)

    print(f"\nModels saved to: {MODELS_DIR} (suffixed _real)")
    print(f"Comparison table: {DATA_PROCESSED_DIR / 'model_comparison_real.csv'}")
    return comparison


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-terrain", action="store_true",
                    help="train on the v2 features even if a terrain table exists (reproduces the pre-terrain model)")
    main(use_terrain=not ap.parse_args().no_terrain)
