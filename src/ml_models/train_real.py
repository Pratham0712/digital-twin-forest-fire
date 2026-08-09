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

from config.config import MODELS_DIR, DATA_PROCESSED_DIR
from src.ml_models.model_trainer import (
    RandomForestModel, XGBoostModel, CNNLSTMModel, evaluate_predictions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "wx_temperature_c", "wx_humidity_pct", "wx_wind_speed_ms", "wx_precipitation_mm",
    "ffmc", "dmc", "dc", "bui", "fwi", "ndvi",
]
TEST_SEASON_START = pd.Timestamp("2025-01-01")


def load_real_dataset() -> pd.DataFrame:
    path = DATA_PROCESSED_DIR / "real_training_data.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run src/ml_models/build_real_dataset.py first."
        )
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def temporal_split(df: pd.DataFrame):
    train = df[df["date"] < TEST_SEASON_START]
    test = df[df["date"] >= TEST_SEASON_START]
    logger.info(
        "Temporal split: train=%d rows (%s to %s), test=%d rows (%s to %s)",
        len(train), train["date"].min().date(), train["date"].max().date(),
        len(test), test["date"].min().date(), test["date"].max().date(),
    )
    logger.info("Train positive rate: %.2f%% | Test positive rate: %.2f%%",
                100 * train["fire_risk_label"].mean(), 100 * test["fire_risk_label"].mean())
    return train, test


def build_real_sequences(df: pd.DataFrame, n_days: int = 7):
    """
    Builds genuine trailing n-day sequences per zone from REAL consecutive
    daily rows (not the synthetic AR(1) walk used for offline/demo mode).
    Zones with gaps or fewer than n_days of history are dropped - real
    station data isn't perfectly gap-free, so this naturally excludes
    incomplete windows rather than fabricating them.
    """
    feature_cols = FEATURE_COLUMNS
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


def main():
    logger.info("=== Loading real dataset ===")
    df = load_real_dataset()
    train_df, test_df = temporal_split(df)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["fire_risk_label"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["fire_risk_label"]

    results = {}

    logger.info("=== Random Forest (real data) ===")
    rf = RandomForestModel().fit(X_train, y_train)
    results["Random Forest"] = rf.evaluate(X_test, y_test)
    logger.info("RF: %s", results["Random Forest"])
    with open(MODELS_DIR / "random_forest_real.pkl", "wb") as f:
        pickle.dump(rf.model, f)

    logger.info("=== XGBoost (real data) ===")
    xgb_model = XGBoostModel().fit(X_train, y_train)
    results["XGBoost"] = xgb_model.evaluate(X_test, y_test)
    logger.info("XGBoost: %s", results["XGBoost"])
    xgb_model.model.save_model(str(MODELS_DIR / "xgboost_real.json"))

    logger.info("=== CNN+LSTM (real data, real 7-day sequences) ===")
    X_seq_train, y_seq_train, feature_cols = build_real_sequences(train_df)
    X_seq_test, y_seq_test, _ = build_real_sequences(test_df)

    if len(X_seq_train) < 50 or len(X_seq_test) < 20:
        logger.warning(
            "Too few real sequences to train CNN+LSTM reliably (train=%d, test=%d) - "
            "skipping. This can happen if the real weather data has date gaps.",
            len(X_seq_train), len(X_seq_test),
        )
    else:
        cnn_lstm = CNNLSTMModel(n_days=X_seq_train.shape[1], n_features=X_seq_train.shape[2])
        cnn_lstm.fit(X_seq_train, y_seq_train, epochs=25, verbose=0)
        results["CNN+LSTM"] = cnn_lstm.evaluate(X_seq_test, y_seq_test)
        logger.info("CNN+LSTM: %s", results["CNN+LSTM"])
        cnn_lstm.model.save(str(MODELS_DIR / "cnn_lstm_real.keras"))
        with open(MODELS_DIR / "cnn_lstm_real_scaler.pkl", "wb") as f:
            pickle.dump(cnn_lstm.scaler, f)

    comparison = pd.DataFrame(results).T.round(4)
    comparison.index.name = "Model"
    comparison.to_csv(DATA_PROCESSED_DIR / "model_comparison_real.csv")

    logger.info("\n=== REAL DATA MODEL COMPARISON (temporal holdout: test = 2025 season) ===\n%s",
                comparison.to_string())

    with open(MODELS_DIR / "training_metadata_real.json", "w") as f:
        json.dump({
            "train_rows": int(len(train_df)), "test_rows": int(len(test_df)),
            "train_date_range": [str(train_df["date"].min().date()), str(train_df["date"].max().date())],
            "test_date_range": [str(test_df["date"].min().date()), str(test_df["date"].max().date())],
            "train_positive_rate": float(train_df["fire_risk_label"].mean()),
            "test_positive_rate": float(test_df["fire_risk_label"].mean()),
            "feature_columns": FEATURE_COLUMNS,
            "split_method": "temporal - train 2023-2024, test 2025 (held-out year)",
        }, f, indent=2)

    print(f"\nModels saved to: {MODELS_DIR} (suffixed _real)")
    print(f"Comparison table: {DATA_PROCESSED_DIR / 'model_comparison_real.csv'}")
    return comparison


if __name__ == "__main__":
    main()
