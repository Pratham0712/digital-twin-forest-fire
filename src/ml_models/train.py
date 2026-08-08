"""
train.py - end-to-end training pipeline: ingestion -> feature engineering ->
train RF/XGBoost/CNN-LSTM -> evaluate -> save models + comparison table.

Run:
    python src/ml_models/train.py            # offline/synthetic data
    python src/ml_models/train.py --live      # real FIRMS/OWM data (needs .env keys)

Output:
    models/random_forest.pkl
    models/xgboost.json
    models/cnn_lstm.keras
    data/processed/model_comparison.csv   <- use this table directly in the paper
"""
import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd

from config.config import MODELS_DIR, DATA_PROCESSED_DIR
from src.data_ingestion.ingestion_module import DataIngestionModule
from src.data_processing.feature_engineering import DataProcessor
from src.ml_models.timeseries_builder import build_history
from src.ml_models.model_trainer import (
    RandomForestModel, XGBoostModel, CNNLSTMModel,
    split_tabular, split_sequence,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(offline: bool = True):
    logger.info("=== STEP 1: Data ingestion ===")
    ingestion = DataIngestionModule(offline=offline)
    unified = ingestion.build_unified_frame()

    logger.info("=== STEP 2: Feature engineering ===")
    processor = DataProcessor()
    processed = processor.transform(unified)
    X, y = processor.get_feature_matrix(processed)
    logger.info("Feature matrix: %s, positive rate: %.2f%%", X.shape, 100 * y.mean())

    results = {}

    # ---- Random Forest ----
    logger.info("=== STEP 3a: Random Forest ===")
    X_train, X_test, y_train, y_test = split_tabular(X, y)
    rf = RandomForestModel().fit(X_train, y_train)
    results["Random Forest"] = rf.evaluate(X_test, y_test)
    logger.info("RF metrics: %s", results["Random Forest"])
    with open(MODELS_DIR / "random_forest.pkl", "wb") as f:
        pickle.dump(rf.model, f)

    # ---- XGBoost ----
    logger.info("=== STEP 3b: XGBoost ===")
    xgb_model = XGBoostModel().fit(X_train, y_train)
    results["XGBoost"] = xgb_model.evaluate(X_test, y_test)
    logger.info("XGBoost metrics: %s", results["XGBoost"])
    xgb_model.model.save_model(str(MODELS_DIR / "xgboost.json"))

    # ---- CNN + LSTM ----
    logger.info("=== STEP 3c: CNN+LSTM (sequence model) ===")
    X_seq, feature_cols = build_history(processed, n_days=7)
    y_seq = y.values
    X_seq_train, X_seq_test, y_seq_train, y_seq_test = split_sequence(X_seq, y_seq)

    cnn_lstm = CNNLSTMModel(n_days=X_seq.shape[1], n_features=X_seq.shape[2])
    cnn_lstm.fit(X_seq_train, y_seq_train, epochs=25, verbose=0)
    results["CNN+LSTM"] = cnn_lstm.evaluate(X_seq_test, y_seq_test)
    logger.info("CNN+LSTM metrics: %s", results["CNN+LSTM"])
    cnn_lstm.model.save(str(MODELS_DIR / "cnn_lstm.keras"))
    with open(MODELS_DIR / "cnn_lstm_scaler.pkl", "wb") as f:
        pickle.dump(cnn_lstm.scaler, f)

    # ---- Comparison table (for report + paper) ----
    comparison = pd.DataFrame(results).T
    comparison.index.name = "Model"
    comparison = comparison.round(4)
    comparison.to_csv(DATA_PROCESSED_DIR / "model_comparison.csv")

    logger.info("\n=== MODEL COMPARISON TABLE ===\n%s", comparison.to_string())

    with open(MODELS_DIR / "training_metadata.json", "w") as f:
        json.dump({
            "n_samples": int(len(X)),
            "n_features": int(X.shape[1]),
            "positive_rate": float(y.mean()),
            "feature_columns": list(X.columns),
            "sequence_features": feature_cols,
            "offline_mode": offline,
        }, f, indent=2)

    print(f"\nModels saved to: {MODELS_DIR}")
    print(f"Comparison table saved to: {DATA_PROCESSED_DIR / 'model_comparison.csv'}")
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                         help="Use real FIRMS/OWM APIs instead of synthetic data (needs .env keys)")
    args = parser.parse_args()
    main(offline=not args.live)
