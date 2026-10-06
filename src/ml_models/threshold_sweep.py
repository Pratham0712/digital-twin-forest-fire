"""
threshold_sweep.py - finds the lowest decision threshold that keeps the
false-negative rate under config.SYSTEM.max_false_negative_rate (default 5%),
then prints a precision/recall/F1 table across thresholds 0.05-0.95.

Run:
    python src/ml_models/threshold_sweep.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from config.config import MODELS_DIR, DATA_PROCESSED_DIR, SYSTEM
from src.ml_models.model_trainer import XGBoostModel, _false_negative_rate
from src.ml_models.train_real import load_real_dataset, temporal_split, FEATURE_COLUMNS
from sklearn.metrics import precision_score, recall_score, f1_score


def sweep(y_true: np.ndarray, y_prob: np.ndarray, max_fnr: float = SYSTEM.max_false_negative_rate):
    thresholds = np.arange(0.05, 0.96, 0.05)
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        rows.append({
            "threshold": round(float(t), 2),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall":    recall_score(y_true, y_pred, zero_division=0),
            "f1":        f1_score(y_true, y_pred, zero_division=0),
            "fnr":       _false_negative_rate(y_true, y_pred),
        })
    df = pd.DataFrame(rows).round(4)

    best_f1_thresh = df.loc[df["f1"].idxmax(), "threshold"]
    satisfying = df[df["fnr"] <= max_fnr]
    if satisfying.empty:
        print(f"\nNo threshold achieves FNR <= {max_fnr:.0%} (min achieved: {df['fnr'].min():.4f}).")
        recommended = best_f1_thresh
    else:
        recommended = satisfying.loc[satisfying["threshold"].idxmin(), "threshold"]
        print(f"\nLowest threshold satisfying FNR <= {max_fnr:.0%}: {recommended}")

    print(f"Best F1 threshold: {best_f1_thresh}  (F1={df.loc[df['threshold']==best_f1_thresh, 'f1'].values[0]:.4f})")
    print(f"Current config threshold: {SYSTEM.alert_threshold_pct / 100:.2f}")
    print(f"Recommended threshold:    {recommended:.2f}")
    print(f"\nFull sweep (XGBoost, temporal test split = 2025 season):\n")
    print(df.to_string(index=False))
    return df, recommended


def main():
    df = load_real_dataset()
    train_df, test_df = temporal_split(df)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["fire_risk_label"]
    X_test  = test_df[FEATURE_COLUMNS]
    y_test  = test_df["fire_risk_label"].values

    model = XGBoostModel()
    saved = MODELS_DIR / "xgboost_real.json"
    if saved.exists():
        # Only load if the saved model's feature names match current FEATURE_COLUMNS.
        # After the fwi_roll7->fwi_lag1 rename the old file is stale - retrain.
        import json as _json
        try:
            import xgboost as _xgb
            _tmp = _xgb.Booster()
            _tmp.load_model(str(saved))
            saved_features = _tmp.feature_names
            if saved_features == FEATURE_COLUMNS:
                model.model.load_model(str(saved))
                print("Loaded saved xgboost_real.json")
            else:
                print(f"Saved model has stale features ({saved_features[:3]}...) - retraining.")
                model.fit(X_train, y_train)
        except Exception:
            print("Could not load saved model - retraining.")
            model.fit(X_train, y_train)
    else:
        print("No saved model found - training now...")
        model.fit(X_train, y_train)

    _, y_prob = model.predict(X_test, threshold=0.5)  # get raw probs
    sweep(y_test, y_prob)


if __name__ == "__main__":
    main()
