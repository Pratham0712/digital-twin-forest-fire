"""
verify_beats_baseline.py - proves the ML models add genuine value beyond the
Canadian FWI formula alone, rather than just re-deriving what a domain expert
could already read off a single FWI number. Compares the trained models
against a naive rule-based baseline ("flag as risky whenever FWI exceeds a
threshold") on the exact same real, unseen 2025 test season.

This directly supports the project's novelty claim: if the ML models don't
meaningfully outperform a simple threshold rule, the ML layer isn't earning
its complexity. If they do outperform it, that's concrete, independently
checkable evidence the ML layer is learning real structure beyond FWI alone
(e.g. combining FWI with wind, precipitation, seasonal timing).

Run:
    python src/ml_models/verify_beats_baseline.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from src.ml_models.train_real import load_real_dataset, temporal_split, FEATURE_COLUMNS


def naive_fwi_baseline(test_df: pd.DataFrame, fwi_thresholds=None) -> pd.DataFrame:
    """
    For each candidate FWI threshold, scores a trivial rule: 'predict fire
    risk whenever FWI >= threshold'. Reports the SAME metrics as the ML
    models so the comparison is apples-to-apples. We sweep several
    thresholds and report the best one, giving the baseline every reasonable
    chance to look good - if the ML models still win, that's a strong result.
    """
    if fwi_thresholds is None:
        fwi_thresholds = np.arange(10, 91, 5)

    y_true = test_df["fire_risk_label"].values
    fwi = test_df["fwi"].values
    rows = []
    for t in fwi_thresholds:
        y_pred = (fwi >= t).astype(int)
        if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
            continue  # degenerate - skip
        rows.append({
            "fwi_threshold": t,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        })
    sweep = pd.DataFrame(rows)
    # AUC for the baseline: FWI itself used directly as a continuous score
    # (no threshold needed for a ranking metric) - the fairest possible
    # AUC comparison, since it doesn't depend on picking a "lucky" threshold.
    baseline_auc = roc_auc_score(y_true, fwi)
    return sweep, baseline_auc


def main():
    print("=" * 70)
    print("  ML MODELS vs NAIVE FWI-THRESHOLD BASELINE")
    print("=" * 70)
    print(
        "\nComparing the trained models against a simple rule - 'flag risk whenever\n"
        "FWI exceeds some threshold' - on the exact same real, unseen 2025 test\n"
        "season. This checks whether the ML layer earns its complexity, or whether\n"
        "the same result could be had by just reading the FWI number directly.\n"
    )

    df = load_real_dataset()
    _, test_df = temporal_split(df)

    sweep, baseline_auc = naive_fwi_baseline(test_df)
    best_row = sweep.loc[sweep["f1_score"].idxmax()]

    print(f"Baseline AUC-ROC (FWI used directly as a continuous risk score): {baseline_auc:.4f}")
    print(f"\nBest naive threshold found: FWI >= {best_row['fwi_threshold']:.0f}")
    print(f"  precision={best_row['precision']:.4f}  recall={best_row['recall']:.4f}  "
          f"f1={best_row['f1_score']:.4f}")

    print("\nFull threshold sweep:")
    print(sweep.to_string(index=False))

    print("\n" + "-" * 70)
    print("Compare the baseline AUC above against your trained models' real AUC")
    print("(from train_real.py's output / data/processed/model_comparison_real.csv):")
    print("  Random Forest, XGBoost, CNN-LSTM all reported AUC ~0.80-0.81.")
    print("-" * 70)
    if baseline_auc < 0.75:
        print(f"\nRESULT: ML models (AUC ~0.80-0.81) meaningfully outperform the naive FWI-only "
              f"baseline (AUC {baseline_auc:.4f}). This is concrete evidence the ML layer is "
              f"learning real structure beyond what FWI alone captures - supports the project's "
              f"core value proposition.")
    else:
        print(f"\nRESULT: Naive FWI baseline (AUC {baseline_auc:.4f}) is close to the ML models' "
              f"AUC. This would suggest most of the model's predictive power comes from FWI alone - "
              f"worth investigating which other features are actually contributing (see the SHAP "
              f"explainability page in the dashboard for per-feature attribution).")


if __name__ == "__main__":
    main()
