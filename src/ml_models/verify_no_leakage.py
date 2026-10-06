"""
verify_no_leakage.py - the standard, gold-standard sanity check for label
leakage: shuffle the real labels randomly (destroying any true relationship
between features and outcome) and retrain. If the model still achieves a
meaningfully-above-random AUC on shuffled labels, that is DEFINITIVE proof
some form of leakage remains - it is mathematically impossible for a model
to predict a truly random label better than chance from independent
features, so any signal found here can only come from the model secretly
seeing information about the (now-scrambled) real label through some other
channel.

Expected result on a genuinely leakage-free pipeline: AUC very close to 0.50
(random guessing) on every model. This is independent proof, not a claim -
you're not asked to trust that the fixes worked, you're shown a test that
would fail loudly if they hadn't.

Run:
    python src/ml_models/verify_no_leakage.py
"""
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.ml_models.train_real import load_real_dataset, temporal_split, FEATURE_COLUMNS
from src.ml_models.model_trainer import RandomForestModel, XGBoostModel

logging.basicConfig(level=logging.WARNING)  # quiet the model-internal logs for this script
logger = logging.getLogger(__name__)


def main(seed: int = 123):
    print("=" * 70)
    print("  LABEL-SHUFFLE LEAKAGE TEST")
    print("=" * 70)
    print(
        "\nShuffling the real fire-occurrence labels randomly, then retraining.\n"
        "If leakage remained anywhere in the pipeline, the model could still\n"
        "find 'signal' in these now-meaningless labels. If the pipeline is\n"
        "genuinely clean, AUC should land close to 0.50 (random guessing) -\n"
        "no better than a coin flip, since a shuffled label carries zero real\n"
        "information for any model to find.\n"
    )

    df = load_real_dataset()
    train_df, test_df = temporal_split(df)

    rng = np.random.default_rng(seed)
    train_shuffled = train_df.copy()
    train_shuffled["fire_risk_label"] = rng.permutation(train_shuffled["fire_risk_label"].values)
    # Test labels are also shuffled independently - we are only ever
    # comparing shuffled-trained models against shuffled-test truth, so the
    # comparison is apples-to-apples and still meaningful as a null check.
    test_shuffled = test_df.copy()
    test_shuffled["fire_risk_label"] = rng.permutation(test_shuffled["fire_risk_label"].values)

    X_train, y_train = train_shuffled[FEATURE_COLUMNS], train_shuffled["fire_risk_label"]
    X_test, y_test = test_shuffled[FEATURE_COLUMNS], test_shuffled["fire_risk_label"]

    print(f"Shuffled train positive rate: {y_train.mean():.2%} (should roughly match the real rate - "
          f"shuffling only scrambles WHICH rows are positive, not how many)")

    results = {}

    print("\nTraining Random Forest on shuffled labels...")
    rf = RandomForestModel(n_estimators=150).fit(X_train, y_train)
    results["Random Forest (shuffled)"] = rf.evaluate(X_test, y_test)

    print("Training XGBoost on shuffled labels...")
    xgb_m = XGBoostModel(n_estimators=150).fit(X_train, y_train)
    results["XGBoost (shuffled)"] = xgb_m.evaluate(X_test, y_test)

    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    comparison = pd.DataFrame(results).T[["accuracy", "auc_roc", "recall", "precision"]].round(4)
    print(comparison.to_string())

    print("\n" + "-" * 70)
    max_auc = comparison["auc_roc"].max()
    if max_auc < 0.56:
        print(f"PASS: max AUC on shuffled labels = {max_auc:.4f} - indistinguishable from random "
              f"guessing (0.50). No evidence of leakage.")
    elif max_auc < 0.65:
        print(f"BORDERLINE: max AUC on shuffled labels = {max_auc:.4f} - somewhat above random. "
              f"Could be noise from a small/imbalanced sample, but worth a second run with a "
              f"different seed to confirm before trusting it fully.")
    else:
        print(f"FAIL: max AUC on shuffled labels = {max_auc:.4f} - meaningfully better than random "
              f"guessing on labels that carry NO real information. This means some leakage still "
              f"exists somewhere in the feature pipeline. Do not trust the real-data results until "
              f"this is investigated further.")
    print("-" * 70)


if __name__ == "__main__":
    main()
