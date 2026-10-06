"""
explain.py - per-zone feature attributions from XGBoost's built-in exact
TreeSHAP (`pred_contribs`). Same values the `shap` package computes for a
tree model, without importing shap/matplotlib (multi-second import) and
computed for the whole grid in milliseconds.

Contributions are in log-odds of the model's raw output; for each row
    bias + sum(contributions) = log-odds of the model probability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb


def contributions(xgb_classifier, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Returns (per-feature contributions DataFrame, bias vector)."""
    booster = xgb_classifier.get_booster()
    dm = xgb.DMatrix(X, feature_names=[str(c) for c in X.columns])
    raw = booster.predict(dm, pred_contribs=True)
    return pd.DataFrame(raw[:, :-1], columns=list(X.columns), index=X.index), raw[:, -1]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))
