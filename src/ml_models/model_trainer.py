"""
MLPredictor layer (Report Ch.6.1.1, Layer 3) - three model implementations
per the project PPT spec: Random Forest, XGBoost, and CNN+LSTM. These replace
the originally-planned Facebook Prophet model, which is a univariate
time-series forecaster and not suited to multi-feature spatial fire-risk
classification.

Each model exposes the same train/predict/evaluate interface so train.py can
loop over them uniformly and build the comparison table required for the
research paper.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

logger = logging.getLogger(__name__)


def evaluate_predictions(y_true, y_pred, y_prob) -> dict:
    """Standard classification metrics used consistently across all three
    models so results are directly comparable in the paper's comparison
    table (agreed requirement: real metrics, not cherry-picked)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
        "false_negative_rate": _false_negative_rate(y_true, y_pred),
    }


def _false_negative_rate(y_true, y_pred) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return fn / (fn + tp) if (fn + tp) > 0 else 0.0


class RandomForestModel:
    """Baseline tabular model - fast, interpretable, robust to noisy features."""

    def __init__(self, n_estimators: int = 300, max_depth: int = 12, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        )
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names_ = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X), self.model.predict_proba(X)[:, 1]

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        y_pred, y_prob = self.predict(X)
        return evaluate_predictions(y, y_pred, y_prob)

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=self.feature_names_).sort_values(ascending=False)


class XGBoostModel:
    """Gradient-boosted trees - typically strongest tabular baseline; used as
    the primary production model behind the dashboard's live risk score."""

    def __init__(self, n_estimators: int = 300, max_depth: int = 6,
                 learning_rate: float = 0.05, random_state: int = 42):
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=random_state, n_jobs=-1,
        )
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names_ = list(X.columns)
        scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
        self.model.set_params(scale_pos_weight=scale_pos_weight)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X), self.model.predict_proba(X)[:, 1]

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        y_pred, y_prob = self.predict(X)
        return evaluate_predictions(y, y_pred, y_prob)

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=self.feature_names_).sort_values(ascending=False)


class CNNLSTMModel:
    """
    Sequence model over the trailing n-day feature history per zone
    (timeseries_builder.build_history). Captures trend (e.g. FWI rising for
    3 straight days) that single-snapshot tabular models cannot see.
    Conv1D extracts local day-to-day patterns; LSTM captures longer trend.
    """

    def __init__(self, n_days: int, n_features: int, random_state: int = 42):
        import tensorflow as tf
        tf.random.set_seed(random_state)
        from tensorflow.keras import layers, models

        self.scaler = StandardScaler()
        self.n_days = n_days
        self.n_features = n_features

        inp = layers.Input(shape=(n_days, n_features))
        x = layers.Conv1D(32, kernel_size=2, activation="relu", padding="same")(inp)
        x = layers.BatchNormalization()(x)
        x = layers.LSTM(32, return_sequences=False)(x)
        x = layers.Dense(16, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        out = layers.Dense(1, activation="sigmoid")(x)

        self.model = models.Model(inp, out)
        self.model.compile(optimizer="adam", loss="binary_crossentropy",
                            metrics=["accuracy"])

    def _scale(self, X_seq: np.ndarray, fit: bool = False) -> np.ndarray:
        n, t, f = X_seq.shape
        flat = X_seq.reshape(-1, f)
        flat = self.scaler.fit_transform(flat) if fit else self.scaler.transform(flat)
        return flat.reshape(n, t, f)

    def fit(self, X_seq: np.ndarray, y: np.ndarray, epochs: int = 25,
            batch_size: int = 32, validation_split: float = 0.15, verbose: int = 0):
        X_scaled = self._scale(X_seq, fit=True)
        import tensorflow as tf
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True)
        self.history_ = self.model.fit(
            X_scaled, y, epochs=epochs, batch_size=batch_size,
            validation_split=validation_split, callbacks=[early_stop], verbose=verbose,
        )
        return self

    def predict(self, X_seq: np.ndarray):
        X_scaled = self._scale(X_seq, fit=False)
        y_prob = self.model.predict(X_scaled, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)
        return y_pred, y_prob

    def evaluate(self, X_seq: np.ndarray, y: np.ndarray) -> dict:
        y_pred, y_prob = self.predict(X_seq)
        return evaluate_predictions(y, y_pred, y_prob)


def split_tabular(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def split_sequence(X_seq: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X_seq, y, test_size=test_size, random_state=random_state, stratify=y)
