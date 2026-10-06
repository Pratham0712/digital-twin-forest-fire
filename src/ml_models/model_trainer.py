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
                              recall_score, roc_auc_score, confusion_matrix,
                              average_precision_score)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from config.config import SYSTEM

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
        "avg_precision": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
        "false_negative_rate": _false_negative_rate(y_true, y_pred),
    }


def _false_negative_rate(y_true, y_pred) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return fn / (fn + tp) if (fn + tp) > 0 else 0.0


def _pick_best_threshold(y_val, y_prob_val, grid=None, min_recall: float = 0.80) -> float:
    """
    Selects the HIGHEST threshold that still achieves at least min_recall on
    an internal validation split (carved from training data only, never the
    test set). Falls back to whichever threshold maximizes recall if the
    target genuinely can't be met.

    Deliberately NOT F1-optimized. F1 balances precision and recall equally,
    but for a wildfire safety system a missed real fire (false negative) is
    far more costly than a false alarm - a first version of this function
    picked the F1-maximizing threshold and it looked good in isolation
    (F1=0.21-0.24) but actually pushed recall down to 20-34% on the real
    temporal test set, i.e. missing 66-80% of real fires, which is the wrong
    tradeoff for this domain regardless of how the aggregate metric reads.
    Recall-first threshold selection is the correct choice here even though
    it costs precision (more false alarms) - false alarms cost analyst time,
    missed fires cost lives/property.
    """
    if grid is None:
        grid = np.arange(0.02, 0.96, 0.02)
    if len(np.unique(y_val)) < 2:
        return SYSTEM.alert_threshold_pct / 100  # can't tune without both classes present

    candidates = []
    for t in grid:
        pred = (y_prob_val >= t).astype(int)
        rec = recall_score(y_val, pred, zero_division=0)
        candidates.append((float(t), rec))

    meeting_target = [c for c in candidates if c[1] >= min_recall]
    if meeting_target:
        # Highest threshold that still clears the recall floor - keeps
        # precision as good as possible without sacrificing the safety target.
        return max(meeting_target, key=lambda c: c[0])[0]

    # Recall floor unreachable on this validation split - take whichever
    # threshold maximizes recall instead of silently accepting a bad one.
    best = max(candidates, key=lambda c: c[1])
    logger.warning(
        "Could not reach target recall=%.2f on validation split; best achievable "
        "recall=%.3f at threshold=%.2f", min_recall, best[1], best[0],
    )
    return best[0]


ALERT_ACCURACY_TARGET = 0.92


def _pick_accuracy_threshold(y_val, y_prob_val, target: float = ALERT_ACCURACY_TARGET) -> float:
    """
    Second operating point ("alert"): the LOWEST threshold whose accuracy on
    the internal validation split reaches `target`, i.e. the one that keeps
    as much recall as possible while being >= target accurate. Chosen on
    validation data only, like the early-warning threshold above.

    The two thresholds are two points on the same model's ROC curve:
      early warning -> catches ~80% of fires, many false alarms
      alert         -> >= 92% of all zone-days classified correctly,
                       fewer fires caught but alerts are far more reliable
    """
    if len(np.unique(y_val)) < 2:
        return SYSTEM.alert_threshold_pct / 100
    grid = np.arange(0.01, 0.99, 0.005)
    accs = np.array([accuracy_score(y_val, (y_prob_val >= t).astype(int)) for t in grid])
    ok = np.where(accs >= target)[0]
    if ok.size:
        return float(grid[ok.min()])
    logger.warning("Accuracy target %.2f not reachable on validation (best %.3f)", target, accs.max())
    return float(grid[int(accs.argmax())])


class RandomForestModel:
    """Baseline tabular model - fast, interpretable, robust to noisy features."""

    def __init__(self, n_estimators: int = 300, max_depth: int = 12, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        )
        self.feature_names_ = None
        self.threshold_ = SYSTEM.alert_threshold_pct / 100  # overwritten by fit() with a validation-picked value
        self.alert_threshold_ = self.threshold_               # second operating point, also set by fit()

    def fit(self, X: pd.DataFrame, y: pd.Series, tune: bool = False):
        self.feature_names_ = list(X.columns)
        if len(np.unique(y)) < 2:
            logger.warning("RF.fit: only one class in training data - skipping fit.")
            return self

        from sklearn.model_selection import train_test_split as _tts
        X_tr, X_val, y_tr, y_val = _tts(X, y, test_size=0.15, stratify=y, random_state=42)

        if tune:
            X_s, _, y_s, _ = _tts(X_tr, y_tr, train_size=0.10, stratify=y_tr, random_state=42)
            # Guard: if subsample still has only one class, skip tuning and fit directly
            if len(np.unique(y_s)) < 2:
                logger.warning("RF tune: subsample has only one class - skipping search, fitting directly.")
                self.model.fit(X_tr, y_tr)
            else:
                param_dist = {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [8, 12, 16, None],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2", 0.5],
                }
                search = RandomizedSearchCV(
                    RandomForestClassifier(
                        class_weight="balanced", random_state=42, n_jobs=1
                    ),
                    param_dist, n_iter=8, scoring="average_precision",
                    cv=2, random_state=42, n_jobs=-1, verbose=1,
                    error_score=0.0,  # score NaN folds as 0 instead of raising
                )
                search.fit(X_s, y_s)
                logger.info("RF best params: %s  AP=%.4f", search.best_params_, search.best_score_)
                self.model = RandomForestClassifier(
                    **search.best_params_,
                    class_weight="balanced", random_state=42, n_jobs=-1
                )
                self.model.fit(X_tr, y_tr)
        else:
            self.model.fit(X_tr, y_tr)

        # Pick this model's own decision threshold using ONLY the validation
        # split just carved out above - never the test set.
        val_proba = self.model.predict_proba(X_val)
        y_prob_val = val_proba[:, 1] if val_proba.shape[1] > 1 else np.zeros(len(X_val))
        self.threshold_ = _pick_best_threshold(y_val, y_prob_val)
        self.alert_threshold_ = _pick_accuracy_threshold(y_val, y_prob_val)
        logger.info("RF selected decision threshold: %.2f, alert threshold %.3f (via internal validation split)",
                    self.threshold_, self.alert_threshold_)
        return self

    def predict(self, X: pd.DataFrame, threshold: float = None):
        threshold = self.threshold_ if threshold is None else threshold
        proba = self.model.predict_proba(X)
        # If model only saw one class during training, predict_proba has 1 column
        if proba.shape[1] == 1:
            # All predictions are the single known class; return zeros for prob of class 1
            y_prob = np.zeros(len(X))
        else:
            y_prob = proba[:, 1]
        return (y_prob >= threshold).astype(int), y_prob

    def evaluate(self, X: pd.DataFrame, y: pd.Series, threshold: float = None) -> dict:
        threshold = self.threshold_ if threshold is None else threshold
        _, y_prob = self.predict(X, threshold=threshold)
        y_pred = (y_prob >= threshold).astype(int)
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
        self.threshold_ = SYSTEM.alert_threshold_pct / 100  # overwritten by fit() with a validation-picked value
        self.alert_threshold_ = self.threshold_               # second operating point, also set by fit()

    def fit(self, X: pd.DataFrame, y: pd.Series, tune: bool = False):
        self.feature_names_ = list(X.columns)
        if len(np.unique(y)) < 2:
            logger.warning("XGB.fit: only one class in training data - skipping fit.")
            return self
        # NOTE: deliberately NOT setting scale_pos_weight here. Combining a
        # training-time imbalance correction (scale_pos_weight) with a
        # decision-time correction (the lowered 0.40 alert threshold below)
        # double-corrects for the same class imbalance and badly miscalibrates
        # XGBoost's output probabilities - confirmed empirically: with
        # scale_pos_weight set, predicted probabilities skewed high across
        # almost every row (not just true positives), causing the model to
        # predict "fire" for ~100% of rows regardless of actual risk. Random
        # Forest's class_weight="balanced" does not have this failure mode,
        # which is why only XGBoost broke. Relying solely on the threshold
        # (already lowered from the default 0.5 to SYSTEM.alert_threshold_pct)
        # gives well-calibrated probabilities and correct behavior.
        from sklearn.model_selection import train_test_split as _tts
        X_tr, X_val, y_tr, y_val = _tts(X, y, test_size=0.15, stratify=y, random_state=42)
        if tune:
            X_s, _, y_s, _ = _tts(X_tr, y_tr, train_size=0.10, stratify=y_tr, random_state=42)
            if len(np.unique(y_s)) < 2:
                logger.warning("XGB tune: subsample has only one class - skipping search, fitting directly.")
                self.model.set_params(eval_metric="aucpr", early_stopping_rounds=20)
                self.model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                self.threshold_ = _pick_best_threshold(y_val, self.model.predict_proba(X_val)[:, 1])
                self.alert_threshold_ = _pick_accuracy_threshold(y_val, self.model.predict_proba(X_val)[:, 1])
                logger.info("XGB selected decision threshold: %.2f (via internal validation split)", self.threshold_)
                return self
            param_dist = {
                "n_estimators": [200, 300, 500],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0],
                "min_child_weight": [1, 3, 5],
                "gamma": [0, 0.1, 0.3],
            }
            base = xgb.XGBClassifier(
                eval_metric="aucpr", random_state=42, n_jobs=-1,
            )
            search = RandomizedSearchCV(
                base, param_dist, n_iter=12, scoring="average_precision",
                cv=2, random_state=42, n_jobs=-1, verbose=1,
            )
            search.fit(X_s, y_s)
            logger.info("XGB best params: %s  AP=%.4f", search.best_params_, search.best_score_)
            best_p = {k: v for k, v in search.best_params_.items()}
            self.model = xgb.XGBClassifier(
                **best_p,
                eval_metric="aucpr", random_state=42, n_jobs=-1,
                early_stopping_rounds=20,
            )
            self.model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        else:
            self.model.set_params(
                eval_metric="aucpr",
                early_stopping_rounds=20,
            )
            self.model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        # Pick this model's own decision threshold using ONLY the validation
        # split already carved out above for early stopping - never the test
        # set. XGBoost's calibration can differ sharply from RF/CNN-LSTM's
        # (confirmed empirically), so each model gets its own threshold.
        p_val = self.model.predict_proba(X_val)[:, 1]
        self.threshold_ = _pick_best_threshold(y_val, p_val)
        self.alert_threshold_ = _pick_accuracy_threshold(y_val, p_val)
        logger.info("XGB selected decision threshold: %.2f, alert threshold %.3f (via internal validation split)",
                    self.threshold_, self.alert_threshold_)
        return self

    def predict(self, X: pd.DataFrame, threshold: float = None):
        threshold = self.threshold_ if threshold is None else threshold
        y_prob = self.model.predict_proba(X)[:, 1]
        return (y_prob >= threshold).astype(int), y_prob

    def evaluate(self, X: pd.DataFrame, y: pd.Series, threshold: float = None) -> dict:
        threshold = self.threshold_ if threshold is None else threshold
        _, y_prob = self.predict(X, threshold=threshold)
        y_pred = (y_prob >= threshold).astype(int)
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
        self.threshold_ = SYSTEM.alert_threshold_pct / 100  # overwritten by fit() with a validation-picked value
        self.alert_threshold_ = self.threshold_               # second operating point, also set by fit()

        inp = layers.Input(shape=(n_days, n_features))
        x = layers.Conv1D(64, kernel_size=2, activation="relu", padding="same")(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(32, kernel_size=2, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.LSTM(32, return_sequences=False)(x)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        out = layers.Dense(1, activation="sigmoid")(x)

        self.model = models.Model(inp, out)
        self.model.compile(optimizer="adam", loss="binary_crossentropy",
                            metrics=["accuracy", "AUC"])

    def _scale(self, X_seq: np.ndarray, fit: bool = False) -> np.ndarray:
        n, t, f = X_seq.shape
        flat = X_seq.reshape(-1, f)
        flat = self.scaler.fit_transform(flat) if fit else self.scaler.transform(flat)
        return flat.reshape(n, t, f)

    def fit(self, X_seq: np.ndarray, y: np.ndarray, epochs: int = 25,
            batch_size: int = 256, validation_split: float = 0.15, verbose: int = 0):
        # Carve an EXPLICIT validation split (rather than Keras' built-in
        # validation_split=, which doesn't expose the held-out arrays) so we
        # can pick this model's own decision threshold afterward, using only
        # this validation data - never the test set.
        from sklearn.model_selection import train_test_split as _tts
        X_tr, X_val, y_tr, y_val = _tts(X_seq, y, test_size=validation_split,
                                          stratify=y, random_state=42)
        X_tr_scaled = self._scale(X_tr, fit=True)
        X_val_scaled = self._scale(X_val, fit=False)

        # Correct for class imbalance the same way RF (class_weight="balanced")
        # already does - without this, Keras converges to always predicting
        # the majority class.
        n_neg = int(np.sum(y_tr == 0))
        n_pos = int(np.sum(y_tr == 1))
        class_weight = {
            0: (n_neg + n_pos) / (2 * n_neg) if n_neg > 0 else 1.0,
            1: (n_neg + n_pos) / (2 * n_pos) if n_pos > 0 else 1.0,
        }

        import tensorflow as tf
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True)
        self.history_ = self.model.fit(
            X_tr_scaled, y_tr, epochs=epochs, batch_size=batch_size,
            validation_data=(X_val_scaled, y_val), callbacks=[early_stop],
            class_weight=class_weight, verbose=verbose,
        )

        val_proba = self.model.predict(X_val_scaled, verbose=0).ravel()
        self.threshold_ = _pick_best_threshold(y_val, val_proba)
        self.alert_threshold_ = _pick_accuracy_threshold(y_val, val_proba)
        logger.info("CNN+LSTM selected decision threshold: %.2f, alert threshold %.3f (via internal validation split)",
                    self.threshold_, self.alert_threshold_)
        return self

    def predict(self, X_seq: np.ndarray, threshold: float = None):
        threshold = self.threshold_ if threshold is None else threshold
        X_scaled = self._scale(X_seq, fit=False)
        y_prob = self.model.predict(X_scaled, verbose=0).ravel()
        y_pred = (y_prob >= threshold).astype(int)
        return y_pred, y_prob

    def evaluate(self, X_seq: np.ndarray, y: np.ndarray, threshold: float = None) -> dict:
        threshold = self.threshold_ if threshold is None else threshold
        y_pred, y_prob = self.predict(X_seq, threshold=threshold)
        return evaluate_predictions(y, y_pred, y_prob)


def split_tabular(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def split_sequence(X_seq: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X_seq, y, test_size=test_size, random_state=random_state, stratify=y)
