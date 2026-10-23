"""
prediction_service.py - the ONE place a fire-risk prediction is made.

Every dashboard value derived from the model (zone risk, region peak / mean,
severity, alerts, Spread Simulation HUD, reports, notifications) is computed
from DigitalTwin.current_snapshot, and the snapshot's scores come from
`predict()` below. No page computes its own risk.

What the numbers mean
  * fire_probability : the model's probability that a satellite-detected fire
                       occurs in the 0.1-degree zone that day (a class
                       probability of a rare event; fires are ~4 % of zone-days).
  * risk_index       : 0-1 RISK INDEX, a monotone map of that probability onto
                       validated operating points (risk_index.py): 20 % = early-
                       warning threshold, 40 % = alert threshold, 60 % = at least
                       half of historical zone-days this high burned. It is an
                       index, not a probability, and is labelled as such.
  * category         : LOW / MODERATE / HIGH / EXTREME from the same bands
                       (AlertEngine.SEVERITY_BANDS).

Input safety: the feature matrix is re-ordered to the model's own feature
schema; infinities become missing values (XGBoost's native missing handling);
missing columns are reported, never silently replaced by invented numbers.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from config.config import MODELS_DIR

logger = logging.getLogger(__name__)

META_FOR_MODEL = {"xgboost_real.json": "training_metadata_real.json",
                  "xgboost_india.json": "training_metadata_india.json"}
SEVERITY_BANDS = ((0.60, "EXTREME"), (0.40, "HIGH"), (0.20, "MODERATE"), (0.0, "LOW"))


def categorize(index_value: float) -> str:
    if index_value is None or not np.isfinite(index_value):
        return "UNKNOWN"
    for cutoff, label in SEVERITY_BANDS:
        if index_value >= cutoff:
            return label
    return "LOW"


@dataclass
class PredictionResult:
    fire_probability: np.ndarray
    risk_index: np.ndarray
    categories: List[str]
    model_version: str
    model_file: str
    predicted_utc: str
    feature_columns: List[str]
    missing_features: List[str] = field(default_factory=list)
    index_available: bool = True
    risk_score: Optional[np.ndarray] = None          # ML-based Fire Risk Score 0-100 (risk_score.py), None if no artefact

    def summary(self) -> dict:
        return {"model_version": self.model_version, "model_file": self.model_file,
                "predicted_utc": self.predicted_utc, "n": int(len(self.risk_index)),
                "missing_features": list(self.missing_features), "index_available": self.index_available,
                "max_risk_index": float(np.max(self.risk_index)) if len(self.risk_index) else 0.0,
                "mean_risk_index": float(np.mean(self.risk_index)) if len(self.risk_index) else 0.0}


def model_metadata(model_file: str, models_dir: Path = MODELS_DIR) -> dict:
    name = META_FOR_MODEL.get(model_file)
    path = Path(models_dir) / name if name else None
    try:
        return json.loads(path.read_text()) if path and path.exists() else {}
    except (OSError, ValueError):
        return {}


def model_version(model_file: str, models_dir: Path = MODELS_DIR) -> str:
    """Registered version of the serving model (model_registry.json), else the
    metadata version, else a content hash - never a made-up label."""
    import hashlib
    path = Path(models_dir) / model_file
    sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else "missing"
    try:
        reg = json.loads((Path(models_dir) / "model_registry.json").read_text())
        for v in reg.get("versions", []):
            if v.get("sha256") == sha:
                return v["version"]
    except (OSError, ValueError):
        pass
    meta = model_metadata(model_file, models_dir)
    if meta.get("model_version"):
        return str(meta["model_version"])
    return f"fs{meta.get('feature_set_version', '?')}-{sha[:8]}"


def feature_schema(model) -> Optional[List[str]]:
    try:
        names = model.model.get_booster().feature_names
        return list(names) if names else None
    except Exception:
        return None


def prepare_features(frame: pd.DataFrame, columns: List[str]):
    """Model-ordered float matrix; inf -> NaN; returns (X, missing_columns)."""
    missing = [c for c in columns if c not in frame.columns]
    X = frame.reindex(columns=columns).copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    X = X.apply(pd.to_numeric, errors="coerce").astype(float)
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, missing


def predict(model, frame: pd.DataFrame, risk_index=None, model_file: str = "xgboost_real.json",
            models_dir: Path = MODELS_DIR) -> PredictionResult:
    columns = feature_schema(model) or list(frame.columns)
    X, missing = prepare_features(frame, columns)
    if missing:
        logger.warning("prediction: %d model feature(s) missing from the input (treated as missing values): %s",
                       len(missing), missing)
    prob = np.clip(model.model.predict_proba(X)[:, 1].astype(float), 0.0, 1.0)
    idx = risk_index.transform(prob) if risk_index is not None else prob
    from src.ml_models.risk_score import RiskScore
    rs = RiskScore.load(model_file, models_dir)
    return PredictionResult(risk_score=None if rs is None else rs.score(prob),
        fire_probability=prob, risk_index=np.asarray(idx, dtype=float),
        categories=[categorize(v) for v in idx], model_version=model_version(model_file, models_dir),
        model_file=model_file, predicted_utc=datetime.now(timezone.utc).isoformat(),
        feature_columns=columns, missing_features=missing, index_available=risk_index is not None)
