"""
risk_score.py - ML-based Fire Risk Score (0-100 %).

Definition
  The score is the percentile rank of the model's predicted fire probability among
  all zone-days of the model's TRAINING fire seasons (2023-2024, January-May):
      score = share (%) of historical zone-days whose predicted probability is
              lower than this prediction.
  "Risk score 82 %" therefore means: the model rates this zone-day as more fire-prone
  than 82 % of the historical fire-season zone-days. It is a relative ranking, NOT
  the probability that a fire occurs (that stays the model's fire-occurrence
  probability, reported separately). Because it is a monotone transform of the model
  output, it ranks exactly as the model does (held-out ROC-AUC is unchanged).

Validation
  The categories are checked on the held-out 2025 season, which played no part in the
  reference distribution: the observed share of zone-days with a satellite-detected
  fire rises steadily from the LOW band to the EXTREME band (table stored with the
  artefact and printed in the report).

Artefact: models/risk_score_real.json (1001 reference quantiles + bands + validation).
Build / refresh (operator, after a model change):
    python -m src.ml_models.risk_score build
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np

from config.config import MODELS_DIR

ARTEFACT = {"xgboost_real.json": "risk_score_real.json"}
# category bands on the 0-100 score (upper bound exclusive), chosen on the score scale and validated on 2025
BANDS = ((50.0, "LOW"), (75.0, "MODERATE"), (90.0, "HIGH"), (95.0, "VERY HIGH"), (101.0, "EXTREME"))
_cache: dict = {}


def category(score: Optional[float]) -> str:
    if score is None or not np.isfinite(score):
        return "UNKNOWN"
    for hi, lab in BANDS:
        if score < hi:
            return lab
    return BANDS[-1][1]


def valid_score(score) -> Optional[float]:
    """A stored ML Fire Risk Score (0-100) or None - never rescaled."""
    try:
        v = float(score)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) and 0.0 <= v <= 100.0 else None


def format_score(score) -> Optional[str]:
    """How every output prints the score: '42%' (same text in the report, e-mail and SMS)."""
    v = valid_score(score)
    return None if v is None else f"{v:.0f}%"


class RiskScore:
    def __init__(self, quantiles, meta: dict):
        self.q = np.maximum.accumulate(np.asarray(quantiles, dtype=float))
        self.meta = meta

    def score(self, p) -> np.ndarray:
        """Percentile (0-100) of probability p within the training reference distribution."""
        p = np.asarray(p, dtype=float)
        grid = np.linspace(0.0, 100.0, len(self.q))
        return np.clip(np.interp(p, self.q, grid, left=0.0, right=100.0), 0.0, 100.0)

    @classmethod
    def load(cls, model_file: str = "xgboost_real.json", models_dir: Path = MODELS_DIR) -> Optional["RiskScore"]:
        name = ARTEFACT.get(model_file)
        if not name:
            return None
        path = Path(models_dir) / name
        key = str(path)
        if key not in _cache:
            if not path.exists():
                _cache[key] = None
            else:
                d = json.loads(path.read_text())
                _cache[key] = cls(d["quantiles"], {k: v for k, v in d.items() if k != "quantiles"})
        return _cache[key]


def build(df=None, models_dir: Path = MODELS_DIR, model_file: str = "xgboost_real.json") -> dict:
    """Reference quantiles from the training seasons + validation table on the held-out season."""
    import xgboost as xgb
    from src.ml_models import model_selection as ms
    df = ms.load_features() if df is None else df
    train, test = ms.split(df)
    meta = json.loads((Path(models_dir) / ms.SERVING_META).read_text())
    cols = meta["feature_columns"]
    m = xgb.XGBClassifier()
    m.load_model(str(Path(models_dir) / model_file))
    p_tr = m.predict_proba(train[cols])[:, 1]
    q = np.quantile(p_tr, np.linspace(0, 1, 1001))
    rs = RiskScore(q, {})
    p_te = m.predict_proba(test[cols])[:, 1]
    s, y = rs.score(p_te), test[ms.LABEL].to_numpy().astype(int)
    rows, lo = [], 0.0
    for hi, lab in BANDS:
        k = (s >= lo) & (s < hi)
        rows.append({"category": lab, "score_from": lo, "score_to": min(hi, 100.0),
                     "share_of_zone_days": round(float(k.mean()), 4),
                     "observed_fire_rate": round(float(y[k].mean()), 4) if k.any() else None,
                     "share_of_fires": round(float(y[k].sum() / max(y.sum(), 1)), 4)})
        lo = hi
    from sklearn.metrics import average_precision_score, roc_auc_score
    out = {"model_file": model_file, "definition": "percentile rank of the model's fire probability among the "
           "training fire-season zone-days", "reference": "training rows " + " to ".join(meta["train_date_range"]),
           "reference_rows": int(len(train)), "validation_season": " to ".join(meta["test_date_range"]),
           "validation": rows, "held_out_roc_auc": round(float(roc_auc_score(y, p_te)), 4),
           "held_out_pr_auc": round(float(average_precision_score(y, p_te)), 4),
           "held_out_base_rate": round(float(y.mean()), 4),
           "bands": [[hi, lab] for hi, lab in BANDS], "quantiles": [float(x) for x in q]}
    (Path(models_dir) / ARTEFACT[model_file]).write_text(json.dumps(out, indent=1))
    _cache.clear()
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        o = build()
        print(json.dumps({k: v for k, v in o.items() if k != "quantiles"}, indent=1))
    else:
        print(__doc__)
