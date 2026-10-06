"""
risk_index.py - turns the model's calibrated fire probability into the 0-100%
risk index the dashboard, alerts and spread simulation use.

Why not show the raw probability?
The model predicts "will a satellite-confirmed fire occur in this 11 km cell
today". Fires are rare (~4% of zone-days), so even a clearly dangerous zone
has a probability of 20-30%, and a fixed "alert above 40%" rule would almost
never fire. Operational systems (e.g. the Canadian FWI danger classes) solve
this the same way: define danger classes from validated operating points,
not from round-number probabilities.

The index is a monotone piecewise-linear map anchored at three points
measured on the TRAINING seasons (never the test season):

    probability                          -> index   severity band
    decision threshold (80% recall)      -> 20%     MODERATE starts
    alert operating point (>=92% acc.)   -> 40%     HIGH starts (alert)
    precision >= EXTREME_PRECISION (50%) -> 60%     EXTREME starts (email)

so "EXTREME" means: historically, at least half of the zones scored this
high really did burn. Ranking is unchanged (monotone), so AUC and every
other ranking metric are identical to the raw model's.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

HIGH_PRECISION = 0.25
EXTREME_PRECISION = 0.50
RISK_INDEX_FILENAME = "risk_index.json"


@dataclass
class RiskIndex:
    t_moderate: float
    t_high: float
    t_extreme: float
    high_precision: float = HIGH_PRECISION
    extreme_precision: float = EXTREME_PRECISION

    @staticmethod
    def _threshold_for_precision(y: np.ndarray, p: np.ndarray, target: float,
                                 min_support: int = 50) -> Optional[float]:
        """Lowest probability cutoff whose precision (share of flagged
        zone-days that truly had a fire) reaches `target`."""
        order = np.argsort(-p)
        p_sorted, y_sorted = p[order], y[order]
        tp = np.cumsum(y_sorted)
        n = np.arange(1, len(y_sorted) + 1)
        precision = tp / n
        ok = np.where((precision >= target) & (n >= min_support))[0]
        return float(p_sorted[ok.max()]) if ok.size else None

    @classmethod
    def fit(cls, y, p, decision_threshold: float,
            high_threshold: Optional[float] = None) -> "RiskIndex":
        """high_threshold: the model's validation-selected alert operating
        point (>= 92% accuracy). When given, HIGH starts exactly there, so a
        HIGH/EXTREME alert on the dashboard is the alert operating point."""
        y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
        t_mod = float(decision_threshold)
        t_high = (float(high_threshold) if high_threshold is not None
                  else cls._threshold_for_precision(y, p, HIGH_PRECISION))
        t_ext = cls._threshold_for_precision(y, p, EXTREME_PRECISION)
        top = float(np.quantile(p, 0.999))
        # Keep the anchors strictly increasing even if a precision target is
        # unreachable on this data (then fall back to high quantiles).
        t_high = t_high if t_high is not None and t_high > t_mod else max(t_mod * 1.5, float(np.quantile(p, 0.99)))
        t_ext = t_ext if t_ext is not None and t_ext > t_high else max(t_high * 1.5, top)
        return cls(t_moderate=t_mod, t_high=float(t_high), t_extreme=float(min(t_ext, 0.99)))

    def transform(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        xs = [0.0, self.t_moderate, self.t_high, self.t_extreme, 1.0]
        ys = [0.0, 0.20, 0.40, 0.60, 1.0]
        return np.clip(np.interp(p, xs, ys), 0.0, 1.0)

    def save(self, models_dir: Path, filename: str = RISK_INDEX_FILENAME):
        Path(models_dir, filename).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, models_dir: Path, filename: str = RISK_INDEX_FILENAME) -> Optional["RiskIndex"]:
        path = Path(models_dir) / filename
        if not path.exists():
            return None
        return cls(**json.loads(path.read_text()))


def tier_performance(y, p, index: RiskIndex) -> list:
    """How each alert tier performed on a labelled set (used on the held-out
    2025 season for the report): share of all fires captured, and hit rate."""
    y, r = np.asarray(y).astype(int), index.transform(p)
    total_fires = max(int(y.sum()), 1)
    rows = []
    for name, lo in (("EXTREME", 0.60), ("HIGH or above", 0.40), ("MODERATE or above", 0.20)):
        flagged = r >= lo
        rows.append({
            "tier": name,
            "share_of_zone_days_flagged": float(flagged.mean()),
            "share_of_fires_captured": float(y[flagged].sum() / total_fires),
            "hit_rate": float(y[flagged].mean()) if flagged.any() else float("nan"),
        })
    return rows
