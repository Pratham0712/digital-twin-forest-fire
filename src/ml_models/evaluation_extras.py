"""
evaluation_extras.py - evaluation that stays honest when fires are rare.

Plain accuracy is misleading when only ~4% of zone-days have a fire (a model
that never warns scores 96%). These helpers report what actually matters for
an early-warning system, all on the held-out season:

  * lift_table          - flag the top X% riskiest zone-days: how many of the
                          real fires are captured, and how much better than
                          picking zone-days at random (lift = precision/base rate)
  * pr_curve            - precision-recall curve (thinned for plotting) + AP
  * persistence_table   - simple "it burned recently" rules vs the model at the
                          SAME flagged share, so the model's added value is measured
  * comparable_metrics  - the same model scored at native resolution and after
                          aggregating cells to a coarser grid, plus base-rate-
                          normalised numbers, so regions with different grids
                          and fire frequencies can be compared fairly
  * summary             - one row of base-rate-normalised headline numbers
"""
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

DEFAULT_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30)


def _arrays(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    return y, p


def lift_table(y, p, fractions: Iterable[float] = DEFAULT_FRACTIONS) -> pd.DataFrame:
    """Flag the top `fraction` of zone-days by score. `lift` is precision divided
    by the base rate: 3.0 means three times better than flagging at random."""
    y, p = _arrays(y, p)
    n, total_pos = len(y), int(y.sum())
    base = total_pos / n if n else float("nan")
    order = np.argsort(-p, kind="mergesort")
    rows = []
    for f in fractions:
        k = max(1, int(np.ceil(f * n)))
        tp = int(y[order[:k]].sum())
        prec = tp / k
        rows.append({
            "flagged_share": round(f, 4), "zone_days_flagged": k,
            "fires_captured": tp,
            "recall": round(tp / total_pos, 4) if total_pos else float("nan"),
            "precision": round(prec, 4),
            "lift_vs_random": round(prec / base, 2) if base else float("nan"),
        })
    return pd.DataFrame(rows)


def pr_curve(y, p, max_points: int = 200) -> pd.DataFrame:
    y, p = _arrays(y, p)
    precision, recall, _ = precision_recall_curve(y, p)
    if len(precision) > max_points:
        idx = np.unique(np.linspace(0, len(precision) - 1, max_points).astype(int))
        precision, recall = precision[idx], recall[idx]
    return pd.DataFrame({"recall": recall.round(4), "precision": precision.round(4)})


def _rule_row(name, flag, y, p) -> dict:
    y = np.asarray(y).astype(int)
    flag = np.asarray(flag).astype(bool)
    n, base = len(y), y.mean()
    k = int(flag.sum())
    tp = int(y[flag].sum())
    share = k / n
    row = {"method": name, "flagged_share": round(share, 4),
           "recall": round(tp / y.sum(), 4) if y.sum() else float("nan"),
           "precision": round(tp / k, 4) if k else float("nan")}
    row["lift_vs_random"] = round(row["precision"] / base, 2) if k and base else float("nan")
    if p is not None and k:
        order = np.argsort(-np.asarray(p, float), kind="mergesort")[:k]
        mtp = int(y[order].sum())
        row["model_recall_same_share"] = round(mtp / y.sum(), 4) if y.sum() else float("nan")
        row["model_precision_same_share"] = round(mtp / k, 4)
    return row


def persistence_table(test_df: pd.DataFrame, prob: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Simple 'it burned recently' rules, each compared with the model when the
    model is allowed to flag the same number of zone-days."""
    y = test_df["fire_risk_label"].values
    rules = {
        "Zone had a fire yesterday": test_df["fire_lag1"].values > 0,
        "Zone had a fire in the last 7 days": test_df["fire_7d"].values > 0,
        "Zone or its neighbours burned in the last 7 days":
            (test_df["fire_7d"].values > 0) | (test_df["nbr5_7d"].values > 0),
    }
    return pd.DataFrame([_rule_row(name, flag, y, prob) for name, flag in rules.items()])


def _auc_ap(y, p):
    y, p = _arrays(y, p)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


def comparable_metrics(test_df: pd.DataFrame, prob: np.ndarray, block_deg: float = 0.5,
                       region: str = "") -> pd.DataFrame:
    """Native-resolution vs coarse-grid scoring. Each coarse block takes the
    highest score of its cells and a fire label if any cell burned that day, so
    a 0.1 degree region is judged on the same footing as the 0.5 degree ones."""
    d = pd.DataFrame({
        "date": test_df["date"].values,
        "blat": np.floor(test_df["latitude"].values / block_deg).astype(int),
        "blon": np.floor(test_df["longitude"].values / block_deg).astype(int),
        "y": test_df["fire_risk_label"].values.astype(int),
        "p": np.asarray(prob, float),
    })
    coarse = d.groupby(["date", "blat", "blon"], sort=False).agg(y=("y", "max"), p=("p", "max"))
    out = []
    for name, y, p in (("native grid", d["y"], d["p"]), (f"{block_deg:g} degree blocks", coarse["y"], coarse["p"])):
        auc, ap = _auc_ap(y, p)
        base = float(np.mean(y))
        out.append({"region": region, "scoring_grid": name, "cell_days": int(len(y)),
                    "fire_rate": round(base, 4), "auc_roc": round(auc, 4),
                    "avg_precision": round(ap, 4),
                    "avg_precision_over_base_rate": round(ap / base, 2) if base else float("nan")})
    return pd.DataFrame(out)


def summary(y, p, alert_threshold: Optional[float] = None) -> dict:
    """Base-rate-normalised headline numbers for one set of predictions."""
    y, p = _arrays(y, p)
    base = float(y.mean())
    auc, ap = _auc_ap(y, p)
    out = {"fire_rate": round(base, 4), "auc_roc": round(auc, 4), "avg_precision": round(ap, 4),
           "ap_over_base_rate": round(ap / base, 2) if base else float("nan")}
    lt = lift_table(y, p, (0.10,)).iloc[0]
    out["recall_top10pct"] = float(lt["recall"])
    out["lift_top10pct"] = float(lt["lift_vs_random"])
    if alert_threshold is not None:
        flag = p >= alert_threshold
        k = int(flag.sum())
        prec = float(y[flag].mean()) if k else float("nan")
        out["alert_flagged_share"] = round(k / len(y), 4)
        out["alert_precision_lift"] = round(prec / base, 2) if k and base else float("nan")
    return out


def season_report(test_df: pd.DataFrame, prob: np.ndarray, season: str, region: str = "",
                  alert_threshold: Optional[float] = None, block_deg: float = 0.5) -> dict:
    """All extra evaluation tables for one held-out season, tagged with `season`.
    Returns {"lift", "pr", "persistence", "comparable", "summary"} DataFrames."""
    y = test_df["fire_risk_label"].values
    tag = lambda df: df.assign(region=region, season=season)      # noqa: E731
    persistence = persistence_table(test_df, prob) if {"fire_lag1", "fire_7d", "nbr5_7d"} <= set(test_df.columns) \
        else pd.DataFrame()
    return {
        "lift": tag(lift_table(y, prob)),
        "pr": tag(pr_curve(y, prob)),
        "persistence": tag(persistence) if len(persistence) else persistence,
        "comparable": comparable_metrics(test_df, prob, block_deg, region).assign(season=season),
        "summary": pd.DataFrame([{"region": region, "season": season, **summary(y, prob, alert_threshold)}]),
    }
