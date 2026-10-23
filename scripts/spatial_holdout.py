"""
spatial_holdout.py - how well does the fire-risk model travel to a state it has
never seen?  Also reports probability calibration.

    python scripts/spatial_holdout.py

1. Leave-one-state-out. For each Indian state: train the pooled model on the OTHER
   states' 2023-24 seasons, score the held-out state on its 2025 and 2026 seasons.
   Compare with the in-state model (trained on all states' 2023-24 incl. that state).
   Spatial + temporal hold-out together is the strictest test available here.
2. Calibration. Reliability of the pooled model's probabilities on the 2025 season
   (Brier score, expected calibration error), before and after a monotone
   (isotonic) recalibration fitted on 2024 only.

Writes data/processed/eval_spatial_holdout.csv and eval_calibration.csv.
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score  # noqa: E402

from config.config import DATA_PROCESSED_DIR, MODELS_DIR  # noqa: E402

logging.getLogger().setLevel(logging.WARNING)


def _fit(X, y, seed=0):
    from xgboost import XGBClassifier
    pos = max(float(y.sum()), 1.0)
    m = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                      min_child_weight=5, reg_lambda=2.0, scale_pos_weight=min((len(y) - pos) / pos, 20.0),
                      eval_metric="logloss", random_state=seed, n_jobs=2, tree_method="hist")
    m.fit(X, y)
    return m


def _metrics(y, p):
    top = max(1, int(round(0.10 * len(y))))
    order = np.argsort(-p)[:top]
    base = float(y.mean())
    return {"fire_rate": round(base, 4), "auc_roc": round(roc_auc_score(y, p), 4),
            "avg_precision": round(average_precision_score(y, p), 4),
            "ap_over_base": round(average_precision_score(y, p) / base, 2) if base else np.nan,
            "recall_top10pct": round(float(y[order].sum() / max(y.sum(), 1)), 4)}


def ece(y, p, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(float(p[m].mean()) - float(y[m].mean()))
    return tot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DATA_PROCESSED_DIR / "india_states_training_data.csv"))
    ap.add_argument("--out", default=str(DATA_PROCESSED_DIR))
    a = ap.parse_args()
    out = Path(a.out)

    import build_india_regions_dataset as bi
    from src.ml_models.fire_history import FEATURE_COLUMNS_V2, FEATURE_COLUMNS_V3
    from src.ml_models.train_real import SECOND_TEST_START, TEST_SEASON_START, add_model_features, add_terrain_training

    states = bi.india_states()
    data = pd.read_csv(a.data)
    data["date"] = pd.to_datetime(data["date"])
    feats = []
    for name, part in data.groupby("region", sort=False):
        df, _ = add_model_features(part.drop(columns=["region"]), states[name])
        df = add_terrain_training(df, states[name])
        df.insert(0, "region", name)
        feats.append(df)
    terrain_on = all("elev_m" in f.columns for f in feats)
    cols = list(FEATURE_COLUMNS_V3 if terrain_on else FEATURE_COLUMNS_V2)
    if not terrain_on:
        feats = [f.drop(columns=["elev_m", "slope_pct"], errors="ignore") for f in feats]
    df = pd.concat(feats, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    tr = df[df["date"] < TEST_SEASON_START]
    seasons = {"2025": df[(df["date"] >= TEST_SEASON_START) & (df["date"] < SECOND_TEST_START)],
               "2026": df[df["date"] >= SECOND_TEST_START]}
    print(f"{len(df):,} rows, {df['region'].nunique()} states, features: {len(cols)} (terrain {terrain_on})")

    full = _fit(tr[cols], tr["fire_risk_label"])
    rows = []
    for state in sorted(df["region"].unique()):
        loso = _fit(tr[tr["region"] != state][cols], tr[tr["region"] != state]["fire_risk_label"])
        for season, sdf in seasons.items():
            g = sdf[sdf["region"] == state]
            y = g["fire_risk_label"].values
            if len(g) == 0 or len(np.unique(y)) < 2:
                continue
            for label, mdl in (("trained incl. this state", full), ("never saw this state", loso)):
                p = mdl.predict_proba(g[cols])[:, 1]
                rows.append({"state": state, "season": season, "model": label, **_metrics(y, p)})
    res = pd.DataFrame(rows)
    res.to_csv(out / "eval_spatial_holdout.csv", index=False)
    piv = res.pivot_table(index=["season", "state"], columns="model", values=["auc_roc", "ap_over_base",
                                                                              "recall_top10pct"])
    print(piv.round(3).to_string())
    summ = res.groupby(["season", "model"])[["auc_roc", "ap_over_base", "recall_top10pct"]].mean().round(3)
    print("\nMean over states:")
    print(summ.to_string())

    # calibration: fit on 2023, recalibrate on 2024, test 2025 (all temporal, no test leakage)
    t23 = tr[tr["date"].dt.year == 2023]
    t24 = tr[tr["date"].dt.year == 2024]
    m23 = _fit(t23[cols], t23["fire_risk_label"])
    p24 = m23.predict_proba(t24[cols])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p24, t24["fire_risk_label"].values)
    cal_rows = []
    for season, sdf in seasons.items():
        y = sdf["fire_risk_label"].values
        p_raw = m23.predict_proba(sdf[cols])[:, 1]
        p_iso = iso.predict(p_raw)
        for label, p in (("raw (class-weighted XGBoost)", p_raw), ("isotonic, fitted on 2024", p_iso)):
            cal_rows.append({"season": season, "probabilities": label, "brier": round(brier_score_loss(y, p), 4),
                             "ece": round(ece(y, p), 4), "mean_predicted": round(float(p.mean()), 4),
                             "observed_rate": round(float(y.mean()), 4)})
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(out / "eval_calibration.csv", index=False)
    print("\nCalibration:")
    print(cal.to_string(index=False))


if __name__ == "__main__":
    main()
