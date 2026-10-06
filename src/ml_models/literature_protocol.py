"""
literature_protocol.py - answers "papers report 93%+ accuracy, why doesn't
yours?" with a like-for-like, clearly labelled comparison.

The SAME data and the SAME XGBoost (fixed settings, no tuning) are scored
under the evaluation protocols that published wildfire papers typically use,
next to the protocol this project uses for its headline numbers. Nothing
here is a new model or a new claim about deployment performance - the
"literature" rows are NOT valid estimates of how well a live system works,
and each row says why.

Protocols
---------
  A  Ours: temporal hold-out. Train 2023-24, test the unseen 2025 season,
     natural fire rate (~4%). Reported at three thresholds.
  B  Random 80/20 split of all zone-days, natural fire rate. Days of the same
     fire and the same zone land on both sides of the split.
  C  Random 80/20 split, then BOTH sides balanced 1:1 (undersample non-fire).
     Typical "balanced dataset" setup.
  D  Small balanced sample (138 fire / 106 non-fire = the 244 rows of the
     Algerian Forest Fire dataset used by many papers), random 70/30 split,
     mean +/- sd over 20 repeats.
  E  Balanced-trained model, tested on the real 2025 season: what happens to
     protocol C's model when it meets real-world class balance.

Each protocol runs on the full v2 feature set and on weather-only features
(what most papers have).

Run:
    python src/ml_models/literature_protocol.py
Writes data/processed/literature_protocol_comparison.csv
"""
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from config.config import DATA_PROCESSED_DIR
from src.ml_models.fire_history import FEATURE_COLUMNS_V2, WEATHER_FEATURE_COLUMNS
from src.ml_models.model_trainer import XGBoostModel
from src.ml_models.train_real import load_real_dataset, temporal_split

logging.basicConfig(level=logging.WARNING)
SEED = 42
LABEL = "fire_risk_label"
FEATURE_SETS = {"v2 (weather + fire history)": FEATURE_COLUMNS_V2,
                "weather only": WEATHER_FEATURE_COLUMNS}


def _clf():
    return xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, eval_metric="logloss", random_state=SEED, n_jobs=-1)


def _balance(df: pd.DataFrame, rng: np.random.Generator, n_pos=None, n_neg=None) -> pd.DataFrame:
    pos, neg = df[df[LABEL] == 1], df[df[LABEL] == 0]
    n_pos = min(len(pos), n_pos or len(pos))
    n_neg = min(len(neg), n_neg or n_pos)
    return pd.concat([pos.sample(n_pos, random_state=int(rng.integers(1 << 30))),
                      neg.sample(n_neg, random_state=int(rng.integers(1 << 30)))])


def _metrics(y, p, threshold=0.5) -> dict:
    pred = (p >= threshold).astype(int)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "auc": roc_auc_score(y, p) if y.nunique() > 1 else np.nan}


def _row(protocol, features, n_tr, n_te, pos_rate, m, note, sd=None):
    r = {"protocol": protocol, "features": features, "train_rows": n_tr, "test_rows": n_te,
         "test_fire_rate": round(pos_rate, 4), **{k: round(float(v), 4) for k, v in m.items()}, "note": note}
    if sd:
        r["accuracy_sd"] = round(float(sd), 4)
    return r


def run(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(SEED)
    train_t, test_t = temporal_split(df)
    base_rate = float(test_t[LABEL].mean())

    for fname, cols in FEATURE_SETS.items():
        # A - temporal hold-out, our operating points
        mdl = XGBoostModel().fit(train_t[cols], train_t[LABEL], tune=False)
        p = mdl.model.predict_proba(test_t[cols])[:, 1]
        for label, thr in [("A  Ours: temporal hold-out 2025, threshold 0.5", 0.5),
                           ("A  Ours: temporal hold-out 2025, ALERT threshold", mdl.alert_threshold_),
                           ("A  Ours: temporal hold-out 2025, EARLY-WARNING threshold", mdl.threshold_)]:
            rows.append(_row(label, fname, len(train_t), len(test_t), base_rate, _metrics(test_t[LABEL], p, thr),
                             f"unseen season, natural fire rate; threshold {thr:.3f}"))

        # B - random split, natural rate
        tr, te = train_test_split(df, test_size=0.2, stratify=df[LABEL], random_state=SEED)
        m = _clf().fit(tr[cols], tr[LABEL])
        rows.append(_row("B  Random 80/20 split, natural fire rate", fname, len(tr), len(te), te[LABEL].mean(),
                         _metrics(te[LABEL], m.predict_proba(te[cols])[:, 1]),
                         "same fires/zones/adjacent days on both sides; 'predict no fire' alone scores "
                         f"{100 * (1 - te[LABEL].mean()):.1f}%"))

        # C - random split, both sides balanced
        trb, teb = _balance(tr, rng), _balance(te, rng)
        mb = _clf().fit(trb[cols], trb[LABEL])
        rows.append(_row("C  Random 80/20 split, balanced 1:1", fname, len(trb), len(teb), teb[LABEL].mean(),
                         _metrics(teb[LABEL], mb.predict_proba(teb[cols])[:, 1]),
                         "test set also balanced, so chance = 50% and easy negatives are discarded"))

        # D - Algerian-sized balanced sample, repeated
        accs, aucs, f1s, recs, precs = [], [], [], [], []
        for i in range(20):
            small = _balance(df, rng, n_pos=138, n_neg=106)
            s_tr, s_te = train_test_split(small, test_size=0.3, stratify=small[LABEL], random_state=i)
            ms = _clf().set_params(n_estimators=100, max_depth=4).fit(s_tr[cols], s_tr[LABEL])
            mm = _metrics(s_te[LABEL], ms.predict_proba(s_te[cols])[:, 1])
            accs.append(mm["accuracy"]); aucs.append(mm["auc"]); f1s.append(mm["f1"])
            recs.append(mm["recall"]); precs.append(mm["precision"])
        rows.append(_row("D  244-row balanced sample, random 70/30 (20 repeats)", fname, 170, 74, 138 / 244,
                         {"accuracy": np.mean(accs), "precision": np.mean(precs), "recall": np.mean(recs),
                          "f1": np.mean(f1s), "auc": np.mean(aucs)},
                         "tiny sample drawn from many days; scores swing with the draw", sd=np.std(accs)))

        # E - balanced-trained model meets the real 2025 season
        trb_t = _balance(train_t, rng)
        me = _clf().fit(trb_t[cols], trb_t[LABEL])
        rows.append(_row("E  Balanced-trained (2023-24), tested on real 2025", fname, len(trb_t), len(test_t),
                         base_rate, _metrics(test_t[LABEL], me.predict_proba(test_t[cols])[:, 1]),
                         "same trees as a 'balanced' paper model, but scored on real-world class balance"))
    return pd.DataFrame(rows)


def main():
    df = load_real_dataset()
    out = run(df)
    path = DATA_PROCESSED_DIR / "literature_protocol_comparison.csv"
    out.to_csv(path, index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print(out.drop(columns=["note"]).to_string(index=False))
    print(f"\nSaved {path}")
    return out


if __name__ == "__main__":
    main()
