"""
validate_spread_hybrid.py - does the physics-based spread automaton add skill on
top of a data-driven ranker?  (src/simulation/event_hybrid.py)

    python scripts/validate_spread_hybrid.py
    python scripts/validate_spread_hybrid.py --events 600 --quick

Gradient-boosted rankers with nested feature sets (distance -> + fire geometry and
the observed growth direction -> + weather -> + automaton arrival times) are
trained on 2023-24 fire events and scored on the held-out 2025 and 2026 seasons
by per-event ROC-AUC over the candidate cells around each fire. Paired bootstrap
confidence intervals compare each model with the distance baseline and with the
model one step below it.

Writes data/processed/eval_spread_hybrid.csv, eval_spread_hybrid_importance.csv.
"""
import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, MODELS_DIR  # noqa: E402
from src.simulation import event_hybrid as eh  # noqa: E402
from src.simulation import event_validation as ev  # noqa: E402

logging.getLogger().setLevel(logging.WARNING)


def _boot(vals, n=4000, seed=0):
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 5:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = rng.choice(vals, (n, len(vals))).mean(axis=1)
    return float(vals.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", type=int, default=800, help="events per held-out season (train gets 2x)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--raw", default=str(DATA_RAW_DIR))
    ap.add_argument("--processed", default=str(DATA_PROCESSED_DIR))
    ap.add_argument("--out", default=str(DATA_PROCESSED_DIR))
    a = ap.parse_args()
    n_ev = 60 if a.quick else a.events
    raw, proc, out = Path(a.raw), Path(a.processed), Path(a.out)

    static = pd.concat([pd.read_csv(MODELS_DIR / f) for f in ("static_heat_sources.csv", "static_heat_sources_india.csv")
                        if (MODELS_DIR / f).exists()], ignore_index=True)
    files = sorted(raw.glob("historical_fires_india_*.csv")) + [raw / "historical_fires_karnataka.csv"]
    px = ev.load_pixels([f for f in files if f.exists()], static)
    cols = ["latitude", "longitude", "date", "ffmc", "bui", "wx_wind_speed_ms"]
    wx = ev.WeatherLookup([pd.read_csv(proc / "real_training_data.csv", usecols=cols),
                           pd.read_csv(proc / "india_states_training_data.csv", usecols=cols)])
    seasons = {"train": ("2023-01-01", "2025-01-01", 2 * n_ev), "2025": ("2025-01-01", "2026-01-01", n_ev),
               "2026": ("2026-01-01", "2027-01-01", n_ev)}
    tables = {}
    for k, (s, e, n) in seasons.items():
        t0 = time.time()
        d = px["date"]
        dates = sorted(d[(d >= pd.Timestamp(s)) & (d < pd.Timestamp(e))].unique())
        events = ev.build_events(px, wx, dates, max_events=n, seed=3)
        tables[k] = eh.build_table(events, key0={"train": 0, "2025": 100000, "2026": 200000}[k])
        print(f"{k}: {len(events)} events -> {len(tables[k])} with growth next day "
              f"({sum(len(t) for t in tables[k]):,} candidate cells) [{time.time()-t0:.0f}s]")

    train = eh.train_sample(tables["train"])
    print(f"\nTraining on {len(train):,} sampled cells ({int(train['y'].sum()):,} grew)")
    models = eh.fit_models(train)

    rows, imp_rows = [], []
    for k in ("2025", "2026"):
        res = eh.score_events(tables[k], models)
        res.to_csv(out / f"eval_spread_hybrid_{k}_detail.csv", index=False)
        names = ["Distance to burning cells", "Rate-of-spread automaton alone"] + list(eh.FEATURE_SETS)
        for subset, mask in (("all events", np.ones(len(res), bool)),
                             ("events with an observed growth direction", res["has_dir"].values == 1)):
            sub = res[mask]
            for nm in names:
                m, lo, hi = _boot(sub[nm].values)
                rows.append({"season": k, "events": subset, "n": int(len(sub)), "ranking": nm,
                             "mean_auc": round(m, 4), "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                             "paired_diff_vs": "", "diff": np.nan, "diff_lo": np.nan, "diff_hi": np.nan})
            pairs = [(names[i], names[0]) for i in range(1, len(names))]
            pairs += [(names[i], names[i - 1]) for i in range(3, len(names))]
            for hi_name, lo_name in pairs:
                m, lo, hi = _boot((sub[hi_name] - sub[lo_name]).values)
                rows.append({"season": k, "events": subset, "n": int(len(sub)), "ranking": hi_name,
                             "mean_auc": np.nan, "ci95_low": np.nan, "ci95_high": np.nan,
                             "paired_diff_vs": lo_name, "diff": round(m, 4), "diff_lo": round(lo, 4),
                             "diff_hi": round(hi, 4)})
    # permutation importance of the full model on 2025
    from sklearn.inspection import permutation_importance
    full = list(eh.FEATURE_SETS)[-1]
    cols_full = eh.FEATURE_SETS[full]
    test = pd.concat(tables["2025"], ignore_index=True)
    smp = test.groupby("event", group_keys=False).apply(lambda g: g.sample(min(len(g), 300), random_state=0))
    pi = permutation_importance(models[full], smp[cols_full], smp["y"], scoring="roc_auc", n_repeats=5,
                                random_state=0)
    for c, v in sorted(zip(cols_full, pi.importances_mean), key=lambda z: -z[1]):
        imp_rows.append({"feature": c, "auc_drop_when_shuffled": round(float(v), 4)})
    res_df, imp_df = pd.DataFrame(rows), pd.DataFrame(imp_rows)
    pd.set_option("display.width", 220)
    print("\nPer-event ranking AUC on held-out seasons (0.5 = random):")
    print(res_df[res_df["paired_diff_vs"] == ""][["season", "events", "n", "ranking", "mean_auc", "ci95_low",
                                                   "ci95_high"]].to_string(index=False))
    print("\nPaired differences (bootstrap 95% CI):")
    print(res_df[res_df["paired_diff_vs"] != ""][["season", "events", "n", "ranking", "paired_diff_vs", "diff",
                                                   "diff_lo", "diff_hi"]].to_string(index=False))
    print("\nPermutation importance (full model, 2025):")
    print(imp_df.to_string(index=False))
    res_df.to_csv(out / "eval_spread_hybrid.csv", index=False)
    imp_df.to_csv(out / "eval_spread_hybrid_importance.csv", index=False)


if __name__ == "__main__":
    main()
