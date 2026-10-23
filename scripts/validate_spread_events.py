"""
validate_spread_events.py - event-level back-test of the rate-of-spread
automaton against VIIRS detections (375 m), with yesterday-to-tomorrow data
assimilation.  See src/simulation/event_validation.py for the design.

    python scripts/validate_spread_events.py
    python scripts/validate_spread_events.py --events 300 --quick

Calibration (spread duration, dilation radius) uses 2023-24 events only. 2025 and
2026 are held out. Writes data/processed/eval_spread_events.csv (pooled F1 per
method), eval_spread_events_auc.csv (per-season ranking AUC) and
eval_spread_events_calibration.csv.
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
from src.simulation import event_validation as ev  # noqa: E402

logging.getLogger().setLevel(logging.WARNING)


def _season_dates(px, start, end):
    d = px["date"]
    return sorted(d[(d >= start) & (d < end)].unique())


def _boot_ci(vals, n=2000, seed=0):
    vals = np.asarray([v for v in vals if v is not None and v == v], float)
    if len(vals) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = [rng.choice(vals, len(vals)).mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", type=int, default=400, help="events sampled per season (default 400)")
    ap.add_argument("--quick", action="store_true", help="small run for a smoke test")
    ap.add_argument("--raw", default=str(DATA_RAW_DIR))
    ap.add_argument("--processed", default=str(DATA_PROCESSED_DIR))
    ap.add_argument("--out", default=str(DATA_PROCESSED_DIR))
    a = ap.parse_args()
    n_ev = 40 if a.quick else a.events
    raw, proc, out = Path(a.raw), Path(a.processed), Path(a.out)

    static = pd.concat([pd.read_csv(MODELS_DIR / f) for f in ("static_heat_sources.csv", "static_heat_sources_india.csv")
                        if (MODELS_DIR / f).exists()], ignore_index=True)
    files = sorted(raw.glob("historical_fires_india_*.csv")) + [raw / "historical_fires_karnataka.csv"]
    px = ev.load_pixels([f for f in files if f.exists()], static)
    print(f"{len(px):,} vegetation-fire pixels, {px['date'].min().date()} to {px['date'].max().date()}")

    cols = ["latitude", "longitude", "date", "ffmc", "bui", "wx_wind_speed_ms"]
    ka = pd.read_csv(proc / "real_training_data.csv", usecols=cols)
    ind = pd.read_csv(proc / "india_states_training_data.csv", usecols=cols)
    wx = ev.WeatherLookup([ka, ind])

    seasons = {"train 2023-24": ("2023-01-01", "2025-01-01"),
               "2025": ("2025-01-01", "2026-01-01"), "2026": ("2026-01-01", "2027-01-01")}
    events = {}
    for k, (s, e) in seasons.items():
        t0 = time.time()
        events[k] = ev.build_events(px, wx, _season_dates(px, pd.Timestamp(s), pd.Timestamp(e)),
                                    max_events=n_ev * (2 if k.startswith("train") else 1), seed=1)
        n_with_truth = sum(1 for x in events[k] if (x.truth & ~x.seeds).any())
        n_prev = sum(1 for x in events[k] if x.prev is not None)
        print(f"{k}: {len(events[k])} events ({n_with_truth} with new cells next day, {n_prev} with a previous "
              f"day) [{time.time()-t0:.0f}s]")

    print("\nCalibrating on 2023-24 events only ...")
    tau, rad, cal = ev.calibrate(events["train 2023-24"])
    print(cal.to_string(index=False))
    print(f"Chosen: CA duration {tau} h, dilation radius {rad} cells ({rad*ev.CELL_KM:.1f} km)")
    cal.to_csv(out / "eval_spread_events_calibration.csv", index=False)

    pooled_rows, auc_rows = [], []
    for k in ("2025", "2026"):
        t0 = time.time()
        df, pooled = ev.evaluate(events[k], tau_global=tau, r_global=rad)
        print(f"\n=== {k}: {len(df)} events scored [{time.time()-t0:.0f}s] ===")
        for name, (tp, npred, nt, n) in pooled.items():
            pr = tp / npred if npred else 0.0
            rc = tp / nt if nt else 0.0
            pooled_rows.append({"season": k, "method": name, "events": n, "precision": round(pr, 4),
                                "recall": round(rc, 4), "f1": round(ev.f1_from(tp, npred, nt), 4)})
        for col, label in (("auc_dist", "Distance to burning cells (no physics)"),
                           ("auc_ca_iso", "ROS automaton, isotropic"),
                           ("auc_ca_dir", "ROS automaton, observed direction")):
            if col not in df:
                continue
            v = df[col].dropna()
            lo, hi = _boot_ci(v.values)
            auc_rows.append({"season": k, "ranking": label, "events": len(v), "mean_auc": round(v.mean(), 4),
                             "ci95_low": round(lo, 4), "ci95_high": round(hi, 4)})
        # paired difference on the events where the direction could be read
        if "auc_ca_dir" in df:
            sub = df.dropna(subset=["auc_ca_dir", "auc_dist", "auc_ca_iso"])
            for c1, c2, lab in (("auc_ca_iso", "auc_dist", "isotropic automaton minus distance"),
                                ("auc_ca_dir", "auc_dist", "directed automaton minus distance"),
                                ("auc_ca_dir", "auc_ca_iso", "directed minus isotropic automaton")):
                dlt = (sub[c1] - sub[c2]).values
                lo, hi = _boot_ci(dlt)
                auc_rows.append({"season": k, "ranking": f"paired: {lab}", "events": len(dlt),
                                 "mean_auc": round(dlt.mean(), 4), "ci95_low": round(lo, 4),
                                 "ci95_high": round(hi, 4)})
        df.to_csv(out / f"eval_spread_events_{k}_detail.csv", index=False)

    pooled_df, auc_df = pd.DataFrame(pooled_rows), pd.DataFrame(auc_rows)
    print("\nPooled precision / recall / F1 of the predicted set (held-out):")
    print(pooled_df.to_string(index=False))
    print("\nRanking AUC (0.5 = random):")
    print(auc_df.to_string(index=False))
    pooled_df.to_csv(out / "eval_spread_events.csv", index=False)
    auc_df.to_csv(out / "eval_spread_events_auc.csv", index=False)


if __name__ == "__main__":
    main()
