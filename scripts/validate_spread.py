"""
validate_spread.py - back-test the fire-spread simulator against real satellite
detections (Karnataka, 0.1 degree grid).

    python scripts/validate_spread.py

For sampled days of the held-out 2025 season (and 2026 if added): seed the
automaton with that day's real detections, drive it with that day's real dryness,
build-up index, wind speed and real terrain, and score its predicted new fire
cells against the cells that really had NEW detections the next day. Two
horizons (2 h = what the app shows, 24 h = the satellite's daily window) and
two reference predictors ("spread to every neighbour", "random cells, same
count") are reported so the numbers can be read.

It also sweeps the automaton's base spread probability on TRAINING-season days
only (2023-24), then tests the chosen value on the held-out season, so any
tuning never sees the test data.

Writes data/processed/eval_spread_validation.csv and eval_spread_calibration.csv.
Takes a few minutes.
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config.config import DATA_PROCESSED_DIR, MODELS_DIR, REGION  # noqa: E402
from src.data_ingestion.ingestion_module import build_region_grid  # noqa: E402
from src.data_ingestion.terrain import TerrainTable  # noqa: E402
from src.ml_models.train_real import SECOND_TEST_START, TEST_SEASON_START  # noqa: E402
from src.simulation import spread_validation as sv  # noqa: E402

logging.getLogger().setLevel(logging.WARNING)   # the simulator logs every run at INFO
DEFAULT_PROB = 0.35     # the value the live system uses


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=40, help="days sampled per season (default 40)")
    ap.add_argument("--replicates", type=int, default=10, help="stochastic runs per day (default 10)")
    a = ap.parse_args()

    path = DATA_PROCESSED_DIR / "real_training_data.csv"
    if not path.exists():
        sys.exit(f"{path} not found - run src/ml_models/build_real_dataset.py first.")
    df = pd.read_csv(path, usecols=["date", "zone_id", "fire_risk_label", "ffmc", "bui", "wx_wind_speed_ms"])
    df["date"] = pd.to_datetime(df["date"])

    grid = build_region_grid(REGION)
    n_rows, n_cols = int(grid["row"].max()) + 1, int(grid["col"].max()) + 1
    terrain = TerrainTable.load(MODELS_DIR, REGION.name)
    elevation = None
    if terrain.available:
        elev = np.zeros((n_rows, n_cols))
        t = terrain.transform(grid)["elev_m"].fillna(0).values
        elev[grid["row"].values.astype(int), grid["col"].values.astype(int)] = t
        elevation = elev
        print("Using real terrain for slope.")
    else:
        print("No terrain table - spread runs on flat ground (run scripts/build_terrain.py).")
    res_deg = REGION.grid_resolution_deg

    train = df[df["date"] < TEST_SEASON_START]
    seasons = {"2025": df[(df["date"] >= TEST_SEASON_START) & (df["date"] < SECOND_TEST_START)],
               "2026": df[df["date"] >= SECOND_TEST_START]}

    print("\nStep 1/2: choosing the base spread probability on 2023-24 days only ...")
    cal_days = sv.eligible_days(train, max_days=min(a.days, 30))
    cal = sv.calibrate(train, grid, elevation, res_deg, cal_days, replicates=max(4, a.replicates // 2))
    print(cal.to_string(index=False))
    best = float(cal.sort_values("f1", ascending=False).iloc[0]["base_spread_prob"])
    print(f"Best on training days: {best}   (live system uses {DEFAULT_PROB})")
    cal.to_csv(DATA_PROCESSED_DIR / "eval_spread_calibration.csv", index=False)

    print("\nStep 2/2: scoring on held-out seasons ...")
    frames = []
    for season, sdf in seasons.items():
        if sdf.empty:
            continue
        days = sv.eligible_days(sdf, max_days=a.days)
        for label, p in (("live setting", DEFAULT_PROB), ("calibrated on 2023-24", best)):
            if label != "live setting" and abs(p - DEFAULT_PROB) < 1e-9:
                continue
            r = sv.backtest(sdf, grid, elevation, res_deg, days, p, replicates=a.replicates)
            r.insert(0, "setting", label)
            r.insert(0, "season", season)
            frames.append(r)
            ctx = r.attrs["context"]
            print(f"\n--- {season}, {label} (base_spread_prob={p}) : {ctx['days']} days, "
                  f"{ctx['new_cells']} new fire cells next day; {ctx['share_of_next_day_fires_on_seed_cells']:.0%} "
                  f"of next-day detections were on cells already burning ---")
            print(r.drop(columns=["season", "setting", "base_spread_prob"]).to_string(index=False))
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(DATA_PROCESSED_DIR / "eval_spread_validation.csv", index=False)
        print(f"\nSaved {DATA_PROCESSED_DIR / 'eval_spread_validation.csv'}")


if __name__ == "__main__":
    main()
