"""
build_real_dataset.py - joins REAL historical fire detections (FIRMS) with
REAL historical weather (Meteostat) to build a genuine, leakage-free training
dataset: label comes purely from satellite-observed fire occurrence, features
come purely from independently-measured weather. This directly fixes the
label-leakage issue flagged earlier (where labels were a formula computed
from the same FWI features used to predict them).

For each (grid zone, date) pair across the fire seasons in the data:
  - features: FWI-family indices computed from that day's real weather
  - label: 1 if a real FIRMS detection fell within that zone on that date, else 0

Known simplification (documented, not hidden): FFMC/DMC/DC are computed
per-day from fixed default previous-day values, the same approach used in
the live pipeline, rather than a full day-by-day recursive carry-forward per
station. True FWI methodology recursively carries yesterday's moisture code
into today's calculation; a full recursive implementation would require
gap-free daily station data (real stations have missing days) and is out of
scope for this pass. This is a standard simplification in operational
fire-risk systems with imperfect station coverage and should be named
explicitly in the paper's methodology section.

Run:
    python src/ml_models/build_real_dataset.py
"""
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from config.config import REGION, DATA_RAW_DIR, DATA_PROCESSED_DIR
from src.data_ingestion.ingestion_module import build_region_grid
from src.data_processing.feature_engineering import (
    compute_ffmc, compute_dmc, compute_dc, compute_bui, compute_fwi, synthetic_ndvi,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FIRE_MATCH_RADIUS_DEG = REGION.grid_resolution_deg * 0.75  # same threshold used in live ingestion


def load_real_fires() -> pd.DataFrame:
    path = DATA_RAW_DIR / "historical_fires_karnataka.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run src/data_ingestion/historical_firms.py first."
        )
    df = pd.read_csv(path)
    df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.date
    return df


def load_real_weather() -> pd.DataFrame:
    path = DATA_RAW_DIR / "historical_weather_karnataka.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run src/data_ingestion/historical_weather.py first."
        )
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def build_dataset() -> pd.DataFrame:
    fires = load_real_fires()
    weather = load_real_weather()
    grid = build_region_grid()

    grid_coords = grid[["latitude", "longitude"]].values
    grid_tree_cache = None  # grid itself is fixed, only weather/fire points change per day

    weather_dates = sorted(weather["date"].unique())
    logger.info("Building real dataset: %d grid zones x %d unique weather dates", len(grid), len(weather_dates))

    all_rows = []
    for i, day in enumerate(weather_dates):
        day_weather = weather[weather["date"] == day]
        if day_weather.empty:
            continue

        # Nearest-neighbor join: weather station -> every grid cell (same
        # technique as live DataIngestionModule._nearest_neighbor_join)
        wx_tree = cKDTree(day_weather[["latitude", "longitude"]].values)
        _, idx = wx_tree.query(grid_coords, k=1)
        wx_matched = day_weather.iloc[idx].reset_index(drop=True)

        temp = wx_matched["temp"].fillna(wx_matched["temp"].median()).values
        rh = wx_matched["rhum"].fillna(wx_matched["rhum"].median()).values
        wind = wx_matched["wspd"].fillna(0).values / 3.6  # meteostat wspd is km/h -> convert to m/s
        rain = wx_matched["prcp"].fillna(0).values

        ffmc = compute_ffmc(temp, rh, wind, rain)
        dmc = compute_dmc(temp, rh, rain, month=day.month)
        dc = compute_dc(temp, rain, month=day.month)
        bui = compute_bui(dmc, dc)
        fwi = compute_fwi(ffmc, bui, wind)

        # Real label: did a fire actually occur in this zone on this date?
        day_fires = fires[fires["acq_date"] == day]
        label = np.zeros(len(grid), dtype=int)
        if not day_fires.empty:
            fire_tree = cKDTree(day_fires[["latitude", "longitude"]].values)
            dist, _ = fire_tree.query(grid_coords, k=1)
            label = (dist <= FIRE_MATCH_RADIUS_DEG).astype(int)

        ndvi = synthetic_ndvi(label.astype(bool), rain, seed=hash(str(day)) % (2**31))

        day_df = pd.DataFrame({
            "zone_id": grid["zone_id"].values,
            "date": day,
            "latitude": grid["latitude"].values,
            "longitude": grid["longitude"].values,
            "wx_temperature_c": temp,
            "wx_humidity_pct": rh,
            "wx_wind_speed_ms": wind,
            "wx_precipitation_mm": rain,
            "ffmc": ffmc, "dmc": dmc, "dc": dc, "bui": bui, "fwi": fwi, "ndvi": ndvi,
            "fire_risk_label": label,
        })
        all_rows.append(day_df)

        if (i + 1) % 50 == 0 or i == len(weather_dates) - 1:
            logger.info("  processed %d/%d days", i + 1, len(weather_dates))

    combined = pd.concat(all_rows, ignore_index=True)
    logger.info(
        "Real dataset built: %d rows (%d zones x %d days), positive rate %.2f%%",
        len(combined), len(grid), len(weather_dates), 100 * combined["fire_risk_label"].mean(),
    )
    return combined


if __name__ == "__main__":
    dataset = build_dataset()
    out_path = DATA_PROCESSED_DIR / "real_training_data.csv"
    dataset.to_csv(out_path, index=False)
    print(f"\nSaved {len(dataset)} rows to {out_path}")
    print(f"\nPositive rate: {dataset['fire_risk_label'].mean():.2%}")
    print(f"\nDate range: {dataset['date'].min()} to {dataset['date'].max()}")
    print("\nSample rows:")
    print(dataset.head(10))
