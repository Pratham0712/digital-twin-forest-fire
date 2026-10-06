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
from src.ml_models.fire_history import (
    FIRE_MATCH_RADIUS_FACTOR, cell_flags_for_detections, confident_detections,
)
from src.data_processing.feature_engineering import (
    compute_ffmc, compute_dmc, compute_dc, compute_bui, compute_fwi, synthetic_ndvi_independent,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FIRE_MATCH_RADIUS_DEG = REGION.grid_resolution_deg * FIRE_MATCH_RADIUS_FACTOR  # tightened: 1.5x was too wide and created false-positive labels in adjacent cells; 0.7x keeps only cells where the hotspot centroid is genuinely inside the cell


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
    # format="mixed": the file may hold plain dates and "2026-01-01 00:00:00" timestamps side by side
    df["date"] = pd.to_datetime(df["date"], format="mixed").dt.date
    return df


def build_dataset_for_region(region, fires: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (grid zone, weather date) for `region`: real weather-derived
    FWI features + a real FIRMS label. Region-agnostic, so the same code
    builds the Karnataka set and the multi-state India set
    (scripts/build_india_regions_dataset.py).
    """
    grid = build_region_grid(region)
    grid_coords = grid[["latitude", "longitude"]].values

    weather_dates = sorted(weather["date"].unique())
    logger.info("Building real dataset for %s: %d grid zones x %d unique weather dates",
                region.name, len(grid), len(weather_dates))

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

        # Plain float64 ndarrays: multi-state weather can arrive as pandas
        # nullable Float64 (pd.NA), which numpy-style code (.clip, ufuncs)
        # cannot handle.
        def _f(col):
            return pd.to_numeric(wx_matched[col], errors="coerce").to_numpy(
                dtype="float64", na_value=np.nan)

        temp, rh, wspd, rain = _f("temp"), _f("rhum"), _f("wspd"), _f("prcp")
        temp = np.where(np.isnan(temp), np.nanmedian(temp) if np.isfinite(temp).any() else 25.0, temp)
        rh = np.where(np.isnan(rh), np.nanmedian(rh) if np.isfinite(rh).any() else 50.0, rh)
        wind = np.nan_to_num(wspd, nan=0.0) / 3.6  # meteostat wspd is km/h -> m/s
        rain = np.nan_to_num(rain, nan=0.0)

        ffmc = compute_ffmc(temp, rh, wind, rain)
        dmc = compute_dmc(temp, rh, rain, month=day.month)
        dc = compute_dc(temp, rain, month=day.month)
        bui = compute_bui(dmc, dc)
        fwi = compute_fwi(ffmc, bui, wind)

        # Real label: did a confident VEGETATION fire occur in this zone on
        # this date? VIIRS nominal/high only (low-confidence detections are
        # often cloud edges), and FIRMS type 0 only - static industrial heat
        # sources (type 2, e.g. the Ballari steel belt) are not forest fires.
        day_fires = confident_detections(fires[fires["acq_date"] == day])
        label = cell_flags_for_detections(
            grid_coords, day_fires[["latitude", "longitude"]].values.astype(float),
            region.grid_resolution_deg).astype(int)

        ndvi = synthetic_ndvi_independent(grid["zone_id"].values, rain)

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
        "Real dataset built for %s: %d rows (%d zones x %d days), positive rate %.2f%%",
        region.name, len(combined), len(grid), len(weather_dates),
        100 * combined["fire_risk_label"].mean(),
    )
    return combined


def build_dataset() -> pd.DataFrame:
    return build_dataset_for_region(REGION, load_real_fires(), load_real_weather())


if __name__ == "__main__":
    from config.config import MODELS_DIR
    from src.ml_models.fire_history import StaticSourceMask
    StaticSourceMask.from_archive(pd.read_csv(DATA_RAW_DIR / "historical_fires_karnataka.csv")).save(MODELS_DIR)
    dataset = build_dataset()
    out_path = DATA_PROCESSED_DIR / "real_training_data.csv"
    dataset.to_csv(out_path, index=False)
    print(f"\nSaved {len(dataset)} rows to {out_path}")
    print(f"\nPositive rate: {dataset['fire_risk_label'].mean():.2%}")
    print(f"\nDate range: {dataset['date'].min()} to {dataset['date'].max()}")
    print("\nSample rows:")
    print(dataset.head(10))
