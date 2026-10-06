"""
add_season.py - add another fire season (default 2026, Jan-May) to the
Karnataka data as a SECOND, untouched hold-out test season.

It downloads the season's NASA FIRMS detections and Meteostat weather, appends
them to data/raw/historical_fires_karnataka.csv and historical_weather_karnataka.csv,
and rebuilds data/processed/real_training_data.csv. Training never uses 2026:
train_real.py trains on 2023-24, reports 2025 as the headline test, and reports
2026 separately (model_comparison_2026.csv).

    python scripts/add_season.py --year 2026
    python scripts/add_season.py --year 2026 --rebuild-only     # data already downloaded

Needs internet and FIRMS_MAP_KEY in .env. Safe to re-run: the year's old rows
are replaced, not duplicated. Afterwards:

    python src/ml_models/train_real.py                 # 2026 comparison table
    python scripts/run_evaluation_extras.py            # lift / PR / baselines per season

For the other states:  python scripts/build_india_regions_dataset.py --train --years 2023 2024 2025 2026
"""
import argparse
import logging
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from config.config import API, DATA_PROCESSED_DIR, DATA_RAW_DIR, MODELS_DIR, REGION  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("add_season")

FIRES = DATA_RAW_DIR / "historical_fires_karnataka.csv"
WEATHER = DATA_RAW_DIR / "historical_weather_karnataka.csv"


def replace_year(path: Path, new: pd.DataFrame, date_col: str, year: int) -> int:
    """Write `new` into `path`, dropping any existing rows of that year first."""
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(old):
        keep = pd.to_datetime(old[date_col], format="mixed").dt.year != year
        old = old[keep]
    out = pd.concat([old, new], ignore_index=True)
    # one date format in the file (new weather rows carry a 00:00:00 time part)
    out[date_col] = pd.to_datetime(out[date_col], format="mixed").dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)
    return len(new)


def download(year: int):
    from src.data_ingestion.historical_firms import fetch_karnataka_fire_seasons
    from src.data_ingestion.historical_weather import fetch_region_historical_weather

    bounds = {"min_lat": REGION.min_lat, "max_lat": REGION.max_lat,
              "min_lon": REGION.min_lon, "max_lon": REGION.max_lon}
    if not API.firms_map_key:
        sys.exit("FIRMS_MAP_KEY is not set in .env - needed to download fire history.")

    fires = fetch_karnataka_fire_seasons(API.firms_map_key, [year], bounds)
    if fires.empty:
        sys.exit(f"No FIRMS detections returned for {year}. The archive may not cover that season yet.")
    n = replace_year(FIRES, fires, "acq_date", year)
    logger.info("Fires %d: saved %d detections", year, n)

    weather = fetch_region_historical_weather(bounds, date(year, 1, 1), date(year, 5, 31))
    if weather.empty:
        sys.exit(f"No weather retrieved for {year}. Fires were saved; re-run to retry the weather.")
    n = replace_year(WEATHER, weather, "date", year)
    logger.info("Weather %d: saved %d rows", year, n)


def rebuild():
    from src.ml_models.build_real_dataset import build_dataset
    from src.ml_models.fire_history import StaticSourceMask
    StaticSourceMask.from_archive(pd.read_csv(FIRES)).save(MODELS_DIR)
    data = build_dataset()
    out = DATA_PROCESSED_DIR / "real_training_data.csv"
    data.to_csv(out, index=False)
    years = pd.to_datetime(data["date"]).dt.year.value_counts().sort_index()
    logger.info("Saved %d rows to %s\nRows per season:\n%s", len(data), out, years.to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--rebuild-only", action="store_true", help="skip downloading, just rebuild the dataset")
    a = ap.parse_args()
    if not a.rebuild_only:
        download(a.year)
    rebuild()


if __name__ == "__main__":
    main()
