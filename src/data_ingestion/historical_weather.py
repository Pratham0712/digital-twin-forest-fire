"""
HistoricalWeatherClient - free historical daily weather via Meteostat, used to
build REAL features to pair with the real FIRMS fire detections pulled by
historical_firms.py. OpenWeatherMap's free tier has no historical endpoint;
Meteostat pulls from public weather station networks with no API key needed.

IMPORTANT: unlike older Meteostat tutorials (1.x API), the installed 2.x API
does NOT auto-interpolate arbitrary lat/lon points - a bare Point object
queried directly against the daily provider fails with "station not found".
Instead we look up the nearest REAL weather stations via stations.nearby()
and pull data from those directly, trying progressively farther stations
until one returns data for the requested date range (rural/hilly regions
like the Western Ghats have sparser station coverage than cities).
"""
import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from meteostat import Point, daily, stations
from meteostat.enumerations import Parameter, Provider

logger = logging.getLogger(__name__)

# Plain daily bulk files do NOT include relative humidity - it's only
# available by aggregating hourly station data (DAILY_DERIVED provider).
# Humidity is essential for FFMC/FWI, so we request both providers and let
# meteostat merge: DAILY for the fast bulk fields, DAILY_DERIVED to fill in
# RHUM from hourly aggregation.
REQUIRED_PARAMETERS = [
    Parameter.TEMP, Parameter.RHUM, Parameter.PRCP,
    Parameter.WSPD, Parameter.WDIR, Parameter.PRES,
]
PROVIDERS = [Provider.DAILY, Provider.DAILY_DERIVED]


def find_nearby_stations(lat: float, lon: float, radius_m: int = 150000, limit: int = 8) -> pd.DataFrame:
    """Returns real station IDs near (lat, lon), ordered by distance."""
    pt = Point(lat, lon)
    return stations.nearby(pt, radius=radius_m, limit=limit)


def fetch_point_weather(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    """
    Fetches daily weather for the nearest real station(s) to (lat, lon),
    trying up to 8 progressively farther stations until one has data for
    the requested range.
    """
    try:
        nearby = find_nearby_stations(lat, lon)
    except Exception as exc:
        logger.error("Station lookup failed for (%.2f, %.2f): %s", lat, lon, exc)
        return pd.DataFrame()

    if nearby.empty:
        logger.warning("No weather stations found within radius of (%.2f, %.2f)", lat, lon)
        return pd.DataFrame()

    for station_id in nearby.index:
        try:
            ts = daily(station_id, start, end, parameters=REQUIRED_PARAMETERS, providers=PROVIDERS)
            df = ts.fetch()
            if df is not None and not df.empty:
                df = df.reset_index().rename(columns={"time": "date"})
                df["latitude"] = lat
                df["longitude"] = lon
                df["station_id"] = station_id
                has_humidity = "rhum" in df.columns and df["rhum"].notna().any()
                logger.info("  -> got %d rows from station %s (humidity: %s)",
                            len(df), station_id, "yes" if has_humidity else "MISSING")
                return df
        except Exception as exc:
            logger.debug("Station %s failed for (%.2f, %.2f): %s", station_id, lat, lon, exc)
            continue

    logger.warning("No data from any nearby station for (%.2f, %.2f)", lat, lon)
    return pd.DataFrame()


def build_weather_query_points(region_bounds: dict, n_per_side: int = 4) -> pd.DataFrame:
    """
    Builds a sparse set of representative points across the region to find
    nearby stations for. n_per_side=4 gives a 4x4=16 point grid, matching the
    coarse-grid approach already used for live weather (WeatherClient).
    """
    lats = np.linspace(region_bounds["min_lat"], region_bounds["max_lat"], n_per_side)
    lons = np.linspace(region_bounds["min_lon"], region_bounds["max_lon"], n_per_side)
    points = [{"latitude": float(lat), "longitude": float(lon)} for lat in lats for lon in lons]
    return pd.DataFrame(points)


def fetch_region_historical_weather(region_bounds: dict, start: date, end: date,
                                     n_per_side: int = 4) -> pd.DataFrame:
    """
    Fetches historical daily weather for every point in a sparse regional
    grid over [start, end]. Returns one row per (point, date).
    """
    points = build_weather_query_points(region_bounds, n_per_side)
    frames = []
    for i, pt in points.iterrows():
        logger.info("Fetching historical weather %d/%d: (%.2f, %.2f)",
                    i + 1, len(points), pt["latitude"], pt["longitude"])
        df = fetch_point_weather(pt["latitude"], pt["longitude"], start, end)
        if not df.empty:
            frames.append(df)
    if not frames:
        logger.warning("No historical weather data retrieved for any point.")
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    logger.info("Historical weather fetch complete: %d rows across %d points", len(combined), len(points))
    return combined


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from config.config import REGION, DATA_RAW_DIR

    logging.basicConfig(level=logging.INFO)

    region_bounds = {
        "min_lat": REGION.min_lat, "max_lat": REGION.max_lat,
        "min_lon": REGION.min_lon, "max_lon": REGION.max_lon,
    }

    # Matches the fire-season window used for historical_firms.py
    all_frames = []
    for year in [2023, 2024, 2025]:
        start = date(year, 1, 1)
        end = date(year, 5, 31)
        logger.info("=== Fetching historical weather for %d (%s to %s) ===", year, start, end)
        df = fetch_region_historical_weather(region_bounds, start, end)
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        print("No historical weather retrieved. Check your internet connection and that "
              "meteostat is installed correctly (pip show meteostat).")
    else:
        combined = pd.concat(all_frames, ignore_index=True)
        out_path = DATA_RAW_DIR / "historical_weather_karnataka.csv"
        combined.to_csv(out_path, index=False)
        print(f"\nSaved {len(combined)} historical weather rows to {out_path}")
        print(combined.head(10))
