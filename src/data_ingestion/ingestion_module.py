"""
DataIngestionModule - orchestrates FIRMSClient + WeatherClient and performs the
spatial nearest-neighbor join described in Report Ch.6 (Module Design) and
implements the 'DataIngestionModule fetches and validates incoming data' step
of Scenario 1 (SRS 6.2.4).
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config.config import API, REGION, SYSTEM
from src.data_ingestion.firms_client import FIRMSClient
from src.data_ingestion.weather_client import WeatherClient

logger = logging.getLogger(__name__)


def build_region_grid(region=REGION) -> pd.DataFrame:
    """
    Builds the fixed geographic grid over the study region (Karnataka / Western
    Ghats) that every other module (ML prediction, CA simulation, dashboard)
    indexes into. Each row is one grid cell centroid with a stable zone_id
    (e.g. 'G-17', matching the naming used in Report Scenario 1).
    """
    lats = np.arange(region.min_lat, region.max_lat, region.grid_resolution_deg)
    lons = np.arange(region.min_lon, region.max_lon, region.grid_resolution_deg)
    rows = []
    zone_idx = 0
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            rows.append({
                "zone_id": f"G-{zone_idx}",
                "row": i, "col": j,
                "latitude": lat + region.grid_resolution_deg / 2,
                "longitude": lon + region.grid_resolution_deg / 2,
            })
            zone_idx += 1
    grid = pd.DataFrame(rows)
    logger.info("Built region grid: %d cells (%d rows x %d cols)", len(grid), len(lats), len(lons))
    return grid


def build_weather_grid(region=REGION) -> pd.DataFrame:
    """
    Builds a much coarser grid purely for weather API calls. Weather fields
    (temperature, humidity, wind) vary smoothly over tens of kilometres, so
    querying at the same 11km resolution as the fire-detection grid wastes
    API quota for no accuracy gain - at 1,400 fire-grid cells this would burn
    through OpenWeatherMap's free-tier 1,000 calls/day limit in a single
    refresh. This grid is nearest-neighbor joined onto the fine grid in
    DataIngestionModule._nearest_neighbor_join, the same technique already
    used for fire hotspots.
    """
    lats = np.arange(region.min_lat, region.max_lat, region.weather_grid_resolution_deg)
    lons = np.arange(region.min_lon, region.max_lon, region.weather_grid_resolution_deg)
    rows = [{"latitude": lat + region.weather_grid_resolution_deg / 2,
             "longitude": lon + region.weather_grid_resolution_deg / 2}
            for lat in lats for lon in lons]
    grid = pd.DataFrame(rows)
    logger.info("Built weather grid: %d points (coarse, %.1f deg spacing)",
                len(grid), region.weather_grid_resolution_deg)
    return grid


class DataIngestionModule:
    """
    Public entry point for Layer 1 (Data Acquisition) of the five-layer
    architecture in Report Ch.6.1.1. Combines satellite hotspots and weather
    observations into one grid-indexed DataFrame ready for feature engineering.
    """

    def __init__(self, offline: bool = False):
        self.offline = offline
        self.firms = FIRMSClient(API.firms_map_key, API.firms_base_url, API.firms_source)
        self.weather = WeatherClient(API.owm_api_key, API.owm_base_url)
        self.grid = build_region_grid()
        self.weather_grid = build_weather_grid()

    def fetch_fire_hotspots(self) -> pd.DataFrame:
        if self.offline or not API.firms_map_key:
            return FIRMSClient.generate_sample(
                {"min_lat": REGION.min_lat, "max_lat": REGION.max_lat,
                 "min_lon": REGION.min_lon, "max_lon": REGION.max_lon},
                n_points=8,
            )
        return self.firms.fetch_hotspots(
            REGION.min_lat, REGION.min_lon, REGION.max_lat, REGION.max_lon,
            API.firms_day_range,
        )

    def fetch_weather(self) -> pd.DataFrame:
        grid_points = self.weather_grid[["latitude", "longitude"]].to_dict("records")
        if self.offline or not API.owm_api_key:
            return WeatherClient.generate_sample(grid_points)
        return self.weather.fetch_grid(grid_points)

    @staticmethod
    def _nearest_neighbor_join(grid: pd.DataFrame, points: pd.DataFrame,
                                value_cols: list, prefix: str,
                                max_distance_deg: Optional[float] = None) -> pd.DataFrame:
        """
        For every grid cell, attaches the value_cols of the nearest point in
        `points` using a KD-tree. Cells beyond max_distance_deg (if given) get
        NaN, so downstream code can distinguish 'no fire nearby' from
        'unknown'. This is the spatial join referenced in the project memory
        (Day 1: 'spatial nearest-neighbor join of fire/weather data').
        """
        out = grid.copy()
        for col in value_cols:
            out[f"{prefix}_{col}"] = np.nan
        out[f"{prefix}_distance_deg"] = np.nan

        if points.empty:
            return out

        tree = cKDTree(points[["latitude", "longitude"]].values)
        dist, idx = tree.query(grid[["latitude", "longitude"]].values, k=1)

        for col in value_cols:
            vals = points[col].values[idx]
            out[f"{prefix}_{col}"] = vals
        out[f"{prefix}_distance_deg"] = dist

        if max_distance_deg is not None:
            mask = dist > max_distance_deg
            for col in value_cols:
                out.loc[mask, f"{prefix}_{col}"] = np.nan
            out.loc[mask, f"{prefix}_distance_deg"] = np.nan

        return out

    def build_unified_frame(self) -> pd.DataFrame:
        """
        Fetches hotspots + weather and joins both onto the region grid,
        producing the single unified DataFrame that feature_engineering.py
        consumes. This mirrors 'Data flows ... into the processing pipeline'
        (Report 6.1.1).
        """
        hotspots = self.fetch_fire_hotspots()
        weather = self.fetch_weather()

        unified = self._nearest_neighbor_join(
            self.grid, weather,
            ["temperature_c", "humidity_pct", "wind_speed_ms", "wind_deg",
             "precipitation_mm", "clouds_pct"],
            prefix="wx",
        )

        if not hotspots.empty:
            unified = self._nearest_neighbor_join(
                unified, hotspots, ["frp", "confidence", "brightness"],
                prefix="fire", max_distance_deg=REGION.grid_resolution_deg * 0.75,
            )
            unified["active_fire_nearby"] = unified["fire_distance_deg"].notna()
        else:
            unified["fire_frp"] = np.nan
            unified["fire_confidence"] = np.nan
            unified["fire_brightness"] = np.nan
            unified["active_fire_nearby"] = False

        unified["ingested_at"] = pd.Timestamp.now(tz="UTC").isoformat()
        logger.info(
            "Unified frame built: %d zones, %d with active fire nearby",
            len(unified), int(unified["active_fire_nearby"].sum()),
        )
        return unified


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    use_offline = not bool(API.firms_map_key)
    if use_offline:
        logger.info("No FIRMS_MAP_KEY found - running in offline/synthetic mode.")
    else:
        logger.info("FIRMS_MAP_KEY found - fetching real satellite data.")
    module = DataIngestionModule(offline=use_offline)
    df = module.build_unified_frame()
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    out_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "unified_sample.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved sample to {out_path}")
