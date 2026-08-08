"""
WeatherClient - wraps the OpenWeatherMap Current Weather API to retrieve the
temperature, humidity, wind speed/direction, and precipitation fields required
by SRS 5.1 (Data Ingestion) and consumed by feature_engineering.py to compute
the Fire Weather Index family (FWI, DMC, BUI, DC).
"""
import logging
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class WeatherClient:
    def __init__(self, api_key: str, base_url: str, timeout: int = 15, max_retries: int = 3):
        if not api_key:
            logger.warning(
                "OWM_API_KEY not set - WeatherClient will only work in offline/"
                "sample mode."
            )
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

        self.session = requests.Session()
        retry = Retry(total=max_retries, backoff_factor=1.5,
                       status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def fetch_point(self, lat: float, lon: float) -> Optional[dict]:
        """Fetch current weather for a single lat/lon grid centroid."""
        params = {"lat": lat, "lon": lon, "appid": self.api_key, "units": "metric"}
        try:
            resp = self.session.get(f"{self.base_url}/weather", params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return {
                "latitude": lat,
                "longitude": lon,
                "temperature_c": data["main"]["temp"],
                "humidity_pct": data["main"]["humidity"],
                "pressure_hpa": data["main"]["pressure"],
                "wind_speed_ms": data.get("wind", {}).get("speed", 0.0),
                "wind_deg": data.get("wind", {}).get("deg", 0.0),
                "precipitation_mm": data.get("rain", {}).get("1h", 0.0),
                "clouds_pct": data.get("clouds", {}).get("all", 0),
                "weather_main": data.get("weather", [{}])[0].get("main", "Unknown"),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
        except (requests.RequestException, KeyError, ValueError) as exc:
            logger.error("OpenWeatherMap request failed for (%s, %s): %s", lat, lon, exc)
            return None

    def fetch_grid(self, grid_points: List[dict]) -> pd.DataFrame:
        """
        Fetch weather for every centroid in a pre-built grid.
        grid_points: list of {"latitude": .., "longitude": ..} dicts, typically
        produced by feature_engineering.build_region_grid().
        Failed points are dropped with a warning rather than raising, per
        SRS 5.2 Reliability (graceful degradation + staleness handling).
        """
        records = []
        for pt in grid_points:
            rec = self.fetch_point(pt["latitude"], pt["longitude"])
            if rec is not None:
                records.append(rec)
        if not records:
            logger.warning("WeatherClient.fetch_grid: no successful responses; returning empty frame.")
            return self._empty_frame()
        return pd.DataFrame(records)

    def _empty_frame(self) -> pd.DataFrame:
        cols = ["latitude", "longitude", "temperature_c", "humidity_pct", "pressure_hpa",
                "wind_speed_ms", "wind_deg", "precipitation_mm", "clouds_pct",
                "weather_main", "fetched_at"]
        return pd.DataFrame(columns=cols)

    @staticmethod
    def generate_sample(grid_points: List[dict], seed: Optional[int] = 7) -> pd.DataFrame:
        """Synthetic weather generator for offline dev/tests/demo mode."""
        import numpy as np
        rng = np.random.default_rng(seed)
        n = len(grid_points)
        df = pd.DataFrame({
            "latitude": [p["latitude"] for p in grid_points],
            "longitude": [p["longitude"] for p in grid_points],
            "temperature_c": rng.uniform(22, 42, n),
            "humidity_pct": rng.uniform(8, 85, n),
            "pressure_hpa": rng.uniform(1005, 1015, n),
            "wind_speed_ms": rng.uniform(0.5, 12, n),
            "wind_deg": rng.uniform(0, 360, n),
            "precipitation_mm": rng.choice([0, 0, 0, 0.5, 2.0], n),
            "clouds_pct": rng.integers(0, 100, n),
            "weather_main": rng.choice(["Clear", "Clouds", "Rain"], n, p=[0.6, 0.3, 0.1]),
        })
        df["fetched_at"] = datetime.now(timezone.utc).isoformat()
        return df
