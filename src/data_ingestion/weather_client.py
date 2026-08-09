"""
WeatherClient - wraps the OpenWeatherMap Current Weather API to retrieve the
temperature, humidity, wind speed/direction, and precipitation fields required
by SRS 5.1 (Data Ingestion) and consumed by feature_engineering.py to compute
the Fire Weather Index family (FWI, DMC, BUI, DC).

Rate limiting: OpenWeatherMap's free tier caps requests at 60/minute. This
client self-paces sequential calls to stay under that limit, and does NOT
retry on HTTP 429 (Too Many Requests) - retrying a rate-limited request just
makes the block worse. Only transient server errors (5xx) are retried.
"""
import logging
import time
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

FREE_TIER_CALLS_PER_MINUTE = 60
MIN_SECONDS_BETWEEN_CALLS = 60.0 / FREE_TIER_CALLS_PER_MINUTE * 1.1  # 10% safety margin


class WeatherClient:
    def __init__(self, api_key: str, base_url: str, timeout: int = 15, max_retries: int = 2):
        if not api_key:
            logger.warning(
                "OWM_API_KEY not set - WeatherClient will only work in offline/"
                "sample mode."
            )
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self._last_call_time = 0.0

        self.session = requests.Session()
        # Deliberately excludes 429 - retrying a rate-limited request just
        # extends the block. Only retry genuine transient server errors.
        retry = Retry(total=max_retries, backoff_factor=1.5,
                       status_forcelist=[500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def _throttle(self):
        """Blocks just long enough to keep this client under 60 calls/minute."""
        elapsed = time.monotonic() - self._last_call_time
        if elapsed < MIN_SECONDS_BETWEEN_CALLS:
            time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)
        self._last_call_time = time.monotonic()

    def fetch_point(self, lat: float, lon: float) -> Optional[dict]:
        """Fetch current weather for a single lat/lon grid centroid."""
        self._throttle()
        params = {"lat": lat, "lon": lon, "appid": self.api_key, "units": "metric"}
        try:
            resp = self.session.get(f"{self.base_url}/weather", params=params, timeout=self.timeout)
            if resp.status_code == 429:
                logger.error(
                    "OpenWeatherMap rate limit hit (429) for (%s, %s). Key may be "
                    "temporarily blocked - this resolves on its own within a few hours.",
                    lat, lon,
                )
                return "RATE_LIMITED"
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
        produced by build_weather_grid() (coarse grid, NOT the fine CA/fire grid).
        Failed points are dropped with a warning rather than raising, per
        SRS 5.2 Reliability (graceful degradation + staleness handling).
        Bails out early on the first 429 - if the key is rate-limited, every
        subsequent call will fail too, so there's no point burning through
        the rest of the grid and making the block last longer.
        """
        records = []
        for pt in grid_points:
            rec = self.fetch_point(pt["latitude"], pt["longitude"])
            if rec == "RATE_LIMITED":
                logger.warning(
                    "Stopping weather fetch early: rate-limited after %d/%d points. "
                    "Falling back to whatever succeeded so far.",
                    len(records), len(grid_points),
                )
                break
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
