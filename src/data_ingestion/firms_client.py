"""
FIRMSClient - wraps NASA FIRMS (Fire Information for Resource Management System)
Area API to retrieve near real-time active-fire hotspot detections from the
VIIRS/MODIS satellite feeds referenced in Report Ch.1 (Existing System 'a')
and SRS 5.1 (Data Ingestion).
"""
import logging
from datetime import datetime, timezone
from io import StringIO
from typing import Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class FIRMSClient:
    """
    Thin, retry-safe client around the FIRMS Area CSV endpoint.

    Endpoint shape:
    https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SOURCE}/{AREA}/{DAY_RANGE}
    where AREA is 'west,south,east,north' in degrees.
    """

    EXPECTED_COLUMNS = [
        "latitude", "longitude", "bright_ti4", "scan", "track",
        "acq_date", "acq_time", "satellite", "confidence", "version",
        "bright_t31", "frp", "daynight",
    ]

    def __init__(self, map_key: str, base_url: str, source: str = "VIIRS_SNPP_NRT",
                 timeout: int = 20, max_retries: int = 3):
        if not map_key:
            logger.warning(
                "FIRMS_MAP_KEY not set - FIRMSClient will only work in offline/"
                "sample mode. Get a free key at https://firms.modaps.eosdis.nasa.gov/api/"
            )
        self.map_key = map_key
        self.base_url = base_url
        self.source = source
        self.timeout = timeout

        self.session = requests.Session()
        retry = Retry(
            total=max_retries, backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def fetch_hotspots(self, min_lat: float, min_lon: float, max_lat: float,
                        max_lon: float, day_range: int = 1) -> pd.DataFrame:
        """
        Fetch active fire detections for a bounding box.
        Returns an empty, correctly-shaped DataFrame (never raises) on failure
        so downstream pipeline stages can apply staleness handling per SRS 5.2
        (Reliability: graceful handling of API downtime using cached data).
        """
        area = f"{min_lon},{min_lat},{max_lon},{max_lat}"
        url = f"{self.base_url}/{self.map_key}/{self.source}/{area}/{day_range}"

        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            if "Invalid" in resp.text[:200] or "error" in resp.text[:50].lower():
                logger.error("FIRMS API returned an error payload: %s", resp.text[:200])
                return self._empty_frame()

            df = pd.read_csv(StringIO(resp.text))
            if df.empty:
                logger.info("FIRMS returned no active hotspots for the requested area.")
                return self._empty_frame()

            df["fetched_at"] = datetime.now(timezone.utc).isoformat()
            df["acq_datetime"] = pd.to_datetime(
                df["acq_date"] + " " + df["acq_time"].astype(str).str.zfill(4),
                format="%Y-%m-%d %H%M", errors="coerce",
            )
            logger.info("FIRMS: fetched %d hotspots", len(df))
            return df

        except requests.RequestException as exc:
            logger.error("FIRMS API request failed: %s", exc)
            return self._empty_frame()
        except pd.errors.ParserError as exc:
            logger.error("FIRMS response could not be parsed as CSV: %s", exc)
            return self._empty_frame()

    def _empty_frame(self) -> pd.DataFrame:
        cols = self.EXPECTED_COLUMNS + ["fetched_at", "acq_datetime"]
        return pd.DataFrame(columns=cols)

    @staticmethod
    def generate_sample(region_bounds: dict, n_points: int = 25,
                         seed: Optional[int] = 42) -> pd.DataFrame:
        """
        Synthetic hotspot generator used for offline development, unit tests,
        and demo mode when no FIRMS_MAP_KEY / internet access is available.
        Distribution is loosely biased toward realistic FRP/confidence ranges.
        """
        import numpy as np
        rng = np.random.default_rng(seed)
        n = n_points
        df = pd.DataFrame({
            "latitude": rng.uniform(region_bounds["min_lat"], region_bounds["max_lat"], n),
            "longitude": rng.uniform(region_bounds["min_lon"], region_bounds["max_lon"], n),
            "bright_ti4": rng.uniform(300, 400, n),
            "scan": rng.uniform(0.3, 1.5, n),
            "track": rng.uniform(0.3, 1.5, n),
            "acq_date": [datetime.now(timezone.utc).date().isoformat()] * n,
            "acq_time": rng.integers(0, 2359, n),
            "satellite": rng.choice(["N", "1"], n),
            "confidence": rng.choice(["l", "n", "h"], n, p=[0.2, 0.5, 0.3]),
            "version": "2.0NRT",
            "bright_t31": rng.uniform(280, 330, n),
            "frp": rng.exponential(15, n),
            "daynight": rng.choice(["D", "N"], n),
        })
        df["fetched_at"] = datetime.now(timezone.utc).isoformat()
        df["acq_datetime"] = pd.to_datetime(df["acq_date"])
        return df
