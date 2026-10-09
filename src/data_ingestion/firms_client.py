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
        # Outcome of the most recent request (never contains the key):
        # {"ok": bool, "error": str | None, "n": int, "fetched_utc": iso, "latest_acq_utc": iso | None}
        self.last_status: dict = {"ok": None, "error": None, "n": 0, "fetched_utc": None,
                                  "latest_acq_utc": None, "source": source}
        self._errors: list = []

        self.session = requests.Session()
        retry = Retry(
            total=max_retries, backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    # The FIRMS area API accepts at most 5 days per request.
    MAX_DAYS_PER_REQUEST = 5

    def fetch_hotspots(self, min_lat: float, min_lon: float, max_lat: float,
                        max_lon: float, day_range: int = 1) -> pd.DataFrame:
        """
        Fetch active fire detections for a bounding box over the last
        `day_range` days (today included). Longer ranges are split into
        <=5-day windows using the API's start-date parameter and stitched.
        Returns an empty, correctly-shaped DataFrame (never raises) on failure
        so downstream pipeline stages can apply staleness handling per SRS 5.2.
        """
        self._errors = []
        if not self.map_key:
            self._errors.append("FIRMS_MAP_KEY is not set")
            return self._finish(None)
        area = f"{min_lon},{min_lat},{max_lon},{max_lat}"
        base = f"{self.base_url}/{self.map_key}/{self.source}/{area}"
        if day_range <= self.MAX_DAYS_PER_REQUEST:
            return self._finish(self._get_csv(f"{base}/{day_range}"))

        today = datetime.now(timezone.utc).date()
        frames, remaining = [], day_range
        while remaining > 0:
            n = min(self.MAX_DAYS_PER_REQUEST, remaining)
            start = today - pd.Timedelta(days=remaining - 1)
            part = self._get_csv(f"{base}/{n}/{start.isoformat()}")
            if part is not None and not part.empty:
                frames.append(part)
            remaining -= n
        if not frames:
            # Dated windows failed (or were empty): fall back to the plain
            # most-recent request so the map still shows today's fires.
            return self._finish(self._get_csv(f"{base}/{self.MAX_DAYS_PER_REQUEST}"))
        df = pd.concat(frames, ignore_index=True).drop_duplicates(
            subset=["latitude", "longitude", "acq_date", "acq_time"])
        return self._finish(df)

    def _get_csv(self, url: str) -> Optional[pd.DataFrame]:
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            if "Invalid" in resp.text[:200] or "error" in resp.text[:50].lower():
                msg = self._clean(resp.text[:160].strip())
                logger.error("FIRMS API returned an error payload: %s", msg)
                self._errors.append(f"FIRMS rejected the request: {msg}")
                return None
            return pd.read_csv(StringIO(resp.text))
        except requests.RequestException as exc:
            msg = self._clean(str(exc))
            logger.error("FIRMS API request failed: %s", msg)
            self._errors.append(f"FIRMS request failed: {msg[:200]}")
            return None
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            logger.error("FIRMS response could not be parsed as CSV: %s", exc)
            self._errors.append("FIRMS response could not be parsed")
            return None

    def _clean(self, text: str) -> str:
        """Never let the MAP_KEY reach logs or the UI."""
        return text.replace(self.map_key, "***") if self.map_key else text

    def _finish(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if df is None or df.empty:
            if df is not None:
                logger.info("FIRMS returned no active hotspots for the requested area.")
            ok = df is not None and not self._errors
            self.last_status = {"ok": ok, "error": None if ok else ("; ".join(dict.fromkeys(self._errors)) or "no response"),
                                "n": 0, "fetched_utc": now, "latest_acq_utc": None, "source": self.source}
            return self._empty_frame()
        df = df.copy()
        df["fetched_at"] = datetime.now(timezone.utc).isoformat()
        df["acq_datetime"] = pd.to_datetime(
            df["acq_date"] + " " + df["acq_time"].astype(str).str.zfill(4),
            format="%Y-%m-%d %H%M", errors="coerce",
        )
        latest = df["acq_datetime"].max()
        self.last_status = {"ok": True, "error": "; ".join(dict.fromkeys(self._errors)) or None, "n": int(len(df)), "fetched_utc": now,
                            "latest_acq_utc": latest.strftime("%Y-%m-%dT%H:%MZ") if pd.notna(latest) else None,
                            "source": self.source}
        logger.info("FIRMS: fetched %d hotspots", len(df))
        return df

    def _empty_frame(self) -> pd.DataFrame:
        cols = self.EXPECTED_COLUMNS + ["fetched_at", "acq_datetime"]
        return pd.DataFrame(columns=cols)

    @staticmethod
    def generate_sample(region_bounds: dict, n_points: int = 25,
                         seed: Optional[int] = None,
                         prior: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Synthetic hotspot generator for offline/demo mode (no FIRMS key or no
        internet). Mimics what the real archive looks like, because the model
        reads the previous 10 days of detections:
          * `n_points` fires are burning today;
          * each has been burning for a few days already (1-6), with a few
            detections in neighbouring cells as it spreads;
          * scattered background detections elsewhere in the region, scaled
            with how many fires are active.
        If `prior` (latitude, longitude, weight) is given, fires start where
        fires have historically been most frequent rather than uniformly.
        """
        import numpy as np
        rng = np.random.default_rng(seed)
        today = pd.Timestamp(datetime.now(timezone.utc).date())

        def locations(k):
            if prior is not None and not prior.empty and prior["weight"].sum() > 0:
                w = prior["weight"].to_numpy(dtype=float)
                idx = rng.choice(len(prior), size=k, p=w / w.sum())
                return (prior["latitude"].to_numpy()[idx] + rng.uniform(-0.04, 0.04, k),
                        prior["longitude"].to_numpy()[idx] + rng.uniform(-0.04, 0.04, k))
            return (rng.uniform(region_bounds["min_lat"], region_bounds["max_lat"], k),
                    rng.uniform(region_bounds["min_lon"], region_bounds["max_lon"], k))

        lat0, lon0 = locations(n_points)
        rows = []
        for la, lo in zip(lat0, lon0):
            rows.append((la, lo, 0))                         # burning today
            for back in range(1, int(rng.integers(1, 7)) + 1):
                rows.append((la + rng.normal(0, 0.01), lo + rng.normal(0, 0.01), back))
                for _ in range(int(rng.integers(0, 4))):     # spread into neighbours
                    rows.append((la + rng.normal(0, 0.08), lo + rng.normal(0, 0.08), back))
        for back in range(1, 11):                            # regional background
            k = int(rng.poisson(4 * n_points))
            bl, bo = locations(k)
            rows += [(a, b, back) for a, b in zip(bl, bo)]

        n = len(rows)
        lat = np.clip([r[0] for r in rows], region_bounds["min_lat"], region_bounds["max_lat"])
        lon = np.clip([r[1] for r in rows], region_bounds["min_lon"], region_bounds["max_lon"])
        df = pd.DataFrame({
            "latitude": lat, "longitude": lon,
            "bright_ti4": rng.uniform(300, 400, n),
            "scan": rng.uniform(0.3, 1.5, n),
            "track": rng.uniform(0.3, 1.5, n),
            "acq_date": [(today - pd.Timedelta(days=r[2])).date().isoformat() for r in rows],
            "acq_time": rng.integers(0, 2359, n),
            "satellite": rng.choice(["N", "1"], n),
            # today's seeded fires are always confident detections
            "confidence": ["h" if r[2] == 0 else c for r, c in
                           zip(rows, rng.choice(["l", "n", "h"], n, p=[0.2, 0.5, 0.3]))],
            "version": "2.0NRT",
            "bright_t31": rng.uniform(280, 330, n),
            "frp": rng.exponential(15, n),
            "daynight": rng.choice(["D", "N"], n),
        })
        df["fetched_at"] = datetime.now(timezone.utc).isoformat()
        df["acq_datetime"] = pd.to_datetime(df["acq_date"])
        return df
