"""
WeatherClient - wraps the OpenWeatherMap Current Weather API to retrieve the
temperature, humidity, wind speed/direction, and precipitation fields required
by SRS 5.1 (Data Ingestion) and consumed by feature_engineering.py to compute
the Fire Weather Index family (FWI, DMC, BUI, DC).

Rate limiting: OpenWeatherMap's free tier caps requests at 60/minute. The
earlier version of this client enforced that by sleeping a fixed ~1.1s
between EVERY call, serially - so a 56-point weather grid (the default
region's actual size) took ~60+ seconds every refresh even though 56 calls
is comfortably under the 60/minute quota. That fixed-interval throttle was
the single biggest source of latency in the whole refresh cycle.

This version instead uses a sliding-window rate limiter (bursts are allowed
up to the quota within any rolling 60s window; it only blocks once actually
near the limit) combined with a small thread pool, so grids at or under the
per-minute quota complete in a few seconds - bound by network round-trip
time, not by an unconditional per-call sleep. Only once a fetch genuinely
approaches 60 calls/minute does it start pacing itself, exactly like the API
requires - no more, no less.

Does NOT retry on HTTP 429 (Too Many Requests) - retrying a rate-limited
request just makes the block worse. Only transient server errors (5xx) are
retried.
"""
import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

FREE_TIER_CALLS_PER_MINUTE = 60
# Small safety margin below the true quota rather than a per-call % margin,
# so a full-speed burst still never actually hits 429.
_SAFE_CALLS_PER_MINUTE = FREE_TIER_CALLS_PER_MINUTE - 3
_WINDOW_SECONDS = 60.0
_MAX_CONCURRENCY = 8


class _SlidingWindowLimiter:
    """Thread-safe: allows up to `max_calls` within any trailing `period`
    seconds. Unlike a fixed per-call delay, callers only wait once genuinely
    at the limit - a burst well under quota pays no delay at all."""

    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self._timestamps: deque = deque()
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            while True:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] > self.period:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.max_calls:
                    self._timestamps.append(now)
                    return
                sleep_for = self.period - (now - self._timestamps[0])
                if sleep_for > 0:
                    time.sleep(sleep_for)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
        # Outcome of the most recent request (never contains the key)
        self.last_error: Optional[str] = None
        self.last_status: dict = {"ok": None, "error": None, "n_ok": 0, "n_total": 0, "fetched_utc": None}
        self._limiter = _SlidingWindowLimiter(_SAFE_CALLS_PER_MINUTE, _WINDOW_SECONDS)

        self.session = requests.Session()
        # requests.Session is thread-safe for concurrent use across threads
        # sharing one connection pool - safe with the thread pool in fetch_grid.
        adapter = HTTPAdapter(
            max_retries=Retry(total=max_retries, backoff_factor=1.5,
                               status_forcelist=[500, 502, 503, 504]),
            pool_maxsize=_MAX_CONCURRENCY,
        )
        self.session.mount("https://", adapter)

    def _clean(self, text: str) -> str:
        """Never let the API key (it is a URL query parameter) reach logs or the UI."""
        return str(text).replace(self.api_key, "***") if self.api_key else str(text)

    def fetch_point(self, lat: float, lon: float) -> Optional[dict]:
        """Fetch current weather for a single lat/lon grid centroid."""
        if not self.api_key:
            self.last_error = "OWM_API_KEY is not set"
            return None
        self._limiter.acquire()
        params = {"lat": lat, "lon": lon, "appid": self.api_key, "units": "metric"}
        try:
            resp = self.session.get(f"{self.base_url}/weather", params=params, timeout=self.timeout)
            if resp.status_code == 429:
                logger.error(
                    "OpenWeatherMap rate limit hit (429) for (%s, %s). Key may be "
                    "temporarily blocked - this resolves on its own within a few hours.",
                    lat, lon,
                )
                self.last_error = "OpenWeatherMap rate limit reached (HTTP 429)"
                return "RATE_LIMITED"
            if resp.status_code == 401:
                self.last_error = "OpenWeatherMap rejected the API key (HTTP 401; new keys can take a few hours to activate)"
                logger.error(self.last_error)
                return None
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
                "weather_description": data.get("weather", [{}])[0].get("description", ""),
                "observed_at": (datetime.fromtimestamp(int(data["dt"]), timezone.utc).isoformat()
                                if data.get("dt") else None),
                "place_name": data.get("name") or "",
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
        except (requests.RequestException, KeyError, ValueError) as exc:
            self.last_error = f"OpenWeatherMap request failed: {self._clean(exc)[:200]}"
            logger.error("OpenWeatherMap request failed for (%s, %s): %s", lat, lon, self._clean(exc))
            return None

    def fetch_forecast_point(self, lat: float, lon: float, horizon_hours: float = 2.0) -> Optional[List[dict]]:
        """3-hourly wind forecast (OpenWeatherMap /forecast, free tier) for one
        point, covering now through at least `horizon_hours` ahead. Returns a
        list of {"dt": unix_seconds, "wind_speed_ms", "wind_deg"} or None on
        failure (or "RATE_LIMITED" on HTTP 429, like fetch_point)."""
        self._limiter.acquire()
        # slots are 3 h apart: enough to bracket the horizon, plus one slot beyond it
        cnt = int(horizon_hours // 3) + 2
        params = {"lat": lat, "lon": lon, "appid": self.api_key, "units": "metric", "cnt": cnt}
        try:
            resp = self.session.get(f"{self.base_url}/forecast", params=params, timeout=self.timeout)
            if resp.status_code == 429:
                logger.error("OpenWeatherMap rate limit hit (429) on forecast for (%s, %s).", lat, lon)
                return "RATE_LIMITED"
            resp.raise_for_status()
            slots = [{"dt": int(it["dt"]),
                      "wind_speed_ms": float(it.get("wind", {}).get("speed", 0.0)),
                      "wind_deg": float(it.get("wind", {}).get("deg", 0.0))}
                     for it in resp.json()["list"]]
            return slots or None
        except (requests.RequestException, KeyError, ValueError, TypeError) as exc:
            logger.error("OpenWeatherMap forecast request failed for (%s, %s): %s", lat, lon, self._clean(exc))
            return None

    def fetch_forecast_wind(self, points: List[dict], horizon_hours: float = 2.0) -> List[List[dict]]:
        """Forecast wind for a few points at once (concurrent, same rate
        limiter as the current-weather calls). Failed points are dropped."""
        out: List[List[dict]] = []
        if not points:
            return out
        with ThreadPoolExecutor(max_workers=min(_MAX_CONCURRENCY, len(points))) as pool:
            futures = [pool.submit(self.fetch_forecast_point, p["latitude"], p["longitude"], horizon_hours)
                       for p in points]
            for f in as_completed(futures):
                res = f.result()
                if isinstance(res, list):
                    out.append(res)
        return out

    def fetch_grid(self, grid_points: List[dict]) -> pd.DataFrame:
        """
        Fetch weather for every centroid in a pre-built grid, concurrently
        (bounded by _MAX_CONCURRENCY workers, all still sharing the single
        rate limiter above so the aggregate call rate never exceeds quota).
        grid_points: list of {"latitude": .., "longitude": ..} dicts, typically
        produced by build_weather_grid() (coarse grid, NOT the fine CA/fire grid).
        Failed points are dropped with a warning rather than raising, per
        SRS 5.2 Reliability (graceful degradation + staleness handling).
        Stops submitting further points on the first 429 - if the key is
        rate-limited, every subsequent call will fail too, so there's no
        point burning through the rest of the grid and extending the block.
        Already in-flight requests at that moment still finish normally.
        """
        if not grid_points:
            return self._empty_frame()
        self.last_error = None
        if not self.api_key:
            self.last_status = {"ok": False, "error": "OWM_API_KEY is not set", "n_ok": 0,
                                "n_total": len(grid_points), "fetched_utc": _now()}
            return self._empty_frame()

        records = []
        rate_limited = threading.Event()

        def _worker(pt):
            if rate_limited.is_set():
                return None
            return self.fetch_point(pt["latitude"], pt["longitude"])

        with ThreadPoolExecutor(max_workers=min(_MAX_CONCURRENCY, len(grid_points))) as pool:
            futures = {pool.submit(_worker, pt): pt for pt in grid_points}
            for future in as_completed(futures):
                rec = future.result()
                if rec == "RATE_LIMITED":
                    rate_limited.set()
                    continue
                if rec is not None:
                    records.append(rec)

        if rate_limited.is_set():
            logger.warning(
                "Weather fetch hit the rate limit partway through: %d/%d points "
                "succeeded. Falling back to whatever succeeded so far.",
                len(records), len(grid_points),
            )
        self.last_status = {"ok": bool(records), "n_ok": len(records), "n_total": len(grid_points),
                            "fetched_utc": _now(),
                            "error": None if len(records) == len(grid_points) else
                            (self.last_error or "some points failed")}
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
    def generate_sample(grid_points: List[dict], seed: Optional[int] = None,
                         temp_c: Optional[float] = None,
                         wind_speed_ms: Optional[float] = None,
                         humidity_pct: Optional[float] = None,
                         wind_from_deg: Optional[float] = None) -> pd.DataFrame:
        """Synthetic weather generator. Pass temp_c/wind_speed_ms/humidity_pct
        to center the spread around a chosen scenario value. wind_from_deg
        (meteorological convention: the bearing the wind blows FROM) gives
        every point that direction +/- 15 deg; without it the direction is
        random per point, as before."""
        import numpy as np
        rng = np.random.default_rng(seed)
        n = len(grid_points)
        temp_lo, temp_hi = (22, 42) if temp_c is None else (temp_c - 4, temp_c + 4)
        wind_lo, wind_hi = (0.5, 12) if wind_speed_ms is None else (max(0.0, wind_speed_ms - 2), wind_speed_ms + 2)
        hum_lo, hum_hi = (8, 85) if humidity_pct is None else (max(0.0, humidity_pct - 10), min(100.0, humidity_pct + 10))
        df = pd.DataFrame({
            "latitude": [p["latitude"] for p in grid_points],
            "longitude": [p["longitude"] for p in grid_points],
            "temperature_c": rng.uniform(temp_lo, temp_hi, n),
            "humidity_pct": rng.uniform(hum_lo, hum_hi, n),
            "pressure_hpa": rng.uniform(1005, 1015, n),
            "wind_speed_ms": rng.uniform(wind_lo, wind_hi, n),
            "wind_deg": rng.uniform(0, 360, n),
            "precipitation_mm": rng.choice([0, 0, 0, 0.5, 2.0], n),
            "clouds_pct": rng.integers(0, 100, n),
            "weather_main": rng.choice(["Clear", "Clouds", "Rain"], n, p=[0.6, 0.3, 0.1]),
        })
        if wind_from_deg is not None:
            # Reuses the uniform draw above (no extra random numbers), so every
            # other column is identical to a run without a wind direction.
            df["wind_deg"] = (float(wind_from_deg) + (df["wind_deg"] - 180.0) / 12.0) % 360.0
        df["fetched_at"] = datetime.now(timezone.utc).isoformat()
        return df
