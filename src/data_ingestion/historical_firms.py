"""
HistoricalFIRMSClient - pulls historical (science-quality archive) fire
hotspot data from NASA FIRMS for a date range. This provides REAL ground-truth
fire occurrence labels for ML training, fixing the label-leakage issue flagged
earlier (where labels were a formula derived from the same FWI features used
to predict them).

Uses "_SP" (Standard Processing) sources rather than "_NRT" (Near Real Time):
NRT only retains ~2 months of rolling data, while SP is the permanent
science-quality archive going back years - what we need for multi-year
training data.

The FIRMS Area API caps each request at 10 days of data, so this client
chunks a wide date range into <=10-day windows and stitches the results
together, with a short pause between requests to be polite to the API.
"""
import logging
import time
from datetime import date, timedelta
from io import StringIO
from typing import List

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

MAX_DAYS_PER_REQUEST = 1  # SP (archive) sources reject day_range>1 with 400 Bad Request;
                            # confirmed empirically - NRT sources may allow up to 10, but
                            # we use SP here for multi-year history, so this stays at 1.


class HistoricalFIRMSClient:
    def __init__(self, map_key: str, base_url: str = "https://firms.modaps.eosdis.nasa.gov/api/area/csv",
                 source: str = "VIIRS_SNPP_SP", timeout: int = 30, max_retries: int = 3):
        if not map_key:
            raise ValueError(
                "HistoricalFIRMSClient requires a FIRMS_MAP_KEY - get one free at "
                "https://firms.modaps.eosdis.nasa.gov/api/"
            )
        self.map_key = map_key
        self.base_url = base_url
        self.source = source
        self.timeout = timeout

        self.session = requests.Session()
        # Deliberately excludes 429, same lesson as WeatherClient: don't
        # hammer an already-rate-limited endpoint.
        retry = Retry(total=max_retries, backoff_factor=2.0, status_forcelist=[500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def _fetch_window(self, min_lat: float, min_lon: float, max_lat: float, max_lon: float,
                       window_start: date, day_range: int) -> pd.DataFrame:
        area = f"{min_lon},{min_lat},{max_lon},{max_lat}"
        date_str = window_start.isoformat()
        url = f"{self.base_url}/{self.map_key}/{self.source}/{area}/{day_range}/{date_str}"

        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 429:
                logger.error("FIRMS rate limit hit (429) for window starting %s. Stopping.", date_str)
                return None
            resp.raise_for_status()
            text = resp.text

            if "Invalid" in text[:200] or "error" in text[:50].lower():
                logger.error("FIRMS historical API returned an error for window starting %s: %s",
                             date_str, text[:200])
                return pd.DataFrame()

            df = pd.read_csv(StringIO(text))
            if df.empty:
                return df

            df["acq_datetime"] = pd.to_datetime(
                df["acq_date"] + " " + df["acq_time"].astype(str).str.zfill(4),
                format="%Y-%m-%d %H%M", errors="coerce",
            )
            return df

        except requests.RequestException as exc:
            logger.error("FIRMS historical request failed for window starting %s: %s", date_str, exc)
            return pd.DataFrame()
        except pd.errors.ParserError as exc:
            logger.error("FIRMS historical response unparsable for window starting %s: %s", date_str, exc)
            return pd.DataFrame()

    def fetch_range(self, min_lat: float, min_lon: float, max_lat: float, max_lon: float,
                     start_date: date, end_date: date, pause_seconds: float = 1.5) -> pd.DataFrame:
        """
        Fetches all hotspots in [start_date, end_date] inclusive, chunked into
        <=10-day windows per the FIRMS API limit, and stitched together.
        """
        all_frames = []
        window_start = start_date
        total_days = (end_date - start_date).days + 1
        windows_done = 0

        while window_start <= end_date:
            days_remaining = (end_date - window_start).days + 1
            day_range = min(MAX_DAYS_PER_REQUEST, days_remaining)
            window_end = window_start + timedelta(days=day_range - 1)

            df = self._fetch_window(min_lat, min_lon, max_lat, max_lon, window_start, day_range)
            windows_done += 1

            if df is None:  # rate limited - stop entirely, don't burn more calls
                logger.warning("Stopping historical fetch early due to rate limiting.")
                break
            if not df.empty:
                all_frames.append(df)

            if windows_done % 10 == 0 or window_end >= end_date:
                pct = 100 * (window_end - start_date).days / max(total_days - 1, 1)
                logger.info("Progress: %s done (%d/%d days, %.0f%%) - %d detections so far",
                            window_end, (window_end - start_date).days + 1, total_days, pct,
                            sum(len(f) for f in all_frames))

            window_start = window_end + timedelta(days=1)
            time.sleep(pause_seconds)

        if not all_frames:
            logger.warning("No historical detections found for %s to %s", start_date, end_date)
            return pd.DataFrame()

        combined = pd.concat(all_frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time"])
        logger.info("Historical fetch complete: %d total detections, %s to %s",
                    len(combined), start_date, end_date)
        return combined


def fetch_karnataka_fire_seasons(map_key: str, years: List[int], region_bounds: dict,
                                  source: str = "VIIRS_SNPP_SP") -> pd.DataFrame:
    """
    Convenience wrapper: pulls the Jan-May fire season (Karnataka/Western
    Ghats peak fire months, per Report Ch.1 scope) for each requested year.
    """
    client = HistoricalFIRMSClient(map_key, source=source)
    frames = []
    for year in years:
        start = date(year, 1, 1)
        end = date(year, 5, 31)
        logger.info("=== Fetching fire season %d (%s to %s) ===", year, start, end)
        df = client.fetch_range(
            region_bounds["min_lat"], region_bounds["min_lon"],
            region_bounds["max_lat"], region_bounds["max_lon"],
            start, end,
        )
        if not df.empty:
            df["fire_season_year"] = year
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from config.config import API, REGION, DATA_RAW_DIR

    logging.basicConfig(level=logging.INFO)

    if not API.firms_map_key:
        print("ERROR: FIRMS_MAP_KEY not set in .env - historical download requires a real key.")
        sys.exit(1)

    region_bounds = {
        "min_lat": REGION.min_lat, "max_lat": REGION.max_lat,
        "min_lon": REGION.min_lon, "max_lon": REGION.max_lon,
    }

    years = [2023, 2024, 2025]
    combined = fetch_karnataka_fire_seasons(API.firms_map_key, years, region_bounds)

    if combined.empty:
        print(
            "\nNo historical fire detections found. Possible causes:\n"
            "  1. VIIRS_SNPP_SP archive may lag behind by a few weeks/months for the most\n"
            "     recent year - try dropping the current year from the 'years' list.\n"
            "  2. Try source='MODIS_SP' instead (longer archive history, coarser resolution).\n"
            "  3. Check the printed URLs above for any 'Invalid' error text from the API."
        )
    else:
        out_path = DATA_RAW_DIR / "historical_fires_karnataka.csv"
        combined.to_csv(out_path, index=False)
        print(f"\nSaved {len(combined)} historical detections to {out_path}")
        print("\nDetections per fire season:")
        print(combined.groupby("fire_season_year").size())
