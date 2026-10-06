"""
ElevationClient - fetches real terrain elevation for the region grid, used to
add slope-driven fire spread to the Cellular Automata simulator (previously a
documented flat-terrain limitation). Uses the free, keyless Open-Elevation API
(https://open-elevation.com/) - no signup required, unlike FIRMS/OWM.

IMPORTANT: like the other live-API clients in this project, the real network
call here has NOT been tested against the live endpoint from the development
sandbox (no route to api.open-elevation.com from that environment). The
request format matches Open-Elevation's documented API exactly, but this is
the first real test of it - same situation as historical_firms.py and
historical_weather.py were when first shipped, both of which needed one
round of live debugging before working. Expect the same here.

Falls back to a synthetic, clearly-labeled procedural terrain when the API is
unavailable, so the CA simulator's slope feature is always demonstrable
offline, but is never silently presented as real elevation data.
"""
import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"
BATCH_SIZE = 200  # Open-Elevation recommends batching, not one point per request


class ElevationClient:
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()

    def fetch_elevations(self, points: pd.DataFrame) -> Optional[np.ndarray]:
        """
        points: DataFrame with 'latitude', 'longitude' columns.
        Returns an array of elevations in meters, same order as input, or
        None if the API call fails entirely (caller should fall back to
        synthetic terrain in that case).
        """
        elevations = np.full(len(points), np.nan)
        records = points[["latitude", "longitude"]].to_dict("records")

        for start in range(0, len(records), BATCH_SIZE):
            batch = records[start:start + BATCH_SIZE]
            payload = {"locations": [{"latitude": r["latitude"], "longitude": r["longitude"]} for r in batch]}
            try:
                resp = self.session.post(OPEN_ELEVATION_URL, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                results = resp.json().get("results", [])
                for i, res in enumerate(results):
                    elevations[start + i] = res.get("elevation", np.nan)
                logger.info("Elevation batch %d-%d fetched OK", start, start + len(batch))
            except requests.RequestException as exc:
                logger.error("Open-Elevation request failed for batch %d-%d: %s", start, start + len(batch), exc)
                return None
            except (ValueError, KeyError) as exc:
                logger.error("Open-Elevation response unparsable for batch %d-%d: %s", start, start + len(batch), exc)
                return None

        if np.isnan(elevations).any():
            logger.warning("Some elevation values missing from API response (%d/%d NaN)",
                            int(np.isnan(elevations).sum()), len(elevations))
        return elevations


def generate_synthetic_terrain(lats: np.ndarray, lons: np.ndarray, seed: int = 7) -> np.ndarray:
    """
    Procedural terrain proxy for offline/demo mode or when the elevation API
    is unavailable. NOT real elevation data - produces a smooth, plausible
    hilly-terrain shape (layered sine ridges + noise, roughly in the Western
    Ghats' real elevation range of ~0-2600m) so the slope-spread feature is
    demonstrable without a live API call. Deterministic per (lat, lon) via
    the seed, so it doesn't change between refreshes.
    """
    rng = np.random.default_rng(seed)
    lat_n = (lats - lats.min()) / max(lats.max() - lats.min(), 1e-6)
    lon_n = (lons - lons.min()) / max(lons.max() - lons.min(), 1e-6)

    ridge1 = np.sin(lat_n * 4.5 + 0.7) * np.cos(lon_n * 3.0)
    ridge2 = np.sin(lon_n * 6.0 + 1.3) * 0.5
    base = (ridge1 + ridge2 + 1.5) / 3.0  # roughly 0-1
    noise = rng.normal(0, 0.04, size=len(lats))
    terrain = np.clip(base + noise, 0, 1)

    elevation_m = 100 + terrain * 1400  # Western Ghats foothill-to-ridge range
    return elevation_m


def get_elevation_grid(points: pd.DataFrame, use_live: bool = True) -> tuple:
    """
    Convenience entry point: tries the real API first (if use_live), falls
    back to synthetic terrain on any failure. Returns (elevations, is_real).
    """
    if use_live:
        client = ElevationClient()
        real = client.fetch_elevations(points)
        if real is not None and not np.isnan(real).any():
            logger.info("Using REAL elevation data from Open-Elevation.")
            return real, True
        logger.warning("Falling back to synthetic terrain (API unavailable or incomplete).")

    synthetic = generate_synthetic_terrain(points["latitude"].values, points["longitude"].values)
    return synthetic, False


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from config.config import DATA_RAW_DIR
    from src.data_ingestion.ingestion_module import build_region_grid

    logging.basicConfig(level=logging.INFO)
    grid = build_region_grid()

    elevations, is_real = get_elevation_grid(grid, use_live=True)
    grid["elevation_m"] = elevations

    print(f"Elevation source: {'REAL (Open-Elevation API)' if is_real else 'SYNTHETIC (fallback terrain)'}")
    print(grid[["zone_id", "latitude", "longitude", "elevation_m"]].describe())

    out_path = DATA_RAW_DIR / "elevation_grid.csv"
    grid[["zone_id", "row", "col", "elevation_m"]].to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
