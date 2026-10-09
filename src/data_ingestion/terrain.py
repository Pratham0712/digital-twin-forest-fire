"""
terrain.py - real terrain (elevation + slope) for the region grid.

Used in two places, both reading the same cached table:
  * the fire-risk model, as the features `elev_m` and `slope_pct`;
  * the cellular-automata spread simulation, as its elevation grid.

How the table is built
----------------------
For every grid cell the elevation is sampled on a 3x3 sub-grid spread across
the cell (offsets of -1/3, 0, +1/3 of the cell width). From that:
  * elev_m    = mean of the 9 samples
  * slope_pct = mean gradient magnitude over the sub-grid, in percent
                (rise / run x 100, run measured in metres at that latitude)
so slope describes the ground INSIDE each cell, not just the difference
between neighbouring cell centres.

Elevation source: OpenTopoData SRTM 90 m (free, no key, 100 points per request,
100,000 points/day), falling back to Open-Meteo and then Open-Elevation. Neither
has been called from the development sandbox (no network route), so the first
real run happens on the user's machine via scripts/build_terrain.py; the
response handling is strict (length checks, NaN checks, retries) so a bad
response fails loudly instead of writing wrong terrain.

The table is keyed by rounded (latitude, longitude), like ZoneClimatology, so
it works for any region's grid.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

TERRAIN_COLUMNS = ["elev_m", "slope_pct"]
TERRAIN_GLOB = "terrain_*.csv"   # lives in models/ next to the climatology, so it deploys with the model
METERS_PER_DEG_LAT = 111_320.0

OPEN_METEO_URL = "https://api.open-meteo.com/v1/elevation"
OPEN_METEO_BATCH = 100
# OpenTopoData public API (SRTM 90 m): 100 points per request, 1 request per
# second, 1000 requests per day (= 100,000 points/day). Open-Meteo, by contrast,
# counts every point against a ~600/minute and 10,000/day allowance, which a
# 12,600-point region cannot fit into one day.
OPENTOPODATA_URL = "https://api.opentopodata.org/v1/srtm90m"
OPENTOPODATA_BATCH = 100
OPENTOPODATA_MIN_INTERVAL_S = 1.1
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"
OPEN_ELEVATION_BATCH = 200

# 3x3 sub-grid offsets in units of (cell_width / 3)
_SUB = np.array([-1.0, 0.0, 1.0])


# ── elevation fetching ────────────────────────────────────────────────────── #

def _fetch_open_meteo(lats: np.ndarray, lons: np.ndarray, timeout: int = 30) -> np.ndarray:
    params = {"latitude": ",".join(f"{v:.5f}" for v in lats),
              "longitude": ",".join(f"{v:.5f}" for v in lons)}
    r = requests.get(OPEN_METEO_URL, params=params, timeout=timeout)
    r.raise_for_status()
    elev = np.asarray(r.json()["elevation"], dtype=float)
    if elev.shape != lats.shape:
        raise ValueError(f"Open-Meteo returned {elev.size} values for {lats.size} points")
    return elev


def _fetch_opentopodata(lats: np.ndarray, lons: np.ndarray, timeout: int = 30) -> np.ndarray:
    locations = "|".join(f"{a:.5f},{b:.5f}" for a, b in zip(lats, lons))
    r = requests.get(OPENTOPODATA_URL, params={"locations": locations}, timeout=timeout)
    r.raise_for_status()
    body = r.json()
    if body.get("status") != "OK":
        raise ValueError(f"OpenTopoData status {body.get('status')}: {body.get('error')}")
    # `null` means no SRTM coverage at that point (open sea): treat as sea level.
    elev = np.asarray([0.0 if x.get("elevation") is None else x["elevation"] for x in body["results"]], dtype=float)
    if elev.shape != lats.shape:
        raise ValueError(f"OpenTopoData returned {elev.size} values for {lats.size} points")
    return elev


def _fetch_open_elevation(lats: np.ndarray, lons: np.ndarray, timeout: int = 60) -> np.ndarray:
    payload = {"locations": [{"latitude": float(a), "longitude": float(b)} for a, b in zip(lats, lons)]}
    r = requests.post(OPEN_ELEVATION_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    elev = np.asarray([x["elevation"] for x in r.json()["results"]], dtype=float)
    if elev.shape != lats.shape:
        raise ValueError(f"Open-Elevation returned {elev.size} values for {lats.size} points")
    return elev


def _retry_wait(exc: Exception, attempt: int) -> float:
    """Seconds to wait before retrying. Rate-limit (HTTP 429) responses honour
    the server's Retry-After header, or back off in steps of 20 s, because the
    free elevation APIs limit requests per minute and a short retry just
    burns attempts."""
    resp = getattr(exc, "response", None)
    if resp is not None and resp.status_code == 429:
        ra = str(resp.headers.get("Retry-After", "")).strip()
        if ra.replace(".", "", 1).isdigit():
            return min(120.0, float(ra) + 1.0)
        return min(120.0, 20.0 * (attempt + 1))
    return 2.0 ** attempt


def fetch_elevations(lats: np.ndarray, lons: np.ndarray,
                     provider: str = "auto", retries: int = 6,
                     pause_s: float = 1.0,
                     progress: Optional[Callable[[int, int], None]] = None,
                     timeout: Optional[float] = None, deadline_s: Optional[float] = None) -> np.ndarray:
    """Elevation in metres for every (lat, lon). Batches, retries with backoff,
    and (provider="auto") falls back from Open-Meteo to Open-Elevation. Raises
    if any value is still missing - never returns partial terrain."""
    lats = np.asarray(lats, float)
    lons = np.asarray(lons, float)
    out = np.full(lats.shape, np.nan)
    t_end = time.monotonic() + deadline_s if deadline_s else None
    kw = {"timeout": timeout} if timeout else {}
    providers = {"opentopodata": [(_fetch_opentopodata, OPENTOPODATA_BATCH)],
                 "open-meteo": [(_fetch_open_meteo, OPEN_METEO_BATCH)],
                 "open-elevation": [(_fetch_open_elevation, OPEN_ELEVATION_BATCH)],
                 "auto": [(_fetch_opentopodata, OPENTOPODATA_BATCH),
                          (_fetch_open_meteo, OPEN_METEO_BATCH),
                          (_fetch_open_elevation, OPEN_ELEVATION_BATCH)],
                 # interactive runs: the fast provider first (Open-Topo-Data is rate-limited to 1 batch/s)
                 "fast": [(_fetch_open_meteo, OPEN_METEO_BATCH),
                          (_fetch_opentopodata, OPENTOPODATA_BATCH)]}[provider]

    for fn, batch in providers:
        todo = np.where(np.isnan(out))[0]
        n_done = 0
        for s in range(0, len(todo), batch):
            idx = todo[s:s + batch]
            if t_end is not None and time.monotonic() > t_end:
                raise RuntimeError(f"elevation fetch exceeded its {deadline_s:.0f} s time budget")
            for attempt in range(retries):
                try:
                    out[idx] = fn(lats[idx], lons[idx], **kw)
                    break
                except (requests.RequestException, ValueError, KeyError) as exc:
                    wait = _retry_wait(exc, attempt)
                    logger.warning("%s batch %d failed (%s); retry in %.0fs",
                                   fn.__name__, s // batch, str(exc)[:120], wait)
                    time.sleep(wait)
            else:
                logger.error("%s gave up on batch %d", fn.__name__, s // batch)
                break                                   # fall through to next provider
            n_done += len(idx)
            if progress:
                progress(n_done, len(todo))
            time.sleep(max(pause_s, OPENTOPODATA_MIN_INTERVAL_S) if fn is _fetch_opentopodata else pause_s)
        if not np.isnan(out).any():
            break
    if np.isnan(out).any():
        raise RuntimeError(f"Elevation missing for {int(np.isnan(out).sum())}/{out.size} points "
                           "- check your internet connection and re-run (it is safe to repeat).")
    return out


def fetch_elevations_cached(lats: np.ndarray, lons: np.ndarray, cache_path: Path,
                            chunk: int = 100, **kwargs) -> np.ndarray:
    """fetch_elevations with a resumable on-disk cache. Every elevation that
    arrives is appended to `cache_path` immediately, so a rate-limit or network
    failure part-way through loses nothing: re-running fetches only what is
    still missing."""
    lats = np.asarray(lats, float)
    lons = np.asarray(lons, float)
    cache_path = Path(cache_path)
    key = lambda a, b: (round(float(a), 5), round(float(b), 5))       # noqa: E731
    known: dict = {}
    if cache_path.exists():
        try:
            df = pd.read_csv(cache_path)
            known = {key(a, b): float(e) for a, b, e in zip(df["latitude"], df["longitude"], df["elevation"])}
        except Exception:
            known = {}
    keys = [key(a, b) for a, b in zip(lats, lons)]
    todo = [i for i, k in enumerate(keys) if k not in known]
    if todo:
        logger.info("Elevation cache: %d of %d points already known, fetching %d",
                    len(keys) - len(todo), len(keys), len(todo))
    for s in range(0, len(todo), chunk):
        idx = todo[s:s + chunk]
        try:
            vals = fetch_elevations(lats[idx], lons[idx], **kwargs)
        except RuntimeError as exc:
            raise RuntimeError(f"{exc} Progress is saved in {cache_path.name}; run the same command again "
                               "in a few minutes and it will continue where it stopped.") from exc
        rows = pd.DataFrame({"latitude": [keys[i][0] for i in idx], "longitude": [keys[i][1] for i in idx],
                             "elevation": vals})
        rows.to_csv(cache_path, mode="a", header=not cache_path.exists(), index=False)
        known.update({keys[i]: float(v) for i, v in zip(idx, vals)})
        print(f"  {min(s + chunk, len(todo))}/{len(todo)} new points fetched", flush=True)
    return np.array([known[k] for k in keys])


# ── terrain table ─────────────────────────────────────────────────────────── #

def subgrid_points(grid: pd.DataFrame, resolution_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """The 9 sample points per cell, shape (n_cells, 3, 3) each for lat and lon
    (axis 1 = south->north, axis 2 = west->east)."""
    step = resolution_deg / 3.0
    lat0 = grid["latitude"].to_numpy(float)[:, None, None]
    lon0 = grid["longitude"].to_numpy(float)[:, None, None]
    lat = lat0 + step * _SUB[None, :, None] + np.zeros((1, 1, 3))
    lon = lon0 + step * _SUB[None, None, :] + np.zeros((1, 3, 1))
    return lat, lon


def slope_stats(elev: np.ndarray, lat_centre: np.ndarray, resolution_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """elev: (n_cells, 3, 3) metres. Returns (mean_elevation_m, mean_slope_pct)."""
    step_deg = resolution_deg / 3.0
    dy = step_deg * METERS_PER_DEG_LAT                                   # metres between sub-rows
    dx = step_deg * METERS_PER_DEG_LAT * np.cos(np.radians(lat_centre))  # metres between sub-cols
    gy = np.gradient(elev, axis=1) / dy
    gx = np.gradient(elev, axis=2) / dx[:, None, None]
    slope = np.sqrt(gx ** 2 + gy ** 2) * 100.0
    return elev.mean(axis=(1, 2)), slope.mean(axis=(1, 2))


def build_terrain_table(grid: pd.DataFrame, resolution_deg: float,
                        fetch: Callable[[np.ndarray, np.ndarray], np.ndarray] = fetch_elevations
                        ) -> pd.DataFrame:
    """grid: zone_id, latitude, longitude (cell centres). Returns a table with
    zone_id, latitude, longitude, elev_m, slope_pct."""
    lat, lon = subgrid_points(grid, resolution_deg)
    elev = np.asarray(fetch(lat.ravel(), lon.ravel()), float).reshape(lat.shape)
    if np.isnan(elev).any():
        raise ValueError("elevation fetch returned NaN values")
    mean_elev, slope = slope_stats(elev, grid["latitude"].to_numpy(float), resolution_deg)
    out = grid[["zone_id", "latitude", "longitude"]].copy()
    out["elev_m"] = mean_elev.round(1)
    out["slope_pct"] = slope.round(2)
    return out


class TerrainTable:
    """Cached per-cell terrain, looked up by rounded coordinates so it can be
    shared by training and live inference. Cells with no entry (a region whose
    terrain hasn't been built) get NaN; XGBoost handles NaN natively, and
    `available` tells callers whether anything real was loaded."""

    def __init__(self, table: Optional[pd.DataFrame] = None):
        self.table = table if table is not None and not table.empty else None
        self._index = None
        if self.table is not None:
            self._index = dict(zip(self._key(self.table["latitude"], self.table["longitude"]),
                                   range(len(self.table))))

    @staticmethod
    def _key(lat, lon):
        return list(zip(np.round(np.asarray(lat, float), 3), np.round(np.asarray(lon, float), 3)))

    @property
    def available(self) -> bool:
        return self.table is not None

    @classmethod
    def load(cls, models_dir: Path, region_name: Optional[str] = None) -> "TerrainTable":
        """Terrain for one region (its own terrain_<region>.csv) when
        `region_name` is given and that file exists; otherwise the union of
        every terrain_*.csv in models_dir. Per-region matters because
        neighbouring regions can share a border cell coordinate at different
        grid resolutions, and the slope differs between them."""
        if region_name:
            own = Path(models_dir) / terrain_filename(region_name)
            if own.exists():
                df = pd.read_csv(own)
                return cls(df if not df.empty else None)
        frames = [pd.read_csv(p) for p in sorted(Path(models_dir).glob(TERRAIN_GLOB))]
        frames = [f for f in frames if not f.empty]
        return cls(pd.concat(frames, ignore_index=True) if frames else None)

    def transform(self, grid: pd.DataFrame) -> pd.DataFrame:
        """zone_id + TERRAIN_COLUMNS for the given cells (NaN where unknown)."""
        out = pd.DataFrame({"zone_id": grid["zone_id"].values})
        for c in TERRAIN_COLUMNS:
            out[c] = np.nan
        if self.table is None:
            return out
        pos = np.array([self._index.get(k, -1) for k in self._key(grid["latitude"], grid["longitude"])])
        ok = pos >= 0
        for c in TERRAIN_COLUMNS:
            vals = self.table[c].to_numpy(float)
            out[c] = np.where(ok, vals[np.clip(pos, 0, None)], np.nan)
        return out

    def coverage(self, grid: pd.DataFrame) -> float:
        """Fraction of the grid's cells that have real terrain."""
        if self.table is None:
            return 0.0
        pos = [self._index.get(k, -1) for k in self._key(grid["latitude"], grid["longitude"])]
        return float(np.mean(np.array(pos) >= 0))


def terrain_filename(region_name: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in region_name).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return f"terrain_{slug}.csv"
