"""
worldcover.py - ESA WorldCover 10 m v200 (2021): the primary land-cover source.

Data: Cloud-Optimised GeoTIFFs in the public AWS bucket `esa-worldcover`
(EPSG:4326, 1/12000-degree pixels, one file per 3 x 3 degree tile named by its
south-west corner, e.g. N09E075 covers 9-12 N, 75-78 E and so all of Bandipur).
Only the COG blocks intersecting each requested 0.01-degree cache tile are read
(HTTP range requests through rasterio/GDAL); the 3-degree file is never
downloaded as a whole.

Why rasterio: reading a window of a remote COG needs the TIFF tile index, the
DEFLATE blocks and their geotransform; rasterio (GDAL) does that robustly and is
also used to reproject Sentinel-2 windows from UTM. It is imported lazily, so
the rest of the application works (and says WorldCover is unavailable) when the
package is not installed.

WorldCover class codes (v200):
  10 tree cover, 20 shrubland, 30 grassland, 40 cropland, 50 built-up,
  60 bare / sparse vegetation, 70 snow and ice, 80 permanent water bodies,
  90 herbaceous wetland, 95 mangroves, 100 moss and lichen, 0 no data.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.config import LAND_COVER
from src.landcover.tile_cache import RasterWindow, Tile, TileCache, load_tiled

logger = logging.getLogger(__name__)

DATASET = "worldcover"
WC_NAMES = {0: "no data", 10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland", 50: "built-up",
            60: "bare / sparse vegetation", 70: "snow and ice", 80: "permanent water", 90: "herbaceous wetland",
            95: "mangroves", 100: "moss and lichen"}


def px_per_tile() -> int:
    return int(round(LAND_COVER.cache_tile_deg / LAND_COVER.worldcover_res_deg))


def wc_tile_name(lat: float, lon: float) -> str:
    """Name of the 3-degree WorldCover file containing (lat, lon)."""
    la = int(np.floor(lat / 3.0) * 3)
    lo = int(np.floor(lon / 3.0) * 3)
    return f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"


def tile_url(lat: float, lon: float) -> str:
    return LAND_COVER.worldcover_url.format(tile=wc_tile_name(lat, lon))


def _gdal_env():
    return dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                GDAL_HTTP_TIMEOUT=str(int(LAND_COVER.http_timeout_s)), GDAL_HTTP_MAX_RETRY="2",
                GDAL_HTTP_RETRY_DELAY="1", VSI_CACHE="TRUE")


def fetch_tiles(tiles: List[Tile]) -> Dict[str, Tuple[Dict[str, np.ndarray], dict]]:
    """Read the cache tiles from the WorldCover COGs (one open file per
    3-degree tile). Raises on any failure (the caller records it)."""
    import rasterio                                   # lazy: optional dependency (see module docstring)
    from rasterio.windows import Window
    px = px_per_tile()
    out: Dict[str, Tuple[Dict[str, np.ndarray], dict]] = {}
    groups: Dict[str, List[Tile]] = {}
    for t in tiles:
        groups.setdefault(tile_url(t.south + t.deg / 2, t.west + t.deg / 2), []).append(t)
    with rasterio.Env(**_gdal_env()):
        for url, group in groups.items():
            with rasterio.open(url) as ds:
                tr = ds.transform
                for t in group:
                    # integer window on the native lattice (the cache tiles are lattice-aligned)
                    col = int(round((t.west - tr.c) / tr.a))
                    row = int(round((t.north - tr.f) / tr.e))
                    win = Window(col, row, px, px)
                    arr = ds.read(1, window=win, boundless=True, fill_value=0)
                    out[t.tile_id] = ({"classes": arr.astype(np.uint8)},
                                      {"source": "ESA WorldCover", "url": url, "res_deg": LAND_COVER.worldcover_res_deg,
                                       "processing": "window read, nearest (native lattice)"})
    return out


def load_window(bounds: Dict[str, float], allow_fetch: bool = True,
                cache: Optional[TileCache] = None) -> Tuple[Optional[RasterWindow], dict]:
    """WorldCover classes covering `bounds` (uint8, 0 = no data) and a status
    dict (status, tiles cached / downloaded / missing, retrieval dates, error)."""
    if not LAND_COVER.use_worldcover:
        return None, {"status": "disabled", "error": "disabled in configuration"}
    win, st = load_tiled(DATASET, LAND_COVER.worldcover_version, bounds, LAND_COVER.worldcover_ttl_days,
                         px_per_tile(), "classes", np.uint8, 0, fetch_tiles, allow_fetch, cache)
    if win is not None:
        when = (st.get("retrieved_last") or "")[:10]
        win.label = f"ESA WorldCover 10 m {LAND_COVER.worldcover_version}" + (f" (retrieved {when})" if when else "")
    if st.get("error"):
        logger.info("WorldCover: %s", st["error"])
    return win, st
