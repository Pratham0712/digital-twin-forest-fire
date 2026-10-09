"""
sentinel2_ndvi.py - Sentinel-2 L2A NDVI: vegetation condition per 10 m pixel.

NDVI = (NIR - Red) / (NIR + Red) with Red = B04 and NIR = B08 (10 m), surface
reflectance from the L2A product. Clouds, cloud shadows, snow and no-data are
removed with the L2A Scene Classification Layer (SCL, 20 m; kept classes in
LAND_COVER.sentinel2_scl_keep). The cloud-free observations of up to
LAND_COVER.sentinel2_max_scenes recent scenes (the least cloudy within
LAND_COVER.sentinel2_lookback_days) are combined with a per-pixel median.

Data access (no account, no key): STAC search on Element84's Earth Search
(collection sentinel-2-l2a), then windowed reads of the COG assets on AWS
(rasterio, HTTP range requests). Each band is read only for the window of the
missing cache tiles and reprojected from the scene's UTM CRS to the
1/12000-degree lat/lon lattice shared with WorldCover.

NDVI is a vegetation-condition proxy. It is NOT fuel load: the fusion turns it
into a RELATIVE fuel-availability factor (documented in config.LandCoverConfig)
and labels it "derived from Sentinel-2 NDVI".
"""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.config import LAND_COVER
from src.landcover.tile_cache import RasterWindow, Tile, TileCache, iso, load_tiled, utc_now

logger = logging.getLogger(__name__)

DATASET = "s2_ndvi"
PROCESSING_VERSION = "ndvi-median-1"


def px_per_tile() -> int:
    return int(round(LAND_COVER.cache_tile_deg / LAND_COVER.worldcover_res_deg))


def stac_search(bbox: Tuple[float, float, float, float]) -> List[dict]:
    """Least-cloudy recent L2A items intersecting bbox (west, south, east, north)."""
    import requests
    end = utc_now()
    start = end - timedelta(days=LAND_COVER.sentinel2_lookback_days)
    body = {"collections": [LAND_COVER.sentinel2_collection], "bbox": list(bbox),
            "datetime": f"{iso(start)}/{iso(end)}", "limit": 50,
            "query": {"eo:cloud_cover": {"lt": LAND_COVER.sentinel2_max_cloud_pct}}}
    r = requests.post(LAND_COVER.stac_url, json=body, timeout=LAND_COVER.http_timeout_s,
                      headers={"User-Agent": "forest-fire-digital-twin/1.0 (BMSCE capstone)"})
    r.raise_for_status()
    feats = r.json().get("features", [])
    feats.sort(key=lambda f: (f.get("properties", {}).get("eo:cloud_cover", 100.0),
                              f.get("properties", {}).get("datetime", "")))
    return feats[:LAND_COVER.sentinel2_max_scenes]


def _asset(item: dict, *names) -> Optional[dict]:
    a = item.get("assets", {})
    for n in names:
        if n in a:
            return a[n]
    return None


def _scale_offset(item: dict, asset: dict) -> Tuple[float, float]:
    """DN -> reflectance. Earth Search publishes raster:bands scale/offset;
    otherwise processing baseline >= 04.00 has a -1000 DN offset."""
    rb = (asset.get("raster:bands") or [{}])[0]
    if "scale" in rb:
        return float(rb.get("scale", 1e-4)), float(rb.get("offset", 0.0))
    base = str(item.get("properties", {}).get("s2:processing_baseline", "0"))
    try:
        newer = float(base) >= 4.0
    except ValueError:
        newer = False
    return 1e-4, (-0.1 if newer else 0.0)


def _read_reprojected(href: str, west: float, south: float, east: float, north: float, shape, resampling):
    import rasterio
    from rasterio.warp import Resampling, reproject, transform_bounds
    from rasterio.transform import from_origin
    from rasterio.windows import from_bounds
    with rasterio.open(href) as ds:
        l, b, r, t = transform_bounds("EPSG:4326", ds.crs, west, south, east, north, densify_pts=21)
        pad = 3 * abs(ds.transform.a)
        win = from_bounds(l - pad, b - pad, r + pad, t + pad, ds.transform).round_offsets().round_lengths()
        src = ds.read(1, window=win, boundless=True, fill_value=0)
        src_tr = ds.window_transform(win)
        dst = np.zeros(shape, dtype=np.float32)
        reproject(src.astype(np.float32), dst, src_transform=src_tr, src_crs=ds.crs, src_nodata=0,
                  dst_transform=from_origin(west, north, LAND_COVER.worldcover_res_deg,
                                            LAND_COVER.worldcover_res_deg),
                  dst_crs="EPSG:4326", dst_nodata=0,
                  resampling=getattr(Resampling, resampling))
    return dst


def fetch_tiles(tiles: List[Tile]) -> Dict[str, Tuple[Dict[str, np.ndarray], dict]]:
    """Median cloud-free NDVI for the missing tiles (one read per band per scene
    for the union window of the tiles). Raises on failure."""
    import rasterio
    px = px_per_tile()
    deg = tiles[0].deg
    i0, i1 = min(t.i_lat for t in tiles), max(t.i_lat for t in tiles)
    j0, j1 = min(t.i_lon for t in tiles), max(t.i_lon for t in tiles)
    south, north, west, east = i0 * deg, (i1 + 1) * deg, j0 * deg, (j1 + 1) * deg
    shape = ((i1 - i0 + 1) * px, (j1 - j0 + 1) * px)
    items = stac_search((west, south, east, north))
    if not items:
        raise LookupError(f"no Sentinel-2 L2A scene with < {LAND_COVER.sentinel2_max_cloud_pct:g}% cloud in the "
                          f"last {LAND_COVER.sentinel2_lookback_days} days")
    stack, scenes = [], []
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES",
               GDAL_HTTP_TIMEOUT=str(int(LAND_COVER.http_timeout_s)), GDAL_HTTP_MAX_RETRY="2", VSI_CACHE="TRUE")
    with rasterio.Env(**env):
        for it in items:
            red_a, nir_a, scl_a = _asset(it, "red", "B04"), _asset(it, "nir", "B08"), _asset(it, "scl", "SCL")
            if not (red_a and nir_a and scl_a):
                continue
            red = _read_reprojected(red_a["href"], west, south, east, north, shape, "bilinear")
            nir = _read_reprojected(nir_a["href"], west, south, east, north, shape, "bilinear")
            scl = _read_reprojected(scl_a["href"], west, south, east, north, shape, "nearest")
            rs, ro = _scale_offset(it, red_a)
            ns, no = _scale_offset(it, nir_a)
            ok = (red > 0) & (nir > 0) & np.isin(np.rint(scl).astype(int), LAND_COVER.sentinel2_scl_keep)
            r = red * rs + ro
            n = nir * ns + no
            with np.errstate(invalid="ignore", divide="ignore"):
                ndvi = np.where(ok & ((n + r) > 0), (n - r) / (n + r), np.nan)
            stack.append(np.clip(ndvi, -1, 1).astype(np.float32))
            p = it.get("properties", {})
            scenes.append({"id": it.get("id"), "datetime": p.get("datetime"),
                           "cloud_cover": p.get("eo:cloud_cover")})
    if not stack:
        raise LookupError("no usable Sentinel-2 scene (missing assets)")
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            comp = np.nanmedian(np.stack(stack), axis=0).astype(np.float32)
    out = {}
    for t in tiles:
        r0 = (i1 - t.i_lat) * px
        c0 = (t.i_lon - j0) * px
        tile = comp[r0:r0 + px, c0:c0 + px].copy()
        if not np.isfinite(tile).any():
            continue                                  # no cloud-free pixel: not cached, retried next time
        out[t.tile_id] = ({"ndvi": tile},
                          {"source": "Sentinel-2 L2A (Earth Search STAC)", "scenes": scenes,
                           "processing": PROCESSING_VERSION, "res_deg": LAND_COVER.worldcover_res_deg})
    return out


def load_window(bounds: Dict[str, float], allow_fetch: bool = True,
                cache: Optional[TileCache] = None) -> Tuple[Optional[RasterWindow], dict]:
    """Median NDVI covering bounds (float32, NaN = no cloud-free observation)."""
    if not LAND_COVER.use_sentinel2:
        return None, {"status": "disabled", "error": "disabled in configuration"}
    win, st = load_tiled(DATASET, PROCESSING_VERSION, bounds, LAND_COVER.ndvi_ttl_days, px_per_tile(), "ndvi",
                         np.float32, np.nan, fetch_tiles, allow_fetch, cache)
    if win is not None:
        scenes = []
        for m in win.meta.get("tile_meta", {}).values():
            scenes += [s.get("datetime", "")[:10] for s in m.get("scenes", []) if s.get("datetime")]
        dates = sorted(set(scenes))
        win.label = ("Sentinel-2 L2A NDVI (median of cloud-free scenes"
                     + (f" {dates[0]} to {dates[-1]}" if len(dates) > 1 else f" {dates[0]}" if dates else "") + ")")
        win.meta["scene_dates"] = dates
    if st.get("error"):
        logger.info("Sentinel-2 NDVI: %s", st["error"])
    return win, st
