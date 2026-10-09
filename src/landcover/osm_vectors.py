"""
osm_vectors.py - OpenStreetMap as the detailed vector refinement layer.

The Overpass download and its disk cache are the existing ones
(src/simulation/fuel_map.fetch_osm). This module turns the features into
SUB-CELL coverage on the s x s supersample of every CA cell, so the fusion can
reason with area fractions instead of "touched = blocked":

  water            natural=water, reservoirs, riverbanks, river / canal lines
  building         building=* polygons (always an explicit barrier)
  landuse_built    residential / commercial / industrial ... land-use polygons
  road             highway=* and railways, centre line buffered by its width
  bare             bare rock, sand, scree, quarries, runways

Continuous firebreaks (config.LandCoverConfig.road_block_classes, railways and
blocking_waterways): every cell their centre line crosses is also marked at
CELL level with a 4-connected supercover, so an 8-neighbour fire cannot slip
through diagonally. Minor roads and tracks only count by their area fraction:
a 4-6 m forest track does not block a 25 m vegetation cell.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config.config import LAND_COVER
from src.landcover.grid import GridSpec
from src.simulation import fuel_map
from src.simulation.fuel_map import BUILT, NON_FUEL, ROAD, WATER

LAYERS = ("water", "building", "landuse_built", "road", "bare")


@dataclass
class OSMLayers:
    sub: Dict[str, np.ndarray]                 # layer -> (rows*s, cols*s) bool
    block_road: np.ndarray                     # (rows, cols) bool, continuous road / rail firebreak
    block_water: np.ndarray                    # (rows, cols) bool, river centre lines
    touched: np.ndarray                        # (rows, cols) bool, any OSM feature in the cell
    n_features: int = 0
    by_layer: Dict[str, int] = field(default_factory=dict)


def feature_layer(tags: dict) -> Optional[Tuple[str, str, float, bool]]:
    """(layer, 'area' | 'line', width_m, continuous_barrier) or None."""
    fc = fuel_map.feature_class(tags or {})
    if fc is None:
        return None
    cls, kind, width = fc
    t = tags or {}
    if cls == WATER:
        return "water", kind, width, kind == "line" and t.get("waterway") in LAND_COVER.blocking_waterways
    if cls == BUILT:
        return ("building" if "building" in t else "landuse_built"), kind, width, False
    if cls == ROAD:
        major = (t.get("highway") in LAND_COVER.road_block_classes
                 or (LAND_COVER.railway_blocks and t.get("railway") is not None))
        return "road", kind, width, bool(major and kind == "line")
    if cls == NON_FUEL:
        return "bare", kind, width, False
    return None


def _fill_ring(mask: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Even-odd fill of one ring (sub-grid units) into a new bool array of
    mask's shape (sample points at index + 0.5)."""
    out = np.zeros(mask.shape, dtype=bool)
    if len(xs) < 3:
        return out
    rows, cols = mask.shape
    r0, r1 = max(0, int(math.floor(ys.min()))), min(rows, int(math.ceil(ys.max())) + 1)
    c0, c1 = max(0, int(math.floor(xs.min()))), min(cols, int(math.ceil(xs.max())) + 1)
    if r0 >= r1 or c0 >= c1:
        return out
    qy = (np.arange(r0, r1) + 0.5)[:, None]
    qx = (np.arange(c0, c1) + 0.5)[None, :]
    inside = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    xj, yj = np.roll(xs, 1), np.roll(ys, 1)
    for xi_, yi_, xj_, yj_ in zip(xs, ys, xj, yj):
        if yi_ == yj_:
            continue
        cond = (yi_ > qy) != (yj_ > qy)                 # (n, 1)
        if not cond.any():
            continue
        xint = xi_ + (qy - yi_) * (xj_ - xi_) / (yj_ - yi_)
        inside ^= cond & (qx < xint)
    out[r0:r1, c0:c1] = inside
    return out


def _segment_band(mask: np.ndarray, x0, y0, x1, y1, half: float):
    """Mark sample points within `half` of the segment (all in grid units)."""
    rows, cols = mask.shape
    r0, r1 = max(0, int(min(y0, y1) - half - 1)), min(rows, int(max(y0, y1) + half + 2))
    c0, c1 = max(0, int(min(x0, x1) - half - 1)), min(cols, int(max(x0, x1) + half + 2))
    if r0 >= r1 or c0 >= c1:
        return
    py = (np.arange(r0, r1) + 0.5)[:, None]
    px = (np.arange(c0, c1) + 0.5)[None, :]
    vx, vy = x1 - x0, y1 - y0
    L2 = vx * vx + vy * vy
    t = np.clip(((px - x0) * vx + (py - y0) * vy) / L2, 0, 1) if L2 > 0 else 0.0
    d = np.hypot(px - (x0 + t * vx), py - (y0 + t * vy))
    mask[r0:r1, c0:c1] |= d <= half


def rasterize_fractional(elements: Sequence[dict], spec: GridSpec, s: int) -> OSMLayers:
    R, C = spec.n_rows, spec.n_cols
    sub = {k: np.zeros((R * s, C * s), dtype=bool) for k in LAYERS}
    block_road = np.zeros((R, C), dtype=bool)
    block_water = np.zeros((R, C), dtype=bool)
    touched_sub = np.zeros((R * s, C * s), dtype=bool)
    dlat, dlon = spec.dlat, spec.dlon

    def to_xy(lat, lon, scale):
        return ((np.asarray(lon, dtype=float) - spec.west) / dlon * scale,
                (spec.north - np.asarray(lat, dtype=float)) / dlat * scale)

    used, by_layer = 0, {k: 0 for k in LAYERS}
    for el in elements or []:
        fl = feature_layer(el.get("tags") or {})
        if fl is None:
            continue
        layer, kind, width_m, barrier = fl
        if el.get("type") == "relation":
            outers, inners = fuel_map._rings_from_members(el.get("members") or [])
            if not outers:
                continue
            m = np.zeros((R * s, C * s), dtype=bool)
            for ring in outers:
                xs, ys = to_xy([p[0] for p in ring], [p[1] for p in ring], s)
                m |= _fill_ring(m, xs, ys)
            for ring in inners:
                xs, ys = to_xy([p[0] for p in ring], [p[1] for p in ring], s)
                m &= ~_fill_ring(m, xs, ys)
            if m.any():
                sub[layer] |= m
                touched_sub |= m
            used += 1
            by_layer[layer] += 1
            continue
        geom = el.get("geometry") or []
        if len(geom) < 2:
            continue
        lat = [p["lat"] for p in geom]
        lon = [p["lon"] for p in geom]
        xs, ys = to_xy(lat, lon, s)
        if (xs.max() < -s or xs.min() > C * s + s or ys.max() < -s or ys.min() > R * s + s):
            continue                                            # entirely outside this grid
        closed = len(geom) >= 4 and lat[0] == lat[-1] and lon[0] == lon[-1]
        if kind == "area" and closed:
            m = _fill_ring(sub[layer], xs, ys)
            mx, my = float(np.mean(xs[:-1])), float(np.mean(ys[:-1]))   # tiny polygons (huts) still count
            if 0 <= my < R * s and 0 <= mx < C * s:
                m[int(my), int(mx)] = True
            sub[layer] |= m
            touched_sub |= m
        else:
            half_sub = width_m / 2.0 / (spec.cell_m / s)
            band = np.zeros((R * s, C * s), dtype=bool)
            for k in range(len(xs) - 1):
                _segment_band(band, xs[k], ys[k], xs[k + 1], ys[k + 1], half_sub)
            sub[layer] |= band
            touched_sub |= band
            if barrier:
                cx, cy = to_xy(lat, lon, 1)
                target = block_water if layer == "water" else block_road
                half_cells = width_m / 2.0 / spec.cell_m
                for k in range(len(cx) - 1):
                    x0, y0, x1, y1 = cx[k], cy[k], cx[k + 1], cy[k + 1]
                    if max(x0, x1) < -1 or min(x0, x1) > C + 1 or max(y0, y1) < -1 or min(y0, y1) > R + 1:
                        continue
                    for r, c in fuel_map._supercover(x0, y0, x1, y1, R, C):
                        target[r, c] = True
                    if half_cells > 0.5:
                        _segment_band(target, x0, y0, x1, y1, half_cells)
        used += 1
        by_layer[layer] += 1
    touched = touched_sub.reshape(R, s, C, s).any(axis=(1, 3)) | block_road | block_water
    return OSMLayers(sub, block_road, block_water, touched, used, by_layer)


OSM_TILE_DEG = 0.02                       # Overpass requests are made per 0.02-degree tile (~2.2 km)
_LAST_FAILURE = [0.0]


def load_osm(spec: GridSpec, allow_fetch: bool = True) -> Tuple[Optional[dict], dict]:
    """Overpass features for the grid, plus a status dict. Features are fetched
    and cached per 0.02-degree tile (so the strips of an expanding domain reuse
    them) and merged; an existing cache file for exactly this area is used
    first. After an Overpass failure no new tile is requested for a few minutes
    (cached tiles are still used), so an outage never stalls a simulation."""
    import time
    from src.landcover.tile_cache import tiles_for
    if not LAND_COVER.use_osm:
        return None, {"status": "disabled", "error": "disabled in configuration"}
    exact = fuel_map._cache_path(fuel_map._snapped_bbox(spec.bounds))
    if exact.exists():
        data = fuel_map.fetch_osm(fuel_map._snapped_bbox(spec.bounds), allow_fetch=False)
        if data is not None and not data.get("_stale"):
            return data, {"status": "cached", "retrieved": str(data.get("fetched_utc", ""))[:10], "error": ""}
    fetch_ok = allow_fetch and time.time() - _LAST_FAILURE[0] > LAND_COVER.retry_after_failure_s
    merged: Dict[tuple, dict] = {}
    dates, missing, stale, n = [], 0, False, 0
    for t in tiles_for(spec.bounds, OSM_TILE_DEG):
        n += 1
        bbox = (round(t.south, 4), round(t.west, 4), round(t.north, 4), round(t.east, 4))
        data = fuel_map.fetch_osm(bbox, allow_fetch=fetch_ok)
        if data is None:
            missing += 1
            if fetch_ok and os.getenv("FIRE_LAND_COVER_FETCH", "1") != "0":
                _LAST_FAILURE[0] = time.time()
                fetch_ok = False
            continue
        stale |= bool(data.get("_stale"))
        dates.append(str(data.get("fetched_utc", "")))
        for el in data.get("elements", []):
            merged[(el.get("type"), el.get("id"), len(merged) if el.get("id") is None else 0)] = el
    if n == missing:
        return None, {"status": "unavailable", "tiles": n, "missing": missing,
                      "error": "OpenStreetMap (Overpass) could not be reached and no cached copy exists for this area"}
    st = "partial" if missing else ("stale" if stale else "cached")
    return ({"fetched_utc": min(dates) if dates else "", "elements": list(merged.values()), "_stale": stale},
            {"status": st, "tiles": n, "missing": missing, "retrieved": (min(dates) if dates else "")[:10],
             "error": f"{missing} of {n} OSM tiles unavailable" if missing else ""})
