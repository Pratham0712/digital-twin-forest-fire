"""
fuel_map.py - land-cover / fuel classification of the local simulation domain.

The fire-spread CA (src/simulation/cellular_automata.py) already has a
NON_FUEL cell state: such cells never ignite and fire never propagates into
them. Until now the local simulation filled that mask from one regional NDVI
value, so it was either all-fuel or all-non-fuel. This module builds a real,
per-cell mask from a geospatial data layer, so fire stops at rivers, lakes,
roads and buildings.

Data source
-----------
OpenStreetMap, through the Overpass API: explicitly tagged features
(natural=water, waterway=river, building=*, highway=*, landuse=residential,
natural=bare_rock, ...). The Google satellite imagery is only the visual base
and is NEVER used for classification (no "green pixel = forest" rule).
Responses are cached on disk (models/land_cover/), so an area is fetched once.
When the data cannot be fetched the mask is all-fuel and the source is
reported as "unavailable" - the UI says so; nothing is guessed.

Classes (priority when several apply: WATER > BUILT > ROAD > NON_FUEL > FUEL)
  FUEL      vegetation / anything not mapped as one of the classes below
  NON_FUEL  bare rock, sand, scree, quarries, runways
  WATER     lakes, reservoirs, rivers (polygons and river/canal centre lines)
  BUILT     buildings, residential / industrial / commercial land
  ROAD      paved road classes and railways (centre line buffered by a
            typical width; forest tracks and footpaths are NOT barriers)

Rasterisation (no GIS dependencies)
  * polygons: a cell belongs to a polygon when its centre is inside it
    (outer rings minus inner rings);
  * lines (roads, rivers, canals): every cell the centre line passes through
    (a 4-connected "supercover" traversal, so a one-cell-wide barrier cannot be
    crossed diagonally by the 8-neighbour CA) plus cells whose centre lies
    within half the feature width;
  * small polygons (buildings) also mark the cell containing their centroid.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config.config import MODELS_DIR

logger = logging.getLogger(__name__)

FUEL, NON_FUEL, WATER, BUILT, ROAD = 0, 1, 2, 3, 4
# UNKNOWN: no land-cover information for the cell (e.g. WorldCover unavailable and
# no OpenStreetMap feature). Not confirmed vegetation: never burnable unless the
# user explicitly runs in DEGRADED LAND-COVER MODE (src/landcover/provider.py).
UNKNOWN = 5
CLASS_NAMES = {FUEL: "fuel", NON_FUEL: "non-fuel (bare)", WATER: "water", BUILT: "built-up", ROAD: "road",
               UNKNOWN: "unverified (no land-cover data)"}
# short names used in ignition messages ("IGNITION REJECTED — ROAD")
CLASS_TAGS = {FUEL: "FUEL", NON_FUEL: "BARE / NON-FUEL", WATER: "WATER", BUILT: "BUILT", ROAD: "ROAD",
              UNKNOWN: "LAND COVER UNVERIFIED"}
PRIORITY = (NON_FUEL, ROAD, BUILT, WATER)             # painted in this order; later wins

LAND_COVER_DIR = MODELS_DIR / "land_cover"
OVERPASS_URLS = ("https://overpass-api.de/api/interpreter",
                 "https://overpass.kumi.systems/api/interpreter")
SNAP_DEG = 0.005                                      # cache tiles snap outward to this grid
TIMEOUT_S = 30
_RETRY_AFTER_S = 300
_FETCH_FAILED_AT: Dict[str, float] = {}

# Typical carriageway / channel widths (m) used to buffer centre lines.
ROAD_WIDTH_M = {"motorway": 20, "motorway_link": 10, "trunk": 14, "trunk_link": 8, "primary": 10,
                "primary_link": 7, "secondary": 8, "secondary_link": 6, "tertiary": 7, "tertiary_link": 5,
                "unclassified": 5, "residential": 6, "living_street": 5, "service": 4}
RAIL_WIDTH_M = 6
WATERWAY_WIDTH_M = {"river": 20, "canal": 8}

ROAD_RE = "|".join(ROAD_WIDTH_M)
QUERY = """[out:json][timeout:25];
(
  way["natural"="water"]({bb}); relation["natural"="water"]({bb});
  way["water"]({bb}); relation["water"]({bb});
  way["waterway"~"^(riverbank|dock)$"]({bb});
  way["waterway"~"^(river|canal)$"]({bb});
  way["landuse"~"^(reservoir|basin)$"]({bb}); relation["landuse"~"^(reservoir|basin)$"]({bb});
  way["building"]({bb});
  way["landuse"~"^(residential|industrial|commercial|retail|railway|construction|garages)$"]({bb});
  relation["landuse"~"^(residential|industrial|commercial|retail)$"]({bb});
  way["highway"~"^(""" + ROAD_RE + """)$"]({bb});
  way["railway"~"^(rail|light_rail|narrow_gauge)$"]({bb});
  way["natural"~"^(bare_rock|scree|sand|beach|shingle|glacier)$"]({bb});
  relation["natural"~"^(bare_rock|scree|sand)$"]({bb});
  way["landuse"="quarry"]({bb});
  way["aeroway"~"^(runway|taxiway|apron)$"]({bb});
  way["natural"~"^(wood|scrub|heath|grassland|shrubbery)$"]({bb});
  way["landuse"~"^(forest|meadow|grass|orchard)$"]({bb});
);
out geom;"""
# v2: the query also returns mapped vegetation polygons (used only by the hypothetical
# ignition-point check, src/simulation/ignition_site.py - never as a fire mask)
QUERY_VERSION = "v2"


@dataclass
class LandCover:
    classes: np.ndarray                     # (rows, cols) int8, FUEL / NON_FUEL / WATER / BUILT / ROAD / UNKNOWN
    source: str                             # "osm" | "fused" | "unavailable" | "none"
    label: str                              # human-readable provenance
    n_features: int = 0
    counts: Dict[str, int] = field(default_factory=dict)
    # Fused land cover (src/landcover/provider.py); None for the legacy OSM-only mask.
    status: str = ""                        # "full" | "degraded" | "unavailable"
    fuel_load: Optional[np.ndarray] = None  # (rows, cols) relative fuel load 0-1 (0 on non-burnable cells)
    fuel_source: Optional[np.ndarray] = None    # (rows, cols) int8, landcover.fusion.FUEL_SRC_*
    confidence: Optional[np.ndarray] = None     # (rows, cols) int8 0 n/a, 1 LOW, 2 MEDIUM, 3 HIGH
    source_mask: Optional[np.ndarray] = None    # (rows, cols) uint8 bits, landcover.fusion.SRC_*
    fractions: Dict[str, np.ndarray] = field(default_factory=dict)
    sources: Dict[str, dict] = field(default_factory=dict)     # per data source: status / error / dates
    provenance: Dict[str, object] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def non_fuel(self) -> np.ndarray:
        """Cells that can never burn (UNKNOWN included: unverified is not fuel)."""
        return self.classes != FUEL

    def non_burnable(self, allow_unverified: bool = False) -> np.ndarray:
        """Non-fuel mask for the CA; allow_unverified=True (explicit DEGRADED
        LAND-COVER MODE) lets UNKNOWN cells burn."""
        if allow_unverified:
            return (self.classes != FUEL) & (self.classes != UNKNOWN)
        return self.classes != FUEL

    @property
    def available(self) -> bool:
        return self.source == "osm" or (self.source == "fused" and self.status in ("full", "degraded"))

    @property
    def has_unverified(self) -> bool:
        return bool((self.classes == UNKNOWN).any())


def all_fuel(shape: Tuple[int, int], source: str = "none", label: str = "No land-cover layer") -> LandCover:
    cls = np.zeros(shape, dtype=np.int8)
    return LandCover(cls, source, label, 0, _counts(cls))


def _counts(cls: np.ndarray) -> Dict[str, int]:
    return {CLASS_NAMES[k]: int((cls == k).sum()) for k in CLASS_NAMES}


# ── feature classification ───────────────────────────────────────────────── #

def feature_class(tags: dict) -> Optional[Tuple[int, str, float]]:
    """(class, geometry kind 'area' | 'line', line width m) for an OSM tag set,
    or None when the feature does not constrain fire."""
    t = tags or {}
    ww = t.get("waterway")
    if t.get("natural") == "water" or "water" in t or ww in ("riverbank", "dock") \
            or t.get("landuse") in ("reservoir", "basin"):
        return WATER, "area", 0.0
    if ww in WATERWAY_WIDTH_M:
        return WATER, "line", _width(t, WATERWAY_WIDTH_M[ww])
    if "building" in t:
        return BUILT, "area", 0.0
    if t.get("landuse") in ("residential", "industrial", "commercial", "retail", "railway", "construction",
                            "garages"):
        return BUILT, "area", 0.0
    hw = t.get("highway")
    if hw in ROAD_WIDTH_M:
        if t.get("area") == "yes":
            return ROAD, "area", 0.0
        return ROAD, "line", _width(t, ROAD_WIDTH_M[hw])
    if t.get("railway") in ("rail", "light_rail", "narrow_gauge"):
        return ROAD, "line", RAIL_WIDTH_M
    if t.get("natural") in ("bare_rock", "scree", "sand", "beach", "shingle", "glacier") \
            or t.get("landuse") == "quarry":
        return NON_FUEL, "area", 0.0
    if t.get("aeroway") in ("runway", "taxiway", "apron"):
        return NON_FUEL, ("area" if t.get("aeroway") == "apron" else "line"), 20.0
    return None


def _width(tags: dict, default: float) -> float:
    try:
        return max(1.0, float(str(tags.get("width", "")).split()[0].replace(",", ".")))
    except (ValueError, IndexError):
        return float(default)


# ── geometry helpers (lat/lon -> fractional grid coordinates) ─────────────── #

class _Grid:
    """Maps lat/lon to fractional (col, row) of the domain grid; row 0 = north."""

    def __init__(self, bounds: Dict[str, float], n_rows: int, n_cols: int, cell_m: float):
        self.b, self.n_rows, self.n_cols, self.cell_m = bounds, n_rows, n_cols, cell_m
        self.dlat = (bounds["north"] - bounds["south"]) / n_rows
        self.dlon = (bounds["east"] - bounds["west"]) / n_cols

    def frac(self, lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return (np.asarray(lon) - self.b["west"]) / self.dlon, (self.b["north"] - np.asarray(lat)) / self.dlat

    def centres(self) -> Tuple[np.ndarray, np.ndarray]:
        c, r = np.meshgrid(np.arange(self.n_cols) + 0.5, np.arange(self.n_rows) + 0.5)
        return c, r


def _points_in_ring(px: np.ndarray, py: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Even-odd (crossing-number) point-in-polygon of grid points px, py
    (2-D arrays) against one ring (N x 2, fractional col/row). Only the points
    inside the ring's bounding box are tested."""
    out = np.zeros(px.shape, dtype=bool)
    if len(ring) < 3:
        return out
    x, y = ring[:, 0], ring[:, 1]
    rows, cols = px.shape
    r0, r1 = max(0, int(math.floor(y.min()))), min(rows, int(math.ceil(y.max())) + 1)
    c0, c1 = max(0, int(math.floor(x.min()))), min(cols, int(math.ceil(x.max())) + 1)
    if r0 >= r1 or c0 >= c1:
        return out
    qx, qy = px[r0:r1, c0:c1], py[r0:r1, c0:c1]
    inside = np.zeros(qx.shape, dtype=bool)
    xj, yj = np.roll(x, 1), np.roll(y, 1)
    for xi_, yi_, xj_, yj_ in zip(x, y, xj, yj):
        if yi_ == yj_:
            continue
        cond = (yi_ > qy) != (yj_ > qy)
        if cond.any():
            xint = xi_ + (qy - yi_) * (xj_ - xi_) / (yj_ - yi_)
            inside ^= cond & (qx < xint)
    out[r0:r1, c0:c1] = inside
    return out


def _clip_segment(x0, y0, x1, y1, xmin, ymin, xmax, ymax):
    """Liang-Barsky clipping of a segment to a rectangle; None if outside."""
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin), (dy, ymax - y0)):
        if p == 0:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return None
    return x0 + t0 * dx, y0 + t0 * dy, x0 + t1 * dx, y0 + t1 * dy


def _supercover(x0: float, y0: float, x1: float, y1: float, n_rows: int, n_cols: int) -> List[Tuple[int, int]]:
    """Every grid cell the segment passes through (4-connected), clipped to the grid."""
    # Clip to the grid (plus a one-cell border) first: a segment starting far outside
    # the grid would otherwise exhaust the step cap before it reaches the grid.
    clipped = _clip_segment(x0, y0, x1, y1, -1.0, -1.0, n_cols + 1.0, n_rows + 1.0)
    if clipped is None:
        return []
    x0, y0, x1, y1 = clipped
    cells = []
    cx, cy = int(math.floor(x0)), int(math.floor(y0))
    ex, ey = int(math.floor(x1)), int(math.floor(y1))
    dx, dy = x1 - x0, y1 - y0
    sx = 1 if dx > 0 else -1
    sy = 1 if dy > 0 else -1
    tdx = abs(1.0 / dx) if dx else math.inf
    tdy = abs(1.0 / dy) if dy else math.inf
    tmx = ((cx + 1 - x0) if dx > 0 else (x0 - cx)) * tdx if dx else math.inf
    tmy = ((cy + 1 - y0) if dy > 0 else (y0 - cy)) * tdy if dy else math.inf
    n = abs(ex - cx) + abs(ey - cy) + 1
    for _ in range(min(n, 4 * (n_rows + n_cols) + 4)):
        if 0 <= cy < n_rows and 0 <= cx < n_cols:
            cells.append((cy, cx))
        if cx == ex and cy == ey:
            break
        if tmx < tmy:
            tmx += tdx
            cx += sx
        else:
            tmy += tdy
            cy += sy
    return cells


def _dist_to_segment(px, py, x0, y0, x1, y1):
    vx, vy = x1 - x0, y1 - y0
    L2 = vx * vx + vy * vy
    t = np.clip(((px - x0) * vx + (py - y0) * vy) / L2, 0, 1) if L2 > 0 else 0.0
    return np.hypot(px - (x0 + t * vx), py - (y0 + t * vy))


def _rings_from_members(members: list) -> Tuple[List[list], List[list]]:
    """Assemble closed outer / inner rings of an OSM multipolygon relation from
    its member ways (joined end to end)."""
    def assemble(parts):
        parts = [list(p) for p in parts if len(p) >= 2]
        rings = []
        while parts:
            ring = parts.pop(0)
            changed = True
            while ring[0] != ring[-1] and changed:
                changed = False
                for k, p in enumerate(parts):
                    if p[0] == ring[-1]:
                        ring += p[1:]
                    elif p[-1] == ring[-1]:
                        ring += p[::-1][1:]
                    elif p[-1] == ring[0]:
                        ring = p[:-1] + ring
                    elif p[0] == ring[0]:
                        ring = p[::-1][:-1] + ring
                    else:
                        continue
                    parts.pop(k)
                    changed = True
                    break
            if len(ring) >= 4:
                rings.append(ring)
        return rings
    outer = [[(g["lat"], g["lon"]) for g in m.get("geometry") or []] for m in members
             if m.get("type") == "way" and m.get("role", "outer") in ("outer", "")]
    inner = [[(g["lat"], g["lon"]) for g in m.get("geometry") or []] for m in members
             if m.get("type") == "way" and m.get("role") == "inner"]
    return assemble(outer), assemble(inner)


# ── rasterisation ────────────────────────────────────────────────────────── #

def rasterize(elements: Sequence[dict], bounds: Dict[str, float], n_rows: int, n_cols: int,
              cell_m: float) -> Tuple[np.ndarray, int]:
    """Class grid (n_rows x n_cols, row 0 = north) from Overpass elements
    (`out geom` format). Returns (classes, number of features used)."""
    g = _Grid(bounds, n_rows, n_cols, cell_m)
    cx, cy = g.centres()
    layers = {k: np.zeros((n_rows, n_cols), dtype=bool) for k in PRIORITY}
    used = 0
    for el in elements:
        fc = feature_class(el.get("tags") or {})
        if fc is None:
            continue
        cls, kind, width_m = fc
        if el.get("type") == "relation":
            outers, inners = _rings_from_members(el.get("members") or [])
            if not outers:
                continue
            mask = np.zeros((n_rows, n_cols), dtype=bool)
            for ring in outers:
                xs, ys = g.frac(np.array([p[0] for p in ring]), np.array([p[1] for p in ring]))
                mask |= _points_in_ring(cx, cy, np.column_stack([xs, ys]))
            for ring in inners:
                xs, ys = g.frac(np.array([p[0] for p in ring]), np.array([p[1] for p in ring]))
                mask &= ~_points_in_ring(cx, cy, np.column_stack([xs, ys]))
            layers[cls] |= mask
            used += 1
            continue
        geom = el.get("geometry") or []
        if len(geom) < 2:
            continue
        xs, ys = g.frac(np.array([p["lat"] for p in geom]), np.array([p["lon"] for p in geom]))
        closed = len(geom) >= 4 and geom[0]["lat"] == geom[-1]["lat"] and geom[0]["lon"] == geom[-1]["lon"]
        if kind == "area" and closed:
            mask = _points_in_ring(cx, cy, np.column_stack([xs, ys]))
            mx, my = float(np.mean(xs[:-1])), float(np.mean(ys[:-1]))        # small polygons (buildings)
            if 0 <= my < n_rows and 0 <= mx < n_cols:
                mask[int(my), int(mx)] = True
            layers[cls] |= mask
        else:
            half = width_m / 2.0 / cell_m                  # half width in cells
            for k in range(len(xs) - 1):
                x0, y0, x1, y1 = xs[k], ys[k], xs[k + 1], ys[k + 1]
                if max(x0, x1) < -1 or min(x0, x1) > n_cols + 1 or max(y0, y1) < -1 or min(y0, y1) > n_rows + 1:
                    continue
                for r, c in _supercover(x0, y0, x1, y1, n_rows, n_cols):
                    layers[cls][r, c] = True
                if half > 0.5:
                    r0, r1 = max(0, int(min(y0, y1) - half - 1)), min(n_rows, int(max(y0, y1) + half + 2))
                    c0, c1 = max(0, int(min(x0, x1) - half - 1)), min(n_cols, int(max(x0, x1) + half + 2))
                    if r0 < r1 and c0 < c1:
                        d = _dist_to_segment(cx[r0:r1, c0:c1], cy[r0:r1, c0:c1], x0, y0, x1, y1)
                        layers[cls][r0:r1, c0:c1] |= d <= half
        used += 1
    classes = np.full((n_rows, n_cols), FUEL, dtype=np.int8)
    for k in PRIORITY:
        classes[layers[k]] = k
    return classes, used


# ── data access (cached Overpass query) ─────────────────────────────────── #

def _snapped_bbox(b: Dict[str, float]) -> Tuple[float, float, float, float]:
    s = math.floor(b["south"] / SNAP_DEG) * SNAP_DEG
    w = math.floor(b["west"] / SNAP_DEG) * SNAP_DEG
    n = math.ceil(b["north"] / SNAP_DEG) * SNAP_DEG
    e = math.ceil(b["east"] / SNAP_DEG) * SNAP_DEG
    return round(s, 4), round(w, 4), round(n, 4), round(e, 4)


def _cache_path(bbox) -> Path:
    key = hashlib.sha1((QUERY_VERSION + ":" + ",".join(f"{v:.4f}" for v in bbox)).encode()).hexdigest()[:16]
    return LAND_COVER_DIR / f"osm_{key}.json"


def fetch_osm(bbox: Tuple[float, float, float, float], allow_fetch: bool = True,
              timeout_s: float = TIMEOUT_S) -> Optional[dict]:
    """Overpass response for bbox (south, west, north, east), from the disk
    cache when present. None when unavailable."""
    path = _cache_path(bbox)
    key = path.name
    if os.getenv("FIRE_LAND_COVER_FETCH", "1") == "0":       # e.g. the test suite: never touch the network
        allow_fetch = False
    cached = None
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Ignoring unreadable land-cover cache %s (%s)", path.name, exc)
    if cached is not None:
        # Expiry (config.LandCoverConfig.osm_ttl_days): an old copy is refreshed when
        # the network allows it; otherwise it is still used, flagged as stale, rather
        # than losing the land cover to a temporary outage.
        if not _expired(cached.get("fetched_utc")):
            return cached
        if not allow_fetch or time.time() - _FETCH_FAILED_AT.get(key, 0) < _RETRY_AFTER_S:
            cached["_stale"] = True
            return cached
    if not allow_fetch or time.time() - _FETCH_FAILED_AT.get(key, 0) < _RETRY_AFTER_S:
        return None
    import requests
    q = QUERY.replace("{bb}", ",".join(f"{v:.4f}" for v in bbox))
    last = None
    for url in OVERPASS_URLS:
        try:
            r = requests.post(url, data={"data": q}, timeout=timeout_s,
                              headers={"User-Agent": "forest-fire-digital-twin/1.0 (BMSCE capstone)"})
            r.raise_for_status()
            data = r.json()
            if "elements" not in data:
                raise ValueError("unexpected Overpass response")
            LAND_COVER_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"bbox": bbox, "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                        "source": "OpenStreetMap (Overpass API)", "query_version": QUERY_VERSION,
                                        "elements": data["elements"]}), encoding="utf-8")
            logger.info("Land cover: %d OSM features for %s (cached in %s)", len(data["elements"]), bbox, path.name)
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:                       # try the next mirror
            last = exc
    _FETCH_FAILED_AT[key] = time.time()
    if cached is not None:
        logger.warning("OpenStreetMap refresh failed for %s (%s); using the cached copy from %s.", bbox, last,
                       str(cached.get("fetched_utc", ""))[:10])
        cached["_stale"] = True
        return cached
    logger.warning("Land-cover data unavailable for %s (%s); fire is not constrained by land cover.", bbox, last)
    return None


def _expired(fetched_utc) -> bool:
    from datetime import datetime, timezone
    from config.config import LAND_COVER
    try:
        t = datetime.strptime(str(fetched_utc), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return True
    return (datetime.now(timezone.utc) - t).total_seconds() > LAND_COVER.osm_ttl_days * 86400


def domain_land_cover(domain, allow_fetch: bool = True) -> LandCover:
    """Per-cell land-cover classes of a SimulationDomain (row 0 = north, same
    layout as the CA grids)."""
    shape = (domain.n_rows, domain.n_cols)
    bbox = _snapped_bbox(domain.bounds)
    data = fetch_osm(bbox, allow_fetch=allow_fetch)
    if data is None:
        return all_fuel(shape, "unavailable",
                        "Land cover unavailable (OpenStreetMap could not be reached); all cells treated as fuel")
    classes, used = rasterize(data.get("elements", []), domain.bounds, domain.n_rows, domain.n_cols,
                              domain.cell_m)
    when = str(data.get("fetched_utc", ""))[:10]
    label = (f"OpenStreetMap land cover ({used} mapped feature{'s' if used != 1 else ''}"
             + (f", retrieved {when}" if when else "") + ")")
    return LandCover(classes, "osm", label, used, _counts(classes))
