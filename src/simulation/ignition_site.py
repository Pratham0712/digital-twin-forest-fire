"""
ignition_site.py - LIGHTWEIGHT check of a hypothetical ignition point.

Purpose: stop obviously inappropriate hypothetical ignitions (a click on a
mapped road, building / built-up area, water body or bare ground). It is NOT a
fuel model and it is NEVER used by the fire propagation: once an ignition is
accepted, the cellular automata spreads the fire with its own parameters.

Data: OpenStreetMap features for the set-up area, through the project's
existing Overpass access and disk cache (src/simulation/fuel_map.fetch_osm) -
one small request per area, then cached. No ESA WorldCover, no Sentinel-2, no
land-cover fusion.

The clicked POINT itself is tested (not a 25 m cell):
  water polygon, or within half the width of a river / canal line   -> WATER
  inside a building polygon                                          -> BUILT-UP AREA
  within half the carriageway width (+ click tolerance) of a road    -> ROAD
  inside a residential / commercial / industrial land-use polygon    -> BUILT-UP AREA
  inside bare rock / sand / scree / quarry / runway                  -> BARE / NON-BURNABLE
  inside mapped forest, wood, scrub, heath, grassland, meadow        -> VEGETATION (accepted)
  anything else (or OSM unavailable)                                 -> LOCATION UNVERIFIED
                                                                        (accepted with a warning;
                                                                        never called vegetation)
Google satellite imagery is never classified.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.simulation import fuel_map
from src.simulation.fuel_map import BUILT, FUEL, NON_FUEL, ROAD, UNKNOWN, WATER

CHECK_TIMEOUT_S = 8.0           # one Overpass request per area (then cached): never a long wait
CLICK_TOLERANCE_M = 2.0          # added to half the road / channel width (map-click precision)
POINT_PAD_DEG = 0.001            # lazy mode: features within ~110 m of the clicked point are loaded
VEGETATION_TAGS = {"natural": ("wood", "scrub", "heath", "grassland", "fell", "shrubbery"),
                   "landuse": ("forest", "meadow", "grass", "orchard", "vineyard", "plant_nursery"),
                   "leisure": ()}
LABELS = {FUEL: "VEGETATION", ROAD: "ROAD", BUILT: "BUILT-UP AREA", WATER: "WATER",
          NON_FUEL: "BARE / NON-BURNABLE", UNKNOWN: "LOCATION UNVERIFIED"}
REJECT_HINT = "Select a vegetation/burnable location."


@dataclass
class SiteCheck:
    code: int                      # fuel_map class: FUEL, ROAD, BUILT, WATER, NON_FUEL or UNKNOWN
    label: str                     # VEGETATION, ROAD, BUILT-UP AREA, WATER, BARE / NON-BURNABLE, LOCATION UNVERIFIED
    accepted: bool
    message: str
    evidence: str = ""             # e.g. "OpenStreetMap highway=trunk"

    def as_dict(self) -> dict:
        return {"code": self.code, "label": self.label, "accepted": self.accepted, "message": self.message,
                "evidence": self.evidence}


def _veg_tag(tags: dict) -> Optional[str]:
    for k, vals in VEGETATION_TAGS.items():
        if tags.get(k) in vals:
            return f"{k}={tags[k]}"
    return None


@dataclass
class IgnitionSiteClassifier:
    """Point classifier over OpenStreetMap features (Overpass `out geom` format)."""
    elements: Sequence[dict] = field(default_factory=list)
    available: bool = True
    source_label: str = "OpenStreetMap"
    retrieved: str = ""

    # LandCover-like flags used by the page (never a grid: no fire mask can be built from this)
    status: str = "point-check"
    classes = None
    # Phase 3 (performance): lazy mode - nothing is fetched when the page renders or the area / domain
    # changes; a small OpenStreetMap tile (~±110 m around the point, snapped to the cache grid) is
    # fetched only when a point is actually checked, and cached on disk. Independent of domain size.
    lazy: bool = False
    allow_fetch: bool = True
    _tiles: Dict[tuple, Optional[dict]] = field(default_factory=dict)

    @classmethod
    def for_domain(cls, domain=None, allow_fetch: bool = True) -> "IgnitionSiteClassifier":
        return cls([], True, "OpenStreetMap", "", lazy=True, allow_fetch=allow_fetch)

    def _tile_for(self, lat: float, lon: float) -> Optional[dict]:
        pad = POINT_PAD_DEG
        bbox = fuel_map._snapped_bbox({"south": lat - pad, "north": lat + pad, "west": lon - pad, "east": lon + pad})
        if bbox not in self._tiles:
            self._tiles[bbox] = fuel_map.fetch_osm(bbox, allow_fetch=self.allow_fetch, timeout_s=CHECK_TIMEOUT_S)
        return self._tiles[bbox]

    def _loaded_elements(self) -> List[dict]:
        if not self.lazy:
            return list(self.elements)
        seen, out = set(), []
        for data in self._tiles.values():
            for el in (data or {}).get("elements", []):
                k = (el.get("type"), el.get("id"), id(el) if el.get("id") is None else None)
                if k not in seen:
                    seen.add(k)
                    out.append(el)
        return out

    # geometry helpers (local metres around the point)
    @staticmethod
    def _xy(lat0: float, lon0: float, pts) -> np.ndarray:
        a = np.asarray(pts, dtype=float)
        return np.column_stack([(a[:, 1] - lon0) * 111320.0 * math.cos(math.radians(lat0)),
                                (a[:, 0] - lat0) * 111320.0])

    @staticmethod
    def _inside(ring: np.ndarray) -> bool:
        """Is the origin inside the ring (even-odd)?"""
        x, y = ring[:, 0], ring[:, 1]
        inside = False
        for xi, yi, xj, yj in zip(x, y, np.roll(x, 1), np.roll(y, 1)):
            if (yi > 0) != (yj > 0) and 0 < xi + (0 - yi) * (xj - xi) / (yj - yi):
                inside = not inside
        return inside

    @staticmethod
    def _dist_to_line(line: np.ndarray) -> float:
        best = math.inf
        for (x0, y0), (x1, y1) in zip(line[:-1], line[1:]):
            vx, vy = x1 - x0, y1 - y0
            L2 = vx * vx + vy * vy
            t = 0.0 if L2 == 0 else max(0.0, min(1.0, -(x0 * vx + y0 * vy) / L2))
            best = min(best, math.hypot(x0 + t * vx, y0 + t * vy))
        return best

    def _polygons(self, el, lat, lon) -> List[Tuple[np.ndarray, List[np.ndarray]]]:
        if el.get("type") == "relation":
            outers, inners = fuel_map._rings_from_members(el.get("members") or [])
            return [(self._xy(lat, lon, o), [self._xy(lat, lon, h) for h in inners]) for o in outers]
        g = el.get("geometry") or []
        if len(g) >= 4 and g[0]["lat"] == g[-1]["lat"] and g[0]["lon"] == g[-1]["lon"]:
            return [(self._xy(lat, lon, [(p["lat"], p["lon"]) for p in g]), [])]
        return []

    def classify(self, lat: float, lon: float) -> SiteCheck:
        lat, lon = float(lat), float(lon)
        if self.lazy:
            data = self._tile_for(lat, lon)
            when = str((data or {}).get("fetched_utc", ""))[:10]
            sub = IgnitionSiteClassifier((data or {}).get("elements", []), data is not None,
                                         "OpenStreetMap" if data is not None else "OpenStreetMap unavailable", when)
            if when:
                self.retrieved = when
            return sub.classify(lat, lon)
        if not self.available:
            return SiteCheck(UNKNOWN, LABELS[UNKNOWN], True,
                             "LOCATION UNVERIFIED — OpenStreetMap is unavailable for this area, so the type of "
                             "ground could not be checked. The hypothetical ignition is accepted, but it is not "
                             "confirmed vegetation.", "no data")
        hits: Dict[int, str] = {}
        veg = None
        for el in self.elements:
            tags = el.get("tags") or {}
            fc = fuel_map.feature_class(tags)
            if fc is None:
                vt = _veg_tag(tags)
                if vt and veg is None:
                    for outer, holes in self._polygons(el, lat, lon):
                        if self._inside(outer) and not any(self._inside(h) for h in holes):
                            veg = vt
                            break
                continue
            code, kind, width = fc
            name = ", ".join(f"{k}={tags[k]}" for k in ("highway", "railway", "waterway", "natural", "water",
                                                        "building", "landuse", "aeroway") if k in tags)
            if kind == "line" or el.get("type") == "way" and not self._polygons(el, lat, lon):
                g = el.get("geometry") or []
                if len(g) < 2:
                    continue
                d = self._dist_to_line(self._xy(lat, lon, [(p["lat"], p["lon"]) for p in g]))
                if d <= width / 2.0 + CLICK_TOLERANCE_M:
                    hits.setdefault(code, name)
            else:
                for outer, holes in self._polygons(el, lat, lon):
                    if self._inside(outer) and not any(self._inside(h) for h in holes):
                        hits.setdefault(code, name)
                        break
        for code in (WATER, BUILT, ROAD, NON_FUEL):           # most obvious first
            if code in hits:
                return SiteCheck(code, LABELS[code], False,
                                 f"IGNITION REJECTED — {LABELS[code]}. {REJECT_HINT}",
                                 f"OpenStreetMap {hits[code]}")
        if veg:
            return SiteCheck(FUEL, LABELS[FUEL], True, "HYPOTHETICAL IGNITION ACCEPTED — VEGETATION",
                             f"OpenStreetMap {veg}")
        return SiteCheck(UNKNOWN, LABELS[UNKNOWN], True,
                         "LOCATION UNVERIFIED — no mapped road, building, water or vegetation at this point. "
                         "The hypothetical ignition is accepted, but it is not confirmed vegetation.",
                         "no mapped feature")

    def context_classes(self, domain) -> Optional[np.ndarray]:
        """Mapped OpenStreetMap features on the domain grid, for the map's Land
        cover CONTEXT layer only (never passed to the fire simulation)."""
        els = self._loaded_elements()
        if not self.available or not els or domain.n_rows * domain.n_cols > 40_000:
            return None                         # (large domains: no context raster - keeps the page fast)
        classes, _ = fuel_map.rasterize(els, domain.bounds, domain.n_rows, domain.n_cols, domain.cell_m)
        return classes if (classes != FUEL).any() else None

    @property
    def label(self) -> str:
        if not self.available:
            return "Ignition-point check: OpenStreetMap unavailable (points are UNVERIFIED)"
        if self.lazy and not self.retrieved:
            return ("Ignition-point check: OpenStreetMap features around each clicked point (cached; not a fire "
                    "barrier)")
        return (f"Ignition-point check: OpenStreetMap"
                + (f", retrieved {self.retrieved}" if self.retrieved else "") + " (not a fire barrier)")
