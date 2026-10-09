"""
study_region.py - the PRIMARY STUDY REGION: Bandipur Tiger Reserve.

The study region is a geographic polygon, separate from every computational
concept (focus area, initial / expanded simulation domain). It never stops a
fire: it is used to label cells, detect boundary crossings and keep Bandipur
statistics separate from REGIONAL EXTENSION statistics.

Boundary sources, in order:

  1. OFFICIAL      data/study_region/bandipur_official.{geojson,json,kml,shp}
                   (put the official notified Tiger Reserve boundary there; an
                   ESRI shapefile needs its .prj, projected CRSs are converted
                   with rasterio). Labelled OFFICIAL.
  2. APPROXIMATE   data/study_region/bandipur_provisional.geojson - for
                   development and testing only, labelled APPROXIMATE /
                   PROVISIONAL everywhere. The shipped file is the latitude /
                   longitude EXTENT published by NTCA, i.e. a box that
                   over-covers the reserve: inside / outside near the real
                   boundary is NOT reliable with it.

An OpenStreetMap National Park polygon is never presented as the complete
1,456 km2 Tiger Reserve.
"""
from __future__ import annotations

import json
import logging
import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from config.config import STUDY_REGION

logger = logging.getLogger(__name__)

OFFICIAL, APPROXIMATE = "OFFICIAL", "APPROXIMATE / PROVISIONAL"
REGIONAL_EXTENSION = "REGIONAL EXTENSION"

Ring = np.ndarray            # (N, 2) lon, lat


@dataclass
class StudyRegion:
    name: str
    status: str                                     # OFFICIAL | APPROXIMATE / PROVISIONAL
    source: str
    polygons: List[Tuple[Ring, List[Ring]]]         # (outer, holes)
    notes: List[str] = field(default_factory=list)
    stated_area_km2: Optional[float] = None
    stated_area_source: str = ""

    @property
    def official(self) -> bool:
        return self.status == OFFICIAL

    @property
    def label(self) -> str:
        return f"{self.name} boundary — {self.status} ({self.source})"

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        pts = np.vstack([o for o, _ in self.polygons])
        return float(pts[:, 1].min()), float(pts[:, 0].min()), float(pts[:, 1].max()), float(pts[:, 0].max())

    @property
    def area_km2(self) -> float:
        return sum(_ring_area_km2(o) - sum(_ring_area_km2(h) for h in holes) for o, holes in self.polygons)

    def contains(self, lat, lon) -> np.ndarray:
        """Boolean array: points (broadcast lat/lon arrays) inside the region."""
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        lat, lon = np.broadcast_arrays(lat, lon)
        out = np.zeros(lat.shape, dtype=bool)
        for outer, holes in self.polygons:
            m = _in_ring(lon, lat, outer)
            for h in holes:
                m &= ~_in_ring(lon, lat, h)
            out |= m
        return out

    def cell_mask(self, grid) -> np.ndarray:
        """(rows, cols) bool: cell centres of a SimulationDomain / GridSpec inside."""
        lat, lon = grid.cell_centres()
        return self.contains(lat, lon)

    def outline(self, max_points: int = 400) -> List[List[List[float]]]:
        """Outer rings as [[lat, lon], ...] lists for the map (decimated)."""
        out = []
        for o, _ in self.polygons:
            step = max(1, int(math.ceil(len(o) / max_points)))
            pts = o[::step]
            if not np.array_equal(pts[-1], o[-1]):
                pts = np.vstack([pts, o[-1:]])
            out.append([[round(float(la), 6), round(float(lo), 6)] for lo, la in pts])
        return out

    def describe(self) -> dict:
        return {"name": self.name, "status": self.status, "source": self.source, "label": self.label,
                "area_km2_polygon": round(self.area_km2, 1), "stated_area_km2": self.stated_area_km2,
                "stated_area_source": self.stated_area_source, "notes": list(self.notes)}


def _in_ring(x: np.ndarray, y: np.ndarray, ring: Ring) -> np.ndarray:
    inside = np.zeros(x.shape, dtype=bool)
    xs, ys = ring[:, 0], ring[:, 1]
    if len(xs) < 3:
        return inside
    box = (x >= xs.min()) & (x <= xs.max()) & (y >= ys.min()) & (y <= ys.max())
    if not box.any():
        return inside
    qx, qy = x[box], y[box]
    res = np.zeros(qx.shape, dtype=bool)
    for xi, yi, xj, yj in zip(xs, ys, np.roll(xs, 1), np.roll(ys, 1)):
        if yi == yj:
            continue
        cond = (yi > qy) != (yj > qy)
        if cond.any():
            res ^= cond & (qx < xi + (qy - yi) * (xj - xi) / (yj - yi))
    inside[box] = res
    return inside


def _ring_area_km2(ring: Ring) -> float:
    lat0 = math.radians(float(np.mean(ring[:, 1])))
    x = np.radians(ring[:, 0]) * 6371.0088 * math.cos(lat0)
    y = np.radians(ring[:, 1]) * 6371.0088
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2.0


# ── readers ──────────────────────────────────────────────────────────────── #

def _close(r: Sequence) -> Ring:
    a = np.asarray(r, dtype=float)[:, :2]
    if not np.array_equal(a[0], a[-1]):
        a = np.vstack([a, a[:1]])
    return a


def read_geojson(path: Path) -> Tuple[List[Tuple[Ring, List[Ring]]], dict]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    feats = d.get("features") if d.get("type") == "FeatureCollection" else [d if d.get("type") == "Feature"
                                                                            else {"geometry": d, "properties": {}}]
    polys, props = [], {}
    for f in feats:
        g = f.get("geometry") or {}
        props = props or (f.get("properties") or {})
        if g.get("type") == "Polygon":
            parts = [g["coordinates"]]
        elif g.get("type") == "MultiPolygon":
            parts = g["coordinates"]
        else:
            continue
        for p in parts:
            polys.append((_close(p[0]), [_close(h) for h in p[1:]]))
    return polys, props


def read_kml(path: Path) -> List[Tuple[Ring, List[Ring]]]:
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    polys = []
    for poly in re.findall(r"<Polygon\b.*?</Polygon>", txt, flags=re.S | re.I):
        def rings(tag):
            out = []
            for blk in re.findall(rf"<{tag}\b.*?</{tag}>", poly, flags=re.S | re.I):
                for c in re.findall(r"<coordinates>(.*?)</coordinates>", blk, flags=re.S | re.I):
                    pts = [tuple(map(float, t.split(",")[:2])) for t in c.split() if t.count(",") >= 1]
                    if len(pts) >= 3:
                        out.append(_close(pts))
            return out
        outer = rings("outerBoundaryIs")
        if outer:
            polys.append((outer[0], rings("innerBoundaryIs")))
    return polys


def read_shapefile(path: Path) -> List[Tuple[Ring, List[Ring]]]:
    """Polygon shapefile (shape types 5 / 15 / 25). Rings are split into outer
    (clockwise) and holes (counter-clockwise) as the format defines; a .prj
    with a projected CRS is converted to WGS84 with rasterio."""
    data = Path(path).read_bytes()
    shp_type = struct.unpack("<i", data[32:36])[0]
    if shp_type not in (5, 15, 25):
        raise ValueError(f"shapefile type {shp_type} is not a polygon layer")
    rings_all: List[np.ndarray] = []
    pos = 100
    while pos + 8 <= len(data):
        _, clen = struct.unpack(">ii", data[pos:pos + 8])
        rec = data[pos + 8:pos + 8 + clen * 2]
        pos += 8 + clen * 2
        if len(rec) < 44 or struct.unpack("<i", rec[:4])[0] == 0:
            continue
        nparts, npts = struct.unpack("<ii", rec[36:44])
        parts = list(struct.unpack(f"<{nparts}i", rec[44:44 + 4 * nparts]))
        off = 44 + 4 * nparts
        xy = np.frombuffer(rec[off:off + 16 * npts], dtype="<f8").reshape(npts, 2)
        for k, a in enumerate(parts):
            b = parts[k + 1] if k + 1 < nparts else npts
            rings_all.append(xy[a:b].copy())
    prj = Path(path).with_suffix(".prj")
    if prj.exists():
        wkt = prj.read_text(encoding="utf-8", errors="replace")
        if "PROJCS" in wkt.upper() or "PROJCRS" in wkt.upper():
            from rasterio.crs import CRS
            from rasterio.warp import transform
            src = CRS.from_wkt(wkt)
            conv = []
            for r in rings_all:
                xs, ys = transform(src, "EPSG:4326", r[:, 0].tolist(), r[:, 1].tolist())
                conv.append(np.column_stack([xs, ys]))
            rings_all = conv
    polys: List[Tuple[Ring, List[Ring]]] = []
    for r in rings_all:
        signed = float(np.dot(r[:, 0], np.roll(r[:, 1], -1)) - np.dot(r[:, 1], np.roll(r[:, 0], -1)))
        if signed < 0 or not polys:                   # clockwise = outer ring (ESRI convention)
            polys.append((_close(r), []))
        else:
            polys[-1][1].append(_close(r))
    return polys


_CACHE: dict = {}


def load_study_region() -> Optional[StudyRegion]:
    """OFFICIAL boundary when present, otherwise the APPROXIMATE one (None if
    neither file exists)."""
    stem = Path(STUDY_REGION.official_boundary_stem)
    for ext in (".geojson", ".json", ".kml", ".shp"):
        p = stem.with_suffix(ext)
        if p.exists():
            key = (str(p), p.stat().st_mtime)
            if key not in _CACHE:
                try:
                    if ext in (".geojson", ".json"):
                        polys, props = read_geojson(p)
                    elif ext == ".kml":
                        polys, props = read_kml(p), {}
                    else:
                        polys, props = read_shapefile(p), {}
                    if not polys:
                        raise ValueError("no polygon found")
                    _CACHE[key] = StudyRegion(STUDY_REGION.name, OFFICIAL, props.get("source") or p.name, polys,
                                              [], STUDY_REGION.stated_area_km2, STUDY_REGION.stated_area_source)
                except Exception as exc:
                    logger.warning("Official study-region boundary %s could not be read (%s)", p.name, exc)
                    _CACHE[key] = None
            if _CACHE[key] is not None:
                return _CACHE[key]
    p = Path(STUDY_REGION.provisional_boundary_path)
    if not p.exists():
        return None
    key = (str(p), p.stat().st_mtime)
    if key not in _CACHE:
        polys, props = read_geojson(p)
        notes = [props.get("note", "Approximate boundary for development and testing only.")]
        _CACHE[key] = StudyRegion(STUDY_REGION.name, APPROXIMATE, props.get("source", p.name), polys, notes,
                                  STUDY_REGION.stated_area_km2, STUDY_REGION.stated_area_source)
    return _CACHE[key]


def split_stats(region: Optional[StudyRegion], grid, affected: np.ndarray, cell_m: float) -> dict:
    """Burned / affected area inside the study region vs REGIONAL EXTENSION."""
    cell_ha = cell_m ** 2 / 1e4
    if region is None:
        return {"region": None, "inside_cells": int(affected.sum()), "outside_cells": 0,
                "inside_ha": round(float(affected.sum()) * cell_ha, 4), "outside_ha": 0.0}
    inside = region.cell_mask(grid)
    return {"region": region.status, "inside_cells": int((affected & inside).sum()),
            "outside_cells": int((affected & ~inside).sum()),
            "inside_ha": round(float((affected & inside).sum()) * cell_ha, 4),
            "outside_ha": round(float((affected & ~inside).sum()) * cell_ha, 4)}


def zone_summary(region: Optional[StudyRegion], processed, risk_scores, alerts=None, hotspots=None) -> Optional[dict]:
    """Regional-twin statistics for the PRIMARY STUDY REGION only: grid zones
    whose centre lies inside the boundary (0.1-degree zones, ~11 km), their
    peak / mean model risk, HIGH/EXTREME alerts and NASA FIRMS detections
    inside the boundary. Regional-extension zones are not mixed in."""
    if region is None or processed is None or len(processed) == 0:
        return None
    p = processed.reset_index(drop=True)
    inside = region.contains(p["latitude"].to_numpy(float), p["longitude"].to_numpy(float))
    risk = np.asarray(risk_scores, dtype=float) if risk_scores is not None and len(risk_scores) == len(p) else None
    ids = set(p.loc[inside, "zone_id"].astype(str))
    out = {"name": region.name, "status": region.status, "zones": int(inside.sum()),
           "peak_risk": float(risk[inside].max()) if risk is not None and inside.any() else None,
           "mean_risk": float(risk[inside].mean()) if risk is not None and inside.any() else None,
           "alerts": sum(1 for a in (alerts or []) if str(a.zone_id) in ids and a.severity in ("HIGH", "EXTREME")),
           "detections": None}
    if hotspots is not None and len(hotspots):
        try:
            out["detections"] = int(region.contains(hotspots["latitude"].to_numpy(float),
                                                    hotspots["longitude"].to_numpy(float)).sum())
        except Exception:
            out["detections"] = None
    return out
