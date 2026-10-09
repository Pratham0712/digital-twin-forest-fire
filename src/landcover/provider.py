"""
provider.py - the land-cover source hierarchy, fallbacks and provenance.

    ESA WorldCover (primary classes)   -- cache tile -> download window -> stale cache -> missing
  + OpenStreetMap (vector refinement)  -- cache -> Overpass -> stale cache -> missing
  + Sentinel-2 NDVI (fuel-load proxy)  -- cache tile -> STAC + COG window -> stale -> class defaults
  + Dynamic World (optional)           -- cached probabilities only
        -> fusion.fuse -> LandCover on exactly the requested CA grid

Status of the result (shown in the UI, never hidden):

  full         WorldCover and OpenStreetMap both available for the whole grid.
  degraded     "DEGRADED LAND-COVER MODE": one primary source is missing.
               * WorldCover missing, OSM present: only explicitly mapped
                 barriers (roads, buildings, water, bare) are known; every other
                 cell is UNKNOWN (unverified), NOT assumed to be vegetation.
               * OSM missing, WorldCover present: classes come from WorldCover
                 only; narrow roads and small buildings are not represented.
               * WorldCover partly missing: those cells are UNKNOWN.
  unavailable  "LAND-COVER DATA UNAVAILABLE": no source at all; every cell is
               UNKNOWN. The application blocks hypothetical simulations.

Sentinel-2 missing does not change the status (classification is unaffected);
the fuel load then comes from class defaults and is labelled so.
"""
from __future__ import annotations

import json
import logging
import zlib
from collections import OrderedDict
from typing import Dict, Optional

import numpy as np

from config.config import LAND_COVER
from src.landcover import dynamic_world, osm_vectors, sentinel2_ndvi, worldcover
from src.landcover.fusion import CONF_NAMES, FUEL_SRC_NAMES, fuse
from src.landcover.grid import GridSpec, supersample_factor
from src.landcover.tile_cache import TileCache
from src.simulation.fuel_map import CLASS_NAMES, UNKNOWN, LandCover, _counts
from src.utils.timing import Timings

logger = logging.getLogger(__name__)

UNAVAILABLE_TITLE = "LAND-COVER DATA UNAVAILABLE"
DEGRADED_TITLE = "DEGRADED LAND-COVER MODE"
_FUSED: "OrderedDict[tuple, LandCover]" = OrderedDict()
_FUSED_MAX = 48


def _fp(win) -> Optional[tuple]:
    """Content fingerprint of a raster window (cheap: the windows are small)."""
    if win is None:
        return None
    return (win.array.shape, win.north, win.west, zlib.crc32(np.ascontiguousarray(win.array).tobytes()))


def _src_line(name: str, st: dict) -> str:
    s = st.get("status", "?")
    extra = []
    if st.get("tiles"):
        extra.append(f"{st['tiles'] - st.get('missing', 0)}/{st['tiles']} tiles")
    if st.get("retrieved") or st.get("retrieved_last"):
        extra.append(f"retrieved {(st.get('retrieved') or st.get('retrieved_last') or '')[:10]}")
    if st.get("error") and s not in ("cached", "downloaded"):
        extra.append(st["error"])
    return f"{name}: {s}" + (f" ({'; '.join(extra)})" if extra else "")


def grid_land_cover(spec: GridSpec, allow_fetch: bool = True, timings: Optional[Timings] = None,
                    cache: Optional[TileCache] = None, fetch_ndvi: bool = True) -> LandCover:
    """Fused land cover for exactly spec.n_rows x spec.n_cols cells."""
    t = timings or Timings()
    s = supersample_factor(spec.cell_m, LAND_COVER)
    with t.stage("land_cover_retrieval"):
        wc, wst = worldcover.load_window(spec.bounds, allow_fetch=allow_fetch, cache=cache)
        data, ost = osm_vectors.load_osm(spec, allow_fetch=allow_fetch)
    with t.stage("ndvi_processing"):
        nd, nst = sentinel2_ndvi.load_window(spec.bounds, allow_fetch=allow_fetch and fetch_ndvi, cache=cache)
    dw, dst = dynamic_world.load_vegetation_probability(spec.bounds, cache=cache)

    key = (spec, s, LAND_COVER.fusion_version, LAND_COVER.rules_version, repr(LAND_COVER),
           wst.get("status"), _fp(wc), ost.get("status"),
           zlib.crc32(json.dumps(data.get("elements", []), sort_keys=True).encode()) if data else None,
           nst.get("status"), _fp(nd), dst.get("status"), _fp(dw))
    if key in _FUSED:
        _FUSED.move_to_end(key)
        t.add("fused_cache_hit", 0.0)
        return _FUSED[key]

    with t.stage("osm_processing"):
        osm = osm_vectors.rasterize_fractional(data.get("elements", []), spec, s) if data is not None else None
    with t.stage("fusion"):
        fused = fuse(spec, s, wc, nd, osm, dw)

    classes = fused.classes
    wc_ok, osm_ok = wc is not None, data is not None
    notes = []
    if wc_ok and osm_ok and not wst.get("missing") and ost.get("status") != "partial":
        status = "full"
    elif wc_ok or osm_ok:
        status = "degraded"
        if not wc_ok:
            notes.append("ESA WorldCover unavailable: only mapped OpenStreetMap barriers (roads, buildings, water, "
                         "bare ground) are known; all other cells are UNVERIFIED, not assumed to be vegetation.")
        elif wst.get("missing"):
            notes.append(f"ESA WorldCover missing for {wst['missing']} of {wst['tiles']} tiles: those cells are "
                         "UNVERIFIED.")
        if not osm_ok:
            notes.append("OpenStreetMap unavailable: classes from WorldCover only; narrow roads and small buildings "
                         "are not represented.")
        elif ost.get("status") == "partial":
            notes.append(f"OpenStreetMap partly unavailable ({ost.get('error')}): roads and buildings may be missing "
                         "in part of the area.")
    else:
        status = "unavailable"
        classes = np.full(spec.shape, UNKNOWN, dtype=np.int8)
    if nd is None and status != "unavailable":
        notes.append("Sentinel-2 NDVI unavailable: fuel load from WorldCover class defaults (DERIVED), not from "
                     "observed vegetation condition.")

    parts = []
    if wc_ok:
        parts.append(wc.label)
    if osm_ok:
        n = osm.n_features if osm is not None else 0
        when = str(data.get("fetched_utc", ""))[:10]
        parts.append(f"OpenStreetMap ({n} mapped feature{'s' if n != 1 else ''}"
                     + (f", retrieved {when}" if when else "") + (", STALE copy" if data.get("_stale") else "") + ")")
    if nd is not None:
        parts.append(nd.label)
    if dw is not None:
        parts.append(dw.label)
    if status == "unavailable":
        label = f"{UNAVAILABLE_TITLE} — " + "; ".join(
            _src_line(n, x) for n, x in (("ESA WorldCover", wst), ("OpenStreetMap", ost)))
    else:
        label = (" + ".join(parts) + (f" — {DEGRADED_TITLE}" if status == "degraded" else ""))
    sources = {"worldcover": wst, "osm": ost, "sentinel2": nst, "dynamic_world": dst}
    prov = {"fusion_version": LAND_COVER.fusion_version, "rules_version": LAND_COVER.rules_version,
            "bounds": spec.bounds, "cell_m": spec.cell_m, "grid": [spec.n_rows, spec.n_cols], "supersample": s,
            "resolution": "WorldCover 10 m raster; Sentinel-2 10 m (SCL 20 m) raster; OpenStreetMap vectors",
            "worldcover_version": LAND_COVER.worldcover_version,
            "worldcover_published_accuracy": LAND_COVER.worldcover_published_accuracy,
            "sources": {k: _src_line(k, v) for k, v in sources.items()},
            "ndvi_scene_dates": (nd.meta.get("scene_dates") if nd is not None else []),
            "wc_label": wc.label if wc is not None else None}
    lc = LandCover(classes, "fused", label, osm.n_features if osm is not None else 0, _counts(classes),
                   status=status, fuel_load=fused.fuel_load, fuel_source=fused.fuel_source,
                   confidence=np.where(classes == UNKNOWN, 0, fused.confidence).astype(np.int8),
                   source_mask=fused.source_mask, fractions=fused.fractions, sources=sources, provenance=prov,
                   notes=notes)
    _FUSED[key] = lc
    while len(_FUSED) > _FUSED_MAX:
        _FUSED.popitem(last=False)
    return lc


def domain_land_cover(domain, allow_fetch: bool = True, timings: Optional[Timings] = None) -> LandCover:
    """Fused land cover of a SimulationDomain (production entry point)."""
    return grid_land_cover(GridSpec.from_domain(domain), allow_fetch=allow_fetch, timings=timings)


def clear_memory_cache():
    _FUSED.clear()


def summary(lc: LandCover) -> Dict[str, object]:
    """Counts per class and confidence, for the UI and reports."""
    out: Dict[str, object] = {"status": lc.status or ("legacy-osm" if lc.source == "osm" else lc.source),
                              "classes": {CLASS_NAMES[k]: int((lc.classes == k).sum()) for k in CLASS_NAMES}}
    if lc.confidence is not None:
        out["confidence"] = {CONF_NAMES[k]: int((lc.confidence == k).sum()) for k in CONF_NAMES}
    if lc.fuel_source is not None:
        out["fuel_load_source"] = {FUEL_SRC_NAMES[k]: int((lc.fuel_source == k).sum()) for k in FUEL_SRC_NAMES
                                   if (lc.fuel_source == k).any()}
    return out
