"""
dynamic_world.py - OPTIONAL refinement from Google Dynamic World probabilities.

Dynamic World (10 m, near-real-time class probabilities) is only available
through Google Earth Engine, which needs an Earth Engine account and
authentication. The simulation therefore never contacts Earth Engine itself and
never waits for it: it only READS probabilities that were exported beforehand
into the tile cache (dataset "dynamic_world", one npz per 0.01-degree tile with
float32 arrays named after the Dynamic World bands: water, trees, grass,
flooded_vegetation, crops, shrub_and_scrub, built, bare, snow_and_ice).

Disabled by default (config.LandCoverConfig.use_dynamic_world). When disabled or
when no cached tile exists, WorldCover + OpenStreetMap + Sentinel-2 work exactly
the same; Dynamic World only raises or lowers the per-cell CONFIDENCE.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from config.config import LAND_COVER
from src.landcover.tile_cache import RasterWindow, TileCache, load_tiled

DATASET = "dynamic_world"
VERSION = "probabilities-v1"
BANDS = ("water", "trees", "grass", "flooded_vegetation", "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice")
VEGETATION = ("trees", "grass", "flooded_vegetation", "crops", "shrub_and_scrub")


def _no_download(tiles):
    raise RuntimeError("Dynamic World is read from the cache only (needs Google Earth Engine to export)")


def load_vegetation_probability(bounds: Dict[str, float],
                                cache: Optional[TileCache] = None) -> Tuple[Optional[RasterWindow], dict]:
    """Cached P(vegetation) = sum of the vegetation-class probabilities, or None."""
    if not LAND_COVER.use_dynamic_world:
        return None, {"status": "disabled", "error": "optional; disabled in configuration"}
    px = int(round(LAND_COVER.cache_tile_deg / LAND_COVER.worldcover_res_deg))
    win, st = load_tiled(DATASET, VERSION, bounds, LAND_COVER.ndvi_ttl_days, px, "p_vegetation", np.float32,
                         np.nan, _no_download, allow_fetch=False, cache=cache)
    if win is not None:
        win.label = "Dynamic World (cached probabilities)"
    return win, st
