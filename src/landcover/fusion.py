"""
fusion.py - deterministic, explainable per-cell land-cover fusion.

For every CA cell, an s x s supersample (about one point per 5 m) is labelled:

  1. ESA WorldCover class of the 10 m pixel under the point (broad truth);
  2. OpenStreetMap overrides at that point (detailed local features), in this
     order, later wins: built-up land use (only where WorldCover does not see
     vegetation, unless configured otherwise), bare ground, road, building,
     water;

then the points of the cell are counted into fractions (water, built, road,
bare, tree, shrub, grass, crop, wetland, mangrove, unknown) and the cell is
classified with the thresholds of config.LandCoverConfig:

  water    >= water_block_fraction   -> WATER
  built    >= built_block_fraction   -> BUILT
  road     >= road_block_fraction    -> ROAD
  bare     >= bare_block_fraction    -> NON_FUEL
  unknown  >= 0.5                    -> UNKNOWN (mostly no land-cover information:
                                        never confirmed vegetation)
  veg      >= min_vegetation_fraction-> FUEL
  mixed:   water+built+road+bare >= mixed_nonburnable_fraction
                                     -> the dominant non-burnable class
           vegetation > 0 and >= unknown share
                                     -> FUEL (its fuel load is reduced by the
                                        non-vegetated share)
           otherwise                 -> dominant non-burnable class, or UNKNOWN

Continuous firebreaks (major roads, railways, rivers) then mark every cell their
centre line crosses (osm_vectors). So a small group of trees next to a road or a
house stays FUEL: only the cells a barrier actually covers are removed.

Fuel load (0-1, relative, per cell) = mean over the cell's points of
  vegetation point:  clip((NDVI - ndvi_bare) / (ndvi_dense - ndvi_bare), 0, 1)
                     when Sentinel-2 has a cloud-free NDVI there, otherwise the
                     class default (config.default_fuel_load)
  other points:      0
NDVI is used as a vegetation-condition proxy for relative fuel availability; it
is not a measurement of fuel load.

Confidence (0 n/a, 1 LOW, 2 MEDIUM, 3 HIGH) comes only from source agreement:
  FUEL:     HIGH   WorldCover vegetation purity >= purity_high AND Sentinel-2
                   NDVI of the cell >= 0.3 (two independent observations agree)
            MEDIUM WorldCover purity >= purity_medium (one source)
            LOW    mixed / ambiguous cell
  WATER / BUILT / NON_FUEL:
            HIGH   OpenStreetMap and WorldCover both show it
            MEDIUM only one of them shows it
            LOW    OpenStreetMap shows it but WorldCover sees (pure) vegetation
  ROAD:     HIGH   OSM road and WorldCover built-up / bare; MEDIUM OSM only (a
            10 m product cannot resolve most roads, so that is not a conflict)
  Dynamic World (optional, cached): P(vegetation) >= 0.6 raises a FUEL cell one
  level, <= 0.3 lowers it one level.
No accuracy percentage is computed or claimed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from config.config import LAND_COVER
from src.landcover.grid import GridSpec
from src.landcover.osm_vectors import OSMLayers
from src.landcover.tile_cache import RasterWindow
from src.simulation.fuel_map import BUILT, FUEL, NON_FUEL, ROAD, UNKNOWN, WATER

# point categories
C_UNK, C_WATER, C_BUILT, C_ROAD, C_BARE, C_TREE, C_SHRUB, C_GRASS, C_CROP, C_WETLAND, C_MANGROVE = range(11)
N_CAT = 11
CAT_NAMES = ("unknown", "water", "built", "road", "bare", "tree", "shrub", "grass", "crop", "wetland", "mangrove")
VEG_CATS = (C_TREE, C_SHRUB, C_GRASS, C_CROP, C_WETLAND, C_MANGROVE)
WC_CODE_OF_CAT = {C_TREE: 10, C_SHRUB: 20, C_GRASS: 30, C_CROP: 40, C_WETLAND: 90, C_MANGROVE: 95}
CONF_NAMES = {0: "n/a", 1: "LOW", 2: "MEDIUM", 3: "HIGH"}
# source bits per cell
SRC_WORLDCOVER, SRC_OSM, SRC_NDVI, SRC_DW = 1, 2, 4, 8
# fuel-load source per cell
FUEL_SRC_NONE, FUEL_SRC_CLASS_DEFAULT, FUEL_SRC_NDVI, FUEL_SRC_ASSUMED = 0, 1, 2, 3
FUEL_SRC_NAMES = {0: "none", 1: "class default (NDVI unavailable)", 2: "Sentinel-2 NDVI proxy",
                  3: "ASSUMED (unverified cell)"}


def _wc_lut(cfg) -> np.ndarray:
    lut = np.zeros(256, dtype=np.int8)                    # 0 (no data) and unknown codes -> C_UNK
    veg = {10: C_TREE, 20: C_SHRUB, 30: C_GRASS, 40: C_CROP, 90: C_WETLAND, 95: C_MANGROVE}
    for code, cat in veg.items():
        lut[code] = cat if code in cfg.worldcover_fuel_classes else C_BARE
    lut[50], lut[60], lut[70], lut[80], lut[100] = C_BUILT, C_BARE, C_BARE, C_WATER, C_BARE
    return lut


def _default_load_lut(cfg) -> np.ndarray:
    lut = np.zeros(N_CAT, dtype=np.float32)
    for cat, code in WC_CODE_OF_CAT.items():
        lut[cat] = float(cfg.default_fuel_load.get(code, 0.0))
    return lut


@dataclass
class FusedGrids:
    classes: np.ndarray                     # (R, C) int8 fuel_map classes incl. UNKNOWN
    fuel_load: np.ndarray                   # (R, C) float32 0-1
    fuel_source: np.ndarray                 # (R, C) int8 FUEL_SRC_*
    confidence: np.ndarray                  # (R, C) int8 0-3
    source_mask: np.ndarray                 # (R, C) uint8 SRC_* bits
    fractions: Dict[str, np.ndarray] = field(default_factory=dict)   # name -> (R, C) float32
    supersample: int = 1


def _cell_counts(cat: np.ndarray, R: int, C: int, s: int, n: int, weights=None) -> np.ndarray:
    """(R, C, n) counts (or weighted sums) of categories per cell."""
    rid = (np.arange(R * s) // s)[:, None] * C
    cid = (np.arange(C * s) // s)[None, :]
    cell = (rid + cid).astype(np.int64)
    idx = (cell * n + cat.astype(np.int64)).ravel()
    w = None if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    return np.bincount(idx, weights=w, minlength=R * C * n).reshape(R, C, n)


def _cell_mean(values: np.ndarray, R: int, C: int, s: int) -> np.ndarray:
    return values.reshape(R, s, C, s).mean(axis=(1, 3))


def fuse(spec: GridSpec, s: int, wc: Optional[RasterWindow], ndvi: Optional[RasterWindow],
         osm: Optional[OSMLayers], dw: Optional[RasterWindow] = None, cfg=LAND_COVER) -> FusedGrids:
    R, C = spec.n_rows, spec.n_cols
    lat_s, lon_s = spec.sub_axes(s)
    if wc is not None:
        wcs = wc.sample(lat_s, lon_s, fill=0)
        raw = _wc_lut(cfg)[wcs]
    else:
        raw = np.zeros((R * s, C * s), dtype=np.int8)
    final = raw.copy()
    is_veg_raw = np.isin(raw, VEG_CATS)
    if osm is not None:
        lb = osm.sub["landuse_built"]
        if wc is None or cfg.landuse_overrides_vegetation:
            final[lb] = C_BUILT
        else:
            final[lb & ~is_veg_raw] = C_BUILT
        final[osm.sub["bare"]] = C_BARE
        final[osm.sub["road"]] = C_ROAD
        final[osm.sub["building"]] = C_BUILT
        final[osm.sub["water"]] = C_WATER

    pts = float(s * s)
    frac = _cell_counts(final, R, C, s, N_CAT) / pts                     # (R, C, 11)
    raw_frac = _cell_counts(raw, R, C, s, N_CAT) / pts
    water, built, road, bare = frac[..., C_WATER], frac[..., C_BUILT], frac[..., C_ROAD], frac[..., C_BARE]
    veg = frac[..., list(VEG_CATS)].sum(axis=-1)
    unknown = frac[..., C_UNK]
    nonburn = water + built + road + bare
    nb_stack = np.stack([water, built, road, bare], axis=-1)
    dominant_nb = np.array([WATER, BUILT, ROAD, NON_FUEL], dtype=np.int8)[np.argmax(nb_stack, axis=-1)]
    leftover = np.where((veg > 0) & (veg >= unknown), FUEL,
                        np.where(nonburn >= unknown, dominant_nb, UNKNOWN)).astype(np.int8)
    classes = np.select(
        [water >= cfg.water_block_fraction, built >= cfg.built_block_fraction, road >= cfg.road_block_fraction,
         bare >= cfg.bare_block_fraction, unknown >= 0.5, veg >= cfg.min_vegetation_fraction,
         nonburn >= cfg.mixed_nonburnable_fraction],
        [WATER, BUILT, ROAD, NON_FUEL, UNKNOWN, FUEL, dominant_nb], default=leftover).astype(np.int8)
    if osm is not None:
        classes[osm.block_road & (classes != WATER)] = ROAD
        classes[osm.block_water] = WATER

    # fuel load
    is_veg = np.isin(final, VEG_CATS)
    default_load = _default_load_lut(cfg)[final]
    nd_ok = np.zeros(final.shape, dtype=bool)
    if ndvi is not None:
        nd = ndvi.sample(lat_s, lon_s, fill=np.nan).astype(np.float32)
        nd_ok = np.isfinite(nd)
        nd_load = np.clip((nd - cfg.ndvi_bare) / (cfg.ndvi_dense - cfg.ndvi_bare), 0.0, 1.0)
        load_pts = np.where(is_veg, np.where(nd_ok, nd_load, default_load), 0.0)
    else:
        nd = None
        load_pts = np.where(is_veg, default_load, 0.0)
    fuel_load = _cell_mean(load_pts.astype(np.float32), R, C, s).astype(np.float32)
    veg_pts = _cell_mean(is_veg.astype(np.float32), R, C, s)
    nd_veg_pts = _cell_mean((is_veg & nd_ok).astype(np.float32), R, C, s)
    fuel_source = np.where(nd_veg_pts > 0.5 * np.maximum(veg_pts, 1e-9), FUEL_SRC_NDVI,
                           FUEL_SRC_CLASS_DEFAULT).astype(np.int8)
    # unverified (UNKNOWN) cells: only used when the user explicitly accepts the
    # degraded mode; NDVI-derived where available, otherwise the configured assumption
    if nd is not None:
        nd_cell = np.where(nd_ok, np.clip((nd - cfg.ndvi_bare) / (cfg.ndvi_dense - cfg.ndvi_bare), 0, 1), 0.0)
        nd_cov = _cell_mean(nd_ok.astype(np.float32), R, C, s)
        nd_mean = _cell_mean(nd_cell.astype(np.float32), R, C, s) / np.maximum(nd_cov, 1e-9)
        nd_raw_mean = _cell_mean(np.where(nd_ok, nd, 0.0).astype(np.float32), R, C, s) / np.maximum(nd_cov, 1e-9)
    else:
        nd_cov = np.zeros((R, C), dtype=np.float32)
        nd_mean = nd_raw_mean = np.zeros((R, C), dtype=np.float32)
    unk = classes == UNKNOWN
    fuel_load = np.where(unk, np.where(nd_cov > 0.5, nd_mean, cfg.unverified_fuel_load), fuel_load)
    fuel_source = np.where(unk, np.where(nd_cov > 0.5, FUEL_SRC_NDVI, FUEL_SRC_ASSUMED), fuel_source)
    burn = (classes == FUEL)
    fuel_load = np.where(burn | unk, fuel_load, 0.0).astype(np.float32)
    fuel_source = np.where(burn | unk, fuel_source, FUEL_SRC_NONE).astype(np.int8)

    # confidence from source agreement
    wc_cov = 1.0 - raw_frac[..., C_UNK]
    raw_veg = raw_frac[..., list(VEG_CATS)].sum(axis=-1)
    purity = raw_frac[..., 1:].max(axis=-1)
    conf = np.zeros((R, C), dtype=np.int8)
    ndvi_agrees = (nd_cov > 0.5) & (nd_raw_mean >= 0.3)
    f_conf = np.where((raw_veg >= cfg.purity_high) & ndvi_agrees, 3,
                      np.where(purity >= cfg.purity_medium, 2, 1))
    f_conf = np.where(wc_cov > 0, f_conf, 1)
    if osm is not None:
        osm_water = _cell_mean(osm.sub["water"].astype(np.float32), R, C, s) + osm.block_water
        osm_built = _cell_mean((osm.sub["building"] | osm.sub["landuse_built"]).astype(np.float32), R, C, s)
        osm_road = _cell_mean(osm.sub["road"].astype(np.float32), R, C, s) + osm.block_road
        osm_bare = _cell_mean(osm.sub["bare"].astype(np.float32), R, C, s)
    else:
        osm_water = osm_built = osm_road = osm_bare = np.zeros((R, C), dtype=np.float32)

    def barrier_conf(osm_has, wc_frac):
        both = (osm_has > 0) & (wc_frac >= 0.25)
        conflict = (osm_has > 0) & (wc_frac < 0.25) & (raw_veg >= cfg.purity_high)
        return np.where(both, 3, np.where(conflict, 1, 2))
    conf = np.where(classes == FUEL, f_conf, conf)
    conf = np.where(classes == WATER, barrier_conf(osm_water, raw_frac[..., C_WATER]), conf)
    conf = np.where(classes == BUILT, barrier_conf(osm_built, raw_frac[..., C_BUILT]), conf)
    conf = np.where(classes == NON_FUEL, barrier_conf(osm_bare, raw_frac[..., C_BARE]), conf)
    road_wc = raw_frac[..., C_BUILT] + raw_frac[..., C_BARE]
    conf = np.where(classes == ROAD, np.where((osm_road > 0) & (road_wc >= 0.25), 3, 2), conf)
    conf = np.where(unk, 0, conf).astype(np.int8)

    src = np.zeros((R, C), dtype=np.uint8)
    src |= np.where(wc_cov > 0, SRC_WORLDCOVER, 0).astype(np.uint8)
    if osm is not None:
        src |= np.where(osm.touched, SRC_OSM, 0).astype(np.uint8)
    src |= np.where(nd_cov > 0, SRC_NDVI, 0).astype(np.uint8)
    if dw is not None:
        pv = dw.sample(lat_s, lon_s, fill=np.nan).astype(np.float32)
        ok = np.isfinite(pv)
        cov = _cell_mean(ok.astype(np.float32), R, C, s)
        pmean = _cell_mean(np.where(ok, pv, 0.0).astype(np.float32), R, C, s) / np.maximum(cov, 1e-9)
        has = cov > 0.5
        src |= np.where(has, SRC_DW, 0).astype(np.uint8)
        up = has & (classes == FUEL) & (pmean >= 0.6)
        down = has & (classes == FUEL) & (pmean <= 0.3)
        conf = np.where(up, np.minimum(conf + 1, 3), np.where(down, np.maximum(conf - 1, 1), conf)).astype(np.int8)

    fractions = {"water": water, "built": built, "road": road, "bare": bare, "vegetation": veg, "unknown": unknown,
                 "tree": frac[..., C_TREE], "shrub": frac[..., C_SHRUB], "grass": frac[..., C_GRASS],
                 "crop": frac[..., C_CROP], "wetland": frac[..., C_WETLAND], "mangrove": frac[..., C_MANGROVE]}
    fractions = {k: v.astype(np.float32) for k, v in fractions.items()}
    return FusedGrids(classes, fuel_load, fuel_source, conf, src, fractions, s)
