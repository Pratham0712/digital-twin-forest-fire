"""
tile_cache.py - persistent window/tile cache for raster land-cover products.

Every raster product is cached in square geographic tiles of
LAND_COVER.cache_tile_deg (0.01 deg, about 1.1 km; exactly 120 x 120 pixels of
the 1/12000-deg WorldCover lattice). A tile is downloaded once and then read
from disk; a simulation only touches the tiles that intersect its domain.

Layout (default root models/land_cover/tiles, override FIRE_LANDCOVER_CACHE_DIR):

    <root>/<dataset>/<version>/<tile_id>.npz    arrays
    <root>/<dataset>/<version>/<tile_id>.json   metadata: dataset, version,
        tile bounds, resolution, source URL (never containing a secret),
        retrieved_utc, expires_utc, processing version, extra details

Expiry: a tile older than the dataset's TTL is "expired". It is still returned
(flagged) when a fresh download is impossible, so one temporary outage never
destroys an otherwise usable cached simulation; the UI shows its retrieval
date. A new dataset/processing version uses a new directory, so stale formats
are never mixed in.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.config import LAND_COVER, MODELS_DIR


def cache_root() -> Path:
    env = os.getenv("FIRE_LANDCOVER_CACHE_DIR", "").strip()
    return Path(env) if env else MODELS_DIR / "land_cover" / "tiles"


def fetch_allowed(allow_fetch: bool = True) -> bool:
    """Network access for land-cover data (FIRE_LAND_COVER_FETCH=0 disables it,
    e.g. for the automated tests)."""
    return bool(allow_fetch) and os.getenv("FIRE_LAND_COVER_FETCH", "1") != "0"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s: str) -> Optional[datetime]:
    try:
        return datetime.strptime(str(s), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


# ── tiles ─────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class Tile:
    i_lat: int                 # south edge = i_lat * deg
    i_lon: int                 # west edge  = i_lon * deg
    deg: float

    @property
    def south(self) -> float:
        return self.i_lat * self.deg

    @property
    def north(self) -> float:
        return (self.i_lat + 1) * self.deg

    @property
    def west(self) -> float:
        return self.i_lon * self.deg

    @property
    def east(self) -> float:
        return (self.i_lon + 1) * self.deg

    @property
    def tile_id(self) -> str:
        return f"{'n' if self.i_lat >= 0 else 's'}{abs(self.i_lat):06d}_{'e' if self.i_lon >= 0 else 'w'}{abs(self.i_lon):06d}"

    @property
    def bounds(self) -> Dict[str, float]:
        return {"north": self.north, "south": self.south, "west": self.west, "east": self.east}


def tiles_for(bounds: Dict[str, float], deg: float = None) -> List[Tile]:
    """All cache tiles intersecting bounds (row-major, north first)."""
    deg = float(deg or LAND_COVER.cache_tile_deg)
    eps = 1e-9
    i0, i1 = math.floor(bounds["south"] / deg + eps), math.floor(bounds["north"] / deg - eps)
    j0, j1 = math.floor(bounds["west"] / deg + eps), math.floor(bounds["east"] / deg - eps)
    return [Tile(i, j, deg) for i in range(i1, i0 - 1, -1) for j in range(j0, j1 + 1)]


@dataclass
class RasterWindow:
    """A north-up raster on a regular lat/lon lattice."""
    array: np.ndarray
    north: float
    west: float
    res_deg: float
    label: str = ""
    meta: Dict = field(default_factory=dict)

    def sample(self, lat: np.ndarray, lon: np.ndarray, fill=0):
        """Nearest-pixel values at the outer product of 1-D lat (rows) and lon
        (cols) arrays -> (len(lat), len(lon)); outside the window -> fill."""
        r = np.floor((self.north - np.asarray(lat)) / self.res_deg + 1e-9).astype(np.int64)
        c = np.floor((np.asarray(lon) - self.west) / self.res_deg + 1e-9).astype(np.int64)
        rok = (r >= 0) & (r < self.array.shape[0])
        cok = (c >= 0) & (c < self.array.shape[1])
        out = self.array[np.ix_(np.clip(r, 0, self.array.shape[0] - 1), np.clip(c, 0, self.array.shape[1] - 1))]
        out = np.array(out, copy=True)
        if not (rok.all() and cok.all()):
            if out.dtype.kind == "f" and fill is None:
                fill = np.nan
            out[~(rok[:, None] & cok[None, :])] = fill
        return out


def mosaic(tiles: List[Tile], arrays: Dict[str, np.ndarray], px: int, dtype, fill) -> RasterWindow:
    """Stitch per-tile arrays (px x px, missing -> fill) into one window."""
    ilats = sorted({t.i_lat for t in tiles}, reverse=True)
    ilons = sorted({t.i_lon for t in tiles})
    out = np.full((len(ilats) * px, len(ilons) * px), fill, dtype=dtype)
    for t in tiles:
        a = arrays.get(t.tile_id)
        if a is None:
            continue
        r = ilats.index(t.i_lat) * px
        c = ilons.index(t.i_lon) * px
        out[r:r + px, c:c + px] = a
    deg = tiles[0].deg
    return RasterWindow(out, (ilats[0] + 1) * deg, ilons[0] * deg, deg / px)


# ── cache ─────────────────────────────────────────────────────────────────── #

class TileCache:
    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else cache_root()
        self.hits = 0
        self.misses = 0

    def _paths(self, dataset: str, version: str, tile_id: str) -> Tuple[Path, Path]:
        d = self.root / dataset / version.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        return d / f"{tile_id}.npz", d / f"{tile_id}.json"

    def get(self, dataset: str, version: str, tile: Tile, ttl_days: float) -> Optional[Tuple[Dict[str, np.ndarray], dict]]:
        """(arrays, meta) or None. meta["expired"] tells whether the TTL passed."""
        npz, js = self._paths(dataset, version, tile.tile_id)
        if not (npz.exists() and js.exists()):
            self.misses += 1
            return None
        try:
            meta = json.loads(js.read_text(encoding="utf-8"))
            with np.load(npz, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
        except Exception:
            self.misses += 1
            return None
        got = parse_iso(meta.get("retrieved_utc", ""))
        meta["expired"] = got is None or utc_now() - got > timedelta(days=float(ttl_days))
        self.hits += 1
        return arrays, meta

    def put(self, dataset: str, version: str, tile: Tile, arrays: Dict[str, np.ndarray], meta: dict,
            ttl_days: float) -> dict:
        npz, js = self._paths(dataset, version, tile.tile_id)
        npz.parent.mkdir(parents=True, exist_ok=True)
        now = utc_now()
        meta = {"dataset": dataset, "version": version, "tile_id": tile.tile_id, "bounds": tile.bounds,
                "retrieved_utc": iso(now), "expires_utc": iso(now + timedelta(days=float(ttl_days))), **meta}
        # atomic writes: a crash never leaves a half-written tile behind
        fd, tmp = tempfile.mkstemp(dir=npz.parent, suffix=".npz")
        os.close(fd)
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, npz)
        fd, tmp = tempfile.mkstemp(dir=npz.parent, suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
        os.replace(tmp, js)
        return {**meta, "expired": False}


_FAILED_AT: Dict[str, float] = {}


def recently_failed(key: str) -> bool:
    return time.time() - _FAILED_AT.get(key, 0.0) < LAND_COVER.retry_after_failure_s


def mark_failed(key: str):
    _FAILED_AT[key] = time.time()


def clear_failures():
    _FAILED_AT.clear()


# ── generic tiled loader (cache first, then one batched download) ─────────── #

def load_tiled(dataset: str, version: str, bounds: Dict[str, float], ttl_days: float, px: int, key: str,
               dtype, fill, fetch_fn, allow_fetch: bool = True, cache: Optional[TileCache] = None):
    """Window covering `bounds` from the tile cache, downloading missing or
    expired tiles with fetch_fn(tiles) -> {tile_id: (arrays, meta)} when
    allowed. Returns (RasterWindow or None, status dict). Nothing is invented:
    a tile that is neither cached nor downloadable stays `fill` and is
    reported as missing."""
    cache = cache or TileCache()
    tiles = tiles_for(bounds)
    arrays: Dict[str, np.ndarray] = {}
    metas: Dict[str, dict] = {}
    need: List[Tile] = []
    stale: Dict[str, Tuple[np.ndarray, dict]] = {}
    for t in tiles:
        got = cache.get(dataset, version, t, ttl_days)
        if got is not None and key in got[0]:
            if got[1].get("expired"):
                stale[t.tile_id] = (got[0][key], got[1])
                need.append(t)
            else:
                arrays[t.tile_id], metas[t.tile_id] = got[0][key], got[1]
        else:
            need.append(t)
    fetched, error = 0, ""
    fkey = f"{dataset}:{version}"
    if need and fetch_allowed(allow_fetch) and not recently_failed(fkey):
        try:
            res = fetch_fn(need) or {}
            for t in need:
                if t.tile_id in res:
                    arr, meta = res[t.tile_id]
                    try:
                        m = cache.put(dataset, version, t, arr, meta, ttl_days)
                    except OSError as exc:            # e.g. file locked by another reader on Windows
                        m = {**meta, "retrieved_utc": iso(utc_now()), "expired": False,
                             "cache_write_error": str(exc)[:200]}
                    arrays[t.tile_id], metas[t.tile_id] = arr[key], m
                    stale.pop(t.tile_id, None)
                    fetched += 1
        except Exception as exc:                      # outage, timeout, missing dependency ...
            mark_failed(fkey)
            error = f"{type(exc).__name__}: {exc}"[:300]
    elif need and not fetch_allowed(allow_fetch):
        error = "download disabled (offline)"
    elif need:
        error = "recent download failure; retry suppressed for a few minutes"
    for tid, (arr, meta) in stale.items():            # expired but usable: better than nothing, flagged
        arrays[tid], metas[tid] = arr, meta
    n, have = len(tiles), len(arrays)
    missing = n - have
    retrieved = sorted(m.get("retrieved_utc", "") for m in metas.values() if m.get("retrieved_utc"))
    status = {"tiles": n, "cached": have - fetched - len(stale), "fetched": fetched, "stale": len(stale),
              "missing": missing, "retrieved_first": retrieved[0] if retrieved else None,
              "retrieved_last": retrieved[-1] if retrieved else None, "error": error,
              "status": ("unavailable" if have == 0 else "partial" if missing else
                         "stale" if stale else "downloaded" if fetched else "cached")}
    if have == 0:
        return None, status
    win = mosaic(tiles, arrays, px, dtype, fill)
    win.meta = {**status, "tile_meta": metas}
    return win, status
