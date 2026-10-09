"""Land-cover SOURCES (OFFLINE FIXTURE TESTS - no download, no real data).

ESA WorldCover and Sentinel-2 are exercised through the production readers on
small synthetic GeoTIFFs written by the tests (same lattice / CRS conventions as
the real products); the tile cache, expiry and fallbacks use counting fakes.
None of this is a real-data verification (see scripts/verify_landcover_sources.py)."""
import json
from datetime import timedelta

import numpy as np
import pytest

from config.config import LAND_COVER
from src.landcover import sentinel2_ndvi, tile_cache, worldcover
from src.landcover.sentinel2_ndvi import load_window as REAL_NDVI_LOAD
from src.landcover.tile_cache import Tile, TileCache, load_tiled, tiles_for
from src.landcover.worldcover import load_window as REAL_WC_LOAD

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

RES = 1.0 / 12000.0
B = {"north": 11.675, "south": 11.665, "west": 76.630, "east": 76.640}     # inside one 0.01 deg tile row


@pytest.fixture
def online(monkeypatch, tmp_path):
    """Allow the (local-file) 'downloads' and isolate the cache."""
    monkeypatch.setenv("FIRE_LAND_COVER_FETCH", "1")
    monkeypatch.setenv("FIRE_LANDCOVER_CACHE_DIR", str(tmp_path / "cache"))
    tile_cache.clear_failures()
    return tmp_path


def _write_wc_tif(path, west=76.62, north=11.69, n=360, painter=None):
    a = np.full((n, n), 10, dtype=np.uint8)                       # tree cover
    if painter:
        painter(a)
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1, dtype="uint8", crs="EPSG:4326",
                       transform=from_origin(west, north, RES, RES), tiled=True, blockxsize=256, blockysize=256,
                       compress="deflate") as ds:
        ds.write(a, 1)
    return a


# 1 ─ WorldCover raster loading (production reader, local COG fixture) ─────────
def test_01_worldcover_window_read_and_cache(online, monkeypatch):
    tif = online / "N09E075.tif"

    def paint(a):
        a[120:240, 120:240] = 80                                   # water: lat 11.67-11.68, lon 76.63-76.64
    _write_wc_tif(tif, painter=paint)
    monkeypatch.setattr(LAND_COVER, "worldcover_url", str(online / "{tile}.tif"))
    monkeypatch.setattr(worldcover, "load_window", REAL_WC_LOAD)
    win, st = worldcover.load_window(B, allow_fetch=True)
    assert st["status"] == "downloaded" and st["missing"] == 0 and st["fetched"] == st["tiles"] == 2
    assert win.res_deg == pytest.approx(RES) and win.label.startswith("ESA WorldCover 10 m")
    # pixel centred at 11.672 N 76.635 E is water, 11.668 N is tree
    assert win.sample(np.array([11.672]), np.array([76.635]))[0, 0] == 80
    assert win.sample(np.array([11.668]), np.array([76.635]))[0, 0] == 10
    tif.unlink()                                                    # second read: from the cache only
    win2, st2 = worldcover.load_window(B, allow_fetch=True)
    assert st2["status"] == "cached" and st2["fetched"] == 0 and np.array_equal(win.array, win2.array)
    meta = json.loads(next((online / "cache" / "worldcover").rglob("*.json")).read_text())
    assert meta["dataset"] == "worldcover" and meta["version"] == LAND_COVER.worldcover_version
    assert {"retrieved_utc", "expires_utc", "bounds", "res_deg", "url"} <= set(meta)


def test_worldcover_tile_naming_and_tiles():
    assert worldcover.wc_tile_name(11.67, 76.63) == "N09E075"            # all of Bandipur
    assert worldcover.wc_tile_name(-0.5, -0.5) == "S03W003"
    ts = tiles_for(B)
    assert len(ts) == 2 and all(t.deg == LAND_COVER.cache_tile_deg for t in ts)
    assert worldcover.px_per_tile() == 120


# 12, 13, 14, 15, 49 ─ cache hit / miss / expiry / fallback / no repeated download ─
def _counting_fetch(calls, value=7, fail=False):
    def fetch(tiles):
        calls.append([t.tile_id for t in tiles])
        if fail:
            raise ConnectionError("outage")
        return {t.tile_id: ({"v": np.full((4, 4), value, np.uint8)}, {"source": "fake"}) for t in tiles}
    return fetch


def test_12_13_49_cache_miss_then_hit_and_no_repeated_download(online):
    calls = []
    cache = TileCache(online / "c")
    w1, s1 = load_tiled("fake", "v1", B, 30, 4, "v", np.uint8, 0, _counting_fetch(calls), True, cache)
    assert s1["status"] == "downloaded" and len(calls) == 1 and cache.misses == 2
    for _ in range(3):
        w2, s2 = load_tiled("fake", "v1", B, 30, 4, "v", np.uint8, 0, _counting_fetch(calls), True, cache)
        assert s2["status"] == "cached"
    assert len(calls) == 1 and cache.hits == 6                       # downloaded exactly once
    assert np.array_equal(w1.array, w2.array)


def test_14_expired_tile_is_refreshed(online):
    calls = []
    cache = TileCache(online / "c")
    load_tiled("fake", "v1", B, 30, 4, "v", np.uint8, 0, _counting_fetch(calls, 1), True, cache)
    for js in (online / "c").rglob("*.json"):                       # age the cache beyond its TTL
        m = json.loads(js.read_text())
        m["retrieved_utc"] = tile_cache.iso(tile_cache.utc_now() - timedelta(days=40))
        js.write_text(json.dumps(m))
    w, s = load_tiled("fake", "v1", B, 30, 4, "v", np.uint8, 0, _counting_fetch(calls, 2), True, cache)
    assert len(calls) == 2 and s["status"] == "downloaded" and (w.array == 2).all()


def test_15_fallback_stale_cache_then_unavailable(online):
    calls = []
    cache = TileCache(online / "c")
    load_tiled("fake", "v1", B, 30, 4, "v", np.uint8, 0, _counting_fetch(calls, 3), True, cache)
    for js in (online / "c").rglob("*.json"):
        m = json.loads(js.read_text())
        m["retrieved_utc"] = tile_cache.iso(tile_cache.utc_now() - timedelta(days=40))
        js.write_text(json.dumps(m))
    w, s = load_tiled("fake", "v1", B, 30, 4, "v", np.uint8, 0, _counting_fetch(calls, fail=True), True, cache)
    assert s["status"] == "stale" and (w.array == 3).all() and "outage" in s["error"]   # outage keeps the cache
    tile_cache.clear_failures()
    w, s = load_tiled("fake", "v2", B, 30, 4, "v", np.uint8, 0, _counting_fetch(calls, fail=True), True, cache)
    assert w is None and s["status"] == "unavailable" and s["missing"] == 2            # nothing invented


def test_download_disabled_offline(online, monkeypatch):
    monkeypatch.setenv("FIRE_LAND_COVER_FETCH", "0")
    calls = []
    w, s = load_tiled("fake", "v1", B, 30, 4, "v", np.uint8, 0, _counting_fetch(calls), True, TileCache(online))
    assert w is None and calls == [] and "offline" in s["error"]


def test_tile_alignment_and_mosaic():
    t = Tile(1166, 7663, 0.01)
    assert (t.south, t.north, t.west, t.east) == pytest.approx((11.66, 11.67, 76.63, 76.64))
    assert t.tile_id == "n001166_e007663"


# Sentinel-2 NDVI through the production reader: UTM GeoTIFF fixtures + mocked STAC ─
def _write_utm(path, value, dtype="uint16"):
    from rasterio.warp import transform as tr
    xs, ys = tr("EPSG:4326", "EPSG:32643", [76.62, 76.65], [11.69, 11.66])
    west, north = min(xs) - 200, max(ys) + 200
    n = int((max(xs) - min(xs) + 400) / 10)
    m = int((max(ys) - min(ys) + 400) / 10)
    arr = np.full((m, n), value, dtype=dtype)
    with rasterio.open(path, "w", driver="GTiff", height=m, width=n, count=1, dtype=dtype, crs="EPSG:32643",
                       transform=from_origin(west, north, 10, 10)) as ds:
        ds.write(arr, 1)


def test_sentinel2_ndvi_reader_with_scl_mask_and_median(online, monkeypatch):
    items = []
    for k, (red, nir, scl) in enumerate([(1500, 4500, 4), (1600, 4800, 4), (1000, 1100, 9)]):   # 3rd = cloud
        paths = {}
        for band, val in (("red", red), ("nir", nir), ("scl", scl)):
            p = online / f"s{k}_{band}.tif"
            _write_utm(p, val, "uint8" if band == "scl" else "uint16")
            paths[band] = str(p)
        items.append({"id": f"S2X_{k}", "properties": {"datetime": f"2026-09-0{k + 1}T05:00:00Z", "eo:cloud_cover": 5,
                                                      "s2:processing_baseline": "05.10"},
                      "assets": {b: {"href": paths[b], "raster:bands": [{"scale": 0.0001, "offset": -0.1}]}
                                 for b in ("red", "nir", "scl")}})
    monkeypatch.setattr(sentinel2_ndvi, "stac_search", lambda bbox: items)
    monkeypatch.setattr(sentinel2_ndvi, "load_window", REAL_NDVI_LOAD)
    win, st = sentinel2_ndvi.load_window(B, allow_fetch=True)
    assert st["status"] == "downloaded" and win.label.startswith("Sentinel-2 L2A NDVI")
    v = win.sample(np.array([11.67]), np.array([76.635]))[0, 0]
    ndvi = lambda r, n: ((n * 1e-4 - 0.1) - (r * 1e-4 - 0.1)) / ((n * 1e-4 - 0.1) + (r * 1e-4 - 0.1))
    assert v == pytest.approx(np.median([ndvi(1500, 4500), ndvi(1600, 4800)]), abs=1e-3)   # cloud scene masked
    assert win.meta["scene_dates"] == ["2026-09-01", "2026-09-02", "2026-09-03"]


def test_sentinel2_unavailable_is_reported(online, monkeypatch):
    def boom(bbox):
        raise ConnectionError("STAC unreachable")
    monkeypatch.setattr(sentinel2_ndvi, "stac_search", boom)
    monkeypatch.setattr(sentinel2_ndvi, "load_window", REAL_NDVI_LOAD)
    win, st = sentinel2_ndvi.load_window(B, allow_fetch=True)
    assert win is None and st["status"] == "unavailable" and "STAC unreachable" in st["error"]
