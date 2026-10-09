"""Performance benchmarks (49-52), OFFLINE FIXTURES. Bounds are deliberately
loose regression guards (CI machines vary); the measured numbers are printed
and reported by scripts/benchmark_simulation.py."""
import time

import numpy as np
import pytest

from src.landcover import provider, sentinel2_ndvi, tile_cache, worldcover
from src.landcover.grid import GridSpec
from src.landcover.worldcover import load_window as REAL_WC_LOAD
from src.simulation.fuel_map import FUEL, LandCover
from src.simulation.local_spread import FocusArea, SimulationDomain, run_local_spread
from tests.landcover_fixtures import ndvi_loader

LAT, LON = 11.6667, 76.6333


def _spec(n, cell=25.0):
    dlat = cell / 111320.0
    dlon = cell / (111320.0 * np.cos(np.radians(LAT)))
    return GridSpec(LAT + n / 2 * dlat, LAT - n / 2 * dlat, LON - n / 2 * dlon, LON + n / 2 * dlon, n, n, cell)


def test_49_50_warm_cache_is_fast_and_downloads_nothing(monkeypatch, tmp_path):
    calls = []

    def fake_fetch(tiles):
        calls.append(len(tiles))
        return {t.tile_id: ({"classes": np.full((120, 120), 10, np.uint8)}, {"source": "fixture"}) for t in tiles}
    monkeypatch.setenv("FIRE_LAND_COVER_FETCH", "1")
    monkeypatch.setenv("FIRE_LANDCOVER_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(worldcover, "load_window", REAL_WC_LOAD)
    monkeypatch.setattr(worldcover, "fetch_tiles", fake_fetch)
    tile_cache.clear_failures()
    spec = _spec(120)
    t0 = time.perf_counter()
    provider.grid_land_cover(spec, allow_fetch=True)
    cold = time.perf_counter() - t0
    provider.clear_memory_cache()                     # warm DISK cache, cold memory cache
    t0 = time.perf_counter()
    lc = provider.grid_land_cover(spec, allow_fetch=True)
    warm = time.perf_counter() - t0
    t0 = time.perf_counter()
    provider.grid_land_cover(spec, allow_fetch=True)
    hot = time.perf_counter() - t0
    print(f"\nland cover 120x120 @25 m: cold {cold:.3f}s (fixture download), warm disk {warm:.3f}s, memory {hot:.4f}s")
    assert len(calls) == 1 and lc.sources["worldcover"]["status"] == "cached"
    assert warm < 3.0 and hot < 0.5


def test_51_fusion_benchmark_largest_initial_domain(monkeypatch):
    monkeypatch.setattr(sentinel2_ndvi, "load_window", ndvi_loader(0.6))
    provider.clear_memory_cache()
    t = {}
    for n in (54, 120, 240):
        t0 = time.perf_counter()
        lc = provider.grid_land_cover(_spec(n), allow_fetch=False)
        t[n] = time.perf_counter() - t0
        assert lc.classes.shape == (n, n)
    print("\nfusion: " + ", ".join(f"{n}x{n} {v:.3f}s" for n, v in t.items()))
    assert t[240] < 8.0


def test_52_ca_expansion_benchmark():
    f = FocusArea(name="t", lat=LAT, lon=LON, width_m=250, height_m=250, cell_m=25)

    def lc_for(spec):
        shape = (spec.n_rows, spec.n_cols)
        return LandCover(np.full(shape, FUEL, np.int8), "fused", "fixture", 0, {}, status="full",
                         fuel_load=np.full(shape, 0.8))
    d0 = SimulationDomain(f, 4)
    hot = {"ffmc": 97.0, "ndvi": 0.8, "bui": 80.0, "risk_score": 0.9, "zone_id": 1, "fwi": 60.0}
    t0 = time.perf_counter()
    res = run_local_spread(f, hot, 9.0, 250.0, domain=d0, land_cover=lc_for(d0), land_provider=lc_for, n_ignition=2,
                           placement="Centre", seed=3, duration_minutes=240, base_spread_prob=1.0)
    dt = time.perf_counter() - t0
    print(f"\nadaptive run 240 min @25 m: {dt:.2f}s, {len(res.expansions)} expansions, final "
          f"{res.domain.n_rows}x{res.domain.n_cols}, timings {res.timings}")
    assert res.expansions and dt < 60.0
