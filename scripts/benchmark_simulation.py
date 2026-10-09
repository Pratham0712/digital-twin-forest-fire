"""
Benchmark of the local simulation pipeline (OFFLINE: synthetic land-cover
fixtures, no download). Prints a table of measured times:

  * fused land cover (WorldCover-format + NDVI fixtures + OSM vectors) per domain size
  * CA run on a fixed domain (current default and larger domains)
  * adaptive run that starts small and expands

    python scripts/benchmark_simulation.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    os.environ["FIRE_LAND_COVER_FETCH"] = "0"
    import numpy as np
    from src.landcover import osm_vectors, provider, sentinel2_ndvi, worldcover
    from src.landcover.grid import GridSpec
    from src.landcover.tile_cache import RasterWindow
    from src.simulation.fuel_map import FUEL, LandCover
    from src.simulation.local_spread import FocusArea, SimulationDomain, domain_for, run_local_spread

    res = 1 / 12000

    def fixture(fill, dtype):
        def load(b, allow_fetch=True, cache=None):
            n = int((b["north"] - b["south"]) / res) + 40
            m = int((b["east"] - b["west"]) / res) + 40
            return RasterWindow(np.full((n, m), fill, dtype), b["north"] + 20 * res, b["west"] - 20 * res, res,
                                "BENCHMARK FIXTURE"), {"status": "fixture", "tiles": 1, "missing": 0}
        return load
    worldcover.load_window = fixture(10, np.uint8)
    sentinel2_ndvi.load_window = fixture(0.6, np.float32)
    lat, lon = 11.6667, 76.6333
    rows = []
    for n in (54, 120, 240, 480):
        d = 25.0 / 111320.0
        dl = 25.0 / (111320.0 * np.cos(np.radians(lat)))
        spec = GridSpec(lat + n / 2 * d, lat - n / 2 * d, lon - n / 2 * dl, lon + n / 2 * dl, n, n, 25.0)
        provider.clear_memory_cache()
        t = time.perf_counter()
        provider.grid_land_cover(spec, allow_fetch=False)
        rows.append((f"fused land cover {n}x{n} @25 m (cold memory)", time.perf_counter() - t))
        t = time.perf_counter()
        provider.grid_land_cover(spec, allow_fetch=False)
        rows.append((f"fused land cover {n}x{n} (in-memory hit)", time.perf_counter() - t))
    hot = {"ffmc": 97.0, "ndvi": 0.8, "bui": 80.0, "risk_score": 0.9, "zone_id": 1, "fwi": 60.0}
    f = FocusArea("bench", lat, lon, 500, 500, 25)
    for dur in (60, 120, 240):
        dom = domain_for(f, dur)
        lc = LandCover(np.full((dom.n_rows, dom.n_cols), FUEL, np.int8), "fused", "fixture", 0, {}, status="full",
                       fuel_load=np.full((dom.n_rows, dom.n_cols), 0.8))
        t = time.perf_counter()
        r = run_local_spread(f, hot, 8.0, 250.0, domain=dom, land_cover=lc, duration_minutes=dur, seed=1,
                             base_spread_prob=1.0)
        rows.append((f"CA fixed domain {dom.n_rows}x{dom.n_cols}, {dur} min (burned {r.final['burned']})",
                     time.perf_counter() - t))

    def lc_for(spec):
        shape = (spec.n_rows, spec.n_cols)
        return LandCover(np.full(shape, FUEL, np.int8), "fused", "fixture", 0, {}, status="full",
                         fuel_load=np.full(shape, 0.8))
    small = SimulationDomain(f, 6)
    t = time.perf_counter()
    r = run_local_spread(f, hot, 8.0, 250.0, domain=small, land_cover=lc_for(small), land_provider=lc_for,
                         duration_minutes=240, seed=1, base_spread_prob=1.0)
    rows.append((f"adaptive 240 min: {small.n_rows}x{small.n_cols} -> {r.domain.n_rows}x{r.domain.n_cols}, "
                 f"{len(r.expansions)} expansions", time.perf_counter() - t))
    print(f"{'benchmark':<78}{'seconds':>10}")
    for name, sec in rows:
        print(f"{name:<78}{sec:>10.3f}")
    print("\nadaptive run stage timings:", r.timings)


if __name__ == "__main__":
    main()
