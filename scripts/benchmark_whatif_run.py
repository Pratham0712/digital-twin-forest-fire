"""
Time the WHAT-IF run path (offline): ignition-aware initial domain, cached DEM
lookup, CA with adaptive expansion, analytics. No land-cover acquisition is
part of this path. Usage: python scripts/benchmark_whatif_run.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("FIRE_LAND_COVER_FETCH", "0")


def main():
    from src.simulation.local_spread import FocusArea, domain_elevation, initial_domain, run_local_spread
    cond = {"ffmc": 93.0, "ndvi": 0.7, "bui": 50.0}
    print(f"{'scenario':<52}{'total s':>9}{'CA s':>8}{'expansions':>12}")
    for size, cell, dur, wind in ((500, 25, 60, 6.0), (500, 25, 120, 6.0), (1000, 25, 240, 8.0), (500, 10, 60, 6.0)):
        f = FocusArea("Bandipur Tiger Reserve", 11.6667, 76.6333, size, size, cell)
        t0 = time.perf_counter()
        dom = initial_domain(f, dur, [])
        elev, src = domain_elevation(dom, allow_fetch=False)
        r = run_local_spread(f, cond, wind, 250.0, placement="Upwind edge", n_ignition=3, duration_minutes=dur,
                             seed=42, domain=dom, elevation=elev, terrain_source=src,
                             elevation_provider=lambda s: domain_elevation(s, allow_fetch=False))
        dt = time.perf_counter() - t0
        print(f"{f'{size} m area, {cell} m cells, {dur} min, wind {wind} m/s':<52}{dt:>9.3f}"
              f"{r.timings.get('ca_simulation', 0):>8.3f}{len(r.expansions):>12}")


if __name__ == "__main__":
    main()
