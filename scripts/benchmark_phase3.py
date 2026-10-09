"""
Phase 3 benchmark (offline, no API calls, no downloads): the WHAT-IF run path
with the legacy probability CA and the default rate-of-spread CA, for the
domain presets. Prints wall-clock time, CA time, time per sub-step, domain
expansion time, number of expansions, burned area and simulated duration.
Usage: python scripts/benchmark_phase3.py
"""
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("FIRE_LAND_COVER_FETCH", "0")
logging.disable(logging.WARNING)


def main():
    from src.simulation.local_spread import FOCUS_AREAS, FocusArea, initial_domain, run_local_spread
    b = FOCUS_AREAS["Bandipur Tiger Reserve"]
    cond = {"ffmc": 93.0, "ndvi": 0.6, "bui": 60.0}
    cases = [  # (label, focus m, cell m, duration min, wind m/s, domain size m or None)
        ("1 km focus, Auto domain, 60 min", 1000, 25, 60, 8.0, None),
        ("1 km focus, Small 2 km domain, 60 min", 1000, 25, 60, 8.0, (2000, 2000)),
        ("1 km focus, Medium 5 km domain, 120 min", 1000, 25, 120, 8.0, (5000, 5000)),
        ("1 km focus, Large 10 km domain, 120 min", 1000, 25, 120, 8.0, (10000, 10000)),
        ("500 m focus, 10 m cells, Auto, 30 min", 500, 10, 30, 6.0, None),
        ("1 km focus, Auto, 240 min, wind 12 m/s", 1000, 25, 240, 12.0, None),
    ]
    print(f"{'scenario':<44}{'model':>8}{'total s':>9}{'CA s':>8}{'ms/sub':>8}{'exp s':>7}{'#exp':>5}"
          f"{'burned ha':>10}{'sim min':>9}{'grid':>12}")
    for label, size, cell, dur, wind, dsize in cases:
        f = FocusArea(b.name, b.lat, b.lon, size, size, cell)
        for model in ("legacy", "ros"):
            t0 = time.perf_counter()
            dom = initial_domain(f, dur, [], domain_size_m=dsize)
            r = run_local_spread(f, cond, wind, 241.0, placement="Centre", n_ignition=3, duration_minutes=dur,
                                 seed=42, domain=dom, model=model)
            dt = time.perf_counter() - t0
            nsub = r.params.get("n_substeps") or len(r.history) - 1
            ca = r.timings.get("ca_simulation", 0.0)
            print(f"{label:<44}{model:>8}{dt:>9.3f}{ca:>8.3f}{1000 * ca / max(nsub, 1):>8.2f}"
                  f"{r.timings.get('domain_expansion', 0.0):>7.3f}{len(r.expansions):>5}"
                  f"{r.final.get('fire_area_ha', 0):>10.2f}{r.final.get('minutes', 0):>9.1f}"
                  f"{f'{r.domain.n_cols}x{r.domain.n_rows}':>12}")


if __name__ == "__main__":
    main()
