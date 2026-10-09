"""
REAL-DATA VERIFICATION of the land-cover sources (run on your own machine).

This is NOT part of the automated test suite (the tests use offline fixtures).
It downloads small windows of the real data for one simulation domain, fills
the local tile cache, and reports exactly what was obtained:

  * ESA WorldCover 10 m v200 (2021)  - windowed COG reads (public AWS bucket)
  * OpenStreetMap (Overpass API)     - roads, buildings, water, bare ground
  * Sentinel-2 L2A NDVI              - Earth Search STAC + windowed COG reads
  * the fused land cover on the CA grid (classes, confidence, fuel-load source)

No API key is used or printed. Expected download for the default 500 m focus
area: well under 10 MB (only the 0.01-degree tiles the domain touches).

Usage (from the repository root, with the project's virtual environment):

    pip install rasterio
    python scripts/verify_landcover_sources.py
    python scripts/verify_landcover_sources.py --lat 11.70 --lon 76.55 --size 1000 --cell 25 --duration 60

Writes models/land_cover/verification_<timestamp>.json and prints a summary.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lat", type=float, default=11.6667, help="focus centre latitude (default: Bandipur preset)")
    ap.add_argument("--lon", type=float, default=76.6333)
    ap.add_argument("--size", type=float, default=500.0, help="focus width = height in metres")
    ap.add_argument("--cell", type=float, default=25.0, help="CA cell size in metres (5, 10, 25 or 50)")
    ap.add_argument("--duration", type=float, default=60.0, help="simulated minutes (sets the initial domain)")
    args = ap.parse_args()
    os.environ.setdefault("FIRE_LAND_COVER_FETCH", "1")

    import numpy as np
    from src.landcover import osm_vectors, provider, sentinel2_ndvi, worldcover
    from src.landcover.grid import GridSpec
    from src.landcover.tile_cache import cache_root, tiles_for
    from src.simulation.local_spread import FocusArea, initial_domain
    from src.utils.timing import Timings

    try:
        import rasterio  # noqa: F401
        has_rasterio = True
    except ImportError:
        has_rasterio = False
    f = FocusArea("verification", args.lat, args.lon, args.size, args.size, args.cell)
    dom = initial_domain(f, args.duration)
    spec = GridSpec.from_domain(dom)
    print(f"Domain: {dom.n_rows} x {dom.n_cols} cells of {args.cell:g} m around ({args.lat}, {args.lon}); "
          f"{len(tiles_for(spec.bounds))} cache tile(s) of 0.01 deg; cache: {cache_root()}")
    if not has_rasterio:
        print("rasterio is NOT installed: WorldCover and Sentinel-2 cannot be read (pip install rasterio).")

    report = {"domain": dom.describe(), "cell_m": args.cell, "rasterio": has_rasterio, "sources": {}}
    t = time.perf_counter()
    wc, wst = worldcover.load_window(spec.bounds, allow_fetch=True)
    report["sources"]["worldcover"] = {**wst, "seconds": round(time.perf_counter() - t, 2)}
    if wc is not None:
        vals, cnt = np.unique(wc.sample(*spec.sub_axes(1)), return_counts=True)
        report["sources"]["worldcover"]["classes_on_grid"] = {worldcover.WC_NAMES.get(int(v), str(v)): int(c)
                                                              for v, c in zip(vals, cnt)}
    t = time.perf_counter()
    data, ost = osm_vectors.load_osm(spec, allow_fetch=True)
    report["sources"]["osm"] = {**ost, "seconds": round(time.perf_counter() - t, 2),
                                "features": len(data.get("elements", [])) if data else 0}
    t = time.perf_counter()
    nd, nst = sentinel2_ndvi.load_window(spec.bounds, allow_fetch=True)
    report["sources"]["sentinel2"] = {**nst, "seconds": round(time.perf_counter() - t, 2),
                                      "scene_dates": nd.meta.get("scene_dates") if nd is not None else []}
    if nd is not None:
        v = nd.sample(*spec.sub_axes(1), fill=None)
        report["sources"]["sentinel2"]["ndvi_valid_fraction"] = round(float(np.isfinite(v).mean()), 3)
        report["sources"]["sentinel2"]["ndvi_median"] = (round(float(np.nanmedian(v)), 3)
                                                         if np.isfinite(v).any() else None)
    tm = Timings()
    provider.clear_memory_cache()
    lc = provider.domain_land_cover(dom, allow_fetch=True, timings=tm)
    report["fused"] = {"status": lc.status, "label": lc.label, "notes": lc.notes, **provider.summary(lc),
                       "timings_s": tm.as_dict()}
    out = ROOT / "models" / "land_cover" / f"verification_{time.strftime('%Y%m%dT%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\nREAL-DATA VERIFICATION")
    for k, v in report["sources"].items():
        print(f"  {k:<11} status={v.get('status')}  tiles={v.get('tiles', '-')}  downloaded={v.get('fetched', '-')}  "
              f"cached={v.get('cached', '-')}  {v.get('seconds')} s  {('error: ' + v['error']) if v.get('error') else ''}")
    print(f"  fused land cover: {lc.status} - {lc.label}")
    print(f"  classes: {report['fused']['classes']}")
    print(f"  confidence: {report['fused'].get('confidence')}")
    print(f"  fuel-load source: {report['fused'].get('fuel_load_source')}")
    print(f"\nReport written to {out}")
    ok = report["sources"]["worldcover"].get("status") in ("downloaded", "cached", "stale")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
