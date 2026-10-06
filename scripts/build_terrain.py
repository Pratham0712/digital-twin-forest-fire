"""
build_terrain.py - download real elevation for a region's grid and save the
per-cell terrain table (elev_m, slope_pct) to models/terrain_<region>.csv.

Karnataka (the default) is 1,400 cells x 9 sample points = 12,600 elevations,
about 126 requests to the free OpenTopoData API (1 request per second) - roughly 2-3 minutes.

Run:
    python scripts/build_terrain.py                       # Karnataka
    python scripts/build_terrain.py --region "Odisha"     # any preset in src/regions.py
    python scripts/build_terrain.py --all                 # every preset region (about 10 minutes)

Safe to re-run. It never writes a partial table: if any elevation is missing
the script stops with an error and leaves the previous file untouched.

After it finishes:
    python src/ml_models/train_real.py      # retrains with terrain features
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.config import MODELS_DIR, REGION
from src.data_ingestion.ingestion_module import build_region_grid
from src.data_ingestion.terrain import build_terrain_table, fetch_elevations_cached, terrain_filename

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default=None, help="preset name from src/regions.py (default: Karnataka)")
    ap.add_argument("--all", action="store_true", help="build terrain for every preset region")
    ap.add_argument("--provider", default="auto", choices=["auto", "opentopodata", "open-meteo", "open-elevation"])
    args = ap.parse_args()

    if args.all:
        from src.regions import REGION_PRESETS
        regions = list({r.name: r for r in REGION_PRESETS.values()}.values())
    else:
        region = REGION
        if args.region:
            from src.regions import REGION_PRESETS
            matches = [r for k, r in REGION_PRESETS.items() if args.region.lower() in k.lower()]
            if not matches:
                sys.exit(f"Unknown region '{args.region}'. Options: {', '.join(REGION_PRESETS)}")
            region = matches[0]
        regions = [region]
    for region in regions:
        build_one(region, args.provider)


def build_one(region, provider):
    grid = build_region_grid(region)
    n = len(grid) * 9
    print(f"{region.name}: {len(grid)} cells, {n} elevation samples")

    def progress(done, total):
        if done == total or done % 1000 < 100:
            print(f"  {done}/{total}", flush=True)

    from config.config import DATA_RAW_DIR
    cache = DATA_RAW_DIR / "elevation_cache.csv"          # shared by all regions, resumable
    table = build_terrain_table(
        grid, region.grid_resolution_deg,
        fetch=lambda la, lo: fetch_elevations_cached(la, lo, cache, provider=provider))

    out = MODELS_DIR / terrain_filename(region.name)
    table.to_csv(out, index=False)
    print(f"\nSaved {out}")
    print(table[["elev_m", "slope_pct"]].describe().round(1).to_string())
    print("\nSteepest cells:")
    print(table.nlargest(5, "slope_pct")[["zone_id", "latitude", "longitude", "elev_m", "slope_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
