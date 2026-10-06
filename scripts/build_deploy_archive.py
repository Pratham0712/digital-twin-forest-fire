"""
Build the small data files the DEPLOYED dashboard needs.

The raw FIRMS archives (~200 MB) are too big to commit. The Historical Time
Machine only needs five columns of detections that fall inside a preset region,
so this writes them to one gzip-compressed file (data/archive/fires_presets.csv.gz).

    python scripts/build_deploy_archive.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.config import DATA_RAW_DIR, REGION  # noqa: E402
from src.regions import REGION_PRESETS  # noqa: E402

KEEP = ["latitude", "longitude", "acq_date", "frp", "confidence"]
OUT = ROOT / "data" / "archive" / "fires_presets.csv.gz"


def main():
    sources = [Path(DATA_RAW_DIR) / "historical_fires_karnataka.csv"]
    sources += sorted(Path(DATA_RAW_DIR).glob("historical_fires_india_*.csv"))
    regions = list({r.name: r for r in REGION_PRESETS.values()}.values())
    parts = []
    for path in sources:
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=lambda c: c in KEEP)
        mask = pd.Series(False, index=df.index)
        for r in regions:
            mask |= (df["latitude"].between(r.min_lat, r.max_lat)
                     & df["longitude"].between(r.min_lon, r.max_lon))
        parts.append(df[mask])
        print(f"{path.name}: kept {int(mask.sum()):,} of {len(df):,}")
    out = pd.concat(parts, ignore_index=True).drop_duplicates()
    out["frp"] = out["frp"].round(1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, compression="gzip")
    print(f"Saved {len(out):,} detections -> {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
