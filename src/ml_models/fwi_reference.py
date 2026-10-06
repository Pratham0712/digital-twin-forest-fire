"""
fwi_reference.py - "this FWI exceeds X% of the historical record" without
reading the 100 MB training CSV on every page view.

The reference distribution is reduced once to 101 quantiles and stored in
models/fwi_quantiles.csv (a few KB, so it deploys with the model). If that
file is missing it is built from data/processed/real_training_data.csv on
first use and written next to the model; after that lookups are instant.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from config.config import DATA_PROCESSED_DIR, MODELS_DIR

logger = logging.getLogger(__name__)

QUANTILE_FILE = "fwi_quantiles.csv"
_cache: dict = {}


def build_quantiles(source: Optional[Path] = None, models_dir: Path = MODELS_DIR) -> Optional[np.ndarray]:
    import pandas as pd
    source = Path(source) if source else Path(DATA_PROCESSED_DIR) / "real_training_data.csv"
    if not source.exists():
        return None
    fwi = pd.read_csv(source, usecols=["fwi"])["fwi"].dropna().to_numpy(float)
    if fwi.size == 0:
        return None
    q = np.quantile(fwi, np.linspace(0, 1, 101))
    try:
        pd.DataFrame({"percentile": np.arange(101), "fwi": q}).to_csv(Path(models_dir) / QUANTILE_FILE, index=False)
    except OSError as exc:                       # read-only deployment: still usable this run
        logger.warning("Could not write %s (%s)", QUANTILE_FILE, exc)
    return q


def load_quantiles(models_dir: Path = MODELS_DIR) -> Optional[np.ndarray]:
    key = str(models_dir)
    if key not in _cache:
        import pandas as pd
        path = Path(models_dir) / QUANTILE_FILE
        q = None
        if path.exists():
            try:
                q = pd.read_csv(path)["fwi"].to_numpy(float)
            except Exception:
                q = None
        _cache[key] = q if q is not None else build_quantiles(models_dir=models_dir)
    return _cache[key]


def fwi_percentile(value: float, models_dir: Path = MODELS_DIR) -> Optional[float]:
    """Share (0-100) of historical zone-days with an FWI below `value`."""
    q = load_quantiles(models_dir)
    if q is None or value is None or not np.isfinite(value):
        return None
    q = np.maximum.accumulate(q)                  # guard: quantiles must be non-decreasing
    return float(np.interp(value, q, np.linspace(0, 100, len(q))))
