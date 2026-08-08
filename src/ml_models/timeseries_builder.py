"""
TimeSeriesBuilder - constructs a multi-day feature history per grid zone.
The CNN+LSTM model (Report PPT spec) needs sequence input (day-over-day trend
in FWI/weather), but a single live API pull only gives one snapshot. This
module simulates a trailing N-day window per zone with realistic day-to-day
autocorrelation (today's weather depends on yesterday's, not independent
random draws) so the sequence has learnable temporal structure.

In production this is replaced by querying stored daily snapshots from the
database (SRS 5.1: system retains rolling history) - the function signatures
are kept identical so swapping the data source doesn't touch model code.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_history(processed_snapshot: pd.DataFrame, n_days: int = 7,
                   seed: Optional[int] = 21) -> np.ndarray:
    """
    Given today's processed snapshot (output of DataProcessor.transform),
    generates a (n_zones, n_days, n_features) array simulating the trailing
    n_days of [ffmc, dmc, dc, bui, fwi, ndvi, wx_temperature_c, wx_humidity_pct]
    leading up to today, using an AR(1)-style walk anchored on today's value so
    the sequence trends realistically instead of being pure noise.
    """
    rng = np.random.default_rng(seed)
    feature_cols = ["ffmc", "dmc", "dc", "bui", "fwi", "ndvi",
                     "wx_temperature_c", "wx_humidity_pct"]
    today = processed_snapshot[feature_cols].values  # (n_zones, n_features)
    n_zones, n_features = today.shape

    sequences = np.zeros((n_zones, n_days, n_features))
    sequences[:, -1, :] = today

    # Walk backward from today with small AR(1) perturbations per feature,
    # scaled to each feature's own std so e.g. DC (slow-changing) drifts less
    # day-to-day than FWI (fast-changing).
    feature_std = processed_snapshot[feature_cols].std().values
    feature_std = np.where(feature_std < 1e-3, 1.0, feature_std)

    for day in range(n_days - 2, -1, -1):
        noise = rng.normal(0, 0.08, size=(n_zones, n_features)) * feature_std
        sequences[:, day, :] = sequences[:, day + 1, :] - noise

    sequences[:, :, feature_cols.index("ndvi")] = np.clip(
        sequences[:, :, feature_cols.index("ndvi")], -1, 1)
    for idx in [feature_cols.index(c) for c in ["ffmc", "dmc", "dc", "bui", "fwi", "wx_humidity_pct"]]:
        sequences[:, :, idx] = np.clip(sequences[:, :, idx], 0, None)

    logger.info("Built time-series history: shape=%s (zones, days, features)", sequences.shape)
    return sequences, feature_cols
