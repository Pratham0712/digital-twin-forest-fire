"""
fire_history.py - the v2 feature set: satellite fire history + zone fire
climatology, computed by ONE shared implementation for both training
(train_real.py) and live inference (DigitalTwin.refresh), so there is no
train/serve skew.

Why these features
------------------
The v1 model only saw same-day weather/FWI. Weather is necessary but not
sufficient: two zones with identical weather have very different fire odds
if one of them has been burning for three days, or sits in a landscape that
burns every single season. Adding that context lifted AUC-ROC on the held-out
2025 season from 0.81 to 0.88 and more than doubled average precision
(0.15 -> 0.33) - see data/processed/model_comparison_real.csv.

Leakage rules (all enforced here, not by convention)
----------------------------------------------------
* Fire-history features for day t use ONLY detections from days t-1 ... t-10.
  Day t's own detections (the label) are never used. At live inference this
  is exactly what the system has: NASA FIRMS near-real-time detections for
  the previous 10 days (FIRMS area API, DAY_RANGE=10).
* Lookback never crosses a gap in the data (the training data has one fire
  season per year, Jan-May, so Jan 1 has no "yesterday" in the data).
* Zone climatology for a training row is computed from the OTHER training
  season(s) only (leave-one-season-out); for test rows and live inference it
  comes from the training seasons only. A row's own label never feeds its
  own feature.

Hotspot -> cell mapping uses the same confidence filter (VIIRS nominal/high)
and the same 0.7 x grid-cell radius as the label definition in
build_real_dataset.py, via the shared helpers below.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HISTORY_DAYS = 10                  # FIRMS NRT area API allows up to 10 days
FIRE_MATCH_RADIUS_FACTOR = 0.7     # x grid_resolution_deg - same as the label

FIRE_HISTORY_COLUMNS = [
    "fire_lag1",          # zone had a confirmed detection yesterday
    "fire_7d",            # detections in zone over the previous 7 days
    "fire_10d",           # ... previous 10 days
    "nbr3_lag1",          # detections yesterday in the 8 surrounding cells
    "nbr5_7d",            # detections in the 5x5 neighbourhood, previous 7 days
    "days_since_fire",    # days since last detection in zone (capped)
    "region_fires_lag1",  # region-wide detections yesterday (burning-period signal)
]
CLIMATOLOGY_COLUMNS = ["zone_clim", "nbr_clim", "latitude", "longitude"]

# The original (v1) weather-only feature set - kept for the baseline row.
BASE_FEATURE_COLUMNS = [
    "wx_temperature_c", "wx_humidity_pct", "wx_wind_speed_ms", "wx_precipitation_mm",
    "ffmc", "dmc", "dc", "bui", "fwi", "ndvi",
    "month", "day_of_year", "fwi_lag1",
]
# v2 drops "ndvi": it was a synthetic per-zone placeholder, not a measurement,
# so it only acted as a zone fingerprint. The per-zone prior now comes from
# the real fire climatology instead.
WEATHER_FEATURE_COLUMNS = [c for c in BASE_FEATURE_COLUMNS if c != "ndvi"]
FEATURE_COLUMNS_V2 = WEATHER_FEATURE_COLUMNS + CLIMATOLOGY_COLUMNS + FIRE_HISTORY_COLUMNS
# v3 = v2 + real terrain (elevation, slope) from models/terrain_*.csv, built by
# scripts/build_terrain.py. train_real.py uses it automatically when that file exists.
from src.data_ingestion.terrain import TERRAIN_COLUMNS  # noqa: E402  (kept next to the column lists it extends)
FEATURE_COLUMNS_V3 = FEATURE_COLUMNS_V2 + TERRAIN_COLUMNS

CLIMATOLOGY_FILENAME = "zone_climatology.csv"
STATIC_SOURCES_FILENAME = "static_heat_sources.csv"
STATIC_SOURCE_RADIUS_DEG = 0.0075   # ~800 m: two VIIRS 375 m pixels


# ── shared primitives ────────────────────────────────────────────────────── #

def confident_detections(fires: pd.DataFrame) -> pd.DataFrame:
    """Keep likely vegetation fires only:
      * VIIRS confidence nominal/high (or numeric >= 30);
      * FIRMS `type` == 0 ("presumed vegetation fire") when the column is
        present (archive data). Type 2 = static land source (steel plants,
        kilns, flares), 3 = offshore - not forest fires. The NRT feed has no
        `type`, so live data is cleaned with StaticSourceMask instead."""
    if fires.empty:
        return fires
    if "type" in fires.columns:
        fires = fires[pd.to_numeric(fires["type"], errors="coerce").fillna(0) == 0]
    if "confidence" not in fires.columns:
        return fires
    conf = fires["confidence"].astype(str).str.strip().str.lower()
    if conf.isin(["l", "n", "h"]).any():
        return fires[conf.isin(["n", "h"])]
    return fires[pd.to_numeric(conf, errors="coerce").fillna(0) >= 30]


class StaticSourceMask:
    """Locations FIRMS flagged as static (industrial) heat sources in the
    archive. The live NRT feed doesn't carry that flag, so live detections
    within STATIC_SOURCE_RADIUS_DEG of a known source are dropped - the same
    cleaning the training labels got."""

    def __init__(self, points: Optional[pd.DataFrame] = None):
        self.points = points if points is not None and not points.empty else None
        self._tree = cKDTree(self.points[["latitude", "longitude"]].values) if self.points is not None else None

    @classmethod
    def from_archive(cls, fires: pd.DataFrame) -> "StaticSourceMask":
        if "type" not in fires.columns:
            return cls(None)
        s = fires[pd.to_numeric(fires["type"], errors="coerce") == 2]
        pts = (s.assign(latitude=s["latitude"].round(3), longitude=s["longitude"].round(3))
                [["latitude", "longitude"]].drop_duplicates().reset_index(drop=True))
        return cls(pts)

    @classmethod
    def load(cls, models_dir: Path) -> "StaticSourceMask":
        """Union of every static_heat_sources*.csv (Karnataka + India builds)."""
        frames = [pd.read_csv(p) for p in sorted(Path(models_dir).glob("static_heat_sources*.csv"))]
        frames = [f for f in frames if not f.empty]
        return cls(pd.concat(frames, ignore_index=True) if frames else None)

    def save(self, models_dir: Path, filename: str = STATIC_SOURCES_FILENAME):
        (self.points if self.points is not None else pd.DataFrame(columns=["latitude", "longitude"])) \
            .to_csv(Path(models_dir) / filename, index=False)

    def filter(self, hotspots: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if self._tree is None or hotspots is None or hotspots.empty:
            return hotspots
        dist, _ = self._tree.query(hotspots[["latitude", "longitude"]].values.astype(float), k=1)
        return hotspots[dist > STATIC_SOURCE_RADIUS_DEG].reset_index(drop=True)


def cell_flags_for_detections(grid_coords: np.ndarray, fire_coords: np.ndarray,
                              grid_resolution_deg: float) -> np.ndarray:
    """1 for every grid cell whose centre lies within the match radius of a
    detection, else 0 - the exact rule used for the training label."""
    flags = np.zeros(len(grid_coords), dtype=np.float32)
    if len(fire_coords) == 0:
        return flags
    dist, _ = cKDTree(fire_coords).query(grid_coords, k=1)
    flags[dist <= grid_resolution_deg * FIRE_MATCH_RADIUS_FACTOR] = 1.0
    return flags


def _box_sum(cube: np.ndarray, r: int) -> np.ndarray:
    """Sum over a (2r+1)x(2r+1) spatial window for every day (zero padded)."""
    p = np.pad(cube, ((0, 0), (r, r), (r, r)))
    c = np.pad(p.cumsum(1).cumsum(2), ((0, 0), (1, 0), (1, 0)))
    k = 2 * r + 1
    return c[:, k:, k:] - c[:, :-k, k:] - c[:, k:, :-k] + c[:, :-k, :-k]


def history_feature_cubes(cube: np.ndarray, segment: np.ndarray) -> dict:
    """
    cube:    (D, n_rows, n_cols) daily 0/1 detection flags, days in order.
    segment: (D,) id of the contiguous run each day belongs to; lookback never
             crosses into a different segment.
    Returns {feature_name: (D, n_rows, n_cols)} where day t's value uses only
    days t-HISTORY_DAYS ... t-1.
    """
    D = cube.shape[0]
    nbr3 = _box_sum(cube, 1) - cube
    nbr5 = _box_sum(cube, 2)
    region_total = cube.sum(axis=(1, 2))

    def lagged(a, w):
        out = np.zeros_like(a)
        for t in range(1, D):
            lo = max(0, t - w)
            while lo < t and segment[lo] != segment[t]:
                lo += 1
            if lo < t:
                out[t] = a[lo:t].sum(axis=0)
        return out

    feats = {
        "fire_lag1": lagged(cube, 1),
        "fire_7d": lagged(cube, 7),
        "fire_10d": lagged(cube, HISTORY_DAYS),
        "nbr3_lag1": lagged(nbr3, 1),
        "nbr5_7d": lagged(nbr5, 7),
    }
    cap = HISTORY_DAYS + 1  # "no detection within the lookback window"
    dsf = np.full(cube.shape, cap, dtype=np.float32)
    last = np.full(cube.shape[1:], -10**6)
    for t in range(D):
        if t > 0 and segment[t] != segment[t - 1]:
            last[:] = -10**6
        dsf[t] = np.minimum(t - last, cap)
        last = np.where(cube[t] > 0, t, last)
    feats["days_since_fire"] = dsf

    rt = np.zeros(D, dtype=np.float32)
    for t in range(1, D):
        if segment[t] == segment[t - 1]:
            rt[t] = region_total[t - 1]
    feats["region_fires_lag1"] = np.broadcast_to(rt[:, None, None], cube.shape).copy()
    return feats


# ── training side ─────────────────────────────────────────────────────────── #

def add_fire_history_training(df: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Adds FIRE_HISTORY_COLUMNS to the real training dataset. `grid` is
    build_region_grid() output (zone_id, row, col)."""
    df = df.merge(grid[["zone_id", "row", "col"]], on="zone_id", how="left")
    dates = np.sort(df["date"].unique())
    day_idx = pd.Series(np.arange(len(dates)), index=pd.DatetimeIndex(dates))
    di = day_idx.reindex(pd.DatetimeIndex(df["date"])).values
    n_rows, n_cols = int(grid["row"].max()) + 1, int(grid["col"].max()) + 1

    cube = np.zeros((len(dates), n_rows, n_cols), dtype=np.float32)
    cube[di, df["row"].values, df["col"].values] = df["fire_risk_label"].values
    # A new segment starts wherever consecutive data days are > 1 day apart.
    gaps = np.diff(pd.DatetimeIndex(dates).values).astype("timedelta64[D]").astype(int) > 1
    segment = np.concatenate([[0], np.cumsum(gaps)])

    for name, c in history_feature_cubes(cube, segment).items():
        df[name] = c[di, df["row"].values, df["col"].values]
    return df.drop(columns=["row", "col"])


def _smooth_on_grid(values: pd.Series, grid: pd.DataFrame) -> pd.Series:
    """5x5 neighbourhood mean of a per-zone value."""
    g = grid.set_index("zone_id")
    arr = np.zeros((int(g["row"].max()) + 1, int(g["col"].max()) + 1), dtype=np.float32)
    v = values.reindex(g.index).fillna(values.mean())
    arr[g["row"].values, g["col"].values] = v.values
    sm = _box_sum(arr[None], 2)[0] / 25.0
    return pd.Series(sm[g["row"].values, g["col"].values], index=g.index)


def add_climatology_training(df: pd.DataFrame, grid: pd.DataFrame,
                             train_mask: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adds zone_clim / nbr_clim without target leakage:
      * training rows of season S get the fire rate from the other training seasons;
      * all other rows (test) get the rate from every training season.
    Returns (df_with_features, climatology_table_for_live_use).
    """
    df = df.copy()
    season = df["date"].dt.year
    train_seasons = sorted(season[train_mask].unique())
    df["zone_clim"] = np.nan
    df["nbr_clim"] = np.nan

    def rates(mask):
        return df.loc[mask].groupby("zone_id")["fire_risk_label"].mean()

    for s in train_seasons:
        others = train_mask & (season != s)
        r = rates(others) if others.any() else rates(train_mask)
        rows = train_mask & (season == s)
        df.loc[rows, "zone_clim"] = df.loc[rows, "zone_id"].map(r).values
        df.loc[rows, "nbr_clim"] = df.loc[rows, "zone_id"].map(_smooth_on_grid(r, grid)).values

    full = rates(train_mask)
    rest = ~train_mask
    df.loc[rest, "zone_clim"] = df.loc[rest, "zone_id"].map(full).values
    df.loc[rest, "nbr_clim"] = df.loc[rest, "zone_id"].map(_smooth_on_grid(full, grid)).values

    table = grid[["zone_id", "latitude", "longitude"]].copy()
    table["zone_clim"] = table["zone_id"].map(full).fillna(full.mean()).values
    table["nbr_clim"] = table["zone_id"].map(_smooth_on_grid(full, grid)).values
    return df, table


# ── live side ─────────────────────────────────────────────────────────────── #

def live_fire_history_features(grid: pd.DataFrame, hotspots: Optional[pd.DataFrame],
                               grid_resolution_deg: float,
                               today: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    FIRE_HISTORY_COLUMNS for every live grid cell, from the FIRMS detections
    of the previous HISTORY_DAYS days. `grid` needs zone_id,row,col,latitude,
    longitude. Runs the same history_feature_cubes() used in training on an
    (HISTORY_DAYS + 1)-day cube whose last day is today.
    """
    now = pd.Timestamp.now(tz="UTC") if today is None else pd.Timestamp(today)
    if now.tzinfo is not None:
        now = now.tz_convert("UTC").tz_localize(None)
    today = now.normalize()
    n_rows, n_cols = int(grid["row"].max()) + 1, int(grid["col"].max()) + 1
    D = HISTORY_DAYS + 1
    cube = np.zeros((D, n_rows, n_cols), dtype=np.float32)
    coords = grid[["latitude", "longitude"]].values

    if hotspots is not None and not hotspots.empty and "acq_date" in hotspots.columns:
        fires = confident_detections(hotspots).copy()
        fires["_d"] = pd.to_datetime(fires["acq_date"], errors="coerce").dt.normalize()
        for k in range(1, D):                       # k days ago -> cube index D-1-k
            day_fires = fires[fires["_d"] == today - pd.Timedelta(days=k)]
            flags = cell_flags_for_detections(
                coords, day_fires[["latitude", "longitude"]].values.astype(float),
                grid_resolution_deg)
            cube[D - 1 - k, grid["row"].values, grid["col"].values] = flags

    feats = history_feature_cubes(cube, np.zeros(D, dtype=int))
    out = pd.DataFrame({"zone_id": grid["zone_id"].values})
    for name in FIRE_HISTORY_COLUMNS:
        out[name] = feats[name][D - 1, grid["row"].values, grid["col"].values]
    return out


class ZoneClimatology:
    """Per-cell historical fire rate learned from the training seasons, looked
    up by coordinates (zone_ids are only unique within a region). Cells with
    no history - e.g. a region the model wasn't trained on - get the training
    mean, so the model still runs but relies on weather + live fire history."""

    def __init__(self, table: Optional[pd.DataFrame] = None):
        self.table = table

    @staticmethod
    def _key(lat, lon):
        return list(zip(np.round(np.asarray(lat, float), 2), np.round(np.asarray(lon, float), 2)))

    @classmethod
    def load(cls, models_dir: Path, filename: str = CLIMATOLOGY_FILENAME) -> "ZoneClimatology":
        path = Path(models_dir) / filename
        return cls(pd.read_csv(path) if path.exists() else None)

    def save(self, models_dir: Path, filename: str = CLIMATOLOGY_FILENAME):
        self.table.to_csv(Path(models_dir) / filename, index=False)

    def transform(self, grid: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame({"zone_id": grid["zone_id"].values})
        if self.table is None or self.table.empty:
            out["zone_clim"] = 0.0
            out["nbr_clim"] = 0.0
            return out
        t = self.table
        idx = dict(zip(self._key(t["latitude"], t["longitude"]), range(len(t))))
        pos = [idx.get(k, -1) for k in self._key(grid["latitude"], grid["longitude"])]
        pos = np.array(pos)
        for col in ("zone_clim", "nbr_clim"):
            vals = t[col].values
            out[col] = np.where(pos >= 0, vals[np.clip(pos, 0, None)], float(vals.mean()))
        return out
