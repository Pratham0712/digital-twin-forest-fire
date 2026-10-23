"""
event_validation.py - validate the rate-of-spread cellular automaton against
what VIIRS satellites actually saw, fire event by fire event, at the sensor's
own resolution (375 m).

The older back-test (spread_validation.py) works on a 0.1 degree zone grid
(~11 km), far coarser than the fire the ROS automaton simulates. This module
asks the real question: given the burning pixels of a fire event on day d, how
well does a spread model rank the pixels that are NEWLY detected on day d+1?

Design (everything below is computed from the archive, nothing is simulated):

* Events. VIIRS detections of one day are clustered (DBSCAN, 3 km); a cluster
  with >= `min_cells` distinct 375 m cells is a fire event. Static heat sources
  (kilns, steel plants) and low-confidence detections are dropped.
* Truth. Cells inside the event's window (seed box + `margin_km`) that have a
  detection on day d+1 and were not already burning on day d.
* Predictors, all scored on the SAME candidate cells:
    - distance to the nearest burning cell (the physics-free baseline; ranking
      by it is exactly "spread to the nearest neighbours first"),
    - ROS automaton arrival time, driven by the day's real FFMC / BUI / wind
      speed. The archive has no wind direction, so the plain automaton averages
      its arrival time over 8 wind directions (no directional information),
    - "assimilated" variants that learn from the PREVIOUS day: the wind
      direction is read off the observed d-1 -> d growth of the same event, and
      the effective spread duration (a single calibration scalar) is re-fitted
      on yesterday's growth before it is used to predict tomorrow's.
* Metrics. Per-event ROC-AUC of the ranking (threshold free; 0.5 = random) and
  pooled precision / recall / F1 of the set predicted at a calibrated
  duration. Calibration uses 2023-24 events only; 2025 and 2026 are held out.

Honest limits (reported with the numbers): flat terrain and uniform fuel (the
archive carries neither at 375 m), weather from the nearest 0.1 / 0.5 degree
zone, satellite overpasses see only part of each fire, and growth on day d+1
may come from a new, unrelated ignition inside the window.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.simulation.cellular_automata import CellState
from src.simulation.fire_behaviour import FuelModel, spread_rates
from src.simulation.ros_ca import RosGrids, cell_hash01, run_ros_ca

CELL_KM = 0.375
KM_PER_DEG_LAT = 110.57
KM_PER_DEG_LON_EQ = 111.32
DIRS8 = tuple(range(0, 360, 45))
TAU_GRID_H = (0.5, 1, 2, 3, 4, 6, 8, 12, 18, 24)
RADIUS_GRID = tuple(range(1, 13))
SIM_HOURS = 24
# At 375 m a VIIRS pixel is 'active' for the whole day, not for the ~4 min a 25 m cell flames: the 45-min
# residence cap of the local simulator would forbid any spread slower than 8 m/min across one pixel.
FLAME_MIN = SIM_HOURS * 60.0
RESIDENCE_MAX_MIN = SIM_HOURS * 60.0
FUEL = FuelModel(min_back_fraction=0.2, max_length_to_breadth=1.8)     # the live system's settings


# --------------------------------------------------------------------------- data
def load_pixels(paths: Sequence, static_xy: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """VIIRS pixels with date, lat, lon. Drops low confidence, non-vegetation
    (type != 0) and known static heat sources."""
    frames = []
    for p in paths:
        df = pd.read_csv(p, usecols=lambda c: c in ("latitude", "longitude", "acq_date", "confidence", "type"))
        frames.append(df)
    d = pd.concat(frames, ignore_index=True).drop_duplicates()
    if "confidence" in d:
        d = d[d["confidence"].astype(str).str.lower().isin(["n", "nominal", "h", "high"]) |
              pd.to_numeric(d["confidence"], errors="coerce").ge(30)]
    if "type" in d:
        d = d[d["type"].fillna(0).eq(0)]
    if static_xy is not None and len(static_xy):
        keys = set(zip(np.round(static_xy["latitude"], 2), np.round(static_xy["longitude"], 2)))
        k = list(zip(np.round(d["latitude"], 2), np.round(d["longitude"], 2)))
        d = d[[kk not in keys for kk in k]]
    d["date"] = pd.to_datetime(d["acq_date"]).dt.normalize()
    return d[["date", "latitude", "longitude"]].reset_index(drop=True)


def _project(lat, lon, lat0, lon0):
    x = (np.asarray(lon) - lon0) * KM_PER_DEG_LON_EQ * math.cos(math.radians(lat0))
    y = (np.asarray(lat) - lat0) * KM_PER_DEG_LAT
    return x, y


def cluster_day(day_px: pd.DataFrame, eps_km: float = 3.0) -> np.ndarray:
    """DBSCAN labels (-1 = noise) for one day's pixels."""
    from sklearn.cluster import DBSCAN
    if len(day_px) == 0:
        return np.zeros(0, int)
    lat0 = float(day_px["latitude"].mean())
    x, y = _project(day_px["latitude"].values, day_px["longitude"].values, lat0, 0.0)
    return DBSCAN(eps=eps_km, min_samples=2).fit_predict(np.c_[x, y])


# --------------------------------------------------------------------------- weather
class WeatherLookup:
    """Nearest-zone FFMC / BUI / wind speed for (lat, lon, date)."""

    def __init__(self, tables: Sequence[pd.DataFrame]):
        cols = ["latitude", "longitude", "date", "ffmc", "bui", "wx_wind_speed_ms"]
        parts = []
        for k, t in enumerate(tables):
            t = t[cols].copy()
            t["src"] = k
            parts.append(t)
        w = pd.concat(parts, ignore_index=True)
        w["date"] = pd.to_datetime(w["date"]).dt.normalize()
        w["zk"] = w["src"].astype(str) + "|" + w["latitude"].round(3).astype(str) + "|" + w["longitude"].round(3).astype(str)
        zones = w.drop_duplicates("zk")[["zk", "latitude", "longitude", "src"]].reset_index(drop=True)
        from scipy.spatial import cKDTree
        self._zones = zones
        self._src_trees = {}
        for s, g in zones.groupby("src"):
            g = g.reset_index(drop=True)
            lat0 = g["latitude"].mean()
            self._src_trees[s] = (cKDTree(np.c_[g["latitude"] * KM_PER_DEG_LAT,
                                                 g["longitude"] * KM_PER_DEG_LON_EQ * math.cos(math.radians(lat0))]),
                                  g, lat0)
        self._w = w.set_index(["zk", "date"]).sort_index()
        self._w = self._w[~self._w.index.duplicated(keep="first")]

    def at(self, lat: float, lon: float, date) -> Optional[Dict[str, float]]:
        best = None
        for s, (tree, g, lat0) in self._src_trees.items():
            d, i = tree.query([lat * KM_PER_DEG_LAT, lon * KM_PER_DEG_LON_EQ * math.cos(math.radians(lat0))])
            if best is None or d < best[0]:
                best = (d, g.iloc[int(i)]["zk"])
        if best is None or best[0] > 60.0:                 # >60 km from any zone: no honest weather
            return None
        try:
            r = self._w.loc[(best[1], pd.Timestamp(date).normalize())]
        except KeyError:
            return None
        return {"ffmc": float(r["ffmc"]), "bui": float(r["bui"]), "wind": float(r["wx_wind_speed_ms"])}


# --------------------------------------------------------------------------- events
@dataclass
class Event:
    date: pd.Timestamp
    lat: float
    lon: float
    shape: Tuple[int, int]
    seeds: np.ndarray                  # bool, day d
    truth: np.ndarray                  # bool, NEW cells day d+1
    prev: Optional[np.ndarray]         # bool, day d-1 cells in the same window (None if absent)
    weather: Dict[str, float]
    weather_prev: Optional[Dict[str, float]] = None
    meta: dict = field(default_factory=dict)


def _cells(lat, lon, lat0, lon0, y_min, x_min, shape):
    x, y = _project(lat, lon, lat0, lon0)
    i = np.floor((y - y_min) / CELL_KM).astype(int)
    j = np.floor((x - x_min) / CELL_KM).astype(int)
    ok = (i >= 0) & (i < shape[0]) & (j >= 0) & (j < shape[1])
    m = np.zeros(shape, bool)
    m[shape[0] - 1 - i[ok], j[ok]] = True          # row 0 = north
    return m


def build_events(px: pd.DataFrame, weather: WeatherLookup, dates: Sequence, min_cells: int = 4,
                 margin_km: float = 10.0, max_events: int = 400, seed: int = 0,
                 max_extent_km: float = 40.0) -> List[Event]:
    """Fire events on `dates` (each needs day d+1 in the archive)."""
    rng = np.random.default_rng(seed)
    by_day = {d: g for d, g in px.groupby("date")}
    days = [pd.Timestamp(d) for d in dates if pd.Timestamp(d) in by_day and
            pd.Timestamp(d) + pd.Timedelta(days=1) in by_day]
    rng.shuffle(days)
    cand: List[Tuple[pd.Timestamp, pd.DataFrame]] = []
    per_day = 6                                  # events per day, so one big day cannot dominate
    for d in days:
        if len(cand) >= max_events:
            break
        g = by_day[d]
        lab = cluster_day(g)
        ok = []
        for k in np.unique(lab[lab >= 0]):
            sub = g[lab == k]
            n_cells = len(set(zip(np.floor(sub["latitude"] / 0.0034).astype(int),
                                  np.floor(sub["longitude"] / 0.0034).astype(int))))
            if n_cells >= min_cells:
                ok.append(sub)
        if len(ok) > per_day:
            ok = [ok[i] for i in rng.choice(len(ok), per_day, replace=False)]
        cand += [(d, sub) for sub in ok]
    cand = cand[:max_events * 2]                 # a margin: some are dropped below (no weather, mega-cluster)
    events: List[Event] = []
    for d, sub in cand:
        lat0, lon0 = float(sub["latitude"].mean()), float(sub["longitude"].mean())
        x, y = _project(sub["latitude"].values, sub["longitude"].values, lat0, lon0)
        if (x.max() - x.min()) > max_extent_km or (y.max() - y.min()) > max_extent_km:
            continue                                   # a mega-cluster: not one fire event
        x_min, y_min = x.min() - margin_km, y.min() - margin_km
        w = int(math.ceil((x.max() + margin_km - x_min) / CELL_KM))
        h = int(math.ceil((y.max() + margin_km - y_min) / CELL_KM))
        shape = (h, w)
        wx = weather.at(lat0, lon0, d)
        if wx is None or not np.isfinite(list(wx.values())).all():
            continue
        def day_cells(day):
            g = by_day.get(day)
            if g is None:
                return None
            near = g[(g["latitude"].between(sub["latitude"].min() - 0.2, sub["latitude"].max() + 0.2)) &
                     (g["longitude"].between(sub["longitude"].min() - 0.2, sub["longitude"].max() + 0.2))]
            return _cells(near["latitude"].values, near["longitude"].values, lat0, lon0, y_min, x_min, shape)
        seeds = day_cells(d)
        nxt = day_cells(d + pd.Timedelta(days=1))
        prev = day_cells(d - pd.Timedelta(days=1))
        truth = nxt & ~seeds
        if prev is not None and not prev.any():
            prev = None
        events.append(Event(d, lat0, lon0, shape, seeds, truth, prev, wx,
                            weather.at(lat0, lon0, d - pd.Timedelta(days=1)) if prev is not None else None,
                            {"n_seed": int(seeds.sum()), "n_truth": int(truth.sum())}))
    return events


# --------------------------------------------------------------------------- predictors
def _grids(shape, key: int) -> RosGrids:
    rows, cols = np.indices(shape)
    thr = 1.0 + 0.45 * (2.0 * cell_hash01(rows, cols, key) - 1.0)
    g = RosGrids(np.ones(shape), np.zeros(shape), np.zeros(shape, bool), thr, CELL_KM * 1000.0)
    g.flame_mult = 0.75 + 0.5 * cell_hash01(rows + 7919, cols - 104729, key + 1)
    return g.build()


def ca_arrival(seeds: np.ndarray, wx: Dict[str, float], wind_from: float, key: int = 0,
               hours: float = SIM_HOURS) -> np.ndarray:
    """Arrival time (minutes, inf = not reached) of the ROS automaton."""
    rates = spread_rates(wx["ffmc"], wx["bui"], wx["wind"], wind_from, FUEL)
    state = np.where(seeds, CellState.BURNING, CellState.UNBURNED).astype(np.int8)
    run = run_ros_ca(state, _grids(seeds.shape, key), lambda t: rates, hours * 60.0, 60.0,
                     flame_min=FLAME_MIN, residence_max_min=RESIDENCE_MAX_MIN, rate_bound=rates.head)
    return run.t_ign


def ca_arrival_iso(seeds, wx, key: int = 0, hours: float = SIM_HOURS) -> np.ndarray:
    """Mean arrival time over 8 wind directions (no directional information)."""
    cap = hours * 60.0 * 1.5
    acc = np.zeros(seeds.shape)
    for k, d in enumerate(DIRS8):
        t = ca_arrival(seeds, wx, d, key + k, hours)
        acc += np.where(np.isfinite(t), t, cap)
    return acc / len(DIRS8)


def dist_to_seeds(seeds: np.ndarray) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt
    return distance_transform_edt(~seeds)


def growth_direction(prev: np.ndarray, cur: np.ndarray) -> Optional[float]:
    """Compass bearing (deg, 0 = north, clockwise) the fire moved between the
    previous and current day, from the centroids of the old and the newly
    burning cells; None if there is too little growth to read a direction."""
    new = cur & ~prev
    if new.sum() < 3 or prev.sum() < 2:
        return None
    ri, ci = np.nonzero(prev)
    rn, cn = np.nonzero(new)
    dy, dx = -(rn.mean() - ri.mean()), (cn.mean() - ci.mean())      # north = up
    if math.hypot(dx, dy) < 0.5:
        return None
    return math.degrees(math.atan2(dx, dy)) % 360.0


def auc(score: np.ndarray, truth: np.ndarray, candidates: np.ndarray) -> Optional[float]:
    """ROC-AUC with higher score = more likely; ties count half."""
    y = truth[candidates]
    s = score[candidates]
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _prf(pred, truth, cand):
    p = pred & cand
    return int((p & truth).sum()), int(p.sum()), int((truth & cand).sum())


def f1_from(tp, npred, ntruth):
    pr = tp / npred if npred else 0.0
    rc = tp / ntruth if ntruth else 0.0
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def best_tau(t_arrival: np.ndarray, seeds, truth, tau_grid=TAU_GRID_H) -> Optional[float]:
    cand = ~seeds
    if not (truth & cand).any():
        return None
    best, bf = None, -1.0
    for h in tau_grid:
        tp, npred, nt = _prf(np.isfinite(t_arrival) & (t_arrival <= h * 60.0), truth, cand)
        f = f1_from(tp, npred, nt)
        if f > bf:
            best, bf = h, f
    return best if bf > 0 else None


def best_radius(dist: np.ndarray, seeds, truth, grid=RADIUS_GRID) -> Optional[int]:
    cand = ~seeds
    if not (truth & cand).any():
        return None
    best, bf = None, -1.0
    for r in grid:
        tp, npred, nt = _prf(dist <= r, truth, cand)
        f = f1_from(tp, npred, nt)
        if f > bf:
            best, bf = r, f
    return best if bf > 0 else None


def evaluate(events: Sequence[Event], tau_global: Optional[float] = None, r_global: Optional[int] = None,
             with_assimilation: bool = True) -> Tuple[pd.DataFrame, dict]:
    """Per-event scores + the arrival/distance maps needed for calibration are
    recomputed here; pooled counts are returned in `pooled` per method."""
    rows = []
    pooled: Dict[str, List[int]] = {}
    for n, ev in enumerate(events):
        cand = ~ev.seeds
        if not (ev.truth & cand).any():
            continue
        dist = dist_to_seeds(ev.seeds)
        t_iso = ca_arrival_iso(ev.seeds, ev.weather, key=n)
        cap = SIM_HOURS * 60.0 * 1.5
        # unreached cells are ranked after every reached one, nearest first
        s_ca = -(np.where(np.isfinite(t_iso), t_iso, cap) + 0.01 * dist)
        s_dist = -dist
        row = {"date": ev.date, "n_seed": ev.meta["n_seed"], "n_truth": ev.meta["n_truth"],
               "auc_dist": auc(s_dist, ev.truth, cand), "auc_ca_iso": auc(s_ca, ev.truth, cand)}
        preds = {}
        if tau_global is not None:
            preds["CA (isotropic), global duration"] = np.isfinite(t_iso) & (t_iso <= tau_global * 60.0)
        if r_global is not None:
            preds["Dilation, global radius"] = dist <= r_global
        # --- assimilation: events with an observed previous day
        row["has_prev"] = False
        if with_assimilation and ev.prev is not None and ev.weather_prev is not None:
            wdir = growth_direction(ev.prev, ev.seeds)
            prev_truth = ev.seeds & ~ev.prev
            t_prev = ca_arrival_iso(ev.prev, ev.weather_prev, key=n + 5000)
            tau_e = best_tau(t_prev, ev.prev, prev_truth)
            r_e = best_radius(dist_to_seeds(ev.prev), ev.prev, prev_truth)
            row["has_prev"] = True
            if wdir is not None:
                t_dir = ca_arrival(ev.seeds, ev.weather, (wdir + 180.0) % 360.0, key=n + 9000)   # wind FROM opposite
                s_dir = -(np.where(np.isfinite(t_dir), t_dir, cap) + 0.01 * dist)
                row["auc_ca_dir"] = auc(s_dir, ev.truth, cand)
                if tau_global is not None:
                    preds["CA (observed direction), global duration"] = (np.isfinite(t_dir) &
                                                                          (t_dir <= tau_global * 60.0))
                if tau_e is not None:
                    preds["CA (observed direction + yesterday-fitted duration)"] = (np.isfinite(t_dir) &
                                                                                    (t_dir <= tau_e * 60.0))
            if tau_e is not None:
                preds["CA (isotropic), yesterday-fitted duration"] = np.isfinite(t_iso) & (t_iso <= tau_e * 60.0)
            if r_e is not None:
                preds["Dilation, yesterday-fitted radius"] = dist <= r_e
            # persistence of growth: predict yesterday's NEW cells shifted by nothing (cells that burned last night)
        for name, p in preds.items():
            tp, npred, nt = _prf(p, ev.truth, cand)
            pooled.setdefault(name, [0, 0, 0, 0])
            a = pooled[name]
            a[0] += tp; a[1] += npred; a[2] += nt; a[3] += 1
            row[f"f1[{name}]"] = f1_from(tp, npred, nt)
        rows.append(row)
    return pd.DataFrame(rows), {k: v for k, v in pooled.items()}


def calibrate(events: Sequence[Event], seed_key: int = 0) -> Tuple[float, int, pd.DataFrame]:
    """Pick the global duration tau and dilation radius on TRAINING events only
    (maximum pooled F1)."""
    tau_stats = {h: [0, 0, 0] for h in TAU_GRID_H}
    r_stats = {r: [0, 0, 0] for r in RADIUS_GRID}
    for n, ev in enumerate(events):
        cand = ~ev.seeds
        if not (ev.truth & cand).any():
            continue
        t_iso = ca_arrival_iso(ev.seeds, ev.weather, key=n + seed_key)
        dist = dist_to_seeds(ev.seeds)
        for h in TAU_GRID_H:
            a = tau_stats[h]
            tp, npred, nt = _prf(np.isfinite(t_iso) & (t_iso <= h * 60.0), ev.truth, cand)
            a[0] += tp; a[1] += npred; a[2] += nt
        for r in RADIUS_GRID:
            a = r_stats[r]
            tp, npred, nt = _prf(dist <= r, ev.truth, cand)
            a[0] += tp; a[1] += npred; a[2] += nt
    rows = [{"method": "CA duration (h)", "value": h, "f1": f1_from(*v)} for h, v in tau_stats.items()]
    rows += [{"method": "dilation radius (cells)", "value": r, "f1": f1_from(*v)} for r, v in r_stats.items()]
    tab = pd.DataFrame(rows)
    tau = float(tab[tab.method == "CA duration (h)"].sort_values("f1", ascending=False).iloc[0]["value"])
    rad = int(tab[tab.method == "dilation radius (cells)"].sort_values("f1", ascending=False).iloc[0]["value"])
    return tau, rad, tab
