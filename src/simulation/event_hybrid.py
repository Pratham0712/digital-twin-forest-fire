"""
event_hybrid.py - does PHYSICS add predictive skill for where a fire grows next?

For every candidate cell around a fire event (see event_validation.py) this module
builds features and trains a gradient-boosted ranker on 2023-24 events. Feature
sets are nested so the contribution of each source of information can be read:

  S0  distance            distance to the nearest burning cell (the naive baseline)
  S1  geometry            + seed density, distance to the fire centroid, event size, the
                            previous day's footprint and the OBSERVED growth direction
                            (information from the data only, no physics)
  S2  weather             + FFMC, BUI, wind speed, day of year
  S3  physics-informed    + arrival times from the rate-of-spread automaton
                            (isotropic, and with the observed wind direction)

If S3 beats S2 on held-out seasons, the automaton adds skill that a data-driven
model with the same inputs cannot find on its own. If not, that is a result too,
and it is reported either way.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, uniform_filter

from src.simulation import event_validation as ev

FEATURE_SETS: Dict[str, List[str]] = {
    "S0 distance only": ["dist"],
    "S1 + fire geometry and observed direction": ["dist", "dens3", "dens8", "dist_centroid", "log_seeds",
                                                  "dist_prev", "has_prev", "cos_align", "has_dir"],
    "S2 + weather": ["dist", "dens3", "dens8", "dist_centroid", "log_seeds", "dist_prev", "has_prev",
                     "cos_align", "has_dir", "ffmc", "bui", "wind", "doy"],
    "S3 + rate-of-spread automaton (physics)": ["dist", "dens3", "dens8", "dist_centroid", "log_seeds",
                                                "dist_prev", "has_prev", "cos_align", "has_dir", "ffmc", "bui",
                                                "wind", "doy", "ca_iso", "ca_dir"],
}
CA_CAP = ev.SIM_HOURS * 60.0 * 1.5


def event_features(e: ev.Event, key: int = 0) -> Optional[pd.DataFrame]:
    """One row per candidate (non-burning) cell of the event window."""
    cand = ~e.seeds
    if not cand.any():
        return None
    h, w = e.seeds.shape
    dist = distance_transform_edt(~e.seeds)
    dens3 = uniform_filter(e.seeds.astype(float), size=7, mode="constant") * 49.0
    dens8 = uniform_filter(e.seeds.astype(float), size=17, mode="constant") * 289.0
    ri, ci = np.nonzero(e.seeds)
    cy, cx = ri.mean(), ci.mean()
    rr, cc = np.indices(e.seeds.shape)
    dist_centroid = np.hypot(rr - cy, cc - cx)
    if e.prev is not None:
        dist_prev = distance_transform_edt(~e.prev)
        wdir = ev.growth_direction(e.prev, e.seeds)
    else:
        dist_prev, wdir = np.full(e.seeds.shape, 99.0), None
    if wdir is not None:
        bearing = (np.degrees(np.arctan2(cc - cx, -(rr - cy))) % 360.0)
        cos_align = np.cos(np.radians(bearing - wdir))
    else:
        cos_align = np.zeros(e.seeds.shape)
    t_iso = ev.ca_arrival_iso(e.seeds, e.weather, key=key)
    ca_iso = np.where(np.isfinite(t_iso), t_iso, CA_CAP)
    if wdir is not None:
        t_dir = ev.ca_arrival(e.seeds, e.weather, (wdir + 180.0) % 360.0, key=key + 9000)
        ca_dir = np.where(np.isfinite(t_dir), t_dir, CA_CAP)
    else:
        ca_dir = ca_iso
    n = int(cand.sum())
    df = pd.DataFrame({
        "dist": dist[cand], "dens3": dens3[cand], "dens8": dens8[cand], "dist_centroid": dist_centroid[cand],
        "log_seeds": np.full(n, math.log1p(e.meta["n_seed"])), "dist_prev": dist_prev[cand],
        "has_prev": np.full(n, float(e.prev is not None)), "cos_align": cos_align[cand],
        "has_dir": np.full(n, float(wdir is not None)),
        "ffmc": np.full(n, e.weather["ffmc"]), "bui": np.full(n, e.weather["bui"]),
        "wind": np.full(n, e.weather["wind"]), "doy": np.full(n, float(e.date.dayofyear)),
        "ca_iso": ca_iso[cand], "ca_dir": ca_dir[cand], "y": e.truth[cand].astype(int),
    })
    return df


def build_table(events: Sequence[ev.Event], key0: int = 0) -> List[pd.DataFrame]:
    out = []
    for n, e in enumerate(events):
        if not (e.truth & ~e.seeds).any():
            continue                                    # nothing to rank: no growth next day
        f = event_features(e, key0 + n)
        if f is not None:
            f["event"] = n
            out.append(f)
    return out


def train_sample(tables: Sequence[pd.DataFrame], neg_per_pos: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for f in tables:
        pos = f[f["y"] == 1]
        neg = f[f["y"] == 0]
        k = min(len(neg), max(50, neg_per_pos * len(pos)))
        parts += [pos, neg.iloc[rng.choice(len(neg), k, replace=False)]]
    return pd.concat(parts, ignore_index=True)


def fit_models(train: pd.DataFrame, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    models = {}
    for name, cols in FEATURE_SETS.items():
        m = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.06, max_iter=250,
                                           l2_regularization=1.0, random_state=seed)
        m.fit(train[cols], train["y"])
        models[name] = m
    return models


def score_events(tables: Sequence[pd.DataFrame], models) -> pd.DataFrame:
    """Per-event AUC for each model and for the two reference rankings."""
    from scipy.stats import rankdata

    def _auc(s, y):
        n1 = int(y.sum()); n0 = len(y) - n1
        if n1 == 0 or n0 == 0:
            return np.nan
        r = rankdata(s)
        return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)

    rows = []
    for f in tables:
        y = f["y"].values
        row = {"event": int(f["event"].iloc[0]), "n_truth": int(y.sum()), "has_dir": float(f["has_dir"].iloc[0]),
               "has_prev": float(f["has_prev"].iloc[0]),
               "Distance to burning cells": _auc(-f["dist"].values, y),
               "Rate-of-spread automaton alone": _auc(-(f["ca_iso"].values + 0.01 * f["dist"].values), y)}
        for name, m in models.items():
            row[name] = _auc(m.predict_proba(f[FEATURE_SETS[name]])[:, 1], y)
        rows.append(row)
    return pd.DataFrame(rows)
