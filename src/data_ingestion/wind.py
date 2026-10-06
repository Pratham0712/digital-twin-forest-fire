"""
wind.py - wind averaging and forecast-schedule helpers for the spread simulation.

Wind direction is an angle, so it must NEVER be averaged arithmetically:
the mean of 350 deg and 10 deg is 0 deg (north), not 180 deg (south). All
averaging here is done on u/v vector components.

Convention: `from_deg` is the meteorological direction the wind blows FROM
(as OpenWeatherMap reports it). u is the eastward and v the northward
component of the air's motion.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def to_uv(speed_ms, from_deg) -> Tuple[np.ndarray, np.ndarray]:
    speed = np.asarray(speed_ms, float)
    rad = np.radians(np.asarray(from_deg, float))
    return -speed * np.sin(rad), -speed * np.cos(rad)


def from_uv(u, v) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (speed_ms, from_deg in [0, 360))."""
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    speed = np.hypot(u, v)
    from_deg = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0
    return speed, from_deg


def mean_wind(speeds: Sequence[float], from_degs: Sequence[float],
              weights: Optional[Sequence[float]] = None) -> Tuple[float, float]:
    """Representative wind for a set of readings: direction from the vector
    mean, speed from the plain mean of the speeds (a vector-mean speed would
    shrink towards zero whenever the readings disagree on direction, which
    understates how hard the wind is actually blowing)."""
    s = np.asarray(speeds, float)
    d = np.asarray(from_degs, float)
    ok = np.isfinite(s) & np.isfinite(d)
    if not ok.any():
        return 0.0, 0.0
    w = np.ones(ok.sum()) if weights is None else np.asarray(weights, float)[ok]
    u, v = to_uv(s[ok], d[ok])
    _, deg = from_uv(np.average(u, weights=w), np.average(v, weights=w))
    return float(np.average(s[ok], weights=w)), float(deg)


def interpolate_schedule(times_s: Sequence[float], speeds: Sequence[float], from_degs: Sequence[float],
                         start_s: float, horizon_min: int, step_min: int) -> List[Tuple[float, float]]:
    """One (speed_ms, from_deg) per simulation step, linearly interpolated in
    u/v between forecast times (so a 350 -> 10 deg swing passes through north,
    not south). Step k covers [start + (k-1)*step, start + k*step]; its wind is
    taken at the middle of that interval. Times outside the forecast range use
    the nearest forecast value."""
    t = np.asarray(times_s, float)
    order = np.argsort(t)
    t = t[order]
    u, v = to_uv(np.asarray(speeds, float)[order], np.asarray(from_degs, float)[order])
    n_steps = max(1, horizon_min // step_min)
    mids = start_s + (np.arange(n_steps) + 0.5) * step_min * 60.0
    ui = np.interp(mids, t, u)
    vi = np.interp(mids, t, v)
    # speed: interpolate scalar speeds (vector interpolation would dip when direction swings)
    si = np.interp(mids, t, np.asarray(speeds, float)[order])
    _, di = from_uv(ui, vi)
    return [(float(a), float(b)) for a, b in zip(si, di)]
