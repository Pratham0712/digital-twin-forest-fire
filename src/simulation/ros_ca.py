"""
ros_ca.py - rate-of-spread cellular automaton for the LOCAL simulation (Phase 3).

States are the project's CellState (UNBURNED / BURNING / BURNED / NON_FUEL).
What changed against the original one-trial-per-step CA, and why:

* Rate, not probability. A burning cell drives each unburned neighbour with a
  rate  r = ROS(direction) x fuel continuity x slope factor / distance  (1/min),
  where ROS(direction) comes from fire_behaviour.spread_rates (FFMC, BUI,
  wind: head / flank / back fire on the wind ellipse). The neighbour's
  ignition progress grows by r x dt (the strongest burning neighbour counts);
  it ignites when the progress reaches its threshold (1 +/- a small seeded
  jitter: natural irregularity of the front). So the fire advances at the
  physical ROS in every direction, faster with wind and drier fuel, slower
  backing into the wind - instead of at most one cell per step.
* 16 neighbours (the 8 adjacent cells plus the 8 "knight" cells at
  atan(1/2) = 26.6 deg) with their true distances: diagonals are corrected
  geometrically (sqrt 2 x cell, sqrt 5 x cell), and the fire ellipse is drawn
  far less octagonal than with 8 neighbours.
* Residence. A cell flames for `flame_min` minutes (x a seeded 0.75-1.25
  factor): the flaming band is deep at the head (ROS x residence) and thin on
  the flanks. Offers already made stay valid, so slow backing fire continues
  (travel times above `residence_max_min` are not offered); with wet fuel
  (ROS ~ 0) the fire stops by itself.
* Adaptive sub-steps. The internal step dt keeps the progress per step <= 0.5,
  so a fast fire is never limited by the step; the frames handed to the
  dashboard are taken every `frame_min` minutes of simulated time, and the run
  always covers the full requested duration (the time axis continues after an
  extinction, so playback and the final state match the selected duration).

Deterministic for a seed: thresholds come from a hash of the absolute cell
position (identical for any domain size / expansion strip on the same
lattice), no other randomness is used.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from src.simulation.cellular_automata import CellState
from src.simulation.fire_behaviour import SpreadRates

# (dr, dc): row - 1 = north, col + 1 = east
OFFSETS: List[Tuple[int, int]] = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1),
                                  (-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1),
                                  (-3, 1), (-1, 3), (1, 3), (3, 1), (3, -1), (1, -3), (-1, -3), (-3, -1)]
BEARINGS = [math.degrees(math.atan2(dc, -dr)) % 360.0 for dr, dc in OFFSETS]
DIST_CELLS = [math.hypot(dr, dc) for dr, dc in OFFSETS]


def _between(dr: int, dc: int) -> List[Tuple[int, int]]:
    """Cells the straight segment from the source (0, 0) to (dr, dc) passes
    through (excluding both ends). A long offset never jumps over non-fuel."""
    cells = []
    for i in range(1, 64):
        t = i / 64.0
        q = (int(round(dr * t)), int(round(dc * t)))
        if q not in cells and q != (0, 0) and q != (dr, dc):
            cells.append(q)
    return cells


BETWEEN = [_between(dr, dc) for dr, dc in OFFSETS]
MAX_REACH = 3                      # cells: the (3, 1) offsets reach three rows / columns


def _shift(a: np.ndarray, dr: int, dc: int, fill) -> np.ndarray:
    """out[r, c] = a[r - dr, c - dc] (value of the cell `offset` behind)."""
    out = np.full(a.shape, fill, dtype=a.dtype)
    R, C = a.shape
    r0, r1 = max(dr, 0), R + min(dr, 0)
    c0, c1 = max(dc, 0), C + min(dc, 0)
    if r1 > r0 and c1 > c0:
        out[r0:r1, c0:c1] = a[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
    return out


def cell_hash01(rows_abs: np.ndarray, cols_abs: np.ndarray, seed: int) -> np.ndarray:
    v = (rows_abs.astype(np.int64) * 73856093) ^ (cols_abs.astype(np.int64) * 19349663) ^ (int(seed) * 83492791)
    v = (v ^ (v >> 13)) * 1274126177
    return ((v ^ (v >> 16)) & 0xFFFF) / 65535.0


@dataclass
class RosGrids:
    """Static per-cell inputs on the current domain grid."""
    continuity: np.ndarray          # fuel continuity multiplier (0 = no spread)
    elevation: np.ndarray
    non_fuel: np.ndarray            # bool
    threshold: np.ndarray           # ignition threshold (1 +/- jitter)
    cell_m: float
    gain: List[np.ndarray] = field(default_factory=list)   # per offset: continuity(target) x slope / distance
    gain_flat: Optional[np.ndarray] = None
    flame_mult: Optional[np.ndarray] = None                   # per-cell residence factor (seeded)

    def build(self):
        self.gain = []
        for (dr, dc), dcell in zip(OFFSETS, DIST_CELLS):
            dist = dcell * self.cell_m
            src_e = _shift(self.elevation, dr, dc, np.nan)
            rise = self.elevation - src_e
            slope = np.where(np.isfinite(rise), np.clip(np.exp(0.06 * rise / dist * 100.0), 0.3, 3.0), 1.0)
            g = self.continuity * slope / dist
            g[self.non_fuel] = 0.0
            if abs(dr) == 1 and abs(dc) == 1:
                # diagonal: no squeezing between two non-fuel cells touching at a corner
                a = _shift(self.non_fuel, dr, 0, False)            # cell (r - dr, c): source row, target column
                b = _shift(self.non_fuel, 0, dc, False)
                g[a & b] = 0.0
            for (ir, ic) in BETWEEN[len(self.gain)]:
                # target (r, c), source (r - dr, c - dc): intermediate cell = source + (ir, ic)
                g[_shift(self.non_fuel, dr - ir, dc - ic, False)] = 0.0
            self.gain.append(g.astype(np.float64))
        self.gain_flat = np.stack([g.reshape(-1) for g in self.gain])     # (24, cells) for vectorised lookups
        return self


@dataclass
class RosRun:
    frames: List[np.ndarray]
    frame_margins: List[Tuple[int, int]]          # (m_north, m_west) of the domain at each frame
    t_ign: np.ndarray                              # minutes, inf = never
    t_out: np.ndarray                              # minutes, inf = never burned out
    ros_ign: np.ndarray                            # m/min of the spread that ignited the cell (0 = never)
    frame_min: float
    dt_min: float
    n_substeps: int
    extinguished_at: Optional[float]
    rates: List[SpreadRates]


def run_ros_ca(state: np.ndarray, grids: RosGrids, rates_at: Callable[[float], SpreadRates],
               duration_min: float, frame_min: float, flame_min: float = 3.0,
               residence_max_min: float = 45.0, max_substeps: int = 20000,
               expand: Optional[Callable] = None, margins: Callable[[], Tuple[int, int]] = lambda: (0, 0),
               rate_bound: Optional[float] = None) -> RosRun:
    """Run to `duration_min` (never shorter). `expand(state, arrays, t)` may grow
    the grid before a sub-step; it returns None or (state, grids, pad) where
    pad = (north, south, west, east) cells added."""
    n_frames = max(1, int(round(duration_min / frame_min)))
    frame_min = duration_min / n_frames
    # sub-step: progress per step <= 0.5 for the fastest rate the run can see
    # domain-independent bound (head ROS x max fuel continuity / cell): the same dt for any domain or
    # expansion, so an expanded run equals a run on the final domain. Arrival times are exact
    # (T_source + travel), dt only sets how often new sources are processed.
    rmax = rate_bound if rate_bound is not None else 0.0
    peak = rmax * 1.25 / grids.cell_m
    sub = max(1, int(math.ceil(frame_min * peak / 0.5))) if peak > 0 else 1
    sub = min(sub, max(1, max_substeps // n_frames))
    dt = frame_min / sub

    S = state.astype(np.int8).copy()
    T = np.where(S == CellState.BURNING, 0.0, np.inf)
    TO = np.full(S.shape, np.inf)
    PR = np.full(S.shape, np.inf)                   # earliest offered arrival time (min)
    RI = np.where(S == CellState.BURNING, 0.0, 0.0)
    frames = [S.copy()]
    fmarg = [margins()]
    rates_used: List[SpreadRates] = []
    ros_dir: List[float] = []
    win = (0, S.shape[0], 0, S.shape[1])
    pending = {"idx": np.zeros(0, dtype=np.int64)}            # offered, not yet ignited cells (flat indices)
    extinct = None
    t = 0.0
    k_total = 0
    for f in range(1, n_frames + 1):
        for _ in range(sub):
            burning = _burning_in(S, win)
            if burning[0].size == 0:
                if pending["idx"].size == 0:
                    if extinct is None:
                        extinct = t
                    break
                # nothing flames right now, but offered cells are still due to ignite (slow spread)
                t_end = t + dt
                pending["idx"] = _ignite_due(S, T, PR, pending["idx"], t_end)
                win = _window_with(_burning_in(S, (0, S.shape[0], 0, S.shape[1])), pending["idx"], S.shape)
                t = t_end
                k_total += 1
                continue
            if expand is not None:
                grown = expand(S, {"t_ign": T, "t_out": TO, "progress": PR, "ros": RI, "burning": burning}, t)
                if grown is not None:
                    old_cols = S.shape[1]
                    S, grids, (pn, ps, pw, pe) = grown
                    def pad(a, fill):
                        out = np.full(S.shape, fill, dtype=a.dtype)
                        out[pn:pn + a.shape[0], pw:pw + a.shape[1]] = a
                        return out
                    T, TO, PR, RI = pad(T, np.inf), pad(TO, np.inf), pad(PR, np.inf), pad(RI, 0.0)
                    burning = (burning[0] + pn, burning[1] + pw)
                    pr_, pc_ = np.divmod(pending["idx"], old_cols)
                    pending["idx"] = (pr_ + pn) * S.shape[1] + (pc_ + pw)
            rates = rates_at(t)
            if not rates_used or rates_used[-1] != rates:
                rates_used.append(rates)
                ros_dir = [rates.towards(b) for b in BEARINGS]
            t_end = t + dt
            _substep(S, T, TO, PR, RI, grids, ros_dir, t, t_end, dt, flame_min, residence_max_min, burning, pending)
            # next window: burning cells (incl. new ignitions next to them) and every offered cell
            win = _window_with(burning, pending["idx"], S.shape)
            t = t_end
            k_total += 1
        t = f * frame_min                       # exact frame time (also after an extinction)
        frames.append(S.copy())
        fmarg.append(margins())
    return RosRun(frames, fmarg, T, TO, RI, frame_min, dt, k_total, extinct, rates_used)


def _window(burning, shape):
    """Rows / cols where cells can be burning after the next sub-step (the
    burning cells' bounding box plus the stencil reach)."""
    br, bc = burning
    if br.size == 0:
        return (0, 0, 0, 0)
    return (max(0, int(br.min()) - MAX_REACH), min(shape[0], int(br.max()) + MAX_REACH + 1),
            max(0, int(bc.min()) - MAX_REACH), min(shape[1], int(bc.max()) + MAX_REACH + 1))


def _window_with(burning, pend_idx, shape):
    """Window covering the burning cells (+ stencil reach) and all offered cells."""
    r0, r1, c0, c1 = _window(burning, shape)
    if pend_idx.size:
        pr, pc = np.divmod(pend_idx, shape[1])
        if r1 <= r0:
            r0, r1, c0, c1 = int(pr.min()), int(pr.max()) + 1, int(pc.min()), int(pc.max()) + 1
        else:
            r0, r1 = min(r0, int(pr.min())), max(r1, int(pr.max()) + 1)
            c0, c1 = min(c0, int(pc.min())), max(c1, int(pc.max()) + 1)
    return r0, r1, c0, c1


def _burning_in(S, win):
    r0, r1, c0, c1 = win
    br, bc = np.nonzero(S[r0:r1, c0:c1] == CellState.BURNING)
    return br + r0, bc + c0


_DR = np.array([o[0] for o in OFFSETS])
_DC = np.array([o[1] for o in OFFSETS])
_DM = np.array(DIST_CELLS)


def _substep(S, T, TO, PR, RI, grids: RosGrids, ros_dir, t0, t1, dt, flame_min, res_max, burning=None,
             pending=None):
    """One sub-step, vectorised over all burning cells x all 24 offsets at once
    (work proportional to the number of burning cells, not the domain size).
    Each burning source offers its neighbours an arrival time
    T_source + threshold / rate (Huygens: travel time across the cell at the
    directional ROS); a cell ignites at the earliest offer once that time is
    reached. Offers that need longer than the residence cap are not made."""
    R, C = S.shape
    br, bc = burning if burning is not None else np.nonzero(S == CellState.BURNING)
    if br.size == 0:
        return
    ts = T[br, bc]
    rd = np.asarray(ros_dir)                                   # (24,)
    tr = br[None, :] + _DR[:, None]                            # (24, n)
    tc = bc[None, :] + _DC[:, None]
    ok = (tr >= 0) & (tr < R) & (tc >= 0) & (tc < C) & (rd[:, None] > 0)
    kk, jj = np.nonzero(ok)
    if kk.size:
        trr, tcc = tr[kk, jj], tc[kk, jj]
        flat = trr * C + tcc
        un = S.reshape(-1)[flat] == CellState.UNBURNED
        kk, jj, flat = kk[un], jj[un], flat[un]
    if kk.size:
        rate = rd[kk] * grids.gain_flat[kk, flat]
        pos = rate > 0
        kk, jj, flat, rate = kk[pos], jj[pos], flat[pos], rate[pos]
        travel = grids.threshold.reshape(-1)[flat] / rate
        keep = travel <= res_max
        kk, jj, flat, rate, travel = kk[keep], jj[keep], flat[keep], rate[keep], travel[keep]
        if flat.size:
            arr = ts[jj] + travel
            prf, rif = PR.reshape(-1), RI.reshape(-1)
            before = prf[flat].copy()
            np.minimum.at(prf, flat, arr)
            win = (arr <= prf[flat]) & (arr < before)          # the offer that set the new earliest arrival
            rif[flat[win]] = (rate * _DM[kk] * grids.cell_m)[win]
            if pending is not None:
                pending["idx"] = np.union1d(pending["idx"], flat)
            else:
                _ignite_due(S, T, PR, np.unique(flat), t1)
    if pending is not None and pending["idx"].size:
        # every offered cell is checked each step - also after the cell that made the offer burned out
        pending["idx"] = _ignite_due(S, T, PR, pending["idx"], t1)
    # burn-out: a cell flames for its residence time (flame_min x a seeded 0.75-1.25 per-cell factor), then
    # it is BURNED. The flaming band is therefore deep at the fast head fire (ROS x residence) and thin on
    # the slow flanks / backing fire - a connected head front instead of a uniform glowing ring. Spread
    # offers already made by a cell stay valid after it burns out (Huygens arrival times), so slow
    # backing fire still advances.
    age = t1 - ts
    fl = flame_min * (grids.flame_mult[br, bc] if grids.flame_mult is not None else 1.0)
    done = age >= fl
    if done.any():
        S[br[done], bc[done]] = CellState.BURNED
        TO[br[done], bc[done]] = t1


def _ignite_due(S, T, PR, idx, t1):
    """Ignite the offered cells whose earliest arrival time has been reached;
    returns the offered cells still waiting."""
    sf, prf = S.reshape(-1), PR.reshape(-1)
    un = sf[idx] == CellState.UNBURNED
    idx = idx[un]
    due = prf[idx] <= t1
    new = idx[due]
    if new.size:
        T.reshape(-1)[new] = np.maximum(prf[new], 0.0)
        sf[new] = CellState.BURNING
    return idx[~due]
