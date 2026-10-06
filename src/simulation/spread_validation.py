"""
spread_validation.py - back-test the fire-spread simulator against what the
satellites actually saw.

For a real day d:
  * seed the cellular automaton with the cells that had a confirmed FIRMS
    detection on day d,
  * drive it with that day's real conditions (FFMC dryness, BUI, wind speed,
    real elevation),
  * compare the cells it predicts will newly catch fire with the cells that
    had a NEW detection on day d+1 (detected on d+1 but not on d).

This is a first-order check, not proof of physical accuracy: the archive has no
wind direction (each run draws a random one), no real fuel map (uniform fuel),
and the satellite sees fires only at its overpasses.

It is scored against two reference predictors so the numbers can be read:
  * "Spread to every neighbour" - the naive rule: any cell next to a burning
    cell catches fire (no physics at all).
  * "Random cells, same count" - pick the same NUMBER of cells at random
    (its expected score is computed exactly, not simulated).
"""
import logging
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.simulation.cellular_automata import CellState, FireSpreadSimulator

logger = logging.getLogger(__name__)

STEP_MIN = 15
HORIZONS_H = (2, 24)


def _grid_index(grid: pd.DataFrame):
    """zone_id -> flat position, plus (rows, cols) arrays in grid order."""
    return (grid["row"].values.astype(int), grid["col"].values.astype(int),
            {z: i for i, z in enumerate(grid["zone_id"].values)})


def eligible_days(df: pd.DataFrame, min_seeds: int = 1, max_days: int = 60) -> List:
    """Days that have seeds AND a following day in the data, thinned evenly."""
    daily = df.groupby("date")["fire_risk_label"].sum()
    dates = sorted(daily.index)
    nxt = {d: dates[i + 1] for i, d in enumerate(dates[:-1])
           if (pd.Timestamp(dates[i + 1]) - pd.Timestamp(d)).days == 1}
    days = [d for d in dates if d in nxt and daily[d] >= min_seeds]
    if len(days) > max_days:
        days = [days[i] for i in np.unique(np.linspace(0, len(days) - 1, max_days).astype(int))]
    return days


def _neighbours(mask: np.ndarray) -> np.ndarray:
    """Cells adjacent (8-neighbourhood) to any True cell, excluding the True cells."""
    out = np.zeros_like(mask)
    r, c = mask.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            sh = np.zeros_like(mask)
            sh[max(dr, 0):r + min(dr, 0), max(dc, 0):c + min(dc, 0)] = \
                mask[max(-dr, 0):r + min(-dr, 0), max(-dc, 0):c + min(-dc, 0)]
            out |= sh
    return out & ~mask


def simulate_day(day_df: pd.DataFrame, grid: pd.DataFrame, elevation: Optional[np.ndarray],
                 seeds: np.ndarray, n_rows: int, n_cols: int, cell_size_deg: float,
                 base_spread_prob: float, replicates: int, horizons_h: Sequence[int],
                 rng: np.random.Generator) -> Dict[int, np.ndarray]:
    """Frequency (0-1) with which each cell is reached, per horizon, over
    `replicates` runs with random wind direction. Arrays are (n_rows, n_cols)."""
    d = day_df.set_index("zone_id").reindex(grid["zone_id"].values)
    rows, cols = grid["row"].values.astype(int), grid["col"].values.astype(int)
    dryness = np.zeros((n_rows, n_cols)); fuel = np.ones((n_rows, n_cols))
    buildup = np.ones((n_rows, n_cols))
    dryness[rows, cols] = (d["ffmc"].fillna(d["ffmc"].median()) / 101.0).clip(0, 1).values
    buildup[rows, cols] = np.clip(d["bui"].fillna(d["bui"].median()).values / 30.0, 0.4, 2.0)
    speed = float(d["wx_wind_speed_ms"].mean())
    horizon_max = max(horizons_h) * 60
    freq = {h: np.zeros((n_rows, n_cols)) for h in horizons_h}
    for _ in range(replicates):
        sim = FireSpreadSimulator(n_rows, n_cols, minutes_per_step=STEP_MIN,
                                  base_spread_prob=base_spread_prob,
                                  random_state=int(rng.integers(1 << 31)), cell_size_deg=cell_size_deg)
        hist = sim.run(ignition_mask=seeds, dryness_grid=dryness, fuel_load_grid=fuel,
                       non_fuel_mask=None, wind_speed_ms=speed, wind_from_deg=float(rng.uniform(0, 360)),
                       horizon_minutes=horizon_max, elevation_grid=elevation, fuel_buildup_grid=buildup)
        for h in horizons_h:
            state = hist[min(h * 60 // STEP_MIN, len(hist) - 1)].state
            freq[h] += np.isin(state, (CellState.BURNING, CellState.BURNED))
    return {h: f / replicates for h, f in freq.items()}


def backtest(df: pd.DataFrame, grid: pd.DataFrame, elevation: Optional[np.ndarray],
             cell_size_deg: float, days: Iterable, base_spread_prob: float = 0.35,
             replicates: int = 10, horizons_h: Sequence[int] = HORIZONS_H,
             vote: float = 0.5, seed: int = 0) -> pd.DataFrame:
    """Score the simulator over `days`. Returns one row per (horizon, method)."""
    rng = np.random.default_rng(seed)
    n_rows, n_cols = int(grid["row"].max()) + 1, int(grid["col"].max()) + 1
    rows, cols = grid["row"].values.astype(int), grid["col"].values.astype(int)
    zone_pos = {z: i for i, z in enumerate(grid["zone_id"].values)}
    by_date = {d: g for d, g in df.groupby("date")}
    dates = sorted(by_date)
    acc = {(h, m): dict(tp=0, pred=0, days=0)
           for h in horizons_h for m in ("CA", "neighbours", "random")}
    truth_total = seeds_total = truth_on_seeds = cand_total = 0
    n_days = 0
    for day in days:
        i = dates.index(day)
        if i + 1 >= len(dates):
            continue
        g0, g1 = by_date[day], by_date[dates[i + 1]]
        seeds = np.zeros((n_rows, n_cols), bool)
        nxt_fire = np.zeros((n_rows, n_cols), bool)
        for g, arr in ((g0, seeds), (g1, nxt_fire)):
            z = g.loc[g["fire_risk_label"] == 1, "zone_id"].map(zone_pos).dropna().astype(int).values
            arr[rows[z], cols[z]] = True
        if not seeds.any():
            continue
        new_truth = nxt_fire & ~seeds
        candidates = int((~seeds).sum())
        n_days += 1
        truth_total += int(new_truth.sum()); seeds_total += int(seeds.sum())
        truth_on_seeds += int((nxt_fire & seeds).sum()); cand_total += candidates
        freq = simulate_day(g0, grid, elevation, seeds, n_rows, n_cols, cell_size_deg,
                            base_spread_prob, replicates, horizons_h, rng)
        nb = _neighbours(seeds)
        for h in horizons_h:
            preds = {"CA": (freq[h] >= vote) & ~seeds, "neighbours": nb}
            for m, p in preds.items():
                a = acc[(h, m)]
                a["tp"] += int((p & new_truth).sum()); a["pred"] += int(p.sum()); a["days"] += 1
            # random baseline with the CA's count: expected true positives, exact
            k = int(preds["CA"].sum())
            a = acc[(h, "random")]
            a["tp"] += k * new_truth.sum() / max(candidates, 1); a["pred"] += k; a["days"] += 1
    out = []
    labels = {"CA": "Cellular automaton (this system)", "neighbours": "Spread to every neighbour (no physics)",
              "random": "Random cells, same count as the automaton"}
    for (h, m), a in acc.items():
        prec = a["tp"] / a["pred"] if a["pred"] else float("nan")
        rec = a["tp"] / truth_total if truth_total else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and (prec + rec) else float("nan")
        out.append({"horizon_hours": h, "method": labels[m], "days_scored": n_days,
                    "cells_predicted": round(a["pred"], 1), "new_fire_cells_next_day": truth_total,
                    "cells_correct": round(a["tp"], 1), "precision": round(prec, 4),
                    "recall": round(rec, 4), "f1": round(f1, 4) if f1 == f1 else f1,
                    "base_spread_prob": base_spread_prob})
    res = pd.DataFrame(out)
    res.attrs["context"] = {"days": n_days, "seed_cells": seeds_total, "new_cells": truth_total,
                            "share_of_next_day_fires_on_seed_cells":
                                truth_on_seeds / max(truth_on_seeds + truth_total, 1)}
    return res


def calibrate(df: pd.DataFrame, grid: pd.DataFrame, elevation, cell_size_deg: float, days,
              probs: Sequence[float] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.35),
              horizon_h: int = 24, replicates: int = 6, seed: int = 0) -> pd.DataFrame:
    """Score several base spread probabilities on TRAINING-season days so one can
    be chosen without looking at the test season."""
    rows = []
    for p in probs:
        r = backtest(df, grid, elevation, cell_size_deg, days, p, replicates, (horizon_h,), seed=seed)
        ca = r[r["method"].str.startswith("Cellular")].iloc[0]
        rows.append({"base_spread_prob": p, "precision": ca["precision"], "recall": ca["recall"], "f1": ca["f1"]})
    return pd.DataFrame(rows)
