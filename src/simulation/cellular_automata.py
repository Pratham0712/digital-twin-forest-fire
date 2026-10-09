"""
FireSpreadSimulator - Layer 4 of the five-layer architecture (Report 6.1.1).

Implements a Cellular Automata (CA) wildfire spread model over the same grid
DataIngestionModule built. Each cell is one of: UNBURNED, BURNING, BURNED,
NON_FUEL. Spread probability between neighbours is a function of:
  - wind speed + direction alignment (fire spreads faster downwind)
  - slope (terrain-aware: fire spreads faster uphill, slower downhill, using
    real or synthetic elevation data - see src/data_ingestion/elevation_client.py)
  - fuel moisture: fast-drying surface fuel via FFMC (dryness), AND deep/
    medium-layer fuel availability via BUI (Buildup Index, combining DMC+DC) -
    more available fuel means faster, more sustained spread, not just higher
    ignition risk
  - fuel load / vegetation continuity (via NDVI)

This is the standard CA wildfire model structure used in the literature
reviewed in Report Ch.3 (e.g. papers using cellular automata for fire spread
prediction), adapted to run on the same grid as the ML layer so ML risk
scores can seed ignition points (Scenario 1 in the report: model flags a
zone -> CA projects 2-hour spread from that zone).
"""
import logging
import math
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CellState(IntEnum):
    UNBURNED = 0
    BURNING = 1
    BURNED = 2
    NON_FUEL = 3  # water bodies, bare rock, urban - never ignites


@dataclass
class SimulationStep:
    step: int
    minutes_elapsed: int
    state: np.ndarray  # (rows, cols) grid of CellState values
    n_burning: int
    n_burned: int


class FireSpreadSimulator:
    """
    8-neighbour Moore-neighbourhood CA. One step = `minutes_per_step` minutes
    of real fire spread, calibrated so the default config
    (SYSTEM.fire_spread_horizon_hours=2) maps to a fixed number of steps.
    """

    def __init__(self, n_rows: int, n_cols: int, minutes_per_step: int = 15,
                 base_spread_prob: float = 0.35, random_state: int = 42,
                 cell_size_deg: float = 0.1, diagonal_correction=False, wind_model: str = "linear"):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.minutes_per_step = minutes_per_step
        self.base_spread_prob = base_spread_prob
        self.rng = np.random.default_rng(random_state)
        # Optional (used by the local simulation): diagonal neighbours are sqrt(2) further
        # away, so their per-step probability is divided by sqrt(2) - equal expected advance in
        # all 8 directions. Off by default (regional CA and the original behaviour unchanged).
        self.diagonal_correction = bool(diagonal_correction)
        self.diagonal_factor = (0.7071067811865476 if diagonal_correction is True
                                else float(diagonal_correction) if diagonal_correction else 1.0)
        # "linear" (original, default): factor clip(1 + 0.8 cos(theta), 0.4, 1.8).
        # "elliptical" (local simulation): the same head-fire factor 1.8, but flank and backing
        # spread follow the elliptical fire shape of fire science (Anderson 1983): rate in
        # direction theta from the head = head rate * (1 - e) / (1 - e cos theta), with the
        # eccentricity e from the wind-dependent length-to-breadth ratio. Calm wind: circle.
        self.wind_model = wind_model
        # Real-world distance between adjacent cell centres, used for slope
        # calculation. ~111.32 km per degree of latitude; longitude distance
        # varies with latitude but this project's region (11.5-15.5N) is close
        # enough to the same order that a single approximation is fine for a
        # slope FACTOR (not a precise distance measurement).
        self.cell_distance_m = cell_size_deg * 111_320

        # 8-neighbour offsets and their compass bearing in degrees, used to
        # compute wind alignment (fire spreads preferentially downwind).
        self._neighbor_offsets = [
            (-1, 0, 0), (-1, 1, 45), (0, 1, 90), (1, 1, 135),
            (1, 0, 180), (1, -1, 225), (0, -1, 270), (-1, -1, 315),
        ]

    def _wind_alignment_factor(self, bearing_deg: float, wind_from_deg: float) -> float:
        """
        Returns a multiplier in [0.4, 1.8]: >1 if spread direction is downwind
        of the wind source, <1 if spreading into the wind.
        wind_from_deg follows meteorological convention (direction wind blows
        FROM), so downwind spread direction = wind_from_deg + 180.
        """
        downwind_deg = (wind_from_deg + 180) % 360
        diff = abs(bearing_deg - downwind_deg)
        diff = min(diff, 360 - diff)  # angular distance, 0-180
        alignment = np.cos(np.radians(diff))  # 1 = perfectly downwind, -1 = upwind
        return float(np.clip(1.0 + 0.8 * alignment, 0.4, 1.8))

    MIDFLAME_FACTOR = 0.4        # mid-flame wind ~ 0.4 x the 10 m wind (open forest)
    MAX_LENGTH_TO_BREADTH = 2.0  # an 8-direction CA cannot represent narrower ellipses: beyond ~2 the
                                 # fire becomes a wedge along the two grid directions next to the wind

    @classmethod
    def length_to_breadth(cls, wind_speed_ms: float) -> float:
        """Fire-ellipse length-to-breadth ratio from wind speed (Anderson 1983,
        mid-flame wind in mph), between 1 (calm: circle) and MAX_LENGTH_TO_BREADTH."""
        u = max(0.0, float(wind_speed_ms)) * 2.23694 * cls.MIDFLAME_FACTOR
        lb = 0.936 * math.exp(0.2566 * u) + 0.461 * math.exp(-0.1548 * u) - 0.397
        return min(max(lb, 1.0), cls.MAX_LENGTH_TO_BREADTH)

    def direction_factor(self, bearing_deg: float, wind_from_deg: float, wind_speed_ms: float) -> float:
        """Directional spread multiplier of the configured wind model."""
        if self.wind_model != "elliptical":
            return self._wind_alignment_factor(bearing_deg, wind_from_deg)
        lb = self.length_to_breadth(wind_speed_ms)
        e = math.sqrt(max(0.0, 1.0 - 1.0 / (lb * lb)))
        downwind = (wind_from_deg + 180) % 360
        d = abs(bearing_deg - downwind) % 360
        theta = math.radians(min(d, 360 - d))
        return 1.8 * (1.0 - e) / (1.0 - e * math.cos(theta))

    def _slope_factor(self, elev_from_m: float, elev_to_m: float, diagonal: bool = False) -> float:
        """
        Returns a multiplier in [0.3, 3.0]: >1 spreading uphill, <1 spreading
        downhill, 1.0 on flat ground. Uses a standard exponential slope-effect
        approximation from wildfire spread literature (rate of spread scales
        roughly exponentially with slope percent, e.g. Rothermel-style models):
            factor = exp(k * slope_percent)
        where slope_percent = rise/run * 100, k=0.06 chosen so a steep ~30%
        grade (Western-Ghats-plausible) gives roughly a 2x uphill boost /
        0.5x downhill reduction - directionally correct and bounded, not a
        precise physical simulation.
        """
        rise = elev_to_m - elev_from_m
        if rise == 0:
            return 1.0                                  # flat: exp(0) = 1 exactly (fast path, same value)
        distance_m = self.cell_distance_m * (1.41421356 if diagonal else 1.0)
        slope_pct = (rise / distance_m) * 100
        factor = float(np.exp(0.06 * slope_pct))
        return min(max(factor, 0.3), 3.0)               # == np.clip for a finite scalar, without its overhead

    def _cell_spread_prob(self, dryness: float, fuel_load: float, fuel_buildup: float,
                           wind_speed_ms: float, wind_factor: float, slope_factor: float) -> float:
        """
        dryness: 0-1, higher = drier fine fuel (derived from FFMC).
        fuel_load: 0-1, higher = more continuous burnable vegetation (from NDVI).
        fuel_buildup: multiplier from BUI (Buildup Index) - more accumulated
            deep/medium-layer fuel available means faster, more sustained
            spread once ignited, not just higher standalone ignition risk.
        wind_speed_ms: boosts spread rate independent of direction.
        slope_factor: >1 uphill, <1 downhill, from _slope_factor.
        """
        wind_speed_boost = 1.0 + min(wind_speed_ms / 10.0, 1.0)
        p = (self.base_spread_prob * dryness * fuel_load * fuel_buildup
             * wind_factor * wind_speed_boost * slope_factor)
        return float(min(max(p, 0.0), 0.97))            # == np.clip(p, 0, 0.97) for a scalar

    def run(self, ignition_mask: np.ndarray, dryness_grid: np.ndarray,
            fuel_load_grid: np.ndarray, non_fuel_mask: Optional[np.ndarray],
            wind_speed_ms: float, wind_from_deg: float,
            horizon_minutes: int = 120,
            elevation_grid: Optional[np.ndarray] = None,
            fuel_buildup_grid: Optional[np.ndarray] = None,
            wind_schedule: Optional[List[Tuple[float, float]]] = None,
            before_step=None) -> List[SimulationStep]:
        """
        ignition_mask: bool (rows, cols) - True where fire starts (from ML
            layer's high-risk zones, e.g. the 15-cell active-fire seed).
        dryness_grid, fuel_load_grid: float (rows, cols), each 0-1.
        non_fuel_mask: bool (rows, cols), True = cell can never burn.
        elevation_grid: float (rows, cols) in meters, optional - enables
            slope-aware spread (faster uphill, slower downhill). If None,
            terrain is treated as flat (factor 1.0 everywhere) - same
            behaviour as before this feature was added.
        fuel_buildup_grid: float (rows, cols), optional - normalised BUI
            (Buildup Index) multiplier for deep-fuel availability. If None,
            defaults to 1.0 everywhere (no effect) - same behaviour as before.
        wind_schedule: optional list of (speed_ms, wind_from_deg), one per
            simulation step (step k uses entry k-1; the last entry repeats if
            the run is longer). When given it replaces the constant
            wind_speed_ms / wind_from_deg - used to drive the spread with a
            forecast that changes over the 2-hour horizon.
        before_step: optional callback(step, state) called before every step.
            It may return None (nothing changes) or a dict with a grown grid:
            {"state", "dryness", "fuel", "buildup", "elevation", "non_fuel"}
            (the adaptive local domain, src/simulation/local_spread.py). The
            step counter, the wind schedule and the random stream continue
            unchanged, so the run is never restarted.
        """
        state = self.start(ignition_mask, non_fuel_mask)

        if elevation_grid is None:
            elevation_grid = np.zeros((self.n_rows, self.n_cols))
        if fuel_buildup_grid is None:
            fuel_buildup_grid = np.ones((self.n_rows, self.n_cols))

        n_steps = max(1, horizon_minutes // self.minutes_per_step)
        history = [SimulationStep(
            step=0, minutes_elapsed=0, state=state.copy(),
            n_burning=int((state == CellState.BURNING).sum()),
            n_burned=int((state == CellState.BURNED).sum()),
        )]

        for step in range(1, n_steps + 1):
            if before_step is not None:
                grown = before_step(step, state)
                if grown:
                    state = grown["state"]
                    dryness_grid, fuel_load_grid = grown["dryness"], grown["fuel"]
                    fuel_buildup_grid, elevation_grid = grown["buildup"], grown["elevation"]
                    self.resize(state.shape[0], state.shape[1], grown["non_fuel"])
            if wind_schedule:
                step_speed, step_from = wind_schedule[min(step - 1, len(wind_schedule) - 1)]
            else:
                step_speed, step_from = wind_speed_ms, wind_from_deg
            state = self._advance(state, dryness_grid, fuel_load_grid, step_speed, step_from,
                                   elevation_grid, fuel_buildup_grid)
            history.append(SimulationStep(
                step=step, minutes_elapsed=step * self.minutes_per_step, state=state.copy(),
                n_burning=int((state == CellState.BURNING).sum()),
                n_burned=int((state == CellState.BURNED).sum()),
            ))
            if (state == CellState.BURNING).sum() == 0:
                logger.info("Fire extinguished naturally at step %d (%d min)", step, step * self.minutes_per_step)
                break

        logger.info(
            "CA simulation complete: %d steps, final burned=%d, burning=%d",
            len(history) - 1, history[-1].n_burned, history[-1].n_burning,
        )
        return history

    # ── step-wise API (used by run() and by the adaptive local domain) ──── #

    def start(self, ignition_mask: np.ndarray, non_fuel_mask: Optional[np.ndarray]) -> np.ndarray:
        """Initial state (int8). HARD RULE: a non-fuel cell is NON_FUEL for the
        whole run - it overrides any ignition and is never a spread candidate
        (see _advance)."""
        state = np.where(ignition_mask, CellState.BURNING, CellState.UNBURNED).astype(np.int8)
        self._non_fuel = (np.asarray(non_fuel_mask, dtype=bool) if non_fuel_mask is not None
                          else np.zeros((self.n_rows, self.n_cols), dtype=bool))
        return np.where(self._non_fuel, CellState.NON_FUEL, state).astype(np.int8)

    def resize(self, n_rows: int, n_cols: int, non_fuel_mask: np.ndarray):
        """The grid grew (adaptive domain): new size and the padded non-fuel
        mask. The RNG stream continues unchanged."""
        self.n_rows, self.n_cols = int(n_rows), int(n_cols)
        self._non_fuel = np.asarray(non_fuel_mask, dtype=bool)

    def advance(self, state: np.ndarray, dryness_grid: np.ndarray, fuel_load_grid: np.ndarray,
                wind_speed_ms: float, wind_from_deg: float,
                elevation_grid: np.ndarray, fuel_buildup_grid: np.ndarray) -> np.ndarray:
        """One CA step (public name of _advance)."""
        return self._advance(state, dryness_grid, fuel_load_grid, wind_speed_ms, wind_from_deg,
                             elevation_grid, fuel_buildup_grid)

    def _advance(self, state: np.ndarray, dryness_grid: np.ndarray, fuel_load_grid: np.ndarray,
                 wind_speed_ms: float, wind_from_deg: float,
                 elevation_grid: np.ndarray, fuel_buildup_grid: np.ndarray) -> np.ndarray:
        new_state = state.copy()
        burning_rows, burning_cols = np.where(state == CellState.BURNING)
        non_fuel = getattr(self, "_non_fuel", None)
        # The wind factor of each of the 8 directions is the same for every cell
        # in a step: computed once here (identical values, so identical results).
        wind_of = {bearing: self.direction_factor(bearing, wind_from_deg, wind_speed_ms)
                   for _, _, bearing in self._neighbor_offsets}

        # Currently burning cells transition to BURNED (fuel consumed) this step
        new_state[burning_rows, burning_cols] = CellState.BURNED

        for r, c in zip(burning_rows, burning_cols):
            for dr, dc, bearing in self._neighbor_offsets:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.n_rows and 0 <= nc < self.n_cols):
                    continue
                if non_fuel is not None and non_fuel[nr, nc]:
                    continue                       # non-fuel neighbour: blocked, no spread probability computed
                if state[nr, nc] != CellState.UNBURNED:
                    continue

                wind_factor = wind_of[bearing]
                slope_factor = self._slope_factor(
                    elevation_grid[r, c], elevation_grid[nr, nc], diagonal=(dr != 0 and dc != 0),
                )
                prob = self._cell_spread_prob(
                    dryness_grid[nr, nc], fuel_load_grid[nr, nc], fuel_buildup_grid[nr, nc],
                    wind_speed_ms, wind_factor, slope_factor,
                )
                if self.diagonal_correction and dr != 0 and dc != 0:
                    prob *= self.diagonal_factor
                if self.rng.random() < prob:
                    new_state[nr, nc] = CellState.BURNING

        return new_state

    @staticmethod
    def grids_from_processed(processed_df, n_rows: int, n_cols: int):
        """
        Converts the flat 'row'/'col'-indexed processed DataFrame (from
        DataIngestionModule.build_region_grid) into 2D arrays the simulator
        needs: dryness (from FFMC, normalised), fuel_load (from NDVI, clipped
        to positive vegetation range), fuel_buildup (from BUI - deep/medium
        layer fuel availability), and a non-fuel mask (very low NDVI = bare
        ground/water, cannot burn).
        """
        dryness_grid = np.zeros((n_rows, n_cols))
        fuel_grid = np.zeros((n_rows, n_cols))
        buildup_grid = np.ones((n_rows, n_cols))
        non_fuel = np.zeros((n_rows, n_cols), dtype=bool)
        ignition = np.zeros((n_rows, n_cols), dtype=bool)

        ffmc_norm = (processed_df["ffmc"] / 101.0).clip(0, 1).values
        ndvi = processed_df["ndvi"].values
        fuel = np.clip(ndvi, 0, 1)
        active_fire = processed_df["active_fire_nearby"].values

        # BUI (Buildup Index) -> fuel-availability multiplier. Typical BUI
        # ranges roughly 0-60 in this project's data (extreme drought can go
        # higher); normalised so BUI=30 gives factor 1.0 (neutral, matches the
        # old no-BUI behaviour), lower BUI reduces spread, higher increases it,
        # bounded to keep the CA simulation stable.
        if "bui" in processed_df.columns:
            buildup = np.clip(processed_df["bui"].values / 30.0, 0.4, 2.0)
        else:
            buildup = np.ones(len(processed_df))

        rows = processed_df["row"].values
        cols = processed_df["col"].values

        dryness_grid[rows, cols] = ffmc_norm
        fuel_grid[rows, cols] = fuel
        buildup_grid[rows, cols] = buildup
        non_fuel[rows, cols] = ndvi < 0.15
        ignition[rows, cols] = active_fire

        return ignition, dryness_grid, fuel_grid, non_fuel, buildup_grid


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))

    from config.config import API, REGION, SYSTEM
    from src.data_ingestion.ingestion_module import DataIngestionModule
    from src.data_processing.feature_engineering import DataProcessor

    logging.basicConfig(level=logging.INFO)

    use_offline = not bool(API.firms_map_key)
    if use_offline:
        logging.info("No FIRMS_MAP_KEY found - running in offline/synthetic mode.")
    else:
        logging.info("FIRMS_MAP_KEY found - fetching real satellite data.")

    ingestion = DataIngestionModule(offline=use_offline)
    unified = ingestion.build_unified_frame()
    processed = DataProcessor().transform(unified)

    n_rows = processed["row"].max() + 1
    n_cols = processed["col"].max() + 1
    ignition, dryness, fuel, non_fuel, buildup = FireSpreadSimulator.grids_from_processed(processed, n_rows, n_cols)

    print(f"Grid: {n_rows}x{n_cols}, ignition points: {ignition.sum()}, non-fuel cells: {non_fuel.sum()}")

    avg_wind_speed = processed["wx_wind_speed_ms"].mean()
    avg_wind_deg = processed["wx_wind_deg"].mean()

    sim = FireSpreadSimulator(n_rows, n_cols)
    history = sim.run(
        ignition_mask=ignition, dryness_grid=dryness, fuel_load_grid=fuel,
        non_fuel_mask=non_fuel, wind_speed_ms=avg_wind_speed, wind_from_deg=avg_wind_deg,
        horizon_minutes=SYSTEM.fire_spread_horizon_hours * 60,
    )

    print(f"\n{'Step':>5} {'Minutes':>8} {'Burning':>8} {'Burned':>8}")
    for h in history:
        print(f"{h.step:>5} {h.minutes_elapsed:>8} {h.n_burning:>8} {h.n_burned:>8}")

    final = history[-1]
    total_cells = n_rows * n_cols - non_fuel.sum()
    print(f"\nFinal burned area: {final.n_burned} cells "
          f"({100 * final.n_burned / total_cells:.2f}% of burnable area)")
