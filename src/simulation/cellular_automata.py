"""
FireSpreadSimulator - Layer 4 of the five-layer architecture (Report 6.1.1).

Implements a Cellular Automata (CA) wildfire spread model over the same grid
DataIngestionModule built. Each cell is one of: UNBURNED, BURNING, BURNED,
NON_FUEL. Spread probability between neighbours is a function of:
  - wind speed + direction alignment (fire spreads faster downwind)
  - slope (not modelled here - flat-terrain assumption, documented limitation)
  - fuel moisture (via NDVI + FFMC as a dryness proxy)
  - fuel load / vegetation continuity (via NDVI)

This is the standard CA wildfire model structure used in the literature
reviewed in Report Ch.3 (e.g. papers using cellular automata for fire spread
prediction), adapted to run on the same grid as the ML layer so ML risk
scores can seed ignition points (Scenario 1 in the report: model flags a
zone -> CA projects 2-hour spread from that zone).
"""
import logging
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
                 base_spread_prob: float = 0.35, random_state: int = 42):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.minutes_per_step = minutes_per_step
        self.base_spread_prob = base_spread_prob
        self.rng = np.random.default_rng(random_state)

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

    def _cell_spread_prob(self, dryness: float, fuel_load: float,
                           wind_speed_ms: float, wind_factor: float) -> float:
        """
        dryness: 0-1, higher = drier fuel (derived from FFMC).
        fuel_load: 0-1, higher = more continuous burnable vegetation (from NDVI).
        wind_speed_ms: boosts spread rate independent of direction.
        """
        wind_speed_boost = 1.0 + min(wind_speed_ms / 10.0, 1.0)
        p = self.base_spread_prob * dryness * fuel_load * wind_factor * wind_speed_boost
        return float(np.clip(p, 0.0, 0.97))

    def run(self, ignition_mask: np.ndarray, dryness_grid: np.ndarray,
            fuel_load_grid: np.ndarray, non_fuel_mask: Optional[np.ndarray],
            wind_speed_ms: float, wind_from_deg: float,
            horizon_minutes: int = 120) -> List[SimulationStep]:
        """
        ignition_mask: bool (rows, cols) - True where fire starts (from ML
            layer's high-risk zones, e.g. the 15-cell active-fire seed).
        dryness_grid, fuel_load_grid: float (rows, cols), each 0-1.
        non_fuel_mask: bool (rows, cols), True = cell can never burn.
        """
        state = np.where(ignition_mask, CellState.BURNING, CellState.UNBURNED).astype(int)
        if non_fuel_mask is not None:
            state = np.where(non_fuel_mask, CellState.NON_FUEL, state)

        n_steps = max(1, horizon_minutes // self.minutes_per_step)
        history = [SimulationStep(
            step=0, minutes_elapsed=0, state=state.copy(),
            n_burning=int((state == CellState.BURNING).sum()),
            n_burned=int((state == CellState.BURNED).sum()),
        )]

        for step in range(1, n_steps + 1):
            state = self._advance(state, dryness_grid, fuel_load_grid, wind_speed_ms, wind_from_deg)
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

    def _advance(self, state: np.ndarray, dryness_grid: np.ndarray, fuel_load_grid: np.ndarray,
                 wind_speed_ms: float, wind_from_deg: float) -> np.ndarray:
        new_state = state.copy()
        burning_rows, burning_cols = np.where(state == CellState.BURNING)

        # Currently burning cells transition to BURNED (fuel consumed) this step
        new_state[burning_rows, burning_cols] = CellState.BURNED

        for r, c in zip(burning_rows, burning_cols):
            for dr, dc, bearing in self._neighbor_offsets:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.n_rows and 0 <= nc < self.n_cols):
                    continue
                if state[nr, nc] != CellState.UNBURNED:
                    continue

                wind_factor = self._wind_alignment_factor(bearing, wind_from_deg)
                prob = self._cell_spread_prob(
                    dryness_grid[nr, nc], fuel_load_grid[nr, nc],
                    wind_speed_ms, wind_factor,
                )
                if self.rng.random() < prob:
                    new_state[nr, nc] = CellState.BURNING

        return new_state

    @staticmethod
    def grids_from_processed(processed_df, n_rows: int, n_cols: int):
        """
        Converts the flat 'row'/'col'-indexed processed DataFrame (from
        DataIngestionModule.build_region_grid) into 2D arrays the simulator
        needs: dryness (from FFMC, normalised), fuel_load (from NDVI, clipped
        to positive vegetation range), and a non-fuel mask (very low NDVI =
        bare ground/water, cannot burn).
        """
        dryness_grid = np.zeros((n_rows, n_cols))
        fuel_grid = np.zeros((n_rows, n_cols))
        non_fuel = np.zeros((n_rows, n_cols), dtype=bool)
        ignition = np.zeros((n_rows, n_cols), dtype=bool)

        ffmc_norm = (processed_df["ffmc"] / 101.0).clip(0, 1).values
        ndvi = processed_df["ndvi"].values
        fuel = np.clip(ndvi, 0, 1)
        active_fire = processed_df["active_fire_nearby"].values

        rows = processed_df["row"].values
        cols = processed_df["col"].values

        dryness_grid[rows, cols] = ffmc_norm
        fuel_grid[rows, cols] = fuel
        non_fuel[rows, cols] = ndvi < 0.15
        ignition[rows, cols] = active_fire

        return ignition, dryness_grid, fuel_grid, non_fuel


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
    ignition, dryness, fuel, non_fuel = FireSpreadSimulator.grids_from_processed(processed, n_rows, n_cols)

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
