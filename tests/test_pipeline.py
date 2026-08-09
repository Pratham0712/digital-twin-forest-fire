"""
Integration + unit tests for the Digital Twin pipeline.
Run: pytest tests/ -v
All tests run in offline/synthetic mode - no API keys or internet required,
so this suite works identically on your machine, a teammate's machine, and
in CI (e.g. a GitHub Actions badge for the README).
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.data_ingestion.ingestion_module import DataIngestionModule, build_region_grid
from src.data_processing.feature_engineering import (
    DataProcessor, compute_ffmc, compute_dmc, compute_dc, compute_bui, compute_fwi,
)
from src.simulation.cellular_automata import FireSpreadSimulator, CellState
from src.digital_twin.twin_state import DigitalTwin, AlertEngine


# --------------------------------------------------------------------------- #
# Layer 1: Data Ingestion
# --------------------------------------------------------------------------- #

def test_region_grid_shape():
    grid = build_region_grid()
    assert len(grid) > 0
    assert set(["zone_id", "row", "col", "latitude", "longitude"]).issubset(grid.columns)
    # every zone_id must be unique
    assert grid["zone_id"].is_unique


def test_ingestion_offline_mode_produces_full_grid():
    module = DataIngestionModule(offline=True)
    unified = module.build_unified_frame()
    assert len(unified) == len(module.grid)
    assert unified["wx_temperature_c"].notna().all()
    assert unified["active_fire_nearby"].dtype == bool


def test_ingestion_active_fire_flag_is_minority_not_universal():
    """Regression test for the spatial-join bug fixed earlier: active_fire_nearby
    must NOT be True for every cell (that bug made every cell 'near' a fire)."""
    module = DataIngestionModule(offline=True)
    unified = module.build_unified_frame()
    pct_active = unified["active_fire_nearby"].mean()
    assert 0 < pct_active < 0.5, f"active_fire_nearby rate {pct_active:.2%} looks like the masking bug is back"


# --------------------------------------------------------------------------- #
# Layer 2: Feature Engineering (FWI system)
# --------------------------------------------------------------------------- #

def test_ffmc_bounded_0_101():
    temp = np.array([25.0, 35.0, 15.0])
    rh = np.array([40.0, 10.0, 90.0])
    wind = np.array([2.0, 8.0, 0.5])
    rain = np.array([0.0, 0.0, 5.0])
    ffmc = compute_ffmc(temp, rh, wind, rain)
    assert np.all((ffmc >= 0) & (ffmc <= 101))


def test_dmc_dc_non_negative_scalar_and_array_inputs():
    """Regression test for the scalar/array clip bug fixed earlier."""
    temp = np.array([30.0, 32.0])
    rh = np.array([20.0, 15.0])
    rain = np.array([0.0, 0.0])
    dmc = compute_dmc(temp, rh, rain, month=5, dmc_prev=6.0)  # scalar dmc_prev
    dc = compute_dc(temp, rain, month=5, dc_prev=15.0)  # scalar dc_prev
    assert np.all(dmc >= 0)
    assert np.all(dc >= 0)
    assert not np.any(np.isnan(dmc))
    assert not np.any(np.isnan(dc))


def test_fwi_pipeline_end_to_end():
    module = DataIngestionModule(offline=True)
    unified = module.build_unified_frame()
    processed = DataProcessor().transform(unified)

    for col in ["ffmc", "dmc", "dc", "bui", "fwi", "ndvi"]:
        assert col in processed.columns
        assert processed[col].notna().all(), f"{col} has NaNs"

    assert processed["ffmc"].between(0, 101).all()
    assert processed["ndvi"].between(-1, 1).all()
    assert (processed["dmc"] >= 0).all()
    assert (processed["dc"] >= 0).all()
    assert (processed["fwi"] >= 0).all()


def test_label_positive_rate_is_realistic():
    """Guards against label leakage regressions: positive rate should be a
    meaningful minority, not near 0% or near 100%."""
    module = DataIngestionModule(offline=True)
    unified = module.build_unified_frame()
    processed = DataProcessor().transform(unified)
    rate = processed["fire_risk_label"].mean()
    assert 0.03 < rate < 0.5, f"Positive rate {rate:.2%} outside sane range - check label logic"


# --------------------------------------------------------------------------- #
# Layer 4: Cellular Automata
# --------------------------------------------------------------------------- #

def test_ca_fire_never_shrinks_burned_count():
    sim = FireSpreadSimulator(n_rows=10, n_cols=10, random_state=1)
    ignition = np.zeros((10, 10), dtype=bool)
    ignition[5, 5] = True
    dryness = np.full((10, 10), 0.8)
    fuel = np.full((10, 10), 0.8)
    history = sim.run(ignition, dryness, fuel, non_fuel_mask=None,
                       wind_speed_ms=3.0, wind_from_deg=180, horizon_minutes=60)
    burned_counts = [h.n_burned for h in history]
    assert burned_counts == sorted(burned_counts), "burned cell count must be monotonically non-decreasing"


def test_ca_non_fuel_cells_never_burn():
    sim = FireSpreadSimulator(n_rows=10, n_cols=10, random_state=1)
    ignition = np.zeros((10, 10), dtype=bool)
    ignition[5, 5] = True
    non_fuel = np.zeros((10, 10), dtype=bool)
    non_fuel[5, 6] = True  # immediate neighbour is non-fuel
    dryness = np.full((10, 10), 0.9)
    fuel = np.full((10, 10), 0.9)
    history = sim.run(ignition, dryness, fuel, non_fuel_mask=non_fuel,
                       wind_speed_ms=5.0, wind_from_deg=270, horizon_minutes=120)
    final_state = history[-1].state
    assert final_state[5, 6] == CellState.NON_FUEL


def test_ca_zero_dryness_zero_fuel_never_spreads():
    sim = FireSpreadSimulator(n_rows=8, n_cols=8, random_state=1)
    ignition = np.zeros((8, 8), dtype=bool)
    ignition[4, 4] = True
    dryness = np.zeros((8, 8))
    fuel = np.zeros((8, 8))
    history = sim.run(ignition, dryness, fuel, non_fuel_mask=None,
                       wind_speed_ms=10.0, wind_from_deg=0, horizon_minutes=120)
    assert history[-1].n_burned == 1  # only the original ignition cell burns out


# --------------------------------------------------------------------------- #
# Layer 5: Digital Twin + Alert Engine
# --------------------------------------------------------------------------- #

def test_alert_engine_respects_threshold():
    engine = AlertEngine(threshold_pct=70.0)
    processed = pd.DataFrame({
        "zone_id": ["A", "B", "C"], "latitude": [12.0] * 3, "longitude": [75.0] * 3,
        "active_fire_nearby": [False, True, False], "fwi": [10, 90, 50],
        "wx_wind_speed_ms": [1, 5, 2],
    })
    scores = np.array([0.2, 0.95, 0.5])
    alerts = engine.generate_alerts(processed, scores)
    assert len(alerts) == 1
    assert alerts[0].zone_id == "B"
    assert alerts[0].severity == "EXTREME"


def test_alert_severity_bands():
    engine = AlertEngine(threshold_pct=70.0)
    assert engine.classify(0.95) == "EXTREME"
    assert engine.classify(0.75) == "HIGH"
    assert engine.classify(0.50) == "MODERATE"
    assert engine.classify(0.10) == "LOW"


def test_digital_twin_full_refresh_cycle():
    twin = DigitalTwin(offline=True)
    snapshot = twin.refresh()
    assert snapshot is not None
    assert len(snapshot.processed_grid) == len(snapshot.risk_scores)
    summary = twin.get_summary()
    assert summary["status"] == "ok"
    assert summary["total_zones"] == len(snapshot.processed_grid)


def test_digital_twin_spread_simulation_requires_refresh_first():
    twin = DigitalTwin(offline=True)
    with pytest.raises(RuntimeError):
        twin.simulate_spread_from_alerts()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
