"""Local (geographically anchored) spread simulation: geometry, conventions
and that it drives the unchanged FireSpreadSimulator."""
import math

import numpy as np
import pandas as pd
import pytest

from config.config import SYSTEM
from src.simulation.cellular_automata import CellState
from src.simulation.local_spread import (FOCUS_AREAS, FocusArea, ca_inputs_from_conditions,
                                         focus_options_for_region, run_local_spread, zone_conditions)

BANDIPUR = FOCUS_AREAS["Bandipur Tiger Reserve"]
COND = {"zone_id": "G-61", "zone_lat": 11.65, "zone_lon": 76.65, "risk_score": 0.3, "ffmc": 93.0,
        "bui": 30.0, "fwi": 30.0, "ndvi": 0.7, "temp_c": 38, "humidity_pct": 15,
        "wind_speed_ms": 6.0, "wind_from_deg": 225.0, "elev_m": 900, "slope_pct": 3, "active_fire_nearby": False}


def test_bandipur_geometry_is_500m_square_of_25m_cells():
    f = BANDIPUR
    assert f.n == 20 and f.cell_m == 25 and f.n * f.cell_m == 500
    b = f.bounds
    ns_m = (b["north"] - b["south"]) * 111_320
    ew_m = (b["east"] - b["west"]) * 111_320 * math.cos(math.radians(f.lat))
    assert ns_m == pytest.approx(500, abs=0.5) and ew_m == pytest.approx(500, abs=0.5)
    assert f.area_km2 == pytest.approx(0.25)
    lat, lon = f.cell_centres()
    assert lat[0, 0] > lat[-1, 0]            # row 0 is north
    assert lon[0, -1] > lon[0, 0]            # col grows east
    assert lat.mean() == pytest.approx(f.lat) and lon.mean() == pytest.approx(f.lon)


def test_time_step_keeps_configured_spread_rate():
    r = run_local_spread(BANDIPUR, COND, 6.0, 225.0, seed=1)
    assert r.step_minutes == pytest.approx(15 * BANDIPUR.cell_m / SYSTEM.ca_cell_size_m)
    assert r.history[1].minutes_elapsed == pytest.approx(r.step_minutes)


def test_same_transforms_as_regional_twin():
    ca = ca_inputs_from_conditions(COND)
    assert ca["dryness"] == pytest.approx(93 / 101)
    assert ca["fuel"] == pytest.approx(0.7)
    assert ca["buildup"] == pytest.approx(1.0)
    assert not ca["non_fuel"]
    assert ca_inputs_from_conditions({**COND, "ndvi": 0.1})["non_fuel"]


def test_lifecycle_and_states_come_from_the_ca():
    r = run_local_spread(BANDIPUR, COND, 6.0, 225.0, seed=3)
    ign, out = r.ignition_step, r.burnout_step
    burned = ign >= 0
    assert burned.sum() >= r.params["n_ignition"]
    # every cell burns for exactly one CA step, then stays burned
    finished = out >= 0
    assert np.all(out[finished] == ign[finished] + 1)
    for h in r.history:
        assert set(np.unique(h.state)) <= {CellState.UNBURNED, CellState.BURNING, CellState.BURNED}
    m = r.metrics
    assert [x["burned"] for x in m] == sorted(x["burned"] for x in m)
    assert m[-1]["burned_ha"] == pytest.approx(m[-1]["burned"] * 0.0625, abs=1e-3)


def _centroid_shift(wind_from):
    shifts = []
    for seed in range(8):
        r = run_local_spread(BANDIPUR, COND, 8.0, wind_from, placement="Centre", seed=seed)
        early = (r.ignition_step >= 0) & (r.ignition_step <= 6)      # before the box edge is reached
        rr, cc = np.where(early)
        shifts.append(((r.focus.n - 1) / 2 - rr.mean(), cc.mean() - (r.focus.n - 1) / 2))   # (north, east)
    return np.mean(shifts, axis=0)


def test_wind_from_southwest_pushes_fire_northeast():
    north, east = _centroid_shift(225.0)
    assert north > 0.3 and east > 0.3


def test_wind_from_north_pushes_fire_south():
    north, east = _centroid_shift(0.0)
    assert north < -0.5 and abs(east) < abs(north)


def test_wind_from_east_pushes_fire_west():
    north, east = _centroid_shift(90.0)
    assert east < -0.5 and abs(north) < abs(east)


def test_upwind_placement_starts_on_the_upwind_side():
    r = run_local_spread(BANDIPUR, COND, 6.0, 225.0, placement="Upwind edge", seed=0)
    rr, cc = np.where(r.ignition_step == 0)
    assert rr.mean() > (r.focus.n - 1) / 2       # south half (wind from SW)
    assert cc.mean() < (r.focus.n - 1) / 2       # west half


def test_forecast_schedule_is_resampled_to_local_steps():
    sched = [(2.0, 90.0)] * 4 + [(9.0, 270.0)] * 4          # 15-min entries over 2 h
    r = run_local_spread(BANDIPUR, COND, 0, 0, seed=0, wind_schedule_15min=sched)
    assert r.wind_schedule[0] == (2.0, 90.0)
    assert r.wind_schedule[-1] == (9.0, 270.0)
    assert len(r.wind_schedule) == 32


def test_zone_conditions_pick_the_zone_containing_the_point():
    df = pd.DataFrame({"zone_id": ["A", "B"], "latitude": [11.65, 11.75], "longitude": [76.65, 76.65],
                       "ffmc": [90, 80], "bui": [10, 20], "fwi": [5, 6], "ndvi": [0.4, 0.5],
                       "wx_temperature_c": [30, 31], "wx_humidity_pct": [40, 41],
                       "wx_wind_speed_ms": [4, 5], "wx_wind_deg": [200, 210]})
    c = zone_conditions(df, np.array([0.2, 0.9]), BANDIPUR.lat, BANDIPUR.lon)
    assert c["zone_id"] == "A" and c["risk_score"] == pytest.approx(0.2)


def test_focus_options_follow_region():
    from config.config import REGION
    from src.regions import REGION_PRESETS
    assert focus_options_for_region(REGION)[0] == "Bandipur Tiger Reserve"
    assert "Bandipur Tiger Reserve" not in focus_options_for_region(REGION_PRESETS["Assam"])


def test_payload_is_deterministic_and_complete():
    from src.dashboard.geo_fire_map import build_payload
    r = run_local_spread(BANDIPUR, COND, 6.0, 225.0, seed=0)
    a = build_payload(BANDIPUR, r, 6, 225, [], "synthetic", 3, "Upwind edge", True, "k", "")
    b = build_payload(BANDIPUR, r, 6, 225, [], "synthetic", 3, "Upwind edge", True, "k", "")
    assert a == b
    assert len(a["ign"]) == len(a["out"]) == len(a["inten"]) == 400
    assert a["lastStep"] == len(a["metrics"]) - 1
    pre = build_payload(BANDIPUR, None, 6, 225, [], "synthetic", 3, "Upwind edge", False, "k", "")
    assert pre["hasRun"] is False and sum(1 for v in pre["ign"] if v == 0) == 3


def test_scenario_wind_direction_reaches_synthetic_weather():
    from src.data_ingestion.weather_client import WeatherClient
    pts = [{"latitude": 12.0, "longitude": 76.0}] * 50
    base = WeatherClient.generate_sample(pts, seed=1)
    w = WeatherClient.generate_sample(pts, seed=1, wind_from_deg=225)
    d = ((w["wind_deg"] - 225 + 180) % 360) - 180
    assert d.abs().max() <= 15.0 + 1e-9
    pd.testing.assert_frame_equal(base.drop(columns=["wind_deg", "fetched_at"]),
                                  w.drop(columns=["wind_deg", "fetched_at"]))
