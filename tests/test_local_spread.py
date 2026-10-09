"""Local (geographically anchored) spread simulation: geometry, focus area vs
simulation domain, duration, conventions, and that it drives the unchanged
FireSpreadSimulator."""
import math

import numpy as np
import pandas as pd
import pytest

from config.config import SYSTEM
from src.simulation.cellular_automata import CellState
from src.simulation.local_spread import (FOCUS_AREAS, FocusArea, ca_inputs_from_conditions, domain_for,
                                         focus_options_for_region, ignition_mask, n_steps_for, run_local_spread,
                                         step_minutes_for, validate_setup, zone_conditions)

BANDIPUR = FOCUS_AREAS["Bandipur Tiger Reserve"]
COND = {"zone_id": "G-61", "zone_lat": 11.65, "zone_lon": 76.65, "risk_score": 0.3, "ffmc": 93.0,
        "bui": 30.0, "fwi": 30.0, "ndvi": 0.7, "temp_c": 38, "humidity_pct": 15,
        "wind_speed_ms": 6.0, "wind_from_deg": 225.0, "elev_m": 900, "slope_pct": 3, "active_fire_nearby": False}


def _extent_m(b, lat):
    return ((b["north"] - b["south"]) * 111_320, (b["east"] - b["west"]) * 111_320 * math.cos(math.radians(lat)))


def test_bandipur_geometry_is_500m_square_of_25m_cells():
    f = BANDIPUR
    assert f.n == 20 and f.n_rows == 20 and f.cell_m == 25 and f.n * f.cell_m == 500
    ns_m, ew_m = _extent_m(f.bounds, f.lat)
    assert ns_m == pytest.approx(500, abs=0.5) and ew_m == pytest.approx(500, abs=0.5)
    assert f.area_km2 == pytest.approx(0.25)
    lat, lon = f.cell_centres()
    assert lat[0, 0] > lat[-1, 0] and lon[0, -1] > lon[0, 0]          # row 0 north, col grows east
    assert lat.mean() == pytest.approx(f.lat) and lon.mean() == pytest.approx(f.lon)


@pytest.mark.parametrize("w,h,cell,grid", [(1000, 1000, 25, (40, 40)), (1500, 1000, 25, (60, 40)),
                                           (250, 250, 25, (10, 10)), (2000, 1500, 50, (40, 30))])
def test_custom_dimensions_give_the_right_grid(w, h, cell, grid):
    f = BANDIPUR.with_size(w, h, cell)
    assert (f.n_cols, f.n_rows) == grid
    ns_m, ew_m = _extent_m(f.bounds, f.lat)
    assert ew_m == pytest.approx(w, abs=0.5) and ns_m == pytest.approx(h, abs=0.5)
    assert f.area_km2 == pytest.approx(w * h / 1e6)
    assert f.with_center(12.0, 76.0).lat == 12.0


def test_domain_margin_covers_the_whole_duration():
    for dur in (1, 5, 30, 60, 120):
        d = domain_for(BANDIPUR, dur)
        assert d.margin == n_steps_for(dur, BANDIPUR.cell_m) + 1
        assert (d.n_rows, d.n_cols) == (20 + 2 * d.margin, 20 + 2 * d.margin)
        r0, c0 = d.focus_slice()
        assert d.focus_mask().sum() == 400 and r0.start == d.margin
        ns_m, ew_m = _extent_m(d.bounds, d.lat)
        assert ew_m == pytest.approx(d.width_m, abs=0.5)


def test_validation_limits():
    assert validate_setup(500, 500, 25, 60) is None
    assert validate_setup(50, 500, 25, 60)                       # below minimum size
    assert validate_setup(500, 500, 7, 60)                        # unsupported cell size
    assert validate_setup(3000, 3000, 5, 240)                     # domain far beyond the cell limit
    assert validate_setup(500, 500, 25, 0)


def test_time_step_and_duration():
    # Phase 3: legacy-CA time axis (3.75 min steps at 25 m). The default "ros" model uses frames of
    # frame_minutes_for(duration) - covered by test_phase3_model.py::test_duration_*.
    r = run_local_spread(BANDIPUR, COND, 6.0, 225.0, seed=1, duration_minutes=60, model="legacy")
    assert r.step_minutes == pytest.approx(step_minutes_for(25)) == pytest.approx(3.75)
    assert r.history[1].minutes_elapsed == pytest.approx(3.75)
    assert len(r.wind_schedule) == n_steps_for(60, 25) == 16
    assert r.end_step == pytest.approx(min(16.0, r.history[-1].step))
    short = run_local_spread(BANDIPUR, COND, 6.0, 225.0, seed=1, duration_minutes=1, model="legacy")
    assert short.end_step == pytest.approx(1 / 3.75)              # playback ends at 1 simulated minute


def test_same_transforms_as_regional_twin():
    ca = ca_inputs_from_conditions(COND)
    assert ca["dryness"] == pytest.approx(93 / 101)
    assert ca["fuel"] == pytest.approx(0.7)
    assert ca["buildup"] == pytest.approx(1.0)
    assert not ca["non_fuel"]
    assert ca_inputs_from_conditions({**COND, "ndvi": 0.1})["non_fuel"]


def test_lifecycle_and_states_come_from_the_ca():
    # legacy CA: a cell burns for exactly one step (the "ros" model keeps a cell burning while it still
    # spreads - see test_phase3_model.py::test_ros_lifecycle)
    r = run_local_spread(BANDIPUR, COND, 6.0, 225.0, seed=3, duration_minutes=60, model="legacy")
    ign, out = r.ignition_step, r.burnout_step
    assert (ign >= 0).sum() >= r.params["n_ignition"]
    finished = out >= 0
    assert np.all(out[finished] == ign[finished] + 1)            # one CA step burning, then burned
    for h in r.history:
        assert set(np.unique(h.state)) <= {CellState.UNBURNED, CellState.BURNING, CellState.BURNED}
    m = r.metrics
    assert [x["burned"] for x in m] == sorted(x["burned"] for x in m)
    assert m[-1]["burned_ha"] == pytest.approx(m[-1]["burned"] * 0.0625, abs=1e-3)


def test_fire_is_not_stopped_by_the_focus_box():
    """Strong fire in a small focus area: it must leave the focus area, and the
    domain margin guarantees it never reaches the domain edge."""
    f = BANDIPUR.with_size(250, 250, 25)
    r = run_local_spread(f, COND, 9.0, 225.0, seed=2, duration_minutes=60, base_spread_prob=1.2)
    fm = r.domain.focus_mask()
    affected = r.ignition_step >= 0
    assert (affected & ~fm).any()                                # burned outside the focus area
    assert any(x["left_focus"] for x in r.metrics)
    assert not any(x["domain_edge_reached"] for x in r.metrics)
    rows = np.where(affected.any(axis=1))[0]
    cols = np.where(affected.any(axis=0))[0]
    assert rows.min() > 0 and cols.min() > 0 and rows.max() < r.domain.n_rows - 1 and cols.max() < r.domain.n_cols - 1


def _centroid_shift(wind_from):
    shifts = []
    for seed in range(8):
        r = run_local_spread(BANDIPUR, COND, 8.0, wind_from, placement="Centre", seed=seed, duration_minutes=60)
        early = (r.ignition_step >= 0) & (r.ignition_step <= 6)
        rr, cc = np.where(early)
        # Phase 3: relative to the ignition cells, not the domain centre - the faster "ros" fire makes the
        # adaptive domain grow asymmetrically, which moves the domain centre
        r0, c0 = np.argwhere(r.ignition_step == 0).mean(axis=0)
        shifts.append((r0 - rr.mean(), cc.mean() - c0))   # (north, east)
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


def test_placements():
    d = domain_for(BANDIPUR, 60)
    mid = (d.n_rows - 1) / 2
    up = np.argwhere(ignition_mask(d, 3, "Upwind edge", 6, 225))
    down = np.argwhere(ignition_mask(d, 3, "Downwind edge", 6, 225))
    centre = np.argwhere(ignition_mask(d, 3, "Centre", 6, 225))
    assert up[:, 0].mean() > mid and up[:, 1].mean() < mid          # SW of centre (wind from SW)
    assert down[:, 0].mean() < mid and down[:, 1].mean() > mid      # NE of centre
    assert abs(centre[:, 0].mean() - mid) < 1.5
    assert d.focus_mask()[tuple(up.T)].all()                        # placed inside the focus area


def test_map_point_ignition_maps_lat_lon_to_the_containing_cell():
    d = domain_for(BANDIPUR, 60)
    lat_c, lon_c = d.cell_centres()
    r, c = 30, 12                                                   # a margin cell outside the focus area
    pts = [(float(lat_c[r, c]) + 0.3 * BANDIPUR.dlat, float(lon_c[r, c]) - 0.3 * BANDIPUR.dlon)]
    m = ignition_mask(d, 3, "Map points", 6, 225, pts)
    assert m.sum() == 1 and m[r, c]
    res = run_local_spread(BANDIPUR, COND, 6.0, 225.0, placement="Map points", ignition_points=pts, seed=0,
                           duration_minutes=60)
    # The run's domain may have grown to the north / west while the fire spread
    # (adaptive domain, audit BUG #1 fix), so the cell is looked up in the
    # result's own domain; it is still exactly the clicked cell, lit at step 0.
    rr, rc = res.domain.cell_of(*pts[0])
    assert (rr - res.domain.m_north, rc - res.domain.m_west) == (r - d.m_north, c - d.m_west)
    assert res.ignition_step[rr, rc] == 0 and int((res.ignition_step == 0).sum()) == 1
    assert ignition_mask(d, 3, "Map points", 6, 225, [(0.0, 0.0)]).sum() == 3   # outside: falls back to centre


def test_forecast_schedule_is_resampled_to_local_steps():
    sched = [(2.0, 90.0)] * 4 + [(9.0, 270.0)] * 4                  # 15-min entries over 2 h
    r = run_local_spread(BANDIPUR, COND, 0, 0, seed=0, wind_schedule_15min=sched, duration_minutes=120,
                         model="legacy")
    assert r.wind_schedule[0] == (2.0, 90.0)
    assert r.wind_schedule[-1] == (9.0, 270.0)
    assert len(r.wind_schedule) == 32
    # default "ros" model: one wind entry per playback frame, same forecast
    q = run_local_spread(BANDIPUR, COND, 0, 0, seed=0, wind_schedule_15min=sched, duration_minutes=120)
    assert q.wind_schedule[0] == (2.0, 90.0) and q.wind_schedule[-1] == (9.0, 270.0)
    assert len(q.wind_schedule) == round(120 / q.step_minutes)


def test_zone_conditions_pick_the_zone_containing_the_point():
    # zones placed relative to the Bandipur preset (Phase 3 moved the preset to a forest interior)
    la, lo = round(BANDIPUR.lat, 2), round(BANDIPUR.lon, 2)
    df = pd.DataFrame({"zone_id": ["A", "B"], "latitude": [la, la + 0.1], "longitude": [lo, lo],
                       "ffmc": [90, 80], "bui": [10, 20], "fwi": [5, 6], "ndvi": [0.4, 0.5],
                       "wx_temperature_c": [30, 31], "wx_humidity_pct": [40, 41],
                       "wx_wind_speed_ms": [4, 5], "wx_wind_deg": [200, 210]})
    c = zone_conditions(df, np.array([0.2, 0.9]), BANDIPUR.lat, BANDIPUR.lon)
    assert c["zone_id"] == "A" and c["risk_score"] == pytest.approx(0.2)
    assert c["zone_distance_km"] < 3


def test_focus_options_follow_region():
    from config.config import REGION
    from src.regions import REGION_PRESETS
    assert focus_options_for_region(REGION)[0] == "Bandipur Tiger Reserve"
    assert "Bandipur Tiger Reserve" not in focus_options_for_region(REGION_PRESETS["Assam"])


def test_payloads_are_deterministic_and_complete():
    from src.dashboard.geo_fire_map import setup_payload, sim_payload
    r = run_local_spread(BANDIPUR, COND, 6.0, 225.0, seed=0, duration_minutes=60)
    layers = {"grid": False, "boundary": True}
    a = sim_payload(BANDIPUR, 60, r, 6, 225, [], "synthetic", 3, "Upwind edge", [], layers, True, "k", "")
    b = sim_payload(BANDIPUR, 60, r, 6, 225, [], "synthetic", 3, "Upwind edge", [], layers, True, "k", "")
    assert a == b and a["uid"] == b["uid"]
    assert len(a["cells"]) == len(a["ign"]) == len(a["out"]) == len(a["inten"]) == int((r.ignition_step >= 0).sum())
    assert a["domain"]["n_rows"] == r.domain.n_rows and a["layers"] == layers
    assert a["lastStep"] == len(a["metrics"]) - 1 and a["endT"] <= a["lastStep"]
    pre = sim_payload(BANDIPUR, 60, None, 6, 225, [], "synthetic", 3, "Upwind edge", [], layers, False, "k", "")
    assert pre["hasRun"] is False and len(pre["preview"]) == 3 and pre["uid"] != a["uid"]
    s = setup_payload(BANDIPUR.with_size(1000, 750), 30, 6, 225, 3, "Centre", [], layers, "k", "")
    assert s["mode"] == "setup" and s["focus"]["n_cols"] == 40 and s["focus"]["n_rows"] == 30 and s["editable"]


def test_scenario_wind_direction_reaches_synthetic_weather():
    from src.data_ingestion.weather_client import WeatherClient
    pts = [{"latitude": 12.0, "longitude": 76.0}] * 50
    base = WeatherClient.generate_sample(pts, seed=1)
    w = WeatherClient.generate_sample(pts, seed=1, wind_from_deg=225)
    d = ((w["wind_deg"] - 225 + 180) % 360) - 180
    assert d.abs().max() <= 15.0 + 1e-9
    pd.testing.assert_frame_equal(base.drop(columns=["wind_deg", "fetched_at"]),
                                  w.drop(columns=["wind_deg", "fetched_at"]))
