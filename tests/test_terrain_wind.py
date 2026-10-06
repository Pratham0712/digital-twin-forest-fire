"""
Tests for real terrain (slope/elevation features), forecast wind and the
vector wind averaging. All offline: elevation and forecast APIs are faked.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.config import RegionConfig
from src.data_ingestion import terrain as terrain_mod
from src.data_ingestion.ingestion_module import build_region_grid
from src.data_ingestion.terrain import (
    TERRAIN_COLUMNS, TerrainTable, build_terrain_table, fetch_elevations, terrain_filename,
)
from src.data_ingestion.wind import from_uv, interpolate_schedule, mean_wind, to_uv
from src.ml_models.fire_history import FEATURE_COLUMNS_V2, FEATURE_COLUMNS_V3
from src.simulation.cellular_automata import CellState, FireSpreadSimulator

SMALL = RegionConfig(name="Test box", min_lat=12.0, max_lat=12.8, min_lon=75.0, max_lon=75.8,
                     grid_resolution_deg=0.1, weather_grid_resolution_deg=0.4)

M_PER_DEG = 111_320.0


# ── terrain ───────────────────────────────────────────────────────────────── #

def _plane(ns_m_per_deg=0.0, ew_m_per_deg=0.0):
    return lambda lat, lon: 500.0 + ns_m_per_deg * (lat - 12.0) + ew_m_per_deg * (lon - 75.0)


def test_slope_of_north_south_ramp_is_exact():
    grid = build_region_grid(SMALL)
    t = build_terrain_table(grid, SMALL.grid_resolution_deg, fetch=_plane(ns_m_per_deg=2000.0))
    assert t["slope_pct"].between(2000 / M_PER_DEG * 100 - 0.02, 2000 / M_PER_DEG * 100 + 0.02).all()
    assert list(t.columns) == ["zone_id", "latitude", "longitude", "elev_m", "slope_pct"]


def test_slope_east_west_uses_metres_at_that_latitude():
    grid = build_region_grid(SMALL)
    t = build_terrain_table(grid, SMALL.grid_resolution_deg, fetch=_plane(ew_m_per_deg=1000.0))
    expected = 1000 / (M_PER_DEG * np.cos(np.radians(grid["latitude"]))) * 100
    assert np.allclose(t["slope_pct"], expected, atol=0.02)


def test_flat_ground_has_zero_slope_and_elevation_is_the_cell_mean():
    grid = build_region_grid(SMALL)
    t = build_terrain_table(grid, SMALL.grid_resolution_deg, fetch=_plane())
    assert (t["slope_pct"] == 0).all() and (t["elev_m"] == 500.0).all()
    tilted = build_terrain_table(grid, SMALL.grid_resolution_deg, fetch=_plane(ns_m_per_deg=3000.0))
    # centre-sample of a planar ramp equals the 9-sample mean
    centre = 500.0 + 3000.0 * (grid["latitude"] - 12.0)
    assert np.allclose(tilted["elev_m"], centre, atol=0.1)


def test_nan_elevations_are_rejected_not_written():
    grid = build_region_grid(SMALL)
    with pytest.raises(ValueError):
        build_terrain_table(grid, SMALL.grid_resolution_deg, fetch=lambda la, lo: np.full(la.shape, np.nan))


def test_terrain_table_lookup_by_coordinates_and_unknown_cells_are_nan(tmp_path):
    grid = build_region_grid(SMALL)
    t = build_terrain_table(grid, SMALL.grid_resolution_deg, fetch=_plane(ns_m_per_deg=2000.0))
    t.to_csv(tmp_path / terrain_filename(SMALL.name), index=False)
    tt = TerrainTable.load(tmp_path)
    assert tt.available and tt.coverage(grid) == 1.0
    got = tt.transform(grid.sample(frac=1.0, random_state=3))            # order must not matter
    merged = got.merge(t, on="zone_id", suffixes=("", "_ref"))
    assert np.allclose(merged["elev_m"], merged["elev_m_ref"])
    other = build_region_grid(RegionConfig(name="Elsewhere", min_lat=20.0, max_lat=20.4, min_lon=80.0,
                                           max_lon=80.4, grid_resolution_deg=0.1,
                                           weather_grid_resolution_deg=0.4))
    assert tt.coverage(other) == 0.0
    assert tt.transform(other)[TERRAIN_COLUMNS].isna().all().all()


def test_no_terrain_files_means_unavailable(tmp_path):
    tt = TerrainTable.load(tmp_path)
    assert not tt.available and tt.coverage(build_region_grid(SMALL)) == 0.0


def test_terrain_filename():
    assert terrain_filename("Karnataka Western Ghats") == "terrain_karnataka_western_ghats.csv"
    assert terrain_filename("Andhra Pradesh / Telangana") == "terrain_andhra_pradesh_telangana.csv"


def test_fetch_falls_back_to_second_provider_and_never_returns_partial(monkeypatch):
    lats, lons = np.linspace(12, 13, 250), np.linspace(75, 76, 250)

    def bad(la, lo, timeout=0):
        raise ValueError("boom")

    monkeypatch.setattr(terrain_mod, "_fetch_opentopodata", bad)
    monkeypatch.setattr(terrain_mod, "_fetch_open_meteo", bad)
    monkeypatch.setattr(terrain_mod, "_fetch_open_elevation", lambda la, lo, timeout=0: la * 100.0)
    monkeypatch.setattr(terrain_mod.time, "sleep", lambda s: None)
    out = fetch_elevations(lats, lons, retries=2)
    assert np.allclose(out, lats * 100.0)

    monkeypatch.setattr(terrain_mod, "_fetch_open_elevation", bad)
    with pytest.raises(RuntimeError):
        fetch_elevations(lats, lons, retries=2)


def test_feature_set_v3_is_v2_plus_terrain():
    assert FEATURE_COLUMNS_V3 == FEATURE_COLUMNS_V2 + TERRAIN_COLUMNS
    assert len(FEATURE_COLUMNS_V3) == len(set(FEATURE_COLUMNS_V3))


def test_training_terrain_merge_matches_live_lookup(monkeypatch):
    from src.ml_models import train_real
    grid = build_region_grid(SMALL)
    t = build_terrain_table(grid, SMALL.grid_resolution_deg, fetch=_plane(ns_m_per_deg=2500.0, ew_m_per_deg=800.0))
    table = TerrainTable(t)
    df = pd.DataFrame({"zone_id": np.repeat(grid["zone_id"].values, 3), "x": 1.0})

    monkeypatch.setattr(train_real, "build_region_grid", lambda region: grid)
    out = train_real.add_terrain_training(df, SMALL, table)
    live = table.transform(grid).set_index("zone_id")
    for c in TERRAIN_COLUMNS:
        assert np.allclose(out[c].values, live.loc[out["zone_id"], c].values)

    # no table / partial coverage -> unchanged, no terrain columns
    assert "elev_m" not in train_real.add_terrain_training(df, SMALL, TerrainTable(None)).columns
    assert "elev_m" not in train_real.add_terrain_training(df, SMALL, TerrainTable(t.iloc[:5])).columns


# ── wind ──────────────────────────────────────────────────────────────────── #

def test_uv_roundtrip():
    for spd, deg in [(5, 0), (5, 90), (3, 225), (10, 359)]:
        s, d = from_uv(*to_uv(spd, deg))
        assert abs(s - spd) < 1e-9 and abs(d - deg) < 1e-6


def test_mean_wind_direction_wraps_around_north():
    speed, deg = mean_wind([5.0, 5.0], [350.0, 10.0])
    assert speed == 5.0
    assert min(deg, 360 - deg) < 1.0           # north - the old arithmetic mean gave 180 (south)


def test_mean_wind_ignores_missing_and_handles_empty():
    assert mean_wind([np.nan], [np.nan]) == (0.0, 0.0)
    speed, deg = mean_wind([3.0, np.nan], [90.0, 10.0])
    assert speed == 3.0 and abs(deg - 90.0) < 1e-6


def test_schedule_interpolates_through_north_not_south():
    t0 = 1_000_000.0
    sched = interpolate_schedule([t0, t0 + 10800], [6.0, 6.0], [350.0, 10.0], t0, 120, 15)
    assert len(sched) == 8
    assert all(min(d, 360 - d) < 25 for _, d in sched)
    assert all(abs(s - 6.0) < 1e-9 for s, _ in sched)


def test_schedule_uses_nearest_value_outside_forecast_range():
    t0 = 2_000_000.0
    sched = interpolate_schedule([t0 + 3600, t0 + 7200], [2.0, 4.0], [90.0, 90.0], t0, 60, 15)
    assert sched[0][0] == 2.0                   # before the first forecast slot
    assert all(abs(d - 90.0) < 1e-6 for _, d in sched)


def _burn(schedule=None, speed=0.0, from_deg=0.0, seed=5):
    n = 21
    sim = FireSpreadSimulator(n, n, random_state=seed)
    ig = np.zeros((n, n), dtype=bool)
    ig[10, 10] = True
    full = np.full((n, n), 0.9)
    hist = sim.run(ig, full, full, None, speed, from_deg, horizon_minutes=120, wind_schedule=schedule)
    burned = hist[-1].state == CellState.BURNED
    rows, cols = np.where(burned)
    return rows.mean() - 10, cols.mean() - 10, hist


def test_ca_schedule_drives_spread_direction():
    _, dc_east, _ = _burn(schedule=[(12.0, 270.0)] * 8)        # wind from the west -> fire runs east
    _, dc_west, _ = _burn(schedule=[(12.0, 90.0)] * 8)         # wind from the east -> fire runs west
    assert dc_east > 0.5 and dc_west < -0.5


def test_ca_schedule_overrides_constant_wind_and_changes_over_time():
    r_const, c_const, _ = _burn(speed=12.0, from_deg=270.0)
    r_sched, c_sched, _ = _burn(schedule=[(12.0, 270.0)] * 8, speed=0.0, from_deg=90.0)
    assert (r_const, c_const) == (r_sched, c_sched)            # schedule wins, same seed -> identical

    # wind that swings from west to south: fire ends up displaced both east and north-to-south
    dr, dc, _ = _burn(schedule=[(12.0, 270.0)] * 4 + [(12.0, 0.0)] * 4)
    assert dc > 0 and dr > 0                                   # east first, then southwards (row index grows)


def test_ca_short_schedule_repeats_last_entry():
    _, _, hist = _burn(schedule=[(8.0, 270.0)])
    assert len(hist) > 1


# ── twin integration ──────────────────────────────────────────────────────── #

def _twin_with_fire():
    from src.digital_twin.twin_state import DigitalTwin
    twin = DigitalTwin(offline=True)
    twin.refresh()
    twin._elevation_grid_cache = np.zeros((twin.current_snapshot.n_rows, twin.current_snapshot.n_cols))
    return twin


def test_offline_spread_uses_current_wind_at_ignition_zones():
    twin = _twin_with_fire()
    hist = twin.simulate_spread_from_top_n(3)
    assert len(hist) > 1
    w = twin.current_snapshot.ca_wind
    assert w["source"] == "current" and 0 <= w["from_deg"] < 360


def test_live_spread_uses_forecast_wind_when_available(monkeypatch):
    from src.digital_twin import twin_state
    twin = _twin_with_fire()
    twin.offline = False
    monkeypatch.setattr(twin_state.API, "owm_api_key", "test-key")
    import time
    now = int(time.time())
    slots = [{"dt": now - 600, "wind_speed_ms": 9.0, "wind_deg": 270.0},
             {"dt": now + 10200, "wind_speed_ms": 9.0, "wind_deg": 270.0}]
    calls = []

    def fake_forecast(points, horizon_hours=2.0):
        calls.append((len(points), horizon_hours))
        return [slots for _ in points]

    monkeypatch.setattr(twin.ingestion.weather, "fetch_forecast_wind", fake_forecast)
    twin.simulate_spread_from_top_n(3)
    w = twin.current_snapshot.ca_wind
    assert w["source"] == "forecast" and abs(w["from_deg"] - 270.0) < 1.0 and abs(w["speed_ms"] - 9.0) < 0.1
    assert 1 <= calls[0][0] <= 4 and calls[0][1] == 2.0

    twin.simulate_spread_from_top_n(3)          # second run within the TTL hits the cache
    assert len(calls) == 1


def test_forecast_failure_falls_back_to_current_wind(monkeypatch):
    from src.digital_twin import twin_state
    twin = _twin_with_fire()
    twin.offline = False
    monkeypatch.setattr(twin_state.API, "owm_api_key", "test-key")
    monkeypatch.setattr(twin.ingestion.weather, "fetch_forecast_wind", lambda *a, **k: [])
    twin.simulate_spread_from_top_n(3)
    assert twin.current_snapshot.ca_wind["source"] == "current"

    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(twin.ingestion.weather, "fetch_forecast_wind", boom)
    twin._forecast_cache.clear()
    twin.simulate_spread_from_top_n(3)
    assert twin.current_snapshot.ca_wind["source"] == "current"


def test_twin_adds_terrain_features_and_uses_them_for_the_ca(monkeypatch):
    from src.digital_twin.twin_state import DigitalTwin
    twin = DigitalTwin(offline=True)
    grid = twin.ingestion.grid
    t = build_terrain_table(grid, twin.region.grid_resolution_deg,
                            fetch=_plane(ns_m_per_deg=4000.0))
    twin.terrain = TerrainTable(t)
    snap = twin.refresh()
    g = snap.processed_grid
    assert g["elev_m"].notna().all() and g["slope_pct"].notna().all()
    assert (g["slope_pct"] > 3.0).all()
    twin.simulate_spread_from_top_n(3)          # elevation comes from the table, no network call
    assert twin._elevation_grid_cache is not None
    r = g.iloc[0]
    assert twin._elevation_grid_cache[int(r["row"]), int(r["col"])] == pytest.approx(r["elev_m"])


def test_twin_without_terrain_table_still_runs():
    from src.digital_twin.twin_state import DigitalTwin
    twin = DigitalTwin(offline=True)
    twin.terrain = TerrainTable(None)
    g = twin.refresh().processed_grid
    assert g["elev_m"].isna().all()


def test_fetch_elevations_cached_resumes(tmp_path, monkeypatch):
    from src.data_ingestion import terrain as terrain_mod
    lats = np.linspace(12, 13, 250)
    lons = np.linspace(75, 76, 250)
    calls = []

    def fake(la, lo, **kw):
        calls.append(len(la))
        if len(calls) == 2:                       # second chunk hits a rate limit
            raise RuntimeError("Elevation missing")
        return la * 10.0

    monkeypatch.setattr(terrain_mod, "fetch_elevations", fake)
    cache = tmp_path / "c.csv"
    import pytest
    with pytest.raises(RuntimeError, match="Progress is saved"):
        terrain_mod.fetch_elevations_cached(lats, lons, cache)
    calls.clear()
    monkeypatch.setattr(terrain_mod, "fetch_elevations", lambda la, lo, **kw: (calls.append(len(la)), la * 10.0)[1])
    out = terrain_mod.fetch_elevations_cached(lats, lons, cache)
    assert np.allclose(out, lats * 10.0)
    assert sum(calls) == 150                      # only the 150 points not yet cached


def test_opentopodata_parsing_treats_null_as_sea_level(monkeypatch):
    class R:
        def raise_for_status(self): pass
        def json(self):
            return {"status": "OK", "results": [{"elevation": 512.0}, {"elevation": None}]}
    monkeypatch.setattr(terrain_mod.requests, "get", lambda *a, **k: R())
    out = terrain_mod._fetch_opentopodata(np.array([12.0, 13.0]), np.array([75.0, 75.0]))
    assert list(out) == [512.0, 0.0]


def test_terrain_load_prefers_the_regions_own_file(tmp_path):
    """Two regions sharing a border-cell coordinate must not mix their slopes."""
    from src.data_ingestion.terrain import terrain_filename
    a = pd.DataFrame({"zone_id": ["G-0"], "latitude": [12.85], "longitude": [77.05],
                      "elev_m": [746.7], "slope_pct": [0.49]})
    b = pd.DataFrame({"zone_id": ["G-9"], "latitude": [12.85], "longitude": [77.05],
                      "elev_m": [748.0], "slope_pct": [1.16]})
    a.to_csv(tmp_path / terrain_filename("Region A"), index=False)
    b.to_csv(tmp_path / terrain_filename("Region B"), index=False)
    grid = pd.DataFrame({"zone_id": ["x"], "latitude": [12.85], "longitude": [77.05]})
    assert TerrainTable.load(tmp_path, "Region A").transform(grid)["slope_pct"].iloc[0] == 0.49
    assert TerrainTable.load(tmp_path, "Region B").transform(grid)["slope_pct"].iloc[0] == 1.16
