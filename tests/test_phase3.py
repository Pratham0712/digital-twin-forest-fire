"""Phase 3 - simulation domain, Bandipur default, weather response, duration,
ignition anywhere in the domain, rendering sync, interaction performance.
All offline: no Google Maps, NASA FIRMS, OpenWeatherMap or downloads."""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.simulation.local_spread import (DOMAIN_AUTO, DOMAIN_CUSTOM, FOCUS_AREAS, FocusArea, SimulationDomain,
                                         domain_from_setup, frame_minutes_for, initial_domain, run_local_spread,
                                         validate_domain)
from src.simulation.cellular_automata import CellState

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "src" / "dashboard" / "components" / "fire_map" / "index.html").read_text(encoding="utf-8")
BP = FOCUS_AREAS["Bandipur Tiger Reserve"]
DRY = {"ffmc": 93.0, "bui": 60.0, "ndvi": 0.6}


def _setup(**kw):
    s = {"location": {"name": BP.name, "lat": BP.lat, "lon": BP.lon, "source": "preset"}, "width_m": 1000.0,
         "height_m": 1000.0, "cell_m": 25.0, "duration_min": 60.0, "placement": "Map points", "n_ignition": 5,
         "ignition_points": [], "ignition_source": "hypothetical", "layers": {"grid": True, "boundary": True},
         "domain_preset": DOMAIN_AUTO, "domain_w_m": 2000.0, "domain_h_m": 2000.0, "max_extent_m": 15000.0}
    s.update(kw)
    return s


def _run(f, cond=DRY, ws=6.0, wd=270.0, dur=60, **kw):
    return run_local_spread(f, cond, ws, wd, placement="Centre", n_ignition=1, duration_minutes=dur, seed=3, **kw)


def _shape(r, wind_from):
    """head / back distance (m) along the downwind axis and width across it, from the ignition cell."""
    aff = r.ignition_step >= 0
    rr, cc = np.nonzero(aff)
    r0, c0 = np.argwhere(r.ignition_step == 0).mean(axis=0)
    x, y = (cc - c0) * r.focus.cell_m, -(rr - r0) * r.focus.cell_m
    dn = math.radians((wind_from + 180) % 360)
    along, across = x * math.sin(dn) + y * math.cos(dn), -x * math.cos(dn) + y * math.sin(dn)
    return along.max(), -along.min(), across.max() - across.min(), x.mean(), y.mean()


# 1 ─ domain presets are real grid dimensions ────────────────────────────────
@pytest.mark.parametrize("preset,side", [("Local · 1 km", 1000), ("Small · 2 km", 2000), ("Medium · 5 km", 5000),
                                         ("Large · 10 km", 10000)])
def test_01_domain_presets_set_the_real_grid(preset, side):
    d = domain_from_setup(_setup(domain_preset=preset))
    assert d.width_m == side and d.height_m == side
    assert (d.n_cols, d.n_rows) == (side // 25, side // 25)
    from src.dashboard.geo_fire_map import setup_payload
    p = setup_payload(d.focus, 60, 5, 270, 3, "Map points", [], {}, "K", "", domain=d, domain_size=(side, side))
    assert p["domain"]["n_cols"] == side // 25 and p["domainSize"] == [side, side]   # the map draws that domain


def test_01b_custom_domain_and_validation():
    d = domain_from_setup(_setup(domain_preset=DOMAIN_CUSTOM, domain_w_m=3000.0, domain_h_m=1500.0))
    assert (d.width_m, d.height_m) == (3000, 1500)
    f = FocusArea("t", BP.lat, BP.lon, 1000, 1000, 25)
    assert validate_domain(f, (800, 800))[0].startswith("The simulation domain (800")           # smaller than focus
    assert "maximum simulation extent" in validate_domain(f, (12000, 12000), {"max_extent_m": 10000})[0]
    assert "cells" in validate_domain(FocusArea("t", BP.lat, BP.lon, 1000, 1000, 10), (10000, 10000))[0]
    assert validate_domain(f, (10000, 10000))[1].startswith("Large grid")                       # warning only


# 2 ─ focus area and simulation domain are distinct ──────────────────────────
def test_02_focus_and_domain_are_distinct():
    d = domain_from_setup(_setup(domain_preset="Small · 2 km"))
    fs = d.focus_slice()
    assert (fs[0].stop - fs[0].start, fs[1].stop - fs[1].start) == (40, 40)                   # 1 km focus
    assert (d.n_rows, d.n_cols) == (80, 80) and d.margin_tuple == (20, 20, 20, 20)            # centred in 2 km
    assert d.bounds["north"] > d.focus.bounds["north"] and d.bounds["west"] < d.focus.bounds["west"]


# 3 ─ expansion preserves fire state and elapsed time ────────────────────────
def test_03_expansion_preserves_state_and_time():
    f = FocusArea("t", BP.lat, BP.lon, 500, 500, 25)
    small = initial_domain(f, 60, domain_size_m=(500, 500))
    big = initial_domain(f, 60, domain_size_m=(4000, 4000))
    a = _run(f, ws=8.0, wd=241.0, domain=small)
    b = _run(f, ws=8.0, wd=241.0, domain=big, expand=False)
    assert a.expansions and not b.expansions
    # the grown run equals the run on a large domain, cell by cell (same lattice, same times)
    ta = {tuple(np.round(xy, 6)): t for xy, t in zip(np.column_stack([c.ravel() for c in a.domain.cell_centres()]),
                                                  a.ignition_time.ravel()) if t >= 0}
    tb = {tuple(np.round(xy, 6)): t for xy, t in zip(np.column_stack([c.ravel() for c in b.domain.cell_centres()]),
                                                  b.ignition_time.ravel()) if t >= 0}
    assert ta.keys() == tb.keys() and all(abs(ta[k] - tb[k]) < 1e-6 for k in ta)   # float rounding of the patch noise
    assert [m["minutes"] for m in a.metrics] == [m["minutes"] for m in b.metrics]             # elapsed time
    burned = [m["burned"] + m["burning"] for m in a.metrics]
    assert burned == sorted(burned)                                                            # never reset


# 4 ─ Bandipur default is a forest interior, used consistently ───────────────
def test_04_bandipur_default_location_is_consistent():
    from src.geo.study_region import load_study_region
    assert (BP.lat, BP.lon) == (11.6560, 76.5800) and (BP.lat, BP.lon) != (11.6667, 76.6333)
    reg = load_study_region()
    assert reg is not None and bool(reg.contains(np.array([BP.lat]), np.array([BP.lon]))[0])
    assert reg.status != "OFFICIAL"                                    # approximate boundary still disclosed
    s = _setup()
    d = domain_from_setup(s)
    lat_c, lon_c = d.cell_centres()
    r, c = d.cell_of(BP.lat, BP.lon)
    assert abs(lat_c[r, c] - BP.lat) <= d.focus.dlat / 2 + 1e-9 and abs(lon_c[r, c] - BP.lon) <= d.focus.dlon / 2 + 1e-9
    assert abs((d.bounds["north"] + d.bounds["south"]) / 2 - BP.lat) < 1e-9                  # map centre = sim


# 5 ─ wind direction sets the spread direction (meteorological FROM) ─────────
@pytest.mark.parametrize("wind_from,expect", [(270.0, "east"), (90.0, "west"), (0.0, "south"), (180.0, "north"),
                                              (225.0, "northeast")])
def test_05_wind_direction_controls_spread_direction(wind_from, expect):
    f = FocusArea("t", BP.lat, BP.lon, 500, 500, 25)
    r = _run(f, ws=8.0, wd=wind_from, dur=30)
    head, back, width, mx, my = _shape(r, wind_from)
    assert head > 2 * max(back, 25.0)          # backing floor 20 % of head (fire also spreads upwind)
    dx, dy = {"east": (1, 0), "west": (-1, 0), "south": (0, -1), "north": (0, 1), "northeast": (1, 1)}[expect]
    assert mx * dx >= 0 and my * dy >= 0 and (abs(mx) + abs(my)) > 50


# 6 ─ sensible, graded response to wind and fire weather ─────────────────────
def test_06_wind_and_fire_weather_change_the_fire_sensibly():
    f = FocusArea("t", BP.lat, BP.lon, 500, 500, 25)
    calm, breezy, windy = (_run(f, ws=w, dur=60) for w in (1.0, 5.0, 10.0))
    hc, hb, hw = (_shape(r, 270.0) for r in (calm, breezy, windy))
    assert hc[0] < hb[0] < hw[0]                                       # head fire faster with wind
    assert hc[0] / hc[2] < hb[0] / hb[2] < hw[0] / hw[2] * 1.5         # and more elongated
    moist = _run(f, {"ffmc": 82.0, "bui": 30.0, "ndvi": 0.6}, ws=5.0)
    dry = _run(f, {"ffmc": 95.0, "bui": 80.0, "ndvi": 0.6}, ws=5.0)
    assert moist.final["fire_area_ha"] < breezy.final["fire_area_ha"] < dry.final["fire_area_ha"]
    wet = _run(f, {"ffmc": 70.0, "bui": 10.0, "ndvi": 0.6}, ws=2.0)
    assert wet.final["burning"] == 0 and wet.params["extinguished_at_min"] is not None   # goes out by itself
    # no artificial burning of everything: even the driest run burns a minority of its domain
    assert (dry.ignition_step >= 0).mean() < 0.5
    # the response is graded, not saturated: 10 m/s is not the same as 5 m/s
    assert windy.final["front_distance_m"] > 1.5 * breezy.final["front_distance_m"]
    # temperature / humidity reach the fire through FFMC: higher FFMC = faster spread
    from src.simulation.fire_behaviour import spread_rates
    assert spread_rates(96, 60, 5, 270).head > 2 * spread_rates(88, 60, 5, 270).head


def test_06b_rates_match_the_fbp_head_rate():
    from src.simulation.fire_behaviour import spread_rates
    f = FocusArea("t", BP.lat, BP.lon, 500, 500, 25)
    for ws, dur in ((5.0, 60), (10.0, 30)):
        r = _run(f, {"ffmc": 93.0, "bui": 60.0, "ndvi": 0.55}, ws=ws, dur=dur)
        head = _shape(r, 270.0)[0]
        expect = spread_rates(93.0, 60.0, ws, 270.0).head * dur
        assert 0.75 * expect <= head <= 1.2 * expect                   # CA advances at the physical ROS


# 7 ─ never beyond the configured maximum extent ─────────────────────────────
def test_07_fire_never_exceeds_the_configured_maximum_extent():
    f = FocusArea("t", BP.lat, BP.lon, 500, 500, 25)
    r = _run(f, ws=12.0, wd=270.0, dur=120, domain=initial_domain(f, 120, domain_size_m=(1000, 1000)),
             limits={"max_extent_m": 2000.0})
    assert max(r.domain.width_m, r.domain.height_m) <= 2000.0
    assert r.extent_limit is not None and "maximum" in r.extent_limit["reason"]


# 8 ─ selected duration is calculated in full ────────────────────────────────
@pytest.mark.parametrize("dur", [1, 10, 30, 60])
def test_08_selected_duration_is_simulated(dur):
    f = FocusArea("t", BP.lat, BP.lon, 500, 500, 25)
    r = _run(f, ws=5.0, dur=dur)
    assert r.duration_minutes == dur and r.history[-1].minutes_elapsed == pytest.approx(dur)
    assert r.metrics[-1]["minutes"] == pytest.approx(dur)
    assert r.end_step == pytest.approx(dur / r.step_minutes) == pytest.approx(len(r.history) - 1)
    assert r.step_minutes == pytest.approx(frame_minutes_for(dur))


def test_08b_extinguished_fire_still_covers_the_duration():
    f = FocusArea("t", BP.lat, BP.lon, 500, 500, 25)
    r = _run(f, {"ffmc": 70.0, "bui": 10.0, "ndvi": 0.6}, ws=2.0, dur=120)
    assert r.params["extinguished_at_min"] < 120 and r.history[-1].minutes_elapsed == pytest.approx(120)
    assert r.metrics[-1]["minutes"] == pytest.approx(120) and r.final["burning"] == 0


# 9 ─ playback speed never changes the computed fire ─────────────────────────
def test_09_playback_speed_is_display_only():
    from src.dashboard.geo_fire_map import playback_seconds_1x, sim_payload
    f = FocusArea("t", BP.lat, BP.lon, 500, 500, 25)
    r = _run(f, ws=5.0, dur=30)
    p = sim_payload(f, 30, r, 5.0, 270, [], "observed", 1, "Centre", [], {}, False, "K", "")
    assert p["secPerStep"] * p["endT"] == pytest.approx(playback_seconds_1x(30), rel=1e-3)
    # the browser only scales the clock: simT += dt * speed / secPerStep (no recomputation)
    assert "simT=Math.min(ENDT,simT+dt*speed/secPerStep)" in HTML
    r2 = _run(f, ws=5.0, dur=30)
    assert np.array_equal(r.ignition_time, r2.ignition_time)                  # deterministic, computed once


# 10 - 12 ─ ignition anywhere inside the simulation domain ───────────────────
def _click(s, pts, land=None):
    from src.dashboard.geo_spread import apply_map_event
    msgs, reasons = [], []
    apply_map_event({"kind": "ignite", "points": pts}, s, None, whatif=True, land=land, messages=msgs,
                    reasons=reasons)
    return msgs


def test_10_ignition_outside_focus_but_inside_domain_is_accepted():
    s = _setup(domain_preset="Medium · 5 km")
    p = [BP.lat + 1500 / 111320.0, BP.lon]                              # 1.5 km north: outside the 1 km focus
    msgs = _click(s, [p])
    assert s["ignition_points"] == [p] and msgs == []
    d = domain_from_setup(s, s["ignition_points"])
    r, c = d.cell_of(*p)
    assert not d.focus_mask()[r, c]                                      # really outside the focus area
    res = run_local_spread(d.focus, DRY, 5.0, 270.0, placement="Map points", ignition_points=[p],
                           strict_points=True, duration_minutes=10, domain=d)
    r2, c2 = res.domain.cell_of(*p)
    assert res.ignition_step[r2, c2] == 0                                 # the fire starts there


def test_11_edge_and_corner_clicks_map_to_their_cells():
    s = _setup(domain_preset="Small · 2 km")
    d = domain_from_setup(s)
    b = d.bounds
    eps_lat, eps_lon = d.focus.dlat * 0.25, d.focus.dlon * 0.25
    cases = {(0, 0): (b["north"] - eps_lat, b["west"] + eps_lon),
             (d.n_rows - 1, d.n_cols - 1): (b["south"] + eps_lat, b["east"] - eps_lon),
             (0, d.n_cols // 2): (b["north"] - eps_lat, (b["west"] + b["east"]) / 2 + eps_lon),
             (d.n_rows // 2, d.n_cols // 2): (BP.lat - eps_lat, BP.lon + eps_lon)}
    for rc, (la, lo) in cases.items():
        assert d.cell_of(la, lo) == rc
    msgs = _click(s, [list(v) for v in cases.values()])
    assert len(s["ignition_points"]) == 4 and msgs == []


def test_12_truly_outside_domain_click_is_rejected():
    from src.dashboard.geo_spread import HYP_OUTSIDE
    s = _setup(domain_preset="Small · 2 km")
    msgs = _click(s, [[BP.lat + 1100 / 111320.0, BP.lon]])                # 1.1 km north: beyond the 2 km domain
    assert s["ignition_points"] == [] and msgs == [HYP_OUTSIDE]
    assert "focusRect.addListener('click',e=>{if(picking)onMapClick(e);});" in HTML   # clicks inside the box work


# 13 - 14 ─ OSM ignition check (lazy, per point) ─────────────────────────────
def test_13_14_lazy_osm_check_rejects_roads_and_marks_missing_data_unverified(monkeypatch):
    from src.simulation import fuel_map
    from src.simulation.ignition_site import IgnitionSiteClassifier
    calls = []
    road = {"type": "way", "tags": {"highway": "secondary"},
            "geometry": [{"lat": BP.lat, "lon": BP.lon - 0.01}, {"lat": BP.lat, "lon": BP.lon + 0.01}]}

    def fake(bbox, allow_fetch=True, timeout_s=None):
        calls.append(bbox)
        return {"elements": [road], "fetched_utc": "2026-10-09T00:00:00"}
    monkeypatch.setattr(fuel_map, "fetch_osm", fake)
    land = IgnitionSiteClassifier.for_domain(domain_from_setup(_setup(domain_preset="Large · 10 km")))
    assert calls == []                                                   # nothing fetched until a point is checked
    assert land.classify(BP.lat, BP.lon).label == "ROAD"
    ok = land.classify(BP.lat + 0.0004, BP.lon)
    assert ok.accepted and ok.label == "LOCATION UNVERIFIED"             # no vegetation mapped: not "vegetation"
    assert len(calls) == 1 and (calls[0][2] - calls[0][0]) < 0.02        # one small tile, not the 10 km domain
    monkeypatch.setattr(fuel_map, "fetch_osm", lambda *a, **k: None)
    off = IgnitionSiteClassifier.for_domain(None).classify(BP.lat, BP.lon)
    assert off.accepted and off.label == "LOCATION UNVERIFIED" and "unavailable" in off.message


# 15 - 16 ─ LIVE never ignites from weather risk / zero detections ───────────
def test_15_16_live_needs_firms_detections():
    from src.dashboard import live_modes as lm
    s = _setup(ignition_source="none")
    f = domain_from_setup(s).focus
    for dets in ([], None):
        plan = lm.plan_ignition(lm.LIVE, s, f, {"ndvi": 0.6, "ffmc": 97.0, "bui": 90.0, "risk_score": 0.99}, dets,
                                "live", allow_fetch=False)
        assert plan["ign"]["source"] == "NONE" and plan["cls"]["n_valid"] == 0
        assert lm.assess(lm.LIVE, "live", plan["cls"], "EXTREME", plan["ign"])["code"] == "high_risk_no_fire"


# 17 ─ rendering follows the simulation state ────────────────────────────────
def test_17_renderer_times_come_from_the_simulation():
    from src.dashboard.geo_fire_map import sim_payload
    f = FocusArea("t", BP.lat, BP.lon, 500, 500, 25)
    r = _run(f, ws=6.0, dur=30)
    p = sim_payload(f, 30, r, 6.0, 270, [], "observed", 1, "Centre", [], {"smoke": True}, False, "K", "")
    q = sim_payload(f, 30, r, 6.0, 270, [], "observed", 1, "Centre", [], {"smoke": False}, False, "K", "")
    assert p["cells"] == q["cells"] and p["ignT"] == q["ignT"] and p["outT"] == q["outT"]
    ign = np.array(p["ign"])
    assert np.all(np.ceil(np.round(np.array(p["ignT"]), 3) - 1e-3) <= ign) and len(p["ignT"]) == len(p["cells"])
    for k, h in enumerate(r.history):                                   # the frame state = the exact times
        t = k * r.step_minutes
        burning_by_time = (r.ignition_time >= 0) & (r.ignition_time <= t + 1e-9) & \
            ((r.burnout_time < 0) | (r.burnout_time > t + 1e-9))
        assert np.array_equal(burning_by_time, h.state == CellState.BURNING)
    assert "IGN[i]=exact?p.ignT[k]:p.ign[k];" in HTML                    # the browser draws those times


# 18 ─ dragging the box never runs the simulation ────────────────────────────
def test_18_area_drag_events_do_not_run_the_simulation(monkeypatch):
    import src.simulation.local_spread as ls
    from src.dashboard.geo_spread import apply_map_event
    monkeypatch.setattr(ls, "run_local_spread", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran")))
    s = _setup()
    assert apply_map_event({"kind": "area", "lat": BP.lat + 0.001, "lon": BP.lon, "width_m": 1200,
                            "height_m": 900}, s, None)
    assert s["width_m"] == 1200 and s["location"]["lat"] == pytest.approx(BP.lat + 0.001)
    # browser: pointer moves only update the live readout / preview; one commit after 650 ms of no movement
    assert "clearTimeout(rectTimer);rectTimer=setTimeout(commitRect,650);" in HTML
    assert "DRAG={x:(x0+x1)/2,y:(y0+y1)/2,w:x1-x0,h:y1-y0};" in HTML


# 19 ─ optional land-cover dependencies are not needed ───────────────────────
def test_19_simulation_does_not_import_the_land_cover_research_module():
    code = ("import sys, logging; logging.disable(logging.WARNING)\n"
            "import src.dashboard.geo_spread, src.dashboard.live_modes\n"
            "from src.simulation.local_spread import FocusArea, run_local_spread\n"
            "r = run_local_spread(FocusArea('t', 11.656, 76.58, 500, 500, 25), {'ffmc': 92, 'bui': 50, 'ndvi': 0.6},"
            " 5, 270, placement='Centre', duration_minutes=20)\n"
            "bad = [m for m in sys.modules if m == 'rasterio' or (m.startswith('src.landcover') and "
            "m != 'src.landcover.grid' and m != 'src.landcover')]\n"
            "print('BAD', bad)")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert "BAD []" in out.stdout, out.stdout + out.stderr


def test_ros_engine_never_burns_non_fuel_and_respects_barriers():
    from src.simulation.fuel_map import ROAD, LandCover, FUEL
    f = FocusArea("t", BP.lat, BP.lon, 500, 500, 25)
    d = initial_domain(f, 30)
    cls = np.full((d.n_rows, d.n_cols), FUEL, np.int8)
    col = d.n_cols // 2 + 3
    cls[:, col] = ROAD
    r = run_local_spread(f, DRY, 10.0, 270.0, placement="Centre", n_ignition=1, duration_minutes=30, domain=d,
                         land_cover=LandCover(cls, "osm", "test"), expand=False)
    assert r.params["model"] == "ros-ca"
    assert (r.ignition_step[:, :col] >= 0).sum() > 5 and not (r.ignition_step[:, col:] >= 0).any()
