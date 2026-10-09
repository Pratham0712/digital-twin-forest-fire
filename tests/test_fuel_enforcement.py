"""Fuel-mask enforcement: NON_FUEL -> NEVER BURN; BURNABLE -> the CA may burn.

The authoritative mask is the existing OpenStreetMap land cover (fuel_map):
water, roads/rail, buildings / built-up land use and bare ground are non-fuel.
Google Satellite is only the base map; nothing here classifies imagery.
Probabilities are pushed to the CA's maximum so any leak would show up."""
import numpy as np
import pytest

from src.simulation.cellular_automata import CellState, FireSpreadSimulator
from src.simulation.fuel_map import BUILT, FUEL, NON_FUEL, ROAD, WATER, LandCover, rasterize
from src.simulation.local_spread import FocusArea, domain_for, run_local_spread

LAT, LON = 11.6667, 76.6333
HOT = {"ffmc": 99.0, "ndvi": 0.9, "bui": 200.0, "risk_score": 0.99, "zone_id": 1, "fwi": 80.0}   # max spread


def _focus(size=500.0, cell=25.0):
    return FocusArea(name="t", lat=LAT, lon=LON, width_m=size, height_m=size, cell_m=cell)


def _land(dom, cells=(), cls=WATER):
    c = np.full((dom.n_rows, dom.n_cols), FUEL, dtype=np.int8)
    for rc in cells:
        c[rc] = cls
    return LandCover(c, "osm", "OpenStreetMap land cover (test)")


def _run(f, land, points=None, placement="Map points", n=0, wind=(20.0, 270.0), seed=0, dur=60):
    return run_local_spread(f, HOT, wind[0], wind[1], n_ignition=n, placement=placement, seed=seed,
                            duration_minutes=dur, ignition_points=points or [], land_cover=land,
                            strict_points=placement == "Map points", base_spread_prob=1.0)


def _ll(dom, r, c):
    lat, lon = dom.cell_centres()
    return [float(lat[r, c]), float(lon[r, c])]


# 1 - 6 ───────────────────────────────────────────────────────────────────
def test_01_burnable_cell_can_ignite():
    f = _focus(); dom = domain_for(f, 60)
    r0, c0 = dom.n_rows // 2, dom.n_cols // 2
    res = _run(f, _land(dom), [_ll(dom, r0, c0)])
    assert res.ignition_step[r0, c0] == 0 and res.params["ignition_cells_on_non_fuel"] == 0


@pytest.mark.parametrize("cls", [NON_FUEL, BUILT, ROAD, WATER],
                         ids=["02_non_fuel", "03_building", "04_road", "05_water_06_bare_is_02"])
def test_02_to_06_non_burnable_cell_cannot_ignite(cls):
    f = _focus(); dom = domain_for(f, 60)
    r0, c0 = dom.n_rows // 2, dom.n_cols // 2
    res = _run(f, _land(dom, [(r0, c0)], cls), [_ll(dom, r0, c0)])
    assert res.ignition_step[r0, c0] == -1
    assert res.params["ignition_cells_on_non_fuel"] == 1 and res.params["n_ignition"] == 0
    assert res.final["burned"] == 0 and res.final["burning"] == 0                  # no fire anywhere


# 7 - 9 ───────────────────────────────────────────────────────────────────
def _sim(n=9, seed=0):
    return FireSpreadSimulator(n, n, minutes_per_step=15, base_spread_prob=1.0, random_state=seed)


def test_07_fire_cannot_enter_a_non_burnable_neighbour():
    n = 9
    ign = np.zeros((n, n), bool); ign[4, 4] = True
    nf = np.zeros((n, n), bool); nf[3:6, 3:6] = True; nf[4, 4] = False           # ring of non-fuel around it
    for seed in range(20):
        hist = _sim(n, seed).run(ign, np.ones((n, n)), np.ones((n, n)), nf, 20.0, 270.0, horizon_minutes=15 * 30,
                                 fuel_buildup_grid=np.full((n, n), 2.0))
        burnt = (hist[-1].state == CellState.BURNED) | (hist[-1].state == CellState.BURNING)
        assert burnt.sum() == 1 and burnt[4, 4]


def test_08_fire_can_enter_a_burnable_neighbour():
    n = 9
    ign = np.zeros((n, n), bool); ign[4, 4] = True
    hist = _sim(n).run(ign, np.ones((n, n)), np.ones((n, n)), np.zeros((n, n), bool), 20.0, 270.0,
                       horizon_minutes=15 * 3, fuel_buildup_grid=np.full((n, n), 2.0))
    assert ((hist[-1].state == CellState.BURNED) | (hist[-1].state == CellState.BURNING)).sum() > 1


def test_09_non_fuel_stays_non_fuel_for_the_whole_run():
    n = 15
    rng = np.random.default_rng(7)
    nf = rng.random((n, n)) < 0.3
    ign = ~nf & (rng.random((n, n)) < 0.1)
    ign[nf] = False
    ign_with_nf = ign | nf                                                          # even if asked to ignite them
    for seed in range(10):
        hist = _sim(n, seed).run(ign_with_nf, np.ones((n, n)), np.ones((n, n)), nf, 20.0, 45.0,
                                 horizon_minutes=15 * 60, fuel_buildup_grid=np.full((n, n), 2.0))
        for h in hist:
            assert (h.state[nf] == CellState.NON_FUEL).all()


# 10 ──────────────────────────────────────────────────────────────────────
def test_10_fire_cannot_cross_a_continuous_barrier():
    f = _focus(); dom = domain_for(f, 60)
    col = dom.n_cols // 2
    barrier = [(r, col) for r in range(dom.n_rows)]                                 # straight road, N-S
    land = _land(dom, barrier, ROAD)
    for seed in range(10):
        # 6 cells upwind of the road (was 3): with flame residence the wind-driven head is narrow near its
        # ignition, so it needs a few cells to show that the fire does spread on its side of the road
        res = _run(f, land, [_ll(dom, dom.n_rows // 2, col - 6)], wind=(25.0, 270.0), seed=seed)
        assert res.final["burned"] + res.final["burning"] > 5                     # fire does spread on its side
        assert (res.ignition_step[:, col:] < 0).all()                             # never on or across the road


def test_10b_osm_road_and_building_from_overpass_geometry_are_barriers():
    """End to end: Overpass-style features -> rasterize -> CA. A diagonal
    tertiary road (supercover, 4-connected) and a building polygon."""
    f = _focus(); dom = domain_for(f, 60)
    b = dom.bounds
    road = {"type": "way", "tags": {"highway": "tertiary"},
            "geometry": [{"lat": b["north"] + 0.001, "lon": b["west"] - 0.001},
                         {"lat": b["south"] - 0.001, "lon": b["east"] + 0.001}]}       # NW -> SE diagonal
    blat, blon = LAT + 0.0016, LON - 0.0016                                          # building NW of centre
    d = 0.00012
    building = {"type": "way", "tags": {"building": "yes"},
                "geometry": [{"lat": blat - d, "lon": blon - d}, {"lat": blat - d, "lon": blon + d},
                             {"lat": blat + d, "lon": blon + d}, {"lat": blat + d, "lon": blon - d},
                             {"lat": blat - d, "lon": blon - d}]}
    classes, used = rasterize([road, building], b, dom.n_rows, dom.n_cols, dom.cell_m)
    assert used == 2 and (classes == ROAD).sum() > dom.n_rows and (classes == BUILT).sum() >= 1
    land = LandCover(classes, "osm", "OpenStreetMap land cover (test)")
    rr, cc = np.mgrid[0:dom.n_rows, 0:dom.n_cols]
    below = rr - cc > 2                                                             # SW side of the NW->SE road
    above = cc - rr > 2
    start = tuple(np.argwhere(below & (classes == FUEL))[len(np.argwhere(below)) // 2])
    for seed in range(10):
        res = _run(f, land, [_ll(dom, *start)], wind=(25.0, 225.0), seed=seed)     # pushed NE, into the road
        assert res.final["burned"] + res.final["burning"] > 5
        assert (res.ignition_step[above] < 0).all()                                 # did not cross diagonally
        assert (res.ignition_step[classes != FUEL] < 0).all()                       # road / building never burn
    # building cell: an ignition there is rejected, a fire around it never enters it
    br, bc = np.argwhere(classes == BUILT)[0]
    fr, fc = next((br + dr, bc + dc) for dr in (-2, -1, 1, 2) for dc in (-2, -1, 1, 2)
                  if classes[br + dr, bc + dc] == FUEL)
    res = _run(f, land, [_ll(dom, br, bc), _ll(dom, fr, fc)], wind=(25.0, 90.0))
    assert res.ignition_step[br, bc] == -1 and res.params["ignition_cells_on_non_fuel"] == 1
    assert res.ignition_step[fr, fc] == 0
    assert (res.ignition_step[classes == BUILT] < 0).all()


def test_mismatched_mask_is_never_silently_ignored():
    f = _focus()
    with pytest.raises(ValueError, match="does not match the simulation domain"):
        _run(f, _land(domain_for(f, 60)), [[LAT, LON]], dur=15)


def test_invariant_guard_rejects_a_non_fuel_burn(monkeypatch):
    """If the engine ever produced fire on a non-fuel cell, the run fails loudly."""
    from config.config import SYSTEM
    monkeypatch.setattr(SYSTEM, "local_ca_model", "legacy")          # this leak is injected into the legacy engine
    f = _focus(); dom = domain_for(f, 60)
    r0, c0 = dom.n_rows // 2, dom.n_cols // 2
    land = _land(dom, [(r0, c0 + 1)], BUILT)
    orig = FireSpreadSimulator.run

    def leaky(self, ignition_mask, *a, **k):
        hist = orig(self, ignition_mask, *a, **k)
        hist[-1].state[r0, c0 + 1] = CellState.BURNING                             # simulated engine bug
        return hist
    monkeypatch.setattr(FireSpreadSimulator, "run", leaky)
    with pytest.raises(RuntimeError, match="fuel-mask violation"):
        _run(f, land, [_ll(dom, r0, c0)])


# 11 - 13 ─────────────────────────────────────────────────────────────────
def _setup(n=3):
    return {"location": {"name": "t", "lat": LAT, "lon": LON, "source": "preset"}, "width_m": 500.0,
            "height_m": 500.0, "cell_m": 25.0, "duration_min": 60.0, "placement": "Map points",
            "n_ignition": n, "ignition_points": [], "ignition_source": "hypothetical",
            "layers": {"grid": True, "boundary": True}}


def test_11_12_13_map_clicks_against_the_fuel_mask():
    from src.dashboard.geo_spread import HYP_NON_FUEL, apply_map_event, focus_from_setup
    s = _setup()
    dom = domain_for(focus_from_setup(s), 60)
    r0, c0 = dom.n_rows // 2, dom.n_cols // 2
    land = _land(dom, [(r0, c0)], BUILT)
    # FINAL requirements (Priority 3C): a HYPOTHETICAL ignition is never rejected because of land cover; the
    # mapped surface is reported and labelled as known land cover instead.
    msgs, notes = [], []
    pb = _ll(dom, r0, c0)
    apply_map_event({"kind": "ignite", "points": [pb]}, s, None, whatif=True, land=land, messages=msgs, notes=notes)
    assert s["ignition_points"] == [pb] and msgs == []                               # accepted, exact point
    assert notes[0].startswith("HYPOTHETICAL IGNITION ACCEPTED ON MAPPED BUILT-UP AREA")
    assert s["ignition_checks"][0]["land_cover"] == "known (mapped)"
    p = [x + 1e-6 for x in _ll(dom, r0, c0 + 2)]
    msgs = []
    apply_map_event({"kind": "ignite", "points": [p]}, s, None, whatif=True, land=land, messages=msgs)
    assert s["ignition_points"] == [p] and msgs == []                               # 11: exactly one, not moved
    apply_map_event({"kind": "ignite", "points": []}, s, None, whatif=True, land=land)
    assert s["ignition_points"] == [] and s["placement"] == "Map points"            # 13: clear still works
    html = open("src/dashboard/components/fire_map/index.html", encoding="utf-8").read()
    assert "land cover never blocks a HYPOTHETICAL ignition" in html


# 14 - 15 ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("placement", ["Upwind edge", "Centre", "Downwind edge"])
def test_14_placement_strategies_still_work_and_skip_non_fuel(placement):
    f = _focus(); dom = domain_for(f, 15)
    free = run_local_spread(f, HOT, 5.0, 270.0, n_ignition=5, placement=placement, duration_minutes=15,
                            land_cover=_land(dom), base_spread_prob=1.0)
    cells = [tuple(x) for x in np.argwhere(free.ignition_step == 0)]
    assert len(cells) == 5
    blocked = run_local_spread(f, HOT, 5.0, 270.0, n_ignition=5, placement=placement, duration_minutes=15,
                               land_cover=_land(dom, cells[:2], WATER), base_spread_prob=1.0)
    assert blocked.params["ignition_cells_requested"] == 5 and blocked.params["ignition_cells_on_non_fuel"] == 2
    assert all(blocked.ignition_step[rc] == -1 for rc in cells[:2])


def test_15_firms_detection_on_non_fuel_is_recorded_not_ignited_or_moved():
    from src.dashboard import live_modes as lm
    from src.simulation.observed_ignition import classify_detections
    f = _focus(); dom = domain_for(f, 60)
    r0, c0 = dom.n_rows // 2, dom.n_cols // 2
    land = _land(dom, [(r0, c0)], WATER)
    det = [{"lat": _ll(dom, r0, c0)[0], "lon": _ll(dom, r0, c0)[1], "sat": "N"}]
    cls = classify_detections(dom, det, land)
    assert cls["n_total"] == 1 and cls["n_non_fuel"] == 1 and cls["n_valid"] == 0
    assert cls["detections"][0]["status"] == "non_fuel" and "water" in cls["detections"][0]["reason"]
    assert cls["detections"][0]["lat"] == det[0]["lat"]                             # the observation is kept as is
    assert lm.ignition_spec(lm.LIVE, {}, cls)["source"] == "NONE"                   # no simulated fire from it
    assert any("non-fuel" in n for n in lm.assess(lm.LIVE, "live", cls, "LOW", lm.ignition_spec(lm.LIVE, {}, cls))["notes"])
