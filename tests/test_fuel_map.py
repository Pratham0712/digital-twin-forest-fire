"""Fuel / non-fuel land-cover mask: OSM features are classified and
rasterised onto the CA grid, and the unchanged FireSpreadSimulator never
ignites or spreads into water, road, built-up or bare cells."""
import json
from collections import deque

import numpy as np
import pytest

from src.simulation import fuel_map
from src.simulation.cellular_automata import CellState
from src.simulation.fuel_map import (BUILT, FUEL, NON_FUEL, ROAD, WATER, LandCover, all_fuel, domain_land_cover,
                                     feature_class, rasterize)
from src.simulation.local_spread import FOCUS_AREAS, domain_for, run_local_spread

BANDIPUR = FOCUS_AREAS["Bandipur Tiger Reserve"]
COND = {"zone_id": "G-61", "zone_lat": 11.65, "zone_lon": 76.65, "risk_score": 0.3, "ffmc": 95.0,
        "bui": 60.0, "fwi": 40.0, "ndvi": 0.75, "temp_c": 40, "humidity_pct": 12,
        "wind_speed_ms": 8.0, "wind_from_deg": 270.0, "elev_m": 900, "slope_pct": 0, "active_fire_nearby": False}


def _grid(n=20):
    """A 20x20 grid of 25 m cells around Bandipur, with helpers to place features
    by fractional (col, row)."""
    f = BANDIPUR
    b = {"north": f.lat + n / 2 * f.dlat, "south": f.lat - n / 2 * f.dlat,
         "west": f.lon - n / 2 * f.dlon, "east": f.lon + n / 2 * f.dlon}
    ll = lambda c, r: {"lat": b["north"] - r * f.dlat, "lon": b["west"] + c * f.dlon}
    return b, ll


def test_feature_classification():
    assert feature_class({"natural": "water"})[0] == WATER
    assert feature_class({"waterway": "river"})[:2] == (WATER, "line")
    assert feature_class({"building": "yes"})[0] == BUILT
    assert feature_class({"highway": "primary"}) == (ROAD, "line", 10)
    assert feature_class({"highway": "secondary", "width": "12"})[2] == 12.0
    assert feature_class({"natural": "bare_rock"})[0] == NON_FUEL
    # forest tracks, footpaths, seasonal streams and forests are not barriers
    for tags in ({"highway": "track"}, {"highway": "footway"}, {"waterway": "stream"}, {"landuse": "forest"},
                 {"natural": "wood"}):
        assert feature_class(tags) is None


def test_lake_polygon_and_island():
    b, ll = _grid()
    lake = {"type": "way", "tags": {"natural": "water"},
            "geometry": [ll(2, 2), ll(10, 2), ll(10, 10), ll(2, 10), ll(2, 2)]}
    cls, used = rasterize([lake], b, 20, 20, 25)
    assert used == 1 and cls[3:10, 3:10].min() == WATER and cls[15, 15] == FUEL
    # multipolygon relation with an island (inner ring) that stays fuel
    rel = {"type": "relation", "tags": {"natural": "water"}, "members": [
        {"type": "way", "role": "outer", "geometry": [ll(2, 2), ll(18, 2), ll(18, 18)]},
        {"type": "way", "role": "outer", "geometry": [ll(18, 18), ll(2, 18), ll(2, 2)]},
        {"type": "way", "role": "inner", "geometry": [ll(8, 8), ll(12, 8), ll(12, 12), ll(8, 12), ll(8, 8)]}]}
    cls, _ = rasterize([rel], b, 20, 20, 25)
    assert cls[4, 4] == WATER and cls[10, 10] == FUEL and cls[0, 0] == FUEL


def _reachable(free, start):
    """8-neighbour flood fill (how the CA spreads) over `free` cells."""
    seen = np.zeros_like(free)
    q = deque([start])
    seen[start] = True
    while q:
        r, c = q.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < free.shape[0] and 0 <= nc < free.shape[1] and free[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
    return seen


@pytest.mark.parametrize("pts", [[(0.3, 0.2), (19.7, 19.8)], [(0.1, 19.9), (7.3, 11.1), (19.9, 0.4)],
                                 [(4.5, 0.0), (15.2, 20.0)]])
def test_narrow_road_is_a_diagonal_proof_barrier(pts):
    """A 5 m road on 25 m cells: the centre-line supercover gives a 4-connected
    line of ROAD cells, which an 8-neighbour fire cannot slip through."""
    b, ll = _grid()
    road = {"type": "way", "tags": {"highway": "unclassified"}, "geometry": [ll(*p) for p in pts]}
    cls, _ = rasterize([road], b, 20, 20, 25)
    assert (cls == ROAD).any()
    free = cls == FUEL
    corners = [(r, c) for r, c in ((0, 19), (19, 0), (0, 0), (19, 19)) if free[r, c]]
    sides = [_reachable(free, s) for s in corners]
    # at least two corners are separated by the road
    assert any(not sides[i][corners[j]] for i in range(len(corners)) for j in range(len(corners)) if i != j)


def test_river_line_width_and_priority():
    b, ll = _grid()
    river = {"type": "way", "tags": {"waterway": "river", "width": "60"}, "geometry": [ll(10, -1), ll(10, 21)]}
    road = {"type": "way", "tags": {"highway": "primary"}, "geometry": [ll(-1, 5), ll(21, 5)]}
    cls, _ = rasterize([road, river], b, 20, 20, 25)
    assert set(np.flatnonzero((cls[12] == WATER))) >= {9, 10}           # 60 m wide -> >= 2 cells
    assert cls[5, 10] == WATER                                           # water wins over road (bridge)
    assert cls[5, 2] == ROAD


def _river_land_cover(domain, col_from_centre=4, width=2):
    cls = np.zeros((domain.n_rows, domain.n_cols), dtype=np.int8)
    c0 = domain.n_cols // 2 + col_from_centre
    cls[:, c0:c0 + width] = WATER
    return LandCover(cls, "osm", "OpenStreetMap land cover (test river)", 1, {}), c0


def test_fire_stops_at_the_river():
    dur = 60
    dom = domain_for(BANDIPUR, dur)
    lc, c0 = _river_land_cover(dom)
    kw = dict(n_ignition=3, placement="Centre", seed=7, duration_minutes=dur)
    free = run_local_spread(BANDIPUR, COND, 8.0, 270.0, **kw)          # west wind pushes fire east
    blocked = run_local_spread(BANDIPUR, COND, 8.0, 270.0, land_cover=lc, **kw)
    assert (free.ignition_step[:, c0 + 2:] >= 0).any(), "without land cover the fire crosses"
    assert not (blocked.ignition_step[:, c0:] >= 0).any(), "fire must not burn the river or beyond"
    assert blocked.land_cover is not None and (blocked.land_cover == WATER).sum() == 2 * dom.n_rows
    for h in blocked.history:
        assert (h.state[:, c0:c0 + 2] == CellState.NON_FUEL).all()


def test_fuel_areas_behave_exactly_as_before():
    dur = 30
    dom = domain_for(BANDIPUR, dur)
    kw = dict(n_ignition=3, placement="Upwind edge", seed=11, duration_minutes=dur)
    a = run_local_spread(BANDIPUR, COND, 6.0, 225.0, **kw)
    b = run_local_spread(BANDIPUR, COND, 6.0, 225.0, land_cover=all_fuel((dom.n_rows, dom.n_cols)), **kw)
    assert np.array_equal(a.ignition_step, b.ignition_step) and np.array_equal(a.burnout_step, b.burnout_step)


def test_ignition_on_water_is_not_lit():
    dur = 15
    dom = domain_for(BANDIPUR, dur)
    cls = np.full((dom.n_rows, dom.n_cols), WATER, dtype=np.int8)
    lc = LandCover(cls, "osm", "OpenStreetMap land cover (all water)", 1, {})
    r = run_local_spread(BANDIPUR, COND, 6.0, 225.0, n_ignition=4, placement="Map points",
                         ignition_points=[(BANDIPUR.lat, BANDIPUR.lon)], duration_minutes=dur, land_cover=lc)
    assert not (r.ignition_step >= 0).any()
    assert r.params["ignition_cells_requested"] >= 1
    assert r.params["ignition_cells_on_non_fuel"] == r.params["ignition_cells_requested"]


def test_domain_land_cover_from_cache_and_when_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(fuel_map, "LAND_COVER_DIR", tmp_path)
    dom = domain_for(BANDIPUR, 15)
    b = dom.bounds
    lake = {"type": "way", "tags": {"natural": "water"},
            "geometry": [{"lat": b["north"], "lon": b["west"]}, {"lat": b["north"], "lon": dom.lon},
                         {"lat": dom.lat, "lon": dom.lon}, {"lat": dom.lat, "lon": b["west"]},
                         {"lat": b["north"], "lon": b["west"]}]}
    bbox = fuel_map._snapped_bbox(b)
    fuel_map._cache_path(bbox).write_text(json.dumps({"bbox": bbox, "fetched_utc": "2026-10-07T00:00:00Z",
                                                     "elements": [lake]}))
    lc = domain_land_cover(dom, allow_fetch=False)
    assert lc.available and lc.n_features == 1 and lc.classes[1, 1] == WATER and lc.classes[-2, -2] == FUEL
    assert lc.label.startswith("OpenStreetMap") and "2026-10-07" in lc.label

    other = domain_for(FOCUS_AREAS["Nagarhole National Park"], 15)        # not cached, network refused

    def refuse(*a, **k):
        raise OSError("network unreachable")
    import requests
    monkeypatch.setattr(requests, "post", refuse)
    lc2 = domain_land_cover(other, allow_fetch=True)
    assert lc2.source == "unavailable" and not lc2.non_fuel.any() and "unavailable" in lc2.label.lower()


def test_map_payload_carries_the_land_cover_layer():
    from src.dashboard.geo_fire_map import sim_payload
    dur = 15
    dom = domain_for(BANDIPUR, dur)
    lc, c0 = _river_land_cover(dom, col_from_centre=3, width=1)
    r = run_local_spread(BANDIPUR, COND, 6.0, 270.0, duration_minutes=dur, land_cover=lc)
    p = sim_payload(BANDIPUR, dur, r, 6.0, 270.0, [], "synthetic", 3, "Centre", [], {"grid": True}, False, "K", "")
    cells = p["landCover"]["cells"][str(WATER)]
    assert len(cells) == dom.n_rows and all(i % dom.n_cols == c0 for i in cells)
    assert p["landCover"]["label"].startswith("OpenStreetMap")
