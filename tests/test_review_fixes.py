"""Regression tests for issues found in the independent review of Phase 1."""
import json

import numpy as np
import pytest

from config.config import DOMAIN
from src.simulation.fuel_map import _supercover
from src.simulation.local_spread import FocusArea, domain_for, initial_domain

LAT, LON = 11.6667, 76.6333


def test_points_outside_the_coverage_never_grow_the_domain():
    f = FocusArea("t", LAT, LON, 500, 500, 5.0)
    base = initial_domain(f, 60)
    far = [(LAT + 0.045, LON)]                                           # ~5 km north: an old, stale point
    assert initial_domain(f, 60, far) == base


def test_ignition_aware_growth_respects_the_limits(monkeypatch):
    monkeypatch.setattr(DOMAIN, "max_side_cells", 60)
    f = FocusArea("t", LAT, LON, 500, 500, 25.0)
    cov = domain_for(f, 60)
    lat, lon = cov.cell_centres()
    d = initial_domain(f, 60, [(float(lat[0, 0]), float(lon[0, 0]))])     # corner point wants growth
    assert d.n_rows <= 60 and d.n_cols <= 60


def test_supercover_is_independent_of_segment_direction():
    a = _supercover(-800, 5.3, 300, 18.7, 24, 148)
    b = _supercover(300, 18.7, -800, 5.3, 24, 148)
    assert len(a) == len(b) == 149 and set(a) == set(b)


def test_click_rules_are_remapped_to_the_grown_domain():
    from src.dashboard.geo_spread import _remap_click_rules
    f = FocusArea("t", LAT, LON, 250, 250, 25.0)
    src = domain_for(f, 15)
    dst = src.grown(north=3, west=5, east=1)
    i = 2 * src.n_cols + 4
    out = _remap_click_rules({"nonfuel_cells": [i], "nonfuel_classes": {"ROAD": [i]}}, src, dst)
    assert out["nonfuel_cells"] == [(2 + 3) * dst.n_cols + (4 + 5)] == out["nonfuel_classes"]["ROAD"]


def test_mostly_unknown_cell_is_not_confirmed_fuel(monkeypatch):
    from src.landcover import osm_vectors, provider, sentinel2_ndvi, worldcover
    from src.landcover.grid import GridSpec
    from tests.landcover_fixtures import paint, unavailable_loader, wc_loader
    n, cell = 10, 25.0
    dlat = cell / 111320.0
    dlon = cell / (111320.0 * np.cos(np.radians(LAT)))
    spec = GridSpec(LAT + n / 2 * dlat, LAT - n / 2 * dlat, LON - n / 2 * dlon, LON + n / 2 * dlon, n, n, cell)

    def holes(w):                                         # WorldCover "no data" over the east half of the grid
        paint(w, spec.south - 1, spec.north + 1, LON + 0.2 * dlon, LON + 9 * dlon, 0)
    monkeypatch.setattr(worldcover, "load_window", wc_loader(10, holes))
    monkeypatch.setattr(sentinel2_ndvi, "load_window", unavailable_loader())
    monkeypatch.setattr(osm_vectors, "load_osm", lambda s, allow_fetch=True: (None, {"status": "unavailable"}))
    provider.clear_memory_cache()
    lc = provider.grid_land_cover(spec, allow_fetch=False)
    unk = lc.fractions["unknown"]
    assert ((unk >= 0.5) <= (lc.classes == 5)).all()                     # mostly-unknown -> UNKNOWN, never FUEL
    assert (lc.classes[:, :4] == 0).all()                                 # the WorldCover-covered half is fuel


def test_osm_tiles_are_merged_and_an_outage_is_not_retried_per_strip(monkeypatch, tmp_path):
    from src.landcover import osm_vectors
    from src.landcover.grid import GridSpec
    from src.simulation import fuel_map
    monkeypatch.setattr(fuel_map, "LAND_COVER_DIR", tmp_path)
    monkeypatch.setenv("FIRE_LAND_COVER_FETCH", "1")
    osm_vectors._LAST_FAILURE[0] = 0.0
    calls = []

    def fake_fetch(bbox, allow_fetch=True):
        calls.append((bbox, allow_fetch))
        if bbox[0] >= 11.66:
            return {"fetched_utc": "2026-10-01T00:00:00Z",
                    "elements": [{"type": "way", "id": 7, "tags": {"building": "yes"}, "geometry": []}]}
        return None                                                      # southern tiles: Overpass outage
    monkeypatch.setattr(fuel_map, "fetch_osm", fake_fetch)
    spec = GridSpec(11.675, 11.645, 76.615, 76.655, 120, 160, 25.0)      # spans 2 x 3 tiles of 0.02 deg
    data, st = osm_vectors.load_osm(spec, allow_fetch=True)
    assert st["status"] == "partial" and st["missing"] >= 1 and len(data["elements"]) == 1   # merged, deduped
    tried = [a for _, a in calls]
    assert tried.count(True) == sum(1 for b, a in calls if a and b[0] >= 11.66) + 1    # one failed try, then none
    calls.clear()
    osm_vectors.load_osm(spec, allow_fetch=True)
    assert all(not a for _, a in calls)                                   # still backing off: cache only


def test_renderer_hatching_follows_the_clock_and_main_skips_risk_ignition_in_live():
    html = open("src/dashboard/components/fire_map/index.html", encoding="utf-8").read()
    assert "Math.floor(simT)!==outbStep" in html
    src = open("main.py", encoding="utf-8").read()
    assert 'getattr(twin, "live_observed_only", False)' in src
