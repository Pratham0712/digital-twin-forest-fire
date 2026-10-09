"""Adaptive / expanding simulation domain (audit BUG #1, requirements 13-17, 20).

A "world" land-cover grid (fixture classes + fuel) is precomputed on a large
domain; the land-cover provider of the run slices it, so the expanded strips
must receive exactly the world's data. The decisive check: a run that starts
on a small domain and grows is IDENTICAL (every step, every cell) to a run on
the final domain from the start - the state, the step counter, the wind
schedule and the random stream are preserved, nothing restarts."""
import hashlib

import numpy as np
import pytest

from config.config import DOMAIN
from src.simulation.cellular_automata import CellState, FireSpreadSimulator
from src.simulation.fuel_map import BUILT, FUEL, ROAD, UNKNOWN, WATER, LandCover
from src.simulation.local_spread import (EXTENT_LIMIT_TITLE, FocusArea, SimulationDomain, domain_for,
                                         initial_domain, run_local_spread)



@pytest.fixture(autouse=True)
def _legacy_ca(monkeypatch):
    """Phase 3: these tests use a fixed 150 x 150-cell synthetic WORLD sized for the LEGACY CA, whose fire
    moves at most one cell per step. The default "ros" model spreads at the physical ROS (far beyond that
    world under HOT weather), so this module pins the legacy engine; adaptive expansion of the "ros"
    model is tested in test_phase3_model.py (state preserved, identical to the final domain, limits)."""
    from config.config import SYSTEM
    monkeypatch.setattr(SYSTEM, "local_ca_model", "legacy")


LAT, LON = 11.6667, 76.6333
HOT = {"ffmc": 97.0, "ndvi": 0.8, "bui": 80.0, "risk_score": 0.9, "zone_id": 1, "fwi": 60.0}
F = FocusArea(name="t", lat=LAT, lon=LON, width_m=250.0, height_m=250.0, cell_m=25.0)     # 10 x 10 cells
WORLD = SimulationDomain(F, 70)                                                          # 150 x 150 cells


def _world(river_col=None, built=()):
    rng = np.random.default_rng(5)
    cls = np.full((WORLD.n_rows, WORLD.n_cols), FUEL, dtype=np.int8)
    if river_col is not None:
        cls[:, river_col:river_col + 2] = WATER
    for (r0, r1, c0, c1) in built:
        cls[r0:r1, c0:c1] = BUILT
    fuel = np.where(cls == FUEL, rng.uniform(0.5, 1.0, cls.shape), 0.0)
    return cls, fuel


def _lc(cls, fuel, label="fixture fused land cover", status="full"):
    return LandCover(cls.copy(), "fused", label, 0, {}, status=status, fuel_load=fuel.copy(),
                     confidence=np.full(cls.shape, 2, np.int8), source_mask=np.ones(cls.shape, np.uint8))


def _offset(dom_or_spec):
    b = dom_or_spec.bounds
    wb = WORLD.bounds
    return int(round((wb["north"] - b["north"]) / F.dlat)), int(round((b["west"] - wb["west"]) / F.dlon))


def provider_for(cls, fuel, calls=None):
    def land(spec):
        if calls is not None:
            calls.append((spec.n_rows, spec.n_cols))
        r0, c0 = _offset(spec)
        sl = (slice(r0, r0 + spec.n_rows), slice(c0, c0 + spec.n_cols))
        assert r0 >= 0 and c0 >= 0 and sl[0].stop <= WORLD.n_rows and sl[1].stop <= WORLD.n_cols
        return _lc(cls[sl], fuel[sl])
    return land


def _slice(dom, arr):
    r0, c0 = _offset(dom)
    return arr[r0:r0 + dom.n_rows, c0:c0 + dom.n_cols]


def _run(dom, cls, fuel, provider=True, **kw):
    args = dict(n_ignition=2, placement="Centre", seed=3, duration_minutes=120, base_spread_prob=1.0)
    args.update(kw)
    return run_local_spread(F, HOT, 9.0, 250.0, domain=dom, land_cover=_lc(_slice(dom, cls), _slice(dom, fuel)),
                            land_provider=provider_for(cls, fuel) if provider else None, **args)


# 26, 27, 25 ───────────────────────────────────────────────────────────────
def test_26_27_25_fire_expands_the_domain_and_crosses_focus_and_initial_domain():
    cls, fuel = _world()
    small = SimulationDomain(F, 4)
    res = _run(small, cls, fuel)
    assert res.expansions, "the fire must grow the domain instead of stopping at its edge"
    assert res.domain.n_rows > small.n_rows or res.domain.n_cols > small.n_cols
    assert res.initial_domain == small and res.extent_limit is None
    # burned well beyond the initial domain and the focus
    r0, c0 = res.domain.m_north - small.m_north, res.domain.m_west - small.m_west
    init_mask = np.zeros(res.ignition_step.shape, bool)
    init_mask[r0:r0 + small.n_rows, c0:c0 + small.n_cols] = True
    burned = res.ignition_step >= 0
    assert (burned & ~init_mask).sum() > 50 and (burned & ~res.domain.focus_mask()).any()
    assert not res.final["domain_edge_reached"]
    rec = res.expansions[0]
    assert {"step", "minutes", "grown_cells", "old", "new", "index"} <= set(rec)
    assert rec["new"]["rows"] * rec["new"]["cols"] > rec["old"]["rows"] * rec["old"]["cols"]
    assert res.params["n_expansions"] == len(res.expansions) and res.params["initial_domain"] == small.describe()


# 29, 30, 31 ─ identical to a fixed run on the final domain ─────────────────
@pytest.mark.parametrize("wind_from,seed", [(250.0, 3), (45.0, 8), (160.0, 1)])
def test_29_expanded_run_is_identical_to_a_run_on_the_final_domain(wind_from, seed):
    cls, fuel = _world(river_col=100, built=[(20, 30, 40, 60)])
    small = SimulationDomain(F, 4)
    grown = run_local_spread(F, HOT, 9.0, wind_from, domain=small, land_cover=_lc(_slice(small, cls),
                             _slice(small, fuel)), land_provider=provider_for(cls, fuel), n_ignition=2,
                             placement="Centre", seed=seed, duration_minutes=120, base_spread_prob=1.0)
    assert grown.expansions
    final = grown.domain
    fixed = run_local_spread(F, HOT, 9.0, wind_from, domain=final, land_cover=_lc(_slice(final, cls),
                             _slice(final, fuel)), n_ignition=2, placement="Centre", seed=seed,
                             duration_minutes=120, base_spread_prob=1.0, expand=False)
    assert np.array_equal(grown.ignition_step, fixed.ignition_step)
    assert np.array_equal(grown.burnout_step, fixed.burnout_step)
    assert len(grown.history) == len(fixed.history)
    for a, b in zip(grown.history, fixed.history):
        assert a.step == b.step and np.array_equal(a.state, b.state)


def test_30_31_no_duplicated_burning_and_no_state_corruption():
    cls, fuel = _world(river_col=95)
    res = _run(SimulationDomain(F, 4), cls, fuel)
    ign, out = res.ignition_step, res.burnout_step
    burned = out >= 0
    assert np.all(out[burned] == ign[burned] + 1)                         # burns exactly one step
    nf = res.non_fuel
    prev = None
    for h in res.history:
        assert set(np.unique(h.state)) <= {0, 1, 2, 3}
        assert (h.state[nf] == CellState.NON_FUEL).all() and not (h.state[~nf] == CellState.NON_FUEL).any()
        assert h.n_burning == int((h.state == CellState.BURNING).sum())
        assert ((h.state == CellState.BURNING) == (ign == h.step)).all()   # each cell burning in ONE frame only
        if prev is not None:
            assert not ((prev == CellState.BURNED) & (h.state != CellState.BURNED)).any()   # never un-burns
        prev = h.state


# 28 ─ new cells use the land cover of their own location ──────────────────
def test_28_expanded_cells_use_the_correct_land_cover():
    cls, fuel = _world(river_col=100)
    calls = []
    small = SimulationDomain(F, 4)
    res = run_local_spread(F, HOT, 12.0, 270.0, domain=small, land_cover=_lc(_slice(small, cls),
                           _slice(small, fuel)), land_provider=provider_for(cls, fuel, calls), n_ignition=2,
                           placement="Centre", seed=2, duration_minutes=180, base_spread_prob=1.2)
    assert calls, "expansion strips are loaded from the land-cover provider"
    assert np.array_equal(res.land_cover, _slice(res.domain, cls))
    assert np.allclose(res.fuel_load[res.land_cover == FUEL], _slice(res.domain, fuel)[res.land_cover == FUEL])
    river = res.land_cover == WATER
    assert river.any() and not (res.ignition_step[river] >= 0).any()      # the world's river still stops it
    east = np.argwhere(river)[:, 1].max()
    assert not (res.ignition_step[:, east + 1:] >= 0).any()


# limits ───────────────────────────────────────────────────────────────────
def test_extent_limit_is_reported_not_disguised_as_natural_stop(monkeypatch):
    monkeypatch.setattr(DOMAIN, "max_side_cells", 24)
    cls, fuel = _world()
    res = _run(SimulationDomain(F, 4), cls, fuel, duration_minutes=150)
    assert res.extent_limit is not None and res.extent_limit["title"] == EXTENT_LIMIT_TITLE
    assert res.params["extent_limit_reached"] and max(res.domain.n_rows, res.domain.n_cols) <= 24
    assert res.final["domain_edge_reached"]


def test_max_expansions_limit(monkeypatch):
    monkeypatch.setattr(DOMAIN, "max_expansions", 1)
    cls, fuel = _world()
    res = _run(SimulationDomain(F, 4), cls, fuel)
    assert len(res.expansions) == 1 and res.extent_limit is not None


def test_unavailable_land_cover_beyond_the_edge_stops_expansion_honestly():
    cls, fuel = _world()

    def none(spec):
        return LandCover(np.full(spec.shape, UNKNOWN, np.int8), "fused", "LAND-COVER DATA UNAVAILABLE — test", 0, {},
                         status="unavailable")
    small = SimulationDomain(F, 4)
    res = run_local_spread(F, HOT, 9.0, 250.0, domain=small, land_cover=_lc(_slice(small, cls), _slice(small, fuel)),
                           land_provider=none, n_ignition=2, placement="Centre", seed=3, duration_minutes=120,
                           base_spread_prob=1.0)
    assert res.extent_limit is not None and "land-cover data unavailable" in res.extent_limit["reason"]
    assert res.domain == small                                            # nothing invented beyond the edge


def test_no_provider_with_a_land_cover_layer_reports_the_limit():
    cls, fuel = _world()
    res = _run(SimulationDomain(F, 4), cls, fuel, provider=False)
    assert res.extent_limit is not None and "no land-cover provider" in res.extent_limit["reason"]


def test_default_full_duration_domain_never_needs_expansion():
    cls, fuel = _world()
    for dur in (15, 60, 120):
        d = domain_for(F, dur)
        res = _run(d, cls, fuel, duration_minutes=dur)
        assert res.expansions == [] and res.domain == d and not res.final["domain_edge_reached"]


# 37, 38 ─ ignition near the initial edge, map coordinates after expansion ───
def test_37_ignition_near_the_initial_edge_is_not_truncated():
    cls, fuel = _world()
    cover = domain_for(F, 60)                                             # the set-up / click coverage
    lat_c, lon_c = cover.cell_centres()
    pt = (float(lat_c[cover.n_rows // 2, 1]), float(lon_c[cover.n_rows // 2, 1]))   # one cell from the west edge
    d0 = initial_domain(F, 60, [pt])
    assert d0.m_west > cover.m_west and d0.m_west - (cover.m_west - 1) >= DOMAIN.ignition_edge_margin_cells
    res = run_local_spread(F, HOT, 9.0, 90.0, domain=d0, land_cover=_lc(_slice(d0, cls), _slice(d0, fuel)),
                           land_provider=provider_for(cls, fuel), placement="Map points", ignition_points=[pt],
                           strict_points=True, seed=1, duration_minutes=60, base_spread_prob=1.0)   # wind from E
    rc = res.domain.cell_of(*pt)
    assert res.ignition_step[rc] == 0 and not res.final["domain_edge_reached"]
    assert (res.ignition_step[:, :rc[1] - 3] >= 0).any()                  # spread further west than the old edge


def test_38_map_coordinates_stay_correct_after_expansion():
    cls, fuel = _world()
    res = _run(SimulationDomain(F, 4), cls, fuel)
    d = res.domain
    lat_c, lon_c = d.cell_centres()
    rng = np.random.default_rng(0)
    for _ in range(300):
        r, c = int(rng.integers(0, d.n_rows)), int(rng.integers(0, d.n_cols))
        la = float(lat_c[r, c]) + float(rng.uniform(-0.45, 0.45)) * F.dlat
        lo = float(lon_c[r, c]) + float(rng.uniform(-0.45, 0.45)) * F.dlon
        assert d.cell_of(la, lo) == (r, c)
    # the focus area did not move: same lat/lon bounds before and after expansion
    fs = d.focus_slice()
    assert lat_c[fs][0, 0] == pytest.approx(F.cell_centres()[0][0, 0], abs=1e-9)
    assert lon_c[fs][0, 0] == pytest.approx(F.cell_centres()[1][0, 0], abs=1e-9)


def test_seeded_runs_are_deterministic():
    cls, fuel = _world(river_col=90)
    a = _run(SimulationDomain(F, 4), cls, fuel, seed=11)
    b = _run(SimulationDomain(F, 4), cls, fuel, seed=11)
    assert np.array_equal(a.ignition_step, b.ignition_step) and a.expansions == b.expansions


# CA engine unchanged (the optimisations keep every random draw identical) ───
def test_ca_engine_matches_the_pre_change_golden_run():
    """Golden values produced by the original FireSpreadSimulator (before the
    per-step wind-factor cache, scalar clips and int8 state)."""
    rng = np.random.default_rng(123)
    n = 48
    ig = np.zeros((n, n), bool)
    ig[24, 10] = ig[30, 30] = True
    d = rng.uniform(0.5, 1, (n, n)); f = rng.uniform(0.2, 1, (n, n)); b = rng.uniform(0.4, 2, (n, n))
    e = rng.normal(900, 40, (n, n)); nf = rng.random((n, n)) < 0.08
    sched = [(float(rng.uniform(0, 15)), float(rng.uniform(0, 360))) for _ in range(30)]
    h = FireSpreadSimulator(n, n, base_spread_prob=0.8, random_state=7, cell_size_deg=25 / 111320).run(
        ignition_mask=ig, dryness_grid=d, fuel_load_grid=f, non_fuel_mask=nf, wind_speed_ms=5, wind_from_deg=200,
        horizon_minutes=30 * 15, elevation_grid=e, fuel_buildup_grid=b, wind_schedule=sched)
    digest = hashlib.md5(b"".join(x.state.astype(np.int8).tobytes() for x in h)).hexdigest()
    assert (len(h), h[-1].n_burned, digest) == (31, 1807, "5e0168fbdaa0102019786afb02adbdf5")
