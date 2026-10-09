"""End-to-end (Streamlit AppTest, stubbed APIs, OFFLINE land-cover FIXTURE):
production fuel path (no synthetic NDVI), LIVE never ignites risk zones
(backend + UI, audit BUG #3), LAND-COVER DATA UNAVAILABLE blocks hypothetical
runs (D6), map payload / renderer contract (asymmetric domain, spill
clipping, land-cover layer = fused mask), stale page removed (BUG #4)."""
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("streamlit.testing.v1")

from src.dashboard.dashboard_common import HIGH_RISK_NO_FIRE, NO_ACTIVE_FIRE
from src.digital_twin.twin_state import DigitalTwin, LiveIgnitionForbidden
from src.landcover import provider, worldcover
from src.simulation.fuel_map import FUEL, LandCover
from src.simulation.local_spread import FocusArea, SimulationDomain, run_local_spread
from tests.test_live_whatif_modes import (BANDIPUR, OPEN, SPREAD, _button, _detection, _md, _open_and_run,  # noqa
                                          _refresh, _whatif, live)
from tests.test_whatif_hypothetical_ignition import PTS, _hyp, _place

DASH = Path(__file__).resolve().parents[1] / "src" / "dashboard"


# Phase 2 ─ the production run uses NO land-cover acquisition, fusion or fuel mask ─
def _forbid_land_cover_pipeline(monkeypatch):
    """Any use of the Phase 1 land-cover pipeline fails the test."""
    from src.landcover import osm_vectors, sentinel2_ndvi

    def boom(*a, **k):
        raise AssertionError("Phase 1 land-cover pipeline used in the fire-simulation path")
    for mod, name in ((worldcover, "load_window"), (sentinel2_ndvi, "load_window"), (osm_vectors, "load_osm"),
                      (provider, "grid_land_cover"), (provider, "domain_land_cover")):
        monkeypatch.setattr(mod, name, boom)


def test_p2_whatif_run_needs_no_worldcover_sentinel2_or_fusion(live, monkeypatch):
    _forbid_land_cover_pipeline(monkeypatch)
    at = _hyp(live())
    _place(at, PTS)
    cfg, res = _open_and_run(at, "RUN WHAT-IF SIMULATION")
    assert not at.exception and res is not None
    assert res.land_cover is None and res.non_fuel.sum() == 0              # no fuel mask controls the CA
    assert res.params["fuel_load_source"].startswith("zone NDVI value (SYNTHETIC estimate)")
    # Phase 3 default model: rate-of-spread CA (geometric diagonal correction, FBP wind ellipse)
    assert res.params["model"] == "ros-ca" and res.params["wind_model"].startswith("FBP ellipse")
    assert not any(k.startswith(("land_cover", "ndvi", "fusion", "osm")) for k in res.timings)
    for r, c in cfg["live"]["ignition"]["cells"]:                           # fire starts at the clicked cells
        rr = r + res.domain.m_north - res.initial_domain.m_north
        cc = c + res.domain.m_west - res.initial_domain.m_west
        assert res.ignition_step[rr, cc] == 0
    prov = next(d.value for d in at.dataframe if "Item" in d.value.columns)
    row = prov[prov["Item"] == "Land cover"].iloc[0]
    assert row["Type"] == "NOT USED FOR SPREAD"
    fuel = prov[prov["Item"] == "Fuel load"].iloc[0]
    assert fuel["Type"] == "SYNTHETIC" and "simulated cell-to-cell variability" in fuel["Value"]


def test_p2_accepted_ignition_is_not_blocked_by_land_cover(live, monkeypatch):
    """Even if every cell around were mapped as water, propagation is the CA's alone."""
    from src.simulation import fuel_map
    _forbid_land_cover_pipeline(monkeypatch)
    la, lo = BANDIPUR
    lake = {"type": "way", "tags": {"natural": "water"},
            "geometry": [{"lat": la - 0.05, "lon": lo - 0.05}, {"lat": la - 0.05, "lon": lo + 0.05},
                         {"lat": la + 0.05, "lon": lo + 0.05}, {"lat": la + 0.05, "lon": lo - 0.05},
                         {"lat": la - 0.05, "lon": lo - 0.05}]}
    monkeypatch.setattr(fuel_map, "fetch_osm", lambda bbox, allow_fetch=True, timeout_s=30:
                        {"fetched_utc": "2026-10-09T00:00:00Z", "elements": [lake]})
    at = _hyp(live())
    _place(at, PTS)                                     # points placed before (the helper skips the check)
    cfg, res = _open_and_run(at, "RUN WHAT-IF SIMULATION")
    assert int((res.ignition_step > 0).sum()) > 0       # the fire spread although the area is mapped as water
    assert "LAND-COVER DATA UNAVAILABLE" not in "\n".join(e.value for e in at.error)


def test_p2_live_run_needs_no_land_cover_pipeline(live, monkeypatch):
    _forbid_land_cover_pipeline(monkeypatch)
    at = live()
    at.feed["detections"] = [_detection(*BANDIPUR)]
    _refresh(at)
    cfg, res = _open_and_run(at, "RUN LIVE SIMULATION")
    assert cfg["live"]["ignition"]["source"] == "OBSERVED_FIRMS" and res.params["ignition_cells_requested"] == 1


# 39, 40, 41 ─ LIVE: no fire without FIRMS, high risk is not fire, no force-ignite ─
def test_41_live_twin_backend_refuses_risk_ignition():
    twin = DigitalTwin(offline=False)
    assert twin.live_observed_only
    twin.current_snapshot = object()                                      # refresh() not needed for the guard
    with pytest.raises(LiveIgnitionForbidden):
        twin.simulate_spread_from_alerts()
    with pytest.raises(LiveIgnitionForbidden):
        twin.simulate_spread_from_top_n(5)
    demo = DigitalTwin(offline=True)
    assert not demo.live_observed_only


def test_39_40_41_live_applied_spread_page_has_no_risk_projection(live):
    at = live()
    _button(at, OPEN).click().run()
    at.switch_page(SPREAD).run()
    assert not at.exception
    assert at.session_state["_sim_twin"].live_observed_only                # flagged for the backend guard
    md = _md(at)
    assert NO_ACTIVE_FIRE in md                                           # 39 (and 40: HIGH RISK variant text)
    assert not [b for b in at.button if b.label.startswith("Force-ignite")]
    assert not [b for b in at.button if b.label == "Run spread simulation"]
    _button(at, "RUN LIVE SIMULATION").click().run()
    assert at.session_state["applied_geo_result"] is None                # zero FIRMS = no fire


def test_40_high_risk_message_text():
    assert HIGH_RISK_NO_FIRE == "HIGH FIRE-WEATHER RISK — NO ACTIVE FIRE DETECTED"


def test_41_current_state_in_live_mode_has_no_projection(live):
    at = live(page=SPREAD)
    at.radio(key="spread_mode").set_value("Current risk state (live/demo, from the sidebar)").run()
    assert not at.exception
    assert not [b for b in at.button if b.label == "Run spread simulation"]
    assert NO_ACTIVE_FIRE in _md(at)


def test_demo_mode_keeps_the_regional_projection_and_force_ignite(live):
    at = live(page=SPREAD, demo=True)
    at.radio(key="spread_mode").set_value("Current risk state (live/demo, from the sidebar)").run()
    assert [b for b in at.button if b.label == "Run spread simulation"]


# 43 ─ LIVE / hypothetical separation preserved on the applied page ───────────
def test_43_observed_detection_runs_live_only_from_firms(live):
    at = live()
    at.feed["detections"] = [_detection(*BANDIPUR)]
    _refresh(at)
    cfg, res = _open_and_run(at, "RUN LIVE SIMULATION")
    assert cfg["live"]["ignition"]["source"] == "OBSERVED_FIRMS"
    assert res is not None and res.params["ignition_cells_requested"] == 1
    assert "HYPOTHETICAL" not in _md(at).split("lm-status")[1][:400]


# 53 - 56 ─ renderer contract ────────────────────────────────────────────────
def _expanded_result():
    f = FocusArea(name="t", lat=BANDIPUR[0], lon=BANDIPUR[1], width_m=250, height_m=250, cell_m=25)
    world = SimulationDomain(f, 70)

    def lc(spec_or_dom):
        # unbounded fixture world: all fuel except a 2-cell barrier at world columns 95-96
        b, wb = spec_or_dom.bounds, world.bounds
        c0 = int(round((b["west"] - wb["west"]) / f.dlon))
        cols = c0 + np.arange(spec_or_dom.n_cols)
        c = np.full((spec_or_dom.n_rows, spec_or_dom.n_cols), FUEL, np.int8)
        c[:, (cols >= 95) & (cols <= 96)] = 3
        return LandCover(c.copy(), "fused", "fixture", 0, {}, status="full", fuel_load=np.where(c == FUEL, 0.9, 0.0))
    d0 = SimulationDomain(f, 4)
    hot = {"ffmc": 97.0, "ndvi": 0.8, "bui": 80.0, "risk_score": 0.9, "zone_id": 1, "fwi": 60.0}
    res = run_local_spread(f, hot, 9.0, 250.0, domain=d0, land_cover=lc(d0), land_provider=lc, n_ignition=2,
                           placement="Centre", seed=3, duration_minutes=120, base_spread_prob=1.0)
    assert res.expansions and not res.domain.symmetric
    return f, res


def _js_cell(p, lat, lon):
    """Python replica of the component's cellOfXY / toXY (index.html)."""
    d, fo = p["domain"], p["focus"]
    cell = fo["cell_m"]
    mlon = 111320 * math.cos(math.radians(fo["lat"]))
    x, y = (lon - fo["lon"]) * mlon, (lat - fo["lat"]) * 111320
    hx, hy = fo["n_cols"] * cell / 2, fo["n_rows"] * cell / 2
    dxw, dxe = hx + d["m_west"] * cell, hx + d["m_east"] * cell
    dys, dyn = hy + d["m_south"] * cell, hy + d["m_north"] * cell
    if x < -dxw or x >= dxe or y <= -dys or y > dyn:
        return -1
    return math.floor((dyn - y) / cell) * d["n_cols"] + math.floor((x + dxw) / cell)


def test_38_53_54_payload_geometry_matches_python_after_asymmetric_expansion():
    from src.dashboard.geo_fire_map import sim_payload
    f, res = _expanded_result()
    p = sim_payload(f, 120, res, 9.0, 250.0, [], "synthetic", 2, "Centre", [], {"grid": True}, False, "K", "")
    d = res.domain
    assert (p["domain"]["m_north"], p["domain"]["m_south"], p["domain"]["m_west"], p["domain"]["m_east"]) == \
        d.margin_tuple
    assert p["expansions"] and p["initialDomain"]["xw"] < p["domain"]["n_cols"] * 25
    lat_c, lon_c = d.cell_centres()
    rng = np.random.default_rng(1)
    bad = 0
    for _ in range(4000):
        r, c = int(rng.integers(0, d.n_rows)), int(rng.integers(0, d.n_cols))
        la = float(lat_c[r, c]) + float(rng.uniform(-0.45, 0.45)) * f.dlat
        lo = float(lon_c[r, c]) + float(rng.uniform(-0.45, 0.45)) * f.dlon
        bad += _js_cell(p, la, lo) != r * d.n_cols + c
    assert bad == 0
    burned = (res.ignition_step >= 0) & ~res.domain.focus_mask()
    assert burned.any()                                                    # 53: fire beyond the focus area
    assert not res.final["domain_edge_reached"]                            # 54: initial domain was no barrier


def test_55_56_effects_follow_ca_state_and_land_cover_layer_is_the_fused_mask():
    from src.dashboard.geo_fire_map import sim_payload
    f, res = _expanded_result()
    p = sim_payload(f, 120, res, 9.0, 250.0, [], "synthetic", 2, "Centre", [], {"grid": True}, False, "K", "")
    assert set(p["cells"]) == set(np.flatnonzero(res.ignition_step.ravel() >= 0).tolist())
    lc = p["landCover"]["cells"]                                            # a caller-supplied layer is shown
    assert sorted(lc["3"]) == np.flatnonzero(res.land_cover.ravel() == 3).tolist()
    html = (DASH / "components" / "fire_map" / "index.html").read_text(encoding="utf-8")
    assert "function cellOfXY(x,y)" in html and "-DXW+(c+0.5)*CELL" in html
    assert "Mapped land cover here: " in html          # final requirements: reported, never a rejection


def test_p2_renderer_fire_first_smoke_secondary():
    """Renderer: organic glow and scar (no square cell clipping), fire clusters
    from a smooth noise field. Phase 3 (issue 4) replaced Phase 2's "much
    lighter smoke" with MEDIUM-density smoke that must never hide the flames:
    smoke opacity is capped, and the fire glow, burning bed and flames are drawn
    after (on top of) the smoke."""
    import re
    html = (DASH / "components" / "fire_map" / "index.html").read_text(encoding="utf-8")
    assert "clipGround" not in html and "sctx.globalCompositeOperation='destination-in'" not in html
    caps = dict(re.findall(r"(\w+):(\d+)", re.search(r"const CAP=\{([^}]*)\}", html).group(1)))
    assert int(caps["smoke"]) <= 320 and int(caps["ember"]) <= 120 and int(caps["ash"]) <= 80
    assert int(caps["flame"]) >= 900
    assert "ctx.globalAlpha=clamp(fade*p.den*0.65,0,0.38)" in html            # semi-transparent smoke, opacity cap 0.38
    assert "core:sprite(" in html and "function patch(x,y,now)" in html
    assert "if(patch(x*1.7,y*1.7,now*3)<0.1)continue;" in html              # a broken plume, few gaps
    # final design: no thin glowing outline ("ribbon") along every perimeter edge; flames at normal zoom;
    # glow + burning bed and flames are drawn ABOVE the smoke
    assert "glowing fire line along the active front" not in html
    assert "if(show.fire&&L>=1&&fs>0.1){" in html
    i_smoke = html.index("// smoke: dense charcoal over intense fire")
    i_glow = html.index("// fire glow + fire bed (drawn ABOVE the smoke")
    i_flames = html.index("// flames: tapered tongues rising from the bed")
    assert i_smoke < i_glow < i_flames


def test_p2_smoke_and_layers_never_change_the_fire_state():
    from src.dashboard.geo_fire_map import sim_payload
    f, res = _expanded_result()
    a = sim_payload(f, 120, res, 9.0, 250.0, [], "synthetic", 2, "Centre", [], {"smoke": True}, False, "K", "")
    b = sim_payload(f, 120, res, 9.0, 250.0, [], "synthetic", 2, "Centre", [], {"smoke": False}, False, "K", "")
    assert a["cells"] == b["cells"] and a["ign"] == b["ign"] and a["out"] == b["out"]


def test_setup_payload_carries_rejection_classes_and_study_region():
    from src.dashboard.geo_fire_map import setup_payload, study_region_payload
    from src.geo.study_region import load_study_region
    f = FocusArea(name="t", lat=BANDIPUR[0], lon=BANDIPUR[1])
    p = setup_payload(f, 60, 3.0, 240.0, 3, "Map points", [], {"grid": True}, "K", "",
                      nonfuel_classes={"ROAD": [5, 6]}, study_region=study_region_payload(load_study_region()))
    assert p["nonfuelClasses"] == {"ROAD": [5, 6]}
    assert p["studyRegion"]["status"].startswith("APPROXIMATE") and p["studyRegion"]["outline"]


# BUG #4 ─ stale What-If page outside pages/ removed ──────────────────────────
def test_bug4_stale_page_removed_and_real_page_kept():
    assert not (DASH / "1_What_If_Simulator.py").exists()
    assert (DASH / "pages" / "1_What_If_Simulator.py").exists()
