"""WHAT-IF hypothetical ignition: ignition-cell count, placement strategies,
Set ignition on map / Clear points, domain + fuel validation, hand-off and run.
LIVE REAL-WORLD must keep using only NASA FIRMS observations.

The map component's click events are applied exactly as the page applies them
(geo_spread.apply_map_event); APIs are stubbed and the network is blocked."""
import numpy as np
import pytest

pytest.importorskip("streamlit.testing.v1")
from pathlib import Path

from src.dashboard import live_modes as lm
from src.dashboard.geo_fire_map import setup_payload
from src.dashboard.geo_spread import (HYP_NON_FUEL, HYP_OUTSIDE, _map_ignition_args, apply_map_event,
                                      focus_from_setup, validate_hypothetical_points)
from src.simulation.fuel_map import FUEL, WATER, LandCover
from src.simulation.local_spread import domain_for
from tests.test_live_whatif_modes import (BANDIPUR, OPEN, SPREAD, _button, _detection, _md, _refresh, _whatif,
                                          live)  # noqa: F401  (live = shared fixture)

DASH = Path(__file__).resolve().parents[1] / "src" / "dashboard"
D = 0.0009                                   # ~100 m: 4 cells at 25 m, well inside a 500 m box
PTS = [[BANDIPUR[0] + D, BANDIPUR[1]], [BANDIPUR[0] - D, BANDIPUR[1]], [BANDIPUR[0], BANDIPUR[1] + D]]


def _setup(n=3, placement="Map points", points=None, source="hypothetical"):
    return {"location": {"name": "Bandipur", "lat": BANDIPUR[0], "lon": BANDIPUR[1], "source": "preset"},
            "width_m": 500.0, "height_m": 500.0, "cell_m": 25.0, "duration_min": 60.0, "placement": placement,
            "n_ignition": n, "ignition_points": points or [], "ignition_source": source,
            "layers": {"grid": True, "boundary": True}}


def _land(setup, water_at=None):
    dom = domain_for(focus_from_setup(setup), setup["duration_min"])
    cls = np.full((dom.n_rows, dom.n_cols), FUEL, dtype=np.int8)
    if water_at:
        cls[dom.cell_of(*water_at)] = WATER
    return LandCover(cls, "osm", "OpenStreetMap land cover (test)")


def _click(setup, points, land=None):
    msgs = []
    assert apply_map_event({"kind": "ignite", "points": points}, setup, None, whatif=True, land=land, messages=msgs)
    return msgs


def _hyp(at, placement="Map points", n=3):
    """WHAT-IF mode, ignition source HYPOTHETICAL, placement and count set through the widgets."""
    _whatif(at)
    at.selectbox(key="wi_setup_igsrc").set_value("hypothetical").run()
    at.selectbox(key="wi_setup_place").set_value(placement).run()
    at.slider(key="wi_setup_nign").set_value(n).run()
    assert not at.exception
    return at


def _place(at, points):
    """Apply the map's ignite event to the page's set-up, as the component does, then rerun."""
    setup = at.session_state["sim_setup"]
    msgs = _click(setup, points, None)              # test land cover is all fuel (no OSM in tests)
    at.session_state["sim_setup"] = setup
    at.session_state["_wi_setup_sync"] = True
    at.run()
    assert not at.exception
    return msgs


def _status(at) -> str:
    return "\n".join(m.value for m in at.markdown if "HYPOTHETICAL IGNITIONS" in m.value)


# 1 ─────────────────────────────────────────────────────────────────────────
def test_01_whatif_none_has_no_ignition_points(live):
    at = _whatif(live())
    assert at.selectbox(key="wi_setup_igsrc").value == "none"
    assert at.selectbox(key="wi_setup_place").disabled and at.slider(key="wi_setup_nign").disabled
    ign = lm.ignition_spec(lm.WHATIF, at.session_state["sim_setup"], None)
    assert ign["source"] == "NONE" and ign["points"] == []
    assert _map_ignition_args(ign)["ignition_kind"] == "none"            # map tools hidden
    assert "WHAT-IF — NO IGNITION SOURCE" in _md(at)


# 2 ─────────────────────────────────────────────────────────────────────────
def test_02_hypothetical_enables_the_controls(live):
    at = _hyp(live())
    assert not at.selectbox(key="wi_setup_place").disabled
    assert not at.slider(key="wi_setup_nign").disabled                   # count enabled, also for Map points
    assert at.selectbox(key="wi_setup_place").value == "Map points"
    assert "0 / 3 SELECTED" in _status(at)
    args = _map_ignition_args(lm.ignition_spec(lm.WHATIF, at.session_state["sim_setup"], None))
    assert args["ignition_kind"] == "hypothetical" and args["max_picks"] == 3     # Set ignition on map / Clear shown
    html = (DASH / "components" / "fire_map" / "index.html").read_text(encoding="utf-8")
    assert "$('tools').classList.toggle('hide',igk!=='hypothetical')" in html
    assert 'id="bPick"' in html and 'id="bClear"' in html


# 3, 4, 5 ───────────────────────────────────────────────────────────────────
def test_03_count_three_accepts_exactly_three_points():
    s = _setup(n=3)
    assert _click(s, PTS) == []
    assert s["ignition_points"] == PTS and s["placement"] == "Map points"
    assert lm.ignition_spec(lm.WHATIF, s, None)["n_selected"] == 3


def test_04_cannot_select_more_than_requested():
    s = _setup(n=3)
    extra = PTS + [[BANDIPUR[0] + D, BANDIPUR[1] + D]]
    msgs = _click(s, extra)
    assert len(s["ignition_points"]) == 3 and s["ignition_points"] == PTS
    assert msgs == ["Maximum of 3 hypothetical ignition points reached."]
    html = (DASH / "components" / "fire_map" / "index.html").read_text(encoding="utf-8")
    assert "'Maximum of '+P.maxPicks+' hypothetical ignition points reached.'" in html


def test_05_point_inside_the_domain_is_kept_exactly():
    s = _setup(n=2)
    p = [BANDIPUR[0] + 0.00031, BANDIPUR[1] - 0.00047]
    assert _click(s, [p]) == []
    assert s["ignition_points"] == [p]                                     # not moved / snapped / randomised
    dom = domain_for(focus_from_setup(s), 60)
    r, c = dom.cell_of(*p)
    ign = lm.plan_ignition(lm.WHATIF, s, focus_from_setup(s), {"ndvi": 0.6, "ffmc": 88.0, "bui": 40.0}, [], "live",
                           allow_fetch=False)["ign"]
    assert ign["cells"] == [[r, c]]


# 6, 7 ──────────────────────────────────────────────────────────────────────
def test_06_point_outside_the_domain_is_rejected():
    s = _setup(n=3)
    msgs = _click(s, [[BANDIPUR[0] + 0.05, BANDIPUR[1]]])                  # ~5.5 km north
    assert s["ignition_points"] == [] and msgs == [HYP_OUTSIDE]
    assert HYP_OUTSIDE == "Select an ignition point inside the simulation area."
    html = (DASH / "components" / "fire_map" / "index.html").read_text(encoding="utf-8")
    assert "toast('Select an ignition point inside the simulation area.')" in html


def test_07_non_fuel_point_is_rejected():
    s = _setup(n=3)
    land = _land(s, water_at=PTS[0])
    msgs = _click(s, PTS, land)
    assert s["ignition_points"] == PTS[1:] and msgs == [HYP_NON_FUEL]
    assert HYP_NON_FUEL == "Selected location is non-burnable. Choose a fuel cell."
    f = focus_from_setup(s)
    p = setup_payload(f, 60, 3, 240, 3, "Map points", s["ignition_points"], s["layers"], "K", "",
                      **_map_ignition_args(lm.ignition_spec(lm.WHATIF, s, None), s, land))
    dom = domain_for(f, 60)
    r, c = dom.cell_of(*PTS[0])
    assert p["nonfuelCells"] == [r * dom.n_cols + c] and p["maxPicks"] == 3     # the map rejects it too


# 8 ─────────────────────────────────────────────────────────────────────────
def test_08_clear_points_removes_them_and_keeps_map_points():
    s = _setup(n=3, points=list(PTS))
    _click(s, [])                                                          # the map's Clear points
    assert s["ignition_points"] == [] and s["placement"] == "Map points" and s["n_ignition"] == 3
    ign = lm.ignition_spec(lm.WHATIF, s, None)
    assert ign["source"] == "NONE" and ign["awaiting_points"] and ign["n_requested"] == 3
    assert lm.hypothetical_status({**ign}) == "HYPOTHETICAL IGNITIONS · 0 / 3 SELECTED"
    assert _map_ignition_args(ign)["ignition_kind"] == "hypothetical"   # tools stay available to place again
    assert _click(s, PTS[:2]) == [] and len(s["ignition_points"]) == 2


# 9 ─────────────────────────────────────────────────────────────────────────
def test_09_points_survive_reruns_parameter_changes_and_navigation(live):
    at = _hyp(live())
    _place(at, PTS)
    assert "3 / 3 SELECTED" in _status(at)
    at.run()
    next(s for s in at.selectbox if s.label == "Simulation duration").select("2 h").run()
    next(s for s in at.slider if s.label.startswith("Temperature")).set_value(35.0).run()
    assert at.session_state["sim_setup"]["ignition_points"] == PTS
    at.switch_page("app.py").run()
    at.switch_page("pages/1_What_If_Simulator.py").run()
    assert not at.exception
    assert at.session_state["sim_setup"]["ignition_points"] == PTS and "3 / 3 SELECTED" in _status(at)


def test_09b_switching_placement_drops_stale_points(live):
    at = _hyp(live())
    _place(at, PTS)
    at.selectbox(key="wi_setup_place").set_value("Upwind edge").run()
    assert at.session_state["sim_setup"]["ignition_points"] == []
    assert "3 cell(s) · Upwind edge" in _status(at)
    at.selectbox(key="wi_setup_place").set_value("Map points").run()
    assert at.session_state["sim_setup"]["ignition_points"] == [] and "0 / 3 SELECTED" in _status(at)


# 10, 11, 12 ────────────────────────────────────────────────────────────────
def test_10_11_12_handoff_label_and_exact_cells(live):
    at = _hyp(live())
    _place(at, PTS)
    _button(at, OPEN).click().run()
    assert not at.exception
    cfg = at.session_state["simulation_config"]
    ign = cfg["live"]["ignition"]
    assert ign["source"] == "HYPOTHETICAL_USER" and ign["kind"] == "HYPOTHETICAL"
    assert ign["points"] == PTS and ign["n_selected"] == 3 and ign["n_requested"] == 3 and len(ign["cells"]) == 3
    assert cfg["setup"]["ignition_points"] == PTS and cfg["setup"]["placement"] == "Map points"
    assert cfg["live"]["baseline"]["values"]["temp_c"] == 22.73 and cfg["live"]["firms"]["status"] == "live"
    assert (cfg["width_m"], cfg["height_m"], cfg["cell_m"], cfg["duration_min"]) == (500, 500, 25, 60)
    assert cfg["region"] and cfg["setup"]["layers"] == {"grid": True, "boundary": True}
    at.switch_page(SPREAD).run()
    _button(at, "RUN WHAT-IF SIMULATION").click().run()
    assert not at.exception
    res = at.session_state["applied_geo_result"]
    assert res.params["ignition_cells_requested"] == 3                     # exactly the three cells, no extras
    for r, c in ign["cells"]:
        assert res.ignition_step[r, c] == 0
    assert int((res.ignition_step == 0).sum()) == 3
    md = _md(at)
    assert "SOURCE: USER HYPOTHETICAL IGNITION" in md
    assert "SOURCE: NASA FIRMS OBSERVED" not in md and "OBSERVED FIRE" not in md


def test_predefined_placement_uses_the_existing_rule(live):
    at = _hyp(live(), placement="Upwind edge", n=4)
    _button(at, OPEN).click().run()
    at.switch_page(SPREAD).run()
    _button(at, "RUN WHAT-IF SIMULATION").click().run()
    res = at.session_state["applied_geo_result"]
    assert res.params["placement"] == "Upwind edge" and res.params["ignition_cells_requested"] == 4


# 13, 14, 16 ────────────────────────────────────────────────────────────────
def test_13_live_mode_exposes_no_hypothetical_ignition(live):
    at = live()
    assert not [s for s in at.selectbox if s.key == "wi_setup_igsrc"]
    assert at.selectbox(key="wi_setup_place").disabled and at.slider(key="wi_setup_nign").disabled
    setup = dict(at.session_state["sim_setup"], placement="Map points", ignition_points=PTS,
                 ignition_source="hypothetical")
    ign = lm.ignition_spec(lm.LIVE, setup, None)                            # stale WHAT-IF points are ignored
    assert ign["source"] == "NONE" and "max_picks" not in _map_ignition_args(ign)


def test_14_16_live_mode_unchanged_and_never_synthetic(live):
    at = _hyp(live())
    _place(at, PTS)                                                         # hypothetical points from WHAT-IF ...
    at.button(key="wl_mode_live").click().run()                             # ... then switch to LIVE
    assert not at.exception
    _button(at, OPEN).click().run()
    at.switch_page(SPREAD).run()
    _button(at, "RUN LIVE SIMULATION").click().run()
    cfg = at.session_state["simulation_config"]
    assert cfg["live"]["ignition"]["source"] == "NONE" and at.session_state["applied_geo_result"] is None
    assert "NO ACTIVE FIRE DETECTED" in _md(at) and "HYPOTHETICAL" not in _md(at).split("lm-status")[1][:400]


# 15 ────────────────────────────────────────────────────────────────────────
def test_15_observed_source_uses_only_firms_detections(live):
    at = _hyp(live())
    _place(at, PTS)
    at.feed["detections"] = [_detection(*BANDIPUR)]
    _refresh(at)
    at.selectbox(key="wi_setup_igsrc").set_value("observed").run()
    _button(at, OPEN).click().run()
    ign = at.session_state["simulation_config"]["live"]["ignition"]
    assert ign["source"] == "OBSERVED_FIRMS" and ign["points"] == [list(BANDIPUR)]
    at.switch_page(SPREAD).run()
    _button(at, "RUN WHAT-IF SIMULATION").click().run()
    res = at.session_state["applied_geo_result"]
    assert res.params["ignition_cells_requested"] == 1                     # the FIRMS cell only, not the 3 points
    assert "SOURCE: NASA FIRMS OBSERVED" in _md(at)


# 17 ────────────────────────────────────────────────────────────────────────
def test_17_no_ignition_source_produces_no_fire(live):
    at = _hyp(live())
    _place(at, PTS)
    at.selectbox(key="wi_setup_igsrc").set_value("none").run()
    _button(at, OPEN).click().run()
    at.switch_page(SPREAD).run()
    _button(at, "RUN WHAT-IF SIMULATION").click().run()
    assert at.session_state["applied_geo_result"] is None
    assert "WHAT-IF — NO IGNITION SOURCE" in _md(at)


def test_17b_map_points_with_none_placed_produce_no_fire(live):
    at = _hyp(live())                                                       # Map points, 0 / 3 selected
    _button(at, OPEN).click().run()
    at.switch_page(SPREAD).run()
    _button(at, "RUN WHAT-IF SIMULATION").click().run()
    assert at.session_state["applied_geo_result"] is None                  # no centre fallback cell
    assert "0 / 3 selected" in _md(at)
