"""Phase 2 - LIGHTWEIGHT hypothetical ignition-point check (Concept A).

The clicked POINT is checked against mapped OpenStreetMap features (Overpass
format, built here - no network): road / building / built-up area / water /
bare ground -> rejected; mapped vegetation -> accepted; anything else ->
LOCATION UNVERIFIED (accepted with a warning, never called vegetation). The
check only decides whether a point may ignite; it is never a fire mask."""
import math

import numpy as np
import pytest

from src.dashboard import live_modes as lm
from src.dashboard.geo_spread import (HYP_NON_FUEL, _map_ignition_args, apply_map_event, focus_from_setup,
                                      rejection_message, validate_hypothetical_points)
from src.simulation.fuel_map import BUILT, FUEL, NON_FUEL, ROAD, UNKNOWN, WATER, LandCover
from src.simulation.ignition_site import IgnitionSiteClassifier
from src.simulation.local_spread import initial_domain

LAT, LON = 11.6667, 76.6333
M_LAT = 1 / 111320.0
M_LON = 1 / (111320.0 * math.cos(math.radians(LAT)))


def P(e_m, n_m):
    """lat/lon of a point e_m east, n_m north of the focus centre."""
    return LAT + n_m * M_LAT, LON + e_m * M_LON


def g(e_m, n_m):
    la, lo = P(e_m, n_m)
    return {"lat": la, "lon": lo}


def box(e0, n0, e1, n1, tags):
    return {"type": "way", "tags": tags, "geometry": [g(e0, n0), g(e1, n0), g(e1, n1), g(e0, n1), g(e0, n0)]}


def line(pts, tags):
    return {"type": "way", "tags": tags, "geometry": [g(*p) for p in pts]}


ELEMENTS = [
    box(-200, -200, 200, 200, {"landuse": "forest"}),                     # mapped forest around the centre
    line([(-300, 50), (300, 50)], {"highway": "unclassified"}),          # 5 m road, 50 m north
    box(80, -100, 92, -88, {"building": "yes"}),                          # 12 m hut
    box(-150, -150, -100, -100, {"natural": "water"}),                    # pond
    line([(-300, -60), (300, -60)], {"waterway": "river", "width": "12"}),
    box(120, 100, 160, 140, {"natural": "bare_rock"}),
    box(-60, 120, -20, 160, {"landuse": "residential"}),
]
CLF = IgnitionSiteClassifier(ELEMENTS, True, "OpenStreetMap", "2026-10-09")


def _setup(n=5):
    return {"location": {"name": "t", "lat": LAT, "lon": LON, "source": "preset"}, "width_m": 500.0,
            "height_m": 500.0, "cell_m": 25.0, "duration_min": 60.0, "placement": "Map points", "n_ignition": n,
            "ignition_points": [], "ignition_source": "hypothetical", "layers": {"grid": True, "boundary": True}}


# 1 ─ vegetation accepted ────────────────────────────────────────────────────
def test_01_vegetation_click_is_accepted():
    c = CLF.classify(*P(0, 0))
    assert c.accepted and c.code == FUEL and c.message == "HYPOTHETICAL IGNITION ACCEPTED — VEGETATION"
    assert "landuse=forest" in c.evidence


# 2 - 5 ─ obvious non-burnable rejected, with the class and the hint ───────────
@pytest.mark.parametrize("pt,code,label", [
    ((10, 50.8), ROAD, "ROAD"),                       # 0.8 m from the road centre line
    ((86, -94), BUILT, "BUILT-UP AREA"),              # inside the hut
    ((-40, 140), BUILT, "BUILT-UP AREA"),             # mapped residential land use
    ((-125, -125), WATER, "WATER"),                   # pond
    ((0, -57), WATER, "WATER"),                       # 3 m from a 12 m wide river centre line
    ((140, 120), NON_FUEL, "BARE / NON-BURNABLE"),
], ids=["02_road", "03_building", "03b_built_up_landuse", "04_water_pond", "04b_river", "05_bare"])
def test_02_to_05_obvious_non_burnable_points_are_rejected(pt, code, label):
    c = CLF.classify(*P(*pt))
    assert not c.accepted and c.code == code and c.label == label
    assert c.message == f"IGNITION REJECTED — {label}. Select a vegetation/burnable location."
    assert c.evidence.startswith("OpenStreetMap")


def test_02b_vegetation_next_to_a_road_is_not_rejected():
    c = CLF.classify(*P(10, 58))                       # 8 m from the road centre (5 m road)
    assert c.accepted and c.code == FUEL


# 6 ─ unknown: no fabricated classification ───────────────────────────────────
def test_06_unmapped_point_is_location_unverified_not_vegetation():
    c = CLF.classify(*P(260, 230))                     # outside every mapped feature
    assert c.accepted and c.code == UNKNOWN and c.label == "LOCATION UNVERIFIED"
    assert "not confirmed vegetation" in c.message and "VEGETATION" not in c.label
    none = IgnitionSiteClassifier([], False, "OpenStreetMap unavailable")
    c = none.classify(*P(0, 0))
    assert c.code == UNKNOWN and c.accepted and "unavailable" in c.message


# 7, 8 ─ exact coordinates, correct CA cell ───────────────────────────────────
def test_07_08_accepted_point_kept_exactly_and_maps_to_its_cell():
    s = _setup()
    p = list(P(13.7, -21.2))
    msgs, reasons, notes = [], [], []
    assert apply_map_event({"kind": "ignite", "points": [p]}, s, None, whatif=True, land=CLF, messages=msgs,
                           reasons=reasons, notes=notes)
    assert s["ignition_points"] == [p] and msgs == [] and reasons == []
    assert notes == ["HYPOTHETICAL IGNITION ACCEPTED — VEGETATION"]
    dom = initial_domain(focus_from_setup(s), 60)
    r, c = dom.cell_of(*p)
    lat_c, lon_c = dom.cell_centres()
    assert abs(lat_c[r, c] - p[0]) <= dom.focus.dlat / 2 and abs(lon_c[r, c] - p[1]) <= dom.focus.dlon / 2
    ign = lm.plan_ignition(lm.WHATIF, s, focus_from_setup(s), {"ndvi": 0.6, "ffmc": 88.0, "bui": 40.0}, [], "live",
                           allow_fetch=False)["ign"]
    assert ign["points"] == [p] and ign["cells"] == [[r, c]]


# 9 ─ rejected ignition does not start a simulation ───────────────────────────
def test_09_point_on_a_mapped_road_is_accepted_and_labelled():
    # FINAL requirements (Priority 3C): a HYPOTHETICAL ignition is never rejected because of land cover; the
    # mapped surface is reported and labelled as known land cover instead.
    s = _setup()
    msgs, reasons, notes = [], [], []
    p = list(P(10, 50.5))
    apply_map_event({"kind": "ignite", "points": [p]}, s, None, whatif=True, land=CLF,
                    messages=msgs, reasons=reasons, notes=notes)
    assert s["ignition_points"] == [p] and msgs == [] and reasons == []
    assert notes[0].startswith("HYPOTHETICAL IGNITION ACCEPTED ON MAPPED ROAD — OpenStreetMap highway=")
    ign = lm.ignition_spec(lm.WHATIF, s, None)
    assert ign["source"] == "HYPOTHETICAL_USER" and ign["n_selected"] == 1 and ign["points"] == [p]


def test_mixed_clicks_keep_only_accepted_points_in_order():
    s = _setup()
    pts = [list(P(0, 0)), list(P(-125, -125)), list(P(260, 230))]
    notes = []
    validate_hypothetical_points(pts, s, CLF, [], [], notes)
    # the water point is accepted too (labelled), order kept; vegetation / water are KNOWN, the last unverified
    assert [c["label"] for c in s["ignition_checks"]] == ["VEGETATION", "WATER", "LOCATION UNVERIFIED"]
    assert [c["land_cover"] for c in s["ignition_checks"]] == ["known (mapped)", "known (mapped)", "unverified"]


def test_production_check_sends_no_cell_mask_to_the_map():
    args = _map_ignition_args({"source": "HYPOTHETICAL_USER", "points": [], "manual": True, "n_requested": 3,
                               "n_selected": 0}, _setup(), CLF)
    assert args["nonfuel_cells"] == [] and "nonfuel_classes" not in args       # point check, not a grid mask


def test_classifier_is_never_a_fire_mask():
    assert CLF.classes is None and not hasattr(CLF, "non_burnable") and not hasattr(CLF, "fuel_load")


def test_grid_land_cover_callers_still_supported():
    s = _setup()
    dom = initial_domain(focus_from_setup(s), 60)
    cls = np.full((dom.n_rows, dom.n_cols), FUEL, np.int8)
    r, c = dom.n_rows // 2, dom.n_cols // 2
    cls[r, c + 2], cls[r, c + 4] = ROAD, UNKNOWN
    lat, lon = dom.cell_centres()
    land = LandCover(cls, "osm", "test")
    msgs, reasons, notes = [], [], []
    validate_hypothetical_points([[float(lat[r, c + 2]), float(lon[r, c + 2])],
                                  [float(lat[r, c + 4]), float(lon[r, c + 4])]], s, land, msgs, reasons, notes)
    assert reasons == [] and msgs == []                                       # final requirements: never rejected
    assert notes[0].startswith("HYPOTHETICAL IGNITION ACCEPTED ON MAPPED ROAD")
    assert notes[1].startswith("LOCATION UNVERIFIED")                        # unknown cell: unverified, accepted
    assert rejection_message(ROAD).startswith("IGNITION REJECTED — ROAD")    # wording kept for LIVE observations


def test_firms_detection_on_mapped_water_is_kept_but_not_ignited():
    from src.simulation.observed_ignition import classify_detections
    from src.simulation.local_spread import FocusArea, domain_for
    dom = domain_for(FocusArea("t", LAT, LON), 60)
    la, lo = P(-125, -125)
    cls = classify_detections(dom, [{"lat": la, "lon": lo}, {"lat": LAT, "lon": LON}], CLF)
    st = {d["lat"]: d["status"] for d in cls["detections"]}
    assert st[la] == "non_fuel" and st[LAT] == "ignition" and cls["n_valid"] == 1


# 44 ─ cell explanation ───────────────────────────────────────────────────────
def test_44_cell_explanation_says_why_a_cell_can_burn():
    from src.dashboard.geo_spread import explain_cell
    from src.simulation.local_spread import FocusArea, run_local_spread
    f = FocusArea("t", LAT, LON, 500, 500, 25)
    res = run_local_spread(f, {"ffmc": 92.0, "ndvi": 0.6, "bui": 40.0}, 5.0, 270.0, placement="Centre",
                           n_ignition=1, duration_minutes=15, seed=1)
    r, c = np.argwhere(res.ignition_step == 0)[0]
    lat_c, lon_c = res.domain.cell_centres()
    info = explain_cell(res, float(lat_c[r, c]), float(lon_c[r, c]))
    assert info["burnability"] == "YES" and info["simulated ignition"] == "T+0 min"
    assert info["land cover"].startswith("not used for fire spread")
    assert explain_cell(res, 0.0, 0.0) is None
