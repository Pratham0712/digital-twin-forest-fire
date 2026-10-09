"""
geo_spread.py - simulation set-up (What-If page) and the geographic spread
panel (Spread Simulation page). Pure UI glue:

  * risk state            -> DigitalTwin (unchanged)
  * fire spread           -> src/simulation/local_spread.py (unchanged FireSpreadSimulator)
  * map, boundary, VFX    -> geo_fire_map.py + components/fire_map (browser)

The simulation set-up is one dict kept in st.session_state["sim_setup"] and
copied into simulation_config["setup"] when the scenario is applied:

    location        {name, lat, lon, source: preset|google_geocoding|google_places|map|highest_risk,
                     preset, address, types, bounds}
    width_m/height_m focus area, metres (snapped to whole cells)
    cell_m          CA cell size
    duration_min    simulated time
    placement, n_ignition, ignition_points [[lat, lon], ...]
    layers          {grid, boundary}
"""
from __future__ import annotations

import io
import logging
import json
import math
import os
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.config import SYSTEM
from src.dashboard.dashboard_common import _compass_direction_name, log_action
from src.dashboard.geo_fire_map import (duration_label, hotspots_near, missing_key_card, new_event,
                                        render_fire_map, setup_payload, sim_payload)
from src.dashboard.ui.live_panel import live_observations, map_hotspots, render_live_conditions
from src.utils.timezone import format_ist
from src.simulation.fuel_map import CLASS_NAMES, FUEL, UNKNOWN
from src.simulation.local_spread import (DEFAULT_FOCUS, DOMAIN_AUTO, DOMAIN_CUSTOM, DOMAIN_MAP, DOMAIN_PRESETS,
                                         domain_box_from_setup,
                                         EXTENT_LIMIT_TITLE, FOCUS_AREAS, HIGHEST_RISK_FOCUS,
                                         MAX_EXTENT_OPTIONS_M, PLACEMENTS, FocusArea, domain_elevation, domain_for,
                                         domain_from_setup, domain_size_from_setup, focus_options_for_region,
                                         frame_minutes_for, initial_domain, limits_from_setup, n_steps_for,
                                         resolve_focus, run_local_spread, step_minutes_for, validate_domain,
                                         validate_setup, zone_conditions)

SPREAD_PAGE = "pages/2_Spread_Simulation.py"
WHATIF_PAGE = "pages/1_What_If_Simulator.py"
SPREAD_MODES = ["Applied What-If scenario", "Current risk state (live/demo, from the sidebar)",
                "Custom weather scenario"]
CUSTOM_LOCATION = "Custom location (Google search or map)"
SIZE_PRESETS = {"250 m": (250, 250), "500 m": (500, 500), "1 km": (1000, 1000), "2 km": (2000, 2000)}
AREA_PRESETS_KM2 = [0.25, 0.5, 1.0, 2.0, 5.0]
SUITABLE_TYPES = {"park", "natural_feature", "national_park", "campground", "tourist_attraction",
                  "hiking_area", "nature_reserve", "colloquial_area"}


def google_maps_key() -> str:
    return os.getenv("GOOGLE_MAPS_API_KEY", "")


def google_maps_map_id() -> str:
    return os.getenv("GOOGLE_MAPS_MAP_ID", "")


# ── set-up state ──────────────────────────────────────────────────────────── #

def _preset_location(name: str, twin) -> dict:
    snap = twin.current_snapshot
    f = resolve_focus(name, snap.processed_grid, snap.risk_scores)
    return {"name": f.name, "lat": f.lat, "lon": f.lon, "source": "preset" if name in FOCUS_AREAS else "highest_risk",
            "preset": name, "address": f.description, "types": [], "bounds": None}


def default_setup(twin) -> dict:
    opts = focus_options_for_region(twin.region)
    first = DEFAULT_FOCUS if DEFAULT_FOCUS in opts else opts[0]
    return {"location": _preset_location(first, twin), "width_m": float(SYSTEM.focus_size_m),
            "height_m": float(SYSTEM.focus_size_m), "cell_m": float(SYSTEM.local_ca_cell_m),
            "duration_min": 60.0, "placement": PLACEMENTS[0], "n_ignition": 3, "ignition_points": [],
            "layers": {"grid": True, "boundary": True}, "domain_preset": DOMAIN_AUTO,
            "domain_w_m": 2000.0, "domain_h_m": 2000.0, "max_extent_m": float(MAX_EXTENT_OPTIONS_M[-1])}


def get_setup(twin, store_key: str = "sim_setup", kp: Optional[str] = None) -> dict:
    """The shared set-up; its location follows the sidebar region (a new
    region starts at that region's first preset)."""
    s = st.session_state.get(store_key)
    if not s or s.get("region") != twin.region.name:
        keep = {k: s[k] for k in ("width_m", "height_m", "cell_m", "duration_min", "layers", "domain_preset",
                                  "domain_w_m", "domain_h_m", "max_extent_m") if k in s} if s else {}
        if keep.get("domain_preset") == DOMAIN_MAP:
            keep["domain_preset"] = DOMAIN_AUTO              # a map-drawn domain does not follow a new region
        s = {**default_setup(twin), **keep, "region": twin.region.name}
        st.session_state[store_key] = s
        if kp:
            st.session_state[f"_{kp}_sync"] = True
    return s


def focus_from_setup(setup: dict) -> FocusArea:
    loc = setup["location"]
    return FocusArea(name=loc["name"], lat=float(loc["lat"]), lon=float(loc["lon"]),
                     width_m=float(setup["width_m"]), height_m=float(setup["height_m"]),
                     cell_m=float(setup["cell_m"]), source=loc.get("source", "preset"))


def _snap(v: float, cell: float) -> float:
    return float(np.clip(round(v / cell) * cell, SYSTEM.focus_min_m, SYSTEM.focus_max_m))


HYP_OUTSIDE = ("Outside the simulation domain — click inside the dashed amber SIMULATION DOMAIN box (it may be "
               "outside the blue focus area), or enlarge the domain under Simulation domain.")
HYP_NON_FUEL = "Ignition rejected — selected location is non-burnable. Choose a fuel cell."
HYP_NO_MASK = ("LOCATION UNVERIFIED: OpenStreetMap is unavailable for this area, so clicked points cannot be "
               "checked for roads, buildings or water. Hypothetical points are accepted as UNVERIFIED (not "
               "confirmed vegetation).")


def rejection_message(cls_code: int) -> str:
    """Per-class rejection shown for a clicked non-burnable location."""
    from src.simulation.ignition_site import LABELS, REJECT_HINT
    return f"IGNITION REJECTED — {LABELS.get(int(cls_code), 'NON-BURNABLE')}. {REJECT_HINT}"


def study_region():
    """The primary study region (Bandipur Tiger Reserve) boundary, or None."""
    from src.geo.study_region import load_study_region
    try:
        return load_study_region()
    except Exception:                                   # a broken boundary file never breaks the simulation
        return None


def validate_hypothetical_points(points, setup: dict, land=None, messages: Optional[list] = None,
                                 reasons: Optional[list] = None, notes: Optional[list] = None) -> list:
    """WHAT-IF hypothetical ignition points as clicked: kept exactly where they
    are (never moved), only inside the set-up coverage (the initial simulation
    domain), at most the requested ignition-cell count, and checked by the
    LIGHTWEIGHT ignition-point check:

      * `land` = IgnitionSiteClassifier (production): the clicked point is
        tested against mapped OpenStreetMap features - road, building / built-up
        area, water, bare ground -> rejected; mapped vegetation -> accepted;
        anything else -> LOCATION UNVERIFIED, accepted with a warning;
      * `land` = a LandCover grid (callers / tests that supply one): the cell's
        class decides; UNKNOWN cells are LOCATION UNVERIFIED (accepted).

    The check only decides whether a point may ignite. It is never a fire
    mask: an accepted ignition spreads with the CA's own parameters.
    Rejections go to `messages` (generic) and `reasons` (per class); accepted
    points' outcomes (VEGETATION / LOCATION UNVERIFIED) go to `notes`."""
    from src.simulation.ignition_site import LABELS, REJECT_HINT
    dom = domain_from_setup(setup)          # the configured simulation domain (NOT the focus area)
    msgs = messages if messages is not None else []
    rs = reasons if reasons is not None else []
    ns = notes if notes is not None else []
    point_check = land is not None and hasattr(land, "classify")
    classes = None
    if not point_check and land is not None and getattr(land, "classes", None) is not None:
        classes = land.classes if land.classes.shape == (dom.n_rows, dom.n_cols) else None
    n_max = int(setup.get("n_ignition") or 1)
    out, checks = [], []
    for lat, lon in points or []:
        rc = dom.cell_of(float(lat), float(lon))
        if rc is None:
            msgs.append(HYP_OUTSIDE)
            continue
        known = False
        if point_check:
            chk = land.classify(float(lat), float(lon))
            ok, label, text = chk.accepted, chk.label, chk.message
            evidence = chk.evidence
        elif classes is not None:
            code = int(classes[rc])
            ok = code in (FUEL, UNKNOWN)
            label = LABELS.get(code, "NON-BURNABLE")
            evidence = "land-cover grid"
            text = ("HYPOTHETICAL IGNITION ACCEPTED — VEGETATION" if code == FUEL else
                    "LOCATION UNVERIFIED — the type of ground at this point is not known. The hypothetical "
                    "ignition is accepted, but it is not confirmed vegetation.")
        else:
            ok, label, text = True, LABELS[UNKNOWN], ("LOCATION UNVERIFIED — no location information is loaded "
                                                      "for this area. The hypothetical ignition is accepted, but "
                                                      "it is not confirmed vegetation.")
        if not ok:
            # final requirements: a HYPOTHETICAL ignition is never rejected because of land cover - the known
            # (mapped) surface is reported with its evidence and labelled, and the point is kept exactly
            known = True
            text = (f"HYPOTHETICAL IGNITION ACCEPTED ON MAPPED {label} — {evidence}. This is known land-cover "
                    "information, not a fuel measurement; the simulated fire starts here as requested.")
        if len(out) >= n_max:
            msgs.append(f"Maximum of {n_max} hypothetical ignition points reached.")
        else:
            out.append([float(lat), float(lon)])
            ns.append(text)
            checks.append({"lat": float(lat), "lon": float(lon), "label": label,
                           "land_cover": "known (mapped)" if (known or label == "VEGETATION") else "unverified"})
    if out or points == []:
        setup["ignition_checks"] = checks
    return out


def apply_map_event(ev: dict, setup: dict, twin, whatif: bool = False, land=None,
                    messages: Optional[list] = None, reasons: Optional[list] = None,
                    notes: Optional[list] = None) -> bool:
    """Apply an event from the map component to the set-up. True if changed.
    whatif=True (LIVE DATA -> WHAT-IF, hypothetical ignition): clicked points are
    validated (domain, fuel mask, count) and "Clear points" keeps the Map points
    placement with 0 points selected instead of falling back to a placement rule."""
    if whatif and ev.get("kind") == "ignite":
        pts = validate_hypothetical_points(ev.get("points") or [], setup, land, messages, reasons, notes)
        setup["ignition_points"] = pts
        setup["placement"] = "Map points"
        setup["ignition_source"] = "hypothetical"
        return True
    kind = ev.get("kind")
    if kind == "domain":
        # the simulation domain was moved / resized on the map: its own bounds, the focus area is unchanged
        setup["domain_box"] = {k: float(ev[k]) for k in ("north", "south", "east", "west")}
        setup["domain_preset"] = DOMAIN_MAP
        return True
    if kind == "area":
        loc = dict(setup["location"])
        moved = abs(loc["lat"] - ev["lat"]) > 1e-6 or abs(loc["lon"] - ev["lon"]) > 1e-6
        loc.update(lat=float(ev["lat"]), lon=float(ev["lon"]))
        if moved and loc.get("source") in ("preset", "highest_risk"):
            loc.update(name=f"{loc['name'].replace(' (adjusted)', '')} (adjusted)", source="map")
        setup.update(location=loc, width_m=_snap(ev["width_m"], setup["cell_m"]),
                     height_m=_snap(ev["height_m"], setup["cell_m"]))
        return True
    if kind == "place":
        setup["location"] = {"name": ev.get("name") or "Selected place", "lat": float(ev["lat"]), "lon": float(ev["lon"]),
                             "source": ev.get("source", "google_geocoding"), "preset": CUSTOM_LOCATION,
                             "address": ev.get("address", ""), "types": ev.get("types") or [],
                             "bounds": ev.get("bounds")}
        setup["ignition_points"] = []
        if setup["placement"] == "Map points":
            setup["placement"] = PLACEMENTS[0]
        return True
    if kind == "ignite":
        pts = [[float(a), float(b)] for a, b in (ev.get("points") or [])][:10]
        setup["ignition_points"] = pts
        if pts and setup.get("ignition_source") == "none":
            setup["ignition_source"] = "hypothetical"           # clicked points are a user (HYPOTHETICAL) ignition
        setup["placement"] = "Map points" if pts else (PLACEMENTS[0] if setup["placement"] == "Map points"
                                                       else setup["placement"])
        return True
    if kind == "layers":
        setup["layers"] = {"grid": bool(ev.get("grid", True)), "boundary": bool(ev.get("boundary", True))}
        return True
    return False


def _sync_widgets(setup: dict, kp: str):
    """Push set-up values into the widget keys (only ever called before the
    widgets are drawn in this run)."""
    st.session_state[f"{kp}_w"] = int(setup["width_m"])
    st.session_state[f"{kp}_h"] = int(setup["height_m"])
    st.session_state[f"{kp}_cell"] = int(setup["cell_m"])
    st.session_state[f"{kp}_place"] = setup["placement"]
    st.session_state[f"{kp}_nign"] = int(setup["n_ignition"])
    st.session_state[f"{kp}_grid"] = bool(setup["layers"]["grid"])
    st.session_state[f"{kp}_bound"] = bool(setup["layers"]["boundary"])
    d = float(setup["duration_min"])
    if d in SYSTEM.duration_presets_minutes:
        st.session_state[f"{kp}_dur"] = duration_label(d)
    else:
        st.session_state[f"{kp}_dur"] = "Custom"
        st.session_state[f"{kp}_dur_c"] = int(d)
    st.session_state[f"{kp}_loc"] = setup["location"].get("preset", CUSTOM_LOCATION)
    st.session_state[f"{kp}_dpre"] = setup.get("domain_preset", DOMAIN_AUTO)
    st.session_state[f"{kp}_dw"] = int(setup.get("domain_w_m", 2000))
    st.session_state[f"{kp}_dh"] = int(setup.get("domain_h_m", 2000))
    st.session_state[f"{kp}_dmax"] = int(setup.get("max_extent_m", MAX_EXTENT_OPTIONS_M[-1]) // 1000)
    st.session_state[f"{kp}_igsrc"] = setup.get("ignition_source", "none")


def render_setup_controls(twin, setup: dict, kp: str, show_location: bool = True,
                          ignition_mode: Optional[str] = None) -> Optional[str]:
    """Location / size / cell / duration / ignition / layer controls. Returns a
    validation error message or None.

    ignition_mode (What-If live modes): "live" - ignition is ONLY observed NASA
    FIRMS cells, the placement controls are locked; "whatif" - an ignition
    source selector (NONE / OBSERVED / HYPOTHETICAL); the placement controls
    apply to a HYPOTHETICAL ignition only. None: the original controls."""
    if st.session_state.pop(f"_{kp}_sync", False) or f"{kp}_w" not in st.session_state:
        _sync_widgets(setup, kp)

    if show_location:
        opts = focus_options_for_region(twin.region) + [CUSTOM_LOCATION]
        if st.session_state.get(f"{kp}_loc") not in opts:
            st.session_state[f"{kp}_loc"] = CUSTOM_LOCATION
        choice = st.selectbox("Forest / location", opts, key=f"{kp}_loc",
                              help="Pick a preset forest, or search Google Maps in the map below (or drag the box) "
                                   "for any other place.")
        if choice != CUSTOM_LOCATION and setup["location"].get("preset") != choice:
            setup["location"] = _preset_location(choice, twin)
            setup["ignition_points"] = []
            if setup["placement"] == "Map points":
                setup["placement"] = PLACEMENTS[0]
                st.session_state[f"{kp}_place"] = PLACEMENTS[0]

    if st.session_state.pop(f"_{kp}_sizemode_reset", False):        # set before the radio exists (quick size)
        st.session_state[f"{kp}_sizemode"] = "Width × height"
    mode = st.radio("Size by", ["Width × height", "Area"], horizontal=True, key=f"{kp}_sizemode")
    c1, c2, c3, c4 = st.columns([1.1, 1.1, 0.9, 1.2])
    cell = c3.selectbox("Cell size (m)", list(SYSTEM.local_cell_options_m), key=f"{kp}_cell",
                        help="25 m is the default. Smaller cells give finer time steps (CA step = 15 min × cell / 100 m).")
    if mode == "Width × height":
        w = c1.number_input("Width (m, east–west)", SYSTEM.focus_min_m, SYSTEM.focus_max_m, step=int(cell), key=f"{kp}_w")
        h = c2.number_input("Height (m, north–south)", SYSTEM.focus_min_m, SYSTEM.focus_max_m, step=int(cell), key=f"{kp}_h")
    else:
        km2 = c1.selectbox("Area (km²)", AREA_PRESETS_KM2, index=0, key=f"{kp}_km2")
        side = math.sqrt(km2) * 1000.0
        w = h = side
        c2.markdown(f"<div style='padding-top:34px;color:#8a96a6'>Square of {side:.0f} m × {side:.0f} m</div>",
                    unsafe_allow_html=True)
    with c4:
        dur_opts = [duration_label(d) for d in SYSTEM.duration_presets_minutes] + ["Custom"]
        dsel = st.selectbox("Simulation duration", dur_opts, key=f"{kp}_dur",
                            help="Simulated time, not playback time. The fire spreads until this time is reached.")
        if dsel == "Custom":
            dur = float(st.number_input("Custom duration (min)", 1, SYSTEM.max_duration_minutes, step=1, key=f"{kp}_dur_c"))
        else:
            dur = float(SYSTEM.duration_presets_minutes[dur_opts.index(dsel)])

    dom_err = _render_domain_controls(setup, kp, float(cell), float(w), float(h), dur)

    q = st.columns(len(SIZE_PRESETS) + 1)
    q[0].caption("Quick size")
    for col, (lbl, (pw, ph)) in zip(q[1:], SIZE_PRESETS.items()):
        if col.button(lbl, key=f"{kp}_sz_{lbl}", use_container_width=True):
            setup.update(width_m=float(pw), height_m=float(ph))
            st.session_state[f"_{kp}_sync"] = True
            st.session_state[f"_{kp}_sizemode_reset"] = True
            st.rerun()

    hyp_locked = False
    prev_placement = setup.get("placement")
    if ignition_mode == "live":
        st.caption("Ignition: **observed NASA FIRMS detections only** (LIVE). Valid detections inside the area ignite "
                   "their grid cell; with none, no fire is simulated. Hypothetical placement is available in "
                   "LIVE DATA → WHAT-IF.")
        hyp_locked = True
    elif ignition_mode == "whatif":
        from src.dashboard.live_modes import IGN_OPTIONS
        if st.session_state.get(f"{kp}_igsrc") not in IGN_OPTIONS:
            st.session_state[f"{kp}_igsrc"] = setup.get("ignition_source", "none") \
                if setup.get("ignition_source") in IGN_OPTIONS else "none"
        src = st.selectbox("Ignition source", list(IGN_OPTIONS), key=f"{kp}_igsrc", format_func=IGN_OPTIONS.get,
                           help="NONE: no fire is simulated. OBSERVED: valid NASA FIRMS detections inside the area. "
                                "HYPOTHETICAL: your ignition (edge, centre or map points), labelled HYPOTHETICAL.")
        setup["ignition_source"] = src
        hyp_locked = src != "hypothetical"

    i1, i2, i3, i4 = st.columns([1.4, 1, 0.8, 0.8])
    placement = i1.selectbox("Ignition" if not ignition_mode else "Hypothetical ignition placement", PLACEMENTS,
                             key=f"{kp}_place", disabled=hyp_locked,
                             help="Upwind edge: the head fire runs across the area. Map points: use 'Set ignition on "
                                  "map' and click the map.")
    whatif_hyp = ignition_mode == "whatif" and not hyp_locked
    n_ign = i2.slider("Ignition cells", 1, 9, key=f"{kp}_nign",
                      disabled=hyp_locked or (placement == "Map points" and not whatif_hyp),
                      help="Number of hypothetical ignition cells; with Map points, the number of points you can "
                           "place on the map." if whatif_hyp else None)
    grid = i3.toggle("Grid", key=f"{kp}_grid")
    boundary = i4.toggle("Boundary", key=f"{kp}_bound")

    cell = float(cell)
    setup.update(width_m=_snap(float(w), cell), height_m=_snap(float(h), cell), cell_m=cell, duration_min=dur,
                 placement=placement, n_ignition=int(n_ign), layers={"grid": bool(grid), "boundary": bool(boundary)})
    if whatif_hyp:
        # WHAT-IF hypothetical ignition: one state (setup) for map and controls.
        if prev_placement != placement and "Map points" in (prev_placement, placement):
            setup["ignition_points"] = []              # no stale points from the previous strategy
        if placement == "Map points":
            pts = setup["ignition_points"]
            if len(pts) > int(n_ign):
                setup["ignition_points"] = pts[:int(n_ign)]
                st.caption(f"Ignition cells lowered to {int(n_ign)}: only the first {int(n_ign)} map point(s) are kept.")
            n_sel = len(setup["ignition_points"])
            st.markdown(f"<div class='lm-ctag'><span class='lm-tag whatif'>HYPOTHETICAL IGNITIONS</span> "
                        f"{n_sel} / {int(n_ign)} SELECTED</div>", unsafe_allow_html=True)
            if not n_sel:
                st.caption("Click **Set ignition on map** on the map, then click inside the simulation area "
                           f"(up to {int(n_ign)} point(s), fuel cells only). No point, no fire.")
        else:
            st.markdown(f"<div class='lm-ctag'><span class='lm-tag whatif'>HYPOTHETICAL IGNITIONS</span> "
                        f"{int(n_ign)} cell(s) · {placement}</div>", unsafe_allow_html=True)
    elif placement == "Map points" and not setup["ignition_points"] and not hyp_locked:
        st.caption("No map points yet: click **Set ignition on map** on the map, then click inside the area. "
                    "Until then the centre is used.")
    return validate_setup(setup["width_m"], setup["height_m"], cell, dur) or dom_err


def _render_domain_controls(setup: dict, kp: str, cell: float, w: float, h: float, dur: float) -> Optional[str]:
    """Simulation domain (Phase 3, issue 1): the grid the fire model runs on,
    distinct from the focus area. Presets are real domain sizes."""
    st.markdown("**Simulation domain** — the computational grid (dashed amber on the map), independent of the "
                "focus area (solid blue). On the map: **Edit domain** to move / resize it (it may lie outside the "
                "focus area), **Edit focus** for the focus area, **Pan map** to move the map, **Set ignition on "
                "map** to place ignitions.")
    if f"{kp}_dpre" not in st.session_state:
        st.session_state[f"{kp}_dpre"] = setup.get("domain_preset", DOMAIN_AUTO)
        st.session_state[f"{kp}_dw"] = int(setup.get("domain_w_m", 2000))
        st.session_state[f"{kp}_dh"] = int(setup.get("domain_h_m", 2000))
        st.session_state[f"{kp}_dmax"] = int(setup.get("max_extent_m", MAX_EXTENT_OPTIONS_M[-1]) // 1000)
    d1, d2, d3, d4 = st.columns([1.5, 1, 1, 1])
    pre = d1.selectbox("Domain size", list(DOMAIN_PRESETS), key=f"{kp}_dpre",
                       help="Auto: focus area plus a margin for the selected duration. The presets are real "
                            "domain sizes (square). The domain still grows while the fire spreads, up to the "
                            "maximum extent.")
    if pre == DOMAIN_MAP and domain_box_from_setup({**setup, "domain_preset": DOMAIN_MAP}) is None:
        # start the map-drawn domain from the domain currently shown
        cur = domain_from_setup({**setup, "domain_preset": setup.get("domain_preset") if
                                 setup.get("domain_preset") != DOMAIN_MAP else DOMAIN_AUTO})
        setup["domain_box"] = {k: round(v, 7) for k, v in cur.bounds.items()}
    custom = pre == DOMAIN_CUSTOM
    dw = d2.number_input("Domain width (m)", 100, 15000, step=int(cell) * 4, key=f"{kp}_dw", disabled=not custom)
    dh = d3.number_input("Domain height (m)", 100, 15000, step=int(cell) * 4, key=f"{kp}_dh", disabled=not custom)
    dmax = d4.selectbox("Maximum extent (km)", [int(v // 1000) for v in MAX_EXTENT_OPTIONS_M], key=f"{kp}_dmax",
                        help="Upper limit for the domain, including automatic expansion while the fire spreads "
                             "(protects memory and keeps the browser responsive).")
    setup.update(domain_preset=pre, domain_w_m=float(dw), domain_h_m=float(dh), max_extent_m=float(dmax) * 1000.0)
    loc = setup["location"]
    f = FocusArea("check", float(loc["lat"]), float(loc["lon"]), _snap(w, cell), _snap(h, cell), cell)
    err, warn = validate_domain(f, domain_size_from_setup(setup), limits_from_setup(setup), dur,
                                domain_box_from_setup(setup))
    if pre == DOMAIN_MAP and not err:
        d = domain_from_setup(setup)
        st.caption(f"Domain drawn on the map: {d.width_m / 1000:.2f} × {d.height_m / 1000:.2f} km, grid "
                   f"{d.n_cols} × {d.n_rows} cells, centre {0.5 * (d.bounds['north'] + d.bounds['south']):.5f}° N, "
                   f"{0.5 * (d.bounds['east'] + d.bounds['west']):.5f}° E.")
    if err:
        st.error(err)
    elif warn:
        st.warning(warn)
    return err


def render_setup_summary(setup: dict, twin):
    f = focus_from_setup(setup)
    dur = setup["duration_min"]
    dom = domain_from_setup(setup)
    step = frame_minutes_for(dur)
    cards = [("Width", f"{f.n_cols * f.cell_m:.0f} m"), ("Height", f"{f.n_rows * f.cell_m:.0f} m"),
             ("Area", f"{f.area_km2:.2f} km²"), ("Cell", f"{f.cell_m:.0f} m"), ("Grid", f"{f.n_cols} × {f.n_rows}"),
             ("Simulated time", duration_label(dur)),
             ("Frame", f"{step * 60:.0f} s" if step < 1 else f"{step:g} min"),
             ("Initial simulation domain", f"{dom.width_m / 1000:.2f} × {dom.height_m / 1000:.2f} km"),
             ("Domain grid", f"{dom.n_cols} × {dom.n_rows}")]
    st.markdown('<div class="kpi-row">' + "".join(
        f'<div class="kpi"><div class="lbl">{a}</div><div class="val" style="font-size:18px">{b}</div></div>'
        for a, b in cards) + "</div>", unsafe_allow_html=True)
    st.caption("Focus area (solid blue box) is the area you analyse; the SIMULATION DOMAIN (dashed amber box) is "
               "the grid the fire model runs on - ignitions may be placed anywhere inside it. The domain grows "
               "automatically before the fire reaches its edge, up to the maximum extent. Simulated time is computed "
               "in full and then played back (1× ≈ 1 simulated minute per second).")
    _location_notes(setup, twin)


def _location_notes(setup: dict, twin):
    loc = setup["location"]
    r = twin.region
    if not (r.min_lat <= loc["lat"] <= r.max_lat and r.min_lon <= loc["lon"] <= r.max_lon):
        snap = twin.current_snapshot
        c = zone_conditions(snap.processed_grid, snap.risk_scores, loc["lat"], loc["lon"])
        st.warning(f"{loc['name']} is outside the selected region ({r.name}). Weather, fuel and risk come from the "
                   f"nearest grid zone, {c['zone_distance_km']:.0f} km away. Choose the matching region in the "
                   f"sidebar for local conditions.")
    if loc.get("source", "").startswith("google") and not (set(loc.get("types") or []) & SUITABLE_TYPES):
        st.warning("Location selected. Verify that this area is suitable for forest-fire simulation "
                   "(the model assumes burnable vegetation inside the area).")


def live_offline() -> bool:
    """The sidebar's demo / offline choice (live by default when keys are set)."""
    from src.dashboard.app_state import is_demo_mode
    return is_demo_mode()


def render_setup_map(setup: dict, kp: str, twin, wind_speed_ms: float, wind_from_deg: float, height: int = 540,
                     hotspots=None, hotspot_kind: str = "observed", hotspot_summary: str = "",
                     ignition: Optional[dict] = None, land=None, whatif: bool = False):
    """Interactive set-up map: search, drag/resize box, click-to-ignite, real
    NASA FIRMS detections around the location."""
    key = google_maps_key()
    if not key:
        missing_key_card()
        return
    f = focus_from_setup(setup)
    if validate_setup(f.width_m, f.height_m, f.cell_m, setup["duration_min"]) or \
            validate_domain(f, domain_size_from_setup(setup), limits_from_setup(setup), setup["duration_min"],
                            domain_box_from_setup(setup))[0]:
        return
    from src.dashboard.geo_fire_map import land_cover_layer, study_region_payload
    dom0 = domain_from_setup(setup)
    lc_layer = None
    if land is not None and hasattr(land, "context_classes"):
        ctx_cls = land.context_classes(dom0)
        if ctx_cls is not None:
            lc_layer = land_cover_layer(ctx_cls, "CONTEXT: OpenStreetMap mapped features (not a fire barrier)",
                                        "context")
    elif land is not None and getattr(land, "classes", None) is not None and \
            land.classes.shape == (dom0.n_rows, dom0.n_cols):
        lc_layer = land_cover_layer(land.classes, land.label, getattr(land, "status", ""))
    payload = setup_payload(f, setup["duration_min"], wind_speed_ms, wind_from_deg, setup["n_ignition"],
                            setup["placement"], setup["ignition_points"], setup["layers"], key,
                            google_maps_map_id(), height=height, hotspots=hotspots or [],
                            hotspot_kind=hotspot_kind, hotspot_summary=hotspot_summary,
                            **_map_ignition_args(ignition, setup, land), land_layer=lc_layer,
                            study_region=study_region_payload(study_region()), domain=dom0,
                            domain_size=domain_size_from_setup(setup),
                            domain_independent=domain_box_from_setup(setup) is not None)
    ev = new_event(render_fire_map(payload, key=f"{kp}_map"), f"{kp}_map")
    msgs: list = []
    reasons: list = []
    notes: list = []
    if ev and apply_map_event(ev, setup, twin, whatif=whatif, land=land, messages=msgs, reasons=reasons,
                              notes=notes):
        st.session_state[f"_{kp}_sync"] = True
        st.session_state[f"_{kp}_ign_msgs"] = sorted(set(reasons + [m for m in msgs if m != HYP_NON_FUEL]))
        st.session_state[f"_{kp}_ign_notes"] = list(dict.fromkeys(notes))
        st.rerun()
    if whatif and land is not None and hasattr(land, "classify") and setup.get("ignition_source") == "hypothetical":
        st.caption(land.label + ". It only checks where a hypothetical ignition may be placed; fire spread is "
                   "not restricted by land cover.")
    _render_ignition_feedback(kp)


def _render_ignition_feedback(kp: str):
    """Outcome of the last ignition-point check (shown once after the click)."""
    for m in st.session_state.pop(f"_{kp}_ign_msgs", []) or []:
        st.warning(m)
    for n in st.session_state.pop(f"_{kp}_ign_notes", []) or []:
        (st.warning if "ON MAPPED" in n else st.success if n.startswith("HYPOTHETICAL IGNITION ACCEPTED")
         else st.info)(n)


def _map_ignition_args(ign: Optional[dict], setup: Optional[dict] = None, land=None) -> dict:
    """Map marker style for an ignition spec (live_modes.ignition_spec); None =
    the original hypothetical placement markers. A WHAT-IF hypothetical ignition
    also sends the point limit (Ignition cells) and the non-fuel cells, so the
    map rejects invalid clicks immediately (Python validates again)."""
    if not ign:
        return {}
    if ign["source"] == "OBSERVED_FIRMS":
        return {"ignition_kind": "observed", "observed_points": ign["points"],
                "ignition_label": f"OBSERVED · {ign.get('n_cells', len(ign['points']))} NASA FIRMS cell(s)"}
    if ign["source"] == "HYPOTHETICAL_USER" or ign.get("awaiting_points"):
        # A LandCover GRID (supplied by a caller) lets the map reject clicks on its non-burnable cells at once.
        # The production point check (IgnitionSiteClassifier) has no grid: the clicked point is checked in
        # Python and the outcome is shown after the click.
        if land is not None and getattr(land, "classes", None) is not None and getattr(land, "available", False):
            from src.simulation.ignition_site import LABELS
            flat = land.classes.ravel()
            nb = (flat != FUEL) & (flat != UNKNOWN)
            nf = np.flatnonzero(nb).tolist()
            nfc = {LABELS[k]: np.flatnonzero(flat == k).tolist()
                   for k in LABELS if k not in (FUEL, UNKNOWN) and (flat == k).any()}
        else:
            nf, nfc = [], {}
        lbl = (f"HYPOTHETICAL · {ign.get('n_selected', 0)} / {ign.get('n_requested')} selected" if ign.get("manual")
               else f"HYPOTHETICAL · {ign.get('n_requested')} cell(s) · {ign.get('placement') or '-'}")
        out = {"ignition_kind": "hypothetical", "ignition_label": lbl, "max_picks": ign.get("n_requested"),
               "nonfuel_cells": nf, "picks_only_for_map_points": True}
        if nfc:
            out["nonfuel_classes"] = nfc
        return out
    return {"ignition_kind": "none", "ignition_label": "NONE (no ignition source)"}


# ── What-If -> Spread hand-off ────────────────────────────────────────────── #

def build_simulation_config(twin, scenario: dict, setup: dict, live: Optional[dict] = None) -> dict:
    """Everything Module 2 needs, taken from the scenario twin that Module 1
    just computed (no second weather or risk calculation).

    live (What-If live modes, see live_modes / the What-If page): mode, live
    baseline, scenario values, weather + FIRMS source/status, the observed
    detections and the ignition source (OBSERVED / HYPOTHETICAL / NONE)."""
    snap = twin.current_snapshot
    loc = setup["location"]
    cond = zone_conditions(snap.processed_grid, snap.risk_scores, loc["lat"], loc["lon"])
    summary = twin.get_summary()
    sev = next((a.severity for a in snap.alerts if a.zone_id == cond["zone_id"]), "LOW")
    r = twin.region
    setup_copy = json.loads(json.dumps(setup))
    return {
        "region": r.name, "region_bounds": [r.min_lat, r.max_lat, r.min_lon, r.max_lon],
        "selected_location": {k: loc.get(k) for k in ("name", "lat", "lon", "source", "bounds", "address")},
        "forest": loc["name"], "latitude": loc["lat"], "longitude": loc["lon"],
        "width_m": setup["width_m"], "height_m": setup["height_m"], "size_m": setup["width_m"],
        "cell_m": setup["cell_m"], "duration_min": setup["duration_min"],
        "temp_c": scenario.get("temp_c"), "humidity_pct": scenario.get("humidity_pct"),
        "wind_speed_ms": scenario.get("wind_speed_ms"), "wind_from_deg": scenario.get("wind_from_deg", 225),
        "n_hotspots": scenario.get("n_hotspots"),
        "risk": {"zone_id": cond["zone_id"], "zone_risk": cond["risk_score"], "zone_severity": sev,
                 "zone_distance_km": cond["zone_distance_km"],
                 "region_peak_risk": summary.get("max_risk_score", 0.0),
                 "region_alerts": summary.get("total_alerts", 0),
                 "ffmc": cond["ffmc"], "bui": cond["bui"], "fwi": cond["fwi"]},
        "simulation": {"n_ignition": setup["n_ignition"], "placement": setup["placement"],
                       "ignition_points": setup["ignition_points"], "duration_min": setup["duration_min"]},
        "setup": setup_copy,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "offline": bool(twin.offline), "scenario": dict(scenario),
        "sim_mode": (live or {}).get("mode", "demo"),
        "live": json.loads(json.dumps(live, default=str)) if live else None,
    }


def apply_and_open_spread(twin, scenario: dict, setup: dict, live: Optional[dict] = None):
    cfg = build_simulation_config(twin, scenario, setup, live)
    st.session_state["simulation_config"] = cfg
    st.session_state["_sim_twin"] = twin
    st.session_state["spread_mode"] = SPREAD_MODES[0]
    st.session_state.pop("applied_geo_result", None)               # a new scenario invalidates the old run
    st.session_state["_applied_set_sync"] = True
    log_action("scenario", f"Scenario sent to Spread Simulation ({cfg['sim_mode']}): {cfg['forest']} "
               f"({cfg['latitude']:.4f}, {cfg['longitude']:.4f}), {cfg['width_m']:.0f}×{cfg['height_m']:.0f} m, "
               f"{duration_label(cfg['duration_min'])}, {cfg['temp_c']}°C, {cfg['humidity_pct']}% RH, "
               f"{cfg['wind_speed_ms']} m/s from {cfg['wind_from_deg']}°", twin.region.name)
    st.switch_page(SPREAD_PAGE)


# ── Spread side ───────────────────────────────────────────────────────────── #

def render_applied_scenario_bar(cfg: dict):
    ts = format_ist(cfg.get("timestamp"))
    risk = cfg.get("risk", {})
    loc = cfg.get("selected_location", {})
    live = cfg.get("live")
    src = {"preset": "preset forest", "google_geocoding": "Google Maps search", "google_places": "Google Maps search",
           "map": "custom map area", "highest_risk": "highest-risk zone"}.get(loc.get("source"), "preset forest")

    def num(v, fmt):
        return "n/a" if v is None else format(float(v), fmt)
    wdir = cfg.get("wind_from_deg")
    wdir_txt = "n/a" if wdir is None else f"{_compass_direction_name(float(wdir))} {float(wdir):.0f}°"
    if live:
        fc = live.get("firms", {})
        last = ("FIRMS detections", f"{fc.get('n', 0)} ({live.get('classification', {}).get('n_valid', 0)} ignition)"
                if fc.get("status") in ("live", "cached") else "UNAVAILABLE")
    else:
        last = ("Hypothetical hotspots", f"{cfg.get('n_hotspots')} (scenario)")
    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi"><div class="lbl">Location</div><div class="val" style="font-size:15px">{cfg.get('forest')}</div></div>
      <div class="kpi"><div class="lbl">Temperature</div><div class="val">{num(cfg.get('temp_c'), '.1f')}°C</div></div>
      <div class="kpi"><div class="lbl">Humidity</div><div class="val">{num(cfg.get('humidity_pct'), '.0f')}%</div></div>
      <div class="kpi"><div class="lbl">Wind</div><div class="val">{num(cfg.get('wind_speed_ms'), '.1f')} m/s</div></div>
      <div class="kpi"><div class="lbl">Wind from</div><div class="val">{wdir_txt}</div></div>
      <div class="kpi"><div class="lbl">{last[0]}</div><div class="val" style="font-size:16px">{last[1]}</div></div>
      <div class="kpi"><div class="lbl">Zone risk</div><div class="val">{risk.get('zone_risk', 0):.0%}</div></div>
    </div>
    """, unsafe_allow_html=True)
    if live:
        from src.dashboard.live_modes import MODE_NAMES
        w = live.get("weather", {})
        kind = ("LIVE WEATHER (real OpenWeatherMap observation)" if live["mode"] == "live"
                else "WHAT-IF SCENARIO INPUT on a real OpenWeatherMap baseline")
        st.caption(f"Mode: {MODE_NAMES.get(live['mode'], live['mode'])} · applied {ts} · {src} · "
                   f"{cfg.get('latitude'):.4f}°N, {cfg.get('longitude'):.4f}°E · {cfg.get('region')} zone "
                   f"{risk.get('zone_id')} ({risk.get('zone_severity')}) · {kind} · weather "
                   f"{str(w.get('status', '-')).upper()}")
    else:
        st.caption(f"Scenario input applied {ts} · {src} · {cfg.get('latitude'):.4f}°N, {cfg.get('longitude'):.4f}°E · "
                   f"{cfg.get('region')} zone {risk.get('zone_id')} ({risk.get('zone_severity')}) · synthetic scenario weather")


def render_geo_spread(twin, key_prefix: str, wind_speed_ms: float, wind_from_deg: float, wind_label: str,
                      setup: dict, wind_schedule_15min=None, scenario: Optional[dict] = None,
                      live: Optional[dict] = None):
    """Google satellite map + CA on the simulation domain + VFX + analytics.

    live (applied What-If live mode, from simulation_config["live"]): the run
    ignites ONLY valid observed NASA FIRMS cells (LIVE), or the WHAT-IF
    ignition source (observed / hypothetical); with no ignition source nothing
    is simulated and the status says why."""
    from src.dashboard import live_modes as lm
    snap = twin.current_snapshot
    kp = key_prefix
    lmode = (live or {}).get("mode") if (live or {}).get("mode") in (lm.LIVE, lm.WHATIF) else None
    st.markdown('<div class="sec-hdr">Geographic fire spread — Google satellite</div>', unsafe_allow_html=True)

    with st.expander("Simulation area, duration and ignition", expanded=False):
        err = render_setup_controls(twin, setup, f"{kp}_set", show_location=(kp != "applied"), ignition_mode=lmode)
    if err:
        st.error(err)
        return
    render_setup_summary(setup, twin)

    f = focus_from_setup(setup)
    loc = setup["location"]
    cond = zone_conditions(snap.processed_grid, snap.risk_scores, f.lat, f.lon)
    plan = sev = None
    if lmode:
        sev = twin.alert_engine.classify(float(cond["risk_score"])) if np.isfinite(cond["risk_score"]) else "UNKNOWN"
        plan = lm.plan_ignition(lmode, setup, f, cond, live.get("detections"), live["firms"]["status"])

    sig = (round(f.lat, 6), round(f.lon, 6), f.width_m, f.height_m, f.cell_m, setup["duration_min"],
           setup["placement"], setup["n_ignition"], json.dumps(setup["ignition_points"]),
           round(float(wind_speed_ms), 2), round(float(wind_from_deg), 1), cond["zone_id"],
           round(cond["ffmc"], 3) if np.isfinite(cond["ffmc"]) else None, id(snap),
           lmode, setup.get("ignition_source") if lmode else None, setup.get("domain_preset"),
           setup.get("domain_w_m"), setup.get("domain_h_m"), setup.get("max_extent_m"))
    res_key, sig_key, seed_key = f"{kp}_geo_result", f"{kp}_geo_sig", f"{kp}_geo_seed"
    status_key = f"{kp}_geo_status"
    if st.session_state.get(sig_key) != sig:
        st.session_state[res_key] = None
        st.session_state[status_key] = None
        st.session_state[sig_key] = sig

    run_label = {lm.LIVE: "RUN LIVE SIMULATION", lm.WHATIF: "RUN WHAT-IF SIMULATION"}.get(lmode, "Run Simulation")
    b1, b2, b3, b4, _ = st.columns([1.3, 0.9, 1.3, 1.2, 2])
    run = b1.button(run_label, type="primary", key=f"{kp}_geo_run", use_container_width=True)
    if b2.button("Reset", key=f"{kp}_geo_reset", use_container_width=True):
        st.session_state[res_key] = None
        st.session_state[status_key] = None
    redraw = b3.button("New random draw", key=f"{kp}_geo_redraw", use_container_width=True)
    if b4.button("New scenario", key=f"{kp}_geo_new", use_container_width=True):
        st.switch_page(WHATIF_PAGE)
    if redraw:
        st.session_state[seed_key] = st.session_state.get(seed_key, 42) + 1
    seed = st.session_state.get(seed_key, 42)

    if (run or redraw) and plan is not None and plan["ign"]["source"] == "NONE":
        # No observed fire / no ignition source: nothing is simulated (no fire from weather or risk alone).
        st.session_state[res_key] = None
        st.session_state[status_key] = lm.result_status(lmode, live["firms"]["status"], plan["cls"], sev,
                                                        plan["ign"], ran=False)
        log_action("simulation", f"{lm.MODE_NAMES[lmode]} at {f.name}: {st.session_state[status_key]['title']} "
                   "(no ignition, nothing simulated)", twin.region.name)
    elif run or redraw:
        result = _run_simulation(f, cond, setup, plan, seed, wind_speed_ms, wind_from_deg, wind_schedule_15min)
        st.session_state[res_key] = result
        st.session_state[status_key] = (lm.result_status(lmode, live["firms"]["status"], plan["cls"], sev,
                                                         plan["ign"], ran=True) if plan is not None else None)
        st.session_state["last_geo_run"] = {"kp": kp}
        from src.dashboard.ui.global_ticker import render_global_ticker
        render_global_ticker()                      # show the new run in the global ticker
        fin = result.final
        ign_txt = f" [{plan['ign']['label']}]" if plan is not None else ""
        log_action("simulation", f"Geographic spread at {f.name} ({f.lat:.4f}, {f.lon:.4f}), "
                   f"{f.width_m:.0f}×{f.height_m:.0f} m, {duration_label(setup['duration_min'])}: "
                   f"{fin.get('burned', 0)} cells burned ({fin.get('burned_ha', 0):.2f} ha), "
                   f"{fin.get('burning', 0)} burning (wind {wind_speed_ms:.1f} m/s from {wind_from_deg:.0f}°)"
                   + ign_txt, twin.region.name)
    result = st.session_state.get(res_key)
    run_status = st.session_state.get(status_key)
    if plan is not None:
        shown = run_status or lm.assess(lmode, live["firms"]["status"], plan["cls"], sev, plan["ign"])
        st.markdown(lm.status_html(shown, plan["ign"]["label"]), unsafe_allow_html=True)

    # Real observations at the focus location (cached; not in demo mode). The map
    # shows real NASA FIRMS detections only; synthetic hotspots appear only in
    # demo mode and are labelled as such. An applied live mode shows exactly the
    # observations that were handed off.
    demo = live_offline()
    if lmode:
        fs_full = live["firms"].get("status_full") or {"mode": live["firms"]["status"]}
        live_data = (live["weather"].get("record"), live["weather"].get("status_full") or
                     {"mode": live["weather"]["status"]}, pd.DataFrame(), fs_full)
        hot = live.get("detections") or []
        hot_kind = "observed"
        hot_summary = (f"{len(hot)} observed" if hot else "none in window") if live["firms"]["status"] in \
            ("live", "cached") else "unavailable"
    else:
        live_data = live_observations(f.lat, f.lon, kp, demo)
        if demo:
            hot = hotspots_near(getattr(twin.ingestion, "last_hotspots", None), f.lat, f.lon)
            hot_kind, hot_summary = "synthetic", (f"{len(hot)} demo (synthetic)" if hot else "-")
        else:
            hot, hot_kind, hot_summary = map_hotspots(live_data[2], live_data[3])
    key = google_maps_key()
    if key:
        payload = _cached_sim_payload(kp, f, setup["duration_min"], result, wind_speed_ms, wind_from_deg, hot,
                              hot_kind, setup["n_ignition"], setup["placement"],
                              setup["ignition_points"], setup["layers"], autoplay=result is not None,
                              api_key=key, map_id=google_maps_map_id(), hotspot_summary=hot_summary,
                              study_region=_region_payload(), domain=domain_from_setup(setup),
                              hud=_hud_info(lmode, cond, twin, setup, plan, result, live),
                              **_remap_click_rules(_map_ignition_args(plan["ign"] if plan is not None else None,
                                                                      setup, plan["land"] if plan is not None
                                                                      else None),
                                                   plan["domain"] if plan is not None else None,
                                                   result.domain if result is not None else None))
        ev = new_event(render_fire_map(payload, key=f"{kp}_simmap"), f"{kp}_simmap")
        msgs: list = []
        reasons: list = []
        if ev and ev.get("kind") == "ignite" and apply_map_event(
                ev, setup, twin, whatif=lmode == lm.WHATIF, land=plan["land"] if plan is not None else None,
                messages=msgs, reasons=reasons):
            st.session_state[f"_{kp}_set_sync"] = True
            st.session_state[f"_{kp}_ign_msgs"] = sorted(set(reasons + [m for m in msgs if m != HYP_NON_FUEL]))
            st.rerun()
        for m in st.session_state.pop(f"_{kp}_ign_msgs", []) or []:
            st.warning(m)
    else:
        missing_key_card()

    st.caption(f"{wind_label}: {wind_speed_ms:.1f} m/s from the {_compass_direction_name(wind_from_deg)} "
               f"({wind_from_deg:.0f}°), pushing fire and smoke towards the "
               f"{_compass_direction_name(wind_from_deg + 180)}.")
    if result is not None:
        from src.dashboard.ui.alerts_ui import render_result_actions
        render_result_actions(_report_snapshot(result, cond, setup, lmode, scenario, live, twin, plan), kp)
    if lmode:
        cls = plan["cls"]
        render_live_conditions(f.lat, f.lon, loc.get("name", ""), kp, demo, data=live_data, refresh=False,
                               detections=live.get("detections"),
                               firms_rows=[("In simulation area", f"{cls['n_in_area']} · valid ignitions "
                                                                  f"{cls['n_valid']}")],
                               scenario_card=lm.scenario_card(lmode, live.get("scenario_values") or {},
                                                              [{**r, "changed": r.get("changed")} for r in
                                                               live.get("changes") or []], live.get("baseline"),
                                                              {"status": live["firms"]["status"],
                                                               "n_in_area": cls["n_in_area"]}, plan["ign"]))
        st.caption(f"Observations as handed off from the What-If Simulator (weather fetched "
                   f"{format_ist(live['weather'].get('fetched_utc'))}, FIRMS fetched "
                   f"{format_ist(live['firms'].get('fetched_utc'))}). Use "
                   "Refresh live data in the What-If Simulator for newer observations.")
    else:
        render_live_conditions(f.lat, f.lon, loc.get("name", ""), kp, demo, scenario=scenario, data=live_data)

    if result is None:
        if plan is None:
            st.markdown('<div class="info-box">Press <b>Run Simulation</b> to compute the spread from the ignition '
                        'shown on the map. Play, pause, speed and the simulation-time slider are on the map.</div>',
                        unsafe_allow_html=True)
        elif run_status is None:
            st.markdown(f'<div class="info-box">Press <b>{run_label}</b>. '
                        + ("It ignites only the valid observed NASA FIRMS cell(s) shown on the map."
                           if plan["ign"]["source"] == "OBSERVED_FIRMS" else
                           "It applies the HYPOTHETICAL ignition shown on the map." if plan["ign"]["source"] ==
                           "HYPOTHETICAL_USER" else "No ignition source: no fire spread will be simulated.")
                        + '</div>', unsafe_allow_html=True)
        _render_provenance(cond, None, f, setup, scenario, wind_label, live=live, plan=plan)
        return
    _render_land_cover_note(result)
    _render_domain_and_region(result)
    _render_analytics(result, setup)
    _render_exports(result, kp)
    _render_provenance(cond, result, f, setup, scenario, wind_label, live=live, plan=plan)
    _render_diagnostics(result)


def _risk_text(cond: dict, twin) -> Optional[str]:
    """ML-predicted risk of the zone (category + score) - PREDICTED, from the twin's model."""
    r = cond.get("risk_score")
    if r is None or not np.isfinite(r):
        return None
    try:
        cat = twin.alert_engine.classify(float(r))
    except Exception:
        cat = "-"
    return f"{cat} · {float(r) * 100:.0f}%"


def _hud_info(lmode, cond: dict, twin, setup: dict, plan, result, live) -> dict:
    from src.dashboard import live_modes as lm
    mode = {lm.LIVE: "LIVE", lm.WHATIF: "WHAT-IF"}.get(lmode, "SCENARIO")
    pts = (plan["ign"]["points"] if plan is not None else setup.get("ignition_points")) or []
    if pts:
        ign = f"{len(pts)} pt · {pts[0][0]:.4f}°, {pts[0][1]:.4f}°"
    else:
        ign = f"{setup.get('n_ignition')} cell(s) · {setup.get('placement')}"
    stale = None
    if live and (live.get("weather") or {}).get("status") not in (None, "live"):
        stale = f"Weather: {(live.get('weather') or {}).get('status')}"
    if live and (live.get("firms") or {}).get("status") not in (None, "live"):
        stale = ((stale + " · ") if stale else "") + f"FIRMS: {(live.get('firms') or {}).get('status')}"
    return {"modeLabel": mode, "riskText": _risk_text(cond, twin), "ignitionText": ign,
            "reportId": (result.params.get("report_id") if result is not None else None),
            "staleNote": (stale + " (not live)") if stale else None}


def _cached_sim_payload(kp: str, *args, **kwargs) -> dict:
    """sim_payload is rebuilt only when its inputs change (a rerun from a widget elsewhere on the page,
    or a map event, reuses the serialised payload instead of re-deriving thousands of cells)."""
    result = args[2] if len(args) > 2 else kwargs.get("result")
    key = json.dumps([id(result), repr(getattr(result, "timings", None)), repr(args[:2]), repr(args[3:]), repr(sorted((k, repr(v)) for k, v in kwargs.items()))],
                     default=str)
    box = st.session_state.get(f"_{kp}_payload_cache")
    if box and box[0] == key:
        return box[1]
    p = sim_payload(*args, **kwargs)
    st.session_state[f"_{kp}_payload_cache"] = (key, p)
    return p


def _remap_cells(cells, src, dst) -> list:
    """Flat cell indices of domain `src` -> the same cells in domain `dst`
    (same lattice; dst may have grown on any side)."""
    dr, dc = dst.m_north - src.m_north, dst.m_west - src.m_west
    out = []
    for i in cells:
        r, c = divmod(int(i), src.n_cols)
        r, c = r + dr, c + dc
        if 0 <= r < dst.n_rows and 0 <= c < dst.n_cols:
            out.append(r * dst.n_cols + c)
    return out


def _remap_click_rules(args: dict, src, dst) -> dict:
    """The map's click rules (non-fuel cells) were computed on the set-up
    coverage; after a run the map shows the run's final domain."""
    if src is None or dst is None or (src.margin_tuple == dst.margin_tuple and src.n_cols == dst.n_cols):
        return args
    args = dict(args)
    if args.get("nonfuel_cells"):
        args["nonfuel_cells"] = _remap_cells(args["nonfuel_cells"], src, dst)
    if args.get("nonfuel_classes"):
        args["nonfuel_classes"] = {k: _remap_cells(v, src, dst) for k, v in args["nonfuel_classes"].items()}
    return args


def _region_payload():
    from src.dashboard.geo_fire_map import study_region_payload
    return study_region_payload(study_region())


def _run_simulation(f: FocusArea, cond: dict, setup: dict, plan: Optional[dict], seed: int,
                    wind_speed_ms: float, wind_from_deg: float, wind_schedule_15min):
    """The production run (WHAT-IF and LIVE): the existing CA with its own
    fire-behaviour inputs (zone FFMC dryness, BUI build-up, fuel, wind, DEM
    slope), the ignition-aware initial domain (audit BUG #1) and adaptive domain
    expansion. NO land-cover acquisition, fusion or fuel mask: the ignition-point
    check already happened when the point was placed, and fire propagation is
    not restricted by land cover."""
    from src.utils.timing import Timings
    strict = plan is not None and (plan["ign"]["source"] == "OBSERVED_FIRMS" or plan["ign"].get("manual"))
    pts = plan["ign"]["points"] if strict else (setup["ignition_points"] if setup["placement"] == "Map points"
                                                 else [])
    tm = Timings()
    dom0 = domain_from_setup(setup, pts)
    with st.status("Running the simulation…", expanded=False) as _status:
        status = _status if _status is not None else type("NoStatus", (), {"update": lambda *a, **k: None})()
        status.update(label="Terrain: cached DEM lattice (network only for new areas, ≤ 12 s)…")
        with tm.stage("terrain"):
            elev, terrain_src = domain_elevation(dom0, allow_fetch=True)   # cached DEM lattice

        def elevation_provider(spec):
            # strips added while the fire spreads: cached DEM only (never a network wait mid-run)
            return domain_elevation(spec, allow_fetch=False)
        common = dict(seed=seed, elevation=elev, terrain_source=terrain_src, wind_schedule_15min=wind_schedule_15min,
                      duration_minutes=setup["duration_min"], domain=dom0, elevation_provider=elevation_provider,
                      study_region=study_region(), timings=tm, limits=limits_from_setup(setup))
        status.update(label=f"Fire spread: {duration_label(setup['duration_min'])} of simulated time…")
        if strict:
            # observed FIRMS cells, or exactly the user's hypothetical map points: no other cell ignites
            res = run_local_spread(f, cond, wind_speed_ms, wind_from_deg, n_ignition=0, placement="Map points",
                                   ignition_points=plan["ign"]["points"], strict_points=True, **common)
        else:
            res = run_local_spread(f, cond, wind_speed_ms, wind_from_deg, n_ignition=setup["n_ignition"],
                                   placement=setup["placement"], ignition_points=setup["ignition_points"], **common)
        t = tm.as_dict()
        status.update(label=f"Simulation complete in {t.get('total', 0):.1f} s (terrain {t.get('terrain', 0):.1f} s, "
                            f"fire model {t.get('ca_simulation', 0):.2f} s)", state="complete")
    res.params["ignition_checks"] = list(setup.get("ignition_checks") or [])
    res.params["report_id"] = new_report_id()
    res.timings = tm.as_dict()
    return res


def _report_snapshot(result, cond, setup, lmode, scenario, live, twin, plan) -> Optional[dict]:
    """The reproducible snapshot of this completed run (built once per report ID, stored in the database)."""
    from src.dashboard import live_modes as lm
    rid = result.params.get("report_id")
    if not rid:
        return None
    box = st.session_state.setdefault("_report_snapshots", {})
    if rid not in box:
        from src.reports.simulation_report import build_snapshot
        mode = {lm.LIVE: "LIVE", lm.WHATIF: "WHAT-IF"}.get(lmode, "SCENARIO")
        snap = build_snapshot(result, cond, setup, mode, scenario if mode != "LIVE" else None, live, twin, plan,
                              actor=st.session_state.get("auth_user", ""))
        box[rid] = snap
        try:
            from src.notifications.alert_store import save_report_snapshot
            save_report_snapshot(rid, snap, actor=st.session_state.get("auth_user", ""))
        except Exception as exc:                       # storage failure never hides the report
            logging.getLogger(__name__).warning("report snapshot not stored: %s", exc)
    return box[rid]


def new_report_id() -> str:
    """Unique ID of a completed simulation (used by the report and any alert about it)."""
    import secrets
    return "FFDT-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2).upper()


def _render_domain_and_region(result):
    """Adaptive domain, extent limit, Bandipur boundary crossing - never hidden."""
    if result.extent_limit:
        lim = result.extent_limit
        st.error(f"**{EXTENT_LIMIT_TITLE}** at T+{lim.get('minutes', 0):g} min ({lim.get('side', '')} edge): "
                 f"{lim['reason']}. The fire did not stop naturally there - it was still able to spread.")
    if result.expansions:
        e = result.expansions[-1]
        st.info(f"Simulation domain expanded {len(result.expansions)}× while the fire spread "
                f"(initial {result.initial_domain.width_m:.0f} × {result.initial_domain.height_m:.0f} m → "
                f"{result.domain.width_m:.0f} × {result.domain.height_m:.0f} m, last at T+{e['minutes']:g} min). "
                "Land cover and terrain were loaded for every new strip; the fire state continued without restart.")
    reg = result.study_region
    if reg:
        tag = "" if reg["status"] == "OFFICIAL" else f" ({reg['status']} boundary)"
        fin = result.final
        if reg.get("ignition_outside_region"):
            st.warning(f"The ignition lies outside the {reg['name']} boundary{tag}: this run is in the "
                       "REGIONAL EXTENSION, not in the primary study region.")
        elif reg.get("crossed"):
            st.warning(f"Fire reached and crossed the {reg['name']} boundary{tag} at T+{reg['crossing_minutes']:g} min "
                       f"({reg.get('crossing_lat')}, {reg.get('crossing_lon')}). The simulation continued into the "
                       "REGIONAL EXTENSION (outside the reserve).")
        st.caption(f"{reg['name']}{tag}: burned {fin.get('burned_ha_bandipur', 0):.2f} ha inside · "
                   f"REGIONAL EXTENSION {fin.get('burned_ha_regional_extension', 0):.2f} ha (kept separate).")


def explain_cell(result, lat: float, lon: float) -> Optional[dict]:
    """Why can (or can't) this cell burn? Every value is read from the run's own
    inputs (fuel grid, conditions, terrain, wind). Fire propagation is not
    restricted by land cover; a caller-supplied land-cover layer is shown if
    the run had one."""
    rc = result.domain.cell_of(float(lat), float(lon))
    if rc is None:
        return None
    r, c = rc
    lc = result.land_cover
    cls = int(lc[rc]) if lc is not None else None
    e = result.elevation
    nb = [float(e[r + dr, c + dc]) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
          if (dr or dc) and 0 <= r + dr < e.shape[0] and 0 <= c + dc < e.shape[1]]
    slope = max(abs(float(e[rc]) - v) for v in nb) / result.focus.cell_m * 100 if nb else 0.0
    p = result.params
    k = max(0, int(result.ignition_step[rc]) - 1) if result.ignition_step[rc] > 0 else 0
    w = result.wind_schedule[min(k, len(result.wind_schedule) - 1)]
    return {"cell (row, col)": f"{r}, {c}",
            "land cover": (CLASS_NAMES.get(cls, "-") if cls is not None
                           else "not used for fire spread (ignition-point check only)"),
            "fuel load (relative 0-1)": round(float(result.fuel_load[rc]), 3) if result.fuel_load is not None else "-",
            "fuel-load source": p.get("fuel_load_source", "-"),
            "FFMC dryness (zone)": round(float(p.get("dryness", float("nan"))) * 101, 1),
            "BUI build-up factor (zone)": round(float(p.get("buildup", float("nan"))), 2),
            "wind": f"{w[0]:.1f} m/s from {_compass_direction_name(w[1])} ({w[1]:.0f}°)",
            "max slope to a neighbour": f"{slope:.1f} %",
            "burnability": "NO" if bool(result.non_fuel[rc]) else "YES",
            "simulated ignition": ((f"T+{round(float(result.ignition_time[rc]), 1):g} min"
                                    if getattr(result, "ignition_time", None) is not None
                                    else f"T+{result.ignition_step[rc] * result.step_minutes:g} min")
                                   if result.ignition_step[rc] >= 0 else "did not burn"),
            **({"fireline intensity": f"{result.fireline_kw[rc]:,.0f} kW/m (SIMULATED)"}
               if getattr(result, "fireline_kw", None) is not None and result.ignition_step[rc] >= 0 else {})}


def _render_diagnostics(result):
    """Developer / debug information: timings, expansions, land-cover sources."""
    with st.expander("Why can this cell burn? (cell inspector)"):
        f = result.focus
        c1, c2 = st.columns(2)
        la = c1.number_input("Latitude", value=float(f.lat), format="%.6f", key=f"insp_lat_{id(result)}")
        lo = c2.number_input("Longitude", value=float(f.lon), format="%.6f", key=f"insp_lon_{id(result)}")
        info = explain_cell(result, la, lo)
        if info is None:
            st.caption("That point is outside the simulation domain.")
        else:
            st.dataframe(pd.DataFrame([(k, str(v)) for k, v in info.items()], columns=["", "value"]),
                         hide_index=True, use_container_width=True)
    with st.expander("Developer diagnostics: timings and domain expansion"):
        t = result.timings or {}
        st.dataframe(pd.DataFrame([{"stage": k, "seconds": v} for k, v in t.items()]), hide_index=True,
                     use_container_width=True)
        st.json({"initial_domain": result.params.get("initial_domain"), "final_domain": result.params.get("final_domain"),
                 "expansions": result.expansions, "extent_limit": result.extent_limit,
                 "fuel_load_source": result.params.get("fuel_load_source"),
                 "fuel_variability": result.params.get("fuel_variability"),
                 "model": {k: result.params.get(k) for k in (
                     "model", "fuel_model", "ros_head_m_per_min", "ros_flank_m_per_min", "ros_back_m_per_min",
                     "length_to_breadth", "isi", "wind_kmh", "frame_minutes", "substep_minutes", "n_substeps",
                     "extinguished_at_min", "neighbourhood", "flame_min", "residence_max_min", "threshold_jitter",
                     "wind_model", "diagonal_correction", "max_spread_m_per_min")},
                 "ignition_checks": result.params.get("ignition_checks"),
                 "study_region": result.study_region}, expanded=False)


def land_cover_summary(result) -> Optional[str]:
    """One line describing the fuel mask the run used (None without a layer)."""
    lc = getattr(result, "land_cover", None)
    if lc is None:
        return None
    if not (result.land_cover_label.startswith("OpenStreetMap") or getattr(result, "land_status", "")):
        return result.land_cover_label
    parts = [f"{int((lc == k).sum())} {name}" for k, name in CLASS_NAMES.items() if k != FUEL and (lc == k).any()]
    return (f"{result.land_cover_label}: " + (", ".join(parts) + " cells excluded from fire spread"
                                               if parts else "no water, road or built-up cells in this domain"))


def _render_land_cover_note(result):
    """One line on how land cover was (not) used by this run."""
    if result.land_cover is not None:                     # a caller-supplied mask (API / tests)
        st.caption((land_cover_summary(result) or result.land_cover_label) + ".")
        return
    checks = result.params.get("ignition_checks") or []
    txt = ("Fire spread is computed by the cellular automata from wind, FFMC (temperature / humidity), BUI, fuel and "
           "slope; it is not restricted by land cover.")
    if checks:
        txt += " Ignition-point check: " + ", ".join(f"{c['label']}" for c in checks) + "."
    st.caption(txt)


def _render_analytics(result, setup: dict):
    m = result.metrics
    fin = result.final
    max_ros = max(x["ros_m_per_min"] for x in m)
    beyond_t = next((x["minutes"] for x in m if x["left_focus"]), None)
    ended = fin["burning"] == 0
    d0 = m[0]["front_distance_m"]                          # radius of the ignition cluster itself
    mean_ros = (fin["front_distance_m"] - d0) / fin["minutes"] if fin["minutes"] else 0.0
    max_int = max(x["max_intensity"] for x in m)
    max_kw = max((x.get("max_intensity_kw_m") or 0.0) for x in m)
    p = result.params
    ext = p.get("extinguished_at_min")
    fcells = result.focus.n_rows * result.focus.n_cols
    pct_focus = 100.0 * fin["burned_in_focus"] / fcells if fcells else 0.0
    dur = setup["duration_min"]
    cards = [("Burned area", f"{fin['burned_ha']:.2f} ha", "crit"), ("Focus area burned", f"{pct_focus:.0f}%", ""),
             ("Fire perimeter", f"{fin['perimeter_m']:.0f} m", ""),
             ("Max spread distance", f"{fin['front_distance_m']:.0f} m", ""),
             ("Mean rate of spread", f"{mean_ros:.1f} m/min", "warn"), ("Peak rate of spread", f"{max_ros:.1f} m/min", "warn"),
             ("Max fire intensity", (f"{max_kw:,.0f} kW/m" if max_kw else f"{max_int * 100:.0f}%"), "warn"),
             ("Fire state", (f"Out at T+{ext:g} min" if ext is not None else f"Out by T+{fin['minutes']:.0f} min")
              if ended else f"Active at T+{min(dur, fin['minutes']):g} min", "ok" if ended else "crit"),
             ("Simulated time", f"{duration_label(fin['minutes'])} of {duration_label(dur)}"
              + (f" · computed in {result.timings.get('total', 0):.1f} s" if result.timings else ""), "ok"),
             ("Beyond focus area", f"From T+{beyond_t:.0f} min" if beyond_t is not None else "No",
              "warn" if beyond_t is not None else "ok")]
    st.markdown('<div class="kpi-row">' + "".join(
        f'<div class="kpi"><div class="lbl">{a}</div><div class="val {c}" style="font-size:19px">{b}</div></div>'
        for a, b, c in cards) + "</div>", unsafe_allow_html=True)
    if p.get("model") == "ros-ca":
        st.caption(f"Fire model: rate-of-spread cellular automaton (Canadian FBP System equations; fuel "
                   f"{p.get('fuel_model')}). Head-fire ROS {p.get('ros_head_m_per_min', 0):.1f} m/min, flank "
                   f"{p.get('ros_flank_m_per_min', 0):.1f}, back {p.get('ros_back_m_per_min', 0):.2f} m/min, fire-ellipse "
                   f"length:breadth {p.get('length_to_breadth', 1):.1f} (ISI {p.get('isi', 0):.1f}, wind "
                   f"{p.get('wind_kmh', 0):.0f} km/h). SIMULATED and uncalibrated for Bandipur: a physically consistent "
                   f"what-if, not an operational forecast. Frames every {p.get('frame_minutes', 0):g} min, "
                   f"{p.get('n_substeps', 0)} internal steps of {60 * (p.get('substep_minutes') or 0):.0f} s. "
                   "Playback speed (0.5×–5×) only changes how fast these computed frames are shown.")
    if fin.get("centroid_lat") is not None:
        st.caption(f"Final fire centroid {fin['centroid_lat']:.5f}°N, {fin['centroid_lon']:.5f}°E · "
                   f"simulation domain {result.domain.width_m:.0f} × {result.domain.height_m:.0f} m "
                   f"(margins N/S/W/E {', '.join(str(m) for m in result.domain.margin_tuple)} cells; it grows "
                   "while the fire spreads, so the fire is never stopped by a box)")

    t = [x["minutes"] for x in m]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=[x["burned_ha"] for x in m], name="Burned area (ha)",
                             line=dict(color="#ff6b4a", width=2), fill="tozeroy", fillcolor="rgba(255,107,74,0.12)"))
    fig.add_trace(go.Scatter(x=t, y=[x["burning"] * result.focus.cell_m ** 2 / 1e4 for x in m],
                             name="Burning area (ha)", line=dict(color="#fbbf24", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=t, y=[x["ros_m_per_min"] for x in m], name="Rate of spread (m/min)",
                             line=dict(color="#8FD3FF", width=1.5), yaxis="y2"))
    fig.update_layout(height=230, margin=dict(l=0, r=0, t=28, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(title="Simulation time (min)", tickfont={"color": "#8a96a6", "size": 10},
                                 gridcolor="#232b36"),
                      yaxis=dict(title="ha", tickfont={"color": "#8a96a6", "size": 10}, gridcolor="#232b36"),
                      yaxis2=dict(title="m/min", overlaying="y", side="right", showgrid=False,
                                  tickfont={"color": "#8a96a6", "size": 10}),
                      legend=dict(orientation="h", y=1.12, x=0, font={"color": "#8a96a6", "size": 10},
                                  bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)


def _render_exports(result, kp: str):
    """Per-step analytics as CSV and the burned cells as GeoJSON polygons."""
    df = pd.DataFrame(result.metrics)
    dom = result.domain
    lat_c, lon_c = dom.cell_centres()
    hl, hw = dom.focus.dlat / 2, dom.focus.dlon / 2
    feats = []
    for r, c in zip(*np.where(result.ignition_step >= 0)):
        la, lo = float(lat_c[r, c]), float(lon_c[r, c])
        feats.append({"type": "Feature",
                      "properties": {"ignition_min": round(float(result.ignition_time[r, c]), 2)
                                     if getattr(result, "ignition_time", None) is not None
                                     else round(float(result.ignition_step[r, c]) * result.step_minutes, 2),
                                     "intensity": round(float(result.intensity[r, c]), 3), "simulated": True},
                      "geometry": {"type": "Polygon", "coordinates": [[[lo - hw, la - hl], [lo + hw, la - hl],
                                                                       [lo + hw, la + hl], [lo - hw, la + hl],
                                                                       [lo - hw, la - hl]]]}})
    gj = {"type": "FeatureCollection", "name": f"simulated_burn_{result.focus.name}", "features": feats}
    e1, e2, _ = st.columns([1, 1, 3])
    e1.download_button("Export analytics (CSV)", df.to_csv(index=False).encode(), "spread_simulation_steps.csv",
                       "text/csv", key=f"{kp}_dl_csv", use_container_width=True)
    e2.download_button("Export burned area (GeoJSON)", json.dumps(gj).encode(), "simulated_burned_area.geojson",
                       "application/geo+json", key=f"{kp}_dl_geo", use_container_width=True)


def _fuel_row(result, cond: dict):
    """Provenance of the CA fuel load."""
    if result is None:
        return ("Fuel load", f"Zone NDVI estimate {cond['ndvi']:.2f} (zone {cond['zone_id']}) with small simulated "
                             "cell-to-cell variability", "SYNTHETIC")
    src = result.params.get("fuel_load_source", "")
    if src.startswith("land cover"):
        return ("Fuel load", f"Per-cell, {src}", "DERIVED")
    var = result.params.get("fuel_variability") or 0
    return ("Fuel load", f"Zone NDVI estimate {cond['ndvi']:.2f} (zone {cond['zone_id']})"
                         + (f" with ±{var * 100:.0f}% simulated cell-to-cell variability (natural patchiness; not "
                            "observed)" if var else ", uniform"), "SYNTHETIC")


def _mask_type(result) -> str:
    if result is None or result.land_cover is None:
        return "NOT USED FOR SPREAD"
    return "CALLER-SUPPLIED MASK"


def _region_row(result) -> str:
    reg = study_region()
    if reg is None:
        return "No study-region boundary file"
    txt = f"{reg.name}: {reg.status} boundary ({reg.source}); polygon area {reg.area_km2:.0f} km²"
    if reg.stated_area_km2:
        txt += f" (stated reserve area {reg.stated_area_km2:,.0f} km²)"
    return txt


def _render_provenance(cond: dict, result, f: FocusArea, setup: dict, scenario: Optional[dict], wind_label: str,
                       live: Optional[dict] = None, plan: Optional[dict] = None):
    p = result.params if result is not None else {}
    lmode = (live or {}).get("mode") if plan is not None else None
    if result is not None:
        e = result.elevation
        terrain = (f"Real DEM (SRTM 90 m via Open-Topo-Data / Open-Meteo), {e.min():.0f}–{e.max():.0f} m"
                   if result.terrain_source == "dem" else "Not available - flat terrain assumed")
    else:
        terrain = "Fetched when the simulation runs"
    loc = setup["location"]
    src = {"preset": "Preset forest coordinates", "google_geocoding": "Google Geocoding search result",
           "google_places": "Google Places search result", "map": "Area placed on the Google map",
           "highest_risk": "Centre of the highest-risk model zone"}.get(loc.get("source"), "Preset")
    rows = [
        ("Real geographic base", "Google satellite imagery", "REAL"),
        ("Location", f"{loc['name']} - {src}", "REAL"),
        ("Terrain", terrain, "REAL" if result is not None and result.terrain_source == "dem" else "-"),
        ("Weather", (f"{scenario.get('temp_c')}°C, {scenario.get('humidity_pct')}% RH (What-If scenario)" if scenario
                     else f"{cond['temp_c']:.1f}°C, {cond['humidity_pct']:.0f}% RH (zone {cond['zone_id']})"),
         {"live": "LIVE WEATHER (REAL OBSERVATION)", "whatif": "SCENARIO INPUT (WHAT-IF)"}.get(
             lmode, "SCENARIO INPUT" if scenario else "LIVE / DEMO")),
        ("Wind", wind_label, {"live": "LIVE WEATHER (REAL OBSERVATION)", "whatif": "SCENARIO INPUT (WHAT-IF)"}.get(
            lmode, "SCENARIO INPUT" if scenario else "LIVE / DEMO")),
        ("FFMC / BUI / FWI", f"{cond['ffmc']:.1f} / {cond['bui']:.1f} / {cond['fwi']:.1f} (zone {cond['zone_id']})", "DERIVED"),
        ("Model risk", f"{cond['risk_score']:.0%}" if np.isfinite(cond["risk_score"]) else "-", "DERIVED (XGBoost)"),
        _fuel_row(result, cond),
        ("Land cover", "Not used for fire propagation. A lightweight OpenStreetMap check of the exact clicked point "
                       "rejects hypothetical ignitions on mapped roads, buildings / built-up areas, water and bare "
                       "ground; unmapped points are LOCATION UNVERIFIED.", _mask_type(result)),
        ("Study region", _region_row(result), "REAL BOUNDARY" if (result is not None and result.study_region and
                                                                   result.study_region["status"] == "OFFICIAL")
         else "APPROXIMATE BOUNDARY"),
        ("Grid", f"{f.cell_m:.0f} m cells, CA step {step_minutes_for(f.cell_m):g} min", "COMPUTATIONAL MODEL"),
        ("Fire spread / burned area", "FireSpreadSimulator (cellular automata)"
         + (f", base spread probability {p.get('base_spread_prob')} (local calibration), seed {p.get('seed')}" if p else ""),
         "SIMULATION OUTPUT"),
        ("Flames, smoke, embers, ash", "Rendered from the simulated cell states", "SIMULATED VISUALIZATION"),
    ]
    if lmode:
        from src.dashboard.live_modes import MODE_NAMES

        def _ts(v):
            return format_ist(v)
        wv, fv, ign = live["weather"], live["firms"], plan["ign"]
        cls = plan["cls"]
        b = (live.get("baseline") or {})
        rows = [("Mode", MODE_NAMES[lmode], "LIVE REAL-WORLD" if lmode == "live" else "WHAT-IF")] + rows + [
            ("Weather source", f"OpenWeatherMap · {str(wv.get('status')).upper()} · fetched {_ts(wv.get('fetched_utc'))}"
                               f" · observed {_ts(wv.get('observed_utc'))}", "REAL OBSERVATION"),
            ("Real baseline", (", ".join(f"{k} {v}" for k, v in (b.get("values") or {}).items()) or "unavailable"),
             "REAL OBSERVATION" if b else "-"),
            ("NASA FIRMS", f"{fv.get('source')} · {str(fv.get('status')).upper()} · fetched {_ts(fv.get('fetched_utc'))} · "
                           f"{fv.get('n', 0)} detection(s) in the search box, {cls['n_in_area']} in the area, "
                           f"{cls['n_valid']} valid", "NASA FIRMS OBSERVATION"),
            ("Ignition source", {"OBSERVED_FIRMS": f"OBSERVED FIRMS ({cls['n_cells']} cell(s))",
                                 "HYPOTHETICAL_USER": f"HYPOTHETICAL USER ({ign.get('placement')})",
                                 "NONE": "NONE (no fire simulated)"}[ign["source"]],
             {"OBSERVED_FIRMS": "OBSERVED FIRE", "HYPOTHETICAL_USER": "HYPOTHETICAL IGNITION", "NONE": "-"}[ign["source"]]),
            ("Simulation", "Cellular-automata fire spread" if result is not None else "Not run", "SIMULATED"),
        ]
    else:
        rows.append(("Ignition source", f"HYPOTHETICAL USER ({setup.get('placement')})", "HYPOTHETICAL IGNITION"))
    with st.expander("Data provenance: real, derived, scenario and simulated"):
        st.dataframe(pd.DataFrame(rows, columns=["Item", "Value", "Type"]), hide_index=True, use_container_width=True)
        st.caption("The 25 m spread probability is a local-scale calibration that has not been validated against "
                   "observed fire perimeters. Simulated fire is not an observed wildfire.")
