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
from src.simulation.local_spread import (DEFAULT_FOCUS, FOCUS_AREAS, HIGHEST_RISK_FOCUS, PLACEMENTS,
                                         FocusArea, domain_elevation, domain_for, focus_options_for_region,
                                         n_steps_for, resolve_focus, run_local_spread, step_minutes_for,
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
            "layers": {"grid": True, "boundary": True}}


def get_setup(twin, store_key: str = "sim_setup", kp: Optional[str] = None) -> dict:
    """The shared set-up; its location follows the sidebar region (a new
    region starts at that region's first preset)."""
    s = st.session_state.get(store_key)
    if not s or s.get("region") != twin.region.name:
        keep = {k: s[k] for k in ("width_m", "height_m", "cell_m", "duration_min", "layers")} if s else {}
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


def apply_map_event(ev: dict, setup: dict, twin) -> bool:
    """Apply an event from the map component to the set-up. True if changed."""
    kind = ev.get("kind")
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


def render_setup_controls(twin, setup: dict, kp: str, show_location: bool = True) -> Optional[str]:
    """Location / size / cell / duration / ignition / layer controls. Returns a
    validation error message or None."""
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

    q = st.columns(len(SIZE_PRESETS) + 1)
    q[0].caption("Quick size")
    for col, (lbl, (pw, ph)) in zip(q[1:], SIZE_PRESETS.items()):
        if col.button(lbl, key=f"{kp}_sz_{lbl}", use_container_width=True):
            setup.update(width_m=float(pw), height_m=float(ph))
            st.session_state[f"_{kp}_sync"] = True
            st.session_state[f"{kp}_sizemode"] = "Width × height"
            st.rerun()

    i1, i2, i3, i4 = st.columns([1.4, 1, 0.8, 0.8])
    placement = i1.selectbox("Ignition", PLACEMENTS, key=f"{kp}_place",
                             help="Upwind edge: the head fire runs across the area. Map points: use 'Set ignition on "
                                  "map' and click the map.")
    n_ign = i2.slider("Ignition cells", 1, 9, key=f"{kp}_nign", disabled=placement == "Map points")
    grid = i3.toggle("Grid", key=f"{kp}_grid")
    boundary = i4.toggle("Boundary", key=f"{kp}_bound")

    cell = float(cell)
    setup.update(width_m=_snap(float(w), cell), height_m=_snap(float(h), cell), cell_m=cell, duration_min=dur,
                 placement=placement, n_ignition=int(n_ign), layers={"grid": bool(grid), "boundary": bool(boundary)})
    if placement == "Map points" and not setup["ignition_points"]:
        st.caption("No map points yet: click **Set ignition on map** on the map, then click inside the area. "
                    "Until then the centre is used.")
    return validate_setup(setup["width_m"], setup["height_m"], cell, dur)


def render_setup_summary(setup: dict, twin):
    f = focus_from_setup(setup)
    dur = setup["duration_min"]
    dom = domain_for(f, dur)
    step = step_minutes_for(f.cell_m)
    cards = [("Width", f"{f.n_cols * f.cell_m:.0f} m"), ("Height", f"{f.n_rows * f.cell_m:.0f} m"),
             ("Area", f"{f.area_km2:.2f} km²"), ("Cell", f"{f.cell_m:.0f} m"), ("Grid", f"{f.n_cols} × {f.n_rows}"),
             ("Duration", duration_label(dur)), ("CA step", f"{step * 60:.0f} s" if step < 1 else f"{step:g} min"),
             ("Simulation domain", f"{dom.width_m / 1000:.2f} × {dom.height_m / 1000:.2f} km")]
    st.markdown('<div class="kpi-row">' + "".join(
        f'<div class="kpi"><div class="lbl">{a}</div><div class="val" style="font-size:18px">{b}</div></div>'
        for a, b in cards) + "</div>", unsafe_allow_html=True)
    if dur < step:
        st.info(f"{duration_label(dur)} is shorter than one CA step ({step:g} min at {f.cell_m:.0f} m cells). At the "
                f"model's spread rate (100 m per 15 min) the fire moves about {dur * 100 / 15:.0f} m in that time, "
                f"less than one cell, so only the ignition will be burning. Use smaller cells or a longer duration "
                f"to see propagation.")
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


def render_setup_map(setup: dict, kp: str, twin, wind_speed_ms: float, wind_from_deg: float, height: int = 540):
    """Interactive set-up map: search, drag/resize box, click-to-ignite."""
    key = google_maps_key()
    if not key:
        missing_key_card()
        return
    f = focus_from_setup(setup)
    if validate_setup(f.width_m, f.height_m, f.cell_m, setup["duration_min"]):
        return
    payload = setup_payload(f, setup["duration_min"], wind_speed_ms, wind_from_deg, setup["n_ignition"],
                            setup["placement"], setup["ignition_points"], setup["layers"], key,
                            google_maps_map_id(), height=height)
    ev = new_event(render_fire_map(payload, key=f"{kp}_map"), f"{kp}_map")
    if ev and apply_map_event(ev, setup, twin):
        st.session_state[f"_{kp}_sync"] = True
        st.rerun()


# ── What-If -> Spread hand-off ────────────────────────────────────────────── #

def build_simulation_config(twin, scenario: dict, setup: dict) -> dict:
    """Everything Module 2 needs, taken from the scenario twin that Module 1
    just computed (no second weather or risk calculation)."""
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
    }


def apply_and_open_spread(twin, scenario: dict, setup: dict):
    cfg = build_simulation_config(twin, scenario, setup)
    st.session_state["simulation_config"] = cfg
    st.session_state["_sim_twin"] = twin
    st.session_state["spread_mode"] = SPREAD_MODES[0]
    st.session_state.pop("applied_geo_result", None)               # a new scenario invalidates the old run
    st.session_state["_applied_set_sync"] = True
    log_action("scenario", f"Scenario sent to Spread Simulation: {cfg['forest']} "
               f"({cfg['latitude']:.4f}, {cfg['longitude']:.4f}), {cfg['width_m']:.0f}×{cfg['height_m']:.0f} m, "
               f"{duration_label(cfg['duration_min'])}, {cfg['temp_c']}°C, {cfg['humidity_pct']}% RH, "
               f"{cfg['wind_speed_ms']} m/s from {cfg['wind_from_deg']}°", twin.region.name)
    st.switch_page(SPREAD_PAGE)


# ── Spread side ───────────────────────────────────────────────────────────── #

def render_applied_scenario_bar(cfg: dict):
    ts = cfg.get("timestamp", "")[:19].replace("T", " ")
    risk = cfg.get("risk", {})
    wdir = float(cfg.get("wind_from_deg", 0))
    loc = cfg.get("selected_location", {})
    src = {"preset": "preset forest", "google_geocoding": "Google Maps search", "google_places": "Google Maps search",
           "map": "custom map area", "highest_risk": "highest-risk zone"}.get(loc.get("source"), "preset forest")
    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi"><div class="lbl">Location</div><div class="val" style="font-size:15px">{cfg.get('forest')}</div></div>
      <div class="kpi"><div class="lbl">Temperature</div><div class="val">{cfg.get('temp_c')}°C</div></div>
      <div class="kpi"><div class="lbl">Humidity</div><div class="val">{cfg.get('humidity_pct')}%</div></div>
      <div class="kpi"><div class="lbl">Wind</div><div class="val">{float(cfg.get('wind_speed_ms') or 0):.1f} m/s</div></div>
      <div class="kpi"><div class="lbl">Wind from</div><div class="val">{_compass_direction_name(wdir)} {wdir:.0f}°</div></div>
      <div class="kpi"><div class="lbl">Seeded hotspots</div><div class="val">{cfg.get('n_hotspots')}</div></div>
      <div class="kpi"><div class="lbl">Zone risk</div><div class="val">{risk.get('zone_risk', 0):.0%}</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.caption(f"Scenario input applied {ts} UTC · {src} · {cfg.get('latitude'):.4f}°N, {cfg.get('longitude'):.4f}°E · "
               f"{cfg.get('region')} zone {risk.get('zone_id')} ({risk.get('zone_severity')}) · synthetic scenario weather")


def render_geo_spread(twin, key_prefix: str, wind_speed_ms: float, wind_from_deg: float, wind_label: str,
                      setup: dict, wind_schedule_15min=None, scenario: Optional[dict] = None):
    """Google satellite map + CA on the simulation domain + VFX + analytics."""
    snap = twin.current_snapshot
    kp = key_prefix
    st.markdown('<div class="sec-hdr">Geographic fire spread — Google satellite</div>', unsafe_allow_html=True)

    with st.expander("Simulation area, duration and ignition", expanded=False):
        err = render_setup_controls(twin, setup, f"{kp}_set", show_location=(kp != "applied"))
    if err:
        st.error(err)
        return
    render_setup_summary(setup, twin)

    f = focus_from_setup(setup)
    loc = setup["location"]
    cond = zone_conditions(snap.processed_grid, snap.risk_scores, f.lat, f.lon)

    sig = (round(f.lat, 6), round(f.lon, 6), f.width_m, f.height_m, f.cell_m, setup["duration_min"],
           setup["placement"], setup["n_ignition"], json.dumps(setup["ignition_points"]),
           round(float(wind_speed_ms), 2), round(float(wind_from_deg), 1), cond["zone_id"],
           round(cond["ffmc"], 3) if np.isfinite(cond["ffmc"]) else None, id(snap))
    res_key, sig_key, seed_key = f"{kp}_geo_result", f"{kp}_geo_sig", f"{kp}_geo_seed"
    if st.session_state.get(sig_key) != sig:
        st.session_state[res_key] = None
        st.session_state[sig_key] = sig

    b1, b2, b3, b4, _ = st.columns([1.3, 0.9, 1.3, 1.2, 2])
    run = b1.button("Run Simulation", type="primary", key=f"{kp}_geo_run", use_container_width=True)
    if b2.button("Reset", key=f"{kp}_geo_reset", use_container_width=True):
        st.session_state[res_key] = None
    redraw = b3.button("New random draw", key=f"{kp}_geo_redraw", use_container_width=True)
    if b4.button("New scenario", key=f"{kp}_geo_new", use_container_width=True):
        st.switch_page(WHATIF_PAGE)
    if redraw:
        st.session_state[seed_key] = st.session_state.get(seed_key, 42) + 1
    seed = st.session_state.get(seed_key, 42)

    if run or redraw:
        with st.spinner("Simulating fire spread on the simulation domain..."):
            dom = domain_for(f, setup["duration_min"])
            elev, terrain_src = domain_elevation(dom, allow_fetch=True)
            result = run_local_spread(f, cond, wind_speed_ms, wind_from_deg, n_ignition=setup["n_ignition"],
                                      placement=setup["placement"], seed=seed, elevation=elev,
                                      terrain_source=terrain_src, wind_schedule_15min=wind_schedule_15min,
                                      duration_minutes=setup["duration_min"],
                                      ignition_points=setup["ignition_points"])
        st.session_state[res_key] = result
        st.session_state["last_geo_run"] = {"kp": kp}
        from src.dashboard.ui.global_ticker import render_global_ticker
        render_global_ticker()                      # show the new run in the global ticker
        fin = result.final
        log_action("simulation", f"Geographic spread at {f.name} ({f.lat:.4f}, {f.lon:.4f}), "
                   f"{f.width_m:.0f}×{f.height_m:.0f} m, {duration_label(setup['duration_min'])}: "
                   f"{fin.get('burned', 0)} cells burned ({fin.get('burned_ha', 0):.2f} ha), "
                   f"{fin.get('burning', 0)} burning (wind {wind_speed_ms:.1f} m/s from {wind_from_deg:.0f}°)",
                   twin.region.name)
    result = st.session_state.get(res_key)

    hot = hotspots_near(getattr(twin.ingestion, "last_hotspots", None), f.lat, f.lon)
    live = (not twin.offline) and bool(os.getenv("FIRMS_MAP_KEY"))
    key = google_maps_key()
    if key:
        payload = sim_payload(f, setup["duration_min"], result, wind_speed_ms, wind_from_deg, hot,
                              "observed" if live else "synthetic", setup["n_ignition"], setup["placement"],
                              setup["ignition_points"], setup["layers"], autoplay=result is not None,
                              api_key=key, map_id=google_maps_map_id())
        ev = new_event(render_fire_map(payload, key=f"{kp}_simmap"), f"{kp}_simmap")
        if ev and ev.get("kind") == "ignite" and apply_map_event(ev, setup, twin):
            st.session_state[f"_{kp}_set_sync"] = True
            st.rerun()
    else:
        missing_key_card()

    st.caption(f"{wind_label}: {wind_speed_ms:.1f} m/s from the {_compass_direction_name(wind_from_deg)} "
               f"({wind_from_deg:.0f}°), pushing fire and smoke towards the "
               f"{_compass_direction_name(wind_from_deg + 180)}.")

    if result is None:
        st.markdown('<div class="info-box">Press <b>Run Simulation</b> to compute the spread from the ignition '
                    'shown on the map. Play, pause, speed and the simulation-time slider are on the map.</div>',
                    unsafe_allow_html=True)
        _render_provenance(cond, None, f, setup, scenario, wind_label)
        return
    _render_analytics(result, setup)
    _render_exports(result, kp)
    _render_provenance(cond, result, f, setup, scenario, wind_label)


def _render_analytics(result, setup: dict):
    m = result.metrics
    fin = result.final
    max_ros = max(x["ros_m_per_min"] for x in m)
    beyond_t = next((x["minutes"] for x in m if x["left_focus"]), None)
    ended = fin["burning"] == 0
    d0 = m[0]["front_distance_m"]                          # radius of the ignition cluster itself
    mean_ros = (fin["front_distance_m"] - d0) / fin["minutes"] if fin["minutes"] else 0.0
    max_int = max(x["max_intensity"] for x in m)
    fcells = result.focus.n_rows * result.focus.n_cols
    pct_focus = 100.0 * fin["burned_in_focus"] / fcells if fcells else 0.0
    dur = setup["duration_min"]
    cards = [("Burned area", f"{fin['burned_ha']:.2f} ha", "crit"), ("Focus area burned", f"{pct_focus:.0f}%", ""),
             ("Fire perimeter", f"{fin['perimeter_m']:.0f} m", ""),
             ("Max spread distance", f"{fin['front_distance_m']:.0f} m", ""),
             ("Mean rate of spread", f"{mean_ros:.1f} m/min", "warn"), ("Peak rate of spread", f"{max_ros:.1f} m/min", "warn"),
             ("Max fire intensity", f"{max_int * 100:.0f}%", "warn"),
             ("Fire state", f"Out at T+{fin['minutes']:.0f} min" if ended else f"Active at T+{min(dur, fin['minutes']):g} min",
              "ok" if ended else "crit"),
             ("Beyond focus area", f"From T+{beyond_t:.0f} min" if beyond_t is not None else "No",
              "warn" if beyond_t is not None else "ok")]
    st.markdown('<div class="kpi-row">' + "".join(
        f'<div class="kpi"><div class="lbl">{a}</div><div class="val {c}" style="font-size:19px">{b}</div></div>'
        for a, b, c in cards) + "</div>", unsafe_allow_html=True)
    if fin.get("centroid_lat") is not None:
        st.caption(f"Final fire centroid {fin['centroid_lat']:.5f}°N, {fin['centroid_lon']:.5f}°E · "
                   f"simulation domain {result.domain.width_m:.0f} × {result.domain.height_m:.0f} m "
                   f"(focus area + {result.domain.margin} cells on every side, so the fire is never stopped by a box)")

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
                      "properties": {"ignition_min": round(float(result.ignition_step[r, c]) * result.step_minutes, 2),
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


def _render_provenance(cond: dict, result, f: FocusArea, setup: dict, scenario: Optional[dict], wind_label: str):
    p = result.params if result is not None else {}
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
         "SCENARIO INPUT" if scenario else "LIVE / DEMO"),
        ("Wind", wind_label, "SCENARIO INPUT" if scenario else "LIVE / DEMO"),
        ("FFMC / BUI / FWI", f"{cond['ffmc']:.1f} / {cond['bui']:.1f} / {cond['fwi']:.1f} (zone {cond['zone_id']})", "DERIVED"),
        ("Model risk", f"{cond['risk_score']:.0%}" if np.isfinite(cond["risk_score"]) else "-", "DERIVED (XGBoost)"),
        ("Fuel", f"NDVI estimate {cond['ndvi']:.2f}, uniform in the domain", "DERIVED"),
        ("Grid", f"{f.cell_m:.0f} m cells, CA step {step_minutes_for(f.cell_m):g} min", "COMPUTATIONAL MODEL"),
        ("Fire spread / burned area", "FireSpreadSimulator (cellular automata)"
         + (f", base spread probability {p.get('base_spread_prob')} (local calibration), seed {p.get('seed')}" if p else ""),
         "SIMULATION OUTPUT"),
        ("Flames, smoke, embers, ash", "Rendered from the simulated cell states", "SIMULATED VISUALIZATION"),
    ]
    with st.expander("Data provenance: real, derived, scenario and simulated"):
        st.dataframe(pd.DataFrame(rows, columns=["Item", "Value", "Type"]), hide_index=True, use_container_width=True)
        st.caption("The 25 m spread probability is a local-scale calibration that has not been validated against "
                   "observed fire perimeters. Simulated fire is not an observed wildfire.")
