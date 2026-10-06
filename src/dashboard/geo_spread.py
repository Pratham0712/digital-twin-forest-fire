"""
geo_spread.py - What-If -> Spread Simulation hand-off and the geographic
(Google satellite) spread panel shared by every mode of the Spread Simulation
page. Pure UI glue: the risk state comes from DigitalTwin, the spread from
src/simulation/local_spread.py (FireSpreadSimulator), the map/VFX from
geo_fire_map.py.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from config.config import SYSTEM
from src.dashboard.dashboard_common import _compass_direction_name, log_action
from src.dashboard.geo_fire_map import (build_payload, hotspots_near, missing_key_card,
                                        render_geo_fire_map)
from src.simulation.local_spread import (DEFAULT_FOCUS, FOCUS_AREAS, HIGHEST_RISK_FOCUS, FocusArea,
                                         focus_options_for_region, local_elevation, resolve_focus,
                                         run_local_spread, zone_conditions)

SPREAD_PAGE = "pages/2_Spread_Simulation.py"
SPREAD_MODES = ["Applied What-If scenario", "Current risk state (live/demo, from the sidebar)",
                "Custom weather scenario"]
PLACEMENTS = ["Upwind edge", "Centre"]


def google_maps_key() -> str:
    return os.getenv("GOOGLE_MAPS_API_KEY", "")


def google_maps_map_id() -> str:
    return os.getenv("GOOGLE_MAPS_MAP_ID", "")


# ── What-If side ──────────────────────────────────────────────────────────── #

def render_focus_selector(region, key: str, label: str = "Simulation focus area") -> str:
    """Selectbox of named protected areas inside the region (Bandipur first for
    Karnataka) plus the scenario's highest-risk zone."""
    options = focus_options_for_region(region)
    current = st.session_state.get(key)
    if current not in options:
        st.session_state[key] = DEFAULT_FOCUS if DEFAULT_FOCUS in options else options[0]
    return st.selectbox(label, options, key=key,
                        help=f"A {SYSTEM.focus_size_m} m × {SYSTEM.focus_size_m} m area on the satellite map "
                             f"where the fire spread is simulated on {SYSTEM.local_ca_cell_m} m cells.")


def build_simulation_config(twin, scenario: dict, focus_choice: str) -> dict:
    """Everything Module 2 needs, taken from the scenario twin that Module 1
    just computed (no second weather or risk calculation)."""
    snap = twin.current_snapshot
    focus = resolve_focus(focus_choice, snap.processed_grid, snap.risk_scores)
    cond = zone_conditions(snap.processed_grid, snap.risk_scores, focus.lat, focus.lon)
    summary = twin.get_summary()
    sev = next((a.severity for a in snap.alerts if a.zone_id == cond["zone_id"]), "LOW")
    r = twin.region
    return {
        "region": r.name, "region_bounds": [r.min_lat, r.max_lat, r.min_lon, r.max_lon],
        "focus_choice": focus_choice, "forest": focus.name,
        "latitude": focus.lat, "longitude": focus.lon,
        "size_m": focus.n * focus.cell_m, "cell_m": focus.cell_m,
        "temp_c": scenario.get("temp_c"), "humidity_pct": scenario.get("humidity_pct"),
        "wind_speed_ms": scenario.get("wind_speed_ms"), "wind_from_deg": scenario.get("wind_from_deg", 225),
        "n_hotspots": scenario.get("n_hotspots"),
        "risk": {"zone_id": cond["zone_id"], "zone_risk": cond["risk_score"], "zone_severity": sev,
                 "region_peak_risk": summary.get("max_risk_score", 0.0),
                 "region_alerts": summary.get("total_alerts", 0),
                 "ffmc": cond["ffmc"], "bui": cond["bui"], "fwi": cond["fwi"]},
        "simulation": {"n_ignition": 3, "placement": PLACEMENTS[0],
                       "horizon_minutes": SYSTEM.fire_spread_horizon_hours * 60},
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "offline": bool(twin.offline), "scenario": dict(scenario),
    }


def apply_and_open_spread(twin, scenario: dict, focus_choice: str):
    cfg = build_simulation_config(twin, scenario, focus_choice)
    st.session_state["simulation_config"] = cfg
    st.session_state["_sim_twin"] = twin
    st.session_state["spread_mode"] = SPREAD_MODES[0]
    st.session_state.pop("applied_geo_result", None)               # a new scenario invalidates the old run
    log_action("scenario", f"Scenario sent to Spread Simulation: {cfg['forest']}, {cfg['temp_c']}°C, "
               f"{cfg['humidity_pct']}% RH, {cfg['wind_speed_ms']} m/s from {cfg['wind_from_deg']}°",
               twin.region.name)
    st.switch_page(SPREAD_PAGE)


# ── Spread side ───────────────────────────────────────────────────────────── #

def render_applied_scenario_bar(cfg: dict):
    ts = cfg.get("timestamp", "")[:19].replace("T", " ")
    risk = cfg.get("risk", {})
    wdir = cfg.get("wind_from_deg", 0)
    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi"><div class="lbl">Focus area</div><div class="val" style="font-size:16px">{cfg.get('forest')}</div></div>
      <div class="kpi"><div class="lbl">Temperature</div><div class="val">{cfg.get('temp_c')}°C</div></div>
      <div class="kpi"><div class="lbl">Humidity</div><div class="val">{cfg.get('humidity_pct')}%</div></div>
      <div class="kpi"><div class="lbl">Wind</div><div class="val">{float(cfg.get('wind_speed_ms') or 0):.1f} m/s</div></div>
      <div class="kpi"><div class="lbl">Wind from</div><div class="val">{_compass_direction_name(wdir)} {wdir:.0f}°</div></div>
      <div class="kpi"><div class="lbl">Seeded hotspots</div><div class="val">{cfg.get('n_hotspots')}</div></div>
      <div class="kpi"><div class="lbl">Zone risk</div><div class="val">{risk.get('zone_risk', 0):.0%}</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.caption(f"Applied {ts} UTC · {cfg.get('region')} · zone {risk.get('zone_id')} "
               f"({risk.get('zone_severity')}) · {cfg.get('latitude'):.4f}°N, {cfg.get('longitude'):.4f}°E · "
               f"synthetic scenario weather")


def _elevation(focus: FocusArea, allow_fetch: bool):
    """Local DEM (cached on disk by local_elevation; a failed fetch is not
    retried for 10 minutes)."""
    return local_elevation(focus, allow_fetch=allow_fetch)


def render_geo_spread(twin, key_prefix: str, wind_speed_ms: float, wind_from_deg: float,
                      wind_label: str, focus_choice: Optional[str] = None,
                      wind_schedule_15min=None):
    """Google satellite map + local CA + VFX + analytics for one twin."""
    snap = twin.current_snapshot
    st.markdown('<div class="sec-hdr">Geographic fire spread — Google satellite</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 1.2, 1])
    with c1:
        if focus_choice is None:
            focus_choice = render_focus_selector(twin.region, key=f"{key_prefix}_focus")
        else:
            st.markdown(f"**Simulation focus area**  \n{focus_choice}")
    with c2:
        placement = st.selectbox("Ignition point", PLACEMENTS, key=f"{key_prefix}_placement",
                                 help="Upwind edge starts the fire on the side the wind comes from, so the "
                                      "head fire runs across the simulation area.")
    with c3:
        n_ign = st.slider("Ignition cells", 1, 9, 3, key=f"{key_prefix}_n_ign")

    focus = resolve_focus(focus_choice, snap.processed_grid, snap.risk_scores)
    cond = zone_conditions(snap.processed_grid, snap.risk_scores, focus.lat, focus.lon)
    r = twin.region
    if not (r.min_lat <= focus.lat <= r.max_lat and r.min_lon <= focus.lon <= r.max_lon):
        st.warning(f"{focus.name} is outside {r.name}; conditions come from the nearest grid zone.")

    sig = (focus.name, round(focus.lat, 5), round(focus.lon, 5), placement, n_ign,
           round(float(wind_speed_ms), 2), round(float(wind_from_deg), 1), cond["zone_id"],
           round(cond["ffmc"], 3) if np.isfinite(cond["ffmc"]) else None, id(snap))
    res_key, sig_key, seed_key = f"{key_prefix}_geo_result", f"{key_prefix}_geo_sig", f"{key_prefix}_geo_seed"
    if st.session_state.get(sig_key) != sig:
        st.session_state[res_key] = None
        st.session_state[sig_key] = sig

    b1, b2, b3, _ = st.columns([1.3, 1, 1.4, 3])
    run = b1.button("Run Simulation", type="primary", key=f"{key_prefix}_geo_run", use_container_width=True)
    if b2.button("Reset", key=f"{key_prefix}_geo_reset", use_container_width=True):
        st.session_state[res_key] = None
    redraw = b3.button("New random draw", key=f"{key_prefix}_geo_redraw", use_container_width=True)
    if redraw:
        st.session_state[seed_key] = st.session_state.get(seed_key, 42) + 1
    seed = st.session_state.get(seed_key, 42)

    if run or redraw:
        with st.spinner("Simulating fire spread on the local grid..."):
            elev, terrain_src = _elevation(focus, allow_fetch=True)
            result = run_local_spread(focus, cond, wind_speed_ms, wind_from_deg, n_ignition=n_ign,
                                      placement=placement, seed=seed, elevation=elev,
                                      terrain_source=terrain_src, wind_schedule_15min=wind_schedule_15min)
        st.session_state[res_key] = result
        fin = result.final
        log_action("simulation", f"Geographic spread at {focus.name}: {fin.get('burned', 0)} cells burned "
                   f"({fin.get('burned_ha', 0):.2f} ha), {fin.get('burning', 0)} burning after "
                   f"{fin.get('minutes', 0):.0f} min (wind {wind_speed_ms:.1f} m/s from {wind_from_deg:.0f}°)",
                   twin.region.name)
    result = st.session_state.get(res_key)

    hot = hotspots_near(getattr(twin.ingestion, "last_hotspots", None), focus.lat, focus.lon)
    live = (not twin.offline) and bool(os.getenv("FIRMS_MAP_KEY"))
    elev_preview = result.elevation if result is not None else _elevation(focus, allow_fetch=False)[0]
    key = google_maps_key()
    payload = build_payload(focus, result, wind_speed_ms, wind_from_deg, hot,
                            "observed" if live else "synthetic", n_ign, placement,
                            autoplay=result is not None, api_key=key, map_id=google_maps_map_id(),
                            elevation=elev_preview)
    if key:
        render_geo_fire_map(payload)
    else:
        missing_key_card()

    st.caption(f"{wind_label}: {wind_speed_ms:.1f} m/s from the {_compass_direction_name(wind_from_deg)} "
               f"({wind_from_deg:.0f}°), pushing the fire towards the {_compass_direction_name(wind_from_deg + 180)}"
               f" · {focus.n} × {focus.n} cells of {focus.cell_m:.0f} m = {focus.n * focus.cell_m:.0f} m × "
               f"{focus.n * focus.cell_m:.0f} m ({focus.area_km2:.2f} km²)")

    if result is None:
        st.markdown('<div class="info-box">Press <b>Run Simulation</b> to project the 2-hour spread from the '
                    'ignition cells shown on the map.</div>', unsafe_allow_html=True)
        _render_conditions(cond, None, focus)
        return
    _render_analytics(result)
    _render_conditions(cond, result, focus)


def _render_analytics(result):
    m = result.metrics
    fin = result.final
    peak_burning = max(x["burning"] for x in m)
    max_ros = max(x["ros_m_per_min"] for x in m)
    boundary_t = next((x["minutes"] for x in m if x["boundary_reached"]), None)
    ended = fin["burning"] == 0
    total_cells = result.focus.n ** 2 - int(result.non_fuel.sum())
    pct = 100.0 * fin["burned"] / total_cells if total_cells else 0.0
    d0 = m[0]["front_distance_m"]                          # radius of the ignition cluster itself
    mean_ros = (fin["front_distance_m"] - d0) / fin["minutes"] if fin["minutes"] else 0.0

    cards = [("Burned area", f"{fin['burned_ha']:.2f} ha", "crit"), ("Area burned", f"{pct:.0f}%", ""),
             ("Fire perimeter", f"{fin['perimeter_m']:.0f} m", ""), ("Mean rate of spread", f"{mean_ros:.1f} m/min", "warn"),
             ("Peak rate of spread", f"{max_ros:.1f} m/min", "warn"), ("Peak burning cells", peak_burning, ""),
             ("Fire state", f"Out at T+{fin['minutes']:.0f} min" if ended else f"Active at T+{fin['minutes']:.0f} min",
              "ok" if ended else "crit"),
             ("Boundary", f"Reached T+{boundary_t:.0f} min" if boundary_t is not None else "Not reached",
              "crit" if boundary_t is not None else "ok")]
    st.markdown('<div class="kpi-row">' + "".join(
        f'<div class="kpi"><div class="lbl">{a}</div><div class="val {c}" style="font-size:20px">{b}</div></div>'
        for a, b, c in cards) + "</div>", unsafe_allow_html=True)

    t = [x["minutes"] for x in m]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=[x["burned_ha"] for x in m], name="Burned area (ha)",
                             line=dict(color="#ff6b4a", width=2), fill="tozeroy", fillcolor="rgba(255,107,74,0.12)"))
    fig.add_trace(go.Scatter(x=t, y=[x["burning"] * result.focus.cell_m ** 2 / 1e4 for x in m],
                             name="Burning area (ha)", line=dict(color="#fbbf24", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=t, y=[x["ros_m_per_min"] for x in m], name="Rate of spread (m/min)",
                             line=dict(color="#60a5fa", width=1.5), yaxis="y2"))
    fig.update_layout(height=230, margin=dict(l=0, r=0, t=28, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(title="Minutes after ignition", tickfont={"color": "#8a96a6", "size": 10},
                                 gridcolor="#232b36"),
                      yaxis=dict(title="ha", tickfont={"color": "#8a96a6", "size": 10}, gridcolor="#232b36"),
                      yaxis2=dict(title="m/min", overlaying="y", side="right", showgrid=False,
                                  tickfont={"color": "#8a96a6", "size": 10}),
                      legend=dict(orientation="h", y=1.12, x=0, font={"color": "#8a96a6", "size": 10},
                                  bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)


def _render_conditions(cond: dict, result, focus: FocusArea):
    p = result.params if result is not None else {}
    terrain = result.terrain_source if result is not None else None
    elev_txt = ""
    if result is not None and terrain == "dem":
        e = result.elevation
        elev_txt = f"DEM {e.min():.0f}–{e.max():.0f} m"
    elif result is not None:
        elev_txt = "flat (no local DEM)"
    rows = [
        ("Regional zone", f"{cond['zone_id']} ({cond['zone_lat']:.2f}°N, {cond['zone_lon']:.2f}°E)"),
        ("Model risk (zone)", f"{cond['risk_score']:.0%}" if np.isfinite(cond["risk_score"]) else "-"),
        ("FFMC / BUI / FWI", f"{cond['ffmc']:.1f} / {cond['bui']:.1f} / {cond['fwi']:.1f}"),
        ("Fuel (NDVI estimate)", f"{cond['ndvi']:.2f}"),
        ("Zone weather", f"{cond['temp_c']:.1f}°C, {cond['humidity_pct']:.0f}% RH"),
    ]
    if result is not None:
        rows += [("Terrain", elev_txt),
                 ("CA", f"{focus.cell_m:.0f} m cells, {result.step_minutes:g} min steps, "
                        f"base spread probability {p.get('base_spread_prob')} (local calibration), seed {p.get('seed')}")]
    with st.expander("Simulation inputs"):
        st.markdown("\n".join(f"- **{a}:** {b}" for a, b in rows))
        st.caption("Real/external: satellite map, terrain DEM (when fetched), FIRMS detections in live mode. "
                   "Derived: FFMC/BUI/FWI, model risk, fuel estimate. Simulated: spread, burned area and every "
                   "visual effect. The 25 m spread probability is a local-scale calibration that has not been "
                   "validated against observed fire perimeters.")
