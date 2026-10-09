"""
Spread Simulation - Module 2. Fire spread on the real map: Google satellite
imagery as the base, a geographically anchored cellular-automata grid over a
500 m x 500 m focus area (e.g. Bandipur Tiger Reserve), and wind-driven fire,
smoke, ember and ash effects rendered from the CA state. Three sources for the
risk state the fire spreads under:
  - "Applied What-If scenario": the scenario (and its twin) sent from the
    What-If Simulator with "Apply Scenario & Open Spread Simulation".
  - "Current risk state": the SAME twin as the Command Center (shared via
    st.session_state["twin"]), live or demo per the sidebar.
  - "Custom weather scenario": an isolated scenario twin built on this page.
The regional 2-hour projection on the 0.1 deg grid (the original animated CA
heatmap) is kept below the map, except in LIVE REAL-WORLD mode: it ignites the
model's risk zones, and risk is not an observed fire (audit BUG #3).
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

import streamlit as st

from src.dashboard.dashboard_common import (
    set_page, build_sidebar, ensure_twin, get_twin, render_header,
    render_ca_simulation, render_scenario_controls, log_action, _region_key, render_live_no_risk_ignition,
    render_study_region_summary,
)
from src.dashboard.geo_spread import (SPREAD_MODES, default_setup, get_setup, render_applied_scenario_bar,
                                     render_geo_spread)
from src.data_ingestion.wind import mean_wind

from src.auth.auth_gate import require_login, render_user_badge_in_sidebar

set_page("Spread Simulation")
require_login()

offline, scenario, region, force_refresh = build_sidebar()
render_user_badge_in_sidebar()
twin = ensure_twin(offline, scenario, region, force_refresh)
summary = twin.get_summary()

cfg = st.session_state.get("simulation_config")
if st.session_state.get("spread_mode") not in SPREAD_MODES:
    st.session_state["spread_mode"] = SPREAD_MODES[0] if cfg else SPREAD_MODES[1]

render_header(summary, offline, "Fire Spread Simulation", region=twin.region)
render_study_region_summary(twin)

st.markdown('<div class="sec-hdr">Simulate from</div>', unsafe_allow_html=True)
mode = st.radio("Source of the risk state to spread from", SPREAD_MODES, horizontal=True,
                label_visibility="collapsed", key="spread_mode")


def _regional_projection(t, key_prefix: str, hypothetical: bool = False):
    """The regional 0.1-degree projection ignites the model's HIGH/EXTREME risk
    zones. LIVE REAL-WORLD: never (audit BUG #3 - risk is not fire; enforced in
    DigitalTwin as well). Scenario / WHAT-IF: labelled HYPOTHETICAL IGNITION.
    Force-ignite only in explicitly marked demo / offline mode."""
    if getattr(t, "live_observed_only", False):
        render_live_no_risk_ignition(t)
        return
    label = ("Regional 2-hour projection — HYPOTHETICAL IGNITION at the model's HIGH/EXTREME zones"
             if hypothetical else "Regional 2-hour projection (0.1° grid, ~11 km cells)")
    with st.expander(label):
        st.caption("The regional CA ignites model risk zones (predictions, not observed fires). Its fuel value per "
                   "11 km zone is a SYNTHETIC NDVI estimate; real land-cover fuel applies to the local simulation "
                   "above.")
        render_ca_simulation(t, key_prefix=key_prefix, allow_force_ignite=bool(offline))


if mode == SPREAD_MODES[0]:
    if not cfg:
        st.markdown('<div class="info-box">No scenario has been applied yet. Set one up in the What-If '
                    'Simulator and press <b>Apply Scenario &amp; Open Spread Simulation</b>.</div>',
                    unsafe_allow_html=True)
        st.page_link("pages/1_What_If_Simulator.py", label="Open the What-If Simulator")
    else:
        sim_twin = st.session_state.get("_sim_twin")
        if sim_twin is None or sim_twin.current_snapshot is None:
            # Session restored without the twin object: rebuild the same scenario.
            from config.config import RegionConfig
            b = cfg["region_bounds"]
            reg = RegionConfig(name=cfg["region"], min_lat=b[0], max_lat=b[1], min_lon=b[2], max_lon=b[3],
                               grid_resolution_deg=twin.region.grid_resolution_deg
                               if twin.region.name == cfg["region"] else 0.5,
                               weather_grid_resolution_deg=twin.region.weather_grid_resolution_deg
                               if twin.region.name == cfg["region"] else 1.5)
            with st.spinner("Rebuilding the applied scenario..."):
                sim_twin = get_twin(offline=True, scenario=cfg["scenario"], region=reg)
                sim_twin.refresh()
            st.session_state["_sim_twin"] = sim_twin
        render_applied_scenario_bar(cfg)
        if "setup" not in cfg:                       # scenario applied by an older version of the page
            cfg["setup"] = {**default_setup(sim_twin), "region": sim_twin.region.name}
        _live = cfg.get("live")
        _wlabel = {"live": "Live wind (OpenWeatherMap observation)", "whatif": "WHAT-IF scenario wind"}.get(
            (_live or {}).get("mode"), "Scenario wind")
        render_geo_spread(sim_twin, key_prefix="applied", wind_speed_ms=float(cfg["wind_speed_ms"]),
                          wind_from_deg=float(cfg["wind_from_deg"]), wind_label=_wlabel,
                          setup=cfg["setup"], scenario=None if _live else cfg.get("scenario"), live=_live)
        # A LIVE hand-off: the scenario twin is built from live observations; it must
        # never ignite risk zones (backend flag, checked by DigitalTwin itself).
        sim_twin.live_observed_only = (_live or {}).get("mode") == "live"
        _regional_projection(sim_twin, "applied_ca", hypothetical=(_live or {}).get("mode") == "whatif")

elif mode == SPREAD_MODES[1]:
    st.caption("Seeded from the same live/demo Digital Twin state as the Command Center.")
    from src.simulation.local_spread import zone_conditions
    snap = twin.current_snapshot
    setup = get_setup(twin, kp="current_set")
    # Wind at the focus zone: the OpenWeatherMap forecast near it when live (same
    # source the regional CA uses), otherwise the zone's current wind.
    lat, lon = setup["location"]["lat"], setup["location"]["lon"]
    cond = zone_conditions(snap.processed_grid, snap.risk_scores, lat, lon)
    schedule, label = None, "Current wind"
    w_speed, w_from = cond["wind_speed_ms"], cond["wind_from_deg"]
    try:
        mask = (snap.processed_grid["zone_id"] == cond["zone_id"]).values
        sched, wind = twin._spread_wind(snap.processed_grid, mask, int(max(15, setup["duration_min"])), 15)
        w_speed, w_from = float(wind["speed_ms"]), float(wind["from_deg"])
        if sched:
            schedule, label = sched, "Forecast wind"
    except Exception:
        w_speed, w_from = mean_wind([cond["wind_speed_ms"]], [cond["wind_from_deg"]])
    render_geo_spread(twin, key_prefix="current", wind_speed_ms=w_speed, wind_from_deg=w_from,
                      wind_label=label, setup=setup, wind_schedule_15min=schedule)
    _regional_projection(twin, "ca")

else:
    st.caption("Pick (or preset) a weather scenario, independent of today's real/demo conditions.")
    scenario_vals, just_applied_preset = render_scenario_controls("spread_scn")
    apply_clicked = st.button("Build this scenario", type="primary", use_container_width=True,
                              key="spread_scn_apply")

    need_new_twin = (
        apply_clicked or just_applied_preset or "_spread_scn_twin" not in st.session_state
        or st.session_state.get("_spread_scn_vals") != scenario_vals
        or st.session_state.get("_spread_scn_region") != _region_key(twin.region)
    )
    if need_new_twin:
        with st.spinner("Building this scenario's risk state..."):
            scn_twin = get_twin(offline=True, scenario=scenario_vals, region=twin.region)
            scn_twin.refresh()
        st.session_state["_spread_scn_twin"] = scn_twin
        st.session_state["_spread_scn_vals"] = scenario_vals
        st.session_state["_spread_scn_region"] = _region_key(twin.region)
        if apply_clicked or just_applied_preset:
            log_action("scenario", f"Spread scenario built: {scenario_vals.get('temp_c')}°C, "
                       f"{scenario_vals.get('wind_speed_ms')} m/s wind, "
                       f"{scenario_vals.get('humidity_pct')}% humidity", scn_twin.region.name)
        st.session_state["custom_scn_history"] = None

    scn_twin = st.session_state["_spread_scn_twin"]
    scn_summary = scn_twin.get_summary()
    bd = scn_summary.get("severity_breakdown", {})
    st.info(
        f"This scenario produces **{scn_summary.get('total_alerts', 0)} alert zone(s)** "
        f"({bd.get('EXTREME', 0)} EXTREME, {bd.get('HIGH', 0)} HIGH) with peak risk "
        f"**{scn_summary.get('max_risk_score', 0):.0%}**."
    )
    render_geo_spread(scn_twin, key_prefix="custom", wind_speed_ms=float(scenario_vals["wind_speed_ms"]),
                      wind_from_deg=float(scenario_vals["wind_from_deg"]), wind_label="Scenario wind",
                      setup=get_setup(scn_twin, kp="custom_set"), scenario=scenario_vals)
    _regional_projection(scn_twin, "custom_scn", hypothetical=True)
