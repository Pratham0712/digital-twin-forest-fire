"""
Spread Simulation - dedicated page for the animated Cellular Automata fire
spread visualization. Two modes:
  - "Current risk state": seeds from the SAME twin as the Command Center
    (shared via st.session_state["twin"]) - whatever alerts exist there,
    right now, in whichever offline/live mode is active.
  - "Custom weather scenario": builds its own isolated scenario twin (same
    mechanism as the What-If Simulator page) so you can deliberately choose
    conditions and watch how THAT scenario would spread, independent of
    today's real/random draw.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

import streamlit as st

from src.dashboard.dashboard_common import (
    set_page, build_sidebar, ensure_twin, get_twin, render_header,
    render_ca_simulation, render_scenario_controls, log_action, _region_key,
)

from src.auth.auth_gate import require_login, render_user_badge_in_sidebar

set_page("Spread Simulation")
require_login()

offline, scenario, region, force_refresh = build_sidebar()
render_user_badge_in_sidebar()
twin = ensure_twin(offline, scenario, region, force_refresh)
summary = twin.get_summary()

render_header(summary, offline, "Fire Spread Simulation", region=twin.region)

with st.expander("About offline mode and simulation source"):
    st.markdown(
        "- **Offline / demo mode** (sidebar toggle) uses synthetic fire and weather data for "
        "this region instead of live satellite and weather feeds.\n"
        "- **Live mode** (toggle off) reads real current fire detections and weather.\n"
        "- **The two modes below** choose which risk state the fire spread is seeded from: "
        "the current situation (live or demo), or a hypothetical weather scenario you set here."
    )

st.markdown('<div class="sec-hdr">Simulate from</div>', unsafe_allow_html=True)
mode = st.radio(
    "Source of the risk state to spread from",
    ["Current risk state (live/demo, from the sidebar)", "Custom weather scenario"],
    horizontal=True, label_visibility="collapsed",
)

if mode.startswith("Current"):
    st.caption(
        "Seeded from the same live/demo Digital Twin state as the Command Center — refresh there "
        "(or use the sidebar here) to update the alert zones this simulation seeds from."
    )
    render_ca_simulation(twin, key_prefix="ca", allow_force_ignite=True)

else:
    st.caption(
        "Pick (or preset) a weather scenario below, independent of today's real/demo conditions, "
        "and see how a fire would spread under exactly those conditions."
    )
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
        f"**{scn_summary.get('max_risk_score', 0):.0%}** — the spread simulation below ignites "
        f"from whichever of those cross the HIGH/EXTREME threshold."
    )
    render_ca_simulation(scn_twin, key_prefix="custom_scn", allow_force_ignite=True)
