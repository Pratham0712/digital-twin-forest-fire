"""
What-If Simulator - dedicated page. Lets the user drag weather sliders (or
click a one-click preset) and immediately see recomputed risk, alerts, and
fire-spread projection, using the same scenario-injection mechanism already
built into DigitalTwin/DataIngestionModule. Only meaningful in offline/demo
mode, since live mode reads real current conditions rather than a
hypothetical scenario.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

import streamlit as st

from src.dashboard.dashboard_common import (
    set_page, build_sidebar, get_twin, render_header, render_kpi_row, render_risk_map,
    render_risk_gauge, render_alerts, render_ca_simulation, render_scenario_controls, SYSTEM,
    log_action,
)

from src.auth.auth_gate import require_login, render_user_badge_in_sidebar

set_page("What-If Simulator")
require_login()

# The scenario is always synthetic, so the offline toggle is hidden here; the
# region picker is shared with every other page and follows the same choice.
_, _, region, _ = build_sidebar(show_offline=False)
render_user_badge_in_sidebar()

st.markdown(f"""
<div class="hero">
  <div class="eyebrow">Forest Fire Digital Twin</div>
  <h1>What-If Scenario Simulator</h1>
  <div class="sub">Adjust hypothetical weather conditions, or click a preset, and watch fire risk
  recompute across every grid zone of <b>{region.name}</b>. Scenarios use synthetic inputs applied to the
  region's real grid, terrain and trained model; they never read live sensors.</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="sec-hdr">Scenario parameters</div>', unsafe_allow_html=True)
st.caption(
    "Click a preset to instantly load and apply a known weather pattern, or drag the sliders "
    "yourself and press **Apply scenario** when ready."
)
scenario, just_applied_preset = render_scenario_controls("wi")

run = st.button("Apply scenario", type="primary", use_container_width=True)

region_sig = (region.name, region.min_lat, region.max_lat, region.min_lon, region.max_lon)
need_recompute = (
    run or just_applied_preset or "_wi_twin" not in st.session_state
    or st.session_state.get("_wi_scenario") != scenario
    or st.session_state.get("_wi_region") != region_sig
)
if need_recompute:
    with st.spinner("Recomputing risk across the region for this scenario..."):
        wi_twin = get_twin(offline=True, scenario=scenario, region=region)
        wi_twin.refresh()
    st.session_state["_wi_twin"] = wi_twin
    st.session_state["_wi_scenario"] = scenario
    st.session_state["_wi_region"] = region_sig
    if run or just_applied_preset:
        _s = wi_twin.get_summary()
        log_action("scenario", f"What-If applied: {scenario.get('temp_c')}°C, "
                   f"{scenario.get('wind_speed_ms')} m/s wind, {scenario.get('humidity_pct')}% humidity, "
                   f"{scenario.get('n_hotspots')} hotspots -> {_s.get('total_alerts', 0)} alerts, "
                   f"peak risk {_s.get('max_risk_score', 0):.0%}", wi_twin.region.name)
    # A new scenario invalidates any spread simulation run from the old one.
    st.session_state["wi_ca_history"] = None

twin = st.session_state["_wi_twin"]
snap = twin.current_snapshot
summary = twin.get_summary()

render_header(summary, True, "Scenario Result", region=twin.region)
render_kpi_row(summary)

# ── plain-language interpretation of the scenario ──
n_hotspots, temp_c = scenario["n_hotspots"], scenario["temp_c"]
wind_speed_ms, humidity_pct = scenario["wind_speed_ms"], scenario["humidity_pct"]
max_r = summary.get("max_risk_score", 0)
mean_r = summary.get("mean_risk_score", 0)
alerts_n = summary.get("total_alerts", 0)
if max_r >= 0.6:
    verdict = "EXTREME conditions"
    verdict_color = "#ff6b4a"
elif max_r >= 0.4:
    verdict = "elevated risk conditions"
    verdict_color = "#fbbf24"
else:
    verdict = "low risk conditions"
    verdict_color = "#34d399"

driver_bits = []
if humidity_pct < 20:
    driver_bits.append(f"very low humidity ({humidity_pct}%)")
if wind_speed_ms > 10:
    driver_bits.append(f"strong wind ({wind_speed_ms:.1f} m/s)")
if temp_c > 38:
    driver_bits.append(f"high temperature ({temp_c}°C)")
if n_hotspots > 15:
    driver_bits.append(f"{n_hotspots} seeded active hotspots")
driver_txt = ", ".join(driver_bits) if driver_bits else "a combination of the selected parameters"

st.markdown(f"""
<div class="briefing-card">
  <h4 style="color:{verdict_color} !important;">Scenario interpretation</h4>
  This scenario (temperature {temp_c}°C, wind {wind_speed_ms:.1f} m/s, humidity {humidity_pct}%,
  {n_hotspots} seeded hotspots) produces <b style="color:{verdict_color}">{verdict}</b> across the
  region — {alerts_n} of {summary.get('total_zones', 0)} zones cross the
  {SYSTEM.alert_threshold_pct:.0f}% alert threshold, with peak risk at {max_r:.0%} and a regional
  mean of {mean_r:.0%}. The primary driver{'s are' if len(driver_bits) > 1 else ' is'} {driver_txt}.
</div>
""", unsafe_allow_html=True)

col_map, col_side = st.columns([3, 1], gap="medium")
with col_map:
    render_risk_map(snap.processed_grid, snap.risk_scores, twin.region, scenario_active=True)
with col_side:
    render_risk_gauge(summary)
    render_alerts(snap.alerts, summary, limit=8)

st.markdown("---")
render_ca_simulation(twin, key_prefix="wi_ca", allow_force_ignite=True)

st.caption(
    "This page runs an isolated scenario twin, separate from the Command Center's live/demo "
    "state; switching pages will not lose your main session."
)
