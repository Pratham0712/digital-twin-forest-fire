"""
What-If Simulator - dedicated page. Lets the user drag weather sliders and
immediately see recomputed risk, alerts, and (optionally) fire spread, using
the same scenario-injection mechanism already built into DigitalTwin/
DataIngestionModule. Only meaningful in offline/demo mode, since live mode
reads real current conditions rather than a hypothetical scenario.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

import streamlit as st

from src.dashboard.dashboard_common import (
    set_page, get_twin, render_header, render_kpi_row, render_risk_map,
    render_risk_gauge, render_alerts, SYSTEM,
)

set_page("What-If Simulator", "🧪")

st.markdown("""
<div class="hero">
  <h1>🧪 What-If Scenario Simulator</h1>
  <div class="sub">Adjust hypothetical weather conditions and watch fire risk recompute across
  every one of the 1,400 grid zones in real time.</div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.get("_last_offline", True):
    st.warning(
        "This page only works in offline/demo mode, since it explores hypothetical "
        "conditions rather than reading live sensors. Toggle 'Offline / demo mode' on "
        "in the sidebar of any other page, then come back here."
    )

st.markdown('<div class="sec-hdr">Scenario parameters</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    n_hotspots = st.slider("Active fire hotspots seeded", 0, 40, 8,
                            help="How many synthetic active-fire detections to seed across the region")
    temp_c = st.slider("Temperature (°C)", 15, 48, 32)
with c2:
    wind_speed_ms = st.slider("Wind speed (m/s)", 0.0, 20.0, 5.0, step=0.5)
    humidity_pct = st.slider("Relative humidity (%)", 0, 100, 40)

preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
preset = None
with preset_col1:
    if st.button("☔ Monsoon calm", use_container_width=True):
        preset = dict(n_hotspots=0, temp_c=24, wind_speed_ms=2.0, humidity_pct=85)
with preset_col2:
    if st.button("🌤️ Typical dry day", use_container_width=True):
        preset = dict(n_hotspots=5, temp_c=33, wind_speed_ms=4.0, humidity_pct=35)
with preset_col3:
    if st.button("🔥 Extreme fire weather", use_container_width=True):
        preset = dict(n_hotspots=20, temp_c=42, wind_speed_ms=12.0, humidity_pct=12)
with preset_col4:
    if st.button("💨 High wind event", use_container_width=True):
        preset = dict(n_hotspots=10, temp_c=36, wind_speed_ms=18.0, humidity_pct=25)

if preset:
    st.session_state["_wi_preset"] = preset
    st.rerun()

if "_wi_preset" in st.session_state:
    p = st.session_state.pop("_wi_preset")
    n_hotspots, temp_c, wind_speed_ms, humidity_pct = p["n_hotspots"], p["temp_c"], p["wind_speed_ms"], p["humidity_pct"]

scenario = {"n_hotspots": n_hotspots, "temp_c": temp_c,
            "wind_speed_ms": wind_speed_ms, "humidity_pct": humidity_pct}

run = st.button("▶ Apply scenario", type="primary", use_container_width=True)

if run or "_wi_twin" not in st.session_state:
    with st.spinner("Recomputing risk across the region for this scenario..."):
        wi_twin = get_twin(offline=True, scenario=scenario, region=None)
        wi_twin.refresh()
    st.session_state["_wi_twin"] = wi_twin
    st.session_state["_wi_scenario"] = scenario

twin = st.session_state["_wi_twin"]
snap = twin.current_snapshot
summary = twin.get_summary()

render_header(summary, True, "Scenario Result")
render_kpi_row(summary)

# ── plain-language interpretation of the scenario ──
max_r = summary.get("max_risk_score", 0)
mean_r = summary.get("mean_risk_score", 0)
alerts_n = summary.get("total_alerts", 0)
if max_r >= 0.6:
    verdict = "EXTREME conditions"
    verdict_color = "#f78166"
elif max_r >= 0.4:
    verdict = "elevated risk conditions"
    verdict_color = "#d29922"
else:
    verdict = "low risk conditions"
    verdict_color = "#3fb950"

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
    render_risk_map(snap.processed_grid, snap.risk_scores, twin.region)
with col_side:
    render_risk_gauge(summary)
    render_alerts(snap.alerts, summary, limit=8)

st.caption(
    "This page runs an isolated scenario twin, separate from the Command Center's live/demo "
    "state — switching pages will not lose your main session."
)
