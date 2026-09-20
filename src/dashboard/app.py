"""
Command Center - entry point of the multi-page Digital Twin dashboard.

Run locally:
    streamlit run src/dashboard/app.py

Deploy on Streamlit Community Cloud:
    - Push repo to GitHub
    - share.streamlit.io -> New app -> point at src/dashboard/app.py
    - Add FIRMS_MAP_KEY / OWM_API_KEY in the app's Secrets manager

Other pages live in src/dashboard/pages/ and appear automatically in the
sidebar navigation (Streamlit's native multi-page convention). All pages
share one DigitalTwin instance via st.session_state, kept in sync by
dashboard_common.ensure_twin().
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st

from src.dashboard.dashboard_common import (
    set_page, build_sidebar, ensure_twin, render_header, render_kpi_row,
    render_risk_map, render_risk_gauge, render_alerts, page_nav_card,
)

set_page("Command Center", "🏠")

offline, scenario, region, force_refresh = build_sidebar()
twin = ensure_twin(offline, scenario, region, force_refresh)
snap = twin.current_snapshot
summary = twin.get_summary()

render_header(summary, offline, "Command Center")
render_kpi_row(summary)

st.markdown('<div class="sec-hdr">Explore</div>', unsafe_allow_html=True)
nav_cols = st.columns(5)
nav_items = [
    ("🧪", "What-If Simulator", "Drag temperature, wind, and humidity sliders and watch risk recompute live.",
     "pages/1_What_If_Simulator.py"),
    ("🕰️", "Historical Time Machine", "Scrub through 3 real fire seasons of satellite-confirmed detections.",
     "pages/3_Historical_Time_Machine.py"),
    ("🔥", "Spread Simulation", "Animated 2-hour Cellular Automata fire spread from today's alert zones.",
     "pages/2_Spread_Simulation.py"),
    ("📋", "AI Situation Briefing", "Plain-English risk summaries auto-generated for any zone.",
     "pages/4_AI_Situation_Briefing.py"),
    ("🧠", "Model Insights", "Real-data model performance plus per-zone SHAP explainability.",
     "pages/5_Model_Insights.py"),
]
for col, (icon, title, desc, target) in zip(nav_cols, nav_items):
    with col:
        page_nav_card(icon, title, desc)
        st.page_link(target, label="Open →", use_container_width=True)

st.markdown("<br/>", unsafe_allow_html=True)

col_map, col_side = st.columns([3, 1], gap="medium")
with col_map:
    render_risk_map(snap.processed_grid, snap.risk_scores, twin.region)
with col_side:
    render_risk_gauge(summary)
    render_alerts(snap.alerts, summary, limit=8)

st.markdown("---")
st.caption(
    "Digital Twin Framework for Forest Fire Prediction · BMSCE ISE · Batch 42 · "
    "Data: NASA FIRMS (VIIRS SNPP), OpenWeatherMap · Canadian FWI System"
    + ("  ·  ⚠ offline/demo mode" if offline else "")
)
