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
    set_page, build_sidebar, ensure_twin, render_kpi_row,
    render_risk_map, render_risk_gauge, render_alerts,
    render_autorefresh_status,
)
from src.dashboard.ui.command_center import (
    render_dashboard_hero, render_explore_cards, render_fire_alert_panel, render_status_strip,
    render_system_status,
)
from src.auth.auth_gate import require_login, render_user_badge_in_sidebar, is_admin

set_page("Command Center")          # also reserves the global ticker at the top of the page
require_login()

offline, scenario, region, force_refresh = build_sidebar()
render_user_badge_in_sidebar()
twin = ensure_twin(offline, scenario, region, force_refresh)
snap = twin.current_snapshot
summary = twin.get_summary()

# Layout: ticker -> hero -> live status strip -> fire alert -> key metrics ->
# explore modules -> risk map / alerts -> data & system status.
render_dashboard_hero()
render_status_strip(twin, summary)
render_autorefresh_status(offline, scenario, region)
render_fire_alert_panel(twin, summary)

st.markdown('<div class="sec-hdr">Key metrics</div>', unsafe_allow_html=True)
render_kpi_row(summary)

st.markdown('<div class="sec-hdr">Explore</div>', unsafe_allow_html=True)
nav_items = [
    ("01", "What-If Simulator", "Drag temperature, wind, and humidity sliders and watch risk recompute live.",
     "pages/1_What_If_Simulator.py", "sliders"),
    ("02", "Historical Time Machine", "Scrub through three real fire seasons of satellite-confirmed detections.",
     "pages/3_Historical_Time_Machine.py", "clock"),
    ("03", "Spread Simulation", "Cellular-automata fire spread on Google satellite geography.",
     "pages/2_Spread_Simulation.py", "flame"),
    ("04", "AI Situation Briefing", "Plain-English risk summaries generated for any zone.",
     "pages/4_AI_Situation_Briefing.py", "spark"),
    ("05", "Model Insights", "Real-data model performance plus per-zone explainability.",
     "pages/5_Model_Insights.py", "avg"),
]
if is_admin():
    nav_items.append(
        ("06", "Admin", "Manage accounts, view stored snapshot and alert history, check notifications.",
         "pages/6_Admin.py", "shield")
    )
render_explore_cards(nav_items)

col_map, col_side = st.columns([3, 1], gap="medium")
with col_map:
    render_risk_map(snap.processed_grid, snap.risk_scores, twin.region)
with col_side:
    render_risk_gauge(summary)
    render_alerts(snap.alerts, summary, limit=8)

st.markdown('<div class="sec-hdr">Data &amp; system status</div>', unsafe_allow_html=True)
render_system_status(twin, summary, offline)
st.markdown("<br/>", unsafe_allow_html=True)
st.caption(
    "Digital Twin Framework for Forest Fire Prediction · BMSCE ISE · Batch 42 · "
    "Data: NASA FIRMS (VIIRS SNPP), OpenWeatherMap · Canadian FWI System"
    + ("  ·  offline / demo mode" if offline else "")
)
