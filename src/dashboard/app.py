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
    set_page, build_sidebar, ensure_twin, render_platform_hero, render_kpi_row,
    render_risk_map, render_risk_gauge, render_alerts, page_nav_card,
    render_autorefresh_status,
)
from src.auth.auth_gate import require_login, render_user_badge_in_sidebar

set_page("Command Center")
require_login()

offline, scenario, region, force_refresh = build_sidebar()
render_user_badge_in_sidebar()
twin = ensure_twin(offline, scenario, region, force_refresh)
snap = twin.current_snapshot
summary = twin.get_summary()

render_platform_hero(summary, offline, region=twin.region)
render_autorefresh_status(offline, scenario, region)
render_kpi_row(summary)

from src.auth.auth_gate import is_admin

st.markdown('<div class="sec-hdr">Explore</div>', unsafe_allow_html=True)
nav_items = [
    ("01", "What-If Simulator", "Drag temperature, wind, and humidity sliders and watch risk recompute live.",
     "pages/1_What_If_Simulator.py"),
    ("02", "Historical Time Machine", "Scrub through three real fire seasons of satellite-confirmed detections.",
     "pages/3_Historical_Time_Machine.py"),
    ("03", "Spread Simulation", "Animated 2-hour cellular-automata fire spread from today's alert zones.",
     "pages/2_Spread_Simulation.py"),
    ("04", "AI Situation Briefing", "Plain-English risk summaries generated for any zone.",
     "pages/4_AI_Situation_Briefing.py"),
    ("05", "Model Insights", "Real-data model performance plus per-zone explainability.",
     "pages/5_Model_Insights.py"),
]
if is_admin():
    nav_items.append(
        ("06", "Admin", "Manage accounts, view stored snapshot and alert history, check notifications.",
         "pages/6_Admin.py")
    )
nav_cols = st.columns(len(nav_items))
for col, (icon, title, desc, target) in zip(nav_cols, nav_items):
    with col:
        page_nav_card(icon, title, desc)
        st.page_link(target, label="Open", use_container_width=True)

st.markdown("<br/>", unsafe_allow_html=True)

col_map, col_side = st.columns([3, 1], gap="medium")
with col_map:
    render_risk_map(snap.processed_grid, snap.risk_scores, twin.region)
with col_side:
    render_risk_gauge(summary)
    render_alerts(snap.alerts, summary, limit=8)

st.markdown("---")

persist_result = st.session_state.get("persist_result")
if persist_result is not None:
    if persist_result.get("pending"):
        status_bits = ["Saving snapshot to the database"]
    elif persist_result.get("persisted"):
        status_bits = ["Snapshot saved to database"]
    else:
        status_bits = ["Snapshot not saved (database unavailable this run)"]
    if persist_result.get("new_extreme_count"):
        email_bit = "email sent" if persist_result.get("emailed") else "email not configured"
        status_bits.append(
            f"{persist_result['new_extreme_count']} newly-extreme zone(s) this refresh, {email_bit}"
        )
    elif not persist_result.get("pending"):
        status_bits.append("no newly-extreme zones this refresh")
    st.caption("Status: " + "  ·  ".join(status_bits))

st.caption(
    "Digital Twin Framework for Forest Fire Prediction · BMSCE ISE · Batch 42 · "
    "Data: NASA FIRMS (VIIRS SNPP), OpenWeatherMap · Canadian FWI System"
    + ("  ·  offline / demo mode" if offline else "")
)
