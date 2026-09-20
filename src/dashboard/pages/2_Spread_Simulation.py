"""
Spread Simulation - dedicated page for the animated Cellular Automata fire
spread visualization. Operates on the SAME twin as the Command Center
(shared via st.session_state["twin"]), so alerts generated there seed the
simulation here consistently.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

import streamlit as st

from src.dashboard.dashboard_common import (
    set_page, build_sidebar, ensure_twin, render_header, render_ca_simulation,
)

set_page("Spread Simulation", "🔥")

offline, scenario, region, force_refresh = build_sidebar()
twin = ensure_twin(offline, scenario, region, force_refresh)
summary = twin.get_summary()

render_header(summary, offline, "Fire Spread Simulation")
st.caption(
    "This page reads the same live Digital Twin state as the Command Center — refresh there "
    "(or use the sidebar here) to update the alert zones this simulation seeds from."
)
render_ca_simulation(twin)
