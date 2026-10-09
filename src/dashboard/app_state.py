"""
app_state.py - the ONE authoritative demo / offline switch.

    st.session_state["_pref_offline_mode"]   True  = demo / offline (synthetic data)
                                             False = real data (live APIs)

The sidebar toggle writes it (build_sidebar -> set_demo_mode); everything else
reads it through is_demo_mode(): the regional twin (ensure_twin), the selected-
location observations (What-If / Spread), the ticker, page badges and the Data &
System Status panel. No other module keeps its own demo flag or default.

Default when the user has not touched the toggle: real data if ANY data API
key (FIRMS / OpenWeatherMap) is configured, demo otherwise. A failed or
unconfigured API never switches the app to demo; it is reported per feed
(UNAVAILABLE / CACHED / NOT CONFIGURED).
"""
import streamlit as st

from config.config import API

DEMO_STATE_KEY = "_pref_offline_mode"
DEMO_TOGGLE_KEY = "demo_mode_toggle"      # the sidebar widget's own key (mirrors DEMO_STATE_KEY)


def default_demo_mode() -> bool:
    return not (API.firms_map_key or API.owm_api_key)


def is_demo_mode() -> bool:
    try:
        return bool(st.session_state.get(DEMO_STATE_KEY, default_demo_mode()))
    except Exception:                       # outside a Streamlit run (scripts, tests)
        return default_demo_mode()


def set_demo_mode(value: bool):
    st.session_state[DEMO_STATE_KEY] = bool(value)
