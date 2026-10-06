"""
dashboard_common.py - shared logic, theme, and render functions used by every
page in the multi-page dashboard. Every page imports from here so the theme,
model loading, and twin session-state handling stay perfectly consistent
across pages, and so functionality already built and tested (risk map, CA
animation, model performance charts) is reused rather than duplicated.
"""
import os
import sys
import time
from typing import Optional
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from config.config import REGION, SYSTEM, MODELS_DIR, DATA_PROCESSED_DIR, DATA_RAW_DIR, RegionConfig
from src.data_ingestion.ingestion_module import region_bounds_error
from src.digital_twin.twin_state import DigitalTwin
from src.simulation.cellular_automata import CellState


# ── secrets / keys ───────────────────────────────────────────────────────────

def _secret(key: str) -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, "")


if _secret("FIRMS_MAP_KEY"):
    os.environ["FIRMS_MAP_KEY"] = _secret("FIRMS_MAP_KEY")
if _secret("OWM_API_KEY"):
    os.environ["OWM_API_KEY"] = _secret("OWM_API_KEY")
for _k in ("GOOGLE_MAPS_API_KEY", "GOOGLE_MAPS_MAP_ID"):
    if _secret(_k):
        os.environ[_k] = _secret(_k)


def set_page(title: str, icon: str = ":material/local_fire_department:"):
    """Every page must call this first. Wrapped in try/except because
    Streamlit raises if set_page_config is called more than once per script
    run, and some Streamlit versions are stricter about call order than
    others across page navigations."""
    if not (isinstance(icon, str) and icon.startswith(":material/")):
        icon = ":material/local_fire_department:"       # legacy callers passed emoji
    try:
        st.set_page_config(page_title=f"{title} | Forest Fire Digital Twin",
                            page_icon=icon, layout="wide", initial_sidebar_state="expanded")
    except Exception:
        try:
            st.set_page_config(page_title=f"{title} | Forest Fire Digital Twin",
                                layout="wide", initial_sidebar_state="expanded")
        except Exception:
            pass
    st.markdown(CSS, unsafe_allow_html=True)


# ── theme ─────────────────────────────────────────────────────────────────── #

CSS = """
<style>

:root {
    --bg: #0a0c10; --surface: #11151b; --surface-2: #161b23; --border: #232b36; --border-2: #303a48;
    --text: #e8edf3; --muted: #8a96a6; --accent: #ff6b35; --accent-2: #ffb347;
    --ok: #34d399; --warn: #fbbf24; --crit: #ff6b4a; --info: #60a5fa;
    --font: 'Plus Jakarta Sans', 'Segoe UI', system-ui, sans-serif;
    --mono: 'JetBrains Mono', 'Consolas', monospace;
}

/* Fonts are applied to text containers only, never to bare span/div, so the
   Material icon font used by Streamlit's own widgets keeps working. */
html, body, .stApp, .stMarkdown, .stMarkdown p, .stCaption, label, li, button, input, textarea,
[data-baseweb="select"], [data-baseweb="tab"], [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
    font-family: var(--font) !important;
}
h1, h2, h3, h4, h5 { font-family: var(--font) !important; color: var(--text) !important;
                     font-weight: 700 !important; letter-spacing: -0.02em; }

.stApp { background: radial-gradient(1200px 500px at 85% -10%, rgba(255,107,53,0.07), transparent 60%), var(--bg); }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #0d1015; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color: #c3ccd8 !important; }
.stMarkdown p, .stCaption, label { color: var(--muted); }

.stTabs [data-baseweb="tab"] { color: var(--muted); font-size: 15px; font-weight: 600; }
.stTabs [aria-selected="true"] { color: var(--text) !important; border-bottom: 2px solid var(--accent); }

.stButton > button, .stDownloadButton > button {
    border-radius: 10px; border: 1px solid var(--border-2); background: var(--surface-2);
    color: var(--text); font-weight: 600; transition: all .16s ease;
}
.stButton > button:hover { border-color: var(--accent); color: #fff; transform: translateY(-1px);
                           box-shadow: 0 6px 18px rgba(255,107,53,.18); }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #ff6b35, #ff9a3c); border: none; color: #170b04; }
.stButton > button[kind="primary"]:hover { color: #000; box-shadow: 0 8px 22px rgba(255,107,53,.35); }

.hero {
    position: relative; overflow: hidden;
    background: linear-gradient(135deg, #131820 0%, #0f1319 100%);
    border: 1px solid var(--border); border-radius: 16px;
    padding: 24px 30px; margin-bottom: 22px; animation: fadeSlideIn .45s ease-out;
}
.hero::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
                background: linear-gradient(180deg, var(--accent), var(--accent-2)); }
.hero::after { content: ""; position: absolute; right: -60px; top: -80px; width: 260px; height: 260px;
               background: radial-gradient(circle, rgba(255,107,53,.16), transparent 65%); pointer-events: none; }
@keyframes fadeSlideIn { from { opacity: 0; transform: translateY(-6px); } to { opacity: 1; transform: translateY(0); } }
.hero .eyebrow { font-size: 11px; font-weight: 700; letter-spacing: .18em; text-transform: uppercase;
                 color: var(--accent); margin-bottom: 6px; }
.hero h1 { margin: 0 0 6px 0; font-size: 30px; line-height: 1.15; }
.hero .sub { color: var(--muted); font-size: 14px; line-height: 1.55; max-width: 900px; }
.hero .badge { display: inline-flex; align-items: center; gap: 6px; padding: 3px 11px; border-radius: 999px;
               font-size: 11px; font-weight: 700; letter-spacing: .1em; margin-left: 12px; vertical-align: middle; }
.hero .badge::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.badge-live { background: rgba(52,211,153,.10); color: var(--ok); border: 1px solid rgba(52,211,153,.35); }
.badge-live::before { animation: livePulse 1.6s ease-in-out infinite; }
.badge-demo { background: rgba(251,191,36,.10); color: var(--warn); border: 1px solid rgba(251,191,36,.35); }
@keyframes livePulse { 0%,100% { opacity: 1; } 50% { opacity: .25; } }

.kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin-bottom: 20px; }
.kpi { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px;
       transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease; animation: fadeSlideIn .45s ease-out; }
.kpi:hover { transform: translateY(-3px); border-color: var(--border-2); box-shadow: 0 8px 22px rgba(0,0,0,.4); }
.kpi .lbl { font-size: 11px; text-transform: uppercase; letter-spacing: .12em; color: var(--muted); font-weight: 600; }
.kpi .val { font-family: var(--mono); font-size: 28px; font-weight: 700; margin-top: 4px; color: var(--text); }
.kpi .val.ok { color: var(--ok); } .kpi .val.warn { color: var(--warn); }
.kpi .val.crit { color: var(--crit); text-shadow: 0 0 14px rgba(255,107,74,.35); }

.alert-card { border-left: 3px solid; border-radius: 10px; padding: 10px 14px; margin-bottom: 8px;
              background: var(--surface); border-top: 1px solid var(--border); border-right: 1px solid var(--border);
              border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: flex-start; }
.alert-card .left .zone { font-weight: 700; font-size: 14px; color: var(--text); }
.alert-card .left .reason { font-size: 12.5px; color: var(--muted); margin-top: 2px; }
.alert-card .score { font-family: var(--mono); font-size: 15px; font-weight: 700; white-space: nowrap; margin-left: 10px; }

.sec-hdr { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .16em; color: var(--muted);
           border-bottom: 1px solid var(--border); padding-bottom: 8px; margin: 22px 0 14px 0; }
.sec-hdr::before { content: ""; display: inline-block; width: 18px; height: 2px; background: var(--accent);
                   vertical-align: middle; margin-right: 10px; }

.legend { display: flex; gap: 18px; font-size: 12.5px; color: var(--muted); flex-wrap: wrap; margin-top: 8px; }
.legend .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }

.info-box { background: var(--surface); border: 1px solid var(--border); border-left: 3px solid var(--info);
            border-radius: 10px; padding: 13px 17px; font-size: 14px; color: #b4bfcd; line-height: 1.6; }
.briefing-card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
                 padding: 20px 24px; margin-bottom: 14px; line-height: 1.75; font-size: 15px; color: #c6d0dc; }
.briefing-card h4 { color: var(--text) !important; margin-top: 0; }

@keyframes pulseGlow { 0% { box-shadow: 0 0 0 0 rgba(255,107,74,0); } 50% { box-shadow: 0 0 22px 3px var(--pulse-color, rgba(255,107,74,.4)); }
                       100% { box-shadow: 0 0 0 0 rgba(255,107,74,0); } }
.scenario-banner { border-radius: 12px; padding: 14px 20px; margin-bottom: 14px; border: 1px solid; font-size: 14.5px;
                   display: flex; align-items: center; gap: 14px; animation: pulseGlow 2.4s ease-in-out infinite; }
.scenario-banner .icon { font-family: var(--mono); font-size: 12px; font-weight: 700; letter-spacing: .1em;
                         padding: 4px 9px; border-radius: 6px; border: 1px solid currentColor; white-space: nowrap; }
.scenario-banner .txt b { font-size: 15px; }

/* first sidebar entry is the entry script (app.py); show a proper name */
[data-testid="stSidebarNav"] ul li:first-child a span { display: none !important; }
[data-testid="stSidebarNav"] ul li:first-child a::after { content: "Command Center"; font-size: 14px; color: #c3ccd8; padding-left: 4px; }
[data-testid="stPageLink"] a { justify-content: center; border: 1px solid var(--border); border-radius: 10px;
                                background: var(--surface-2); }
[data-testid="stPageLink"] a:hover { border-color: var(--accent); }
[data-testid="stPageLink"] a p { color: var(--text) !important; font-weight: 600; }
.nav-card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 18px 20px; height: 250px; overflow: hidden;
            transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease; }
.nav-card:hover { transform: translateY(-4px); border-color: var(--accent); box-shadow: 0 10px 26px rgba(255,107,53,.14); }
.nav-card .idx { font-family: var(--mono); font-size: 11px; color: var(--accent); letter-spacing: .14em; margin-bottom: 8px; }
.nav-card h4 { color: var(--text) !important; margin: 0 0 6px 0; font-size: 16px; }
.nav-card p { font-size: 13px; color: var(--muted); margin: 0; line-height: 1.55; }

/* ── sidebar navigation: rounded glass cards, blue / pink accents ── */
[data-testid="stSidebarNav"] { padding-top: 8px; }
[data-testid="stSidebarNav"] ul { gap: 6px; display: flex; flex-direction: column; }
[data-testid="stSidebarNav"] ul li { margin: 0 !important; }
[data-testid="stSidebarNav"] a {
    position: relative; display: flex !important; align-items: center; gap: 10px; min-height: 40px;
    padding: 8px 12px 8px 40px !important; border-radius: 12px !important;
    background: rgba(255,255,255,0.025) !important; border: 1px solid rgba(143,211,255,0.10);
    transition: background .18s ease, border-color .18s ease, box-shadow .18s ease, transform .18s ease; }
[data-testid="stSidebarNav"] a span { color: #c3ccd8 !important; font-weight: 600 !important; font-size: 14px !important; }
[data-testid="stSidebarNav"] a:hover {
    background: linear-gradient(90deg, rgba(143,211,255,0.08), rgba(255,140,207,0.06)) !important;
    border-color: rgba(143,211,255,0.38); box-shadow: 0 0 0 1px rgba(143,211,255,0.06), 0 6px 18px rgba(112,215,255,0.10);
    transform: translateX(2px); }
[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: linear-gradient(90deg, rgba(143,211,255,0.16), rgba(217,140,255,0.12)) !important;
    border: 1px solid transparent;
    background-clip: padding-box;
    box-shadow: inset 0 0 0 1px rgba(143,211,255,0.55), 0 0 18px rgba(255,140,207,0.16); }
[data-testid="stSidebarNav"] a[aria-current="page"] span,
[data-testid="stSidebarNav"] ul li:first-child a[aria-current="page"]::after { color: #ffffff !important; }
[data-testid="stSidebarNav"] a::before {
    content: ""; position: absolute; left: 12px; top: 50%; width: 18px; height: 18px; transform: translateY(-50%);
    background: linear-gradient(135deg, #8FD3FF, #FF8CCF); -webkit-mask: var(--ico) center / contain no-repeat;
    mask: var(--ico) center / contain no-repeat; opacity: .85; }
[data-testid="stSidebarNav"] a[aria-current="page"]::before { opacity: 1; }
[data-testid="stSidebarNav"] li:first-child a::before { --ico: url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='3' width='7' height='7' rx='1.5'/%3E%3Crect x='14' y='3' width='7' height='7' rx='1.5'/%3E%3Crect x='3' y='14' width='7' height='7' rx='1.5'/%3E%3Crect x='14' y='14' width='7' height='7' rx='1.5'/%3E%3C/svg%3E"); }
[data-testid="stSidebarNav"] a[href*="What_If"]::before { --ico: url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0'/%3E%3Ccircle cx='16' cy='6' r='2'/%3E%3Ccircle cx='10' cy='12' r='2'/%3E%3Ccircle cx='18' cy='18' r='2'/%3E%3C/svg%3E"); }
[data-testid="stSidebarNav"] a[href*="Spread_Simulation"]::before { --ico: url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 3c1 3 4 4.5 4 8.5a4 4 0 0 1-8 0c0-1.6.8-2.7 1.6-3.6.3 1.4 1.2 2.1 1.9 2.1-.6-2.5.5-5 .5-7z'/%3E%3Cpath d='M5 20h14'/%3E%3C/svg%3E"); }
[data-testid="stSidebarNav"] a[href*="Historical"]::before { --ico: url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='8.5'/%3E%3Cpath d='M12 7.5V12l3 2'/%3E%3C/svg%3E"); }
[data-testid="stSidebarNav"] a[href*="AI_Situation"]::before { --ico: url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 3l1.8 4.6L18.5 9l-4.7 1.6L12 15l-1.8-4.4L5.5 9l4.7-1.4z'/%3E%3Cpath d='M18 15l.8 2 2 .8-2 .8-.8 2-.8-2-2-.8 2-.8z'/%3E%3C/svg%3E"); }
[data-testid="stSidebarNav"] a[href*="Model_Insights"]::before { --ico: url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4 20V10M10 20V4M16 20v-7M22 20H2'/%3E%3C/svg%3E"); }
[data-testid="stSidebarNav"] a[href*="Admin"]::before { --ico: url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 3l7 3v5c0 4.5-3 8-7 10-4-2-7-5.5-7-10V6z'/%3E%3Cpath d='M9 12l2 2 4-4'/%3E%3C/svg%3E"); }
[data-testid="stSidebarNav"] a[href*="Activity_Log"]::before { --ico: url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M8 6h12M8 12h12M8 18h12'/%3E%3Ccircle cx='4' cy='6' r='1'/%3E%3Ccircle cx='4' cy='12' r='1'/%3E%3Ccircle cx='4' cy='18' r='1'/%3E%3C/svg%3E"); }
[data-testid="stSidebar"] { background: radial-gradient(420px 260px at 0% 0%, rgba(143,211,255,0.06), transparent 70%),
                            radial-gradient(380px 260px at 100% 100%, rgba(217,140,255,0.05), transparent 70%), #0b0e13 !important; }

/* ── Command Center platform hero ── */
.platform-hero { position: relative; display: grid; grid-template-columns: minmax(0,1.05fr) minmax(0,1fr); gap: 0;
    border-radius: 18px; overflow: hidden; margin-bottom: 18px; border: 1px solid rgba(143,211,255,0.16);
    background: linear-gradient(135deg, rgba(17,21,27,0.96), rgba(13,16,22,0.96));
    box-shadow: 0 18px 50px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.03); }
.platform-hero .copy { padding: 26px 28px; display: flex; flex-direction: column; justify-content: center; }
.platform-hero .eyebrow { font-size: 11px; font-weight: 700; letter-spacing: .18em; text-transform: uppercase;
    background: linear-gradient(90deg, #8FD3FF, #FF8CCF); -webkit-background-clip: text; background-clip: text; color: transparent; }
.platform-hero h1 { font-size: 28px; line-height: 1.18; margin: 8px 0 8px 0; }
.platform-hero .sub { color: var(--muted); font-size: 13.5px; line-height: 1.6; max-width: 560px; }
.platform-hero .chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 14px; }
.platform-hero .chip { font-size: 11px; color: #cfe9ff; padding: 4px 10px; border-radius: 999px;
    border: 1px solid rgba(143,211,255,0.25); background: rgba(143,211,255,0.06); }
.platform-hero .chip.fire { color: #ffc29a; border-color: rgba(255,140,60,0.35); background: rgba(255,107,53,0.08); }
.platform-hero .meta { margin-top: 14px; font-size: 12px; color: var(--muted); }
.platform-hero .visual { position: relative; min-height: 250px; }
.platform-hero .visual svg { position: absolute; inset: 0; width: 100%; height: 100%; }
@media (max-width: 900px) { .platform-hero { grid-template-columns: 1fr; } .platform-hero .visual { min-height: 200px; } }

[data-testid="stMetricValue"] { font-family: var(--mono) !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 10px; }
hr { border-color: var(--border) !important; }
</style>
"""

SEV_COLOR = {"EXTREME": "#ff6b4a", "HIGH": "#fbbf24", "MODERATE": "#60a5fa", "LOW": "#34d399"}

RISK_CS = [
    [0.00, "#152238"], [0.20, "#1e3a5f"],
    [0.40, "#8a6410"], [0.65, "#e0a21b"],
    [0.80, "#ff8a3d"], [1.00, "#ff5a3c"],
]

CA_CS = [
    [0.00, "#1a3d1f"], [0.24, "#1a3d1f"],
    [0.25, "#ff8a00"], [0.49, "#ff2d00"],
    [0.50, "#232b36"], [0.74, "#232b36"],
    [0.75, "#1f3a5f"], [1.00, "#1f3a5f"],
]


# ── model / twin ──────────────────────────────────────────────────────────── #

@st.cache_resource(show_spinner=False)
def _load_model_file(model_file: str):
    from src.ml_models.model_registry import load_xgb
    return load_xgb(model_file)


def load_ml_model(region=None):
    """Trained model serving `region` (Karnataka model by default)."""
    from src.ml_models.model_registry import choose_for_region
    return _load_model_file(choose_for_region(region).model_file)


def get_twin(offline: bool, scenario: dict = None, region=None) -> DigitalTwin:
    return DigitalTwin(ml_model=load_ml_model(region), offline=offline,
                        scenario=scenario, region=region)


from src.regions import REGION_PRESETS  # noqa: E402 - shared with the scheduler


def _pref(key: str, default):
    return st.session_state.get(f"_pref_{key}", default)


def _remember(key: str, value):
    """Sidebar choices are kept under `_pref_<key>`, which is never a widget
    key. Streamlit drops (or resets) the state of keyed widgets when you move
    between pages, which snapped the region back to Karnataka. Widgets here
    therefore carry NO key: each is created with its initial value taken from
    the remembered choice, so whatever the frontend does with widget state on
    a page change, the sidebar comes back showing the user's choice."""
    st.session_state[f"_pref_{key}"] = value


DEFAULT_PRESET = next(iter(REGION_PRESETS))


def current_region():
    """The region chosen in the sidebar, readable without drawing it (used by
    pages that show the region but never render the widget)."""
    choice = _pref("region_preset_choice", DEFAULT_PRESET)
    if choice == "Custom":
        return _custom_region()
    return REGION_PRESETS.get(choice, REGION)


def _custom_region():
    return RegionConfig(name=_pref("custom_region_name", "Custom region"),
                        min_lat=_pref("custom_min_lat", 11.5), max_lat=_pref("custom_max_lat", 15.5),
                        min_lon=_pref("custom_min_lon", 74.0), max_lon=_pref("custom_max_lon", 77.5),
                        grid_resolution_deg=0.5, weather_grid_resolution_deg=1.5)


def build_sidebar(show_scenario: bool = False, show_offline: bool = True):
    """Shared sidebar: offline toggle + region picker + refresh. Present on
    every page, and both choices persist across pages (see _remember)."""
    st.sidebar.markdown("### Controls")
    offline_default = _pref("offline_mode", not bool(os.getenv("FIRMS_MAP_KEY")))
    if show_offline:
        offline = st.sidebar.toggle(
            "Offline / demo mode", value=offline_default,
            help="Uses synthetic data. Turn off to use live satellite and weather feeds "
                 "(needs API keys in .env or Streamlit Secrets).")
    else:
        offline = bool(offline_default)
    _remember("offline_mode", offline)

    scenario = st.session_state.get("scenario") if not show_scenario else None
    if show_scenario and offline:
        st.sidebar.markdown("**Scenario**")
        n_hotspots = st.sidebar.slider("Active fire hotspots", 0, 40, _pref("scenario_hotspots", 8))
        temp_c = st.sidebar.slider("Temperature (°C)", 15, 48, _pref("scenario_temp", 32))
        wind_speed_ms = st.sidebar.slider("Wind speed (m/s)", 0.0, 20.0, float(_pref("scenario_wind", 5.0)))
        humidity_pct = st.sidebar.slider("Humidity (%)", 0, 100, _pref("scenario_humidity", 40))
        for k, v in (("scenario_hotspots", n_hotspots), ("scenario_temp", temp_c),
                     ("scenario_wind", wind_speed_ms), ("scenario_humidity", humidity_pct)):
            _remember(k, v)
        scenario = {"n_hotspots": n_hotspots, "temp_c": temp_c,
                    "wind_speed_ms": wind_speed_ms, "humidity_pct": humidity_pct}
        st.session_state["scenario"] = scenario

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Region**")
    options = list(REGION_PRESETS.keys()) + ["Custom"]
    remembered = _pref("region_preset_choice", DEFAULT_PRESET)
    if remembered not in options:
        remembered = DEFAULT_PRESET
    choice = st.sidebar.selectbox("Preset", options, index=options.index(remembered))
    _remember("region_preset_choice", choice)

    if choice == "Custom":
        c1, c2 = st.sidebar.columns(2)
        vals = {
            "custom_min_lat": c1.number_input("Min lat", value=float(_pref("custom_min_lat", 11.5)), format="%.2f"),
            "custom_max_lat": c2.number_input("Max lat", value=float(_pref("custom_max_lat", 15.5)), format="%.2f"),
            "custom_min_lon": c1.number_input("Min lon", value=float(_pref("custom_min_lon", 74.0)), format="%.2f"),
            "custom_max_lon": c2.number_input("Max lon", value=float(_pref("custom_max_lon", 77.5)), format="%.2f"),
            "custom_region_name": st.sidebar.text_input("Name", value=_pref("custom_region_name", "Custom region")),
        }
        for k, v in vals.items():
            _remember(k, v)
        region = _custom_region()
        problem = region_bounds_error(region)
        if problem:
            # An inverted or oversized box has no grid cells and used to crash the
            # data pipeline with a KeyError. Say what is wrong and stop cleanly.
            st.sidebar.error(problem)
            st.error(f"The custom region is not valid. {problem} Fix the values in the sidebar, "
                     "or pick a preset region.")
            st.stop()
    else:
        region = REGION_PRESETS[choice]

    st.sidebar.markdown("---")
    refresh = st.sidebar.button("Refresh data", use_container_width=True, key="refresh_btn")
    cell_km = region.grid_resolution_deg * 111  # rough deg->km at this latitude band
    st.sidebar.caption(f"{region.name}  ·  alert threshold {SYSTEM.alert_threshold_pct:.0f}%  ·  "
                        f"grid {region.grid_resolution_deg}° (~{cell_km:.0f} km cells)")

    return offline, scenario, region, refresh


def log_action(event_type: str, detail: str, region_name: Optional[str] = None):
    """Activity-log entry attributed to the signed-in user (never raises)."""
    from src.storage import database as db
    db.log_activity(event_type, detail, actor=st.session_state.get("auth_user") or "system",
                    region_name=region_name)


def _sim_detail(history, forced: bool = False) -> str:
    if not history or len(history) <= 1:
        return "Spread simulation run - no HIGH/EXTREME zones to ignite"
    last = history[-1]
    return (f"Spread simulation{' (forced ignition, top 5 zones)' if forced else ''}: "
            f"{int(last.n_burned)} cells burned, {int(last.n_burning)} still burning "
            f"after {int(last.minutes_elapsed)} min")


def _region_key(region) -> tuple:
    return (region.name, round(region.min_lat, 3), round(region.max_lat, 3),
            round(region.min_lon, 3), round(region.max_lon, 3))


def _refresh_reason(offline: bool, scenario: dict, region, force_refresh: bool,
                    trigger: Optional[str]) -> Optional[str]:
    """Why this run needs a refresh (None = no refresh needed). The text is
    what appears in the activity log."""
    if trigger:
        return trigger
    if "twin" not in st.session_state:
        return "Session started"
    if force_refresh:
        return "Manual refresh"
    if st.session_state.get("_last_region_key") != _region_key(region):
        return f"Region changed to {region.name}"
    if st.session_state.get("_last_offline") != offline:
        return "Switched to offline data" if offline else "Switched to live satellite + weather data"
    if st.session_state.get("_last_scenario") != scenario:
        return "Scenario changed"
    return None


def _persist_async(twin: DigitalTwin, actor: str, trigger: str) -> dict:
    """Snapshot + alert + activity-log writes go to a remote database (several
    round trips), so they run on a background thread instead of blocking the
    page. The returned dict is filled in when the thread finishes."""
    import threading
    from src.digital_twin.twin_state import persist_snapshot_and_notify
    holder = {"persisted": None, "new_extreme_count": 0, "emailed": False, "pending": True}

    def _run():
        try:
            holder.update(persist_snapshot_and_notify(twin, actor=actor, trigger=trigger))
        except Exception as exc:  # never let persistence break the dashboard
            holder.update(persisted=False, error=str(exc))
        holder["pending"] = False

    threading.Thread(target=_run, daemon=True, name="persist-snapshot").start()
    return holder


_SHARED_TWINS: dict = {}          # process-wide: (region, mode, scenario) -> (twin, timestamp)
_SHARED_LOCK = __import__("threading").Lock()


def _twin_key(offline: bool, scenario: Optional[dict], region) -> tuple:
    return (region.name, round(region.min_lat, 3), round(region.max_lat, 3),
            round(region.min_lon, 3), round(region.max_lon, 3), bool(offline),
            tuple(sorted(scenario.items())) if scenario else None)


def ensure_twin(offline: bool, scenario: dict, region, force_refresh: bool,
                trigger: Optional[str] = None) -> DigitalTwin:
    """Central session-state gatekeeper: every page calls this instead of
    building its own twin, so switching pages never silently loses or
    duplicates the live state. Twins are also kept per (region, mode,
    scenario) for one refresh interval, so flipping back to a region already
    visited is instant instead of re-fetching everything."""
    reason = _refresh_reason(offline, scenario, region, force_refresh, trigger)
    if reason:
        key = _twin_key(offline, scenario, region)
        cache = st.session_state.setdefault("_twin_cache", {})
        max_age = SYSTEM.min_refresh_interval_minutes * 60
        hit = cache.get(key)
        if hit is None or (time.time() - hit[1]) >= max_age:
            # Another visitor/tab may already have loaded this region within the
            # refresh interval; sharing it makes sign-in and reloads instant.
            with _SHARED_LOCK:
                shared = _SHARED_TWINS.get(key)
            if shared is not None and (time.time() - shared[1]) < max_age:
                hit = shared
        reusable = (hit is not None and not force_refresh and not trigger
                    and (time.time() - hit[1]) < max_age)
        if reusable:
            twin, ts = hit
            st.session_state["twin"] = twin
            st.session_state["_last_refresh_ts"] = ts
            cache[key] = hit
        else:
            with st.spinner(f"Loading {region.name}..."):
                twin = get_twin(offline, scenario, region)
                twin.refresh()
            st.session_state["persist_result"] = _persist_async(
                twin, st.session_state.get("auth_user") or "system", reason)
            st.session_state["twin"] = twin
            st.session_state["_last_refresh_ts"] = time.time()
            cache[key] = (twin, st.session_state["_last_refresh_ts"])
            with _SHARED_LOCK:
                _SHARED_TWINS[key] = cache[key]
                for k in [k for k, v in _SHARED_TWINS.items() if time.time() - v[1] > 2 * max_age]:
                    _SHARED_TWINS.pop(k, None)
            for k in [k for k, v in cache.items() if time.time() - v[1] > 2 * max_age]:
                cache.pop(k, None)           # keep the per-session cache small
        st.session_state["ca_history"] = None
        st.session_state["_last_region"] = region.name
        st.session_state["_last_region_key"] = _region_key(region)
        st.session_state["_last_offline"] = offline
        st.session_state["_last_scenario"] = scenario
    return st.session_state["twin"]


def render_autorefresh_status(offline: bool, scenario: dict, region):
    """Self-refreshing status widget: reruns itself every
    SYSTEM.min_refresh_interval_minutes for as long as this page has an open
    browser session, with NO separate process required. This is what makes
    the system "keep refreshing and storing itself" after deployment (e.g.
    Streamlit Community Cloud), where you can't leave a second terminal
    running src/scheduler/scheduler.py alongside the app.

    When its timer fires, it refreshes the twin, persists a new snapshot +
    alert rows to the database (same as every other refresh path - see
    ensure_twin/persist_snapshot_and_notify), then triggers ONE full-app
    rerun so the map/KPIs/gauge elsewhere on the page pick up the new data.
    That full rerun is safe: every stateful widget in this file carries an
    explicit `key` with an initialize-only-if-absent guard, so it no longer
    gets reset by a rerun (see the note in build_sidebar).

    Limitation worth knowing for the review: this only fires while at least
    one browser tab/session is connected to the running app - it is not an
    OS-level cron independent of any visitor. On a host that can put an idle
    app to sleep (e.g. Streamlit Community Cloud's free tier), a fully
    unvisited app can still go to sleep; the first visit after that just
    wakes it and refreshes immediately. For a guaranteed wake-up on a fixed
    clock regardless of visitors, you'd additionally need an external pinger
    (e.g. a scheduled GitHub Actions job or UptimeRobot hitting the app's
    URL) - not needed to demonstrate self-refresh + self-storage, only to
    guarantee it while genuinely nobody is looking.
    """

    # Watcher: checks every 5 s whether a refresh is due and does nothing
    # else, so nothing on screen re-renders (no flicker, JS countdown keeps
    # ticking). The countdown is measured from the last DATA refresh, so a
    # single long run_every tick could land before it hits zero and skip a
    # whole interval; frequent cheap checks make the refresh fire at 0.
    @st.fragment(run_every=timedelta(seconds=5))
    def _watcher():
        last_ts = st.session_state.get("_last_refresh_ts")
        due = last_ts is None or (time.time() - last_ts) >= SYSTEM.min_refresh_interval_minutes * 60
        if due:
            ensure_twin(offline, scenario, region, force_refresh=True,
                        trigger="Scheduled auto-refresh")
            st.rerun()

    _watcher()

    last_ts = st.session_state.get("_last_refresh_ts") or time.time()
    now = time.time()
    remaining = max(0, int(SYSTEM.min_refresh_interval_minutes * 60 - (now - last_ts)))
    # A plain st.caption would only visually update whenever Streamlit
    # happens to rerun (i.e. once per interval) - not a smooth countdown.
    # This renders a tiny client-side timer instead, so the number
    # actually ticks down second-by-second in the browser.
    components.html(
        f"""
        <div style="font-family:'Plus Jakarta Sans','Segoe UI',sans-serif;font-size:13px;
                    color:#8a96a6;display:flex;align-items:center;gap:8px;
                    margin-top:2px;">
          <span style="width:8px;height:8px;border-radius:50%;
                       background:#34d399;display:inline-block;"></span>
          <span>Next automatic update in
            <b id="dt-cd" style="color:#e8edf3;font-family:'JetBrains Mono',monospace;"></b></span>
        </div>
        <script>
          let remaining = {remaining};
          const el = document.getElementById('dt-cd');
          function render() {{
            const m = Math.floor(remaining / 60);
            const s = remaining % 60;
            el.textContent = m + 'm ' + String(s).padStart(2, '0') + 's';
          }}
          render();
          const t = setInterval(() => {{
            remaining = Math.max(0, remaining - 1);
            render();
            if (remaining <= 0) {{ clearInterval(t); el.textContent = 'refreshing...'; }}
          }}, 1000);
        </script>
        """,
        height=26,
    )




# ── shared: weather-scenario controls (What-If page + Spread Simulation) ──── #

SCENARIO_PRESETS = {
    "Monsoon calm": dict(n_hotspots=0, temp_c=24, wind_speed_ms=2.0, humidity_pct=85),
    "Typical dry day": dict(n_hotspots=5, temp_c=33, wind_speed_ms=4.0, humidity_pct=35),
    "Extreme fire weather": dict(n_hotspots=20, temp_c=42, wind_speed_ms=12.0, humidity_pct=12),
    "High wind event": dict(n_hotspots=10, temp_c=36, wind_speed_ms=18.0, humidity_pct=25),
}


def render_scenario_controls(key_prefix: str):
    """
    Renders the weather-scenario sliders + one-click preset buttons, shared
    between the What-If Simulator and the Spread Simulation page's "custom
    scenario" mode. Every widget is explicitly keyed and every preset button
    writes directly into those keys before st.rerun() - this is what makes
    clicking a preset actually move the sliders (previously the sliders had
    no key, so a preset only changed a local variable for one script run
    without moving the widget itself or forcing a recompute - it looked like
    clicking the button did nothing).

    Returns (scenario: dict, just_applied_preset: bool). just_applied_preset
    is True only on the single rerun right after a preset button was
    clicked, so callers can auto-trigger a recompute for presets while still
    requiring an explicit "Apply" click for manual slider drags (recomputing
    ~1400 zones on every drag tick would be janky).
    """
    defaults = dict(n_hotspots=8, temp_c=32, wind_speed_ms=5.0, humidity_pct=40, wind_from_deg=225)
    for k, v in defaults.items():
        sk = f"{key_prefix}_{k}"
        if sk not in st.session_state:
            st.session_state[sk] = v

    preset_cols = st.columns(len(SCENARIO_PRESETS))
    for col, (label, vals) in zip(preset_cols, SCENARIO_PRESETS.items()):
        with col:
            if st.button(label, use_container_width=True, key=f"{key_prefix}_preset_{label}"):
                st.session_state[f"{key_prefix}_n_hotspots"] = vals["n_hotspots"]
                st.session_state[f"{key_prefix}_temp_c"] = vals["temp_c"]
                st.session_state[f"{key_prefix}_wind_speed_ms"] = vals["wind_speed_ms"]
                st.session_state[f"{key_prefix}_humidity_pct"] = vals["humidity_pct"]
                st.session_state[f"_{key_prefix}_just_applied"] = label
                st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        n_hotspots = st.slider("Active fire hotspots seeded", 0, 40, key=f"{key_prefix}_n_hotspots",
                                help="How many synthetic active-fire detections to seed across the region")
        temp_c = st.slider("Temperature (°C)", 15, 48, key=f"{key_prefix}_temp_c")
    with c2:
        wind_speed_ms = st.slider("Wind speed (m/s)", 0.0, 20.0, step=0.5, key=f"{key_prefix}_wind_speed_ms")
        wind_from_deg = st.slider("Wind direction (blowing FROM, °)", 0, 355, step=5,
                                  key=f"{key_prefix}_wind_from_deg",
                                  help="Meteorological convention: 225° = wind from the south-west, "
                                       "so fire and smoke are pushed towards the north-east.")
        st.caption(f"Wind from the {_compass_direction_name(wind_from_deg)} → pushes fire towards the "
                   f"{_compass_direction_name(wind_from_deg + 180)}")
        humidity_pct = st.slider("Relative humidity (%)", 0, 100, key=f"{key_prefix}_humidity_pct")

    just_applied_label = st.session_state.pop(f"_{key_prefix}_just_applied", None)
    if just_applied_label:
        st.success(f"Preset loaded: {just_applied_label} — sliders updated above.")

    scenario = {"n_hotspots": n_hotspots, "temp_c": temp_c,
                "wind_speed_ms": wind_speed_ms, "humidity_pct": humidity_pct,
                "wind_from_deg": wind_from_deg}
    return scenario, bool(just_applied_label)


# ── shared: wind compass ───────────────────────────────────────────────────── #

def _compass_direction_name(deg: float) -> str:
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return dirs[round((deg % 360) / 22.5) % 16]


def render_wind_compass(wind_from_deg: float, wind_speed_ms: float, label: str = "Wind"):
    """Small inline-SVG compass showing which way the wind is blowing FROM
    (meteorological convention - matches what the CA simulator uses to bias
    spread direction), so the spread animation's shape is visually
    explainable rather than looking arbitrary."""
    # The arrow points in the direction the wind blows TOWARD (i.e. spread
    # direction), which is wind_from_deg + 180.
    arrow_deg = (wind_from_deg + 180) % 360
    dir_name = _compass_direction_name(wind_from_deg)
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:14px; background:#11151b;
                border:1px solid #303a48; border-radius:10px; padding:12px 16px;">
      <svg width="64" height="64" viewBox="0 0 64 64">
        <circle cx="32" cy="32" r="30" fill="#0a0c10" stroke="#303a48" stroke-width="1.5"/>
        <text x="32" y="10" text-anchor="middle" font-size="7" fill="#8a96a6">N</text>
        <text x="32" y="60" text-anchor="middle" font-size="7" fill="#8a96a6">S</text>
        <text x="6" y="35" text-anchor="middle" font-size="7" fill="#8a96a6">W</text>
        <text x="58" y="35" text-anchor="middle" font-size="7" fill="#8a96a6">E</text>
        <g transform="rotate({arrow_deg} 32 32)">
          <line x1="32" y1="46" x2="32" y2="16" stroke="#ff6b4a" stroke-width="3" stroke-linecap="round"/>
          <polygon points="32,10 26,22 38,22" fill="#ff6b4a"/>
        </g>
      </svg>
      <div>
        <div style="color:#8a96a6; font-size:12px; text-transform:uppercase; letter-spacing:1px;">{label}</div>
        <div style="color:#e8edf3; font-size:18px; font-weight:700;">{wind_speed_ms:.1f} m/s</div>
        <div style="color:#8a96a6; font-size:13px;">from the {dir_name} ({wind_from_deg:.0f}°)</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── render: header ────────────────────────────────────────────────────────── #

def render_header(summary: dict, offline: bool, subtitle: str = None, region=None):
    region = region or REGION
    badge_cls = "badge-demo" if offline else "badge-live"
    badge_txt = "DEMO" if offline else "LIVE"
    ts = summary.get("timestamp", "")[:19].replace("T", " ")
    title = subtitle or "Forest Fire Digital Twin"
    st.markdown(f"""
    <div class="hero">
      <div class="eyebrow">Forest Fire Digital Twin</div>
      <h1>{title}<span class="badge {badge_cls}">{badge_txt}</span></h1>
      <div class="sub">{region.name} &nbsp;·&nbsp; Last refreshed {ts} UTC
        &nbsp;·&nbsp; BMS College of Engineering · ISE Batch 42</div>
    </div>
    """, unsafe_allow_html=True)


def render_platform_hero(summary: dict, offline: bool, region=None):
    """Command Center hero: platform identity + a self-contained SVG illustration
    (procedural terrain, 25 m grid, simulation box, fire front, wind-driven
    plume). No external image is needed; it is an illustration, not data."""
    region = region or REGION
    ts = summary.get("timestamp", "")[:19].replace("T", " ")
    badge_cls, badge_txt = ("badge-demo", "DEMO") if offline else ("badge-live", "LIVE")
    grid = "".join(f'<line x1="{x}" y1="70" x2="{x - 60}" y2="250" />' for x in range(250, 520, 22))
    grid += "".join(f'<line x1="{250 - k * 3.3:.1f}" y1="{70 + k * 10}" x2="{520 - k * 3.3:.1f}" y2="{70 + k * 10}" />'
                    for k in range(0, 19))
    st.markdown(f"""
<div class="platform-hero">
  <div class="copy">
    <div class="eyebrow">Forest Fire Research Command Center</div>
    <h1>Forest Fire Digital Twin &amp; Wildfire Spread Simulation Platform
      <span class="badge {badge_cls}" style="font-size:11px;vertical-align:middle">{badge_txt}</span></h1>
    <div class="sub">Satellite fire detections and weather feed a trained risk model for every grid zone of
      <b>{region.name}</b>; a cellular-automata model then projects how a fire would spread across real
      Google satellite geography.</div>
    <div class="chips">
      <span class="chip">Google satellite base</span><span class="chip">NASA FIRMS VIIRS</span>
      <span class="chip">OpenWeatherMap</span><span class="chip">Canadian FWI</span>
      <span class="chip">XGBoost risk model</span><span class="chip fire">Cellular-automata spread</span>
    </div>
    <div class="meta">Last refreshed {ts} UTC · BMS College of Engineering · ISE Batch 42</div>
  </div>
  <div class="visual">
    <svg viewBox="0 0 560 300" preserveAspectRatio="xMidYMid slice" role="img" aria-label="Illustration: simulated fire front on a gridded forest">
      <defs>
        <filter id="ph-terrain" x="0" y="0" width="100%" height="100%">
          <feTurbulence type="fractalNoise" baseFrequency="0.018 0.026" numOctaves="5" seed="7" result="n"/>
          <feColorMatrix in="n" type="matrix" values="0 0 0 0 0.10  0 0 0 0 0.19  0 0 0 0 0.10  0 0 0 1.25 -0.25" result="g"/>
          <feTurbulence type="fractalNoise" baseFrequency="0.22" numOctaves="2" seed="3" result="f"/>
          <feColorMatrix in="f" type="matrix" values="0 0 0 0 0.05  0 0 0 0 0.12  0 0 0 0 0.06  0 0 0 0.9 -0.2" result="fg"/>
          <feBlend in="fg" in2="g" mode="multiply"/>
        </filter>
        <linearGradient id="ph-fade" x1="0" x2="1"><stop offset="0" stop-color="#11151b" stop-opacity="1"/>
          <stop offset="0.28" stop-color="#11151b" stop-opacity="0"/></linearGradient>
        <linearGradient id="ph-fire" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="#b3260a"/>
          <stop offset="0.45" stop-color="#ff7a1f"/><stop offset="1" stop-color="#ffe9a8"/></linearGradient>
        <filter id="ph-blur"><feGaussianBlur stdDeviation="6"/></filter>
        <filter id="ph-glow"><feGaussianBlur stdDeviation="3.2"/></filter>
        <radialGradient id="ph-smoke"><stop offset="0" stop-color="#b9b6b2" stop-opacity=".55"/>
          <stop offset="1" stop-color="#b9b6b2" stop-opacity="0"/></radialGradient>
      </defs>
      <rect width="560" height="300" fill="#16241a"/>
      <rect width="560" height="300" filter="url(#ph-terrain)"/>
      <path d="M0 236 C120 214 210 250 330 222 S 500 188 560 196" stroke="#8a7a5c" stroke-opacity=".45" stroke-width="3" fill="none"/>
      <g stroke="#e8f0ff" stroke-opacity=".16" stroke-width=".8">{grid}</g>
      <polygon points="250,70 520,70 460,250 190,250" fill="none" stroke="#8FD3FF" stroke-width="1.8" stroke-opacity=".9"/>
      <path d="M232 222 C262 205 286 214 312 194 S 360 176 384 160 S 420 146 442 128" fill="none" stroke="#2a1a12" stroke-width="34" stroke-opacity=".55" stroke-linecap="round" filter="url(#ph-blur)"/>
      <path d="M262 236 C292 220 318 226 344 206 S 392 188 414 172 S 452 158 476 140" fill="none" stroke="#ff6b1f" stroke-width="9" stroke-opacity=".75" stroke-linecap="round" filter="url(#ph-glow)"/>
      <g fill="url(#ph-fire)" opacity=".95">
        <path d="M270 234 q4 -16 2 -26 q7 10 6 26z"/><path d="M296 222 q5 -22 1 -32 q9 13 7 32z"/>
        <path d="M322 214 q4 -18 0 -28 q8 12 6 28z"/><path d="M350 202 q6 -24 1 -36 q10 15 7 36z"/>
        <path d="M378 188 q4 -17 1 -26 q8 11 6 26z"/><path d="M404 176 q6 -23 2 -34 q9 14 6 34z"/>
        <path d="M430 162 q4 -16 0 -24 q8 10 6 24z"/><path d="M456 150 q5 -20 1 -30 q9 12 7 30z"/>
      </g>
      <g filter="url(#ph-blur)">
        <ellipse cx="380" cy="140" rx="46" ry="28" fill="url(#ph-smoke)"/><ellipse cx="420" cy="112" rx="62" ry="34" fill="url(#ph-smoke)"/>
        <ellipse cx="470" cy="86" rx="76" ry="40" fill="url(#ph-smoke)"/><ellipse cx="525" cy="60" rx="90" ry="46" fill="url(#ph-smoke)"/>
      </g>
      <g font-family="JetBrains Mono, Consolas, monospace" font-size="10" fill="#cfe9ff">
        <text x="254" y="64">500 m × 500 m · 25 m cells</text>
        <text x="372" y="276" fill="#ffc29a">simulated fire front (illustration)</text>
      </g>
      <g transform="translate(222 92)"><circle r="15" fill="#0b0f15" fill-opacity=".8" stroke="#8FD3FF" stroke-opacity=".5"/>
        <path d="M-6 6 L6 -6 M6 -6 L0 -6 M6 -6 L6 0" stroke="#8FD3FF" stroke-width="2" fill="none" stroke-linecap="round"/></g>
      <rect width="560" height="300" fill="url(#ph-fade)"/>
    </svg>
  </div>
</div>
""", unsafe_allow_html=True)


def render_kpi_row(summary: dict):
    bd = summary.get("severity_breakdown", {})
    max_r = summary.get("max_risk_score", 0)
    alerts = summary.get("total_alerts", 0)
    max_cls = "crit" if max_r > 0.6 else ("warn" if max_r > 0.4 else "ok")
    al_cls = "crit" if alerts > 100 else ("warn" if alerts > 0 else "ok")

    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi"><div class="lbl">Grid zones</div>
        <div class="val">{summary.get('total_zones', 0)}</div></div>
      <div class="kpi"><div class="lbl">Active alerts</div>
        <div class="val {al_cls}">{alerts}</div></div>
      <div class="kpi"><div class="lbl">Extreme</div>
        <div class="val {'crit' if bd.get('EXTREME',0) else 'ok'}">{bd.get('EXTREME',0)}</div></div>
      <div class="kpi"><div class="lbl">High</div>
        <div class="val {'warn' if bd.get('HIGH',0) else 'ok'}">{bd.get('HIGH',0)}</div></div>
      <div class="kpi"><div class="lbl">Moderate</div>
        <div class="val">{bd.get('MODERATE',0)}</div></div>
      <div class="kpi"><div class="lbl">Peak risk</div>
        <div class="val {max_cls}">{max_r:.0%}</div></div>
      <div class="kpi"><div class="lbl">Mean risk</div>
        <div class="val">{summary.get('mean_risk_score',0):.0%}</div></div>
    </div>
    """, unsafe_allow_html=True)


# ── render: risk map ──────────────────────────────────────────────────────── #

@st.cache_data(show_spinner=False)
def _build_grid_geojson(zone_ids: tuple, lats: tuple, lons: tuple, half_deg: float):
    """
    Builds a GeoJSON FeatureCollection of filled rectangles, one per grid
    zone, from each zone's centroid +/- half the grid resolution. This
    replaces the old circular-scatter-marker map: at low zoom, hundreds of
    overlapping circles merge into an undifferentiated blob (flagged in
    review as 'grid looks fully covered when zoomed out'). Real filled grid
    tiles stay crisp and legible at any zoom level, and read as a genuine
    GIS risk product rather than a scatter plot.
    Cached because the grid geometry itself never changes between refreshes
    for a fixed region - only the risk VALUES change, which are passed
    separately as the choropleth's z data.
    """
    features = []
    for zid, lat, lon in zip(zone_ids, lats, lons):
        w, e = round(lon - half_deg, 4), round(lon + half_deg, 4)     # 4 dp ~ 11 m: keeps the payload small
        s_, n = round(lat - half_deg, 4), round(lat + half_deg, 4)
        features.append({
            "type": "Feature", "id": zid,
            "geometry": {"type": "Polygon", "coordinates": [[[w, s_], [e, s_], [e, n], [w, n], [w, s_]]]},
        })
    return {"type": "FeatureCollection", "features": features}


def _map_figure(processed: pd.DataFrame, risk_scores: np.ndarray, region, height: int):
    df = processed[["zone_id", "latitude", "longitude", "fwi", "active_fire_nearby"]].copy()
    df["risk_score"] = risk_scores
    df["risk_pct"] = (df["risk_score"] * 100).round(1)

    geojson = _build_grid_geojson(
        tuple(df["zone_id"]), tuple(df["latitude"]), tuple(df["longitude"]),
        half_deg=region.grid_resolution_deg / 2,
    )
    fig = go.Figure(go.Choroplethmap(
        geojson=geojson, locations=df["zone_id"], z=df["risk_score"],
        colorscale=RISK_CS, zmin=0, zmax=1, marker_opacity=0.78,
        marker_line_width=0.3, marker_line_color="#0a0c10",
        colorbar=dict(title="Risk", tickformat=".0%", tickfont={"color": "#8a96a6", "size": 10},
                      title_font={"color": "#8a96a6"}, len=0.65, thickness=10, bgcolor="rgba(0,0,0,0)"),
        customdata=np.stack([df["zone_id"], df["fwi"], df["risk_pct"]], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>FWI: %{customdata[1]:.1f}<br>"
                      "Risk: %{customdata[2]}%<extra></extra>",
    ))
    active = df[df["active_fire_nearby"]]
    if not active.empty:
        fig.add_trace(go.Scattermap(
            lat=active["latitude"], lon=active["longitude"], mode="markers",
            marker=dict(size=24, color="rgba(255,138,0,0.22)"), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scattermap(
            lat=active["latitude"], lon=active["longitude"], mode="markers",
            marker=dict(size=10, color="#ff9a3c"), hoverinfo="skip", showlegend=False))
    fig.update_layout(
        map=dict(style="carto-darkmatter", zoom=_zoom_for(region),
                  center={"lat": (region.min_lat + region.max_lat) / 2,
                          "lon": (region.min_lon + region.max_lon) / 2}),
        margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", height=height,
        font=dict(family="Plus Jakarta Sans, sans-serif"),
    )
    return fig, int(len(active))


def _zoom_for(region) -> float:
    """Zoom that fits the region's bounding box (a single fixed zoom cut off
    large states and wasted space on small ones)."""
    span = max(region.max_lat - region.min_lat, (region.max_lon - region.min_lon) * 0.85, 0.5)
    return float(np.clip(np.log2(360.0 / span) - 1.15, 4.6, 8.0))


def render_risk_map(processed: pd.DataFrame, risk_scores: np.ndarray, region=REGION, height=520,
                     scenario_active: bool = False):
    st.markdown('<div class="sec-hdr">Regional risk map</div>', unsafe_allow_html=True)
    # The figure only changes when the snapshot does, so build it once per
    # snapshot instead of on every Streamlit rerun (tab switch, slider, etc.).
    cache = st.session_state.setdefault("_map_cache", {})
    key = (id(risk_scores), region.name, height)
    if key not in cache:
        if len(cache) > 6:
            cache.clear()
        cache[key] = _map_figure(processed, risk_scores, region, height)
    fig, n_active = cache[key]
    st.plotly_chart(fig, use_container_width=True)

    if scenario_active and n_active:
        st.markdown(f"""
        <div class="scenario-banner" style="--pulse-color: rgba(255,154,60,0.4);
             background:rgba(255,154,60,0.08); border-color:rgba(255,154,60,0.4); color:#ff9a3c;">
          <span class="icon">HOTSPOTS</span>
          <span class="txt" style="color:#e8edf3"><b>{n_active} active hotspot zone{'s' if n_active != 1 else ''}</b>
          seeded by this scenario, shown as glowing markers on the map above.</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="legend">
      <span><span class="dot" style="background:#1e3a5f"></span>Low risk</span>
      <span><span class="dot" style="background:#8a6410"></span>Moderate</span>
      <span><span class="dot" style="background:#e0a21b"></span>Elevated</span>
      <span><span class="dot" style="background:#ff8a3d"></span>High</span>
      <span><span class="dot" style="background:#ff5a3c"></span>Extreme</span>
      <span><span class="dot" style="background:#ff9a3c;box-shadow:0 0 0 4px rgba(255,138,0,.25)"></span>Active hotspot zone</span>
    </div>
    """, unsafe_allow_html=True)


def render_risk_gauge(summary: dict):
    mean_r = summary.get("mean_risk_score", 0) * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=mean_r,
        number={"suffix": "%", "font": {"color": "#e8edf3", "size": 30}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8a96a6", "tickfont": {"color": "#8a96a6", "size": 9}},
            "bar": {"color": "#f0883e", "thickness": 0.22}, "bgcolor": "#11151b", "borderwidth": 0,
            "steps": [{"range": [0, 40], "color": "#1c2128"},
                      {"range": [40, 70], "color": "#2d2200"},
                      {"range": [70, 100], "color": "#3d1a1a"}],
            "threshold": {"line": {"color": "#ff6b4a", "width": 2}, "thickness": 0.8,
                          "value": SYSTEM.alert_threshold_pct},
        },
    ))
    fig.update_layout(height=200, margin=dict(l=16, r=16, t=10, b=0),
                       paper_bgcolor="rgba(0,0,0,0)", font={"color": "#8a96a6"})
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Mean regional risk · alert threshold at {SYSTEM.alert_threshold_pct:.0f}%")


def render_alerts(alerts, summary: dict, limit: int = 15):
    st.markdown('<div class="sec-hdr">Active alerts</div>', unsafe_allow_html=True)
    bd = summary.get("severity_breakdown", {})
    if bd:
        sevs = ["EXTREME", "HIGH", "MODERATE", "LOW"]
        counts = [bd.get(s, 0) for s in sevs]
        colors = [SEV_COLOR[s] for s in sevs]
        fig = go.Figure(go.Bar(x=sevs, y=counts, marker_color=colors,
                                text=counts, textposition="outside",
                                textfont={"color": "#8a96a6", "size": 11}))
        fig.update_layout(height=160, margin=dict(l=0, r=0, t=10, b=0),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           xaxis=dict(tickfont={"color": "#8a96a6", "size": 11}, gridcolor="#232b36"),
                           yaxis=dict(visible=False), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    if not alerts:
        st.markdown('<div class="info-box">No zones above the alert threshold.</div>', unsafe_allow_html=True)
        return

    for a in alerts[:limit]:
        c = SEV_COLOR.get(a.severity, "#8a96a6")
        st.markdown(f"""
        <div class="alert-card" style="border-left-color:{c}">
          <div class="left">
            <div class="zone">{a.zone_id} &nbsp;<span style="color:{c};font-size:11px">{a.severity}</span></div>
            <div class="reason">{a.reason}</div>
          </div>
          <div class="score" style="color:{c}">{a.risk_score:.0%}</div>
        </div>
        """, unsafe_allow_html=True)
    if len(alerts) > limit:
        st.caption(f"+ {len(alerts) - limit} more zones above threshold")


# ── render: CA simulation ─────────────────────────────────────────────────── #

def render_ca_simulation(twin: DigitalTwin, key_prefix: str = "ca", allow_force_ignite: bool = True):
    """
    key_prefix keeps this page's two modes (current live/demo state vs a
    custom what-if scenario) from overwriting each other's stored animation
    in session_state - each mode gets its own ca_history slot.
    """
    history_key = f"{key_prefix}_history"

    st.markdown('<div class="sec-hdr">Fire spread simulation — 2-hour projection</div>', unsafe_allow_html=True)
    st.caption("8-neighbour cellular automata · spreads faster downwind, slower into wet/low-fuel cells, "
               "biased uphill by real/synthetic terrain slope")

    if st.button("Run spread simulation", type="primary", key=f"{key_prefix}_run_btn"):
        with st.spinner("Simulating fire spread across the region..."):
            history = twin.simulate_spread_from_alerts()
        st.session_state[history_key] = history
        st.session_state[f"{key_prefix}_forced"] = False
        log_action("simulation", _sim_detail(history), twin.region.name)

    history = st.session_state.get(history_key)

    # IMPORTANT: `history is None` (button never clicked yet) and `history == []`
    # (button WAS clicked, and the simulation correctly found nothing to
    # ignite this cycle) must be told apart. An empty list is falsy in
    # Python, so a naive `if not history:` treats a genuine "ran, found
    # nothing" result identically to "never run" - showing the same "click
    # Run simulation" message before and after the click, which is exactly
    # why this looked like the button did nothing.
    if history is None:
        st.markdown('<div class="info-box">Click <b>Run spread simulation</b> above to project spread from '
                    'this scenario\'s current HIGH/EXTREME risk zones.</div>', unsafe_allow_html=True)
        return

    if len(history) <= 1:
        st.markdown(
            '<div class="info-box">No HIGH/EXTREME zones in this scenario right now, so there is nothing '
            'above the alert threshold to ignite from. This is a real result, not a bug - it means current '
            'conditions genuinely don\'t support fast fire spread (e.g. real live data during the monsoon '
            'season, or a calm synthetic draw in demo mode).</div>', unsafe_allow_html=True)
        if allow_force_ignite:
            if st.button("Force-ignite the 5 highest-risk zones anyway (demo mode)",
                         key=f"{key_prefix}_force_btn"):
                with st.spinner("Simulating fire spread from the top 5 riskiest zones..."):
                    history = twin.simulate_spread_from_top_n(5)
                st.session_state[history_key] = history
                st.session_state[f"{key_prefix}_forced"] = True
                log_action("simulation", _sim_detail(history, forced=True), twin.region.name)
                st.rerun()
        return

    if st.session_state.get(f"{key_prefix}_forced"):
        st.caption("Forced ignition: these 5 zones were the highest-risk this cycle, but none actually "
                   "crossed the HIGH/EXTREME alert threshold - this projection is illustrative, not a real alert.")

    processed = twin.current_snapshot.processed_grid
    # The wind that actually drove this simulation (forecast near the fire when
    # live, otherwise current wind at the ignition zones).
    wind = getattr(twin.current_snapshot, "ca_wind", None)
    if wind:
        mean_wind_speed, mean_wind_deg = float(wind["speed_ms"]), float(wind["from_deg"])
        wind_label = "Forecast wind" if wind["source"] == "forecast" else "Current wind"
    else:
        from src.data_ingestion.wind import mean_wind as _mean_wind
        mean_wind_speed, mean_wind_deg = _mean_wind(processed["wx_wind_speed_ms"].values,
                                                    processed["wx_wind_deg"].values)
        wind_label = "Current wind"

    frames = [go.Frame(data=[go.Heatmap(z=h.state, colorscale=CA_CS, zmin=0, zmax=3, showscale=False)],
                        name=str(h.minutes_elapsed)) for h in history]
    fig = go.Figure(data=[go.Heatmap(z=history[0].state, colorscale=CA_CS, zmin=0, zmax=3, showscale=False)],
                     frames=frames)
    fig.update_layout(
        height=440, margin=dict(l=0, r=0, t=10, b=60),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed", visible=False, scaleanchor="x"), xaxis=dict(visible=False),
        updatemenus=[{"type": "buttons", "showactive": False, "x": 0.0, "y": -0.12, "xanchor": "left",
                      "bgcolor": "#11151b", "bordercolor": "#303a48", "font": {"color": "#e8edf3"},
                      "buttons": [
                          {"label": "Play", "method": "animate",
                           "args": [None, {"frame": {"duration": 600, "redraw": True}, "fromcurrent": True,
                                            "transition": {"duration": 150}}]},
                          {"label": "Pause", "method": "animate",
                           "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
                      ]}],
        sliders=[{"active": 0, "x": 0.12, "len": 0.88, "y": -0.10, "bgcolor": "#11151b", "bordercolor": "#303a48",
                  "font": {"color": "#8a96a6", "size": 10},
                  "currentvalue": {"prefix": "T+", "suffix": " min", "font": {"color": "#e8edf3", "size": 12}, "xanchor": "right"},
                  "steps": [{"label": str(h.minutes_elapsed), "method": "animate",
                             "args": [[str(h.minutes_elapsed)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]}
                            for h in history]}],
    )

    final = history[-1]
    pct_burned = (final.n_burned / final.state.size * 100) if final.state.size else 0

    if pct_burned >= 40:
        verdict, v_color, v_icon = "Severe, fast-moving spread", "#ff6b4a", "SEVERE"
    elif pct_burned >= 15:
        verdict, v_color, v_icon = "Moderate spread", "#fbbf24", "MODERATE"
    elif pct_burned >= 3:
        verdict, v_color, v_icon = "Contained, slow spread", "#60a5fa", "CONTAINED"
    else:
        verdict, v_color, v_icon = "Minimal spread", "#34d399", "MINIMAL"

    dir_name = _compass_direction_name(mean_wind_deg)
    st.markdown(f"""
    <div class="scenario-banner" style="border-color:{v_color}55; background:{v_color}14; color:{v_color};
         --pulse-color:{v_color}55;">
      <div class="icon">{v_icon}</div>
      <div class="txt" style="color:#e8edf3"><b style="color:{v_color}">{verdict}</b> — projected to burn
        <b>{pct_burned:.0f}%</b> of the simulated area ({final.n_burned} of {final.state.size} cells)
        within {final.minutes_elapsed} minutes, pushed by wind from the <b>{dir_name}</b>
        at {mean_wind_speed:.1f} m/s.</div>
    </div>
    """, unsafe_allow_html=True)

    st.plotly_chart(fig, use_container_width=True)

    col_compass, col_kpis = st.columns([1, 3], gap="medium")
    with col_compass:
        render_wind_compass(mean_wind_deg, mean_wind_speed, label=wind_label)
    with col_kpis:
        c1, c2, c3, c4 = st.columns(4)
        for col, (lbl, val, cls) in zip([c1, c2, c3, c4],
            [("Horizon", f"{final.minutes_elapsed} min", ""), ("Steps simulated", len(history) - 1, ""),
             ("Cells burning now", final.n_burning, "warn"), ("Total cells burned", final.n_burned, "crit")]):
            col.markdown(f'<div class="kpi"><div class="lbl">{lbl}</div><div class="val {cls}">{val}</div></div>',
                          unsafe_allow_html=True)

    burned_series = [h.n_burned for h in history]
    burning_series = [h.n_burning for h in history]
    times = [h.minutes_elapsed for h in history]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=times, y=burned_series, name="Burned", line=dict(color="#ff6b4a", width=2),
                               fill="tozeroy", fillcolor="rgba(247,129,102,0.12)"))
    fig2.add_trace(go.Scatter(x=times, y=burning_series, name="Burning", line=dict(color="#fbbf24", width=2, dash="dot")))
    fig2.update_layout(height=160, margin=dict(l=0, r=0, t=20, b=0),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(title="Minutes", tickfont={"color": "#8a96a6", "size": 10}, gridcolor="#232b36"),
                        yaxis=dict(title="Cells", tickfont={"color": "#8a96a6", "size": 10}, gridcolor="#232b36"),
                        legend=dict(font={"color": "#8a96a6", "size": 10}, bgcolor="rgba(0,0,0,0)"),
                        title=dict(text="Spread over time", font={"color": "#8a96a6", "size": 11}, x=0))
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("""
    <div class="legend">
      <span><span class="dot" style="background:#1a3d1f"></span>Unburned forest</span>
      <span><span class="dot" style="background:#ff8a00"></span>Burning</span>
      <span><span class="dot" style="background:#232b36"></span>Burned / charred</span>
      <span><span class="dot" style="background:#1f3a5f"></span>Non-fuel (water / bare ground)</span>
    </div>
    """, unsafe_allow_html=True)


# ── render: model performance ─────────────────────────────────────────────── #

def render_model_performance():
    st.markdown('<div class="sec-hdr">Model performance — real data, temporal holdout (test = 2025 season)</div>',
                unsafe_allow_html=True)
    path = DATA_PROCESSED_DIR / "model_comparison_real.csv"
    if not path.exists():
        st.markdown('<div class="info-box">Run <code>python src/ml_models/train_real.py</code> to generate results.</div>',
                    unsafe_allow_html=True)
        return None

    alert_path = DATA_PROCESSED_DIR / "model_comparison_alert.csv"
    options = (["Alert (high accuracy)", "Early warning (high recall)"] if alert_path.exists()
               else ["Early warning (high recall)"])
    if st.session_state.get("perf_op_point") not in options:
        st.session_state["perf_op_point"] = options[0]
    op = st.radio("Operating point", options, key="perf_op_point", horizontal=True)
    alert_mode = op.startswith("Alert")
    st.caption("Same trained models at two decision thresholds, both chosen on training data: "
               "**Alert** classifies at least 92% of zone-days correctly; "
               "**Early warning** catches about 80% of fires at the cost of more false alarms.")
    if alert_mode:
        path = alert_path

    df = pd.read_csv(path, index_col=0)
    radar_cols = ["accuracy", "precision", "recall", "f1_score", "auc_roc", "avg_precision"]
    radar_cols = [c for c in radar_cols if c in df.columns]
    colors = ["#ff6b4a", "#fbbf24", "#34d399", "#8a96a6"]
    fill_colors = ["rgba(247,129,102,0.15)", "rgba(210,153,34,0.15)", "rgba(63,185,80,0.15)",
                   "rgba(139,148,158,0.10)"]

    fig = go.Figure()
    for i, model in enumerate(df.index):
        vals = df.loc[model, radar_cols].tolist()
        fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=radar_cols + [radar_cols[0]], fill="toself",
                                       name=model, line_color=colors[i % len(colors)],
                                       fillcolor=fill_colors[i % len(fill_colors)], opacity=0.9))
    fig.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(visible=True, range=[0, 1], tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                                    tickfont={"color": "#8a96a6", "size": 9}, gridcolor="#232b36", linecolor="#232b36"),
                   angularaxis=dict(tickfont={"color": "#c3ccd8", "size": 11}, gridcolor="#232b36", linecolor="#232b36")),
        showlegend=True, legend={"font": {"color": "#c3ccd8"}, "bgcolor": "rgba(0,0,0,0)"},
        paper_bgcolor="rgba(0,0,0,0)", height=400, margin=dict(l=50, r=50, t=30, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    metric_display = {"auc_roc": "AUC-ROC", "recall": "Recall", "precision": "Precision",
                       "f1_score": "F1", "avg_precision": "Avg Precision"}
    x_labels = [metric_display.get(c, c) for c in metric_display if c in df.columns]
    for i, model in enumerate(df.index):
        y_vals = [df.loc[model, c] for c in metric_display if c in df.columns]
        fig2.add_trace(go.Bar(name=model, x=x_labels, y=y_vals, marker_color=colors[i % len(colors)],
                               text=[f"{v:.3f}" for v in y_vals], textposition="outside",
                               textfont={"color": "#8a96a6", "size": 10}))
    fig2.update_layout(barmode="group", height=300, margin=dict(l=0, r=0, t=20, b=0),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(tickfont={"color": "#c3ccd8"}, gridcolor="#232b36"),
                        yaxis=dict(range=[0, 1.15], tickfont={"color": "#8a96a6"}, gridcolor="#232b36"),
                        legend=dict(font={"color": "#c3ccd8"}, bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig2, use_container_width=True)

    display_df = df.copy()
    fnr_col = "false_negative_rate" if "false_negative_rate" in display_df.columns else None
    other_cols = [c for c in display_df.columns if c != fnr_col]
    styled = display_df.style.format("{:.4f}")
    styled = styled.highlight_max(subset=other_cols, axis=0, props="background-color:#1a3d2b;color:#34d399")
    if fnr_col:
        styled = styled.highlight_min(subset=[fnr_col], axis=0, props="background-color:#1a3d2b;color:#34d399")
    st.dataframe(styled, use_container_width=True)

    v2 = df.drop(index=[i for i in df.index if "v1" in str(i)], errors="ignore")
    v1 = df[[("v1" in str(i)) for i in df.index]]
    best = v2["auc_roc"].idxmax() if not v2.empty else None
    if best is not None:
        txt = (f"Held-out 2025 season (never seen in training). Best AUC-ROC "
               f"<b>{v2.loc[best, 'auc_roc']:.3f}</b> ({best})")
        if not v1.empty:
            txt += (f" vs <b>{v1['auc_roc'].iloc[0]:.3f}</b> for the weather-only baseline; "
                    f"average precision {v2['avg_precision'].max():.2f} vs {v1['avg_precision'].iloc[0]:.2f}")
        if alert_mode:
            txt += (f". At the alert threshold, accuracy is <b>{v2['accuracy'].max():.1%}</b> with "
                    f"{v2.loc[v2['accuracy'].idxmax(), 'recall']:.0%} of fires caught.")
        else:
            txt += (f". At the early-warning threshold, {v2['recall'].max():.0%} of fires are caught; "
                    "precision is low because only about 4% of zone-days burn.")
        st.markdown(f'<div class="info-box">{txt}</div>', unsafe_allow_html=True)

    tiers_path = DATA_PROCESSED_DIR / "alert_tier_performance.csv"
    if tiers_path.exists():
        t = pd.read_csv(tiers_path)
        st.markdown('<div class="sec-hdr">Alert tiers on the held-out 2025 season</div>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Tier": t["tier"],
            "Zone-days flagged": (t["share_of_zone_days_flagged"] * 100).round(1).astype(str) + "%",
            "Fires captured": (t["share_of_fires_captured"] * 100).round(1).astype(str) + "%",
            "Hit rate": (t["hit_rate"] * 100).round(1).astype(str) + "%",
        }), use_container_width=True, hide_index=True)
    return df


def page_nav_card(icon: str, title: str, desc: str):
    """`icon` is now a short index label such as "01" (no emoji)."""
    st.markdown(f"""
    <div class="nav-card">
        <div class="idx">{icon}</div>
        <h4>{title}</h4>
        <p>{desc}</p>
    </div>
    """, unsafe_allow_html=True)
