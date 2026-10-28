"""Appearance: Default Dark (the existing theme) and Professional Navy Blue.

How it works
  * Default Dark is the dashboard exactly as it was: no rule below applies to it.
  * Every Professional Navy Blue rule is scoped to html[data-ff-theme="navy"]. The
    attribute is set in the browser by the theme_switch component
    (components/theme_switch/index.html), which also draws the single "⋮" Appearance
    menu in the header and remembers the choice in the browser (localStorage). Switching
    is therefore a CSS-only change: no Streamlit rerun, no page reload, no change to
    session state, simulation parameters, regions or model results.
  * The navy theme first re-points the shared design tokens (--bg, --surface, --text,
    ...) that the dashboard's own styles use, then restyles the components that use
    fixed colours (sidebar, ticker, alert panel, Streamlit widgets, charts, tables).
  * Data colours keep their meaning (green = low / healthy, amber = elevated,
    red = critical fire); satellite imagery, fire / smoke / ember effects and report
    documents are never tinted.
"""
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

_theme_switch = components.declare_component(
    "theme_switch", path=str(Path(__file__).resolve().parents[1] / "components" / "theme_switch"))

THEMES = (("dark", "Default Dark"), ("navy", "Professional Navy Blue"))
N = 'html[data-ff-theme="navy"]'

# Professional Navy Blue palette (reference: dark navy geospatial monitoring dashboard)
NAVY = {
    "bg": "#061426", "sidebar": "#03101F", "header": "#04172B", "card": "#081D33", "card2": "#0B2542",
    "input": "#0A2038", "primary": "#0878F9", "nav_active": "#0756C9", "cyan": "#00B8F0",
    "text": "#F3F7FC", "text2": "#B6C7DC", "muted": "#8298B3", "border": "#163A5E", "border2": "#1F4C78",
    "ok": "#20D995", "warn": "#FFB547", "crit": "#FF4D5A",
}

# Header "⋮" Appearance button + menu (styled for both themes; Default Dark look below)
MENU_CSS = """
/* The ONE application menu: fixed in the top-right of the header area. Streamlit's own toolbar menu is
   switched off with the supported config client.toolbarMode = "minimal" (.streamlit/config.toml). */
#ff-theme-wrap { position: fixed; top: 12px; right: 16px; display: flex; align-items: center; z-index: 1000001; }
#ff-theme-btn { width: 34px; height: 34px; border-radius: 9px; display: grid; place-items: center; padding: 0; cursor: pointer;
  background: transparent; border: 1px solid transparent; color: #C9D2DD; transition: background .15s, border-color .15s; }
#ff-theme-btn svg { width: 18px; height: 18px; fill: currentColor; }
#ff-theme-btn:hover, #ff-theme-btn[aria-expanded="true"] { background: rgba(255,255,255,0.06); border-color: rgba(255,255,255,0.14); }
#ff-theme-btn:focus-visible { outline: 2px solid #FFB347; outline-offset: 2px; }
#ff-theme-menu { position: absolute; top: calc(100% + 6px); right: 0; min-width: 244px; padding: 6px; border-radius: 12px;
  background: #111722; border: 1px solid #2A3445; box-shadow: 0 16px 40px rgba(0,0,0,0.5); }
#ff-theme-menu[hidden] { display: none; }
#ff-theme-menu .ff-tm-title { font: 700 10.5px/1 'JetBrains Mono', Consolas, monospace; letter-spacing: .14em; text-transform: uppercase;
  color: #8A96A6; padding: 8px 10px 8px 10px; }
#ff-theme-menu [data-theme] { width: 100%; display: flex; align-items: center; gap: 10px; padding: 9px 10px; border-radius: 8px;
  background: transparent; border: 0; cursor: pointer; color: #E8EDF3; font: 600 13px/1.2 'Plus Jakarta Sans', 'Segoe UI', sans-serif; text-align: left; }
#ff-theme-menu [data-theme]:hover, #ff-theme-menu [data-theme]:focus-visible { background: rgba(255,255,255,0.06); outline: none; }
#ff-theme-menu .ff-tm-l { flex: 1 1 auto; }
#ff-theme-menu .ff-tm-sep { height: 1px; margin: 6px 4px; background: rgba(255,255,255,0.10); }
#ff-theme-menu [data-action] { width: 100%; display: flex; align-items: center; gap: 10px; padding: 9px 10px; border-radius: 8px;
  background: transparent; border: 0; cursor: pointer; color: #E8EDF3; font: 600 13px/1.2 'Plus Jakarta Sans', 'Segoe UI', sans-serif; text-align: left; }
#ff-theme-menu [data-action]:hover, #ff-theme-menu [data-action]:focus-visible { background: rgba(255,255,255,0.06); outline: none; }
#ff-theme-menu [data-action] svg { width: 16px; height: 16px; flex: 0 0 auto; fill: none; stroke: currentColor; stroke-width: 2;
  stroke-linecap: round; stroke-linejoin: round; opacity: .8; }
#ff-theme-menu .ff-tm-k { margin-left: auto; font: 600 11px/1 'JetBrains Mono', Consolas, monospace; color: #8A96A6;
  border: 1px solid rgba(255,255,255,0.14); border-radius: 5px; padding: 3px 6px; }
#ff-theme-menu [data-action]:disabled { opacity: .45; cursor: default; }
#ff-theme-menu .ff-tm-sw { width: 18px; height: 18px; border-radius: 5px; flex: 0 0 auto; border: 1px solid rgba(255,255,255,0.25); }
#ff-theme-menu .ff-tm-dark { background: linear-gradient(135deg, #080B12 50%, #FF7A18 50%); }
#ff-theme-menu .ff-tm-navy { background: linear-gradient(135deg, #061426 50%, #0878F9 50%); }
#ff-theme-menu .ff-tm-ck { width: 16px; height: 16px; fill: none; stroke: #35D07F; stroke-width: 2.4; stroke-linecap: round;
  stroke-linejoin: round; visibility: hidden; }
#ff-theme-menu [aria-checked="true"] .ff-tm-ck { visibility: visible; }
#ff-theme-menu [aria-checked="true"] { background: rgba(77,163,255,0.10); }
/* the controller iframe takes no space (its script still runs while hidden) */
.st-key-ff_theme_host, [data-testid="stLayoutWrapper"]:has(> .st-key-ff_theme_host),
[data-testid="stElementContainer"]:has(> .st-key-ff_theme_host) { display: none !important; }
"""

C = NAVY
NAVY_CSS = f"""
/* ═════ Professional Navy Blue — every rule scoped to {N} ═════ */
{N} {{
  --bg: {C['bg']}; --surface: {C['card']}; --surface-2: {C['card2']}; --border: {C['border']}; --border-2: {C['border2']};
  --text: {C['text']}; --muted: {C['muted']};
  --accent: {C['primary']}; --accent-2: {C['cyan']}; --info: {C['cyan']};
  --ok: {C['ok']}; --warn: {C['warn']}; --crit: {C['crit']}; --alert: {C['crit']};
  --nav-blue: {C['primary']}; --nav-cyan: {C['cyan']}; color-scheme: dark;
}}
{N} body, {N} .stApp {{ color: {C['text']}; }}
{N} .stApp {{ background: radial-gradient(1100px 520px at 85% -12%, rgba(8,120,249,0.12), transparent 60%),
    radial-gradient(900px 420px at -10% 110%, rgba(0,184,240,0.06), transparent 60%), {C['bg']} !important; }}
{N} [data-testid="stHeader"] {{ background: transparent !important; }}   /* as in Default Dark: the ticker sits right under it */
{N} [data-testid="stHeader"] button, {N} [data-testid="stHeader"] a, {N} [data-testid="stHeader"] span {{ color: {C['text2']} !important; }}
{N} h1, {N} h2, {N} h3, {N} h4, {N} h5, {N} h6 {{ color: {C['text']} !important; }}
{N} .stMarkdown p, {N} .stMarkdown li, {N} label, {N} [data-testid="stWidgetLabel"] p {{ color: {C['text2']} !important; }}
{N} .stCaption, {N} [data-testid="stCaptionContainer"], {N} [data-testid="stCaptionContainer"] p {{ color: {C['muted']} !important; }}
{N} .stMarkdown strong, {N} .stMarkdown b {{ color: {C['text']}; }}
{N} a {{ color: {C['cyan']}; }}
{N} code {{ background: rgba(8,120,249,0.14) !important; color: #9FD8FF !important; }}
{N} hr {{ border-color: {C['border']} !important; }}
{N} ::selection {{ background: rgba(8,120,249,0.45); }}

/* header ⋮ menu */
{N} #ff-theme-btn {{ color: {C['text2']}; }}
{N} #ff-theme-btn:hover, {N} #ff-theme-btn[aria-expanded="true"] {{ background: rgba(8,120,249,0.14); border-color: {C['border2']}; }}
{N} #ff-theme-btn:focus-visible {{ outline-color: {C['cyan']}; }}
{N} #ff-theme-menu {{ background: {C['card2']}; border-color: {C['border2']}; box-shadow: 0 18px 44px rgba(0,0,0,0.55); }}
{N} #ff-theme-menu .ff-tm-title {{ color: {C['muted']}; }}
{N} #ff-theme-menu [data-theme] {{ color: {C['text']}; }}
{N} #ff-theme-menu [data-theme]:hover, {N} #ff-theme-menu [data-theme]:focus-visible {{ background: rgba(8,120,249,0.14); }}
{N} #ff-theme-menu [aria-checked="true"] {{ background: rgba(8,120,249,0.22); }}
{N} #ff-theme-menu .ff-tm-ck {{ stroke: {C['cyan']}; }}
{N} #ff-theme-menu .ff-tm-sep {{ background: {C['border']}; }}
{N} #ff-theme-menu [data-action] {{ color: {C['text']}; }}
{N} #ff-theme-menu [data-action]:hover, {N} #ff-theme-menu [data-action]:focus-visible {{ background: rgba(8,120,249,0.14); }}
{N} #ff-theme-menu .ff-tm-k {{ color: {C['muted']}; border-color: {C['border2']}; }}

/* focus ring */
{N} a:focus-visible, {N} button:focus-visible, {N} [role="tab"]:focus-visible, {N} input:focus-visible,
{N} [data-testid="stPageLink"] a:focus-visible {{ outline-color: {C['cyan']} !important; }}

/* ── sidebar / navigation ── */
{N} [data-testid="stSidebar"] {{ background: {C['sidebar']} !important; border-right: 1px solid {C['border']} !important; }}
{N} [data-testid="stSidebar"] p, {N} [data-testid="stSidebar"] label, {N} [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{ color: {C['text2']} !important; }}
{N} [data-testid="stSidebar"] h1, {N} [data-testid="stSidebar"] h2, {N} [data-testid="stSidebar"] h3,
{N} [data-testid="stSidebar"] strong, {N} [data-testid="stSidebar"] b {{ color: {C['text']} !important; }}
{N} [data-testid="stSidebar"] .stCaption, {N} [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{ color: {C['muted']} !important; }}
{N} [data-testid="stSidebar"] hr {{ border-color: {C['border']} !important; }}
{N} [data-testid="stSidebarNav"] a {{ background: {C['card']} !important; border: 1px solid {C['border']} !important; }}
{N} [data-testid="stSidebarNav"] a span, {N} [data-testid="stSidebarNav"] a span p,
{N} [data-testid="stSidebarNav"] ul li:first-child a::after {{ color: {C['text2']} !important; }}
{N} [data-testid="stSidebarNav"] a::before {{ background: {C['muted']} !important; }}
{N} [data-testid="stSidebarNav"] a:hover {{ background: {C['card2']} !important; border-color: {C['border2']} !important;
    box-shadow: inset 2px 0 0 {C['cyan']} !important; }}
{N} [data-testid="stSidebarNav"] a:hover span, {N} [data-testid="stSidebarNav"] a:hover span p {{ color: {C['text']} !important; }}
{N} [data-testid="stSidebarNav"] a[aria-current="page"] {{ background: linear-gradient(90deg, {C['nav_active']}, {C['primary']}) !important;
    border-color: rgba(0,184,240,0.45) !important; box-shadow: 0 6px 18px rgba(8,120,249,0.30), inset 0 0 0 1px rgba(255,255,255,0.10) !important; }}
{N} [data-testid="stSidebarNav"] a[aria-current="page"] span, {N} [data-testid="stSidebarNav"] a[aria-current="page"] span p,
{N} [data-testid="stSidebarNav"] ul li:first-child a[aria-current="page"]::after {{ color: #FFFFFF !important; }}
{N} [data-testid="stSidebarNav"] a[aria-current="page"]::before {{ background: #FFFFFF !important; }}
{N} [data-testid="stSidebar"] .stButton > button {{ background: {C['card2']} !important; color: {C['text']} !important; border-color: {C['border2']} !important; }}
{N} [data-testid="stSidebar"] .stButton > button:hover {{ border-color: {C['primary']} !important; }}

/* ── Streamlit widgets (Streamlit draws them in the app's charcoal / fire-orange base palette) ── */
{N} input, {N} textarea, {N} [data-baseweb="input"], {N} [data-baseweb="base-input"], {N} [data-baseweb="textarea"],
{N} [data-baseweb="select"] > div, {N} [data-testid="stTextInputRootElement"], {N} [data-testid="stTextAreaRootElement"],
{N} [data-testid="stNumberInputContainer"], {N} [data-testid="stDateInputField"], {N} [data-testid="stTimeInputField"],
{N} [data-testid="stSelectbox"] [role="group"], {N} [data-testid="stMultiSelect"] [role="group"] {{
    background: {C['input']} !important; color: {C['text']} !important; border-color: {C['border2']} !important; }}
{N} [data-testid="stTextInputRootElement"]:focus-within, {N} [data-testid="stTextAreaRootElement"]:focus-within,
{N} [data-testid="stNumberInputContainer"]:focus-within, {N} [data-testid="stSelectbox"] [role="group"]:focus-within {{
    border-color: {C['primary']} !important; box-shadow: 0 0 0 1px rgba(8,120,249,0.35); }}
{N} input::placeholder, {N} textarea::placeholder {{ color: {C['muted']} !important; opacity: 1; }}
{N} [data-testid="stSelectbox"] [role="group"] *, {N} [data-testid="stMultiSelect"] [role="group"] *,
{N} [data-baseweb="select"] span, {N} [data-baseweb="select"] svg {{ color: {C['text']} !important; }}
{N} [data-testid="stTextInputRootElement"] button, {N} [data-testid="stTextInputRootElement"] [data-testid="stIconMaterial"] {{
    color: {C['text2']} !important; background: transparent !important; }}
{N} [data-testid="stNumberInput"] button {{ background: {C['card2']} !important; border-color: {C['border2']} !important; }}
{N} [data-testid="stNumberInput"] button svg {{ color: {C['text2']} !important; }}
{N} [data-baseweb="popover"] [role="listbox"], {N} [data-baseweb="popover"] ul, {N} [data-baseweb="menu"] {{
    background: {C['card2']} !important; border: 1px solid {C['border2']} !important; }}
{N} [data-baseweb="popover"] li, {N} [role="option"] {{ color: {C['text']} !important; background: transparent; }}
{N} [data-baseweb="popover"] li:hover, {N} [role="option"]:hover, {N} [role="option"][aria-selected="true"] {{ background: rgba(8,120,249,0.22) !important; }}
{N} [data-baseweb="tag"] {{ background: rgba(8,120,249,0.25) !important; color: {C['text']} !important; }}
{N} [data-baseweb="tooltip"], {N} [data-testid="stTooltipContent"] {{ background: {C['card2']} !important; color: {C['text']} !important;
    border: 1px solid {C['border2']} !important; }}
{N} [data-baseweb="tooltip"] *, {N} [data-testid="stTooltipContent"] * {{ color: {C['text']} !important; }}
{N} [data-testid="stTooltipIcon"] svg {{ color: {C['muted']} !important; }}

{N} .stButton > button, {N} .stDownloadButton > button, {N} [data-testid="stFormSubmitButton"] button,
{N} [data-testid="stBaseButton-secondary"] {{ background: {C['card2']} !important; color: {C['text']} !important;
    border: 1px solid {C['border2']} !important; }}
{N} .stButton > button p, {N} .stDownloadButton > button p, {N} [data-testid="stFormSubmitButton"] button p {{ color: inherit !important; }}
{N} .stButton > button:hover, {N} .stDownloadButton > button:hover, {N} [data-testid="stFormSubmitButton"] button:hover {{
    border-color: {C['primary']} !important; color: #FFFFFF !important; box-shadow: 0 6px 18px rgba(8,120,249,0.22) !important; }}
{N} .stButton > button:disabled, {N} .stDownloadButton > button:disabled {{ opacity: .5; }}
{N} .stButton > button[kind="primary"], {N} [data-testid="stBaseButton-primary"], {N} [data-testid="stBaseButton-primaryFormSubmit"] {{
    background: {C['primary']} !important; color: #FFFFFF !important; border: 1px solid {C['primary']} !important; }}
{N} .stButton > button[kind="primary"] p, {N} [data-testid="stBaseButton-primary"] p {{ color: #FFFFFF !important; }}
{N} .stButton > button[kind="primary"]:hover {{ background: #1A86FF !important; box-shadow: 0 8px 22px rgba(8,120,249,0.38) !important; }}
{N} [data-testid="stPageLink"] a {{ background: {C['card2']} !important; border-color: {C['border2']} !important; }}
{N} [data-testid="stPageLink"] a p {{ color: {C['text']} !important; }}
{N} [data-testid="stPageLink"] a:hover {{ border-color: {C['primary']} !important; }}

{N} [data-testid="stTab"] > div:not([data-testid]), {N} .stTabs [data-baseweb="tab-highlight"] {{ background-color: {C['primary']} !important; }}
{N} [data-testid="stTab"] p, {N} .stTabs [data-baseweb="tab"] p {{ color: {C['muted']} !important; }}
{N} [data-testid="stTab"][aria-selected="true"] p, {N} .stTabs [aria-selected="true"] p {{ color: {C['text']} !important; }}
{N} .stTabs [aria-selected="true"] {{ border-bottom-color: {C['primary']} !important; }}
{N} [data-baseweb="tab-border"] {{ background: {C['border']} !important; }}

{N} [data-testid="stSlider"] [role="group"] > div > div:nth-child(-n+2) {{ filter: hue-rotate(186deg) saturate(1.2); }}
{N} [data-testid="stSliderThumbValue"], {N} [data-testid="stSliderThumbValue"] p {{ color: {C['text']} !important; }}
{N} [data-testid="stSliderTickBarMin"], {N} [data-testid="stSliderTickBarMax"] {{ color: {C['muted']} !important; }}
{N} label:has(input[type="checkbox"]:checked) > div:not([data-testid]) {{ background-color: {C['primary']} !important; border-color: {C['primary']} !important; }}
{N} label:has(input[role="switch"]:not(:checked)) > div:not([data-testid]) {{ background-color: {C['border2']} !important; }}
{N} label:has(input[type="checkbox"]:not([role="switch"]):not(:checked)) > div:not([data-testid]) {{ background-color: {C['input']} !important;
    border-color: {C['border2']} !important; }}
{N} [data-testid="stRadioOption"]:has(input:checked) > div > div > div:first-child {{ background-color: {C['primary']} !important; }}
{N} [data-testid="stRadioOption"]:not(:has(input:checked)) > div > div > div:first-child {{ background-color: {C['border2']} !important; }}
{N} [data-testid="stRadioOption"]:not(:has(input:checked)) > div > div > div:first-child > div {{ background-color: {C['input']} !important; }}
{N} [data-testid="stProgress"] [role="progressbar"] > div > div {{ background-color: {C['primary']} !important; }}

{N} [data-testid="stExpander"] details {{ background: {C['card']} !important; border-color: {C['border']} !important; border-radius: 12px; }}
{N} [data-testid="stExpander"] summary, {N} [data-testid="stExpander"] summary p, {N} [data-testid="stExpander"] summary svg {{ color: {C['text']} !important; }}
{N} [data-testid="stExpander"] summary:hover {{ background: rgba(8,120,249,0.10) !important; }}
{N} [data-testid="stForm"] {{ border-color: {C['border']} !important; background: rgba(8,29,51,0.6); }}
{N} [data-testid="stVerticalBlockBorderWrapper"] {{ border-color: {C['border']} !important; }}
{N} [data-testid="stMetric"] {{ background: {C['card']}; border: 1px solid {C['border']}; border-radius: 12px; padding: 10px 12px; }}
{N} [data-testid="stMetricValue"], {N} [data-testid="stMetricValue"] div {{ color: {C['text']} !important; }}
{N} [data-testid="stMetricLabel"], {N} [data-testid="stMetricLabel"] p {{ color: {C['muted']} !important; }}
/* st.success / st.info / st.warning / st.error: ONE surface per message. Streamlit paints the
   background on stAlertContainer; the content element inside stays transparent (painting it as
   well produced a second, darker strip behind the text). */
{N} [data-testid="stAlertContainer"]:has(> [data-testid="stAlertContentSuccess"]) {{ background: #0B332D !important;
    border: 1px solid #176B52 !important; border-radius: 12px !important; padding: 12px 16px !important; color: #D1FAE5 !important; }}
{N} [data-testid="stAlertContentSuccess"] {{ background: transparent !important; display: flex; align-items: flex-start; gap: 10px; }}
{N} [data-testid="stAlertContentSuccess"]::before {{ content: ""; flex: 0 0 18px; width: 18px; height: 18px; margin-top: 2px; background: #34E0A1;
    -webkit-mask: url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='9'/%3E%3Cpath d='M8 12.5l2.7 2.7L16 9.8'/%3E%3C/svg%3E") center / contain no-repeat; mask: url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='9'/%3E%3Cpath d='M8 12.5l2.7 2.7L16 9.8'/%3E%3C/svg%3E") center / contain no-repeat; }}
{N} [data-testid="stAlertContentSuccess"] p, {N} [data-testid="stAlertContentSuccess"] li, {N} [data-testid="stAlertContentSuccess"] [data-testid="stMarkdownContainer"] {{
    color: #D1FAE5 !important; line-height: 1.5; margin: 0; }}
{N} [data-testid="stAlertContainer"]:has(> [data-testid="stAlertContentInfo"]) {{ background: #0A2A47 !important;
    border: 1px solid #1D5A8C !important; border-radius: 12px !important; padding: 12px 16px !important; color: #D6EEFF !important; }}
{N} [data-testid="stAlertContentInfo"] {{ background: transparent !important; display: flex; align-items: flex-start; gap: 10px; }}
{N} [data-testid="stAlertContentInfo"]::before {{ content: ""; flex: 0 0 18px; width: 18px; height: 18px; margin-top: 2px; background: #38BDF8;
    -webkit-mask: url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='9'/%3E%3Cpath d='M12 11v5M12 8h.01'/%3E%3C/svg%3E") center / contain no-repeat; mask: url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='9'/%3E%3Cpath d='M12 11v5M12 8h.01'/%3E%3C/svg%3E") center / contain no-repeat; }}
{N} [data-testid="stAlertContentInfo"] p, {N} [data-testid="stAlertContentInfo"] li, {N} [data-testid="stAlertContentInfo"] [data-testid="stMarkdownContainer"] {{
    color: #D6EEFF !important; line-height: 1.5; margin: 0; }}
{N} [data-testid="stAlertContainer"]:has(> [data-testid="stAlertContentWarning"]) {{ background: #33270B !important;
    border: 1px solid #7A5A16 !important; border-radius: 12px !important; padding: 12px 16px !important; color: #FFE7B8 !important; }}
{N} [data-testid="stAlertContentWarning"] {{ background: transparent !important; display: flex; align-items: flex-start; gap: 10px; }}
{N} [data-testid="stAlertContentWarning"]::before {{ content: ""; flex: 0 0 18px; width: 18px; height: 18px; margin-top: 2px; background: #FFB547;
    -webkit-mask: url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 4l9 16H3z'/%3E%3Cpath d='M12 10v4M12 17h.01'/%3E%3C/svg%3E") center / contain no-repeat; mask: url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 4l9 16H3z'/%3E%3Cpath d='M12 10v4M12 17h.01'/%3E%3C/svg%3E") center / contain no-repeat; }}
{N} [data-testid="stAlertContentWarning"] p, {N} [data-testid="stAlertContentWarning"] li, {N} [data-testid="stAlertContentWarning"] [data-testid="stMarkdownContainer"] {{
    color: #FFE7B8 !important; line-height: 1.5; margin: 0; }}
{N} [data-testid="stAlertContainer"]:has(> [data-testid="stAlertContentError"]) {{ background: #3A1219 !important;
    border: 1px solid #7F2532 !important; border-radius: 12px !important; padding: 12px 16px !important; color: #FFD9DD !important; }}
{N} [data-testid="stAlertContentError"] {{ background: transparent !important; display: flex; align-items: flex-start; gap: 10px; }}
{N} [data-testid="stAlertContentError"]::before {{ content: ""; flex: 0 0 18px; width: 18px; height: 18px; margin-top: 2px; background: #FF6B76;
    -webkit-mask: url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='9'/%3E%3Cpath d='M9 9l6 6M15 9l-6 6'/%3E%3C/svg%3E") center / contain no-repeat; mask: url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='9'/%3E%3Cpath d='M9 9l6 6M15 9l-6 6'/%3E%3C/svg%3E") center / contain no-repeat; }}
{N} [data-testid="stAlertContentError"] p, {N} [data-testid="stAlertContentError"] li, {N} [data-testid="stAlertContentError"] [data-testid="stMarkdownContainer"] {{
    color: #FFD9DD !important; line-height: 1.5; margin: 0; }}
{N} [data-testid="stToast"] {{ background: {C['card2']} !important; border: 1px solid {C['border2']} !important; }}
{N} [data-testid="stToast"] * {{ color: {C['text']} !important; }}
{N} [data-testid="stCodeBlock"] pre, {N} pre {{ background: {C['card']} !important; }}
{N} [data-testid="stDialog"] [role="dialog"] {{ background: {C['card']} !important; border: 1px solid {C['border2']} !important; }}

/* tables: Streamlit draws its data grid on a canvas in the app's own near-black base palette
   (white text, unchanged cell highlights), which reads cleanly on navy; the frame follows the theme */
{N} [data-testid="stDataFrame"], {N} [data-testid="stTable"] {{ border: 1px solid {C['border']} !important; border-radius: 10px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.25); }}
{N} [data-testid="stTable"] th {{ background: {C['card2']} !important; color: {C['text2']} !important; }}
{N} [data-testid="stTable"] td {{ color: {C['text']} !important; border-color: {C['border']} !important; }}

/* Plotly charts (transparent backgrounds): text, axes, grid, legend and hover follow the theme;
   data colours are untouched so severities stay distinguishable */
{N} [data-testid="stPlotlyChart"], {N} [data-testid="stVegaLiteChart"] {{ background: {C['card']}; border: 1px solid {C['border']};
    border-radius: 12px; }}
{N} .js-plotly-plot .main-svg .xtick text, {N} .js-plotly-plot .main-svg .ytick text, {N} .js-plotly-plot .main-svg .legendtext,
{N} .js-plotly-plot .main-svg .xtitle, {N} .js-plotly-plot .main-svg .ytitle, {N} .js-plotly-plot .main-svg .gtitle,
{N} .js-plotly-plot .main-svg .angularaxistick text, {N} .js-plotly-plot .main-svg .radialaxistick text,
{N} .js-plotly-plot .main-svg .cbtitle text, {N} .js-plotly-plot .main-svg .cbaxis text {{ fill: {C['text2']} !important; }}
{N} .js-plotly-plot .main-svg .annotation-text, {N} .js-plotly-plot .main-svg .number {{ fill: {C['text']} !important; }}
{N} .js-plotly-plot .gridlayer path, {N} .js-plotly-plot .polar .angularaxis path, {N} .js-plotly-plot .polar .radialaxis path,
{N} .js-plotly-plot .polar .gridlayer path {{ stroke: rgba(31,76,120,0.70) !important; }}
{N} .js-plotly-plot .zerolinelayer path {{ stroke: rgba(130,152,179,0.55) !important; }}
{N} .js-plotly-plot .legend .bg {{ fill: rgba(8,29,51,0.85) !important; stroke: {C['border']} !important; }}
{N} .js-plotly-plot .hoverlayer .hovertext path {{ fill: {C['card2']} !important; stroke: {C['border2']} !important; }}
{N} .js-plotly-plot .hoverlayer .hovertext text {{ fill: {C['text']} !important; }}
{N} .js-plotly-plot .modebar-btn path {{ fill: {C['muted']} !important; }}
{N} .js-plotly-plot .modebar {{ background: transparent !important; }}
{N} .js-plotly-plot .indicatorlayer .angular > g.bg-arc:nth-of-type(1) path {{ fill: {C['card2']} !important; }}
{N} .js-plotly-plot .indicatorlayer .angular > g.bg-arc:nth-of-type(2) path {{ fill: #0E2C4A !important; }}
{N} .js-plotly-plot .indicatorlayer .angular > g.bg-arc:nth-of-type(3) path {{ fill: #3A2C0A !important; }}
{N} .js-plotly-plot .indicatorlayer .angular > g.bg-arc:nth-of-type(4) path {{ fill: #401722 !important; }}
{N} .js-plotly-plot .indicatorlayer path {{ stroke: {C['border']} !important; }}
{N} .js-plotly-plot .indicatorlayer .threshold-arc path {{ stroke: {C['crit']} !important; }}
{N} .js-plotly-plot .indicatorlayer .xangularaxistick path {{ stroke: {C['muted']} !important; }}
{N} .js-plotly-plot .updatemenu-button rect {{ fill: {C['card2']} !important; stroke: {C['border2']} !important; }}
{N} .js-plotly-plot .updatemenu-button text {{ fill: {C['text']} !important; }}
{N} .js-plotly-plot .slider-rail-rect {{ fill: {C['card2']} !important; stroke: {C['border2']} !important; }}
{N} .js-plotly-plot .slider-grip-rect {{ fill: {C['primary']} !important; stroke: {C['text']} !important; }}
{N} .js-plotly-plot .slider-label, {N} .js-plotly-plot .slider-container text {{ fill: {C['text2']} !important; }}
{N} [data-testid="stVegaLiteChart"] text {{ fill: {C['text2']} !important; }}

/* ── shared components (dashboard_common / theme.py) ── */
{N} .hero {{ background: linear-gradient(135deg, {C['card2']} 0%, {C['card']} 100%) !important; border-color: {C['border']} !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.30); }}
{N} .hero::before {{ background: linear-gradient(180deg, {C['primary']}, {C['cyan']}) !important; }}
{N} .hero::after {{ background: radial-gradient(circle, rgba(8,120,249,.20), transparent 65%) !important; }}
{N} .hero .eyebrow {{ color: {C['cyan']} !important; }}
{N} .hero .sub {{ color: {C['text2']} !important; }}
{N} .kpi, {N} .alert-card, {N} .info-box, {N} .briefing-card, {N} .nav-card, {N} .lv-card, {N} .lm-card, {N} .sys-item,
{N} .cc-pill, {N} [class*="st-key-cc_explore_"] {{ background: {C['card']} !important; border-color: {C['border']} !important; }}
{N} .kpi:hover, {N} .nav-card:hover, {N} [class*="st-key-cc_explore_"]:hover {{ border-color: {C['primary']} !important;
    box-shadow: inset 0 2px 0 rgba(0,184,240,0.55), 0 10px 26px rgba(0,0,0,0.35) !important; }}
{N} .kpi .val {{ color: {C['text']} !important; }}
{N} .kpi .lbl, {N} .kpi .ico {{ color: {C['muted']} !important; }} {N} .kpi .ctx {{ color: {C['muted']} !important; }}
{N} .kpi.k-neutral, {N} .kpi:not([class*="k-"]) {{ --k-accent: {C['border2']}; }}
{N} .info-box {{ color: {C['text2']} !important; border-left-color: {C['cyan']} !important; }}
{N} .briefing-card {{ color: {C['text2']} !important; }}
{N} .sec-hdr {{ color: {C['text2']} !important; border-bottom-color: {C['border']} !important; }}
{N} .sec-hdr::before {{ background: {C['primary']} !important; }}
{N} .nav-card .idx {{ color: {C['cyan']} !important; }}
{N} .legend {{ color: {C['text2']} !important; }}

/* global ticker */
{N} .gt {{ background: linear-gradient(100deg, {C['header']} 0%, #061C34 100%) padding-box,
    linear-gradient(90deg, rgba(8,120,249,0.80), rgba(0,184,240,0.70)) border-box !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.40), 0 0 20px rgba(8,120,249,0.10) !important; }}
{N} .gt::before {{ background: radial-gradient(420px 60px at 8% 0%, rgba(8,120,249,0.16), transparent 70%) !important; }}
{N} .gt-tag {{ background: rgba(3,16,31,0.80) !important; border-right-color: {C['border2']} !important; }}
{N} .gt-item {{ color: {C['text2']} !important; }} {N} .gt-item b {{ color: #6FC3F5 !important; }} {N} .gt-item .v {{ color: {C['text']} !important; }}
{N} .gt-sep {{ color: rgba(0,184,240,0.55) !important; }} {N} .gt-dot.data {{ background: {C['cyan']} !important; }}

/* Command Center */
{N} .cc-hero-note {{ color: {C['muted']} !important; }}
{N} .cc-pill {{ color: {C['text2']} !important; }} {N} .cc-pill b {{ color: {C['muted']} !important; }}
{N} .fa {{ background: linear-gradient(120deg, rgba(64,14,24,0.90), rgba(8,29,51,0.95) 60%) !important; border-color: rgba(255,77,90,0.45) !important; }}
{N} .fa.calm {{ background: linear-gradient(120deg, rgba(6,48,40,0.85), rgba(8,29,51,0.95) 60%) !important; border-color: rgba(32,217,149,0.38) !important; }}
{N} .fa.warn {{ background: linear-gradient(120deg, rgba(58,40,8,0.88), rgba(8,29,51,0.95) 60%) !important; border-color: rgba(255,181,71,0.42) !important; }}
{N} .fa .thumb::after {{ background: linear-gradient(90deg, transparent 55%, rgba(8,29,51,0.96)) !important; }}
{N} .fa .meta, {N} .fa .stat .l {{ color: {C['muted']} !important; }}
{N} .fa .stat {{ background: rgba(3,16,31,0.55) !important; border-color: {C['border']} !important; }}
{N} .fa .chip {{ border-color: {C['border2']} !important; background: rgba(8,120,249,0.08) !important; }}
{N} .ex-card .ex-ico {{ background: rgba(8,120,249,0.14) !important; border-color: rgba(8,120,249,0.38) !important; color: #6FC3F5 !important; }}
{N} .ex-card .idx {{ color: {C['muted']} !important; }} {N} .ex-card p {{ color: {C['text2']} !important; }}
{N} [class*="st-key-cc_explore_"] [data-testid="stPageLink"] a {{ background: transparent !important; border: none !important; }}
{N} [class*="st-key-cc_explore_"] [data-testid="stPageLink"] a p {{ color: #4FB4FF !important; }}
{N} [class*="st-key-cc_explore_"] [data-testid="stPageLink"] a:hover p {{ color: #FFFFFF !important; }}
{N} .st-key-cc_regionmap iframe {{ box-shadow: 0 0 0 1px {C['border2']}, 0 14px 34px rgba(0,0,0,0.45) !important; }}
{N} .sys-item .v {{ color: {C['text']} !important; }} {N} .sys-item .l, {N} .lv-card .l {{ color: {C['muted']} !important; }}

/* real observations / What-If live modes */
{N} .lv-card .g2 span:nth-child(odd) {{ color: {C['muted']} !important; }} {N} .lv-card .g2 span:nth-child(even) {{ color: {C['text']} !important; }}
{N} .lv-card .ft, {N} .lm-ref, {N} .lm-note, {N} .lm-desc {{ color: {C['muted']} !important; }}
{N} .lm-chip {{ border-color: {C['border']} !important; color: {C['muted']} !important; }}
{N} .lm-chip.on, {N} .lm-card.on {{ color: {C['text']} !important; border-color: {C['cyan']} !important; background: rgba(0,184,240,0.10) !important;
    box-shadow: 0 0 0 1px rgba(0,184,240,0.20) !important; }}
{N} .lm-cmp th {{ border-bottom-color: {C['border']} !important; }} {N} .lm-cmp td {{ border-bottom-color: rgba(22,58,94,0.6) !important; }}

/* Export Simulation Report panel */
{N} .st-key-export_panel {{ background: linear-gradient(160deg, rgba(11,37,66,0.98), rgba(8,29,51,0.98)) padding-box,
    linear-gradient(135deg, {C['primary']}, {C['cyan']}) border-box !important; box-shadow: 0 18px 48px rgba(0,0,0,0.55) !important; }}
{N} .st-key-export_panel .xp-ico {{ background: rgba(8,120,249,0.18) !important; border-color: rgba(0,184,240,0.40) !important; }}
{N} .st-key-export_panel .xp-sub, {N} .st-key-export_panel .xp-sum {{ color: {C['text2']} !important; }}
{N} .st-key-export_panel .xp-chip, {N} .st-key-export_panel .xp-chip.pink {{ color: #CFEAFF !important; background: rgba(8,120,249,0.16) !important;
    border-color: rgba(8,120,249,0.45) !important; }}
{N} .st-key-export_panel [data-testid="stExpander"] summary:hover p {{ color: {C['cyan']} !important; }}
{N} .st-key-export_panel [data-testid="stDownloadButton"] button {{ background: {C['primary']} !important; border-color: {C['primary']} !important;
    color: #FFFFFF !important; }}
{N} .st-key-export_panel [data-testid="stExpander"] details {{ border-color: {C['border2']} !important; }}

/* sign-in / create account */
{N} .stApp:has(.st-key-login_card) {{ background: linear-gradient(180deg, rgba(6,20,38,0.82), rgba(3,16,31,0.94)),
    var(--ff-login-bg, none) center / cover no-repeat fixed, {C['bg']} !important; }}
{N} .st-key-login_card {{ background: rgba(8,29,51,0.80) !important; border-color: {C['border2']} !important;
    box-shadow: 0 30px 80px rgba(0,0,0,0.55), inset 0 1px 0 rgba(255,255,255,0.05) !important; }}
{N} .lg-mark {{ background: linear-gradient(135deg, rgba(8,120,249,0.26), rgba(0,184,240,0.10)) !important;
    border-color: rgba(0,184,240,0.45) !important; color: #6FC3F5 !important; }}
{N} .lg-inst {{ color: {C['cyan']} !important; }} {N} .lg-sub, {N} .lg-hint, {N} .lg-foot {{ color: {C['muted']} !important; }}
{N} .lg-note {{ background: rgba(8,120,249,0.06) !important; border-color: {C['border']} !important; color: {C['text2']} !important; }}
{N} .st-key-login_card input {{ background: {C['input']} !important; }}
{N} .st-key-login_card [data-testid="stFormSubmitButton"] button {{ background: {C['primary']} !important; color: #FFFFFF !important; }}
{N} .st-key-login_card [data-testid="stFormSubmitButton"] button:hover {{ box-shadow: 0 10px 26px rgba(8,120,249,0.40) !important; }}

@media (max-width: 900px) {{ {N} .fa .thumb::after {{ background: linear-gradient(180deg, transparent 50%, rgba(8,29,51,0.96)) !important; }} }}
"""

# Inline-styled pieces. Streamlit renders markdown HTML with React, which rewrites
# style="background:#11151b" as "background: rgb(17, 21, 27)"; match that exact form.
INLINE_CSS = (
    "/* wind compass card (inline-styled div + SVG) */\n"
    f'{N} [style*="background: rgb(17, 21, 27)"] {{ background: {C["card"]} !important; border-color: {C["border"]} !important; }}\n'
    f'{N} [style*="background: rgb(17, 21, 27)"] svg circle {{ fill: {C["sidebar"]}; stroke: {C["border2"]}; }}\n'
    f'{N} [style*="background: rgb(17, 21, 27)"] svg text {{ fill: {C["muted"]}; }}\n'
)

# For small components.html iframes: mirror the dashboard's theme as class "ff-navy" on the
# iframe's own <html>, so its CSS can restyle itself (same-origin; harmless if not).
IFRAME_THEME_SYNC = (
    "<script>(function(){function s(){var t='dark';try{t=window.parent.document.documentElement"
    ".getAttribute('data-ff-theme')||'dark';}catch(e){}document.documentElement.classList.toggle('ff-navy',t==='navy');}"
    "s();try{new MutationObserver(s).observe(window.parent.document.documentElement,{attributes:true,"
    "attributeFilter:['data-ff-theme']});}catch(e){}})();</script>")

# starts on its own line: it is appended to the shared <style> markdown in set_page and must open
# its own HTML block (CommonMark ends a <style> block at the line holding </style>)
THEME_SWITCH_CSS = "\n<style>" + MENU_CSS + NAVY_CSS + INLINE_CSS + "</style>\n"


def render_theme_switch():
    """Draw the (invisible) application-menu controller; THEME_SWITCH_CSS must be on the page
    (set_page adds it to the shared style block). Same key and no arguments on every run, so
    Streamlit keeps the same iframe across reruns.

    Theme changes never reach Python (CSS only). The menu's Rerun and Clear cache items send
    {action, nonce}: the value itself triggers a normal script rerun (= Rerun); Clear cache also
    clears st.cache_data and st.cache_resource, as Streamlit's own menu item does. Each nonce is
    handled once, so later reruns that return the same value do nothing."""
    with st.container(key="ff_theme_host"):
        value = _theme_switch(key="ff_theme_switch", default=None)
    handle_menu_action(value)


def handle_menu_action(value) -> str:
    """Act on one menu request; returns the action performed ('' if none / already handled)."""
    if not isinstance(value, dict) or not value.get("nonce"):
        return ""
    if st.session_state.get("_ff_menu_nonce") == value["nonce"]:
        return ""
    st.session_state["_ff_menu_nonce"] = value["nonce"]
    action = value.get("action")
    if action == "clear_cache":
        st.cache_data.clear()
        st.cache_resource.clear()
        st.toast("Cache cleared")
        return "clear_cache"
    if action == "rerun":
        return "rerun"                  # the component value already started this rerun
    return ""
