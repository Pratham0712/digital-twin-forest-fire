"""
dashboard_common.py - shared logic, theme, and render functions used by every
page in the multi-page dashboard. Every page imports from here so the theme,
model loading, and twin session-state handling stay perfectly consistent
across pages, and so functionality already built and tested (risk map, CA
animation, model performance charts) is reused rather than duplicated.
"""
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.config import REGION, SYSTEM, MODELS_DIR, DATA_PROCESSED_DIR, DATA_RAW_DIR, RegionConfig
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


def set_page(title: str, icon: str = "🔥"):
    """Every page must call this first. Wrapped in try/except because
    Streamlit raises if set_page_config is called more than once per script
    run, and some Streamlit versions are stricter about call order than
    others across page navigations."""
    try:
        st.set_page_config(page_title=f"{title} — Forest Fire Digital Twin",
                            page_icon=icon, layout="wide", initial_sidebar_state="expanded")
    except Exception:
        pass
    st.markdown(CSS, unsafe_allow_html=True)


# ── theme ─────────────────────────────────────────────────────────────────── #

CSS = """
<style>
.stApp { background: #0b0d0f; }
[data-testid="stSidebar"] { background: #0f1114; border-right: 1px solid #1e2329; }
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
h1,h2,h3 { color: #e6edf3 !important; font-weight: 600 !important; letter-spacing: -0.3px; }
p, span, label, div { color: #8b949e; }
.stTabs [data-baseweb="tab"] { color: #8b949e; font-size: 13px; }
.stTabs [aria-selected="true"] { color: #e6edf3 !important; border-bottom: 2px solid #f78166; }

.hero {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d; border-radius: 12px;
    padding: 20px 26px; margin-bottom: 20px;
    animation: fadeSlideIn 0.5s ease-out;
}
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(-8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.hero h1 { margin: 0 0 4px 0; font-size: 22px; color: #e6edf3 !important; }
.hero .flame { display: inline-block; animation: flicker 1.8s ease-in-out infinite; }
@keyframes flicker {
    0%, 100% { transform: scale(1) rotate(0deg); filter: drop-shadow(0 0 4px rgba(255,138,0,0.5)); }
    25%      { transform: scale(1.06) rotate(-2deg); filter: drop-shadow(0 0 9px rgba(255,138,0,0.8)); }
    50%      { transform: scale(0.97) rotate(1deg); filter: drop-shadow(0 0 5px rgba(255,138,0,0.5)); }
    75%      { transform: scale(1.04) rotate(2deg); filter: drop-shadow(0 0 8px rgba(255,138,0,0.7)); }
}
.hero .sub { color: #8b949e; font-size: 12px; }
.hero .badge {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600; margin-left: 8px; vertical-align: middle;
}
.badge-live { background: #1a4731; color: #3fb950; border: 1px solid #238636; }
.badge-demo { background: #2d1f00; color: #d29922; border: 1px solid #9e6a03; }

.kpi-row { display: flex; gap: 12px; margin-bottom: 18px; flex-wrap: wrap; }
.kpi {
    flex: 1; min-width: 110px;
    background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 14px 16px;
    transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
    animation: fadeSlideIn 0.5s ease-out;
}
.kpi:hover {
    transform: translateY(-3px);
    border-color: #484f58;
    box-shadow: 0 6px 18px rgba(0,0,0,0.35);
}
.kpi .lbl { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: #8b949e; }
.kpi .val { font-size: 28px; font-weight: 700; margin-top: 2px; color: #e6edf3; }
.kpi .val.ok   { color: #3fb950; }
.kpi .val.warn { color: #d29922; }
.kpi .val.crit { color: #f78166; text-shadow: 0 0 12px rgba(247,129,102,0.4); }

.alert-card {
    border-left: 3px solid; border-radius: 8px;
    padding: 9px 13px; margin-bottom: 7px;
    background: #161b22;
    display: flex; justify-content: space-between; align-items: flex-start;
}
.alert-card .left .zone { font-weight: 600; font-size: 13px; color: #e6edf3; }
.alert-card .left .reason { font-size: 11px; color: #8b949e; margin-top: 2px; }
.alert-card .score { font-size: 13px; font-weight: 700; white-space: nowrap; margin-left: 10px; }

.sec-hdr {
    font-size: 11px; text-transform: uppercase; letter-spacing: 1.2px;
    color: #8b949e; border-bottom: 1px solid #21262d;
    padding-bottom: 6px; margin: 18px 0 12px 0;
}

.legend { display: flex; gap: 16px; font-size: 11px; color: #8b949e; flex-wrap: wrap; margin-top: 8px; }
.legend .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }

.info-box {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 12px 16px; font-size: 12px; color: #8b949e;
}

.briefing-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 18px 22px; margin-bottom: 14px; line-height: 1.6; font-size: 13px;
    color: #c9d1d9;
}
.briefing-card h4 { color: #e6edf3 !important; margin-top: 0; }

@keyframes pulseGlow {
    0%   { box-shadow: 0 0 0px 0px rgba(247,129,102,0.0); }
    50%  { box-shadow: 0 0 22px 4px var(--pulse-color, rgba(247,129,102,0.45)); }
    100% { box-shadow: 0 0 0px 0px rgba(247,129,102,0.0); }
}
.scenario-banner {
    border-radius: 10px; padding: 14px 20px; margin-bottom: 14px;
    border: 1px solid; font-size: 13px; display: flex; align-items: center; gap: 12px;
    animation: pulseGlow 2.2s ease-in-out infinite;
}
.scenario-banner .icon { font-size: 22px; line-height: 1; }
.scenario-banner .txt b { font-size: 14px; }

@keyframes markerPulse {
    0%   { opacity: 0.35; }
    50%  { opacity: 0.9; }
    100% { opacity: 0.35; }
}
.nav-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 16px 18px; height: 100%;
    transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}
.nav-card:hover {
    transform: translateY(-4px);
    border-color: #f0883e;
    box-shadow: 0 8px 20px rgba(240,136,62,0.15);
}
.nav-card h4 { color: #e6edf3 !important; margin: 0 0 6px 0; font-size: 15px; }
.nav-card p { font-size: 12px; color: #8b949e; margin: 0; }
</style>
"""

SEV_COLOR = {"EXTREME": "#f78166", "HIGH": "#d29922", "MODERATE": "#58a6ff", "LOW": "#3fb950"}

RISK_CS = [
    [0.00, "#1c2128"], [0.20, "#1f3a5f"],
    [0.40, "#9e6a03"], [0.65, "#d29922"],
    [0.80, "#f0883e"], [1.00, "#f78166"],
]

CA_CS = [
    [0.00, "#1a3d1f"], [0.24, "#1a3d1f"],
    [0.25, "#ff8a00"], [0.49, "#ff2d00"],
    [0.50, "#21262d"], [0.74, "#21262d"],
    [0.75, "#1f3a5f"], [1.00, "#1f3a5f"],
]


# ── model / twin ──────────────────────────────────────────────────────────── #

@st.cache_resource(show_spinner=False)
def load_ml_model():
    path = MODELS_DIR / "xgboost_real.json"
    if not path.exists():
        return None
    from src.ml_models.model_trainer import XGBoostModel
    w = XGBoostModel()
    w.model.load_model(str(path))
    return w


def get_twin(offline: bool, scenario: dict = None, region=None) -> DigitalTwin:
    return DigitalTwin(ml_model=load_ml_model(), offline=offline,
                        scenario=scenario, region=region)


REGION_PRESETS = {
    "Karnataka / Western Ghats (validated)": REGION,
    "California": RegionConfig(name="California", min_lat=32.5, max_lat=42.0,
                                min_lon=-124.5, max_lon=-114.0,
                                grid_resolution_deg=0.5, weather_grid_resolution_deg=1.5),
    "Australia (SE)": RegionConfig(name="Australia (SE)", min_lat=-39.0, max_lat=-28.0,
                                    min_lon=140.0, max_lon=154.0,
                                    grid_resolution_deg=0.5, weather_grid_resolution_deg=1.5),
    "Mediterranean": RegionConfig(name="Mediterranean", min_lat=36.0, max_lat=44.0,
                                   min_lon=-5.0, max_lon=20.0,
                                   grid_resolution_deg=0.5, weather_grid_resolution_deg=1.5),
}


def build_sidebar(show_scenario: bool = False):
    """Shared sidebar: offline toggle + region picker + refresh, present on
    every page for consistency. show_scenario=True additionally renders the
    weather-scenario sliders inline (used by the What-If page; other pages
    leave this off and instead point the user to the dedicated page)."""
    st.sidebar.markdown("### Controls")
    offline = st.sidebar.toggle(
        "Offline / demo mode",
        value=not bool(os.getenv("FIRMS_MAP_KEY")),
        help="Uses synthetic data. Disable once API keys are set in .env or Streamlit Secrets.",
    )

    scenario = st.session_state.get("scenario") if not show_scenario else None
    if show_scenario and offline:
        st.sidebar.markdown("**Scenario** (see What-If Simulator page for the full view)")
        n_hotspots = st.sidebar.slider("Active fire hotspots", 0, 40, 8)
        temp_c = st.sidebar.slider("Temperature (°C)", 15, 48, 32)
        wind_speed_ms = st.sidebar.slider("Wind speed (m/s)", 0.0, 20.0, 5.0)
        humidity_pct = st.sidebar.slider("Humidity (%)", 0, 100, 40)
        scenario = {"n_hotspots": n_hotspots, "temp_c": temp_c,
                    "wind_speed_ms": wind_speed_ms, "humidity_pct": humidity_pct}
        st.session_state["scenario"] = scenario

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Region**")
    choice = st.sidebar.selectbox("Preset", list(REGION_PRESETS.keys()) + ["Custom"],
                                   key="region_preset_choice")
    if choice == "Custom":
        c1, c2 = st.sidebar.columns(2)
        min_lat = c1.number_input("Min lat", value=11.5, format="%.2f")
        max_lat = c2.number_input("Max lat", value=15.5, format="%.2f")
        min_lon = c1.number_input("Min lon", value=74.0, format="%.2f")
        max_lon = c2.number_input("Max lon", value=77.5, format="%.2f")
        name = st.sidebar.text_input("Name", value="Custom region")
        region = RegionConfig(name=name, min_lat=min_lat, max_lat=max_lat,
                               min_lon=min_lon, max_lon=max_lon,
                               grid_resolution_deg=0.5, weather_grid_resolution_deg=1.5)
    else:
        region = REGION_PRESETS[choice]

    if region.name != REGION.name:
        st.sidebar.warning(
            "Model trained on Karnataka data only. "
            "Other regions are architecture demos, not validated predictions."
        )

    st.sidebar.markdown("---")
    refresh = st.sidebar.button("🔄  Refresh data", use_container_width=True)
    st.sidebar.caption(f"Alert threshold: {SYSTEM.alert_threshold_pct:.0f}%  ·  "
                        f"Grid: {REGION.grid_resolution_deg}° (~11 km cells)")

    return offline, scenario, region, refresh


def ensure_twin(offline: bool, scenario: dict, region, force_refresh: bool) -> DigitalTwin:
    """Central session-state gatekeeper: every page calls this instead of
    building its own twin, so switching pages never silently loses or
    duplicates the live state."""
    need_refresh = (
        "twin" not in st.session_state
        or force_refresh
        or st.session_state.get("_last_region") != region.name
        or st.session_state.get("_last_offline") != offline
        or st.session_state.get("_last_scenario") != scenario
    )
    if need_refresh:
        with st.spinner("Refreshing digital twin state..."):
            twin = get_twin(offline, scenario, region)
            twin.refresh()
        st.session_state["twin"] = twin
        st.session_state["ca_history"] = None
        st.session_state["_last_region"] = region.name
        st.session_state["_last_offline"] = offline
        st.session_state["_last_scenario"] = scenario
    return st.session_state["twin"]


# ── render: header ────────────────────────────────────────────────────────── #

def render_header(summary: dict, offline: bool, subtitle: str = None):
    badge_cls = "badge-demo" if offline else "badge-live"
    badge_txt = "DEMO" if offline else "LIVE"
    ts = summary.get("timestamp", "")[:19].replace("T", " ")
    title = subtitle or "Forest Fire Digital Twin"
    st.markdown(f"""
    <div class="hero">
      <h1><span class="flame">🔥</span> {title}
        <span class="badge {badge_cls}">{badge_txt}</span>
      </h1>
      <div class="sub">{REGION.name} &nbsp;·&nbsp; Last refreshed {ts} UTC
        &nbsp;·&nbsp; BMS College of Engineering · ISE Batch 42</div>
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
        features.append({
            "type": "Feature", "id": zid,
            "geometry": {"type": "Polygon", "coordinates": [[
                [lon - half_deg, lat - half_deg], [lon + half_deg, lat - half_deg],
                [lon + half_deg, lat + half_deg], [lon - half_deg, lat + half_deg],
                [lon - half_deg, lat - half_deg],
            ]]},
        })
    return {"type": "FeatureCollection", "features": features}


def render_risk_map(processed: pd.DataFrame, risk_scores: np.ndarray, region=REGION, height=520,
                     scenario_active: bool = False):
    st.markdown('<div class="sec-hdr">Regional risk map</div>', unsafe_allow_html=True)
    df = processed[["zone_id", "latitude", "longitude", "fwi", "active_fire_nearby"]].copy()
    df["risk_score"] = risk_scores
    df["risk_pct"] = (df["risk_score"] * 100).round(1)

    geojson = _build_grid_geojson(
        tuple(df["zone_id"]), tuple(df["latitude"]), tuple(df["longitude"]),
        half_deg=region.grid_resolution_deg / 2,
    )

    fig = go.Figure(go.Choroplethmap(
        geojson=geojson, locations=df["zone_id"], z=df["risk_score"],
        colorscale=RISK_CS, zmin=0, zmax=1, marker_opacity=0.75,
        marker_line_width=0.4, marker_line_color="#0b0d0f",
        colorbar=dict(title="Risk", tickformat=".0%", tickfont={"color": "#8b949e", "size": 10},
                      title_font={"color": "#8b949e"}, len=0.65, thickness=12, bgcolor="rgba(0,0,0,0)"),
        customdata=np.stack([df["zone_id"], df["fwi"], df["risk_pct"]], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>FWI: %{customdata[1]:.1f}<br>"
                      "Risk: %{customdata[2]}%<extra></extra>",
    ))

    # Glowing fire markers on zones with an active seeded hotspot - this is
    # the direct, ON-THE-MAP visual response to the What-If simulator's
    # hotspot-count slider, not just a colour shift (per review feedback:
    # "changing values only changes the map colours, show something on the
    # map itself").
    active = df[df["active_fire_nearby"]]
    if not active.empty:
        fig.add_trace(go.Scattermap(
            lat=active["latitude"], lon=active["longitude"], mode="markers",
            marker=dict(size=22, color="rgba(255,138,0,0.25)"), hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scattermap(
            lat=active["latitude"], lon=active["longitude"], mode="markers+text",
            marker=dict(size=11, color="#ff8a00"), text=["🔥"] * len(active),
            textfont=dict(size=14), hoverinfo="skip", showlegend=False,
        ))

    fig.update_layout(
        map=dict(style="carto-darkmatter", zoom=6.5,
                  center={"lat": (region.min_lat + region.max_lat) / 2,
                          "lon": (region.min_lon + region.max_lon) / 2}),
        margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", height=height,
    )
    st.plotly_chart(fig, use_container_width=True)

    if scenario_active and not active.empty:
        st.markdown(f"""
        <div class="scenario-banner" style="--pulse-color: rgba(255,138,0,0.45);
             background:#2d1f00; border-color:#9e6a03;">
          <span class="icon">🔥</span>
          <span class="txt"><b>{len(active)} active hotspot zone{'s' if len(active)!=1 else ''}</b>
          seeded by this scenario — shown as glowing markers on the map above.</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="legend">
      <span><span class="dot" style="background:#1f3a5f"></span>Low risk</span>
      <span><span class="dot" style="background:#9e6a03"></span>Moderate</span>
      <span><span class="dot" style="background:#d29922"></span>Elevated</span>
      <span><span class="dot" style="background:#f0883e"></span>High</span>
      <span><span class="dot" style="background:#f78166"></span>Extreme</span>
      <span>🔥 = active seeded hotspot zone</span>
    </div>
    """, unsafe_allow_html=True)


def render_risk_gauge(summary: dict):
    mean_r = summary.get("mean_risk_score", 0) * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=mean_r,
        number={"suffix": "%", "font": {"color": "#e6edf3", "size": 30}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8b949e", "tickfont": {"color": "#8b949e", "size": 9}},
            "bar": {"color": "#f0883e", "thickness": 0.22}, "bgcolor": "#161b22", "borderwidth": 0,
            "steps": [{"range": [0, 40], "color": "#1c2128"},
                      {"range": [40, 70], "color": "#2d2200"},
                      {"range": [70, 100], "color": "#3d1a1a"}],
            "threshold": {"line": {"color": "#f78166", "width": 2}, "thickness": 0.8,
                          "value": SYSTEM.alert_threshold_pct},
        },
    ))
    fig.update_layout(height=200, margin=dict(l=16, r=16, t=10, b=0),
                       paper_bgcolor="rgba(0,0,0,0)", font={"color": "#8b949e"})
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
                                textfont={"color": "#8b949e", "size": 11}))
        fig.update_layout(height=160, margin=dict(l=0, r=0, t=10, b=0),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           xaxis=dict(tickfont={"color": "#8b949e", "size": 11}, gridcolor="#21262d"),
                           yaxis=dict(visible=False), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    if not alerts:
        st.markdown('<div class="info-box">No zones above the alert threshold.</div>', unsafe_allow_html=True)
        return

    for a in alerts[:limit]:
        c = SEV_COLOR.get(a.severity, "#8b949e")
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

def render_ca_simulation(twin: DigitalTwin):
    st.markdown('<div class="sec-hdr">Fire spread simulation — 2-hour projection</div>', unsafe_allow_html=True)
    st.caption("Seeded from HIGH/EXTREME alert zones · 8-neighbour cellular automata · wind-aligned spread")

    if st.button("▶  Run spread simulation", type="primary"):
        with st.spinner("Simulating fire spread..."):
            history = twin.simulate_spread_from_alerts()
        st.session_state["ca_history"] = history

    history = st.session_state.get("ca_history")
    if not history:
        st.markdown('<div class="info-box">Click <b>Run spread simulation</b> to project spread from current alert zones.</div>',
                    unsafe_allow_html=True)
        return
    if len(history) <= 1:
        st.markdown('<div class="info-box">No HIGH/EXTREME zones this cycle — nothing to simulate.</div>',
                    unsafe_allow_html=True)
        return

    frames = [go.Frame(data=[go.Heatmap(z=h.state, colorscale=CA_CS, zmin=0, zmax=3, showscale=False)],
                        name=str(h.minutes_elapsed)) for h in history]
    fig = go.Figure(data=[go.Heatmap(z=history[0].state, colorscale=CA_CS, zmin=0, zmax=3, showscale=False)],
                     frames=frames)
    fig.update_layout(
        height=440, margin=dict(l=0, r=0, t=10, b=60),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed", visible=False, scaleanchor="x"), xaxis=dict(visible=False),
        updatemenus=[{"type": "buttons", "showactive": False, "x": 0.0, "y": -0.12, "xanchor": "left",
                      "bgcolor": "#161b22", "bordercolor": "#30363d", "font": {"color": "#e6edf3"},
                      "buttons": [
                          {"label": "▶ Play", "method": "animate",
                           "args": [None, {"frame": {"duration": 600, "redraw": True}, "fromcurrent": True,
                                            "transition": {"duration": 150}}]},
                          {"label": "⏸ Pause", "method": "animate",
                           "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
                      ]}],
        sliders=[{"active": 0, "x": 0.12, "len": 0.88, "y": -0.10, "bgcolor": "#161b22", "bordercolor": "#30363d",
                  "font": {"color": "#8b949e", "size": 10},
                  "currentvalue": {"prefix": "T+", "suffix": " min", "font": {"color": "#e6edf3", "size": 12}, "xanchor": "right"},
                  "steps": [{"label": str(h.minutes_elapsed), "method": "animate",
                             "args": [[str(h.minutes_elapsed)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]}
                            for h in history]}],
    )
    st.plotly_chart(fig, use_container_width=True)

    final = history[-1]
    c1, c2, c3, c4 = st.columns(4)
    for col, (lbl, val, cls) in zip([c1, c2, c3, c4],
        [("Horizon", f"{final.minutes_elapsed} min", ""), ("Steps simulated", len(history) - 1, ""),
         ("Cells burning", final.n_burning, "warn"), ("Cells burned", final.n_burned, "crit")]):
        col.markdown(f'<div class="kpi"><div class="lbl">{lbl}</div><div class="val {cls}">{val}</div></div>',
                      unsafe_allow_html=True)

    burned_series = [h.n_burned for h in history]
    burning_series = [h.n_burning for h in history]
    times = [h.minutes_elapsed for h in history]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=times, y=burned_series, name="Burned", line=dict(color="#f78166", width=2),
                               fill="tozeroy", fillcolor="rgba(247,129,102,0.12)"))
    fig2.add_trace(go.Scatter(x=times, y=burning_series, name="Burning", line=dict(color="#d29922", width=2, dash="dot")))
    fig2.update_layout(height=160, margin=dict(l=0, r=0, t=20, b=0),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(title="Minutes", tickfont={"color": "#8b949e", "size": 10}, gridcolor="#21262d"),
                        yaxis=dict(title="Cells", tickfont={"color": "#8b949e", "size": 10}, gridcolor="#21262d"),
                        legend=dict(font={"color": "#8b949e", "size": 10}, bgcolor="rgba(0,0,0,0)"),
                        title=dict(text="Spread over time", font={"color": "#8b949e", "size": 11}, x=0))
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("""
    <div class="legend">
      <span><span class="dot" style="background:#1a3d1f"></span>Unburned forest</span>
      <span><span class="dot" style="background:#ff8a00"></span>Burning</span>
      <span><span class="dot" style="background:#21262d"></span>Burned / charred</span>
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

    df = pd.read_csv(path, index_col=0)
    radar_cols = ["accuracy", "precision", "recall", "f1_score", "auc_roc", "avg_precision"]
    radar_cols = [c for c in radar_cols if c in df.columns]
    colors = ["#f78166", "#d29922", "#3fb950"]
    fill_colors = ["rgba(247,129,102,0.15)", "rgba(210,153,34,0.15)", "rgba(63,185,80,0.15)"]

    fig = go.Figure()
    for i, model in enumerate(df.index):
        vals = df.loc[model, radar_cols].tolist()
        fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=radar_cols + [radar_cols[0]], fill="toself",
                                       name=model, line_color=colors[i % len(colors)],
                                       fillcolor=fill_colors[i % len(fill_colors)], opacity=0.9))
    fig.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(visible=True, range=[0, 1], tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                                    tickfont={"color": "#8b949e", "size": 9}, gridcolor="#21262d", linecolor="#21262d"),
                   angularaxis=dict(tickfont={"color": "#c9d1d9", "size": 11}, gridcolor="#21262d", linecolor="#21262d")),
        showlegend=True, legend={"font": {"color": "#c9d1d9"}, "bgcolor": "rgba(0,0,0,0)"},
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
                               textfont={"color": "#8b949e", "size": 10}))
    fig2.update_layout(barmode="group", height=300, margin=dict(l=0, r=0, t=20, b=0),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(tickfont={"color": "#c9d1d9"}, gridcolor="#21262d"),
                        yaxis=dict(range=[0, 1.15], tickfont={"color": "#8b949e"}, gridcolor="#21262d"),
                        legend=dict(font={"color": "#c9d1d9"}, bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig2, use_container_width=True)

    display_df = df.copy()
    fnr_col = "false_negative_rate" if "false_negative_rate" in display_df.columns else None
    other_cols = [c for c in display_df.columns if c != fnr_col]
    styled = display_df.style.format("{:.4f}")
    styled = styled.highlight_max(subset=other_cols, axis=0, props="background-color:#1a3d2b;color:#3fb950")
    if fnr_col:
        styled = styled.highlight_min(subset=[fnr_col], axis=0, props="background-color:#1a3d2b;color:#3fb950")
    st.dataframe(styled, use_container_width=True)

    st.markdown("""
    <div class="info-box">
    <b>Confirmed real-data results</b> (temporal holdout: train 2023-2024, test 2025 season,
    4.5% positive rate): AUC-ROC ~0.80-0.81 across all three models — consistent with published
    wildfire-prediction literature (typically 0.75-0.90). Each model's decision threshold was
    selected on an internal validation split (never the test set) to target 80% recall — missing
    a real fire is far costlier than a false alarm for this application, so recall is prioritized
    over precision/F1. Recall lands at 70-78% on the actual 2025 test season, slightly below the
    80% validation target — this reflects genuine year-to-year weather variation between the
    training period and the held-out season, not a modeling error. The resulting ~11-12%
    precision is the expected, honest cost of prioritizing recall on this rare-event class.
    </div>
    """, unsafe_allow_html=True)
    return df


def page_nav_card(icon: str, title: str, desc: str):
    st.markdown(f"""
    <div class="nav-card">
        <h4>{icon} {title}</h4>
        <p>{desc}</p>
    </div>
    """, unsafe_allow_html=True)
