"""
Streamlit Dashboard - Layer 5 (Report Ch.6.1.1): operator-facing view of
the Digital Twin. Reads state from DigitalTwin only (never touches lower
layers directly), matching the layered architecture.

Run locally:
    streamlit run src/dashboard/app.py

Deploy on Streamlit Community Cloud:
    - Push repo to GitHub
    - share.streamlit.io -> New app -> point at src/dashboard/app.py
    - Add FIRMS_MAP_KEY / OWM_API_KEY in the app's Secrets manager
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

from config.config import REGION, SYSTEM, MODELS_DIR, DATA_PROCESSED_DIR, RegionConfig
from src.digital_twin.twin_state import DigitalTwin
from src.simulation.cellular_automata import CellState


# ── helpers ────────────────────────────────────────────────────────────────

def _secret(key: str) -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, "")


if _secret("FIRMS_MAP_KEY"):
    os.environ["FIRMS_MAP_KEY"] = _secret("FIRMS_MAP_KEY")
if _secret("OWM_API_KEY"):
    os.environ["OWM_API_KEY"] = _secret("OWM_API_KEY")


st.set_page_config(
    page_title="Forest Fire Digital Twin — Karnataka & Western Ghats",
    page_icon="🔥", layout="wide", initial_sidebar_state="expanded",
)

# ── theme ───────────────────────────────────────────────────────────────────

CSS = """
<style>
/* ── base ── */
.stApp { background: #0b0d0f; }
[data-testid="stSidebar"] { background: #0f1114; border-right: 1px solid #1e2329; }
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
h1,h2,h3 { color: #e6edf3 !important; font-weight: 600 !important; letter-spacing: -0.3px; }
p, span, label, div { color: #8b949e; }
.stTabs [data-baseweb="tab"] { color: #8b949e; font-size: 13px; }
.stTabs [aria-selected="true"] { color: #e6edf3 !important; border-bottom: 2px solid #f78166; }

/* ── hero banner ── */
.hero {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d; border-radius: 12px;
    padding: 20px 26px; margin-bottom: 20px;
}
.hero h1 { margin: 0 0 4px 0; font-size: 22px; color: #e6edf3 !important; }
.hero .sub { color: #8b949e; font-size: 12px; }
.hero .badge {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600; margin-left: 8px; vertical-align: middle;
}
.badge-live { background: #1a4731; color: #3fb950; border: 1px solid #238636; }
.badge-demo { background: #2d1f00; color: #d29922; border: 1px solid #9e6a03; }

/* ── metric cards ── */
.kpi-row { display: flex; gap: 12px; margin-bottom: 18px; flex-wrap: wrap; }
.kpi {
    flex: 1; min-width: 110px;
    background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 14px 16px;
}
.kpi .lbl { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: #8b949e; }
.kpi .val { font-size: 28px; font-weight: 700; margin-top: 2px; color: #e6edf3; }
.kpi .val.ok   { color: #3fb950; }
.kpi .val.warn { color: #d29922; }
.kpi .val.crit { color: #f78166; text-shadow: 0 0 12px rgba(247,129,102,0.4); }

/* ── alert cards ── */
.alert-card {
    border-left: 3px solid; border-radius: 8px;
    padding: 9px 13px; margin-bottom: 7px;
    background: #161b22;
    display: flex; justify-content: space-between; align-items: flex-start;
}
.alert-card .left .zone { font-weight: 600; font-size: 13px; color: #e6edf3; }
.alert-card .left .reason { font-size: 11px; color: #8b949e; margin-top: 2px; }
.alert-card .score { font-size: 13px; font-weight: 700; white-space: nowrap; margin-left: 10px; }

/* ── section headers ── */
.sec-hdr {
    font-size: 11px; text-transform: uppercase; letter-spacing: 1.2px;
    color: #8b949e; border-bottom: 1px solid #21262d;
    padding-bottom: 6px; margin: 18px 0 12px 0;
}

/* ── legend row ── */
.legend { display: flex; gap: 16px; font-size: 11px; color: #8b949e; flex-wrap: wrap; margin-top: 8px; }
.legend .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }

/* ── info box ── */
.info-box {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 12px 16px; font-size: 12px; color: #8b949e;
}
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


# ── model loader ────────────────────────────────────────────────────────────

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


# ── render: header ──────────────────────────────────────────────────────────

def render_header(summary: dict, offline: bool):
    st.markdown(CSS, unsafe_allow_html=True)
    badge_cls = "badge-demo" if offline else "badge-live"
    badge_txt = "DEMO" if offline else "LIVE"
    ts = summary.get("timestamp", "")[:19].replace("T", " ")
    st.markdown(f"""
    <div class="hero">
      <h1>🔥 Forest Fire Digital Twin
        <span class="badge {badge_cls}">{badge_txt}</span>
      </h1>
      <div class="sub">{REGION.name} &nbsp;·&nbsp; Last refreshed {ts} UTC
        &nbsp;·&nbsp; BMS College of Engineering · ISE Batch 42</div>
    </div>
    """, unsafe_allow_html=True)

    bd = summary.get("severity_breakdown", {})
    max_r = summary.get("max_risk_score", 0)
    alerts = summary.get("total_alerts", 0)
    max_cls = "crit" if max_r > 0.6 else ("warn" if max_r > 0.4 else "ok")
    al_cls  = "crit" if alerts > 100 else ("warn" if alerts > 0 else "ok")

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


# ── render: risk map ────────────────────────────────────────────────────────

def render_risk_map(processed: pd.DataFrame, risk_scores: np.ndarray, region=REGION):
    st.markdown('<div class="sec-hdr">Regional risk map</div>', unsafe_allow_html=True)

    df = processed[["zone_id", "latitude", "longitude", "fwi", "active_fire_nearby"]].copy()
    df["risk_score"] = risk_scores
    df["risk_pct"] = (df["risk_score"] * 100).round(1)

    # Use a fixed pixel size derived from grid resolution so cells tile cleanly
    # without overlap. At zoom 6.4 over Karnataka, 0.1 deg ~ 8px on screen.
    # We drive size via a constant column so every marker is identical size —
    # color alone encodes risk, eliminating the blocky-overlap problem.
    df["sz"] = 6

    fig = px.scatter_map(
        df, lat="latitude", lon="longitude",
        color="risk_score",
        size="sz", size_max=6,
        color_continuous_scale=RISK_CS,
        range_color=(0, 1),
        zoom=6.5,
        center={"lat": (region.min_lat + region.max_lat) / 2,
                "lon": (region.min_lon + region.max_lon) / 2},
        hover_data={"zone_id": True, "fwi": ":.1f",
                    "risk_pct": True, "active_fire_nearby": True,
                    "sz": False},
        map_style="carto-darkmatter",
        height=520,
        opacity=0.85,
    )
    fig.update_traces(marker=dict(allowoverlap=True, sizemode="diameter"))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_colorbar=dict(
            title="Risk", tickformat=".0%",
            tickfont={"color": "#8b949e", "size": 10},
            title_font={"color": "#8b949e"},
            len=0.65, thickness=12,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <div class="legend">
      <span><span class="dot" style="background:#1f3a5f"></span>Low risk</span>
      <span><span class="dot" style="background:#9e6a03"></span>Moderate</span>
      <span><span class="dot" style="background:#d29922"></span>Elevated</span>
      <span><span class="dot" style="background:#f0883e"></span>High</span>
      <span><span class="dot" style="background:#f78166"></span>Extreme</span>
    </div>
    """, unsafe_allow_html=True)


# ── render: gauge ───────────────────────────────────────────────────────────

def render_risk_gauge(summary: dict):
    mean_r = summary.get("mean_risk_score", 0) * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=mean_r,
        number={"suffix": "%", "font": {"color": "#e6edf3", "size": 30}},
        gauge={
            "axis": {"range": [0, 100],
                     "tickcolor": "#8b949e",
                     "tickfont": {"color": "#8b949e", "size": 9}},
            "bar": {"color": "#f0883e", "thickness": 0.22},
            "bgcolor": "#161b22",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  40], "color": "#1c2128"},
                {"range": [40, 70], "color": "#2d2200"},
                {"range": [70, 100], "color": "#3d1a1a"},
            ],
            "threshold": {
                "line": {"color": "#f78166", "width": 2},
                "thickness": 0.8,
                "value": SYSTEM.alert_threshold_pct,
            },
        },
    ))
    fig.update_layout(
        height=200, margin=dict(l=16, r=16, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#8b949e"},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Mean regional risk · alert threshold at {SYSTEM.alert_threshold_pct:.0f}%")


# ── render: alerts ──────────────────────────────────────────────────────────

def render_alerts(alerts, summary: dict):
    st.markdown('<div class="sec-hdr">Active alerts</div>', unsafe_allow_html=True)

    bd = summary.get("severity_breakdown", {})
    if bd:
        sevs = ["EXTREME", "HIGH", "MODERATE", "LOW"]
        counts = [bd.get(s, 0) for s in sevs]
        colors = [SEV_COLOR[s] for s in sevs]
        fig = go.Figure(go.Bar(
            x=sevs, y=counts,
            marker_color=colors,
            text=counts, textposition="outside",
            textfont={"color": "#8b949e", "size": 11},
        ))
        fig.update_layout(
            height=160, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickfont={"color": "#8b949e", "size": 11}, gridcolor="#21262d"),
            yaxis=dict(visible=False),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    if not alerts:
        st.markdown('<div class="info-box">No zones above the alert threshold.</div>',
                    unsafe_allow_html=True)
        return

    for a in alerts[:15]:
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
    if len(alerts) > 15:
        st.caption(f"+ {len(alerts) - 15} more zones above threshold")


# ── render: CA simulation ───────────────────────────────────────────────────

def render_ca_simulation(twin: DigitalTwin):
    st.markdown('<div class="sec-hdr">Fire spread simulation — 2-hour projection</div>',
                unsafe_allow_html=True)
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

    # animated heatmap
    frames = [
        go.Frame(
            data=[go.Heatmap(z=h.state, colorscale=CA_CS, zmin=0, zmax=3,
                             showscale=False)],
            name=str(h.minutes_elapsed),
        )
        for h in history
    ]
    fig = go.Figure(
        data=[go.Heatmap(z=history[0].state, colorscale=CA_CS, zmin=0, zmax=3,
                         showscale=False)],
        frames=frames,
    )
    fig.update_layout(
        height=440,
        margin=dict(l=0, r=0, t=10, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed", visible=False, scaleanchor="x"),
        xaxis=dict(visible=False),
        updatemenus=[{
            "type": "buttons", "showactive": False,
            "x": 0.0, "y": -0.12, "xanchor": "left",
            "bgcolor": "#161b22", "bordercolor": "#30363d",
            "font": {"color": "#e6edf3"},
            "buttons": [
                {"label": "▶ Play", "method": "animate",
                 "args": [None, {"frame": {"duration": 600, "redraw": True},
                                 "fromcurrent": True,
                                 "transition": {"duration": 150}}]},
                {"label": "⏸ Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                   "mode": "immediate"}]},
            ],
        }],
        sliders=[{
            "active": 0, "x": 0.12, "len": 0.88, "y": -0.10,
            "bgcolor": "#161b22", "bordercolor": "#30363d",
            "font": {"color": "#8b949e", "size": 10},
            "currentvalue": {"prefix": "T+", "suffix": " min",
                             "font": {"color": "#e6edf3", "size": 12},
                             "xanchor": "right"},
            "steps": [
                {"label": str(h.minutes_elapsed), "method": "animate",
                 "args": [[str(h.minutes_elapsed)],
                          {"frame": {"duration": 0, "redraw": True},
                           "mode": "immediate"}]}
                for h in history
            ],
        }],
    )
    st.plotly_chart(fig, use_container_width=True)

    # stats row
    final = history[-1]
    c1, c2, c3, c4 = st.columns(4)
    for col, (lbl, val, cls) in zip(
        [c1, c2, c3, c4],
        [("Horizon", f"{final.minutes_elapsed} min", ""),
         ("Steps simulated", len(history) - 1, ""),
         ("Cells burning", final.n_burning, "warn"),
         ("Cells burned", final.n_burned, "crit")],
    ):
        col.markdown(f"""
        <div class="kpi">
          <div class="lbl">{lbl}</div>
          <div class="val {cls}">{val}</div>
        </div>
        """, unsafe_allow_html=True)

    # spread-over-time sparkline
    burned_series = [h.n_burned for h in history]
    burning_series = [h.n_burning for h in history]
    times = [h.minutes_elapsed for h in history]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=times, y=burned_series, name="Burned",
                              line=dict(color="#f78166", width=2), fill="tozeroy",
                              fillcolor="rgba(247,129,102,0.12)"))
    fig2.add_trace(go.Scatter(x=times, y=burning_series, name="Burning",
                              line=dict(color="#d29922", width=2, dash="dot")))
    fig2.update_layout(
        height=160, margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Minutes", tickfont={"color": "#8b949e", "size": 10},
                   gridcolor="#21262d", title_font={"color": "#8b949e"}),
        yaxis=dict(title="Cells", tickfont={"color": "#8b949e", "size": 10},
                   gridcolor="#21262d", title_font={"color": "#8b949e"}),
        legend=dict(font={"color": "#8b949e", "size": 10},
                    bgcolor="rgba(0,0,0,0)"),
        title=dict(text="Spread over time", font={"color": "#8b949e", "size": 11},
                   x=0),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    <div class="legend">
      <span><span class="dot" style="background:#1a3d1f"></span>Unburned forest</span>
      <span><span class="dot" style="background:#ff8a00"></span>Burning</span>
      <span><span class="dot" style="background:#21262d"></span>Burned / charred</span>
      <span><span class="dot" style="background:#1f3a5f"></span>Non-fuel (water / bare ground)</span>
    </div>
    """, unsafe_allow_html=True)


# ── render: model performance ───────────────────────────────────────────────

def render_model_performance():
    st.markdown('<div class="sec-hdr">Model performance — real data, temporal holdout (test = 2025 season)</div>',
                unsafe_allow_html=True)

    path = DATA_PROCESSED_DIR / "model_comparison_real.csv"
    if not path.exists():
        st.markdown('<div class="info-box">Run <code>python src/ml_models/train_real.py</code> to generate results.</div>',
                    unsafe_allow_html=True)
        return

    df = pd.read_csv(path, index_col=0)

    # ── radar: display metrics where higher = better ──
    radar_cols = ["accuracy", "precision", "recall", "f1_score", "auc_roc", "avg_precision"]
    radar_cols = [c for c in radar_cols if c in df.columns]
    colors      = ["#f78166", "#d29922", "#3fb950"]
    fill_colors = ["rgba(247,129,102,0.15)", "rgba(210,153,34,0.15)", "rgba(63,185,80,0.15)"]

    fig = go.Figure()
    for i, model in enumerate(df.index):
        vals = df.loc[model, radar_cols].tolist()
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=radar_cols + [radar_cols[0]],
            fill="toself",
            name=model,
            line_color=colors[i % len(colors)],
            fillcolor=fill_colors[i % len(fill_colors)],
            opacity=0.9,
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                tickfont={"color": "#8b949e", "size": 9},
                gridcolor="#21262d", linecolor="#21262d",
            ),
            angularaxis=dict(
                tickfont={"color": "#c9d1d9", "size": 11},
                gridcolor="#21262d", linecolor="#21262d",
            ),
        ),
        showlegend=True,
        legend={"font": {"color": "#c9d1d9"}, "bgcolor": "rgba(0,0,0,0)"},
        paper_bgcolor="rgba(0,0,0,0)",
        height=400,
        margin=dict(l=50, r=50, t=30, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── bar chart: AUC-ROC side by side (most meaningful single metric) ──
    fig2 = go.Figure()
    metric_display = {
        "auc_roc": "AUC-ROC", "recall": "Recall", "precision": "Precision",
        "f1_score": "F1", "avg_precision": "Avg Precision",
    }
    x_labels = [metric_display.get(c, c) for c in metric_display if c in df.columns]
    for i, model in enumerate(df.index):
        y_vals = [df.loc[model, c] for c in metric_display if c in df.columns]
        fig2.add_trace(go.Bar(
            name=model, x=x_labels, y=y_vals,
            marker_color=colors[i % len(colors)],
            text=[f"{v:.3f}" for v in y_vals],
            textposition="outside",
            textfont={"color": "#8b949e", "size": 10},
        ))
    fig2.update_layout(
        barmode="group", height=300,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickfont={"color": "#c9d1d9"}, gridcolor="#21262d"),
        yaxis=dict(range=[0, 1.15], tickfont={"color": "#8b949e"},
                   gridcolor="#21262d"),
        legend=dict(font={"color": "#c9d1d9"}, bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── table ──
    display_df = df.copy()
    # FNR: lower is better — highlight_min, not highlight_max
    fnr_col = "false_negative_rate" if "false_negative_rate" in display_df.columns else None
    other_cols = [c for c in display_df.columns if c != fnr_col]

    styled = display_df.style.format("{:.4f}")
    styled = styled.highlight_max(subset=other_cols, axis=0,
                                  props="background-color:#1a3d2b;color:#3fb950")
    if fnr_col:
        styled = styled.highlight_min(subset=[fnr_col], axis=0,
                                      props="background-color:#1a3d2b;color:#3fb950")
    st.dataframe(styled, use_container_width=True)

    st.markdown("""
    <div class="info-box">
    <b>Note:</b> AUC-ROC ~0.80 on a genuine temporal holdout (train 2023-2024, test 2025 season).
    Low precision / F1 reflects the 4.5% positive rate (rare-event class imbalance) —
    see Known Limitations in the README. Threshold tuned to 0.40 for best recall/F1 balance.
    </div>
    """, unsafe_allow_html=True)


# ── sidebar ─────────────────────────────────────────────────────────────────

def build_sidebar():
    st.sidebar.markdown("### Controls")
    offline = st.sidebar.toggle(
        "Offline / demo mode",
        value=not bool(os.getenv("FIRMS_MAP_KEY")),
        help="Uses synthetic data. Disable once API keys are set in .env or Streamlit Secrets.",
    )

    scenario = None
    if offline:
        st.sidebar.markdown("**Scenario**")
        n_hotspots    = st.sidebar.slider("Active fire hotspots", 0, 40, 8)
        temp_c        = st.sidebar.slider("Temperature (°C)", 15, 48, 32)
        wind_speed_ms = st.sidebar.slider("Wind speed (m/s)", 0.0, 20.0, 5.0)
        humidity_pct  = st.sidebar.slider("Humidity (%)", 0, 100, 40)
        scenario = {"n_hotspots": n_hotspots, "temp_c": temp_c,
                    "wind_speed_ms": wind_speed_ms, "humidity_pct": humidity_pct}

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Region**")
    PRESETS = {
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
        "Custom": None,
    }
    choice = st.sidebar.selectbox("Preset", list(PRESETS.keys()))
    if choice == "Custom":
        c1, c2 = st.sidebar.columns(2)
        min_lat = c1.number_input("Min lat", value=11.5, format="%.2f")
        max_lat = c2.number_input("Max lat", value=15.5, format="%.2f")
        min_lon = c1.number_input("Min lon", value=74.0, format="%.2f")
        max_lon = c2.number_input("Max lon", value=77.5, format="%.2f")
        name    = st.sidebar.text_input("Name", value="Custom region")
        region  = RegionConfig(name=name, min_lat=min_lat, max_lat=max_lat,
                               min_lon=min_lon, max_lon=max_lon,
                               grid_resolution_deg=0.5, weather_grid_resolution_deg=1.5)
    else:
        region = PRESETS[choice]

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


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    offline, scenario, region, force_refresh = build_sidebar()

    need_refresh = (
        "twin" not in st.session_state
        or force_refresh
        or st.session_state.get("_last_region") != region.name
        or st.session_state.get("_last_offline") != offline
    )

    if need_refresh:
        with st.spinner("Refreshing digital twin state..."):
            twin = get_twin(offline, scenario, region)
            twin.refresh()
        st.session_state["twin"] = twin
        st.session_state["ca_history"] = None
        st.session_state["_last_region"] = region.name
        st.session_state["_last_offline"] = offline

    twin: DigitalTwin = st.session_state["twin"]
    snap = twin.current_snapshot
    summary = twin.get_summary()

    render_header(summary, offline)

    tab1, tab2, tab3 = st.tabs(["🗺  Risk map & alerts", "🔥  Spread simulation", "📊  Model performance"])

    with tab1:
        col_map, col_side = st.columns([3, 1], gap="medium")
        with col_map:
            render_risk_map(snap.processed_grid, snap.risk_scores, twin.region)
        with col_side:
            render_risk_gauge(summary)
            render_alerts(snap.alerts, summary)

    with tab2:
        render_ca_simulation(twin)

    with tab3:
        render_model_performance()

    st.markdown("---")
    st.caption(
        "Digital Twin Framework for Forest Fire Prediction · BMSCE ISE · Batch 42 · "
        "Data: NASA FIRMS (VIIRS SNPP), OpenWeatherMap · Canadian FWI System"
        + ("  ·  ⚠ offline/demo mode" if offline else "")
    )


if __name__ == "__main__":
    main()
