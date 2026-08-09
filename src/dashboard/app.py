"""
Streamlit Dashboard - Layer 5 (Report Ch.6.1.1): the operator-facing view of
the Digital Twin. Reads state from DigitalTwin only (never touches the lower
layers directly), matching the layered architecture.

Run locally:
    streamlit run src/dashboard/app.py

Deploy on Streamlit Community Cloud:
    - Push repo to GitHub
    - share.streamlit.io -> New app -> point at src/dashboard/app.py
    - Add FIRMS_MAP_KEY / OWM_API_KEY in the app's Secrets manager (not .env)
"""
import os
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.config import REGION, SYSTEM, MODELS_DIR, DATA_PROCESSED_DIR
from src.digital_twin.twin_state import DigitalTwin
from src.simulation.cellular_automata import CellState


def _get_secret_or_env(key: str) -> str:
    """Reads from Streamlit Secrets when deployed, falls back to .env/os.environ locally."""
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, "")


if _get_secret_or_env("FIRMS_MAP_KEY"):
    os.environ["FIRMS_MAP_KEY"] = _get_secret_or_env("FIRMS_MAP_KEY")
if _get_secret_or_env("OWM_API_KEY"):
    os.environ["OWM_API_KEY"] = _get_secret_or_env("OWM_API_KEY")


st.set_page_config(
    page_title="Forest Fire Digital Twin - Karnataka & Western Ghats",
    page_icon="🔥", layout="wide", initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Theme: dark "command center" look. Fire-domain palette - char black base,
# ember amber/orange for active risk, deep red for extreme, cool slate for
# calm/unburned. Injected once at the top of the page.
# --------------------------------------------------------------------------- #
THEME_CSS = """
<style>
.stApp { background: radial-gradient(circle at 20% 0%, #1a1410 0%, #0d0b0a 55%, #050403 100%); }
[data-testid="stSidebar"] { background: #0d0b0a; border-right: 1px solid #2a221a; }
h1, h2, h3 { color: #f5ede3 !important; font-weight: 600 !important; }
p, span, label { color: #cbbfb0; }

.dtw-hero {
    background: linear-gradient(120deg, #1f1611 0%, #2a1810 45%, #1a1108 100%);
    border: 1px solid #3d2a1a; border-radius: 14px;
    padding: 22px 28px; margin-bottom: 18px;
    box-shadow: 0 0 40px rgba(255,120,40,0.06);
}
.dtw-hero h1 { margin: 0; font-size: 28px; letter-spacing: 0.3px; }
.dtw-hero .sub { color: #a8917a; font-size: 13px; margin-top: 4px; }

.dtw-metric {
    background: #14100c; border: 1px solid #2e241a; border-radius: 12px;
    padding: 14px 16px; text-align: left;
}
.dtw-metric .label { color: #8a7a68; font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; }
.dtw-metric .value { color: #f5ede3; font-size: 26px; font-weight: 600; margin-top: 2px; }
.dtw-metric .value.warn { color: #ff8a3d; }
.dtw-metric .value.danger { color: #ff4d4d; text-shadow: 0 0 14px rgba(255,77,77,0.5); }
.dtw-metric .value.ok { color: #5fd68a; }

.dtw-alert-card {
    border-left: 3px solid; border-radius: 8px; padding: 10px 14px;
    margin-bottom: 8px; background: #14100c;
}
.dtw-alert-card .zone { font-weight: 600; font-size: 14px; color: #f5ede3; }
.dtw-alert-card .reason { font-size: 12px; color: #a8917a; margin-top: 2px; }
.dtw-alert-card .score { float: right; font-size: 13px; font-weight: 600; }

.dtw-legend { display: flex; gap: 18px; font-size: 12px; color: #a8917a; margin-top: 8px; flex-wrap: wrap; }
.dtw-legend span.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }
</style>
"""

SEVERITY_COLORS = {
    "EXTREME": "#ff2d2d", "HIGH": "#ff8a3d", "MODERATE": "#ffce54", "LOW": "#5fd68a",
}

# Fire-domain colorscale: cool slate (calm) -> ember amber -> deep red (extreme)
RISK_COLORSCALE = [
    [0.0, "#2b3a4a"], [0.25, "#4a6178"], [0.5, "#e8a33d"],
    [0.75, "#ff6a3d"], [1.0, "#ff1f1f"],
]


@st.cache_resource(show_spinner=False)
def load_ml_model():
    model_path = MODELS_DIR / "xgboost.json"
    if not model_path.exists():
        return None
    from src.ml_models.model_trainer import XGBoostModel
    wrapper = XGBoostModel()
    wrapper.model.load_model(str(model_path))
    return wrapper


def get_twin(offline: bool) -> DigitalTwin:
    model = load_ml_model()
    return DigitalTwin(ml_model=model, offline=offline)


def render_header(summary: dict, offline: bool):
    mode_badge = "OFFLINE / DEMO" if offline else "LIVE"
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="dtw-hero">
        <h1>Digital twin — forest fire prediction</h1>
        <div class="sub">{REGION.name} · {mode_badge} · Last refreshed {summary.get('timestamp', 'N/A')[:19].replace('T', ' ')} UTC</div>
    </div>
    """, unsafe_allow_html=True)

    breakdown = summary.get("severity_breakdown", {})
    max_risk = summary.get("max_risk_score", 0)
    max_risk_class = "danger" if max_risk > 0.7 else ("warn" if max_risk > 0.4 else "ok")

    cols = st.columns(5)
    metrics = [
        ("Grid zones", summary.get("total_zones", 0), ""),
        ("Active alerts", summary.get("total_alerts", 0), "warn" if summary.get("total_alerts", 0) else "ok"),
        ("Extreme", breakdown.get("EXTREME", 0), "danger" if breakdown.get("EXTREME", 0) else "ok"),
        ("High", breakdown.get("HIGH", 0), "warn" if breakdown.get("HIGH", 0) else "ok"),
        ("Peak risk", f"{max_risk:.0%}", max_risk_class),
    ]
    for col, (label, value, cls) in zip(cols, metrics):
        col.markdown(f"""
        <div class="dtw-metric">
            <div class="label">{label}</div>
            <div class="value {cls}">{value}</div>
        </div>
        """, unsafe_allow_html=True)


def render_risk_gauge(summary: dict):
    mean_risk = summary.get("mean_risk_score", 0) * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=mean_risk,
        number={"suffix": "%", "font": {"color": "#f5ede3", "size": 34}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8a7a68", "tickfont": {"color": "#8a7a68", "size": 10}},
            "bar": {"color": "#ff6a3d", "thickness": 0.25},
            "bgcolor": "#14100c",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "#1f3040"},
                {"range": [40, 70], "color": "#4a3520"},
                {"range": [70, 100], "color": "#4a1f1f"},
            ],
            "threshold": {
                "line": {"color": "#ff1f1f", "width": 3},
                "thickness": 0.85, "value": SYSTEM.alert_threshold_pct,
            },
        },
    ))
    fig.update_layout(
        height=220, margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#cbbfb0"},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Regional average risk · red line marks the alert threshold")


def render_risk_map(processed: pd.DataFrame, risk_scores: np.ndarray):
    st.subheader("Regional risk map")
    df = processed[["zone_id", "latitude", "longitude", "fwi", "active_fire_nearby"]].copy()
    df["risk_score"] = risk_scores
    df["glow_size"] = np.clip(df["risk_score"] * 22 + 4, 4, 26)

    fig = px.scatter_map(
        df, lat="latitude", lon="longitude", color="risk_score",
        size="glow_size", size_max=26,
        color_continuous_scale=RISK_COLORSCALE, range_color=(0, 1), zoom=6.4,
        center={"lat": (REGION.min_lat + REGION.max_lat) / 2, "lon": (REGION.min_lon + REGION.max_lon) / 2},
        hover_data={"zone_id": True, "fwi": ":.1f", "risk_score": ":.2f", "glow_size": False},
        map_style="carto-darkmatter", height=540,
        opacity=0.85,
    )
    fig.update_traces(marker=dict(allowoverlap=True))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_colorbar=dict(
            title="Risk", tickfont={"color": "#cbbfb0"}, title_font={"color": "#cbbfb0"},
            len=0.7,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <div class="dtw-legend">
        <span><span class="dot" style="background:#4a6178"></span>Calm</span>
        <span><span class="dot" style="background:#e8a33d"></span>Elevated</span>
        <span><span class="dot" style="background:#ff6a3d"></span>High</span>
        <span><span class="dot" style="background:#ff1f1f"></span>Extreme</span>
        <span>Marker size and glow scale with risk score</span>
    </div>
    """, unsafe_allow_html=True)


def render_alerts_table(alerts):
    st.subheader(f"Active alerts ({len(alerts)})")
    if not alerts:
        st.success("No zones currently above the alert threshold.")
        return
    for a in alerts[:20]:
        color = SEVERITY_COLORS.get(a.severity, "#888")
        st.markdown(f"""
        <div class="dtw-alert-card" style="border-left-color:{color}">
            <span class="score" style="color:{color}">{a.risk_score:.0%}</span>
            <div class="zone">{a.zone_id} · {a.severity}</div>
            <div class="reason">{a.reason}</div>
        </div>
        """, unsafe_allow_html=True)
    if len(alerts) > 20:
        st.caption(f"+ {len(alerts) - 20} more zones above threshold")


def _ca_colorscale():
    """
    Fire-realistic discrete colorscale for the CA grid:
    unburned = deep forest green, burning = hot ember gradient (amber->red),
    burned = charred ash gray-black, non-fuel = cool slate blue (water/bare rock).
    """
    return [
        [0.00, "#1a3d1f"], [0.24, "#1a3d1f"],
        [0.25, "#ff8a00"], [0.49, "#ff2d00"],
        [0.50, "#2b2420"], [0.74, "#2b2420"],
        [0.75, "#2e4a5e"], [1.00, "#2e4a5e"],
    ]


def render_ca_simulation(twin: DigitalTwin):
    st.subheader("Fire spread simulation")
    st.caption(f"Seeded from HIGH/EXTREME alert zones · {SYSTEM.fire_spread_horizon_hours}-hour projection · "
               f"8-neighbour cellular automata, wind-aligned spread")

    if st.button("Run spread simulation", type="primary"):
        with st.spinner("Simulating fire spread..."):
            history = twin.simulate_spread_from_alerts()
        st.session_state["ca_history"] = history

    history = st.session_state.get("ca_history")
    if not history:
        st.info("Click 'Run spread simulation' to project spread from current alert zones.")
        return
    if len(history) <= 1:
        st.warning("No HIGH/EXTREME zones this cycle - nothing to simulate.")
        return

    frames = [
        go.Frame(data=[go.Heatmap(z=h.state, colorscale=_ca_colorscale(), zmin=0, zmax=3, showscale=False)],
                  name=str(h.minutes_elapsed))
        for h in history
    ]
    fig = go.Figure(
        data=[go.Heatmap(z=history[0].state, colorscale=_ca_colorscale(), zmin=0, zmax=3, showscale=False)],
        frames=frames,
    )
    fig.update_layout(
        height=460, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed", visible=False),
        xaxis=dict(visible=False),
        updatemenus=[{
            "type": "buttons", "showactive": False, "x": 0.0, "y": -0.06, "xanchor": "left",
            "buttons": [
                {"label": "Play", "method": "animate",
                 "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True, "transition": {"duration": 200}}]},
                {"label": "Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
            ],
        }],
        sliders=[{
            "active": 0, "x": 0.08, "len": 0.92, "y": -0.06,
            "currentvalue": {"prefix": "Minutes elapsed: ", "font": {"color": "#cbbfb0"}},
            "steps": [
                {"label": str(h.minutes_elapsed), "method": "animate",
                 "args": [[str(h.minutes_elapsed)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]}
                for h in history
            ],
        }],
    )
    st.plotly_chart(fig, use_container_width=True)

    final = history[-1]
    c1, c2, c3 = st.columns(3)
    for col, (label, value, cls) in zip(
        [c1, c2, c3],
        [("Final horizon", f"{final.minutes_elapsed} min", ""),
         ("Cells burning", final.n_burning, "warn"),
         ("Cells burned", final.n_burned, "danger")],
    ):
        col.markdown(f"""
        <div class="dtw-metric">
            <div class="label">{label}</div>
            <div class="value {cls}">{value}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="dtw-legend">
        <span><span class="dot" style="background:#1a3d1f"></span>Unburned forest</span>
        <span><span class="dot" style="background:#ff2d00"></span>Burning</span>
        <span><span class="dot" style="background:#2b2420"></span>Burned / charred</span>
        <span><span class="dot" style="background:#2e4a5e"></span>Non-fuel (water/bare ground)</span>
    </div>
    """, unsafe_allow_html=True)


def render_model_comparison():
    st.subheader("Model performance")
    path = DATA_PROCESSED_DIR / "model_comparison.csv"
    if not path.exists():
        st.info("Run `python src/ml_models/train.py` first to generate the comparison table.")
        return
    df = pd.read_csv(path, index_col=0)

    fig = go.Figure()
    metric_cols = [c for c in df.columns if c != "false_negative_rate"]
    colors = ["#ff8a3d", "#ffce54", "#5fd68a", "#4aa3ff", "#c58aff"]
    for i, model in enumerate(df.index):
        fig.add_trace(go.Scatterpolar(
            r=df.loc[model, metric_cols].values, theta=metric_cols, fill="toself",
            name=model, line_color=colors[i % len(colors)], opacity=0.75,
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], tickfont={"color": "#8a7a68", "size": 9}, gridcolor="#2e241a"),
            angularaxis=dict(tickfont={"color": "#cbbfb0"}, gridcolor="#2e241a"),
        ),
        showlegend=True, legend={"font": {"color": "#cbbfb0"}},
        paper_bgcolor="rgba(0,0,0,0)", height=420, margin=dict(l=40, r=40, t=30, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df.style.highlight_max(axis=0, color="#2e4a2244"), use_container_width=True)


def main():
    st.sidebar.header("Controls")
    offline = st.sidebar.toggle(
        "Offline / demo mode", value=not bool(os.getenv("FIRMS_MAP_KEY")),
        help="Uses synthetic data. Turn off once FIRMS_MAP_KEY and OWM_API_KEY are set.",
    )
    st.sidebar.caption(f"Alert threshold: {SYSTEM.alert_threshold_pct:.0f}% (config.py)")

    if "twin" not in st.session_state or st.sidebar.button("Refresh data"):
        with st.spinner("Refreshing digital twin state..."):
            twin = get_twin(offline)
            twin.refresh()
        st.session_state["twin"] = twin
        st.session_state["ca_history"] = None

    twin: DigitalTwin = st.session_state["twin"]
    snapshot = twin.current_snapshot
    summary = twin.get_summary()

    render_header(summary, offline)

    tab1, tab2, tab3 = st.tabs(["Risk map & alerts", "Spread simulation", "Model performance"])
    with tab1:
        col_map, col_side = st.columns([2, 1])
        with col_map:
            render_risk_map(snapshot.processed_grid, snapshot.risk_scores)
        with col_side:
            render_risk_gauge(summary)
            render_alerts_table(snapshot.alerts)
    with tab2:
        render_ca_simulation(twin)
    with tab3:
        render_model_comparison()

    st.markdown("---")
    st.caption(
        "Digital Twin Framework for Forest Fire Prediction · BMSCE ISE · "
        "Batch 42 · Data: NASA FIRMS, OpenWeatherMap"
        + (" (offline/demo mode — synthetic data)" if offline else "")
    )


if __name__ == "__main__":
    main()
