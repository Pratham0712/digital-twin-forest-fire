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


# Make secrets available to config.API before any module reads them, so the
# same code path works locally (.env) and deployed (Streamlit Secrets).
if _get_secret_or_env("FIRMS_MAP_KEY"):
    os.environ["FIRMS_MAP_KEY"] = _get_secret_or_env("FIRMS_MAP_KEY")
if _get_secret_or_env("OWM_API_KEY"):
    os.environ["OWM_API_KEY"] = _get_secret_or_env("OWM_API_KEY")


st.set_page_config(
    page_title="Forest Fire Digital Twin - Karnataka & Western Ghats",
    page_icon="🔥", layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_ml_model():
    """Loads the trained XGBoost model if present (models/xgboost.json from
    train.py). Falls back to None -> DigitalTwin uses FWI-based risk instead."""
    model_path = MODELS_DIR / "xgboost.json"
    if not model_path.exists():
        return None
    import xgboost as xgb
    from src.ml_models.model_trainer import XGBoostModel
    wrapper = XGBoostModel()
    wrapper.model.load_model(str(model_path))
    return wrapper


def get_twin(offline: bool) -> DigitalTwin:
    model = load_ml_model()
    return DigitalTwin(ml_model=model, offline=offline)


def severity_color(sev: str) -> str:
    return {"EXTREME": "#8B0000", "HIGH": "#FF4500", "MODERATE": "#FFA500", "LOW": "#2E8B57"}.get(sev, "#808080")


def render_header(summary: dict):
    st.title("🔥 Digital Twin — Forest Fire Prediction")
    st.caption(f"{REGION.name}  ·  Last refreshed: {summary.get('timestamp', 'N/A')}")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Grid Zones", summary.get("total_zones", 0))
    c2.metric("Active Alerts", summary.get("total_alerts", 0))
    breakdown = summary.get("severity_breakdown", {})
    c3.metric("Extreme", breakdown.get("EXTREME", 0))
    c4.metric("High", breakdown.get("HIGH", 0))
    c5.metric("Max Risk Score", f"{summary.get('max_risk_score', 0):.0%}")


def render_risk_map(processed: pd.DataFrame, risk_scores: np.ndarray):
    st.subheader("Regional Risk Map")
    df = processed[["zone_id", "latitude", "longitude", "fwi", "active_fire_nearby"]].copy()
    df["risk_score"] = risk_scores

    fig = px.scatter_map(
        df, lat="latitude", lon="longitude", color="risk_score",
        size=np.clip(df["risk_score"] * 15 + 3, 3, 18),
        color_continuous_scale=["#2E8B57", "#FFA500", "#FF4500", "#8B0000"],
        range_color=(0, 1), zoom=6.2,
        center={"lat": (REGION.min_lat + REGION.max_lat) / 2, "lon": (REGION.min_lon + REGION.max_lon) / 2},
        hover_data={"zone_id": True, "fwi": ":.1f", "risk_score": ":.2f"},
        map_style="carto-positron", height=520,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)


def render_alerts_table(alerts):
    st.subheader(f"Active Alerts ({len(alerts)})")
    if not alerts:
        st.success("No zones currently above the alert threshold.")
        return
    df = pd.DataFrame([{
        "Zone": a.zone_id, "Severity": a.severity, "Risk Score": f"{a.risk_score:.0%}",
        "Reason": a.reason, "Lat": round(a.latitude, 3), "Lon": round(a.longitude, 3),
    } for a in alerts])

    def highlight_severity(row):
        color = severity_color(row["Severity"])
        return [f"background-color: {color}22"] * len(row)

    st.dataframe(df.style.apply(highlight_severity, axis=1), use_container_width=True, height=320)


def render_ca_simulation(twin: DigitalTwin):
    st.subheader("Fire Spread Simulation (Cellular Automata)")
    st.caption(f"Seeded from HIGH/EXTREME alert zones · {SYSTEM.fire_spread_horizon_hours}-hour projection")

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

    step = st.slider("Minutes elapsed", 0, history[-1].minutes_elapsed,
                      value=history[-1].minutes_elapsed, step=SYSTEM.ca_grid_max_seconds * 3)
    closest = min(history, key=lambda h: abs(h.minutes_elapsed - step))

    color_map = {CellState.UNBURNED: 0, CellState.BURNING: 1, CellState.BURNED: 2, CellState.NON_FUEL: 3}
    fig = go.Figure(data=go.Heatmap(
        z=closest.state, colorscale=[
            [0.0, "#E8F5E9"], [0.25, "#E8F5E9"],
            [0.25, "#FF4500"], [0.5, "#FF4500"],
            [0.5, "#3E2723"], [0.75, "#3E2723"],
            [0.75, "#B0BEC5"], [1.0, "#B0BEC5"],
        ], showscale=False,
    ))
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Minutes elapsed", closest.minutes_elapsed)
    c2.metric("Cells burning", closest.n_burning)
    c3.metric("Cells burned", closest.n_burned)
    st.caption("🟢 Unburned · 🟠 Burning · ⬛ Burned · ⬜ Non-fuel (water/bare ground)")


def render_model_comparison():
    st.subheader("Model Comparison")
    path = DATA_PROCESSED_DIR / "model_comparison.csv"
    if not path.exists():
        st.info("Run `python src/ml_models/train.py` first to generate the comparison table.")
        return
    df = pd.read_csv(path, index_col=0)
    st.dataframe(df.style.highlight_max(axis=0, color="#2E8B5744"), use_container_width=True)


def main():
    st.sidebar.header("Controls")
    offline = st.sidebar.toggle(
        "Offline / demo mode", value=not bool(os.getenv("FIRMS_MAP_KEY")),
        help="Uses synthetic data. Turn off once FIRMS_MAP_KEY and OWM_API_KEY are set.",
    )
    st.sidebar.caption(f"Alert threshold: {SYSTEM.alert_threshold_pct:.0f}% (config.py)")

    if "twin" not in st.session_state or st.sidebar.button("🔄 Refresh data"):
        with st.spinner("Refreshing digital twin state..."):
            twin = get_twin(offline)
            twin.refresh()
        st.session_state["twin"] = twin
        st.session_state["ca_history"] = None

    twin: DigitalTwin = st.session_state["twin"]
    snapshot = twin.current_snapshot
    summary = twin.get_summary()

    render_header(summary)
    st.divider()

    tab1, tab2, tab3 = st.tabs(["🗺️ Risk Map & Alerts", "🔥 Spread Simulation", "📊 Model Performance"])
    with tab1:
        col_map, col_alerts = st.columns([2, 1])
        with col_map:
            render_risk_map(snapshot.processed_grid, snapshot.risk_scores)
        with col_alerts:
            render_alerts_table(snapshot.alerts)
    with tab2:
        render_ca_simulation(twin)
    with tab3:
        render_model_comparison()

    st.divider()
    st.caption(
        "Digital Twin Framework for Forest Fire Prediction · BMSCE ISE · "
        "Batch 42 · Data: NASA FIRMS, OpenWeatherMap"
        + (" (offline/demo mode — synthetic data)" if offline else "")
    )


if __name__ == "__main__":
    main()
