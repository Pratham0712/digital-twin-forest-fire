"""
Historical Time Machine - NEW feature page. Animates real satellite-confirmed
fire detections (NASA FIRMS VIIRS archive) across the 2023-2025 fire seasons,
month by month, on the same dark map style used elsewhere. This is the
project's clearest differentiator: almost no comparable student project has
3 real fire seasons of ground-truth satellite data to show.

Falls back to a clearly-labeled synthetic preview if the real historical CSV
(data/raw/historical_fires_karnataka.csv, built by
src/data_ingestion/historical_firms.py) isn't present locally, so the page
is never blank - but the fallback is never silently presented as real data.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.dashboard_common import set_page, REGION, DATA_RAW_DIR

from src.auth.auth_gate import require_login

set_page("Historical Time Machine", "🕰️")
require_login()

st.markdown("""
<div class="hero">
  <h1>🕰️ Historical Time Machine</h1>
  <div class="sub">Real NASA FIRMS satellite detections across three full fire seasons
  (2023–2025), animated month by month.</div>
</div>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_real_history():
    path = DATA_RAW_DIR / "historical_fires_karnataka.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    df["year_month"] = df["acq_date"].dt.strftime("%Y-%m")
    return df


@st.cache_data(show_spinner=False)
def generate_sample_history(seed: int = 99):
    """Clearly-labeled SAMPLE PREVIEW only - used when the real historical
    CSV hasn't been pulled locally yet. Same seasonal shape as the real data
    (heavier Feb-Apr, tapering into May) so the visualization is meaningful,
    but every render of this data is labeled as synthetic in the UI."""
    rng = np.random.default_rng(seed)
    rows = []
    for year in [2023, 2024, 2025]:
        for month in range(1, 6):
            # seasonal weighting: peak in March/April, matching real pattern
            weight = {1: 0.3, 2: 0.7, 3: 1.0, 4: 0.9, 5: 0.5}[month]
            n = int(rng.poisson(120 * weight))
            for _ in range(n):
                day = rng.integers(1, 29)
                rows.append({
                    "latitude": rng.uniform(REGION.min_lat, REGION.max_lat),
                    "longitude": rng.uniform(REGION.min_lon, REGION.max_lon),
                    "frp": rng.exponential(15),
                    "confidence": rng.choice(["nominal", "high"], p=[0.6, 0.4]),
                    "acq_date": pd.Timestamp(year=year, month=month, day=day),
                })
    df = pd.DataFrame(rows)
    df["year_month"] = df["acq_date"].dt.strftime("%Y-%m")
    return df


real_df = load_real_history()
using_real = real_df is not None
df = real_df if using_real else generate_sample_history()

if using_real:
    st.success(f"✅ Using REAL data: {len(df):,} satellite-confirmed detections, "
               f"{df['acq_date'].min().date()} to {df['acq_date'].max().date()}")
else:
    st.warning("⚠️ Real historical data not found. Showing a synthetic preview instead.")

# ── year-by-year summary ──
st.markdown('<div class="sec-hdr">Detections per fire season</div>', unsafe_allow_html=True)
year_counts = df["acq_date"].dt.year.value_counts().sort_index()
cols = st.columns(len(year_counts))
for col, (year, count) in zip(cols, year_counts.items()):
    col.markdown(f'<div class="kpi"><div class="lbl">{year} season</div>'
                 f'<div class="val">{count:,}</div></div>', unsafe_allow_html=True)

# ── monthly seasonal pattern (all years overlaid) ──
st.markdown('<div class="sec-hdr">Seasonal pattern</div>', unsafe_allow_html=True)
df["month_name"] = df["acq_date"].dt.strftime("%b")
df["year"] = df["acq_date"].dt.year
month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
monthly = df.groupby(["year", "month_name"]).size().reset_index(name="detections")
fig_season = go.Figure()
year_colors = {2023: "#f78166", 2024: "#d29922", 2025: "#58a6ff"}
for year in sorted(df["year"].unique()):
    sub = monthly[monthly["year"] == year].set_index("month_name").reindex(month_order).fillna(0)
    fig_season.add_trace(go.Bar(x=month_order, y=sub["detections"], name=str(year),
                                 marker_color=year_colors.get(year, "#8b949e")))
fig_season.update_layout(
    barmode="group", height=260, margin=dict(l=0, r=0, t=10, b=0),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(tickfont={"color": "#c9d1d9"}, gridcolor="#21262d"),
    yaxis=dict(title="Detections", tickfont={"color": "#8b949e"}, gridcolor="#21262d",
               title_font={"color": "#8b949e"}),
    legend=dict(font={"color": "#c9d1d9"}, bgcolor="rgba(0,0,0,0)"),
)
st.plotly_chart(fig_season, use_container_width=True)
st.caption("Fire activity concentrates in Feb–Apr and tapers by May, consistent with the "
           "Western Ghats' known pre-monsoon fire season.")

# ── animated map, month by month ──
st.markdown('<div class="sec-hdr">Animated detection map</div>', unsafe_allow_html=True)
months_sorted = sorted(df["year_month"].unique())

frames = []
for ym in months_sorted:
    sub = df[df["year_month"] == ym]
    frames.append(go.Frame(
        data=[go.Scattermap(
            lat=sub["latitude"], lon=sub["longitude"], mode="markers",
            marker=dict(size=7, color="#f78166", opacity=0.75),
        )],
        name=ym,
    ))

first = df[df["year_month"] == months_sorted[0]]
fig_map = go.Figure(
    data=[go.Scattermap(lat=first["latitude"], lon=first["longitude"], mode="markers",
                         marker=dict(size=7, color="#f78166", opacity=0.75))],
    frames=frames,
)
fig_map.update_layout(
    map=dict(style="carto-darkmatter", zoom=6.2,
              center={"lat": (REGION.min_lat + REGION.max_lat) / 2,
                      "lon": (REGION.min_lon + REGION.max_lon) / 2}),
    height=520, margin=dict(l=0, r=0, t=10, b=60), paper_bgcolor="rgba(0,0,0,0)",
    updatemenus=[{"type": "buttons", "showactive": False, "x": 0.0, "y": -0.08, "xanchor": "left",
                  "bgcolor": "#161b22", "bordercolor": "#30363d", "font": {"color": "#e6edf3"},
                  "buttons": [
                      {"label": "▶ Play", "method": "animate",
                       "args": [None, {"frame": {"duration": 700, "redraw": True}, "fromcurrent": True,
                                        "transition": {"duration": 200}}]},
                      {"label": "⏸ Pause", "method": "animate",
                       "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
                  ]}],
    sliders=[{"active": 0, "x": 0.1, "len": 0.9, "y": -0.06, "bgcolor": "#161b22", "bordercolor": "#30363d",
              "font": {"color": "#8b949e", "size": 10},
              "currentvalue": {"prefix": "Month: ", "font": {"color": "#e6edf3", "size": 13}},
              "steps": [{"label": ym, "method": "animate",
                         "args": [[ym], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]}
                        for ym in months_sorted]}],
)
st.plotly_chart(fig_map, use_container_width=True)
st.caption(("Real" if using_real else "Synthetic preview") +
           f" — {len(months_sorted)} months animated, {len(df):,} total detections plotted.")
