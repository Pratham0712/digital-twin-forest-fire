"""
Historical Time Machine - NEW feature page. Animates real satellite-confirmed
fire detections (NASA FIRMS VIIRS archive) across every fire season in the archive (2023 onward),
month by month, on the same dark map style used elsewhere. This is the
project's clearest differentiator: almost no comparable student project has
3 real fire seasons of ground-truth satellite data to show.

Follows the region chosen in the sidebar. Karnataka reads
data/raw/historical_fires_karnataka.csv; every other state reads its share of
the pan-India archive (data/raw/historical_fires_india_<year>.csv, built by
scripts/build_india_regions_dataset.py) filtered to the state's bounding box.
If a region has no archive yet, the page says so plainly (Karnataka alone falls
back to a labelled synthetic preview) - synthetic points are never shown for
a region they don't describe.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.dashboard_common import (
    set_page, build_sidebar, REGION, DATA_RAW_DIR, _zoom_for,
)
from src.auth.auth_gate import require_login, render_user_badge_in_sidebar

set_page("Historical Time Machine")
require_login()
_, _, region, _ = build_sidebar(show_offline=False)
render_user_badge_in_sidebar()

st.markdown(f"""
<div class="hero">
  <div class="eyebrow">Forest Fire Digital Twin</div>
  <h1>Historical Time Machine</h1>
  <div class="sub">Real NASA FIRMS satellite detections for <b>{region.name}</b> across the
  2023 to 2026 fire seasons, animated month by month.</div>
</div>
""", unsafe_allow_html=True)

COLORS = {2023: "#ff6b4a", 2024: "#fbbf24", 2025: "#60a5fa", 2026: "#34d399"}
_KEEP = ["latitude", "longitude", "acq_date", "frp", "confidence"]


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    df["year_month"] = df["acq_date"].dt.strftime("%Y-%m")
    df["year"] = df["acq_date"].dt.year
    df["month_name"] = df["acq_date"].dt.strftime("%b")
    return df


@st.cache_resource(show_spinner=False)
def _read_archive(path: str, mtime: float):
    """The compact archive is parsed once per process, not once per region."""
    return pd.read_csv(path, usecols=lambda c: c in _KEEP)


@st.cache_data(show_spinner=False)
def load_real_history(name: str, min_lat: float, max_lat: float, min_lon: float, max_lon: float,
                      is_karnataka: bool):
    """Archive detections inside the region's box, or None if no archive exists."""
    frames = []
    slim = Path(DATA_RAW_DIR).parent / "archive" / "fires_presets.csv.gz"
    if slim.exists():
        # Compact archive committed with the app (scripts/build_deploy_archive.py);
        # the multi-hundred-MB raw FIRMS files are not deployed.
        df = _read_archive(str(slim), slim.stat().st_mtime)
        df = df[df["latitude"].between(min_lat, max_lat) & df["longitude"].between(min_lon, max_lon)].copy()
        return _finish(df) if len(df) else None
    if is_karnataka:
        path = DATA_RAW_DIR / "historical_fires_karnataka.csv"
        if path.exists():
            frames.append(pd.read_csv(path, usecols=lambda c: c in _KEEP))
    else:
        for path in sorted(Path(DATA_RAW_DIR).glob("historical_fires_india_*.csv")):
            df = pd.read_csv(path, usecols=lambda c: c in _KEEP)
            frames.append(df[df["latitude"].between(min_lat, max_lat) &
                             df["longitude"].between(min_lon, max_lon)])
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df = df[df["latitude"].between(min_lat, max_lat) & df["longitude"].between(min_lon, max_lon)]
    return _finish(df) if len(df) else None


@st.cache_data(show_spinner=False)
def generate_sample_history(seed: int = 99):
    """Labelled SAMPLE PREVIEW for Karnataka only, used when its archive
    hasn't been pulled locally. Same seasonal shape as the real data."""
    rng = np.random.default_rng(seed)
    rows = []
    for year in [2023, 2024, 2025, 2026]:
        for month in range(1, 6):
            weight = {1: 0.3, 2: 0.7, 3: 1.0, 4: 0.9, 5: 0.5}[month]
            for _ in range(int(rng.poisson(120 * weight))):
                rows.append({
                    "latitude": rng.uniform(REGION.min_lat, REGION.max_lat),
                    "longitude": rng.uniform(REGION.min_lon, REGION.max_lon),
                    "frp": rng.exponential(15), "confidence": rng.choice(["nominal", "high"], p=[0.6, 0.4]),
                    "acq_date": pd.Timestamp(year=year, month=month, day=int(rng.integers(1, 29))),
                })
    return _finish(pd.DataFrame(rows))


is_ka = region.name == REGION.name
real_df = load_real_history(region.name, region.min_lat, region.max_lat, region.min_lon, region.max_lon, is_ka)
if real_df is not None:
    df, using_real = real_df, True
    st.success(f"Using real data: {len(df):,} satellite-confirmed detections in {region.name}, "
               f"{df['acq_date'].min().date()} to {df['acq_date'].max().date()}")
elif is_ka:
    df, using_real = generate_sample_history(), False
    st.warning("Real historical data not found for Karnataka. Showing a synthetic preview instead.")
else:
    st.markdown(f"""
    <div class="info-box">No historical detections are stored for <b>{region.name}</b> yet.
    Run <code>python scripts/build_india_regions_dataset.py</code> to download the FIRMS archive (2023 to 2026)
    for the Indian states; this page then fills in automatically.</div>
    """, unsafe_allow_html=True)
    st.stop()

# ── year-by-year summary ──
st.markdown('<div class="sec-hdr">Detections per fire season</div>', unsafe_allow_html=True)
year_counts = df["year"].value_counts().sort_index()
cols = st.columns(len(year_counts))
for col, (year, count) in zip(cols, year_counts.items()):
    col.markdown(f'<div class="kpi"><div class="lbl">{year} season</div>'
                 f'<div class="val">{count:,}</div></div>', unsafe_allow_html=True)

# ── monthly seasonal pattern (all years overlaid) ──
st.markdown('<div class="sec-hdr">Seasonal pattern</div>', unsafe_allow_html=True)
month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
monthly = df.groupby(["year", "month_name"]).size().reset_index(name="detections")
fig_season = go.Figure()
for year in sorted(df["year"].unique()):
    sub = monthly[monthly["year"] == year].set_index("month_name").reindex(month_order).fillna(0)
    fig_season.add_trace(go.Bar(x=month_order, y=sub["detections"], name=str(year),
                                 marker_color=COLORS.get(year, "#8a96a6")))
fig_season.update_layout(
    barmode="group", height=260, margin=dict(l=0, r=0, t=10, b=0),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(tickfont={"color": "#c3ccd8"}, gridcolor="#232b36"),
    yaxis=dict(title="Detections", tickfont={"color": "#8a96a6"}, gridcolor="#232b36", title_font={"color": "#8a96a6"}),
    legend=dict(font={"color": "#c3ccd8"}, bgcolor="rgba(0,0,0,0)"),
    font=dict(family="Plus Jakarta Sans, sans-serif"),
)
st.plotly_chart(fig_season, use_container_width=True)
peak = monthly.groupby("month_name")["detections"].sum().reindex(month_order).fillna(0)
st.caption(f"In {region.name}, detections peak in {peak.idxmax()} across the {df['year'].nunique()} seasons "
           f"({int(peak.max()):,} of {int(peak.sum()):,} detections in Jan to Jun).")

# ── animated map, month by month ──
st.markdown('<div class="sec-hdr">Animated detection map</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def build_map(name: str, _df: pd.DataFrame, min_lat, max_lat, min_lon, max_lon, real: bool):
    months = sorted(_df["year_month"].unique())
    by_month = {ym: g for ym, g in _df.groupby("year_month")}

    def trace(ym):
        sub = by_month[ym]
        return go.Scattermap(lat=sub["latitude"], lon=sub["longitude"], mode="markers",
                             marker=dict(size=7, color="#ff6b4a", opacity=0.75))

    fig = go.Figure(data=[trace(months[0])], frames=[go.Frame(data=[trace(ym)], name=ym) for ym in months])
    reg = type("R", (), dict(min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon))
    fig.update_layout(
        map=dict(style="carto-darkmatter", zoom=_zoom_for(reg),
                  center={"lat": (min_lat + max_lat) / 2, "lon": (min_lon + max_lon) / 2}),
        height=520, margin=dict(l=0, r=0, t=10, b=60), paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif"),
        updatemenus=[{"type": "buttons", "showactive": False, "x": 0.0, "y": -0.08, "xanchor": "left",
                      "bgcolor": "#11151b", "bordercolor": "#303a48", "font": {"color": "#e8edf3"},
                      "buttons": [
                          {"label": "Play", "method": "animate",
                           "args": [None, {"frame": {"duration": 700, "redraw": True}, "fromcurrent": True,
                                            "transition": {"duration": 200}}]},
                          {"label": "Pause", "method": "animate",
                           "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
                      ]}],
        sliders=[{"active": 0, "x": 0.1, "len": 0.9, "y": -0.06, "bgcolor": "#11151b", "bordercolor": "#303a48",
                  "font": {"color": "#8a96a6", "size": 10},
                  "currentvalue": {"prefix": "Month: ", "font": {"color": "#e8edf3", "size": 13}},
                  "steps": [{"label": ym, "method": "animate",
                             "args": [[ym], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]}
                            for ym in months]}],
    )
    return fig, len(months)


fig_map, n_months = build_map(region.name, df[["latitude", "longitude", "year_month"]],
                              region.min_lat, region.max_lat, region.min_lon, region.max_lon, using_real)
st.plotly_chart(fig_map, use_container_width=True)
st.caption(("Real" if using_real else "Synthetic preview") +
           f": {n_months} months animated, {len(df):,} total detections plotted.")
