"""
Model Insights - reuses the existing model-performance charts (radar,
bar comparison, table) and adds NEW per-zone SHAP explainability: a
waterfall chart showing exactly which features pushed a specific zone's
risk score up or down, plus a global feature-importance summary across
the current snapshot. This directly strengthens both the demo ("why did
the model flag this zone") and the paper (model transparency, not just
a black-box score).
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.dashboard.dashboard_common import (
    set_page, build_sidebar, ensure_twin, render_header, render_model_performance,
    load_ml_model,
)
from src.data_processing.feature_engineering import DataProcessor

from src.auth.auth_gate import require_login

set_page("Model Insights", "🧠")
require_login()

# Dark matplotlib theme to match the dashboard, applied once for all SHAP plots.
plt.rcParams.update({
    "figure.facecolor": "#0b0d0f", "axes.facecolor": "#0b0d0f",
    "savefig.facecolor": "#0b0d0f", "text.color": "#c9d1d9",
    "axes.labelcolor": "#c9d1d9", "xtick.color": "#8b949e", "ytick.color": "#8b949e",
    "axes.edgecolor": "#30363d",
})

offline, scenario, region, force_refresh = build_sidebar()
twin = ensure_twin(offline, scenario, region, force_refresh)
snap = twin.current_snapshot
summary = twin.get_summary()

render_header(summary, offline, "Model Insights", region=twin.region)

st.markdown('<div class="sec-hdr">Real-data model comparison</div>', unsafe_allow_html=True)
render_model_performance()

st.markdown('<div class="sec-hdr">Per-zone explainability (SHAP)</div>', unsafe_allow_html=True)
st.caption(
    "Shows exactly which features pushed a specific zone's risk score up or down, using SHAP "
    "(SHapley Additive exPlanations) against the real-trained XGBoost model — not a black-box "
    "score, a fully attributable one."
)

ml_model = load_ml_model()
if ml_model is None:
    st.warning("No real-trained model found (models/xgboost_real.json). Run "
               "`python src/ml_models/train_real.py` first.")
else:
    processed = snap.processed_grid.reset_index(drop=True)
    processor = DataProcessor()
    X, _ = processor.get_feature_matrix(processed, real=True)

    zone_ids = processed["zone_id"].tolist()
    default_idx = 0
    if snap.alerts:
        try:
            default_idx = zone_ids.index(snap.alerts[0].zone_id)
        except ValueError:
            pass
    chosen = st.selectbox("Zone to explain", zone_ids, index=default_idx, key="shap_zone")
    row_idx = processed[processed["zone_id"] == chosen].index[0]

    with st.spinner("Computing SHAP values..."):
        explainer = shap.TreeExplainer(ml_model.model)
        sv_all = explainer(X)

    risk_score = float(snap.risk_scores[row_idx])
    col_wf, col_summary = st.columns([2, 1])

    with col_wf:
        st.markdown(f"**Why {chosen} scored {risk_score:.0%} risk**")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        plt.sca(ax)
        shap.plots.waterfall(sv_all[row_idx], show=False, max_display=10)
        fig.patch.set_facecolor("#0b0d0f")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_summary:
        st.markdown("**Global feature importance**")
        st.caption("Mean |SHAP value| across all zones in this snapshot")
        mean_abs = np.abs(sv_all.values).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": X.columns, "importance": mean_abs,
        }).sort_values("importance", ascending=True).tail(10)

        import plotly.graph_objects as go
        fig_imp = go.Figure(go.Bar(
            x=importance_df["importance"], y=importance_df["feature"], orientation="h",
            marker_color="#f78166",
        ))
        fig_imp.update_layout(
            height=380, margin=dict(l=0, r=10, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickfont={"color": "#8b949e"}, gridcolor="#21262d"),
            yaxis=dict(tickfont={"color": "#c9d1d9", "size": 10}),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("""
    <div class="info-box">
    Red bars push risk higher, blue bars push it lower. The base value (E[f(X)]) is the
    model's average prediction across all training data; each feature's SHAP value shows
    exactly how much it moved this specific zone's prediction away from that baseline.
    </div>
    """, unsafe_allow_html=True)
