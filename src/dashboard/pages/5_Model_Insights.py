"""
Model Insights - real-data model comparison, regional model coverage, and
per-zone explainability. Attributions come from XGBoost's exact TreeSHAP
(`pred_contribs`), computed once per snapshot, so the page loads fast and
does not need the heavy `shap`/matplotlib imports.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.dashboard_common import (
    set_page, build_sidebar, ensure_twin, render_header, render_model_performance,
    load_ml_model, REGION_PRESETS,
)
from src.data_processing.feature_engineering import DataProcessor
from src.ml_models.explain import contributions, sigmoid
from src.ml_models.model_registry import choose_for_region, india_state_metrics

from src.auth.auth_gate import require_login

set_page("Model Insights")
require_login()

offline, scenario, region, force_refresh = build_sidebar()
twin = ensure_twin(offline, scenario, region, force_refresh)
snap = twin.current_snapshot
summary = twin.get_summary()

render_header(summary, offline, "Model Insights", region=twin.region)

st.markdown('<div class="sec-hdr">Real-data model comparison</div>', unsafe_allow_html=True)
render_model_performance()


# ── regional coverage ──────────────────────────────────────────────────────── #

@st.cache_data(show_spinner=False)
def _coverage_table(bust: tuple):
    rows = []
    for name, r in REGION_PRESETS.items():
        choice = choose_for_region(r)
        m = india_state_metrics(r.name) if choice.model_file != "xgboost_real.json" else None
        rows.append({
            "Region": r.name, "Model": choice.label, "Grid": f"{r.grid_resolution_deg}°",
            "Held-out 2025 AUC-ROC": None if m is None else round(float(m["auc_roc"]), 3),
        })
    return pd.DataFrame(rows)


st.markdown('<div class="sec-hdr">Model used per region</div>', unsafe_allow_html=True)
_bust = tuple((Path(p).stat().st_mtime if Path(p).exists() else 0)
              for p in (Path(__file__).resolve().parents[3] / "models" / "xgboost_india.json",
                        Path(__file__).resolve().parents[3] / "data" / "processed" / "india_model_by_state.csv"))
cov = _coverage_table(_bust)
st.dataframe(cov, use_container_width=True, hide_index=True,
             column_config={"Held-out 2025 AUC-ROC": st.column_config.NumberColumn(format="%.3f")})
st.caption("AUC-ROC is measured on the 2025 fire season, which the model never saw during training. "
           "The Karnataka figures are in the comparison above.")


# ── evaluation beyond accuracy ─────────────────────────────────────────────── #

_PROC = Path(__file__).resolve().parents[3] / "data" / "processed"


@st.cache_data(show_spinner=False)
def _eval_csv(name: str, mtime: float):
    path = _PROC / name
    return pd.read_csv(path) if path.exists() else None


def _load(name: str):
    path = _PROC / name
    return _eval_csv(name, path.stat().st_mtime if path.exists() else 0.0)


_PLOT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
             font=dict(family="Plus Jakarta Sans, sans-serif", color="#c3ccd8"),
             margin=dict(l=0, r=10, t=10, b=0), height=330,
             xaxis=dict(gridcolor="#232b36", zerolinecolor="#303a48"),
             yaxis=dict(gridcolor="#232b36", zerolinecolor="#303a48"))
_SEASON_COLOR = {"2025": "#ff6b35", "2026": "#60a5fa"}

st.markdown('<div class="sec-hdr">Evaluation beyond accuracy</div>', unsafe_allow_html=True)
st.caption("Fires are rare (a few percent of zone-days), so plain accuracy flatters any model. "
           "These views measure what an early-warning system is for: how many real fires it "
           "catches for how many alerts, and how that compares with simple alternatives.")

_summary, _lift, _pr = _load("eval_summary.csv"), _load("eval_lift_table.csv"), _load("eval_pr_curve.csv")
_persist, _comp = _load("eval_persistence.csv"), _load("eval_comparable.csv")
_spread, _cmp26 = _load("eval_spread_validation.csv"), _load("model_comparison_2026.csv")

if _summary is None:
    st.markdown('''<div class="info-box">Run <code>python scripts/run_evaluation_extras.py</code> to generate
    the lift, precision-recall and baseline tables for the saved Karnataka model.</div>''', unsafe_allow_html=True)
else:
    t_lift, t_pr, t_rules, t_seasons, t_cmp, t_spread = st.tabs(
        ["Lift", "Precision-recall", "Versus simple rules", "Second season", "Comparing regions", "Fire-spread check"])

    with t_lift:
        for _, r in _summary.iterrows():
            st.markdown(f"**{r['season']} season** - fire rate {r['fire_rate']:.1%}. Flagging the riskiest 10% of "
                        f"zone-days catches **{r['recall_top10pct']:.0%}** of fires "
                        f"(**{r['lift_top10pct']:.1f}x** better than random). At the alert threshold, precision is "
                        f"**{r['alert_precision_lift']:.1f}x** the base rate.")
        fig = go.Figure()
        for season, g in _lift.groupby("season"):
            fig.add_trace(go.Scatter(x=g["flagged_share"] * 100, y=g["recall"] * 100, mode="lines+markers",
                                     name=f"{season} season", line=dict(color=_SEASON_COLOR.get(str(season), "#8a96a6"))))
        fig.add_trace(go.Scatter(x=[0, 30], y=[0, 30], mode="lines", name="Random", line=dict(color="#5b6675", dash="dash")))
        fig.update_layout(**_PLOT, xaxis_title="Share of zone-days flagged (%)", yaxis_title="Fires captured (%)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(_lift.drop(columns=["region"], errors="ignore"), use_container_width=True, hide_index=True)

    with t_pr:
        fig = go.Figure()
        for season, g in _pr.groupby("season"):
            fig.add_trace(go.Scatter(x=g["recall"], y=g["precision"], mode="lines", name=f"{season} season",
                                     line=dict(color=_SEASON_COLOR.get(str(season), "#8a96a6"))))
        for _, r in _summary.iterrows():
            fig.add_hline(y=r["fire_rate"], line=dict(color="#5b6675", dash="dot"),
                          annotation_text=f"random ({r['season']})", annotation_font_color="#8a96a6")
        fig.update_layout(**_PLOT, xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Precision stays modest because most zone-days with elevated risk do not record a fire on that "
                   "particular day. The dotted line is what random flagging would achieve.")

    with t_rules:
        if _persist is None:
            st.info("Not available - re-run scripts/run_evaluation_extras.py.")
        else:
            st.caption("Each rule flags zone-days from recent fire history alone. The last two columns show the "
                       "model when it is allowed to flag the same number of zone-days.")
            show = _persist.drop(columns=["region"], errors="ignore").rename(columns={
                "method": "Rule", "flagged_share": "Flagged share", "recall": "Rule recall", "precision": "Rule precision",
                "lift_vs_random": "Rule lift", "model_recall_same_share": "Model recall",
                "model_precision_same_share": "Model precision", "season": "Season"})
            st.dataframe(show, use_container_width=True, hide_index=True)

    with t_seasons:
        if _cmp26 is None:
            st.markdown('''<div class="info-box">No second hold-out season yet. Run
            <code>python scripts/add_season.py --year 2026</code>, then <code>python src/ml_models/train_real.py</code>.</div>''',
                        unsafe_allow_html=True)
        else:
            st.caption("2026 was never used for training, tuning or choosing thresholds. Same models, same thresholds.")
            st.dataframe(_cmp26, use_container_width=True, hide_index=False)
            _s26 = _load("india_model_by_state_2026.csv")
            if _s26 is not None:
                st.markdown("**Other states, 2026 season**")
                st.dataframe(_s26[["region", "test_fire_days", "auc_roc", "avg_precision", "recall"]],
                             use_container_width=True, hide_index=True)

    with t_cmp:
        st.caption("Regions use different grid sizes and have different fire frequencies, so raw precision and "
                   "accuracy are not comparable. Karnataka is shown on its own 0.1 degree grid and re-scored on "
                   "0.5 degree blocks (the states' grid). The last column divides average precision by the fire rate.")
        if _comp is not None:
            st.dataframe(_comp.drop(columns=["region"], errors="ignore"), use_container_width=True, hide_index=True)
        _ind = _load("india_model_by_state.csv")
        if _ind is not None and "ap_over_base_rate" in _ind.columns:
            st.markdown("**Other states (0.5 degree grid), 2025 season**")
            st.dataframe(_ind[["region", "positive_rate", "auc_roc", "ap_over_base_rate", "lift_top10pct",
                               "recall_top10pct"]].rename(columns={
                "positive_rate": "Fire rate", "auc_roc": "AUC-ROC", "ap_over_base_rate": "Avg precision / fire rate",
                "lift_top10pct": "Lift at top 10%", "recall_top10pct": "Recall at top 10%"}),
                use_container_width=True, hide_index=True)

    with t_spread:
        if _spread is None:
            st.markdown('''<div class="info-box">Run <code>python scripts/validate_spread.py</code> to back-test the
            spread simulator against real satellite detections.</div>''', unsafe_allow_html=True)
        else:
            st.caption("Seeded with one day's real detections and that day's dryness, wind speed and terrain, the "
                       "simulator is scored on the NEW fire cells detected the next day, against two reference predictors. "
                       "No wind direction or fuel map exists in the archive, so runs use random wind direction and uniform fuel.")
            st.dataframe(_spread.drop(columns=["days_scored"], errors="ignore"), use_container_width=True, hide_index=True)


# ── per-zone explainability ────────────────────────────────────────────────── #

st.markdown('<div class="sec-hdr">Per-zone explainability</div>', unsafe_allow_html=True)
st.caption(
    "Exact TreeSHAP attributions from the trained XGBoost model: how much each feature pushed a "
    "zone's fire probability up or down, measured in log-odds relative to the model's average prediction."
)

ml_model = load_ml_model(twin.region)
if ml_model is None:
    st.warning("No trained model file found for this region. Run "
               "`python src/ml_models/train_real.py` (Karnataka) or "
               "`python scripts/build_india_regions_dataset.py --train` (other states).")
    st.stop()


def _attributions(snapshot):
    """One computation per snapshot, kept in session_state."""
    key = ("_attr", id(snapshot))
    if st.session_state.get("_attr_key") != key:
        processed = snapshot.processed_grid.reset_index(drop=True)
        X, _ = DataProcessor().get_feature_matrix(processed, real=True, columns=twin._model_columns())
        contrib, bias = contributions(ml_model.model, X)
        st.session_state["_attr_key"] = key
        st.session_state["_attr_val"] = (processed, X, contrib, bias)
    return st.session_state["_attr_val"]


processed, X, contrib, bias = _attributions(snap)
zone_ids = processed["zone_id"].tolist()
default_idx = 0
if snap.alerts:
    try:
        default_idx = zone_ids.index(snap.alerts[0].zone_id)
    except ValueError:
        pass
if st.session_state.get("shap_zone") not in zone_ids:
    st.session_state["shap_zone"] = zone_ids[default_idx]
chosen = st.selectbox("Zone to explain", zone_ids, key="shap_zone")
row_idx = zone_ids.index(chosen)
risk_score = float(snap.risk_scores[row_idx])
model_prob = float(sigmoid(contrib.iloc[row_idx].sum() + bias[row_idx]))

col_wf, col_summary = st.columns([2, 1], gap="large")
with col_wf:
    st.markdown(f"**Why {chosen} scored {risk_score:.0%} risk**")
    st.caption(f"Model probability {model_prob:.1%}. The dashboard risk score maps that probability "
               "onto the validated risk bands.")
    c = contrib.iloc[row_idx]
    top = c.reindex(c.abs().sort_values(ascending=False).index).head(9)
    rest = c.drop(top.index).sum()
    labels = [f"{f} = {X.iloc[row_idx][f]:.3g}" for f in top.index] + ["other features"]
    vals = list(top.values) + [rest]
    fig = go.Figure(go.Waterfall(
        orientation="h", measure=["relative"] * len(vals), y=labels[::-1], x=vals[::-1],
        increasing=dict(marker=dict(color="#ff6b4a")), decreasing=dict(marker=dict(color="#60a5fa")),
        connector=dict(line=dict(color="#303a48", width=1)),
        text=[f"{v:+.2f}" for v in vals[::-1]], textposition="outside",
    ))
    fig.update_layout(height=420, margin=dict(l=0, r=30, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                      xaxis=dict(title="Contribution (log-odds)", gridcolor="#232b36", zerolinecolor="#303a48",
                                 tickfont={"color": "#8a96a6"}, title_font={"color": "#8a96a6"}),
                      yaxis=dict(tickfont={"color": "#c3ccd8", "size": 11}),
                      font=dict(family="Plus Jakarta Sans, sans-serif"))
    st.plotly_chart(fig, use_container_width=True)

with col_summary:
    st.markdown("**Global feature importance**")
    st.caption("Mean absolute contribution across all zones in this snapshot")
    imp = contrib.abs().mean().sort_values().tail(10)
    fig_imp = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color="#ff6b35"))
    fig_imp.update_layout(height=420, margin=dict(l=0, r=10, t=10, b=0),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(tickfont={"color": "#8a96a6"}, gridcolor="#232b36"),
                          yaxis=dict(tickfont={"color": "#c3ccd8", "size": 10}),
                          font=dict(family="Plus Jakarta Sans, sans-serif"))
    st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("""
<div class="info-box">
Orange bars push the risk up, blue bars push it down. The values add up (with the model's baseline)
to the log-odds of this zone's predicted fire probability, so every score is fully attributable.
</div>
""", unsafe_allow_html=True)
