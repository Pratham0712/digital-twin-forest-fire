"""
AI Situation Briefing - auto-generates a situation report for any zone,
entirely template-based (no external LLM API call) so it works offline, has
zero cost, and never fails mid-demo due to a network/API issue. References
the zone's real FWI components, weather, and (when available) the real
historical detection record for added context - e.g. "similar to N% of days
in the 2023-2025 record" - which is a genuinely data-grounded statement, not
a generic canned line.

Rendered as a structured card (severity banner, gauge, wind compass, stat
tiles, bulleted factors/actions) rather than one wall-of-text paragraph.
"""
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.append(str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.utils.timezone import utc_to_ist
from src.dashboard.dashboard_common import (
    set_page, build_sidebar, ensure_twin, render_header, render_wind_compass,
    SEV_COLOR, SYSTEM, DATA_RAW_DIR, log_action,
)

from src.auth.auth_gate import require_login

set_page("AI Situation Briefing")
require_login()

offline, scenario, region, force_refresh = build_sidebar()
twin = ensure_twin(offline, scenario, region, force_refresh)
snap = twin.current_snapshot
summary = twin.get_summary()

render_header(summary, offline, "AI Situation Briefing", region=twin.region)
st.caption(
    "Auto-generated from this session's live risk scores and real Fire Weather Index "
    "components for the selected zone."
)


def historical_percentile(fwi_value: float):
    """Compares an FWI value against the real historical record (a 101-point
    quantile table, so no large file is read while the page renders)."""
    from src.ml_models.fwi_reference import fwi_percentile
    return fwi_percentile(fwi_value)


def build_briefing_data(zone_row: pd.Series, risk_score: float, severity: str) -> dict:
    """Returns a plain dict of every fact the card needs, kept separate from
    rendering so the same data can drive both the visual card and the plain-
    text download without re-deriving anything twice."""
    fwi = zone_row.get("fwi", np.nan)
    temp = zone_row.get("wx_temperature_c", np.nan)
    humidity = zone_row.get("wx_humidity_pct", np.nan)
    wind_speed = zone_row.get("wx_wind_speed_ms", np.nan)
    wind_deg = zone_row.get("wx_wind_deg", np.nan)
    active_fire = bool(zone_row.get("active_fire_nearby", False))

    fwi_desc = None
    if not np.isnan(fwi):
        fwi_desc = "extreme" if fwi > 60 else ("elevated" if fwi > 35 else "moderate" if fwi > 15 else "low")

    key_factors = []
    if not np.isnan(fwi):
        key_factors.append(("FWI", f"Fire Weather Index {fwi:.1f} — {fwi_desc} severity on the Canadian FWI scale"))
    if not np.isnan(humidity) and humidity < 25:
        key_factors.append(("Humidity", f"Low relative humidity ({humidity:.0f}%) — vegetation dries quickly"))
    if not np.isnan(wind_speed) and wind_speed > 8:
        key_factors.append(("Wind", f"Strong wind ({wind_speed:.1f} m/s) — accelerates spread once ignited"))
    if not np.isnan(temp) and temp > 38:
        key_factors.append(("Heat", f"High air temperature ({temp:.0f}°C)"))
    if active_fire:
        key_factors.append(("Satellite", "Active satellite-confirmed fire hotspot detected within this zone"))
    if not key_factors:
        key_factors.append(("Note", "No individual factor stands out — risk is driven by the combined score"))

    hist_pct = historical_percentile(fwi) if not np.isnan(fwi) else None

    if severity in ("EXTREME", "HIGH"):
        actions = [
            "Increase monitoring frequency for this zone and its immediate neighbours",
            "Pre-position suppression resources if adjacent zones show similar severity",
            "Notify local fire authorities if this persists into the next refresh cycle",
        ]
    elif severity == "MODERATE":
        actions = [
            "Keep this zone in the standard monitoring rotation",
            "Re-check after the next scheduled refresh for any upward trend",
        ]
    else:
        actions = ["Routine monitoring — no action required at this time"]

    return dict(
        zone_id=zone_row["zone_id"], lat=zone_row.get("latitude"), lon=zone_row.get("longitude"),
        risk_score=risk_score, severity=severity, fwi=fwi, fwi_desc=fwi_desc,
        temp=temp, humidity=humidity, wind_speed=wind_speed, wind_deg=wind_deg,
        active_fire=active_fire, key_factors=key_factors, hist_pct=hist_pct, actions=actions,
    )


def render_briefing_card(d: dict):
    c = SEV_COLOR.get(d["severity"], "#8a96a6")
    sev_icon = d["severity"]

    st.markdown(f"""
    <div class="scenario-banner" style="border-color:{c}55; background:{c}14; color:{c}; --pulse-color:{c}55;">
      <div class="icon">{sev_icon}</div>
      <div class="txt" style="color:#e8edf3">Zone <b>{d['zone_id']}</b>
        ({d['lat']:.3f}°N, {d['lon']:.3f}°E) &nbsp;·&nbsp;
        <b style="color:{c}">{d['risk_score']:.0%} risk</b> against the
        {SYSTEM.alert_threshold_pct:.0f}% alert threshold</div>
    </div>
    """, unsafe_allow_html=True)

    col_gauge, col_compass, col_tiles = st.columns([1, 1, 2], gap="medium")
    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=d["risk_score"] * 100,
            number={"suffix": "%", "font": {"color": "#e8edf3", "size": 24}},
            gauge={"axis": {"range": [0, 100], "tickcolor": "#8a96a6", "tickfont": {"color": "#8a96a6", "size": 8}},
                   "bar": {"color": c, "thickness": 0.25}, "bgcolor": "#11151b", "borderwidth": 0,
                   "steps": [{"range": [0, 40], "color": "#1c2128"}, {"range": [40, 70], "color": "#2d2200"},
                             {"range": [70, 100], "color": "#3d1a1a"}],
                   "threshold": {"line": {"color": "#ff6b4a", "width": 2}, "thickness": 0.8,
                                 "value": SYSTEM.alert_threshold_pct}},
        ))
        fig.update_layout(height=140, margin=dict(l=10, r=10, t=10, b=0),
                           paper_bgcolor="rgba(0,0,0,0)", font={"color": "#8a96a6"})
        # Explicit key: batch mode renders one gauge per zone in a loop, and
        # without a key Streamlit can't tell otherwise-identical-looking
        # plotly_chart calls apart, raising StreamlitDuplicateElementId.
        st.plotly_chart(fig, use_container_width=True, key=f"briefing_gauge_{d['zone_id']}")
    with col_compass:
        if not np.isnan(d["wind_deg"]) and not np.isnan(d["wind_speed"]):
            render_wind_compass(float(d["wind_deg"]), float(d["wind_speed"]), label="Wind")
    with col_tiles:
        t1, t2, t3 = st.columns(3)
        fwi_val = f"{d['fwi']:.1f}" if not np.isnan(d["fwi"]) else "—"
        temp_val = f"{d['temp']:.0f}°C" if not np.isnan(d["temp"]) else "—"
        hum_val = f"{d['humidity']:.0f}%" if not np.isnan(d["humidity"]) else "—"
        for col, lbl, val in [(t1, "FWI", fwi_val), (t2, "Temperature", temp_val), (t3, "Humidity", hum_val)]:
            col.markdown(f'<div class="kpi"><div class="lbl">{lbl}</div><div class="val">{val}</div></div>',
                          unsafe_allow_html=True)
        if d["hist_pct"] is not None:
            st.markdown(
                f'<div class="info-box" style="margin-top:10px;">This FWI level exceeds approximately '
                f'<b>{d["hist_pct"]:.0f}%</b> of all zone-days in the real 2023–2025 historical record.</div>',
                unsafe_allow_html=True,
            )

    fc1, fc2 = st.columns(2, gap="medium")
    with fc1:
        st.markdown("**Key factors**")
        for tag, text in d["key_factors"]:
            st.markdown(f"- **{tag}.** {text}")
    with fc2:
        st.markdown("**Recommended actions**")
        for action in d["actions"]:
            st.markdown(f"- {action}")


def briefing_data_to_text(d: dict) -> str:
    lines = [
        f"=== Zone {d['zone_id']} — {d['severity']} ({d['risk_score']:.0%} risk) ===",
        f"Location: {d['lat']:.3f} N, {d['lon']:.3f} E",
    ]
    if not np.isnan(d["fwi"]):
        lines.append(f"FWI: {d['fwi']:.1f} ({d['fwi_desc']})")
    if not np.isnan(d["temp"]):
        lines.append(f"Temperature: {d['temp']:.0f}C")
    if not np.isnan(d["humidity"]):
        lines.append(f"Humidity: {d['humidity']:.0f}%")
    if not np.isnan(d["wind_speed"]):
        lines.append(f"Wind: {d['wind_speed']:.1f} m/s")
    if d["hist_pct"] is not None:
        lines.append(f"Historical context: exceeds ~{d['hist_pct']:.0f}% of 2023-2025 zone-days")
    lines.append("Key factors:")
    lines += [f"  - {text}" for _, text in d["key_factors"]]
    lines.append("Recommended actions:")
    lines += [f"  - {a}" for a in d["actions"]]
    return "\n".join(lines)


processed = snap.processed_grid.reset_index(drop=True)
risk_scores = snap.risk_scores
alerts_by_zone = {a.zone_id: a for a in snap.alerts}

st.markdown('<div class="sec-hdr">Select a zone</div>', unsafe_allow_html=True)
mode = st.radio("Briefing scope", ["Single zone", "All EXTREME zones (batch report)"], horizontal=True)

if mode == "Single zone":
    zone_ids = processed["zone_id"].tolist()
    default_idx = 0
    if snap.alerts:
        try:
            default_idx = zone_ids.index(snap.alerts[0].zone_id)
        except ValueError:
            pass
    chosen = st.selectbox("Zone", zone_ids, index=default_idx)
    row = processed[processed["zone_id"] == chosen].iloc[0]
    idx = processed[processed["zone_id"] == chosen].index[0]
    risk = float(risk_scores[idx])
    severity = alerts_by_zone[chosen].severity if chosen in alerts_by_zone else "LOW"

    data = build_briefing_data(row, risk, severity)
    render_briefing_card(data)

    st.download_button("Download as text", briefing_data_to_text(data),
                        file_name=f"briefing_{chosen}_{utc_to_ist(datetime.now(timezone.utc)).strftime('%Y%m%d_%H%M')}_IST.txt",
                        on_click=log_action, args=("report", f"Downloaded briefing for {chosen} "
                                                   f"({severity}, risk {risk:.0%})", twin.region.name))

else:
    extreme_zones = [a for a in snap.alerts if a.severity == "EXTREME"]
    if not extreme_zones:
        st.info("No zones currently classified EXTREME — nothing to batch-report on this cycle.")
    else:
        st.write(f"Generating briefings for **{len(extreme_zones)} EXTREME zones**...")
        full_report = []
        for a in extreme_zones[:25]:
            row = processed[processed["zone_id"] == a.zone_id].iloc[0]
            data = build_briefing_data(row, a.risk_score, a.severity)
            render_briefing_card(data)
            full_report.append(briefing_data_to_text(data))
            st.markdown("---")
        if len(extreme_zones) > 25:
            st.caption(f"+ {len(extreme_zones) - 25} more EXTREME zones not shown (batch limit 25)")
        st.download_button("Download full batch report", "\n\n".join(full_report),
                            file_name=f"batch_report_{utc_to_ist(datetime.now(timezone.utc)).strftime('%Y%m%d_%H%M')}_IST.txt",
                            on_click=log_action, args=("report", f"Downloaded batch report "
                                                       f"({len(extreme_zones)} EXTREME zones)", twin.region.name))
