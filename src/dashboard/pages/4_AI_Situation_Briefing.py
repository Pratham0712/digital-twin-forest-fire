"""
AI Situation Briefing - NEW feature page. Auto-generates a plain-English
situation report for any zone, entirely template-based (no external LLM API
call) so it works offline, has zero cost, and never fails mid-demo due to a
network/API issue. References the zone's real FWI components, weather, and
(when available) the real historical detection record for added context —
e.g. "similar to N% of days in the 2023-2025 record" — which is a genuinely
data-grounded statement, not a generic canned line.
"""
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.append(str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard.dashboard_common import (
    set_page, build_sidebar, ensure_twin, render_header, SYSTEM, DATA_RAW_DIR,
)

set_page("AI Situation Briefing", "📋")

offline, scenario, region, force_refresh = build_sidebar()
twin = ensure_twin(offline, scenario, region, force_refresh)
snap = twin.current_snapshot
summary = twin.get_summary()

render_header(summary, offline, "AI Situation Briefing")
st.caption(
    "Every briefing below is generated from this session's live risk scores and real FWI "
    "components — template-based, not a hosted LLM call, so it works with zero internet "
    "dependency and never fails mid-demo."
)


@st.cache_data(show_spinner=False)
def historical_percentile(fwi_value: float):
    """Compares a given FWI value against the real historical record, if
    available, for a genuinely data-grounded contextual statement."""
    path = DATA_RAW_DIR / "historical_fires_karnataka.csv"
    fwi_ref_path = Path(__file__).resolve().parents[3] / "data" / "processed" / "real_training_data.csv"
    if not fwi_ref_path.exists():
        return None
    try:
        ref = pd.read_csv(fwi_ref_path, usecols=["fwi"])
        pct = (ref["fwi"] < fwi_value).mean() * 100
        return pct
    except Exception:
        return None


def generate_briefing(zone_row: pd.Series, risk_score: float, severity: str) -> str:
    zone_id = zone_row["zone_id"]
    fwi = zone_row.get("fwi", np.nan)
    temp = zone_row.get("wx_temperature_c", np.nan)
    humidity = zone_row.get("wx_humidity_pct", np.nan)
    wind = zone_row.get("wx_wind_speed_ms", np.nan)
    active_fire = bool(zone_row.get("active_fire_nearby", False))
    lat, lon = zone_row.get("latitude"), zone_row.get("longitude")

    sev_phrase = {
        "EXTREME": "an EXTREME wildfire risk situation",
        "HIGH": "a HIGH wildfire risk condition",
        "MODERATE": "a MODERATE, worth-monitoring risk level",
        "LOW": "a LOW risk, routine condition",
    }.get(severity, "an assessed risk condition")

    lines = []
    lines.append(
        f"<b>Zone {zone_id}</b> ({lat:.3f}°N, {lon:.3f}°E) is currently assessed at "
        f"<b>{risk_score:.0%} risk</b> — {sev_phrase}, against the "
        f"{SYSTEM.alert_threshold_pct:.0f}% operational alert threshold."
    )

    if not np.isnan(fwi):
        fwi_desc = "extreme" if fwi > 60 else ("elevated" if fwi > 35 else "moderate" if fwi > 15 else "low")
        lines.append(
            f"The Fire Weather Index reads {fwi:.1f} ({fwi_desc} fire-weather severity on the "
            f"Canadian FWI scale)."
        )

    weather_bits = []
    if not np.isnan(temp):
        weather_bits.append(f"{temp:.0f}°C air temperature")
    if not np.isnan(humidity):
        weather_bits.append(f"{humidity:.0f}% relative humidity")
    if not np.isnan(wind):
        weather_bits.append(f"{wind:.1f} m/s wind speed")
    if weather_bits:
        lines.append("Current conditions: " + ", ".join(weather_bits) + ".")

    if active_fire:
        lines.append("⚠️ An active satellite-confirmed fire hotspot has been detected within this zone.")

    if not np.isnan(fwi):
        pct = historical_percentile(fwi)
        if pct is not None:
            lines.append(
                f"For context: this FWI level exceeds approximately {pct:.0f}% of all "
                f"zone-days recorded in the real 2023–2025 historical dataset."
            )

    if severity in ("EXTREME", "HIGH"):
        lines.append(
            "<b>Recommended action:</b> increase monitoring frequency for this zone; "
            "consider pre-positioning suppression resources if multiple adjacent zones "
            "show similar severity."
        )
    else:
        lines.append("<b>Recommended action:</b> routine monitoring — no immediate action required.")

    return " ".join(lines)


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

    briefing_html = generate_briefing(row, risk, severity)
    st.markdown(f'<div class="briefing-card"><h4>Situation Briefing — {chosen}</h4>{briefing_html}</div>',
                unsafe_allow_html=True)

    plain_text = briefing_html.replace("<b>", "").replace("</b>", "")
    st.download_button("⬇ Download as text", plain_text,
                        file_name=f"briefing_{chosen}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.txt")

else:
    extreme_zones = [a for a in snap.alerts if a.severity == "EXTREME"]
    if not extreme_zones:
        st.info("No zones currently classified EXTREME — nothing to batch-report on this cycle.")
    else:
        st.write(f"Generating briefings for **{len(extreme_zones)} EXTREME zones**...")
        full_report = []
        for a in extreme_zones[:25]:
            row = processed[processed["zone_id"] == a.zone_id].iloc[0]
            briefing_html = generate_briefing(row, a.risk_score, a.severity)
            st.markdown(f'<div class="briefing-card"><h4>{a.zone_id}</h4>{briefing_html}</div>',
                        unsafe_allow_html=True)
            full_report.append(f"=== {a.zone_id} ===\n{briefing_html}".replace("<b>", "").replace("</b>", ""))
        if len(extreme_zones) > 25:
            st.caption(f"+ {len(extreme_zones) - 25} more EXTREME zones not shown (batch limit 25)")
        st.download_button("⬇ Download full batch report", "\n\n".join(full_report),
                            file_name=f"batch_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.txt")
