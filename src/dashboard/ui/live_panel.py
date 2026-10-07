"""Real observations at the selected simulation location, kept visibly apart
from the What-If scenario inputs.

  REAL WEATHER      OpenWeatherMap current weather at the location's lat/lon
  NASA FIRMS        satellite fire detections in a box around the location
  SCENARIO INPUT    the hypothetical values the simulation uses (What-If)

Nothing here feeds the simulation; it only fetches (cached, see
src/data_ingestion/live_point.py) and displays. Refresh buttons force a new
request. In demo / offline mode the APIs are not called.
"""
import html
from typing import Optional

import pandas as pd
import streamlit as st

from src.data_ingestion.live_point import FIRMS_BOX_KM, FIRMS_DAYS, area_hotspots, hotspot_markers, point_weather
from src.dashboard.ui.global_ticker import compass

_TAG = {"live": ("live", "LIVE"), "cached": ("warn", "CACHED"), "error": ("crit", "UNAVAILABLE"),
        "not_configured": ("warn", "NOT CONFIGURED"), "demo": ("warn", "DEMO / OFFLINE")}


def _utc(iso) -> str:
    if not iso:
        return "-"
    return str(iso)[:16].replace("T", " ").replace("+00:00", "").rstrip("Z") + " UTC"


def _card(title: str, mode: str, rows, foot: str, cls: str = "") -> str:
    tcls, ttxt = _TAG.get(mode, ("warn", mode.upper()))
    if cls == "scn":
        tcls, ttxt = "warn", "SCENARIO INPUT"
    body = "".join(f"<span>{html.escape(str(a))}</span><span>{html.escape(str(b))}</span>" for a, b in rows)
    return (f'<div class="lv-card {cls}"><div class="l">{html.escape(title)}<span class="tag {tcls}">{ttxt}</span></div>'
            f'<div class="g2">{body}</div><div class="ft">{html.escape(foot)}</div></div>')


def live_observations(lat: float, lon: float, kp: str, offline: bool):
    """Fetch (cached) real weather + FIRMS detections for the location.
    Returns (weather, weather_status, detections_df, firms_status)."""
    if offline:
        demo = {"mode": "demo", "label": "Demo / offline mode: live APIs not called", "fetched_utc": None, "error": None}
        return None, demo, pd.DataFrame(), {**demo, "n": 0}
    force_w = st.session_state.pop(f"_{kp}_force_w", False)
    force_f = st.session_state.pop(f"_{kp}_force_f", False)
    w, ws = point_weather(lat, lon, force=force_w)
    d, fs = area_hotspots(lat, lon, force=force_f)
    if w:
        st.session_state["_last_point_weather"] = (f"{w.get('place_name') or 'selected point'} "
                                                   f"({lat:.3f}, {lon:.3f}) fetched {_utc(ws.get('fetched_utc'))}")
    return w, ws, d, fs


def map_hotspots(detections: pd.DataFrame, firms_status: dict) -> tuple:
    """(markers, kind, summary) for the simulation maps: real detections only."""
    mode = firms_status.get("mode")
    markers = hotspot_markers(detections)
    if mode in ("live", "cached"):
        n = len(markers)
        summ = f"{n} observed" if n else "none in window"
        return markers, "observed", summ + (" (cached)" if mode == "cached" else "")
    if mode == "not_configured":
        return [], "observed", "key not set"
    if mode == "demo":
        return [], "observed", "demo mode: not queried"
    return [], "observed", "unavailable"


def render_live_conditions(lat: float, lon: float, place: str, kp: str, offline: bool,
                           scenario: Optional[dict] = None, data=None):
    """Cards: REAL WEATHER (OpenWeatherMap) | NASA FIRMS | SCENARIO INPUT."""
    w, ws, d, fs = data if data is not None else live_observations(lat, lon, kp, offline)
    st.markdown('<div class="sec-hdr">Real-world observations at the selected location</div>',
                unsafe_allow_html=True)
    cards = []
    if w:
        rows = [("Temperature", f"{w['temperature_c']:.1f} °C"), ("Humidity", f"{w['humidity_pct']:.0f} %"),
                ("Wind", f"{float(w['wind_speed_ms']):.1f} m/s from {compass(float(w['wind_deg']))} "
                         f"({float(w['wind_deg']):.0f}°)"),
                ("Pressure", f"{w['pressure_hpa']} hPa"),
                ("Conditions", (w.get("weather_description") or w.get("weather_main") or "-").capitalize()),
                ("Clouds", f"{w.get('clouds_pct', 0)} %"), ("Rain (1 h)", f"{float(w.get('precipitation_mm') or 0):.1f} mm")]
        foot = (f"OpenWeatherMap current weather at {lat:.4f}°N, {lon:.4f}°E"
                + (f" ({w['place_name']})" if w.get("place_name") else "")
                + f" · observed {_utc(w.get('observed_at'))} · fetched {_utc(ws.get('fetched_utc'))}")
        if ws.get("mode") == "cached":
            foot += f" · live request failed: {ws.get('error')}"
    else:
        rows = [("Status", ws.get("label", "-"))]
        foot = (ws.get("error") or "") if ws.get("mode") == "error" else (
            "Real weather is shown here when OWM_API_KEY is set and the request succeeds. Nothing is estimated.")
    cards.append(_card("REAL WEATHER · OPENWEATHERMAP", ws.get("mode", "error"), rows, foot))

    n = int(fs.get("n", 0) or 0)
    box = fs.get("box") or {}
    if fs.get("mode") in ("live", "cached"):
        rows = [("Detections", f"{n} in {FIRMS_BOX_KM:.0f} × {FIRMS_BOX_KM:.0f} km, last {FIRMS_DAYS} days"),
                ("Latest observation", _utc(fs.get("latest_acq_utc")) if n else "-"),
                ("Product", fs.get("source", "NASA FIRMS")), ("Fetched", _utc(fs.get("fetched_utc")))]
        foot = ("Observed satellite fire detections (cyan circles on the map, VIIRS 375 m pixel)."
                if n else "No NASA FIRMS fire detections available for this area/time window.")
        if fs.get("mode") == "cached":
            foot += f" Live request failed: {fs.get('error')}"
    else:
        rows = [("Status", fs.get("label", "-"))]
        foot = (fs.get("error") or "") if fs.get("mode") == "error" else "No detections are invented."
    cards.append(_card("NASA FIRMS · SATELLITE FIRE DETECTIONS", fs.get("mode", "error"), rows, foot))

    if scenario:
        rows = [("Temperature", f"{scenario.get('temp_c')} °C"), ("Humidity", f"{scenario.get('humidity_pct')} %"),
                ("Wind", f"{float(scenario.get('wind_speed_ms', 0)):.1f} m/s from "
                         f"{compass(float(scenario.get('wind_from_deg', 225)))} ({float(scenario.get('wind_from_deg', 225)):.0f}°)"),
                ("Seeded hotspots", f"{scenario.get('n_hotspots', '-')} (synthetic)")]
        cards.append(_card("WHAT-IF SCENARIO", "", rows,
                           "Hypothetical values chosen by you. They drive the simulation; they are not observations.",
                           cls="scn"))
    st.markdown(f'<div class="lv-grid">{"".join(cards)}</div>', unsafe_allow_html=True)
    if not offline:
        b1, b2, _ = st.columns([1, 1, 3])
        if b1.button("Refresh weather", key=f"{kp}_lv_rw", use_container_width=True):
            st.session_state[f"_{kp}_force_w"] = True
            st.rerun()
        if b2.button("Refresh FIRMS", key=f"{kp}_lv_rf", use_container_width=True):
            st.session_state[f"_{kp}_force_f"] = True
            st.rerun()
