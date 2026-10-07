"""Command Center building blocks: hero artwork, live status strip, wildfire
alert panel, Explore cards and the data / system status grid.

Honesty rules: the hero is the supplied illustrative artwork (captioned as
such; its painted numbers are not data). The alert panel shows either the
latest fire-spread run of this session (labelled SIMULATED, with its source
scenario) or the model's highest-risk zone (labelled MODEL PREDICTION and
DEMO DATA / LIVE DATA); it never claims an observed fire it does not have.
"""
import html
import os

import streamlit as st

from src.dashboard.ui.assets import ALERT_THUMB, HERO_IMAGE, HERO_URL, data_uri
from src.dashboard.ui.global_ticker import (compass, data_badge, feed_status, feed_text, latest_spread, maps_item,
                                           regional_wind)

SPREAD_PAGE = "pages/2_Spread_Simulation.py"

ICONS = {
    "grid": '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/>'
            '<rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
    "bell": '<path d="M6 16V11a6 6 0 0 1 12 0v5l1.5 2h-15z"/><path d="M10 20a2 2 0 0 0 4 0"/>',
    "flame": '<path d="M12 3c1 3 4 4.5 4 8.5a4 4 0 0 1-8 0c0-1.6.8-2.7 1.6-3.6.3 1.4 1.2 2.1 1.9 2.1-.6-2.5.5-5 .5-7z"/>',
    "alert": '<path d="M12 4l9 16H3z"/><path d="M12 10v4M12 17v.5"/>',
    "eye": '<path d="M2 12s3.6-6 10-6 10 6 10 6-3.6 6-10 6S2 12 2 12z"/><circle cx="12" cy="12" r="2.6"/>',
    "gauge": '<path d="M4 18a8 8 0 1 1 16 0"/><path d="M12 18l4-6"/>',
    "avg": '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    "sliders": '<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12"/><circle cx="16" cy="6" r="2"/>'
               '<circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>',
    "clock": '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>',
    "spark": '<path d="M12 3l1.8 4.6L18.5 9l-4.7 1.6L12 15l-1.8-4.4L5.5 9l4.7-1.4z"/>',
    "shield": '<path d="M12 3l7 3v5c0 4.5-3 8-7 10-4-2-7-5.5-7-10V6z"/><path d="M9 12l2 2 4-4"/>',
}


def svg_icon(name: str) -> str:
    return ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICONS.get(name, "")}</svg>')


# ── hero ───────────────────────────────────────────────────────────────────── #

def render_dashboard_hero():
    """The supplied artwork, full width, natural aspect ratio (no crop, no
    stretch), rounded with a subtle border. Rendered as a plain <img> from
    Streamlit's static file server: no st.image toolbar, so no fullscreen /
    expand button appears over it. Falls back to an inline copy if static
    serving is switched off."""
    if not HERO_IMAGE.exists():
        return
    src = HERO_URL if _static_serving_enabled() else data_uri(HERO_IMAGE)
    st.markdown(
        f'<div class="cc-hero"><img src="{src}" alt="Forest Fire Digital Twin: from satellite observations to '
        f'predictive action (illustrative artwork)" draggable="false" loading="eager" decoding="async"/></div>'
        '<div class="cc-hero-note">Illustrative concept artwork. Figures painted into the image (wind, burned '
        'area, spread rate) are not live values; the live and simulated values are in the panels below.</div>',
        unsafe_allow_html=True)


def _static_serving_enabled() -> bool:
    try:
        return bool(st.get_option("server.enableStaticServing"))
    except Exception:
        return False


# ── status strip ───────────────────────────────────────────────────────────── #

def _pill(cls: str, label: str, value: str) -> str:
    return (f'<span class="cc-pill"><i class="gt-dot {cls}"></i><b>{html.escape(label)}</b>'
            f'{html.escape(value)}</span>')


def render_status_strip(twin, summary: dict):
    feeds = feed_status(twin)
    bd = summary.get("severity_breakdown", {}) or {}
    pills = [_pill("ok", "SYSTEM", "Operational")]
    pills.append(_pill({"live": "data", "error": "crit"}.get(feeds["overall"], "warn"), "DATA",
                       {"live": "Live · NASA FIRMS + OpenWeatherMap", "cached": "Cached (API unavailable)",
                        "error": "API unavailable", "partial": "Partly live"}.get(feeds["overall"],
                                                                                 "Demo / offline (synthetic)")))
    if bd.get("EXTREME"):
        pills.append(_pill("crit", "RISK", f"{bd['EXTREME']} extreme zone(s)"))
    elif bd.get("HIGH"):
        pills.append(_pill("warn", "RISK", f"{bd['HIGH']} high zone(s)"))
    else:
        pills.append(_pill("ok", "RISK", "No high / extreme zones"))
    spread = latest_spread()
    if spread is not None and spread[1].final.get("burning"):
        pills.append(_pill("crit", "SPREAD", "Simulated fire active"))
    elif spread is not None:
        pills.append(_pill("ok", "SPREAD", "Simulated fire burned out"))
    else:
        pills.append(_pill("data", "SPREAD", "Ready"))
    pills.append(_pill("data" if feeds["maps"] else "warn", "MAP",
                       "Google satellite" if feeds["maps"] else "Maps key missing"))
    persist = st.session_state.get("persist_result") or {}
    if persist.get("pending"):
        pills.append(_pill("data", "DATABASE", "Saving snapshot"))
    elif persist.get("persisted"):
        pills.append(_pill("ok", "DATABASE", "Snapshot saved"))
    elif persist:
        pills.append(_pill("warn", "DATABASE", "Not saved this run"))
    st.markdown(f'<div class="cc-strip" role="status">{"".join(pills)}</div>', unsafe_allow_html=True)


# ── wildfire alert panel ───────────────────────────────────────────────────── #

def _stat(label: str, value: str, hot: bool = False) -> str:
    return (f'<div class="stat"><div class="l">{html.escape(label)}</div>'
            f'<div class="n{" hot" if hot else ""}">{html.escape(value)}</div></div>')


def alert_panel_model(twin, summary: dict) -> dict:
    """Content of the panel, from the simulation if one ran, else the model."""
    feeds = feed_status(twin)
    data_chip = {"live": ("live", "LIVE DATA"), "cached": ("demo", "CACHED DATA"), "error": ("demo", "API UNAVAILABLE"),
                 "partial": ("demo", "PARTLY LIVE")}.get(feeds["overall"], ("demo", "DEMO DATA"))
    spread = latest_spread()
    if spread is not None:
        src, res = spread
        fin = res.final
        burning = int(fin.get("burning", 0))
        ros = max((m.get("ros_m_per_min", 0) for m in res.metrics), default=0.0) * 60
        w_speed, w_from = fin.get("wind_speed_ms", 0.0), fin.get("wind_from_deg", 0.0)
        return {
            "state": "" if burning else "warn",
            "kicker": "WILDFIRE ALERT · SPREAD SIMULATION" if burning else "SPREAD SIMULATION · BURNED OUT",
            "chips": [("sim", "SIMULATED"),
                      data_chip if src.startswith("Current") else ("demo", "SCENARIO INPUT")],
            "title": f"{'Active simulated fire front' if burning else 'Simulated fire'} · {res.focus.name}",
            "meta": (f"{res.focus.lat:.4f}°N, {res.focus.lon:.4f}°E · {src} · wind {w_speed:.1f} m/s from "
                     f"{compass(w_from)} · cellular-automata projection, not an observed fire"),
            "stats": [("Burning cells", f"{burning}", burning > 0),
                      ("Burned area", f"{fin.get('burned_ha', 0):.2f} ha", False),
                      ("Peak spread rate", f"{ros:.0f} m/h", False),
                      ("Simulation time", f"T+{res.duration_minutes:.0f} min", False)],
            "thumb_tag": "Illustrative image",
        }
    alerts = getattr(twin.current_snapshot, "alerts", None) or []
    w = regional_wind(twin)
    wind_txt = f"{w[0]:.1f} m/s {compass(w[1])}" if w else "n/a"
    if alerts:
        a = alerts[0]
        sev = a.severity
        return {
            "state": "" if sev == "EXTREME" else ("warn" if sev == "HIGH" else "calm"),
            "kicker": f"FIRE-RISK ALERT · {sev}",
            "chips": [("", "MODEL PREDICTION"), data_chip],
            "title": f"Highest-risk zone {a.zone_id} · {twin.region.name}",
            "meta": f"{a.latitude:.3f}°N, {a.longitude:.3f}°E · {a.reason or 'model risk score above threshold'}",
            "stats": [("Risk score", f"{a.risk_score:.0%}", sev == "EXTREME"),
                      ("Severity", sev, sev == "EXTREME"),
                      ("Alert zones", f"{summary.get('total_alerts', 0)}", False),
                      ("Regional wind", wind_txt, False)],
            "thumb_tag": "Illustrative image",
        }
    return {
        "state": "calm", "kicker": "NO ACTIVE ALERT", "chips": [("", "MODEL PREDICTION"), data_chip],
        "title": f"No zone above the alert threshold · {twin.region.name}",
        "meta": "Run a spread simulation to project how a fire would behave under current or scenario weather.",
        "stats": [("Peak risk", f"{summary.get('max_risk_score', 0):.0%}", False),
                  ("Mean risk", f"{summary.get('mean_risk_score', 0):.0%}", False),
                  ("Zones", f"{summary.get('total_zones', 0)}", False), ("Regional wind", wind_txt, False)],
        "thumb_tag": "Illustrative image",
    }


def render_fire_alert_panel(twin, summary: dict):
    m = alert_panel_model(twin, summary)
    chips = "".join(f'<span class="chip {c}">{html.escape(t)}</span>' for c, t in m["chips"])
    stats = "".join(_stat(*s) for s in m["stats"])
    thumb = data_uri(ALERT_THUMB)
    with st.container(key="cc_alert"):
        st.markdown(f"""
<div class="fa {m['state']}" role="alert" aria-live="polite">
  <div class="thumb" style="background-image:url('{thumb}')"><span class="tag">{html.escape(m['thumb_tag'])}</span></div>
  <div class="body">
    <div class="head"><span class="pulse" aria-hidden="true"></span><span class="kicker">{html.escape(m['kicker'])}</span>{chips}</div>
    <h3>{html.escape(m['title'])}</h3>
    <div class="meta">{html.escape(m['meta'])}</div>
    <div class="stats">{stats}</div>
  </div>
</div>""", unsafe_allow_html=True)
        st.page_link(SPREAD_PAGE, label="OPEN SPREAD SIMULATION  →", width="stretch")


# ── Explore cards ──────────────────────────────────────────────────────────── #

def render_explore_cards(items):
    """items: (index, title, description, page path, icon name)."""
    per_row = 3
    for start in range(0, len(items), per_row):
        cols = st.columns(per_row, gap="small")
        for j, (idx, title, desc, target, icon) in enumerate(items[start:start + per_row]):
            _explore_card(cols[j], start + j, idx, title, desc, target, icon)


def _explore_card(col, i, idx, title, desc, target, icon):
    with col, st.container(key=f"cc_explore_{i}"):
        st.markdown(f'<div class="ex-card"><div class="ex-ico">{svg_icon(icon)}</div>'
                    f'<div class="idx">{idx}</div><h4>{html.escape(title)}</h4>'
                    f'<p>{html.escape(desc)}</p></div>', unsafe_allow_html=True)
        st.page_link(target, label=f"Open {title} →", width="stretch")


# ── data / system status ───────────────────────────────────────────────────── #

def _dot(mode: str) -> str:
    return {"live": "ok", "cached": "warn", "error": "crit"}.get(mode, "warn")


def render_system_status(twin, summary: dict, offline: bool):
    """NASA FIRMS / OpenWeatherMap / Google Maps / risk model / database status.
    Only states that were actually observed are reported; no key is shown."""
    feeds = feed_status(twin)
    fs, ws = feeds["firms_status"], feeds["weather_status"]
    persist = st.session_state.get("persist_result") or {}
    from src.storage import database as db
    fallback = None
    try:
        fallback = db.fallback_reason()
    except Exception:
        pass
    if persist.get("pending"):
        db_cls, db_txt = "data", "Saving snapshot to the database"
    elif persist.get("persisted"):
        db_cls, db_txt = "ok", "Connected · snapshot saved"
        if persist.get("new_extreme_count"):
            db_txt += (f" · {persist['new_extreme_count']} newly-extreme zone(s), "
                       + ("email sent" if persist.get("emailed") else "email not configured"))
    elif persist:
        db_cls, db_txt = "warn", "Snapshot not saved (database unavailable this run)"
    else:
        db_cls, db_txt = "ok", "Connected"
    if fallback:
        db_txt += " · local SQLite fallback (cloud database unreachable)"
    firms_extra = []
    if fs.get("fetched_utc"):
        firms_extra.append(f"fetched {str(fs['fetched_utc'])[:16].replace('T', ' ')} UTC")
    if fs.get("error") and feeds["firms_mode"] in ("error", "cached"):
        firms_extra.append(f"reason: {str(fs['error'])[:120]}")
    w_extra = []
    if feeds["weather_mode"] in ("live", "cached") and ws.get("n") is not None:
        w_extra.append(f"{ws.get('n')}/{ws.get('n_total', ws.get('n'))} weather grid points of {twin.region.name}")
    pw = st.session_state.get("_last_point_weather")
    if pw:
        w_extra.append(f"selected location: {pw}")
    if ws.get("error") and feeds["weather_mode"] in ("error", "cached"):
        w_extra.append(f"reason: {str(ws['error'])[:120]}")
    m_cls, _, m_txt = maps_item(feeds)
    from src.dashboard.dashboard_common import model_status_text
    ts = (summary.get("timestamp") or "")[:19].replace("T", " ")
    rows = [
        (_dot(feeds["firms_mode"]), "NASA FIRMS · SATELLITE FIRE DETECTIONS",
         " · ".join([feed_text("firms", feeds)] + firms_extra)),
        (_dot(feeds["weather_mode"]), "OPENWEATHERMAP · CURRENT WEATHER",
         " · ".join([feed_text("weather", feeds)] + w_extra)),
        (m_cls, "GOOGLE MAPS · SATELLITE", m_txt[0].upper() + m_txt[1:]),
        ("ok", "RISK MODEL · XGBOOST", f"Loaded · {model_status_text(twin.region)} · "
                                       f"{summary.get('total_zones', 0)} zones scored {ts} UTC (model prediction)"),
        (db_cls, "DATABASE", db_txt),
    ]
    items = "".join(f'<div class="sys-item"><div class="l"><i class="gt-dot {c}"></i>{html.escape(l)}</div>'
                    f'<div class="v">{html.escape(v)}</div></div>' for c, l, v in rows)
    st.markdown(f'<div class="sys-grid">{items}</div>', unsafe_allow_html=True)


# ── Command Center regional map: Google satellite + risk zones + detections ── #

def region_hotspots(twin) -> tuple:
    """(markers, kind, summary) for the regional map. Real FIRMS detections of
    the last ~48 h when the feed is live (or cached); synthetic ones only in
    demo mode and labelled as such; nothing when the API failed."""
    from src.data_ingestion.live_point import hotspot_markers
    feeds = feed_status(twin)
    mode = feeds["firms_mode"]
    h = getattr(twin.ingestion, "last_hotspots", None)
    if h is not None and len(h) and "acq_date" in h.columns:
        import pandas as pd
        d = pd.to_datetime(h["acq_date"], errors="coerce")
        h = h[d >= pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=1)]
    markers = hotspot_markers(h) if h is not None else []
    if mode in ("live", "cached"):
        n = len(markers)
        summary = (f"{n} NASA FIRMS detection(s), last 48 h" if n else "No NASA FIRMS detections, last 48 h")
        if mode == "cached":
            summary += " (cached)"
        return markers, "observed", summary
    if mode == "error":
        return [], "observed", "NASA FIRMS unavailable: none shown"
    return markers, "synthetic", f"{len(markers)} demo hotspot(s), synthetic"


def render_region_map(twin, height: int = 560):
    """Regional risk map on Google satellite (the same fire_map component and
    Google Maps configuration as the What-If and Spread Simulation pages)."""
    from src.dashboard.geo_fire_map import missing_key_card, region_payload, render_fire_map
    from src.dashboard.geo_spread import google_maps_key, google_maps_map_id
    st.markdown('<div class="sec-hdr">Regional risk map · Google satellite</div>', unsafe_allow_html=True)
    key = google_maps_key()
    snap = twin.current_snapshot
    if not key:
        missing_key_card()
        from src.dashboard.dashboard_common import render_risk_map
        render_risk_map(snap.processed_grid, snap.risk_scores, twin.region, height=height, header=False)
        return
    markers, kind, hsum = region_hotspots(twin)
    feeds = feed_status(twin)
    note = ("Zone colour = XGBoost fire-risk prediction (model output, not an observation). "
            + ("Dots = NASA FIRMS VIIRS satellite detections (observed)." if kind == "observed"
               else "Dots = synthetic demo hotspots (not observed)."))
    if feeds["overall"] in ("error", "cached"):
        note += " Some live data is unavailable; see Data & system status."
    payload = region_payload(twin, markers, kind, hsum, note, key, google_maps_map_id(), height=height)
    with st.container(key="cc_regionmap"):
        render_fire_map(payload, key="cc_region_map")
