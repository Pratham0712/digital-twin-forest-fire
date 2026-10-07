"""Global status ticker shown at the top of every signed-in page.

Content comes only from state the app already holds:
  * the shared Digital Twin (st.session_state["twin"]): region, risk level,
    alert counts, regional mean wind, last refresh time, live/demo mode;
  * the fire-spread simulation results kept by geo_spread (labelled SIMULATED);
  * whether the FIRMS / OpenWeatherMap / Google Maps keys are configured.
When no twin has been loaded yet the ticker says "DEMO / OFFLINE MODE" and does
not invent any value.

The slot is created by set_page() (so the ticker sits above every page's
content) and filled by render_global_ticker(); ensure_twin() fills it again
once a fresh twin is available, so the text always matches the page.
"""
import html
import os
import time
from typing import List, Optional, Tuple

import streamlit as st

SLOT_KEY = "_global_ticker_slot"
SECONDS_PER_ITEM = 7.0

_COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
SPREAD_KEYS = (("applied", "Applied What-If scenario"), ("current", "Current risk state"),
               ("custom", "Custom weather scenario"))


def compass(deg: float) -> str:
    return _COMPASS[int((float(deg) % 360) / 22.5 + 0.5) % 16]


def latest_spread() -> Optional[Tuple[str, object]]:
    """(source label, LocalSpreadResult) of the most recent spread simulation
    run in this session, or None."""
    last = st.session_state.get("last_geo_run") or {}
    order = [k for k, _ in SPREAD_KEYS]
    if last.get("kp") in order:
        order.remove(last["kp"])
        order.insert(0, last["kp"])
    labels = dict(SPREAD_KEYS)
    for kp in order:
        res = st.session_state.get(f"{kp}_geo_result")
        if res is not None and getattr(res, "metrics", None):
            return labels[kp], res
    return None


def regional_wind(twin) -> Optional[Tuple[float, float]]:
    try:
        from src.data_ingestion.wind import mean_wind
        g = twin.current_snapshot.processed_grid
        return mean_wind(g["wx_wind_speed_ms"].values, g["wx_wind_deg"].values)
    except Exception:
        return None


MODE_LABEL = {"live": "LIVE", "cached": "CACHED", "error": "UNAVAILABLE", "demo": "DEMO",
              "not_configured": "NOT CONFIGURED", "pending": "PENDING", "none": "DEMO"}


def feed_status(twin) -> dict:
    """What each feed ACTUALLY delivered on the twin's last refresh (from the
    ingestion layer's source_status), not what the sidebar asked for. LIVE only
    when the real request succeeded; synthetic data is never called LIVE."""
    ss = getattr(getattr(twin, "ingestion", None), "source_status", None) or {}
    if twin is None:
        f = w = "none"
    else:
        f = (ss.get("firms") or {}).get("mode") or ("demo" if getattr(twin, "offline", True) else "pending")
        w = (ss.get("weather") or {}).get("mode") or ("demo" if getattr(twin, "offline", True) else "pending")
    modes = {f, w}
    if modes == {"live"}:
        overall = "live"
    elif "error" in modes:
        overall = "error"
    elif "cached" in modes:
        overall = "cached"
    elif "live" in modes:
        overall = "partial"
    else:
        overall = "demo"
    return {"live": overall == "live", "overall": overall,
            "firms": MODE_LABEL.get(f, f.upper()), "weather": MODE_LABEL.get(w, w.upper()),
            "firms_mode": f, "weather_mode": w,
            "firms_status": ss.get("firms") or {}, "weather_status": ss.get("weather") or {},
            "maps": bool(os.getenv("GOOGLE_MAPS_API_KEY"))}


def data_badge(feeds: dict) -> Tuple[str, str]:
    """(css class, text) for header / ticker badges."""
    return {"live": ("live", "LIVE · REAL DATA"), "cached": ("demo", "CACHED DATA"),
            "error": ("demo", "API UNAVAILABLE"), "partial": ("demo", "PARTLY LIVE")}.get(
        feeds["overall"], ("demo", "DEMO / OFFLINE MODE"))


def _utc(iso: Optional[str]) -> str:
    return (iso or "")[:16].replace("T", " ").replace("+00:00", "").rstrip("Z") + (" UTC" if iso else "")


def feed_text(kind: str, feeds: dict) -> str:
    """One-line description of a feed for the ticker / status panels."""
    mode = feeds[f"{kind}_mode"]
    st_ = feeds[f"{kind}_status"]
    if kind == "firms":
        if mode == "live":
            n = st_.get("n", 0)
            return (f"LIVE · {n} detection(s) · latest observation {_utc(st_.get('latest_acq_utc'))}" if n else
                    "LIVE · no NASA FIRMS detections in this region/time window")
        if mode == "cached":
            return f"CACHED · {st_.get('n', 0)} detection(s) fetched {_utc(st_.get('fetched_utc'))} (API unavailable)"
        if mode == "error":
            return "UNAVAILABLE · no detections shown"
        if mode == "not_configured":
            return "NOT CONFIGURED · synthetic detections (not observed)"
        return "DEMO · synthetic detections"
    if mode == "live":
        return f"LIVE · fetched {_utc(st_.get('fetched_utc'))}"
    if mode == "cached":
        return f"CACHED · fetched {_utc(st_.get('fetched_utc'))} (API unavailable)"
    if mode == "error":
        return "UNAVAILABLE · risk computed without current weather"
    if mode == "not_configured":
        return "NOT CONFIGURED · synthetic weather"
    return "DEMO · synthetic weather"


def risk_level(summary: dict) -> Tuple[str, str]:
    bd = summary.get("severity_breakdown", {}) or {}
    for sev, cls in (("EXTREME", "crit"), ("HIGH", "warn"), ("MODERATE", "data")):
        if bd.get(sev):
            return sev, cls
    return "LOW", "ok"


def ticker_items(twin=None) -> Tuple[dict, List[Tuple[str, str, str]]]:
    """(feed status, [(dot class, label, value), ...])."""
    feeds = feed_status(twin)
    items: List[Tuple[str, str, str]] = []
    summary = twin.get_summary() if twin is not None and twin.current_snapshot is not None else None
    if summary is None:
        items.append(("warn", "DATA", "DEMO / OFFLINE MODE · no risk state loaded yet"))
    else:
        level, cls = risk_level(summary)
        bd = summary.get("severity_breakdown", {}) or {}
        items.append(("data", "REGION", twin.region.name))
        items.append((cls, "RISK LEVEL", f"{level} · peak {summary.get('max_risk_score', 0):.0%}"
                                         f" · mean {summary.get('mean_risk_score', 0):.0%}"))
        items.append(("crit" if bd.get("EXTREME") else ("warn" if summary.get("total_alerts") else "ok"),
                      "ALERT ZONES", f"{summary.get('total_alerts', 0)} "
                                     f"({bd.get('EXTREME', 0)} extreme, {bd.get('HIGH', 0)} high)"))
        w = regional_wind(twin)
        if w is not None:
            items.append(("data", "WIND", f"{w[0]:.1f} m/s from {compass(w[1])} ({w[1]:.0f}°) · regional mean"
                                          + {"live": "", "cached": " · cached"}.get(feeds["weather_mode"], " · demo")))
        ts = (summary.get("timestamp") or "")[:16].replace("T", " ")
        if ts:
            items.append(("data", "LAST REFRESH", f"{ts} UTC"))
    spread = latest_spread()
    if spread is not None:
        src, res = spread
        fin = res.final
        items.append(("fire" if fin.get("burning") else "ok", "SPREAD SIMULATION",
                      f"SIMULATED · {fin.get('burned_ha', 0):.2f} ha burned, {fin.get('burning', 0)} cells burning "
                      f"at {res.focus.name} (T+{res.duration_minutes:.0f} min · {src})"))
    elif st.session_state.get("simulation_config"):
        items.append(("warn", "SPREAD SIMULATION", "scenario applied · ready to run"))
    else:
        items.append(("ok", "SPREAD SIMULATION", "idle · no run this session"))
    dot = {"live": "ok", "error": "crit"}
    items.append((dot.get(feeds["firms_mode"], "warn"), "NASA FIRMS", feed_text("firms", feeds)))
    items.append((dot.get(feeds["weather_mode"], "warn"), "OPENWEATHERMAP", feed_text("weather", feeds)))
    items.append(maps_item(feeds))
    return feeds, items


def maps_item(feeds: dict) -> Tuple[str, str, str]:
    gs = st.session_state.get("gmaps_status") or {}
    if not feeds["maps"]:
        return "warn", "GOOGLE MAPS", "NOT CONFIGURED · set GOOGLE_MAPS_API_KEY"
    if gs and not gs.get("ok"):
        return "crit", "GOOGLE MAPS", "ERROR · " + str(gs.get("error") or "map did not load")[:80]
    if gs.get("ok"):
        return "ok", "GOOGLE MAPS", "satellite map loaded" + (" · 3D vector" if gs.get("vector") else " · 2D")
    return "data", "GOOGLE MAPS", "key configured · satellite map loads in the browser"


def ticker_html(twin=None) -> str:
    feeds, items = ticker_items(twin)
    parts = []
    for cls, label, value in items:
        parts.append(f'<span class="gt-item"><i class="gt-dot {cls}"></i><b>{html.escape(label)}</b>'
                     f'<span class="v">{html.escape(value)}</span></span><span class="gt-sep">•</span>')
    track = "".join(parts)
    dur = max(30.0, SECONDS_PER_ITEM * len(items))
    # A negative delay derived from the clock keeps the scroll position
    # continuous across Streamlit reruns instead of restarting at zero.
    delay = -(time.time() % dur)
    tag_cls, tag_txt = data_badge(feeds)
    return (f'<div class="gt" role="region" aria-label="System status ticker">'
            f'<div class="gt-tag {tag_cls}"><span class="dot"></span>{tag_txt}</div>'
            f'<div class="gt-view"><div class="gt-track" style="--gt-dur:{dur:.0f}s;animation-delay:{delay:.1f}s">'
            f'<span>{track}</span><span aria-hidden="true">{track}</span></div></div></div>')


def create_ticker_slot():
    """Called by set_page(): reserve the ticker's place at the top of the page."""
    st.session_state[SLOT_KEY] = st.empty()


def render_global_ticker(twin=None):
    """Draw (or redraw) the ticker into the slot reserved by set_page(). Uses
    the shared twin when none is passed. Signed-in pages only."""
    if not st.session_state.get("auth_user"):
        return
    slot = st.session_state.get(SLOT_KEY)
    if slot is None:
        return
    if twin is None:
        twin = st.session_state.get("twin")
    try:
        slot.markdown(ticker_html(twin), unsafe_allow_html=True)
    except Exception:            # presentation only: never break a page
        pass
