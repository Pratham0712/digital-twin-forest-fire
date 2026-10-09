"""
live_modes.py - the What-If Simulator's two real-data modes.

  LIVE REAL-WORLD SIMULATION   weather = the latest OpenWeatherMap observation at
                               the selected location, locked; FIRMS = the latest
                               NASA FIRMS detections; ignition = ONLY valid
                               observed FIRMS cells inside the simulation area.
  LIVE DATA -> WHAT-IF         the same observation is loaded as the REAL
                               BASELINE (kept, never overwritten by edits), the
                               controls unlock, the user's values are SCENARIO
                               INPUT; FIRMS is context; ignition = retained
                               observed FIRMS cells or a HYPOTHETICAL user
                               ignition, or none.

Offline / demo mode keeps the original synthetic What-If behaviour (DEMO).
Nothing here invents a value: a missing observation is shown as "Unavailable
from live source", and no fire is ever started from temperature, wind, FWI or
model risk alone.

State (st.session_state):
    wi_sim_mode       "live" | "whatif"
    wi_baseline       REAL BASELINE loaded into WHAT-IF (see baseline_from_observation)
    wl_<input>        the four control widgets (temp_c, humidity_pct, wind_speed_ms, wind_from_deg)
"""
from __future__ import annotations

import html
import math
from typing import Optional

import pandas as pd
import streamlit as st

from src.dashboard.ui.global_ticker import compass
from src.utils.timezone import format_ist

LIVE, WHATIF, DEMO = "live", "whatif", "demo"
MODE_KEY = "wi_sim_mode"
BASE_KEY = "wi_baseline"
SCN_KEY = "wi_whatif_values"   # the user's WHAT-IF control values (not a widget key, survives page changes)
CTRL = "wl"
MODE_NAMES = {LIVE: "LIVE REAL-WORLD", WHATIF: "LIVE DATA → WHAT-IF", DEMO: "DEMO / OFFLINE MODE"}
LIVE_DESC = ("Fetch the latest real weather and NASA FIRMS observations and evaluate the selected location "
             "using current conditions.")
WHATIF_DESC = ("Load the latest real-world conditions into the simulator, then modify them to test a "
               "hypothetical scenario.")
UNAVAILABLE = "Unavailable from live source"

# key, OpenWeatherMap field, label, unit, slider min, max, step, decimals
INPUTS = (("temp_c", "temperature_c", "Temperature", "°C", -40.0, 60.0, 0.1, 1),
          ("humidity_pct", "humidity_pct", "Relative humidity", "%", 0.0, 100.0, 1.0, 0),
          ("wind_speed_ms", "wind_speed_ms", "Wind speed", "m/s", 0.0, 60.0, 0.1, 1),
          ("wind_from_deg", "wind_deg", "Wind direction (blowing FROM)", "°", 0.0, 359.0, 1.0, 0))
SCENARIO_START = {"temp_c": 32.0, "humidity_pct": 40.0, "wind_speed_ms": 5.0, "wind_from_deg": 225.0}

IGN_NONE, IGN_OBSERVED, IGN_HYPOTHETICAL = "none", "observed", "hypothetical"
MAP_POINTS = "Map points"                    # local_spread.PLACEMENTS: user-clicked hypothetical cells
IGN_OPTIONS = {IGN_NONE: "NONE — no ignition source", IGN_OBSERVED: "OBSERVED — NASA FIRMS detections",
               IGN_HYPOTHETICAL: "HYPOTHETICAL — user-selected ignition points"}
SOURCE_LABEL = {"OBSERVED_FIRMS": "SOURCE: NASA FIRMS OBSERVED", "HYPOTHETICAL_USER": "SOURCE: USER HYPOTHETICAL IGNITION",
                "NONE": "SOURCE: NO IGNITION"}
HIGH = ("HIGH", "EXTREME")
COLORS = {"green": "#35d07f", "yellow": "#fbbf24", "orange": "#ff9a2e", "red": "#ff6b4a", "grey": "#8a96a6",
          "blue": "#8FD3FF"}


# ── mode ──────────────────────────────────────────────────────────────────── #

def current_mode(demo: bool) -> str:
    if demo:
        return DEMO
    m = st.session_state.get(MODE_KEY)
    return m if m in (LIVE, WHATIF) else LIVE


def set_mode(mode: str):
    st.session_state[MODE_KEY] = mode


# ── baseline / scenario values ────────────────────────────────────────────── #

def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def baseline_from_observation(w: Optional[dict], ws: dict, lat: float, lon: float) -> Optional[dict]:
    """The REAL observation as a baseline record (None without one). Values a
    response did not contain are None, never a default."""
    if not w:
        return None
    missing = set(w.get("missing_fields") or [])
    values = {k: (None if src in missing else _num(w.get(src))) for k, src, *_ in INPUTS}
    extra = {"pressure_hpa": _num(w.get("pressure_hpa")), "precipitation_mm": _num(w.get("precipitation_mm")) or 0.0,
             "clouds_pct": _num(w.get("clouds_pct")), "weather_main": w.get("weather_main"),
             "weather_description": w.get("weather_description")}
    return {"values": values, "extra": extra, "observed_utc": w.get("observed_at"),
            "fetched_utc": ws.get("fetched_utc"), "status": ws.get("mode"), "lat": float(lat), "lon": float(lon),
            "place": w.get("place_name") or ""}


def complete(values: Optional[dict]) -> bool:
    return bool(values) and all(values.get(k) is not None for k, *_ in INPUTS)


def _round(key: str, v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    dec = next(i[7] for i in INPUTS if i[0] == key)
    return float(round(float(v), dec))


def scenario_values(sliders: dict, baseline: Optional[dict]) -> dict:
    """Effective WHAT-IF values: a slider left on the (rounded) baseline value
    means "unchanged" and keeps the exact observed value."""
    base = (baseline or {}).get("values") or {}
    out = {}
    for k, *_ in INPUTS:
        s, b = sliders.get(k), base.get(k)
        out[k] = b if (b is not None and s is not None and _round(k, b) == _round(k, s)) else s
    return out


def changes(baseline: Optional[dict], values: dict) -> list:
    """Per-input REAL BASELINE vs WHAT-IF SCENARIO comparison."""
    base = (baseline or {}).get("values") or {}
    rows = []
    for k, _src, label, unit, *_r, dec in INPUTS:
        b, s = base.get(k), values.get(k)
        if b is None:
            delta, changed = None, s is not None
        else:
            delta = (s - b) if s is not None else None
            if k == "wind_from_deg" and delta is not None:
                delta = (delta + 180.0) % 360.0 - 180.0           # shortest turn, -180..180
            changed = delta is not None and abs(delta) >= 10 ** -(dec + 1) / 2
        rows.append({"key": k, "label": label, "unit": unit, "dec": dec, "baseline": b, "scenario": s, "change": delta,
                     "changed": bool(changed), "text": change_text(label, unit, b, s, delta, dec)})
    return rows


def _fmt(v, unit, dec) -> str:
    if v is None:
        return "unavailable"
    return f"{v:.{dec}f}{'' if unit in ('°', '%') else ' '}{unit}" if unit != "°C" else f"{v:.{dec}f}°C"


def change_text(label, unit, b, s, delta, dec) -> str:
    """e.g. 'Temperature — Baseline: 22.7°C / Scenario: 35.0°C / Change: +12.3°C'."""
    ch = "n/a" if delta is None else (f"{delta:+.{dec}f}{'' if unit in ('°', '%') else ' '}{unit}"
                                      if unit != "°C" else f"{delta:+.{dec}f}°C")
    return f"{label} — Baseline: {_fmt(b, unit, dec)} / Scenario: {_fmt(s, unit, dec)} / Change: {ch}"


def is_modified(baseline: Optional[dict], values: dict) -> bool:
    return any(r["changed"] for r in changes(baseline, values))


def newer_observation(saved: Optional[dict], obs: Optional[dict]) -> bool:
    """True when the latest observation differs from the WHAT-IF baseline (a
    newer fetch, other values, or the location moved): it becomes the baseline."""
    if not obs:
        return False
    if not saved:
        return True
    moved = abs(saved["lat"] - obs["lat"]) > 5e-4 or abs(saved["lon"] - obs["lon"]) > 5e-4
    return (moved or obs.get("values") != saved.get("values") or obs.get("extra") != saved.get("extra")
            or obs.get("observed_utc") != saved.get("observed_utc") or obs.get("fetched_utc") != saved.get("fetched_utc"))


# ── scenario for the risk twin (XGBoost + FWI, unchanged pipeline) ─────────── #

FIRMS_COLS = ("latitude", "longitude", "bright_ti4", "scan", "track", "acq_date", "acq_time", "satellite",
              "confidence", "version", "bright_t31", "frp", "daynight")


def firms_records(df: Optional[pd.DataFrame]) -> list:
    """Real FIRMS rows as plain records (for the scenario twin and the hand-off)."""
    if df is None or df.empty:
        return []
    out = []
    for _, r in df.iterrows():
        rec = {}
        for c in FIRMS_COLS:
            if c in r.index:
                v = r[c]
                rec[c] = None if (v is None or (isinstance(v, float) and not math.isfinite(v)) or
                                  (not isinstance(v, (str, bytes)) and pd.isna(v))) else (
                    v if isinstance(v, (str, int, float, bool)) else str(v))
        out.append(rec)
    return out


def twin_scenario(mode: str, values: dict, extra: Optional[dict], weather_status: str,
                  detections: Optional[pd.DataFrame], firms_status: str) -> dict:
    """Scenario dict for DigitalTwin(offline=True, scenario=...): the existing
    risk pipeline runs on exactly these weather values (real in LIVE, the
    user's in WHAT-IF) and on the real FIRMS detections - no synthetic weather
    noise, no seeded hotspots."""
    pw = None
    if complete(values):
        ex = extra or {}
        pw = {"temperature_c": float(values["temp_c"]), "humidity_pct": float(values["humidity_pct"]),
              "wind_speed_ms": float(values["wind_speed_ms"]), "wind_deg": float(values["wind_from_deg"]),
              "pressure_hpa": ex.get("pressure_hpa") if ex.get("pressure_hpa") is not None else 1013.0,
              "precipitation_mm": float(ex.get("precipitation_mm") or 0.0),
              "clouds_pct": ex.get("clouds_pct") if ex.get("clouds_pct") is not None else 0,
              "weather_main": ex.get("weather_main") or "Unknown"}
    if mode == LIVE:
        wmode, wlabel = weather_status, "OpenWeatherMap observation at the selected location (LIVE)"
    else:
        wmode, wlabel = "scenario", "WHAT-IF scenario weather (user values on the real baseline)"
    firms_ok = firms_status in ("live", "cached")
    return {"sim_mode": mode, "temp_c": values.get("temp_c"), "humidity_pct": values.get("humidity_pct"),
            "wind_speed_ms": values.get("wind_speed_ms"), "wind_from_deg": values.get("wind_from_deg"),
            "n_hotspots": 0, "point_weather": pw, "weather_mode": wmode if pw else "error",
            "weather_label": wlabel, "observed_hotspots": firms_records(detections) if firms_ok else [],
            "firms_mode": firms_status if firms_ok else "error"}


# ── risk at the location ──────────────────────────────────────────────────── #

def zone_risk(twin, lat: float, lon: float) -> dict:
    from src.simulation.local_spread import zone_conditions
    snap = twin.current_snapshot
    cond = zone_conditions(snap.processed_grid, snap.risk_scores, lat, lon)
    r = cond.get("risk_score")
    sev = twin.alert_engine.classify(float(r)) if r is not None and math.isfinite(float(r)) else "UNKNOWN"
    return {**cond, "severity": sev}


# ── ignition ──────────────────────────────────────────────────────────────── #

def ignition_source(mode: str, setup: dict) -> str:
    if mode == LIVE:
        return IGN_OBSERVED
    if mode == WHATIF:
        s = setup.get("ignition_source", IGN_NONE)
        return s if s in IGN_OPTIONS else IGN_NONE
    return IGN_HYPOTHETICAL                         # demo / legacy: the placement controls


def ignition_spec(mode: str, setup: dict, cls: Optional[dict]) -> dict:
    """What will ignite: OBSERVED FIRMS cells, a HYPOTHETICAL user ignition, or NONE."""
    src = ignition_source(mode, setup)
    n_valid = (cls or {}).get("n_valid", 0)
    if src == IGN_OBSERVED and n_valid:
        return {"source": "OBSERVED_FIRMS", "kind": "OBSERVED", "points": (cls or {}).get("valid_points", []),
                "n_valid": n_valid, "n_cells": (cls or {}).get("n_cells", 0), "placement": None,
                "label": SOURCE_LABEL["OBSERVED_FIRMS"]}
    if src == IGN_HYPOTHETICAL:
        n_req = int(setup.get("n_ignition") or 1)
        manual = setup.get("placement") == MAP_POINTS
        pts = [list(map(float, q)) for q in (setup.get("ignition_points") or [])][:n_req] if manual else []
        if manual and not pts:
            # Map points chosen but none placed yet: nothing ignites (no fallback cell is invented).
            return {"source": "NONE", "kind": None, "points": [], "n_valid": 0, "placement": MAP_POINTS,
                    "awaiting_points": True, "n_requested": n_req, "n_selected": 0, "manual": True,
                    "label": f"SOURCE: NO IGNITION (0 / {n_req} HYPOTHETICAL POINTS SELECTED)"}
        return {"source": "HYPOTHETICAL_USER", "kind": "HYPOTHETICAL", "points": pts, "n_valid": 0,
                "placement": setup.get("placement"), "n_ignition": n_req, "n_requested": n_req,
                "n_selected": len(pts) if manual else n_req, "manual": manual,
                "label": SOURCE_LABEL["HYPOTHETICAL_USER"]}
    return {"source": "NONE", "kind": None, "points": [], "n_valid": 0, "placement": None,
            "label": SOURCE_LABEL["NONE"]}


def hypothetical_status(ign: dict) -> Optional[str]:
    """'HYPOTHETICAL IGNITIONS · 2 / 3 SELECTED' (map points) or the placement summary."""
    if ign.get("manual"):
        return f"HYPOTHETICAL IGNITIONS · {ign.get('n_selected', 0)} / {ign.get('n_requested', 0)} SELECTED"
    if ign.get("source") == "HYPOTHETICAL_USER":
        return f"HYPOTHETICAL IGNITIONS · {ign.get('n_requested')} cell(s) · {ign.get('placement')}"
    return None


# ── status ────────────────────────────────────────────────────────────────── #

def _notes(cls: Optional[dict], firms_status: str) -> list:
    c = cls or {}
    out = []
    if c.get("n_non_fuel"):
        out.append(f"{c['n_non_fuel']} NASA FIRMS detection(s) inside the area fell on water, road, built-up or "
                   "non-fuel cells and were not used as ignitions.")
    if c.get("n_outside"):
        out.append(f"{c['n_outside']} NASA FIRMS detection(s) in the search box are outside the simulation area "
                   "(shown on the map, not ignited).")
    if c.get("n_valid"):
        out.append(f"{c['n_valid']} of {c.get('n_total', 0)} observation(s) became valid ignitions "
                   f"({c.get('n_cells', 0)} grid cell(s)).")
    if firms_status == "cached":
        out.append("NASA FIRMS data is CACHED (the latest request failed).")
    return out


def _awaiting_msg(ign: dict) -> str:
    return (f"No hypothetical ignition points placed yet (0 / {ign.get('n_requested')} selected). Click "
            "Set ignition on map, then click inside the simulation area. No fire spread will be simulated until an "
            "ignition source is provided.")


def assess(mode: str, firms_status: str, cls: Optional[dict], severity: str, ign: dict,
           weather_ok: bool = True) -> dict:
    """Status shown before running (What-If page): code, colour, title, message."""
    sev = severity or "UNKNOWN"
    high = sev in HIGH
    notes = _notes(cls, firms_status)
    if mode == LIVE:
        if firms_status not in ("live", "cached"):
            st_ = ("firms_unavailable", "grey", "NASA FIRMS UNAVAILABLE",
                   "NASA FIRMS data unavailable. Current observed-fire status cannot be confirmed.")
        elif ign["source"] == "OBSERVED_FIRMS":
            st_ = ("observed_fire", "red", "OBSERVED FIRE DETECTION",
                   "Active NASA FIRMS fire detection found. Fire spread simulation initialized from observed "
                   "ignition location(s).")
        elif high:
            st_ = ("high_risk_no_fire", "orange" if sev == "EXTREME" else "yellow",
                   f"{sev} FIRE-WEATHER RISK but NO OBSERVED ACTIVE FIRE",
                   "High fire-weather risk detected, but no active NASA FIRMS fire detection is available. "
                   "No observed fire spread simulated.")
        else:
            st_ = ("no_fire", "green", "NO FIRE",
                   "No active fire detected. Current conditions do not indicate an observed fire. "
                   "No fire spread simulated.")
        if not weather_ok:
            notes.insert(0, "OPENWEATHERMAP UNAVAILABLE: the risk is computed without current weather "
                            "(nothing invented).")
    elif mode == WHATIF:
        if ign["source"] == "HYPOTHETICAL_USER":
            st_ = ("whatif_hypothetical", "blue", "WHAT-IF SIMULATION — HYPOTHETICAL IGNITION",
                   "WHAT-IF SIMULATION: Hypothetical ignition applied.")
        elif ign["source"] == "OBSERVED_FIRMS":
            st_ = ("whatif_observed", "red", "WHAT-IF SIMULATION — OBSERVED IGNITION RETAINED",
                   "WHAT-IF SIMULATION: the observed NASA FIRMS ignition location(s) are kept and spread under the "
                   "scenario weather.")
        elif high:
            st_ = ("whatif_high_no_ignition", "orange", "WHAT-IF — HIGH FIRE RISK, NO FIRE SIMULATED",
                   "WHAT-IF CONDITIONS INDICATE HIGH FIRE RISK, BUT NO IGNITION SOURCE WAS PROVIDED.")
        elif ign.get("awaiting_points"):
            st_ = ("whatif_no_ignition", "grey", "WHAT-IF — NO IGNITION SOURCE", _awaiting_msg(ign))
        else:
            st_ = ("whatif_no_ignition", "grey", "WHAT-IF — NO IGNITION SOURCE",
                   f"No ignition source selected. Weather conditions indicate {sev} risk, but no fire spread will "
                   "be simulated until an ignition source is provided.")
        if ign["source"] == "NONE" and high and not ign.get("awaiting_points"):
            notes.insert(0, f"No ignition source selected. Weather conditions indicate {sev} risk, but no fire "
                            "spread will be simulated until an ignition source is provided.")
    else:
        st_ = ("demo", "grey", "DEMO / OFFLINE MODE", "Synthetic demo inputs; the live APIs are not called.")
    code, color, title, msg = st_
    return {"code": code, "color": color, "title": title, "message": msg, "notes": notes, "severity": sev}


def result_status(mode: str, firms_status: str, cls: Optional[dict], severity: str, ign: dict,
                  ran: bool) -> dict:
    """Status of a run (Spread Simulation): what was, or was not, simulated."""
    sev = severity or "UNKNOWN"
    high = sev in HIGH
    notes = _notes(cls, firms_status)
    if mode == LIVE:
        if ran:
            r = ("OBSERVED FIRE — SIMULATION ACTIVE", "red",
                 "Fire spread simulation initialized from observed NASA FIRMS ignition location(s).")
        elif firms_status not in ("live", "cached"):
            r = ("NASA FIRMS UNAVAILABLE — FIRE PRESENCE UNCONFIRMED", "grey",
                 "NASA FIRMS data unavailable. Current observed-fire status cannot be confirmed. No fire spread "
                 "was simulated and no detections were invented.")
        elif high:
            r = ("HIGH FIRE-WEATHER RISK — NO ACTIVE FIRE", "orange",
                 "Current weather conditions indicate elevated fire potential, but NASA FIRMS returned no active fire "
                 "detection. No observed fire spread was simulated.")
            notes.insert(0, f"Fire-weather risk is {sev}, but no active satellite fire detection was found. The system "
                            "does not simulate an existing fire without an ignition source.")
        else:
            r = ("NO ACTIVE FIRE DETECTED", "green",
                 "NASA FIRMS returned no fire detections within the selected area/time window. No fire spread "
                 "was simulated.")
            if (cls or {}).get("n_in_area"):
                r = (r[0], r[1], "No valid NASA FIRMS ignition inside the selected area (detections on non-fuel "
                                 "cells are not used). No fire spread was simulated.")
        if not ran:
            notes.insert(0, f"Current fire-weather conditions: {sev}.")
            if firms_status in ("live", "cached"):
                notes.insert(1, "No fire spread was simulated because no valid ignition source was detected.")
    else:
        if ran:
            r = ("WHAT-IF SIMULATION — HYPOTHETICAL IGNITION" if ign["source"] == "HYPOTHETICAL_USER"
                 else "WHAT-IF SIMULATION — OBSERVED IGNITION", "blue",
                 "WHAT-IF SIMULATION: Hypothetical ignition applied." if ign["source"] == "HYPOTHETICAL_USER"
                 else "WHAT-IF SIMULATION: observed NASA FIRMS ignition(s) under scenario weather.")
        elif high:
            r = ("WHAT-IF — HIGH FIRE RISK, NO FIRE SIMULATED", "orange",
                 "WHAT-IF CONDITIONS INDICATE HIGH FIRE RISK, BUT NO IGNITION SOURCE WAS PROVIDED.")
        elif ign.get("awaiting_points"):
            r = ("WHAT-IF — NO IGNITION SOURCE", "grey", _awaiting_msg(ign))
        else:
            r = ("WHAT-IF — NO IGNITION SOURCE", "grey",
                 f"No ignition source selected. Weather conditions indicate {sev} risk, but no fire spread will be "
                 "simulated until an ignition source is provided.")
    return {"title": r[0], "color": r[1], "message": r[2], "notes": notes, "ran": ran}


ICONS = {"green": "🟢", "yellow": "🟡", "orange": "🟠", "red": "🔴", "grey": "⚪", "blue": "🔵"}


def status_html(s: dict, source_label: Optional[str] = None) -> str:
    c = COLORS.get(s["color"], COLORS["grey"])
    icon = ICONS.get(s["color"], "")
    notes = "".join(f"<div class='lm-note'>{html.escape(n)}</div>" for n in s.get("notes") or [])
    src = f"<span class='lm-src'>{html.escape(source_label)}</span>" if source_label else ""
    return (f"<div class='lm-status' style='border-color:{c}66;background:{c}14'>"
            f"<div class='lm-title' style='color:{c}'><span>{icon}</span>{html.escape(s['title'])}{src}</div>"
            f"<div class='lm-msg'>{html.escape(s['message'])}</div>{notes}</div>")


# ── UI ────────────────────────────────────────────────────────────────────── #

def render_mode_selector(demo: bool) -> str:
    """SIMULATION MODE [ LIVE REAL-WORLD ] [ LIVE DATA → WHAT-IF ]."""
    mode = current_mode(demo)

    def chip(m):
        on = "on" if mode == m else ""
        return f"<span class='lm-chip {on}'>[ {MODE_NAMES[m]} ]</span>"
    note = ("<div class='lm-desc'>DEMO / OFFLINE MODE is on (sidebar): synthetic inputs, live APIs not called. "
            "Switch <b>Offline / demo mode</b> OFF to use the live modes.</div>") if demo else ""
    st.markdown(f"<div class='lm-modes'><span class='lm-lbl'>SIMULATION MODE</span>{chip(LIVE)}{chip(WHATIF)}"
                f"</div>{note}", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<div class='lm-card {'on' if mode == LIVE else ''}'><b>LIVE REAL-WORLD SIMULATION</b>"
                    f"<div class='lm-desc'>{LIVE_DESC}</div></div>", unsafe_allow_html=True)
        if st.button("LIVE REAL-WORLD SIMULATION", key=f"{CTRL}_mode_live", use_container_width=True,
                     type="primary" if mode == LIVE else "secondary", disabled=demo):
            set_mode(LIVE)
            st.rerun()
    with c2:
        st.markdown(f"<div class='lm-card {'on' if mode == WHATIF else ''}'><b>LIVE DATA → WHAT-IF</b>"
                    f"<div class='lm-desc'>{WHATIF_DESC}</div></div>", unsafe_allow_html=True)
        if st.button("LIVE DATA → WHAT-IF", key=f"{CTRL}_mode_whatif", use_container_width=True,
                     type="primary" if mode == WHATIF else "secondary", disabled=demo):
            set_mode(WHATIF)
            st.session_state[f"_{CTRL}_load_baseline"] = True       # load the latest observation, then unlock
            st.rerun()
    return mode


def _set_sliders(values: dict):
    for k, *_ in INPUTS:
        v = values.get(k)
        if v is not None:
            st.session_state[f"{CTRL}_{k}"] = _round(k, v)


def render_live_controls(mode: str, obs: Optional[dict]) -> dict:
    """Weather controls for the live modes. Returns {"values", "baseline",
    "modified", "changes", "complete"}. LIVE: locked to the observation.
    WHAT-IF: unlocked, starting from the saved REAL BASELINE."""
    if mode == LIVE:
        values = dict((obs or {}).get("values") or {k: None for k, *_ in INPUTS})
        _set_sliders(values)                          # locked controls always show the latest observation
        baseline = obs
        st.markdown("<div class='lm-lock'><span class='lm-tag live'>LIVE OBSERVATION</span>"
                    "<span class='lm-tag'>LOCKED TO LIVE DATA</span></div>", unsafe_allow_html=True)
    else:
        if st.session_state.pop(f"_{CTRL}_load_baseline", False) or BASE_KEY not in st.session_state:
            st.session_state[BASE_KEY] = obs           # the REAL BASELINE (None if OWM is unavailable)
            start = {k: ((obs or {}).get("values") or {}).get(k) for k, *_ in INPUTS}
            _set_sliders({k: (v if v is not None else (st.session_state.get(f"{CTRL}_{k}") or SCENARIO_START[k]))
                          for k, v in start.items()})
            st.session_state[SCN_KEY] = {k: st.session_state.get(f"{CTRL}_{k}") for k, *_ in INPUTS}
        # Streamlit drops a widget's state when a run does not draw it (another page,
        # an early rerun): the user's WHAT-IF values are kept in SCN_KEY and restored.
        saved = st.session_state.get(SCN_KEY) or {}
        for k, *_ in INPUTS:
            if st.session_state.get(f"{CTRL}_{k}") is None:
                b = ((st.session_state.get(BASE_KEY) or {}).get("values") or {}).get(k)
                v = saved.get(k) if saved.get(k) is not None else (b if b is not None else SCENARIO_START[k])
                st.session_state[f"{CTRL}_{k}"] = _round(k, v)
        baseline = st.session_state.get(BASE_KEY)
        # A newer successful observation (refresh, cache expiry, new location) becomes the
        # REAL BASELINE before the controls are drawn. The controls follow it only while
        # the user has not edited them; edited WHAT-IF values are never overwritten. A
        # failed request (obs None) never replaces the baseline.
        updated_note = None
        if obs is not None and newer_observation(baseline, obs):
            current = {k: st.session_state.get(f"{CTRL}_{k}") for k, *_ in INPUTS}
            if baseline:
                edited = is_modified(baseline, scenario_values(current, baseline))
            else:                                   # placeholders only: edited if moved off them
                edited = any(_round(k, current.get(k)) != _round(k, SCENARIO_START[k]) for k, *_ in INPUTS)
            st.session_state[BASE_KEY] = baseline = obs
            if edited:
                updated_note = (f"Baseline updated to the latest real observation (fetched "
                                f"{format_ist(obs.get('fetched_utc'))}). Your WHAT-IF "
                                f"scenario was kept and now differs from the latest baseline.")
            else:
                _set_sliders({k: (v if v is not None else current.get(k)) for k, v in (obs.get("values") or {}).items()})
                st.session_state[SCN_KEY] = {k: st.session_state.get(f"{CTRL}_{k}") for k, *_ in INPUTS}
        if baseline:
            st.success("Baseline loaded from current real-world observations.")
            if updated_note:
                st.info(updated_note)
        else:
            st.warning("OPENWEATHERMAP UNAVAILABLE: no real baseline could be loaded. The values below are "
                       "SCENARIO INPUT only (not observations).")

    locked = mode == LIVE
    tag = "[LIVE]" if locked else "[SCENARIO INPUT]"
    base_vals = (baseline or {}).get("values") or {}
    cols = st.columns(2)
    sliders = {}
    for i, (k, _src, label, unit, lo, hi, step, dec) in enumerate(INPUTS):
        with cols[i % 2]:
            if locked and base_vals.get(k) is None:
                st.markdown(f"<div class='lm-unavail'><b>{label} ({unit}) {tag}</b><br>{UNAVAILABLE}</div>",
                            unsafe_allow_html=True)
                sliders[k] = None
                continue
            v = st.session_state.get(f"{CTRL}_{k}")
            if v is not None and not (lo <= v <= hi):          # never clip an observation silently
                lo, hi = min(lo, v), max(hi, v)
            sliders[k] = st.slider(f"{label} ({unit}) {tag}", lo, hi, step=step, key=f"{CTRL}_{k}", disabled=locked,
                                   format=f"%.{dec}f",
                                   help=("Locked to the live observation. Switch to LIVE DATA → WHAT-IF to modify."
                                         if locked else "Hypothetical value (SCENARIO INPUT)."))
            if k == "wind_from_deg" and sliders[k] is not None:
                st.caption(f"Wind from the {compass(sliders[k])} → pushes fire towards the {compass(sliders[k] + 180)}")
            st.markdown(_control_tag(k, locked, sliders[k], base_vals.get(k), baseline), unsafe_allow_html=True)
    if not locked:
        st.session_state[SCN_KEY] = dict(sliders)
    values = dict(base_vals) if locked else scenario_values(sliders, baseline)
    rows = changes(baseline, values)
    if not locked:
        _render_comparison(rows, baseline)
        for r in rows:
            if r["changed"]:
                st.caption(r["text"])
    return {"values": values, "baseline": baseline, "modified": any(r["changed"] for r in rows), "changes": rows,
            "complete": complete(values)}


def _render_comparison(rows: list, baseline: Optional[dict]):
    if not baseline:
        return
    ts = format_ist(baseline.get("fetched_utc"))
    body = "".join(
        f"<tr class='{'chg' if r['changed'] else ''}'><td>{html.escape(r['label'])}</td>"
        f"<td>{html.escape(_fmt(r['baseline'], r['unit'], r['dec']))}</td>"
        f"<td>{html.escape(_fmt(r['scenario'], r['unit'], r['dec']))}</td>"
        f"<td>{html.escape(r['text'].split('Change: ')[1])}</td></tr>" for r in rows)
    st.markdown(f"<table class='lm-cmp'><tr><th></th><th>REAL BASELINE<br><small>OpenWeatherMap · fetched {ts}"
                f"</small></th><th>WHAT-IF SCENARIO<br><small>SCENARIO INPUT</small></th><th>Change</th></tr>"
                f"{body}</table>", unsafe_allow_html=True)


def _control_tag(key: str, locked: bool, slider_value, base_value, baseline: Optional[dict]) -> str:
    """Per-control source tag: LIVE (locked), REAL BASELINE (WHAT-IF, unchanged) or
    WHAT-IF (changed, with the baseline value next to it)."""
    unit, dec = next((i[3], i[7]) for i in INPUTS if i[0] == key)
    if locked:
        return f"<div class='lm-ctag'><span class='lm-tag live'>LIVE</span> {html.escape(_fmt(base_value, unit, dec))}</div>"
    eff = scenario_values({key: slider_value}, {"values": {key: base_value}}).get(key)
    if baseline and base_value is not None and _round(key, eff) == _round(key, base_value):
        return (f"<div class='lm-ctag'><span class='lm-tag live'>REAL BASELINE</span> "
                f"{html.escape(_fmt(base_value, unit, dec))}</div>")
    ref = f"baseline {_fmt(base_value, unit, dec)}" if (baseline and base_value is not None) else "no real baseline"
    return (f"<div class='lm-ctag'><span class='lm-tag whatif'>WHAT-IF</span> {html.escape(_fmt(eff, unit, dec))}"
            f" <span class='lm-ref'>({html.escape(ref)})</span></div>")


# ── ignition plan shared by the What-If page and Spread Simulation ─────────── #

def plan_ignition(mode: str, setup: dict, focus, cond: dict, detections: Optional[list], firms_status: str,
                  allow_fetch: bool = True) -> dict:
    """Classify the real detections against the simulation area and decide the
    ignition. The LIGHTWEIGHT ignition-point check (OpenStreetMap, cached;
    src/simulation/ignition_site.py) is only loaded when a detection lies inside
    the area or a hypothetical ignition is set. It only decides whether a point
    may ignite - it is never a fire-propagation mask. The coverage is the
    initial simulation domain."""
    from src.simulation.fuel_map import all_fuel
    from src.simulation.ignition_site import IgnitionSiteClassifier
    from src.simulation.local_spread import (ca_inputs_from_conditions, domain_size_from_setup, initial_domain,
                                             limits_from_setup)
    from src.simulation.observed_ignition import classify_detections
    dom = initial_domain(focus, setup["duration_min"], domain_size_m=domain_size_from_setup(setup),
                         limits=limits_from_setup(setup))
    dets = list(detections or []) if firms_status in ("live", "cached") else []
    zone_nf = bool(ca_inputs_from_conditions(cond)["non_fuel"])
    cls = classify_detections(dom, dets, None, zone_non_fuel=zone_nf)
    land = None
    if cls["n_in_area"] or ignition_source(mode, setup) == IGN_HYPOTHETICAL:
        land = IgnitionSiteClassifier.for_domain(dom, allow_fetch=allow_fetch)   # validates ignition points
        cls = classify_detections(dom, dets, land, zone_non_fuel=zone_nf)
    ign = ignition_spec(mode, setup, cls)
    ign["cells"] = [list(rc) if rc else None for rc in (dom.cell_of(float(a), float(b)) for a, b in ign["points"])]
    return {"cls": cls, "ign": ign, "land": land if land is not None else all_fuel((dom.n_rows, dom.n_cols)),
            "domain": dom}


def live_handoff(mode: str, obs: Optional[dict], baseline: Optional[dict], values: dict, rows: list,
                 weather: Optional[dict], ws: dict, fs: dict, detections: list, plan: dict, risk: dict) -> dict:
    """The live part of simulation_config: everything Spread Simulation needs
    to reproduce the run and label its provenance."""
    cls = plan["cls"]
    return {
        "mode": mode, "mode_name": MODE_NAMES[mode],
        "weather": {"source": "OpenWeatherMap", "status": ws.get("mode"), "fetched_utc": ws.get("fetched_utc"),
                    "observed_utc": (weather or {}).get("observed_at"), "error": ws.get("error"),
                    "record": weather, "status_full": ws},
        "baseline": baseline if mode == WHATIF else obs,
        "scenario_values": values, "changes": [{k: r[k] for k in ("label", "baseline", "scenario", "change", "changed",
                                                                  "text")} for r in rows],
        "weather_role": "LIVE WEATHER" if mode == LIVE else "SCENARIO INPUT",
        "firms": {"source": fs.get("source", "NASA FIRMS"), "status": fs.get("mode"), "fetched_utc": fs.get("fetched_utc"),
                  "latest_acq_utc": fs.get("latest_acq_utc"), "n": int(fs.get("n", 0) or 0), "box": fs.get("box"),
                  "days": fs.get("days"), "size_km": fs.get("size_km"), "error": fs.get("error"), "status_full": fs},
        "detections": detections if fs.get("mode") in ("live", "cached") else [],
        "classification": {k: cls[k] for k in ("n_total", "n_in_area", "n_valid", "n_non_fuel", "n_outside", "n_cells")},
        "ignition": plan["ign"],
        "risk": {"severity": risk.get("severity"), "zone_id": risk.get("zone_id"),
                 "zone_risk": risk.get("risk_score"), "ffmc": risk.get("ffmc"), "bui": risk.get("bui"),
                 "fwi": risk.get("fwi")},
    }


def scenario_card(mode: str, values: dict, rows: list, baseline: Optional[dict], firms: Optional[dict] = None,
                  ign: Optional[dict] = None) -> dict:
    """Third observation card: the inputs the simulation will use. firms:
    {"status", "n_in_area"}; ign: live_modes.ignition_spec."""
    def v(k, unit, dec):
        x = values.get(k)
        return UNAVAILABLE if x is None else _fmt(x, unit, dec)
    wd = values.get("wind_from_deg")
    body = [("Temperature", v("temp_c", "°C", 1)), ("Humidity", v("humidity_pct", "%", 0)),
            ("Wind", (v("wind_speed_ms", "m/s", 1) + (f" from {compass(wd)} ({wd:.0f}°)" if wd is not None else "")))]
    if firms is not None:
        obs_fire = (f"{firms.get('n_in_area', 0)} (NASA FIRMS, in the area)"
                    if firms.get("status") in ("live", "cached") else "UNAVAILABLE (none invented)")
        body.append(("Observed fire detections", obs_fire))
    if mode == WHATIF and ign is not None:
        body.append(("Ignition", {"HYPOTHETICAL_USER": f"Hypothetical ignition(s) · {ign.get('placement') or '-'}",
                                  "OBSERVED_FIRMS": "Observed NASA FIRMS ignition(s), retained",
                                  "NONE": "None selected"}[ign["source"]]))
    if mode == LIVE:
        return {"title": "SIMULATION INPUT", "tag": "LIVE OBSERVATION", "rows": body,
                "foot": "LIVE mode: the simulation uses the real observation unchanged (controls locked)."}
    changed = [r for r in rows if r["changed"]]
    if not baseline:
        status = "no real baseline (OpenWeatherMap unavailable)"
    elif changed:
        status = "modifies the baseline: " + "; ".join(
            f"{r['label'].split(' (')[0].lower()} {r['text'].split('Change: ')[1]}" for r in changed)
    else:
        status = "matches the real baseline (no change yet)"
    return {"title": "WHAT-IF SCENARIO", "tag": "SCENARIO INPUT", "rows": body + [("Baseline", status)],
            "foot": "Hypothetical values chosen by you on top of the real baseline. They drive the WHAT-IF "
                    "simulation; they are not observations."}
