"""
geo_fire_map.py - Python side of the geographic simulation map.

The map itself is a Streamlit custom component (components/fire_map/index.html):
one Google Maps instance + canvases per page, created once. Python sends a
payload; on reruns with the same payload the browser does nothing, with a new
one it updates in place (no new map, no new canvas, no reload). The component
sends back user actions: a moved/resized focus area, a Google search result,
map-clicked ignition points, grid/boundary visibility.

Payload contents:
  * focus area + simulation domain geometry (lat/lon, metres, cells);
  * for a run: the CA history in compact form - for every cell that ever burns,
    the step it ignited, the step it burned out, and its visual intensity -
    plus per-step wind and analytics. Play / pause / reset / speed / timeline /
    layer toggles all run in the browser.

Labelling: everything on the canvases is SIMULATED (CA output + visual
effects). Satellite fire detections are drawn as circles of the VIIRS 375 m
footprint and labelled observed (live FIRMS) or synthetic (demo).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from config.config import SYSTEM
from src.simulation.local_spread import (FocusArea, LocalSpreadResult, SimulationDomain, domain_for,
                                         ignition_mask)

_COMPONENT_DIR = Path(__file__).resolve().parent / "components" / "fire_map"
_fire_map = components.declare_component("fire_map", path=str(_COMPONENT_DIR))


# ── helpers ───────────────────────────────────────────────────────────────── #

def hotspots_near(hotspots: Optional[pd.DataFrame], lat: float, lon: float,
                  radius_km: float = 40.0, max_points: int = 150, days: int = 1) -> List[dict]:
    """Recent hotspot detections within radius_km of (lat, lon)."""
    if hotspots is None or len(hotspots) == 0:
        return []
    h = hotspots
    if "acq_date" in h.columns:
        d = pd.to_datetime(h["acq_date"], errors="coerce")
        cutoff = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=days)
        h = h[d >= cutoff]
    if h.empty:
        return []
    dy = (h["latitude"].to_numpy(float) - lat) * 111.32
    dx = (h["longitude"].to_numpy(float) - lon) * 111.32 * math.cos(math.radians(lat))
    dist = np.hypot(dx, dy)
    keep = np.argsort(dist)[:max_points]
    keep = keep[dist[keep] <= radius_km]
    return [{"lat": round(float(h.iloc[int(i)]["latitude"]), 5), "lon": round(float(h.iloc[int(i)]["longitude"]), 5),
             "date": str(h.iloc[int(i)].get("acq_date", "")), "km": round(float(dist[i]), 1)} for i in keep]


def duration_label(minutes: float) -> str:
    if minutes < 1:
        return f"{minutes * 60:.0f} s"
    if minutes < 60 or minutes % 60:
        return f"{minutes:g} min"
    return f"{minutes / 60:g} h"


def _geometry(focus: FocusArea, domain: SimulationDomain) -> Tuple[dict, dict]:
    b, db = focus.bounds, domain.bounds
    f = {"name": focus.name, "lat": focus.lat, "lon": focus.lon, "cell_m": focus.cell_m,
         "n_rows": focus.n_rows, "n_cols": focus.n_cols,
         "width_m": focus.n_cols * focus.cell_m, "height_m": focus.n_rows * focus.cell_m, **b}
    d = {"n_rows": domain.n_rows, "n_cols": domain.n_cols, "margin": domain.margin, **db}
    return f, d


def _uid(payload: dict) -> str:
    blob = json.dumps({k: v for k, v in payload.items() if k != "uid"}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def _playback_seconds_per_step(end_t: float) -> float:
    """Playback pacing at 1x: long runs take ~1 s per CA step, short runs are
    stretched so they still last at least ~8 s on screen."""
    total = float(np.clip(8 + end_t * 1.1, 8, 70))
    return total / max(end_t, 1e-6)


# ── payloads ──────────────────────────────────────────────────────────────── #

def setup_payload(focus: FocusArea, duration_minutes: float, wind_speed_ms: float, wind_from_deg: float,
                  n_ignition: int, placement: str, ignition_points: Sequence[Tuple[float, float]],
                  layers: dict, api_key: str, map_id: str, height: int = 560, search: bool = True) -> dict:
    domain = domain_for(focus, duration_minutes)
    f, d = _geometry(focus, domain)
    preview = ignition_mask(domain, n_ignition, placement, wind_speed_ms, wind_from_deg, ignition_points)
    p = {"mode": "setup", "key": api_key, "mapId": map_id or "DEMO_MAP_ID", "height": height,
         "focus": f, "domain": d, "hasRun": False, "editable": True, "search": search,
         "limits": {"min": SYSTEM.focus_min_m, "max": SYSTEM.focus_max_m},
         "placement": placement, "preview": np.flatnonzero(preview.ravel()).tolist(),
         "ignitionPoints": [list(p) for p in ignition_points or []],
         "wind": [[round(float(wind_speed_ms), 2), round(float(wind_from_deg) % 360, 1)]],
         "metrics": [], "layers": layers, "hotspots": [], "hotspotKind": "synthetic"}
    p["uid"] = _uid(p)
    return p


def sim_payload(focus: FocusArea, duration_minutes: float, result: Optional[LocalSpreadResult],
                wind_speed_ms: float, wind_from_deg: float, hotspots: List[dict], hotspot_kind: str,
                n_ignition: int, placement: str, ignition_points: Sequence[Tuple[float, float]],
                layers: dict, autoplay: bool, api_key: str, map_id: str, height: int = 660) -> dict:
    domain = result.domain if result is not None else domain_for(focus, duration_minutes)
    f, d = _geometry(focus, domain)
    p = {"mode": "sim", "key": api_key, "mapId": map_id or "DEMO_MAP_ID", "height": height,
         "focus": f, "domain": d, "editable": False, "search": False,
         "durationMin": float(duration_minutes), "durationLabel": duration_label(duration_minutes),
         "hotspots": hotspots, "hotspotKind": hotspot_kind, "layers": layers, "autoplay": bool(autoplay),
         "placement": placement, "ignitionPoints": [list(p) for p in ignition_points or []]}
    if result is None:
        preview = ignition_mask(domain, n_ignition, placement, wind_speed_ms, wind_from_deg, ignition_points)
        p.update({"hasRun": False, "stepMin": 0, "endT": 0, "lastStep": 0, "cells": [], "ign": [], "out": [],
                  "inten": [], "nonfuel": False, "preview": np.flatnonzero(preview.ravel()).tolist(),
                  "wind": [[round(float(wind_speed_ms), 2), round(float(wind_from_deg) % 360, 1)]], "metrics": []})
    else:
        ign = result.ignition_step.ravel()
        cells = np.flatnonzero(ign >= 0)
        end_t = result.end_step
        keys = ("minutes", "burning", "burned", "burned_ha", "fire_area_ha", "perimeter_m", "front_distance_m",
                "ros_m_per_min", "intensity_class", "mean_intensity", "max_intensity", "centroid_lat",
                "centroid_lon", "left_focus", "burned_in_focus")
        p.update({
            "hasRun": True, "stepMin": result.step_minutes, "endT": round(end_t, 4),
            "lastStep": int(result.history[-1].step), "secPerStep": round(_playback_seconds_per_step(end_t), 3),
            "cells": cells.tolist(), "ign": ign[cells].astype(int).tolist(),
            "out": result.burnout_step.ravel()[cells].astype(int).tolist(),
            "inten": np.round(result.intensity.ravel()[cells], 2).tolist(),
            "nonfuel": bool(result.non_fuel.all()), "preview": [],
            "wind": [[round(float(s), 2), round(float(dg), 1)] for s, dg in result.wind_schedule],
            "metrics": [{k: x.get(k) for k in keys} for x in result.metrics],
            "landCover": _land_cover_payload(result),
        })
    p["uid"] = _uid(p)
    return p


def _land_cover_payload(result: LocalSpreadResult) -> Optional[dict]:
    """Non-fuel cells by class (sparse, domain cell indices) for the map's
    Land cover layer; the simulation itself already excluded them."""
    lc = getattr(result, "land_cover", None)
    if lc is None:
        return None
    flat = lc.ravel()
    return {"label": result.land_cover_label,
            "cells": {str(int(k)): np.flatnonzero(flat == k).tolist() for k in np.unique(flat) if int(k) != 0}}


def missing_key_card():
    st.markdown("""
    <div class="info-box" style="border-color:rgba(255,107,74,.45);">
      <b style="color:#ff6b4a;">Google Maps API key not set.</b> Add <code>GOOGLE_MAPS_API_KEY=...</code>
      (Maps JavaScript API enabled, billing on) to <code>.env</code> or Streamlit Secrets and restart.
      The simulation, analytics and charts below still run without the map.
    </div>
    """, unsafe_allow_html=True)


def render_fire_map(payload: dict, key: str) -> Optional[dict]:
    """Draw the map component; returns the latest event it sent (or None).
    Events carry a unique `nonce`; callers handle each nonce once."""
    return _fire_map(payload=payload, key=key, default=None)


def new_event(event: Optional[dict], state_key: str) -> Optional[dict]:
    """The event if it has not been handled yet (dedupe by nonce)."""
    if not event or not isinstance(event, dict):
        return None
    nonce = event.get("nonce")
    seen = st.session_state.setdefault("_fire_map_nonces", {})
    if nonce is None or seen.get(state_key) == nonce:
        return None
    seen[state_key] = nonce
    return event
