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
                                         ignition_mask, initial_domain)

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
    d = {"n_rows": domain.n_rows, "n_cols": domain.n_cols, "margin": domain.margin, **db,
         "m_north": domain.m_north, "m_south": domain.m_south, "m_west": domain.m_west, "m_east": domain.m_east}
    return f, d


def _extent_m(focus: FocusArea, domain: SimulationDomain) -> dict:
    """Domain extent around the focus centre in metres (west, east, south, north)."""
    hx, hy = focus.n_cols * focus.cell_m / 2.0, focus.n_rows * focus.cell_m / 2.0
    c = focus.cell_m
    return {"xw": hx + domain.m_west * c, "xe": hx + domain.m_east * c,
            "ys": hy + domain.m_south * c, "yn": hy + domain.m_north * c}


def study_region_payload(region) -> Optional[dict]:
    if region is None:
        return None
    return {"name": region.name, "status": region.status, "label": region.label, "outline": region.outline()}


def land_cover_layer(classes: Optional[np.ndarray], label: str = "", status: str = "", notes=None,
                     summary: str = "") -> Optional[dict]:
    """Sparse non-fuel / unverified cells by class for the map's Land cover layer."""
    if classes is None:
        return None
    DEGRADED_TITLE, UNAVAILABLE_TITLE = "DEGRADED LAND-COVER MODE", "LAND-COVER DATA UNAVAILABLE"  # no module import
    flat = classes.ravel()
    title = {"full": "fused", "degraded": DEGRADED_TITLE, "unavailable": UNAVAILABLE_TITLE,
             "context": "CONTEXT ONLY (not a fire barrier)"}.get(status, "no fuel")
    return {"label": label, "status": status, "statusTitle": title, "summary": summary,
            "cells": {str(int(k)): np.flatnonzero(flat == k).tolist() for k in np.unique(flat) if int(k) != 0}}


def _uid(payload: dict) -> str:
    blob = json.dumps({k: v for k, v in payload.items() if k != "uid"}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def _playback_seconds_per_step(end_t: float) -> float:
    """Legacy pacing at 1x (kept for callers): ~1 s per CA step, at least ~8 s."""
    total = float(np.clip(8 + end_t * 1.1, 8, 70))
    return total / max(end_t, 1e-6)


def playback_seconds_1x(duration_minutes: float) -> float:
    """Wall-clock length of the whole playback at 1x: one simulated minute per
    second, at least 12 s (very short runs) and at most 120 s (long runs).
    Playback speed (1x / 2x / 5x) only changes this display time - never the
    computed fire (the run is computed once, for the selected simulated time)."""
    return float(np.clip(float(duration_minutes), 12.0, 120.0))


# ── payloads ──────────────────────────────────────────────────────────────── #

def setup_payload(focus: FocusArea, duration_minutes: float, wind_speed_ms: float, wind_from_deg: float,
                  n_ignition: int, placement: str, ignition_points: Sequence[Tuple[float, float]],
                  layers: dict, api_key: str, map_id: str, height: int = 560, search: bool = True,
                  hotspots: Optional[List[dict]] = None, hotspot_kind: str = "observed",
                  hotspot_summary: str = "", ignition_kind: str = "hypothetical",
                  observed_points: Optional[Sequence[Tuple[float, float]]] = None,
                  ignition_label: str = "", max_picks: Optional[int] = None,
                  nonfuel_cells: Optional[List[int]] = None, picks_only_for_map_points: bool = False,
                  nonfuel_classes: Optional[dict] = None, land_layer: Optional[dict] = None,
                  study_region: Optional[dict] = None, domain: Optional[SimulationDomain] = None,
                  domain_size: Optional[Tuple[float, float]] = None, domain_independent: bool = False) -> dict:
    """ignition_kind: "hypothetical" (placement / map points, the default),
    "observed" (only `observed_points`: valid NASA FIRMS cells) or "none".
    The map shows the set-up coverage: the initial simulation domain, whose
    land cover is loaded for validating clicked ignition points."""
    domain = domain or initial_domain(focus, duration_minutes)
    f, d = _geometry(focus, domain)
    preview, picks = _ignition_preview(domain, ignition_kind, n_ignition, placement, wind_speed_ms, wind_from_deg,
                                       ignition_points, observed_points, picks_only_for_map_points)
    p = {"mode": "setup", **_pick_rules(max_picks, nonfuel_cells), "key": api_key, "mapId": map_id or "DEMO_MAP_ID", "height": height,
         "focus": f, "domain": d, "hasRun": False, "editable": True, "search": search,
         "limits": {"min": SYSTEM.focus_min_m, "max": SYSTEM.focus_max_m},
         "placement": placement, "preview": preview, "ignitionKind": ignition_kind, "ignitionLabel": ignition_label,
         "ignitionPoints": picks,
         "wind": [[round(float(wind_speed_ms), 2), round(float(wind_from_deg) % 360, 1)]],
         "metrics": [], "layers": layers, "hotspots": _display_hotspots(hotspots), "hotspotKind": hotspot_kind,
         "hotspotSummary": hotspot_summary}
    if domain_size:
        p["domainSize"] = [float(domain_size[0]), float(domain_size[1])]
    p["domainIndependent"] = bool(domain_independent)     # drawn on the map: does not follow the focus area
    if nonfuel_classes:
        p["nonfuelClasses"] = nonfuel_classes
    if land_layer:
        p["landCover"] = land_layer
    if study_region:
        p["studyRegion"] = study_region
    p["uid"] = _uid(p)
    return p


def sim_payload(focus: FocusArea, duration_minutes: float, result: Optional[LocalSpreadResult],
                wind_speed_ms: float, wind_from_deg: float, hotspots: List[dict], hotspot_kind: str,
                n_ignition: int, placement: str, ignition_points: Sequence[Tuple[float, float]],
                layers: dict, autoplay: bool, api_key: str, map_id: str, height: int = 660,
                hotspot_summary: str = "", ignition_kind: str = "hypothetical",
                observed_points: Optional[Sequence[Tuple[float, float]]] = None, ignition_label: str = "",
                max_picks: Optional[int] = None, nonfuel_cells: Optional[List[int]] = None,
                picks_only_for_map_points: bool = False, nonfuel_classes: Optional[dict] = None,
                study_region: Optional[dict] = None, domain: Optional[SimulationDomain] = None,
                hud: Optional[dict] = None) -> dict:
    domain = result.domain if result is not None else (domain or initial_domain(focus, duration_minutes))
    f, d = _geometry(focus, domain)
    preview, picks = _ignition_preview(domain, ignition_kind, n_ignition, placement, wind_speed_ms, wind_from_deg,
                                       ignition_points, observed_points, picks_only_for_map_points)
    p = {"mode": "sim", **_pick_rules(max_picks, nonfuel_cells), "key": api_key, "mapId": map_id or "DEMO_MAP_ID", "height": height,
         "focus": f, "domain": d, "editable": False, "search": False,
         "durationMin": float(duration_minutes), "durationLabel": duration_label(duration_minutes),
         "hotspots": _display_hotspots(hotspots), "hotspotKind": hotspot_kind, "hotspotSummary": hotspot_summary,
         "layers": layers, "autoplay": bool(autoplay), "ignitionKind": ignition_kind, "ignitionLabel": ignition_label,
         "placement": placement, "ignitionPoints": picks}
    if result is None:
        p.update({"hasRun": False, "stepMin": 0, "endT": 0, "lastStep": 0, "cells": [], "ign": [], "out": [],
                  "inten": [], "nonfuel": False, "preview": preview,
                  "wind": [[round(float(wind_speed_ms), 2), round(float(wind_from_deg) % 360, 1)]], "metrics": []})
    else:
        ign = result.ignition_step.ravel()
        cells = np.flatnonzero(ign >= 0)
        end_t = result.end_step
        keys = ("minutes", "burning", "burned", "burned_ha", "fire_area_ha", "perimeter_m", "front_distance_m",
                "ros_m_per_min", "intensity_class", "mean_intensity", "max_intensity", "centroid_lat",
                "centroid_lon", "left_focus", "burned_in_focus", "burned_ha_bandipur",
                "burned_ha_regional_extension", "max_intensity_kw_m")
        p.update({
            "hasRun": True, "stepMin": result.step_minutes, "endT": round(end_t, 4),
            "lastStep": int(result.history[-1].step),
            "secPerStep": round(playback_seconds_1x(duration_minutes) / max(end_t, 1e-6), 4),
            "playback1x": round(playback_seconds_1x(duration_minutes), 2),
            "model": result.params.get("model", "legacy-ca"),
            "extinguishedAt": result.params.get("extinguished_at_min"),
            "cells": cells.tolist(), "ign": ign[cells].astype(int).tolist(),
            "out": result.burnout_step.ravel()[cells].astype(int).tolist(),
            "inten": np.round(result.intensity.ravel()[cells], 2).tolist(),
            "nonfuel": bool(result.non_fuel.all()), "preview": [],
            "wind": [[round(float(s), 2), round(float(dg), 1)] for s, dg in result.wind_schedule],
            "metrics": [{k: x.get(k) for k in keys} for x in result.metrics],
            "landCover": _land_cover_payload(result),
            "expansions": [{"step": e["step"], "minutes": e["minutes"], "new": e["new"]} for e in result.expansions],
            "initialDomain": _extent_m(focus, result.initial_domain) if result.initial_domain is not None else None,
            "extentLimit": ({**result.extent_limit, "step": round(result.extent_limit["step"], 3)}
                            if result.extent_limit else None),
        })
        if getattr(result, "ignition_time", None) is not None:
            # exact simulated ignition / burn-out time (in frames) of every burned cell: smooth playback
            fm = result.step_minutes
            it = result.ignition_time.ravel()[cells] / fm
            ot = result.burnout_time.ravel()[cells]
            ot = np.where(ot >= 0, ot / fm, -1.0)
            p["ignT"] = np.round(it, 3).tolist()
            p["outT"] = np.round(ot, 3).tolist()
        reg = result.study_region
        if reg and reg.get("crossing_step") is not None and not reg.get("ignition_outside_region"):
            p["crossing"] = {"step": reg["crossing_step"], "minutes": reg["crossing_minutes"],
                             "lat": reg.get("crossing_lat"), "lon": reg.get("crossing_lon")}
        if result.inside_region is not None:
            outside = (result.ignition_step >= 0) & ~result.inside_region
            p["outsideBurned"] = np.flatnonzero(outside.ravel()).tolist()
    if nonfuel_classes:
        p["nonfuelClasses"] = nonfuel_classes
    if study_region:
        p["studyRegion"] = study_region
    for k, v in (hud or {}).items():          # concise HUD: mode, ML risk, ignition, report ID, stale-data note
        if v is not None:
            p[k] = v
    p["uid"] = _uid(p)
    return p


def _display_hotspots(hotspots: Optional[List[dict]]) -> List[dict]:
    """Copies of the detection markers with their observation time shown in IST
    (the data layer's UTC value is not modified)."""
    from src.utils.timezone import format_ist
    return [{**h, "date": format_ist(h["date"], "dot", missing=str(h["date"]))} if h.get("date") else dict(h)
            for h in (hotspots or [])]


def _pick_rules(max_picks: Optional[int], nonfuel_cells: Optional[List[int]]) -> dict:
    """Map-click rules for a WHAT-IF hypothetical ignition (absent = original behaviour)."""
    out = {}
    if max_picks:
        out["maxPicks"] = int(max_picks)
    if nonfuel_cells:
        out["nonfuelCells"] = [int(i) for i in nonfuel_cells]
    return out


def _ignition_preview(domain, kind: str, n_ignition: int, placement: str, wind_speed_ms: float,
                      wind_from_deg: float, ignition_points, observed_points,
                      picks_only_for_map_points: bool = False) -> Tuple[list, list]:
    """(preview cell indices, editable map points) for the ignition markers.
    picks_only_for_map_points (WHAT-IF): map points are shown only with the Map
    points placement, and Map points with no point shows no marker (nothing
    ignites, no centre fallback)."""
    if kind == "none":
        return [], []
    if picks_only_for_map_points and kind == "hypothetical":
        if placement == "Map points":
            pts = [list(q) for q in ignition_points or []]
            mask = ignition_mask(domain, 0, "Map points", wind_speed_ms, wind_from_deg, pts, strict_points=True)
            return np.flatnonzero(mask.ravel()).tolist(), pts
        mask = ignition_mask(domain, n_ignition, placement, wind_speed_ms, wind_from_deg)
        return np.flatnonzero(mask.ravel()).tolist(), []
    if kind == "observed":
        mask = ignition_mask(domain, 0, "Map points", wind_speed_ms, wind_from_deg, observed_points or [],
                             strict_points=True)
        return np.flatnonzero(mask.ravel()).tolist(), []
    mask = ignition_mask(domain, n_ignition, placement, wind_speed_ms, wind_from_deg, ignition_points)
    return np.flatnonzero(mask.ravel()).tolist(), [list(q) for q in ignition_points or []]


def _land_cover_payload(result: LocalSpreadResult) -> Optional[dict]:
    """Non-fuel cells by class (sparse, domain cell indices) for the map's
    Land cover layer; the simulation itself already excluded them."""
    lc = getattr(result, "land_cover", None)
    if lc is None:
        return None
    summary = ""
    conf = getattr(result, "land_confidence", None)
    if conf is not None:
        from src.landcover.fusion import CONF_NAMES
        summary = "Confidence (source agreement): " + ", ".join(
            f"{CONF_NAMES[k]} {int((conf == k).sum())}" for k in (3, 2, 1) if (conf == k).any())
    return land_cover_layer(lc, result.land_cover_label, getattr(result, "land_status", ""), summary=summary)


_SEV_CODE = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "EXTREME": 3}


def region_payload(twin, hotspots: List[dict], hotspot_kind: str, hotspot_summary: str, source_note: str,
                   api_key: str, map_id: str, height: int = 560) -> dict:
    """Command Center overview: model risk per grid zone (MODEL PREDICTION) and
    satellite detections (OBSERVED, or synthetic in demo mode) on Google
    satellite. Only display data - nothing here changes the risk values."""
    snap = twin.current_snapshot
    g = snap.processed_grid
    r = twin.region
    risk = np.asarray(snap.risk_scores, float)
    engine = getattr(twin, "alert_engine", None)
    sev = [_SEV_CODE.get(engine.classify(float(v)), 0) if engine is not None else 0 for v in risk]
    fwi = g["fwi"].to_numpy(float) if "fwi" in g else np.full(len(g), np.nan)
    act = g["active_fire_nearby"].astype(bool).to_numpy() if "active_fire_nearby" in g else np.zeros(len(g), bool)
    p = {"mode": "region", "key": api_key, "mapId": map_id or "DEMO_MAP_ID", "height": height,
         "region": {"name": r.name, "south": r.min_lat, "north": r.max_lat, "west": r.min_lon, "east": r.max_lon},
         "res": float(r.grid_resolution_deg), "alerts": int(len(snap.alerts)),
         "zones": {"lat": np.round(g["latitude"].to_numpy(float), 4).tolist(),
                   "lon": np.round(g["longitude"].to_numpy(float), 4).tolist(),
                   "risk": np.round(risk, 3).tolist(), "sev": sev,
                   "fwi": [None if not np.isfinite(v) else round(float(v), 1) for v in fwi],
                   "act": [int(v) for v in act], "ids": g["zone_id"].astype(str).tolist()},
         "hotspots": _display_hotspots(hotspots), "hotspotKind": hotspot_kind, "hotspotSummary": hotspot_summary,
         "sourceNote": source_note, "timestamp": str(snap.timestamp)}
    p["uid"] = _uid(p)
    return p


def record_map_status(event: Optional[dict]):
    """The browser reports whether Google Maps loaded (once per tab); kept for
    the Data & System Status panel."""
    if isinstance(event, dict) and event.get("kind") == "mapstatus":
        if (st.session_state.get("gmaps_status") or {}).get("nonce") == event.get("nonce"):
            return
        from datetime import datetime, timezone
        st.session_state["gmaps_status"] = {"ok": bool(event.get("ok")), "vector": bool(event.get("vector")),
                                            "error": event.get("error"), "nonce": event.get("nonce"),
                                            "utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}


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
    ev = _fire_map(payload=payload, key=key, default=None)
    record_map_status(ev)
    return ev


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
