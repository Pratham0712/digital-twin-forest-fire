"""
simulation_report.py - technical report of ONE completed simulation (PDF / DOCX / printable HTML).

Everything comes from the snapshot taken when the run completed (`build_snapshot`):
the same conditions, observations and results the dashboard used - nothing is
fetched again. Values are labelled OBSERVED (satellite / weather station data),
PREDICTED (ML model), DERIVED (computed indices), SCENARIO INPUT (user-chosen)
or SIMULATED (fire-spread model). Missing data is stated as missing; charts are
drawn only from recorded series.

Dependencies: reportlab (PDF), python-docx (DOCX), matplotlib (charts, optional:
without it the report states that charts are unavailable).
"""
from __future__ import annotations

import io
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

TITLE = "Digital Twin Framework for Forest Fire Prediction"
REPORT_TITLE = "Forest Fire Simulation and Predictive Risk Assessment"
DISCLAIMER = ("This report supports situational awareness and planning. It does not replace official fire monitoring, "
              "forest-department assessments or emergency instructions.")
# Interpretation bands for the Fire Weather Index components (general fire-danger guidance)
FWI_BANDS = {
    "ffmc": ([(70, "low ignition potential"), (85, "moderate ignition potential"), (90, "high ignition potential")],
             "very high to extreme ignition potential", "Fine Fuel Moisture Code - dryness of litter and fine fuels"),
    "dmc": ([(21, "low"), (41, "moderate"), (61, "high")], "very high",
            "Duff Moisture Code - dryness of loosely compacted organic layers"),
    "dc": ([(200, "low"), (300, "moderate"), (500, "high")], "very high",
           "Drought Code - deep, compact organic layers / seasonal drought"),
    "bui": ([(20, "low"), (40, "moderate"), (60, "high")], "very high",
            "Buildup Index - total fuel available for combustion"),
    "fwi": ([(5, "low"), (10, "moderate"), (20, "high"), (30, "very high")], "extreme",
            "Fire Weather Index - potential fire intensity"),
}
COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def compass(deg) -> str:
    return COMPASS[int(round((float(deg) % 360) / 22.5)) % 16]


def _f(v, nd=1):
    try:
        x = float(v)
        return None if not math.isfinite(x) else round(x, nd)
    except (TypeError, ValueError):
        return None


def build_snapshot(result, cond: dict, setup: dict, mode: str, scenario: Optional[dict], live: Optional[dict],
                   twin=None, plan: Optional[dict] = None, actor: str = "") -> dict:
    """Reproducible snapshot of a completed run (JSON-serialisable)."""
    from src.dashboard.geo_fire_map import duration_label
    p = result.params
    fin = result.final
    m = result.metrics
    loc = setup.get("location") or {}
    risk_score = cond.get("risk_score")
    cat = None
    if twin is not None and risk_score is not None and np.isfinite(risk_score):
        try:
            cat = twin.alert_engine.classify(float(risk_score))
        except Exception:
            cat = None
    mc = getattr(twin, "model_choice", None)
    # weather actually used by the run
    if live and live.get("mode") == "live":
        rec = (live.get("weather") or {}).get("record") or {}
        weather = {"label": "OBSERVED (OpenWeatherMap)", "temp_c": _f(rec.get("temperature_c")),
                   "humidity_pct": _f(rec.get("humidity_pct"), 0), "wind_speed_ms": _f(rec.get("wind_speed_ms")),
                   "wind_from_deg": _f(rec.get("wind_deg"), 0), "source": "OpenWeatherMap",
                   "timestamp_utc": (live.get("weather") or {}).get("observed_utc") or
                   (live.get("weather") or {}).get("fetched_utc"),
                   "status": (live.get("weather") or {}).get("status")}
    elif scenario:
        weather = {"label": "SCENARIO INPUT (user-chosen what-if weather)", "temp_c": _f(scenario.get("temp_c")),
                   "humidity_pct": _f(scenario.get("humidity_pct"), 0), "wind_speed_ms": _f(scenario.get("wind_speed_ms")),
                   "wind_from_deg": _f(scenario.get("wind_from_deg"), 0), "source": "What-If scenario",
                   "timestamp_utc": None, "status": "scenario"}
    else:
        weather = {"label": "MODEL GRID ZONE (live or demo, see status)", "temp_c": _f(cond.get("temp_c")),
                   "humidity_pct": _f(cond.get("humidity_pct"), 0), "wind_speed_ms": _f(cond.get("wind_speed_ms")),
                   "wind_from_deg": _f(cond.get("wind_from_deg"), 0), "source": "regional twin zone "
                   + str(cond.get("zone_id")), "timestamp_utc": None,
                   "status": "demo" if getattr(twin, "offline", False) else "live/cached"}
    w_used = result.wind_schedule[0] if result.wind_schedule else (0.0, 0.0)
    # FIRMS
    firms = {"status": "not used (no LIVE observation hand-off)", "n_total": 0, "n_in_area": 0, "n_valid": 0,
             "detections": [], "fetched_utc": None, "latest_acq_utc": None}
    if live:
        fs = live.get("firms") or {}
        cls = (plan or {}).get("cls") or {}
        firms = {"status": fs.get("status"), "fetched_utc": fs.get("fetched_utc"),
                 "latest_acq_utc": fs.get("latest_acq_utc"), "n_total": int(fs.get("n", 0) or 0),
                 "n_in_area": int(cls.get("n_in_area", 0) or 0), "n_valid": int(cls.get("n_valid", 0) or 0),
                 "detections": [{k: d.get(k) for k in ("lat", "lon", "acq_utc", "date", "confidence", "conf",
                                                       "satellite", "sat", "instrument", "frp")
                                 if d.get(k) is not None}
                                for d in (live.get("detections") or [])[:25]]}
    # historical (training-season fire climatology of the zone), if the model has one
    hist = {"available": False, "note": "Historical fire records are not available for this zone in this session."}
    try:
        clim = getattr(twin, "climatology", None)
        if clim is not None and clim.table is not None and not clim.table.empty:
            import pandas as pd
            v = clim.transform(pd.DataFrame({"zone_id": [cond.get("zone_id")], "latitude": [cond.get("zone_lat")],
                                             "longitude": [cond.get("zone_lon")]}))
            hist = {"available": True, "zone_fire_rate": _f(v["zone_clim"].iloc[0], 4),
                    "neighbour_fire_rate": _f(v["nbr_clim"].iloc[0], 4),
                    "source": f"NASA FIRMS archive, model training seasons ({getattr(mc, 'climatology_file', '-')})",
                    "note": "Per-cell historical fire rate learned from the training seasons (DERIVED)."}
    except Exception:
        pass
    series = [{"minutes": x["minutes"], "burning_ha": round(x["burning"] * result.focus.cell_m ** 2 / 1e4, 3),
               "burned_ha": x["burned_ha"], "front_m": x["front_distance_m"], "ros": x["ros_m_per_min"],
               "burning_cells": int(x["burning"]), "burned_cells": int(x["burned"]),
               "perimeter_m": x.get("perimeter_m")} for x in m]
    prediction = _prediction_block(cond, twin, cat)
    peak = max(m, key=lambda x: x["ros_m_per_min"]) if m else {}
    clock = p.get("clock") or {"requested_min": float(result.duration_minutes),
                               "simulated_min": float(m[-1]["minutes"]) if m else 0.0,
                               "requested_s": float(result.duration_minutes) * 60.0,
                               "simulated_s": (float(m[-1]["minutes"]) if m else 0.0) * 60.0,
                               "frame_s": float(result.step_minutes) * 60.0, "n_frames": max(0, len(m) - 1),
                               "status": "completed"}
    import hashlib
    import json as _json
    sim_id = "SIM-" + hashlib.sha1(_json.dumps(
        [p.get("seed"), p.get("placement"), result.duration_minutes, result.focus.lat, result.focus.lon,
         list(map(list, result.wind_schedule or [])), clock.get("started_utc")], default=str).encode()).hexdigest()[:10].upper()
    dom0 = result.initial_domain or result.domain
    snap = {
        "report_id": p.get("report_id"), "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "actor": actor, "title": TITLE, "mode": mode,
        "region": getattr(getattr(twin, "region", None), "name", None),
        "location": {"name": loc.get("name"), "lat": float(loc.get("lat", result.focus.lat)),
                     "lon": float(loc.get("lon", result.focus.lon))},
        "focus": {"width_m": result.focus.n_cols * result.focus.cell_m, "height_m": result.focus.n_rows * result.focus.cell_m,
                  "cell_m": result.focus.cell_m},
        "domain_initial": {"width_m": dom0.width_m, "height_m": dom0.height_m},
        "domain_final": {"width_m": result.domain.width_m, "height_m": result.domain.height_m,
                         "expansions": len(result.expansions)},
        "ignition": {"kind": "OBSERVED (NASA FIRMS)" if mode == "LIVE" else "HYPOTHETICAL (user)",
                     "points": (plan or {}).get("ign", {}).get("points") or setup.get("ignition_points") or [],
                     "placement": setup.get("placement"), "checks": p.get("ignition_checks") or []},
        "weather": weather,
        "wind_used": {"speed_ms": round(float(w_used[0]), 2), "from_deg": round(float(w_used[1]), 1)},
        "simulation_id": sim_id, "clock": clock, "prediction": prediction,
        "land_cover": {"label": getattr(result, "land_cover_label", None), "status": getattr(result, "land_status", "")},
        "fuel_source": p.get("fuel_source"),
        "risk": {"score_pct": None if risk_score is None or not np.isfinite(risk_score) else round(100 * float(risk_score)),
                 "category": cat, "zone": cond.get("zone_id"), "zone_distance_km": cond.get("zone_distance_km"),
                 "model": getattr(mc, "label", None), "model_file": getattr(mc, "model_file", None)},
        "fwi": {k: _f(cond.get(k)) for k in ("ffmc", "dmc", "dc", "bui", "fwi")},
        "firms": firms, "historical": hist,
        "results": {"duration_min": result.duration_minutes, "duration_label": duration_label(result.duration_minutes),
                    "burned_ha": fin.get("burned_ha", 0.0), "fire_area_ha": fin.get("fire_area_ha", 0.0),
                    "burning_ha": round(fin.get("burning", 0) * result.focus.cell_m ** 2 / 1e4, 3),
                    "perimeter_m": fin.get("perimeter_m"), "front_distance_m": fin.get("front_distance_m"),
                    "peak_ros": max((x["ros_m_per_min"] for x in m), default=0.0),
                    "peak_ros_at_min": peak.get("minutes"), "final_ros": m[-1]["ros_m_per_min"] if m else 0.0,
                    "burning_cells": int(fin.get("burning", 0)), "burned_cells": int(fin.get("burned", 0)),
                    "cell_area_ha": round(result.focus.cell_m ** 2 / 1e4, 4),
                    "spread_towards": compass(w_used[1] + 180), "extinguished_at_min": p.get("extinguished_at_min"),
                    "left_focus": bool(fin.get("left_focus")), "max_kw_m": max((x.get("max_intensity_kw_m") or 0)
                                                                              for x in m) if m else 0},
        "model": {k: p.get(k) for k in ("model", "fuel_model", "ros_head_m_per_min", "ros_flank_m_per_min",
                                        "ros_back_m_per_min", "length_to_breadth", "isi", "frame_minutes")},
        "terrain": result.terrain_source,
        "series": series,
        "expansion_minutes": [float(e.get("minutes", 0)) for e in (result.expansions or [])],
    }
    return snap


def _prediction_block(cond: dict, twin, category) -> dict:
    """The single model prediction the dashboard showed for this location (same snapshot,
    no new inference): probability, risk index, category, thresholds, model version,
    timestamp, grid zone and the feature values the model received."""
    from src.ml_models.prediction_service import SEVERITY_BANDS
    out = {"available": False, "probability": _f(cond.get("fire_probability"), 6),
           "risk_index": _f(cond.get("risk_score"), 4), "category": category,
           "model_version": cond.get("model_version"), "predicted_utc": cond.get("predicted_utc"),
           "zone": cond.get("zone_id"), "zone_lat": _f(cond.get("zone_lat"), 4), "zone_lon": _f(cond.get("zone_lon"), 4),
           "zone_distance_km": cond.get("zone_distance_km"), "resolution": None, "thresholds": None,
           "bands": [[float(c), lab] for c, lab in SEVERITY_BANDS], "features": []}
    if out["risk_index"] is not None:
        out["available"] = True
    reg = getattr(twin, "region", None)
    if reg is not None and getattr(reg, "grid_resolution_deg", None):
        out["resolution"] = f"{reg.grid_resolution_deg:g}° grid zone (about {reg.grid_resolution_deg * 111:.0f} km)"
    ri = getattr(twin, "risk_index", None)
    if ri is not None:
        out["thresholds"] = {"moderate_from_probability": ri.t_moderate, "high_from_probability": ri.t_high,
                             "extreme_from_probability": ri.t_extreme}
    try:
        from src.ml_models.prediction_service import feature_schema
        cols = feature_schema(twin.ml_model) or []
        g = twin.current_snapshot.processed_grid
        row = g[g["zone_id"].astype(str) == str(cond.get("zone_id"))].iloc[0]
        out["features"] = [[c, _f(row[c], 3) if c in row.index else None] for c in cols]
        out["scenario_inputs"] = {lab: (None if _f(row.get(c)) is None else f"{_f(row.get(c)):g}{u}")
                                  for c, lab, u in (("wx_temperature_c", "temperature", " °C"),
                                                    ("wx_humidity_pct", "humidity", " %"),
                                                    ("wx_wind_speed_ms", "wind", " m/s"),
                                                    ("wx_precipitation_mm", "rain", " mm"),
                                                    ("ffmc", "FFMC", ""), ("fwi", "FWI", ""))}
        out["out_of_range"] = _out_of_range(dict(out["features"]), getattr(twin.model_choice, "model_file", ""))
    except Exception:
        pass
    _add_risk_score(out, cond, twin)
    return out


def _add_risk_score(out: dict, cond: dict, twin) -> None:
    """ML-based Fire Risk Score (risk_score.py): percentile of the stored model probability - computed from the
    snapshot's prediction, no new inference."""
    from src.ml_models.risk_score import RiskScore, category
    mf = getattr(getattr(twin, "model_choice", None), "model_file", None) or "xgboost_real.json"
    rs = RiskScore.load(mf)
    p = out.get("probability")
    if rs is None or p is None:
        out["risk_score"] = None
        return
    sc = cond.get("ml_risk_score")
    sc = float(sc) if sc is not None and np.isfinite(sc) else float(rs.score(p))
    out["risk_score"] = round(sc, 1)
    out["risk_score_category"] = category(sc)
    out["risk_score_meta"] = {k: rs.meta.get(k) for k in ("definition", "reference", "validation_season", "validation",
                                                          "held_out_roc_auc", "held_out_pr_auc", "held_out_base_rate",
                                                          "bands")}
    f = dict(out.get("features") or [])
    try:
        from src.ml_models.fwi_reference import fwi_percentile
        out["fwi_percentile"] = None if f.get("fwi") is None else round(float(fwi_percentile(f["fwi"])), 1)
    except Exception:
        out["fwi_percentile"] = None
    out["factors"] = _factors(f, out)


def _factors(f: dict, out: dict) -> list:
    """Plain statements of what drives this prediction (facts from the model's inputs, no invented weights)."""
    res = []
    hist = [f.get(k) or 0 for k in ("fire_lag1", "fire_7d", "fire_10d", "nbr3_lag1", "nbr5_7d")]
    if f.get("fire_7d") is not None:
        res.append("Recent satellite fire activity (the model's strongest input): "
                   + ("none detected in or around the zone in the last 10 days - this keeps the prediction low."
                      if not any(hist) else
                      f"{int(f.get('fire_10d') or 0)} detection day(s) in the zone and {int(f.get('nbr5_7d') or 0)} in "
                      "neighbouring zones in the last 10 days - this raises the prediction."))
    if f.get("ffmc") is not None:
        fp = out.get("fwi_percentile")
        res.append(f"Fire weather: FFMC {f['ffmc']:g}, FWI {f.get('fwi', 0):g}"
                   + (f" (FWI higher than {fp:.0f}% of historical fire-season days)" if fp is not None else "")
                   + " - drier, hotter, windier weather raises the prediction; humidity and rain lower it.")
    if f.get("zone_clim") is not None:
        res.append(f"Zone fire history: {f['zone_clim'] * 100:.1f}% of fire-season days in this zone had a "
                   "satellite-detected fire in the training seasons.")
    if f.get("month") is not None:
        mo = int(f["month"])
        res.append("Season: " + (f"month {mo} is inside the model's training season (January-May)." if 1 <= mo <= 5 else
                                 f"month {mo} is outside the model's training season (January-May); the model treats "
                                 "it like the nearest trained month."))
    return res


def _out_of_range(values: dict, model_file: str) -> list:
    """Model inputs outside the 1st-99th percentile of the training rows (models/feature_ranges_*.json)."""
    import json
    from pathlib import Path
    from config.config import MODELS_DIR
    path = Path(MODELS_DIR) / ("feature_ranges_real.json" if model_file == "xgboost_real.json" else "-")
    if not path.exists():
        return []
    rng = json.loads(path.read_text()).get("ranges", {})
    names = {"wx_temperature_c": "temperature", "wx_humidity_pct": "humidity", "wx_wind_speed_ms": "wind speed",
             "ffmc": "FFMC", "fwi": "FWI", "month": "month", "bui": "BUI", "dc": "DC", "dmc": "DMC"}
    out = []
    for c, lab in names.items():
        v = values.get(c)
        if v is None or c not in rng:
            continue
        lo, hi = rng[c]
        if v < lo or v > hi:
            out.append([lab, float(v), float(lo), float(hi)])
    return out




# ── formatting helpers ──────────────────────────────────────────────────────

def _ist(v) -> str:
    if not v:
        return "not available"
    from src.utils.timezone import format_ist
    return format_ist(v)


def _num(v, fmt="{:.1f}", unit="", missing="not available"):
    x = _f(v, 6)
    return missing if x is None else fmt.format(x) + (f" {unit}" if unit else "")


def _dur(minutes) -> str:
    m = float(minutes or 0)
    if m < 1:
        return f"{m * 60:.0f} s"
    return f"{m:g} min ({m * 60:,.0f} s)" if m < 120 else f"{m / 60:g} h ({m * 60:,.0f} s)"


def fwi_interpretation(key: str, value) -> str:
    v = _f(value, 3)
    if v is None:
        return "not calculated for this zone"
    bands, top, _ = FWI_BANDS[key]
    for lim, label in bands:
        if v < lim:
            return label
    return top


def _banner(snap) -> str:
    return ("SIMULATION ONLY - hypothetical scenario, not a confirmed fire" if snap["mode"] != "LIVE"
            else "LIVE - ignition from NASA FIRMS satellite detections; spread is SIMULATED")


# ── charts (only from recorded series / computed values) ────────────────────

PALETTE = {"navy": "#13233a", "orange": "#e8710a", "red": "#b3401a", "amber": "#f2a33a", "blue": "#2f6db3",
           "grey": "#6b7785", "light": "#e9eef4"}


def _fig_bytes(fig) -> bytes:
    import matplotlib.pyplot as plt
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=170, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return b.getvalue()


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=9.5, loc="left", color=PALETTE["navy"], fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.grid(alpha=.28, lw=.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


_CHART_CACHE: "OrderedDict[str, Dict[str, bytes]]" = None
_CHART_LOCK = __import__("threading").Lock()
_CHART_CACHE_MAX = 16
_DRAW_LOCK = __import__("threading").Lock()


_DOC_CACHE: "OrderedDict[tuple, bytes]" = None
_DOC_LOCK = __import__("threading").Lock()
_DOC_CACHE_MAX = 8
EXPORT_FORMATS = {"pdf": ("application/pdf", ".pdf"),
                  "docx": ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx")}


def export_document(snap: dict, kind: str) -> bytes:
    """The ONE requested report file ("pdf" or "docx"), built from the stored simulation snapshot only - no new
    simulation, model run, database query or API call. Built at most once per identical snapshot and format (a
    double click or a second download reuses it); the other format is never built."""
    global _DOC_CACHE
    from collections import OrderedDict
    if kind not in EXPORT_FORMATS:
        raise ValueError(f"unsupported export format: {kind}")
    key = (_chart_key(snap), kind)
    with _DOC_LOCK:                                       # also serialises concurrent clicks: built once
        if _DOC_CACHE is None:
            _DOC_CACHE = OrderedDict()
        hit = _DOC_CACHE.get(key)
        if hit is not None:
            _DOC_CACHE.move_to_end(key)
            return hit
        data = render_pdf(snap) if kind == "pdf" else render_docx(snap)
        _DOC_CACHE[key] = data
        while len(_DOC_CACHE) > _DOC_CACHE_MAX:
            _DOC_CACHE.popitem(last=False)
        return data


def prewarm_charts(snap: dict) -> None:
    """Draw a finished simulation's report charts in a background thread, so a later PDF / Word export only
    assembles the files. Never raises; the result is exactly what charts() would draw on demand."""
    import threading

    def run():
        try:
            charts(snap)
        except Exception:
            pass
    threading.Thread(target=run, daemon=True, name="report-charts").start()


def _chart_key(snap: dict) -> str:
    """Content hash of everything charts() reads, so a cached image set is reused only for identical inputs."""
    import hashlib
    import json as _json
    keys = ("report_id", "generated_utc", "series", "clock", "results", "model", "fwi", "prediction", "risk",
            "expansion_minutes", "weather", "wind_used", "mode")
    blob = _json.dumps({k: snap.get(k) for k in keys}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def charts(snap: dict) -> Dict[str, bytes]:
    """The report's chart images, drawn once per identical snapshot: the HTML preview / print view, the PDF and
    the Word file all embed the same charts, so they are rendered once instead of once per format (matplotlib
    is most of a report's build time). Bounded in-process cache; the bytes are exactly those of _draw_charts."""
    global _CHART_CACHE
    from collections import OrderedDict
    key = _chart_key(snap)
    with _CHART_LOCK:
        if _CHART_CACHE is None:
            _CHART_CACHE = OrderedDict()
        hit = _CHART_CACHE.get(key)
        if hit is not None:
            _CHART_CACHE.move_to_end(key)
            return dict(hit)
    with _DRAW_LOCK:                                      # pyplot is not thread-safe: one drawing at a time
        with _CHART_LOCK:
            hit = _CHART_CACHE.get(key)                   # drawn meanwhile (e.g. by prewarm_charts)
        if hit is not None:
            return dict(hit)
        out = _draw_charts(snap)
    with _CHART_LOCK:
        _CHART_CACHE[key] = dict(out)
        while len(_CHART_CACHE) > _CHART_CACHE_MAX:
            _CHART_CACHE.popitem(last=False)
    return out


def _draw_charts(snap: dict) -> Dict[str, bytes]:
    """Four analytical charts. Time-series charts need >= 2 recorded frames; the bar
    charts use the run's model rates / FWI / prediction values. Nothing is interpolated."""
    out: Dict[str, bytes] = {}
    s = snap.get("series") or []
    if len(s) < 2:
        return out
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return out
    t = [x["minutes"] for x in s]
    dur = float((snap.get("clock") or {}).get("requested_min") or t[-1])
    ext = snap["results"].get("extinguished_at_min")
    # 1 - area
    fig, ax = plt.subplots(figsize=(7.4, 2.55))
    ax.fill_between(t, [x["burned_ha"] for x in s], color=PALETTE["red"], alpha=.18, lw=0)
    ax.plot(t, [x["burned_ha"] for x in s], color=PALETTE["red"], lw=2.0, label="Cumulative burned area")
    ax.plot(t, [x["burning_ha"] for x in s], color=PALETTE["orange"], lw=1.8, ls="--", label="Active burning area")
    for k, em in enumerate(snap.get("expansion_minutes") or []):
        ax.axvline(em, color=PALETTE["grey"], lw=.8, ls=":", label="Domain expansion" if k == 0 else None)
    if ext is not None:
        ax.axvline(float(ext), color=PALETTE["navy"], lw=1, ls="-.", label="Fire activity ended")
    ax.set_xlim(0, dur)
    ax.set_ylim(bottom=0)
    _style(ax, "Burned area and active burning area over time", "Simulated time (min)", "Area (ha)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    out["area"] = _fig_bytes(fig)
    # 2 - front distance + ROS
    fig, ax = plt.subplots(figsize=(7.4, 2.55))
    ax.plot(t, [x["front_m"] for x in s], color=PALETTE["blue"], lw=2.0, label="Fire-front distance (m)")
    ax2 = ax.twinx()
    ax2.plot(t, [x["ros"] for x in s], color=PALETTE["orange"], lw=1.4, marker="o", ms=2.2,
             label="Rate of spread (m/min)")
    pk = max(range(len(s)), key=lambda i: s[i]["ros"])
    if s[pk]["ros"] > 0:
        ax2.annotate(f"peak {s[pk]['ros']:.1f} m/min", (t[pk], s[pk]["ros"]), fontsize=7, color=PALETTE["orange"],
                     xytext=(4, 6), textcoords="offset points")
    ax.set_xlim(0, dur)
    ax.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0, top=max(1.0, max(x["ros"] for x in s) * 1.25))
    _style(ax, f"Fire-front distance and rate of spread (towards the {snap['results']['spread_towards']})",
           "Simulated time (min)", "Front distance (m)")
    ax2.set_ylabel("Rate of spread (m/min)", fontsize=8)
    ax2.tick_params(labelsize=7.5)
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.5, loc="upper left")
    out["front"] = _fig_bytes(fig)
    # 3 - fire-behaviour rates (one unit: m/min)
    mdl = snap.get("model") or {}
    rates = [("Head", mdl.get("ros_head_m_per_min")), ("Flank", mdl.get("ros_flank_m_per_min")),
             ("Backing", mdl.get("ros_back_m_per_min")), ("Peak observed\nin the run", snap["results"].get("peak_ros"))]
    rates = [(k, float(v)) for k, v in rates if _f(v) is not None]
    if rates:
        fig, ax = plt.subplots(figsize=(7.4, 2.45))
        cols = [PALETTE["red"], PALETTE["orange"], PALETTE["amber"], PALETTE["blue"]][:len(rates)]
        bars = ax.bar([k for k, _ in rates], [v for _, v in rates], color=cols, width=.55)
        for b_, (_, v) in zip(bars, rates):
            ax.annotate(f"{v:.1f}", (b_.get_x() + b_.get_width() / 2, v), ha="center", va="bottom", fontsize=8,
                        xytext=(0, 2), textcoords="offset points")
        ax.set_ylim(0, max(v for _, v in rates) * 1.25 or 1)
        _style(ax, "Fire-behaviour spread rates", "", "Rate of spread (m/min)")
        out["rates"] = _fig_bytes(fig)
    # 4 - FWI components | model outputs (separate panels, separate units)
    fwi = snap.get("fwi") or {}
    comp = [(k.upper(), fwi.get(k)) for k in ("ffmc", "dmc", "dc", "bui", "fwi") if fwi.get(k) is not None]
    pr = snap.get("prediction") or {}
    probs = [(lab, v * 100) for lab, v in (("Model fire\nprobability", pr.get("probability")),
                                            ("Risk index", pr.get("risk_index"))) if v is not None]
    if comp or probs:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 2.6), gridspec_kw={"width_ratios": [1.7, 1]})
        if comp:
            bars = a1.bar([k for k, _ in comp], [v for _, v in comp], color=PALETTE["navy"], width=.55)
            for b_, (_, v) in zip(bars, comp):
                a1.annotate(f"{v:.1f}", (b_.get_x() + b_.get_width() / 2, v), ha="center", va="bottom", fontsize=7.5,
                            xytext=(0, 2), textcoords="offset points")
            a1.set_ylim(0, max(v for _, v in comp) * 1.22 or 1)
        _style(a1, "Fire Weather Index components", "", "Index value")
        if probs:
            bars = a2.bar([k for k, _ in probs], [v for _, v in probs], color=[PALETTE["grey"], PALETTE["orange"]][:len(probs)],
                          width=.5)
            for b_, (_, v) in zip(bars, probs):
                a2.annotate(f"{v:.1f}%", (b_.get_x() + b_.get_width() / 2, v), ha="center", va="bottom", fontsize=7.5,
                            xytext=(0, 2), textcoords="offset points")
            for y, lab in ((20, "MODERATE"), (40, "HIGH"), (60, "EXTREME")):
                a2.axhline(y, color=PALETTE["red"], lw=.7, ls="--", alpha=.6)
                a2.text(1.42, y, lab, fontsize=6, color=PALETTE["red"], va="center")
            a2.set_ylim(0, max(70, max(v for _, v in probs) * 1.2))
        _style(a2, "Model prediction", "", "Percent (%)")
        fig.tight_layout(w_pad=2.2)
        out["indicators"] = _fig_bytes(fig)
    return out


def chart_captions(snap: dict) -> Dict[str, str]:
    s, res = snap.get("series") or [], snap["results"]
    if not s:
        return {}
    first, last = s[0], s[-1]
    grew = last["burned_ha"] + last["burning_ha"] - (first["burned_ha"] + first["burning_ha"])
    cap = {"area": (f"Cumulative burned area rose from {first['burned_ha']:.2f} ha to {last['burned_ha']:.2f} ha; "
                    f"the area actively burning was {last['burning_ha']:.2f} ha at the end. The total fire footprint "
                    f"grew by {grew:.2f} ha over {snap['clock'].get('simulated_min', res['duration_min']):g} simulated "
                    f"minutes, sampled every {snap['clock'].get('frame_s', 0):g} s."
                    + (" Cells keep flaming for several minutes before they count as burned, so in short runs most of "
                       "the footprint is still burning." if last["burned_ha"] < last["burning_ha"] else "")),
           "front": (f"The fire front moved from {first['front_m']:.0f} m to {last['front_m']:.0f} m from the ignition "
                     f"centre. Rate of spread peaked at {res['peak_ros']:.1f} m/min"
                     + (f" at T+{res['peak_ros_at_min']:g} min" if res.get("peak_ros_at_min") is not None else "")
                     + f" and was {res.get('final_ros', 0):.1f} m/min at the end (5-minute moving window)."
                     + (" A rate of 0 means the front had not yet crossed a whole cell."
                        if res['peak_ros'] == 0 else ""))}
    mdl = snap.get("model") or {}
    if _f(mdl.get("ros_head_m_per_min")) is not None:
        cap["rates"] = (f"Potential spread rates for the run's wind at full fuel continuity: head "
                        f"{float(mdl['ros_head_m_per_min']):.1f}, flank {float(mdl.get('ros_flank_m_per_min') or 0):.1f} "
                        f"and backing {float(mdl.get('ros_back_m_per_min') or 0):.1f} m/min (length-to-breadth "
                        f"{mdl.get('length_to_breadth')}). The peak observed rate reflects patchy fuel and terrain.")
    fwi = snap.get("fwi") or {}
    pr = snap.get("prediction") or {}
    bits = [f"FWI {fwi['fwi']:.1f} ({fwi_interpretation('fwi', fwi['fwi'])})"] if fwi.get("fwi") is not None else []
    if fwi.get("ffmc") is not None:
        bits.append(f"FFMC {fwi['ffmc']:.1f} ({fwi_interpretation('ffmc', fwi['ffmc'])})")
    if pr.get("risk_index") is not None:
        bits.append(f"model risk index {pr['risk_index'] * 100:.0f}% ({pr.get('category') or '-'})"
                    + (f" from a fire probability of {pr['probability'] * 100:.2f}%" if pr.get("probability") is not None else ""))
    if bits:
        cap["indicators"] = "Indicators at the location: " + "; ".join(bits) + ". Dashed lines mark the risk-index category thresholds."
    return cap


# ── content model shared by PDF / DOCX / HTML ───────────────────────────────
# section = {"title", "blocks": [("p", text) | ("bullets", [..]) | ("table", header, rows, widths) |
#                                ("kpis", [(label, value, note)]) | ("chart", key)]}

def _risk_line(snap) -> str:
    pr, r = snap.get("prediction") or {}, snap["risk"]
    if r.get("score_pct") is None:
        return "not available"
    from src.ml_models.risk_score import format_score
    score = (f"ML Fire Risk Score {format_score(pr['risk_score'])} ({pr.get('risk_score_category')}), "
             if format_score(pr.get("risk_score")) else "")
    prob = f"fire-occurrence probability {_pct(pr['probability'])}" if pr.get("probability") is not None else ""
    return f"{score}{prob} - PREDICTED"


def _fmt_score(v) -> str:
    from src.ml_models.risk_score import format_score
    return format_score(v) or "n/a"


def _pct(p) -> str:
    """A probability as a percentage with enough precision to read small values (0.18%, 4.09%, 38.7%)."""
    v = float(p) * 100
    return f"{v:.2f}%" if v < 10 else f"{v:.1f}%"


def findings(snap) -> List[str]:
    res, s, c = snap["results"], snap.get("series") or [], snap.get("clock") or {}
    out = [f"The simulation ran for the full {c.get('simulated_s', 0):,.0f} s of simulated time "
           f"({c.get('n_frames', 0)} recorded frames, status: {c.get('status', 'completed')})."]
    out.append(f"Fire footprint at the end: {res['fire_area_ha']:.2f} ha ({res['burned_ha']:.2f} ha burned, "
               f"{res['burning_ha']:.2f} ha still burning), perimeter {res['perimeter_m'] or 0:,.0f} m.")
    out.append(f"The front reached {res['front_distance_m'] or 0:,.0f} m from the ignition centre, spreading towards "
               f"the {res['spread_towards']} (wind from the {compass(snap['wind_used']['from_deg'])} at "
               f"{snap['wind_used']['speed_ms']} m/s); peak rate of spread {res['peak_ros']:.1f} m/min.")
    if len(s) >= 3:
        mid = s[len(s) // 2]
        out.append(f"At mid-run (T+{mid['minutes']:g} min) the footprint was {mid['burned_ha'] + mid['burning_ha']:.2f} ha; "
                   f"it reached {s[-1]['burned_ha'] + s[-1]['burning_ha']:.2f} ha by the end.")
    out.append("The fire went out by itself at T+{:g} min.".format(res["extinguished_at_min"])
               if res.get("extinguished_at_min") is not None else "The fire was still active at the end of the run.")
    if res.get("left_focus"):
        out.append(f"The fire left the {snap['focus']['width_m']:.0f} × {snap['focus']['height_m']:.0f} m focus area; "
                   f"the domain grew {snap['domain_final']['expansions']} time(s) to keep the whole fire inside.")
    out.append(f"Model risk at the location: {_risk_line(snap)}.")
    return out


def sections(snap: dict) -> List[dict]:
    loc, w, r, f, res = snap["location"], snap["weather"], snap["risk"], snap["firms"], snap["results"]
    c, pr, mdl, fwi, hist = snap.get("clock") or {}, snap.get("prediction") or {}, snap["model"], snap["fwi"], snap["historical"]
    mode = snap["mode"]
    wl = w["label"].split(" ")[0]
    pts = snap["ignition"]["points"] or []
    sec = []
    # cover / overview
    sec.append({"title": "Report overview", "cover": True, "blocks": [
        ("table", ("Item", "Value"), [
            ("Report ID", snap["report_id"]), ("Simulation ID", snap.get("simulation_id", "-")),
            ("Generated", _ist(snap["generated_utc"])),
            ("Region / location", f"{loc['name']} - {snap.get('region') or '-'}"),
            ("Coordinates", f"{loc['lat']:.5f}° N, {loc['lon']:.5f}° E"),
            ("Simulation mode", "LIVE (observed ignition)" if mode == "LIVE" else f"{mode} (hypothetical scenario)"),
            ("Duration selected / completed", f"{_dur(c.get('requested_min', res['duration_min']))} / "
                                              f"{_dur(c.get('simulated_min', res['duration_min']))}"),
            ("Overall risk (model)", _risk_line(snap))], (0.32, 0.68)),
        ("kpis", [("Burned area", f"{res['burned_ha']:.2f} ha", "SIMULATED"),
                  ("Active burning", f"{res['burning_ha']:.2f} ha", "at completion"),
                  ("Front distance", f"{res['front_distance_m'] or 0:,.0f} m", f"towards {res['spread_towards']}"),
                  ("Peak spread rate", f"{res['peak_ros']:.1f} m/min", "SIMULATED"),
                  ("Perimeter", f"{res['perimeter_m'] or 0:,.0f} m", "at completion"),
                  (("ML Fire Risk Score", _fmt_score(pr['risk_score']), pr.get("risk_score_category") or "")
                   if pr.get("risk_score") is not None else
                   ("ML fire probability", _pct(pr["probability"]) if pr.get("probability") is not None else "n/a",
                    "PREDICTED"))])]})
    # 1 executive summary
    interp = (f"This {'hypothetical WHAT-IF scenario' if mode != 'LIVE' else 'LIVE simulation started from satellite fire detections'} "
              f"simulated fire spread at {loc['name']} ({loc['lat']:.4f}° N, {loc['lon']:.4f}° E) for "
              f"{_dur(c.get('simulated_min', res['duration_min']))} of simulated time with "
              f"{len(pts) or snap['ignition'].get('placement') or 'the configured'} ignition point(s). "
              f"Conditions: {w.get('temp_c', '-')} °C, {w.get('humidity_pct', '-')} % relative humidity, wind "
              f"{snap['wind_used']['speed_ms']} m/s from the {compass(snap['wind_used']['from_deg'])}.")
    sec.append({"title": "1. Executive summary", "blocks": [("p", interp), ("bullets", findings(snap)), ("p", DISCLAIMER)]})
    # 2 scenario and environment
    ign_txt = "; ".join(f"{float(p_[0]):.5f}, {float(p_[1]):.5f}" for p_ in pts[:6]) or (snap["ignition"].get("placement") or "-")
    sec.append({"title": "2. Scenario and environmental conditions", "blocks": [
        ("p", "Inputs exactly as supplied to the simulation, each labelled with its provenance."),
        ("table", ("Input", "Value", "Type"), [
            ("Focus area", f"{snap['focus']['width_m']:.0f} × {snap['focus']['height_m']:.0f} m, "
                           f"{snap['focus']['cell_m']:.0f} m cells", "SCENARIO INPUT"),
            ("Ignition", f"{snap['ignition']['kind']} - {snap['ignition'].get('placement') or 'map points'}",
             "OBSERVED" if mode == "LIVE" else "SCENARIO INPUT"),
            ("Ignition coordinates", ign_txt, "OBSERVED" if mode == "LIVE" else "SCENARIO INPUT"),
            ("Temperature", _num(w.get("temp_c"), "{:.1f}", "°C"), wl),
            ("Relative humidity", _num(w.get("humidity_pct"), "{:.0f}", "%"), wl),
            ("Wind used by the simulation", f"{snap['wind_used']['speed_ms']} m/s from {compass(snap['wind_used']['from_deg'])} "
                                            f"({snap['wind_used']['from_deg']:.0f}°)", wl),
            ("Weather source / time", f"{w.get('source')} · {_ist(w.get('timestamp_utc'))} · status {w.get('status')}", wl),
            ("Terrain", "~90 m elevation model (slope applied)" if snap.get("terrain") == "dem" else "flat ground (no elevation data)",
             "DERIVED"),
            ("Land cover", f"{(snap.get('land_cover') or {}).get('label') or '-'}", "OBSERVED / DERIVED")], (0.30, 0.50, 0.20))]})
    # 3 ML risk
    feats = pr.get("features") or []
    th = pr.get("thresholds") or {}
    ml_rows = [("Model", f"{r.get('model') or '-'} ({r.get('model_file') or '-'})"),
               ("Model version", pr.get("model_version") or "not recorded"),
               ("Prediction time", _ist(pr.get("predicted_utc"))),
               ("Prediction location", f"grid zone {pr.get('zone') or r.get('zone')} centred {pr.get('zone_lat')}° N, "
                                       f"{pr.get('zone_lon')}° E ({pr.get('zone_distance_km')} km from the site)"),
               ("Spatial resolution", pr.get("resolution") or "grid zone"),
               ("ML Fire Risk Score (main figure)", (f"{pr['risk_score']:.0f}% - {pr.get('risk_score_category')}: the model "
                                                     f"rates this zone-day as more fire-prone than {pr['risk_score']:.0f}% of "
                                                     "historical fire-season zone-days (relative ranking, not a probability)")
                if pr.get("risk_score") is not None else "not available"),
               ("Risk-score categories", "LOW < 50% ≤ MODERATE < 75% ≤ HIGH < 90% ≤ VERY HIGH < 95% ≤ EXTREME"),
               ("ML fire-occurrence probability", (f"{_pct(pr['probability'])} (raw model output, no re-calibration applied; on the "
                                                  "held-out 2025 season: mean predicted 4.2% vs observed 4.4% "
                                                  "fire rate, average calibration error 0.3 percentage points)") if pr.get("probability") is not None
                else "not available"),
               ("What the probability means", "chance of a satellite-detected fire in this ~11 km grid zone on the "
                                              "prediction day - not the chance that the simulated fire occurs, and not "
                                              f"tied to the {snap['results']['duration_label']} simulation"),
               ("Hypothetical ignition", "not a model input - the simulated ignition does not change the prediction"),
               ("Risk index / category", f"{r['score_pct']}% / {r.get('category')}" if r.get("score_pct") is not None
                else "not available"),
               ("Category thresholds (risk index)", "LOW < 20% ≤ MODERATE < 40% ≤ HIGH < 60% ≤ EXTREME"),
               ("Probability at the thresholds", (f"MODERATE from {th['moderate_from_probability'] * 100:.1f}%, HIGH from "
                                                  f"{th['high_from_probability'] * 100:.1f}%, EXTREME from "
                                                  f"{th['extreme_from_probability'] * 100:.1f}% fire probability")
                if th else "not recorded")]
    sc = pr.get("scenario_inputs") or {}
    if sc:
        ml_rows.insert(5, ("Scenario inputs received by the model",
                           ", ".join(f"{k} {v}" for k, v in sc.items())))
    oor = pr.get("out_of_range") or []
    blocks = [("p", "Three different outputs are reported and are not interchangeable: the ML Fire Risk Score (a "
                    "0-100% ranking of the model output against historical fire-season days), the ML fire-occurrence "
                    "probability (the model's estimate of a satellite-detected fire in the grid zone that day) and the "
                    "SIMULATED fire-spread results (section 6). The operational risk index used by dashboard alerts is "
                    "listed for completeness. All are model PREDICTIONS for the zone, not observations of fire."),
              ("table", ("Item", "Value"), ml_rows, (0.32, 0.68))]
    if pr.get("factors"):
        blocks += [("p", "Factors influencing this prediction:"), ("bullets", pr["factors"])]
    meta = pr.get("risk_score_meta") or {}
    if meta.get("validation"):
        blocks += [("p", f"Validation on the held-out {meta.get('validation_season')} season (never used to build the "
                         f"score): ROC-AUC {meta.get('held_out_roc_auc')}, PR-AUC {meta.get('held_out_pr_auc')} "
                         f"(base fire rate {meta.get('held_out_base_rate', 0) * 100:.1f}%). Observed fire rate by "
                         "risk-score category:"),
                   ("table", ("Category", "Score range", "Share of zone-days", "Observed fire rate", "Share of fires"),
                    [(v["category"], f"{v['score_from']:.0f}-{v['score_to']:.0f}%", f"{v['share_of_zone_days'] * 100:.1f}%",
                      "-" if v["observed_fire_rate"] is None else f"{v['observed_fire_rate'] * 100:.1f}%",
                      f"{v['share_of_fires'] * 100:.1f}%") for v in meta["validation"]], (0.2, 0.18, 0.2, 0.22, 0.2)),
                   ("p", "Model limitations: trained on January-May station weather for Karnataka (2023-2024); recent "
                         "satellite fire activity dominates the prediction; weather beyond the training range is not "
                         "extrapolated; the score ranks risk, it does not forecast a fire within the simulated period.")]
    if oor:
        blocks.append(("p", "Inputs outside the model's training range ("
                            + "; ".join(f"{n} {v:g} vs trained {lo:g}-{hi:g}" for n, v, lo, hi in oor)
                            + "). The model was trained on observed station weather; beyond that range a tree model "
                              "does not extrapolate, so the probability may stop rising even if conditions get more extreme."))
    if feats:
        half = (len(feats) + 1) // 2
        rows = [(feats[i][0], "-" if feats[i][1] is None else f"{feats[i][1]:g}",
                 feats[i + half][0] if i + half < len(feats) else "",
                 ("-" if feats[i + half][1] is None else f"{feats[i + half][1]:g}") if i + half < len(feats) else "")
                for i in range(half)]
        blocks += [("p", "Input features supplied to the model for this zone (weather, fire-weather indices, season, "
                         "satellite fire history and terrain):"),
                   ("table", ("Feature", "Value", "Feature", "Value"), rows, (0.28, 0.22, 0.28, 0.22))]
    blocks.append(("p", (f"Historical fire indicators: {hist['note']} Zone fire rate {hist.get('zone_fire_rate')}, "
                         f"neighbourhood {hist.get('neighbour_fire_rate')} (source: {hist.get('source')})."
                         if hist.get("available") else "Historical fire records: " + hist["note"])))
    sec.append({"title": "3. ML-based fire-risk assessment", "blocks": blocks})
    # 4 FWI
    sec.append({"title": "4. Fire Weather Index analysis", "blocks": [
        ("p", "Fire Weather Index components for the prediction zone, with their general fire-danger interpretation."),
        ("table", ("Component", "Value", "Interpretation", "Meaning"),
         [(k.upper(), "not calculated" if v is None else f"{v:g}", fwi_interpretation(k, v), FWI_BANDS[k][2])
          for k, v in fwi.items()], (0.12, 0.13, 0.27, 0.48)),
        ("p", "Values are DERIVED with the standard Fire Weather Index equations from the weather supplied to the grid "
              "zone; the indices are dimensionless. Interpretation uses general fire-danger bands.")]})
    # 5 method / configuration
    sec.append({"title": "5. Simulation method and configuration", "blocks": [
        ("p", "The simulation clock advances in simulated seconds until exactly the selected duration; computer runtime "
              "is recorded separately."),
        ("table", ("Setting", "Value"), [
            ("Selected duration", _dur(c.get("requested_min", res["duration_min"]))),
            ("Completed simulated time", f"{c.get('simulated_s', 0):,.0f} s ({c.get('status', 'completed')})"),
            ("Recording interval / frames", f"{c.get('frame_s', 0):g} s / {c.get('n_frames', 0)} frames"),
            ("Model time-step", (f"adaptive, mean {c['substep_s']:g} s ({c.get('n_substeps')} sub-steps)"
                                 if c.get("substep_s") else "-")),
            ("Grid resolution", f"{snap['focus']['cell_m']:.0f} m cells"),
            ("Domain (initial → final)", f"{snap['domain_initial']['width_m'] / 1000:.2f} × {snap['domain_initial']['height_m'] / 1000:.2f} km"
                                         f" → {snap['domain_final']['width_m'] / 1000:.2f} × {snap['domain_final']['height_m'] / 1000:.2f} km "
                                         f"({snap['domain_final']['expansions']} expansion(s))"),
            ("Fire-spread model", "rate-of-spread cellular automaton with an elliptical wind model (head / flank / "
                                  "backing rates) and slope effect"),
            ("Fuel", "dry-deciduous forest / grass fuel blend; cell fuel load from the zone vegetation index"),
            ("Wind / slope inputs", f"{snap['wind_used']['speed_ms']} m/s from {snap['wind_used']['from_deg']:.0f}°; "
                                    + ("slope from the elevation model" if snap.get("terrain") == "dem" else "flat ground")),
            ("Started / completed", f"{_ist(c.get('started_utc'))} / {_ist(c.get('completed_utc'))}"),
            ("Computational runtime", f"{c['runtime_s']:.2f} s" if c.get("runtime_s") is not None else "not recorded")],
         (0.32, 0.68))]})
    # 6 results
    sec.append({"title": "6. Simulation results", "blocks": [
        ("p", "Values at completion of the run, using the simulation engine's own definitions (SIMULATED)."),
        ("table", ("Result", "Value", "Definition"), [
            ("Total burned area", f"{res['burned_ha']:.2f} ha", "cells that finished flaming"),
            ("Active burning area (end)", f"{res['burning_ha']:.2f} ha", "cells flaming at completion"),
            ("Fire perimeter", f"{res['perimeter_m'] or 0:,.0f} m", "outline of burning + burned cells"),
            ("Fire-front distance", f"{res['front_distance_m'] or 0:,.0f} m", "farthest affected cell from the ignition centre"),
            ("Head / flank / backing rate", f"{_num(mdl.get('ros_head_m_per_min'))} / {_num(mdl.get('ros_flank_m_per_min'))} / "
                                            f"{_num(mdl.get('ros_back_m_per_min'))} m/min", "model rates for the run's wind"),
            ("Peak rate of spread", f"{res['peak_ros']:.1f} m/min" + (f" at T+{res['peak_ros_at_min']:g} min"
                                                                    if res.get("peak_ros_at_min") is not None else ""),
             "maximum of the 5-min moving window"),
            ("Final rate of spread", f"{res.get('final_ros', 0):.1f} m/min", "last frame, 5-min moving window"),
            ("Max fireline intensity", f"{res['max_kw_m']:,.0f} kW/m" if res.get("max_kw_m") else "-", "Byram intensity"),
            ("Active burning cells (end)", f"{res.get('burning_cells', 0):,}", f"{res.get('cell_area_ha', 0):g} ha each"),
            ("Burned cells (end)", f"{res.get('burned_cells', 0):,}", "cumulative"),
            ("Spread direction", f"towards {res['spread_towards']}", "downwind of the wind used"),
            ("Domain expansions", str(snap["domain_final"]["expansions"]), "automatic growth around the fire"),
            ("Final fire state", (f"out at T+{res['extinguished_at_min']:g} min" if res.get("extinguished_at_min") is not None
                                  else "still active"), "-")], (0.30, 0.30, 0.40))]})
    # 7 charts
    sec.append({"title": "7. Graphical analysis", "min_space_mm": 105, "blocks": [
        ("p", "All four charts are drawn from this run's recorded frames and computed values - nothing is interpolated."),
        ("chart", "area"), ("chart", "front"), ("chart", "rates"),
                                                             ("chart", "indicators")]})
    # 8 data sources
    det_rows = [(f"{d.get('lat', 0):.4f}, {d.get('lon', 0):.4f}", str(d.get("acq_utc") or d.get("date") or "-"),
                 str(d.get("confidence") or d.get("conf") or "-"), str(d.get("satellite") or d.get("sat") or
                                                                    d.get("instrument") or "-"))
                for d in f.get("detections") or []]
    firms_txt = (f"NASA FIRMS: status {f.get('status')}, fetched {_ist(f.get('fetched_utc'))}, latest acquisition "
                 f"{_ist(f.get('latest_acq_utc'))}; {f.get('n_total', 0)} detection(s) in the search box, "
                 f"{f.get('n_in_area', 0)} in the simulation area, {f.get('n_valid', 0)} used as ignitions. Hotspots are "
                 "satellite OBSERVATIONS, not ground-verified fires."
                 if f.get("status") is not None and not str(f.get("status")).startswith("not used") else
                 "NASA FIRMS: not part of this run (WHAT-IF hypothetical ignition) - no observed fire is claimed.")
    blocks = [("p", "Where each input came from and how current it is."),
              ("table", ("Source", "Status / provenance"), [
        ("Weather", f"{w['label']} - {w.get('source')}, {_ist(w.get('timestamp_utc'))}, status {w.get('status')}"),
        ("Satellite fire detections", firms_txt),
        ("Terrain", "elevation model (~90 m) applied" if snap.get("terrain") == "dem" else "no elevation data - flat ground used"),
        ("Land cover", f"{(snap.get('land_cover') or {}).get('label') or '-'} {(snap.get('land_cover') or {}).get('status') or ''}".strip()),
        ("Historical fire data", hist.get("source") if hist.get("available") else hist["note"]),
        ("Prediction model", f"{r.get('model') or '-'}, version {pr.get('model_version') or 'not recorded'}, "
                             f"predicted {_ist(pr.get('predicted_utc'))}")], (0.25, 0.75))]
    if det_rows:
        blocks.append(("table", ("Detection (lat, lon)", "Acquired (UTC)", "Confidence", "Sensor"), det_rows,
                       (0.3, 0.3, 0.2, 0.2)))
    sec.append({"title": "8. Data sources and data quality", "blocks": blocks})
    # 9 findings
    trend = ""
    s_ = snap.get("series") or []
    if len(s_) >= 3:
        q = [x["burned_ha"] + x["burning_ha"] for x in s_]
        half = len(q) // 2
        g1, g2 = q[half] - q[0], q[-1] - q[half]
        trend = ("Growth accelerated in the second half of the run" if g2 > g1 * 1.15 else
                 "Growth slowed in the second half of the run" if g2 < g1 * 0.85 else "Growth was steady through the run")
        trend += f" (+{g1:.2f} ha, then +{g2:.2f} ha)."
    sec.append({"title": "9. Findings and operational summary", "blocks": [
        ("bullets", [x for x in (
            f"Simulated extent: {res['fire_area_ha']:.2f} ha footprint, front {res['front_distance_m'] or 0:,.0f} m towards "
            f"the {res['spread_towards']} after {_dur(c.get('simulated_min', res['duration_min']))}.", trend,
            f"Fire weather: FWI {fwi.get('fwi')} and FFMC {fwi.get('ffmc')} ({fwi_interpretation('ffmc', fwi.get('ffmc'))}); "
            f"model risk {_risk_line(snap)}.") if x]),
        ("p", "Use the simulated extent and direction to plan where a fire starting at this point could reach under "
              "these conditions; confirm any real fire through official forest-department channels.")]})
    return sec


# ── PDF ─────────────────────────────────────────────────────────────────────

NAVY, ORANGE = "#13233a", "#e8710a"


def render_pdf(snap: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.platypus import (CondPageBreak, Image, KeepTogether, ListFlowable, ListItem, Paragraph,
                                    SimpleDocTemplate, Spacer, Table, TableStyle)
    from xml.sax.saxutils import escape as esc
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=9, leading=12.2, spaceAfter=3)
    small = ParagraphStyle("s", parent=body, fontSize=8, leading=10, spaceAfter=0)
    cap = ParagraphStyle("cap", parent=body, fontSize=8, leading=10.5, textColor=colors.HexColor("#46505c"),
                         spaceBefore=1, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12, spaceBefore=8,
                        spaceAfter=5, textColor=colors.HexColor(NAVY), keepWithNext=1)
    kpi_v = ParagraphStyle("kv", parent=body, fontName="Helvetica-Bold", fontSize=13, leading=15, alignment=TA_CENTER,
                           textColor=colors.HexColor(NAVY), spaceAfter=0)
    kpi_l = ParagraphStyle("kl", parent=small, alignment=TA_CENTER, textColor=colors.HexColor("#5b6673"))
    W = 180 * mm
    ch, caps = charts(snap), chart_captions(snap)
    footer_left = f"{TITLE}  |  Report {snap['report_id']}"

    class NumberedCanvas(rl_canvas.Canvas):          # two-pass "Page X of Y"
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._pages = []

        def showPage(self):
            self._pages.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            n = len(self._pages)
            for st_ in self._pages:
                self.__dict__.update(st_)
                self._chrome(n)
                super().showPage()
            super().save()

        def _chrome(self, n):
            self.saveState()
            self.setStrokeColor(colors.HexColor("#c9d1db"))
            self.setLineWidth(0.5)
            self.line(15 * mm, 13 * mm, 195 * mm, 13 * mm)
            self.setFont("Helvetica", 7.5)
            self.setFillColor(colors.HexColor("#5b6673"))
            self.drawString(15 * mm, 8.5 * mm, footer_left)
            self.drawRightString(195 * mm, 8.5 * mm, f"Page {self._pageNumber} of {n}")
            if self._pageNumber > 1:
                self.setFillColor(colors.HexColor("#b3401a"))
                self.drawString(15 * mm, 289 * mm, _banner(snap))
                self.setFillColor(colors.HexColor("#5b6673"))
                self.drawRightString(195 * mm, 289 * mm, f"{snap['report_id']}  ·  {_ist(snap['generated_utc'])}")
                self.line(15 * mm, 287 * mm, 195 * mm, 287 * mm)
            self.restoreState()

    def table(header, rows, widths):
        data = [[Paragraph(f"<b>{esc(str(x))}</b>", small) for x in header]] + \
               [[Paragraph(esc(str(c_)), small) for c_ in row] for row in rows]
        t = Table(data, colWidths=[W * x for x in widths], repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7ecf2")),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(ORANGE)),
            ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#d5dbe3")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2.6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fb")])]))
        return t

    def kpis(items):
        cells = [[Paragraph(esc(v), kpi_v), Paragraph(esc(lab), kpi_l), Paragraph(esc(note), kpi_l)]
                 for lab, v, note in items]
        row = [Table([[c_[0]], [c_[1]], [c_[2]]], colWidths=[W / len(items) - 4]) for c_ in cells]
        t = Table([row], colWidths=[W / len(items)] * len(items))
        t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#d5dbe3")),
                               ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d5dbe3")),
                               ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7f9fb")),
                               ("LINEABOVE", (0, 0), (-1, 0), 2, colors.HexColor(ORANGE)),
                               ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        return t

    def flow(block):
        kind = block[0]
        if kind == "p":
            return [Paragraph(esc(block[1]), body)]
        if kind == "bullets":
            return [ListFlowable([ListItem(Paragraph(esc(x), body), leftIndent=10, value="•") for x in block[1]],
                                 bulletType="bullet", start="•", leftIndent=10, bulletFontSize=8)]
        if kind == "table":
            return [table(block[1], block[2], block[3]), Spacer(1, 7)]
        if kind == "kpis":
            return [Spacer(1, 4), kpis(block[1]), Spacer(1, 8)]
        if kind == "chart":
            key = block[1]
            if key not in ch:
                return [Paragraph("Chart not available: fewer than two recorded frames.", cap)] if key == "area" else []
            from reportlab.lib.utils import ImageReader
            iw, ih = ImageReader(io.BytesIO(ch[key])).getSize()
            img = Image(io.BytesIO(ch[key]), width=W, height=W * ih / iw)
            return [KeepTogether([img, Paragraph(esc(caps.get(key, "")), cap)])]
        return []

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=16 * mm,
                            bottomMargin=20 * mm, title=f"{REPORT_TITLE} - {snap['report_id']}", author=TITLE)
    title_tbl = Table([[Paragraph(f"<font color='#f2a33a' size='8'><b>{esc(TITLE.upper())}</b></font>", body)],
                       [Paragraph(f"<font color='white' size='17'><b>{esc(REPORT_TITLE)}</b></font>",
                                  ParagraphStyle("t", parent=body, leading=21))],
                       [Paragraph(f"<font color='#c9d6e6' size='8.5'>Report {esc(snap['report_id'])}  ·  "
                                  f"{esc(snap.get('simulation_id', ''))}  ·  {esc(_ist(snap['generated_utc']))}</font>", body)]],
                      colWidths=[W])
    title_tbl.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(NAVY)),
                                   ("LINEBELOW", (0, -1), (-1, -1), 3, colors.HexColor(ORANGE)),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (0, 0), 9),
                                   ("BOTTOMPADDING", (0, -1), (-1, -1), 9)]))
    story = [title_tbl, Spacer(1, 6),
             Paragraph(f"<font color='#b3401a'><b>{esc(_banner(snap))}</b></font>", body), Spacer(1, 4)]
    for sec in sections(snap):
        blocks = [f_ for b in sec["blocks"] for f_ in flow(b)]
        if not blocks:
            continue
        if not sec.get("cover"):
            story.append(CondPageBreak(sec.get("min_space_mm", 35) * mm))          # never a heading stranded at a page bottom
            story.append(Paragraph(esc(sec["title"]), h2))     # keepWithNext: heading stays with its first block
            story += blocks
        else:
            story += blocks
    doc.build(story, canvasmaker=NumberedCanvas)
    return buf.getvalue()


# ── DOCX ────────────────────────────────────────────────────────────────────

def _docx_field(run, code):
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    for tag, txt in (("begin", None), (None, code), ("end", None)):
        if tag:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), tag)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = txt
        run._r.append(el)


def _shade(cell, hex_fill):
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    tc = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:color"), "auto")
    sh.set(qn("w:fill"), hex_fill)
    tc.append(sh)


def render_docx(snap: dict) -> bytes:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Mm, Pt, RGBColor
    d = Document()
    for s in d.sections:
        s.left_margin = s.right_margin = Mm(15)
        s.top_margin, s.bottom_margin = Mm(15), Mm(18)
        s.footer_distance = Mm(8)
        p = s.footer.paragraphs[0]
        p.text = f"{TITLE}  |  Report {snap['report_id']}  |  Page "
        _docx_field(p.add_run(), "PAGE")
        p.add_run(" of ")
        _docx_field(p.add_run(), "NUMPAGES")
        for run in p.runs:
            run.font.size = Pt(7.5)
            run.font.color.rgb = RGBColor(0x5B, 0x66, 0x73)
        hp = s.header.paragraphs[0]
        hr = hp.add_run(_banner(snap))
        hr.font.size, hr.font.color.rgb = Pt(7.5), RGBColor(0xB3, 0x40, 0x1A)
    st = d.styles["Normal"]
    st.font.name, st.font.size = "Calibri", Pt(9.5)
    st.paragraph_format.space_after = Pt(3)
    # title block
    t = d.add_table(rows=1, cols=1)
    c = t.rows[0].cells[0]
    _shade(c, NAVY[1:])
    p = c.paragraphs[0]
    r = p.add_run(TITLE.upper())
    r.bold, r.font.size, r.font.color.rgb = True, Pt(8), RGBColor(0xF2, 0xA3, 0x3A)
    p = c.add_paragraph()
    r = p.add_run(REPORT_TITLE)
    r.bold, r.font.size, r.font.color.rgb = True, Pt(17), RGBColor(0xFF, 0xFF, 0xFF)
    p = c.add_paragraph()
    r = p.add_run(f"Report {snap['report_id']}  ·  {snap.get('simulation_id', '')}  ·  {_ist(snap['generated_utc'])}")
    r.font.size, r.font.color.rgb = Pt(8.5), RGBColor(0xC9, 0xD6, 0xE6)
    ban = d.add_paragraph()
    run = ban.add_run(_banner(snap))
    run.bold, run.font.color.rgb = True, RGBColor(0xB3, 0x40, 0x1A)
    ch, caps = charts(snap), chart_captions(snap)
    for sec in sections(snap):
        if sec.get("min_space_mm", 0) >= 100:
            d.add_page_break()                         # charts section starts on a fresh page
        if not sec.get("cover"):
            h = d.add_heading(sec["title"], level=1)
            for rr in h.runs:
                rr.font.size, rr.font.color.rgb = Pt(13), RGBColor(0x13, 0x23, 0x3A)
            h.paragraph_format.keep_with_next = True
        for b in sec["blocks"]:
            kind = b[0]
            if kind == "p":
                d.add_paragraph(b[1])
            elif kind == "bullets":
                for x in b[1]:
                    d.add_paragraph(x, style="List Bullet")
            elif kind == "table":
                tb = d.add_table(rows=1, cols=len(b[1]))
                tb.style = "Table Grid"
                for i, x in enumerate(b[1]):
                    cell = tb.rows[0].cells[i]
                    cell.text = str(x)
                    cell.paragraphs[0].runs[0].bold = True
                    _shade(cell, "E7ECF2")
                for row in b[2]:
                    cells = tb.add_row().cells
                    for i, x in enumerate(row):
                        cells[i].text = str(x)
                for row in tb.rows:
                    for i, cell in enumerate(row.cells):
                        cell.width = Mm(180 * b[3][i])
                        for pp in cell.paragraphs:
                            for rr in pp.runs:
                                rr.font.size = Pt(8.5)
                d.add_paragraph()
            elif kind == "kpis":
                tb = d.add_table(rows=2, cols=len(b[1]))
                tb.style = "Table Grid"
                for i, (lab, v, note) in enumerate(b[1]):
                    _shade(tb.rows[0].cells[i], "F7F9FB")
                    _shade(tb.rows[1].cells[i], "F7F9FB")
                    p = tb.rows[0].cells[i].paragraphs[0]
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    rr = p.add_run(v)
                    rr.bold, rr.font.size, rr.font.color.rgb = True, Pt(12), RGBColor(0x13, 0x23, 0x3A)
                    p = tb.rows[1].cells[i].paragraphs[0]
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    rr = p.add_run(f"{lab}\n{note}")
                    rr.font.size = Pt(7.5)
                d.add_paragraph()
            elif kind == "chart" and b[1] in ch:
                d.add_picture(io.BytesIO(ch[b[1]]), width=Mm(178))
                cp = d.add_paragraph(caps.get(b[1], ""))
                for rr in cp.runs:
                    rr.font.size, rr.font.color.rgb = Pt(8), RGBColor(0x46, 0x50, 0x5C)
            elif kind == "chart" and b[1] == "area":
                d.add_paragraph("Chart not available: fewer than two recorded frames.")
    out = io.BytesIO()
    d.save(out)
    return out.getvalue()


# ── printable HTML (print view = the report only) ───────────────────────────

def render_html(snap: dict, auto_print: bool = False) -> str:
    import base64
    from html import escape
    ch, caps = charts(snap), chart_captions(snap)
    parts = [f"<header class='title'><div class='k'>{escape(TITLE.upper())}</div><h1>{escape(REPORT_TITLE)}</h1>"
             f"<div class='m'>Report {escape(snap['report_id'])} · {escape(snap.get('simulation_id', ''))} · "
             f"{escape(_ist(snap['generated_utc']))}</div></header><p class='ban'>{escape(_banner(snap))}</p>"]
    for sec in sections(snap):
        parts.append("<section>" + ("" if sec.get("cover") else f"<h2>{escape(sec['title'])}</h2>"))
        for b in sec["blocks"]:
            if b[0] == "p":
                parts.append(f"<p>{escape(b[1])}</p>")
            elif b[0] == "bullets":
                parts.append("<ul>" + "".join(f"<li>{escape(x)}</li>" for x in b[1]) + "</ul>")
            elif b[0] == "table":
                parts.append("<table><tr>" + "".join(f"<th>{escape(str(x))}</th>" for x in b[1]) + "</tr>" +
                             "".join("<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in row) + "</tr>" for row in b[2])
                             + "</table>")
            elif b[0] == "kpis":
                parts.append("<div class='kpis'>" + "".join(
                    f"<div><b>{escape(v)}</b><span>{escape(lab)}</span><i>{escape(n)}</i></div>" for lab, v, n in b[1])
                    + "</div>")
            elif b[0] == "chart" and b[1] in ch:
                parts.append(f"<figure><img src='data:image/png;base64,{base64.b64encode(ch[b[1]]).decode()}'>"
                             f"<figcaption>{escape(caps.get(b[1], ''))}</figcaption></figure>")
        parts.append("</section>")
    css = ("body{font-family:Segoe UI,Arial,sans-serif;font-size:12px;color:#1d2733;background:#fff;margin:14px}"
           ".title{background:#13233a;color:#fff;padding:12px 14px;border-bottom:3px solid #e8710a}"
           ".title .k{color:#f2a33a;font-size:10px;font-weight:700;letter-spacing:.06em}.title h1{font-size:20px;margin:4px 0}"
           ".title .m{color:#c9d6e6;font-size:11px}.ban{color:#b3401a;font-weight:700}"
           "h2{font-size:15px;color:#13233a;border-bottom:1px solid #d5dbe3;padding-bottom:2px;margin:16px 0 6px;"
           "break-after:avoid}table{border-collapse:collapse;width:100%;margin:6px 0;break-inside:auto}"
           "tr{break-inside:avoid}td,th{border-bottom:1px solid #d5dbe3;padding:3px 6px;vertical-align:top;text-align:left}"
           "th{background:#e7ecf2;border-bottom:1.5px solid #e8710a}.kpis{display:flex;gap:6px;margin:8px 0}"
           ".kpis div{flex:1;border:1px solid #d5dbe3;border-top:3px solid #e8710a;background:#f7f9fb;text-align:center;padding:6px}"
           ".kpis b{display:block;font-size:16px;color:#13233a}.kpis span,.kpis i{display:block;font-size:10px;color:#5b6673}"
           "figure{margin:8px 0;break-inside:avoid}img{width:100%;max-width:760px}figcaption{font-size:11px;color:#46505c}"
           ".bar{position:sticky;top:0;background:#fff;padding:4px 0}"
           "@media print{.bar{display:none}body{margin:0}@page{size:A4;margin:14mm 13mm 16mm}}")
    btn = "<div class='bar'><button onclick='window.print()' style='padding:6px 14px'>🖨 Print report</button></div>"
    js = "<script>setTimeout(()=>window.print(),400)</script>" if auto_print else ""
    return (f"<html><head><meta charset='utf-8'><title>{escape(snap['report_id'])}</title><style>{css}</style></head>"
            f"<body>{btn}{''.join(parts)}{js}</body></html>")
