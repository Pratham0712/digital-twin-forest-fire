"""
live_point.py - real observations for ONE selected location (the simulation
focus area / a Google search result), separate from the regional twin grid.

  * point_weather(lat, lon)   OpenWeatherMap current weather at that point
  * area_hotspots(lat, lon)   NASA FIRMS satellite fire detections in a box
                              around that point

Rules
  * Real data only: when a request fails, the last successful response for the
    same place is returned and labelled "cached" (with its own timestamp);
    when there is none, nothing is returned and the status is "error". No
    value is ever invented.
  * Calls are cached per place/time window (TTL below), shared by all sessions
    of the process, so Streamlit reruns never hit the APIs. `force=True` (the
    Refresh buttons) bypasses the TTL.
  * API keys come from the environment (config.API, .env); they are never
    returned, logged or put in an error message.

Status dict returned with every result:
    {"mode": "live" | "cached" | "error" | "not_configured",
     "label": human-readable, "fetched_utc": iso | None, "error": str | None, ...}
"""
from __future__ import annotations

import math
import threading
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

import pandas as pd

from config.config import API

WEATHER_TTL_S = 10 * 60
FIRMS_TTL_S = 15 * 60
FIRMS_BOX_KM = 25.0            # search box around the selected location (side length)
FIRMS_DAYS = 2                 # today + yesterday (UTC): near-real-time window

_LOCK = threading.Lock()
_WEATHER: dict = {}            # key -> {"t": monotonic, "data": dict | None, "status": dict, "good": (data, status)}
_FIRMS: dict = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _owm_key() -> str:
    return API.owm_api_key or ""


def _firms_key() -> str:
    return API.firms_map_key or ""


# ── OpenWeatherMap: current weather at the selected point ────────────────── #

def point_weather(lat: float, lon: float, force: bool = False, client=None) -> Tuple[Optional[dict], dict]:
    """(weather dict or None, status). Weather dict keys: temperature_c,
    humidity_pct, wind_speed_ms, wind_deg, pressure_hpa, weather_main,
    weather_description, clouds_pct, precipitation_mm, observed_at,
    place_name, fetched_at."""
    if not _owm_key():
        return None, {"mode": "not_configured", "label": "OWM_API_KEY not set in .env",
                      "fetched_utc": None, "error": None, "source": "OpenWeatherMap"}
    key = (round(float(lat), 3), round(float(lon), 3))
    with _LOCK:
        hit = _WEATHER.get(key)
    if hit and not force and time.monotonic() - hit["t"] < WEATHER_TTL_S:
        return hit["data"], hit["status"]
    if client is None:
        from src.data_ingestion.weather_client import WeatherClient
        client = WeatherClient(_owm_key(), API.owm_base_url)
    rec = client.fetch_point(float(lat), float(lon))
    good = hit.get("good") if hit else None
    if isinstance(rec, dict):
        status = {"mode": "live", "label": "OpenWeatherMap current weather", "fetched_utc": _now_iso(),
                  "observed_utc": rec.get("observed_at"), "error": None, "source": "OpenWeatherMap"}
        entry = {"t": time.monotonic(), "data": rec, "status": status, "good": (rec, status)}
    else:
        err = getattr(client, "last_error", None) or "no response"
        if rec == "RATE_LIMITED":
            err = "OpenWeatherMap rate limit reached (HTTP 429)"
        if good is not None:
            data, gst = good
            status = {"mode": "cached", "label": f"OpenWeatherMap unavailable; cached from {gst['fetched_utc']}",
                      "fetched_utc": gst["fetched_utc"], "observed_utc": gst.get("observed_utc"), "error": err,
                      "source": "OpenWeatherMap"}
        else:
            data = None
            status = {"mode": "error", "label": "OpenWeatherMap unavailable", "fetched_utc": _now_iso(),
                      "error": err, "source": "OpenWeatherMap"}
        entry = {"t": time.monotonic(), "data": data, "status": status, "good": good}
    with _LOCK:
        _WEATHER[key] = entry
    return entry["data"], entry["status"]


# ── NASA FIRMS: satellite fire detections around the selected point ──────── #

def firms_box(lat: float, lon: float, size_km: float = FIRMS_BOX_KM) -> dict:
    half_lat = size_km / 2 / 111.32
    half_lon = size_km / 2 / (111.32 * max(0.1, math.cos(math.radians(lat))))
    return {"south": round(lat - half_lat, 4), "north": round(lat + half_lat, 4),
            "west": round(lon - half_lon, 4), "east": round(lon + half_lon, 4)}


def area_hotspots(lat: float, lon: float, size_km: float = FIRMS_BOX_KM, days: int = FIRMS_DAYS,
                  force: bool = False, client=None) -> Tuple[pd.DataFrame, dict]:
    """(detections, status). Detections are real FIRMS rows (latitude,
    longitude, acq_datetime, satellite, confidence, frp, ...), possibly empty;
    never synthetic."""
    box = firms_box(lat, lon, size_km)
    base = {"source": f"NASA FIRMS {API.firms_source}", "box": box, "days": days, "size_km": size_km}
    if not _firms_key():
        return _empty(), {**base, "mode": "not_configured", "label": "FIRMS_MAP_KEY not set in .env",
                          "fetched_utc": None, "error": None, "n": 0}
    key = (box["south"], box["west"], box["north"], box["east"], days, API.firms_source)
    with _LOCK:
        hit = _FIRMS.get(key)
    if hit and not force and time.monotonic() - hit["t"] < FIRMS_TTL_S:
        return hit["data"].copy(), hit["status"]
    if client is None:
        from src.data_ingestion.firms_client import FIRMSClient
        client = FIRMSClient(_firms_key(), API.firms_base_url, API.firms_source)
    df = client.fetch_hotspots(box["south"], box["west"], box["north"], box["east"], days)
    st = dict(client.last_status)
    good = hit.get("good") if hit else None
    if st.get("ok"):
        n = int(len(df))
        status = {**base, "mode": "live", "n": n, "fetched_utc": st["fetched_utc"],
                  "latest_acq_utc": st.get("latest_acq_utc"), "error": st.get("error"),
                  "label": (f"{n} NASA FIRMS detection(s)" if n else
                            "No NASA FIRMS fire detections available for this area/time window.")}
        entry = {"t": time.monotonic(), "data": df, "status": status, "good": (df, status)}
    elif good is not None:
        gdf, gst = good
        status = {**gst, "mode": "cached", "error": st.get("error"),
                  "label": f"NASA FIRMS unavailable; cached detections from {gst['fetched_utc']}"}
        entry = {"t": time.monotonic(), "data": gdf, "status": status, "good": good}
    else:
        status = {**base, "mode": "error", "n": 0, "fetched_utc": st.get("fetched_utc") or _now_iso(),
                  "error": st.get("error") or "no response", "label": "NASA FIRMS unavailable; no detections shown"}
        entry = {"t": time.monotonic(), "data": _empty(), "status": status, "good": None}
    with _LOCK:
        _FIRMS[key] = entry
    return entry["data"].copy(), entry["status"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["latitude", "longitude", "acq_datetime", "satellite", "confidence", "frp"])


def hotspot_markers(df: pd.DataFrame, limit: int = 400) -> list:
    """Map markers for real detections: position + the observation details."""
    out = []
    if df is None or df.empty:
        return out
    for _, r in df.head(limit).iterrows():
        t = r.get("acq_datetime")
        out.append({"lat": round(float(r["latitude"]), 5), "lon": round(float(r["longitude"]), 5),
                    "date": (pd.Timestamp(t).strftime("%Y-%m-%d %H:%M UTC") if pd.notna(t) else str(r.get("acq_date", ""))),
                    "sat": str(r.get("satellite", "") or ""), "conf": str(r.get("confidence", "") or ""),
                    "frp": (round(float(r["frp"]), 1) if "frp" in r and pd.notna(r["frp"]) else None)})
    return out


def clear_caches():
    """Tests only."""
    with _LOCK:
        _WEATHER.clear()
        _FIRMS.clear()
