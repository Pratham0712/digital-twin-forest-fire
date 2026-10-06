"""
local_spread.py - fine-grid, geographically anchored fire spread for one
focus area (for example a 500 m x 500 m block of Bandipur Tiger Reserve).

Why this exists
---------------
The regional digital twin runs on a 0.1 deg grid (~11 km cells). A 500 m
simulation area is far smaller than one of those cells, so it cannot be
"zoomed into" the regional CA without inventing data. Instead this module
runs the SAME FireSpreadSimulator (src/simulation/cellular_automata.py,
algorithm unchanged) on a local grid of real lat/lon cells:

  * every cell is a real geographic square (default 25 m x 25 m), so the grid
    can be drawn exactly on the Google satellite map;
  * the conditions (FFMC dryness, BUI fuel build-up, NDVI fuel load, wind)
    come from the regional twin's grid zone that contains the focus area -
    the same values the regional model and CA use;
  * elevation comes from a real DEM sample per local cell (Open-Topo-Data /
    Open-Meteo, cached on disk) when it can be fetched, otherwise the terrain
    is treated as flat and labelled as such;
  * the CA time step is scaled with the cell size so the physical spread rate
    stays the one the project is configured for:
        step_minutes = 15 min * cell_m / SYSTEM.ca_cell_size_m (100 m)
    i.e. 25 m cells -> 3.75 min steps -> the same 100 m per 15 min.

Nothing here changes the CA itself; it only prepares its inputs and relabels
its time axis.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.config import MODELS_DIR, SYSTEM
from src.simulation.cellular_automata import CellState, FireSpreadSimulator

logger = logging.getLogger(__name__)

METERS_PER_DEG_LAT = 111_320.0
CA_STEP_MINUTES = 15                  # FireSpreadSimulator's native step at SYSTEM.ca_cell_size_m


# ── focus areas ───────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class FocusArea:
    """A square simulation area centred on (lat, lon)."""
    name: str
    lat: float
    lon: float
    size_m: float = float(SYSTEM.focus_size_m)
    cell_m: float = float(SYSTEM.local_ca_cell_m)
    state: str = ""
    description: str = ""

    @property
    def n(self) -> int:
        return max(2, int(round(self.size_m / self.cell_m)))

    @property
    def dlat(self) -> float:
        """Cell height in degrees of latitude."""
        return self.cell_m / METERS_PER_DEG_LAT

    @property
    def dlon(self) -> float:
        """Cell width in degrees of longitude (shrinks with latitude)."""
        return self.cell_m / (METERS_PER_DEG_LAT * math.cos(math.radians(self.lat)))

    @property
    def bounds(self) -> Dict[str, float]:
        half = self.n / 2.0
        return {"north": self.lat + half * self.dlat, "south": self.lat - half * self.dlat,
                "west": self.lon - half * self.dlon, "east": self.lon + half * self.dlon}

    @property
    def area_km2(self) -> float:
        return (self.n * self.cell_m) ** 2 / 1e6

    def cell_centres(self) -> Tuple[np.ndarray, np.ndarray]:
        """(lat, lon) of every cell centre, shape (n, n). Row 0 is the NORTH
        edge, column 0 the WEST edge (the same orientation as the regional grid
        and the CA's neighbour bearings: row-1 = north, col+1 = east)."""
        b = self.bounds
        r = np.arange(self.n)
        lat = b["north"] - (r + 0.5) * self.dlat
        lon = b["west"] + (r + 0.5) * self.dlon
        return np.repeat(lat[:, None], self.n, axis=1), np.repeat(lon[None, :], self.n, axis=0)

    def with_center(self, lat: float, lon: float, name: Optional[str] = None) -> "FocusArea":
        return FocusArea(name=name or self.name, lat=lat, lon=lon, size_m=self.size_m,
                         cell_m=self.cell_m, state=self.state, description=self.description)


# Named protected areas (centre coordinates of the reserve / park).
FOCUS_AREAS: Dict[str, FocusArea] = {
    "Bandipur Tiger Reserve": FocusArea(
        name="Bandipur Tiger Reserve", lat=11.6667, lon=76.6333, state="Karnataka",
        description="Tiger reserve in Chamarajanagar district, Nilgiri Biosphere Reserve."),
    "Nagarhole National Park": FocusArea(
        name="Nagarhole National Park", lat=12.0306, lon=76.1558, state="Karnataka",
        description="Rajiv Gandhi (Nagarhole) National Park, Kodagu / Mysuru districts."),
    "Kudremukh National Park": FocusArea(
        name="Kudremukh National Park", lat=13.2167, lon=75.2500, state="Karnataka",
        description="Shola forest and grassland, Chikkamagaluru district."),
}
DEFAULT_FOCUS = "Bandipur Tiger Reserve"
HIGHEST_RISK_FOCUS = "Highest-risk zone of this scenario"


def focus_options_for_region(region) -> List[str]:
    """Named focus areas that lie inside `region`, plus the data-driven option."""
    names = [k for k, f in FOCUS_AREAS.items()
             if region.min_lat <= f.lat <= region.max_lat and region.min_lon <= f.lon <= region.max_lon]
    return names + [HIGHEST_RISK_FOCUS]


def resolve_focus(choice: str, processed: pd.DataFrame, risk_scores: np.ndarray) -> FocusArea:
    """FocusArea for a selectbox choice. The highest-risk option centres the
    area on the centre of the zone with the highest model risk score."""
    if choice in FOCUS_AREAS:
        return FOCUS_AREAS[choice]
    i = int(np.argmax(risk_scores)) if len(risk_scores) else 0
    row = processed.reset_index(drop=True).iloc[i]
    return FocusArea(name=f"Highest-risk zone {row['zone_id']}", lat=float(row["latitude"]),
                     lon=float(row["longitude"]))


# ── conditions from the regional twin ─────────────────────────────────────── #

def zone_conditions(processed: pd.DataFrame, risk_scores: Optional[np.ndarray],
                    lat: float, lon: float) -> dict:
    """Values of the regional grid zone nearest to (lat, lon)."""
    p = processed.reset_index(drop=True)
    d2 = (p["latitude"] - lat) ** 2 + ((p["longitude"] - lon) * math.cos(math.radians(lat))) ** 2
    i = int(d2.idxmin())
    row = p.iloc[i]

    def g(col, default=float("nan")):
        return float(row[col]) if col in p.columns and pd.notna(row[col]) else default

    return {
        "zone_id": str(row["zone_id"]), "zone_lat": float(row["latitude"]), "zone_lon": float(row["longitude"]),
        "risk_score": float(risk_scores[i]) if risk_scores is not None and len(risk_scores) > i else float("nan"),
        "ffmc": g("ffmc"), "bui": g("bui"), "fwi": g("fwi"), "ndvi": g("ndvi"),
        "temp_c": g("wx_temperature_c"), "humidity_pct": g("wx_humidity_pct"),
        "wind_speed_ms": g("wx_wind_speed_ms", 0.0), "wind_from_deg": g("wx_wind_deg", 0.0),
        "elev_m": g("elev_m"), "slope_pct": g("slope_pct"),
        "active_fire_nearby": bool(row["active_fire_nearby"]) if "active_fire_nearby" in p.columns else False,
    }


def ca_inputs_from_conditions(cond: dict) -> dict:
    """The exact per-cell transforms DigitalTwin._simulate_spread applies to a
    zone (FFMC/101 dryness, NDVI fuel clipped to 0-1, NDVI < 0.15 = non-fuel,
    BUI/30 clipped to 0.4-2.0), so the local run sees the same conditions."""
    ffmc = cond.get("ffmc")
    ndvi = cond.get("ndvi")
    bui = cond.get("bui")
    dryness = float(np.clip((ffmc if np.isfinite(ffmc) else 85.0) / 101.0, 0, 1))
    fuel = float(np.clip(ndvi if np.isfinite(ndvi) else 0.5, 0, 1))
    buildup = float(np.clip((bui if np.isfinite(bui) else 30.0) / 30.0, 0.4, 2.0))
    non_fuel = bool(np.isfinite(ndvi) and ndvi < 0.15)
    return {"dryness": dryness, "fuel": fuel, "buildup": buildup, "non_fuel": non_fuel}


# ── local terrain ─────────────────────────────────────────────────────────── #

FOCUS_TERRAIN_DIR = MODELS_DIR / "focus_terrain"
_FETCH_FAILED_AT: Dict[str, float] = {}         # don't retry a failed DEM fetch on every rerun
_RETRY_AFTER_S = 600


def _terrain_cache_path(focus: FocusArea) -> Path:
    slug = "".join(ch if ch.isalnum() else "_" for ch in focus.name.lower()).strip("_")
    return FOCUS_TERRAIN_DIR / f"{slug}_{focus.lat:.4f}_{focus.lon:.4f}_{int(focus.size_m)}m_{int(focus.cell_m)}m.csv"


def local_elevation(focus: FocusArea, allow_fetch: bool = True) -> Tuple[np.ndarray, str]:
    """Elevation (m) at every local cell centre and where it came from:
    "dem" (real DEM sample, cached), or "flat" (no DEM available - the CA then
    treats the area as level ground, exactly as it does without terrain)."""
    path = _terrain_cache_path(focus)
    lat, lon = focus.cell_centres()
    if path.exists():
        try:
            df = pd.read_csv(path)
            elev = df["elevation"].to_numpy(dtype=float)
            if elev.size == lat.size and np.isfinite(elev).all():
                return elev.reshape(lat.shape), "dem"
        except Exception as exc:                   # corrupt cache: refetch below
            logger.warning("Ignoring unreadable focus terrain cache %s (%s)", path.name, exc)

    key = str(path)
    if allow_fetch and time.time() - _FETCH_FAILED_AT.get(key, 0) > _RETRY_AFTER_S:
        try:
            from src.data_ingestion.terrain import fetch_elevations
            elev = fetch_elevations(lat.ravel(), lon.ravel(), provider="auto", retries=1, pause_s=0.2)
            FOCUS_TERRAIN_DIR.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"latitude": lat.ravel().round(6), "longitude": lon.ravel().round(6),
                          "elevation": elev}).to_csv(path, index=False)
            logger.info("Focus terrain for %s fetched and cached (%d points)", focus.name, elev.size)
            return elev.reshape(lat.shape), "dem"
        except Exception as exc:
            _FETCH_FAILED_AT[key] = time.time()
            logger.warning("Focus terrain unavailable for %s (%s); using flat terrain.", focus.name, exc)
    return np.zeros(lat.shape), "flat"


# ── the run ───────────────────────────────────────────────────────────────── #

INTENSITY_CLASSES = [(0.35, "LOW"), (0.55, "MODERATE"), (0.75, "HIGH"), (1.01, "EXTREME")]


def intensity_class(v: float) -> str:
    for upper, name in INTENSITY_CLASSES:
        if v < upper:
            return name
    return "EXTREME"


@dataclass
class LocalSpreadResult:
    focus: FocusArea
    step_minutes: float
    history: list                       # SimulationStep list (minutes relabelled to local time)
    ignition_step: np.ndarray           # (n, n) int, -1 = never burned
    burnout_step: np.ndarray            # (n, n) int, -1 = never burned out
    intensity: np.ndarray               # (n, n) float 0-1, 0 = never burned
    non_fuel: np.ndarray                # (n, n) bool
    elevation: np.ndarray               # (n, n) metres
    terrain_source: str                 # "dem" | "flat"
    wind_schedule: List[Tuple[float, float]]
    conditions: dict
    params: dict
    metrics: List[dict] = field(default_factory=list)

    @property
    def final(self) -> dict:
        return self.metrics[-1] if self.metrics else {}


def _ignition_cells(n: int, count: int, placement: str, wind_speed: float, wind_from: float) -> np.ndarray:
    """Boolean (n, n) mask of `count` contiguous cells. "Upwind edge" puts the
    ignition 30 % of the box width upwind of the centre so the head fire runs
    across the simulation area; in calm air (< 0.5 m/s) it falls back to the
    centre."""
    cr = cc = (n - 1) / 2.0
    if placement.lower().startswith("upwind") and wind_speed >= 0.5:
        # wind_from is a compass bearing: north = row - 1, east = col + 1
        off = 0.3 * n
        cr -= math.cos(math.radians(wind_from)) * off
        cc += math.sin(math.radians(wind_from)) * off
    rr, ccs = np.mgrid[0:n, 0:n]
    d = (rr - cr) ** 2 + (ccs - cc) ** 2
    order = np.argsort(d.ravel(), kind="stable")[:max(1, count)]
    mask = np.zeros(n * n, dtype=bool)
    mask[order] = True
    return mask.reshape(n, n)


def _perimeter_m(affected: np.ndarray, cell_m: float) -> float:
    """Length of the boundary between affected (burning or burned) and
    unaffected ground, counting the simulation-area edge as unaffected."""
    a = np.pad(affected, 1, constant_values=False).astype(np.int8)
    edges = (np.abs(np.diff(a, axis=0)).sum() + np.abs(np.diff(a, axis=1)).sum())
    return float(edges) * cell_m


def run_local_spread(focus: FocusArea, conditions: dict, wind_speed_ms: float, wind_from_deg: float,
                     n_ignition: int = 3, placement: str = "Upwind edge", seed: int = 42,
                     horizon_minutes: Optional[int] = None,
                     elevation: Optional[np.ndarray] = None, terrain_source: str = "flat",
                     wind_schedule_15min: Optional[Sequence[Tuple[float, float]]] = None,
                     base_spread_prob: Optional[float] = None) -> LocalSpreadResult:
    """Run FireSpreadSimulator on the focus area's local grid.

    wind_speed_ms / wind_from_deg: constant wind (meteorological FROM bearing,
        the convention the CA and the dashboard compass use).
    wind_schedule_15min: optional forecast schedule at the CA's native 15-min
        steps (as produced by DigitalTwin._spread_wind); it is resampled to
        the local step.
    base_spread_prob: FireSpreadSimulator's existing calibration parameter;
        defaults to SYSTEM.local_ca_base_spread_prob (see config.py).
    """
    n = focus.n
    horizon = int(horizon_minutes or SYSTEM.fire_spread_horizon_hours * 60)
    step_min = CA_STEP_MINUTES * focus.cell_m / float(SYSTEM.ca_cell_size_m)
    n_steps = max(1, int(round(horizon / step_min)))

    ca = ca_inputs_from_conditions(conditions)
    dryness = np.full((n, n), ca["dryness"])
    fuel = np.full((n, n), ca["fuel"])
    buildup = np.full((n, n), ca["buildup"])
    non_fuel = np.full((n, n), ca["non_fuel"], dtype=bool)
    if elevation is None:
        elevation = np.zeros((n, n))

    if wind_schedule_15min:
        sched = [tuple(map(float, wind_schedule_15min[min(int(k * step_min // CA_STEP_MINUTES),
                                                          len(wind_schedule_15min) - 1)]))
                 for k in range(n_steps)]
    else:
        sched = [(float(wind_speed_ms), float(wind_from_deg) % 360)] * n_steps

    ignition = _ignition_cells(n, n_ignition, placement, sched[0][0], sched[0][1]) & ~non_fuel

    base = float(SYSTEM.local_ca_base_spread_prob if base_spread_prob is None else base_spread_prob)
    sim = FireSpreadSimulator(n, n, minutes_per_step=CA_STEP_MINUTES, base_spread_prob=base,
                              random_state=seed, cell_size_deg=focus.cell_m / METERS_PER_DEG_LAT)
    history = sim.run(ignition_mask=ignition, dryness_grid=dryness, fuel_load_grid=fuel,
                      non_fuel_mask=non_fuel, wind_speed_ms=sched[0][0], wind_from_deg=sched[0][1],
                      horizon_minutes=n_steps * CA_STEP_MINUTES, elevation_grid=elevation,
                      fuel_buildup_grid=buildup, wind_schedule=sched)
    for h in history:                                    # relabel to the local time step
        h.minutes_elapsed = round(h.step * step_min, 2)

    ign = np.full((n, n), -1, dtype=int)
    out = np.full((n, n), -1, dtype=int)
    for h in history:
        ign[(h.state == CellState.BURNING) & (ign < 0)] = h.step
        out[(h.state == CellState.BURNED) & (out < 0)] = h.step

    # Visual fire intensity (0-1): the CA's own spread potential for the cell
    # (dryness x fuel x build-up x wind speed, no direction) scaled by how the
    # fire reached it - head fire (driven downwind) burns hotter than flank or
    # backing fire. Derived from the CA state, used only to scale the VFX.
    potential = sim._cell_spread_prob(ca["dryness"], ca["fuel"], ca["buildup"],
                                      sched[0][0], 1.0, 1.0)
    pot_norm = min(1.0, potential / (0.5 * base))
    intensity = np.zeros((n, n))
    offsets = sim._neighbor_offsets
    for r, c in zip(*np.where(ign >= 0)):
        s = ign[r, c]
        if s == 0:
            align = 1.0
        else:
            prev = history[s - 1].state
            spd, frm = sched[min(s - 1, len(sched) - 1)]
            align = 0.4
            for dr, dc, bearing in offsets:
                pr, pc = r - dr, c - dc                  # neighbour that could have ignited (r, c)
                if 0 <= pr < n and 0 <= pc < n and prev[pr, pc] == CellState.BURNING:
                    align = max(align, sim._wind_alignment_factor(bearing, frm))
        intensity[r, c] = float(np.clip((align / 1.8) * (0.45 + 0.55 * pot_norm), 0.12, 1.0))

    # Per-step analytics, all read from the CA state.
    centre_r, centre_c = np.argwhere(ignition).mean(axis=0) if ignition.any() else ((n - 1) / 2, (n - 1) / 2)
    rr, cc = np.mgrid[0:n, 0:n]
    dist_m = np.hypot(rr - centre_r, cc - centre_c) * focus.cell_m
    cell_area_ha = focus.cell_m ** 2 / 1e4
    metrics, prev_d = [], 0.0
    for h in history:
        burning = h.state == CellState.BURNING
        burned = h.state == CellState.BURNED
        affected = burning | burned
        d = float(dist_m[affected].max()) if affected.any() else 0.0
        edge = affected[0, :].any() or affected[-1, :].any() or affected[:, 0].any() or affected[:, -1].any()
        mean_int = float(intensity[burning].mean()) if burning.any() else 0.0
        metrics.append({
            "step": h.step, "minutes": h.minutes_elapsed,
            "burning": int(burning.sum()), "burned": int(burned.sum()),
            "burned_ha": round(float(burned.sum()) * cell_area_ha, 3),
            "fire_area_ha": round(float(affected.sum()) * cell_area_ha, 3),
            "perimeter_m": _perimeter_m(affected, focus.cell_m),
            "front_distance_m": round(d, 1),
            "ros_m_per_min": round((d - prev_d) / step_min, 2) if h.step else 0.0,
            "mean_intensity": round(mean_int, 3), "intensity_class": intensity_class(mean_int) if mean_int else "-",
            "boundary_reached": bool(edge),
            "wind_speed_ms": sched[min(max(h.step - 1, 0), len(sched) - 1)][0],
            "wind_from_deg": sched[min(max(h.step - 1, 0), len(sched) - 1)][1],
        })
        prev_d = d

    params = {"n_ignition": int(ignition.sum()), "placement": placement, "seed": int(seed),
              "horizon_minutes": horizon, "n_steps_planned": n_steps, "spread_potential": round(potential, 4),
              "base_spread_prob": base, "cell_m": focus.cell_m, "size_m": focus.n * focus.cell_m,
              **ca}
    return LocalSpreadResult(focus=focus, step_minutes=step_min, history=history, ignition_step=ign,
                             burnout_step=out, intensity=intensity, non_fuel=non_fuel,
                             elevation=elevation, terrain_source=terrain_source,
                             wind_schedule=sched, conditions=conditions, params=params, metrics=metrics)
