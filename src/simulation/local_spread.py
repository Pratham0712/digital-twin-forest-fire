"""
local_spread.py - fine-grid, geographically anchored fire spread around a
user-chosen focus area (for example a 500 m x 500 m block of Bandipur Tiger
Reserve, a Google-Maps search result, or a box dragged on the map).

Why this exists
---------------
The regional digital twin runs on a 0.1 deg grid (~11 km cells). A focus area
of a few hundred metres is far smaller than one of those cells, so it cannot be
"zoomed into" the regional CA without inventing data. Instead this module runs
the SAME FireSpreadSimulator (src/simulation/cellular_automata.py, algorithm
unchanged) on a local grid of real lat/lon cells:

  * every cell is a real geographic square (default 25 m), so the grid can be
    drawn exactly on the Google satellite map;
  * the conditions (FFMC dryness, BUI fuel build-up, NDVI fuel load, wind)
    come from the regional twin's grid zone that contains the focus area -
    the same values the regional model and CA use;
  * elevation comes from a real DEM (Open-Topo-Data SRTM 90 m / Open-Meteo)
    sampled on a fixed ~90 m lattice, cached on disk and interpolated to the
    cells; without it the terrain is treated as flat and labelled as such;
  * the CA time step is scaled with the cell size so the physical spread rate
    stays the one the project is configured for:
        step_minutes = 15 min * cell_m / SYSTEM.ca_cell_size_m (100 m)
    i.e. 25 m cells -> 3.75 min steps -> the same 100 m per 15 min.

Focus area vs simulation domain
-------------------------------
The focus area is what the user selected and looks at. The CA runs on a larger
SIMULATION DOMAIN: the focus area plus a margin on every side. A Moore-
neighbourhood CA moves the fire at most one cell per step, so a margin of
(number of steps + 1) cells guarantees the fire can never reach the domain edge
within the chosen duration: the fire is never stopped by a box, only by the
end of the simulated time (or by running out of fuel).

Nothing here changes the CA itself; it only prepares its inputs and relabels
its time axis.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.config import MODELS_DIR, SYSTEM
from src.simulation.cellular_automata import CellState, FireSpreadSimulator

logger = logging.getLogger(__name__)

METERS_PER_DEG_LAT = 111_320.0
CA_STEP_MINUTES = 15                  # FireSpreadSimulator's native step at SYSTEM.ca_cell_size_m


def step_minutes_for(cell_m: float) -> float:
    """Simulated minutes per CA step for a cell size (constant spread rate)."""
    return CA_STEP_MINUTES * float(cell_m) / float(SYSTEM.ca_cell_size_m)


def n_steps_for(duration_minutes: float, cell_m: float) -> int:
    """CA steps needed to cover `duration_minutes` (at least one)."""
    return max(1, int(math.ceil(float(duration_minutes) / step_minutes_for(cell_m) - 1e-9)))


# ── focus area (what the user selects) ────────────────────────────────────── #

@dataclass(frozen=True)
class FocusArea:
    """A rectangular area centred on (lat, lon), width (east-west) x height
    (north-south) in metres, divided into square cells of `cell_m`."""
    name: str
    lat: float
    lon: float
    width_m: float = float(SYSTEM.focus_size_m)
    height_m: float = float(SYSTEM.focus_size_m)
    cell_m: float = float(SYSTEM.local_ca_cell_m)
    state: str = ""
    description: str = ""
    source: str = "preset"            # preset | google_search | map | highest_risk

    # grid size
    @property
    def n_cols(self) -> int:
        return max(2, int(round(self.width_m / self.cell_m)))

    @property
    def n_rows(self) -> int:
        return max(2, int(round(self.height_m / self.cell_m)))

    @property
    def n(self) -> int:
        """Cells per side for a square area (kept for the original 500 m box)."""
        return self.n_cols

    @property
    def size_m(self) -> float:
        return self.n_cols * self.cell_m

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
        return {"north": self.lat + self.n_rows / 2.0 * self.dlat, "south": self.lat - self.n_rows / 2.0 * self.dlat,
                "west": self.lon - self.n_cols / 2.0 * self.dlon, "east": self.lon + self.n_cols / 2.0 * self.dlon}

    @property
    def area_km2(self) -> float:
        return self.n_rows * self.n_cols * self.cell_m ** 2 / 1e6

    def cell_centres(self) -> Tuple[np.ndarray, np.ndarray]:
        """(lat, lon) of every cell centre, shape (n_rows, n_cols). Row 0 is the
        NORTH edge, column 0 the WEST edge (the CA's orientation: row-1 =
        north, col+1 = east)."""
        b = self.bounds
        lat = b["north"] - (np.arange(self.n_rows) + 0.5) * self.dlat
        lon = b["west"] + (np.arange(self.n_cols) + 0.5) * self.dlon
        return (np.repeat(lat[:, None], self.n_cols, axis=1), np.repeat(lon[None, :], self.n_rows, axis=0))

    def with_center(self, lat: float, lon: float, name: Optional[str] = None, source: Optional[str] = None) -> "FocusArea":
        return replace(self, lat=float(lat), lon=float(lon), name=name or self.name, source=source or self.source)

    def with_size(self, width_m: float, height_m: float, cell_m: Optional[float] = None) -> "FocusArea":
        return replace(self, width_m=float(width_m), height_m=float(height_m),
                       cell_m=float(cell_m if cell_m is not None else self.cell_m))


@dataclass(frozen=True)
class SimulationDomain:
    """The grid the CA actually runs on: the focus area plus `margin` cells on
    every side, same centre, same cell size."""
    focus: FocusArea
    margin: int

    @property
    def cell_m(self) -> float:
        return self.focus.cell_m

    @property
    def n_rows(self) -> int:
        return self.focus.n_rows + 2 * self.margin

    @property
    def n_cols(self) -> int:
        return self.focus.n_cols + 2 * self.margin

    @property
    def width_m(self) -> float:
        return self.n_cols * self.cell_m

    @property
    def height_m(self) -> float:
        return self.n_rows * self.cell_m

    @property
    def lat(self) -> float:
        return self.focus.lat

    @property
    def lon(self) -> float:
        return self.focus.lon

    @property
    def bounds(self) -> Dict[str, float]:
        f = self.focus
        return {"north": f.lat + self.n_rows / 2.0 * f.dlat, "south": f.lat - self.n_rows / 2.0 * f.dlat,
                "west": f.lon - self.n_cols / 2.0 * f.dlon, "east": f.lon + self.n_cols / 2.0 * f.dlon}

    @property
    def area_km2(self) -> float:
        return self.n_rows * self.n_cols * self.cell_m ** 2 / 1e6

    def focus_slice(self) -> Tuple[slice, slice]:
        m = self.margin
        return slice(m, m + self.focus.n_rows), slice(m, m + self.focus.n_cols)

    def focus_mask(self) -> np.ndarray:
        mask = np.zeros((self.n_rows, self.n_cols), dtype=bool)
        mask[self.focus_slice()] = True
        return mask

    def cell_centres(self) -> Tuple[np.ndarray, np.ndarray]:
        f = self.focus
        b = self.bounds
        lat = b["north"] - (np.arange(self.n_rows) + 0.5) * f.dlat
        lon = b["west"] + (np.arange(self.n_cols) + 0.5) * f.dlon
        return (np.repeat(lat[:, None], self.n_cols, axis=1), np.repeat(lon[None, :], self.n_rows, axis=0))

    def cell_of(self, lat: float, lon: float) -> Optional[Tuple[int, int]]:
        """(row, col) of the domain cell containing (lat, lon), or None."""
        b = self.bounds
        r = int(math.floor((b["north"] - lat) / self.focus.dlat))
        c = int(math.floor((lon - b["west"]) / self.focus.dlon))
        if 0 <= r < self.n_rows and 0 <= c < self.n_cols:
            return r, c
        return None


def domain_for(focus: FocusArea, duration_minutes: float) -> SimulationDomain:
    """Domain large enough that the fire cannot reach its edge within the
    duration (the CA moves at most one cell per step)."""
    return SimulationDomain(focus=focus, margin=n_steps_for(duration_minutes, focus.cell_m) + 1)


def validate_setup(width_m: float, height_m: float, cell_m: float, duration_minutes: float) -> Optional[str]:
    """Human-readable problem with a requested setup, or None if it is valid."""
    lo, hi = SYSTEM.focus_min_m, SYSTEM.focus_max_m
    if not (lo <= width_m <= hi and lo <= height_m <= hi):
        return f"Width and height must each be between {lo} m and {hi} m."
    if cell_m not in SYSTEM.local_cell_options_m:
        return f"Cell size must be one of {', '.join(str(c) for c in SYSTEM.local_cell_options_m)} m."
    if width_m < 4 * cell_m or height_m < 4 * cell_m:
        return "The area must be at least 4 cells wide and high."
    if not (0 < duration_minutes <= SYSTEM.max_duration_minutes):
        return f"Duration must be between 1 and {SYSTEM.max_duration_minutes} minutes."
    d = domain_for(FocusArea("check", 0.0, 0.0, width_m, height_m, cell_m), duration_minutes)
    if max(d.n_rows, d.n_cols) > SYSTEM.max_domain_cells:
        return (f"That needs a {d.n_cols} x {d.n_rows}-cell simulation domain (the area plus room for "
                f"{duration_minutes:g} min of spread), above the {SYSTEM.max_domain_cells}-cell limit that keeps the "
                f"browser responsive. Use larger cells, a smaller area or a shorter duration.")
    return None


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
    """FocusArea for a preset choice. The highest-risk option centres the area
    on the centre of the zone with the highest model risk score."""
    if choice in FOCUS_AREAS:
        return FOCUS_AREAS[choice]
    i = int(np.argmax(risk_scores)) if len(risk_scores) else 0
    row = processed.reset_index(drop=True).iloc[i]
    return FocusArea(name=f"Highest-risk zone {row['zone_id']}", lat=float(row["latitude"]),
                     lon=float(row["longitude"]), source="highest_risk")


# ── conditions from the regional twin ─────────────────────────────────────── #

def zone_conditions(processed: pd.DataFrame, risk_scores: Optional[np.ndarray],
                    lat: float, lon: float) -> dict:
    """Values of the regional grid zone nearest to (lat, lon), plus the
    distance from the point to that zone's centre."""
    p = processed.reset_index(drop=True)
    d2 = (p["latitude"] - lat) ** 2 + ((p["longitude"] - lon) * math.cos(math.radians(lat))) ** 2
    i = int(d2.idxmin())
    row = p.iloc[i]

    def g(col, default=float("nan")):
        return float(row[col]) if col in p.columns and pd.notna(row[col]) else default

    return {
        "zone_id": str(row["zone_id"]), "zone_lat": float(row["latitude"]), "zone_lon": float(row["longitude"]),
        "zone_distance_km": round(float(math.sqrt(d2.iloc[i])) * 111.32, 2),
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


# ── terrain (real DEM on a fixed lattice, shared cache) ───────────────────── #

FOCUS_TERRAIN_DIR = MODELS_DIR / "focus_terrain"
DEM_LATTICE_DEG = 0.0008              # ~89 m: the SRTM 90 m resolution the DEM APIs serve
DEM_CACHE = FOCUS_TERRAIN_DIR / "dem_lattice.csv"
_FETCH_FAILED_AT: Dict[str, float] = {}         # don't retry a failed DEM fetch on every rerun
_RETRY_AFTER_S = 600
MAX_DEM_POINTS = 2500


def _lattice(b: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, float]:
    """Lattice coordinates (snapped to multiples of the step, so moving or
    resizing the area reuses cached points) covering bounds b."""
    step = DEM_LATTICE_DEG
    while True:
        lat0, lat1 = math.floor(b["south"] / step) * step, math.ceil(b["north"] / step) * step
        lon0, lon1 = math.floor(b["west"] / step) * step, math.ceil(b["east"] / step) * step
        lats = np.round(np.arange(lat0, lat1 + step / 2, step), 6)
        lons = np.round(np.arange(lon0, lon1 + step / 2, step), 6)
        if lats.size * lons.size <= MAX_DEM_POINTS:
            return lats, lons, step
        step *= 2


def domain_elevation(domain, allow_fetch: bool = True) -> Tuple[np.ndarray, str]:
    """Elevation (m) at every domain cell centre and its source: "dem" (real
    DEM lattice, bilinearly interpolated) or "flat" (no DEM available - the
    CA then treats the area as level ground, as it does without terrain)."""
    lat_c, lon_c = domain.cell_centres()
    lats, lons, _ = _lattice(domain.bounds)
    glat, glon = np.meshgrid(lats, lons, indexing="ij")
    key = f"{lats[0]:.4f},{lons[0]:.4f},{lats.size}x{lons.size}"
    known: Dict[Tuple[float, float], float] = {}
    if DEM_CACHE.exists():
        try:
            df = pd.read_csv(DEM_CACHE)
            known = {(round(a, 6), round(b, 6)): float(e)
                     for a, b, e in zip(df["latitude"], df["longitude"], df["elevation"])}
        except Exception as exc:
            logger.warning("Ignoring unreadable DEM cache (%s)", exc)
    pts = list(zip(np.round(glat.ravel(), 6), np.round(glon.ravel(), 6)))
    missing = [p for p in pts if p not in known]
    if missing:
        if not allow_fetch or time.time() - _FETCH_FAILED_AT.get(key, 0) < _RETRY_AFTER_S:
            return np.zeros(lat_c.shape), "flat"
        try:
            from src.data_ingestion.terrain import fetch_elevations
            vals = fetch_elevations(np.array([p[0] for p in missing]), np.array([p[1] for p in missing]),
                                    provider="auto", retries=1, pause_s=0.2)
            FOCUS_TERRAIN_DIR.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"latitude": [p[0] for p in missing], "longitude": [p[1] for p in missing],
                          "elevation": vals}).to_csv(DEM_CACHE, mode="a", header=not DEM_CACHE.exists(), index=False)
            known.update({p: float(v) for p, v in zip(missing, vals)})
            logger.info("DEM: fetched %d lattice points (cached in %s)", len(missing), DEM_CACHE.name)
        except Exception as exc:
            _FETCH_FAILED_AT[key] = time.time()
            logger.warning("DEM unavailable for this area (%s); using flat terrain.", exc)
            return np.zeros(lat_c.shape), "flat"
    grid = np.array([known[p] for p in pts]).reshape(glat.shape)
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((lats, lons), grid, bounds_error=False, fill_value=None)
    elev = f(np.column_stack([lat_c.ravel(), lon_c.ravel()])).reshape(lat_c.shape)
    return elev, "dem"


def local_elevation(focus: FocusArea, allow_fetch: bool = True) -> Tuple[np.ndarray, str]:
    """Elevation for the focus area itself (no margin)."""
    return domain_elevation(SimulationDomain(focus, 0), allow_fetch=allow_fetch)


# ── the run ───────────────────────────────────────────────────────────────── #

INTENSITY_CLASSES = [(0.35, "LOW"), (0.55, "MODERATE"), (0.75, "HIGH"), (1.01, "EXTREME")]
PLACEMENTS = ["Upwind edge", "Centre", "Downwind edge", "Map points"]


def intensity_class(v: float) -> str:
    for upper, name in INTENSITY_CLASSES:
        if v < upper:
            return name
    return "EXTREME"


@dataclass
class LocalSpreadResult:
    focus: FocusArea
    domain: SimulationDomain
    step_minutes: float
    duration_minutes: float
    history: list                       # SimulationStep list (minutes relabelled to local time)
    ignition_step: np.ndarray           # (rows, cols) of the DOMAIN, -1 = never burned
    burnout_step: np.ndarray            # (rows, cols), -1 = never burned out
    intensity: np.ndarray               # (rows, cols) 0-1, 0 = never burned
    non_fuel: np.ndarray                # (rows, cols) bool
    elevation: np.ndarray               # (rows, cols) metres
    terrain_source: str                 # "dem" | "flat"
    wind_schedule: List[Tuple[float, float]]
    conditions: dict
    params: dict
    metrics: List[dict] = field(default_factory=list)
    land_cover: Optional[np.ndarray] = None     # (rows, cols) fuel_map classes, None = no land-cover layer
    land_cover_label: str = "No land-cover layer"

    @property
    def final(self) -> dict:
        return self.metrics[-1] if self.metrics else {}

    @property
    def end_step(self) -> float:
        """The requested duration in CA steps (may end part-way through a step)."""
        return min(self.duration_minutes / self.step_minutes, float(self.history[-1].step))


def _placement_mask(rows: int, cols: int, count: int, placement: str, wind_speed: float,
                    wind_from: float) -> np.ndarray:
    """`count` contiguous cells inside a rows x cols area. Upwind edge puts the
    ignition 30 % of the area upwind of the centre (so the head fire runs
    across it), downwind edge 30 % downwind; in calm air (< 0.5 m/s) both fall
    back to the centre."""
    cr, cc = (rows - 1) / 2.0, (cols - 1) / 2.0
    p = placement.lower()
    if wind_speed >= 0.5 and (p.startswith("upwind") or p.startswith("downwind")):
        sign = 1.0 if p.startswith("upwind") else -1.0
        # wind_from is a compass bearing: north = row - 1, east = col + 1
        cr -= sign * math.cos(math.radians(wind_from)) * 0.3 * rows
        cc += sign * math.sin(math.radians(wind_from)) * 0.3 * cols
    rr, ccs = np.mgrid[0:rows, 0:cols]
    order = np.argsort(((rr - cr) ** 2 + (ccs - cc) ** 2).ravel(), kind="stable")[:max(1, count)]
    mask = np.zeros(rows * cols, dtype=bool)
    mask[order] = True
    return mask.reshape(rows, cols)


def _ignition_cells(n: int, count: int, placement: str, wind_speed: float, wind_from: float) -> np.ndarray:
    """Square-area version kept for callers of the original 500 m box."""
    return _placement_mask(n, n, count, placement, wind_speed, wind_from)


def ignition_mask(domain: SimulationDomain, n_ignition: int, placement: str, wind_speed: float,
                  wind_from: float, points: Optional[Sequence[Tuple[float, float]]] = None,
                  strict_points: bool = False) -> np.ndarray:
    """Ignition cells on the domain grid: placed inside the focus area, or at
    user-picked map points (each point ignites the cell that contains it).
    strict_points=True (observed NASA FIRMS ignitions): only the given points,
    never a fallback placement - no point inside the domain, no ignition."""
    mask = np.zeros((domain.n_rows, domain.n_cols), dtype=bool)
    if strict_points or (placement.lower().startswith("map") and points):
        for lat, lon in points or []:
            rc = domain.cell_of(lat, lon)
            if rc is not None:
                mask[rc] = True
        if mask.any() or strict_points:
            return mask
    f = domain.focus
    mask[domain.focus_slice()] = _placement_mask(f.n_rows, f.n_cols, n_ignition, placement, wind_speed, wind_from)
    return mask


def _perimeter_m(affected: np.ndarray, cell_m: float) -> float:
    """Length of the boundary between affected (burning or burned) and
    unaffected ground."""
    a = np.pad(affected, 1, constant_values=False).astype(np.int8)
    edges = (np.abs(np.diff(a, axis=0)).sum() + np.abs(np.diff(a, axis=1)).sum())
    return float(edges) * cell_m


def run_local_spread(focus: FocusArea, conditions: dict, wind_speed_ms: float, wind_from_deg: float,
                     n_ignition: int = 3, placement: str = "Upwind edge", seed: int = 42,
                     horizon_minutes: Optional[float] = None,
                     elevation: Optional[np.ndarray] = None, terrain_source: str = "flat",
                     wind_schedule_15min: Optional[Sequence[Tuple[float, float]]] = None,
                     base_spread_prob: Optional[float] = None,
                     duration_minutes: Optional[float] = None,
                     ignition_points: Optional[Sequence[Tuple[float, float]]] = None,
                     land_cover=None, strict_points: bool = False) -> LocalSpreadResult:
    """Run FireSpreadSimulator on the simulation domain around `focus`.

    duration_minutes (or horizon_minutes): simulated time to cover; defaults
        to SYSTEM.fire_spread_horizon_hours.
    wind_speed_ms / wind_from_deg: constant wind (meteorological FROM bearing,
        the convention the CA and the dashboard compass use).
    wind_schedule_15min: optional forecast schedule at the CA's native 15-min
        steps (as produced by DigitalTwin._spread_wind); resampled to the local
        step.
    elevation: (domain rows, domain cols) metres; flat if None.
    base_spread_prob: FireSpreadSimulator's existing calibration parameter;
        defaults to SYSTEM.local_ca_base_spread_prob (see config.py).
    ignition_points: (lat, lon) pairs, used when placement == "Map points".
    strict_points: ignite ONLY ignition_points (observed NASA FIRMS cells), with
        no fallback placement.
    land_cover: optional fuel_map.LandCover for the domain. Its WATER / BUILT /
        ROAD / NON_FUEL cells join the CA's existing non-fuel mask, so they never
        ignite and fire never propagates into them; FUEL cells behave exactly as
        before.
    """
    duration = float(duration_minutes or horizon_minutes or SYSTEM.fire_spread_horizon_hours * 60)
    domain = domain_for(focus, duration)
    rows, cols = domain.n_rows, domain.n_cols
    step_min = step_minutes_for(focus.cell_m)
    n_steps = n_steps_for(duration, focus.cell_m)

    ca = ca_inputs_from_conditions(conditions)
    dryness = np.full((rows, cols), ca["dryness"])
    fuel = np.full((rows, cols), ca["fuel"])
    buildup = np.full((rows, cols), ca["buildup"])
    non_fuel = np.full((rows, cols), ca["non_fuel"], dtype=bool)
    lc_classes, lc_label = None, "No land-cover layer"
    if land_cover is not None and getattr(land_cover, "classes", None) is not None \
            and land_cover.classes.shape == (rows, cols):
        lc_classes, lc_label = land_cover.classes.copy(), land_cover.label
        non_fuel |= land_cover.non_fuel
    if elevation is None or elevation.shape != (rows, cols):
        elevation, terrain_source = np.zeros((rows, cols)), "flat"

    if wind_schedule_15min:
        sched = [tuple(map(float, wind_schedule_15min[min(int(k * step_min // CA_STEP_MINUTES),
                                                          len(wind_schedule_15min) - 1)]))
                 for k in range(n_steps)]
    else:
        sched = [(float(wind_speed_ms), float(wind_from_deg) % 360)] * n_steps

    requested = ignition_mask(domain, n_ignition, placement, sched[0][0], sched[0][1], ignition_points,
                              strict_points=strict_points)
    ignition = requested & ~non_fuel                     # water / road / built / bare cells never ignite

    base = float(SYSTEM.local_ca_base_spread_prob if base_spread_prob is None else base_spread_prob)
    sim = FireSpreadSimulator(rows, cols, minutes_per_step=CA_STEP_MINUTES, base_spread_prob=base,
                              random_state=seed, cell_size_deg=focus.cell_m / METERS_PER_DEG_LAT)
    history = sim.run(ignition_mask=ignition, dryness_grid=dryness, fuel_load_grid=fuel,
                      non_fuel_mask=non_fuel, wind_speed_ms=sched[0][0], wind_from_deg=sched[0][1],
                      horizon_minutes=n_steps * CA_STEP_MINUTES, elevation_grid=elevation,
                      fuel_buildup_grid=buildup, wind_schedule=sched)
    for h in history:                                    # relabel to the local time step
        h.minutes_elapsed = round(h.step * step_min, 3)

    ign = np.full((rows, cols), -1, dtype=int)
    out = np.full((rows, cols), -1, dtype=int)
    for h in history:
        ign[(h.state == CellState.BURNING) & (ign < 0)] = h.step
        out[(h.state == CellState.BURNED) & (out < 0)] = h.step

    # Visual fire intensity (0-1): the CA's own spread potential for the cell
    # (dryness x fuel x build-up x wind speed, no direction) scaled by how the
    # fire reached it - head fire (driven downwind) burns hotter than flank or
    # backing fire. Derived from the CA state, used only to scale the VFX.
    potential = sim._cell_spread_prob(ca["dryness"], ca["fuel"], ca["buildup"], sched[0][0], 1.0, 1.0)
    pot_norm = min(1.0, potential / (0.5 * base))
    intensity = np.zeros((rows, cols))
    offsets = sim._neighbor_offsets
    for r, c in zip(*np.where(ign >= 0)):
        s = ign[r, c]
        if s == 0:
            align = 1.0
        else:
            prev = history[s - 1].state
            frm = sched[min(s - 1, len(sched) - 1)][1]
            align = 0.4
            for dr, dc, bearing in offsets:
                pr, pc = r - dr, c - dc                  # neighbour that could have ignited (r, c)
                if 0 <= pr < rows and 0 <= pc < cols and prev[pr, pc] == CellState.BURNING:
                    align = max(align, sim._wind_alignment_factor(bearing, frm))
        intensity[r, c] = float(np.clip((align / 1.8) * (0.45 + 0.55 * pot_norm), 0.12, 1.0))

    # Per-step analytics, all read from the CA state.
    focus_mask = domain.focus_mask()
    lat_c, lon_c = domain.cell_centres()
    centre_r, centre_c = np.argwhere(ignition).mean(axis=0) if ignition.any() else ((rows - 1) / 2, (cols - 1) / 2)
    rr, cc = np.mgrid[0:rows, 0:cols]
    dist_m = np.hypot(rr - centre_r, cc - centre_c) * focus.cell_m
    cell_area_ha = focus.cell_m ** 2 / 1e4
    metrics, prev_d = [], 0.0
    for h in history:
        burning = h.state == CellState.BURNING
        burned = h.state == CellState.BURNED
        affected = burning | burned
        d = float(dist_m[affected].max()) if affected.any() else 0.0
        outside = bool((affected & ~focus_mask).any())
        edge = bool(affected[0, :].any() or affected[-1, :].any() or affected[:, 0].any() or affected[:, -1].any())
        mean_int = float(intensity[burning].mean()) if burning.any() else 0.0
        max_int = float(intensity[burning].max()) if burning.any() else 0.0
        w = sched[min(max(h.step - 1, 0), len(sched) - 1)]
        metrics.append({
            "step": h.step, "minutes": h.minutes_elapsed,
            "burning": int(burning.sum()), "burned": int(burned.sum()),
            "burned_ha": round(float(burned.sum()) * cell_area_ha, 4),
            "fire_area_ha": round(float(affected.sum()) * cell_area_ha, 4),
            "burned_in_focus": int((burned & focus_mask).sum()),
            "perimeter_m": _perimeter_m(affected, focus.cell_m),
            "front_distance_m": round(d, 1),
            "ros_m_per_min": round((d - prev_d) / step_min, 2) if h.step else 0.0,
            "mean_intensity": round(mean_int, 3), "max_intensity": round(max_int, 3),
            "intensity_class": intensity_class(mean_int) if mean_int else "-",
            "centroid_lat": round(float(lat_c[affected].mean()), 6) if affected.any() else None,
            "centroid_lon": round(float(lon_c[affected].mean()), 6) if affected.any() else None,
            "left_focus": outside, "boundary_reached": outside, "domain_edge_reached": edge,
            "wind_speed_ms": w[0], "wind_from_deg": w[1],
        })
        prev_d = d
    if metrics and metrics[-1]["domain_edge_reached"]:          # cannot happen by construction
        logger.warning("Fire reached the simulation-domain edge; margin=%d cells", domain.margin)

    params = {"n_ignition": int(ignition.sum()), "placement": placement, "seed": int(seed),
              "duration_minutes": duration, "horizon_minutes": duration, "n_steps_planned": n_steps,
              "spread_potential": round(potential, 4), "base_spread_prob": base,
              "cell_m": focus.cell_m, "size_m": focus.size_m, "width_m": focus.n_cols * focus.cell_m,
              "height_m": focus.n_rows * focus.cell_m, "domain_margin": domain.margin,
              "ignition_cells_requested": int(requested.sum()),
              "ignition_cells_on_non_fuel": int((requested & non_fuel).sum()),
              "non_fuel_cells": int(non_fuel.sum()), "land_cover": lc_label, **ca}
    return LocalSpreadResult(focus=focus, domain=domain, step_minutes=step_min, duration_minutes=duration,
                             history=history, ignition_step=ign, burnout_step=out, intensity=intensity,
                             non_fuel=non_fuel, elevation=elevation, terrain_source=terrain_source,
                             wind_schedule=sched, conditions=conditions, params=params, metrics=metrics,
                             land_cover=lc_classes, land_cover_label=lc_label)
