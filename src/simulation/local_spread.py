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
    """The grid the CA actually runs on: the focus area plus a margin of cells
    on every side, same cell lattice. `margins` (north, south, west, east)
    overrides the symmetric `margin` once the domain has grown asymmetrically
    (adaptive expansion, ignition-aware initial domain)."""
    focus: FocusArea
    margin: int
    margins: Optional[Tuple[int, int, int, int]] = None

    @property
    def m_north(self) -> int:
        return self.margins[0] if self.margins else self.margin

    @property
    def m_south(self) -> int:
        return self.margins[1] if self.margins else self.margin

    @property
    def m_west(self) -> int:
        return self.margins[2] if self.margins else self.margin

    @property
    def m_east(self) -> int:
        return self.margins[3] if self.margins else self.margin

    @property
    def margin_tuple(self) -> Tuple[int, int, int, int]:
        return self.m_north, self.m_south, self.m_west, self.m_east

    @property
    def symmetric(self) -> bool:
        return len(set(self.margin_tuple)) == 1

    @property
    def cell_m(self) -> float:
        return self.focus.cell_m

    @property
    def n_rows(self) -> int:
        return self.focus.n_rows + self.m_north + self.m_south

    @property
    def n_cols(self) -> int:
        return self.focus.n_cols + self.m_west + self.m_east

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
        if self.margins is None:
            return {"north": f.lat + self.n_rows / 2.0 * f.dlat, "south": f.lat - self.n_rows / 2.0 * f.dlat,
                    "west": f.lon - self.n_cols / 2.0 * f.dlon, "east": f.lon + self.n_cols / 2.0 * f.dlon}
        return {"north": f.lat + (f.n_rows / 2.0 + self.m_north) * f.dlat,
                "south": f.lat - (f.n_rows / 2.0 + self.m_south) * f.dlat,
                "west": f.lon - (f.n_cols / 2.0 + self.m_west) * f.dlon,
                "east": f.lon + (f.n_cols / 2.0 + self.m_east) * f.dlon}

    @property
    def area_km2(self) -> float:
        return self.n_rows * self.n_cols * self.cell_m ** 2 / 1e6

    def focus_slice(self) -> Tuple[slice, slice]:
        """Rows / cols of the focus area inside the domain (clipped: a domain moved on the map may
        cover only part of the focus area, or none of it)."""
        r0, c0 = self.m_north, self.m_west
        return (slice(max(0, r0), max(0, min(self.n_rows, r0 + self.focus.n_rows))),
                slice(max(0, c0), max(0, min(self.n_cols, c0 + self.focus.n_cols))))

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

    def grown(self, north: int = 0, south: int = 0, west: int = 0, east: int = 0) -> "SimulationDomain":
        """The same domain with extra cells on some sides (same lattice)."""
        mn, ms, mw, me = self.margin_tuple
        return SimulationDomain(self.focus, self.margin, (mn + north, ms + south, mw + west, me + east))

    def describe(self) -> Dict[str, object]:
        b = self.bounds
        return {"rows": self.n_rows, "cols": self.n_cols, "margins_nswe": list(self.margin_tuple),
                "bounds": {k: round(v, 6) for k, v in b.items()},
                "width_m": round(self.width_m, 1), "height_m": round(self.height_m, 1)}


def domain_for(focus: FocusArea, duration_minutes: float) -> SimulationDomain:
    """Domain large enough that the fire cannot reach its edge within the
    duration (the CA moves at most one cell per step). This is the area whose
    land cover is loaded for set-up (map clicks are validated against it) and
    the default initial simulation domain."""
    return SimulationDomain(focus=focus, margin=n_steps_for(duration_minutes, focus.cell_m) + 1)


# Phase 3 - user-configurable initial simulation domain (What-If "Simulation domain").
# Each preset is a real domain size (metres per side) used by the engine and drawn on the map.
DOMAIN_AUTO = "Auto (focus + duration margin)"
DOMAIN_CUSTOM = "Custom"
DOMAIN_MAP = "Drawn on map"           # moved / resized directly on the map (independent of the focus area)
DOMAIN_PRESETS: Dict[str, Optional[float]] = {DOMAIN_AUTO: None, "Local · 1 km": 1000.0, "Small · 2 km": 2000.0,
                                               "Medium · 5 km": 5000.0, "Large · 10 km": 10000.0,
                                               DOMAIN_CUSTOM: None, DOMAIN_MAP: None}
MAX_EXTENT_OPTIONS_M = (2000.0, 5000.0, 10000.0, 15000.0)
LARGE_GRID_WARN_CELLS = 100_000


def domain_box_from_setup(setup: dict) -> Optional[Dict[str, float]]:
    """Explicit domain bounds (north / south / east / west) when the domain was drawn on the map."""
    b = setup.get("domain_box")
    if (setup.get("domain_preset") == DOMAIN_MAP and isinstance(b, dict)
            and all(k in b for k in ("north", "south", "east", "west"))):
        return {k: float(b[k]) for k in ("north", "south", "east", "west")}
    return None


def domain_size_from_setup(setup: dict) -> Optional[Tuple[float, float]]:
    """(width, height) in metres of the configured initial domain, or None for Auto."""
    p = setup.get("domain_preset") or DOMAIN_AUTO
    if p == DOMAIN_MAP:
        b = domain_box_from_setup(setup)
        if b is None:
            return None
        lat = 0.5 * (b["north"] + b["south"])
        return ((b["east"] - b["west"]) * METERS_PER_DEG_LAT * math.cos(math.radians(lat)),
                (b["north"] - b["south"]) * METERS_PER_DEG_LAT)
    if p == DOMAIN_CUSTOM:
        return float(setup.get("domain_w_m") or 0.0), float(setup.get("domain_h_m") or 0.0)
    v = DOMAIN_PRESETS.get(p)
    return (float(v), float(v)) if v else None


def limits_from_setup(setup: dict) -> dict:
    return {"max_extent_m": float(setup.get("max_extent_m") or 0.0)} if setup.get("max_extent_m") else {}


def focus_of_setup(setup: dict) -> FocusArea:
    loc = setup["location"]
    return FocusArea(name=loc["name"], lat=float(loc["lat"]), lon=float(loc["lon"]),
                     width_m=float(setup["width_m"]), height_m=float(setup["height_m"]),
                     cell_m=float(setup["cell_m"]), source=loc.get("source", "preset"))


def domain_from_setup(setup: dict, ignition_points=None) -> SimulationDomain:
    """THE initial simulation domain of a set-up: the map overlay, ignition
    validation and the engine all use this one function."""
    return initial_domain(focus_of_setup(setup), float(setup["duration_min"]), ignition_points,
                          domain_size_m=domain_size_from_setup(setup), limits=limits_from_setup(setup),
                          domain_box=domain_box_from_setup(setup))


def validate_domain(focus: FocusArea, size_m: Optional[Tuple[float, float]], limits: Optional[dict] = None,
                    duration_minutes: float = 60.0, domain_box: Optional[dict] = None
                    ) -> Tuple[Optional[str], Optional[str]]:
    """(error, warning) for a configured domain; error blocks the run."""
    L = _limits(limits)
    if domain_box is not None:
        d = initial_domain(focus, duration_minutes, domain_box=domain_box, limits=limits)
        w, h = d.width_m, d.height_m
        if min(d.n_rows, d.n_cols) < 4:
            return "The simulation domain drawn on the map is too small (at least 4 cells per side).", None
        if max(w, h) > L["max_extent_m"] + 1e-6:
            return (f"The simulation domain drawn on the map ({w / 1000:.1f} × {h / 1000:.1f} km) exceeds the "
                    f"maximum simulation extent ({L['max_extent_m'] / 1000:.0f} km). Make it smaller on the map or "
                    "raise the maximum extent."), None
        n = d.n_rows * d.n_cols
        if max(d.n_rows, d.n_cols) > L["max_side_cells"] or n > L["max_total_cells"]:
            return (f"That domain needs {d.n_cols} × {d.n_rows} = {n:,} cells at {focus.cell_m:.0f} m, above the safe "
                    f"limit ({L['max_side_cells']} per side, {L['max_total_cells']:,} in total). Use larger cells or "
                    "draw a smaller domain."), None
        warns = []
        fm = d.focus_mask()
        if not fm.any():
            warns.append("The focus area lies outside the simulation domain: focus-area statistics will be 0.")
        elif fm.sum() < focus.n_rows * focus.n_cols:
            warns.append("The simulation domain covers only part of the focus area.")
        if n > LARGE_GRID_WARN_CELLS:
            warns.append(f"Large grid: {d.n_cols} × {d.n_rows} = {n:,} cells; the run and the map will be slower.")
        return None, (" ".join(warns) or None)
    if size_m is not None:
        w, h = size_m
        fw, fh = focus.n_cols * focus.cell_m, focus.n_rows * focus.cell_m
        if not (w > 0 and h > 0):
            return "Enter a simulation-domain width and height.", None
        if w + 1e-6 < fw or h + 1e-6 < fh:
            return (f"The simulation domain ({w:.0f} × {h:.0f} m) must contain the focus area ({fw:.0f} × "
                    f"{fh:.0f} m). Choose a larger domain or a smaller focus area."), None
        if max(w, h) > L["max_extent_m"] + 1e-6:
            return (f"The simulation domain ({w / 1000:.1f} × {h / 1000:.1f} km) exceeds the maximum simulation "
                    f"extent ({L['max_extent_m'] / 1000:.0f} km). Raise the maximum extent or choose a smaller "
                    "domain."), None
    d = initial_domain(focus, duration_minutes, domain_size_m=size_m, limits=limits)
    n = d.n_rows * d.n_cols
    if max(d.n_rows, d.n_cols) > L["max_side_cells"] or n > L["max_total_cells"]:
        return (f"That domain needs {d.n_cols} × {d.n_rows} = {n:,} cells at {focus.cell_m:.0f} m, above the safe "
                f"limit ({L['max_side_cells']} cells per side, {L['max_total_cells']:,} in total) that keeps the "
                f"simulation and the browser responsive. Use larger cells (e.g. 50 m) or a smaller domain."), None
    warn = None
    if n > LARGE_GRID_WARN_CELLS:
        warn = (f"Large grid: {d.n_cols} × {d.n_rows} = {n:,} cells. The run and the map will be slower; larger "
                f"cells keep it responsive.")
    return None, warn


def initial_domain(focus: FocusArea, duration_minutes: float,
                   ignition_points: Optional[Sequence[Tuple[float, float]]] = None,
                   domain_size_m: Optional[Tuple[float, float]] = None,
                   limits: Optional[dict] = None, domain_box: Optional[dict] = None) -> SimulationDomain:
    """Starting computational domain of a run.

    * the full-duration domain (domain_for) when it is at most
      DOMAIN.initial_domain_max_cells per side - the fire cannot reach its edge;
    * otherwise the focus area plus DOMAIN.compact_margin_cells: the domain then
      grows while the fire spreads (adaptive expansion), instead of allocating
      a huge grid up front;
    * ignition-aware (audit BUG #1): every ignition point gets at least
      ignition_edge_margin + safety_margin cells to the edge, so an ignition
      placed near the edge is never truncated."""
    from config.config import DOMAIN
    if domain_box is not None:
        # drawn on the map: its own position and size, on the focus area's cell lattice; margins may be
        # negative (the domain can be moved away from the focus area - they are independent objects)
        fb = focus.bounds
        mn = int(round((domain_box["north"] - fb["north"]) / focus.dlat))
        ms = int(round((fb["south"] - domain_box["south"]) / focus.dlat))
        mw = int(round((fb["west"] - domain_box["west"]) / focus.dlon))
        me = int(round((domain_box["east"] - fb["east"]) / focus.dlon))
        if focus.n_rows + mn + ms < 4:
            ms = 4 - focus.n_rows - mn
        if focus.n_cols + mw + me < 4:
            me = 4 - focus.n_cols - mw
        d = SimulationDomain(focus, min(mn, ms, mw, me), (mn, ms, mw, me))
    elif domain_size_m is not None:
        # user-configured domain: at least w x h metres, centred on the focus area (same cell lattice)
        w, h = domain_size_m
        ex = max(0, int(math.ceil((float(w) - focus.n_cols * focus.cell_m) / focus.cell_m - 1e-9)))
        ey = max(0, int(math.ceil((float(h) - focus.n_rows * focus.cell_m) / focus.cell_m - 1e-9)))
        d = SimulationDomain(focus, min(ex // 2, ey // 2), (ey // 2, ey - ey // 2, ex // 2, ex - ex // 2))
        if len(set(d.margin_tuple)) == 1:
            d = SimulationDomain(focus, d.margin_tuple[0])
    else:
        d = domain_for(focus, duration_minutes)
        if max(d.n_rows, d.n_cols) > DOMAIN.initial_domain_max_cells:
            d = SimulationDomain(focus, min(d.margin, DOMAIN.compact_margin_cells))
    need = DOMAIN.ignition_edge_margin_cells + DOMAIN.safety_margin_cells
    grow = [0, 0, 0, 0]
    for lat, lon in ignition_points or []:
        rc = d.cell_of(float(lat), float(lon))
        if rc is None:                    # outside the loaded coverage: never ignited, never grows the domain
            continue
        r, c = rc
        grow[0] = max(grow[0], need - r)
        grow[1] = max(grow[1], need - (d.n_rows - 1 - r))
        grow[2] = max(grow[2], need - c)
        grow[3] = max(grow[3], need - (d.n_cols - 1 - c))
    grow = [max(0, int(g)) for g in grow]
    # never beyond the configured limits (the run reports SIMULATION EXTENT LIMIT REACHED instead)
    L = _limits(limits)
    max_side = min(L["max_side_cells"], int(L["max_extent_m"] // focus.cell_m))
    while any(grow) and (d.n_rows + grow[0] + grow[1] > max_side or d.n_cols + grow[2] + grow[3] > max_side
                         or (d.n_rows + grow[0] + grow[1]) * (d.n_cols + grow[2] + grow[3]) > L["max_total_cells"]):
        k = int(np.argmax(grow))
        grow[k] -= 1
    if any(grow):
        d = d.grown(*grow)
    return d


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
    from config.config import DOMAIN
    d = initial_domain(FocusArea("check", 0.0, 0.0, width_m, height_m, cell_m), duration_minutes)
    if max(d.n_rows, d.n_cols) > DOMAIN.max_side_cells or d.n_rows * d.n_cols > DOMAIN.max_total_cells:
        return (f"That needs a {d.n_cols} x {d.n_rows}-cell initial simulation domain (the area plus a margin), "
                f"above the cell limit ({DOMAIN.max_side_cells} cells per side, {DOMAIN.max_total_cells:,} cells in "
                f"total) that keeps the simulation and the browser responsive. Use larger cells or a smaller area.")
    return None


# Named protected areas (centre coordinates of the reserve / park).
FOCUS_AREAS: Dict[str, FocusArea] = {
    "Bandipur Tiger Reserve": FocusArea(
        # Phase 3: forest interior instead of the old reserve "centre" point (11.6667, 76.6333), which lies
        # on the Bandipur campus / NH 766 roadside (buildings, clearings). 11.6560 N, 76.5800 E is closed
        # dry deciduous canopy west of the Gundlupet-Ooty road, chosen by visual inspection of Google
        # satellite imagery (Imagery (c) 2026 Airbus / Maxar, checked 2026-10-09): nearest forest track
        # ~300 m north, no buildings, water or farmland within ~1 km. Not a land-cover classification;
        # inside the APPROXIMATE / PROVISIONAL Bandipur boundary of data/study_region.
        name="Bandipur Tiger Reserve", lat=11.6560, lon=76.5800, state="Karnataka",
        description="Tiger reserve in Chamarajanagar district, Nilgiri Biosphere Reserve "
                    "(default view: forest interior west of the Gundlupet-Ooty road)."),
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
        "ffmc": g("ffmc"), "bui": g("bui"), "fwi": g("fwi"), "ndvi": g("ndvi"), "dmc": g("dmc"), "dc": g("dc"),
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
MAX_DEM_POINTS = 900                  # ~90 m lattice up to ~2.6 km domains, coarser beyond (slope factor only)
DEM_FETCH_BUDGET_S = 12.0             # a RUN never waits longer than this for terrain (then: flat, labelled)


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
                                    provider="fast", retries=1, pause_s=0.05, timeout=6,
                                    deadline_s=DEM_FETCH_BUDGET_S)
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
    domain: SimulationDomain            # FINAL domain (after any expansion); all arrays are on this grid
    step_minutes: float
    duration_minutes: float
    history: list                       # SimulationStep list (minutes relabelled to local time), final grid
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
    # adaptive domain, fuel, study region, timings (see run_local_spread)
    initial_domain: Optional[SimulationDomain] = None
    expansions: List[dict] = field(default_factory=list)
    extent_limit: Optional[dict] = None         # set when a configured limit / missing data stopped expansion
    fuel_load: Optional[np.ndarray] = None      # (rows, cols) fuel load the CA used
    land_confidence: Optional[np.ndarray] = None
    land_source_mask: Optional[np.ndarray] = None
    land_status: str = ""
    study_region: Optional[dict] = None         # Bandipur boundary info, crossing, split statistics
    ignition_time: Optional[np.ndarray] = None  # ros-ca: simulated minute each cell ignited (-1 = never)
    burnout_time: Optional[np.ndarray] = None   # ros-ca: minute it stopped flaming (-1 = never)
    fireline_kw: Optional[np.ndarray] = None    # ros-ca: Byram fireline intensity (kW/m) of the spread into the cell
    inside_region: Optional[np.ndarray] = None  # (rows, cols) bool, cell centre inside the study region
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def final(self) -> dict:
        return self.metrics[-1] if self.metrics else {}

    @property
    def end_step(self) -> float:
        """The requested duration in CA steps (may end part-way through a step)."""
        return min(self.duration_minutes / self.step_minutes, float(self.history[-1].step))


EXTENT_LIMIT_TITLE = "SIMULATION EXTENT LIMIT REACHED"


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


def fuel_variability_field(lat_c: np.ndarray, lon_c: np.ndarray, dlat: float, dlon: float,
                           amplitude: float, sharpness: Optional[float] = None) -> np.ndarray:
    """Multiplier 1 +/- amplitude: smooth value noise (two octaves, ~4 and ~11
    cells) seeded by the ABSOLUTE cell position, so it is fixed for a place and
    identical for any domain / expansion strip on the same lattice. It models
    natural patchiness of fuel between cells; it is SIMULATED, not observed."""
    def octave(scale):
        gx = np.asarray(lon_c, dtype=float) / dlon / scale
        gy = np.asarray(lat_c, dtype=float) / dlat / scale
        x0, y0 = np.floor(gx), np.floor(gy)
        fx, fy = gx - x0, gy - y0
        fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
        x0, y0 = x0.astype(np.int64), y0.astype(np.int64)

        def h(x, y):
            v = (x * 73856093) ^ (y * 19349663) ^ (scale * 83492791)
            v = (v ^ (v >> 13)) * 1274126177
            return ((v ^ (v >> 16)) & 0xFFFF) / 65535.0
        a, b, c, d = h(x0, y0), h(x0 + 1, y0), h(x0, y0 + 1), h(x0 + 1, y0 + 1)
        return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy
    n = 2.0 * (0.6 * octave(4) + 0.4 * octave(11)) - 1.0
    # patch contrast: gamma < 1 pushes the smooth noise towards distinct dense / sparse fuel patches, which
    # gives the irregular, fingered perimeter of real fires (fuel INPUT only - the spread model is unchanged)
    g = float(SYSTEM.local_ca_fuel_patch_gamma if sharpness is None else sharpness)
    if g != 1.0:
        n = np.clip(np.sign(n) * np.abs(n) ** g * 1.8, -1.0, 1.0)
    return 1.0 + float(amplitude) * n


def _grid_of(domain: SimulationDomain):
    from src.landcover.grid import GridSpec
    return GridSpec.from_domain(domain)


def _active_sides(state: np.ndarray, safety: int, steps_left: int) -> Dict[str, bool]:
    """Sides that must grow before the next step: a BURNING cell is closer than
    `safety` cells to the edge AND could still reach it within the remaining
    steps (the CA moves at most one cell per step). A burning cell d cells from
    the edge can be burning ON the edge d steps later, so a side is only grown
    when d <= steps_left + 1 - expansion is never wasted on a fire that cannot
    get there, and the fire never even touches the computational edge."""
    br, bc = np.nonzero(state == CellState.BURNING)
    if br.size == 0:
        return {"north": False, "south": False, "west": False, "east": False}
    rows, cols = state.shape
    d = {"north": int(br.min()), "south": rows - 1 - int(br.max()),
         "west": int(bc.min()), "east": cols - 1 - int(bc.max())}
    return {k: v < safety and v <= steps_left + 1 for k, v in d.items()}


def run_local_spread(focus: FocusArea, conditions: dict, wind_speed_ms: float, wind_from_deg: float,
                     n_ignition: int = 3, placement: str = "Upwind edge", seed: int = 42,
                     horizon_minutes: Optional[float] = None,
                     elevation: Optional[np.ndarray] = None, terrain_source: str = "flat",
                     wind_schedule_15min: Optional[Sequence[Tuple[float, float]]] = None,
                     base_spread_prob: Optional[float] = None,
                     duration_minutes: Optional[float] = None,
                     ignition_points: Optional[Sequence[Tuple[float, float]]] = None,
                     land_cover=None, strict_points: bool = False,
                     domain: Optional[SimulationDomain] = None,
                     land_provider=None, elevation_provider=None, expand: bool = True,
                     allow_unverified_fuel: bool = False, study_region=None,
                     timings=None, fuel_variability: Optional[float] = None,
                     diagonal_correction: Optional[bool] = None, model: Optional[str] = None,
                     limits: Optional[dict] = None) -> LocalSpreadResult:
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
    strict_points: ignite ONLY ignition_points (observed NASA FIRMS ignitions), with
        no fallback placement.
    land_cover: optional fuel_map.LandCover for the INITIAL domain (exactly its
        grid, otherwise ValueError). Its WATER / BUILT / ROAD / NON_FUEL (and
        UNKNOWN, unless allow_unverified_fuel) cells join the CA's non-fuel mask,
        so they never ignite and fire never propagates into them. A fused land
        cover also provides the per-cell fuel load (Sentinel-2 NDVI proxy or
        WorldCover class default) instead of the zone's (synthetic) NDVI.
    domain: initial domain (default domain_for(focus, duration)).
    expand / land_provider / elevation_provider: ADAPTIVE DOMAIN. Before every
        CA step, a side with a burning cell closer than DOMAIN.safety_margin_cells
        to the edge grows; the new cells get land cover from
        land_provider(GridSpec) and elevation from elevation_provider(GridSpec).
        The CA state, the step counter, the wind schedule and the random stream
        continue unchanged (no restart). Without land_provider the new cells
        can only be filled when there is no land-cover layer at all (legacy
        all-fuel run); otherwise expansion stops and the result reports
        SIMULATION EXTENT LIMIT REACHED. Limits: DOMAIN.max_side_cells,
        max_total_cells, max_extent_m, max_expansions.
    study_region: src.geo.study_region.StudyRegion (Bandipur) for the
        boundary crossing and the Bandipur / REGIONAL EXTENSION statistics.
    """
    from config.config import DOMAIN
    from src.simulation.fuel_map import UNKNOWN
    from src.utils.timing import Timings
    tm = timings or Timings()
    duration = float(duration_minutes or horizon_minutes or SYSTEM.fire_spread_horizon_hours * 60)
    dom = domain or domain_for(focus, duration)
    dom0 = dom
    rows, cols = dom.n_rows, dom.n_cols
    step_min = step_minutes_for(focus.cell_m)
    n_steps = n_steps_for(duration, focus.cell_m)

    var_spec = None
    with tm.stage("ca_setup"):
        ca = ca_inputs_from_conditions(conditions)
        dryness = np.full((rows, cols), ca["dryness"])
        buildup = np.full((rows, cols), ca["buildup"])
        lc_classes, lc_label, lc_fuel, lc_conf, lc_src, lc_status = None, "No land-cover layer", None, None, None, ""
        if land_cover is not None and getattr(land_cover, "classes", None) is not None:
            if land_cover.classes.shape != (rows, cols):
                # never run without the authoritative mask by accident (it would let every cell burn)
                raise ValueError(f"land cover grid {land_cover.classes.shape} does not match the simulation domain "
                                 f"{(rows, cols)}")
            lc_classes, lc_label = land_cover.classes.copy(), land_cover.label
            lc_status = getattr(land_cover, "status", "") or ""
            if getattr(land_cover, "fuel_load", None) is not None:
                lc_fuel = np.asarray(land_cover.fuel_load, dtype=float).copy()
                lc_conf = None if land_cover.confidence is None else land_cover.confidence.copy()
                lc_src = None if land_cover.source_mask is None else land_cover.source_mask.copy()
        if lc_fuel is not None:
            # per-cell fuel from the fused land cover; the zone's NDVI (synthetic) is not used
            fuel = lc_fuel
            fuel_source = "land cover: " + ("Sentinel-2 NDVI proxy" if "Sentinel-2" in lc_label or "NDVI" in lc_label
                                            else "WorldCover class defaults (NDVI unavailable)")
            non_fuel = np.zeros((rows, cols), dtype=bool)
        else:
            fuel = np.full((rows, cols), ca["fuel"])
            fuel_source = "zone NDVI value (SYNTHETIC estimate, uniform)"
            amp = float(SYSTEM.local_ca_fuel_variability if fuel_variability is None else fuel_variability)
            if amp > 0:
                lat_c, lon_c = dom.cell_centres()
                fuel = np.clip(fuel * fuel_variability_field(lat_c, lon_c, focus.dlat, focus.dlon, amp), 0.0, 1.0)
                fuel_source = (f"zone NDVI value (SYNTHETIC estimate) with ±{amp * 100:.0f}% simulated "
                               "cell-to-cell variability")
                var_spec = (amp, focus.dlat, focus.dlon)
            non_fuel = np.full((rows, cols), ca["non_fuel"], dtype=bool)
        if lc_classes is not None:
            non_fuel |= (land_cover.non_burnable(allow_unverified_fuel) if hasattr(land_cover, "non_burnable")
                         else land_cover.non_fuel)
        if elevation is None or elevation.shape != (rows, cols):
            elevation, terrain_source = np.zeros((rows, cols)), "flat"
        elevation = np.asarray(elevation, dtype=float)

        if wind_schedule_15min:
            sched = [tuple(map(float, wind_schedule_15min[min(int(k * step_min // CA_STEP_MINUTES),
                                                              len(wind_schedule_15min) - 1)]))
                     for k in range(n_steps)]
        else:
            sched = [(float(wind_speed_ms), float(wind_from_deg) % 360)] * n_steps

        requested = ignition_mask(dom, n_ignition, placement, sched[0][0], sched[0][1], ignition_points,
                                  strict_points=strict_points)
        ignition = requested & ~non_fuel                     # water / road / built / bare cells never ignite

        base = float(SYSTEM.local_ca_base_spread_prob if base_spread_prob is None else base_spread_prob)
        diag = SYSTEM.local_ca_diagonal_correction if diagonal_correction is None else diagonal_correction
        sim = FireSpreadSimulator(rows, cols, minutes_per_step=CA_STEP_MINUTES, base_spread_prob=base,
                                  random_state=seed, cell_size_deg=focus.cell_m / METERS_PER_DEG_LAT,
                                  diagonal_correction=diag, wind_model=SYSTEM.local_ca_wind_model)

    mdl = (model or SYSTEM.local_ca_model or "legacy").lower()
    if mdl == "ros":
        return _run_ros(focus, dom, conditions, ca, dryness, buildup, fuel, non_fuel, elevation, terrain_source,
                        float(wind_speed_ms), float(wind_from_deg) % 360, wind_schedule_15min, ignition, requested,
                        duration, seed, placement, expand, land_provider, elevation_provider, allow_unverified_fuel,
                        study_region, tm, fuel_source, var_spec, lc_classes, lc_label, lc_conf, lc_src, lc_status,
                        lc_fuel, limits)

    # ── CA run; before every step the adaptive domain may grow ──────────── #
    expansions: List[dict] = []
    limit_box: Dict[str, Optional[dict]] = {"limit": None}
    blocked: set = set()
    safety = max(2, int(DOMAIN.safety_margin_cells))
    grid_arrays = {"dryness": dryness, "fuel": fuel, "buildup": buildup, "elev": elevation, "non_fuel": non_fuel,
                   "fuel_var": var_spec}
    lc_arrays = {"classes": lc_classes, "conf": lc_conf, "src": lc_src}
    dom_box = {"dom": dom}
    margins_at = {0: (dom.m_north, dom.m_west)}

    def before_step(step, state):
        cur = dom_box["dom"]
        grown = None
        if expand:
            sides = {k: v and k not in blocked
                     for k, v in _active_sides(state, safety, n_steps - step).items()}
            if any(sides.values()):
                with tm.stage("domain_expansion"):
                    new_dom, new_state, rec, lim = _expand(cur, sides, state, grid_arrays, lc_arrays, ca,
                                                           land_provider, elevation_provider, allow_unverified_fuel,
                                                           lc_fuel is not None, step, step_min, tm, blocked,
                                                           len(expansions))
                limit_box["limit"] = limit_box["limit"] or lim
                if rec is not None:
                    expansions.append(rec)
                    dom_box["dom"] = cur = new_dom
                    grown = {"state": new_state, "dryness": grid_arrays["dryness"], "fuel": grid_arrays["fuel"],
                             "buildup": grid_arrays["buildup"], "elevation": grid_arrays["elev"],
                             "non_fuel": grid_arrays["non_fuel"]}
        margins_at[step] = (cur.m_north, cur.m_west)
        return grown

    t_sim = time.perf_counter()
    raw_history = sim.run(ignition_mask=ignition, dryness_grid=dryness, fuel_load_grid=fuel, non_fuel_mask=non_fuel,
                          wind_speed_ms=sched[0][0], wind_from_deg=sched[0][1],
                          horizon_minutes=n_steps * CA_STEP_MINUTES, elevation_grid=elevation,
                          fuel_buildup_grid=buildup, wind_schedule=sched, before_step=before_step)
    tm.add("ca_simulation", time.perf_counter() - t_sim - tm.seconds.get("domain_expansion", 0.0))
    dom, limit = dom_box["dom"], limit_box["limit"]
    frames = [(h.state, margins_at.get(h.step, margins_at[0])[0], margins_at.get(h.step, margins_at[0])[1], h.step)
              for h in raw_history]

    # ── history on the final grid (earlier frames padded) ───────────────── #
    rows, cols = dom.n_rows, dom.n_cols
    non_fuel = grid_arrays["non_fuel"]
    fuel, dryness, buildup, elevation = grid_arrays["fuel"], grid_arrays["dryness"], grid_arrays["buildup"], \
        grid_arrays["elev"]
    lc_classes, lc_conf, lc_src = lc_arrays["classes"], lc_arrays["conf"], lc_arrays["src"]
    from src.simulation.cellular_automata import SimulationStep
    history = []
    blank = np.where(non_fuel, CellState.NON_FUEL, CellState.UNBURNED).astype(np.int8)
    for st_, mn, mw, k in frames:
        if st_.shape != (rows, cols):
            full = blank.copy()
            r0, c0 = dom.m_north - mn, dom.m_west - mw
            full[r0:r0 + st_.shape[0], c0:c0 + st_.shape[1]] = st_
            st_ = full
        history.append(SimulationStep(step=k, minutes_elapsed=round(k * step_min, 3), state=st_,
                                      n_burning=int((st_ == CellState.BURNING).sum()),
                                      n_burned=int((st_ == CellState.BURNED).sum())))
    # ignition mask / requested on the final grid
    r0, c0 = dom.m_north - dom0.m_north, dom.m_west - dom0.m_west
    ign0 = np.zeros((rows, cols), dtype=bool)
    ign0[r0:r0 + dom0.n_rows, c0:c0 + dom0.n_cols] = ignition
    req0 = np.zeros((rows, cols), dtype=bool)
    req0[r0:r0 + dom0.n_rows, c0:c0 + dom0.n_cols] = requested
    ignition, requested = ign0, req0

    ign = np.full((rows, cols), -1, dtype=int)
    out = np.full((rows, cols), -1, dtype=int)
    for h in history:
        ign[(h.state == CellState.BURNING) & (ign < 0)] = h.step
        out[(h.state == CellState.BURNED) & (out < 0)] = h.step
    # Invariant of the fuel mask: a non-fuel cell never burns (checked on the real output).
    if ((ign >= 0) & non_fuel).any() or ((out >= 0) & non_fuel).any():
        raise RuntimeError("fuel-mask violation: a non-fuel cell reached BURNING/BURNED")

    with tm.stage("analytics"):
        result = _analyse(focus, dom, dom0, history, ign, out, ignition, requested, non_fuel, fuel, dryness, buildup,
                          elevation, terrain_source, sched, conditions, ca, sim, base, seed, placement, duration,
                          n_steps, step_min, lc_classes, lc_label, lc_conf, lc_src, lc_status, fuel_source,
                          expansions, limit, study_region)
    result.params["fuel_variability"] = var_spec[0] if var_spec else 0.0
    result.timings = tm.as_dict()
    return result


def _limits(limits: Optional[dict] = None) -> dict:
    """Configured extent limits, optionally narrowed by the user's maximum
    simulation extent (never above the DOMAIN safety limits)."""
    from config.config import DOMAIN
    L = {"max_side_cells": DOMAIN.max_side_cells, "max_total_cells": DOMAIN.max_total_cells,
         "max_extent_m": DOMAIN.max_extent_m, "max_expansions": DOMAIN.max_expansions}
    for k, v in (limits or {}).items():
        if k in L and v:
            L[k] = min(L[k], type(L[k])(v))
    return L


def _expand(dom: SimulationDomain, sides: Dict[str, bool], state: np.ndarray, g: dict, lc: dict, ca: dict,
            land_provider, elevation_provider, allow_unverified: bool, fused_fuel: bool, step: int, step_min: float,
            tm, blocked: set, n_done: int, limits: Optional[dict] = None):
    """Grow `dom` on the requested sides (one side at a time, north, south,
    west, east), padding every grid in place in `g` / `lc`. Returns (domain,
    state, expansion record or None, limit dict or None)."""
    from config.config import DOMAIN
    L = _limits(limits)
    cell = dom.cell_m
    old = dom.describe()
    grown = {"north": 0, "south": 0, "west": 0, "east": 0}
    terrain: Dict[str, str] = {}
    limit = None
    unavailable = []
    for side in ("north", "south", "west", "east"):
        if not sides[side]:
            continue
        vertical = side in ("north", "south")
        size = dom.n_rows if vertical else dom.n_cols
        other = dom.n_cols if vertical else dom.n_rows
        k = max(int(DOMAIN.expansion_chunk_cells), int(DOMAIN.expansion_chunk_frac * size))
        k = min(k, L["max_side_cells"] - size, int(L["max_extent_m"] // cell) - size,
                (L["max_total_cells"] - dom.n_rows * dom.n_cols) // max(1, other))
        if side in blocked:
            continue
        if n_done >= L["max_expansions"]:
            k = 0
        if k <= 0:
            blocked.add(side)
            limit = limit or {"title": EXTENT_LIMIT_TITLE, "reason": "configured maximum simulation extent / cell count",
                              "side": side, "step": step, "minutes": round(step * step_min, 2),
                              "max_side_cells": L["max_side_cells"], "max_total_cells": L["max_total_cells"],
                              "max_extent_m": L["max_extent_m"]}
            continue
        new = dom.grown(**{side: k})
        spec_new = _grid_of(new)
        if side == "north":
            sl = (slice(0, k), slice(0, new.n_cols))
        elif side == "south":
            sl = (slice(new.n_rows - k, new.n_rows), slice(0, new.n_cols))
        elif side == "west":
            sl = (slice(0, new.n_rows), slice(0, k))
        else:
            sl = (slice(0, new.n_rows), slice(new.n_cols - k, new.n_cols))
        strip = spec_new.sub(sl[0].start, sl[0].stop, sl[1].start, sl[1].stop)
        shape = (sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
        # land cover / fuel of the new strip
        s_classes = s_conf = s_src = None
        if lc["classes"] is not None:
            if land_provider is None:
                blocked.add(side)
                limit = limit or {"title": EXTENT_LIMIT_TITLE, "reason": "land-cover data for the expansion area is "
                                  "not available (no land-cover provider)", "side": side, "step": step,
                                  "minutes": round(step * step_min, 2)}
                continue
            with tm.stage("land_cover_expansion"):
                slc = land_provider(strip)
            if slc.classes.shape != shape:
                raise ValueError(f"land cover strip {slc.classes.shape} does not match the expansion strip {shape}")
            if getattr(slc, "status", "") == "unavailable":
                unavailable.append(side)
                blocked.add(side)
                limit = limit or {"title": EXTENT_LIMIT_TITLE, "reason": "required land-cover data unavailable "
                                  "beyond this edge (" + slc.label + ")", "side": side, "step": step,
                                  "minutes": round(step * step_min, 2)}
                continue
            s_classes = slc.classes
            s_nf = slc.non_burnable(allow_unverified) if hasattr(slc, "non_burnable") else slc.non_fuel
            s_fuel = (np.asarray(slc.fuel_load, dtype=float) if (fused_fuel and slc.fuel_load is not None)
                      else np.full(shape, ca["fuel"]))
            s_conf = slc.confidence if slc.confidence is not None else np.zeros(shape, np.int8)
            s_src = slc.source_mask if slc.source_mask is not None else np.zeros(shape, np.uint8)
        else:
            s_nf = np.full(shape, ca["non_fuel"], dtype=bool)          # no land-cover layer (production path)
            s_fuel = np.full(shape, ca["fuel"])
        if g.get("fuel_var") is not None and not (fused_fuel and lc["classes"] is not None):
            amp, dl_, dn_ = g["fuel_var"]
            s_fuel = np.clip(s_fuel * fuel_variability_field(*strip.cell_centres(), dl_, dn_, amp), 0.0, 1.0)
        s_elev = None
        if elevation_provider is not None:
            with tm.stage("terrain_expansion"):
                e_, e_src = elevation_provider(strip)
            if e_src == "dem" and np.shape(e_) == shape:
                s_elev = np.asarray(e_, dtype=float)
                terrain[side] = "dem"
        if s_elev is None:
            # no DEM for the strip: extend the edge values (no artificial cliff; labelled)
            e = g["elev"]
            edge = {"north": e[:1, :], "south": e[-1:, :], "west": e[:, :1], "east": e[:, -1:]}[side]
            s_elev = np.broadcast_to(edge, shape).astype(float).copy()
            terrain[side] = "edge-extended (no DEM for this strip)" if e.any() else "flat"

        def pad(arr, strip_vals):
            out = np.empty((new.n_rows, new.n_cols), dtype=arr.dtype)
            out[sl] = strip_vals
            rest = [slice(None), slice(None)]
            if side == "north":
                rest[0] = slice(k, None)
            elif side == "south":
                rest[0] = slice(0, new.n_rows - k)
            elif side == "west":
                rest[1] = slice(k, None)
            else:
                rest[1] = slice(0, new.n_cols - k)
            out[tuple(rest)] = arr
            return out
        g["dryness"] = pad(g["dryness"], np.full(shape, ca["dryness"]))
        g["buildup"] = pad(g["buildup"], np.full(shape, ca["buildup"]))
        g["fuel"] = pad(g["fuel"], s_fuel)
        g["elev"] = pad(g["elev"], s_elev)
        g["non_fuel"] = pad(g["non_fuel"], s_nf)
        state = pad(state, np.where(s_nf, CellState.NON_FUEL, CellState.UNBURNED).astype(np.int8))
        if lc["classes"] is not None:
            lc["classes"] = pad(lc["classes"], s_classes)
            if lc["conf"] is not None:
                lc["conf"] = pad(lc["conf"], s_conf)
            if lc["src"] is not None:
                lc["src"] = pad(lc["src"], s_src)
        dom = new
        grown[side] = k
    if not any(grown.values()):
        return dom, state, None, limit
    rec = {"step": step, "minutes": round(step * step_min, 2), "grown_cells": grown, "old": old,
           "new": dom.describe(), "unavailable_sides": unavailable, "terrain": terrain, "index": n_done + 1}
    return dom, state, rec, limit


def lc_fuel_present(fuel_source: str) -> bool:
    return fuel_source.startswith("land cover")


def _analyse(focus, dom, dom0, history, ign, out, ignition, requested, non_fuel, fuel, dryness, buildup, elevation,
             terrain_source, sched, conditions, ca, sim, base, seed, placement, duration, n_steps, step_min,
             lc_classes, lc_label, lc_conf, lc_src, lc_status, fuel_source, expansions, limit, study_region,
             intensity_override=None, potential_override=None, class_fn=None, kw=None, extra_params=None):
    rows, cols = dom.n_rows, dom.n_cols
    if intensity_override is not None:
        intensity, potential = intensity_override, float(potential_override or 0.0)
    else:
        intensity, potential = _legacy_intensity(ign, fuel, dryness, buildup, non_fuel, sched, ca, sim, base, rows,
                                                 cols)
    class_fn = class_fn or intensity_class
    return _analyse_rest(focus, dom, dom0, history, ign, out, ignition, requested, non_fuel, fuel, elevation,
                         terrain_source, sched, conditions, ca, sim, base, seed, placement, duration, n_steps,
                         step_min, lc_classes, lc_label, lc_conf, lc_src, lc_status, fuel_source, expansions, limit,
                         study_region, intensity, potential, class_fn, kw, extra_params)


def _legacy_intensity(ign, fuel, dryness, buildup, non_fuel, sched, ca, sim, base, rows, cols):
    # Visual fire intensity (0-1): the CA's own spread potential for the cell
    # (dryness x fuel x build-up x wind speed, no direction) scaled by how the
    # fire reached it - head fire (driven downwind) burns hotter than flank or
    # backing fire. Derived from the CA state, used only to scale the VFX.
    boost = 1.0 + min(sched[0][0] / 10.0, 1.0)
    pot = np.clip(base * dryness * fuel * buildup * boost, 0.0, 0.97)
    potential = float(sim._cell_spread_prob(ca["dryness"], float(np.mean(fuel[~non_fuel])) if (~non_fuel).any()
                                            else ca["fuel"], ca["buildup"], sched[0][0], 1.0, 1.0))
    pot_norm = np.minimum(1.0, pot / (0.5 * base)) if base > 0 else np.zeros_like(pot)
    # How the fire reached each cell: the best wind alignment over the neighbours
    # that were burning one step earlier (a cell burns for exactly one step, so
    # "burning at s-1" is "ignited at s-1"). Vectorised over the grid.
    offsets = sim._neighbor_offsets
    last = int(ign.max()) if (ign >= 0).any() else 0
    wf = np.array([[sim.direction_factor(b, sched[min(k, len(sched) - 1)][1], sched[min(k, len(sched) - 1)][0])
                    for _, _, b in offsets] for k in range(max(last, 1))])   # wf[s-1, k]
    pad = np.pad(ign, 1, constant_values=-5)
    align = np.where(ign == 0, 1.0, 0.4)
    later = ign > 0
    prev_step = np.where(later, ign - 1, 0)
    for k, (dr, dc, _) in enumerate(offsets):
        nb = pad[1 - dr:1 - dr + rows, 1 - dc:1 - dc + cols]            # value at (r - dr, c - dc)
        hit = later & (nb == ign - 1)
        align = np.where(hit, np.maximum(align, wf[prev_step, k]), align)
    intensity = np.where(ign >= 0, np.clip((align / 1.8) * (0.45 + 0.55 * pot_norm), 0.12, 1.0), 0.0)
    return intensity, potential


def _analyse_rest(focus, dom, dom0, history, ign, out, ignition, requested, non_fuel, fuel, elevation,
                  terrain_source, sched, conditions, ca, sim, base, seed, placement, duration, n_steps, step_min,
                  lc_classes, lc_label, lc_conf, lc_src, lc_status, fuel_source, expansions, limit, study_region,
                  intensity, potential, class_fn, kw, extra_params):
    rows, cols = dom.n_rows, dom.n_cols
    # Study region (Bandipur): cell labels, boundary crossing, split statistics.
    inside = None
    region_info = None
    if study_region is not None:
        inside = study_region.cell_mask(dom)
        burned_any = ign >= 0
        outside_burn = burned_any & ~inside
        cross = int(ign[outside_burn].min()) if outside_burn.any() else None
        start_out = bool((ignition & ~inside).any())
        region_info = {**study_region.describe(),
                       "ignition_outside_region": start_out,
                       "crossed": cross is not None and not start_out,
                       "crossing_step": cross, "crossing_minutes": None if cross is None else round(cross * step_min, 2),
                       "domain_cells_outside": int((~inside).sum())}
        if cross is not None:
            rr, cc = np.argwhere(outside_burn & (ign == cross))[0]
            lat_c, lon_c = dom.cell_centres()
            region_info["crossing_lat"] = round(float(lat_c[rr, cc]), 6)
            region_info["crossing_lon"] = round(float(lon_c[rr, cc]), 6)

    # Per-step analytics, all read from the CA state. Every frame's burning / burned
    # cells lie inside the bounding box of the cells that ever ignite, so the
    # arrays are sliced to that box (same values, much less work on a big domain).
    focus_mask = dom.focus_mask()
    lat_c, lon_c = dom.cell_centres()
    centre_r, centre_c = np.argwhere(ignition).mean(axis=0) if ignition.any() else ((rows - 1) / 2, (cols - 1) / 2)
    ever = ign >= 0
    if ever.any():
        rs_, cs_ = np.nonzero(ever)
        r0, r1, c0, c1 = int(rs_.min()), int(rs_.max()) + 1, int(cs_.min()), int(cs_.max()) + 1
    else:
        r0, r1, c0, c1 = 0, 1, 0, 1
    W = (slice(r0, r1), slice(c0, c1))
    rr, cc = np.mgrid[r0:r1, c0:c1]
    dist_m = np.hypot(rr - centre_r, cc - centre_c) * focus.cell_m
    fm, la_w, lo_w, it_w = focus_mask[W], lat_c[W], lon_c[W], intensity[W]
    ins_w = inside[W] if inside is not None else None
    cell_area_ha = focus.cell_m ** 2 / 1e4
    metrics, prev_d = [], 0.0
    # rate of spread over a ~5-minute window (one frame can be shorter than the time the front needs to
    # cross a cell, which would show 0 / spikes)
    ros_win = max(1, int(round(5.0 / step_min))) if step_min > 0 else 1
    dists: List[float] = []
    for h in history:
        st_w = h.state[W]
        burning = st_w == CellState.BURNING
        burned = st_w == CellState.BURNED
        affected = burning | burned
        any_aff = bool(affected.any())
        d = float(dist_m[affected].max()) if any_aff else 0.0
        outside = bool((affected & ~fm).any())
        edge = bool((r0 == 0 and affected[0, :].any()) or (r1 == rows and affected[-1, :].any())
                    or (c0 == 0 and affected[:, 0].any()) or (c1 == cols and affected[:, -1].any()))
        mean_int = float(it_w[burning].mean()) if burning.any() else 0.0
        max_int = float(it_w[burning].max()) if burning.any() else 0.0
        w = sched[min(max(h.step - 1, 0), len(sched) - 1)]
        m = {
            "step": h.step, "minutes": h.minutes_elapsed,
            "burning": int(burning.sum()), "burned": int(burned.sum()),
            "burned_ha": round(float(burned.sum()) * cell_area_ha, 4),
            "fire_area_ha": round(float(affected.sum()) * cell_area_ha, 4),
            "burned_in_focus": int((burned & fm).sum()),
            "perimeter_m": _perimeter_m(affected, focus.cell_m),
            "front_distance_m": round(d, 1),
            "ros_m_per_min": (round((d - dists[max(0, len(dists) - ros_win)]) / (min(ros_win, len(dists)) * step_min), 2)
                              if h.step and dists else 0.0),
            "mean_intensity": round(mean_int, 3), "max_intensity": round(max_int, 3),
            "intensity_class": class_fn(mean_int) if mean_int else "-",
            "centroid_lat": round(float(la_w[affected].mean()), 6) if any_aff else None,
            "centroid_lon": round(float(lo_w[affected].mean()), 6) if any_aff else None,
            "left_focus": outside, "boundary_reached": outside, "domain_edge_reached": edge,
            "wind_speed_ms": w[0], "wind_from_deg": w[1],
        }
        if kw is not None:
            kw_w = kw[W]
            m["max_intensity_kw_m"] = round(float(kw_w[burning].max()), 1) if burning.any() else 0.0
            m["mean_intensity_kw_m"] = round(float(kw_w[burning].mean()), 1) if burning.any() else 0.0
        if ins_w is not None:
            m["burned_ha_bandipur"] = round(float((burned & ins_w).sum()) * cell_area_ha, 4)
            m["burned_ha_regional_extension"] = round(float((burned & ~ins_w).sum()) * cell_area_ha, 4)
            m["fire_area_ha_bandipur"] = round(float((affected & ins_w).sum()) * cell_area_ha, 4)
            m["fire_area_ha_regional_extension"] = round(float((affected & ~ins_w).sum()) * cell_area_ha, 4)
        metrics.append(m)
        prev_d = d
        dists.append(d)
    if metrics and metrics[-1]["domain_edge_reached"] and limit is None:
        logger.warning("Fire reached the simulation-domain edge without a recorded limit")

    params = {"n_ignition": int(ignition.sum()), "placement": placement, "seed": int(seed),
              "duration_minutes": duration, "horizon_minutes": duration, "n_steps_planned": n_steps,
              "spread_potential": round(potential, 4), "base_spread_prob": base,
              "cell_m": focus.cell_m, "size_m": focus.size_m, "width_m": focus.n_cols * focus.cell_m,
              "height_m": focus.n_rows * focus.cell_m, "domain_margin": dom.margin,
              "domain_margins_nswe": list(dom.margin_tuple),
              "initial_domain": dom0.describe(), "final_domain": dom.describe(),
              "n_expansions": len(expansions), "extent_limit_reached": limit is not None,
              "ignition_cells_requested": int(requested.sum()),
              "ignition_cells_on_non_fuel": int((requested & non_fuel).sum()),
              "non_fuel_cells": int(non_fuel.sum()), "land_cover": lc_label, "land_cover_status": lc_status,
              "fuel_load_source": fuel_source + (" (initial domain; expansion strips use Sentinel-2 NDVI only "
                                                 "where already cached, otherwise class defaults)"
                                                 if expansions and lc_fuel_present(fuel_source) else ""),
              "fuel_load_mean_burnable": round(float(fuel[~non_fuel].mean()), 4) if (~non_fuel).any() else 0.0,
              **ca}
    if sim is not None:
        params.update({"model": "legacy-ca", "max_spread_m_per_min": round(focus.cell_m / step_min, 3),
                       "diagonal_correction": (sim.diagonal_factor if sim.diagonal_correction else False),
                       "wind_model": sim.wind_model})
    params.update(extra_params or {})
    return LocalSpreadResult(focus=focus, domain=dom, step_minutes=step_min, duration_minutes=duration,
                             history=history, ignition_step=ign, burnout_step=out, intensity=intensity,
                             non_fuel=non_fuel, elevation=elevation, terrain_source=terrain_source,
                             wind_schedule=sched, conditions=conditions, params=params, metrics=metrics,
                             land_cover=lc_classes, land_cover_label=lc_label, initial_domain=dom0,
                             expansions=expansions, extent_limit=limit, fuel_load=fuel, land_confidence=lc_conf,
                             land_source_mask=lc_src, land_status=lc_status, study_region=region_info,
                             inside_region=inside)


# ── Phase 3: rate-of-spread CA (default local model) ─────────────────────── #

def frame_minutes_for(duration_minutes: float) -> float:
    """Simulated minutes between the frames handed to the dashboard: about 60
    frames per run, never coarser than 5 min or finer than 15 s; the run's
    duration is always an exact number of frames."""
    d = float(duration_minutes)
    target = min(5.0, max(0.25, d / 60.0))
    n = max(1, int(round(d / target)))
    return d / n


def _lattice_index(dom: SimulationDomain) -> Tuple[np.ndarray, np.ndarray]:
    lat_c, lon_c = dom.cell_centres()
    return (np.round(lat_c / dom.focus.dlat).astype(np.int64), np.round(lon_c / dom.focus.dlon).astype(np.int64))


def _ros_grids(dom: SimulationDomain, fuel: np.ndarray, elevation: np.ndarray, non_fuel: np.ndarray, seed: int,
               base_fuel: Optional[float] = None):
    from src.simulation.fire_behaviour import fuel_continuity
    if base_fuel is not None:
        # the simulated fuel variability acts at half strength on the spread RATE: full-strength noise
        # channels a strong wind-driven head into fingers / forks that real fronts do not show
        fuel = base_fuel + SYSTEM.local_ros_fuel_noise_weight * (np.asarray(fuel, dtype=float) - base_fuel)
    from src.simulation.ros_ca import RosGrids, cell_hash01
    jit = float(SYSTEM.local_ros_threshold_jitter)
    ra, ca_ = _lattice_index(dom)
    thr = 1.0 + jit * (2.0 * cell_hash01(ra, ca_, seed) - 1.0)
    g = RosGrids(fuel_continuity(fuel), np.asarray(elevation, dtype=float), np.asarray(non_fuel, dtype=bool),
                 thr, dom.cell_m)
    g.flame_mult = 0.75 + 0.5 * cell_hash01(ra + 7919, ca_ - 104729, seed + 1)
    return g.build()


def _run_ros(focus, dom, conditions, ca, dryness, buildup, fuel, non_fuel, elevation, terrain_source, wind_speed_ms,
             wind_from_deg, wind_schedule_15min, ignition, requested, duration, seed, placement, expand, land_provider,
             elevation_provider, allow_unverified_fuel, study_region, tm, fuel_source, var_spec, lc_classes, lc_label,
             lc_conf, lc_src, lc_status, lc_fuel, limits):
    from config.config import DOMAIN
    from src.simulation import fire_behaviour as fb
    from src.simulation.ros_ca import MAX_REACH, OFFSETS, run_ros_ca
    fuel_model = fb.FuelModel(w_deciduous=SYSTEM.local_fbp_w_deciduous, w_grass=1.0 - SYSTEM.local_fbp_w_deciduous,
                              grass_curing_pct=SYSTEM.local_fbp_grass_curing_pct,
                              max_length_to_breadth=SYSTEM.local_ros_max_length_to_breadth,
                              min_back_fraction=SYSTEM.local_ros_min_back_fraction)
    ffmc, bui = conditions.get("ffmc", float("nan")), conditions.get("bui", float("nan"))
    sched15 = ([tuple(map(float, w)) for w in wind_schedule_15min] if wind_schedule_15min
               else [(wind_speed_ms, wind_from_deg)])
    rate_cache: Dict[Tuple[float, float], object] = {}

    def rates_at(t):
        w = sched15[min(int(t // CA_STEP_MINUTES), len(sched15) - 1)]
        if w not in rate_cache:
            rate_cache[w] = fb.spread_rates(ffmc, bui, w[0], w[1] % 360, fuel_model)
        return rate_cache[w]
    head_max = max(fb.spread_rates(ffmc, bui, w[0], w[1] % 360, fuel_model).head for w in sched15)
    frame_min = frame_minutes_for(duration)
    n_frames = int(round(duration / frame_min))

    state0 = np.where(ignition, CellState.BURNING, CellState.UNBURNED).astype(np.int8)
    state0 = np.where(non_fuel, CellState.NON_FUEL, state0).astype(np.int8)
    grid_arrays = {"dryness": dryness, "fuel": fuel, "buildup": buildup, "elev": elevation, "non_fuel": non_fuel,
                   "fuel_var": var_spec}
    lc_arrays = {"classes": lc_classes, "conf": lc_conf, "src": lc_src}
    with tm.stage("ca_setup"):
        base_f = ca["fuel"] if (var_spec and lc_fuel is None) else None
        grids = _ros_grids(dom, fuel, elevation, non_fuel, seed, base_f)
    expansions: List[dict] = []
    limit_box: Dict[str, Optional[dict]] = {"limit": None}
    blocked: set = set()
    safety = max(int(DOMAIN.safety_margin_cells), MAX_REACH + 3)
    box = {"dom": dom}
    dom_init = dom

    def margins():
        return box["dom"].m_north, box["dom"].m_west

    def expand_hook(state, arrays, t_now):
        if not expand:
            return None
        cur = box["dom"]
        br, bc = arrays["burning"]
        if br.size == 0:
            return None
        left_cells = int(math.ceil(max(0.0, duration - t_now) * head_max * 3.0 / cur.cell_m)) + 2
        rows_, cols_ = state.shape
        dist = {"north": int(br.min()), "south": rows_ - 1 - int(br.max()),
                "west": int(bc.min()), "east": cols_ - 1 - int(bc.max())}
        sides = {k: v < safety and v <= left_cells + 1 and k not in blocked for k, v in dist.items()}
        if not any(sides.values()):
            return None
        step_f = t_now / frame_min
        with tm.stage("domain_expansion"):
            new_dom, new_state, rec, lim = _expand(cur, sides, state, grid_arrays, lc_arrays, ca, land_provider,
                                                   elevation_provider, allow_unverified_fuel, lc_fuel is not None,
                                                   step_f, frame_min, tm, blocked, len(expansions), limits)
        limit_box["limit"] = limit_box["limit"] or lim
        if rec is None:
            return None
        expansions.append(rec)
        pad = (new_dom.m_north - cur.m_north, new_dom.m_south - cur.m_south,
               new_dom.m_west - cur.m_west, new_dom.m_east - cur.m_east)
        box["dom"] = new_dom
        g = _ros_grids(new_dom, grid_arrays["fuel"], grid_arrays["elev"], grid_arrays["non_fuel"], seed, base_f)
        return new_state, g, pad

    t_sim = time.perf_counter()
    run = run_ros_ca(state0, grids, rates_at, duration, frame_min, flame_min=SYSTEM.local_ros_flame_min,
                     residence_max_min=SYSTEM.local_ros_residence_max_min, expand=expand_hook, margins=margins,
                     rate_bound=head_max, max_substeps=SYSTEM.local_ros_max_substeps)
    tm.add("ca_simulation", time.perf_counter() - t_sim - tm.seconds.get("domain_expansion", 0.0))
    dom, limit = box["dom"], limit_box["limit"]
    rows, cols = dom.n_rows, dom.n_cols
    non_fuel = grid_arrays["non_fuel"]
    fuel, dryness, buildup, elevation = grid_arrays["fuel"], grid_arrays["dryness"], grid_arrays["buildup"], \
        grid_arrays["elev"]
    lc_classes, lc_conf, lc_src = lc_arrays["classes"], lc_arrays["conf"], lc_arrays["src"]

    from src.simulation.cellular_automata import SimulationStep
    blank = np.where(non_fuel, CellState.NON_FUEL, CellState.UNBURNED).astype(np.int8)
    history = []
    for k, (st_, (mn, mw)) in enumerate(zip(run.frames, run.frame_margins)):
        if st_.shape != (rows, cols):
            full = blank.copy()
            r0, c0 = dom.m_north - mn, dom.m_west - mw
            full[r0:r0 + st_.shape[0], c0:c0 + st_.shape[1]] = st_
            st_ = full
        history.append(SimulationStep(step=k, minutes_elapsed=round(k * frame_min, 4), state=st_,
                                      n_burning=int((st_ == CellState.BURNING).sum()),
                                      n_burned=int((st_ == CellState.BURNED).sum())))
    dom0 = dom_init
    r0, c0 = dom.m_north - dom0.m_north, dom.m_west - dom0.m_west
    ign0 = np.zeros((rows, cols), dtype=bool)
    ign0[r0:r0 + dom0.n_rows, c0:c0 + dom0.n_cols] = ignition
    req0 = np.zeros((rows, cols), dtype=bool)
    req0[r0:r0 + dom0.n_rows, c0:c0 + dom0.n_cols] = requested
    T, TO = run.t_ign, run.t_out
    ever = np.isfinite(T)
    ign = np.where(ever, np.ceil(np.round(T / frame_min, 9)), -1).astype(int)
    out = np.where(np.isfinite(TO), np.ceil(np.round(TO / frame_min, 9)), -1).astype(int)
    out = np.where(ever & (out >= 0), np.maximum(out, ign + 1), out)
    if ((ign >= 0) & non_fuel).any():
        raise RuntimeError("fuel-mask violation: a non-fuel cell reached BURNING/BURNED")
    # fireline intensity from the spread that reached each cell (ignition cells: flank rate)
    r_last = run.rates[-1] if run.rates else rates_at(0.0)
    ros_c = np.where(ever, np.where(run.ros_ign > 0, run.ros_ign, r_last.flank), 0.0)
    kw = fb.byram_intensity_kw_m(ros_c, SYSTEM.local_ros_fuel_consumed_kg_m2) * np.clip(
        fb.fuel_continuity(fuel), 0.0, 1.25)
    # visual intensity: absolute fireline intensity, weighted by the cell's share of the head-fire rate so the
    # head front is the brightest part and flanks / backing fire are visibly weaker
    rel = np.sqrt(np.clip(ros_c / max(head_max, 1e-6), 0.0, 1.0))
    vis = np.where(ever, np.clip(fb.visual_intensity(kw) * (0.35 + 0.65 * rel), 0.12, 1.0), 0.0)
    thresholds = [float(fb.visual_intensity(k)) for k, _ in fb.INTENSITY_KW_CLASSES[:3]]

    def class_fn(v):
        for th, (_, name) in zip(thresholds, fb.INTENSITY_KW_CLASSES):
            if v < th:
                return name
        return "EXTREME"
    r0_ = rates_at(0.0)
    sched_frames = [(float(w[0]), float(w[1]) % 360)
                    for w in (sched15[min(int(k * frame_min // CA_STEP_MINUTES), len(sched15) - 1)]
                              for k in range(max(1, n_frames)))]
    extra = {"model": "ros-ca", "fuel_model": fuel_model.label, **fb.describe(r0_),
             "frame_minutes": round(frame_min, 4), "substep_minutes": round(run.dt_min, 5),
             "n_substeps": run.n_substeps,
             "extinguished_at_min": None if run.extinguished_at is None else round(run.extinguished_at, 2),
             "max_spread_m_per_min": round(head_max, 3), "neighbourhood": len(OFFSETS),
             "flame_min": SYSTEM.local_ros_flame_min, "residence_max_min": SYSTEM.local_ros_residence_max_min,
             "threshold_jitter": SYSTEM.local_ros_threshold_jitter, "diagonal_correction": "geometric (true distance)",
             "wind_model": "FBP ellipse (head / flank / back)"}
    with tm.stage("analytics"):
        result = _analyse(focus, dom, dom0, history, ign, out, ign0, req0, non_fuel, fuel, dryness, buildup,
                          elevation, terrain_source, sched_frames, conditions, ca, None, 0.0, seed, placement,
                          duration, n_frames, frame_min, lc_classes, lc_label, lc_conf, lc_src, lc_status,
                          fuel_source, expansions, limit, study_region, intensity_override=vis,
                          potential_override=r0_.head, class_fn=class_fn, kw=kw, extra_params=extra)
    result.params["fuel_variability"] = var_spec[0] if var_spec else 0.0
    result.ignition_time = np.where(ever, T, -1.0)
    result.burnout_time = np.where(np.isfinite(TO), TO, -1.0)
    result.fireline_kw = kw
    result.timings = tm.as_dict()
    return result


def _initial_of(dom: SimulationDomain, expansions: List[dict]) -> SimulationDomain:
    """The domain before the recorded expansions (same lattice)."""
    g = {"north": 0, "south": 0, "west": 0, "east": 0}
    for e in expansions:
        for k, v in e["grown_cells"].items():
            g[k] += v
    mn, ms, mw, me = dom.margin_tuple
    return SimulationDomain(dom.focus, dom.margin, (mn - g["north"], ms - g["south"], mw - g["west"], me - g["east"]))
