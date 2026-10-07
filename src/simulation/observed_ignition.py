"""
observed_ignition.py - NASA FIRMS detections -> OBSERVED ignition cells.

The only way a LIVE simulation gets fire: a real satellite detection inside the
selected simulation area, on a cell that can burn. Temperature, wind, FWI or the
model risk never create an ignition.

  * Each detection is mapped to the CA grid cell that contains it (the nearest
    cell centre). It is never moved to another cell: relocating an observation
    would invent a fire location.
  * Detections outside the focus area (the user's box) are context only.
  * Detections on non-fuel cells (OpenStreetMap water, roads, buildings, bare
    ground - the same mask the CA uses - or a zone whose vegetation index is
    below the CA's fuel threshold) are skipped and reported.
  * Several detections may ignite several cells; detections that fall in the
    same cell ignite it once.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

from src.simulation.fuel_map import CLASS_NAMES, FUEL

IGNITION = "ignition"
OUTSIDE = "outside_area"
NON_FUEL = "non_fuel"


def classify_detections(domain, detections: Iterable[dict], land_cover=None,
                        zone_non_fuel: bool = False) -> dict:
    """Classify real detections ({lat, lon, ...}) against a SimulationDomain.

    Returns {"detections": [... each with status / reason / cell],
             "n_total", "n_in_area", "n_valid", "n_non_fuel", "n_outside",
             "valid_points": [[lat, lon], ...], "n_cells": unique ignition cells,
             "land_cover": label of the mask used}."""
    focus = domain.focus_mask()
    classes = None
    if land_cover is not None and getattr(land_cover, "classes", None) is not None \
            and land_cover.classes.shape == focus.shape:
        classes = land_cover.classes
    out: List[dict] = []
    cells = set()
    for d in detections or []:
        lat, lon = float(d["lat"]), float(d["lon"])
        rc = domain.cell_of(lat, lon)
        rec = {**d, "lat": lat, "lon": lon, "cell": list(rc) if rc else None}
        if rc is None or not focus[rc]:
            rec.update(status=OUTSIDE, reason="outside the selected simulation area")
        elif zone_non_fuel:
            rec.update(status=NON_FUEL, reason="zone vegetation index below the fuel threshold (NDVI < 0.15)")
        elif classes is not None and int(classes[rc]) != FUEL:
            rec.update(status=NON_FUEL, reason=f"{CLASS_NAMES.get(int(classes[rc]), 'non-fuel')} cell (OpenStreetMap)")
        else:
            rec.update(status=IGNITION, reason="valid fuel cell")
            cells.add(tuple(rc))
        out.append(rec)
    valid = [r for r in out if r["status"] == IGNITION]
    return {"detections": out, "n_total": len(out),
            "n_in_area": sum(r["status"] != OUTSIDE for r in out),
            "n_valid": len(valid), "n_non_fuel": sum(r["status"] == NON_FUEL for r in out),
            "n_outside": sum(r["status"] == OUTSIDE for r in out),
            "valid_points": [[r["lat"], r["lon"]] for r in valid], "n_cells": len(cells),
            "land_cover": getattr(land_cover, "label", None) if land_cover is not None else None}


def observed_mask(domain, valid_points) -> np.ndarray:
    """Boolean ignition mask from classified (valid) observation points only -
    no fallback placement: no valid point, no ignition."""
    mask = np.zeros((domain.n_rows, domain.n_cols), dtype=bool)
    for lat, lon in valid_points or []:
        rc = domain.cell_of(float(lat), float(lon))
        if rc is not None:
            mask[rc] = True
    return mask


def empty_classification(n_total: int = 0, label: Optional[str] = None) -> dict:
    return {"detections": [], "n_total": n_total, "n_in_area": 0, "n_valid": 0, "n_non_fuel": 0,
            "n_outside": n_total, "valid_points": [], "n_cells": 0, "land_cover": label}
