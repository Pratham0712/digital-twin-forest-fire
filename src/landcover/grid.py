"""
grid.py - exact geometry of a CA grid (row 0 = north, column 0 = west).

GridSpec is the land-cover side's view of a SimulationDomain (or of a strip
added when the domain expands): the same bounds, the same cell lattice, so a
land-cover array computed for a GridSpec lines up 1:1 with the CA cells.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    north: float
    south: float
    west: float
    east: float
    n_rows: int
    n_cols: int
    cell_m: float

    @classmethod
    def from_domain(cls, domain) -> "GridSpec":
        b = domain.bounds
        return cls(b["north"], b["south"], b["west"], b["east"], int(domain.n_rows), int(domain.n_cols),
                   float(domain.cell_m))

    @property
    def bounds(self) -> Dict[str, float]:
        return {"north": self.north, "south": self.south, "west": self.west, "east": self.east}

    @property
    def shape(self) -> Tuple[int, int]:
        return self.n_rows, self.n_cols

    @property
    def dlat(self) -> float:
        return (self.north - self.south) / self.n_rows

    @property
    def dlon(self) -> float:
        return (self.east - self.west) / self.n_cols

    def cell_centres(self) -> Tuple[np.ndarray, np.ndarray]:
        lat = self.north - (np.arange(self.n_rows) + 0.5) * self.dlat
        lon = self.west + (np.arange(self.n_cols) + 0.5) * self.dlon
        return (np.repeat(lat[:, None], self.n_cols, axis=1), np.repeat(lon[None, :], self.n_rows, axis=0))

    def sub_axes(self, s: int) -> Tuple[np.ndarray, np.ndarray]:
        """1-D latitudes (n_rows*s, north->south) and longitudes (n_cols*s,
        west->east) of an s x s supersample of every cell."""
        lat = self.north - (np.arange(self.n_rows * s) + 0.5) * (self.dlat / s)
        lon = self.west + (np.arange(self.n_cols * s) + 0.5) * (self.dlon / s)
        return lat, lon

    def sub(self, r0: int, r1: int, c0: int, c1: int) -> "GridSpec":
        """The cells [r0:r1, c0:c1] (may lie outside 0..n, for strips added
        when a domain expands) as a GridSpec on the same lattice."""
        return GridSpec(self.north - r0 * self.dlat, self.north - r1 * self.dlat,
                        self.west + c0 * self.dlon, self.west + c1 * self.dlon, r1 - r0, c1 - c0, self.cell_m)


def supersample_factor(cell_m: float, cfg) -> int:
    """Supersample points per cell side: about one per cfg.supersample_m."""
    return int(np.clip(math.ceil(float(cell_m) / cfg.supersample_m), cfg.supersample_min, cfg.supersample_max))
