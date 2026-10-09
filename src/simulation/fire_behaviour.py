"""
fire_behaviour.py - surface fire rate of spread (ROS) for the LOCAL simulation.

Why this exists (Phase 3 audit)
-------------------------------
The original local CA turned the weather into a spread PROBABILITY:
    p = base x FFMC/101 x NDVI x BUI/30 x wind_factor x (1 + min(U/10, 1)) x slope, capped at 0.97
with exactly one trial per neighbour and a burn that lasts one step. Measured
consequences (see the Phase 3 report):
  * FFMC/101 is almost flat over the realistic range (FFMC 80 -> 96 changes it
    from 0.79 to 0.95), so temperature / humidity barely changed the fire;
  * the wind-speed boost saturated at 10 m/s, and the fire could never move
    faster than one cell per step (6.7 m/min at 25 m cells), whatever the
    weather;
  * a cell got one chance from each neighbour, so weak flanks died out and
    fires often stopped after a few steps.

This module replaces that probability by a physical spread RATE, using the
published equations of the Canadian Forest Fire Behavior Prediction (FBP)
System (Forestry Canada Fire Danger Group 1992, ST-X-3; Wotton, Alexander &
Taylor 2009, GLC-X-10) - the companion of the Canadian FWI System this project
already uses for FFMC / BUI / FWI:

  fine-fuel moisture   m    = 147.2 (101 - FFMC) / (59.5 + FFMC)
  moisture function    f(F) = 91.9 exp(-0.1386 m) (1 + m^5.31 / 4.93e7)
  wind function        f(W) = exp(0.05039 W)               W <= 40 km/h
                              12 (1 - exp(-0.0818 (W - 28))) W  > 40 km/h
  Initial Spread Index ISI  = 0.208 f(W) f(F)
  rate of spread       ROS  = a (1 - exp(-b ISI))^c  x BE(BUI)  [m/min]
  back ROS             from ISI with f(W) = exp(-0.05039 W)
  length : breadth     LB   = 1 + 8.729 (1 - exp(-0.030 W))^2.155
  flank ROS            FROS = (ROS + BROS) / (2 LB)

Fuel type: Bandipur is mostly dry deciduous forest with a grass / litter
understorey. The FBP System has no Indian fuel types, so the head rate is a
BLEND of FBP D-1 (leafless deciduous) and O-1a (grass, partly cured). The
blend weights, curing and the NDVI fuel-continuity factor are ASSUMPTIONS,
not calibrated against observed Bandipur fires (configurable in config.py).
The result is a plausible, physically consistent response to wind, fuel
moisture (temperature / humidity through FFMC) and build-up (BUI) - not a
validated operational forecast.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

# FBP fuel-type constants (ST-X-3, table 6): a, b, c [, q, BUI0]
FBP_D1 = (30.0, 0.0232, 1.6, 0.9, 32.0)
FBP_O1A = (190.0, 0.0310, 1.4)


def fine_fuel_moisture(ffmc: float) -> float:
    ffmc = float(np.clip(ffmc, 0.0, 101.0))
    return 147.2 * (101.0 - ffmc) / (59.5 + ffmc)


def moisture_function(ffmc: float) -> float:
    m = fine_fuel_moisture(ffmc)
    return 91.9 * math.exp(-0.1386 * m) * (1.0 + m ** 5.31 / 4.93e7)


def wind_function(wind_kmh: float) -> float:
    w = max(0.0, float(wind_kmh))
    return math.exp(0.05039 * w) if w <= 40.0 else 12.0 * (1.0 - math.exp(-0.0818 * (w - 28.0)))


def initial_spread_index(ffmc: float, wind_kmh: float) -> float:
    return 0.208 * wind_function(wind_kmh) * moisture_function(ffmc)


def buildup_effect(bui: float, q: float, bui0: float) -> float:
    if not np.isfinite(bui) or bui <= 0:
        return 1.0
    return math.exp(50.0 * math.log(q) * (1.0 / bui - 1.0 / bui0))


def curing_factor(curing_pct: float) -> float:
    c = float(curing_pct)
    return 0.005 * (math.exp(0.061 * c) - 1.0) if c < 58.8 else 0.176 + 0.02 * (c - 58.8)


def length_to_breadth(wind_kmh: float) -> float:
    w = max(0.0, float(wind_kmh))
    return 1.0 + 8.729 * (1.0 - math.exp(-0.030 * w)) ** 2.155


@dataclass(frozen=True)
class FuelModel:
    """Head-rate blend (see module docstring). ASSUMED, uncalibrated."""
    w_deciduous: float = 0.7          # FBP D-1 share
    w_grass: float = 0.3              # FBP O-1a share
    grass_curing_pct: float = 80.0    # dry-season curing of the grass understorey
    max_length_to_breadth: float = 5.0  # a 16-neighbour grid cannot draw narrower ellipses
    min_back_fraction: float = 0.0      # floor of backing ROS as a share of the head ROS (0 = pure FBP)
    label: str = "FBP D-1 / O-1a blend (assumed for dry deciduous forest; uncalibrated)"

    def ros(self, isi: float, bui: float) -> float:
        a, b, c, q, bui0 = FBP_D1
        d1 = a * (1.0 - math.exp(-b * isi)) ** c * buildup_effect(bui, q, bui0)
        a2, b2, c2 = FBP_O1A
        o1 = a2 * (1.0 - math.exp(-b2 * isi)) ** c2 * curing_factor(self.grass_curing_pct)
        return self.w_deciduous * d1 + self.w_grass * o1


def fuel_continuity(fuel_load) -> np.ndarray:
    """Relative spread multiplier from the CA's 0-1 fuel load (NDVI proxy):
    0.15 (the existing non-fuel threshold) -> ~0, 0.55 (closed forest) -> 1."""
    return np.clip((np.asarray(fuel_load, dtype=float) - 0.15) / 0.40, 0.05, 1.25)


@dataclass(frozen=True)
class SpreadRates:
    """Rates for one wind / fuel-moisture state (open, flat ground, fuel continuity 1)."""
    head: float            # m/min
    back: float
    flank: float
    length_to_breadth: float
    isi: float
    wind_kmh: float
    wind_from_deg: float

    def towards(self, bearing_deg: float) -> float:
        """ROS (m/min) in compass direction `bearing_deg` from the ignition
        point: the distance to the fire ellipse (head / back / flank rates,
        ignition at the rear focus region) along that direction."""
        R, B = self.head, self.back
        F = max(self.flank, 1e-9)
        a = 0.5 * (R + B)
        c = 0.5 * (R - B)
        downwind = (self.wind_from_deg + 180.0) % 360.0
        th = math.radians(bearing_deg - downwind)
        ct, s = math.cos(th), math.sin(th)
        A = ct * ct / (a * a) + s * s / (F * F)
        Bq = -2.0 * c * ct / (a * a)          # ignition point at x = -c from the ellipse centre
        C = c * c / (a * a) - 1.0
        disc = max(0.0, Bq * Bq - 4.0 * A * C)
        return max(0.0, (-Bq + math.sqrt(disc)) / (2.0 * A))


def spread_rates(ffmc: float, bui: float, wind_ms: float, wind_from_deg: float,
                 fuel: FuelModel = FuelModel()) -> SpreadRates:
    w = max(0.0, float(wind_ms)) * 3.6                         # m/s (10 m open wind) -> km/h
    ffmc = 85.0 if not np.isfinite(ffmc) else float(ffmc)
    ff = moisture_function(ffmc)
    isi = 0.208 * wind_function(w) * ff
    head = fuel.ros(isi, bui)
    back = fuel.ros(0.208 * math.exp(-0.05039 * w) * ff, bui)
    back = max(back, fuel.min_back_fraction * head)       # local gusts / eddies: some backing spread in strong wind
    lb = min(length_to_breadth(w), fuel.max_length_to_breadth)
    flank = (head + back) / (2.0 * lb)
    return SpreadRates(head, min(back, head), flank, lb, isi, w, float(wind_from_deg) % 360.0)


def byram_intensity_kw_m(ros_m_min, fuel_consumed_kg_m2=1.0):
    """Byram fireline intensity I = H w R (H = 18 000 kJ/kg)."""
    return 18000.0 * fuel_consumed_kg_m2 * np.asarray(ros_m_min, dtype=float) / 60.0


INTENSITY_KW_CLASSES = [(500.0, "LOW"), (2000.0, "MODERATE"), (4000.0, "HIGH"), (float("inf"), "EXTREME")]


def intensity_class_kw(kw: float) -> str:
    for upper, name in INTENSITY_KW_CLASSES:
        if kw < upper:
            return name
    return "EXTREME"


def visual_intensity(kw):
    """0-1 visual scale: 50 kW/m -> 0, 10 000 kW/m -> 1 (log)."""
    v = np.log10(np.maximum(np.asarray(kw, dtype=float), 1e-6) / 50.0) / math.log10(10000.0 / 50.0)
    return np.clip(v, 0.0, 1.0)


def describe(r: SpreadRates) -> Dict[str, float]:
    return {"ros_head_m_per_min": round(r.head, 3), "ros_back_m_per_min": round(r.back, 3),
            "ros_flank_m_per_min": round(r.flank, 3), "length_to_breadth": round(r.length_to_breadth, 3),
            "isi": round(r.isi, 3), "wind_kmh": round(r.wind_kmh, 2)}
