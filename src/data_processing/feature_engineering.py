"""
DataProcessor - cleans, normalizes, and engineers model-ready features from the
unified ingestion frame, per SRS 5.1 (Data Processing): Fire Weather Index (FWI),
Duff Moisture Code (DMC), Buildup Index (BUI), Drought Code (DC), and NDVI.

The FWI-family formulas below implement the Canadian Forest Fire Weather Index
System (Van Wagner, 1987) — the same indices used as engineered inputs in
several of the reviewed papers (Report Ch.3, e.g. papers on Algerian Forest
Fire dataset which report FWI/FFMC/DMC/DC as core predictors).
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Canadian FWI System component calculations
# --------------------------------------------------------------------------- #

def compute_ffmc(temp_c: np.ndarray, rh_pct: np.ndarray, wind_ms: np.ndarray,
                  rain_mm: np.ndarray, ffmc_prev: float = 85.0) -> np.ndarray:
    """Fine Fuel Moisture Code - moisture of fine litter fuels (proxy: 0-101)."""
    wind_kmh = wind_ms * 3.6
    mo = 147.2 * (101 - ffmc_prev) / (59.5 + ffmc_prev)

    rain_effect = np.where(rain_mm > 0.5, rain_mm - 0.5, 0.0)
    mo = np.where(
        rain_mm > 0.5,
        mo + 42.5 * rain_effect * np.exp(-100 / (251 - mo)) * (1 - np.exp(-6.93 / rain_effect.clip(min=1e-6))),
        mo,
    )
    mo = np.clip(mo, 0, 250)

    ed = 0.942 * rh_pct ** 0.679 + 11 * np.exp((rh_pct - 100) / 10) + \
        0.18 * (21.1 - temp_c) * (1 - np.exp(-0.115 * rh_pct))
    ew = 0.618 * rh_pct ** 0.753 + 10 * np.exp((rh_pct - 100) / 10) + \
        0.18 * (21.1 - temp_c) * (1 - np.exp(-0.115 * rh_pct))

    m = np.where(mo > ed, ed, np.where(mo < ew, ew, mo))
    kd = 0.424 * (1 - (rh_pct / 100) ** 1.7) + 0.0694 * np.sqrt(wind_kmh) * (1 - (rh_pct / 100) ** 8)
    kw = kd * 0.581 * np.exp(0.0365 * temp_c)

    m_new = np.where(mo > ed, ed + (mo - ed) * 10 ** (-kd),
                      np.where(mo < ew, ew - (ew - mo) * 10 ** (-kw), m))
    ffmc = (59.5 * (250 - m_new)) / (147.2 + m_new)
    return np.clip(ffmc, 0, 101)


def compute_dmc(temp_c: np.ndarray, rh_pct: np.ndarray, rain_mm: np.ndarray,
                 month: int = 5, dmc_prev: float = 6.0) -> np.ndarray:
    """Duff Moisture Code - moisture of loosely-compacted, moderate-depth organic layers."""
    day_length_factor = {1: 6.5, 2: 7.5, 3: 9.0, 4: 12.8, 5: 13.9, 6: 13.9,
                          7: 12.4, 8: 10.9, 9: 9.4, 10: 8.0, 11: 7.0, 12: 6.0}
    le = day_length_factor.get(month, 9.0)

    rk = 1.894 * (temp_c + 1.1) * (100 - rh_pct) * le * 1e-4
    rk = np.clip(rk, 0, None)
    dmc_prev_arr = np.full_like(temp_c, dmc_prev, dtype=float) if np.isscalar(dmc_prev) else dmc_prev
    dmc = dmc_prev_arr + rk

    log_safe = np.clip(dmc_prev_arr, 1e-6, None)
    b = np.where(dmc_prev_arr <= 33, 100 / (0.5 + 0.3 * dmc_prev_arr),
                 np.where(dmc_prev_arr <= 65, 14 - 1.3 * np.log(log_safe),
                          6.2 * np.log(log_safe) - 17.2))
    rain_excess = np.clip(rain_mm - 1.27, 1e-6, None)
    mr = dmc_prev_arr + (1000 * (rain_mm - 1.27)) / (48.77 + b * rain_excess)
    dmc = np.where(rain_mm > 1.5, np.clip(mr, 0, None) + rk, dmc)
    return np.clip(dmc, 0, None)


def compute_dc(temp_c: np.ndarray, rain_mm: np.ndarray, month: int = 5,
                dc_prev: float = 15.0) -> np.ndarray:
    """Drought Code - deep, slow-drying organic layer moisture; fire-season severity."""
    day_factor = {1: -1.6, 2: -1.6, 3: -1.6, 4: 0.9, 5: 3.8, 6: 5.8,
                  7: 6.4, 8: 5.0, 9: 2.4, 10: 0.4, 11: -1.6, 12: -1.6}
    lf = day_factor.get(month, 3.8)

    v = np.clip(0.36 * (temp_c + 2.8) + lf, 0, None)
    dc_prev_arr = np.full_like(temp_c, dc_prev, dtype=float) if np.isscalar(dc_prev) else dc_prev
    dc = dc_prev_arr + 0.5 * v

    rd = np.where(rain_mm > 2.8, 0.83 * rain_mm - 1.27, 0.0)
    qo = 800 * np.exp(-dc_prev_arr / 400)
    qr = np.clip(qo + 3.937 * rd, 1e-6, None)
    dr = 400 * np.log(800 / qr)
    dc = np.where(rain_mm > 2.8, np.clip(dr, 0, None) + 0.5 * v, dc)
    return np.clip(dc, 0, None)


def compute_bui(dmc: np.ndarray, dc: np.ndarray) -> np.ndarray:
    """Buildup Index - combines DMC and DC into a single fuel-availability index."""
    bui = np.where(
        dmc <= 0.4 * dc,
        (0.8 * dmc * dc) / (dmc + 0.4 * dc).clip(min=1e-6),
        dmc - (1 - 0.8 * dc / (dmc + 0.4 * dc).clip(min=1e-6)) *
        (0.92 + (0.0114 * dmc) ** 1.7),
    )
    return np.clip(bui, 0, None)


def compute_fwi(ffmc: np.ndarray, bui: np.ndarray, wind_ms: np.ndarray) -> np.ndarray:
    """Fire Weather Index - overall fire intensity potential, combines ISI and BUI."""
    wind_kmh = wind_ms * 3.6
    f_ffmc = 91.9 * np.exp(-0.1386 * (101 - ffmc)) * (1 + (101 - ffmc) ** 5.31 / 4.93e7)
    isi = f_ffmc * np.exp(0.05039 * wind_kmh)

    fd = np.where(bui <= 80, 0.626 * bui ** 0.809 + 2,
                  1000 / (25 + 108.64 * np.exp(-0.023 * bui)))
    b = 0.1 * isi * fd
    # np.where evaluates BOTH branches for every element before selecting -
    # for rows where b<=1 (which take the first branch), the second branch's
    # log(b)**0.647 can hit a negative base and warn "invalid value in power"
    # even though that value is discarded. Suppress just this expected,
    # harmless warning rather than the b<=1 rows' correct results.
    with np.errstate(invalid="ignore"):
        b_high = np.exp(2.72 * (0.434 * np.log(b.clip(min=1e-6))) ** 0.647)
    fwi = np.where(b <= 1, b, b_high)
    return np.clip(fwi, 0, None)


def synthetic_ndvi(active_fire_nearby: np.ndarray, rain_mm: np.ndarray,
                    seed: Optional[int] = 11) -> np.ndarray:
    """
    NDVI proxy for offline/demo mode. Real deployments should replace this with
    a Sentinel-2 / MODIS NDVI raster lookup per grid cell (Report SRS 5.4 lists
    GeoPandas for geospatial handling). Values follow the expected -1..1 range,
    biased lower (drier vegetation) near active fires.
    """
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.35, 0.75, len(active_fire_nearby))
    base = np.where(active_fire_nearby, base * rng.uniform(0.2, 0.5, len(base)), base)
    base = np.where(rain_mm > 1.0, np.clip(base * 1.1, 0, 0.9), base)
    return np.clip(base, -1, 1)


class DataProcessor:
    """
    Layer 2 (Data Processing) of the five-layer architecture (Report 6.1.1).
    Consumes the unified frame from DataIngestionModule and produces the
    model-ready feature matrix consumed by MLPredictor.
    """

    FEATURE_COLUMNS = [
        "wx_temperature_c", "wx_humidity_pct", "wx_wind_speed_ms",
        "wx_precipitation_mm", "ffmc", "dmc", "dc", "bui", "fwi", "ndvi",
        "fire_frp", "active_fire_nearby",
    ]

    def __init__(self, month: Optional[int] = None):
        self.month = month or pd.Timestamp.now(tz="UTC").month

    def transform(self, unified: pd.DataFrame) -> pd.DataFrame:
        df = unified.copy()

        # Basic cleaning: impute missing weather with grid-wide median (SRS 5.1)
        weather_cols = ["wx_temperature_c", "wx_humidity_pct", "wx_wind_speed_ms", "wx_precipitation_mm"]
        for col in weather_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        df["fire_frp"] = df["fire_frp"].fillna(0.0)
        df["active_fire_nearby"] = df["active_fire_nearby"].fillna(False).astype(bool)

        temp = df["wx_temperature_c"].values
        rh = df["wx_humidity_pct"].values
        wind = df["wx_wind_speed_ms"].values
        rain = df["wx_precipitation_mm"].values

        df["ffmc"] = compute_ffmc(temp, rh, wind, rain)
        df["dmc"] = compute_dmc(temp, rh, rain, month=self.month)
        df["dc"] = compute_dc(temp, rain, month=self.month)
        df["bui"] = compute_bui(df["dmc"].values, df["dc"].values)
        df["fwi"] = compute_fwi(df["ffmc"].values, df["bui"].values, wind)
        df["ndvi"] = synthetic_ndvi(df["active_fire_nearby"].values, rain)

        # Weather-derived fire-risk label for supervised training when ground-truth
        # fire occurrence isn't available: a zone is "fire" if it currently has an
        # active hotspot OR its FWI is in the extreme range (Scenario 1: FWI=89 -> Extreme)
        df["fire_risk_label"] = ((df["active_fire_nearby"]) | (df["fwi"] > 60)).astype(int)

        logger.info(
            "Feature engineering complete: %d zones, %d positive labels (%.1f%%)",
            len(df), df["fire_risk_label"].sum(),
            100 * df["fire_risk_label"].mean(),
        )
        return df

    def get_feature_matrix(self, processed: pd.DataFrame):
        X = processed[self.FEATURE_COLUMNS].copy()
        X["active_fire_nearby"] = X["active_fire_nearby"].astype(int)
        y = processed["fire_risk_label"]
        return X, y


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.data_ingestion.ingestion_module import DataIngestionModule

    logging.basicConfig(level=logging.INFO)
    ingestion = DataIngestionModule(offline=True)
    unified = ingestion.build_unified_frame()

    processor = DataProcessor()
    processed = processor.transform(unified)
    X, y = processor.get_feature_matrix(processed)
    print(processed[["zone_id", "ffmc", "dmc", "dc", "bui", "fwi", "ndvi", "fire_risk_label"]].head(10))
    print(f"\nFeature matrix: {X.shape}, positive rate: {y.mean():.2%}")

    out_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "features_sample.csv"
    processed.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
