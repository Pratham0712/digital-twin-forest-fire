"""
Central configuration for the Digital Twin Framework for Forest Fire Prediction.
All tunable parameters, API endpoints, and thresholds are defined here so that
no module hardcodes values that Chapter 5 (SRS) specifies as configurable.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

for _dir in (DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


@dataclass
class APIConfig:
    # NASA FIRMS - fire hotspot API. Get a free MAP_KEY at https://firms.modaps.eosdis.nasa.gov/api/
    firms_map_key: str = os.getenv("FIRMS_MAP_KEY", "")
    firms_base_url: str = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
    firms_source: str = "VIIRS_SNPP_NRT"  # SRS 5.1: MODIS/VIIRS satellite feeds
    firms_day_range: int = 1

    # OpenWeatherMap - free tier ~1000 calls/day (SRS 5.4)
    owm_api_key: str = os.getenv("OWM_API_KEY", "")
    owm_base_url: str = "https://api.openweathermap.org/data/2.5"


@dataclass
class RegionConfig:
    """Study region: Karnataka and the Western Ghats (Report Ch.1 Scope)."""
    name: str = "Karnataka Western Ghats"
    min_lat: float = 11.5
    max_lat: float = 15.5
    min_lon: float = 74.0
    max_lon: float = 77.5
    grid_resolution_deg: float = 0.1  # ~11 km grid cells - used for fire detection + CA sim
    weather_grid_resolution_deg: float = 0.5  # ~55 km - weather is spatially smoother, fetched sparser


@dataclass
class SystemConfig:
    # SRS 5.1 Data Ingestion
    min_refresh_interval_minutes: int = 15
    # SRS 5.1 Alert System
    alert_threshold_pct: float = 70.0
    # SRS 5.2 Performance
    max_processing_latency_sec: int = 30
    ca_grid_max_seconds: int = 5
    # SRS 5.2 Accuracy targets
    target_f1: float = 0.88
    target_auc: float = 0.92
    max_false_negative_rate: float = 0.05
    # Fire spread simulation horizon (Scenario 1: 2-hour projection)
    fire_spread_horizon_hours: int = 2
    ca_cell_size_m: int = 100


API = APIConfig()
REGION = RegionConfig()
SYSTEM = SystemConfig()
