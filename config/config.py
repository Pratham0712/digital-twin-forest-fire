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
ENV_FILE = BASE_DIR / ".env"
# Loaded here, before APIConfig below reads os.getenv(). Values already set in
# the process environment win (override=False).
load_dotenv(ENV_FILE)

API_KEY_NAMES = ("FIRMS_MAP_KEY", "OWM_API_KEY", "GOOGLE_MAPS_API_KEY")


def env_file_problems(path: Path = ENV_FILE) -> list:
    """Safe diagnostics for .env lines python-dotenv cannot use: a variable
    name with spaces / wrong separators ("FIRMS MAP_KEY", "OWM API KEY") is
    skipped by the parser, so the key silently stays unset. Returns messages
    with line numbers and NAMES only - never a value."""
    import re
    out = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return out
    wanted = {re.sub(r"[^A-Z0-9]", "", k): k for k in API_KEY_NAMES + ("GOOGLE_MAPS_MAP_ID",)}
    for n, line in enumerate(lines, 1):
        t = line.strip()
        if not t or t.startswith("#") or "=" not in t:
            continue
        name = t.split("=", 1)[0].strip()
        if name.startswith("export "):
            name = name[7:].strip()
        expected = wanted.get(re.sub(r"[^A-Z0-9]", "", name.upper()))
        if expected and name != expected:
            out.append(f".env line {n}: variable name '{name}' is not valid, so it is ignored; rename it to {expected}")
    return out


def key_configured() -> dict:
    """{name: True/False} - whether each API key is visible to this process."""
    return {k: bool(os.getenv(k, "").strip()) for k in API_KEY_NAMES}


for _msg in env_file_problems():
    import logging as _logging
    _logging.getLogger(__name__).warning(_msg)

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
    # 10 = the FIRMS API maximum. The model's fire-history features need the
    # previous 10 days of detections; the map only shows the last ~48h as active.
    firms_day_range: int = 10

    # OpenWeatherMap - free tier ~1000 calls/day (SRS 5.4)
    owm_api_key: str = os.getenv("OWM_API_KEY", "")
    owm_base_url: str = "https://api.openweathermap.org/data/2.5"

    # Google Maps Platform (Maps JavaScript API) - satellite base map of the
    # Spread Simulation page. Optional map ID enables vector rendering (tilt /
    # rotation); without one Google's DEMO_MAP_ID is used.
    google_maps_api_key: str = os.getenv("GOOGLE_MAPS_API_KEY", "")
    google_maps_map_id: str = os.getenv("GOOGLE_MAPS_MAP_ID", "")


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
    alert_threshold_pct: float = 40.0
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
    # Local, geographically anchored spread simulation (Spread Simulation page):
    # a focus_size_m x focus_size_m area on real lat/lon cells of local_ca_cell_m,
    # run by the same FireSpreadSimulator. Its time step is scaled with the cell
    # size so the spread rate stays ca_cell_size_m per 15 min. base_spread_prob
    # is a local-scale calibration (the regional CA keeps its default 0.35):
    # at 0.35 every preset burns out within ~10 min on the 25 m grid; at 0.8 a
    # monsoon-calm fire still dies out while extreme / high-wind fires cross the
    # area. Not yet validated against observed fire perimeters.
    focus_size_m: int = 500
    local_ca_cell_m: int = 25
    local_ca_base_spread_prob: float = 0.8
    # User-configurable simulation area (What-If / Spread Simulation). The CA
    # domain is the area plus a margin of (steps + 1) cells, so the fire is
    # limited by the simulated duration, never by the drawn box.
    focus_min_m: int = 100
    focus_max_m: int = 3000
    local_cell_options_m: tuple = (5, 10, 25, 50)
    max_domain_cells: int = 240          # legacy fixed-domain limit; the adaptive domain uses DomainConfig
    # Natural-looking local spread (inputs / neighbour distance only; the CA rules are unchanged):
    # * diagonal correction: a diagonal neighbour is sqrt(2) further away and touches the cell
    #   only at a corner, so its per-step spread probability is scaled by this factor. An
    #   uncorrected 8-neighbour CA burns a SQUARE under calm wind (diagonal / axis extent
    #   1.22-1.40 measured for the presets); 0.5 gives a near-circular calm-wind fire
    #   (1.02-1.19), so wind and slope - not the grid - shape the fire. Geometric calibration
    #   only (measured with scripts/benchmark_simulation.py conditions); 1.0 / False = off;
    # * fuel variability: smooth, fixed (location-seeded) +/- variation of the zone fuel value
    #   between cells - natural patchiness that gives irregular fronts. SIMULATED, labelled so.
    local_ca_diagonal_correction: float = 0.5
    # * wind model: "elliptical" keeps the CA's head-fire factor but lets flank and backing
    #   fire spread slower, following the elliptical fire shape (Anderson 1983 length-to-breadth
    #   ratio from mid-flame wind, capped at 2 for the 8-direction grid). With the original
    #   "linear" factor the flanks (factor 1.0) and
    #   even the backing fire (0.4) reach the 0.97 probability cap under high FWI, so the fire
    #   grows as a square regardless of the wind. Not calibrated against observed perimeters.
    local_ca_wind_model: str = "elliptical"
    local_ca_fuel_variability: float = 0.45
    local_ca_fuel_patch_gamma: float = 0.6        # <1: contrasting dense / sparse fuel patches (fingered perimeter)
    # Phase 3 - local fire model. "ros" (default): rate-of-spread CA driven by the
    # Canadian FBP System equations (src/simulation/fire_behaviour.py, ros_ca.py):
    # ROS from FFMC (temperature / humidity), wind and BUI, elliptical head / flank /
    # back spread, 16 neighbours with true distances, adaptive sub-steps. "legacy":
    # the original one-trial-per-step probability CA (kept for comparison / tests).
    # The fuel blend, curing, flame / residence times and threshold jitter are
    # ASSUMPTIONS (not calibrated against observed Bandipur fires).
    local_ca_model: str = "ros"
    local_fbp_w_deciduous: float = 0.7          # FBP D-1 share; the rest is O-1a grass
    local_fbp_grass_curing_pct: float = 80.0
    local_ros_max_length_to_breadth: float = 1.8   # irregular fire that also widens / backs, not only a thin oval
    local_ros_threshold_jitter: float = 0.45    # ignition threshold 1 +/- 0.45: stochastic, patchy perimeter (seeded)
    local_ros_min_back_fraction: float = 0.2   # backing / upwind spread >= 20 % of the head rate (spread in all directions)
    local_ros_fuel_noise_weight: float = 1.0    # weight of the simulated fuel patchiness on the spread rate
    local_ros_flame_min: float = 4.0            # flaming residence of a cell (min, x0.75-1.25 per cell)
    local_ros_residence_max_min: float = 45.0   # a front cell stops burning after this even if not spread
    local_ros_max_substeps: int = 20000         # computation guard (reported if it limits the step)
    local_ros_fuel_consumed_kg_m2: float = 1.0  # for the Byram fireline intensity
    max_duration_minutes: int = 240
    duration_presets_minutes: tuple = (1, 5, 10, 15, 30, 60, 120)


# ── Local land cover (fuel / non-fuel) - fusion of real geospatial layers ── #
# ESA WorldCover (broad land-cover classes) + OpenStreetMap (roads, buildings,
# water as vectors) + Sentinel-2 NDVI (vegetation condition -> relative fuel
# load). Every threshold the fusion uses lives here, documented, so nothing in
# src/landcover is an unexplained magic number. None of these values has been
# calibrated against observed fires; they are transparent, documented defaults.
@dataclass
class LandCoverConfig:
    fusion_version: str = "fusion-1.0"
    rules_version: str = "rules-1.0"
    # data sources (switches; tests run fully offline with fixtures)
    use_worldcover: bool = True
    use_osm: bool = True
    use_sentinel2: bool = True
    use_dynamic_world: bool = False      # needs a Google Earth Engine account; cached probabilities only
    worldcover_url: str = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
                           "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")
    worldcover_version: str = "v200 (2021)"
    worldcover_res_deg: float = 1.0 / 12000.0       # 10 m product on a 1/12000 deg lattice (EPSG:4326)
    stac_url: str = "https://earth-search.aws.element84.com/v1/search"
    sentinel2_collection: str = "sentinel-2-l2a"
    sentinel2_lookback_days: int = 45
    sentinel2_max_cloud_pct: float = 40.0
    sentinel2_max_scenes: int = 4
    # Scene Classification Layer values kept for NDVI: 4 vegetation, 5 bare soil,
    # 6 water, 7 unclassified. Dropped: 0 no data, 1 saturated, 2 dark/shadow,
    # 3 cloud shadow, 8-10 cloud / cirrus, 11 snow.
    sentinel2_scl_keep: tuple = (4, 5, 6, 7)
    # cache: window tiles of tile_deg x tile_deg; expiry per dataset
    cache_tile_deg: float = 0.01
    worldcover_ttl_days: float = 3650.0  # static 2021 product: invalidated by version, not by age
    osm_ttl_days: float = 30.0
    ndvi_ttl_days: float = 15.0
    fused_ttl_days: float = 15.0
    http_timeout_s: float = 25.0
    retry_after_failure_s: float = 300.0
    # supersampling of each CA cell (points per side = clip(ceil(cell / supersample_m), min, max))
    supersample_m: float = 5.0
    supersample_min: int = 2
    supersample_max: int = 10
    # per-cell classification thresholds (fractions of the cell area)
    water_block_fraction: float = 0.5
    built_block_fraction: float = 0.5
    road_block_fraction: float = 0.5     # minor roads: fractional coverage only
    bare_block_fraction: float = 0.6
    min_vegetation_fraction: float = 0.3
    mixed_nonburnable_fraction: float = 0.7   # mixed cell: NON_FUEL when water+built+road+bare >= this
    # Major roads / railways are continuous firebreaks: every cell their centre
    # line crosses is ROAD (4-connected, so an 8-neighbour fire cannot slip
    # through diagonally). Minor roads and tracks only count by area fraction.
    road_block_classes: tuple = ("motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
                                 "secondary", "secondary_link")
    railway_blocks: bool = True
    blocking_waterways: tuple = ("river",)
    # OSM built-up land use (residential, commercial ...) often encloses gardens
    # and trees: with WorldCover it removes only points WorldCover does NOT see as
    # vegetation. Buildings, water and roads are always explicit barriers.
    landuse_overrides_vegetation: bool = False
    # WorldCover class -> fuel group. Cropland and herbaceous wetland are burnable
    # (seasonal); snow/ice and moss/lichen are treated as bare.
    worldcover_fuel_classes: tuple = (10, 20, 30, 40, 90, 95)
    # relative fuel load when NDVI is unavailable (class default, DERIVED)
    default_fuel_load: dict = field(default_factory=lambda: {10: 0.80, 20: 0.70, 30: 0.60, 40: 0.45,
                                                             90: 0.40, 95: 0.70})
    # NDVI -> relative fuel availability proxy: (NDVI - bare) / (dense - bare), clipped 0-1
    ndvi_bare: float = 0.15
    ndvi_dense: float = 0.80
    # relative fuel load of an UNVERIFIED cell (no land-cover data, no NDVI); only
    # used when the user explicitly runs in DEGRADED LAND-COVER MODE
    unverified_fuel_load: float = 0.6
    # confidence: WorldCover class purity needed for HIGH / MEDIUM
    purity_high: float = 0.8
    purity_medium: float = 0.5
    # published global overall accuracy of WorldCover 2021 v200 (ESA validation
    # report); quoted for context only - local accuracy is unknown
    worldcover_published_accuracy: str = "about 77% global overall accuracy (ESA WorldCover 2021 v200 validation)"


# ── Adaptive / expanding local simulation domain ─────────────────────────── #
@dataclass
class DomainConfig:
    # Initial domain = focus area + margin. The margin is the full-duration
    # margin (steps + 1 cells, the fire cannot reach the edge) while that domain
    # is at most initial_domain_max_cells per side; larger set-ups start with a
    # compact margin and grow while the fire spreads.
    initial_domain_max_cells: int = 120
    compact_margin_cells: int = 24
    # Expansion: when a burning cell is within safety_margin_cells of an edge,
    # that side grows by max(expansion_chunk_cells, expansion_chunk_frac * size).
    safety_margin_cells: int = 6
    expansion_chunk_cells: int = 24
    expansion_chunk_frac: float = 0.25
    # Hard limits (reported as "SIMULATION EXTENT LIMIT REACHED")
    max_extent_m: float = 15000.0
    max_side_cells: int = 600
    max_total_cells: int = 250_000
    max_expansions: int = 40
    # an ignition point closer than this to the initial domain edge extends the
    # initial domain first (no truncated ignition, audit BUG #1)
    ignition_edge_margin_cells: int = 6


# ── Primary study region ─────────────────────────────────────────────────── #
@dataclass
class StudyRegionConfig:
    name: str = "Bandipur Tiger Reserve"
    # Official boundary: put the official polygon here (GeoJSON, KML or ESRI
    # shapefile with .prj). It is used automatically and labelled OFFICIAL.
    official_boundary_stem: str = str(BASE_DIR / "data" / "study_region" / "bandipur_official")
    # Development fallback, labelled APPROXIMATE / PROVISIONAL everywhere.
    provisional_boundary_path: str = str(BASE_DIR / "data" / "study_region" / "bandipur_provisional.geojson")
    stated_area_km2: float = 1456.309    # NTCA brief note: core 872.24 + buffer 584.069 km2
    stated_area_source: str = "NTCA Project Tiger brief note (core 872.24 km² + buffer 584.07 km²)"


API = APIConfig()
REGION = RegionConfig()
SYSTEM = SystemConfig()
LAND_COVER = LandCoverConfig()
DOMAIN = DomainConfig()
STUDY_REGION = StudyRegionConfig()
