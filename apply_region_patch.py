import sys

def patch(path, old, new, label):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if old not in content:
        print(f"[SKIP] {label}: pattern not found -> {path}")
        return
    if content.count(old) > 1:
        print(f"[FAIL] {label}: multiple matches -> {path}")
        sys.exit(1)
    content = content.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK]   {label} -> {path}")

patch(
    "src/data_ingestion/ingestion_module.py",
    '    def __init__(self, offline: bool = False, scenario: Optional[dict] = None):\n        self.offline = offline\n        self.scenario = scenario or {}\n        self.firms = FIRMSClient(API.firms_map_key, API.firms_base_url, API.firms_source)\n        self.weather = WeatherClient(API.owm_api_key, API.owm_base_url)\n        self.grid = build_region_grid()\n        self.weather_grid = build_weather_grid()\n\n    def fetch_fire_hotspots(self) -> pd.DataFrame:\n        if self.offline or not API.firms_map_key:\n            return FIRMSClient.generate_sample(\n                {"min_lat": REGION.min_lat, "max_lat": REGION.max_lat,\n                 "min_lon": REGION.min_lon, "max_lon": REGION.max_lon},\n                n_points=self.scenario.get("n_hotspots", 8),\n            )\n        return self.firms.fetch_hotspots(\n            REGION.min_lat, REGION.min_lon, REGION.max_lat, REGION.max_lon,\n            API.firms_day_range,\n        )',
    '    def __init__(self, offline: bool = False, scenario: Optional[dict] = None, region=None):\n        self.offline = offline\n        self.scenario = scenario or {}\n        self.region = region or REGION\n        self.firms = FIRMSClient(API.firms_map_key, API.firms_base_url, API.firms_source)\n        self.weather = WeatherClient(API.owm_api_key, API.owm_base_url)\n        self.grid = build_region_grid(self.region)\n        self.weather_grid = build_weather_grid(self.region)\n\n    def fetch_fire_hotspots(self) -> pd.DataFrame:\n        if self.offline or not API.firms_map_key:\n            return FIRMSClient.generate_sample(\n                {"min_lat": self.region.min_lat, "max_lat": self.region.max_lat,\n                 "min_lon": self.region.min_lon, "max_lon": self.region.max_lon},\n                n_points=self.scenario.get("n_hotspots", 8),\n            )\n        return self.firms.fetch_hotspots(\n            self.region.min_lat, self.region.min_lon, self.region.max_lat, self.region.max_lon,\n            API.firms_day_range,\n        )',
    "ingestion_module.py: DataIngestionModule accepts custom region",
)

patch(
    "src/data_ingestion/ingestion_module.py",
    'prefix="fire", max_distance_deg=REGION.grid_resolution_deg * 0.75,',
    'prefix="fire", max_distance_deg=self.region.grid_resolution_deg * 0.75,',
    "ingestion_module.py: nearest-neighbor join uses custom resolution",
)

patch(
    "src/digital_twin/twin_state.py",
    '    def __init__(self, ml_model=None, offline: bool = True, use_real_model: bool = True,\n                 scenario: Optional[dict] = None):',
    '    def __init__(self, ml_model=None, offline: bool = True, use_real_model: bool = True,\n                 scenario: Optional[dict] = None, region=None):',
    "twin_state.py: DigitalTwin accepts region",
)
patch(
    "src/digital_twin/twin_state.py",
    '        self.offline = offline\n        self.ingestion = DataIngestionModule(offline=offline, scenario=scenario)',
    '        self.offline = offline\n        self.region = region or REGION\n        self.ingestion = DataIngestionModule(offline=offline, scenario=scenario, region=self.region)',
    "twin_state.py: DigitalTwin stores + forwards region",
)
patch(
    "src/digital_twin/twin_state.py",
    '            "status": "ok",\n            "timestamp": snap.timestamp,\n            "region": REGION.name,',
    '            "status": "ok",\n            "timestamp": snap.timestamp,\n            "region": self.region.name,',
    "twin_state.py: summary reports the active region",
)

patch(
    "src/dashboard/app.py",
    'from config.config import REGION, SYSTEM, MODELS_DIR, DATA_PROCESSED_DIR',
    'from config.config import REGION, SYSTEM, MODELS_DIR, DATA_PROCESSED_DIR, RegionConfig',
    "app.py: import RegionConfig",
)
patch(
    "src/dashboard/app.py",
    'def get_twin(offline: bool, scenario: dict = None) -> DigitalTwin:\n    model = load_ml_model()\n    return DigitalTwin(ml_model=model, offline=offline, scenario=scenario)',
    'def get_twin(offline: bool, scenario: dict = None, region=None) -> DigitalTwin:\n    model = load_ml_model()\n    return DigitalTwin(ml_model=model, offline=offline, scenario=scenario, region=region)',
    "app.py: get_twin() accepts region",
)
patch(
    "src/dashboard/app.py",
    '    if "twin" not in st.session_state or st.sidebar.button("Refresh data"):\n        with st.spinner("Refreshing digital twin state..."):\n            twin = get_twin(offline, scenario)\n            twin.refresh()\n        st.session_state["twin"] = twin\n        st.session_state["ca_history"] = None',
    '    st.sidebar.subheader("Region")\n    PRESET_REGIONS = {\n        "Karnataka Western Ghats (default, model-validated)": REGION,\n        "California": RegionConfig(name="California", min_lat=32.5, max_lat=42.0,\n                                    min_lon=-124.5, max_lon=-114.0,\n                                    grid_resolution_deg=0.5, weather_grid_resolution_deg=1.5),\n        "Australia (SE)": RegionConfig(name="Australia (SE)", min_lat=-39.0, max_lat=-28.0,\n                                        min_lon=140.0, max_lon=154.0,\n                                        grid_resolution_deg=0.5, weather_grid_resolution_deg=1.5),\n        "Mediterranean (S. Europe)": RegionConfig(name="Mediterranean", min_lat=36.0, max_lat=44.0,\n                                                   min_lon=-5.0, max_lon=20.0,\n                                                   grid_resolution_deg=0.5, weather_grid_resolution_deg=1.5),\n        "Custom (define below)": None,\n    }\n    region_choice = st.sidebar.selectbox("Region preset", list(PRESET_REGIONS.keys()))\n    if region_choice == "Custom (define below)":\n        c1, c2 = st.sidebar.columns(2)\n        min_lat = c1.number_input("Min lat", value=11.5, format="%.2f")\n        max_lat = c2.number_input("Max lat", value=15.5, format="%.2f")\n        min_lon = c1.number_input("Min lon", value=74.0, format="%.2f")\n        max_lon = c2.number_input("Max lon", value=77.5, format="%.2f")\n        custom_name = st.sidebar.text_input("Region name", value="Custom region")\n        region = RegionConfig(name=custom_name, min_lat=min_lat, max_lat=max_lat,\n                               min_lon=min_lon, max_lon=max_lon,\n                               grid_resolution_deg=0.5, weather_grid_resolution_deg=1.5)\n    else:\n        region = PRESET_REGIONS[region_choice]\n    if region.name != REGION.name:\n        st.sidebar.warning(\n            "Model was trained only on Karnataka data. Predictions outside "\n            "Karnataka are for architecture demonstration only, not validated accuracy."\n        )\n\n    if ("twin" not in st.session_state or st.sidebar.button("Refresh data")\n            or st.session_state.get("_last_region") != region.name):\n        with st.spinner("Refreshing digital twin state..."):\n            twin = get_twin(offline, scenario, region)\n            twin.refresh()\n        st.session_state["twin"] = twin\n        st.session_state["ca_history"] = None\n        st.session_state["_last_region"] = region.name',
    "app.py: region picker (presets + custom bounding box) wired to refresh",
)

patch(
    "src/dashboard/app.py",
    'center={"lat": (REGION.min_lat + REGION.max_lat) / 2, "lon": (REGION.min_lon + REGION.max_lon) / 2},',
    'center={"lat": (twin.region.min_lat + twin.region.max_lat) / 2, "lon": (twin.region.min_lon + twin.region.max_lon) / 2},',
    "app.py: risk map centers on the active region",
)

print("\nPatch script finished.")
