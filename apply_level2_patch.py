import sys

def patch(path, old, new, label):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if old not in content:
        print(f"[SKIP] {label}: pattern not found (already patched?) -> {path}")
        return
    if content.count(old) > 1:
        print(f"[FAIL] {label}: pattern matches more than once -> {path}")
        sys.exit(1)
    content = content.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK]   {label} -> {path}")

patch(
    "src/data_ingestion/firms_client.py",
    '    def generate_sample(region_bounds: dict, n_points: int = 25,\n'
    '                         seed: Optional[int] = 42) -> pd.DataFrame:',
    '    def generate_sample(region_bounds: dict, n_points: int = 25,\n'
    '                         seed: Optional[int] = None) -> pd.DataFrame:',
    "firms_client.py: random seed",
)

patch(
    "src/data_ingestion/weather_client.py",
    '    @staticmethod\n'
    '    def generate_sample(grid_points: List[dict], seed: Optional[int] = 7) -> pd.DataFrame:\n'
    '        """Synthetic weather generator for offline dev/tests/demo mode."""\n'
    '        import numpy as np\n'
    '        rng = np.random.default_rng(seed)\n'
    '        n = len(grid_points)\n'
    '        df = pd.DataFrame({\n'
    '            "latitude": [p["latitude"] for p in grid_points],\n'
    '            "longitude": [p["longitude"] for p in grid_points],\n'
    '            "temperature_c": rng.uniform(22, 42, n),\n'
    '            "humidity_pct": rng.uniform(8, 85, n),\n'
    '            "pressure_hpa": rng.uniform(1005, 1015, n),\n'
    '            "wind_speed_ms": rng.uniform(0.5, 12, n),\n'
    '            "wind_deg": rng.uniform(0, 360, n),\n'
    '            "precipitation_mm": rng.choice([0, 0, 0, 0.5, 2.0], n),\n'
    '            "clouds_pct": rng.integers(0, 100, n),\n'
    '            "weather_main": rng.choice(["Clear", "Clouds", "Rain"], n, p=[0.6, 0.3, 0.1]),\n'
    '        })\n'
    '        df["fetched_at"] = datetime.now(timezone.utc).isoformat()\n'
    '        return df',
    '    @staticmethod\n'
    '    def generate_sample(grid_points: List[dict], seed: Optional[int] = None,\n'
    '                         temp_c: Optional[float] = None,\n'
    '                         wind_speed_ms: Optional[float] = None,\n'
    '                         humidity_pct: Optional[float] = None) -> pd.DataFrame:\n'
    '        """Synthetic weather generator. Pass temp_c/wind_speed_ms/humidity_pct\n'
    '        to center the spread around a chosen scenario value."""\n'
    '        import numpy as np\n'
    '        rng = np.random.default_rng(seed)\n'
    '        n = len(grid_points)\n'
    '        temp_lo, temp_hi = (22, 42) if temp_c is None else (temp_c - 4, temp_c + 4)\n'
    '        wind_lo, wind_hi = (0.5, 12) if wind_speed_ms is None else (max(0.0, wind_speed_ms - 2), wind_speed_ms + 2)\n'
    '        hum_lo, hum_hi = (8, 85) if humidity_pct is None else (max(0.0, humidity_pct - 10), min(100.0, humidity_pct + 10))\n'
    '        df = pd.DataFrame({\n'
    '            "latitude": [p["latitude"] for p in grid_points],\n'
    '            "longitude": [p["longitude"] for p in grid_points],\n'
    '            "temperature_c": rng.uniform(temp_lo, temp_hi, n),\n'
    '            "humidity_pct": rng.uniform(hum_lo, hum_hi, n),\n'
    '            "pressure_hpa": rng.uniform(1005, 1015, n),\n'
    '            "wind_speed_ms": rng.uniform(wind_lo, wind_hi, n),\n'
    '            "wind_deg": rng.uniform(0, 360, n),\n'
    '            "precipitation_mm": rng.choice([0, 0, 0, 0.5, 2.0], n),\n'
    '            "clouds_pct": rng.integers(0, 100, n),\n'
    '            "weather_main": rng.choice(["Clear", "Clouds", "Rain"], n, p=[0.6, 0.3, 0.1]),\n'
    '        })\n'
    '        df["fetched_at"] = datetime.now(timezone.utc).isoformat()\n'
    '        return df',
    "weather_client.py: scenario overrides",
)

patch(
    "src/data_ingestion/ingestion_module.py",
    '    def __init__(self, offline: bool = False):\n'
    '        self.offline = offline\n'
    '        self.firms = FIRMSClient(API.firms_map_key, API.firms_base_url, API.firms_source)\n'
    '        self.weather = WeatherClient(API.owm_api_key, API.owm_base_url)\n'
    '        self.grid = build_region_grid()\n'
    '        self.weather_grid = build_weather_grid()\n'
    '\n'
    '    def fetch_fire_hotspots(self) -> pd.DataFrame:\n'
    '        if self.offline or not API.firms_map_key:\n'
    '            return FIRMSClient.generate_sample(\n'
    '                {"min_lat": REGION.min_lat, "max_lat": REGION.max_lat,\n'
    '                 "min_lon": REGION.min_lon, "max_lon": REGION.max_lon},\n'
    '                n_points=8,\n'
    '            )\n'
    '        return self.firms.fetch_hotspots(\n'
    '            REGION.min_lat, REGION.min_lon, REGION.max_lat, REGION.max_lon,\n'
    '            API.firms_day_range,\n'
    '        )\n'
    '\n'
    '    def fetch_weather(self) -> pd.DataFrame:\n'
    '        grid_points = self.weather_grid[["latitude", "longitude"]].to_dict("records")\n'
    '        if self.offline or not API.owm_api_key:\n'
    '            return WeatherClient.generate_sample(grid_points)\n'
    '        return self.weather.fetch_grid(grid_points)',
    '    def __init__(self, offline: bool = False, scenario: Optional[dict] = None):\n'
    '        self.offline = offline\n'
    '        self.scenario = scenario or {}\n'
    '        self.firms = FIRMSClient(API.firms_map_key, API.firms_base_url, API.firms_source)\n'
    '        self.weather = WeatherClient(API.owm_api_key, API.owm_base_url)\n'
    '        self.grid = build_region_grid()\n'
    '        self.weather_grid = build_weather_grid()\n'
    '\n'
    '    def fetch_fire_hotspots(self) -> pd.DataFrame:\n'
    '        if self.offline or not API.firms_map_key:\n'
    '            return FIRMSClient.generate_sample(\n'
    '                {"min_lat": REGION.min_lat, "max_lat": REGION.max_lat,\n'
    '                 "min_lon": REGION.min_lon, "max_lon": REGION.max_lon},\n'
    '                n_points=self.scenario.get("n_hotspots", 8),\n'
    '            )\n'
    '        return self.firms.fetch_hotspots(\n'
    '            REGION.min_lat, REGION.min_lon, REGION.max_lat, REGION.max_lon,\n'
    '            API.firms_day_range,\n'
    '        )\n'
    '\n'
    '    def fetch_weather(self) -> pd.DataFrame:\n'
    '        grid_points = self.weather_grid[["latitude", "longitude"]].to_dict("records")\n'
    '        if self.offline or not API.owm_api_key:\n'
    '            return WeatherClient.generate_sample(\n'
    '                grid_points,\n'
    '                temp_c=self.scenario.get("temp_c"),\n'
    '                wind_speed_ms=self.scenario.get("wind_speed_ms"),\n'
    '                humidity_pct=self.scenario.get("humidity_pct"),\n'
    '            )\n'
    '        return self.weather.fetch_grid(grid_points)',
    "ingestion_module.py: scenario threading",
)

patch(
    "src/digital_twin/twin_state.py",
    '    def __init__(self, ml_model=None, offline: bool = True, use_real_model: bool = True):',
    '    def __init__(self, ml_model=None, offline: bool = True, use_real_model: bool = True,\n'
    '                 scenario: Optional[dict] = None):',
    "twin_state.py: accept scenario",
)
patch(
    "src/digital_twin/twin_state.py",
    '        self.offline = offline\n'
    '        self.ingestion = DataIngestionModule(offline=offline)',
    '        self.offline = offline\n'
    '        self.ingestion = DataIngestionModule(offline=offline, scenario=scenario)',
    "twin_state.py: forward scenario",
)

patch(
    "src/dashboard/app.py",
    'def get_twin(offline: bool) -> DigitalTwin:\n'
    '    model = load_ml_model()\n'
    '    return DigitalTwin(ml_model=model, offline=offline)',
    'def get_twin(offline: bool, scenario: dict = None) -> DigitalTwin:\n'
    '    model = load_ml_model()\n'
    '    return DigitalTwin(ml_model=model, offline=offline, scenario=scenario)',
    "app.py: get_twin scenario param",
)
patch(
    "src/dashboard/app.py",
    '    offline = st.sidebar.toggle(\n'
    '        "Offline / demo mode", value=not bool(os.getenv("FIRMS_MAP_KEY")),\n'
    '        help="Uses synthetic data. Turn off once FIRMS_MAP_KEY and OWM_API_KEY are set.",\n'
    '    )\n'
    '    st.sidebar.caption(f"Alert threshold: {SYSTEM.alert_threshold_pct:.0f}% (config.py)")\n'
    '\n'
    '    if "twin" not in st.session_state or st.sidebar.button("Refresh data"):\n'
    '        with st.spinner("Refreshing digital twin state..."):\n'
    '            twin = get_twin(offline)\n'
    '            twin.refresh()\n'
    '        st.session_state["twin"] = twin\n'
    '        st.session_state["ca_history"] = None',
    '    offline = st.sidebar.toggle(\n'
    '        "Offline / demo mode", value=not bool(os.getenv("FIRMS_MAP_KEY")),\n'
    '        help="Uses synthetic data. Turn off once FIRMS_MAP_KEY and OWM_API_KEY are set.",\n'
    '    )\n'
    '    st.sidebar.caption(f"Alert threshold: {SYSTEM.alert_threshold_pct:.0f}% (config.py)")\n'
    '\n'
    '    scenario = None\n'
    '    if offline:\n'
    '        st.sidebar.subheader("Scenario simulator")\n'
    '        n_hotspots = st.sidebar.slider("Active fire hotspots", 0, 40, 8)\n'
    '        temp_c = st.sidebar.slider("Avg temperature (C)", 15, 48, 32)\n'
    '        wind_speed_ms = st.sidebar.slider("Avg wind speed (m/s)", 0.0, 20.0, 5.0)\n'
    '        humidity_pct = st.sidebar.slider("Avg humidity (%)", 0, 100, 40)\n'
    '        scenario = {\n'
    '            "n_hotspots": n_hotspots,\n'
    '            "temp_c": temp_c,\n'
    '            "wind_speed_ms": wind_speed_ms,\n'
    '            "humidity_pct": humidity_pct,\n'
    '        }\n'
    '\n'
    '    if "twin" not in st.session_state or st.sidebar.button("Refresh data"):\n'
    '        with st.spinner("Refreshing digital twin state..."):\n'
    '            twin = get_twin(offline, scenario)\n'
    '            twin.refresh()\n'
    '        st.session_state["twin"] = twin\n'
    '        st.session_state["ca_history"] = None',
    "app.py: sidebar scenario sliders",
)

print("\nPatch script finished.")