"""What-If Simulator: LIVE REAL-WORLD and LIVE DATA -> WHAT-IF modes.

Every OpenWeatherMap / NASA FIRMS request is replaced by an in-process stub
(the conftest also refuses all network), with FAKE keys so the app runs in
real-data mode. Nothing here reaches an external API or uses quota."""
import socket

import numpy as np
import pandas as pd
import pytest
import requests

pytest.importorskip("streamlit.testing.v1")
from pathlib import Path

from streamlit.testing.v1 import AppTest

from config.config import API
from src.dashboard import live_modes as lm
from src.data_ingestion.firms_client import FIRMSClient
from src.data_ingestion.weather_client import WeatherClient
from src.simulation.fuel_map import FUEL, WATER, LandCover
from src.simulation.local_spread import FocusArea, domain_for, run_local_spread
from src.simulation.observed_ignition import classify_detections
from tests.conftest import ExternalNetworkBlocked

DASH = Path(__file__).resolve().parents[1] / "src" / "dashboard"
WHATIF = "pages/1_What_If_Simulator.py"
SPREAD = "pages/2_Spread_Simulation.py"
BANDIPUR = (11.6667, 76.6333)                      # default What-If location (preset)
OPEN = "Apply Scenario & Open Spread Simulation"
INPUT_LABEL = {"temp_c": "Temperature", "humidity_pct": "Relative humidity", "wind_speed_ms": "Wind speed",
               "wind_from_deg": "Wind direction"}


def _weather(**kw):
    w = {"temperature_c": 22.73, "humidity_pct": 64, "pressure_hpa": 1009, "wind_speed_ms": 3.09, "wind_deg": 240,
         "precipitation_mm": 0.0, "clouds_pct": 20, "weather_main": "Clouds", "weather_description": "few clouds",
         "observed_at": "2026-10-07T08:00:00+00:00", "place_name": "Bandipur", "missing_fields": []}
    w.update(kw)
    return w


def _detection(lat, lon, frp=12.5):
    return {"latitude": lat, "longitude": lon, "bright_ti4": 340.1, "scan": 0.39, "track": 0.36,
            "acq_date": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"), "acq_time": "0812", "satellite": "N",
            "confidence": "n", "version": "2.0NRT", "bright_t31": 291.2, "frp": frp, "daynight": "D"}


@pytest.fixture
def live(tmp_path, monkeypatch):
    """Factory: signed-in app in REAL-DATA mode (fake keys) with stubbed APIs.
    `feed` controls what the stubs return; `calls` counts the requests."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}")
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "TESTKEY")
    monkeypatch.setattr(API, "firms_map_key", "FAKE_FIRMS_KEY")
    monkeypatch.setattr(API, "owm_api_key", "FAKE_OWM_KEY")
    from src.dashboard import dashboard_common as dc
    from src.data_ingestion import ingestion_module as im
    from src.data_ingestion import live_point
    dc._SHARED_TWINS.clear()
    im._LIVE_CACHE.clear()
    live_point.clear_caches()
    feed = {"weather": _weather(), "detections": [], "firms_ok": True, "owm_ok": True}
    calls = {"firms": 0, "owm": 0}

    def fake_firms(self, *a, **k):
        calls["firms"] += 1
        if not feed["firms_ok"]:
            self._errors = ["FIRMS request failed: simulated outage"]
            return self._finish(None)
        self._errors = []
        return self._finish(pd.DataFrame(feed["detections"]) if feed["detections"]
                            else pd.DataFrame(columns=FIRMSClient.EXPECTED_COLUMNS))

    def fake_point(self, lat, lon):
        calls["owm"] += 1
        if not feed["owm_ok"]:
            self.last_error = "OpenWeatherMap request failed: simulated outage"
            return None
        return {"latitude": lat, "longitude": lon, **feed["weather"],
                "fetched_at": pd.Timestamp.now(tz="UTC").isoformat()}

    monkeypatch.setattr(FIRMSClient, "fetch_hotspots", fake_firms)
    monkeypatch.setattr(WeatherClient, "fetch_point", fake_point)
    monkeypatch.setattr(WeatherClient, "fetch_forecast_point", lambda self, *a, **k: None)

    def make(page=WHATIF, demo=False):
        at = AppTest.from_file(str(DASH / "app.py"), default_timeout=300)
        at.session_state["auth_user"] = "admin"
        at.session_state["auth_role"] = "admin"
        at.session_state["_pref_offline_mode"] = demo
        at.run()
        assert not at.exception
        if page:
            at.switch_page(page).run()
            assert not at.exception
        at.feed, at.calls = feed, calls
        return at
    return make


def _md(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def _slider(at, key):
    return next(s for s in at.slider if s.label.startswith(INPUT_LABEL[key]) and "[" in s.label)


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def _whatif(at):
    at.button(key="wl_mode_whatif").click().run()
    assert not at.exception
    return at


def _refresh(at):
    _button(at, "Refresh live data").click().run()
    assert not at.exception


def _open_and_run(at, run_label):
    _button(at, OPEN).click().run()
    assert not at.exception
    at.switch_page(SPREAD).run()
    assert not at.exception
    _button(at, run_label).click().run()
    assert not at.exception
    return at.session_state["simulation_config"], at.session_state["applied_geo_result"]


def _domain(cell=25.0, size=500.0, duration=60.0):
    f = FocusArea(name="t", lat=BANDIPUR[0], lon=BANDIPUR[1], width_m=size, height_m=size, cell_m=cell)
    return f, domain_for(f, duration)


COND = {"ffmc": 92.0, "ndvi": 0.6, "bui": 60.0, "risk_score": 0.9, "zone_id": 1, "fwi": 40.0}


# ── 1-4  LIVE REAL-WORLD ─────────────────────────────────────────────────── #

def test_01_live_mode_loads_openweathermap(live):
    at = live()
    assert at.calls["owm"] >= 1
    md = _md(at)
    assert "REAL WEATHER · OPENWEATHERMAP" in md and "22.7 °C" in md and "LIVE" in md
    assert "[ LIVE REAL-WORLD ]" in md and "lm-chip on'>[ LIVE REAL-WORLD ]" in md


def test_02_live_mode_loads_nasa_firms(live):
    at = live()
    assert at.calls["firms"] >= 1
    md = _md(at)
    assert "NASA FIRMS · SATELLITE FIRE DETECTIONS" in md and "Last FIRMS fetch:" in "\n".join(
        c.value for c in at.caption)
    assert at.session_state["_wi_twin"].ingestion.source_status["firms"]["mode"] == "live"


def test_03_live_mode_populates_sliders_from_the_observation(live):
    at = live()
    assert _slider(at, "temp_c").value == pytest.approx(22.7)
    assert _slider(at, "humidity_pct").value == pytest.approx(64)
    assert _slider(at, "wind_speed_ms").value == pytest.approx(3.1)
    assert _slider(at, "wind_from_deg").value == pytest.approx(240)
    pw = at.session_state["_wi_twin"].ingestion.scenario["point_weather"]          # the exact observation drives risk
    assert (pw["temperature_c"], pw["humidity_pct"], pw["wind_speed_ms"], pw["wind_deg"]) == (22.73, 64, 3.09, 240)
    assert at.session_state["_wi_twin"].ingestion.scenario["n_hotspots"] == 0      # no seeded synthetic hotspots


def test_04_live_controls_are_locked(live):
    at = live()
    for k in INPUT_LABEL:
        s = _slider(at, k)
        assert s.disabled and s.label.endswith("[LIVE]")
    md = _md(at)
    assert "LIVE OBSERVATION" in md and "LOCKED TO LIVE DATA" in md
    assert at.selectbox(key="wi_setup_place").disabled                             # no hypothetical placement in LIVE


# ── 5-10  FIRMS -> ignition, no fire without an observation ──────────────── #

def test_05_zero_firms_means_no_fire(live):
    at = live()
    md = _md(at)
    assert "NO FIRE" in md and "No active fire detected. Current conditions do not indicate an observed fire. " \
                                "No fire spread simulated." in md
    cfg, res = _open_and_run(at, "RUN LIVE SIMULATION")
    assert cfg["live"]["ignition"]["source"] == "NONE" and res is None
    md = _md(at)
    assert "NO ACTIVE FIRE DETECTED" in md and ("NASA FIRMS returned no fire detections within the selected "
                                                "area/time window. No fire spread was simulated.") in md


def test_06_valid_firms_detection_becomes_an_observed_ignition(live):
    at = live()
    at.feed["detections"] = [_detection(*BANDIPUR), _detection(11.9, 76.9)]           # 2nd: outside the area
    _refresh(at)
    md = _md(at)
    assert "OBSERVED FIRE DETECTION" in md and ("Active NASA FIRMS fire detection found. Fire spread simulation "
                                                "initialized from observed ignition location(s).") in md
    assert "1 of 2 observation(s) became valid ignitions" in md
    cfg, res = _open_and_run(at, "RUN LIVE SIMULATION")
    ign = cfg["live"]["ignition"]
    assert ign["source"] == "OBSERVED_FIRMS" and ign["kind"] == "OBSERVED" and ign["points"] == [list(BANDIPUR)]
    assert res is not None and res.params["ignition_cells_requested"] == 1            # exactly the observed cell
    r, c = res.domain.cell_of(*BANDIPUR)
    assert res.ignition_step[r, c] == 0
    md = _md(at)
    assert "OBSERVED FIRE — SIMULATION ACTIVE" in md and "SOURCE: NASA FIRMS OBSERVED" in md


def test_07_detection_on_non_fuel_is_skipped():
    f, dom = _domain()
    r, c = dom.cell_of(*BANDIPUR)
    classes = np.full((dom.n_rows, dom.n_cols), FUEL, dtype=np.int8)
    classes[r, c] = WATER
    land = LandCover(classes, "osm", "OpenStreetMap land cover (test)")
    near = (BANDIPUR[0] + 0.0012, BANDIPUR[1])                                         # a fuel cell, inside
    cls = classify_detections(dom, [{"lat": BANDIPUR[0], "lon": BANDIPUR[1]}, {"lat": near[0], "lon": near[1]}], land)
    assert cls["n_in_area"] == 2 and cls["n_non_fuel"] == 1 and cls["n_valid"] == 1
    assert cls["detections"][0]["status"] == "non_fuel" and "water" in cls["detections"][0]["reason"]
    assert cls["valid_points"] == [[near[0], near[1]]]                                 # not moved to another cell
    only_water = classify_detections(dom, [{"lat": BANDIPUR[0], "lon": BANDIPUR[1]}], land)
    spec = lm.ignition_spec(lm.LIVE, {}, only_water)
    assert spec["source"] == "NONE"                                                    # no ignition, no fallback
    st_ = lm.assess(lm.LIVE, "live", only_water, "LOW", spec)
    assert st_["code"] == "no_fire" and any("non-fuel" in n for n in st_["notes"])
    # the strict path never falls back to a placement
    res = run_local_spread(f, COND, 5, 225, n_ignition=0, placement="Map points", duration_minutes=60,
                           ignition_points=[[BANDIPUR[0], BANDIPUR[1]]], land_cover=land, strict_points=True)
    assert res.params["n_ignition"] == 0 and res.final["burned"] == 0 and res.final["burning"] == 0


def test_08_high_temperature_alone_creates_no_fire(live):
    at = live()
    at.feed["weather"] = _weather(temperature_c=46.0, humidity_pct=6, wind_speed_ms=14.0)
    _refresh(at)
    assert _slider(at, "temp_c").value == pytest.approx(46.0)
    cfg, res = _open_and_run(at, "RUN LIVE SIMULATION")
    assert cfg["temp_c"] == 46.0 and cfg["live"]["ignition"]["source"] == "NONE"
    assert res is None                                                                 # nothing simulated
    assert "SIMULATION ACTIVE" not in _md(at)


def test_09_high_fwi_alone_creates_no_fire():
    _, dom = _domain()
    f = FocusArea(name="t", lat=BANDIPUR[0], lon=BANDIPUR[1], width_m=500, height_m=500, cell_m=25)
    extreme = {**COND, "ffmc": 99.0, "bui": 200.0, "fwi": 120.0, "risk_score": 0.99}
    plan = lm.plan_ignition(lm.LIVE, {"duration_min": 60}, f, extreme, [], "live")
    assert plan["ign"]["source"] == "NONE" and plan["cls"]["n_valid"] == 0
    assert lm.assess(lm.LIVE, "live", plan["cls"], "EXTREME", plan["ign"])["code"] == "high_risk_no_fire"


def test_10_high_risk_and_zero_firms_message():
    cls = classify_detections(_domain()[1], [])
    ign = lm.ignition_spec(lm.LIVE, {}, cls)
    pre = lm.assess(lm.LIVE, "live", cls, "HIGH", ign)
    assert pre["title"] == "HIGH FIRE-WEATHER RISK but NO OBSERVED ACTIVE FIRE" and pre["color"] in ("yellow", "orange")
    assert pre["message"] == ("High fire-weather risk detected, but no active NASA FIRMS fire detection is available. "
                              "No observed fire spread simulated.")
    post = lm.result_status(lm.LIVE, "live", cls, "HIGH", ign, ran=False)
    assert post["title"] == "HIGH FIRE-WEATHER RISK — NO ACTIVE FIRE"
    assert post["message"] == ("Current weather conditions indicate elevated fire potential, but NASA FIRMS returned no "
                               "active fire detection. No observed fire spread was simulated.")
    assert ("Fire-weather risk is HIGH, but no active satellite fire detection was found. The system does not simulate "
            "an existing fire without an ignition source.") in post["notes"]
    assert "Current fire-weather conditions: HIGH." in post["notes"]


# ── 11-20  LIVE DATA -> WHAT-IF ──────────────────────────────────────────── #

def test_11_whatif_starts_from_the_real_weather(live):
    at = _whatif(live())
    base = at.session_state[lm.BASE_KEY]
    assert base["values"] == {"temp_c": 22.73, "humidity_pct": 64.0, "wind_speed_ms": 3.09, "wind_from_deg": 240.0}
    assert base["status"] == "live" and base["fetched_utc"]
    assert _slider(at, "temp_c").value == pytest.approx(22.7) and _slider(at, "wind_from_deg").value == 240
    assert any(s.value == "Baseline loaded from current real-world observations." for s in at.success)
    assert "REAL BASELINE" in _md(at) and "WHAT-IF SCENARIO" in _md(at)


def test_12_whatif_controls_are_editable(live):
    at = _whatif(live())
    for k in INPUT_LABEL:
        s = _slider(at, k)
        assert not s.disabled and s.label.endswith("[SCENARIO INPUT]")
    assert "DEMO" not in next(m.value for m in at.markdown if "badge" in m.value and "Scenario Result" in m.value)


@pytest.mark.parametrize("key,new,field,text", [
    ("temp_c", 35.0, "temperature_c", "Temperature — Baseline: 22.7°C / Scenario: 35.0°C / Change: +12.3°C"),   # 13
    ("humidity_pct", 20.0, "humidity_pct", "Relative humidity — Baseline: 64% / Scenario: 20% / Change: -44%"),  # 14
    ("wind_speed_ms", 12.5, "wind_speed_ms", "Wind speed — Baseline: 3.1 m/s / Scenario: 12.5 m/s / Change: +9.4 m/s"),  # 15
    ("wind_from_deg", 90.0, "wind_deg", "Wind direction (blowing FROM) — Baseline: 240° / Scenario: 90° / Change: -150°"),  # 16
], ids=["13_temperature", "14_humidity", "15_wind_speed", "16_wind_direction"])
def test_13_to_16_scenario_change_differs_from_baseline(live, key, new, field, text):
    at = _whatif(live())
    _slider(at, key).set_value(new).run()
    assert not at.exception
    assert text in [c.value for c in at.caption]
    assert at.session_state["_wi_twin"].ingestion.scenario["point_weather"][field] == new   # scenario drives risk
    assert at.session_state[lm.BASE_KEY]["values"][key] != new                              # baseline untouched
    assert "modifies the baseline" in _md(at)


def test_17_baseline_is_retained(live):
    at = _whatif(live())
    before = dict(at.session_state[lm.BASE_KEY]["values"])
    _slider(at, "temp_c").set_value(40.0).run()
    _slider(at, "humidity_pct").set_value(10.0).run()
    at.run()
    assert at.session_state[lm.BASE_KEY]["values"] == before
    at.switch_page("app.py").run()                                                  # leave the page and come back
    at.switch_page(WHATIF).run()
    assert not at.exception and at.session_state[lm.MODE_KEY] == lm.WHATIF
    assert _slider(at, "temp_c").value == pytest.approx(40.0) and _slider(at, "humidity_pct").value == 10
    assert at.session_state[lm.BASE_KEY]["values"] == before
    _button(at, OPEN).click().run()
    live_cfg = at.session_state["simulation_config"]["live"]
    assert live_cfg["baseline"]["values"] == before and live_cfg["scenario_values"]["temp_c"] == 40.0


def test_18_hypothetical_ignition_is_labelled(live):
    at = _whatif(live())
    at.selectbox(key="wi_setup_igsrc").set_value("hypothetical").run()
    md = _md(at)
    assert "WHAT-IF SIMULATION — HYPOTHETICAL IGNITION" in md and "WHAT-IF SIMULATION: Hypothetical ignition applied." in md
    assert "SOURCE: USER HYPOTHETICAL IGNITION" in md
    from src.dashboard.geo_fire_map import setup_payload
    f, _ = _domain()
    p = setup_payload(f, 60, 3, 240, 3, "Centre", [], {"grid": True, "boundary": True}, "K", "",
                      ignition_kind="hypothetical", ignition_label="HYPOTHETICAL · Centre")
    assert p["ignitionKind"] == "hypothetical" and p["preview"] and p["ignitionLabel"].startswith("HYPOTHETICAL")
    html = (DASH / "components" / "fire_map" / "index.html").read_text(encoding="utf-8")
    assert "HYPOTHETICAL IGNITION" in html and "Hypothetical ignition (user)" in html


def test_19_whatif_without_ignition_simulates_no_fire(live):
    at = _whatif(live())
    _slider(at, "temp_c").set_value(46.0).run()
    _slider(at, "humidity_pct").set_value(5.0).run()
    assert at.selectbox(key="wi_setup_igsrc").value == "none"
    md = _md(at)
    assert "WHAT-IF —" in md and "no fire spread will be simulated until an ignition source is provided" in md
    cfg, res = _open_and_run(at, "RUN WHAT-IF SIMULATION")
    assert cfg["live"]["ignition"]["source"] == "NONE" and res is None
    assert "WHAT-IF — " in _md(at) and "SIMULATION — HYPOTHETICAL" not in _md(at)


def test_20_whatif_with_hypothetical_ignition_runs_the_simulation(live):
    at = _whatif(live())
    _slider(at, "temp_c").set_value(40.0).run()
    at.selectbox(key="wi_setup_igsrc").set_value("hypothetical").run()
    cfg, res = _open_and_run(at, "RUN WHAT-IF SIMULATION")
    assert cfg["live"]["ignition"]["source"] == "HYPOTHETICAL_USER" and cfg["setup"]["ignition_source"] == "hypothetical"
    assert res is not None and res.final["burned"] + res.final["burning"] >= 1
    assert res.params["placement"] == cfg["setup"]["placement"]                     # the existing placement path
    md = _md(at)
    assert "WHAT-IF SIMULATION — HYPOTHETICAL IGNITION" in md and "SOURCE: USER HYPOTHETICAL IGNITION" in md


# ── 21-23  unavailable / cached sources ──────────────────────────────────── #

def test_21_firms_unavailable_means_no_synthetic_fire(live, monkeypatch):
    at = live(page=None)
    at.feed["firms_ok"] = False
    at.switch_page(WHATIF).run()
    assert not at.exception
    md = _md(at)
    assert "NASA FIRMS UNAVAILABLE" in md and ("NASA FIRMS data unavailable. Current observed-fire status cannot be "
                                               "confirmed.") in md
    t = at.session_state["_wi_twin"]
    assert t.ingestion.scenario["observed_hotspots"] == [] and t.ingestion.last_hotspots.empty
    cfg, res = _open_and_run(at, "RUN LIVE SIMULATION")
    assert cfg["live"]["detections"] == [] and res is None
    assert "NASA FIRMS UNAVAILABLE — FIRE PRESENCE UNCONFIRMED" in _md(at)


def test_22_owm_unavailable_means_no_fake_weather(live):
    at = live(page=None)
    at.feed["owm_ok"] = False
    at.switch_page(WHATIF).run()
    assert not at.exception
    md = _md(at)
    assert md.count(lm.UNAVAILABLE) >= 4 and "OPENWEATHERMAP UNAVAILABLE" in md
    assert not [s for s in at.slider if "[LIVE]" in s.label]                        # no slider shows an invented value
    assert _button(at, OPEN).disabled
    t = at.session_state["_wi_twin"]
    assert t.ingestion.scenario["point_weather"] is None
    assert t.ingestion.source_status["weather"]["mode"] == "error"


def test_23_cached_data_is_labelled_cached(live):
    at = live()
    at.feed["firms_ok"] = at.feed["owm_ok"] = False
    _refresh(at)                                                                     # live requests fail now
    md = _md(at)
    assert md.count(">CACHED<") >= 2 and "CACHED: the latest request failed" in md
    assert _slider(at, "temp_c").value == pytest.approx(22.7)                        # last real values, labelled


# ── 24-26  refresh and hand-off ──────────────────────────────────────────── #

def test_24_refresh_updates_live_controls(live):
    at = live()
    at.feed["weather"] = _weather(temperature_c=30.0, wind_deg=90)
    _refresh(at)
    assert _slider(at, "temp_c").value == pytest.approx(30.0) and _slider(at, "wind_from_deg").value == 90
    assert at.session_state["_wi_twin"].ingestion.scenario["point_weather"]["temperature_c"] == 30.0


def test_25_refresh_never_overwrites_a_modified_whatif(live):
    at = _whatif(live())
    _slider(at, "temp_c").set_value(35.0).run()
    at.feed["weather"] = _weather(temperature_c=30.0)
    _refresh(at)
    assert _slider(at, "temp_c").value == pytest.approx(35.0)                          # user's value kept
    assert at.session_state[lm.BASE_KEY]["values"]["temp_c"] == 30.0                  # baseline follows the refresh
    assert any("Your WHAT-IF scenario was kept and now differs from the latest baseline" in i.value for i in at.info)
    assert "Temperature — Baseline: 30.0°C / Scenario: 35.0°C / Change: +5.0°C" in [c.value for c in at.caption]
    assert at.session_state["_wi_twin"].ingestion.scenario["point_weather"]["temperature_c"] == 35.0


def test_26_handoff_preserves_mode_weather_firms_ignition_location_duration(live):
    at = live()
    at.feed["detections"] = [_detection(*BANDIPUR)]
    _refresh(at)
    next(s for s in at.selectbox if s.label == "Simulation duration").select("2 h").run()
    _button(at, OPEN).click().run()
    assert not at.exception
    cfg = at.session_state["simulation_config"]
    lv = cfg["live"]
    assert cfg["sim_mode"] == lv["mode"] == "live" and lv["mode_name"] == "LIVE REAL-WORLD"
    assert (cfg["latitude"], cfg["longitude"]) == pytest.approx(BANDIPUR) and cfg["duration_min"] == 120
    assert (cfg["width_m"], cfg["height_m"]) == (500, 500)
    assert lv["weather"]["source"] == "OpenWeatherMap" and lv["weather"]["status"] == "live"
    assert lv["weather"]["fetched_utc"] and lv["weather"]["record"]["temperature_c"] == 22.73
    assert lv["baseline"]["values"]["temp_c"] == 22.73 and lv["scenario_values"]["temp_c"] == 22.73
    assert lv["firms"]["status"] == "live" and lv["firms"]["n"] == 1 and lv["firms"]["fetched_utc"]
    assert lv["detections"][0]["lat"] == BANDIPUR[0] and lv["classification"]["n_valid"] == 1
    assert lv["ignition"]["source"] == "OBSERVED_FIRMS" and lv["ignition"]["kind"] == "OBSERVED"
    assert lv["risk"]["severity"] in ("LOW", "MODERATE", "HIGH", "EXTREME") and "fwi" in lv["risk"]
    at.switch_page(SPREAD).run()
    assert not at.exception
    md = _md(at)
    assert "SOURCE: NASA FIRMS OBSERVED" in md and "Mode: LIVE REAL-WORLD" in "\n".join(c.value for c in at.caption)
    prov = next(df.value for df in at.dataframe if "Item" in df.value.columns)
    items = dict(zip(prov["Item"], prov["Type"]))
    assert items["Mode"] == "LIVE REAL-WORLD" and items["Ignition source"] == "OBSERVED FIRE"
    assert items["NASA FIRMS"] == "NASA FIRMS OBSERVATION" and items["Simulation"] == "SIMULATED"


# ── 27-30  backward compatibility and isolation ──────────────────────────── #

def test_27_offline_demo_mode_still_works(live):
    at = live(demo=True)
    assert at.calls == {"firms": 0, "owm": 0}
    assert [b.disabled for b in at.button if b.label in ("LIVE REAL-WORLD SIMULATION", "LIVE DATA → WHAT-IF")] == [True, True]
    s = next(x for x in at.slider if x.label == "Temperature (°C)")                    # the original demo sliders
    s.set_value(41).run()
    assert not at.exception and at.session_state["_wi_twin"].ingestion.scenario["temp_c"] == 41
    assert "DEMO / OFFLINE MODE" in _md(at) and at.calls == {"firms": 0, "owm": 0}
    _button(at, OPEN).click().run()
    assert at.session_state["simulation_config"]["sim_mode"] == "demo"
    assert at.session_state["simulation_config"]["live"] is None


def test_28_demo_toggle_persistence_is_intact(live):
    at = live(page=None, demo=True)
    at.switch_page(WHATIF).run()
    assert [t.value for t in at.sidebar.toggle if t.label == "Offline / demo mode"] == [True]
    at.sidebar.toggle[0].set_value(False).run()                                         # flip ON the What-If page
    assert at.session_state["_pref_offline_mode"] is False and not at.exception
    for page in (SPREAD, "app.py", WHATIF):
        at.switch_page(page).run()
        assert not at.exception and at.session_state["_pref_offline_mode"] is False
    assert [t.value for t in at.sidebar.toggle if t.label == "Offline / demo mode"] == [False]
    assert lm.current_mode(False) in (lm.LIVE, lm.WHATIF)


def test_29_no_real_external_api_calls(live, _no_external_network):
    at = live()
    at.feed["detections"] = [_detection(*BANDIPUR)]
    _refresh(at)
    _open_and_run(at, "RUN LIVE SIMULATION")
    assert at.calls["firms"] >= 1 and at.calls["owm"] >= 1                            # all served by the stubs
    hosts = set(_no_external_network)
    assert not hosts & {"firms.modaps.eosdis.nasa.gov", "api.openweathermap.org"}


def test_30_network_isolation_is_unchanged(_no_external_network):
    with pytest.raises(ExternalNetworkBlocked):
        requests.get("https://firms.modaps.eosdis.nasa.gov/api/area/csv/x/VIIRS_SNPP_NRT/world/1", timeout=2)
    with pytest.raises(ExternalNetworkBlocked):
        socket.create_connection(("93.184.216.34", 443), timeout=2)
    assert "firms.modaps.eosdis.nasa.gov" in _no_external_network                     # the guard saw and refused it
    assert API.firms_map_key == "" and API.owm_api_key == ""                             # no real keys in tests
    _no_external_network.clear()                                                        # deliberate probe, not a leak
