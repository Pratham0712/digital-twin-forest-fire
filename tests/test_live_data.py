"""Real-API integration (NASA FIRMS, OpenWeatherMap, Google Maps config) with
mocked HTTP: status reporting, cached fallback, no fabricated observations,
no key leakage, real-vs-scenario labelling, and the map clean-up."""
import re
from pathlib import Path

import pandas as pd
import pytest
import requests

from config.config import API
from src.data_ingestion import ingestion_module as im
from src.data_ingestion import live_point
from src.data_ingestion.firms_client import FIRMSClient
from src.data_ingestion.weather_client import WeatherClient

ROOT = Path(__file__).resolve().parents[1]
FAKE_FIRMS = "FIRMSKEY0123456789abcdef"
FAKE_OWM = "OWMKEY0123456789abcdef0"
CSV = ("latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,confidence,version,bright_t31,frp,daynight\n"
       "11.70,76.60,330.1,0.4,0.4,2026-10-07,0620,N,n,2.0NRT,290.1,5.4,D\n"
       "11.72,76.62,335.0,0.4,0.4,2026-10-07,0748,N,h,2.0NRT,291.0,12.8,D\n")
OWM_JSON = {"main": {"temp": 29.4, "humidity": 67, "pressure": 1009}, "wind": {"speed": 3.8, "deg": 240},
            "clouds": {"all": 40}, "weather": [{"main": "Clouds", "description": "scattered clouds"}],
            "dt": 1791300000, "name": "Gundlupet"}


class Resp:
    def __init__(self, status=200, text="", js=None):
        self.status_code, self.text, self._js = status, text, js

    def json(self):
        return self._js

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error for url: https://x/?appid={FAKE_OWM}")


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setattr(API, "firms_map_key", FAKE_FIRMS)
    monkeypatch.setattr(API, "owm_api_key", FAKE_OWM)
    live_point.clear_caches()
    im._LIVE_CACHE.clear()
    yield
    live_point.clear_caches()
    im._LIVE_CACHE.clear()


def _no_key(obj):
    s = repr(obj)
    assert FAKE_FIRMS not in s and FAKE_OWM not in s


# ── configuration / security ─────────────────────────────────────────────── #

def test_env_names_only_and_env_ignored():
    ex = (ROOT / ".env.example").read_text()
    for k in ("FIRMS_MAP_KEY", "OWM_API_KEY", "GOOGLE_MAPS_API_KEY"):
        assert re.search(rf"^{k}=\s*$", ex, re.M), k                 # name only, no value
    assert re.search(r"^\.env$", (ROOT / ".gitignore").read_text(), re.M)


def test_keys_come_from_the_environment(monkeypatch):
    src = (ROOT / "config" / "config.py").read_text()
    for k in ("FIRMS_MAP_KEY", "OWM_API_KEY", "GOOGLE_MAPS_API_KEY"):
        assert f'os.getenv("{k}"' in src                                   # read from env / .env, never hard-coded
    assert "load_dotenv(ENV_FILE)" in src and 'ENV_FILE = BASE_DIR / ".env"' in src
    from src.dashboard.geo_spread import google_maps_key
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "gkey")
    assert google_maps_key() == "gkey"
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY")
    assert google_maps_key() == ""


# ── NASA FIRMS client ────────────────────────────────────────────────────── #

def test_firms_success_status():
    c = FIRMSClient(FAKE_FIRMS, API.firms_base_url, "VIIRS_SNPP_NRT")
    c.session.get = lambda url, timeout: Resp(200, CSV)
    df = c.fetch_hotspots(11.5, 76.4, 11.9, 76.8, 2)
    assert len(df) == 2 and c.last_status["ok"] and c.last_status["n"] == 2
    assert c.last_status["latest_acq_utc"] == "2026-10-07T07:48Z"
    _no_key(c.last_status)


def test_firms_failure_is_reported_not_faked():
    c = FIRMSClient(FAKE_FIRMS, API.firms_base_url, "VIIRS_SNPP_NRT")

    def boom(url, timeout):
        raise requests.ConnectionError(f"cannot reach {url}")
    c.session.get = boom
    df = c.fetch_hotspots(11.5, 76.4, 11.9, 76.8, 2)
    assert df.empty and c.last_status["ok"] is False and "FIRMS request failed" in c.last_status["error"]
    _no_key(c.last_status)
    c.session.get = lambda url, timeout: Resp(200, "Invalid MAP_KEY.")
    assert c.fetch_hotspots(11.5, 76.4, 11.9, 76.8, 1).empty and not c.last_status["ok"]


def test_firms_missing_key():
    c = FIRMSClient("", API.firms_base_url)
    assert c.fetch_hotspots(11.5, 76.4, 11.9, 76.8, 1).empty
    assert c.last_status["ok"] is False and "FIRMS_MAP_KEY" in c.last_status["error"]


# ── OpenWeatherMap client ────────────────────────────────────────────────── #

def test_owm_point_success_and_failures():
    c = WeatherClient(FAKE_OWM, API.owm_base_url)
    c.session.get = lambda url, params, timeout: Resp(200, js=OWM_JSON)
    rec = c.fetch_point(11.66, 76.63)
    assert rec["temperature_c"] == 29.4 and rec["wind_deg"] == 240 and rec["pressure_hpa"] == 1009
    assert rec["weather_description"] == "scattered clouds" and rec["observed_at"].startswith("2026")
    c.session.get = lambda url, params, timeout: Resp(401)
    assert c.fetch_point(11.66, 76.63) is None and "401" in c.last_error
    c.session.get = lambda url, params, timeout: Resp(500)
    assert c.fetch_point(11.66, 76.63) is None
    _no_key(c.last_error)


# ── selected-location observations (live_point) ─────────────────────────── #

class FakeOWM:
    def __init__(self, ok=True):
        self.ok, self.calls, self.last_error = ok, 0, None

    def fetch_point(self, lat, lon):
        self.calls += 1
        if self.ok:
            return {"temperature_c": 29.4, "humidity_pct": 67, "wind_speed_ms": 3.8, "wind_deg": 240,
                    "pressure_hpa": 1009, "weather_main": "Clouds", "observed_at": "2026-10-07T06:20:00+00:00"}
        self.last_error = "OpenWeatherMap request failed: timeout"
        return None


def test_point_weather_live_cache_and_failure():
    ok = FakeOWM()
    w, s = live_point.point_weather(11.6667, 76.6333, client=ok)
    assert s["mode"] == "live" and w["temperature_c"] == 29.4 and s["fetched_utc"]
    live_point.point_weather(11.6667, 76.6333, client=ok)              # within TTL: no new request
    assert ok.calls == 1
    bad = FakeOWM(ok=False)
    w2, s2 = live_point.point_weather(11.6667, 76.6333, force=True, client=bad)
    assert s2["mode"] == "cached" and w2["temperature_c"] == 29.4 and "timeout" in s2["error"]
    w3, s3 = live_point.point_weather(12.5, 75.5, client=bad)        # never fetched here: nothing invented
    assert w3 is None and s3["mode"] == "error"


def test_point_weather_not_configured(monkeypatch):
    monkeypatch.setattr(API, "owm_api_key", "")
    w, s = live_point.point_weather(11.6, 76.6)
    assert w is None and s["mode"] == "not_configured"


class FakeFIRMS:
    def __init__(self, df=None, ok=True):
        self.df, self.ok, self.calls = df, ok, 0
        self.last_status = {}

    def fetch_hotspots(self, s, w, n, e, days):
        self.calls += 1
        if self.ok:
            self.last_status = {"ok": True, "n": len(self.df), "fetched_utc": "2026-10-07T06:30:00+00:00",
                                "latest_acq_utc": "2026-10-07T06:20Z", "error": None}
            return self.df
        self.last_status = {"ok": False, "error": "FIRMS request failed: timeout", "fetched_utc": "x"}
        return pd.DataFrame()


def test_area_hotspots_live_empty_and_failure():
    df = pd.read_csv(pd.io.common.StringIO(CSV))
    df["acq_datetime"] = pd.to_datetime(df.acq_date + " " + df.acq_time.astype(str).str.zfill(4))
    d, s = live_point.area_hotspots(11.6667, 76.6333, client=FakeFIRMS(df))
    assert s["mode"] == "live" and s["n"] == 2 and len(d) == 2 and s["box"]["south"] < 11.6667 < s["box"]["north"]
    marks = live_point.hotspot_markers(d)
    assert marks[0]["date"].endswith("UTC") and marks[1]["frp"] == 12.8
    d2, s2 = live_point.area_hotspots(11.6667, 76.6333, force=True, client=FakeFIRMS(ok=False))
    assert s2["mode"] == "cached" and len(d2) == 2                         # last real data, labelled cached
    d3, s3 = live_point.area_hotspots(13.0, 75.0, client=FakeFIRMS(ok=False))
    assert s3["mode"] == "error" and d3.empty                              # no synthetic replacement
    d4, s4 = live_point.area_hotspots(14.0, 75.0, client=FakeFIRMS(df.iloc[:0]))
    assert s4["mode"] == "live" and s4["n"] == 0 and "No NASA FIRMS fire detections" in s4["label"]


# ── ingestion: what each feed actually delivered ─────────────────────────── #

def test_ingestion_status_live_cached_error():
    from config.config import REGION
    m = im.DataIngestionModule(offline=False, region=REGION)
    m.firms.session.get = lambda url, timeout: Resp(200, CSV)
    assert len(m.fetch_fire_hotspots()) == 2 and m.source_status["firms"]["mode"] == "live"

    def boom(url, timeout):
        raise requests.ConnectionError("down")
    m.firms.session.get = boom
    df = m.fetch_fire_hotspots()
    assert m.source_status["firms"]["mode"] == "cached" and len(df) == 2
    im._LIVE_CACHE.clear()
    df = m.fetch_fire_hotspots()
    assert m.source_status["firms"]["mode"] == "error" and df.empty           # no synthetic fallback
    # weather: every point fails -> error, empty (risk then runs without weather)
    m.weather.fetch_point = lambda lat, lon: None
    assert m.fetch_weather().empty and m.source_status["weather"]["mode"] == "error"
    _no_key(m.source_status)


def test_offline_and_unconfigured_are_never_live(monkeypatch):
    from config.config import REGION
    m = im.DataIngestionModule(offline=True, region=REGION)
    m.fetch_fire_hotspots()
    m.fetch_weather()
    assert m.source_status["firms"]["mode"] == "demo" and m.source_status["weather"]["mode"] == "demo"
    monkeypatch.setattr(API, "firms_map_key", "")
    monkeypatch.setattr(API, "owm_api_key", "")
    m2 = im.DataIngestionModule(offline=False, region=REGION)
    assert m2.fetch_fire_hotspots().empty and m2.fetch_weather().empty          # no synthetic data in real mode
    assert m2.source_status["firms"]["mode"] == "not_configured"
    assert m2.source_status["weather"]["mode"] == "not_configured"


# ── labels: LIVE only for real successful data; DEMO only when demo mode is ON ── #

class _Twin:
    def __init__(self, firms, weather, offline=False):
        self.offline = offline
        self.ingestion = type("I", (), {"source_status": {"firms": {"mode": firms, "n": 3},
                                                          "weather": {"mode": weather}}})()


@pytest.mark.parametrize("f,w,overall,badge", [
    ("live", "live", "live", "LIVE · REAL DATA"),
    ("live", "error", "error", "REAL DATA · API UNAVAILABLE"),
    ("error", "error", "error", "REAL DATA · API UNAVAILABLE"),
    ("cached", "live", "cached", "REAL DATA · CACHED"),
    ("not_configured", "live", "not_configured", "REAL DATA · KEY MISSING"),
    ("not_configured", "not_configured", "not_configured", "REAL DATA · KEY MISSING")])
def test_real_mode_is_never_labelled_demo(f, w, overall, badge):
    from src.dashboard.ui.global_ticker import data_badge, feed_status, feed_text
    fs = feed_status(_Twin(f, w), demo=False)
    assert fs["overall"] == overall and data_badge(fs)[1] == badge
    assert "DEMO" not in data_badge(fs)[1] and not fs["demo"]
    if f != "live":
        assert not feed_text("firms", fs).startswith("LIVE")


def test_demo_mode_is_labelled_demo():
    from src.dashboard.ui.global_ticker import data_badge, feed_status, feed_text
    fs = feed_status(_Twin("demo", "demo", offline=True), demo=True)
    assert fs["overall"] == "demo" and data_badge(fs)[1] == "DEMO / OFFLINE MODE"
    assert feed_text("firms", fs).startswith("DEMO")
    # a demo twin while the switch is already OFF is "connecting", never LIVE or DEMO
    fs2 = feed_status(_Twin("demo", "demo", offline=True), demo=False)
    assert fs2["overall"] == "pending" and "DEMO" not in data_badge(fs2)[1]


def test_default_mode_follows_configured_keys(monkeypatch):
    from src.dashboard.app_state import default_demo_mode
    assert default_demo_mode() is False                                   # keys set -> real data by default
    monkeypatch.setattr(API, "firms_map_key", "")
    monkeypatch.setattr(API, "owm_api_key", "")
    assert default_demo_mode() is True


def test_single_authoritative_demo_state():
    """Only app_state.py owns the demo / offline default; nobody else derives
    one from the environment or keeps a second flag."""
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        if path.name == "app_state.py":
            continue
        txt = path.read_text(encoding="utf-8")
        if re.search(r'_pref\(\s*"offline_mode"|_pref_offline_mode|DEMO_MODE\s*=\s*True|setdefault\(\s*"demo_mode"', txt):
            offenders.append(path.name)
        if re.search(r'not bool\(os\.getenv\("FIRMS_MAP_KEY"\)\)', txt):
            offenders.append(path.name + " (env-derived default)")
    assert offenders == []


def test_env_file_diagnostics_name_problems_without_values(tmp_path):
    from config.config import env_file_problems
    secret = "abcdef0123456789abcdef0123456789"
    env = tmp_path / ".env"
    env.write_bytes(f"FIRMS MAP_KEY={secret}\r\nOWM API KEY={secret}\r\nGOOGLE_MAPS_API_KEY=AIzaX\n".encode())
    probs = env_file_problems(env)
    assert len(probs) == 2 and "FIRMS_MAP_KEY" in probs[0] and "line 2" in probs[1] and "OWM_API_KEY" in probs[1]
    assert all(secret not in p for p in probs)
    from dotenv import dotenv_values
    assert "FIRMS_MAP_KEY" not in dotenv_values(env)                    # why the key was invisible


def test_keys_never_logged(caplog):
    c = FIRMSClient(FAKE_FIRMS, API.firms_base_url)

    def boom(url, timeout):
        raise requests.ConnectionError(f"cannot reach {url}")
    c.session.get = boom
    c.fetch_hotspots(11.5, 76.4, 11.9, 76.8, 1)
    w = WeatherClient(FAKE_OWM, API.owm_base_url)
    w.session.get = lambda url, params, timeout: Resp(500)
    w.fetch_point(11.6, 76.6)
    assert FAKE_FIRMS not in caplog.text and FAKE_OWM not in caplog.text and "***" in caplog.text


def test_map_hotspots_never_synthetic_outside_demo():
    from src.dashboard.ui.live_panel import map_hotspots
    for mode in ("error", "not_configured", "demo"):
        m, kind, _ = map_hotspots(pd.DataFrame({"latitude": [11.7], "longitude": [76.6]}), {"mode": mode})
        assert m == [] and kind == "observed"


def test_location_search_moves_the_focus():
    from src.dashboard.geo_spread import apply_map_event, default_setup
    setup = {"location": {"name": "Bandipur Tiger Reserve", "lat": 11.6667, "lon": 76.6333, "source": "preset"},
             "placement": "Upwind edge", "ignition_points": [], "cell_m": 25}
    ev = {"kind": "place", "name": "Coorg", "lat": 12.4244, "lon": 75.7382, "source": "google_geocoding",
          "address": "Kodagu, Karnataka, India"}
    assert apply_map_event(ev, setup, None)
    assert setup["location"]["lat"] == 12.4244 and setup["location"]["source"] == "google_geocoding"
    assert live_point.firms_box(12.4244, 75.7382)["south"] < 12.4244


# ── maps: Command Center on Google satellite, What-If without the extra map ── #

def test_region_payload(monkeypatch):
    from config.config import REGION
    from src.dashboard.dashboard_common import get_twin
    from src.dashboard.geo_fire_map import region_payload
    from src.dashboard.ui.command_center import region_hotspots
    monkeypatch.setattr(API, "firms_map_key", "")                      # no keys -> demo mode by default
    monkeypatch.setattr(API, "owm_api_key", "")
    t = get_twin(offline=True, region=REGION)
    t.refresh()
    marks, kind, summ = region_hotspots(t)
    assert kind == "synthetic" and "synthetic" in summ                 # demo mode is labelled
    p = region_payload(t, marks, kind, summ, "note", "K", "")
    n = len(t.current_snapshot.processed_grid)
    assert p["mode"] == "region" and len(p["zones"]["risk"]) == n and len(p["zones"]["sev"]) == n
    assert p["mapId"] == "DEMO_MAP_ID" and p["hotspotKind"] == "synthetic"
    # real-data mode (keys set, real twin) with a failed FIRMS request: nothing shown, not synthetic
    monkeypatch.setattr(API, "firms_map_key", FAKE_FIRMS)
    t.offline = False
    t.ingestion.source_status["firms"] = {"mode": "error"}
    assert region_hotspots(t)[0] == [] and region_hotspots(t)[1] == "observed"


@pytest.fixture
def app(tmp_path, monkeypatch):
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}")
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "TESTKEY")
    at = AppTest.from_file(str(ROOT / "src" / "dashboard" / "app.py"), default_timeout=240)
    at.session_state["auth_user"] = "admin"
    at.session_state["auth_role"] = "admin"
    at.session_state["_pref_offline_mode"] = True
    at.run()
    assert not at.exception
    return at


def test_command_center_uses_google_map_and_what_if_has_no_extra_map(app):
    md = "\n".join(m.value for m in app.markdown)
    assert "Regional risk map · Google satellite" in md
    assert not app.get("plotly_chart") or all("carto" not in str(c.proto) for c in app.get("plotly_chart"))
    assert "NASA FIRMS" in md and "OPENWEATHERMAP" in md and "GOOGLE MAPS" in md and "RISK MODEL" in md
    app.switch_page("pages/1_What_If_Simulator.py").run()
    assert not app.exception
    md = "\n".join(m.value for m in app.markdown)
    assert "Regional risk map" not in md
    assert "Real-world observations at the selected location" in md and "SCENARIO INPUT" in md


def _app_with(monkeypatch, tmp_path, demo, firms_ok=True):
    """Command Center with fake FIRMS / OWM HTTP (no network) and the sidebar
    switch in the given position."""
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}")
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "TESTKEY")
    calls = {"firms": 0, "owm": 0}
    from src.dashboard import dashboard_common as dc
    dc._SHARED_TWINS.clear()                                             # no twin reuse across tests

    def fake_firms(self, min_lat, min_lon, max_lat, max_lon, day_range=1):
        calls["firms"] += 1
        if not firms_ok:
            self._errors = ["FIRMS request failed: simulated outage"]
            return self._finish(None)
        self._errors = []
        df = pd.read_csv(pd.io.common.StringIO(CSV))
        df["acq_date"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
        return self._finish(df)

    def fake_point(self, lat, lon):
        calls["owm"] += 1
        return {"latitude": lat, "longitude": lon, "temperature_c": 29.0, "humidity_pct": 60, "pressure_hpa": 1008,
                "wind_speed_ms": 3.0, "wind_deg": 240, "precipitation_mm": 0, "clouds_pct": 20,
                "weather_main": "Clouds", "fetched_at": "2026-10-07T08:00:00+00:00"}
    monkeypatch.setattr(FIRMSClient, "fetch_hotspots", fake_firms)
    monkeypatch.setattr(WeatherClient, "fetch_point", fake_point)
    at = AppTest.from_file(str(ROOT / "src" / "dashboard" / "app.py"), default_timeout=240)
    at.session_state["auth_user"] = "admin"
    at.session_state["auth_role"] = "admin"
    at.session_state["_pref_offline_mode"] = demo
    at.run()
    assert not at.exception
    return at, calls


def _ticker(at) -> str:
    return next(m.value for m in at.markdown if 'class="gt"' in m.value)


def test_sidebar_off_means_real_data(monkeypatch, tmp_path):
    at, calls = _app_with(monkeypatch, tmp_path, demo=False)
    assert [t.value for t in at.sidebar.toggle if t.label == "Offline / demo mode"] == [False]
    t = _ticker(at)
    assert "DEMO / OFFLINE MODE" not in t and "LIVE · REAL DATA" in t
    assert "NASA FIRMS" in t and "OPENWEATHERMAP" in t and "NOT CONFIGURED" not in t
    assert calls["firms"] >= 1 and calls["owm"] >= 1                      # real requests were attempted
    md = "\n".join(m.value for m in at.markdown)
    assert "Real data (sidebar toggle OFF)" in md


def test_sidebar_off_with_failing_firms_is_unavailable_not_demo(monkeypatch, tmp_path):
    at, _ = _app_with(monkeypatch, tmp_path, demo=False, firms_ok=False)
    t = _ticker(at)
    assert "DEMO / OFFLINE MODE" not in t and "REAL DATA · API UNAVAILABLE" in t and "UNAVAILABLE" in t


def test_sidebar_off_with_missing_key_is_not_configured_not_demo(monkeypatch, tmp_path):
    monkeypatch.setattr(API, "firms_map_key", "")
    at, _ = _app_with(monkeypatch, tmp_path, demo=False)
    t = _ticker(at)
    assert "DEMO / OFFLINE MODE" not in t and "KEY MISSING" in t and "NOT CONFIGURED" in t


def test_sidebar_on_means_demo(monkeypatch, tmp_path):
    at, calls = _app_with(monkeypatch, tmp_path, demo=True)
    assert [t.value for t in at.sidebar.toggle if t.label == "Offline / demo mode"] == [True]
    t = _ticker(at)
    assert "DEMO / OFFLINE MODE" in t and "LIVE" not in t.split("DEMO / OFFLINE MODE")[1][:40]
    assert calls["firms"] == 0 and calls["owm"] == 0                      # demo mode does not call the APIs
    # flipping the switch OFF changes every reader at once
    at.sidebar.toggle[0].set_value(False).run()
    assert "DEMO / OFFLINE MODE" not in _ticker(at) and at.session_state["_pref_offline_mode"] is False
    # ... and back ON, then OFF again: every flip sticks (no lost clicks)
    for want in (True, False, True):
        at.sidebar.toggle[0].set_value(want).run()
        assert at.session_state["_pref_offline_mode"] is want and at.sidebar.toggle[0].value is want
        assert ("DEMO / OFFLINE MODE" in _ticker(at)) is want
