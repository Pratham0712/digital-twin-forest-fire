"""The Offline / demo mode switch is a global setting: it persists across
reruns and page navigation and changes only when the user flips it.

Regression for the dangerous case: demo mode ON, navigate to another page, the
toggle silently fell back to OFF and the app started real FIRMS /
OpenWeatherMap requests. HTTP is stubbed (and the conftest refuses all network),
so the counters below show exactly which requests the app attempted."""
import pytest

pytest.importorskip("streamlit.testing.v1")
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from config.config import API
from src.dashboard.app_state import DEMO_STATE_KEY
from src.data_ingestion.firms_client import FIRMSClient
from src.data_ingestion.weather_client import WeatherClient

DASH = Path(__file__).resolve().parents[1] / "src" / "dashboard"
TOGGLE = "Offline / demo mode"
TOUR = ["pages/1_What_If_Simulator.py", "pages/2_Spread_Simulation.py", "pages/4_AI_Situation_Briefing.py", "app.py"]
ALL_PAGES = ["pages/1_What_If_Simulator.py", "pages/2_Spread_Simulation.py", "pages/3_Historical_Time_Machine.py",
             "pages/4_AI_Situation_Briefing.py", "pages/5_Model_Insights.py", "pages/6_Admin.py",
             "pages/7_Activity_Log.py", "app.py"]


@pytest.fixture
def app(tmp_path, monkeypatch):
    """Keys configured (fake) so the default is real-data mode; FIRMS / OWM
    replaced by counting stubs."""
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
    calls = {"firms": 0, "owm": 0}

    def fake_firms(self, *a, **k):
        calls["firms"] += 1
        self._errors = []
        return self._finish(pd.DataFrame(columns=FIRMSClient.EXPECTED_COLUMNS))

    def fake_point(self, lat, lon):
        calls["owm"] += 1
        return {"latitude": lat, "longitude": lon, "temperature_c": 29.0, "humidity_pct": 60, "pressure_hpa": 1008,
                "wind_speed_ms": 3.0, "wind_deg": 240, "precipitation_mm": 0, "clouds_pct": 20,
                "weather_main": "Clouds", "fetched_at": "2026-10-07T08:00:00+00:00"}
    monkeypatch.setattr(FIRMSClient, "fetch_hotspots", fake_firms)
    def fake_forecast(self, lat, lon, horizon_hours=2.0):
        calls["owm"] += 1
        return None
    monkeypatch.setattr(WeatherClient, "fetch_point", fake_point)
    monkeypatch.setattr(WeatherClient, "fetch_forecast_point", fake_forecast)
    at = AppTest.from_file(str(DASH / "app.py"), default_timeout=240)
    at.session_state["auth_user"] = "admin"
    at.session_state["auth_role"] = "admin"
    at.run()
    assert not at.exception
    at.calls = calls
    return at


def _demo(at) -> bool:
    return at.session_state[DEMO_STATE_KEY] if DEMO_STATE_KEY in at.session_state else None


def _toggle(at):
    vals = [t.value for t in at.sidebar.toggle if t.label == TOGGLE]
    return vals[0] if vals else None                      # What-If / Historical do not draw the toggle


def _tag(at) -> str:
    tick = next(m.value for m in at.markdown if 'class="gt"' in m.value)
    return "DEMO" if "DEMO / OFFLINE MODE" in tick else "REAL"


def _flip(at, value: bool):
    [t for t in at.sidebar.toggle if t.label == TOGGLE][0].set_value(value).run()
    assert not at.exception


def _visit(at, page, expected: bool):
    """Render a page and check every reader agrees with the authoritative value."""
    at.switch_page(page).run()
    assert not at.exception, page
    assert _demo(at) is expected, page                                  # rendering did not change it
    assert _toggle(at) in (expected, None), page                        # sidebar shows the same value
    assert _tag(at) == ("DEMO" if expected else "REAL"), page           # ticker reads the same value


def test_default_is_real_data_when_keys_are_configured(app):
    assert _toggle(app) is False and _tag(app) == "REAL"
    assert app.calls["firms"] >= 1 and app.calls["owm"] >= 1            # real-data mode requests data


def test_demo_on_persists_across_reruns_and_navigation(app):
    _flip(app, True)
    assert _demo(app) is True and _toggle(app) is True and _tag(app) == "DEMO"
    app.run()                                                           # plain rerun
    assert _demo(app) is True and _toggle(app) is True
    for page in TOUR:                                                   # What-If -> Spread -> AI -> Command Center
        _visit(app, page, True)


def test_demo_on_makes_no_api_requests_after_navigation(app):
    """The dangerous scenario: ON, navigate, must still be ON and request nothing."""
    _flip(app, True)
    app.calls.update(firms=0, owm=0)
    for page in ALL_PAGES:
        _visit(app, page, True)
    assert app.calls == {"firms": 0, "owm": 0}


def test_demo_off_persists_across_navigation(app):
    _flip(app, True)
    _flip(app, False)
    assert _demo(app) is False and _toggle(app) is False and _tag(app) == "REAL"
    for page in TOUR + TOUR:
        _visit(app, page, False)


def test_every_flip_sticks_and_only_the_user_changes_it(app):
    for want in (True, False, True, True, False):
        _flip(app, want)
        assert _demo(app) is want
        for page in ("pages/2_Spread_Simulation.py", "pages/5_Model_Insights.py", "app.py"):
            _visit(app, page, want)


def test_single_authoritative_value():
    """The widget key mirrors the one authoritative key; nothing else stores a mode."""
    src = (DASH / "dashboard_common.py").read_text(encoding="utf-8")
    assert "st.session_state[DEMO_TOGGLE_KEY] = offline_default" in src           # widget seeded from state
    assert "on_change=lambda: set_demo_mode(" in src                               # written only on user change
    assert "set_demo_mode(offline)" not in src                                     # no write-back on render
