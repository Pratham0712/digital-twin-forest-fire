"""Headless dashboard checks (Streamlit AppTest): the region chosen on one
page must be the region shown and used on every other page."""
import pytest

pytest.importorskip("streamlit.testing.v1")
from pathlib import Path

from streamlit.testing.v1 import AppTest

DASH = Path(__file__).resolve().parents[1] / "src" / "dashboard"
PAGES = ["pages/1_What_If_Simulator.py", "pages/3_Historical_Time_Machine.py", "pages/2_Spread_Simulation.py",
         "pages/5_Model_Insights.py", "pages/7_Activity_Log.py", "app.py", "pages/4_AI_Situation_Briefing.py"]


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}")
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    at = AppTest.from_file(str(DASH / "app.py"), default_timeout=240)
    at.session_state["auth_user"] = "admin"
    at.session_state["auth_role"] = "admin"
    at.session_state["_pref_offline_mode"] = True       # explicit demo mode: no real API calls
    at.run()
    assert not at.exception
    return at


def _preset(at):
    return [s.value for s in at.sidebar.selectbox if s.label == "Preset"]


def test_region_survives_every_page(app):
    assert _preset(app) == ["Karnataka / Western Ghats"]
    app.sidebar.selectbox[0].select("Assam").run()
    for page in PAGES:
        app.switch_page(page).run()
        assert not app.exception, page
        assert _preset(app) == ["Assam"], page


def test_what_if_uses_selected_region(app):
    app.sidebar.selectbox[0].select("Odisha").run()
    app.switch_page("pages/1_What_If_Simulator.py").run()
    assert app.session_state["_wi_twin"].region.name == "Odisha"
    app.sidebar.selectbox[0].select("Maharashtra").run()
    assert app.session_state["_wi_twin"].region.name == "Maharashtra"


def test_no_karnataka_only_warning(app):
    app.sidebar.selectbox[0].select("Uttarakhand").run()
    text = " ".join(w.value for w in app.sidebar.warning) + " ".join(i.value for i in app.sidebar.info)
    assert "Karnataka data only" not in text


def test_invalid_custom_region_shows_an_error_not_a_crash(app):
    app.sidebar.selectbox[0].select("Custom").run()
    assert not app.exception
    # inverted latitude box: min above max
    mins = [n for n in app.sidebar.number_input if n.label == "Min lat"][0]
    mins.set_value(40.0).run()
    assert not app.exception
    assert any("Latitude" in e.value for e in app.sidebar.error)
    # fixing it brings the page back
    maxs = [n for n in app.sidebar.number_input if n.label == "Max lat"][0]
    maxs.set_value(45.0).run()
    assert not app.exception and not list(app.sidebar.error)


def test_region_bounds_error_rules():
    from config.config import RegionConfig
    from src.data_ingestion.ingestion_module import build_region_grid, region_bounds_error
    ok = RegionConfig(name="ok", min_lat=10, max_lat=12, min_lon=70, max_lon=72,
                      grid_resolution_deg=0.5, weather_grid_resolution_deg=1.5)
    assert region_bounds_error(ok) is None
    bad = RegionConfig(name="bad", min_lat=12, max_lat=10, min_lon=70, max_lon=72,
                       grid_resolution_deg=0.5, weather_grid_resolution_deg=1.5)
    assert "Latitude" in region_bounds_error(bad)
    huge = RegionConfig(name="huge", min_lat=0, max_lat=60, min_lon=0, max_lon=10,
                        grid_resolution_deg=0.5, weather_grid_resolution_deg=1.5)
    assert "no larger" in region_bounds_error(huge)
    with pytest.raises(ValueError):
        build_region_grid(bad)

