"""What-If -> Spread Simulation hand-off (Streamlit AppTest, headless)."""
import pytest

pytest.importorskip("streamlit.testing.v1")
from pathlib import Path

from streamlit.testing.v1 import AppTest

DASH = Path(__file__).resolve().parents[1] / "src" / "dashboard"


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}")
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)
    at = AppTest.from_file(str(DASH / "app.py"), default_timeout=240)
    at.session_state["auth_user"] = "admin"
    at.session_state["auth_role"] = "admin"
    at.session_state["_pref_offline_mode"] = True       # explicit demo mode: no real API calls
    at.run()
    assert not at.exception
    return at


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def test_apply_and_open_transfers_the_scenario(app):
    app.switch_page("pages/1_What_If_Simulator.py").run()
    assert not app.exception
    assert next(s for s in app.selectbox if s.label == "Forest / location").value == "Bandipur Tiger Reserve"
    next(s for s in app.slider if s.label.startswith("Wind direction")).set_value(90).run()
    next(s for s in app.slider if s.label == "Temperature (°C)").set_value(41).run()
    _button(app, "Apply Scenario & Open Spread Simulation").click().run()
    assert not app.exception
    cfg = app.session_state["simulation_config"]
    assert cfg["forest"] == "Bandipur Tiger Reserve"
    assert cfg["wind_from_deg"] == 90 and cfg["temp_c"] == 41
    from src.simulation.local_spread import FOCUS_AREAS
    bp = FOCUS_AREAS["Bandipur Tiger Reserve"]                 # Phase 3: preset moved to a forest interior
    assert cfg["latitude"] == pytest.approx(bp.lat) and cfg["longitude"] == pytest.approx(bp.lon)
    assert cfg["width_m"] == 500 and cfg["height_m"] == 500 and cfg["cell_m"] == 25
    assert cfg["duration_min"] == 60 and cfg["setup"]["layers"] == {"grid": True, "boundary": True}
    assert cfg["selected_location"]["source"] == "preset"
    assert cfg["scenario"]["wind_from_deg"] == 90
    assert app.session_state["_sim_twin"].ingestion.scenario["temp_c"] == 41
    # Module 2 opened in the applied-scenario mode (AppTest keeps its own page
    # pointer, so point it at the page st.switch_page opened)
    assert app.session_state["spread_mode"].startswith("Applied")
    app.switch_page("pages/2_Spread_Simulation.py").run()
    assert not app.exception
    _button(app, "Run Simulation").click().run()
    assert not app.exception
    res = app.session_state["applied_geo_result"]
    assert res.focus.name == "Bandipur Tiger Reserve"
    assert res.wind_schedule[0] == (cfg["wind_speed_ms"], 90.0)


def test_spread_page_without_a_scenario_falls_back_to_current_state(app):
    app.switch_page("pages/2_Spread_Simulation.py").run()
    assert not app.exception
    assert app.session_state["spread_mode"].startswith("Current")
    _button(app, "Run Simulation").click().run()
    assert not app.exception
    assert app.session_state["current_geo_result"] is not None


def _number(at, label):
    return next(n for n in at.number_input if n.label == label)


def test_custom_dimensions_duration_and_layers_transfer(app):
    app.switch_page("pages/1_What_If_Simulator.py").run()
    _number(app, "Width (m, east–west)").set_value(1000).run()
    _number(app, "Height (m, north–south)").set_value(750).run()
    next(s for s in app.selectbox if s.label == "Simulation duration").select("15 min").run()
    next(t for t in app.toggle if t.label == "Grid").set_value(False).run()
    assert not app.exception
    setup = app.session_state["sim_setup"]
    assert (setup["width_m"], setup["height_m"], setup["duration_min"]) == (1000, 750, 15)
    assert setup["layers"] == {"grid": False, "boundary": True}
    _button(app, "Apply Scenario & Open Spread Simulation").click().run()
    cfg = app.session_state["simulation_config"]
    assert (cfg["width_m"], cfg["height_m"], cfg["duration_min"]) == (1000, 750, 15)
    app.switch_page("pages/2_Spread_Simulation.py").run()
    _button(app, "Run Simulation").click().run()
    assert not app.exception
    res = app.session_state["applied_geo_result"]
    assert (res.focus.n_cols, res.focus.n_rows) == (40, 30) and res.duration_minutes == 15
    # grid hidden: the simulation still ran on the full domain
    assert res.domain.n_cols == 40 + 2 * res.domain.margin
    _button(app, "Reset").click().run()
    assert app.session_state["applied_geo_result"] is None


def test_invalid_setup_blocks_the_hand_off(app):
    app.switch_page("pages/1_What_If_Simulator.py").run()
    next(s for s in app.selectbox if s.label == "Cell size (m)").select(5).run()
    _number(app, "Width (m, east–west)").set_value(3000).run()
    _number(app, "Height (m, north–south)").set_value(3000).run()
    assert not app.exception
    assert any("cell limit" in e.value for e in app.error)
    assert _button(app, "Apply Scenario & Open Spread Simulation").disabled


def test_map_events_update_the_setup():
    """Google search result, dragged box, clicked ignition and layer toggles
    coming back from the map component."""
    from types import SimpleNamespace

    from src.dashboard.geo_spread import CUSTOM_LOCATION, apply_map_event
    setup = {"location": {"name": "Bandipur Tiger Reserve", "lat": 11.6667, "lon": 76.6333, "source": "preset",
                          "preset": "Bandipur Tiger Reserve"},
             "width_m": 500.0, "height_m": 500.0, "cell_m": 25.0, "duration_min": 60.0,
             "placement": "Upwind edge", "n_ignition": 3, "ignition_points": [],
             "layers": {"grid": True, "boundary": True}}
    twin = SimpleNamespace()
    assert apply_map_event({"kind": "place", "name": "Coorg", "lat": 12.42, "lon": 75.74, "source": "google_geocoding",
                            "types": ["administrative_area_level_3"], "bounds": {"north": 12.5}}, setup, twin)
    loc = setup["location"]
    assert (loc["name"], loc["lat"], loc["lon"], loc["source"], loc["preset"]) == \
        ("Coorg", 12.42, 75.74, "google_geocoding", CUSTOM_LOCATION)
    assert apply_map_event({"kind": "area", "lat": 12.43, "lon": 75.75, "width_m": 812, "height_m": 640}, setup, twin)
    assert (setup["width_m"], setup["height_m"]) == (800, 650)       # snapped to whole 25 m cells
    assert setup["location"]["lat"] == 12.43
    assert apply_map_event({"kind": "ignite", "points": [[12.431, 75.751], [12.432, 75.752]]}, setup, twin)
    assert setup["placement"] == "Map points" and len(setup["ignition_points"]) == 2
    assert apply_map_event({"kind": "ignite", "points": []}, setup, twin)
    assert setup["placement"] == "Upwind edge"
    assert apply_map_event({"kind": "layers", "grid": False, "boundary": False}, setup, twin)
    assert setup["layers"] == {"grid": False, "boundary": False}


def test_map_component_renders_with_a_key(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}")
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "test-key")
    at = AppTest.from_file(str(DASH / "app.py"), default_timeout=240)
    at.session_state["auth_user"] = "admin"
    at.session_state["auth_role"] = "admin"
    at.session_state["_pref_offline_mode"] = True       # explicit demo mode: no real API calls
    at.run()
    at.switch_page("pages/1_What_If_Simulator.py").run()
    assert not at.exception
    at.switch_page("pages/2_Spread_Simulation.py").run()
    _button(at, "Run Simulation").click().run()
    assert not at.exception
