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
    at.run()
    assert not at.exception
    return at


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def test_apply_and_open_transfers_the_scenario(app):
    app.switch_page("pages/1_What_If_Simulator.py").run()
    assert not app.exception
    assert next(s for s in app.selectbox if s.label == "Simulation focus area").value == "Bandipur Tiger Reserve"
    next(s for s in app.slider if s.label.startswith("Wind direction")).set_value(90).run()
    next(s for s in app.slider if s.label == "Temperature (°C)").set_value(41).run()
    _button(app, "Apply Scenario & Open Spread Simulation").click().run()
    assert not app.exception
    cfg = app.session_state["simulation_config"]
    assert cfg["forest"] == "Bandipur Tiger Reserve"
    assert cfg["wind_from_deg"] == 90 and cfg["temp_c"] == 41
    assert cfg["latitude"] == pytest.approx(11.6667) and cfg["longitude"] == pytest.approx(76.6333)
    assert cfg["size_m"] == 500 and cfg["cell_m"] == 25
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
