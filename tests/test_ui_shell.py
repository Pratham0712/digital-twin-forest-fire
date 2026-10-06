"""UI shell: global ticker, active navigation styling, Command Center panels
and the login screen. Checks that every value shown is labelled honestly
(DEMO / SIMULATED / MODEL PREDICTION) and that nothing is invented when no
state is loaded."""
import pytest

pytest.importorskip("streamlit.testing.v1")
from pathlib import Path

from streamlit.testing.v1 import AppTest

DASH = Path(__file__).resolve().parents[1] / "src" / "dashboard"
PAGES = ["app.py", "pages/1_What_If_Simulator.py", "pages/2_Spread_Simulation.py",
         "pages/3_Historical_Time_Machine.py", "pages/4_AI_Situation_Briefing.py",
         "pages/5_Model_Insights.py", "pages/6_Admin.py", "pages/7_Activity_Log.py"]


def _md(at) -> str:
    return "\n".join(m.value for m in at.markdown)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}")
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)


@pytest.fixture
def app(env):
    at = AppTest.from_file(str(DASH / "app.py"), default_timeout=240)
    at.session_state["auth_user"] = "admin"
    at.session_state["auth_role"] = "admin"
    at.run()
    assert not at.exception
    return at


def test_ticker_without_state_says_demo_offline():
    from src.dashboard.ui.global_ticker import ticker_html
    h = ticker_html(None)
    assert "DEMO / OFFLINE MODE" in h
    assert "REGION" not in h and "RISK LEVEL" not in h          # nothing invented


def test_active_nav_css_per_page():
    from src.dashboard.ui.theme import THEME_CSS, active_nav_css
    assert 'a[href*="Spread_Simulation"]' in active_nav_css("Spread Simulation")
    assert "li:first-child a" in active_nav_css("Command Center")
    assert active_nav_css("Unknown page") == ""
    assert '[aria-current="page"]' in THEME_CSS and "#E86CFF" in THEME_CSS


def test_command_center_layout(app):
    md = _md(app)
    assert 'class="gt"' in md and "DEMO / OFFLINE MODE" in md          # ticker, demo-labelled
    assert "Karnataka" in md                                          # region from the twin
    assert "Illustrative concept artwork" in md                       # hero is labelled as artwork
    assert 'class="fa' in md and "MODEL PREDICTION" in md and "DEMO DATA" in md
    assert 'class="cc-strip"' in md and 'class="kpi-row"' in md and 'class="sys-grid"' in md
    assert "OPEN SPREAD SIMULATION  →" in [p.label for p in app.get("page_link")]
    order = [md.index(s) for s in ('class="gt"', "Illustrative concept artwork", 'class="cc-strip"',
                                   'class="fa', 'class="kpi-row"', 'class="ex-card"', 'class="sys-grid"')]
    assert order == sorted(order)


def test_ticker_on_every_page(app):
    for page in PAGES:
        app.switch_page(page).run()
        assert not app.exception, page
        assert 'class="gt"' in _md(app), page


def test_simulation_run_is_labelled_simulated(app):
    app.switch_page("pages/2_Spread_Simulation.py").run()
    from src.dashboard.geo_spread import SPREAD_MODES
    app.radio(key="spread_mode").set_value(SPREAD_MODES[1]).run()
    [b for b in app.button if b.label == "Run Simulation"][0].click().run()
    assert not app.exception
    assert "SIMULATED" in _md(app)
    app.switch_page("app.py").run()
    md = _md(app)
    assert "SPREAD SIMULATION" in md and "SIMULATED" in md


def test_login_screen(env):
    at = AppTest.from_file(str(DASH / "app.py"), default_timeout=240)
    at.run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Sign in", "Create account"]
    md = _md(at)
    assert "FOREST FIRE DIGITAL TWIN" in md and "Secure Research Command Center" in md
    assert "no self-registration" in md                               # no fake sign-up
    assert 'class="gt"' not in md                                     # no ticker before sign-in
    at.text_input[0].input("admin")
    at.text_input[1].input("wrong-password")
    at.button[0].click().run()
    assert any("Incorrect" in e.value for e in at.error)
    at.text_input[1].input("changeme123")
    at.button[0].click().run()
    assert at.session_state["auth_user"] == "admin"
