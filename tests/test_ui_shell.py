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
    """Signed-in app with Offline / demo mode EXPLICITLY switched ON (the
    authoritative state), so these UI tests are deterministic and never call
    the real APIs, whatever keys the machine's .env holds."""
    at = AppTest.from_file(str(DASH / "app.py"), default_timeout=240)
    at.session_state["auth_user"] = "admin"
    at.session_state["auth_role"] = "admin"
    at.session_state["_pref_offline_mode"] = True
    at.run()
    assert not at.exception
    return at


def _keys(monkeypatch, firms="FAKE_FIRMS_KEY", owm="FAKE_OWM_KEY"):
    from config.config import API
    monkeypatch.setattr(API, "firms_map_key", firms)
    monkeypatch.setattr(API, "owm_api_key", owm)


def test_ticker_without_state_never_assumes_demo(monkeypatch):
    """No twin and no session state (keys configured): the ticker reports real-
    data mode still connecting, not DEMO, and invents no values."""
    from src.dashboard.ui.global_ticker import ticker_html
    _keys(monkeypatch)
    h = ticker_html(None)
    assert "DEMO / OFFLINE MODE" not in h
    assert "REAL DATA · CONNECTING" in h and "REAL DATA · no risk state loaded yet" in h
    assert "CONNECTING · fetching real detections" in h and "CONNECTING · fetching real weather" in h
    assert "REGION" not in h and "RISK LEVEL" not in h and "LIVE" not in h      # nothing invented / claimed


def test_ticker_follows_the_authoritative_switch(monkeypatch):
    """Explicit OFF (even with keys missing) is real-data mode; explicit ON is demo."""
    from src.dashboard import app_state
    from src.dashboard.ui.global_ticker import ticker_html
    _keys(monkeypatch, firms="", owm="")
    monkeypatch.setattr(app_state, "is_demo_mode", lambda: False)
    h = ticker_html(None)
    assert "DEMO / OFFLINE MODE" not in h and "REAL DATA" in h
    monkeypatch.setattr(app_state, "is_demo_mode", lambda: True)
    h = ticker_html(None)
    assert "DEMO / OFFLINE MODE" in h and "REAL DATA" not in h


def test_active_nav_css_per_page():
    from src.dashboard.ui.theme import THEME_CSS, active_nav_css
    assert 'a[href*="Spread_Simulation"]' in active_nav_css("Spread Simulation")
    assert "li:first-child a" in active_nav_css("Command Center")
    assert active_nav_css("Unknown page") == ""
    assert '[aria-current="page"]' in THEME_CSS and "#E86CFF" in THEME_CSS


def test_command_center_layout(app, monkeypatch):
    md = _md(app)
    assert 'class="gt"' in md                                         # global ticker present
    # demo mode was switched ON explicitly (fixture) -> labelled demo everywhere
    assert app.session_state["_pref_offline_mode"] is True
    assert [t.value for t in app.sidebar.toggle if t.label == "Offline / demo mode"] == [True]
    assert "DEMO / OFFLINE MODE" in md and "Demo / offline (sidebar toggle ON)" in md
    assert "Karnataka" in md                                          # region from the twin
    assert "Illustrative concept artwork" in md                       # hero is labelled as artwork
    # hero is a plain <img> (static file, or inline fallback): no st.image, so no fullscreen button
    assert 'class="cc-hero"' in md and ("app/static/dashboard_hero.jpg" in md or "data:image/jpeg;base64" in md)
    assert not app.get("imgs")
    assert 'class="fa' in md and "MODEL PREDICTION" in md and "DEMO DATA" in md
    assert 'class="cc-strip"' in md and 'class="kpi-row"' in md and 'class="sys-grid"' in md
    assert "OPEN SPREAD SIMULATION  →" in [p.label for p in app.get("page_link")]
    order = [md.index(s) for s in ('class="gt"', 'class="cc-hero"', "Illustrative concept artwork", 'class="cc-strip"',
                                   'class="fa', 'class="kpi-row"', 'class="ex-card"', 'class="sys-grid"')]
    assert order == sorted(order)
    # switching it OFF -> real-data mode: never DEMO, the actual feed outcome is shown.
    # HTTP is stubbed to fail fast (no network in tests): both feeds UNAVAILABLE.
    from src.data_ingestion.firms_client import FIRMSClient
    from src.data_ingestion.weather_client import WeatherClient
    from src.dashboard import dashboard_common as dc
    from src.data_ingestion import ingestion_module as im
    dc._SHARED_TWINS.clear()                                          # no twin / data reuse from other tests
    im._LIVE_CACHE.clear()
    _keys(monkeypatch)
    monkeypatch.setattr(FIRMSClient, "fetch_hotspots",
                        lambda self, *a, **k: (self._errors.clear(), self._errors.append("stubbed outage"),
                                               self._finish(None))[-1])
    monkeypatch.setattr(WeatherClient, "fetch_point", lambda self, lat, lon: None)
    app.sidebar.toggle[0].set_value(False).run()
    assert not app.exception
    md = _md(app)
    tick = next(m.value for m in app.markdown if 'class="gt"' in m.value)
    assert "DEMO / OFFLINE MODE" not in tick and "REAL DATA · API UNAVAILABLE" in tick
    assert "Real data (sidebar toggle OFF)" in md and "DEMO DATA" not in md


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
