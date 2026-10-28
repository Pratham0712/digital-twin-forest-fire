"""Every Plotly chart in the dashboard is drawn without the Plotly mode bar (camera / zoom / pan / reset)
and Streamlit's fullscreen toolbar is hidden only inside Plotly chart elements. Static checks; offline."""
import re
from pathlib import Path

from src.dashboard.ui import charts

ROOT = Path(__file__).resolve().parents[1]


def test_every_plotly_chart_call_disables_the_mode_bar():
    calls = []
    for f in sorted((ROOT / "src").rglob("*.py")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if "st.plotly_chart(" in line and not line.strip().startswith("#"):
                calls.append((f.name, line))
    assert len(calls) >= 15
    for name, line in calls:
        assert re.search(r"config=(CHART_CONFIG|MAP_CONFIG)\)", line), (name, line)


def test_config_values():
    assert charts.CHART_CONFIG == {"displayModeBar": False, "displaylogo": False}
    assert charts.MAP_CONFIG["displayModeBar"] is False and charts.MAP_CONFIG["scrollZoom"] is True   # map still zooms
    assert "staticPlot" not in charts.CHART_CONFIG                       # hover tooltips stay


def test_fallback_css_is_scoped_to_plotly_elements_only():
    css = charts.CHART_TOOLBAR_CSS
    assert css.startswith('[data-testid="stElementContainer"]:has([data-testid="stPlotlyChart"]) '
                          '[data-testid="stElementToolbar"]')
    assert css.count("{") == 1 and "button" not in css and "svg" not in css   # no global hiding of buttons / icons
    common = (ROOT / "src/dashboard/dashboard_common.py").read_text(encoding="utf-8")
    assert "CHART_TOOLBAR_CSS" in common
