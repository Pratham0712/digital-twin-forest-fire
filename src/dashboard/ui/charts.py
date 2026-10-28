"""
charts.py - one Plotly configuration for every chart in the dashboard.

The Plotly mode bar (camera, zoom, pan, box select, reset, autoscale) is switched off with Plotly's own
supported option; hover tooltips stay on. The animated detection map keeps mouse navigation (drag to pan,
scroll to zoom) without the buttons.

Streamlit draws a separate fullscreen button above Plotly charts that has no configuration switch;
CHART_TOOLBAR_CSS hides only that toolbar, only inside Plotly chart elements (nothing else is matched).
"""
from __future__ import annotations

CHART_CONFIG = {"displayModeBar": False, "displaylogo": False}
MAP_CONFIG = {**CHART_CONFIG, "scrollZoom": True}

CHART_TOOLBAR_CSS = (
    '[data-testid="stElementContainer"]:has([data-testid="stPlotlyChart"]) [data-testid="stElementToolbar"]'
    " { display: none !important; }"
)
