"""Image assets bundled with the dashboard (src/dashboard/assets)."""
import base64
from functools import lru_cache
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
STATIC_DIR = Path(__file__).resolve().parents[1] / "static"   # served by Streamlit at app/static/

# The supplied hero artwork (unmodified composition). Served as a static file and
# drawn with a plain <img>, so Streamlit adds no image toolbar / fullscreen button.
HERO_IMAGE = STATIC_DIR / "dashboard_hero.jpg"
HERO_URL = "app/static/dashboard_hero.jpg"
# Command Center hero carousel: the three supplied images, in display order.
HERO_SLIDES = (
    {"path": HERO_IMAGE, "url": HERO_URL, "title": "Forest Fire Digital Twin Command Center",
     "alt": "Forest Fire Digital Twin: from satellite observations to predictive action (illustrative artwork)"},
    {"path": STATIC_DIR / "hero" / "forest_fire_command_center.png",
     "url": "app/static/hero/forest_fire_command_center.png",
     "title": "Forest Fire Command Center",
     "alt": "Forest department operations room: officers reviewing a wall map with a fire alert, predicted "
            "spread, environmental conditions and recommended actions (illustrative artwork)"},
    {"path": STATIC_DIR / "hero" / "satellite_intelligence.jpg", "url": "app/static/hero/satellite_intelligence.jpg",
     "title": "Satellite-Based Forest Intelligence",
     "alt": "A satellite observing the forest with geospatial data layers and weather indicators "
            "(illustrative artwork)"},
)
LOGIN_BACKGROUND = ASSETS_DIR / "login_background.jpg"  # blurred, downscaled copy of the hero artwork
ALERT_THUMB = ASSETS_DIR / "alert_thumb.jpg"            # fire-front crop of the hero artwork
ALERT_THUMB_STATIC = STATIC_DIR / "alert_thumb.jpg"     # identical copy, served at app/static/
ALERT_THUMB_URL = "app/static/alert_thumb.jpg"


@lru_cache(maxsize=8)
def data_uri(path: Path) -> str:
    """Small images only (login background, alert thumbnail). Returns "" when
    the file is missing so the page still renders without it."""
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return ""
    mime = "image/png" if str(path).lower().endswith(".png") else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
