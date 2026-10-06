"""Image assets bundled with the dashboard (src/dashboard/assets)."""
import base64
from functools import lru_cache
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"

HERO_IMAGE = ASSETS_DIR / "dashboard_hero.jpg"          # the supplied hero artwork, unmodified aspect
LOGIN_BACKGROUND = ASSETS_DIR / "login_background.jpg"  # blurred, downscaled copy of the hero artwork
ALERT_THUMB = ASSETS_DIR / "alert_thumb.jpg"            # fire-front crop of the hero artwork


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
