"""OFFLINE TEST FIXTURES for the land-cover pipeline.

Synthetic rasters in the WorldCover / NDVI window format, so the automated
tests never download ESA WorldCover or Sentinel-2. They are labelled as test
fixtures everywhere and are never used by the application itself.
"""
import numpy as np

from src.landcover.tile_cache import RasterWindow

RES = 1.0 / 12000.0
FIXTURE_WC_LABEL = "OFFLINE TEST FIXTURE — synthetic WorldCover-format raster (not real ESA WorldCover data)"
FIXTURE_NDVI_LABEL = "OFFLINE TEST FIXTURE — synthetic NDVI raster (not real Sentinel-2 data)"


def window(bounds, fill, dtype=np.uint8, pad=60, label=FIXTURE_WC_LABEL):
    n = int(np.ceil((bounds["north"] - bounds["south"]) / RES)) + 2 * pad
    m = int(np.ceil((bounds["east"] - bounds["west"]) / RES)) + 2 * pad
    return RasterWindow(np.full((n, m), fill, dtype=dtype), bounds["north"] + pad * RES, bounds["west"] - pad * RES,
                        RES, label, {"fixture": True})


def paint(win, lat_s, lat_n, lon_w, lon_e, value):
    """Set the pixels whose centres fall in [lat_s, lat_n] x [lon_w, lon_e]."""
    rows = win.north - (np.arange(win.array.shape[0]) + 0.5) * win.res_deg
    cols = win.west + (np.arange(win.array.shape[1]) + 0.5) * win.res_deg
    r = (rows >= lat_s) & (rows <= lat_n)
    c = (cols >= lon_w) & (cols <= lon_e)
    win.array[np.ix_(r, c)] = value
    return win


def wc_loader(code=10, painter=None):
    """Replacement for worldcover.load_window returning a uniform class (and
    optional painted patches) for any requested bounds."""
    def load(bounds, allow_fetch=True, cache=None):
        w = window(bounds, code)
        if painter:
            painter(w)
        return w, {"status": "fixture", "tiles": 1, "missing": 0, "error": "",
                   "retrieved_last": "fixture"}
    return load


def ndvi_loader(value=0.6, painter=None):
    def load(bounds, allow_fetch=True, cache=None):
        w = window(bounds, value, dtype=np.float32, label=FIXTURE_NDVI_LABEL)
        if painter:
            painter(w)
        w.meta["scene_dates"] = []
        return w, {"status": "fixture", "tiles": 1, "missing": 0, "error": ""}
    return load


def unavailable_loader(reason="offline test: source unavailable"):
    def load(bounds, allow_fetch=True, cache=None):
        return None, {"status": "unavailable", "tiles": 1, "missing": 1, "error": reason}
    return load
