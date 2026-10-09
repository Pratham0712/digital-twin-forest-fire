"""Shared test settings: the automated suite never touches the network and
never uses the developer's real API keys.

Production:       real .env  -> real NASA FIRMS / OpenWeatherMap / Google APIs
Automated tests:  no keys + explicit demo state, or mocked HTTP -> no external calls

  * The data-API keys that config.py loaded from the developer's .env are
    blanked for every test (tests that exercise the live path set FAKE keys
    and stub the HTTP layer themselves). Keys are never read or printed here.
  * Every HTTP request through `requests` is refused with a ConnectionError
    subclass (production code treats it like an outage), and any attempt to
    reach NASA FIRMS or OpenWeatherMap FAILS the test, so an accidental live
    call - and quota use - cannot go unnoticed.
  * Raw sockets to non-loopback addresses are refused as a backstop.
"""
import socket
from urllib.parse import urlparse

import pytest
import requests
import requests.adapters

QUOTA_API_HOSTS = ("firms.modaps.eosdis.nasa.gov", "api.openweathermap.org")


class ExternalNetworkBlocked(requests.ConnectionError):
    """Raised instead of sending a real HTTP request during tests."""


@pytest.fixture(autouse=True)
def _no_land_cover_downloads(monkeypatch):
    """Keep the suite offline and deterministic: land cover comes only from the
    disk cache (tests that need it build their own layer)."""
    monkeypatch.setenv("FIRE_LAND_COVER_FETCH", "0")


@pytest.fixture(autouse=True)
def _offline_land_cover_fixture(monkeypatch, tmp_path):
    """OFFLINE TEST FIXTURE for the fused land cover: a synthetic
    WorldCover-format raster of uniform tree cover (labelled as a fixture, never
    real data) and no Sentinel-2 NDVI, so simulations in the suite run without
    downloads. OpenStreetMap still comes only from the existing disk cache.
    Tests of the pipeline itself replace these loaders explicitly."""
    from tests.landcover_fixtures import unavailable_loader, wc_loader
    from src.landcover import provider, sentinel2_ndvi, tile_cache, worldcover
    monkeypatch.setenv("FIRE_LANDCOVER_CACHE_DIR", str(tmp_path / "landcover_tiles"))
    monkeypatch.setattr(worldcover, "load_window", wc_loader(10))
    monkeypatch.setattr(sentinel2_ndvi, "load_window", unavailable_loader("offline test: no Sentinel-2 fixture"))
    provider.clear_memory_cache()
    tile_cache.clear_failures()
    yield
    provider.clear_memory_cache()


@pytest.fixture(autouse=True)
def _no_real_api_keys(monkeypatch):
    """Blank the FIRMS / OpenWeatherMap keys loaded from the developer's .env,
    so no test silently runs in real-data mode with real credentials."""
    from config.config import API
    monkeypatch.setattr(API, "firms_map_key", "")
    monkeypatch.setattr(API, "owm_api_key", "")
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    monkeypatch.delenv("OWM_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def _no_external_network(monkeypatch):
    attempts = []

    def blocked_send(self, request, *args, **kwargs):
        host = urlparse(request.url).hostname or "?"
        attempts.append(host)                      # host only: URLs can carry API keys
        raise ExternalNetworkBlocked(f"network access is blocked in tests ({host})")

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", blocked_send)

    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def _external(sock, address) -> bool:
        if sock.family not in (socket.AF_INET, socket.AF_INET6):
            return False
        host = address[0] if isinstance(address, tuple) else str(address)
        return host not in ("127.0.0.1", "::1", "localhost")

    def guarded_connect(sock, address):
        if _external(sock, address):
            attempts.append(str(address[0]) if isinstance(address, tuple) else "?")
            raise ExternalNetworkBlocked("network access is blocked in tests")
        return real_connect(sock, address)

    def guarded_connect_ex(sock, address):
        if _external(sock, address):
            attempts.append(str(address[0]) if isinstance(address, tuple) else "?")
            raise ExternalNetworkBlocked("network access is blocked in tests")
        return real_connect_ex(sock, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    yield attempts
    quota_calls = sorted({h for h in attempts if h in QUOTA_API_HOSTS})
    assert not quota_calls, f"test attempted a real API call to {quota_calls}; use demo mode or mock the HTTP layer"
