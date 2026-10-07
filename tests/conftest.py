"""Shared test settings."""
import pytest


@pytest.fixture(autouse=True)
def _no_land_cover_downloads(monkeypatch):
    """Keep the suite offline and deterministic: land cover comes only from the
    disk cache (tests that need it build their own layer)."""
    monkeypatch.setenv("FIRE_LAND_COVER_FETCH", "0")
