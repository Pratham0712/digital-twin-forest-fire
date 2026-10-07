"""Display timestamps in IST (Asia/Kolkata); the data layer stays UTC."""
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from src.utils.timezone import IST, format_ist, format_ist_series, localize_text, to_utc, utc_to_ist

EXPECTED = "07 Oct 2026, 10:36 PM IST"


# ── conversion utility ──────────────────────────────────────────────────── #

@pytest.mark.parametrize("value", [
    "2026-10-07 17:06 UTC",                      # the reported example
    "2026-10-07T17:06:00Z",                      # ISO with Z
    "2026-10-07T17:06Z",                         # NASA FIRMS latest_acq_utc format
    "2026-10-07T17:06:00+00:00",                 # datetime.isoformat() of an aware UTC time
    "2026-10-07T17:06:00.123456+00:00",
    "2026-10-07 17:06:00",                       # naive = UTC (project convention)
    datetime(2026, 10, 7, 17, 6, tzinfo=timezone.utc),
    datetime(2026, 10, 7, 17, 6),
    pd.Timestamp("2026-10-07T17:06:00Z"),
    int(datetime(2026, 10, 7, 17, 6, tzinfo=timezone.utc).timestamp()),          # Unix seconds
    int(datetime(2026, 10, 7, 17, 6, tzinfo=timezone.utc).timestamp()) * 1000,   # Unix milliseconds
])
def test_utc_example_becomes_ist(value):
    assert format_ist(value) == EXPECTED


def test_display_styles():
    v = "2026-10-07T17:06:00Z"
    assert format_ist(v, "compact") == "10:36 PM IST"
    assert format_ist(v, "dot") == "07 Oct 2026 · 10:36 PM IST"
    assert format_ist("2026-10-07T17:06:05Z", "seconds") == "07 Oct 2026, 10:36:05 PM IST"


def test_midnight_utc():
    assert format_ist("2026-10-07T00:00:00Z") == "07 Oct 2026, 05:30 AM IST"


def test_crossing_midnight_into_the_next_ist_day():
    assert format_ist("2026-10-07T18:30:00Z") == "08 Oct 2026, 12:00 AM IST"
    assert format_ist("2026-12-31T20:00:00Z") == "01 Jan 2027, 01:30 AM IST"   # and the next year


def test_timezone_aware_datetimes_of_other_zones():
    ny = timezone(timedelta(hours=-4))
    assert format_ist(datetime(2026, 10, 7, 13, 6, tzinfo=ny)) == EXPECTED
    assert utc_to_ist(datetime(2026, 10, 7, 17, 6, tzinfo=timezone.utc)).utcoffset() == timedelta(hours=5, minutes=30)


def test_already_localized_values_are_not_shifted_twice():
    assert format_ist("2026-10-07T22:36:00+05:30") == EXPECTED
    assert format_ist(datetime(2026, 10, 7, 22, 36, tzinfo=IST)) == EXPECTED
    assert format_ist("2026-10-07 22:36 IST") == EXPECTED
    assert format_ist(utc_to_ist("2026-10-07T17:06:00Z")) == EXPECTED          # converting the result again


@pytest.mark.parametrize("value", [None, "", "   ", "not a time", float("nan"), pd.NaT, True, [], {}])
def test_invalid_or_missing(value):
    assert format_ist(value) == "-"
    assert format_ist(value, missing="") == ""
    assert to_utc(value) is None


def test_date_only_is_not_shifted():
    assert format_ist("2026-10-07") == "07 Oct 2026"
    assert format_ist(date(2026, 10, 7)) == "07 Oct 2026"


def test_series_and_embedded_text():
    s = pd.Series(["2026-10-07T17:06:00+00:00", None])
    assert format_ist_series(s).tolist() == [EXPECTED, "-"]
    assert localize_text("OpenWeatherMap unavailable; cached from 2026-10-07T17:06:00+00:00") == \
        f"OpenWeatherMap unavailable; cached from {EXPECTED}"
    assert localize_text("No timestamp here") == "No timestamp here"


# ── the dashboard shows IST, the data layer keeps UTC ───────────────────── #

def test_live_cards_show_ist_and_keep_observed_and_fetched_separate(monkeypatch):
    from src.dashboard.ui import live_panel
    shown = []
    monkeypatch.setattr(live_panel.st, "markdown", lambda body, **k: shown.append(body))
    monkeypatch.setattr(live_panel.st, "caption", lambda body, **k: shown.append(body))
    monkeypatch.setattr(live_panel.st, "columns", lambda spec: [type("C", (), {"button": lambda *a, **k: False})()
                                                               for _ in range(len(spec) if isinstance(spec, list) else spec)])
    w = {"temperature_c": 22.0, "humidity_pct": 81, "wind_speed_ms": 2.1, "wind_deg": 159, "pressure_hpa": 1008,
         "observed_at": "2026-10-07T16:50:00+00:00", "place_name": "Bandipur"}
    ws = {"mode": "live", "fetched_utc": "2026-10-07T17:06:00+00:00"}
    fs = {"mode": "live", "n": 1, "fetched_utc": "2026-10-07T17:07:00+00:00", "latest_acq_utc": "2026-10-07T08:12Z",
          "source": "NASA FIRMS VIIRS_SNPP_NRT"}
    live_panel.render_live_conditions(11.66, 76.63, "Bandipur", "t", False, data=(w, ws, pd.DataFrame(), fs))
    html = "\n".join(shown)
    assert "<span>Observed</span><span>07 Oct 2026, 10:20 PM IST</span>" in html       # observation time
    assert "<span>Fetched</span><span>07 Oct 2026, 10:36 PM IST</span>" in html        # fetch time (weather)
    assert "<span>Latest observation</span><span>07 Oct 2026, 01:42 PM IST</span>" in html
    assert "<span>Fetched</span><span>07 Oct 2026, 10:37 PM IST</span>" in html        # fetch time (FIRMS)
    assert "Last weather fetch: 07 Oct 2026, 10:36 PM IST · Last FIRMS fetch: 07 Oct 2026, 10:37 PM IST" in html
    assert "17:06 UTC" not in html and "2026-10-07 17:06" not in html
    assert ws["fetched_utc"] == "2026-10-07T17:06:00+00:00"                             # input untouched


def test_ticker_shows_ist(monkeypatch):
    from src.dashboard.ui import global_ticker as gt
    feeds = {"firms_mode": "live", "weather_mode": "live",
             "firms_status": {"mode": "live", "n": 2, "latest_acq_utc": "2026-10-07T17:06Z"},
             "weather_status": {"mode": "live", "fetched_utc": "2026-10-07T17:06:00+00:00"}}
    assert gt.feed_text("firms", feeds).endswith("latest observation 07 Oct 2026 · 10:36 PM IST")
    assert gt.feed_text("weather", feeds) == "LIVE · fetched 07 Oct 2026 · 10:36 PM IST"


def test_map_tooltips_show_ist_but_markers_keep_utc():
    from src.dashboard.geo_fire_map import setup_payload
    from src.data_ingestion.live_point import hotspot_markers
    from src.simulation.local_spread import FocusArea
    df = pd.DataFrame([{"latitude": 11.67, "longitude": 76.64, "acq_datetime": pd.Timestamp("2026-10-07 17:06"),
                        "satellite": "N", "confidence": "n", "frp": 5.0}])
    marks = hotspot_markers(df)
    assert marks[0]["date"] == "2026-10-07 17:06 UTC"                                  # data layer: UTC
    f = FocusArea(name="t", lat=11.6667, lon=76.6333, width_m=500, height_m=500, cell_m=25)
    p = setup_payload(f, 60, 3, 240, 3, "Centre", [], {"grid": True, "boundary": True}, "K", "", hotspots=marks)
    assert p["hotspots"][0]["date"] == "07 Oct 2026 · 10:36 PM IST"                   # display: IST
    assert marks[0]["date"] == "2026-10-07 17:06 UTC"                                  # not mutated


def test_data_layer_stays_utc():
    from src.data_ingestion import live_point
    from src.data_ingestion.weather_client import _now
    for iso in (live_point._now_iso(), _now()):
        assert iso.endswith("+00:00")


def test_admin_tables_and_activity_log_use_the_shared_converter():
    from pathlib import Path
    dash = Path(__file__).resolve().parents[1] / "src" / "dashboard" / "pages"
    admin = (dash / "6_Admin.py").read_text(encoding="utf-8")
    log = (dash / "7_Activity_Log.py").read_text(encoding="utf-8")
    assert "format_ist_series" in admin and '"created (IST)"' in admin
    assert "from src.utils.timezone import IST as LOCAL_TZ, format_ist_series" in log
    assert 'LOCAL_TZ = "Asia/Kolkata"' not in log                                      # no private converter
