"""Regression: refreshed real-world data must reach the simulator, not only the
REAL WEATHER card.

Bug report: the REAL WEATHER card showed the fetched OpenWeatherMap values
(22.0 °C, 81 %, 2.1 m/s from 159°) while the scenario still showed the old
defaults (32 °C, 40 %, 5.0 m/s from 225°, "Seeded hotspots: 8 (synthetic)") and
the result was badged DEMO.

All API traffic is stubbed (fake keys, conftest network block); nothing here
reaches NASA FIRMS or OpenWeatherMap."""
import pytest

pytest.importorskip("streamlit.testing.v1")

from src.dashboard import live_modes as lm
from tests.test_live_whatif_modes import (BANDIPUR, OPEN, SPREAD, _button, _detection, _md, _refresh, _slider,
                                          _weather, _whatif, live)  # noqa: F401  (live = shared fixture)

OLD_DEFAULTS = {"temp_c": 32.0, "humidity_pct": 40.0, "wind_speed_ms": 5.0, "wind_from_deg": 225.0}
REPORTED = dict(temperature_c=22.0, humidity_pct=81, wind_speed_ms=2.1, wind_deg=159)       # from the bug report
REFRESHED = dict(temperature_c=22.4, humidity_pct=79, wind_speed_ms=2.3, wind_deg=165)


def _controls(at) -> dict:
    return {k: _slider(at, k).value for k in ("temp_c", "humidity_pct", "wind_speed_ms", "wind_from_deg")}


def _twin_weather(at) -> dict:
    """The authoritative model input: the weather the risk twin (XGBoost + FWI) ran on."""
    return at.session_state["_wi_twin"].ingestion.scenario["point_weather"]


def _start(live, **weather):
    at = live(page=None)
    at.feed["weather"] = _weather(**weather)
    at.switch_page("pages/1_What_If_Simulator.py").run()
    assert not at.exception
    return at


def test_01_initial_live_weather_populates_the_simulator(live):
    at = _start(live, **REPORTED)
    assert _controls(at) == {"temp_c": 22.0, "humidity_pct": 81.0, "wind_speed_ms": 2.1, "wind_from_deg": 159.0}
    assert _controls(at) != OLD_DEFAULTS
    w = _twin_weather(at)
    assert (w["temperature_c"], w["humidity_pct"], w["wind_speed_ms"], w["wind_deg"]) == (22.0, 81, 2.1, 159)


def test_02_refresh_updates_the_real_weather_card(live):
    at = _start(live, **REPORTED)
    assert "22.0 °C" in _md(at)
    at.feed["weather"] = _weather(**REFRESHED)
    _refresh(at)
    md = _md(at)
    assert "22.4 °C" in md and "79 %" in md and "2.3 m/s from SSE (165°)" in md


def test_03_refresh_updates_the_live_controls(live):
    at = _start(live, **REPORTED)
    at.feed["weather"] = _weather(**REFRESHED)
    _button(at, "Refresh weather").click().run()                                     # the per-source button too
    assert not at.exception
    assert _controls(at) == {"temp_c": 22.4, "humidity_pct": 79.0, "wind_speed_ms": 2.3, "wind_from_deg": 165.0}
    assert all(_slider(at, k).disabled for k in ("temp_c", "humidity_pct", "wind_speed_ms", "wind_from_deg"))


def test_04_refresh_changes_the_authoritative_live_weather_state(live):
    at = _start(live, **REPORTED)
    before = at.session_state["_wi_twin"]
    at.feed["weather"] = _weather(**REFRESHED)
    _refresh(at)
    w = _twin_weather(at)
    assert at.session_state["_wi_twin"] is not before                                 # risk recomputed
    assert (w["temperature_c"], w["humidity_pct"], w["wind_speed_ms"], w["wind_deg"]) == (22.4, 79, 2.3, 165)
    tags = "\n".join(m.value for m in at.markdown if "lm-ctag" in m.value)
    assert "22.4°C" in tags and tags.count(">LIVE<") == 4                               # every control tagged LIVE


def test_05_whatif_initially_follows_the_latest_baseline(live):
    at = _whatif(_start(live, **REPORTED))
    assert at.session_state[lm.BASE_KEY]["values"] == {"temp_c": 22.0, "humidity_pct": 81.0, "wind_speed_ms": 2.1,
                                                       "wind_from_deg": 159.0}
    assert _controls(at) == {"temp_c": 22.0, "humidity_pct": 81.0, "wind_speed_ms": 2.1, "wind_from_deg": 159.0}
    tags = "\n".join(m.value for m in at.markdown if "lm-ctag" in m.value)
    assert tags.count("REAL BASELINE") == 4 and "WHAT-IF</span>" not in tags


def test_06_unmodified_whatif_follows_a_refresh(live):
    at = _whatif(_start(live, **REPORTED))
    at.feed["weather"] = _weather(**REFRESHED)
    _refresh(at)
    assert at.session_state[lm.BASE_KEY]["values"]["temp_c"] == 22.4                  # baseline
    assert _controls(at) == {"temp_c": 22.4, "humidity_pct": 79.0, "wind_speed_ms": 2.3, "wind_from_deg": 165.0}
    assert _twin_weather(at)["temperature_c"] == 22.4                                  # scenario values
    assert not [i for i in at.info if "differs from the latest baseline" in i.value]


def test_07_refresh_updates_the_baseline_but_keeps_a_modified_scenario(live):
    at = _whatif(_start(live, **REPORTED))
    for k, v in (("temp_c", 35.0), ("humidity_pct", 30.0), ("wind_speed_ms", 8.0), ("wind_from_deg", 225.0)):
        _slider(at, k).set_value(v).run()
    tags = "\n".join(m.value for m in at.markdown if "lm-ctag" in m.value)
    assert tags.count("WHAT-IF</span>") == 4 and "(baseline 22.0°C)" in tags
    at.feed["weather"] = _weather(**REFRESHED)
    _refresh(at)
    assert at.session_state[lm.BASE_KEY]["values"] == {"temp_c": 22.4, "humidity_pct": 79.0, "wind_speed_ms": 2.3,
                                                       "wind_from_deg": 165.0}
    assert _controls(at) == {"temp_c": 35.0, "humidity_pct": 30.0, "wind_speed_ms": 8.0, "wind_from_deg": 225.0}
    assert any("differs from the latest baseline" in i.value for i in at.info)
    caps = [c.value for c in at.caption]
    assert "Temperature — Baseline: 22.4°C / Scenario: 35.0°C / Change: +12.6°C" in caps
    assert _twin_weather(at)["temperature_c"] == 35.0


def test_08_firms_refresh_updates_the_detection_state(live):
    at = _start(live, **REPORTED)
    assert "OBSERVED FIRE DETECTION" not in _md(at)
    at.feed["detections"] = [_detection(*BANDIPUR)]
    _button(at, "Refresh FIRMS").click().run()
    assert not at.exception
    md = _md(at)
    assert "OBSERVED FIRE DETECTION" in md and "Observed fire detections</span><span>1 (NASA FIRMS, in the area)" in md
    t = at.session_state["_wi_twin"]
    assert len(t.ingestion.scenario["observed_hotspots"]) == 1 and t.ingestion.source_status["firms"]["n"] == 1


def test_09_10_zero_firms_is_zero_observed_fire_and_no_synthetic_fire(live):
    at = _start(live, **REPORTED)
    md = _md(at)
    assert "Observed fire detections</span><span>0 (NASA FIRMS, in the area)" in md
    t = at.session_state["_wi_twin"]
    assert t.ingestion.scenario["n_hotspots"] == 0 and t.ingestion.scenario["observed_hotspots"] == []
    assert t.ingestion.last_hotspots.empty                                             # nothing seeded
    cfg, res = _run_live(at)
    assert cfg["live"]["ignition"]["source"] == "NONE" and res is None


def test_11_firms_detection_creates_observed_ignition(live):
    at = _start(live, **REPORTED)
    at.feed["detections"] = [_detection(*BANDIPUR)]
    _refresh(at)
    cfg, res = _run_live(at)
    assert cfg["live"]["ignition"]["source"] == "OBSERVED_FIRMS" and res is not None
    assert res.params["ignition_cells_requested"] == 1


def test_12_owm_failure_keeps_the_last_real_weather(live):
    at = _start(live, **REPORTED)
    at.feed["owm_ok"] = False
    _refresh(at)
    assert _controls(at) == {"temp_c": 22.0, "humidity_pct": 81.0, "wind_speed_ms": 2.1, "wind_from_deg": 159.0}
    assert _twin_weather(at)["temperature_c"] == 22.0                                 # not zero, not a default


def test_13_cached_weather_stays_labelled_cached(live):
    at = _start(live, **REPORTED)
    at.feed["owm_ok"] = False
    _refresh(at)
    at.run()
    md = _md(at)
    assert ">CACHED<" in md and "CACHED: the latest request failed" in md


def test_14_refresh_makes_one_request_and_no_loop(live):
    at = _start(live, **REPORTED)
    n = dict(at.calls)
    at.run()
    at.run()
    assert at.calls == n                                                               # reruns use the cache
    _refresh(at)
    assert at.calls == {"firms": n["firms"] + 1, "owm": n["owm"] + 1}                  # exactly one of each
    at.run()
    assert at.calls == {"firms": n["firms"] + 1, "owm": n["owm"] + 1}


def test_15_live_simulation_config_uses_the_latest_weather(live):
    at = _start(live, **REPORTED)
    at.feed["weather"] = _weather(**REFRESHED)
    _refresh(at)
    _button(at, OPEN).click().run()
    cfg = at.session_state["simulation_config"]
    assert (cfg["temp_c"], cfg["humidity_pct"], cfg["wind_speed_ms"], cfg["wind_from_deg"]) == (22.4, 79, 2.3, 165)
    assert cfg["live"]["weather"]["record"]["temperature_c"] == 22.4
    assert at.session_state["_sim_twin"].ingestion.scenario["point_weather"]["temperature_c"] == 22.4


def test_16_whatif_handoff_receives_the_latest_values(live):
    at = _whatif(_start(live, **REPORTED))
    _slider(at, "temp_c").set_value(35.0).run()
    at.feed["weather"] = _weather(**REFRESHED)
    _refresh(at)
    _button(at, OPEN).click().run()
    lv = at.session_state["simulation_config"]["live"]
    assert lv["baseline"]["values"]["temp_c"] == 22.4 and lv["scenario_values"]["temp_c"] == 35.0
    assert lv["scenario_values"]["humidity_pct"] == 81.0       # a modified scenario is kept as a whole (spec rule)
    assert at.session_state["simulation_config"]["temp_c"] == 35.0


def test_17_no_seeded_synthetic_hotspots_in_live_mode(live):
    at = _start(live, **REPORTED)
    md = _md(at)
    assert "Seeded hotspots" not in md and "(synthetic)" not in md
    _run_live(at)
    md = _md(at)
    assert "Seeded hotspots" not in md and "(synthetic)" not in md
    badge = next(m.value for m in at.markdown if "Fire Spread Simulation" in m.value and "badge" in m.value)
    assert "DEMO" not in badge


def test_18_no_fire_result_panel(live):
    at = _start(live, **REPORTED)
    _, res = _run_live(at)
    assert res is None
    md = _md(at)
    assert "🟢" in md and "NO ACTIVE FIRE DETECTED" in md
    assert "NASA FIRMS returned no fire detections within the selected area/time window." in md
    assert "Current fire-weather conditions: " in md
    assert "No fire spread was simulated because no valid ignition source was detected." in md
    assert "OBSERVED FIRE — SIMULATION ACTIVE" not in md


def test_19_high_risk_without_fire(live):
    at = _start(live, **REPORTED)
    cls = {"n_total": 0, "n_in_area": 0, "n_valid": 0, "n_non_fuel": 0, "n_outside": 0, "n_cells": 0}
    ign = lm.ignition_spec(lm.LIVE, {}, cls)
    s = lm.result_status(lm.LIVE, "live", cls, "EXTREME", ign, ran=False)
    assert s["title"] == "HIGH FIRE-WEATHER RISK — NO ACTIVE FIRE" and s["color"] == "orange"
    assert "🟠" in lm.status_html(s)
    assert s["message"] == ("Current weather conditions indicate elevated fire potential, but NASA FIRMS returned no "
                            "active fire detection. No observed fire spread was simulated.")
    assert "Current fire-weather conditions: EXTREME." in s["notes"]
    # the page uses the same panel; the result badge is never DEMO in LIVE
    badge = next(m.value for m in at.markdown if "Scenario Result" in m.value and "badge" in m.value)
    assert "LIVE REAL-WORLD" in badge and "DEMO" not in badge


def _run_live(at):
    _button(at, OPEN).click().run()
    assert not at.exception
    at.switch_page(SPREAD).run()
    assert not at.exception
    _button(at, "RUN LIVE SIMULATION").click().run()
    assert not at.exception
    return at.session_state["simulation_config"], at.session_state["applied_geo_result"]
