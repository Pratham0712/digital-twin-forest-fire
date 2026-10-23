"""Animation continuity (the map component must not restart a running playback when Streamlit re-sends the
same simulation or re-mounts the component, and must not throttle the fire on isolated slow frames) and
weather-responsive spread through the real pipeline (weather -> FWI codes -> fire-spread model)."""
import re
from pathlib import Path

import pytest

HTML = (Path(__file__).resolve().parents[1] / "src/dashboard/components/fire_map/index.html").read_text(encoding="utf-8")


def _fn(name):
    i = HTML.index(f"function {name}(")
    return HTML[i:HTML.index("\nfunction ", i + 10)]


def test_resent_same_simulation_keeps_the_clock_playback_and_particles():
    ap = _fn("applyPayload")
    assert "simSig(P)===simSig(p)" in ap                                  # same completed run detected
    assert "if(sameSim){simT=keep.t;" in ap and "else{simT=0;" in ap      # clock kept, reset only for a new run
    assert "clearParticles()" in ap.split("else{simT=0;")[1]               # particles cleared only for a new run
    assert "setPlaying(sameSim&&keep.playing)" in ap                       # a running playback keeps running
    assert "p.autoplay&&p.hasRun&&!sameSim" in ap                          # no delayed autoplay restart
    sig = _fn("simSig")
    for k in ("reportId", "endT", "cells", "n_rows", "n_cols"):
        assert k in sig


def test_component_remount_restores_the_running_clock():
    assert "function saveClock()" in HTML and "function loadClock(sg)" in HTML
    assert re.search(r"Date\.now\(\)-c\.ts<60000", HTML)                    # only a fresh, same-run clock
    assert "loop.saveT" in _fn("loop") and "saveClock()" in _fn("loop")
    assert "P.autoplay&&P.hasRun&&!RESTORED" in HTML


def test_clock_advances_from_real_time_and_quality_does_not_oscillate():
    lp = _fn("loop")
    assert "simT=Math.min(ENDT,simT+dt*speed/secPerStep)" in lp            # the clock is the simulation time
    assert "Math.min(dt*1000,34)" in lp                                     # one slow frame cannot throttle the fire
    assert "else if(!playing&&frameMs<15&&RS<1" in lp                       # no canvas resize mid-playback
    rs = _fn("renderState")
    assert "countLE(SORT_IGN,t)" in rs and "key===stateKey" in rs           # surface rebuilt only on a state change


# ── weather-responsive spread (actual calculations) ─────────────────────── #

def _run(t=30.0, rh=40.0, w=5.0, rain=0.0, dur=15):
    import logging
    logging.disable(logging.WARNING)
    from src.dashboard.live_modes import zone_risk
    from src.digital_twin.twin_state import DigitalTwin
    from src.simulation.local_spread import FOCUS_AREAS, FocusArea, run_local_spread
    b = FOCUS_AREAS["Bandipur Tiger Reserve"]
    pw = {"temperature_c": t, "humidity_pct": rh, "wind_speed_ms": w, "wind_deg": 81.0, "pressure_hpa": 1010.0,
          "precipitation_mm": rain, "clouds_pct": 0, "weather_main": "x"}
    tw = DigitalTwin(offline=True, scenario={"point_weather": pw, "weather_mode": "scenario"})
    tw.refresh()
    cond = zone_risk(tw, b.lat, b.lon)
    r = run_local_spread(FocusArea("t", b.lat, b.lon, 500, 500, 25), cond, w, 81.0, duration_minutes=dur, seed=1,
                         placement="Upwind edge", n_ignition=3)
    return cond, r


def _area(r):
    return r.final["fire_area_ha"]


def test_humidity_reduces_fuel_dryness_and_spread():
    (c1, r1), (c2, r2), (c3, r3) = (_run(rh=h) for h in (20, 50, 80))
    assert c1["ffmc"] > c2["ffmc"] > c3["ffmc"]
    assert r1.params["ros_head_m_per_min"] > r2.params["ros_head_m_per_min"] > r3.params["ros_head_m_per_min"]
    assert _area(r1) > _area(r2) >= _area(r3)


def test_temperature_and_wind_increase_spread():
    (_, cool), (_, hot) = _run(t=20), _run(t=42)
    assert hot.params["ros_head_m_per_min"] > cool.params["ros_head_m_per_min"] and _area(hot) > _area(cool)
    (_, calm), (_, windy) = _run(w=1), _run(w=10)
    assert windy.params["ros_head_m_per_min"] > 3 * calm.params["ros_head_m_per_min"] and _area(windy) > _area(calm)


def test_rain_wets_fine_fuels_and_reduces_spread_without_forcing_fire():
    (c0, dry), (c1, wet), (c2, soaked) = _run(rain=0), _run(rain=2), _run(rain=10)
    assert c0["ffmc"] > c1["ffmc"] > c2["ffmc"]
    assert _area(dry) > _area(wet) >= _area(soaked)
    assert soaked.params["clock"]["simulated_s"] == 900                    # clock still runs the full duration


def test_fuel_moisture_code_drives_the_spread_rate_monotonically():
    from src.simulation.fire_behaviour import spread_rates
    heads = [spread_rates(f, 40.0, 5.0, 270.0).head for f in (70, 80, 86, 90, 94)]
    assert all(b > a for a, b in zip(heads, heads[1:]))
