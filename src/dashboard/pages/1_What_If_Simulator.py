"""
What-If Simulator - dedicated page.

Real-data mode (sidebar Offline / demo mode OFF), two clearly separated modes,
both starting from the latest real OpenWeatherMap + NASA FIRMS observations at
the selected location (see src/dashboard/live_modes.py):
  * LIVE REAL-WORLD SIMULATION - weather locked to the observation; only valid
    observed FIRMS detections can ignite.
  * LIVE DATA -> WHAT-IF - the observation is the REAL BASELINE; the user edits
    the controls (SCENARIO INPUT); ignition is observed, hypothetical or none.
Offline / demo mode ON: the original synthetic What-If sliders and presets
(labelled DEMO), no API calls.

Risk (XGBoost + FWI) is recomputed by the unchanged DigitalTwin scenario
mechanism; the Spread Simulation receives the full set-up via the hand-off.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

import streamlit as st

from src.dashboard.dashboard_common import (
    set_page, build_sidebar, get_twin, render_header, render_kpi_row,
    render_risk_gauge, render_alerts, render_ca_simulation, render_scenario_controls, SYSTEM,
    log_action,
)

from src.auth.auth_gate import require_login, render_user_badge_in_sidebar
from src.dashboard import live_modes as lm
from src.dashboard.geo_spread import (apply_and_open_spread, focus_from_setup, get_setup, live_offline,
                                     render_setup_controls, render_setup_map, render_setup_summary)
from src.dashboard.ui.live_panel import live_observations, map_hotspots, render_live_conditions
from src.data_ingestion.live_point import hotspot_markers

set_page("What-If Simulator")
require_login()

# The Offline / demo switch is shown here too: it decides between the live modes
# (real data) and the original synthetic What-If (demo).
_, _, region, _ = build_sidebar(show_offline=True)
render_user_badge_in_sidebar()
demo = live_offline()

st.markdown(f"""
<div class="hero">
  <div class="eyebrow">Forest Fire Digital Twin</div>
  <h1>What-If Scenario Simulator</h1>
  <div class="sub">{'Start from the latest <b>real</b> OpenWeatherMap weather and NASA FIRMS detections at the selected '
                    'location: evaluate current conditions (LIVE REAL-WORLD), or load them as a baseline and change '
                    'them to test a hypothetical scenario (LIVE DATA → WHAT-IF). Fire is only simulated from an '
                    'observed NASA FIRMS detection or an ignition you place, labelled HYPOTHETICAL.'
                    if not demo else
                    f'DEMO / OFFLINE MODE: adjust synthetic weather conditions, or click a preset, and watch fire '
                    f'risk recompute across every grid zone of <b>{region.name}</b>. Demo inputs are synthetic; '
                    f'the live APIs are not called.'}</div>
</div>
""", unsafe_allow_html=True)

mode = lm.render_mode_selector(demo)                  # SIMULATION MODE [ LIVE REAL-WORLD ] [ LIVE DATA → WHAT-IF ]

region_sig = (region.name, region.min_lat, region.max_lat, region.min_lon, region.max_lon)
obs = ctl = None
w = ws = d = fs = None
if mode == lm.DEMO:
    st.markdown('<div class="sec-hdr">Scenario parameters</div>', unsafe_allow_html=True)
    st.caption(
        "Click a preset to instantly load and apply a known weather pattern, or drag the sliders "
        "yourself and press **Apply scenario** when ready."
    )
    scenario, just_applied_preset = render_scenario_controls("wi")
else:
    # Location first (it persists across reruns), then the real observations there,
    # then the controls populated from them.
    _setup0 = st.session_state.get("sim_setup")
    if not _setup0 or _setup0.get("region") != region.name:
        _base = st.session_state.get("_wi_twin")
        if _base is None or st.session_state.get("_wi_region") != region_sig:
            with st.spinner("Preparing the region..."):
                _base = get_twin(offline=True, scenario=None, region=region)   # only for the default location
                _base.refresh()
        _setup0 = get_setup(_base, kp="wi_setup")
    _loc0 = dict(_setup0["location"])
    w, ws, d, fs = live_observations(_loc0["lat"], _loc0["lon"], "wi", False)
    obs = lm.baseline_from_observation(w, ws, _loc0["lat"], _loc0["lon"])
    st.markdown(f'<div class="sec-hdr">{"Live conditions" if mode == lm.LIVE else "What-If parameters"} · '
                f'{_loc0.get("name", "")}</div>', unsafe_allow_html=True)
    ctl = lm.render_live_controls(mode, obs)
    _extra = ((ctl["baseline"] if mode == lm.WHATIF else obs) or {}).get("extra")
    scenario = lm.twin_scenario(mode, ctl["values"], _extra, ws.get("mode"), d, fs.get("mode"))
    just_applied_preset = False


def _recompute(log: bool):
    with st.spinner("Recomputing risk across the region for this scenario..."):
        wi_twin = get_twin(offline=True, scenario=scenario, region=region)
        wi_twin.refresh()
    st.session_state["_wi_twin"] = wi_twin
    st.session_state["_wi_scenario"] = scenario
    st.session_state["_wi_region"] = region_sig
    if log:
        _s = wi_twin.get_summary()
        log_action("scenario", f"What-If applied ({lm.MODE_NAMES[mode]}): {scenario.get('temp_c')}°C, "
                   f"{scenario.get('wind_speed_ms')} m/s wind from {scenario.get('wind_from_deg')}°, "
                   f"{scenario.get('humidity_pct')}% humidity, "
                   f"{scenario.get('n_hotspots')} seeded hotspots -> {_s.get('total_alerts', 0)} alerts, "
                   f"peak risk {_s.get('max_risk_score', 0):.0%}", wi_twin.region.name)
    # A new scenario invalidates any spread simulation run from the old one.
    st.session_state["wi_ca_history"] = None


if (just_applied_preset or "_wi_twin" not in st.session_state
        or st.session_state.get("_wi_scenario") != scenario
        or st.session_state.get("_wi_region") != region_sig):
    _recompute(log=just_applied_preset)
twin = st.session_state["_wi_twin"]

# ── simulation set-up: location, area, cell size, duration, ignition ──
st.markdown('<div class="sec-hdr">Simulation set-up</div>', unsafe_allow_html=True)
setup = get_setup(twin, kp="wi_setup")
setup_error = render_setup_controls(twin, setup, "wi_setup", ignition_mode=None if demo else mode)
if mode != lm.DEMO:
    _l = setup["location"]
    if abs(_l["lat"] - _loc0["lat"]) > 1e-9 or abs(_l["lon"] - _loc0["lon"]) > 1e-9:
        st.rerun()                                   # new location: fetch its observations first
plan = risk = None
markers = []
if setup_error:
    st.error(setup_error)
else:
    render_setup_summary(setup, twin)
    _loc = setup["location"]
    if mode == lm.DEMO:
        # Real observations are not requested in demo mode (cards say so).
        _live = live_observations(_loc["lat"], _loc["lon"], "wi", True)
        _hot, _kind, _hsum = map_hotspots(_live[2], _live[3])
        render_setup_map(setup, "wi_setup", twin, float(scenario["wind_speed_ms"]), float(scenario["wind_from_deg"]),
                         hotspots=_hot, hotspot_kind=_kind, hotspot_summary=_hsum)
        render_live_conditions(_loc["lat"], _loc["lon"], _loc.get("name", ""), "wi", True, scenario=scenario,
                               data=_live)
    else:
        markers = hotspot_markers(d) if fs.get("mode") in ("live", "cached") else []
        risk = lm.zone_risk(twin, _loc["lat"], _loc["lon"])
        plan = lm.plan_ignition(mode, setup, focus_from_setup(setup), risk, markers, fs.get("mode"))
        status = lm.assess(mode, fs.get("mode"), plan["cls"], risk["severity"], plan["ign"],
                           weather_ok=ctl["complete"])
        st.markdown(lm.status_html(status, plan["ign"]["label"]), unsafe_allow_html=True)
        _hot, _kind, _hsum = map_hotspots(d, fs)
        _v = ctl["values"]
        render_setup_map(setup, "wi_setup", twin, float(_v.get("wind_speed_ms") or 0.0),
                         float(_v.get("wind_from_deg") or 0.0), hotspots=_hot, hotspot_kind=_kind,
                         hotspot_summary=_hsum, ignition=plan["ign"], land=plan["land"],
                         whatif=mode == lm.WHATIF)
        cls = plan["cls"]
        render_live_conditions(_loc["lat"], _loc["lon"], _loc.get("name", ""), "wi", False, data=(w, ws, d, fs),
                               detections=markers, refresh_all=True,
                               firms_rows=[("In simulation area", f"{cls['n_in_area']} · valid ignitions "
                                                                  f"{cls['n_valid']} ({cls['n_cells']} cell(s))")],
                               scenario_card=lm.scenario_card(mode, ctl["values"], ctl["changes"], ctl["baseline"],
                                                              {"status": fs.get("mode"), "n_in_area": cls["n_in_area"]},
                                                              plan["ign"]))

blocked = bool(setup_error)
if mode == lm.LIVE and ctl is not None and not ctl["complete"]:
    st.warning("OPENWEATHERMAP UNAVAILABLE: the LIVE simulation needs real temperature, humidity, wind speed and "
               "wind direction. Nothing is estimated - refresh the live data, or use LIVE DATA → WHAT-IF to supply "
               "scenario values.")
    blocked = True
b_apply, b_open = st.columns(2)
run = b_apply.button("Re-evaluate live conditions" if mode == lm.LIVE else "Apply scenario", use_container_width=True)
open_spread = b_open.button("Apply Scenario & Open Spread Simulation", type="primary",
                            use_container_width=True, disabled=blocked)
if run or open_spread:
    _recompute(log=True)
    twin = st.session_state["_wi_twin"]
if open_spread:
    # Same twin, same scenario, same set-up: Module 2 receives exactly what was configured here.
    live = None
    if mode != lm.DEMO and plan is not None:
        risk = lm.zone_risk(twin, setup["location"]["lat"], setup["location"]["lon"])
        live = lm.live_handoff(mode, obs, ctl["baseline"], ctl["values"], ctl["changes"], w, ws, fs, markers,
                               plan, risk)
    apply_and_open_spread(twin, scenario, setup, live)
snap = twin.current_snapshot
summary = twin.get_summary()

render_header(summary, True, "Scenario Result", region=twin.region,
              badge={lm.LIVE: ("badge-live", "LIVE REAL-WORLD"), lm.WHATIF: ("badge-demo", "WHAT-IF / SIMULATED")}.get(mode))
render_kpi_row(summary)

# ── plain-language interpretation of the scenario ──
n_hotspots, temp_c = scenario["n_hotspots"], scenario["temp_c"]
wind_speed_ms, humidity_pct = scenario["wind_speed_ms"], scenario["humidity_pct"]
max_r = summary.get("max_risk_score", 0)
mean_r = summary.get("mean_risk_score", 0)
alerts_n = summary.get("total_alerts", 0)
if max_r >= 0.6:
    verdict = "EXTREME conditions"
    verdict_color = "#ff6b4a"
elif max_r >= 0.4:
    verdict = "elevated risk conditions"
    verdict_color = "#fbbf24"
else:
    verdict = "low risk conditions"
    verdict_color = "#34d399"

driver_bits = []
if humidity_pct is not None and humidity_pct < 20:
    driver_bits.append(f"very low humidity ({humidity_pct:.0f}%)")
if wind_speed_ms is not None and wind_speed_ms > 10:
    driver_bits.append(f"strong wind ({wind_speed_ms:.1f} m/s)")
if temp_c is not None and temp_c > 38:
    driver_bits.append(f"high temperature ({temp_c:.1f}°C)")
if n_hotspots > 15:
    driver_bits.append(f"{n_hotspots} seeded active hotspots")
driver_txt = ", ".join(driver_bits) if driver_bits else "a combination of the selected parameters"


def _v(x, fmt, unit):
    return "unavailable" if x is None else f"{format(float(x), fmt)}{unit}"


if mode == lm.DEMO:
    what = (f"This scenario (temperature {temp_c}°C, wind {wind_speed_ms:.1f} m/s, humidity {humidity_pct}%, "
            f"{n_hotspots} seeded hotspots)")
else:
    n_obs = len(scenario.get("observed_hotspots") or [])
    what = (("Current observed conditions" if mode == lm.LIVE else "This WHAT-IF scenario")
            + f" (temperature {_v(temp_c, '.1f', '°C')}, wind {_v(wind_speed_ms, '.1f', ' m/s')}, humidity "
              f"{_v(humidity_pct, '.0f', '%')}, {n_obs} real NASA FIRMS detection(s) nearby)")
st.markdown(f"""
<div class="briefing-card">
  <h4 style="color:{verdict_color} !important;">{'Scenario interpretation' if mode != lm.LIVE else 'Current-conditions interpretation'}</h4>
  {what} produce{'s' if mode == lm.DEMO else ''} <b style="color:{verdict_color}">{verdict}</b> across the
  region — {alerts_n} of {summary.get('total_zones', 0)} zones cross the
  {SYSTEM.alert_threshold_pct:.0f}% alert threshold, with peak risk at {max_r:.0%} and a regional
  mean of {mean_r:.0%}. The primary driver{'s are' if len(driver_bits) > 1 else ' is'} {driver_txt}.
  {'' if mode == lm.DEMO else '<br><small>Model risk (MODEL PREDICTION) is not a fire: fire is only simulated from an '
   'observed NASA FIRMS detection or a HYPOTHETICAL ignition you provide.</small>'}
</div>
""", unsafe_allow_html=True)

# (The regional CARTO risk map that used to sit here was removed: the Google
# satellite map above is the one map of the simulation location on this page.)
col_gauge, col_alerts = st.columns([1, 2], gap="medium")
with col_gauge:
    render_risk_gauge(summary)
with col_alerts:
    render_alerts(snap.alerts, summary, limit=8)

st.markdown("---")
if mode == lm.DEMO:
    render_ca_simulation(twin, key_prefix="wi_ca", allow_force_ignite=True)
elif mode == lm.LIVE:
    st.markdown('<div class="info-box">The regional 2-hour projection ignites the model\'s HIGH/EXTREME risk zones, '
                'which are predictions, not observed fires, so it is not run in LIVE mode. Fire spread at the '
                'selected location starts only from observed NASA FIRMS detections (Spread Simulation).</div>',
                unsafe_allow_html=True)
else:
    with st.expander("Regional 2-hour projection — HYPOTHETICAL IGNITION at the model's HIGH/EXTREME zones"):
        st.caption("WHAT-IF only: this projection ignites the zones the model rates HIGH/EXTREME under your scenario. "
                   "Those ignitions are HYPOTHETICAL, not observed fires.")
        render_ca_simulation(twin, key_prefix="wi_ca", allow_force_ignite=False)

st.caption(
    "This page runs an isolated scenario twin, separate from the Command Center's live/demo "
    "state; switching pages will not lose your main session."
)
