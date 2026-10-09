"""Final implementation: fire geometry / rendering, movable domain, Run speed,
PDF / DOCX reports and the opt-in emergency-alert workflow. Offline only: the
suite's conftest refuses every HTTP request; providers run in mock or
unconfigured mode, so no real message can be sent."""
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from src.simulation.local_spread import (DOMAIN_MAP, FOCUS_AREAS, FocusArea, domain_from_setup, initial_domain,
                                         run_local_spread)
from src.simulation.cellular_automata import CellState

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "src" / "dashboard" / "components" / "fire_map" / "index.html").read_text(encoding="utf-8")
BP = FOCUS_AREAS["Bandipur Tiger Reserve"]
HOT = {"ffmc": 95.0, "bui": 80.0, "ndvi": 0.6, "risk_score": 0.8, "zone_id": "Z1"}
COOL = {"ffmc": 75.0, "bui": 15.0, "ndvi": 0.6, "risk_score": 0.1, "zone_id": "Z1"}


def _run(cond=HOT, ws=8.0, wd=241.0, dur=60, **kw):
    f = kw.pop("focus", FocusArea("Bandipur Tiger Reserve", BP.lat, BP.lon, 1000, 1000, 25))
    return run_local_spread(f, cond, ws, wd, placement="Centre", n_ignition=1, duration_minutes=dur, seed=5, **kw)


def _setup(**kw):
    s = {"location": {"name": BP.name, "lat": BP.lat, "lon": BP.lon, "source": "preset"}, "width_m": 1000.0,
         "height_m": 1000.0, "cell_m": 25.0, "duration_min": 60.0, "placement": "Map points", "n_ignition": 5,
         "ignition_points": [], "ignition_source": "hypothetical", "layers": {"grid": True, "boundary": True},
         "domain_preset": "Auto (focus + duration margin)", "max_extent_m": 15000.0}
    s.update(kw)
    return s


# ── fire geometry and rendering ─────────────────────────────────────────────
def test_fire_region_is_connected_and_burning_cells_match_burned_area():
    from scipy import ndimage
    r = _run()
    for k in (15, 30, len(r.history) - 1):
        st = r.history[k].state
        aff = (st == CellState.BURNING) | (st == CellState.BURNED)
        # the ROS stencil reaches 2 cells (knight / (3,1) moves) and the patchier fuel field can leave a
        # one-cell unburned gap, so connectivity is checked at the stencil reach (no detached strips)
        _, n = ndimage.label(ndimage.binary_dilation(aff, np.ones((3, 3))), structure=np.ones((3, 3)))
        assert n == 1
        t = k * r.step_minutes
        ign = (r.ignition_time >= 0) & (r.ignition_time <= t + 1e-9)
        assert np.array_equal(aff, ign)                                   # burning + burned = cells ignited so far


def test_flaming_band_is_deep_at_the_head_and_thin_at_the_back():
    r = _run(ws=10.0, wd=270.0, dur=40)
    st = r.history[-1].state
    rr, cc = np.nonzero(st == CellState.BURNING)
    r0, c0 = np.argwhere(r.ignition_step == 0)[0]
    east = cc[cc > c0 + 5]
    west = cc[cc < c0]
    assert east.size > 3 * max(1, west.size)            # flames concentrate on the wind-driven head (east), not a ring


def test_renderer_layers_glow_flames_smoke_and_toggles():
    assert "glowing fire line along the active front" not in HTML        # no thin ribbon outline
    for k in ("fire", "smoke", "embers", "ash", "heat", "grid", "boundary", "front", "wind", "veg"):
        assert f'data-t="{k}"' in HTML                                     # every layer has its own toggle
    assert "if(show.fire&&L>=1&&fs>0.1){" in HTML                          # flames at normal zoom
    assert "if(show.heat){ctx.globalCompositeOperation='lighter';" in HTML
    assert "p.x+=(wx*DK*tk*k+tu)*dt;p.y+=(wy*DK*tk*k+tv)*dt;" in HTML       # smoke drifts with the wind vector
    assert "to=(w[1]+180)*Math.PI/180" in HTML                             # downwind = FROM + 180 (wind agrees)


# ── movable / resizable domain (independent of the focus area) ─────────────
def test_domain_drawn_on_map_is_independent_of_the_focus_area():
    from src.dashboard.geo_spread import apply_map_event
    s = _setup()
    box = {"north": BP.lat + 0.03, "south": BP.lat + 0.006, "east": BP.lon + 0.01, "west": BP.lon - 0.01}
    assert apply_map_event({"kind": "domain", **box}, s, None)
    assert s["domain_preset"] == DOMAIN_MAP and s["width_m"] == 1000.0           # focus unchanged
    d = domain_from_setup(s)
    assert abs(d.bounds["north"] - box["north"]) < d.focus.dlat and abs(d.bounds["west"] - box["west"]) < d.focus.dlon
    assert not d.focus_mask().any()                                                # moved outside the focus area
    apply_map_event({"kind": "area", "lat": BP.lat - 0.002, "lon": BP.lon, "width_m": 1500, "height_m": 800}, s, None)
    d2 = domain_from_setup(s)
    assert abs(d2.bounds["north"] - d.bounds["north"]) < d.focus.dlat                # resizing the focus kept the domain
    # ignition inside that domain (outside the focus) is accepted and burns; outside it is rejected
    from src.dashboard.geo_spread import HYP_OUTSIDE
    msgs = []
    p_in = [BP.lat + 0.018, BP.lon]
    apply_map_event({"kind": "ignite", "points": [p_in, [BP.lat - 0.02, BP.lon]]}, s, None, whatif=True,
                    messages=msgs)
    assert s["ignition_points"] == [p_in] and msgs == [HYP_OUTSIDE]
    dom = domain_from_setup(s, s["ignition_points"])
    res = run_local_spread(dom.focus, HOT, 6.0, 270.0, placement="Map points", ignition_points=[p_in],
                           strict_points=True, duration_minutes=20, domain=dom)
    assert res.ignition_step[res.domain.cell_of(*p_in)] == 0 and res.final["fire_area_ha"] > 0.1
    assert "kind:'domain'" in HTML and "applyEditMode" in HTML and 'data-m="domain"' in HTML


def test_hypothetical_ignition_on_mapped_road_is_accepted_and_labelled():
    from src.dashboard.geo_spread import apply_map_event
    from src.simulation.ignition_site import IgnitionSiteClassifier
    road = {"type": "way", "tags": {"highway": "secondary"},
            "geometry": [{"lat": BP.lat, "lon": BP.lon - 0.01}, {"lat": BP.lat, "lon": BP.lon + 0.01}]}
    s, notes, msgs = _setup(), [], []
    apply_map_event({"kind": "ignite", "points": [[BP.lat, BP.lon]]}, s, None, whatif=True,
                    land=IgnitionSiteClassifier([road], True, "OpenStreetMap", "2026-10-09"), messages=msgs, notes=notes)
    assert s["ignition_points"] == [[BP.lat, BP.lon]] and msgs == []
    assert "ON MAPPED ROAD" in notes[0] and s["ignition_checks"][0]["land_cover"] == "known (mapped)"


# ── Run speed ───────────────────────────────────────────────────────────────
def test_run_fetches_terrain_once_and_makes_no_api_calls(monkeypatch):
    import src.dashboard.geo_spread as gs
    calls = []
    real = gs.domain_elevation

    def counting(dom, allow_fetch=True):
        calls.append(allow_fetch)
        return real(dom, allow_fetch=False)
    monkeypatch.setattr(gs, "domain_elevation", counting)
    s = _setup(placement="Centre", n_ignition=3)
    f = gs.focus_from_setup(s)
    res = gs._run_simulation(f, HOT, s, None, 1, 8.0, 241.0, None)
    assert calls.count(True) == 1                    # one (cached) terrain lookup for the run, none mid-run
    assert res.params["report_id"].startswith("FFDT-") and res.final["minutes"] == 60.0


def test_payload_is_reused_across_reruns(monkeypatch):
    import src.dashboard.geo_spread as gs
    import streamlit as st
    r = _run(dur=20)
    n = {"k": 0}
    real = gs.sim_payload

    def counting(*a, **k):
        n["k"] += 1
        return real(*a, **k)
    monkeypatch.setattr(gs, "sim_payload", counting)
    args = (r.focus, 20, r, 8.0, 241.0, [], "observed", 1, "Centre", [], {})
    a = gs._cached_sim_payload("t", *args, autoplay=True, api_key="K", map_id="")
    b = gs._cached_sim_payload("t", *args, autoplay=True, api_key="K", map_id="")
    assert a is b and n["k"] == 1


def test_dem_fetch_has_a_time_budget():
    import inspect
    from src.data_ingestion import terrain
    from src.simulation import local_spread as ls
    assert "deadline_s" in inspect.signature(terrain.fetch_elevations).parameters
    assert ls.DEM_FETCH_BUDGET_S <= 15 and ls.MAX_DEM_POINTS <= 1000


# ── reports ─────────────────────────────────────────────────────────────────
def _snapshot(mode="WHAT-IF", live=None, plan=None, dur=60):
    from src.reports.simulation_report import build_snapshot
    r = _run(dur=dur)
    r.params["report_id"] = "FFDT-TEST-1"
    setup = {"location": {"name": BP.name, "lat": BP.lat, "lon": BP.lon}, "placement": "Centre",
             "ignition_points": []}
    sc = {"temp_c": 38, "humidity_pct": 18, "wind_speed_ms": 8, "wind_from_deg": 241} if mode != "LIVE" else None
    return r, build_snapshot(r, HOT, setup, mode, sc, live, None, plan)


def test_pdf_and_docx_are_generated_from_the_snapshot():
    from src.reports.simulation_report import render_docx, render_html, render_pdf
    r, s = _snapshot()
    json.dumps(s)                                                       # reproducible / storable
    assert s["results"]["burned_ha"] == r.final["burned_ha"] and s["series"][-1]["minutes"] == 60.0
    pdf = render_pdf(s)
    assert pdf[:5] == b"%PDF-" and len(pdf) > 20_000
    doc = zipfile.ZipFile(io.BytesIO(render_docx(s))).read("word/document.xml").decode()
    assert "FFDT-TEST-1" in doc and "SIMULATED" in doc and "SIMULATION ONLY" in doc
    html = render_html(s)
    assert "window.print()" in html and "Executive summary" in html


def test_report_states_missing_data_and_claims_no_observed_fire():
    from src.reports.simulation_report import charts, sections
    r, s = _snapshot()
    txt = json.dumps(sections(s))
    assert "NASA FIRMS: not part of this run" in txt and "no observed fire is claimed" in txt
    assert "Historical fire records are not available" in txt          # no twin -> stated as missing
    assert "confirmed fire" not in txt.replace("not a confirmed fire", "")
    one = dict(s, series=s["series"][:1])
    assert charts(one) == {}                                          # no chart from a single recorded value


def test_live_report_labels_firms_as_observations():
    from src.reports.simulation_report import sections
    live = {"mode": "live", "weather": {"status": "live", "record": {"temperature_c": 31, "humidity_pct": 40,
                                                                      "wind_speed_ms": 4, "wind_deg": 200},
                                        "fetched_utc": "2026-10-09T06:00:00+00:00"},
            "firms": {"status": "live", "n": 2, "fetched_utc": "2026-10-09T06:00:00+00:00"},
            "detections": [{"lat": BP.lat, "lon": BP.lon, "confidence": "h", "satellite": "N20"}]}
    plan = {"cls": {"n_in_area": 1, "n_valid": 1}, "ign": {"points": [[BP.lat, BP.lon]]}}
    _, s = _snapshot("LIVE", live, plan)
    txt = json.dumps(sections(s))
    assert "satellite OBSERVATIONS, not ground-verified" in txt and s["weather"]["label"].startswith("OBSERVED")


# ── emergency alerts ────────────────────────────────────────────────────────
@pytest.fixture()
def alerts_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'alerts.db'}")
    for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN",
              "TWILIO_FROM_NUMBER", "ALERT_PROVIDER_MODE"):
        monkeypatch.delenv(k, raising=False)
    from src.notifications import alert_store as store
    return store


def _recipient(store, verified=True, **kw):
    rid = store.add_recipient(kw.get("name", "Range Officer"), "Karnataka Forest Dept", "RFO",
                              "rfo@example.org", "+919800000001", {"email": True, "sms": True})
    if verified:
        store.update_recipient(rid, email_verified=1, phone_verified=1)
    return rid


def test_no_send_without_confirmation_or_authorisation(alerts_db):
    from src.notifications import emergency as em
    _, s = _snapshot(dur=10)
    rid = _recipient(alerts_db)
    with pytest.raises(PermissionError):
        em.send_alert(s, [(rid, "email")], "admin", "admin", confirmed=False)
    with pytest.raises(PermissionError):
        em.send_alert(s, [(rid, "email")], "viewer1", "viewer", confirmed=True)
    assert alerts_db.alert_history() == []                            # nothing recorded, nothing sent


def test_whatif_wording_mock_provider_audit_and_duplicates(alerts_db, monkeypatch):
    from src.notifications import emergency as em
    monkeypatch.setenv("ALERT_PROVIDER_MODE", "mock")
    _, s = _snapshot(dur=10)
    rid = _recipient(alerts_db)
    msg = em.build_message(s, "ALERT-X")
    assert msg["body"].startswith(em.SIM_ONLY) and "SIMULATION ONLY" in msg["subject"]
    out = em.send_alert(s, [(rid, "email"), (rid, "sms")], "admin", "admin", confirmed=True)
    assert out["alert_type"] == "SIMULATION ONLY" and out["counts"] == {"mock": 2}   # mock is never "delivered"
    again = em.send_alert(s, [(rid, "email")], "admin", "admin", confirmed=True)
    assert again["counts"] == {"duplicate": 1}                          # duplicate-send protection
    hist = alerts_db.alert_history()
    assert len(hist) == 2 and {d["status"] for d in hist[1]["deliveries"]} == {"mock"}
    assert hist[0]["sender"] == "admin" and hist[0]["report_id"] == "FFDT-TEST-1"
    assert alerts_db.unacked_inapp("someone")[0]["alert_type"] == "SIMULATION ONLY"   # receiver-side in-app alert


def test_missing_credentials_unverified_and_partial_failures(alerts_db):
    from src.notifications import emergency as em
    from src.notifications.providers import SendResult
    _, s = _snapshot(dur=10)
    ok_r = _recipient(alerts_db, name="A")
    unv = _recipient(alerts_db, verified=False, name="B")
    out = em.send_alert(s, [(ok_r, "email"), (unv, "email")], "admin", "admin", confirmed=True)
    st = {x["name"]: x["status"] for x in out["results"]}
    assert st == {"A": "not_configured", "B": "failed"}                 # no SMTP creds; B not verified

    class Boom:
        def send(self, *a, **k):
            return SendResult("failed", "smtp", error="SMTPServerDisconnected")

    class Fine:
        def send(self, *a, **k):
            return SendResult("queued", "twilio", "SM123")
    out = em.send_alert(dict(s, report_id="FFDT-TEST-2"), [(ok_r, "email"), (ok_r, "sms")], "admin", "admin",
                        confirmed=True, email_provider=Boom(), sms_provider=Fine())
    assert out["counts"] == {"failed": 1, "queued": 1}                  # partial outcome reported truthfully
    assert em.describe_status("queued").startswith("queued") and "not" in em.describe_status("accepted")


def test_observation_based_wording_only_with_valid_firms(alerts_db):
    from src.notifications import emergency as em
    _, s = _snapshot(dur=10)
    assert em.alert_type_of(dict(s, mode="LIVE")) == "SIMULATION ONLY"          # LIVE without valid detections
    obs = dict(s, mode="LIVE", firms={"n_valid": 2, "latest_acq_utc": None, "fetched_utc": None})
    m = em.build_message(obs, "ALERT-Y")
    assert em.alert_type_of(obs) == "OBSERVATION-BASED" and "not independently ground-verified" in m["body"]
    assert em.SIM_ONLY not in m["body"]


def test_recipient_verification_codes(alerts_db):
    rid = _recipient(alerts_db, verified=False)
    code = alerts_db.create_verification(rid, "email")
    assert not alerts_db.confirm_verification(rid, "email", "000000" if code != "000000" else "111111")
    assert alerts_db.confirm_verification(rid, "email", code)
    r = alerts_db.get_recipient(rid)
    assert r["email_verified"] == 1 and r["phone_verified"] == 0
    assert alerts_db.mask("rfo@example.org") == "rf***@example.org" and alerts_db.mask("+919800000001") == "***001"


def test_receiver_alarm_controls():
    src = (ROOT / "src" / "dashboard" / "ui" / "alerts_ui.py").read_text(encoding="utf-8")
    assert "Mute alert sound" in src and "Test alarm (no message is sent)" in src and "Acknowledge" in src
    assert "each alert sounds once per browser session" in src


def test_cell_states_render_as_one_contoured_ground_layer_not_tiles():
    from pathlib import Path
    html = Path("src/dashboard/components/fire_map/index.html").read_text(encoding="utf-8")
    assert '<canvas id="statefx"' in html and "function renderState(" in html and "function mapGround(" in html
    assert "quad(i)" not in html                                         # no per-cell square overlay any more
    for col in ("o[k]=80;o[k+1]=100;o[k+2]=140", "0.30*a", "Math.round(30+225*f)", "Math.round(10+70*f)",
                "0.82*(1+0.2*n)*(1-f)+0.65*0.5*f"):
        assert col in html                                               # CA state colours kept
    assert "x=clamp(x,cx-CELL*0.47" not in html                          # flame bases no longer square-clamped
    assert "drawState();" in html and "mapGround(ctxT,stAll,bx)" in html    # same ground mapping in 2D and tilted 3D


def test_light_emitters_are_screen_blended_above_the_ground_layers():
    from pathlib import Path
    html = Path("src/dashboard/components/fire_map/index.html").read_text(encoding="utf-8")
    assert "#lightfx{z-index:7;mix-blend-mode:screen}" in html and '<canvas id="lightfx"' in html
    assert "#statefx{z-index:5}" in html                                 # state surface stays below flames / glow
    d = html[html.index("function draw(t,now){"):html.index("// vegetation representation")]
    i_glow, i_fl, i_ash = d.index("// fire glow + fire bed"), d.index("// flames: tapered"), d.index("// ash: small")
    assert d.rfind("ctx=ctxL;", 0, i_glow) > d.rfind("ctx=ctxFx;", 0, i_glow)   # glow + flames go to the light layer
    assert d.rfind("ctx=ctxFx;", 0, i_ash) > i_fl                       # smoke / ash stay on the normal layer


def test_ash_layers_use_one_fine_ground_mesh_for_every_tilt():
    from pathlib import Path
    html = Path("src/dashboard/components/fire_map/index.html").read_text(encoding="utf-8")
    g = html[html.index("function mapGround(g,img,box){"):html.index("// ── CA cell-state layer as ONE")]
    assert "if(TILT<0.05){" in g                                         # affine only when exactly top-down
    assert "Math.ceil((bx1-bx0)/(2*CELL))" in g                          # ~2-cell triangles: no drift off terrain
    assert "mapGround(ctxS,scar,FIRE_BOX)" in html                       # burn scar uses the same mesh as the ash
