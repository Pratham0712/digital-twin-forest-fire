"""Performance changes must not change results: cached report charts are byte-identical, the export panel builds
the report once per rerun, FIRMS / weather run side by side with identical output, the Historical map keeps its
positions, alert history returns the same rows with one query, and the alert thumbnail is the same image.
Offline: no network, temporary databases."""
from __future__ import annotations

import inspect
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def snap():
    from tests.test_duration_report_alerts import _snap
    return _snap(15)[1]


# ── report charts ─────────────────────────────────────────────────────────

def test_cached_charts_are_identical_and_shared_by_all_formats(snap, monkeypatch):
    from src.reports import simulation_report as rep
    rep._CHART_CACHE = None
    fresh = rep._draw_charts(snap)
    assert fresh and rep.charts(snap) == fresh and rep.charts(snap) == fresh       # same bytes, from cache
    calls = []
    real = rep._draw_charts
    monkeypatch.setattr(rep, "_draw_charts", lambda s: calls.append(1) or real(s))
    rep._CHART_CACHE = None
    html1 = rep.render_html(snap)
    rep.render_pdf(snap); rep.render_docx(snap)
    assert len(calls) == 1                                                          # drawn once for 3 formats
    assert rep.render_html(snap) == html1                                           # report unchanged


def test_chart_cache_never_mixes_simulations_and_is_bounded(snap):
    from src.reports import simulation_report as rep
    other = dict(snap, report_id="FFDT-OTHER", results=dict(snap["results"], burned_ha=snap["results"]["burned_ha"] + 5))
    assert rep._chart_key(snap) != rep._chart_key(other)
    assert rep._chart_key(snap) != rep._chart_key(dict(snap, generated_utc="2030-01-01T00:00:00+00:00"))
    rep._CHART_CACHE = None
    for i in range(rep._CHART_CACHE_MAX + 5):
        rep._CHART_CACHE = rep._CHART_CACHE or __import__("collections").OrderedDict()
        rep._CHART_CACHE[f"k{i}"] = {}
        while len(rep._CHART_CACHE) > rep._CHART_CACHE_MAX:
            rep._CHART_CACHE.popitem(last=False)
    rep.charts(snap)
    assert len(rep._CHART_CACHE) <= rep._CHART_CACHE_MAX


def _menu_app(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'x.db'}")
    st_testing = pytest.importorskip("streamlit.testing.v1")
    page = """
import streamlit as st
from src.dashboard.ui import alerts_ui
from tests.test_duration_report_alerts import _snap
if "snap" not in st.session_state:
    st.session_state["snap"] = _snap(15)[1]
alerts_ui._export_menu(st.session_state["snap"], "t")
"""
    return st_testing.AppTest.from_string(page, default_timeout=120)


def test_export_menu_builds_nothing_until_a_format_is_chosen(snap, tmp_path, monkeypatch):
    from src.reports import simulation_report as rep
    calls = []
    for fn in ("render_pdf", "render_docx", "render_html", "_draw_charts"):
        real = getattr(rep, fn)
        monkeypatch.setattr(rep, fn, (lambda f, r: (lambda s: calls.append(f) or r(s)))(fn, real))
    at = _menu_app(monkeypatch, tmp_path)
    at.run(); at.run()                                                              # open / rerun the page
    assert not at.exception and calls == []                                         # no document, no chart
    labels = [b.proto.label for b in at.get("download_button")]
    assert labels == ["PDF (.pdf)", "Word (.docx)"]                                 # exactly two formats
    assert not at.get("iframe") and not at.expander                                 # no preview / print view
    assert "Preview" not in " ".join(m.value for m in at.markdown)


def test_each_format_builds_only_itself_with_valid_files(snap, monkeypatch):
    from src.reports import simulation_report as rep
    rep._DOC_CACHE = None
    calls = []
    for fn in ("render_pdf", "render_docx", "render_html"):
        real = getattr(rep, fn)
        monkeypatch.setattr(rep, fn, (lambda f, r: (lambda s: calls.append(f) or r(s)))(fn, real))
    pdf = rep.export_document(snap, "pdf")
    assert calls == ["render_pdf"] and pdf[:5] == b"%PDF-" and pdf.rstrip().endswith(b"%%EOF")
    docx = rep.export_document(snap, "docx")
    assert calls == ["render_pdf", "render_docx"]
    import io
    import zipfile
    z = zipfile.ZipFile(io.BytesIO(docx))
    assert "word/document.xml" in z.namelist() and z.testzip() is None
    text = z.read("word/document.xml").decode("utf-8")
    assert snap["report_id"] in text and "ML Fire Risk Score" in text                # content preserved


def test_export_never_reruns_model_or_simulation(snap, monkeypatch):
    from src.reports import simulation_report as rep
    import src.simulation.local_spread as ls
    import src.ml_models.prediction_service as ps
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-ran model / simulation"))
    monkeypatch.setattr(ps, "predict", boom)
    for name in dir(ls):
        if name.startswith("run") and callable(getattr(ls, name)):
            monkeypatch.setattr(ls, name, boom)
    import requests
    monkeypatch.setattr(requests.Session, "request", boom)                          # no API call either
    rep._DOC_CACHE = None
    assert rep.export_document(snap, "pdf")[:4] == b"%PDF" and rep.export_document(snap, "docx")[:2] == b"PK"


def test_export_errors_are_reported_not_hidden(snap, monkeypatch):
    from src.reports import simulation_report as rep
    rep._DOC_CACHE = None
    monkeypatch.setattr(rep, "render_pdf", lambda s: (_ for _ in ()).throw(RuntimeError("bad chart")))
    with pytest.raises(RuntimeError):                                               # Streamlit then shows
        rep.export_document(snap, "pdf")                                            # "Failed to generate file"
    assert rep._DOC_CACHE == {} or not rep._DOC_CACHE                               # nothing cached


def test_missing_library_disables_only_that_format(tmp_path, monkeypatch):
    from src.dashboard.ui import alerts_ui as ui
    import importlib.util
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: None if n == "docx" else real(n, *a))
    at = _menu_app(monkeypatch, tmp_path)
    at.run()
    assert [b.proto.label for b in at.get("download_button")] == ["PDF (.pdf)"]
    word = at.button(key="t_xp_opt_docx")
    assert word.disabled and word.label == "Word (.docx)"


# ── live feeds ────────────────────────────────────────────────────────────

def test_live_feeds_run_side_by_side_with_identical_results(monkeypatch):
    from src.data_ingestion.ingestion_module import DataIngestionModule
    ing = DataIngestionModule(offline=True)
    ing.offline = False                                                             # exercise the live branch
    hot = pd.DataFrame({"latitude": [12.0], "longitude": [76.0], "frp": [3.0]})
    wx = pd.DataFrame({"latitude": [12.0], "longitude": [76.0], "temperature_c": [30.0]})

    def slow(df):
        def f():
            time.sleep(0.4)
            return df.copy()
        return f
    monkeypatch.setattr(ing, "fetch_fire_hotspots", slow(hot))
    monkeypatch.setattr(ing, "fetch_weather", slow(wx))
    t0 = time.perf_counter()
    h, w = ing._fetch_feeds()
    elapsed = time.perf_counter() - t0
    assert h.equals(hot) and w.equals(wx)
    assert elapsed < 0.7                                                            # ~0.4 s, not 0.8 s
    assert set(ing.last_fetch_timings) == {"firms_s", "weather_s"} and min(ing.last_fetch_timings.values()) >= 0.39


def test_live_feed_errors_still_surface(monkeypatch):
    from src.data_ingestion.ingestion_module import DataIngestionModule
    ing = DataIngestionModule(offline=True)
    ing.offline = False
    monkeypatch.setattr(ing, "fetch_fire_hotspots", lambda: (_ for _ in ()).throw(RuntimeError("firms down")))
    monkeypatch.setattr(ing, "fetch_weather", lambda: pd.DataFrame())
    with pytest.raises(RuntimeError, match="firms down"):
        ing._fetch_feeds()


def test_unified_frame_identical_sequential_vs_parallel(monkeypatch):
    """Same feed data in, same unified frame out - whether the feeds are fetched one after the other (old) or
    side by side (new). Demo data is random per refresh, so the feeds are frozen first."""
    from src.data_ingestion.ingestion_module import DataIngestionModule
    ing = DataIngestionModule(offline=True)
    hot, wx = ing.fetch_fire_hotspots(), ing.fetch_weather()                     # one frozen draw
    monkeypatch.setattr(ing, "fetch_fire_hotspots", lambda: hot.copy())
    monkeypatch.setattr(ing, "fetch_weather", lambda: wx.copy())
    seq = ing.build_unified_frame()                                               # offline: sequential branch
    ing.offline = False
    par = ing.build_unified_frame()                                               # live: thread pool branch
    assert "ingested_at" in seq.columns                                           # wall-clock stamp, differs by design
    pd.testing.assert_frame_equal(seq.drop(columns="ingested_at"), par.drop(columns="ingested_at"))


def test_twin_refresh_records_feed_timings():
    from src.digital_twin.twin_state import DigitalTwin
    t = DigitalTwin(offline=True)
    t.refresh()
    assert {"firms_s", "weather_s", "ingest_s", "total_s"} <= set(t.last_timings)


# ── Historical map ────────────────────────────────────────────────────────

def test_historical_map_positions_unchanged_by_float32():
    lat = np.random.default_rng(1).uniform(11.0, 16.0, 50000).round(5)
    lon = np.random.default_rng(2).uniform(74.0, 78.5, 50000).round(5)
    assert np.abs(lat.astype(np.float32) - lat).max() < 1e-5                      # < 1.1 m
    assert np.abs(lon.astype(np.float32) - lon).max() < 1e-5
    src = (ROOT / "src/dashboard/pages/3_Historical_Time_Machine.py").read_text(encoding="utf-8")
    assert 'to_numpy(np.float32)' in src


# ── alert history / thumbnail ─────────────────────────────────────────────

def test_alert_history_single_query_matches_previous_result(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'h.db'}")
    from src.notifications import alert_store as store
    from src.storage import database as db
    import sqlalchemy as sa
    for i in range(30):
        store.record_alert(f"A-{i}", "admin", f"R-{i}", "SIMULATION ONLY", "x", "reg", "s", "body", {"n": i})
        for j in range(i % 4):
            store.record_delivery(f"A-{i}", f"R-{i}", 0, "sms" if j % 2 else "email", "fake", "accepted", f"m{i}{j}")
    new = store.alert_history(20)
    with db.get_connection() as c:                                                 # the previous per-alert queries
        old = [dict(r._mapping) for r in c.execute(sa.select(store.T_ALERTS).order_by(store.T_ALERTS.c.id.desc()).limit(20))]
        for a in old:
            a["summary"] = __import__("json").loads(a.get("summary") or "{}")
            a["deliveries"] = [dict(r._mapping) for r in c.execute(
                sa.select(store.T_DELIV).where(store.T_DELIV.c.alert_id == a["alert_id"]).order_by(store.T_DELIV.c.id))]
    assert new == old
    insp = sa.inspect(db.get_engine())
    assert any(i["column_names"] == ["alert_id"] for i in insp.get_indexes("alert_deliveries"))


def test_index_added_to_an_existing_database_without_losing_rows(tmp_path, monkeypatch):
    import sqlalchemy as sa
    path = tmp_path / "old.db"
    eng = sa.create_engine(f"sqlite:///{path}")
    with eng.begin() as c:                                                         # a pre-existing table, no index
        c.execute(sa.text("CREATE TABLE alert_deliveries (id INTEGER PRIMARY KEY, alert_id VARCHAR(40) NOT NULL, "
                          "report_id VARCHAR(40), recipient_id INTEGER NOT NULL, channel VARCHAR(8) NOT NULL, "
                          "provider VARCHAR(40) NOT NULL, status VARCHAR(24) NOT NULL, provider_message_id VARCHAR(120), "
                          "error TEXT, timestamp_utc VARCHAR(40) NOT NULL)"))
        c.execute(sa.text("INSERT INTO alert_deliveries (alert_id, recipient_id, channel, provider, status, timestamp_utc) "
                          "VALUES ('A-1', 0, 'sms', 'fake', 'accepted', '2026-01-01')"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{path}")
    from src.notifications import alert_store as store
    store._READY.discard(f"sqlite:///{path}")
    store.ensure_tables()
    insp = sa.inspect(sa.create_engine(f"sqlite:///{path}"))
    assert any(i["column_names"] == ["alert_id"] for i in insp.get_indexes("alert_deliveries"))
    assert store.recent_deliveries(5)[0]["alert_id"] == "A-1"


def test_alert_thumbnail_is_the_same_static_image():
    from src.dashboard.ui import assets
    assert assets.ALERT_THUMB_STATIC.read_bytes() == assets.ALERT_THUMB.read_bytes()
    from src.dashboard.ui import command_center as cc
    src = inspect.getsource(cc.render_fire_alert_panel)
    assert "ALERT_THUMB_URL if _static_serving_enabled()" in src and "data_uri(ALERT_THUMB)" in src   # fallback kept


def test_prewarm_draws_once_and_export_reuses_it(snap, monkeypatch):
    import threading
    from src.reports import simulation_report as rep
    rep._CHART_CACHE = None
    calls = []
    real = rep._draw_charts
    monkeypatch.setattr(rep, "_draw_charts", lambda s: calls.append(1) or real(s))
    rep.prewarm_charts(snap)
    t = threading.Thread(target=lambda: rep.charts(snap)); t.start()             # Export opened while drawing
    out = rep.charts(snap); t.join()
    for th in [x for x in threading.enumerate() if x.name == "report-charts"]:
        th.join(10)
    assert len(calls) == 1 and out == real(snap)                                  # drawn once, identical bytes
