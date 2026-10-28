"""Command Center hero carousel: the three supplied images, served from the
static folder, passed to one declared component whose arguments stay identical
across reruns (so the iframe and its timer are reused, never duplicated), plus
static checks of the component's behaviour contract. Offline; no browser."""
import json
import re
from pathlib import Path

from PIL import Image
from streamlit.testing.v1 import AppTest

from src.dashboard.ui import assets
from src.dashboard.ui import command_center as cc

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "src/dashboard/components/hero_carousel/index.html").read_text(encoding="utf-8")

SCRIPT = '''
import sys
sys.path.insert(0, %r)
import streamlit as st
from src.dashboard.ui.command_center import render_dashboard_hero
st.button("poke")
render_dashboard_hero()
''' % str(ROOT)


def _hero(at):
    found = [c for c in at.get("component_instance") if "cc_hero_carousel" in c.proto.id]
    assert len(found) == 1
    return found[0]


def test_three_supplied_images_in_order_and_served_from_static():
    assert len(assets.HERO_SLIDES) == 3
    for s in assets.HERO_SLIDES:
        assert s["path"].exists() and s["path"].parent.is_relative_to(assets.STATIC_DIR)
        assert s["url"] == "app/static/" + s["path"].relative_to(assets.STATIC_DIR).as_posix()
        w, h = Image.open(s["path"]).size
        assert w >= 1300 and 1.6 < w / h < 1.8          # landscape artwork, full resolution
        assert s["alt"] and "illustrative" in s["alt"].lower()


def test_component_args_stable_across_reruns_and_note_present():
    at = AppTest.from_string(SCRIPT, default_timeout=30).run()
    assert not at.exception
    first = _hero(at)
    args = json.loads(first.proto.json_args)
    assert [s["src"] for s in args["slides"]] == [s["url"] for s in assets.HERO_SLIDES]
    assert args["interval_ms"] == 6000 and 500 <= args["fade_ms"] <= 800
    assert abs(args["layout"]["fit_ratio"] - 1627 / 967) < 1e-3
    assert any("Illustrative concept artwork" in m.value for m in at.markdown)
    at.button[0].click().run()                          # a rerun: same key + same args -> same iframe kept
    again = _hero(at)
    assert again.proto.id == first.proto.id and again.proto.json_args == first.proto.json_args


def test_inline_fallback_when_static_serving_off(monkeypatch):
    seen = {}
    monkeypatch.setattr(cc, "_static_serving_enabled", lambda: False)
    monkeypatch.setattr(cc, "_hero_carousel", lambda **kw: seen.update(kw))
    monkeypatch.setattr(cc.st, "markdown", lambda *a, **k: None)
    cc.render_dashboard_hero()
    assert len(seen["slides"]) == 3 and all(s["src"].startswith(("data:image/jpeg;base64,", "data:image/png;base64,")) for s in seen["slides"])


def test_slide_order_and_new_slide_2_is_the_supplied_file_unmodified():
    p = [s["path"] for s in assets.HERO_SLIDES]
    assert p[0].name == "dashboard_hero.jpg"
    assert p[1].name == "forest_fire_command_center.png" and p[1].parent.name == "hero"
    assert p[2].name == "satellite_intelligence.jpg"
    assert Image.open(p[1]).size == (1627, 967) and Image.open(p[1]).format == "PNG"   # original, not re-encoded


def test_height_follows_width_12_percent_shorter():
    L = cc.HERO_LAYOUT

    def h(w):  # mirrors heightFor() in index.html (1 px border)
        return (w - 2) / L["fit_ratio"] * L["height_scale"] + 2
    assert 0.85 <= L["height_scale"] <= 0.90                     # 10-15 % shorter than the full image height
    assert 680 <= h(1320) <= 700 and 395 <= h(768) <= 410 and 185 <= h(358) <= 195
    # full width for every slide: the stage is wider (relative to height) than every image,
    # so object-fit: cover fills the width and only trims top / bottom, never the sides much
    stage = L["fit_ratio"] / L["height_scale"]
    for s in assets.HERO_SLIDES:
        w, hh = Image.open(s["path"]).size
        assert w / hh <= stage
        assert 1 - (w / hh) / stage <= 0.13                      # at most ~12 % of the height trimmed
    assert "c.fit_ratio" in HTML and "c.height_scale" in HTML and "max_height" not in HTML


def test_full_width_cover_no_side_panels_no_arrows():
    rule = re.search(r"\.slide img \{([^}]*)\}", HTML).group(1)
    for decl in ("inset: 0", "width: 100%", "height: 100%", "object-fit: cover", "object-position: 50% 8%"):
        assert decl in rule
    assert "blur(" not in HTML and 'className = "bg"' not in HTML and "object-fit: contain" not in HTML
    assert HTML.count("new Image()") == 1 and "objectPosition" not in HTML
    assert "class=\"arrow" not in HTML and ".arrow" not in HTML and 'id="prev"' not in HTML


def test_component_behaviour_contract():
    # built once per iframe: reruns re-send "render" but never rebuild / add timers
    assert "if (started) return;" in HTML
    assert "clearTimeout(timer)" in HTML and "setInterval" not in HTML
    # continuous autoplay: paused only while the tab is hidden (no hover / focus pause)
    assert "document.hidden" in HTML and "visibilitychange" in HTML
    assert "pointerenter" not in HTML and "focusin" not in HTML
    # images preloaded + decoded before a slide may show
    assert "decode" in HTML and "ready[i].then" in HTML
    # wrap-around 3 -> 1; indicator dots
    assert "((i % n) + n) % n" in HTML and '"aria-current"' in HTML
    # crossfade, honouring reduced-motion; theme-aware
    assert "transition: opacity var(--fade)" in HTML and "prefers-reduced-motion" in HTML
    assert 'th.base === "light"' in HTML
    # no invented text drawn over the artwork
    assert not re.search(r"<(h[1-6]|p)[ >]", HTML)
