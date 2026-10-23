"""Appearance switcher: Default Dark (unchanged) and Professional Navy Blue.

Static checks of the CSS-only design (every light rule is scoped to
html[data-ff-theme="navy"], so Default Dark cannot change), of the controller
component (no Streamlit rerun, persisted in the browser, one instance per tab)
and an AppTest run showing every page draws the controller. Offline; no browser."""
import re
from pathlib import Path

import pytest

from src.dashboard.ui import theme_switch as ts

ROOT = Path(__file__).resolve().parents[1]
COMP = (ROOT / "src/dashboard/components/theme_switch/index.html").read_text(encoding="utf-8")
FIRE_MAP = (ROOT / "src/dashboard/components/fire_map/index.html").read_text(encoding="utf-8")
HERO = (ROOT / "src/dashboard/components/hero_carousel/index.html").read_text(encoding="utf-8")
L = 'html[data-ff-theme="navy"]'


def _rules(css: str):
    """(selector, body) pairs of a flat stylesheet; @media blocks are unwrapped."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    css = re.sub(r"@media[^{]*\{((?:[^{}]*\{[^{}]*\})*)\s*\}", r"\1", css)
    return [(sel.strip(), body) for sel, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css)]


def test_palette_is_the_requested_navy():
    expect = {"bg": "#061426", "sidebar": "#03101F", "header": "#04172B", "card": "#081D33", "card2": "#0B2542",
              "primary": "#0878F9", "nav_active": "#0756C9", "cyan": "#00B8F0", "text": "#F3F7FC", "text2": "#B6C7DC",
              "muted": "#8298B3", "border": "#163A5E", "ok": "#20D995", "warn": "#FFB547", "crit": "#FF4D5A"}
    for k, v in expect.items():
        assert ts.NAVY[k] == v and v in ts.NAVY_CSS, k
    assert ts.THEMES == (("dark", "Default Dark"), ("navy", "Professional Navy Blue"))


def test_no_light_blue_theme_left_anywhere():
    for src in (ts.THEME_SWITCH_CSS, COMP, FIRE_MAP, HERO):
        for old in ("#D5EAFF", "#B5DCFF", "#8FC9FF", "#1767B7", "ff-light", 'data-ff-theme="light"', "Light Blue"):
            assert old not in src, old


def test_every_navy_rule_is_scoped_so_default_dark_is_untouched():
    rules = _rules(ts.NAVY_CSS + ts.INLINE_CSS)
    assert len(rules) > 150
    for sel, _ in rules:
        for part in re.split(r",(?![^\[]*\])", sel):          # commas outside [attr="..."]
            assert part.strip().startswith(L), part
    # the always-on part only styles the new menu / hidden controller, never the dashboard's tokens
    for sel, body in _rules(ts.MENU_CSS):
        assert ":root" not in sel and "--bg" not in body and "--surface" not in body
        assert all(p.strip().startswith(("#ff-theme", ".st-key-ff_theme_host", '[data-testid="stLayoutWrapper"]:has',
                                          '[data-testid="stElementContainer"]:has')) for p in sel.split(",")), sel


def test_map_imagery_and_fire_effects_are_never_tinted():
    css = ts.NAVY_CSS
    assert "filter" not in re.sub(r'\[data-testid="stDataFrame"\][^{]*\{[^}]*\}|\[data-testid="stSlider"\][^{]*\{[^}]*\}', "", css)
    assert "iframe" not in css.replace(".st-key-cc_regionmap iframe", "")        # only the map frame's shadow
    # in the map component only panels / buttons change; canvases and the map are not styled
    navy_map = "\n".join(l for l in FIRE_MAP.splitlines() if l.startswith("html.ff-navy"))
    assert navy_map and "canvas" not in navy_map and "#map{" not in navy_map and "filter" not in navy_map


def test_inline_styles_match_reacts_normalised_form():
    # React writes style="background:#11151b" as "background: rgb(17, 21, 27)"
    assert '[style*="background: rgb(17, 21, 27)"]' in ts.INLINE_CSS and "#11151b" not in ts.INLINE_CSS.lower()


def test_controller_component_contract():
    # default dark; persisted in the browser; two themes, labelled as requested; check on the active one
    assert 'KEY = "ffdt-theme"' in COMP and 'v === "navy" ? "navy" : "dark"' in COMP
    assert '["dark", "Default Dark"], ["navy", "Professional Navy Blue"]' in COMP
    assert 'role="menuitemradio"' in COMP and "aria-checked" in COMP and ">Appearance<" in COMP
    assert 'aria-label="Application menu"' in COMP and "circle" in COMP                # the single ⋮
    # application items: real actions only (Rerun / Clear cache via Python, Print via the browser)
    assert '["rerun", "Rerun"' in COMP and '["clear_cache", "Clear cache"' in COMP and '["print", "Print"' in COMP
    assert "window.print()" in COMP and "Record screen" not in COMP and "Auto rerun" not in COMP
    # theme switching never sends a value (no rerun); only Rerun / Clear cache do
    assert COMP.count('send("streamlit:setComponentValue"') == 1 and "function set(t)" in COMP
    set_fn = COMP[COMP.index("function set(t)"):COMP.index("function act(a)")]
    assert "bridge" not in set_fn and "location.reload" not in COMP
    # one controller per tab; own fixed element - Streamlit's header is not touched
    assert "P.__ffTheme && P.__ffTheme.version >= VERSION" in COMP and 'id = "ff-theme-controller"' in COMP
    assert "stToolbar" not in COMP and "stAppDeployButton" not in COMP and "stMainMenu" not in COMP
    assert "document.body.appendChild(wrapEl)" in COMP and '"ArrowDown"' in COMP and '"Escape"' in COMP


def test_streamlit_native_menu_switched_off_by_supported_config():
    cfg = (ROOT / ".streamlit/config.toml").read_text(encoding="utf-8")       # (tomllib needs Python 3.11)
    client = cfg.split("[client]", 1)[1].split("\n[", 1)[0]
    assert re.search(r'^toolbarMode\s*=\s*"minimal"\s*$', client, re.M)
    assert "position: fixed" in ts.MENU_CSS and "ff-floating" not in ts.MENU_CSS


def test_menu_actions_are_handled_once(monkeypatch):
    calls = []
    state = {}
    monkeypatch.setattr(ts.st, "session_state", state)
    monkeypatch.setattr(ts.st.cache_data, "clear", lambda: calls.append("data"))
    monkeypatch.setattr(ts.st.cache_resource, "clear", lambda: calls.append("resource"))
    monkeypatch.setattr(ts.st, "toast", lambda *a, **k: calls.append("toast"))
    assert ts.handle_menu_action(None) == "" and ts.handle_menu_action({"action": "rerun"}) == ""
    assert ts.handle_menu_action({"action": "clear_cache", "nonce": "1"}) == "clear_cache"
    assert calls == ["data", "resource", "toast"]
    assert ts.handle_menu_action({"action": "clear_cache", "nonce": "1"}) == ""          # same value on a later rerun
    assert calls == ["data", "resource", "toast"]
    assert ts.handle_menu_action({"action": "rerun", "nonce": "2"}) == "rerun" and len(calls) == 3


def test_alerts_have_one_surface_in_navy():
    css = ts.NAVY_CSS
    for kind, bg, bd, fg in (("Success", "#0B332D", "#176B52", "#D1FAE5"), ("Info", "#0A2A47", "#1D5A8C", "#D6EEFF"),
                             ("Warning", "#33270B", "#7A5A16", "#FFE7B8"), ("Error", "#3A1219", "#7F2532", "#FFD9DD")):
        cont = re.search(r'\[data-testid="stAlertContainer"\]:has\(> \[data-testid="stAlertContent%s"\]\) \{([^}]*)\}' % kind, css).group(1)
        assert bg in cont and bd in cont and fg in cont
        inner = re.search(r'\[data-testid="stAlertContent%s"\] \{([^}]*)\}' % kind, css).group(1)
        assert "background: transparent" in inner                      # no second strip behind the text
    assert not re.search(r'\[data-testid="stAlertContent\w+"\] \{[^}]*background: rgba', css)


def test_iframes_follow_the_dashboard_theme():
    for src in (FIRE_MAP, HERO, ts.IFRAME_THEME_SYNC):
        assert "data-ff-theme" in src and "MutationObserver" in src
    from src.dashboard.ui import alerts_ui                  # (the print-button iframe was removed with Print)
    assert not hasattr(alerts_ui, "PRINT_BUTTON_HTML")


def test_style_block_opens_its_own_html_block():
    # appended to set_page's shared <style> markdown: must start on a new line (CommonMark)
    assert ts.THEME_SWITCH_CSS.startswith("\n<style>") and ts.THEME_SWITCH_CSS.rstrip().endswith("</style>")


@pytest.mark.parametrize("page", ["app.py", "pages/1_What_If_Simulator.py", "pages/6_Admin.py"])
def test_every_page_draws_the_controller(page, tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")    # never the project's real database
    st_testing = pytest.importorskip("streamlit.testing.v1")
    at = st_testing.AppTest.from_file(str(ROOT / "src/dashboard" / page), default_timeout=120)
    at.session_state["auth_user"] = "tester"
    at.session_state["auth_role"] = "admin"
    at.session_state["_pref_offline_mode"] = True
    at.run()
    assert not at.exception
    comps = [c for c in at.get("component_instance") if "ff_theme_switch" in c.proto.id]
    assert len(comps) == 1
    styles = "".join(m.value for m in at.markdown)
    assert 'html[data-ff-theme="navy"]' in styles and "#ff-theme-btn" in styles
