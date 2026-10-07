"""Design system: palette tokens and the shared component styles.

Colour rules
  * base surfaces #080B12 / #0D111A / #111722, fire accents #FF7A18 / #FF4D1C,
    alert #FF3B3B, success #35D07F;
  * the blue -> purple -> pink gradient (#4DA3FF -> #9B7BFF -> #E86CFF) is used
    ONLY for the active sidebar navigation button (and, faintly, the hover edge
    of sidebar buttons / Explore cards). Nothing else uses it.
All animations are CSS only (transform / opacity), and are switched off under
`prefers-reduced-motion`.
"""

# Page title (as passed to set_page) -> selector of its sidebar link. Used to
# highlight the active page on the very first paint, independent of the
# aria-current attribute Streamlit also sets on the active link.
NAV_SELECTORS = {
    "Command Center": "ul li:first-child a",
    "What-If Simulator": 'a[href*="What_If"]',
    "Spread Simulation": 'a[href*="Spread_Simulation"]',
    "Historical Time Machine": 'a[href*="Historical"]',
    "AI Situation Briefing": 'a[href*="AI_Situation"]',
    "Model Insights": 'a[href*="Model_Insights"]',
    "Admin": 'a[href*="Admin"]',
    "Activity Log": 'a[href*="Activity_Log"]',
}

_NAV = '[data-testid="stSidebarNav"]'


def _active_rules(selector: str) -> str:
    return f"""
{selector} {{
    background: linear-gradient(90deg, #4DA3FF 0%, #9B7BFF 55%, #E86CFF 100%) !important;
    border-color: transparent !important;
    box-shadow: 0 6px 20px rgba(155,123,255,0.30), inset 0 0 0 1px rgba(255,255,255,0.16) !important;
    transform: none !important; }}
{selector} span, {selector} span p, {selector}::after {{ color: #ffffff !important; font-weight: 700 !important; }}
{selector}::before {{ background: #ffffff !important; opacity: 1 !important; }}
"""


def active_nav_css(title: str) -> str:
    """Highlight for the current page's sidebar button, emitted by set_page()."""
    sel = NAV_SELECTORS.get(title)
    if not sel:
        return ""
    return "<style>" + _active_rules(f"{_NAV} {sel}") + "</style>"


_CONTOURS = (
    "url(\"data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='900' height='700' "
    "viewBox='0 0 900 700' fill='none' stroke='white' stroke-opacity='0.035' stroke-width='1.2'%3E"
    "%3Cpath d='M520 120c90-40 210-20 260 60s10 190-80 230-230 30-280-50-10-200 100-240z'/%3E"
    "%3Cpath d='M545 160c70-28 160-12 196 48s6 140-62 170-172 20-208-40-4-150 74-178z'/%3E"
    "%3Cpath d='M572 200c48-18 108-6 132 34s2 92-44 112-114 12-138-28 0-100 50-118z'/%3E"
    "%3Cpath d='M600 238c26-10 58-2 70 20s0 48-24 58-60 6-72-16 0-52 26-62z'/%3E"
    "%3Cpath d='M80 520c120-60 300-50 380 30s-20 160-180 160S-20 600 80 520z'/%3E"
    "%3Cpath d='M130 545c90-40 220-32 276 22s-14 112-130 112S60 590 130 545z'/%3E"
    "%3Cpath d='M720 470h150M720 500h150M720 530h150M750 450v100M790 450v100M830 450v100' stroke-opacity='0.03'/%3E"
    "%3C/svg%3E\")"
)

THEME_CSS = """
<style>
:root {
    --bg: #080B12; --surface: #0D111A; --surface-2: #111722; --border: #1E2633; --border-2: #2A3445;
    --text: #E8EDF3; --muted: #8A96A6;
    --accent: #FF7A18; --accent-2: #FFB347; --fire: #FF7A18; --fire-2: #FF4D1C;
    --ok: #35D07F; --warn: #FBBF24; --crit: #FF3B3B; --info: #4DA3FF; --alert: #FF3B3B;
    --nav-blue: #4DA3FF; --nav-cyan: #5BD6FF; --nav-purple: #9B7BFF; --nav-pink: #E86CFF;
    --radius: 14px;
}
.stApp {
    background: """ + _CONTOURS + """ right -120px top 40px / 900px 700px no-repeat,
                radial-gradient(1100px 480px at 88% -12%, rgba(255,122,24,0.07), transparent 60%),
                var(--bg) !important;
}

/* ── accessibility: visible keyboard focus everywhere ── */
a:focus-visible, button:focus-visible, [role="tab"]:focus-visible, input:focus-visible,
[data-testid="stPageLink"] a:focus-visible {
    outline: 2px solid #FFB347 !important; outline-offset: 2px !important; border-radius: 10px; }

/* ── sidebar: compact rounded buttons with icons ── */
[data-testid="stSidebar"] { background: #0A0E15 !important; border-right: 1px solid var(--border); }
[data-testid="stSidebarNav"] { padding-top: 6px; }
[data-testid="stSidebarNav"] ul { gap: 5px; display: flex; flex-direction: column; }
[data-testid="stSidebarNav"] a {
    position: relative; display: flex !important; align-items: center; gap: 10px; min-height: 38px;
    padding: 7px 12px 7px 40px !important; border-radius: 11px !important;
    background: #0F141E !important; border: 1px solid rgba(255,255,255,0.06) !important;
    box-shadow: none !important;
    transition: border-color .16s ease, box-shadow .16s ease, background .16s ease; }
[data-testid="stSidebarNav"] a span, [data-testid="stSidebarNav"] a span p { color: #B9C3D0 !important; font-weight: 600 !important; font-size: 13.5px !important; }
[data-testid="stSidebarNav"] a::before { background: #8A96A6 !important; opacity: 1 !important; }
[data-testid="stSidebarNav"] a:hover {
    background: #121927 !important; transform: none !important;
    border-color: rgba(77,163,255,0.38) !important;
    box-shadow: inset 2px 0 0 rgba(77,163,255,0.85), inset -2px 0 0 rgba(232,108,255,0.55) !important; }
[data-testid="stSidebarNav"] a:hover span, [data-testid="stSidebarNav"] a:hover span p, [data-testid="stSidebarNav"] li:first-child a:hover::after { color: #ffffff !important; }
""" + _active_rules(f'{_NAV} a[aria-current="page"]') + """

/* ── buttons / page links: fire accent, no blue / pink ── */
.stButton > button:hover { border-color: var(--fire); box-shadow: 0 6px 18px rgba(255,122,24,.18); }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #FF7A18, #FF9A3C); color: #170b04; }
[data-testid="stPageLink"] a:hover { border-color: var(--fire) !important; }

/* ── global ticker: dark navy glass, thin blue -> purple -> pink gradient border, pink as accent only ── */
.gt { position: relative; display: flex; align-items: stretch; height: 36px; margin: -14px 0 16px 0; border-radius: 12px;
      overflow: hidden; isolation: isolate;
      border: 1px solid transparent;
      background:
        linear-gradient(100deg, rgba(10,16,34,0.94) 0%, rgba(14,18,40,0.94) 55%, rgba(22,14,36,0.94) 100%) padding-box,
        linear-gradient(90deg, rgba(77,163,255,0.85), rgba(155,123,255,0.75) 55%, rgba(232,108,255,0.80)) border-box;
      backdrop-filter: blur(10px) saturate(130%); -webkit-backdrop-filter: blur(10px) saturate(130%);
      box-shadow: 0 0 0 1px rgba(77,163,255,0.06), 0 8px 26px rgba(8,12,30,0.55), 0 0 22px rgba(77,163,255,0.12),
                  0 0 30px rgba(232,108,255,0.07); }
.gt::before { content: ""; position: absolute; inset: 0; z-index: -1; pointer-events: none;
      background: radial-gradient(420px 60px at 8% 0%, rgba(77,163,255,0.16), transparent 70%),
                  radial-gradient(380px 60px at 92% 100%, rgba(232,108,255,0.11), transparent 70%); }
.gt::after { content: ""; position: absolute; left: 0; right: 0; top: 0; height: 1px; pointer-events: none;
      background: linear-gradient(90deg, transparent, rgba(255,255,255,0.18), transparent); }
.gt-tag { display: flex; align-items: center; gap: 7px; padding: 0 14px; white-space: nowrap; flex: 0 0 auto;
          font: 700 10.5px/1 var(--mono); letter-spacing: .14em;
          border-right: 1px solid rgba(155,123,255,0.28); background: rgba(9,13,30,0.75); }
.gt-tag.live { color: var(--ok); box-shadow: inset 0 0 18px rgba(53,208,127,0.10); }
.gt-tag.demo { color: var(--warn); box-shadow: inset 0 0 18px rgba(251,191,36,0.08); }
.gt-tag .dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; box-shadow: 0 0 8px currentColor;
               animation: gtBlink 1.6s ease-in-out infinite; }
.gt-view { position: relative; flex: 1 1 auto; overflow: hidden;
           -webkit-mask-image: linear-gradient(90deg, transparent, #000 4%, #000 96%, transparent);
                   mask-image: linear-gradient(90deg, transparent, #000 4%, #000 96%, transparent); }
.gt-track { display: inline-flex; align-items: center; height: 100%; white-space: nowrap; will-change: transform;
            animation: gtScroll var(--gt-dur, 60s) linear infinite; }
.gt:hover .gt-track { animation-play-state: paused; }
.gt-item { display: inline-flex; align-items: center; gap: 7px; padding: 0 18px; font-size: 12.5px; color: #D3DAEA; }
.gt-item b { font: 700 10.5px/1 var(--mono); letter-spacing: .12em; color: #8FB4F0; }
.gt-item .v { color: #F2F4FA; font-weight: 600; }
.gt-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; flex: 0 0 auto; }
.gt-dot.ok { background: var(--ok); } .gt-dot.warn { background: var(--warn); }
.gt-dot.crit { background: var(--crit); box-shadow: 0 0 8px rgba(255,59,59,.7); } .gt-dot.data { background: #4DA3FF; }
.gt-dot.fire { background: var(--fire); }
.gt-sep { color: rgba(232,108,255,0.55); }
@keyframes gtScroll { from { transform: translateX(0); } to { transform: translateX(-50%); } }
@keyframes gtBlink { 0%,100% { opacity: 1; } 50% { opacity: .3; } }

/* ── Command Center hero (supplied artwork, never cropped or stretched; plain <img>, no toolbar) ── */
.cc-hero { position: relative; width: 100%; border-radius: 18px; overflow: hidden; line-height: 0;
    border: 1px solid rgba(255,255,255,0.08); box-shadow: 0 22px 60px rgba(0,0,0,0.45); animation: heroIn .9s ease-out both; }
.cc-hero img { display: block; width: 100%; height: auto; max-width: 100%; object-fit: contain;
    user-select: none; -webkit-user-drag: none; pointer-events: none; }
.cc-hero-note { font-size: 11.5px; color: #6F7B8B; margin: 8px 4px 0 4px; }
@keyframes heroIn { from { opacity: 0; transform: scale(1.012); } to { opacity: 1; transform: scale(1); } }

/* ── live status strip ── */
.cc-strip { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0 12px 0; }
.cc-pill { display: inline-flex; align-items: center; gap: 8px; padding: 7px 13px; border-radius: 999px;
           background: var(--surface); border: 1px solid var(--border); font-size: 12.5px; color: #C9D2DD; }
.cc-pill b { font: 700 10.5px/1 var(--mono); letter-spacing: .12em; color: var(--muted); }
.cc-pill .gt-dot.ok, .cc-pill .gt-dot.crit { animation: gtBlink 2s ease-in-out infinite; }

/* ── wildfire alert panel ── */
.fa { position: relative; display: grid; grid-template-columns: 230px minmax(0,1fr); gap: 0; overflow: hidden;
      border-radius: 16px; border: 1px solid rgba(255,59,59,0.38);
      background: linear-gradient(120deg, rgba(40,10,10,0.88), rgba(13,17,26,0.92) 60%);
      box-shadow: 0 0 0 1px rgba(255,59,59,0.06), 0 0 34px rgba(255,59,59,0.14);
      animation: faGlow 3.2s ease-in-out infinite; }
.fa.calm { border-color: rgba(53,208,127,0.32); background: linear-gradient(120deg, rgba(10,30,20,0.8), rgba(13,17,26,0.92) 60%);
           box-shadow: none; animation: none; }
.fa.warn { border-color: rgba(251,191,36,0.40); background: linear-gradient(120deg, rgba(40,28,6,0.85), rgba(13,17,26,0.92) 60%);
           box-shadow: 0 0 26px rgba(251,191,36,0.10); animation: none; }
.fa .thumb { position: relative; min-height: 170px; background-size: cover; background-position: center; }
.fa .thumb::after { content: ""; position: absolute; inset: 0; background: linear-gradient(90deg, transparent 55%, rgba(13,17,26,0.95)); }
.fa .thumb .tag { position: absolute; left: 10px; bottom: 10px; z-index: 1; font: 600 10px/1.3 var(--mono);
                  color: #FFD9C2; background: rgba(0,0,0,0.55); padding: 3px 7px; border-radius: 6px; }
.fa .body { padding: 16px 20px 14px 18px; min-width: 0; }
.fa .head { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; }
.fa .pulse { position: relative; width: 11px; height: 11px; border-radius: 50%; background: var(--alert); flex: 0 0 auto; }
.fa .pulse::after { content: ""; position: absolute; inset: -5px; border-radius: 50%; border: 2px solid var(--alert);
                    animation: faPing 1.8s ease-out infinite; }
.fa.calm .pulse { background: var(--ok); } .fa.calm .pulse::after { border-color: var(--ok); }
.fa.warn .pulse { background: var(--warn); } .fa.warn .pulse::after { border-color: var(--warn); }
.fa .kicker { font: 700 11px/1 var(--mono); letter-spacing: .16em; color: #FF8A80; }
.fa.calm .kicker { color: var(--ok); } .fa.warn .kicker { color: var(--warn); }
.fa .chip { font: 700 10px/1 var(--mono); letter-spacing: .1em; padding: 4px 8px; border-radius: 6px;
            border: 1px solid rgba(255,255,255,0.18); color: #E8EDF3; background: rgba(255,255,255,0.05); }
.fa .chip.demo { color: var(--warn); border-color: rgba(251,191,36,0.4); }
.fa .chip.sim { color: #FFB38A; border-color: rgba(255,122,24,0.45); }
.fa .chip.live { color: var(--ok); border-color: rgba(53,208,127,0.4); }
.fa h3 { margin: 9px 0 3px 0 !important; font-size: 19px !important; }
.fa .meta { font-size: 12.5px; color: var(--muted); }
.fa .stats { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 10px; margin-top: 12px; }
.fa .stat { background: rgba(0,0,0,0.28); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 8px 11px; }
.fa .stat .l { font-size: 10.5px; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
.fa .stat .n { font: 700 18px/1.3 var(--mono); color: #F4F6F8; }
.fa .stat .n.hot { color: #FF8A5B; }
.st-key-cc_alert { gap: 10px !important; }
.st-key-cc_alert .fa { margin-bottom: 18px; }
.st-key-cc_alert [data-testid="stPageLink"] a { justify-content: center; min-height: 42px; background: rgba(255,59,59,0.12) !important;
    border: 1px solid rgba(255,59,59,0.45) !important; border-radius: 10px; }
.st-key-cc_alert [data-testid="stPageLink"] a p { color: #FFD3CC !important; font-weight: 700; letter-spacing: .12em; font-size: 12.5px; }
.st-key-cc_alert [data-testid="stPageLink"] a:hover { background: rgba(255,59,59,0.22) !important; border-color: #FF3B3B !important; }
@keyframes faPing { 0% { transform: scale(.6); opacity: .9; } 100% { transform: scale(1.8); opacity: 0; } }
@keyframes faGlow { 0%,100% { box-shadow: 0 0 0 1px rgba(255,59,59,0.06), 0 0 22px rgba(255,59,59,0.10); }
                    50% { box-shadow: 0 0 0 1px rgba(255,59,59,0.12), 0 0 38px rgba(255,59,59,0.22); } }

/* ── metric cards ── */
.kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr)); gap: 12px; margin-bottom: 20px; }
.kpi { position: relative; overflow: hidden; background: var(--surface); border: 1px solid var(--border);
       border-radius: 13px; padding: 14px 16px 13px 16px; animation: fadeSlideIn .5s ease-out both; }
.kpi::before { content: ""; position: absolute; left: 0; right: 0; top: 0; height: 2px; background: var(--k-accent, #3A4556); }
.kpi.k-ok { --k-accent: var(--ok); } .kpi.k-warn { --k-accent: var(--warn); } .kpi.k-crit { --k-accent: var(--crit); }
.kpi.k-fire { --k-accent: var(--fire); } .kpi.k-neutral { --k-accent: #3A4556; }
.kpi .top { display: flex; align-items: center; justify-content: space-between; }
.kpi .ico { width: 17px; height: 17px; color: var(--muted); }
.kpi .ico svg { width: 17px; height: 17px; }
.kpi .ctx { font-size: 11.5px; color: #6F7B8B; margin-top: 2px; }
.kpi-row .kpi:nth-child(2) { animation-delay: .04s; } .kpi-row .kpi:nth-child(3) { animation-delay: .08s; }
.kpi-row .kpi:nth-child(4) { animation-delay: .12s; } .kpi-row .kpi:nth-child(5) { animation-delay: .16s; }
.kpi-row .kpi:nth-child(6) { animation-delay: .20s; } .kpi-row .kpi:nth-child(7) { animation-delay: .24s; }

/* ── Explore cards ── */
[class*="st-key-cc_explore_"] { background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
    padding: 16px 16px 12px 16px; height: 100%; gap: 6px !important;
    transition: border-color .18s ease, box-shadow .18s ease, transform .18s ease; animation: fadeSlideIn .5s ease-out both; }
[class*="st-key-cc_explore_"]:hover { transform: translateY(-3px); border-color: rgba(77,163,255,0.45);
    box-shadow: inset 0 2px 0 rgba(77,163,255,0.6), 0 10px 26px rgba(232,108,255,0.10); }
.ex-card .ex-ico { width: 36px; height: 36px; border-radius: 10px; display: grid; place-items: center;
    background: rgba(255,122,24,0.10); border: 1px solid rgba(255,122,24,0.25); color: #FFB38A; margin-bottom: 10px; }
.ex-card .ex-ico svg { width: 19px; height: 19px; }
.ex-card .idx { font: 600 10.5px/1 var(--mono); color: #6F7B8B; letter-spacing: .14em; }
.ex-card h4 { margin: 6px 0 6px 0 !important; font-size: 15.5px !important; }
.ex-card { margin-bottom: 16px; }
.ex-card p { font-size: 12.8px; color: var(--muted); line-height: 1.5; margin: 0; min-height: 40px; }
[class*="st-key-cc_explore_"] [data-testid="stPageLink"] a { justify-content: flex-start; background: transparent !important;
    border: none !important; padding-left: 0 !important; }
[class*="st-key-cc_explore_"] [data-testid="stPageLink"] a p { color: #FFB38A !important; font-weight: 700; font-size: 13px;
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
[class*="st-key-cc_explore_"] [data-testid="stPageLink"] a:hover p { color: #FFFFFF !important; }

/* ── data / system status ── */
.sys-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px; margin-top: 6px; }
.sys-item { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 11px 14px; }
.sys-item .l { display: flex; align-items: center; gap: 8px; font: 700 10.5px/1 var(--mono); letter-spacing: .12em; color: var(--muted); }
.sys-item .v { margin-top: 6px; font-size: 13px; color: #D5DCE5; }

@media (max-width: 900px) {
    .fa { grid-template-columns: 1fr; } .fa .thumb { min-height: 120px; }
    .fa .thumb::after { background: linear-gradient(180deg, transparent 50%, rgba(13,17,26,0.95)); }
    .fa .stats { grid-template-columns: repeat(2, minmax(0,1fr)); }
    .gt { margin-top: -6px; }
}
@media (prefers-reduced-motion: reduce) {
    .gt-track { animation: none; } .gt-view { overflow-x: auto; }
    .fa, .fa .pulse::after, .gt-tag .dot, .cc-pill .gt-dot, .kpi, .cc-hero,
    [class*="st-key-cc_explore_"] { animation: none !important; }
}
</style>
"""
