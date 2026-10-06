"""
geo_fire_map.py - Google satellite base map + geographically anchored fire
spread visualization for the Spread Simulation page.

Architecture (one Streamlit component, one map, one canvas):

    Streamlit (Python)                      Browser (iframe from components.html)
    ------------------                      -------------------------------------
    LocalSpreadResult  --JSON payload-->     Google Maps JavaScript API (satellite)
      (CA state per cell, ignition /           |- google.maps.Data: one polygon per
       burn-out step, intensity, wind,         |    CA cell (grid, burned, non-fuel),
       per-step analytics)                     |    restyled only when a cell changes
                                               |- OverlayView: map projection only
                                               '- ONE <canvas> over the map: fire,
                                                  glow, smoke, embers, ash,
                                                  vegetation, wind - every particle
                                                  is in local metres (east, north,
                                                  altitude) and projected through a
                                                  per-frame ground-plane homography,
                                                  so it stays on the same lat/lon
                                                  through pan / zoom / tilt / rotate.

The whole CA history is sent once. Play, pause, reset, speed and the timeline
run in the browser, so a Streamlit rerun never re-creates the map or the
canvas while it animates (the HTML is deterministic: identical inputs give an
identical iframe, which Streamlit keeps instead of reloading).

Canvas 2D with pre-rendered sprite textures and additive blending is used
rather than Three.js: Google's WebGLOverlayView needs a vector map, which is
not guaranteed for satellite imagery, and a single 2D canvas stays fast on an
integrated GPU. Particle counts are hard-capped and scale with zoom (LOD).

Labelling: everything drawn on the canvas is SIMULATED (CA output + visual
effects). Satellite fire detections are drawn separately as circles of the
VIIRS 375 m footprint and labelled observed (live FIRMS) or synthetic (demo).
"""
from __future__ import annotations

import json
import math
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.simulation.local_spread import FocusArea, LocalSpreadResult, _ignition_cells


# ── payload ───────────────────────────────────────────────────────────────── #

def hotspots_near(hotspots: Optional[pd.DataFrame], lat: float, lon: float,
                  radius_km: float = 40.0, max_points: int = 150, days: int = 1) -> List[dict]:
    """Recent hotspot detections within radius_km of (lat, lon)."""
    if hotspots is None or len(hotspots) == 0:
        return []
    h = hotspots
    if "acq_date" in h.columns:
        d = pd.to_datetime(h["acq_date"], errors="coerce")
        cutoff = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=days)
        h = h[d >= cutoff]
    if h.empty:
        return []
    dy = (h["latitude"].to_numpy(float) - lat) * 111.32
    dx = (h["longitude"].to_numpy(float) - lon) * 111.32 * math.cos(math.radians(lat))
    dist = np.hypot(dx, dy)
    keep = np.argsort(dist)[:max_points]
    keep = keep[dist[keep] <= radius_km]
    out = []
    for i in keep:
        row = h.iloc[int(i)]
        out.append({"lat": round(float(row["latitude"]), 5), "lon": round(float(row["longitude"]), 5),
                    "date": str(row.get("acq_date", "")), "km": round(float(dist[i]), 1)})
    return out


def build_payload(focus: FocusArea, result: Optional[LocalSpreadResult], wind_speed_ms: float,
                  wind_from_deg: float, hotspots: List[dict], hotspot_kind: str,
                  n_ignition: int, placement: str, autoplay: bool, api_key: str, map_id: str,
                  elevation: Optional[np.ndarray] = None) -> dict:
    n = focus.n
    b = focus.bounds
    base = {
        "key": api_key, "mapId": map_id or "DEMO_MAP_ID",
        "focus": {"name": focus.name, "lat": focus.lat, "lon": focus.lon, "n": n, "cell_m": focus.cell_m,
                  "size_m": n * focus.cell_m, "north": b["north"], "south": b["south"],
                  "west": b["west"], "east": b["east"]},
        "hotspots": hotspots, "hotspotKind": hotspot_kind,
        "autoplay": bool(autoplay),
    }
    if elevation is not None and np.isfinite(elevation).all() and np.ptp(elevation) > 0:
        gy, gx = np.gradient(elevation, focus.cell_m)
        slope = np.hypot(gx, gy) * 100.0
        base["slopePct"] = np.round(slope.ravel(), 1).tolist()
    if result is None:
        ign = _ignition_cells(n, n_ignition, placement, wind_speed_ms, wind_from_deg)
        base.update({
            "hasRun": False, "stepMin": 0, "lastStep": 0,
            "ign": np.where(ign.ravel(), 0, -1).astype(int).tolist(),
            "out": [-1] * (n * n), "inten": [0] * (n * n), "nonfuel": [0] * (n * n),
            "wind": [[float(wind_speed_ms), float(wind_from_deg) % 360]], "metrics": [],
        })
        return base
    m = result.metrics
    base.update({
        "hasRun": True, "stepMin": result.step_minutes, "lastStep": int(result.history[-1].step),
        "ign": result.ignition_step.ravel().astype(int).tolist(),
        "out": result.burnout_step.ravel().astype(int).tolist(),
        "inten": np.round(result.intensity.ravel(), 2).tolist(),
        "nonfuel": result.non_fuel.ravel().astype(int).tolist(),
        "wind": [[round(float(s), 2), round(float(d), 1)] for s, d in result.wind_schedule],
        "metrics": [{k: x[k] for k in ("minutes", "burning", "burned", "burned_ha", "fire_area_ha",
                                        "perimeter_m", "front_distance_m", "ros_m_per_min",
                                        "intensity_class", "boundary_reached")} for x in m],
    })
    return base


def missing_key_card():
    st.markdown("""
    <div class="info-box" style="border-color:rgba(255,107,74,.45);">
      <b style="color:#ff6b4a;">Google Maps API key not set.</b> Add <code>GOOGLE_MAPS_API_KEY=...</code>
      (Maps JavaScript API enabled, billing on) to <code>.env</code> or Streamlit Secrets and restart.
      The simulation, analytics and charts below still run without the map.
    </div>
    """, unsafe_allow_html=True)


def render_geo_fire_map(payload: dict, height: int = 640):
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = _HTML.replace("__HEIGHT__", str(int(height))).replace("__PAYLOAD__", blob)
    components.html(html, height=height + 4, scrolling=False)


# ── browser side ──────────────────────────────────────────────────────────── #

_HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
:root{--bg:#0a0c10;--surface:rgba(13,16,21,.86);--border:#303a48;--text:#e8edf3;--muted:#8a96a6;--accent:#ff6b35;}
html,body{margin:0;height:100%;background:var(--bg);font-family:'Plus Jakarta Sans','Segoe UI',system-ui,sans-serif;color:var(--text);overflow:hidden}
#wrap{position:relative;width:100%;height:__HEIGHT__px;border-radius:12px;overflow:hidden;border:1px solid #232b36;background:#0a0c10}
#map{position:absolute;inset:0}
#fx{position:absolute;inset:0;pointer-events:none;z-index:5}
#scarfx{position:absolute;inset:0;pointer-events:none;z-index:4;mix-blend-mode:multiply}
.panel{position:absolute;z-index:10;background:var(--surface);border:1px solid var(--border);border-radius:10px;backdrop-filter:blur(6px);font-size:11px}
#hud{top:10px;left:10px;padding:10px 12px;min-width:200px;max-width:240px;pointer-events:none}
#hud .t{font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);font-weight:700}
#hud .nm{font-size:13px;font-weight:700;margin:2px 0 6px}
#hud .clock{font-family:'JetBrains Mono',Consolas,monospace;font-size:20px;font-weight:700;margin:2px 0 6px}
#hud .g{display:grid;grid-template-columns:1fr auto;gap:2px 10px}
#hud .g span:nth-child(odd){color:var(--muted)} #hud .g span:nth-child(even){text-align:right;font-weight:600}
#hud .warn{margin-top:6px;color:#ff6b4a;font-weight:700;display:none}
#hud .sim{margin-top:6px;color:var(--muted);font-size:10px;line-height:1.35}
#legend{left:10px;bottom:62px;padding:8px 10px;line-height:1.7;pointer-events:none}
#legend i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px;vertical-align:-1px;border:1px solid rgba(255,255,255,.18)}
#cam{right:10px;top:56px;padding:6px;display:flex;flex-direction:column;gap:4px}
#layers{left:50%;transform:translateX(-50%);top:10px;padding:5px 8px;display:flex;align-items:center;gap:4px;white-space:nowrap}
#layers .lbl{color:var(--muted);font-size:10px;letter-spacing:.12em;text-transform:uppercase;margin-right:4px}
#ctrl{left:50%;transform:translateX(-50%);bottom:10px;padding:6px 10px;display:flex;align-items:center;gap:6px;white-space:nowrap}
button{background:#161b23;color:var(--text);border:1px solid var(--border);border-radius:7px;padding:5px 9px;font-family:inherit;font-weight:600;font-size:11px;cursor:pointer}
button:hover{border-color:var(--accent)} button.on{border-color:var(--accent);color:#ffb347;background:rgba(255,107,53,.12)}
button:disabled{opacity:.4;cursor:default}
#layers button{padding:4px 7px;font-size:10.5px}
#tl{width:260px;accent-color:#ff6b35}
#tlab{font-family:'JetBrains Mono',Consolas,monospace;min-width:84px;text-align:right}
#err{position:absolute;inset:0;z-index:30;display:none;align-items:center;justify-content:center;text-align:center;padding:30px;background:rgba(10,12,16,.95);color:#ff6b4a;font-size:13px}
#note{right:10px;bottom:62px;padding:6px 9px;max-width:230px;color:var(--muted);display:none}
.sep{width:1px;height:18px;background:var(--border);margin:0 2px}
</style></head>
<body>
<div id="wrap">
  <div id="map"></div>
  <canvas id="scarfx"></canvas>
  <canvas id="fx"></canvas>
  <div id="hud" class="panel">
    <div class="t">Simulated fire spread</div>
    <div class="nm" id="hname"></div>
    <div class="clock" id="hclock">T+0 min</div>
    <div class="g">
      <span>Burning cells</span><span id="hburn">0</span>
      <span>Burned cells</span><span id="hburned">0</span>
      <span>Burned area</span><span id="harea">0 ha</span>
      <span>Fire perimeter</span><span id="hper">0 m</span>
      <span>Front distance</span><span id="hdist">0 m</span>
      <span>Rate of spread</span><span id="hros">0 m/min</span>
      <span>Intensity</span><span id="hint">-</span>
      <span>Wind</span><span id="hwind"></span>
    </div>
    <div class="warn" id="hwarn">Fire reached the simulation boundary</div>
    <div class="sim" id="hsim"></div>
  </div>
  <div id="layers" class="panel">
    <span class="lbl">Layers</span>
    <button class="tog on" data-t="fire">Fire</button><button class="tog on" data-t="smoke">Smoke</button>
    <button class="tog on" data-t="embers">Embers</button><button class="tog on" data-t="ash">Ash</button>
    <button class="tog on" data-t="heat">Heat</button><span class="sep"></span>
    <button class="tog on" data-t="grid">Grid</button><button class="tog" data-t="front">Front</button>
    <button class="tog on" data-t="wind">Wind</button><button class="tog" data-t="veg">Vegetation</button>
  </div>
  <div id="legend" class="panel">
    <div><i style="background:transparent;border:2px solid #4dd0e1;border-radius:50%"></i><span id="lhot">Observed hotspot</span></div>
    <div><i style="background:linear-gradient(0deg,#b3260a,#ff9a2e,#fff0b0)"></i>Burning (simulated)</div>
    <div><i style="background:#2a160e"></i>Recently burned</div>
    <div><i style="background:#3a3532"></i>Burned / ash</div>
    <div><i style="background:#3f5878"></i>Non-fuel</div>
    <div><i style="background:transparent;border:2px solid #ffd166"></i>Ignition</div>
    <div><i style="background:linear-gradient(90deg,transparent,#e8edf3)"></i>Simulated spread direction</div>
    <div><i style="background:linear-gradient(90deg,#3b3836,#a9a7a3)"></i>Smoke (wind-driven)</div>
  </div>
  <div id="cam" class="panel">
    <button id="cOver">Overview</button>
    <button id="cSim">Simulation</button>
    <button id="cFront">Fire front</button>
    <button id="cClose">Close-up</button>
    <button id="cReset">Reset view</button>
    <button id="cTilt">3D tilt</button>
    <button id="cFull">Fullscreen</button>
  </div>
  <div id="ctrl" class="panel">
    <button id="bPlay">Run</button>
    <button id="bReset">Reset</button>
    <span class="sep"></span>
    <button class="spd on" data-s="1">1×</button><button class="spd" data-s="2">2×</button><button class="spd" data-s="5">5×</button>
    <span class="sep"></span>
    <input id="tl" type="range" min="0" max="0" step="0.01" value="0">
    <span id="tlab">T+0 min</span>
  </div>
  <div id="note" class="panel"></div>
  <div id="err"></div>
</div>
<script>
"use strict";
const P = __PAYLOAD__;
const F = P.focus, N = F.n, CELL = F.cell_m, NC = N * N;
const LAT0 = F.lat, LON0 = F.lon;
const MLAT = 111320, MLON = 111320 * Math.cos(LAT0 * Math.PI / 180);
const HALF = N * CELL / 2;
const SEC_PER_STEP = 1.1;            // 1x playback: one CA step per 1.1 s
const DRIFT_K = 9;                   // visual drift (m per real second per m/s of wind) x playback speed: smoke
                                     // must outrun the time-compressed fire front, as it does in reality
const CAP = {flame: 950, smoke: 380, ember: 200, ash: 170, streak: 70, emitters: 110};

// ── helpers ────────────────────────────────────────────────────────────────
function rng(seed){return function(){seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
const R = rng(1234567);
const clamp = (v,a,b)=>v<a?a:(v>b?b:v);
const hash = (i,k)=>{const h=Math.sin(i*12.9898+k*78.233)*43758.5453;return h-Math.floor(h);};   // deterministic per cell
function ll(x,y){return {lat: LAT0 + y / MLAT, lng: LON0 + x / MLON};}
function cellXY(i){const r=(i/N)|0,c=i%N;return [(c+0.5-N/2)*CELL,(N/2-r-0.5)*CELL];}
function showErr(msg){const e=document.getElementById('err');e.innerHTML=msg;e.style.display='flex';}
window.gm_authFailure=function(){showErr('<div><b>Google Maps authentication failed.</b><br>Check that the API key has the Maps JavaScript API enabled, billing is active and the key is not restricted away from localhost.</div>');};

// ── simulation clock (browser side; Streamlit never reruns during playback) ──
const LAST = P.lastStep || 0;
let simT = 0, playing = false, speed = 1;
function windAt(t){return P.wind[clamp(Math.floor(t),0,P.wind.length-1)];}
function windVec(t){const w=windAt(t);const to=(w[1]+180)*Math.PI/180;return [Math.sin(to)*w[0],Math.cos(to)*w[0],w[0],w[1]];}
const IGN = P.ign, OUT = P.out.map((o,i)=>o>=0?o:(IGN[i]>=0?IGN[i]+1:-1)), INT = P.inten, NF = P.nonfuel;
// per-cell CA state at clock t: 0 unburned, 1 burning, 2 recently burned, 3 burned, 4 non-fuel, 5 ignition (before a run)
function cellState(i,t){
  if(NF[i]) return 4;
  if(!P.hasRun) return IGN[i]===0?5:0;
  const a=IGN[i]; if(a<0||t<a) return 0;
  if(t<OUT[i]) return 1;
  return (t-OUT[i])<3?2:3;
}
// Visual interpolation between CA snapshots (the CA decides WHICH cell burns WHEN; this
// only shapes the transition): a cell that the CA ignites at step a pre-heats over the last
// 0.4 step, flares up, burns, dies down after burn-out, then smoulders.
function fireStrength(i,t){
  const a=IGN[i]; if(!P.hasRun||a<0||NF[i]||t<a-0.4) return 0;
  if(t<a) return 0.18*(t-(a-0.4))/0.4;
  const o=OUT[i], p=(t-a)/(o-a);
  if(p<1) return p<0.25 ? 0.3+0.7*(p/0.25) : (p<0.75 ? 1 : 1-0.4*(p-0.75)/0.25);
  const s=t-o; return s<1 ? 0.45*Math.pow(1-s,1.8) : 0;    // dying flames over the burned-out cell
}
function smokeStrength(i,t){
  const a=IGN[i]; if(!P.hasRun||a<0||t<a||NF[i]) return 0;
  const o=OUT[i]; if(t<o) return 0.5+0.5*clamp((t-a)/0.3,0,1);
  const s=t-o; return s<3 ? 0.55*Math.pow(1-s/3,1.3) : 0;    // smoke decays gradually while it smoulders
}

// ── sprites (rendered once, reused) ────────────────────────────────────────
function sprite(w,h,fn){const c=document.createElement('canvas');c.width=w;c.height=h;fn(c.getContext('2d'),w,h);return c;}
function radial(stops){return (g,w,h)=>{const r=g.createRadialGradient(w/2,h/2,0,w/2,h/2,w/2);stops.forEach(([o,c])=>r.addColorStop(o,c));g.fillStyle=r;g.fillRect(0,0,w,h);};}
// Procedural flame: a ragged crown of 3-5 overlapping tongues of different heights (never a single
// clean cone); red rim -> orange body -> yellow-white core, softened with blur.
function tongue(seed){
  const r=rng(seed);
  return sprite(128,176,(g,w,h)=>{
    const shape=(cx,by,bw,th,wob)=>{
      const k1=(r()-0.5)*wob,k2=(r()-0.5)*wob,tipx=cx+(r()-0.5)*wob;
      g.beginPath();g.moveTo(cx-bw,by);
      g.bezierCurveTo(cx-bw*1.1+k1,by-th*0.3,cx-bw*0.5+k2,by-th*0.72,tipx,by-th);
      g.bezierCurveTo(cx+bw*0.5-k2,by-th*0.72,cx+bw*1.1-k1,by-th*0.3,cx+bw,by);
      g.quadraticCurveTo(cx,by+bw*0.3,cx-bw,by);g.closePath();
    };
    const by=h-8, n=3+((r()*3)|0), tongues=[];
    for(let j=0;j<n;j++) tongues.push([w/2+(r()-0.5)*w*0.28, w*(0.08+r()*0.07), h*(0.45+r()*0.42)]);
    tongues.sort((a,b)=>b[2]-a[2]);
    g.filter='blur(3.5px)';
    for(const [cx,bw,th] of tongues){                     // outer flame body
      shape(cx,by,bw,th,w*0.25);
      const gr=g.createLinearGradient(0,by,0,by-th);
      gr.addColorStop(0,'rgba(200,50,8,0)');gr.addColorStop(0.06,'rgba(215,65,12,0.75)');gr.addColorStop(0.22,'rgba(255,130,28,0.95)');gr.addColorStop(0.55,'rgba(250,100,18,0.85)');
      gr.addColorStop(0.85,'rgba(200,45,8,0.5)');gr.addColorStop(1,'rgba(120,18,4,0)');g.fillStyle=gr;g.fill();}
    g.globalCompositeOperation='lighter';
    for(const [cx,bw,th] of tongues){                     // bright inner core, lower and narrower
      shape(cx+(r()-0.5)*3,by-2,bw*0.45,th*0.45,w*0.12);
      const gr=g.createLinearGradient(0,by,0,by-th*0.45);
      gr.addColorStop(0,'rgba(255,245,200,0)');gr.addColorStop(0.1,'rgba(255,245,200,0.85)');gr.addColorStop(0.4,'rgba(255,215,120,0.7)');gr.addColorStop(1,'rgba(255,150,40,0)');
      g.fillStyle=gr;g.fill();}
    // feather the sprite borders: blur can bleed to the canvas edge, and hundreds of additive
    // copies would then show faint rectangles
    g.filter='none'; g.globalCompositeOperation='destination-in';
    const mx=g.createLinearGradient(0,0,w,0); mx.addColorStop(0,'rgba(0,0,0,0)');mx.addColorStop(0.18,'#000');mx.addColorStop(0.82,'#000');mx.addColorStop(1,'rgba(0,0,0,0)');
    g.fillStyle=mx; g.fillRect(0,0,w,h);
    const my=g.createLinearGradient(0,0,0,h); my.addColorStop(0,'rgba(0,0,0,0)');my.addColorStop(0.08,'#000');my.addColorStop(1,'#000');
    g.fillStyle=my; g.fillRect(0,0,w,h);
  });
}
const SPR = {
  tongue: Array.from({length:12},(_,k)=>tongue(101+k*17)),
  glow: sprite(128,128,radial([[0,'rgba(255,150,50,.9)'],[0.3,'rgba(255,100,20,.45)'],[0.7,'rgba(200,50,0,.12)'],[1,'rgba(150,30,0,0)']])),
  bed: sprite(128,128,radial([[0,'rgba(255,190,90,.9)'],[0.22,'rgba(240,100,25,.7)'],[0.55,'rgba(150,30,6,.4)'],[1,'rgba(60,10,0,0)']])),
  glint: sprite(16,16,radial([[0,'rgba(255,240,190,1)'],[0.35,'rgba(255,140,40,.9)'],[1,'rgba(200,40,0,0)']])),
  ash: sprite(8,8,radial([[0,'rgba(215,210,205,.95)'],[1,'rgba(170,165,160,0)']])),
  smoke: [], smokeShadow: [], tree: [], treeLit: null, treeBurnt: null
};
// Smoke puffs: fractal clusters of soft blobs (irregular, never a clean circle), pre-tinted in
// four shades: charcoal (dense, near intense fire) -> dark grey -> grey -> light haze.
const SHADES=[[92,84,78],[134,128,122],[178,175,171],[210,208,205]];   // fire-darkened -> sunlit grey-white
const PUFFS=[];
for(let k=0;k<8;k++){
  const r2=rng(77+k*13), blobs=[];
  for(let j=0;j<34;j++){const a=r2()*Math.PI*2,d=Math.pow(r2(),0.7)*0.27,rad=0.07+r2()*0.2;blobs.push([0.5+Math.cos(a)*d,0.5+Math.sin(a)*d,rad,0.12+r2()*0.16,(r2()-0.5)*26]);}
  PUFFS.push(blobs);
}
function puff(blobs,rgb,alphaMul){
  return sprite(128,128,(g,s)=>{for(const [x,y,rad,al,dv] of blobs){
    // each lobe lit from the north-west (top-left) and shadowed on the far side -> billowing volume
    const X=x*s,Y=y*s,RR=rad*s,gr=g.createRadialGradient(X-RR*0.35,Y-RR*0.35,0,X,Y,RR);
    const hi=rgb.map(v=>clamp(Math.round(v+dv+30),0,255)), lo=rgb.map(v=>clamp(Math.round(v+dv-28),0,255));
    gr.addColorStop(0,`rgba(${hi[0]},${hi[1]},${hi[2]},${al*alphaMul})`);gr.addColorStop(0.55,`rgba(${rgb[0]},${rgb[1]},${rgb[2]},${al*alphaMul*0.7})`);
    gr.addColorStop(0.85,`rgba(${lo[0]},${lo[1]},${lo[2]},${al*alphaMul*0.3})`);gr.addColorStop(1,`rgba(${lo[0]},${lo[1]},${lo[2]},0)`);
    g.fillStyle=gr;g.fillRect(0,0,s,s);}});
}
SHADES.forEach(rgb=>SPR.smoke.push(PUFFS.map(b=>puff(b,rgb,1.6))));
SPR.smokeShadow=PUFFS.map(b=>puff(b,[0,0,0],1.2));
// vegetation-density representation (off by default; the satellite already shows the real canopy)
function canopy(seed,base,light,rim){
  const r2=rng(seed);
  return sprite(48,48,(g,s)=>{const lobes=3+((r2()*4)|0);
    for(let j=0;j<lobes;j++){const a=r2()*Math.PI*2,d=s*0.12*r2(),x=s/2+Math.cos(a)*d,y=s/2+Math.sin(a)*d,rad=s*(0.2+r2()*0.14);
      const gr=g.createRadialGradient(x-rad*0.35,y-rad*0.35,rad*0.1,x,y,rad);
      gr.addColorStop(0,light);gr.addColorStop(0.55,base);gr.addColorStop(0.9,rim);gr.addColorStop(1,'rgba(0,0,0,0)');
      g.fillStyle=gr;g.beginPath();g.arc(x,y,rad,0,Math.PI*2);g.fill();}});
}
[['#2c4a26','#4f7a3a','#16260f'],['#355a2b','#5f8c43','#1a2c12'],['#28452e','#4b7350','#132416'],['#3d5a2c','#6b8a45','#1e2c12']]
  .forEach((c,k)=>SPR.tree.push(canopy(900+k*7,c[0],c[1],c[2])));
SPR.treeLit=canopy(31,'#8a3a12','#ffb347','#3a1206');
SPR.treeBurnt=canopy(57,'#221d1a','#3a332e','#0d0b0a');
const TREES=[];
(function(){
  const r2=rng(4242);
  const dens=(x,y)=>0.55+0.3*Math.sin(x/37+1.3)*Math.cos(y/53-0.4)+0.25*Math.sin((x+y)/23);
  for(let i=0;i<NC;i++){ if(NF[i]) continue; const [cx,cy]=cellXY(i);
    let k=Math.round(clamp(dens(cx,cy),0.05,1.1)*9);
    if(P.slopePct) k=Math.round(k*clamp(1-P.slopePct[i]/120,0.35,1));
    for(let j=0;j<k;j++) TREES.push({x:cx+(r2()-0.5)*CELL*0.95,y:cy+(r2()-0.5)*CELL*0.95,r:1.6+r2()*r2()*4.2,v:(r2()*SPR.tree.length)|0,c:i});}
  TREES.sort((a,b)=>b.y-a.y);
})();

// ── particles (pooled; nothing is created or destroyed per frame) ──────────
function makePool(n){const a=[];for(let i=0;i<n;i++)a.push({on:false});return a;}
const POOL={flame:makePool(CAP.flame),smoke:makePool(CAP.smoke),ember:makePool(CAP.ember),ash:makePool(CAP.ash),streak:makePool(CAP.streak)};
const LIVE={flame:0,smoke:0,ember:0,ash:0,streak:0};
function spawn(kind,o){const p=POOL[kind];for(let i=0;i<p.length;i++){if(!p[i].on){Object.assign(p[i],o);p[i].on=true;p[i].age=0;LIVE[kind]++;return p[i];}}return null;}
function clearParticles(){for(const k in POOL){POOL[k].forEach(p=>p.on=false);LIVE[k]=0;}for(const k in ACC)ACC[k].fill(0);}
const ACC={flame:new Float32Array(NC),smoke:new Float32Array(NC),ember:new Float32Array(NC),ash:new Float32Array(NC)};

// ── map + projection ───────────────────────────────────────────────────────
let map=null, proj=null, data=null, focusLine=null, cellFeat=new Array(NC), cellCache=new Int8Array(NC).fill(-1);
let H=null, PXM=1, TILTF=0.2, ZOOM=12, budget=1.0;
const show={fire:true,smoke:true,embers:true,ash:true,heat:true,grid:true,front:false,wind:true,veg:false};
const cvs=document.getElementById('fx'), ctx=cvs.getContext('2d');
// burn scar lives on its own canvas, blended with the satellite imagery by CSS multiply
const cvs2=document.getElementById('scarfx'), ctx2=cvs2.getContext('2d');
const DPR0=Math.min(window.devicePixelRatio||1,1.5); let RS=1, DPR=DPR0, CW=0, CH=0;   // RS: adaptive render scale
function resize(){const w=document.getElementById('wrap');CW=w.clientWidth;CH=w.clientHeight;DPR=DPR0*RS;cvs.width=Math.round(CW*DPR);cvs.height=Math.round(CH*DPR);cvs.style.width=CW+'px';cvs.style.height=CH+'px';
  cvs2.width=cvs.width;cvs2.height=cvs.height;cvs2.style.width=CW+'px';cvs2.style.height=CH+'px';}
new ResizeObserver(resize).observe(document.getElementById('wrap')); resize();

// Ground-plane homography (local metres -> container pixels) from 4 projected corners.
// Every effect is positioned in metres east/north of the focus centre (i.e. lat/lon) and
// projected through it each frame, so it stays on the ground through pan/zoom/rotate/tilt.
function solveH(src,dst){
  const A=[],b=[];
  for(let k=0;k<4;k++){const [x,y]=src[k],[u,v]=dst[k];
    A.push([x,y,1,0,0,0,-u*x,-u*y]);b.push(u);A.push([0,0,0,x,y,1,-v*x,-v*y]);b.push(v);}
  for(let i=0;i<8;i++){let m=i;for(let r=i+1;r<8;r++)if(Math.abs(A[r][i])>Math.abs(A[m][i]))m=r;
    [A[i],A[m]]=[A[m],A[i]];[b[i],b[m]]=[b[m],b[i]];const d=A[i][i];if(Math.abs(d)<1e-12)return null;
    for(let r=0;r<8;r++){if(r===i)continue;const f=A[r][i]/d;if(!f)continue;for(let c=i;c<8;c++)A[r][c]-=f*A[i][c];b[r]-=f*b[i];}}
  return b.map((v,i)=>v/A[i][i]);
}
function W2S(x,y){const h=H,w=h[6]*x+h[7]*y+1;return [(h[0]*x+h[1]*y+h[2])/w,(h[3]*x+h[4]*y+h[5])/w];}
function updateProjection(){
  if(!proj) return false;
  const E=Math.max(HALF*4,1500), src=[[-E,-E],[E,-E],[E,E],[-E,E]], dst=[];
  for(const [x,y] of src){const p=proj.fromLatLngToContainerPixel(new google.maps.LatLng(ll(x,y)));if(!p)return false;dst.push([p.x,p.y]);}
  H=solveH(src,dst); if(!H) return false;
  const a=W2S(0,0),c=W2S(20,0); PXM=Math.hypot(c[0]-a[0],c[1]-a[1])/20;
  const tilt=(map.getTilt&&map.getTilt())||0; TILTF=0.22+0.78*Math.sin(tilt*Math.PI/180);
  ZOOM=map.getZoom()||12; return true;
}

// ── data layer: the computational grid only (fire states are drawn as VFX, not tiles) ──
function styleFn(feat){const s=feat.getProperty('s')|0, g=show.grid;
  if(s===4) return {fillColor:'#3f5878',fillOpacity:0.30,strokeColor:'#5b7aa3',strokeOpacity:g?0.3:0,strokeWeight:0.6,clickable:false};
  if(s===5) return {fillColor:'#ffd166',fillOpacity:0.15,strokeColor:'#ffd166',strokeOpacity:0.95,strokeWeight:2,clickable:false,zIndex:5};
  if(s===1) return {fillOpacity:0,strokeColor:'#ffb347',strokeOpacity:g?0.45:0,strokeWeight:0.8,clickable:false,zIndex:2};
  return {fillOpacity:0,strokeColor:'#ffffff',strokeOpacity:g?(s===0?0.14:0.08):0,strokeWeight:0.6,clickable:false};}
function buildCells(){
  data=new google.maps.Data({map});
  const dlat=CELL/MLAT,dlon=CELL/MLON;
  for(let i=0;i<NC;i++){
    const r=(i/N)|0,c=i%N,n=F.north-r*dlat,s=n-dlat,w=F.west+c*dlon,e=w+dlon;
    cellFeat[i]=data.add({geometry:new google.maps.Data.Polygon([[{lat:n,lng:w},{lat:n,lng:e},{lat:s,lng:e},{lat:s,lng:w}]]),properties:{s:0}});
  }
  data.setStyle(styleFn);
  focusLine=new google.maps.Polyline({map,path:[{lat:F.north,lng:F.west},{lat:F.north,lng:F.east},{lat:F.south,lng:F.east},{lat:F.south,lng:F.west},{lat:F.north,lng:F.west}],
    strokeColor:'#ffd166',strokeOpacity:0.95,strokeWeight:2.2,clickable:false,zIndex:20});
}
function syncCells(t){                         // restyle only the cells whose CA state changed
  let boundary=false;
  for(let i=0;i<NC;i++){const s=cellState(i,t);
    if(s!==cellCache[i]){cellCache[i]=s;if(cellFeat[i])cellFeat[i].setProperty('s',s);}
    if(s>=1&&s<=3){const r=(i/N)|0,c=i%N;if(r===0||c===0||r===N-1||c===N-1)boundary=true;}}
  if(focusLine) focusLine.setOptions({strokeColor:boundary?'#ff4d3a':'#ffd166'});
  document.getElementById('hwarn').style.display=boundary?'block':'none';
  scarDirty=true; frontDirty=true;
}
function hotspotLayer(){
  const obs=P.hotspotKind==='observed';
  document.getElementById('lhot').textContent=obs?'Observed hotspot (NASA FIRMS VIIRS)':'Demo hotspot (synthetic, not observed)';
  (P.hotspots||[]).forEach(h=>{
    new google.maps.Circle({map,center:{lat:h.lat,lng:h.lon},radius:187.5,strokeColor:obs?'#4dd0e1':'#b39ddb',strokeOpacity:0.95,strokeWeight:2,
      fillColor:obs?'#4dd0e1':'#b39ddb',fillOpacity:0.08,clickable:false,zIndex:15});
  });
}

// ── burn scar: organic charred ground, rendered off-screen in metre space ──
// Each affected cell lays down several soft, irregular char blobs that overlap their
// neighbours, so the scar is one continuous organic shape (never square tiles). Colour moves
// from scorched (under the flames) -> fresh char with a warm tint -> grey ash over 3 steps.
const SCAR_RES=1.6, SCAR_E=HALF+CELL;                      // px per metre, half-extent (m)
const scar=document.createElement('canvas'); scar.width=scar.height=Math.round(2*SCAR_E*SCAR_RES);
const sctx=scar.getContext('2d'); let scarDirty=true, scarT=-1;
function renderScar(t){
  // Colours here are MULTIPLY factors applied to the satellite image (drawScar), so the real
  // ground texture stays visible: scorched (warm, light) under the flames -> fresh char
  // (very dark, brownish) -> grey ash over 3 CA steps.
  const S=scar.width; sctx.clearRect(0,0,S,S);
  const toPx=(x,y)=>[(x+SCAR_E)*SCAR_RES,(SCAR_E-y)*SCAR_RES];
  for(let i=0;i<NC;i++){
    const a=IGN[i]; if(!P.hasRun||a<0||t<a||NF[i]) continue;
    const o=OUT[i], burning=t<o, prog=burning?clamp((t-a)/(o-a),0,1):1, age=burning?0:t-o;
    const fresh=clamp(1-age/3,0,1);
    let c;
    if(burning) c=[150-80*prog,95-50*prog,70-35*prog];
    else c=[70+45*(1-fresh),48+60*(1-fresh),38+64*(1-fresh)];
    const alpha=burning?0.45+0.45*prog:0.92;
    const [x,y]=cellXY(i);
    for(let k=0;k<4;k++){
      const ox=(hash(i,k)-0.5)*CELL*0.8, oy=(hash(i,k+9)-0.5)*CELL*0.8, rad=CELL*(0.48+0.34*hash(i,k+17))*SCAR_RES;
      const v=0.85+0.3*hash(i,k+23), cc=c.map(z=>Math.round(clamp(z*v,0,255)));
      const [px,py]=toPx(x+ox,y+oy), gr=sctx.createRadialGradient(px,py,0,px,py,rad);
      gr.addColorStop(0,`rgba(${cc[0]},${cc[1]},${cc[2]},${alpha})`);gr.addColorStop(0.6,`rgba(${cc[0]},${cc[1]},${cc[2]},${alpha*0.8})`);
      gr.addColorStop(1,`rgba(${cc[0]},${cc[1]},${cc[2]},0)`);
      sctx.fillStyle=gr; sctx.fillRect(px-rad,py-rad,2*rad,2*rad);
    }
  }
  scarT=t; scarDirty=false;
}
function drawScar(){
  const S=scar.width, p0=W2S(-SCAR_E,SCAR_E), p1=W2S(SCAR_E,SCAR_E), p2=W2S(-SCAR_E,-SCAR_E);
  ctx2.setTransform(1,0,0,1,0,0); ctx2.clearRect(0,0,cvs2.width,cvs2.height);
  ctx2.setTransform(DPR*(p1[0]-p0[0])/S,DPR*(p1[1]-p0[1])/S,DPR*(p2[0]-p0[0])/S,DPR*(p2[1]-p0[1])/S,DPR*p0[0],DPR*p0[1]);
  ctx2.drawImage(scar,0,0);
}

// ── fire front topology (recomputed only when the CA step changes) ─────────
// For every burning cell: which sides face unburned fuel. Flames are emitted mostly along
// those edges, so adjacent burning cells read as one continuous, irregular fire line.
let frontDirty=true, FRONT=[], PERIM=[];
const NB4=[[-1,0,0,1],[1,0,0,-1],[0,1,1,0],[0,-1,-1,0]];   // dr,dc, edge normal (east, north)
function rebuildFront(t){
  FRONT=[]; PERIM=[];
  const aff=i=>{const s=cellState(i,t);return s>=1&&s<=3;};
  for(let i=0;i<NC;i++){
    const r=(i/N)|0,c=i%N,st=cellState(i,t);
    if(!(st>=1&&st<=3)) continue;
    const edges=[];
    for(const [dr,dc,ex,ey] of NB4){const nr=r+dr,nc=c+dc;
      const open=nr<0||nc<0||nr>=N||nc>=N?false:(!aff(nr*N+nc)&&!NF[nr*N+nc]);
      const outside=nr<0||nc<0||nr>=N||nc>=N||!aff(nr*N+nc);
      if(open) edges.push([ex,ey]);
      if(outside) PERIM.push([i,ex,ey,st===1&&open]);}
    if(st===1) FRONT.push([i,edges]);
  }
  frontDirty=false;
}

// ── camera ─────────────────────────────────────────────────────────────────
let camAnim=null;
function flyTo(target,ms){
  const from={lat:map.getCenter().lat(),lng:map.getCenter().lng(),zoom:map.getZoom(),tilt:map.getTilt()||0,heading:map.getHeading()||0};
  const to=Object.assign({},from,target), t0=performance.now();
  if(camAnim) cancelAnimationFrame(camAnim);
  const ease=x=>x<.5?4*x*x*x:1-Math.pow(-2*x+2,3)/2;
  const step=now=>{const k=ease(clamp((now-t0)/ms,0,1));
    map.moveCamera({center:{lat:from.lat+(to.lat-from.lat)*k,lng:from.lng+(to.lng-from.lng)*k},zoom:from.zoom+(to.zoom-from.zoom)*k,
      tilt:from.tilt+(to.tilt-from.tilt)*k,heading:from.heading+(((to.heading-from.heading+540)%360)-180)*k});
    if(k<1) camAnim=requestAnimationFrame(step); else camAnim=null;};
  camAnim=requestAnimationFrame(step);
}
function centroid(t,states){let sx=0,sy=0,k=0;for(let i=0;i<NC;i++){if(states.includes(cellState(i,t))){const [x,y]=cellXY(i);sx+=x;sy+=y;k++;}}return k?[sx/k,sy/k]:null;}
function frontCentre(){const c=centroid(simT,[1,5])||centroid(simT,[2,3])||[0,0];return ll(c[0],c[1]);}
const IGN0=(()=>{let sx=0,sy=0,k=0;for(let i=0;i<NC;i++)if(IGN[i]===0){const [x,y]=cellXY(i);sx+=x;sy+=y;k++;}return k?[sx/k,sy/k]:[0,0];})();
const SIMZOOM=17;
function camSim(ms){flyTo({lat:LAT0,lng:LON0,zoom:SIMZOOM,tilt:0,heading:0},ms||1200);}
function note(msg){const n=document.getElementById('note');n.textContent=msg;n.style.display='block';clearTimeout(note.t);note.t=setTimeout(()=>n.style.display='none',4000);}
function checkTilt(t){setTimeout(()=>{if((map.getTilt()||0)<1&&t>0)note('3D tilt is not available for this imagery here; the view stays top-down.');},800);}

// ── UI ─────────────────────────────────────────────────────────────────────
const fmtT=t=>'T+'+(t*(P.stepMin||0)).toFixed(0)+' min';
const tl=document.getElementById('tl'), bPlay=document.getElementById('bPlay');
tl.max=String(LAST); tl.disabled=!P.hasRun; bPlay.disabled=!P.hasRun;
function setPlaying(v){playing=v&&P.hasRun;bPlay.textContent=playing?'Pause':(simT>0&&simT<LAST?'Resume':'Run');bPlay.classList.toggle('on',playing);}
bPlay.onclick=()=>{if(simT>=LAST){simT=0;clearParticles();}setPlaying(!playing);};
document.getElementById('bReset').onclick=()=>{simT=0;clearParticles();setPlaying(false);tl.value='0';updateHud(true);if(data)syncCells(0);};
document.querySelectorAll('.spd').forEach(b=>b.onclick=()=>{speed=+b.dataset.s;document.querySelectorAll('.spd').forEach(x=>x.classList.toggle('on',x===b));});
document.querySelectorAll('.tog').forEach(b=>b.onclick=()=>{const k=b.dataset.t;show[k]=!show[k];b.classList.toggle('on',show[k]);if(k==='grid'&&data)data.setStyle(styleFn);});
tl.oninput=()=>{simT=+tl.value;setPlaying(false);updateHud(true);if(data)syncCells(simT);};
document.getElementById('cOver').onclick=()=>flyTo({lat:LAT0,lng:LON0,zoom:13.5,tilt:0,heading:0},1400);
document.getElementById('cSim').onclick=()=>camSim();
document.getElementById('cFront').onclick=()=>{const c=frontCentre();flyTo({lat:c.lat,lng:c.lng,zoom:18.3},1200);};
document.getElementById('cClose').onclick=()=>{const c=frontCentre();flyTo({lat:c.lat,lng:c.lng,zoom:19,tilt:60},1500);checkTilt(60);};
document.getElementById('cReset').onclick=()=>camSim(900);
document.getElementById('cTilt').onclick=()=>{const t=(map.getTilt()||0)>5?0:55;map.setTilt(t);checkTilt(t);};
document.getElementById('cFull').onclick=()=>{const w=document.getElementById('wrap');if(document.fullscreenElement)document.exitFullscreen();else if(w.requestFullscreen)w.requestFullscreen();};
document.addEventListener('fullscreenchange',()=>{const w=document.getElementById('wrap');w.style.height=document.fullscreenElement?'100vh':'__HEIGHT__px';resize();});

const COMPASS=['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'];
let lastHudStep=-2;
function updateHud(force){
  const k=clamp(Math.floor(simT+1e-6),0,Math.max(0,P.metrics.length-1));
  const w=windAt(simT);
  document.getElementById('hclock').textContent=fmtT(simT);
  document.getElementById('tlab').textContent=fmtT(simT);
  document.getElementById('hwind').textContent=w[0].toFixed(1)+' m/s from '+COMPASS[Math.round((w[1]%360)/22.5)%16]+' ('+Math.round(w[1])+'°)';
  if(!force&&k===lastHudStep) return; lastHudStep=k;
  const m=P.metrics[k], set=(id,v)=>document.getElementById(id).textContent=v;
  if(!m){set('hburn','0');set('hburned','0');set('harea','0 ha');set('hper','0 m');set('hdist','0 m');set('hros','0 m/min');set('hint','-');return;}
  set('hburn',m.burning);set('hburned',m.burned);set('harea',m.burned_ha.toFixed(2)+' ha');
  set('hper',Math.round(m.perimeter_m)+' m');set('hdist',Math.round(m.front_distance_m)+' m');
  set('hros',m.ros_m_per_min.toFixed(1)+' m/min');set('hint',m.intensity_class);
}
document.getElementById('hname').textContent=F.name;
document.getElementById('hsim').textContent=(P.hasRun?'Cellular automata, ':'Ignition shown; press Run Simulation. ')+F.size_m+' m × '+F.size_m+' m ('+(F.size_m*F.size_m/1e6).toFixed(2)+' km²), '+CELL+' m cells. Flames, smoke, embers and ash are visual representations of the simulated state.';

// ── per-frame visual effects (driven by the CA state at the current clock) ─
function lod(){ return ZOOM<14?0:(ZOOM<16?1:(ZOOM<17.6?2:3)); }   // regional / forest / simulation / close
function emitters(t){
  const list=[];
  for(let i=0;i<NC;i++){const fs=fireStrength(i,t),ss=smokeStrength(i,t);if(fs>0||ss>0)list.push([i,fs,ss]);}
  if(list.length>CAP.emitters){list.sort((a,b)=>(b[1]*INT[b[0]]+b[2]*.3)-(a[1]*INT[a[0]]+a[2]*.3));list.length=CAP.emitters;}
  return list;
}
const EDGES=new Map();
function stepFx(dt,t,now){
  const L=lod(), wv=windVec(t), wx=wv[0], wy=wv[1], spd=wv[2], q=budget;
  const tk=playing?speed:1;
  if(frontDirty) rebuildFront(t);
  EDGES.clear(); for(const [i,e] of FRONT) EDGES.set(i,e);
  for(const [i,fs,ss] of emitters(t)){
    const I=Math.max(INT[i],0.12), [cx,cy]=cellXY(i), edges=EDGES.get(i)||[];
    // flames: rate, height and width all scale with the CA-derived intensity
    if(show.fire&&L>=2&&fs>0.1){
      ACC.flame[i]+=dt*fs*(14+60*I*I+30*edges.length*I)*q*(L===3?1:0.55);
      while(ACC.flame[i]>=1){ACC.flame[i]-=1; if(LIVE.flame>=CAP.flame*q) continue;
        let x=cx+(R()-.5)*CELL*.9, y=cy+(R()-.5)*CELL*.9;
        if(edges.length&&R()<0.7){const [ex,ey]=edges[(R()*edges.length)|0];   // on the edge facing unburned fuel
          const along=(R()-.5)*CELL*1.05; x=cx+ex*CELL*(0.32+R()*0.22)+(ey?along:0); y=cy+ey*CELL*(0.32+R()*0.22)+(ex?along:0);}
        const life=0.55+R()*0.65;
        spawn('flame',{x,y,z:0,life,h:(8+30*I)*(0.4+0.8*R()*R()+0.3*R())*(0.6+0.4*fs),w:(4+9*I)*(0.6+0.7*R()),
          v:(R()*SPR.tongue.length)|0,seed:R()*6.28,fl:5+R()*9,str:fs,dx:(R()-.5)*0.6,dy:(R()-.5)*0.6});}}
    if(show.smoke&&L>=1&&ss>0){
      ACC.smoke[i]+=dt*ss*(0.6+2.6*I)*q*(L===1?0.55:1);
      while(ACC.smoke[i]>=1){ACC.smoke[i]-=1; if(LIVE.smoke>=CAP.smoke*q) continue;
        const flaming=fs>0.3, dense=flaming?I:I*0.35;           // smouldering smoke is thinner and lighter
        spawn('smoke',{x:cx+(R()-.5)*CELL*.8,y:cy+(R()-.5)*CELL*.8,z:1+R()*3,vz:3+R()*3+7*dense,life:9+R()*8,
          s0:6+10*I,s1:55+130*dense+R()*45,rot:R()*6.28,vr:(R()-.5)*0.18,v:(R()*8)|0,
          shade:dense>0.65?0:(dense>0.35?1:2),den:(0.35+0.45*ss)*(0.4+0.6*I)*(flaming?1:0.45),seed:R()*100});}}
    if(show.embers&&L>=2&&fs>0.35&&I>0.25){
      ACC.ember[i]+=dt*fs*Math.pow(I,1.5)*(1.4+spd*0.25)*q*(L===3?1:0.5);
      while(ACC.ember[i]>=1){ACC.ember[i]-=1;
        spawn('ember',{x:cx+(R()-.5)*CELL*.7,y:cy+(R()-.5)*CELL*.7,z:4+R()*10*I,vx:(R()-.5)*3,vy:(R()-.5)*3,vz:7+R()*12*I,
          life:1.5+R()*2.2,sz:0.8+R()*1.6,seed:R()*6.28});}}
    if(show.ash&&L>=2){const st=cellState(i,t);
      if(st===2||(st===1&&fs>0.5)){ACC.ash[i]+=dt*(st===2?0.9:0.5)*q;
        while(ACC.ash[i]>=1){ACC.ash[i]-=1;
          spawn('ash',{x:cx+(R()-.5)*CELL,y:cy+(R()-.5)*CELL,z:3+R()*12,vz:0.4+R()*1.2,life:5+R()*5,seed:R()*6.28,sz:1+R()*1.4});}}}
  }
  if(show.wind&&L>=1&&spd>0.3&&LIVE.streak<CAP.streak*q){const E=HALF*1.6;spawn('streak',{x:(R()-.5)*2*E,y:(R()-.5)*2*E,life:1.6+R()*1.6});}
  const shear=z=>0.45+0.55*Math.min(1,z/70);
  for(const p of POOL.flame) if(p.on){p.age+=dt;if(p.age>=p.life){p.on=false;LIVE.flame--;}}
  for(const p of POOL.smoke) if(p.on){p.age+=dt;if(p.age>=p.life){p.on=false;LIVE.smoke--;continue;}
    const a=p.age/p.life, k=shear(p.z), turb=1.2+5*a;           // turbulence grows as the plume ages
    const tu=Math.sin(now*0.0007+p.seed+p.y*0.013)*turb+Math.sin(now*0.0019+p.seed*2.1)*turb*0.5;
    const tv=Math.cos(now*0.0006+p.seed*1.3+p.x*0.011)*turb+Math.cos(now*0.0023+p.seed)*turb*0.5;
    p.x+=(wx*DRIFT_K*tk*k+tu)*dt; p.y+=(wy*DRIFT_K*tk*k+tv)*dt; p.z+=p.vz*dt; p.vz=Math.max(0.6,p.vz*0.992); p.rot+=p.vr*dt;}
  for(const p of POOL.ember) if(p.on){p.age+=dt;if(p.age>=p.life||p.z<0){p.on=false;LIVE.ember--;continue;}
    const up=p.age<0.5;                                          // rise in the plume first, then drift downwind
    p.vz-=(up?1.5:5)*dt; p.px=p.x; p.py=p.y; p.pz=p.z;
    p.x+=(p.vx+wx*DRIFT_K*tk*(up?0.3:0.95)+Math.sin(now*0.01+p.seed)*2)*dt; p.y+=(p.vy+wy*DRIFT_K*tk*(up?0.3:0.95)+Math.cos(now*0.012+p.seed)*2)*dt; p.z+=p.vz*dt;}
  for(const p of POOL.ash) if(p.on){p.age+=dt;if(p.age>=p.life){p.on=false;LIVE.ash--;continue;}
    p.x+=(wx*DRIFT_K*tk*0.4+Math.sin(now*0.003+p.seed)*1.2)*dt; p.y+=(wy*DRIFT_K*tk*0.4+Math.cos(now*0.0025+p.seed)*1.2)*dt; p.z+=p.vz*dt;}
  for(const p of POOL.streak) if(p.on){p.age+=dt;if(p.age>=p.life){p.on=false;LIVE.streak--;continue;} p.x+=wx*DRIFT_K*0.9*dt;p.y+=wy*DRIFT_K*0.9*dt;}
}

// ── per-frame drawing ──────────────────────────────────────────────────────
function onScreen(s,m){return s[0]>-m&&s[1]>-m&&s[0]<CW+m&&s[1]<CH+m;}
function draw(t,now){
  ctx.setTransform(DPR,0,0,DPR,0,0); ctx.clearRect(0,0,CW,CH);
  if(!H) return;
  // Altitude -> screen offset. Flames are upright billboards anchored at their ground point;
  // smoke, embers and ash use an oblique factor (at least ~0.5 even when the map is top-down)
  // so the plume's vertical structure stays readable; the ground anchor never moves.
  const L=lod(), ex=clamp(1.2/PXM,1,2.6), up=PXM*Math.max(0.5,TILTF), wv=windVec(t);
  // screen direction of the downwind vector -> flame lean
  const c0=W2S(0,0), c1=W2S(wv[0]*10/(wv[2]||1),wv[1]*10/(wv[2]||1));
  const lean=wv[2]>0.3?clamp(wv[2]/12,0,1)*0.55*clamp((c1[0]-c0[0])/(Math.hypot(c1[0]-c0[0],c1[1]-c0[1])||1),-1,1):0;

  // 1. burn scar (organic, continuous)
  if(P.hasRun){ if(scarDirty||Math.abs(t-scarT)>0.12) renderScar(t); drawScar(); }
  else { ctx2.setTransform(1,0,0,1,0,0); ctx2.clearRect(0,0,cvs2.width,cvs2.height); }

  // 2. vegetation-density representation (optional)
  if(show.veg&&L>=2){ctx.globalCompositeOperation='source-over';
    for(const tr of TREES){const s=W2S(tr.x,tr.y);if(!onScreen(s,20))continue;
      const st=cellState(tr.c,t); let sp=SPR.tree[tr.v], r=tr.r;
      if(st===1)sp=SPR.treeLit; else if(st===2||st===3){sp=SPR.treeBurnt;r*=0.72;}
      const d=Math.max(2.5,r*2*PXM); ctx.globalAlpha=L===3?0.6:0.45; ctx.drawImage(sp,s[0]-d/2,s[1]-d/2,d,d);}}

  // 3. smoke shadows on the ground (sun from the north-west) - gives the plume volume from above
  if(show.smoke&&L>=1&&budget>0.5){ctx.globalCompositeOperation='source-over';
    for(const p of POOL.smoke) if(p.on){const a=p.age/p.life; if(a<0.15)continue;
      const s=W2S(p.x+p.z*0.55,p.y-p.z*0.55), size=Math.min(300,(p.s0+(p.s1-p.s0)*Math.sqrt(a))*PXM*Math.max(1,ex*0.8));
      if(!onScreen(s,size))continue;
      ctx.globalAlpha=clamp(0.1*Math.pow(1-a,1.2)*p.den,0,0.12);
      ctx.drawImage(SPR.smokeShadow[p.v],s[0]-size/2,s[1]-size/2,size,size);}}

  // 4. combustion base: dark-red/orange glowing bed under every burning cell, irregular and
  //    overlapping its neighbours so the front reads as one band
  if(show.fire||show.heat) for(let i=0;i<NC;i++){const fs=fireStrength(i,t);if(fs<=0)continue;
    const [x,y]=cellXY(i), s=W2S(x,y); if(!onScreen(s,140))continue;
    const I=Math.max(INT[i],0.12), fl=0.82+0.12*Math.sin(now*0.011+i*1.7)+0.08*Math.sin(now*0.029+i);
    if(show.heat){ctx.globalCompositeOperation='lighter';
      const r=Math.max(14,CELL*PXM*(0.8+1.3*I)*fl*(L===0?1.7:1));
      ctx.globalAlpha=clamp((0.07+0.2*I)*fs*fl,0,0.35); ctx.drawImage(SPR.glow,s[0]-r,s[1]-r,2*r,2*r);}
    if(show.fire&&L>=1){ctx.globalCompositeOperation='lighter';
      const nb=L>=2?5:2;                     // irregular burning fuel bed: blobs of different size,
      for(let k=0;k<nb;k++){                 // brightness and flicker, overlapping the neighbours
        const b=W2S(x+(hash(i,k)-0.5)*CELL*1.05,y+(hash(i,k+5)-0.5)*CELL*1.05);
        const fk=0.7+0.3*Math.sin(now*0.0013*(6+5*hash(i,k+3))+k*2+i);
        const rr=Math.max(4,CELL*PXM*(0.22+0.38*hash(i,k+11))*(0.6+0.6*I)*fs*fk);
        ctx.globalAlpha=clamp((0.18+0.32*hash(i,k+29))*fs*fk,0,0.5);
        ctx.drawImage(SPR.bed,b[0]-rr*1.3,b[1]-rr*0.7,2.6*rr,1.4*rr);}}}

  // 5. residual embers glowing in freshly burned ground
  if(show.heat&&L>=2){ctx.globalCompositeOperation='lighter';
    for(let i=0;i<NC;i++){if(cellState(i,t)!==2)continue; const s0=t-OUT[i]; if(s0>2)continue;
      const [x,y]=cellXY(i);
      for(let k=0;k<4;k++){const b=W2S(x+(hash(i,k+60)-0.5)*CELL,y+(hash(i,k+70)-0.5)*CELL); if(!onScreen(b,10))continue;
        const f=0.5+0.5*Math.sin(now*0.004*(1+hash(i,k))+k*3+i); ctx.globalAlpha=clamp((1-s0/2)*0.8*f,0,1);
        const d=3+2*hash(i,k+80); ctx.drawImage(SPR.glint,b[0]-d/2,b[1]-d/2,d,d);}}}

  // 6. smoke: dense charcoal at the source, rising, expanding, lightening and drifting downwind
  if(show.smoke){ctx.globalCompositeOperation='source-over';
    for(const p of POOL.smoke) if(p.on){const s=W2S(p.x,p.y), a=p.age/p.life;
      const size=Math.min(320,(p.s0+(p.s1-p.s0)*Math.sqrt(a))*PXM*Math.max(1,ex*0.8));
      const sy=s[1]-p.z*up; if(!onScreen([s[0],sy],size))continue;
      const fade=a<0.08?a/0.08:Math.pow(1-(a-0.08)/0.92,1.25);
      const shade=Math.min(3,p.shade+Math.floor(a*5));
      ctx.globalAlpha=clamp(fade*p.den*0.85,0,0.7);
      const cr=Math.cos(p.rot)*DPR, sr=Math.sin(p.rot)*DPR;
      ctx.setTransform(cr,sr,-sr,cr,s[0]*DPR,sy*DPR);
      ctx.drawImage(SPR.smoke[shade][p.v],-size/2,-size/2,size,size);}
    ctx.setTransform(DPR,0,0,DPR,0,0);}

  // 7. flames: tapered tongues rising from the base, leaning downwind, each with its own
  //    height, width, flicker and lifetime; additive so overlapping tongues build a hot core
  if(show.fire){ctx.globalCompositeOperation='screen';   // screen, not add: dense walls stay orange instead of blowing out to white
    for(const p of POOL.flame) if(p.on){const s=W2S(p.x,p.y); if(!onScreen(s,80))continue;
      const a=p.age/p.life, grow=a<0.2?a/0.2:1-0.55*(a-0.2)/0.8;
      const flick=0.85+0.15*Math.sin(now*0.001*p.fl*6.28+p.seed);
      const h=Math.max(6,p.h*PXM*ex*grow*flick), w=Math.max(3,p.w*PXM*ex*(0.8+0.3*grow));
      ctx.globalAlpha=clamp((a<0.15?a/0.15:Math.pow(1-(a-0.15)/0.85,0.8))*(0.45+0.4*p.str),0,0.75);
      const rot=lean+p.dx*0.35+Math.sin(now*0.004+p.seed)*0.08;
      const cs=Math.cos(rot)*DPR, sn=Math.sin(rot)*DPR;
      ctx.setTransform(cs,sn,-sn,cs,s[0]*DPR,s[1]*DPR);
      ctx.drawImage(SPR.tongue[p.v],-w/2,-h*0.9,w,h);}      // base sinks slightly into the glowing bed
    ctx.setTransform(DPR,0,0,DPR,0,0);}

  // 8. embers: small glowing sparks with a short trail, rising then carried downwind
  if(show.embers){ctx.globalCompositeOperation='lighter';ctx.lineCap='round';
    for(const p of POOL.ember) if(p.on&&p.px!==undefined){const s=W2S(p.x,p.y),s0=W2S(p.px,p.py); if(!onScreen(s,20))continue;
      const a=p.age/p.life, f=0.55+0.45*Math.sin(now*0.03+p.seed*9);
      const y1=s[1]-p.z*up*ex, y0=s0[1]-p.pz*up*ex, dx=s[0]-s0[0], dy=y1-y0, n=Math.hypot(dx,dy)||1, len=Math.min(2.5,0.6+n*0.5);
      ctx.globalAlpha=clamp((1-a)*f,0,0.9); ctx.strokeStyle=a<0.35?'#ffd27a':(a<0.7?'#ff9433':'#e0521c');
      ctx.lineWidth=0.9+0.5*p.sz*Math.min(1.5,PXM);
      ctx.beginPath(); ctx.moveTo(s[0],y1); ctx.lineTo(s[0]-dx/n*len,y1-dy/n*len); ctx.stroke();}}

  // 9. ash: faint grey flakes, slow, drifting farther than embers
  if(show.ash&&L>=2){ctx.globalCompositeOperation='source-over';
    for(const p of POOL.ash) if(p.on){const s=W2S(p.x,p.y); if(!onScreen(s,10))continue;
      const a=p.age/p.life; ctx.globalAlpha=0.45*Math.sin(Math.PI*a); const d=p.sz*Math.max(1,Math.min(2,PXM));
      ctx.drawImage(SPR.ash,s[0]-d/2,s[1]-p.z*up-d/2,d,d);}}

  // 10. optional fire-front outline: perimeter of the burned area, leading edge brighter
  if(show.front&&P.hasRun){ctx.globalCompositeOperation='source-over';ctx.lineWidth=1.6;
    for(const [i,ex2,ey,lead] of PERIM){const [x,y]=cellXY(i),h2=CELL/2;
      const p0=W2S(x+ex2*h2+ey*h2,y+ey*h2+ex2*h2), p1=W2S(x+ex2*h2-ey*h2,y+ey*h2-ex2*h2);
      ctx.globalAlpha=lead?0.9:0.45; ctx.strokeStyle=lead?'#ffcf6b':'#e8edf3';
      ctx.beginPath();ctx.moveTo(p0[0],p0[1]);ctx.lineTo(p1[0],p1[1]);ctx.stroke();}}

  // 11. wind streaks + simulated spread direction (ignition centroid -> current fire centroid;
  //     wind direction until the fire has moved)
  if(show.wind){ctx.globalCompositeOperation='source-over';
    const n=Math.hypot(wv[0],wv[1])||1, ux=wv[0]/n, uy=wv[1]/n;
    ctx.strokeStyle='rgba(232,237,243,0.5)';ctx.lineWidth=1;
    for(const p of POOL.streak) if(p.on){const a=p.age/p.life;ctx.globalAlpha=Math.sin(a*Math.PI)*0.4;
      const s0=W2S(p.x,p.y),s1=W2S(p.x-ux*18,p.y-uy*18);ctx.beginPath();ctx.moveTo(s0[0],s0[1]);ctx.lineTo(s1[0],s1[1]);ctx.stroke();}
    const cur=P.hasRun?(centroid(t,[1])||centroid(t,[2,3])):null;
    let dx=ux,dy=uy,ox=IGN0[0],oy=IGN0[1];
    if(cur){const mx=cur[0]-IGN0[0],my=cur[1]-IGN0[1],mm=Math.hypot(mx,my);if(mm>CELL*1.5){dx=mx/mm;dy=my/mm;ox=cur[0];oy=cur[1];}else{ox=cur[0];oy=cur[1];}}
    if(cur||(wv[2]>0.3)){const L2=HALF*0.45,a0=W2S(ox,oy),a1=W2S(ox+dx*L2,oy+dy*L2),hx=a1[0]-a0[0],hy=a1[1]-a0[1],hl=Math.hypot(hx,hy)||1;
      ctx.globalAlpha=0.75;ctx.strokeStyle='#e8edf3';ctx.lineWidth=2;ctx.setLineDash([6,5]);
      ctx.beginPath();ctx.moveTo(a0[0],a0[1]);ctx.lineTo(a1[0],a1[1]);ctx.stroke();ctx.setLineDash([]);
      const bx=hx/hl,by=hy/hl;ctx.fillStyle='#e8edf3';ctx.beginPath();ctx.moveTo(a1[0]+bx*9,a1[1]+by*9);
      ctx.lineTo(a1[0]-by*6,a1[1]+bx*6);ctx.lineTo(a1[0]+by*6,a1[1]-bx*6);ctx.closePath();ctx.fill();}}
  ctx.globalAlpha=1; ctx.globalCompositeOperation='source-over';
}

// ── main loop: one requestAnimationFrame for everything ────────────────────
let last=null, raf=null, frameMs=16, lastStep=-1;
function loop(now){
  raf=requestAnimationFrame(loop);
  if(last===null) last=now; const dt=Math.min((now-last)/1000,0.05); last=now;
  frameMs=frameMs*0.95+dt*1000*0.05;
  if(frameMs>30) budget=Math.max(0.35,budget-0.01); else if(frameMs<18) budget=Math.min(1,budget+0.005);
  // CPU-only laptops are fill-rate bound: lower the canvas resolution before dropping more effects
  if(budget<=0.36&&frameMs>32&&RS>0.6&&now-(loop.rsT||0)>1500){RS=Math.round((RS-0.1)*10)/10;resize();loop.rsT=now;}
  else if(frameMs<15&&RS<1&&now-(loop.rsT||0)>3000){RS=Math.round((RS+0.1)*10)/10;resize();loop.rsT=now;}
  if(playing){simT=Math.min(LAST,simT+dt*speed/SEC_PER_STEP);tl.value=String(simT);if(simT>=LAST)setPlaying(false);}
  const st=Math.floor(simT+1e-6);
  if(data&&st!==lastStep){lastStep=st;syncCells(simT);}
  if(map&&updateProjection()){stepFx(dt,simT,now);draw(simT,now);}
  updateHud(false);
}
document.addEventListener('visibilitychange',()=>{if(document.hidden){cancelAnimationFrame(raf);raf=null;}else if(!raf){last=null;raf=requestAnimationFrame(loop);}});
window.addEventListener('pagehide',()=>{cancelAnimationFrame(raf);raf=null;});

// ── Google Maps bootstrap (official dynamic library loader) ────────────────
(g=>{var h,a,k,p="The Google Maps JavaScript API",c="google",l="importLibrary",q="__ib__",m=document,b=window;b=b[c]||(b[c]={});var d=b.maps||(b.maps={}),r=new Set,e=new URLSearchParams,u=()=>h||(h=new Promise(async(f,n)=>{await (a=m.createElement("script"));e.set("libraries",[...r]+"");for(k in g)e.set(k.replace(/[A-Z]/g,t=>"_"+t[0].toLowerCase()),g[k]);e.set("callback",c+".maps."+q);a.src=`https://maps.googleapis.com/maps/api/js?`+e;d[q]=f;a.onerror=()=>h=n(Error(p+" could not load."));a.nonce=m.querySelector("script[nonce]")?.nonce||"";m.head.append(a)}));d[l]?console.warn(p+" only loads once. Ignoring:",g):d[l]=(f,...n)=>r.add(f)&&u().then(()=>d[l](f,...n))})({key:P.key,v:"weekly"});

async function init(){
  try{
    const {Map,RenderingType}=await google.maps.importLibrary('maps');
    map=new Map(document.getElementById('map'),{
      center:{lat:LAT0,lng:LON0},zoom:12,mapId:P.mapId,mapTypeId:'satellite',
      renderingType:RenderingType?RenderingType.VECTOR:undefined,tilt:0,heading:0,
      tiltInteractionEnabled:true,headingInteractionEnabled:true,
      mapTypeControl:true,mapTypeControlOptions:{position:google.maps.ControlPosition.TOP_RIGHT,
        style:google.maps.MapTypeControlStyle.HORIZONTAL_BAR,mapTypeIds:['satellite','hybrid','terrain','roadmap']},
      zoomControl:true,zoomControlOptions:{position:google.maps.ControlPosition.RIGHT_BOTTOM},
      scaleControl:true,streetViewControl:false,rotateControl:true,fullscreenControl:false,clickableIcons:false,
      gestureHandling:'greedy',isFractionalZoomEnabled:true});
    const ov=new google.maps.OverlayView(); ov.onAdd=function(){}; ov.onRemove=function(){};
    ov.draw=function(){proj=this.getProjection();}; ov.setMap(map);
    buildCells(); hotspotLayer(); syncCells(0); updateHud(true);
    google.maps.event.addListenerOnce(map,'idle',()=>{setTimeout(()=>{camSim(2200);
      if(P.autoplay&&P.hasRun) setTimeout(()=>setPlaying(true),2400);},350);});
    raf=requestAnimationFrame(loop);
  }catch(e){showErr('<div><b>Google Maps could not load.</b><br>'+String(e&&e.message||e)+'</div>');console.error(e);}
}
init();
</script>
</body></html>
"""
