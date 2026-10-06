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
#ctrl{left:50%;transform:translateX(-50%);bottom:10px;padding:6px 10px;display:flex;align-items:center;gap:6px;white-space:nowrap}
button{background:#161b23;color:var(--text);border:1px solid var(--border);border-radius:7px;padding:5px 9px;font:600 11px inherit;cursor:pointer}
button:hover{border-color:var(--accent)} button.on{border-color:var(--accent);color:#ffb347;background:rgba(255,107,53,.12)}
button:disabled{opacity:.4;cursor:default}
#tl{width:220px;accent-color:#ff6b35}
#tlab{font-family:'JetBrains Mono',Consolas,monospace;min-width:84px;text-align:right}
#err{position:absolute;inset:0;z-index:30;display:none;align-items:center;justify-content:center;text-align:center;padding:30px;background:rgba(10,12,16,.95);color:#ff6b4a;font-size:13px}
#note{right:10px;bottom:62px;padding:6px 9px;max-width:230px;color:var(--muted);display:none}
.sep{width:1px;height:18px;background:var(--border);margin:0 2px}
</style></head>
<body>
<div id="wrap">
  <div id="map"></div>
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
  <div id="legend" class="panel">
    <div><i style="background:transparent;border:2px solid #4dd0e1;border-radius:50%"></i><span id="lhot">Observed hotspot</span></div>
    <div><i style="background:#ff8a1e"></i>Burning (simulated)</div>
    <div><i style="background:#5a2a14"></i>Recently burned</div>
    <div><i style="background:#1a1512"></i>Burned / charred</div>
    <div><i style="background:#3f5878"></i>Non-fuel</div>
    <div><i style="background:transparent;border:2px solid #ffd166"></i>Ignition</div>
    <div><i style="background:linear-gradient(90deg,transparent,#e8edf3)"></i>Predicted spread direction</div>
    <div><i style="background:#7d7a76"></i>Smoke (wind-driven)</div>
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
    <span class="sep"></span>
    <button class="tog on" data-t="grid">Grid</button><button class="tog on" data-t="smoke">Smoke</button>
    <button class="tog on" data-t="veg">Vegetation</button><button class="tog on" data-t="wind">Wind</button>
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
const SEC_PER_STEP = 0.9;            // 1x playback: one CA step per 0.9 s
const DRIFT_K = 9;                   // visual drift (m per real second per m/s of wind), x playback speed:
                                     // smoke must outrun the time-compressed fire front, as it does in reality
const CAP = {flame: 900, smoke: 200, ember: 220, ash: 140, streak: 70, emitters: 90};

// ── helpers ────────────────────────────────────────────────────────────────
function rng(seed){return function(){seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
const R = rng(1234567);
const clamp = (v,a,b)=>v<a?a:(v>b?b:v);
function ll(x,y){return {lat: LAT0 + y / MLAT, lng: LON0 + x / MLON};}
function cellXY(i){const r=(i/N)|0,c=i%N;return [(c+0.5-N/2)*CELL,(N/2-r-0.5)*CELL];}
function showErr(msg){const e=document.getElementById('err');e.innerHTML=msg;e.style.display='flex';}
window.gm_authFailure=function(){showErr('<div><b>Google Maps authentication failed.</b><br>Check that the API key has the Maps JavaScript API enabled, billing is active and the key is not restricted away from localhost.</div>');};

// ── simulation clock (browser side; Streamlit never reruns during playback) ──
const LAST = P.lastStep || 0;
let simT = 0, playing = false, speed = 1, simDone = false;
function windAt(t){const w=P.wind[clamp(Math.floor(t),0,P.wind.length-1)];return w;}
function windVec(t){const w=windAt(t);const to=(w[1]+180)*Math.PI/180;return [Math.sin(to)*w[0],Math.cos(to)*w[0],w[0],w[1]];}
const IGN = P.ign, OUT = P.out.map((o,i)=>o>=0?o:(IGN[i]>=0?IGN[i]+1:-1)), INT = P.inten, NF = P.nonfuel;
// per-cell visual state at clock t: 0 unburned,1 burning,2 recently burned,3 charred,4 non-fuel,5 ignition (not yet run)
function cellState(i,t){
  if(NF[i]) return 4;
  if(!P.hasRun) return IGN[i]===0?5:0;
  const a=IGN[i]; if(a<0||t<a) return 0;
  if(t<OUT[i]) return 1;
  return (t-OUT[i])<3?2:3;
}
// fire / smoke strength of a cell at clock t (0..1), from the CA lifecycle
function fireStrength(i,t){
  const a=IGN[i]; if(!P.hasRun||a<0||t<a||NF[i]) return 0;
  const o=OUT[i], d=o-a, p=(t-a)/d;
  if(p<1){ const env = p<0.22 ? 0.25+0.75*(p/0.22) : (p<0.8 ? 1 : 1-0.45*(p-0.8)/0.2); return env; }
  const s=t-o; return s<0.7 ? 0.32*(1-s/0.7) : 0;        // last flames over the burned-out cell
}
function smokeStrength(i,t){
  const a=IGN[i]; if(!P.hasRun||a<0||t<a||NF[i]) return 0;
  const o=OUT[i]; if(t<o) return 0.45+0.55*clamp((t-a)/0.35,0,1);
  const s=t-o; return s<2.5 ? 0.42*(1-s/2.5) : 0;           // residual smoke while it smoulders
}

// ── sprites (rendered once) ────────────────────────────────────────────────
function sprite(sz,fn){const c=document.createElement('canvas');c.width=c.height=sz;fn(c.getContext('2d'),sz);return c;}
function radial(stops){return (g,s)=>{const r=g.createRadialGradient(s/2,s/2,0,s/2,s/2,s/2);stops.forEach(([o,c])=>r.addColorStop(o,c));g.fillStyle=r;g.fillRect(0,0,s,s);};}
const SPR = {
  flame:[radial([[0,'rgba(255,255,235,1)'],[0.25,'rgba(255,236,150,.9)'],[0.6,'rgba(255,170,40,.45)'],[1,'rgba(255,120,0,0)']]),
         radial([[0,'rgba(255,214,110,1)'],[0.35,'rgba(255,140,30,.8)'],[0.7,'rgba(240,80,10,.35)'],[1,'rgba(200,40,0,0)']]),
         radial([[0,'rgba(255,120,40,.9)'],[0.4,'rgba(210,50,10,.6)'],[0.75,'rgba(120,20,5,.25)'],[1,'rgba(60,10,0,0)']])].map(f=>sprite(64,f)),
  glow: sprite(128, radial([[0,'rgba(255,150,50,.85)'],[0.3,'rgba(255,100,20,.45)'],[0.7,'rgba(200,50,0,.12)'],[1,'rgba(150,30,0,0)']])),
  ember: sprite(16, radial([[0,'rgba(255,250,210,1)'],[0.3,'rgba(255,180,60,.9)'],[1,'rgba(255,90,0,0)']])),
  ash: sprite(8, radial([[0,'rgba(205,200,195,.95)'],[1,'rgba(160,155,150,0)']])),
  smoke: [], tree: [], treeLit: null, treeBurnt: null
};
// procedural smoke puffs: many soft blobs -> irregular, non-repeating shapes
for(let k=0;k<6;k++){
  const r2=rng(77+k*13);
  SPR.smoke.push(sprite(128,(g,s)=>{
    for(let j=0;j<22;j++){
      const a=r2()*Math.PI*2, d=r2()*s*0.24, x=s/2+Math.cos(a)*d, y=s/2+Math.sin(a)*d, rad=s*(0.12+r2()*0.2);
      const gr=g.createRadialGradient(x,y,0,x,y,rad); const v=180+Math.floor(r2()*40);
      gr.addColorStop(0,`rgba(${v},${v-4},${v-8},${0.10+r2()*0.10})`); gr.addColorStop(1,`rgba(${v},${v},${v},0)`);
      g.fillStyle=gr; g.fillRect(0,0,s,s);
    }
  }));
}
// darker, fire-lit variants for young smoke near the flames (pre-rendered: no per-frame filters)
SPR.smokeDark=SPR.smoke.map(src=>sprite(128,(g,s)=>{g.drawImage(src,0,0);g.globalCompositeOperation='source-atop';
  g.fillStyle='rgba(48,36,30,0.62)';g.fillRect(0,0,s,s);}));
// irregular canopy crowns: several soft lobes, sun-lit from the north-west, darker rim
function canopy(seed,base,light,rim){
  const r2=rng(seed);
  return sprite(48,(g,s)=>{
    const lobes=3+((r2()*4)|0);
    for(let j=0;j<lobes;j++){
      const a=r2()*Math.PI*2,d=s*0.12*r2(),x=s/2+Math.cos(a)*d,y=s/2+Math.sin(a)*d,rad=s*(0.2+r2()*0.14);
      const gr=g.createRadialGradient(x-rad*0.35,y-rad*0.35,rad*0.1,x,y,rad);
      gr.addColorStop(0,light);gr.addColorStop(0.55,base);gr.addColorStop(0.9,rim);gr.addColorStop(1,'rgba(0,0,0,0)');
      g.fillStyle=gr;g.beginPath();g.arc(x,y,rad,0,Math.PI*2);g.fill();
    }
  });
}
const GREENS=[['#2c4a26','#4f7a3a','#16260f'],['#355a2b','#5f8c43','#1a2c12'],['#28452e','#4b7350','#132416'],
              ['#3d5a2c','#6b8a45','#1e2c12'],['#2f4f33','#557e4f','#15261a'],['#43602f','#78955a','#22301a']];
GREENS.forEach((c,k)=>SPR.tree.push(canopy(900+k*7,c[0],c[1],c[2])));
SPR.treeLit=canopy(31,'#8a3a12','#ffb347','#3a1206');
SPR.treeBurnt=canopy(57,'#221d1a','#3a332e','#0d0b0a');
SPR.shadow=sprite(48,radial([[0,'rgba(0,0,0,.55)'],[0.6,'rgba(0,0,0,.25)'],[1,'rgba(0,0,0,0)']]));

// ── procedural vegetation (inside the simulation area only) ────────────────
const TREES=[];
(function(){
  const r2=rng(4242);
  const dens=(x,y)=>0.55+0.3*Math.sin(x/37+1.3)*Math.cos(y/53-0.4)+0.25*Math.sin((x+y)/23);   // clustered, not rows
  for(let i=0;i<NC;i++){
    if(NF[i]) continue;
    const [cx,cy]=cellXY(i);
    let k=Math.round(clamp(dens(cx,cy),0.05,1.1)*9);
    if(P.slopePct) k=Math.round(k*clamp(1-P.slopePct[i]/120,0.35,1));   // sparser on steep ground
    for(let j=0;j<k;j++){
      TREES.push({x:cx+(r2()-0.5)*CELL*0.95,y:cy+(r2()-0.5)*CELL*0.95,r:1.6+r2()*r2()*4.2,v:(r2()*SPR.tree.length)|0,c:i});
    }
  }
  TREES.sort((a,b)=>b.y-a.y);                 // north first -> painter's order for tilted views
})();

// ── particles (pooled) ─────────────────────────────────────────────────────
function makePool(n){const a=[];for(let i=0;i<n;i++)a.push({on:false});return a;}
const POOL={flame:makePool(CAP.flame),smoke:makePool(CAP.smoke),ember:makePool(CAP.ember),ash:makePool(CAP.ash),streak:makePool(CAP.streak)};
const LIVE={flame:0,smoke:0,ember:0,ash:0,streak:0};
function spawn(kind,o){const p=POOL[kind];for(let i=0;i<p.length;i++){if(!p[i].on){Object.assign(p[i],o);p[i].on=true;p[i].age=0;LIVE[kind]++;return p[i];}}return null;}
function clearParticles(){for(const k in POOL){POOL[k].forEach(p=>p.on=false);LIVE[k]=0;}}
const ACC={flame:new Float32Array(NC),smoke:new Float32Array(NC),ember:new Float32Array(NC),ash:new Float32Array(NC)};

// ── map + projection ───────────────────────────────────────────────────────
let map=null, proj=null, data=null, focusLine=null, cellFeat=new Array(NC), cellCache=new Int8Array(NC).fill(-1);
let H=null, PXM=1, TILTF=0.2, ZOOM=12, show={grid:true,smoke:true,veg:true,wind:true}, budget=1.0;
const cvs=document.getElementById('fx'), ctx=cvs.getContext('2d');
const DPR0=Math.min(window.devicePixelRatio||1,1.5); let RS=1, DPR=DPR0, CW=0, CH=0;   // RS: adaptive render scale
function resize(){const w=document.getElementById('wrap');CW=w.clientWidth;CH=w.clientHeight;DPR=DPR0*RS;cvs.width=Math.round(CW*DPR);cvs.height=Math.round(CH*DPR);cvs.style.width=CW+'px';cvs.style.height=CH+'px';}
new ResizeObserver(resize).observe(document.getElementById('wrap')); resize();

// Ground-plane homography (local metres -> container pixels) from 4 projected
// corners. Exact for a planar map under any zoom, rotation or tilt.
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
  const tilt=(map.getTilt&&map.getTilt())||0; TILTF=0.18+0.82*Math.sin(tilt*Math.PI/180);
  ZOOM=map.getZoom()||12; return true;
}

// ── data layer: CA grid cells ──────────────────────────────────────────────
const CELL_STYLE=[
  {f:'#000000',fo:0,   s:'#ffffff',so:0.16},   // unburned
  {f:'#ff8a1e',fo:0.20,s:'#ffb347',so:0.55},   // burning
  {f:'#5a2a14',fo:0.50,s:'#7a3a1a',so:0.35},   // recently burned
  {f:'#1a1512',fo:0.66,s:'#2a2420',so:0.30},   // charred
  {f:'#3f5878',fo:0.30,s:'#5b7aa3',so:0.30},   // non-fuel
  {f:'#ffd166',fo:0.18,s:'#ffd166',so:0.95},   // ignition (before run)
];
function styleFn(feat){const s=feat.getProperty('s')|0, st=CELL_STYLE[s];
  const gridOff=!show.grid&&(s===0);
  return {fillColor:st.f,fillOpacity:st.fo,strokeColor:st.s,strokeOpacity:gridOff?0:st.so,strokeWeight:s===5?2:0.6,clickable:false,zIndex:s};}
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
function syncCells(t){                         // restyle only cells whose state changed
  let boundary=false;
  for(let i=0;i<NC;i++){const s=cellState(i,t);
    if(s!==cellCache[i]){cellCache[i]=s;cellFeat[i].setProperty('s',s);}
    if(s>=1&&s<=3){const r=(i/N)|0,c=i%N;if(r===0||c===0||r===N-1||c===N-1)boundary=true;}}
  if(focusLine) focusLine.setOptions({strokeColor:boundary?'#ff4d3a':'#ffd166'});
  document.getElementById('hwarn').style.display=boundary?'block':'none';
}
function hotspotLayer(){
  const obs=P.hotspotKind==='observed';
  document.getElementById('lhot').textContent=obs?'Observed hotspot (NASA FIRMS VIIRS)':'Demo hotspot (synthetic, not observed)';
  (P.hotspots||[]).forEach(h=>{
    new google.maps.Circle({map,center:{lat:h.lat,lng:h.lon},radius:187.5,strokeColor:obs?'#4dd0e1':'#b39ddb',strokeOpacity:0.95,strokeWeight:2,
      fillColor:obs?'#4dd0e1':'#b39ddb',fillOpacity:0.08,clickable:false,zIndex:15});
  });
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
function frontCentre(){let sx=0,sy=0,k=0;for(let i=0;i<NC;i++){const s=cellState(i,simT);if(s===1||s===5||(!P.hasRun&&IGN[i]===0)){const [x,y]=cellXY(i);sx+=x;sy+=y;k++;}}
  return k?ll(sx/k,sy/k):{lat:LAT0,lng:LON0};}
const SIMZOOM=17;
function camSim(ms){flyTo({lat:LAT0,lng:LON0,zoom:SIMZOOM,tilt:0,heading:0},ms||1200);}
function note(msg){const n=document.getElementById('note');n.textContent=msg;n.style.display='block';clearTimeout(note.t);note.t=setTimeout(()=>n.style.display='none',4000);}
function tiltTo(t){map.setTilt(t);setTimeout(()=>{if((map.getTilt()||0)<1&&t>0)note('3D tilt is not available for this imagery here; the view stays top-down.');},700);}

// ── UI ─────────────────────────────────────────────────────────────────────
const fmtT=t=>'T+'+(t*(P.stepMin||0)).toFixed(0)+' min';
const tl=document.getElementById('tl'), bPlay=document.getElementById('bPlay');
tl.max=String(LAST); tl.disabled=!P.hasRun; bPlay.disabled=!P.hasRun;
function setPlaying(v){playing=v&&P.hasRun;bPlay.textContent=playing?'Pause':(simT>0&&simT<LAST?'Resume':'Run');bPlay.classList.toggle('on',playing);}
bPlay.onclick=()=>{if(simT>=LAST){simT=0;clearParticles();}setPlaying(!playing);};
document.getElementById('bReset').onclick=()=>{simT=0;simDone=false;clearParticles();ACC.flame.fill(0);ACC.smoke.fill(0);setPlaying(false);tl.value='0';updateHud(true);if(data)syncCells(0);};
document.querySelectorAll('.spd').forEach(b=>b.onclick=()=>{speed=+b.dataset.s;document.querySelectorAll('.spd').forEach(x=>x.classList.toggle('on',x===b));});
document.querySelectorAll('.tog').forEach(b=>b.onclick=()=>{const k=b.dataset.t;show[k]=!show[k];b.classList.toggle('on',show[k]);if(k==='grid'&&data)data.setStyle(styleFn);});
tl.oninput=()=>{simT=+tl.value;setPlaying(false);updateHud(true);if(data)syncCells(simT);};
document.getElementById('cOver').onclick=()=>flyTo({lat:LAT0,lng:LON0,zoom:13,tilt:0,heading:0},1400);
document.getElementById('cSim').onclick=()=>camSim();
document.getElementById('cFront').onclick=()=>{const c=frontCentre();flyTo({lat:c.lat,lng:c.lng,zoom:18},1200);};
document.getElementById('cClose').onclick=()=>{const c=frontCentre();flyTo({lat:c.lat,lng:c.lng,zoom:18.6,tilt:60},1500);setTimeout(()=>{if((map.getTilt()||0)<1)note('3D tilt is not available for this imagery here; the view stays top-down.');},1700);};
document.getElementById('cReset').onclick=()=>camSim(900);
document.getElementById('cTilt').onclick=()=>tiltTo((map.getTilt()||0)>5?0:55);
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
  const m=P.metrics[k];
  const set=(id,v)=>document.getElementById(id).textContent=v;
  if(!m){set('hburn','0');set('hburned','0');set('harea','0 ha');set('hper','0 m');set('hdist','0 m');set('hros','0 m/min');set('hint','-');return;}
  set('hburn',m.burning);set('hburned',m.burned);set('harea',m.burned_ha.toFixed(2)+' ha');
  set('hper',Math.round(m.perimeter_m)+' m');set('hdist',Math.round(m.front_distance_m)+' m');
  set('hros',m.ros_m_per_min.toFixed(1)+' m/min');set('hint',m.intensity_class);
}
document.getElementById('hname').textContent=F.name;
document.getElementById('hsim').textContent=(P.hasRun?'Cellular automata, ':'Ignition shown; press Run Simulation. ')+F.size_m+' m × '+F.size_m+' m ('+(F.size_m*F.size_m/1e6).toFixed(2)+' km²), '+CELL+' m cells. Flames, smoke, embers, ash and vegetation are visual representations of the simulated state.';

// ── per-frame simulation of the visual effects ─────────────────────────────
function lod(){            // 0 regional, 1 forest, 2 simulation, 3 close
  return ZOOM<14?0:(ZOOM<16?1:(ZOOM<17.5?2:3));
}
function emitters(t){      // burning / smouldering cells, strongest first, capped
  const list=[];
  for(let i=0;i<NC;i++){const fs=fireStrength(i,t),ss=smokeStrength(i,t);if(fs>0||ss>0)list.push([i,fs,ss]);}
  if(list.length>CAP.emitters){list.sort((a,b)=>(b[1]*INT[b[0]]+b[2]*.3)-(a[1]*INT[a[0]]+a[2]*.3));list.length=CAP.emitters;}
  return list;
}
function stepFx(dt,t,now){
  const L=lod(), wv=windVec(t), wx=wv[0], wy=wv[1], spd=wv[2];
  const em=emitters(t), q=budget;
  for(const [i,fs,ss] of em){
    const I=Math.max(INT[i],0.15), [cx,cy]=cellXY(i);
    if(L>=2&&fs>0){ACC.flame[i]+=dt*fs*(14+34*I)*q*(L===3?1:0.6);
      while(ACC.flame[i]>=1){ACC.flame[i]-=1;if(LIVE.flame>=CAP.flame*q)continue;const life=0.45+R()*0.55,h=(2.5+9*I)*(0.6+0.5*fs);
        spawn('flame',{x:cx+(R()-.5)*CELL*.9,y:cy+(R()-.5)*CELL*.9,z:0,vx:wx*0.35+(R()-.5)*1.5,vy:wy*0.35+(R()-.5)*1.5,vz:h/life,
          life,size:(3+6.5*I)*(0.7+0.6*R()),seed:R()*6.28,str:fs});}}
    if(show.smoke&&L>=1&&ss>0){ACC.smoke[i]+=dt*ss*(0.6+1.6*I)*q*(L===1?0.5:(L===3?0.75:1));
      while(ACC.smoke[i]>=1){ACC.smoke[i]-=1;if(LIVE.smoke>=CAP.smoke*q)continue;
        spawn('smoke',{x:cx+(R()-.5)*CELL*.8,y:cy+(R()-.5)*CELL*.8,z:2+R()*3,vz:2.4+R()*2.2*I,life:7+R()*6,
          s0:8+12*I,s1:45+70*I+R()*30,rot:R()*6.28,vr:(R()-.5)*0.25,v:(R()*SPR.smoke.length)|0,den:(0.35+0.65*ss)*(0.5+0.5*I),seed:R()*100});}}
    if(L>=2&&fs>0.3&&I>0.3){ACC.ember[i]+=dt*fs*I*(2+spd*0.5)*q*(L===3?1:0.5);
      while(ACC.ember[i]>=1){ACC.ember[i]-=1;
        spawn('ember',{x:cx+(R()-.5)*CELL*.6,y:cy+(R()-.5)*CELL*.6,z:3+R()*6*I,vx:(R()-.5)*3,vy:(R()-.5)*3,vz:6+R()*9*I,life:1.4+R()*1.8,sz:1+R()*1.8,seed:R()*6.28});}}
    if(L===3&&(ss>0)){ACC.ash[i]+=dt*(0.35+0.6*ss)*q;
      while(ACC.ash[i]>=1){ACC.ash[i]-=1;
        spawn('ash',{x:cx+(R()-.5)*CELL,y:cy+(R()-.5)*CELL,z:8+R()*22,vz:-(0.6+R()*0.8),life:4+R()*4,seed:R()*6.28});}}
  }
  // wind streaks around the simulation area (direction + relative speed only)
  if(show.wind&&L>=1&&spd>0.3&&LIVE.streak<CAP.streak*q){
    const E=HALF*1.6; spawn('streak',{x:(R()-.5)*2*E,y:(R()-.5)*2*E,life:1.6+R()*1.6});
  }
  const shear=z=>0.5+0.5*Math.min(1,z/60), tk=playing?speed:1;
  for(const p of POOL.flame) if(p.on){p.age+=dt;if(p.age>=p.life){p.on=false;LIVE.flame--;continue;}
    p.z+=p.vz*dt;p.vz*=0.985;p.x+=(p.vx+Math.sin(now*0.006+p.seed)*1.2)*dt;p.y+=(p.vy+Math.cos(now*0.005+p.seed)*1.2)*dt;}
  for(const p of POOL.smoke) if(p.on){p.age+=dt;if(p.age>=p.life){p.on=false;LIVE.smoke--;continue;}
    const k=shear(p.z),tu=Math.sin(now*0.0007+p.seed+p.y*0.01)*1.6,tv=Math.cos(now*0.0006+p.seed*1.3+p.x*0.01)*1.6;
    p.x+=(wx*DRIFT_K*tk*k+tu)*dt;p.y+=(wy*DRIFT_K*tk*k+tv)*dt;p.z+=p.vz*dt;p.vz=Math.max(0.5,p.vz*0.995);p.rot+=p.vr*dt;}
  for(const p of POOL.ember) if(p.on){p.age+=dt;if(p.age>=p.life||p.z<0){p.on=false;LIVE.ember--;continue;}
    p.vz-=4.5*dt;p.x+=(p.vx+wx*DRIFT_K*tk*0.9)*dt;p.y+=(p.vy+wy*DRIFT_K*tk*0.9)*dt;p.z+=p.vz*dt;}
  for(const p of POOL.ash) if(p.on){p.age+=dt;if(p.age>=p.life||p.z<0){p.on=false;LIVE.ash--;continue;}
    p.x+=(wx*DRIFT_K*tk*0.35+Math.sin(now*0.003+p.seed)*0.8)*dt;p.y+=(wy*DRIFT_K*tk*0.35+Math.cos(now*0.0025+p.seed)*0.8)*dt;p.z+=p.vz*dt;}
  for(const p of POOL.streak) if(p.on){p.age+=dt;if(p.age>=p.life){p.on=false;LIVE.streak--;continue;}
    p.x+=wx*DRIFT_K*0.9*dt;p.y+=wy*DRIFT_K*0.9*dt;}
}

// ── per-frame drawing ──────────────────────────────────────────────────────
function onScreen(s,m){return s[0]>-m&&s[1]>-m&&s[0]<CW+m&&s[1]<CH+m;}
function draw(t,now){
  ctx.setTransform(DPR,0,0,DPR,0,0); ctx.clearRect(0,0,CW,CH);
  if(!H) return;
  const L=lod(), ex=clamp(1.15/PXM,1,2.6), up=PXM*TILTF;   // ex: small-scale exaggeration so effects stay visible when zoomed out
  // vegetation representation
  if(show.veg&&L>=2){
    ctx.globalCompositeOperation='source-over';
    const va=L===3?0.62:0.45;
    for(const tr of TREES){const s=W2S(tr.x,tr.y);if(!onScreen(s,20))continue;
      const st=cellState(tr.c,t); let sp=SPR.tree[tr.v], r=tr.r;
      if(st===1){sp=SPR.treeLit;} else if(st===2||st===3){sp=SPR.treeBurnt;r*=0.72;}
      const d=Math.max(2.5,r*2*PXM);
      ctx.globalAlpha=va*0.6; ctx.drawImage(SPR.shadow,s[0]-d/2+d*0.18,s[1]-d/2+d*0.14,d,d);   // sun from the north-west
      ctx.globalAlpha=va; ctx.drawImage(sp,s[0]-d/2,s[1]-d/2-r*up*0.6,d,d);}
  }
  ctx.globalAlpha=1;
  // ground glow under burning and smouldering cells
  ctx.globalCompositeOperation='lighter';
  for(let i=0;i<NC;i++){const fs=fireStrength(i,t);if(fs<=0)continue;
    const [x,y]=cellXY(i),s=W2S(x,y);if(!onScreen(s,120))continue;
    const fl=0.85+0.15*Math.sin(now*0.013+i*1.7)+0.08*Math.sin(now*0.031+i);
    const r=Math.max(12,CELL*PXM*(1.1+1.1*INT[i])*fl*(L===0?1.6:1));
    ctx.globalAlpha=clamp((0.35+0.55*INT[i])*fs,0,0.9); ctx.drawImage(SPR.glow,s[0]-r,s[1]-r,2*r,2*r);
    const r2=r*0.45; ctx.globalAlpha=clamp(0.5*fs,0,0.7); ctx.drawImage(SPR.glow,s[0]-r2,s[1]-r2,2*r2,2*r2);
    // burning fuel bed: a few flickering flame clusters spread over the cell, so adjacent
    // burning cells merge into one continuous front instead of separate icons
    if(L>=1){const nb=L>=2?4:2;
      for(let k=0;k<nb;k++){const h1=Math.sin(i*12.9898+k*78.233)*43758.5453, h2=Math.sin(i*39.346+k*11.135)*24634.6345;
        const ox=(h1-Math.floor(h1)-0.5)*CELL*0.85, oy=(h2-Math.floor(h2)-0.5)*CELL*0.85, b=W2S(x+ox,y+oy);
        const fk=0.7+0.3*Math.sin(now*0.017+k*2.1+i)+0.15*Math.sin(now*0.043+k+i*0.7);
        const w=Math.max(4,CELL*PXM*(0.32+0.38*INT[i])*fs*fk*ex*0.8), hh=w*(1.35+0.35*Math.sin(now*0.011+k+i));
        ctx.globalAlpha=clamp(0.75*fs,0,0.9); ctx.drawImage(SPR.flame[k%2],b[0]-w/2,b[1]-hh*0.8,w,hh);}}
  }
  // flames: stretched, swaying additive tongues; colour shifts white -> orange -> red with age
  for(const p of POOL.flame) if(p.on){const s=W2S(p.x,p.y);if(!onScreen(s,40))continue;
    const a=p.age/p.life, sp=SPR.flame[a<0.3?0:(a<0.65?1:2)];
    const w=Math.max(2,p.size*PXM*ex*(1-a*0.55)), h=w*(1.5+0.8*Math.sin(p.seed+now*0.01));
    ctx.globalAlpha=clamp(Math.pow(1-a,0.7)*(0.55+0.45*p.str),0,1);
    ctx.drawImage(sp,s[0]-w/2,s[1]-p.z*up*ex-h*0.75,w,h);}
  // embers
  for(const p of POOL.ember) if(p.on){const s=W2S(p.x,p.y);if(!onScreen(s,20))continue;
    const a=p.age/p.life, fl=0.55+0.45*Math.sin(now*0.025+p.seed*9);
    ctx.globalAlpha=clamp((1-a)*fl,0,1); const d=p.sz*2.4*Math.max(1,Math.min(2,PXM));
    ctx.drawImage(SPR.ember,s[0]-d/2,s[1]-p.z*up*ex-d/2,d,d);}
  // smoke: soft procedural puffs, darker and denser near the fire, lighter and diffuse downwind
  ctx.globalCompositeOperation='source-over';
  if(show.smoke) for(const p of POOL.smoke) if(p.on){const s=W2S(p.x,p.y);
    const a=p.age/p.life, size=Math.min(280,(p.s0+(p.s1-p.s0)*Math.sqrt(a))*PXM*Math.max(1,ex*0.8));   // fill-rate cap
    const sy=s[1]-p.z*up; if(!onScreen([s[0],sy],size))continue;
    const fade=a<0.12?a/0.12:Math.pow(1-(a-0.12)/0.88,1.4);
    ctx.globalAlpha=clamp(fade*p.den*0.9,0,0.92);
    const cr=Math.cos(p.rot)*DPR, sr=Math.sin(p.rot)*DPR;
    ctx.setTransform(cr,sr,-sr,cr,s[0]*DPR,sy*DPR);
    ctx.drawImage((a<0.22?SPR.smokeDark:SPR.smoke)[p.v],-size/2,-size/2,size,size);}   // dark, fire-lit near the source
  ctx.setTransform(DPR,0,0,DPR,0,0);
  // ash
  if(L===3) for(const p of POOL.ash) if(p.on){const s=W2S(p.x,p.y);if(!onScreen(s,10))continue;
    const a=p.age/p.life; ctx.globalAlpha=0.35*(1-a); const d=2.2;
    ctx.drawImage(SPR.ash,s[0]-d/2,s[1]-p.z*up-d/2,d,d);}
  // wind streaks + predicted spread direction
  if(show.wind){
    const wv=windVec(t), n=Math.hypot(wv[0],wv[1])||1, ux=wv[0]/n, uy=wv[1]/n;
    ctx.strokeStyle='rgba(232,237,243,0.5)';ctx.lineWidth=1;
    for(const p of POOL.streak) if(p.on){const a=p.age/p.life;ctx.globalAlpha=Math.sin(a*Math.PI)*0.45;
      const s0=W2S(p.x,p.y),s1=W2S(p.x-ux*18,p.y-uy*18);ctx.beginPath();ctx.moveTo(s0[0],s0[1]);ctx.lineTo(s1[0],s1[1]);ctx.stroke();}
    if(wv[2]>0.3){const c=frontCentreXY(t);if(c){
      const L2=HALF*0.55,a0=W2S(c[0],c[1]),a1=W2S(c[0]+ux*L2,c[1]+uy*L2),hx=a1[0]-a0[0],hy=a1[1]-a0[1],hl=Math.hypot(hx,hy)||1;
      ctx.globalAlpha=0.8;ctx.strokeStyle='#e8edf3';ctx.lineWidth=2;ctx.setLineDash([6,5]);
      ctx.beginPath();ctx.moveTo(a0[0],a0[1]);ctx.lineTo(a1[0],a1[1]);ctx.stroke();ctx.setLineDash([]);
      const bx=hx/hl,by=hy/hl;ctx.fillStyle='#e8edf3';ctx.beginPath();ctx.moveTo(a1[0]+bx*9,a1[1]+by*9);
      ctx.lineTo(a1[0]-by*6,a1[1]+bx*6);ctx.lineTo(a1[0]+by*6,a1[1]-bx*6);ctx.closePath();ctx.fill();}}
  }
  ctx.globalAlpha=1;
}
function frontCentreXY(t){let sx=0,sy=0,k=0;for(let i=0;i<NC;i++){const s=cellState(i,t);if(s===1||s===5){const [x,y]=cellXY(i);sx+=x;sy+=y;k++;}}return k?[sx/k,sy/k]:null;}

// ── main loop: one requestAnimationFrame for everything ────────────────────
let last=null, raf=null, frameMs=16, lastStep=-1;
function loop(now){
  raf=requestAnimationFrame(loop);
  if(last===null) last=now; const dt=Math.min((now-last)/1000,0.05); last=now;
  frameMs=frameMs*0.95+dt*1000*0.05;
  if(frameMs>30) budget=Math.max(0.35,budget-0.01); else if(frameMs<18) budget=Math.min(1,budget+0.005);
  // a CPU-only laptop is fill-rate bound: drop the canvas resolution before dropping effects further
  if(budget<=0.36&&frameMs>32&&RS>0.6&&now-(loop.rsT||0)>1500){RS=Math.round((RS-0.1)*10)/10;resize();loop.rsT=now;}
  else if(frameMs<15&&RS<1&&now-(loop.rsT||0)>3000){RS=Math.round((RS+0.1)*10)/10;resize();loop.rsT=now;}
  if(playing){simT=Math.min(LAST,simT+dt*speed/SEC_PER_STEP);tl.value=String(simT);if(simT>=LAST){setPlaying(false);simDone=true;}}
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
