"""Public static site, rebuilt on a schedule: an overview, one page per area, and a methodology page.

Inputs are SQLite databases: the live AIS recording, plus any databases that hold processed passes
(`passes` and `detections` tables). Each pass is assigned to an area by where its detections are.
"""
from __future__ import annotations

import html
import json
import os
import sqlite3
import time
from datetime import datetime, timezone

import pandas as pd

from . import config as C
from .insight import CROWDED, priors
from .live import ais_status, recording_box

TITLES = {"mumbai": "Mumbai", "denmark": "Øresund"}
SUBTITLES = {"mumbai": "Coast and harbour, India", "denmark": "Copenhagen–Malmö strait, Denmark and Sweden"}
REASONS = {"beyond_ground_receiver_range": "Beyond coastal AIS range", "ais_feed_gap": "AIS feed was down",
           "no_ais_in_window": "No AIS available for this pass", "same_spot_in_another_pass": "Probably a fixed object"}
LIVE_WITHIN_MIN = 30      # a message this recent means the feed counts as live when the site is built


def utc(t, fmt="%d %b %Y, %H:%M UTC"):
    return datetime.fromtimestamp(t, timezone.utc).strftime(fmt) if t else "never"


def region_of(lon, lat) -> str | None:
    for name, (w, s, e, n) in C.REGIONS.items():
        if w <= lon <= e and s <= lat <= n:
            return name
    return None


def collect_passes(result_dbs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    passes, dets = [], []
    for db in result_dbs:
        if not os.path.exists(db):
            continue
        con = sqlite3.connect(db)
        try:
            p = pd.read_sql("SELECT * FROM passes", con)
            d = pd.read_sql("SELECT * FROM detections", con)
        except Exception:
            continue
        finally:
            con.close()
        passes.append(p); dets.append(d)
    if not passes:
        return pd.DataFrame(), pd.DataFrame()
    p, d = pd.concat(passes, ignore_index=True), pd.concat(dets, ignore_index=True)
    centre = d.groupby("pass_id")[["lon", "lat"]].median()
    p["region"] = [region_of(*centre.loc[i]) if i in centre.index else None for i in p.pass_id]
    p = p.dropna(subset=["region"]).drop_duplicates("pass_id", keep="last").sort_values("time")
    return p, d[d.pass_id.isin(p.pass_id)]


def live_positions(live_db: str, aoi, minutes: int = 30) -> list[dict]:
    if not os.path.exists(live_db):
        return []
    w, s, e, n = recording_box(aoi)
    con = sqlite3.connect(live_db)
    try:
        df = pd.read_sql("SELECT mmsi, time, lon, lat, sog, cog FROM ais WHERE time > ? AND lon BETWEEN ? AND ? AND lat BETWEEN ? AND ?",
                         con, params=(time.time() - minutes * 60, w, e, s, n))
    finally:
        con.close()
    last = df.sort_values("time").groupby("mmsi").tail(1)
    return [dict(lon=r.lon, lat=r.lat, sog=round(r.sog, 1), age=int((time.time() - r.time) / 60)) for r in last.itertuples()]


CSS = """
:root{--bg:#070b0f;--panel:#0d141a;--line:#1b2a33;--ink:#e3eeee;--muted:#7f9498;
--green:#2ef2b0;--red:#ff5c6c;--amber:#ffb547;--cyan:#4fc3ff;
--mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Consolas,monospace;--sans:'Inter',system-ui,-apple-system,Segoe UI,sans-serif}
*{box-sizing:border-box}html{background:var(--bg)}
body{margin:0;color:var(--ink);font:15px/1.55 var(--sans);background:var(--bg);min-height:100vh;display:flex;flex-direction:column}
a{color:var(--green);text-decoration:none}a:hover{text-decoration:underline}
.wrap{width:100%;max-width:1120px;margin:0 auto;padding:0 16px}
header.top{border-bottom:1px solid var(--line);background:var(--panel)}
.bar{display:flex;align-items:center;gap:24px;min-height:60px;flex-wrap:wrap}
.brand{display:flex;align-items:center;gap:10px;color:var(--ink);font-weight:700;font-size:16px}.brand:hover{text-decoration:none}
.scope{width:26px;height:26px;border-radius:50%;flex:none;border:1px solid var(--green);
 background:conic-gradient(from 0deg,rgba(46,242,176,.75),rgba(46,242,176,0) 90deg,transparent)}
nav.tabs{display:flex;gap:4px;flex:1}
nav.tabs a{color:var(--muted);padding:6px 10px;border-radius:6px;font-weight:600;font-size:14px}
nav.tabs a:hover{color:var(--ink);text-decoration:none}nav.tabs a.on{color:var(--ink);background:#16222a}
.pill{font-size:13px;color:var(--muted);white-space:nowrap}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:7px;background:var(--muted)}
.dot.live{background:var(--green);animation:pulse 1.8s ease-in-out infinite}
@keyframes pulse{50%{opacity:.4}}@media (prefers-reduced-motion:reduce){.dot.live{animation:none}}
main{padding:28px 0 40px;flex:1}
h1{font:700 30px/1.2 var(--sans);margin:0 0 4px;letter-spacing:-.01em}
h2{font:600 17px/1.3 var(--sans);margin:34px 0 12px}
.sub{color:var(--muted);margin:0 0 20px}
.titlerow{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.tag{font-size:12px;font-weight:600;border:1px solid var(--line);border-radius:999px;padding:3px 10px;color:var(--muted)}
.tag.amber{color:var(--amber);border-color:rgba(255,181,71,.45)}.tag.green{color:var(--green);border-color:rgba(46,242,176,.45)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px 16px;min-width:0}
.label{font-size:13px;color:var(--muted);margin-bottom:6px}
.value{font:600 30px/1.1 var(--mono);color:var(--ink)}
.value.green{color:var(--green)}.value.red{color:var(--red)}.value.cyan{color:var(--cyan)}.value.amber{color:var(--amber)}
.area{display:block;color:inherit}.area:hover{border-color:rgba(46,242,176,.5);text-decoration:none}
.area h3{font:600 18px/1.3 var(--sans);margin:0}.area .sub{margin:2px 0 12px;font-size:14px}
.kv{display:flex;justify-content:space-between;gap:12px;font-size:14px;padding:5px 0;border-top:1px solid var(--line)}
.kv span:first-child{color:var(--muted)}.kv span:last-child{font-family:var(--mono)}
.go{color:var(--green);font-weight:600;font-size:14px;margin-top:10px}
.inspect{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:12px}
.map{height:620px;border:1px solid var(--line);border-radius:8px;background:#0a0f13;min-width:0}
.side{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px 16px;height:620px;overflow-y:auto;min-width:0}
.side .empty{color:var(--muted);height:100%;display:flex;align-items:center;justify-content:center;text-align:center;padding:0 24px}
.side h3{font:600 18px/1.3 var(--sans);margin:6px 0 10px;overflow-wrap:anywhere}
.side h4{font:600 13px/1.3 var(--sans);color:var(--muted);margin:16px 0 6px}
.side img{width:100%;display:block;border:1px solid var(--line);border-radius:6px;image-rendering:pixelated;background:#000}
.side .kv span:last-child{text-align:right;overflow-wrap:anywhere}
.side .note{color:var(--amber);font-size:13px;margin:10px 0 0}.side .hint{color:var(--muted);font-size:12.5px;margin:6px 0 0}
.bar2{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:2px 10px;font-size:14px;margin:6px 0}
.bar2 i{grid-column:1/3;height:5px;border-radius:3px;background:var(--line);overflow:hidden;display:block}
.bar2 i b{display:block;height:100%;background:var(--cyan)}
tr.pick{cursor:pointer}tr.pick:hover td,tr.pick:focus td{background:#16222a}tr.pick:focus{outline:none}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:14px;background:var(--panel)}
th{font-weight:600;font-size:13px;color:var(--muted);text-align:right}
td,th{padding:9px 12px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}td:first-child,th:first-child{text-align:left}
td{font-variant-numeric:tabular-nums}tr:last-child td{border-bottom:0}
.more{color:var(--muted);font-size:13px;margin:8px 0 0}
.chips{list-style:none;padding:0;margin:0;display:flex;flex-wrap:wrap;gap:8px}
.chips li{font-size:13px;border:1px solid var(--line);background:var(--panel);padding:6px 10px;border-radius:6px;color:var(--muted)}
.chart{border:1px solid var(--line);border-radius:8px;display:block;width:100%}
footer{border-top:1px solid var(--line);padding:16px 0 24px;font-size:13px;color:var(--muted)}
footer .wrap{display:flex;justify-content:space-between;gap:8px 24px;flex-wrap:wrap}
.leaflet-container{background:#0a0f13;font:13px var(--sans)}
.radar-tiles{filter:invert(100%) hue-rotate(180deg) grayscale(70%) brightness(78%) contrast(118%)}
.leaflet-popup-content-wrapper,.leaflet-popup-tip{background:var(--panel);color:var(--ink);border:1px solid var(--line);box-shadow:0 6px 24px rgba(0,0,0,.6)}
.leaflet-popup-content{line-height:1.5}.leaflet-popup-content img{border:1px solid var(--line);margin-top:6px;image-rendering:pixelated}
.leaflet-control-layers,.leaflet-bar a{background:var(--panel)!important;color:var(--ink)!important;border-color:var(--line)!important}
.leaflet-control-layers i{display:inline-block;width:10px;height:10px;border-radius:50%;margin:0 6px 0 2px}
.leaflet-control-attribution{background:rgba(7,11,15,.75)!important;color:var(--muted)!important}.leaflet-control-attribution a{color:var(--muted)}
@media (max-width:900px){.inspect{grid-template-columns:minmax(0,1fr)}.map{height:420px}.side{height:auto;max-height:none}.side .empty{padding:24px}}
@media (max-width:760px){h1{font-size:25px}.value{font-size:25px}nav.tabs{flex-basis:100%;order:3}.bar{padding:10px 0;gap:8px 16px}}
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">')

# "Updated 5 min ago" is worked out in the viewer's browser, so the page stays truthful however long ago it was built:
# if the system stops, the page quietly shows its age rather than claiming to be live.
STATUS_JS = """<script>(function(){const s=document.getElementById('status'),b=+s.dataset.built,live=s.dataset.live==='1';
const f=()=>{const m=Math.round((Date.now()/1000-b)/60),a=m<1?'just now':m<60?m+' min ago':m<1440?Math.round(m/60)+' h ago':Math.round(m/1440)+' d ago',ok=live&&m<180;
s.innerHTML='<span class="dot'+(ok?' live':'')+'"></span>'+(ok?'Live &middot; updated ':'Updated ')+a;};f();setInterval(f,60000);})();</script>"""


def page(title: str, body: str, depth: int = 0, leaflet: bool = False, active: str = "overview", feed: dict | None = None) -> str:
    up = "../" * depth
    tabs = [("overview", f"{up}index.html", "Overview")] + [(r, f"{up}{r}/index.html", TITLES[r]) for r in C.SITE_REGIONS]
    nav = "".join(f'<a class="{"on" if k == active else ""}" href="{href}">{label}</a>' for k, href, label in tabs)
    lf = ('<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">'
          '<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>') if leaflet else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark"><title>{html.escape(title)}</title>{FONTS}<style>{CSS}</style>{lf}</head><body>
<header class="top"><div class="wrap bar">
 <a class="brand" href="{up}index.html"><span class="scope"></span>Dark Vessel Detection</a>
 <nav class="tabs">{nav}</nav>
 <span class="pill" id="status" data-built="{int(time.time())}" data-live="{1 if (feed or {}).get('live') else 0}"></span>
</div></header>
<main class="wrap">{body}</main>
<footer><div class="wrap"><span>Flags are for human review, not accusations.</span>
<span>Contains modified Copernicus Sentinel data 2026 &middot; AIS: aisstream.io, Danish Maritime Authority &middot;
&copy; OpenStreetMap contributors &middot; <a href="{up}nn/index.html">Methodology</a></span></div></footer>
{STATUS_JS}
</body></html>"""


MAP_JS = """<script>
const D=__DATA__,LIVE=__LIVE__,B=[[__S__,__W__],[__N__,__E__]],R=__REASONS__,SHOW_UNVERIFIED=__SHOW__,WHEN=__WHEN__,PRIOR_N=__PRIOR_N__;
const m=L.map('map',{zoomSnap:.25}).fitBounds(B);
// a map created while its container has no size (e.g. a page opened in a background tab) falls back to zoom 0, so
// re-fit whenever the container's size changes, until the visitor has moved the map themselves
let moved=false;m.on('dragstart',()=>moved=true);m.getContainer().addEventListener('wheel',()=>moved=true,{passive:true});
const refit=()=>{m.invalidateSize();if(!moved)m.fitBounds(B,{animate:false});};
new ResizeObserver(refit).observe(m.getContainer());window.addEventListener('load',refit);
// standard OSM tiles (no API key), turned into a dark basemap with a CSS filter (.radar-tiles)
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,className:'radar-tiles',
 attribution:'&copy; OpenStreetMap contributors'}).addTo(m);
const tracked=L.layerGroup().addTo(m),untracked=L.layerGroup().addTo(m),unverified=L.layerGroup(),live=L.layerGroup().addTo(m);
if(SHOW_UNVERIFIED)unverified.addTo(m);
const side=document.getElementById('side'),byId={};let ring=null,trackLine=null;
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const kv=(k,v)=>v===null||v===undefined||v===''?'':'<div class="kv"><span>'+k+'</span><span>'+esc(v)+'</span></div>';
function inspect(d){
 const t=d.status=='matched',low=d.confidence=='low',info=d.info||{},a=info.ais;
 const c=t?'#2ef2b0':(low?'#ffb547':'#ff5c6c'),label=t?'Tracked by AIS':(low?'Unverified':'Untracked');
 if(ring)ring.remove();if(trackLine){trackLine.remove();trackLine=null;}
 ring=L.circleMarker([d.lat,d.lon],{radius:13,color:'#fff',weight:2,fill:false,interactive:false}).addTo(m);
 if(a&&a.track&&a.track.length>1)trackLine=L.polyline(a.track,{color:'#2ef2b0',weight:2,opacity:.8,dashArray:'5 5',interactive:false}).addTo(m);
 let h='<span class="tag" style="color:'+c+';border-color:'+c+'">'+label+'</span><h3>'+esc((a&&a.name)||(t?'Tracked ship':(low?'Unverified target':'Untracked target')))+'</h3>';
 if(d.crop)h+='<img src="'+d.crop+'" alt="radar image of this target">';
 h+='<h4>Radar</h4>'+(info.est_length_m&&!(a&&a.length_m)?kv('Estimated length','about '+info.est_length_m+' m'):'')+kv('Measured on radar',d.length_m.toFixed(0)+' m')
  +kv('From shore',d.shore_km.toFixed(1)+' km')+kv('Position',d.lat.toFixed(4)+', '+d.lon.toFixed(4))
  +kv('Surroundings',info.setting)+(t||d.ship_prob===''||d.ship_prob===null?'':kv('Ship likelihood',(Number(d.ship_prob)*100).toFixed(0)+'%'))+kv('Imaged',WHEN);
 if(a){h+='<h4>Reported by the ship</h4>'+kv('Type',a.type)+kv('Length',a.length_m?a.length_m.toFixed(0)+' m':null)+kv('Speed',a.sog===null?null:a.sog+' kn')
  +kv('Course',a.cog===null?null:a.cog.toFixed(0)+'°')+kv('Destination',a.destination);
  if(a.track&&a.track.length>1)h+='<p class="hint">Its AIS track around the image time is shown on the map.</p>';}
 if(!(a&&a.type)&&info.types&&info.types.length){h+='<h4>Likely type</h4>'+info.types.map(([n,p])=>'<div class="bar2"><span>'+esc(n)+'</span><span>'+(p*100).toFixed(0)+'%</span><i><b style="width:'+(p*100).toFixed(0)+'%"></b></i></div>').join('')
  +'<p class="hint">Estimate from length, based on '+PRIOR_N.toLocaleString()+' ships in Danish AIS records.</p>';}
 if(info.ports&&info.ports.length){h+='<h4>Nearest ports</h4>'+info.ports.map(([n,k])=>kv(n,k+' km')).join('');
  if(!t)h+='<p class="hint">Distance only. Its real origin is unknown without AIS.</p>';}
 if(d.reason)h+='<p class="note">'+d.reason.split(',').map(x=>esc(R[x]||x)).join(' &middot; ')+'</p>';
 side.innerHTML=h;side.scrollTop=0;
 if(innerWidth<=900)side.scrollIntoView({behavior:'smooth',block:'start'});
}
D.forEach(d=>{const t=d.status=='matched',low=d.confidence=='low',c=t?'#2ef2b0':(low?'#ffb547':'#ff5c6c'),layer=t?tracked:(low?unverified:untracked);
byId[d.det_id]={d,layer};L.circleMarker([d.lat,d.lon],{radius:t?5:7,color:c,fillColor:c,fillOpacity:.85,weight:t?1:2}).on('click',()=>inspect(d)).addTo(layer);});
document.querySelectorAll('tr.pick').forEach(r=>{const go=()=>{const x=byId[r.dataset.id];if(!x)return;if(!m.hasLayer(x.layer))x.layer.addTo(m);inspect(x.d);m.panTo([x.d.lat,x.d.lon]);
 document.getElementById('map').scrollIntoView({behavior:'smooth',block:'center'});};r.addEventListener('click',go);r.addEventListener('keydown',e=>{if(e.key==='Enter')go();});});
LIVE.forEach(p=>L.circleMarker([p.lat,p.lon],{radius:3,color:'#4fc3ff',fillColor:'#4fc3ff',fillOpacity:.9,weight:1}).bindPopup('<b>Broadcasting now</b><br>'+p.sog+' kn &middot; '+p.age+' min ago').addTo(live));
const dot=c=>'<i style="background:'+c+'"></i>';
L.control.layers(null,{[dot('#2ef2b0')+'Tracked by AIS']:tracked,[dot('#ff5c6c')+'Untracked']:untracked,
 [dot('#ffb547')+'Unverified']:unverified,[dot('#4fc3ff')+'Broadcasting now']:live},{collapsed:false}).addTo(m);
</script>"""


def readout(label: str, value, cls: str = "") -> str:
    return f'<div class="panel"><div class="label">{label}</div><div class="value {cls}">{value}</div></div>'


def shown_reason(reason, info) -> str:
    """The reasons worth showing. Passes saved before the crowded-area rule may still carry the same-spot note, which
    means little where ships crowd together, so it is dropped here as well as no longer being added by the pipeline."""
    setting = (json.loads(info) if info else {}).get("setting")
    return ",".join(r for r in str(reason or "").split(",") if r and not (r == "same_spot_in_another_pass" and setting in CROWDED))


def plain_reason(reason) -> str:
    return " · ".join(REASONS.get(r, r) for r in str(reason or "").split(",") if r)


def review_table(d: pd.DataFrame, limit: int = 12) -> str:
    """Untracked and unverified targets of a pass: the ones a person would look at. Strongest first."""
    c = d[d.status == "candidate"].copy()
    if c.empty:
        return "<div class='panel muted'>Nothing to review on this pass.</div>"
    c["rank"] = (c.confidence == "low").astype(int)
    c = c.sort_values(["rank", "ship_prob"], ascending=[True, False])
    rows = "".join(
        f"<tr class='pick' tabindex='0' data-id='{int(r.det_id)}'><td>{'Unverified' if r.confidence == 'low' else 'Untracked'}</td><td>{r.lat:.3f}, {r.lon:.3f}</td><td>{r.length_m / C.RADAR_LENGTH_BIAS:.0f} m</td>"
        f"<td>{r.shore_km:.1f} km</td><td>{'&ndash;' if pd.isna(r.ship_prob) else f'{r.ship_prob:.0%}'}</td>"
        f"<td>{html.escape(plain_reason(r.reason))}</td></tr>" for r in c.head(limit).itertuples())
    more = f"<p class='more'>{len(c) - limit} more on the map.</p>" if len(c) > limit else ""
    return ("<div class='tablewrap'><table><thead><tr><th>status</th><th>position</th><th>est. length</th><th>from shore</th><th>ship likelihood</th>"
            f"<th>note</th></tr></thead><tbody>{rows}</tbody></table></div>{more}")


def region_page(region: str, passes: pd.DataFrame, dets: pd.DataFrame, live_db: str, out: str, scenes: list[dict], feed: dict) -> dict:
    aoi = C.REGIONS[region]
    os.makedirs(out, exist_ok=True)
    rp = passes[passes.region == region] if len(passes) else passes
    live = live_positions(live_db, aoi)

    if len(rp):
        latest = rp.iloc[-1]
        d = dets[(dets.pass_id == latest.pass_id) & (dets.status != "filtered")]
        d = d.assign(reason=[shown_reason(r, i) for r, i in zip(d.reason, d["info"])])
        data = d.reindex(columns=["det_id", "lon", "lat", "length_m", "shore_km", "status", "reason", "confidence", "crop",
                                  "ship_prob", "info"]).astype(object).fillna("").to_dict("records")
        for x in data:
            x["info"] = json.loads(x["info"]) if x["info"] else {}
        shown, seen, n_m = utc(latest.time), int(latest.n_det), int(latest.n_matched)
        n_low = int(latest.n_low_conf)
        n_high = int(latest.n_candidates) - n_low
        radar_only = int(latest.ais_msgs) == 0
        review = review_table(d)
    else:
        data, shown, seen, n_m, n_high, n_low, radar_only = [], "No pass yet", 0, 0, 0, 0, False
        review = "<div class='panel muted'>No pass yet.</div>"
    w, s, e, n = aoi
    js = (MAP_JS.replace("__DATA__", json.dumps(data)).replace("__LIVE__", json.dumps(live)).replace("__REASONS__", json.dumps(REASONS))
          .replace("__SHOW__", "true" if radar_only else "false").replace("__WHEN__", json.dumps(shown))
          .replace("__PRIOR_N__", str((priors() or {}).get("ships", 0)))
          .replace("__S__", str(s)).replace("__W__", str(w)).replace("__N__", str(n)).replace("__E__", str(e)))

    rows = "".join(f"<tr><td>{utc(r.time)}</td><td>{r.n_det}</td><td>{r.n_candidates}</td></tr>" for r in rp.iloc[::-1].itertuples())
    table = ("<div class='tablewrap'><table><thead><tr><th>date</th><th>ships seen</th><th>not tracked</th></tr></thead>"
             f"<tbody>{rows}</tbody></table></div>") if rows else "<div class='panel muted'>No pass yet.</div>"
    acq = "".join(f"<li>{html.escape(x['start'][:16].replace('T', ' '))} UTC</li>" for x in scenes[:8])
    recent = f"<h2>Recent satellite passes</h2><ul class='chips'>{acq}</ul>" if acq else ""
    tag = "<span class='tag amber'>Radar only</span>" if radar_only else "<span class='tag green'>Radar + AIS</span>"
    body = f"""<div class="titlerow"><h1>{TITLES[region]}</h1>{tag}</div><p class="sub">{SUBTITLES[region]}</p>
<div class="grid">
 {readout("Broadcasting now", len(live), "cyan")}
 {readout("Ships seen, latest pass", seen)}
 {readout("Tracked by AIS", n_m, "green")}
 {readout("Untracked", n_high, "red")}
</div>
<h2>Latest pass &middot; {shown}</h2>
<div class="inspect"><div id="map" class="map"></div>
<aside id="side" class="side" aria-live="polite"><div class="empty">Select a dot on the map to inspect it</div></aside></div>{js}
<h2>To review</h2>{review}
<h2>Pass history</h2>{table}
{recent}"""
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(page(f"{TITLES[region]} · Dark Vessel Detection", body, depth=1, leaflet=True, active=region, feed=feed))
    return dict(region=region, passes=len(rp), live=len(live), shown=shown, seen=seen, latest_matched=n_m, latest_high=n_high,
                radar_only=radar_only, matched=int(rp.n_matched.sum()) if len(rp) else 0,
                high=int((rp.n_candidates - rp.n_low_conf).sum()) if len(rp) else 0)


def nn_page(out: str, feed: dict, report_path: str = os.path.join("models", "ship_cnn_report.json")) -> dict | None:
    """Methodology: the neural network's architecture and how it scores on the held-out Oresund. Linked from the footer."""
    if not os.path.exists(report_path):
        return None
    rep = json.load(open(report_path))
    os.makedirs(out, exist_ok=True)
    arch = "".join(f"<tr><td>{html.escape(r['layer'])}</td><td>{r['output']}</td><td>{html.escape(r['activation'])}</td>"
                   f"<td>{r['weights']:,}</td></tr>" for r in rep["architecture"])
    t, h, b = rep["test"]["cnn"], rep["test"].get("cnn_at_0_5", rep["test"]["cnn"]), rep["test"]["cfar"]
    thr = rep.get("threshold", 0.5)

    tuned = thr != 0.5       # show a separate 0.5 column only when the operating threshold differs from it

    def row(name, k, fmt="{:.3f}"):
        mid = f"<td>{fmt.format(h[k])}</td>" if tuned else ""
        return f"<tr><td>{name}</td><td>{fmt.format(t[k])}</td>{mid}<td>{fmt.format(b[k])}</td></tr>"

    comp = "".join(row(n, k) for n, k in [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"),
                                         ("F1 score", "f1"), ("ROC-AUC", "roc_auc")])
    comp += "".join(row(n, k, "{:d}") for n, k in [("True positives (ships found)", "tp"), ("False positives", "fp"),
                                                  ("True negatives", "tn"), ("False negatives (ships missed)", "fn")])
    body = f"""<h1>Methodology</h1><p class="sub">ShipCNN, the neural network behind the detector</p>
<div class="grid">{readout("Weights", f"{rep['weights']:,}")}{readout("Layers with weights", 6)}
{readout("Test F1", f"{t['f1']:.2f}")}{readout("Test ROC-AUC", f"{t['roc_auc']:.2f}", "cyan")}
{readout("Threshold", f"{thr:g}", "amber")}</div>
<h2>Architecture</h2>
<div class="tablewrap"><table><thead><tr><th>layer</th><th>output (channels x height x width)</th><th>activation</th><th>weights</th></tr></thead>
<tbody><tr><td>0. Input: radar chip, log10 backscatter / 2</td><td>1 x 64 x 64</td><td>-</td><td>0</td></tr>{arch}</tbody></table></div>
<h2>Test on the held-out Øresund</h2>
<div class="tablewrap"><table><thead><tr><th>metric</th><th>ShipCNN @ {thr:g}</th>{"<th>ShipCNN @ 0.5</th>" if tuned else ""}<th>CFAR only</th></tr></thead>
<tbody>{comp}</tbody></table></div>
<h2>Test chips</h2>
<img class="chart" src="examples.png" alt="example radar chips with predicted probabilities" style="max-width:820px">"""
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(page("Methodology · Dark Vessel Detection", body, depth=1, active="nn", feed=feed))
    return rep


def build(out: str, live_db: str, result_dbs: list[str], check_catalogue: bool = True) -> str:
    os.makedirs(out, exist_ok=True)
    passes, dets = collect_passes(result_dbs)
    con = sqlite3.connect(live_db) if os.path.exists(live_db) else None
    fresh = [ais_status(con, aoi=a).get("minutes_since_last") for a in (C.REGIONS[r] for r in C.SITE_REGIONS)] if con else []
    feed = dict(live=any(f is not None and f < LIVE_WITHIN_MIN for f in fresh))
    for region in set(C.REGIONS) - set(C.SITE_REGIONS):
        stale = os.path.join(out, region, "index.html")    # a page from before the area was hidden
        if os.path.exists(stale):
            os.remove(stale)
    summaries = []
    for region in C.SITE_REGIONS:
        scenes = []
        if check_catalogue:
            try:
                from .live import find_scenes
                scenes = list({x["start"][:16]: x for x in find_scenes(days=14, aoi=C.REGIONS[region])}.values())
            except Exception:
                scenes = []
        summaries.append(region_page(region, passes, dets, live_db, os.path.join(out, region), scenes, feed))
    nn_page(os.path.join(out, "nn"), feed)

    areas = "".join(f"""<a class="panel area" href="{s['region']}/index.html"><div class="titlerow"><h3>{TITLES[s['region']]}</h3>
<span class="tag {'amber' if s['radar_only'] else 'green'}">{'Radar only' if s['radar_only'] else 'Radar + AIS'}</span></div>
<p class="sub">{SUBTITLES[s['region']]}</p>
<div class="kv"><span>Latest pass</span><span>{s['shown']}</span></div>
<div class="kv"><span>Ships seen</span><span>{s['seen']}</span></div>
<div class="kv"><span>Tracked by AIS</span><span>{s['latest_matched']}</span></div>
<div class="kv"><span>Untracked</span><span>{s['latest_high']}</span></div>
<div class="kv"><span>Broadcasting now</span><span>{s['live']}</span></div>
<div class="go">Open map &rarr;</div></a>""" for s in summaries)
    body = f"""<h1>Ships the radar sees that nobody is tracking</h1>
<p class="sub">Radar satellite images checked against ship broadcasts, pass by pass</p>
<div class="grid">{readout("Satellite passes", sum(s['passes'] for s in summaries))}
{readout("Ships tracked by AIS", sum(s['matched'] for s in summaries), "green")}
{readout("Untracked ships", sum(s['high'] for s in summaries), "red")}
{readout("Broadcasting now", sum(s['live'] for s in summaries), "cyan")}</div>
<h2>Areas</h2><div class="grid">{areas}</div>"""
    path = os.path.join(out, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page("Dark Vessel Detection", body, active="overview", feed=feed))
    return path
