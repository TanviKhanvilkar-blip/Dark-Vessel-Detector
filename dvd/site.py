"""Public static site, rebuilt on a schedule: overview, one page per region, and the simulation demo.

Inputs are SQLite databases: the live AIS recording, plus any databases that hold processed passes
(`passes` and `detections` tables). Each pass is assigned to a region by where its detections are.
"""
from __future__ import annotations

import html
import json
import os
import shutil
import sqlite3
import time
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config as C
from .live import ais_status, recording_box

TITLES = {"mumbai": "Mumbai coast and harbour, India", "denmark": "Oresund, Copenhagen (Denmark/Sweden)"}
NOTES = {
    "mumbai": "The project's target area. Sentinel-1 images it about once every 12 days, and free AIS comes only from "
              "coastal receivers, so many unmatched ships here are simply out of receiver range.",
    "denmark": "Validation area. Sentinel-1 images it almost daily and coastal AIS coverage is dense, so the full "
               "match between radar and AIS can be shown and checked.",
}


def utc(t, fmt="%Y-%m-%d %H:%M UTC"):
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


def ais_hourly(live_db: str, aoi, hours: int = 48) -> pd.Series:
    if not os.path.exists(live_db):
        return pd.Series(dtype=float)
    w, s, e, n = recording_box(aoi)
    con = sqlite3.connect(live_db)
    try:
        df = pd.read_sql("SELECT CAST(time / 3600 AS INTEGER) AS h, COUNT(*) AS n FROM ais WHERE time > ? AND lon BETWEEN ? AND ? "
                         "AND lat BETWEEN ? AND ? GROUP BY h", con, params=(time.time() - hours * 3600, w, e, s, n))
    finally:
        con.close()
    return pd.Series(df.n.values, index=pd.to_datetime(df.h * 3600, unit="s", utc=True))


PAL = dict(bg="#05090d", panel="#0a1218", line="#16303a", ink="#d3ece9", muted="#6d8c91", green="#2ef2b0",
           red="#ff4d5e", amber="#ffb547", cyan="#4fc3ff")


def charts(region: str, passes: pd.DataFrame, hourly: pd.Series, path: str) -> None:
    with plt.rc_context({"font.family": "monospace", "text.color": PAL["ink"], "axes.labelcolor": PAL["muted"],
                         "xtick.color": PAL["muted"], "ytick.color": PAL["muted"], "axes.edgecolor": PAL["line"]}):
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.2), facecolor=PAL["panel"])
        for a in ax:
            a.set_facecolor(PAL["panel"])
            a.grid(axis="y", color=PAL["line"], linewidth=0.6)
            a.set_axisbelow(True)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
            a.tick_params(labelsize=8)
        if len(hourly):
            ax[0].bar(hourly.index, hourly.values, width=1 / 24 * 0.85, align="edge", color=PAL["cyan"])  # bar covers its hour
            now = pd.Timestamp.now(tz="UTC")
            ax[0].set_xlim(now - pd.Timedelta(hours=48), now + pd.Timedelta(hours=1))
            import matplotlib.dates as mdates
            ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
            ax[0].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
            ax[0].tick_params(axis="x", labelrotation=30)
        else:
            ax[0].text(0.5, 0.5, "NO AIS RECEIVED YET", ha="center", va="center", color=PAL["muted"], transform=ax[0].transAxes)
            ax[0].set_xticks([]); ax[0].set_yticks([])
        ax[0].set_title("LIVE AIS MESSAGES / HOUR (48 H)", fontsize=9, loc="left", color=PAL["ink"])
        if len(passes):
            x = [utc(t, "%d %b %H:%M") for t in passes.time]
            ax[1].bar(x, passes.n_matched, width=0.5, color=PAL["green"], label="matched to AIS")
            ax[1].bar(x, passes.n_candidates, width=0.5, bottom=passes.n_matched, color=PAL["red"], label="not publicly tracked")
            ax[1].set_xlim(-0.75, max(len(x), 4) - 0.25)     # keep bars narrow even with one or two passes
            ax[1].set_ylim(0, (passes.n_matched + passes.n_candidates).max() * 1.3 + 1)
            leg = ax[1].legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
            for t in leg.get_texts():
                t.set_color(PAL["ink"])
            ax[1].tick_params(axis="x", labelrotation=30, labelsize=8)
        else:
            ax[1].text(0.5, 0.5, "NO PASSES PROCESSED YET", ha="center", va="center", color=PAL["muted"], transform=ax[1].transAxes)
            ax[1].set_xticks([]); ax[1].set_yticks([])
        ax[1].set_title("SHIPS DETECTED PER SATELLITE PASS", fontsize=9, loc="left", color=PAL["ink"])
        fig.tight_layout()
        fig.savefig(path, dpi=110, facecolor=PAL["panel"])
        plt.close(fig)


CSS = """
:root{--bg:#05090d;--panel:#0a1218;--panel2:#0d1820;--line:#16303a;--ink:#d3ece9;--muted:#6d8c91;
--green:#2ef2b0;--red:#ff4d5e;--amber:#ffb547;--cyan:#4fc3ff;--glow:0 0 12px rgba(46,242,176,.35);
--mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Consolas,monospace;--sans:'Inter',system-ui,-apple-system,Segoe UI,sans-serif}
*{box-sizing:border-box}html{background:var(--bg)}
body{margin:0;color:var(--ink);font:15px/1.6 var(--sans);background:
 radial-gradient(1200px 500px at 70% -10%,rgba(46,242,176,.08),transparent 60%),
 linear-gradient(rgba(46,242,176,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(46,242,176,.035) 1px,transparent 1px),var(--bg);
 background-size:auto,44px 44px,44px 44px,auto;min-height:100vh}
a{color:var(--green);text-decoration:none}a:hover{text-shadow:var(--glow)}
.wrap{max-width:1120px;margin:0 auto;padding:0 16px}
header.hud{border-bottom:1px solid var(--line);background:linear-gradient(180deg,rgba(10,18,24,.95),rgba(5,9,13,.85));position:sticky;top:0;z-index:1000;backdrop-filter:blur(6px)}
.hudrow{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:12px 0;flex-wrap:wrap}
.brand{display:flex;align-items:center;gap:14px}
.scope{width:52px;height:52px;border-radius:50%;position:relative;overflow:hidden;flex:none;border:1px solid rgba(46,242,176,.5);
 background:repeating-radial-gradient(circle,transparent 0 9px,rgba(46,242,176,.22) 9px 10px),
 linear-gradient(rgba(46,242,176,.25),rgba(46,242,176,.25)) center/1px 100% no-repeat,
 linear-gradient(90deg,rgba(46,242,176,.25),rgba(46,242,176,.25)) center/100% 1px no-repeat,#03110c;box-shadow:var(--glow)}
.sweep{position:absolute;inset:0;border-radius:50%;background:conic-gradient(from 0deg,rgba(46,242,176,.65),rgba(46,242,176,0) 75deg,transparent);animation:spin 3.2s linear infinite}
.blip{position:absolute;width:4px;height:4px;border-radius:50%;background:var(--red);box-shadow:0 0 6px var(--red);animation:blink 3.2s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}@keyframes blink{0%,60%{opacity:.15}70%{opacity:1}100%{opacity:.15}}
@media (prefers-reduced-motion:reduce){.sweep,.blip,.dot.live{animation:none}}
.kicker{font:600 11px/1.2 var(--mono);letter-spacing:.18em;color:var(--muted)}
.title{font:700 20px/1.2 var(--mono);letter-spacing:.06em;color:var(--ink)}
.title b{color:var(--green);font-weight:700}
.status{display:flex;gap:10px;flex-wrap:wrap;font:12px/1 var(--mono)}
.pill{border:1px solid var(--line);background:var(--panel);padding:7px 10px;border-radius:999px;color:var(--muted);white-space:nowrap}
.pill strong{color:var(--ink);font-weight:600}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:7px;vertical-align:0}
.dot.live{background:var(--green);box-shadow:0 0 8px var(--green);animation:pulse 1.6s ease-in-out infinite}.dot.off{background:var(--red)}
@keyframes pulse{50%{opacity:.35}}
nav.tabs{display:flex;gap:4px;overflow-x:auto;padding:0 0 0;margin-top:-1px}
nav.tabs a{font:600 12px/1 var(--mono);letter-spacing:.12em;color:var(--muted);padding:12px 14px;border-bottom:2px solid transparent;white-space:nowrap}
nav.tabs a.on{color:var(--green);border-bottom-color:var(--green);text-shadow:var(--glow)}nav.tabs a:hover{color:var(--ink)}
main{padding:24px 0 40px}
h1{font:700 28px/1.2 var(--sans);margin:4px 0 6px;letter-spacing:-.01em}
h2{font:600 12px/1 var(--mono);letter-spacing:.18em;color:var(--muted);margin:34px 0 12px;text-transform:uppercase;display:flex;align-items:center;gap:10px}
h2::after{content:"";flex:1;height:1px;background:var(--line)}
.lede{color:#a9c6c3;max-width:760px}.muted{color:var(--muted)}.mono{font-family:var(--mono)}
.coords{font:12px/1.4 var(--mono);color:var(--muted);letter-spacing:.06em}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}
.panel{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:6px;padding:14px 16px;position:relative}
.panel::before,.panel::after{content:"";position:absolute;width:10px;height:10px;border:1px solid var(--green);opacity:.55}
.panel::before{top:-1px;left:-1px;border-right:0;border-bottom:0}.panel::after{bottom:-1px;right:-1px;border-left:0;border-top:0}
.label{font:600 10.5px/1.2 var(--mono);letter-spacing:.16em;color:var(--muted);text-transform:uppercase;margin-bottom:8px}
.value{font:600 30px/1.1 var(--mono);color:var(--green);text-shadow:var(--glow)}
.value.red{color:var(--red);text-shadow:0 0 12px rgba(255,77,94,.35)}.value.cyan{color:var(--cyan);text-shadow:0 0 12px rgba(79,195,255,.35)}
.value.small{font-size:16px}.ok{color:var(--green)}.bad{color:var(--red)}
.alert{border:1px solid rgba(255,181,71,.45);background:rgba(255,181,71,.07);border-radius:6px;padding:12px 16px;color:#f3dcb4}
.alert b{color:var(--amber)}
.flow{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}
.step{border:1px solid var(--line);background:var(--panel);border-radius:6px;padding:12px}
.step .n{font:700 11px/1 var(--mono);color:var(--green);letter-spacing:.14em}.step .t{font:600 14px/1.3 var(--sans);margin:6px 0 4px}
.step .d{font-size:12.5px;color:var(--muted);line-height:1.45}
@media (max-width:760px){.flow{grid-template-columns:1fr 1fr}.title{font-size:16px}h1{font-size:23px}.value{font-size:24px}
 header.hud{position:static}.status .pill:nth-child(3){display:none}.map{height:420px}}
.region{display:block;color:inherit}.region:hover{border-color:rgba(46,242,176,.5)}
.region .rt{font:600 16px/1.3 var(--sans);color:var(--ink);margin-bottom:6px}.region .go{font:600 11px/1 var(--mono);color:var(--green);letter-spacing:.14em;margin-top:10px}
.kv{display:flex;justify-content:space-between;font:13px/1.9 var(--mono);border-bottom:1px dashed var(--line)}.kv span:last-child{color:var(--ink)}
.map{height:520px;border:1px solid var(--line);border-radius:6px;background:#0a0f13}
.legend{display:flex;flex-wrap:wrap;gap:16px;font:12px/1 var(--mono);color:var(--muted);margin:0 0 10px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:6px}
table{border-collapse:collapse;width:100%;font:13px/1.4 var(--mono);background:var(--panel)}
th{font-weight:600;font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);background:var(--panel2)}
td,th{padding:9px 12px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}td:first-child,th:first-child{text-align:left}
tr:last-child td{border-bottom:0}tbody tr:hover td{background:rgba(46,242,176,.04)}
.chart{border:1px solid var(--line);border-radius:6px;display:block;width:100%}
ul.acq{list-style:none;padding:0;margin:0;display:flex;flex-wrap:wrap;gap:8px}
ul.acq li{font:12px/1 var(--mono);border:1px solid var(--line);background:var(--panel);padding:8px 10px;border-radius:4px}
footer{border-top:1px solid var(--line);padding:18px 0 28px;font:12px/1.6 var(--mono);color:var(--muted)}
.leaflet-container{background:#0a0f13;font:12px var(--mono)}
.radar-tiles{filter:invert(100%) hue-rotate(180deg) grayscale(70%) brightness(78%) contrast(118%)}
.leaflet-popup-content-wrapper,.leaflet-popup-tip{background:#0d1820;color:var(--ink);border:1px solid var(--line);box-shadow:0 6px 24px rgba(0,0,0,.6)}
.leaflet-popup-content b{color:var(--green)}.leaflet-popup-content img{border:1px solid var(--line);margin-top:6px;image-rendering:pixelated}
.leaflet-control-layers,.leaflet-bar a{background:#0d1820!important;color:var(--ink)!important;border-color:var(--line)!important}
.leaflet-control-attribution{background:rgba(5,9,13,.75)!important;color:var(--muted)!important}.leaflet-control-attribution a{color:var(--muted)}
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">')


def page(title: str, body: str, depth: int = 0, leaflet: bool = False, active: str = "overview", feed: dict | None = None) -> str:
    up = "../" * depth
    tabs = [("overview", f"{up}index.html", "OVERVIEW")] + [(r, f"{up}{r}/index.html", r.upper()) for r in C.REGIONS] + \
           [("demo", f"{up}demo/index.html", "SIMULATION")]
    nav = "".join(f'<a class="{"on" if k == active else ""}" href="{href}">{label}</a>' for k, href, label in tabs)
    feed = feed or {}
    dot = "live" if feed.get("live") else "off"
    feed_txt = "AIS FEED <strong>LIVE</strong>" if feed.get("live") else "AIS FEED <strong>OFFLINE</strong>"
    lf = ('<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">'
          '<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>') if leaflet else ""
    blips = '<span class="blip" style="left:32px;top:14px"></span><span class="blip" style="left:15px;top:33px;animation-delay:1.1s"></span>'
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark"><title>{html.escape(title)}</title>{FONTS}<style>{CSS}</style>{lf}</head><body>
<header class="hud"><div class="wrap"><div class="hudrow">
 <a class="brand" href="{up}index.html"><div class="scope"><div class="sweep"></div>{blips}</div>
  <div><div class="kicker">SAR &times; AIS &middot; MARITIME AWARENESS</div><div class="title">DARK VESSEL <b>DETECTION</b></div></div></a>
 <div class="status"><span class="pill"><span class="dot {dot}"></span>{feed_txt}</span>
  <span class="pill">UTC <strong id="utc">--:--:--</strong></span>
  <span class="pill">UPDATED <strong>{utc(time.time(), '%d %b %H:%M')}</strong></span></div>
</div><nav class="tabs">{nav}</nav></div></header>
<main class="wrap">{body}</main>
<footer><div class="wrap">Data: Copernicus Sentinel-1 (ESA) &middot; AIS via aisstream.io and the Danish Maritime Authority &middot;
coastlines, structures and basemap &copy; OpenStreetMap contributors. Flags are candidates for review, not accusations.</div></footer>
<script>(function(){{const e=document.getElementById('utc');const t=()=>e.textContent=new Date().toISOString().slice(11,19);t();setInterval(t,1000);}})();</script>
</body></html>"""


MAP_JS = """<script>
const D=__DATA__,LIVE=__LIVE__,B=[[__S__,__W__],[__N__,__E__]];const m=L.map('map',{zoomSnap:.25}).fitBounds(B);
// re-fit once layout has settled: if the container had no size when the map was created, Leaflet falls back to zoom 0
const refit=()=>{m.invalidateSize();m.fitBounds(B);};setTimeout(refit,0);window.addEventListener('load',refit);
// standard OSM tiles (no API key), turned into a dark "radar console" basemap with a CSS filter (.radar-tiles)
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,className:'radar-tiles',
 attribution:'&copy; OpenStreetMap contributors'}).addTo(m);
L.rectangle(B,{color:'#2ef2b0',weight:1,dashArray:'4 6',fill:false,opacity:.6}).addTo(m);
const det=L.layerGroup().addTo(m),live=L.layerGroup().addTo(m);
D.forEach(d=>{const c=d.status=='matched'?'#2ef2b0':(d.confidence=='low'?'#ffb547':'#ff4d5e');
let h='<b>'+(d.status=='matched'?'MATCHED TO AIS':'NOT PUBLICLY TRACKED')+'</b><br>length ~'+d.length_m.toFixed(0)+' m &middot; score '+d.score.toFixed(2)+'<br>'+d.shore_km.toFixed(1)+' km from shore';
if(d.reason)h+='<br><span style="color:#ffb547">caveat: '+d.reason.replaceAll('_',' ').replaceAll(',',', ')+'</span>';if(d.crop)h+='<br><img src="'+d.crop+'" width="160">';
L.circleMarker([d.lat,d.lon],{radius:d.status=='matched'?4.5:7,color:c,fillColor:c,fillOpacity:.85,weight:d.status=='matched'?1:2,opacity:1}).bindPopup(h).addTo(det);});
LIVE.forEach(p=>L.circleMarker([p.lat,p.lon],{radius:3,color:'#4fc3ff',fillColor:'#4fc3ff',fillOpacity:.9,weight:1}).bindPopup('<b>LIVE AIS</b><br>'+p.sog+' kn &middot; '+p.age+' min ago').addTo(live));
L.control.layers(null,{'Radar detections (selected pass)':det,'Live AIS, last 30 min':live},{collapsed:false}).addTo(m);
</script>"""


def readout(label: str, value, cls: str = "") -> str:
    return f'<div class="panel"><div class="label">{label}</div><div class="value {cls}">{value}</div></div>'


def region_page(region: str, passes: pd.DataFrame, dets: pd.DataFrame, live_db: str, out: str, scenes: list[dict], feed: dict) -> dict:
    aoi = C.REGIONS[region]
    os.makedirs(out, exist_ok=True)
    rp = passes[passes.region == region] if len(passes) else passes
    st = ais_status(sqlite3.connect(live_db), aoi=aoi) if os.path.exists(live_db) else {}
    live = live_positions(live_db, aoi)
    charts(region, rp, ais_hourly(live_db, aoi), os.path.join(out, "charts.png"))

    if len(rp):
        latest = rp.iloc[-1]
        d = dets[(dets.pass_id == latest.pass_id) & (dets.status != "filtered")]
        data = d[["lon", "lat", "length_m", "score", "shore_km", "status", "reason", "confidence", "crop"]].fillna("").to_dict("records")
        shown = f"Showing pass <span class='mono'>{html.escape(latest.pass_id)}</span> &middot; {utc(latest.time)}"
        n_m, n_c = int(latest.n_matched), int(latest.n_candidates)
    else:
        data, shown, n_m, n_c = [], "No satellite pass processed for this area yet.", 0, 0
    w, s, e, n = aoi
    js = (MAP_JS.replace("__DATA__", json.dumps(data)).replace("__LIVE__", json.dumps(live))
          .replace("__S__", str(s)).replace("__W__", str(w)).replace("__N__", str(n)).replace("__E__", str(e)))

    rows = "".join(
        f"<tr><td>{utc(r.time)}</td><td>{r.n_det}</td><td class='ok'>{r.n_matched}</td><td class='bad'>{r.n_candidates - r.n_low_conf}</td>"
        f"<td style='color:var(--amber)'>{r.n_low_conf}</td><td>{'&ndash;' if pd.isna(r.match_rate) else f'{r.match_rate:.0%}'}</td><td>{r.ais_msgs}</td></tr>"
        for r in rp.iloc[::-1].itertuples())
    table = ("<div class='tablewrap'><table><thead><tr><th>pass (UTC)</th><th>ships</th><th>matched</th><th>not tracked</th>"
             f"<th>low conf.</th><th>match rate</th><th>AIS msgs</th></tr></thead><tbody>{rows}</tbody></table></div>") if rows \
        else "<p class='muted mono'>No passes processed yet.</p>"
    acq = "".join(f"<li>{html.escape(x['start'][:16].replace('T', ' '))} UTC</li>" for x in scenes[:8]) or "<li>catalogue unavailable</li>"
    fresh = st.get("minutes_since_last")
    rec = (f"<span class='ok'>RECEIVING</span>" if fresh is not None and fresh < 10 else "<span class='bad'>NO SIGNAL</span>")
    since = f"last message {fresh:.0f} min ago" if fresh is not None else "no message received yet"
    quiet = ("<div class='alert' style='margin-top:12px'><b>Live AIS coverage.</b> The free aisstream network has delivered no "
             "messages for this area so far: it relies on volunteer coastal receivers, and none is reporting here. "
             "This is the free-data blind spot the project measures.</div>") if not st.get("messages") else ""
    body = f"""<div class="coords">SECTOR // {region.upper()} &middot; {s:.2f}&ndash;{n:.2f}&deg;N &middot; {w:.2f}&ndash;{e:.2f}&deg;E</div>
<h1>{TITLES[region]}</h1><p class="lede">{NOTES[region]}</p>
<div class="grid">
 <div class="panel"><div class="label">Live AIS feed</div><div class="value small">{rec}</div><div class="muted mono" style="font-size:12px;margin-top:6px">{since}</div></div>
 {readout("Ships broadcasting &middot; 30 min", len(live), "cyan")}
 {readout("Matched &middot; latest pass", n_m)}
 {readout("Not tracked &middot; latest pass", n_c, "red")}
</div>{quiet}
<h2>Tactical map</h2><p class="muted mono" style="font-size:12.5px;margin:-4px 0 10px">{shown}</p>
<div class="legend"><span><i style="background:var(--green)"></i>matched to AIS</span><span><i style="background:var(--red)"></i>not tracked &middot; high confidence</span>
<span><i style="background:var(--amber)"></i>not tracked &middot; low confidence</span><span><i style="background:var(--cyan)"></i>live AIS position</span></div>
<div id="map" class="map"></div>{js}
<h2>Pass log</h2>{table}
<h2>Telemetry</h2><img class="chart" src="charts.png" alt="AIS messages per hour and ships detected per pass">
<p class="muted mono" style="font-size:12px">Feed gaps in the last 24 h: {st.get('gaps_last_24h', 0)} ({st.get('gap_minutes_last_24h', 0):.0f} min) &middot; AIS messages in the last hour: {st.get('messages_last_hour', 0)}</p>
<h2>Recent Sentinel-1 acquisitions</h2><ul class="acq">{acq}</ul>"""
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(page(TITLES[region], body, depth=1, leaflet=True, active=region, feed=feed))
    return dict(region=region, passes=len(rp), live=len(live), st=st,
                matched=int(rp.n_matched.sum()) if len(rp) else 0, cands=int(rp.n_candidates.sum()) if len(rp) else 0)


def demo_page(demo_dir: str, out: str, feed: dict) -> None:
    """The simulation (python -m dvd demo) in the same layout. Synthetic scenes with known ground truth."""
    os.makedirs(out, exist_ok=True)
    if os.path.isdir(os.path.join(demo_dir, "crops")):
        shutil.copytree(os.path.join(demo_dir, "crops"), os.path.join(out, "crops"), dirs_exist_ok=True)
    con = sqlite3.connect(os.path.join(demo_dir, "dvd.sqlite"))
    try:
        p = pd.read_sql("SELECT * FROM passes ORDER BY time", con)
        d = pd.read_sql("SELECT * FROM detections", con)
    finally:
        con.close()
    if p.empty:
        return
    p = p.assign(region="demo")
    charts("demo", p, pd.Series(dtype=float), os.path.join(out, "charts.png"))
    latest = p.iloc[-1]
    dd = d[(d.pass_id == latest.pass_id) & (d.status != "filtered")]
    data = dd[["lon", "lat", "length_m", "score", "shore_km", "status", "reason", "confidence", "crop"]].fillna("").to_dict("records")
    w, s, e, n = C.AOI
    js = (MAP_JS.replace("__DATA__", json.dumps(data)).replace("__LIVE__", "[]")
          .replace("__S__", str(s)).replace("__W__", str(w)).replace("__N__", str(n)).replace("__E__", str(e)))
    rows = "".join(
        f"<tr><td>{html.escape(r.pass_id)}</td><td>{r.n_det}</td><td class='ok'>{r.n_matched}</td><td class='bad'>{r.n_candidates - r.n_low_conf}</td>"
        f"<td style='color:var(--amber)'>{r.n_low_conf}</td><td>{r.match_rate:.0%}</td><td>{r.ais_msgs}</td><td>{r.ais_gap_s / 60:.0f} min</td></tr>"
        for r in p.iloc[::-1].itertuples())
    body = f"""<div class="coords">SIMULATION // SYNTHETIC SCENES WITH KNOWN GROUND TRUTH</div>
<h1>Simulation: the whole pipeline with a known answer</h1>
<p class="lede">Four synthetic Sentinel-1-like scenes with planted ships, simulated AIS, two fixed platforms, rougher sea in pass 3 and an
AIS feed outage in pass 4. Because every ship is known, detection accuracy and the "dark" labels can be checked exactly.
Synthetic radar is easier than real radar; real results are on the sector pages.</p>
<div class="grid">{readout("Passes simulated", len(p))}{readout("Ships matched", int(p.n_matched.sum()))}
{readout("Not tracked", int(p.n_candidates.sum()), "red")}{readout("Flags downgraded", int(p.n_low_conf.sum()), "cyan")}</div>
<h2>Map &middot; pass with AIS outage</h2><p class="muted mono" style="font-size:12.5px;margin:-4px 0 10px">Showing {html.escape(latest.pass_id)}: every
flag is low confidence because the AIS feed had a 25-minute gap, so the system refuses to call these ships dark.</p>
<div class="legend"><span><i style="background:var(--green)"></i>matched to AIS</span><span><i style="background:var(--red)"></i>not tracked &middot; high confidence</span>
<span><i style="background:var(--amber)"></i>not tracked &middot; low confidence</span></div>
<div id="map" class="map"></div>{js}
<h2>Pass log</h2><div class="tablewrap"><table><thead><tr><th>pass</th><th>ships</th><th>matched</th><th>not tracked</th><th>low conf.</th>
<th>match rate</th><th>AIS msgs</th><th>feed gap</th></tr></thead><tbody>{rows}</tbody></table></div>
<h2>Telemetry</h2><img class="chart" src="charts.png" alt="ships detected per simulated pass">"""
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(page("Simulation", body, depth=1, leaflet=True, active="demo", feed=feed))


FLOW = [("01", "Radar image", "Sentinel-1 sees every large ship, day or night, through cloud."),
        ("02", "Detect", "CFAR finds bright targets on dark sea; land, bridges and buoys are masked."),
        ("03", "Align AIS", "Each ship's last AIS report is moved along its course to the exact image time."),
        ("04", "Match", "The Hungarian algorithm pairs radar targets with AIS ships within a speed-aware radius."),
        ("05", "Flag", "Unmatched targets become candidates, with an honest confidence label.")]


def build(out: str, live_db: str, result_dbs: list[str], demo_dir: str | None = None, check_catalogue: bool = True) -> str:
    os.makedirs(out, exist_ok=True)
    passes, dets = collect_passes(result_dbs)
    con = sqlite3.connect(live_db) if os.path.exists(live_db) else None
    fresh = [ais_status(con, aoi=a).get("minutes_since_last") for a in C.REGIONS.values()] if con else []
    feed = dict(live=any(f is not None and f < 10 for f in fresh))
    summaries = []
    for region in C.REGIONS:
        scenes = []
        if check_catalogue:
            try:
                from .live import find_scenes
                scenes = list({x["start"][:16]: x for x in find_scenes(days=14, aoi=C.REGIONS[region])}.values())
            except Exception:
                scenes = []
        summaries.append(region_page(region, passes, dets, live_db, os.path.join(out, region), scenes, feed))

    if demo_dir and os.path.exists(os.path.join(demo_dir, "dvd.sqlite")):
        demo_page(demo_dir, os.path.join(out, "demo"), feed)

    log_path = os.path.join(out, "runlog.json")
    try:
        log = json.load(open(log_path))
    except Exception:
        log = []
    log.append(dict(t=time.time(), **{s["region"]: dict(live=s["live"], msgs_hour=s["st"].get("messages_last_hour", 0)) for s in summaries}))
    log = log[-500:]
    json.dump(log, open(log_path, "w"))

    total_m = sum(s["matched"] for s in summaries)
    total_c = sum(s["cands"] for s in summaries)
    total_p = sum(s["passes"] for s in summaries)
    total_live = sum(s["live"] for s in summaries)
    flow = "".join(f'<div class="step"><div class="n">{n}</div><div class="t">{t}</div><div class="d">{d}</div></div>' for n, t, d in FLOW)
    regions = "".join(f"""<a class="panel region" href="{s['region']}/index.html"><div class="label">Sector // {s['region'].upper()}</div>
<div class="rt">{TITLES[s['region']]}</div>
<div class="kv"><span>passes processed</span><span>{s['passes']}</span></div>
<div class="kv"><span>ships matched to AIS</span><span class="ok">{s['matched']}</span></div>
<div class="kv"><span>not publicly tracked</span><span class="bad">{s['cands']}</span></div>
<div class="kv"><span>broadcasting now (30 min)</span><span style="color:var(--cyan)">{s['live']}</span></div>
<div class="go">OPEN SECTOR &rarr;</div></a>""" for s in summaries)
    runs = "".join(f"<tr><td>{utc(r['t'], '%Y-%m-%d %H:%M')}</td>" + "".join(
        f"<td>{r.get(k, {}).get('live', '')}</td><td>{r.get(k, {}).get('msgs_hour', '')}</td>" for k in C.REGIONS) + "</tr>"
        for r in log[::-1][:24])
    heads = "".join(f"<th>{k} live</th><th>{k} msg/h</th>" for k in C.REGIONS)
    body = f"""<div class="coords">SYSTEM OVERVIEW</div>
<h1>Ships the radar sees that nobody is tracking</h1>
<p class="lede">Radar satellites see ships whether or not they announce themselves. AIS is what ships broadcast about themselves.
This system compares the two after every Sentinel-1 pass and lists the ships that appear on radar with no matching AIS signal.</p>
<div class="grid" style="margin-top:18px">{readout("Satellite passes processed", total_p)}{readout("Ships matched to AIS", total_m)}
{readout("Not publicly tracked", total_c, "red")}{readout("Broadcasting now &middot; 30 min", total_live, "cyan")}</div>
<h2>Pipeline</h2><div class="flow">{flow}</div>
<h2>Sectors</h2><div class="grid">{regions}</div>
<h2>Read before trusting a flag</h2>
<div class="alert"><b>"Not publicly tracked" is not "switched off".</b> Small boats often carry no public AIS, free AIS comes from coastal
receivers with limited range, and ships under about 40 m are not reliably visible to the satellite. Low-confidence flags are the ones the
data cannot support: beyond receiver range, during an AIS feed gap, or with no AIS recorded.</div>
<h2>Automatic runs</h2><p class="muted mono" style="font-size:12.5px;margin-top:-4px">Rebuilt on a schedule from the live AIS recording. Newest first.</p>
<div class="tablewrap"><table><thead><tr><th>run (UTC)</th>{heads}</tr></thead><tbody>{runs}</tbody></table></div>"""
    path = os.path.join(out, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page("Dark Vessel Detection", body, active="overview", feed=feed))
    return path
