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

TITLES = {"mumbai": "Mumbai coast and harbour, India", "denmark": "Great Belt (Storebaelt), Denmark"}
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


def charts(region: str, passes: pd.DataFrame, hourly: pd.Series, path: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3))
    if len(hourly):
        ax[0].bar(hourly.index, hourly.values, width=1 / 24, color="#3b6ea5")
        ax[0].tick_params(axis="x", labelrotation=30)
    else:
        ax[0].text(0.5, 0.5, "no AIS recorded yet", ha="center", va="center", transform=ax[0].transAxes)
    ax[0].set_title("live AIS messages per hour (last 48 h)", fontsize=10)
    if len(passes):
        x = [utc(t, "%d %b %H:%M") for t in passes.time]
        ax[1].bar(x, passes.n_matched, color="#2e7d32", label="matched to AIS")
        ax[1].bar(x, passes.n_candidates, bottom=passes.n_matched, color="#d32f2f", label="not tracked (candidates)")
        ax[1].legend(fontsize=8); ax[1].tick_params(axis="x", labelrotation=30, labelsize=8)
    ax[1].set_title("ships detected per satellite pass", fontsize=10)
    for a in ax:
        a.tick_params(labelsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2330;--muted:#5d6675;--line:#e1e4e9;--accent:#3b6ea5}
@media (prefers-color-scheme:dark){:root{--bg:#14171c;--card:#1c2027;--ink:#e6e8eb;--muted:#9aa3b0;--line:#2c323b;--accent:#7aa7d9}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1060px;margin:0 auto;padding:16px}a{color:var(--accent)}h1{font-size:24px;margin:8px 0 4px}h2{font-size:18px;margin:24px 0 8px}
nav{display:flex;gap:16px;flex-wrap:wrap;padding:10px 16px;background:var(--card);border-bottom:1px solid var(--line)}
.muted{color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.big{font-size:26px;font-weight:600}.note{background:#fff7e0;color:#4a3b00;border:1px solid #ecd48a;border-radius:10px;padding:10px 14px}
@media (prefers-color-scheme:dark){.note{background:#3a3110;color:#f3e3a6;border-color:#6b5a1d}}
.map{height:480px;border-radius:10px;border:1px solid var(--line)}table{border-collapse:collapse;width:100%;background:var(--card);font-size:14px}
td,th{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right}td:first-child,th:first-child{text-align:left}
.tablewrap{overflow-x:auto}.key span{display:inline-block;width:11px;height:11px;border-radius:50%;margin:0 4px 0 12px;vertical-align:middle}
img{max-width:100%}.ok{color:#2e7d32}.bad{color:#c62828}
"""


def page(title: str, body: str, depth: int = 0, leaflet: bool = False) -> str:
    up = "../" * depth
    nav = f'<nav><a href="{up}index.html">Overview</a>' + "".join(
        f'<a href="{up}{r}/index.html">{r.capitalize()}</a>' for r in C.REGIONS) + f'<a href="{up}demo/index.html">Simulation</a></nav>'
    lf = ('<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">'
          '<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>') if leaflet else ""
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(title)}</title><style>{CSS}</style>{lf}</head><body>{nav}<main>{body}</main></body></html>')


MAP_JS = """<script>
const D=__DATA__,LIVE=__LIVE__,B=[[__S__,__W__],[__N__,__E__]];const m=L.map('map').fitBounds(B);
// re-fit once layout has settled: if the container had no size when the map was created, Leaflet falls back to zoom 0
const refit=()=>{m.invalidateSize();m.fitBounds(B);};setTimeout(refit,0);window.addEventListener('load',refit);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:'&copy; OpenStreetMap'}).addTo(m);
const det=L.layerGroup().addTo(m),live=L.layerGroup().addTo(m);
D.forEach(d=>{const c=d.status=='matched'?'#2e7d32':(d.confidence=='low'?'#ef9a2f':'#d32f2f');
let h='<b>'+(d.status=='matched'?'matched to AIS':'not publicly tracked')+'</b><br>length ~'+d.length_m.toFixed(0)+' m, score '+d.score.toFixed(2)+'<br>'+d.shore_km.toFixed(1)+' km from shore';
if(d.reason)h+='<br>caveat: '+d.reason.replaceAll('_',' ').replaceAll(',',', ');if(d.crop)h+='<br><img src="'+d.crop+'" width="160">';
L.circleMarker([d.lat,d.lon],{radius:d.status=='matched'?5:8,color:c,fillColor:c,fillOpacity:.75,weight:1}).bindPopup(h).addTo(det);});
LIVE.forEach(p=>L.circleMarker([p.lat,p.lon],{radius:3,color:'#3b6ea5',fillOpacity:.9,weight:1}).bindPopup('live AIS: '+p.sog+' kn, '+p.age+' min ago').addTo(live));
L.control.layers(null,{'radar detections (selected pass)':det,'live AIS, last 30 min':live}).addTo(m);
</script>"""


def region_page(region: str, passes: pd.DataFrame, dets: pd.DataFrame, live_db: str, out: str, scenes: list[dict]) -> dict:
    aoi = C.REGIONS[region]
    os.makedirs(out, exist_ok=True)
    rp = passes[passes.region == region]
    st = ais_status(sqlite3.connect(live_db), aoi=aoi) if os.path.exists(live_db) else {}
    live = live_positions(live_db, aoi)
    charts(region, rp, ais_hourly(live_db, aoi), os.path.join(out, "charts.png"))

    if len(rp):
        latest = rp.iloc[-1]
        d = dets[(dets.pass_id == latest.pass_id) & (dets.status != "filtered")]
        data = d[["lon", "lat", "length_m", "score", "shore_km", "status", "reason", "confidence", "crop"]].fillna("").to_dict("records")
        shown = f"Map shows pass <b>{html.escape(latest.pass_id)}</b> ({utc(latest.time)})."
    else:
        data, shown = [], "No satellite pass processed for this region yet."
    w, s, e, n = aoi
    js = (MAP_JS.replace("__DATA__", json.dumps(data)).replace("__LIVE__", json.dumps(live))
          .replace("__S__", str(s)).replace("__W__", str(w)).replace("__N__", str(n)).replace("__E__", str(e)))

    rows = "".join(
        f"<tr><td>{utc(r.time)}</td><td>{r.n_det}</td><td>{r.n_matched}</td><td>{r.n_candidates - r.n_low_conf}</td>"
        f"<td>{r.n_low_conf}</td><td>{'' if pd.isna(r.match_rate) else f'{r.match_rate:.0%}'}</td><td>{r.ais_msgs}</td></tr>"
        for r in rp.iloc[::-1].itertuples())
    table = ("<div class='tablewrap'><table><tr><th>pass (UTC)</th><th>ships</th><th>matched</th><th>candidates</th>"
             f"<th>low-confidence</th><th>match rate</th><th>AIS msgs</th></tr>{rows}</table></div>") if rows else "<p class='muted'>None yet.</p>"
    acq = "".join(f"<li>{html.escape(x['start'][:16].replace('T', ' '))} UTC</li>" for x in scenes[:6]) or "<li>catalogue unavailable</li>"
    fresh = st.get("minutes_since_last")
    rec = (f"<span class='ok'>recording</span>, last message {fresh:.0f} min ago" if fresh is not None and fresh < 10
           else "<span class='bad'>not receiving</span>" + (f", last message {fresh / 60:.1f} h ago" if fresh else ""))
    body = f"""<h1>{TITLES[region]}</h1><p class="muted">{NOTES[region]}</p>
<div class="cards">
 <div class="card"><div class="muted">Live AIS</div><div>{rec}</div></div>
 <div class="card"><div class="muted">Ships broadcasting, last 30 min</div><div class="big">{len(live)}</div></div>
 <div class="card"><div class="muted">AIS messages, last hour</div><div class="big">{st.get('messages_last_hour', 0)}</div></div>
 <div class="card"><div class="muted">Satellite passes processed</div><div class="big">{len(rp)}</div></div>
</div>
<h2>Map</h2><p class="muted">{shown} Switch layers at the top right.</p>
<div class="key"><span style="background:#2e7d32"></span>matched to AIS<span style="background:#d32f2f"></span>not tracked, high confidence
<span style="background:#ef9a2f"></span>not tracked, low confidence<span style="background:#3b6ea5"></span>live AIS position</div>
<div id="map" class="map"></div>{js}
<h2>Pass log</h2>{table}
<h2>Monitoring</h2><img src="charts.png" alt="AIS messages per hour and ships per pass">
<h2>Recent Sentinel-1 acquisitions over this area</h2><ul>{acq}</ul>
<p class="muted">Feed gaps in the last 24 h: {st.get('gaps_last_24h', 0)} ({st.get('gap_minutes_last_24h', 0):.0f} min).</p>"""
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(page(TITLES[region], body, depth=1, leaflet=True))
    return dict(region=region, passes=len(rp), live=len(live), st=st,
                matched=int(rp.n_matched.sum()) if len(rp) else 0, cands=int(rp.n_candidates.sum()) if len(rp) else 0)


def build(out: str, live_db: str, result_dbs: list[str], demo_dir: str | None = None, check_catalogue: bool = True) -> str:
    os.makedirs(out, exist_ok=True)
    passes, dets = collect_passes(result_dbs)
    summaries = []
    for region in C.REGIONS:
        scenes = []
        if check_catalogue:
            try:
                from .live import find_scenes
                scenes = find_scenes(days=14, aoi=C.REGIONS[region])
                scenes = list({x["start"][:16]: x for x in scenes}.values())
            except Exception:
                scenes = []
        summaries.append(region_page(region, passes, dets, live_db, os.path.join(out, region), scenes))

    if demo_dir and os.path.exists(os.path.join(demo_dir, "index.html")):
        dst = os.path.join(out, "demo")
        shutil.copytree(demo_dir, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.sqlite"))

    log_path = os.path.join(out, "runlog.json")
    try:
        log = json.load(open(log_path))
    except Exception:
        log = []
    log.append(dict(t=time.time(), **{s["region"]: dict(live=s["live"], msgs_hour=s["st"].get("messages_last_hour", 0)) for s in summaries}))
    log = log[-500:]
    json.dump(log, open(log_path, "w"))

    cards = "".join(f"""<div class="card"><h2 style="margin-top:0"><a href="{s['region']}/index.html">{TITLES[s['region']]}</a></h2>
<div>{s['passes']} satellite passes processed: {s['matched']} ships matched to AIS, {s['cands']} not publicly tracked</div>
<div>{s['live']} ships broadcasting AIS in the last 30 min</div></div>""" for s in summaries)
    runs = "".join(f"<tr><td>{utc(r['t'])}</td>" + "".join(f"<td>{r.get(k, {}).get('live', '')}</td><td>{r.get(k, {}).get('msgs_hour', '')}</td>"
                                                         for k in C.REGIONS) + "</tr>" for r in log[::-1][:24])
    heads = "".join(f"<th>{k}: ships live</th><th>{k}: msgs/h</th>" for k in C.REGIONS)
    body = f"""<h1>Dark vessel detection</h1>
<p>Radar satellites (Sentinel-1) see ships whether or not they announce themselves. AIS is what ships broadcast. This system detects
ships in radar images with a CFAR detector, projects every AIS ship to the exact image time, pairs them with the Hungarian algorithm,
and lists the ships that radar saw but AIS did not: candidates for review, never accusations.</p>
<div class="note"><b>How to read a flag.</b> "Not publicly tracked" is not "switched off". Small boats often carry no public AIS, free AIS
comes from coastal receivers with limited range, and ships under ~40 m are not reliably visible. Low-confidence flags are those the
data cannot support (beyond receiver range, AIS feed gap, or no AIS recorded).</div>
<h2>Areas</h2><div class="cards">{cards}</div>
<h2>Automatic runs</h2><p class="muted">The site is rebuilt on a schedule from the live AIS recording. Last update: {utc(time.time(), '%Y-%m-%d %H:%M:%S UTC')}.</p>
<div class="tablewrap"><table><tr><th>run (UTC)</th>{heads}</tr>{runs}</table></div>
<p class="muted">Data: Copernicus Sentinel-1 (ESA), AIS via aisstream.io and the Danish Maritime Authority, maps and coastlines &copy; OpenStreetMap contributors.</p>"""
    path = os.path.join(out, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page("Dark vessel detection", body))
    return path
