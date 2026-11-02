"""Static site: Leaflet map of the latest pass, pass log, and monitoring chart. Rebuilt after every pass."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config as C

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dark vessel candidates - Mumbai-Gujarat coast</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
body{font:15px/1.5 system-ui,sans-serif;margin:0;color:#1d2330;background:#f6f7f9}
header,section{max-width:1000px;margin:0 auto;padding:16px}
h1{font-size:22px;margin:8px 0}#map{height:520px;border-radius:8px;border:1px solid #cfd4dc}
table{border-collapse:collapse;width:100%;background:#fff}td,th{padding:6px 10px;border-bottom:1px solid #e3e6ea;text-align:right}
th:first-child,td:first-child{text-align:left}.note{background:#fff7e0;border:1px solid #ecd48a;border-radius:8px;padding:10px 14px}
.key span{display:inline-block;width:11px;height:11px;border-radius:50%;margin:0 4px 0 12px;vertical-align:middle}
img{max-width:100%}
</style></head><body>
<header><h1>Ships seen by radar that were not publicly tracked</h1>
<p>Latest pass: <b>__PASS__</b>. Candidates are for human review, not verdicts or accusations.</p>
<div class="note"><b>Read before trusting a flag.</b> "Not tracked" is not "switched off". Free AIS comes from coastal receivers that
reach roughly __RX__ km offshore, many boats carry no public AIS at all, and ships under ~40 m are not reliably seen.
Orange = low confidence (beyond receiver range, or the AIS feed had a gap). Demo data is synthetic.</div></header>
<section><div class="key"><span style="background:#2e7d32"></span>matched to AIS<span style="background:#d32f2f"></span>candidate (high confidence)<span style="background:#ef9a2f"></span>candidate (low confidence)</div>
<div id="map"></div></section>
<section><h2>Pass log</h2>__TABLE__<h2>Monitoring</h2><img src="monitoring.png" alt="monitoring chart"></section>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const D=__DATA__;const m=L.map('map').fitBounds([[__S__,__W__],[__N__,__E__]]);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:'&copy; OpenStreetMap'}).addTo(m);
D.forEach(d=>{const c=d.status=='matched'?'#2e7d32':(d.confidence=='low'?'#ef9a2f':'#d32f2f');
let h='<b>'+d.status+'</b><br>length ~'+d.length_m.toFixed(0)+' m, score '+d.score.toFixed(2)+'<br>'+d.shore_km.toFixed(1)+' km from shore';
if(d.reason)h+='<br>caveat: '+d.reason.replaceAll('_',' ');if(d.crop)h+='<br><img src="'+d.crop+'" width="160">';
L.circleMarker([d.lat,d.lon],{radius:d.status=='matched'?5:8,color:c,fillColor:c,fillOpacity:.7}).bindPopup(h).addTo(m);});
</script></body></html>"""


def monitoring_chart(passes: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(11, 2.8))
    x = [datetime.fromtimestamp(t, timezone.utc).strftime("%m-%d") for t in passes["time"]]
    ax[0].bar(x, passes["n_det"], color="#5c6bc0"); ax[0].set_title("detections per pass")
    ax[1].plot(x, passes["match_rate"], "o-", color="#2e7d32"); ax[1].set_ylim(0, 1); ax[1].set_title("match rate")
    ax[2].plot(x, passes["mean_score"], "o-", color="#8e24aa"); ax[2].set_ylim(0, 1); ax[2].set_title("mean detection score")
    for a in ax:
        a.tick_params(labelsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def build_site(con, out_dir: str, aoi=C.AOI) -> str:
    os.makedirs(out_dir, exist_ok=True)
    passes = pd.read_sql("SELECT * FROM passes ORDER BY time", con)
    last = passes.iloc[-1]
    det = pd.read_sql("SELECT * FROM detections WHERE pass_id=? AND status!='filtered'", con, params=(last.pass_id,))
    monitoring_chart(passes, os.path.join(out_dir, "monitoring.png"))

    shown = passes.assign(time=[datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M") for t in passes["time"]],
                          match_rate=passes["match_rate"].round(2), mean_score=passes["mean_score"].round(2))
    shown = shown[["pass_id", "time", "n_det", "n_matched", "n_candidates", "n_low_conf", "match_rate", "ais_msgs", "ais_gap_s"]]
    shown.columns = ["pass", "time (UTC)", "detections", "matched", "candidates", "low conf.", "match rate", "AIS msgs", "feed gap s"]
    w, s, e, n = aoi
    html = (PAGE.replace("__PASS__", f"{last.pass_id}").replace("__RX__", f"{C.RX_RANGE_KM:.0f}")
            .replace("__TABLE__", shown.to_html(index=False, border=0))
            .replace("__DATA__", json.dumps(det[["lon", "lat", "length_m", "score", "shore_km", "status", "reason", "confidence", "crop"]]
                                            .to_dict("records")))
            .replace("__S__", str(s)).replace("__W__", str(w)).replace("__N__", str(n)).replace("__E__", str(e)))
    path = os.path.join(out_dir, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path
