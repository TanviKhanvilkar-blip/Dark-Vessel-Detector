"""Automatic pass processing: watch the satellite catalogue and run the full pipeline on every new pass.

One sweep (`python -m dvd auto`, run by scripts/publish_site.ps1 every 15 minutes):
  1. ask Planetary Computer for Sentinel-1 GRD scenes over each sector in the last few days
  2. keep scenes that cover most of the sector, that were taken while the AIS recorder was running, that are
     old enough for AIS to exist up to 30 min after the image, and that have not been processed yet
  3. for each: download the sector crop, copy the AIS (and logged feed gaps) around the image time from the live
     recording into the sector's results database, and run the pipeline with ShipCNN
The results database is the same one the sector's earlier passes live in, so the repeat-spot check sees them all.
"""
from __future__ import annotations

import os
import sqlite3
import time
from datetime import datetime

import pandas as pd
import requests
from shapely.geometry import box, shape

from . import config as C
from . import store
from .live import recording_box

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
RESULT_DBS = {"mumbai": "mumbai_results.sqlite", "denmark": "denmark_hist.sqlite"}
MIN_COVERAGE = 0.65         # fraction of the sector inside the scene footprint; the uncovered part is treated as no-data
SETTLE_S = 35 * 60          # AIS must exist up to 30 min after the image before matching makes sense


def search(aoi, days: float) -> list[dict]:
    since = datetime.utcfromtimestamp(time.time() - days * 86400).strftime("%Y-%m-%dT%H:%M:%SZ")
    r = requests.post(STAC, json={"collections": ["sentinel-1-grd"], "bbox": list(aoi), "datetime": f"{since}/..",
                                  "limit": 50}, timeout=60)
    r.raise_for_status()
    return r.json().get("features", [])


def item_time(item: dict) -> float:
    return pd.Timestamp(item["properties"]["datetime"]).timestamp()


def select_passes(items: list[dict], aoi, processed: set[str], recording_start: float | None, now: float) -> list[tuple[dict, str]]:
    """Decide what to do with each catalogue item. Returns [(item, decision)], decision 'process' or a reason to skip."""
    sector = box(*aoi)
    out = []
    for it in sorted(items, key=item_time):
        t = item_time(it)
        cov = shape(it["geometry"]).intersection(sector).area / sector.area
        if it["id"] in processed:
            d = "already processed"
        elif cov < MIN_COVERAGE:
            d = f"covers only {cov:.0%} of the sector"
        elif recording_start is None or t - C.MATCH_WINDOW_S < recording_start:
            d = "taken before the AIS recorder was running"
        elif now - t < SETTLE_S:
            d = "too recent: waiting for AIS up to 30 min after the image"
        else:
            d = "process"
        out.append((it, d))
    return out


def processed_ids(db: str) -> set[str]:
    if not os.path.exists(db):
        return set()
    con = sqlite3.connect(db)
    try:
        return {r[0] for r in con.execute("SELECT pass_id FROM passes")}
    except sqlite3.OperationalError:
        return set()
    finally:
        con.close()


def copy_ais(live_db: str, results_db: str, aoi, t0: float) -> int:
    """AIS and feed gaps around t0 from the live recording into the results database (the pipeline reads both there)."""
    window = C.MATCH_WINDOW_S * 2
    src = store.connect(live_db)
    try:
        ais = store.ais_window(src, t0, window, recording_box(aoi))
        gaps = src.execute("SELECT start, end FROM ais_gaps WHERE end > ? AND start < ?", (t0 - window, t0 + window)).fetchall()
        ships = pd.read_sql("SELECT * FROM ships", src)
    finally:
        src.close()
    dst = store.connect(results_db)
    if results_db != live_db:       # replace any earlier copy of this window, so a retried pass never duplicates AIS
        w, s, e, n = recording_box(aoi)
        dst.execute("DELETE FROM ais WHERE time BETWEEN ? AND ? AND lon BETWEEN ? AND ? AND lat BETWEEN ? AND ?",
                    (t0 - window, t0 + window, w, e, s, n))
        dst.execute("DELETE FROM ais_gaps WHERE end > ? AND start < ?", (t0 - window, t0 + window))
        dst.commit()
    store.add_ais(dst, ais)
    store.add_ships(dst, ships)
    for s, e in gaps:
        store.add_gap(dst, s, e)
    dst.close()
    return len(ais)


def _retry(fn, log, attempts: int = 3, wait_s: float = 20):
    for k in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if k == attempts:
                raise
            log(f"   download attempt {k} failed ({type(exc).__name__}); retrying in {wait_s:.0f}s")
            time.sleep(wait_s)


def recording_start(live_db: str, aoi) -> float | None:
    if not os.path.exists(live_db):
        return None
    con = sqlite3.connect(live_db, timeout=30)
    try:
        w, s, e, n = recording_box(aoi)
        return con.execute("SELECT MIN(time) FROM ais WHERE lon BETWEEN ? AND ? AND lat BETWEEN ? AND ?", (w, e, s, n)).fetchone()[0]
    finally:
        con.close()


def sweep(live_db: str = "live.sqlite", days: float = 3, dry_run: bool = False, log=print,
          result_dbs: dict | None = None, out_root: str = "docs", scenes_root: str = os.path.join("scenes", "auto")) -> list[str]:
    """One pass over the catalogue for every sector. Returns the pass ids processed.
    `result_dbs`, `out_root` and `scenes_root` can point elsewhere to rehearse the whole path without touching the site."""
    result_dbs = result_dbs or RESULT_DBS
    from .infra import Structures
    from .pc import fetch_scene
    from .pipeline import run_pass
    from .sar import load_scene

    done = []
    for region, aoi in C.REGIONS.items():
        db = result_dbs[region]
        try:
            items = search(aoi, days)
        except Exception as exc:                 # catalogue down: try again next sweep
            log(f"[{region}] catalogue unavailable ({exc!r})")
            continue
        start = recording_start(live_db, aoi)
        if start is None and region == "mumbai":
            # free AIS never reaches Mumbai; process anyway once the recorder has been running at all, so the
            # radar result still appears (every flag will honestly be low confidence)
            start = recording_start(live_db, (-180, -90, 180, 90))
        for it, decision in select_passes(items, aoi, processed_ids(db), start, time.time()):
            log(f"[{region}] {it['id']}: {decision}")
            if decision != "process" or dry_run:
                continue
            t0 = item_time(it)
            parts = it["id"].split("_")
            folder = os.path.join(scenes_root, f"{region}_{parts[0]}_{parts[4]}")   # short: Windows path limit
            try:
                # one failed pass (e.g. a storage timeout) must not stop the sweep; it is not marked as processed,
                # so the next sweep simply tries it again
                tif, xml = _retry(lambda: fetch_scene(it["id"], aoi, folder), log)
                n_ais = copy_ais(live_db, db, aoi, t0)
                land = os.path.join("data", region, "coast_osm.geojson")
                scene = load_scene(tif, aoi=aoi, calibration_xml=xml, land_geojson=land if os.path.exists(land) else None, time=t0)
                con = store.connect(db)
                summary, _ = run_pass(scene, con, it["id"], out_dir=os.path.join(out_root, region),
                                      structures=Structures.load(os.path.join("data", region, "infra_osm.geojson")), use_cnn=True)
                con.close()
            except Exception as exc:
                log(f"[{region}] {it['id']}: FAILED ({exc!r}); will retry next sweep")
                continue
            log(f"[{region}] processed {it['id']}: {summary['n_matched']} matched, {summary['n_candidates']} candidates "
                f"({summary['n_low_conf']} low confidence), {n_ais} AIS messages around the image")
            done.append(it["id"])
    return done
