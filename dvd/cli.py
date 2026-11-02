"""Command line: `python -m dvd demo | record-ais | find-scenes | export-yolo`."""
from __future__ import annotations

import argparse
import asyncio
import os

from . import config as C
from . import report, store
from .evaluate import classify_candidates, score_detections, truth_frame
from .infra import Structures
from .pipeline import run_pass
from .synth import make_ais, make_scene

# (seed, hours after first pass, sea_state, ais outage as (start offset s, end offset s) or None)
DEMO_PASSES = [(1, 0, 0.5, None), (2, 24 * 6, 1.0, None), (3, 24 * 12, 2.0, None), (4, 24 * 18, 1.0, (-900, 600))]
T_START = 1_780_000_000.0


def demo(out: str) -> None:
    os.makedirs(out, exist_ok=True)
    db = os.path.join(out, "dvd.sqlite")
    if os.path.exists(db):
        os.remove(db)
    con = store.connect(db)
    for seed, hours, sea, gap in DEMO_PASSES:
        t0 = T_START + hours * 3600
        scene = make_scene(seed, t0=t0, sea_state=sea)
        ais = make_ais(scene, seed, gap=(t0 + gap[0], t0 + gap[1]) if gap else None)
        store.add_ais(con, ais)
        if gap:
            store.add_gap(con, t0 + gap[0], t0 + gap[1])
        summary, det = run_pass(scene, con, f"S1-demo-{seed}", out_dir=out)

        truth = truth_frame(scene)
        s = score_detections(det[det.status != "filtered"], truth)
        c = classify_candidates(det, truth)
        print(f"\n{summary['pass_id']} sea_state={sea} ais_gap={'yes' if gap else 'no'}")
        print(f"  detection: precision={s['precision']:.2f} recall={s['recall']:.2f} (tp={s['tp']} fp={s['fp']} fn={s['fn']})")
        print(f"  recall by size: { {k: (round(v, 2) if v is not None else None) for k, v in s['recall_by_size'].items()} }")
        print(f"  ships: matched={summary['n_matched']} candidates={summary['n_candidates']} (low confidence {summary['n_low_conf']})")
        if len(c):
            print("  what the candidates really were:", c.groupby(["confidence", "truth"]).size().to_dict())
    print("\nsite:", report.build_site(con, out))


def region_aoi(a):
    return tuple(a.aoi) if getattr(a, "aoi", None) else C.ALL_AREAS[a.region]


def region_file(region: str, name: str) -> str | None:
    path = os.path.join("data", region, name)
    return path if os.path.exists(path) else None


def run_scene(a) -> None:
    """Process one real Sentinel-1 scene against the AIS database recorded by `record-ais`."""
    import pandas as pd
    from .sar import load_scene
    out = a.out or os.path.join("docs", a.region)    # crops go straight into the published site
    os.makedirs(out, exist_ok=True)
    aoi = region_aoi(a)
    land = a.land or region_file(a.region, "coast_osm.geojson")
    structures_path = a.structures or region_file(a.region, "infra_osm.geojson")
    print(f"region={a.region} aoi={aoi} land={land} structures={structures_path}")
    # config.FIXED_STRUCTURES holds demo placeholders, so real scenes use only the structures file
    structures = Structures.load(structures_path)
    scene = load_scene(a.tiff, aoi=aoi, calibration_xml=a.cal, land_geojson=land, time=a.time)
    con = store.connect(a.db)
    pass_id = a.pass_id or os.path.basename(a.tiff).split(".")[0][:40]
    summary, det = run_pass(scene, con, pass_id, out_dir=out, detector=a.detector, structures=structures, use_cnn=a.cnn)
    print({k: summary[k] for k in ("n_det", "n_matched", "n_candidates", "n_low_conf", "n_filtered", "ais_msgs", "ais_gap_s")})
    print("filtered:", det[det.status == "filtered"].reason.value_counts().to_dict())
    if summary["ais_msgs"] == 0:
        print("WARNING: no AIS in the +/-30 min window; every ship is flagged low confidence. Was the recorder running?")
    if a.reference:   # e.g. a Global Fishing Watch detections export with lon/lat columns
        ref = pd.read_csv(a.reference)
        ref = ref.assign(length_m=ref.get("length_m", 100.0))
        s = score_detections(det[det.status != "filtered"], ref, min_length_m=0)
        print(f"agreement with reference: precision={s['precision']:.2f} recall={s['recall']:.2f} (tp={s['tp']} fp={s['fp']} fn={s['fn']})")
    print(f"saved pass {pass_id} to {a.db}; run `python -m dvd build-site` to refresh the website")


def main() -> None:
    p = argparse.ArgumentParser(prog="dvd")
    sub = p.add_subparsers(dest="cmd", required=True)

    def area_args(sp, all_ok=False):
        choices = (list(C.REGIONS) + ["all"]) if all_ok else list(C.ALL_AREAS)   # training areas only where a single area is meant
        sp.add_argument("--region", choices=choices, default="all" if all_ok else "mumbai")
        sp.add_argument("--aoi", type=float, nargs=4, metavar=("WEST", "SOUTH", "EAST", "NORTH"), help="override the region's box")

    d = sub.add_parser("demo", help="run the full pipeline on synthetic passes and build the site")
    d.add_argument("--out", default="site")
    r = sub.add_parser("record-ais", help="record aisstream (needs a free API key in AISSTREAM_KEY)")
    r.add_argument("--db", default="live.sqlite")
    area_args(r, all_ok=True)
    t = sub.add_parser("ais-status", help="check the AIS recording, optionally for a pass time")
    t.add_argument("--db", default="live.sqlite")
    t.add_argument("--at", help="pass time in UTC, e.g. 2026-11-05T01:02:38")
    area_args(t, all_ok=True)
    f = sub.add_parser("find-scenes", help="list recent Sentinel-1 GRD scenes over a region")
    f.add_argument("--days", type=int, default=7)
    area_args(f)
    y = sub.add_parser("export-yolo", help="write a YOLO-format training set from synthetic scenes")
    y.add_argument("--out", default="yolo_data")
    y.add_argument("--scenes", type=int, default=6)
    s = sub.add_parser("run-scene", help="run the pipeline on a real Sentinel-1 GRD measurement GeoTIFF")
    s.add_argument("tiff"); s.add_argument("--cal", help="calibration-*.xml from the SAFE annotation folder")
    area_args(s)
    s.add_argument("--land", help="land polygons GeoJSON (default data/<region>/coast_osm.geojson)")
    s.add_argument("--structures", help="fixed structures GeoJSON (default data/<region>/infra_osm.geojson)")
    s.add_argument("--db", default="live.sqlite", help="AIS is read from here and the pass is saved here")
    s.add_argument("--out", help="where image crops go (default docs/<region>)")
    b = sub.add_parser("build-site", help="rebuild the public website in docs/")
    b.add_argument("--out", default="docs"); b.add_argument("--live", default="live.sqlite")
    b.add_argument("--results", nargs="*", default=["mumbai_results.sqlite", "denmark_hist.sqlite", "live.sqlite"])
    b.add_argument("--demo", default="site", help="folder with the simulation demo (python -m dvd demo)")
    b.add_argument("--no-catalogue", action="store_true", help="skip the Copernicus catalogue check")
    s.add_argument("--time", type=float, help="epoch seconds; default parsed from file name")
    s.add_argument("--pass-id"); s.add_argument("--detector", default="cfar", choices=["cfar", "yolo"])
    s.add_argument("--cnn", action="store_true", help="score every CFAR target with ShipCNN (models/ship_cnn.pt)")
    s.add_argument("--reference", help="CSV with lon,lat[,length_m] of reference detections")
    k = sub.add_parser("fetch-coast", help="build a detailed land mask from OpenStreetMap coastlines")
    area_args(k)
    i = sub.add_parser("fetch-infra", help="download bridges, piers, platforms and buoys from OpenStreetMap")
    area_args(i)
    u = sub.add_parser("auto", help="process every new Sentinel-1 pass over the sectors (catalogue -> crop -> pipeline)")
    u.add_argument("--days", type=float, default=3, help="how far back to look in the catalogue")
    u.add_argument("--live", default="live.sqlite")
    u.add_argument("--dry-run", action="store_true", help="only print what would be processed")
    n = sub.add_parser("train-cnn", help="build the AIS-labelled dataset, train ShipCNN and evaluate on the Oresund")
    n.add_argument("--rebuild", action="store_true", help="recreate the chip dataset from the scenes")
    n.add_argument("--epochs", type=int, default=60)
    g = sub.add_parser("fetch-scene", help="download just the region's part of a Sentinel-1 scene from Planetary Computer")
    g.add_argument("item", help="scene id, e.g. S1D_IW_GRDH_1SDV_20261029T170847_20261029T170912_005233_009F0B")
    g.add_argument("--out", help="default scenes/<YYYYMMDD>")
    area_args(g)
    m = sub.add_parser("import-dma", help="import Danish Maritime Authority historical AIS around a pass time")
    m.add_argument("--at", required=True, help="pass time in UTC, e.g. 2026-10-29T17:08:47")
    m.add_argument("--zip", help="local aisdk-YYYY-MM-DD.zip (downloaded if missing)")
    m.add_argument("--db", default="denmark_hist.sqlite")
    area_args(m)
    a = p.parse_args()

    if a.cmd == "auto":
        from .auto import sweep
        done = sweep(a.live, days=a.days, dry_run=a.dry_run)
        print(f"{len(done)} new pass(es) processed" if not a.dry_run else "dry run: nothing processed")
    elif a.cmd == "train-cnn":
        from .train_cnn import main as train_main
        train_main(rebuild=a.rebuild, epochs=a.epochs)
    elif a.cmd == "fetch-scene":
        from .pc import fetch_scene
        out = a.out or os.path.join("scenes", a.item.split("_")[4][:8])
        tif, xml = fetch_scene(a.item, region_aoi(a), out)
        print(f"image crop: {tif} ({os.path.getsize(tif) / 1e6:.1f} MB)\ncalibration: {xml}")
    elif a.cmd == "import-dma":
        from datetime import datetime, timezone
        from .dma import day_url, import_window
        from .live import recording_box
        at = datetime.fromisoformat(a.at).replace(tzinfo=timezone.utc)
        zpath = a.zip or os.path.join("data", "dma", f"aisdk-{at:%Y-%m-%d}.zip")
        if not os.path.exists(zpath):
            import requests
            os.makedirs(os.path.dirname(zpath), exist_ok=True)
            print("downloading", day_url(f"{at:%Y-%m-%d}"), "->", zpath, flush=True)
            with requests.get(day_url(f"{at:%Y-%m-%d}"), stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(zpath + ".part", "wb") as fh:
                    for block in r.iter_content(1 << 20):
                        fh.write(block)
            os.replace(zpath + ".part", zpath)
        t = at.timestamp()
        n = import_window(zpath, a.db, recording_box(region_aoi(a)), t - 3600, t + 3600)
        print(f"imported {n} AIS messages within 1 h of {a.at} into {a.db}")
    elif a.cmd in ("fetch-coast", "fetch-infra"):
        aoi, folder = region_aoi(a), os.path.join("data", a.region)
        os.makedirs(folder, exist_ok=True)
        if a.cmd == "fetch-coast":
            from .coast import fetch_coastline, write_land_geojson
            raw, out = os.path.join(folder, "coast_osm_raw.json"), os.path.join(folder, "coast_osm.geojson")
            print(fetch_coastline(aoi, raw), "coastline ways ->", raw)
            print(write_land_geojson(raw, out, aoi), "land polygons ->", out)
        else:
            from .infra import fetch_infrastructure, to_geojson
            raw, out = os.path.join(folder, "infra_osm_raw.json"), os.path.join(folder, "infra_osm.geojson")
            print(fetch_infrastructure(aoi, raw), "OSM elements ->", raw)
            print(to_geojson(raw, out), "->", out)
    elif a.cmd == "build-site":
        from .site import build
        print("site:", build(a.out, a.live, a.results, demo_dir=a.demo, check_catalogue=not a.no_catalogue))
    elif a.cmd == "run-scene":
        run_scene(a)
    elif a.cmd == "demo":
        demo(a.out)
    elif a.cmd == "record-ais":
        from .live import AuthError, record_ais
        key = os.environ.get("AISSTREAM_KEY", "").strip()   # from the environment so the key never appears in shell history
        if not key:
            raise SystemExit("Set the AISSTREAM_KEY environment variable to your aisstream.io API key first.")
        if a.aoi:
            aois = [tuple(a.aoi)]
        else:
            aois = list(C.REGIONS.values()) if a.region == "all" else [C.REGIONS[a.region]]
        print("recording areas:", aois, flush=True)
        try:
            asyncio.run(record_ais(key, a.db, aois=aois))
        except KeyboardInterrupt:
            print("stopped; buffered messages were saved")
        except AuthError:
            raise SystemExit(1)
    elif a.cmd == "ais-status":
        from datetime import datetime, timezone
        from .live import ais_status
        at = datetime.fromisoformat(a.at).replace(tzinfo=timezone.utc).timestamp() if a.at else None
        con = store.connect(a.db)
        for reg in ([a.region] if a.region != "all" else list(C.REGIONS)):
            st = ais_status(con, at=at, aoi=tuple(a.aoi) if a.aoi else C.REGIONS[reg])
            print(f"[{reg}]")
            for k, v in st.items():
                if k in ("first", "last") and v:
                    v = datetime.fromtimestamp(v, timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
                print(f"  {k:22s} {round(v, 1) if isinstance(v, float) else v}")
    elif a.cmd == "find-scenes":
        from .live import find_scenes
        for s in find_scenes(a.days, aoi=region_aoi(a)):
            print(s["start"], s["name"])
    elif a.cmd == "export-yolo":
        from .yolo import export_dataset
        scenes = [make_scene(100 + i) for i in range(a.scenes)]
        n = export_dataset(scenes[:-1], a.out, "train") + export_dataset(scenes[-1:], a.out, "val")
        print(f"wrote {n} tiles to {a.out}")
