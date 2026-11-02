# Dark Vessel Detection (Mumbai–Gujarat coast)

Radar satellites (Sentinel-1) see ships whether or not they broadcast. AIS shows ships that do. A ship seen by radar with
no matching AIS is a **"not publicly tracked" candidate**: for human review, never an accusation.

```
python -m dvd demo            # full pipeline on 4 synthetic passes -> site/index.html (+ metrics in the terminal)
python -m pytest              # 10 tests
python -m dvd find-scenes     # recent Sentinel-1 GRD scenes over the AOI (needs network)
python -m dvd record-ais --key YOUR_AISSTREAM_KEY   # always-on AIS recorder (free key at aisstream.io)
python -m dvd export-yolo     # YOLO-format training tiles (then: pip install ultralytics; yolo detect train data=yolo_data/data.yaml model=yolov8n.pt imgsz=640)
```

## Running on a real Sentinel-1 scene

1. Register (free) at Copernicus Data Space; `python -m dvd find-scenes` lists recent GRD scenes over the AOI. Download one and unzip the `.SAFE`.
2. Get land polygons for the AOI as GeoJSON (Natural Earth or OSM coastline; any polygons work).
3. Run, with the AIS recorder having covered the pass time:

```
python -m dvd run-scene path/to/measurement/s1c-iw-grd-vv-20260104t013200-....tiff \
    --cal path/to/annotation/calibration/calibration-s1c-iw-grd-vv-....xml \
    --land coast.geojson --db live.sqlite --out site [--reference gfw_detections.csv]
```

Add `--aoi WEST SOUTH EAST NORTH` to process a different area than `config.AOI` (e.g. `--aoi 72.55 18.80 72.95 19.10` to include the Mumbai
coast). `run-scene` ignores the placeholder `FIXED_STRUCTURES`; put real coordinates in `config.py` before relying on that filter.
Land masks: `data/coast.geojson` is Natural Earth 10 m land, coarse (kilometre-scale errors around Mumbai's creeks and harbour).
Prefer `data/coast_osm.geojson`, built from OpenStreetMap coastlines with
`python -m dvd fetch-coast --aoi 72.55 18.80 72.95 19.10` (Overpass API; it is often busy, just retry on a 502).

Fixed structures: `python -m dvd fetch-infra --aoi …` pulls bridges, piers, breakwaters, platforms, buoys, beacons and rocks
from OpenStreetMap into `data/infra_osm.geojson`; pass it with `--structures`. Detections within 150 m of a point or 120 m of a
line (`config.py`) are filtered as `fixed_structure`. Chart-only seamarks (traffic lanes, submarine cables, anchorage areas)
are dropped, because masking them would hide real ships.

Georeferencing: GRD products carry ~250 control points over a 250 km swath. GDAL's default single polynomial fit put this scene
~250 m west of OpenStreetMap; thin-plate-spline warping (now the default, `gcp_method="tps"`) cuts the measured offset to ~140 m,
which is within the noise of the measurement (radar-land vs OSM-land cross-correlation).

First real result (24 Oct 2026 scene, VH, OSM land + structures, TPS): 684 raw detections, 153 candidates, all low confidence
because no AIS was recorded for that date. 174 detections were removed as fixed structures (mainly Atal Setu and the
Bandra–Worli/coastal-road bridges, plus piers). Offshore ships and the Mumbai harbour anchorage look right by eye; 9 candidates
still sit within 300 m of a mapped structure and are worth inspecting.

`dvd/sar.py` reads only the pixels covering the AOI, converts DN to sigma0 with the calibration XML, resamples onto the AOI grid,
masks land and the black swath border, and scales calm sea to ~1. Use VH if you can (better ships-vs-rough-sea contrast).
Tested on generated GeoTIFFs with both a plain geotransform and GCP georeferencing (the way GRD products are referenced), and the
calibration interpolation is unit-tested; **not yet run on an actual Copernicus file**. Expect to tune `CFAR_THRESHOLD` on real data.
Moving ships appear displaced along track in SAR (Doppler), which the matching gate does not yet model.

## Recording AIS for a pass

aisstream has no history, so AIS must be recorded *while* the satellite passes.

1. Get a free API key at aisstream.io (sign in, then create a key). Keep it out of chats and files.
2. Start the recorder and leave it running (it asks for the key once per session):
   `powershell -ExecutionPolicy Bypass -File scripts\record_ais.ps1`
   It records the Mumbai area plus a ~16 km margin (`config.LIVE_AOI`, `AIS_PAD_DEG`) into `live.sqlite`, reconnects on drops,
   reconnects if the feed goes silent for 3 min, and logs every silence over 60 s (including recorder downtime and the PC
   sleeping) to `ais_gaps`. Flags overlapping a gap longer than 5 min are downgraded to low confidence.
3. Keep the PC awake for at least 00:30–01:35 UTC on the pass day.
4. Check coverage: `python -m dvd ais-status --at 2026-11-05T01:02:38` (`pass_ready True` = recording covers ±30 min with no long gap).
5. When the scene appears (`python -m dvd find-scenes`), download it and run `run-scene ... --db live.sqlite`.

Sentinel-1 schedule seen so far for this area: Sentinel-1D only, 01:02 UTC (06:32 IST), every 12 days (12 and 24 Oct 2026).
The next pass is therefore expected on **5 Nov 2026 ~01:02 UTC**; this is a prediction from that pattern, not an official plan.

## Status against the build plan

| Phase | Status |
|---|---|
| 1. Detector | CFAR baseline done and tested on synthetic scenes. YOLO: dataset export + inference adapter written, **not trained or run** (needs `ultralytics` and real/large data). |
| 2. Matching | Done: dead-reckoning to image time, Hungarian assignment, speed- and size-aware gate. Tested. |
| 3. Live | AIS recorder with reconnect + gap logging, and Copernicus scene search written, **untested against the live services** (need keys/network). Fixed-structure coordinates are placeholders. |
| 4. Site | Done as a static site rebuilt after every pass: map, pass log, monitoring chart. Scheduling not wired (see next steps). |

## How it works

1. **Scene** (`synth.py` now, GeoTIFF later): a north-up intensity array + a `Grid` that maps pixels to lon/lat.
2. **Detect** (`detect.py`): the scene is cut into 512-px tiles with a 64-px halo. In each tile, CA-CFAR flags pixels more than
   5.5× the local sea background (ring around the pixel, guard window excluded, land excluded). Blobs are cleaned, labelled, and
   turned into lon/lat, an estimated length (PCA of the blob: a rectangle of length L has variance L²/12) and a score.
   A detection is kept only by the tile whose core contains its centre, so nothing is double-counted.
3. **Align** (`match.py`): from AIS within ±30 min, take each ship's report nearest the image time and move it along its
   speed and heading to that exact time.
4. **Match** (`match.py`): cost = distance ÷ gate, gate = 100 m + half the ship length + 15% of the distance the ship could have
   travelled in the time gap. The Hungarian algorithm (`linear_sum_assignment`) finds the best one-to-one pairing; pairs outside
   the gate are rejected.
5. **Filter & label** (`pipeline.py`): drop fixed structures, low scores, ships under 40 m and anything within 0.5 km of shore.
   Unmatched ships become candidates, tagged **low confidence** if they are beyond free-receiver range (18 km, approximate),
   if the AIS feed had an outage in the window, or if there was no AIS at all.
6. **Store & publish** (`store.py`, `report.py`): everything goes to SQLite; the site shows the latest pass plus the history.
7. **Evaluate** (`evaluate.py`): precision/recall, recall by ship size, and what the candidates *really* were.

## Read this before believing any number

- The demo uses **synthetic** scenes. Real SAR has ghosts, wake, rough-sea clutter, azimuth displacement of moving ships and
  mixed ship sizes. Perfect-looking recall here (≈1.0) says the code is correct, not that ships are easy to find. The real test is
  phase 2/3 on real scenes against Global Fishing Watch detections.
- In the synthetic world I *know* which unmatched ships are truly transponder-less and which are merely out of receiver range.
  The demo prints that split: even with "low confidence" tagging, some out-of-range ships are flagged high confidence, because
  the pipeline's shore distance (nearest land pixel) differs from the generator's (east–west) near the 18 km boundary.
  That is the project's central honesty problem in miniature.
- Site map uses OpenStreetMap tiles and Leaflet from public CDNs, so it needs internet when opened.
- Receiver range, thresholds and platform coordinates are all in `dvd/config.py` and are placeholders until calibrated on real data.

## Next steps

1. Register at Copernicus, download one GRD scene, read it with `rasterio`, calibrate to sigma0, and swap it in for `make_scene`
   (the pipeline only needs an intensity array, a land mask and a `Grid`).
2. Replace placeholder platform coordinates; build the land mask from a coastline file (GeoPandas).
3. Run the AIS recorder continuously on a small server; schedule `find-scenes` → pipeline (cron / GitHub Actions) for each pass.
4. Train YOLO on SSDD/xView3 tiles and compare with CFAR; validate against Global Fishing Watch.
