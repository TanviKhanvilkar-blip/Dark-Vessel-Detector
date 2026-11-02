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

Re-run with the final pipeline and ShipCNN (`--cnn`): ShipCNN rejects 38 targets as not ships, leaving **115 candidates, all low
confidence** (no AIS exists for that date; 9 are also beyond the assumed receiver range). Note that ShipCNN was trained only on
Danish waters, so how well it transfers to Mumbai is untested.

`dvd/sar.py` reads only the pixels covering the AOI, converts DN to sigma0 with the calibration XML, resamples onto the AOI grid,
masks land and the black swath border, and scales calm sea to ~1. Use VH if you can (better ships-vs-rough-sea contrast).
Tested on generated GeoTIFFs with both a plain geotransform and GCP georeferencing (the way GRD products are referenced), and the
calibration interpolation is unit-tested; **not yet run on an actual Copernicus file**. Expect to tune `CFAR_THRESHOLD` on real data.
Moving ships appear displaced along track in SAR (Doppler), which the matching gate does not yet model.

## Validation area: Oresund (Copenhagen), first matched pass

Mumbai gets ~1 pass every 12 days and almost no free live AIS, so the end-to-end match is demonstrated on the Oresund,
which Sentinel-1 covers fully on most passes and where the Danish Maritime Authority publishes full historical AIS.
(The Great Belt was tried first: on both available dates it sat on the swath edge, 21% and 72% covered.)

```
python -m dvd fetch-coast --region denmark && python -m dvd fetch-infra --region denmark
python -m dvd import-dma --region denmark --at 2026-10-27T05:23:38          # one AIS day, ~620 MB, cut to the area +/- 1 h
python -m dvd fetch-scene S1C_IW_GRDH_1SDV_20261027T052338_20261027T052403_010066_01417A --region denmark   # area crop only, no login
python -m dvd run-scene scenes/20261027/<...>-vh-aoi.tiff --cal scenes/20261027/calibration-<...>.xml --region denmark --db denmark_hist.sqlite
```

Result, 27 Oct 2026 05:23 UTC: 79 ships broadcasting AIS inside the area at image time (11 moving); 68 of them (86%) had a radar
detection within 300 m. After filtering: **26 ships matched to AIS, 4 candidates not publicly tracked** (none had any AIS report
within 0.9 km at image time; all four are compact targets on open water). Lessons from the real data, now built in:
- Structure and shore filters apply only to *unmatched* detections; before this, 62 moored ships at quays were thrown away.
- Moving ships are imaged displaced along the satellite track (~110 s x their velocity toward the satellite, ~500 m at 10 kn),
  so the match radius grows with speed. Stationary ships keep the tight radius.
- 29 broadcasting ships (mostly small harbour craft) gave detections below the score threshold: the detector's real limit.
- Offshore wind turbines (Middelgrunden) are now pulled from OpenStreetMap with the other fixed structures.

Second pass, 26 Oct 2026 05:32 UTC: 88 ships broadcasting inside the area, 75 (85%) with a radar detection within 300 m;
**24 matched, 3 candidates**. All three were within ~400 m of 27 Oct candidates: an untracked target at the same spot on
consecutive days is almost certainly fixed (unmapped structure, mooring, anchored hulk). Such candidates are now tagged
`same_spot_in_another_pass` and downgraded to low confidence (`REPEAT_RADIUS_M`). Across both passes this leaves **one
high-confidence candidate** (27 Oct, 55.669 N 12.670 E) out of 50 matched ships.

## The neural network: ShipCNN

A two-stage detector: CFAR proposes every bright target, then **ShipCNN**, our own convolutional network written from scratch in
PyTorch (`dvd/cnn.py`), looks at a 64 x 64 chip (640 m at 10 m/px, land blanked to sea level) and outputs P(ship).

| # | Layer | Output | Activation | Weights |
|---|---|---|---|---|
| 0 | Input: log10 backscatter / 2 | 1 x 64 x 64 | - | 0 |
| 1 | Conv 3x3, 16 filters + BatchNorm + MaxPool 2x2 | 16 x 32 x 32 | ReLU | 192 |
| 2 | Conv 3x3, 32 filters + BatchNorm + MaxPool 2x2 | 32 x 16 x 16 | ReLU | 4,704 |
| 3 | Conv 3x3, 64 filters + BatchNorm + MaxPool 2x2 | 64 x 8 x 8 | ReLU | 18,624 |
| 4 | Conv 3x3, 64 filters + BatchNorm | 64 x 8 x 8 | ReLU | 37,056 |
| 5 | Global average pool | 64 | - | 0 |
| 6 | Fully connected + Dropout 0.3 | 32 | ReLU | 2,080 |
| 7 | Output | 1 | Sigmoid | 33 |

62,689 weights. Adam (learning rate 3e-4, weight decay 1e-4), binary cross-entropy with the ship class weighted, batch 64, 60 epochs,
weights kept from the epoch with the lowest validation loss; augmentation: 90-degree rotations, flips, +/-3 px shifts.

**Labels come from AIS** (as in xView3): ship = radar target matched to a broadcasting ship; not ship = target on a mapped point
structure (bridge pylon, wind turbine, buoy, beacon, light, pile, rock) with no AIS ship within 300 m, or open sea; everything else is
excluded (it could be a boat without AIS). Piers and quays are excluded too, because boats moor there.

**Data:** trained on 7 Danish areas (Great Belt, Little Belt, Fehmarn Belt, Kattegat, Skagen, and the Rodsand and Anholt wind farms;
10 scene crops over 26, 27 and 29 Oct 2026); validation split by location; **tested on the Oresund, which training never sees**.

```
powershell -ExecutionPolicy Bypass -File scripts\fetch_training_data.ps1     # masks + scene crops (AIS: import-dma, see above)
python -m dvd train-cnn --rebuild                                              # dataset, training, test report -> models/, docs/nn/
python -m dvd run-scene ... --region denmark --db denmark_hist.sqlite --cnn    # pipeline with the network
```

Held-out Oresund (375 chips, 50 ships), operating threshold 0.69 chosen on validation as the highest that keeps >= 90% of ships:

| | ShipCNN | CFAR only |
|---|---|---|
| Accuracy | 0.827 | 0.765 |
| Precision | 0.430 | 0.362 |
| Recall | 0.920 | 1.000 (by construction) |
| F1 | 0.586 | 0.532 |
| ROC-AUC | 0.922 | 0.865 |

Where it matters, on the 138 targets CFAR accepts as ships: ShipCNN removes 63 of 88 non-ships and loses 4 of 50 real ships,
raising precision from 0.36 to 0.65. In the full pipeline it rejects 5 unmatched targets across the two passes; every AIS-matched ship
scores above the threshold (median P = 0.99).

How it got there (each step is in the git history): the first model scored F1 0.41 because "not ship" labels included piers where
boats moor (fixed: point structures only), it called 87% of wind turbines ships (fixed: added two wind farms to training, now 23%),
and it judged the city around harbour targets instead of the target (fixed: land blanked in chips; ROC-AUC 0.88 -> 0.92).
Remaining weakness: buoys and small navigation lights, and quay edges that the coastline does not count as land.

## Automatic pass processing

`python -m dvd auto` (run by `scripts/publish_site.ps1` every 15 minutes, before the site is rebuilt) asks Planetary Computer for
Sentinel-1 scenes over each sector from the last 3 days and processes a scene when it covers >= 65% of the sector (the uncovered
part is treated as no-data, so ships there are simply not assessed), was taken while
the AIS recorder was running, is at least 35 min old (AIS must exist up to 30 min after the image) and is not in the sector's
results database yet. Processing = download the sector crop, copy the AIS and feed gaps around the image time from `live.sqlite`
into the sector's results database (`mumbai_results.sqlite`, `denmark_hist.sqlite`), run the pipeline with ShipCNN.
`python -m dvd auto --dry-run` prints the decision for every scene without processing anything.

Rehearsed end to end on 26 and 27 Oct (historical Danish AIS standing in for the recording): 25 matched / 0 candidates and
27 matched / 2 low-confidence, one more match each than the manual runs because the automatic path uses the catalogue's exact
image-centre time rather than the slice start time in the file name; a second sweep skipped both as already processed.
A failed download is logged and retried next sweep instead of stopping it (a storage timeout in the first rehearsal exposed this).
Next passes covering the sectors (predicted from the 6-day repeat): Mumbai ~5 Nov 01:02 UTC, Oresund ~5 Nov 17:00 UTC.

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
