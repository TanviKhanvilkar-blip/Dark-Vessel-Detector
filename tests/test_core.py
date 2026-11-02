import json

import numpy as np
import pandas as pd

from dvd import config as C
from dvd import store
from dvd.detect import detect_scene, tiles
from dvd.evaluate import score_detections, truth_frame
from dvd.geo import Grid, destination, haversine_m
from dvd.live import parse_position_report
from dvd.match import match, project_ais
from dvd.pipeline import run_pass
from dvd.synth import make_ais, make_scene


def test_destination_roundtrip():
    lon, lat = destination(72.5, 19.0, 90, 1000)
    assert abs(haversine_m(72.5, 19.0, lon, lat) - 1000) < 1
    lon2, lat2 = destination(lon, lat, 90, -1000)
    assert haversine_m(72.5, 19.0, lon2, lat2) < 1


def test_grid_roundtrip():
    g = Grid.from_aoi(C.AOI, C.RES_DEG)
    lon, lat = g.lonlat(100.0, 200.0)
    r, c = g.rowcol(lon, lat)
    assert abs(r - 100) < 1e-6 and abs(c - 200) < 1e-6


def test_tiles_cover_scene_exactly_once():
    seen = np.zeros((1300, 1700), int)
    for _, _, (r0, r1, c0, c1) in tiles(seen.shape):
        seen[r0:r1, c0:c1] += 1
    assert (seen == 1).all()


def test_cfar_finds_planted_ships_and_ignores_empty_sea():
    sc = make_scene(7, n_ships=10)
    det = detect_scene(sc.image, sc.land, sc.grid)
    # raw CFAR also fires on the platforms; the pipeline filters those, so drop them for this check
    near = np.zeros(len(det), bool)
    for lo, la in C.FIXED_STRUCTURES:
        near |= haversine_m(det.lon.values, det.lat.values, lo, la) < C.STRUCTURE_RADIUS_M
    s = score_detections(det[~near], truth_frame(sc))
    assert s["recall"] >= 0.9 and s["precision"] >= 0.9
    clean = make_scene(8, n_ships=0)
    assert len(detect_scene(clean.image, clean.land, clean.grid)) <= 2  # only the fixed structures


def test_length_estimate_is_reasonable():
    sc = make_scene(9, n_ships=12)
    det = detect_scene(sc.image, sc.land, sc.grid)
    truth = truth_frame(sc)
    errs = []
    for _, d in det.iterrows():
        j = int(np.argmin(haversine_m(d.lon, d.lat, truth.lon.values, truth.lat.values)))
        if haversine_m(d.lon, d.lat, truth.lon[j], truth.lat[j]) < 100 and truth.length_m[j] > 60:
            errs.append(abs(d.length_m - truth.length_m[j]) / truth.length_m[j])
    assert errs and np.median(errs) < 0.25


def test_projection_moves_report_along_track():
    ais = pd.DataFrame([dict(mmsi=1, time=0.0, lon=72.5, lat=19.0, sog=10.0, cog=90.0)])
    p = project_ais(ais, t0=600.0)
    assert abs(haversine_m(72.5, 19.0, p.lon[0], p.lat[0]) - 10 * C.KNOT_MS * 600) < 5


def test_hungarian_prefers_global_optimum_and_respects_gate():
    det = pd.DataFrame(dict(lon=[72.5, 72.5003], lat=[19.0, 19.0], length_m=[80, 80]))
    proj = pd.DataFrame(dict(lon=[72.5001, 72.9], lat=[19.0, 19.0], sog=[0, 0], dt_s=[0, 0]))
    pairs = match(det, proj)
    assert len(pairs) == 1 and list(pairs.values()) == [0]   # the far ship is outside every gate


def test_dark_ships_become_candidates_and_gap_downgrades_confidence(tmp_path):
    for gap in (False, True):
        sc = make_scene(11, n_ships=20)
        con = store.connect(str(tmp_path / f"{gap}.sqlite"))
        g = (sc.time - 900, sc.time + 600) if gap else None
        store.add_ais(con, make_ais(sc, 11, gap=g))
        if gap:
            store.add_gap(con, *g)
        _, det = run_pass(sc, con, "t")
        cand = det[det.status == "candidate"]
        assert len(cand) > 0
        if gap:
            assert (cand.confidence == "low").all()
        # every AIS-broadcasting ship must be matched, not flagged
        truth = truth_frame(sc)
        for _, c in cand.iterrows():
            j = int(np.argmin(haversine_m(c.lon, c.lat, truth.lon.values, truth.lat.values)))
            if not gap and haversine_m(c.lon, c.lat, truth.lon[j], truth.lat[j]) < 150:
                assert truth.category[j] != "ais"


def test_fixed_structures_are_filtered(tmp_path):
    sc = make_scene(12, n_ships=3)
    con = store.connect(str(tmp_path / "s.sqlite"))
    _, det = run_pass(sc, con, "t")
    assert (det.reason == "fixed_structure").sum() == len(C.FIXED_STRUCTURES)


def test_parse_position_report():
    raw = json.dumps({"MessageType": "PositionReport", "MetaData": {"MMSI": 123, "time_utc": "2026-10-04 10:00:00.123 +0000 UTC"},
                      "Message": {"PositionReport": {"Latitude": 19.0, "Longitude": 72.6, "Sog": 8.2, "Cog": 45.0}}})
    row = parse_position_report(raw)
    assert row["mmsi"] == 123 and row["lat"] == 19.0 and row["time"] > 0
    assert parse_position_report(json.dumps({"MessageType": "StaticData"})) is None
