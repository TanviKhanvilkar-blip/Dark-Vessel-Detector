import json

from dvd import config as C
from dvd.infra import Structures, to_geojson


def test_near_uses_point_and_line_buffers():
    s = Structures.load(None, [(72.80, 19.00)])
    deg = 1 / (111_320 * 0.9455)  # one metre east-west at 19 N
    near = s.near([72.80 + 100 * deg, 72.80 + 400 * deg], [19.0, 19.0])
    assert list(near) == [True, False]          # 100 m inside the 150 m point buffer, 400 m outside


def test_line_structure(tmp_path):
    p = tmp_path / "s.geojson"
    p.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"kind": "bridge"},
         "geometry": {"type": "LineString", "coordinates": [[72.80, 18.95], [72.80, 19.05]]}}]}))
    s = Structures.load(str(p))
    deg = 1 / (111_320 * 0.9455)
    near = s.near([72.80 + 50 * deg, 72.80 + (C.STRUCTURE_LINE_BUFFER_M + 60) * deg], [19.0, 19.0])
    assert list(near) == [True, False]


def test_virtual_seamarks_are_dropped(tmp_path):
    raw = {"elements": [
        {"type": "node", "lon": 72.8, "lat": 19.0, "tags": {"seamark:type": "buoy_lateral"}},
        {"type": "way", "tags": {"seamark:type": "separation_lane"}, "geometry": [{"lon": 72.7, "lat": 19}, {"lon": 72.9, "lat": 19}]},
        {"type": "way", "tags": {"seamark:type": "cable_submarine"}, "geometry": [{"lon": 72.7, "lat": 19}, {"lon": 72.9, "lat": 19}]},
        {"type": "way", "tags": {"highway": "primary", "bridge": "yes"}, "geometry": [{"lon": 72.7, "lat": 19}, {"lon": 72.9, "lat": 19}]},
    ]}
    src, out = tmp_path / "raw.json", tmp_path / "out.geojson"
    src.write_text(json.dumps(raw))
    assert to_geojson(str(src), str(out)) == {"buoy_lateral": 1, "bridge": 1}


def test_no_structures_flags_nothing():
    assert not Structures.load(None).near([72.8], [19.0]).any()
