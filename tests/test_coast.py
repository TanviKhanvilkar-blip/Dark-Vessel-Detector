import json

from shapely.geometry import Point

from dvd.coast import land_polygons


def _raw(tmp_path, *ways):
    els = [{"type": "way", "geometry": [{"lon": x, "lat": y} for x, y in w]} for w in ways]
    p = tmp_path / "raw.json"
    p.write_text(json.dumps({"elements": els}))
    return str(p)


def test_land_is_left_of_coastline(tmp_path):
    # a coastline running north along x=0.5: OSM puts land on the left, i.e. the west half
    polys = land_polygons(_raw(tmp_path, [(0.5, -0.1), (0.5, 1.1)]), (0, 0, 1, 1))
    assert len(polys) == 1
    assert polys[0].contains(Point(0.25, 0.5)) and not polys[0].contains(Point(0.75, 0.5))
    assert abs(polys[0].area - 0.5) < 1e-9


def test_reversed_way_flips_land_side(tmp_path):
    polys = land_polygons(_raw(tmp_path, [(0.5, 1.1), (0.5, -0.1)]), (0, 0, 1, 1))
    assert len(polys) == 1 and polys[0].contains(Point(0.75, 0.5))


def test_island_inside_aoi(tmp_path):
    # closed counter-clockwise ring = island (land on the left = inside)
    ring = [(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6), (0.4, 0.4)]
    polys = land_polygons(_raw(tmp_path, ring), (0, 0, 1, 1))
    assert len(polys) == 1 and polys[0].contains(Point(0.5, 0.5)) and abs(polys[0].area - 0.04) < 1e-9
