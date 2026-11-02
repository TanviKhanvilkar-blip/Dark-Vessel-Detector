"""Detailed land mask from OpenStreetMap coastlines.

OSM stores the coastline as lines (`natural=coastline` ways), not polygons. We cut the AOI box with those
lines, and every resulting face is either land or water. OSM's convention is that land lies to the LEFT of
each coastline way, so a face is land if a point just to the left of a coastline segment falls inside it.
"""
from __future__ import annotations

import json

import numpy as np
from shapely.geometry import LineString, Point, box, mapping
from shapely.ops import linemerge, polygonize, unary_union

from . import config as C

OVERPASS = "https://overpass-api.de/api/interpreter"


def fetch_coastline(aoi, path: str, pad: float = 0.05) -> int:
    """Download coastline ways covering the AOI (+ padding) from the Overpass API. Returns number of ways."""
    import requests

    west, south, east, north = aoi
    q = f'[out:json][timeout:90];way["natural"="coastline"]({south - pad},{west - pad},{north + pad},{east + pad});out geom;'
    r = requests.post(OVERPASS, data={"data": q}, timeout=180,
                      headers={"User-Agent": "dark-vessel-detector/0.1 (research project)", "Accept": "application/json"})
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return sum(e["type"] == "way" for e in r.json()["elements"])


def _ways(raw_path: str) -> list[LineString]:
    els = json.load(open(raw_path, encoding="utf-8"))["elements"]
    return [LineString([(p["lon"], p["lat"]) for p in e["geometry"]]) for e in els
            if e["type"] == "way" and len(e.get("geometry", [])) >= 2]


def land_polygons(raw_path: str, aoi, probe_m: float = 3.0):
    """Split the AOI box by coastline ways and keep the faces on the land (left) side."""
    frame = box(*aoi)
    ways = _ways(raw_path)
    clipped = [w.intersection(frame) for w in ways]
    merged = unary_union([g for g in clipped if not g.is_empty] + [frame.boundary])
    faces = list(polygonize(merged))
    if not faces:
        return []

    probe = probe_m / 111_320
    votes = np.zeros(len(faces))
    for w in ways:
        xy = np.asarray(w.coords)
        for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            if not frame.contains(Point(mx, my)):
                continue
            dx, dy = x1 - x0, y1 - y0
            n = np.hypot(dx, dy)
            if n == 0:
                continue
            left, right = Point(mx - dy / n * probe, my + dx / n * probe), Point(mx + dy / n * probe, my - dx / n * probe)
            for i, f in enumerate(faces):
                if f.contains(left):
                    votes[i] += 1
                elif f.contains(right):
                    votes[i] -= 1
    return [f for f, v in zip(faces, votes) if v > 0]


def write_land_geojson(raw_path: str, out_path: str, aoi=C.AOI) -> int:
    polys = land_polygons(raw_path, aoi)
    fc = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": mapping(p)} for p in polys]}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fc, f)
    return len(polys)
