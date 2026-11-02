"""Fixed infrastructure that looks like ships on radar: bridges, piers, breakwaters, platforms, buoys, beacons.

Pulled from OpenStreetMap so the filter covers every mapped structure, not a hand-typed list. Detections within
a buffer of any of these are filtered with reason `fixed_structure` (kept in the database, hidden from alerts).
"""
from __future__ import annotations

import json
import re

import numpy as np
from shapely import STRtree
from shapely.geometry import LineString, Point, mapping, shape

from . import config as C
from .coast import OVERPASS


def fetch_infrastructure(aoi, path: str, pad: float = 0.02) -> int:
    import requests

    west, south, east, north = aoi
    bb = f"{south - pad},{west - pad},{north + pad},{east + pad}"
    q = (f'[out:json][timeout:120];('
         f'way["man_made"~"^(pier|breakwater|groyne|bridge|jetty|quay|offshore_platform)$"]({bb});'
         f'way["bridge"]["bridge"!="no"]({bb});'
         f'way["seamark:type"]({bb});'
         f'node["seamark:type"]({bb});'
         f'node["man_made"~"^(lighthouse|offshore_platform|beacon|mast|tower)$"]({bb});'
         f');out geom;')
    r = requests.post(OVERPASS, data={"data": q}, timeout=240,
                      headers={"User-Agent": "dark-vessel-detector/0.1 (research project)", "Accept": "application/json"})
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return len(r.json()["elements"])


# Seamark types that are lines/zones on a chart, not objects a radar can see. Masking them would hide real ships
# (traffic lanes are exactly where ships are; submarine cables lie on the sea bed).
VIRTUAL = re.compile(r"^(separation_|cable_|pipeline_|anchorage|anchor_berth|restricted_area|fairway|recommended_track|"
                     r"navigation_line|harbour|sea_area|production_area|dredged_area|precautionary_area|inshore_traffic_zone|"
                     r"ferry_route|calling-in_point|radio_calling-in_point)")


def _kind(tags: dict) -> str:
    if tags.get("bridge") and tags["bridge"] != "no":
        return "bridge"
    return tags.get("man_made") or tags.get("seamark:type") or "structure"


def to_geojson(raw_path: str, out_path: str) -> dict[str, int]:
    """Overpass JSON -> GeoJSON FeatureCollection (points and lines) with a `kind` property. Returns counts by kind."""
    feats, counts = [], {}
    for e in json.load(open(raw_path, encoding="utf-8"))["elements"]:
        tags = e.get("tags", {})
        if e["type"] == "node":
            geom = Point(e["lon"], e["lat"])
        elif e["type"] == "way" and len(e.get("geometry", [])) >= 2:
            geom = LineString([(p["lon"], p["lat"]) for p in e["geometry"]])
        else:
            continue
        k = _kind(tags)
        if VIRTUAL.match(k):
            continue
        counts[k] = counts.get(k, 0) + 1
        feats.append({"type": "Feature", "properties": {"kind": k}, "geometry": mapping(geom)})
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)
    return counts


class Structures:
    """Spatial index of structures; `near(lon, lat)` says which detections fall inside a buffer."""

    def __init__(self, geoms: list):
        self.geoms = geoms
        self.tree = STRtree(geoms) if geoms else None

    @classmethod
    def load(cls, geojson_path: str | None, points: list[tuple[float, float]] = ()) -> "Structures":
        geoms = [Point(lo, la) for lo, la in points]
        if geojson_path:
            geoms += [shape(f["geometry"]) for f in json.load(open(geojson_path, encoding="utf-8"))["features"]]
        return cls(geoms)

    def near(self, lon, lat) -> np.ndarray:
        lon, lat = np.atleast_1d(lon), np.atleast_1d(lat)
        out = np.zeros(len(lon), bool)
        if self.tree is None:
            return out
        # degree distances -> metres with the east-west scale (the smaller one), so the buffer never comes out too small
        m_per_deg = 111_320 * np.cos(np.radians(np.mean(lat))) if len(lat) else 111_320
        reach = max(C.STRUCTURE_RADIUS_M, C.STRUCTURE_LINE_BUFFER_M) / m_per_deg
        for i, (x, y) in enumerate(zip(lon, lat)):
            p = Point(x, y)
            for j in self.tree.query(p, predicate="dwithin", distance=reach):
                g = self.geoms[j]
                buf = C.STRUCTURE_RADIUS_M if isinstance(g, Point) else C.STRUCTURE_LINE_BUFFER_M
                if g.distance(p) * m_per_deg <= buf:
                    out[i] = True
                    break
        return out
