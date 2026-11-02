"""Load a real Sentinel-1 GRD measurement GeoTIFF into the same `Scene` the synthetic generator produces.

Steps: find the pixel window covering the AOI (GRD products are geo-referenced by ground control points,
not a simple transform), read only that window, convert DN to sigma0 with the product's calibration XML,
resample onto the AOI `Grid`, rasterise a land mask, and scale so calm sea is ~1.0 (the detector works
on ratios, and crops/scores assume this scale).
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import numpy as np
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import GCPTransformer, from_origin
from rasterio.warp import reproject
from rasterio.windows import Window
from scipy.interpolate import RegularGridInterpolator

from . import config as C
from .geo import Grid
from .synth import Scene


def time_from_name(path: str) -> float:
    """Sentinel-1 file names embed the start time, e.g. s1a-iw-grd-vv-20260104t013200-....tiff"""
    m = re.search(r"(\d{8})[tT](\d{6})", path)
    if not m:
        raise ValueError("cannot read acquisition time from the file name; pass time=")
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).timestamp()


def _aoi_window(ds, aoi, margin=300) -> Window:
    west, south, east, north = aoi
    lons, lats = [west, east, west, east], [south, south, north, north]
    gcps = ds.gcps[0]
    if gcps:
        rows, cols = GCPTransformer(gcps).rowcol(lons, lats)
    else:
        rows, cols = zip(*(ds.index(x, y) for x, y in zip(lons, lats)))
    r0, r1 = max(int(min(rows)) - margin, 0), min(int(max(rows)) + margin, ds.height)
    c0, c1 = max(int(min(cols)) - margin, 0), min(int(max(cols)) + margin, ds.width)
    if r1 <= r0 or c1 <= c0:
        raise ValueError("the AOI does not overlap this scene")
    return Window(c0, r0, c1 - c0, r1 - r0)


def sigma0_lut(xml_path: str, win: Window) -> np.ndarray:
    """Per-pixel sigma-nought look-up values for a window, from a Sentinel-1 calibration XML."""
    vecs = ET.parse(xml_path).getroot().iter("calibrationVector")
    lines, table = [], []
    for v in vecs:
        lines.append(int(v.findtext("line")))
        pix = np.array(v.findtext("pixel").split(), dtype=float)
        table.append(np.array(v.findtext("sigmaNought").split(), dtype=float))
    interp = RegularGridInterpolator((np.array(lines, float), pix), np.vstack(table), bounds_error=False, fill_value=None)
    rr, cc = np.mgrid[int(win.row_off):int(win.row_off + win.height), int(win.col_off):int(win.col_off + win.width)]
    return interp(np.stack([rr.ravel(), cc.ravel()], axis=1)).reshape(rr.shape).astype(np.float32)


def load_scene(tiff: str, aoi=C.AOI, res_deg: float = C.RES_DEG, calibration_xml: str | None = None,
               land_geojson: str | None = None, time: float | None = None, gcp_method: str = "tps") -> Scene:
    """gcp_method: 'tps' (thin-plate spline, exact at every GCP) or 'polynomial' (GDAL's single global fit, which can be
    hundreds of metres off locally on a 250 km wide swath)."""
    grid = Grid.from_aoi(aoi, res_deg)
    with rasterio.open(tiff) as ds:
        win = _aoi_window(ds, aoi)
        dn = ds.read(1, window=win).astype(np.float32)
        nodata_src = (dn == 0)
        intensity = dn ** 2
        if calibration_xml:
            lut = sigma0_lut(calibration_xml, win)
            intensity = intensity / np.maximum(lut, 1e-6) ** 2
        # re-express the full-image GCPs / transform relative to the window
        gcps, gcp_crs = ds.gcps
        kw = {}
        if gcps:
            shifted = [GroundControlPoint(g.row - win.row_off, g.col - win.col_off, g.x, g.y, g.z) for g in gcps]
            kw = dict(gcps=shifted, src_crs=gcp_crs)
            if gcp_method == "tps":
                kw["SRC_METHOD"] = "GCP_TPS"
        else:
            kw = dict(src_transform=ds.window_transform(win), src_crs=ds.crs)

    dst_tf = from_origin(grid.west, grid.north, res_deg, res_deg)
    out = np.zeros((grid.height, grid.width), np.float32)
    valid = np.zeros((grid.height, grid.width), np.float32)
    common = dict(dst_transform=dst_tf, dst_crs="EPSG:4326", resampling=Resampling.average)
    reproject(intensity, out, **kw, **common)
    reproject((~nodata_src).astype(np.float32), valid, **kw, **common)

    no_data = valid < 0.99          # outside the swath or on the black border
    land = no_data.copy()
    if land_geojson:
        import json
        from shapely.geometry import shape
        feats = json.load(open(land_geojson))
        feats = feats.get("features", [feats])
        geoms = [shape(f["geometry"] if "geometry" in f else f) for f in feats]
        land |= rasterize([(g, 1) for g in geoms], out_shape=out.shape, transform=dst_tf, dtype="uint8").astype(bool)

    sea_vals = out[~land & (out > 0)]
    if sea_vals.size == 0:
        raise ValueError("no sea pixels in the AOI: check the AOI, the scene footprint and the land mask")
    out /= np.median(sea_vals)
    return Scene(out, land, grid, time if time is not None else time_from_name(tiff))
