"""Sentinel-1 GRD scenes from Microsoft Planetary Computer: no account, and only the AOI is downloaded.

The VH image there is a cloud-optimised GeoTIFF, so GDAL can read just the pixels around the AOI over HTTP.
We save that crop locally (with its GCPs shifted and its offset in the original image recorded as tags, which the
calibration lookup needs) plus the scene's VH calibration XML, so `run-scene` then works offline.
"""
from __future__ import annotations

import os

import rasterio
import requests
from rasterio.control import GroundControlPoint

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-grd"   # public, anonymous read token
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
                GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", GDAL_HTTP_MAX_RETRY="4")


def _signed(href: str, token: str) -> str:
    return f"{href}{'&' if '?' in href else '?'}{token}"


def fetch_scene(item_id: str, aoi, out_dir: str, pol: str = "vh", margin: int = 400) -> tuple[str, str]:
    """Download the AOI crop of one scene's `pol` band and its calibration XML. Returns (tiff path, xml path)."""
    from .sar import _aoi_window

    item = requests.get(f"{STAC}/collections/sentinel-1-grd/items/{item_id}", timeout=60)
    item.raise_for_status()
    assets = item.json()["assets"]
    token = requests.get(SAS, timeout=60).json()["token"]
    os.makedirs(out_dir, exist_ok=True)

    cal_href = assets[f"schema-calibration-{pol}"]["href"]
    # Planetary Computer uses generic file names (iw-vh.tiff); name files after the scene so its time can be parsed
    stem = f"{item_id.lower()}-{pol}"
    xml_path = os.path.join(out_dir, f"calibration-{stem}.xml")
    r = requests.get(_signed(cal_href, token), timeout=120)
    r.raise_for_status()
    with open(xml_path, "wb") as f:
        f.write(r.content)

    img_href = assets[pol]["href"]
    tif_path = os.path.join(out_dir, f"{stem}-aoi.tiff")
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(_signed(img_href, token)) as src:
            win = _aoi_window(src, aoi, margin=margin)
            data = src.read(1, window=win)
            gcps, crs = src.gcps
            shifted = [GroundControlPoint(g.row - win.row_off, g.col - win.col_off, g.x, g.y, g.z) for g in gcps]
            profile = dict(driver="GTiff", height=data.shape[0], width=data.shape[1], count=1, dtype=data.dtype,
                           compress="deflate", tiled=True)
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(data, 1)
        dst.gcps = (shifted, crs)
        dst.update_tags(DVD_ROW_OFF=int(win.row_off), DVD_COL_OFF=int(win.col_off), DVD_SOURCE=f"planetarycomputer:{item_id}")
    return tif_path, xml_path
