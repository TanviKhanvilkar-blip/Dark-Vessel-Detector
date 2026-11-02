"""Ship detection on a SAR intensity scene: tiled CA-CFAR baseline, with a hook for a YOLO model.

CFAR (constant false alarm rate): a pixel is a ship candidate if it is much brighter than the sea
around it. The background is estimated from a ring around each pixel (a guard window is excluded so
the ship itself does not raise its own threshold) and land is excluded from the estimate.
"""
from __future__ import annotations

from typing import Callable, Iterator

import numpy as np
import pandas as pd
from scipy import ndimage

from . import config as C
from .geo import Grid

Detector = Callable[[np.ndarray, np.ndarray], list[dict]]  # (tile, sea_mask) -> blobs with row/col/size in tile px


def _box_sum(x, size):
    return ndimage.uniform_filter(x, size=size, mode="constant") * size * size


def cfar_blobs(tile: np.ndarray, sea: np.ndarray, grid: Grid, row_off: int = 0) -> list[dict]:
    """Detect bright blobs in one tile. Returns row/col (tile pixels), area, contrast and length in metres."""
    sea_f = sea.astype(np.float32)
    img = np.where(sea, tile, 0).astype(np.float32)
    big, small = C.CFAR_BG_PX, C.CFAR_GUARD_PX
    n_ring = _box_sum(sea_f, big) - _box_sum(sea_f, small)
    s_ring = _box_sum(img, big) - _box_sum(img, small)
    bg = np.where(n_ring > 50, s_ring / np.maximum(n_ring, 1), np.nan)

    hit = (tile > C.CFAR_THRESHOLD * bg) & sea
    hit = ndimage.binary_closing(hit, structure=np.ones((3, 3)))
    labels, n = ndimage.label(hit, structure=np.ones((3, 3)))
    if n == 0:
        return []

    out = []
    for lab, sl in enumerate(ndimage.find_objects(labels), start=1):
        mask = labels[sl] == lab
        area = int(mask.sum())
        if area < C.MIN_AREA_PX:
            continue
        rr, cc = np.nonzero(mask)
        rr, cc = rr + sl[0].start, cc + sl[1].start
        lat = grid.north - (rr + row_off + 0.5) * grid.res_deg
        mx, my = grid.m_per_px(lat.mean())
        pts = np.stack([cc * mx, rr * my], axis=1)
        pts = pts - pts.mean(axis=0)
        eig = np.linalg.eigvalsh(np.cov(pts.T)) if area > 2 else np.array([0.0, 0.0])
        length = float(np.sqrt(12 * max(eig[-1], 0.0)))   # rectangle of length L has variance L^2/12 along its axis
        contrast = float(tile[rr, cc].mean() / np.nanmean(bg[rr, cc]))
        out.append(dict(row=float(rr.mean()), col=float(cc.mean()), area_px=area, length_m=length, contrast=contrast))
    return out


def tiles(shape, core=C.TILE_CORE_PX, halo=C.TILE_HALO_PX) -> Iterator[tuple[slice, slice, tuple[int, int, int, int]]]:
    """Yield (padded row slice, padded col slice, core box r0,r1,c0,c1). Detections are kept only if their
    centroid lies in the core, so each ship is reported exactly once without de-duplication."""
    h, w = shape
    for r0 in range(0, h, core):
        for c0 in range(0, w, core):
            r1, c1 = min(r0 + core, h), min(c0 + core, w)
            yield (slice(max(r0 - halo, 0), min(r1 + halo, h)), slice(max(c0 - halo, 0), min(c1 + halo, w)),
                   (r0, r1, c0, c1))


def detect_scene(image: np.ndarray, land: np.ndarray, grid: Grid, detector: str = "cfar") -> pd.DataFrame:
    """Run the detector over every tile and return one row per ship with lon/lat, length and score."""
    sea = ~ndimage.binary_dilation(land, iterations=C.LAND_BUFFER_PX)
    rows = []
    for rs, cs, (r0, r1, c0, c1) in tiles(image.shape):
        tile, tsea = image[rs, cs], sea[rs, cs]
        if detector == "cfar":
            blobs = cfar_blobs(tile, tsea, grid, row_off=rs.start)
        elif detector == "yolo":
            from .yolo import yolo_blobs
            blobs = yolo_blobs(tile, tsea, grid, row_off=rs.start)
        else:
            raise ValueError(f"unknown detector {detector!r}")
        for b in blobs:
            row, col = b["row"] + rs.start, b["col"] + cs.start
            if r0 <= row < r1 and c0 <= col < c1:
                rows.append({**b, "row": row, "col": col})
    df = pd.DataFrame(rows, columns=["row", "col", "area_px", "length_m", "contrast"])
    if df.empty:
        return df.assign(lon=[], lat=[], score=[])
    df["lon"], df["lat"] = grid.lonlat(df["row"], df["col"])
    df["score"] = 1 - np.exp(-df["contrast"] / 10)       # monotone squash of contrast into 0..1
    return df.reset_index(drop=True)
