"""YOLO path (phase 1b): export labelled tiles in YOLO format, and a detector with the same blob interface as CFAR.

Training itself is a one-liner once `ultralytics` is installed (see README); it is not run here.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

from . import config as C
from .detect import tiles
from .evaluate import truth_frame

TILE = 640


def to_uint8(tile: np.ndarray) -> np.ndarray:
    """Log-scale SAR intensity into 8 bit; the same transform must be used for training and inference."""
    return (255 * np.log1p(tile) / np.log1p(60.0)).clip(0, 255).astype(np.uint8)


def export_dataset(scenes, out_dir: str, split: str = "train") -> int:
    """Cut scenes into TILE x TILE images with YOLO labels (class 0 = ship). Returns number of tiles written."""
    img_dir, lab_dir = (os.path.join(out_dir, d, split) for d in ("images", "labels"))
    os.makedirs(img_dir, exist_ok=True); os.makedirs(lab_dir, exist_ok=True)
    n = 0
    for si, sc in enumerate(scenes):
        truth = truth_frame(sc)
        truth = truth[truth.length_m >= C.MIN_LENGTH_M]
        rows, cols = sc.grid.rowcol(truth.lon.values, truth.lat.values)
        mx, my = sc.grid.m_per_px(sc.grid.centre_lat)
        h, w = sc.image.shape
        for r0 in range(0, h - TILE + 1, TILE):
            for c0 in range(0, w - TILE + 1, TILE):
                lines = []
                for (r, c, L) in zip(rows, cols, truth.length_m):
                    if r0 <= r < r0 + TILE and c0 <= c < c0 + TILE:
                        side = max(L / mx, 6) * 1.2          # square box: orientation is not tracked here
                        lines.append(f"0 {(c - c0) / TILE:.5f} {(r - r0) / TILE:.5f} {side / TILE:.5f} {side / TILE:.5f}")
                name = f"s{si}_{r0}_{c0}"
                Image.fromarray(to_uint8(sc.image[r0:r0 + TILE, c0:c0 + TILE])).save(os.path.join(img_dir, name + ".png"))
                with open(os.path.join(lab_dir, name + ".txt"), "w") as f:
                    f.write("\n".join(lines))
                n += 1
    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(out_dir)}\ntrain: images/train\nval: images/val\nnames:\n  0: ship\n")
    return n


_model = None


def yolo_blobs(tile: np.ndarray, sea: np.ndarray, grid, row_off: int = 0, weights: str = "runs/detect/train/weights/best.pt") -> list[dict]:
    global _model
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError("pip install ultralytics to use the YOLO detector") from e
    _model = _model or YOLO(weights)
    img = np.stack([to_uint8(tile)] * 3, axis=-1)
    out = []
    for b in _model.predict(img, imgsz=TILE, conf=0.25, verbose=False)[0].boxes:
        x0, y0, x1, y1 = b.xyxy[0].tolist()
        r, c = (y0 + y1) / 2, (x0 + x1) / 2
        if not sea[int(r), int(c)]:
            continue
        mx, _ = grid.m_per_px(grid.north - (r + row_off) * grid.res_deg)
        out.append(dict(row=r, col=c, area_px=int((x1 - x0) * (y1 - y0)), length_m=float(max(x1 - x0, y1 - y0) * mx),
                        contrast=float(-10 * np.log(1 - float(b.conf)))))   # invert the CFAR score squash so `score` == YOLO confidence
    return out
