"""One satellite pass end to end: detect -> align AIS -> match -> filter -> label -> save."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from . import config as C
from . import store
from .detect import detect_scene
from .geo import haversine_m
from .infra import Structures
from .match import match, project_ais


def shore_distance_km(land: np.ndarray, grid, rows, cols) -> np.ndarray:
    px_m = float(np.mean(grid.m_per_px(grid.centre_lat)))
    dist = ndimage.distance_transform_edt(~land) * px_m / 1000
    return dist[np.clip(np.round(rows).astype(int), 0, grid.height - 1), np.clip(np.round(cols).astype(int), 0, grid.width - 1)]


def save_crop(image: np.ndarray, row: float, col: float, path: str, half: int = 48) -> None:
    r, c = int(row), int(col)
    patch = image[max(r - half, 0):r + half, max(c - half, 0):c + half]
    v = np.log1p(patch)
    v = (255 * (v - 0.0) / max(np.log1p(60.0), 1e-6)).clip(0, 255).astype(np.uint8)
    Image.fromarray(v).resize((192, 192), Image.NEAREST).save(path)


def run_pass(scene, store_con, pass_id: str, out_dir: str | None = None, detector: str = "cfar",
             structures: Structures | None = None) -> tuple[dict, pd.DataFrame]:
    """Process one scene. AIS is read back from the database exactly as the live system would.
    `structures` defaults to the points in config.FIXED_STRUCTURES."""
    t0, grid = scene.time, scene.grid
    structures = structures or Structures.load(None, C.FIXED_STRUCTURES)
    det = detect_scene(scene.image, scene.land, grid, detector)
    # only AIS near this scene: the database also holds other recorded areas
    pad = C.AIS_PAD_DEG
    bbox = (grid.west - pad, grid.north - grid.height * grid.res_deg - pad,
            grid.west + grid.width * grid.res_deg + pad, grid.north + pad)
    ais = store.ais_window(store_con, t0, C.MATCH_WINDOW_S, bbox)
    gap_s = store.gap_seconds(store_con, t0, C.MATCH_WINDOW_S)
    proj = project_ais(ais, t0)

    det["shore_km"] = shore_distance_km(scene.land, grid, det["row"], det["col"]) if len(det) else []
    # only plausible ships compete for AIS partners, so faint clutter (e.g. a wake) cannot take a real ship's match
    eligible = det.index[(det.score >= C.MIN_SCORE) & (det.length_m >= C.MIN_LENGTH_M)] if len(det) else det.index
    sub_pairs = match(det.loc[eligible].reset_index(drop=True), proj)
    pairs = {int(eligible[k]): v for k, v in sub_pairs.items()}
    near_struct = structures.near(det["lon"].values, det["lat"].values) if len(det) else []

    # candidates from other passes in this database: a target that recurs at the same spot is probably fixed
    try:
        others = pd.read_sql("SELECT lon, lat FROM detections WHERE status='candidate' AND pass_id != ? "
                             "AND lon BETWEEN ? AND ? AND lat BETWEEN ? AND ?", store_con,
                             params=(pass_id, bbox[0], bbox[2], bbox[1], bbox[3]))
    except Exception:
        others = pd.DataFrame(columns=["lon", "lat"])

    def recurring(lon, lat) -> bool:
        return len(others) > 0 and bool((haversine_m(lon, lat, others.lon.values, others.lat.values) < C.REPEAT_RADIUS_M).any())

    status, reason, conf = [], [], []
    for i, d in det.iterrows():
        # A ship that matches AIS is a ship, even beside a pier or bridge: the structure and shore filters only exist
        # to stop structures being called dark, so they apply to unmatched detections. Score and size filters come
        # first because only plausible ships take part in matching (see `eligible` above).
        if d.score < C.MIN_SCORE:
            s, r, c = "filtered", "low_score", "n/a"
        elif d.length_m < C.MIN_LENGTH_M:
            s, r, c = "filtered", "too_small", "n/a"
        elif i in pairs:
            s, r, c = "matched", "", "high"
        elif near_struct[i]:
            s, r, c = "filtered", "fixed_structure", "n/a"
        elif d.shore_km < C.MIN_SHORE_KM:
            s, r, c = "filtered", "too_close_to_shore", "n/a"
        else:
            # Unmatched. Only trust "dark" where a ground receiver could plausibly have heard the ship
            # and the AIS feed was healthy; otherwise it is merely "not publicly tracked".
            reasons = []
            if d.shore_km > C.RX_RANGE_KM:
                reasons.append("beyond_ground_receiver_range")
            if gap_s > C.MAX_AIS_GAP_S:
                reasons.append("ais_feed_gap")
            if proj.empty:
                reasons.append("no_ais_in_window")
            if recurring(d.lon, d.lat):
                reasons.append("same_spot_in_another_pass")
            s, r, c = "candidate", ",".join(reasons), "low" if reasons else "high"
        status.append(s); reason.append(r); conf.append(c)
    det["status"], det["reason"], det["confidence"] = status, reason, conf
    det["pass_id"], det["det_id"] = pass_id, np.arange(len(det))
    det["crop"] = ""

    if out_dir:
        os.makedirs(os.path.join(out_dir, "crops"), exist_ok=True)
        for i, d in det[det.status == "candidate"].iterrows():
            rel = f"crops/{pass_id}_{int(d.det_id)}.png"
            save_crop(scene.image, d.row, d.col, os.path.join(out_dir, rel))
            det.at[i, "crop"] = rel

    kept = det[det.status != "filtered"]
    n_match, n_cand = int((det.status == "matched").sum()), int((det.status == "candidate").sum())
    summary = dict(
        pass_id=pass_id, time=t0, n_det=len(kept), n_matched=n_match, n_candidates=n_cand,
        n_low_conf=int(((det.status == "candidate") & (det.confidence == "low")).sum()),
        n_filtered=int((det.status == "filtered").sum()),
        match_rate=n_match / len(kept) if len(kept) else float("nan"),
        mean_score=float(kept.score.mean()) if len(kept) else float("nan"),
        ais_msgs=len(ais), ais_gap_s=gap_s, sea_state=float(getattr(scene, "sea_state", np.nan)))
    store.save_pass(store_con, summary, det)
    return summary, det
