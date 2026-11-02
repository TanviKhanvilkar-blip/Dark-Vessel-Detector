"""Score detections against synthetic ground truth, and quantify the free-AIS blind spot.

With real scenes the same functions take Global Fishing Watch's published detections as `truth`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import haversine_m

SIZE_BINS = [(0, 60), (60, 100), (100, 200), (200, 1e9)]


def truth_frame(scene) -> pd.DataFrame:
    return pd.DataFrame([vars(s) for s in scene.ships])


def score_detections(det: pd.DataFrame, truth: pd.DataFrame, radius_m: float = 150.0, min_length_m: float = 40.0) -> dict:
    """Greedy nearest matching of detections to truth ships (both sides use ships >= min_length_m)."""
    truth = truth[truth.length_m >= min_length_m].reset_index(drop=True)
    det = det.reset_index(drop=True)
    hit_truth = set()
    tp = 0
    for _, d in det.iterrows():
        if truth.empty:
            break
        dist = haversine_m(d.lon, d.lat, truth.lon.values, truth.lat.values)
        j = int(np.argmin(dist))
        if dist[j] <= radius_m and j not in hit_truth:
            hit_truth.add(j)
            tp += 1
    fp, fn = len(det) - tp, len(truth) - tp
    by_size = {}
    for lo, hi in SIZE_BINS:
        idx = truth.index[(truth.length_m >= lo) & (truth.length_m < hi)]
        by_size[f"{lo}-{int(hi) if hi < 1e8 else '+'}m"] = (sum(i in hit_truth for i in idx) / len(idx)) if len(idx) else None
    return dict(tp=tp, fp=fp, fn=fn, precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), recall_by_size=by_size)


def classify_candidates(det: pd.DataFrame, truth: pd.DataFrame, radius_m: float = 150.0) -> pd.DataFrame:
    """For each candidate, what was the ship really? ('dark' = no transponder, 'out_of_range' = transmitting but unheard)."""
    rows = []
    for _, d in det[det.status == "candidate"].iterrows():
        dist = haversine_m(d.lon, d.lat, truth.lon.values, truth.lat.values)
        j = int(np.argmin(dist))
        rows.append(dict(confidence=d.confidence, truth=truth.category.iloc[j] if dist[j] <= radius_m else "false_alarm"))
    return pd.DataFrame(rows, columns=["confidence", "truth"])
