"""Align AIS to the image time and match radar detections to AIS ships (the fuzzy join)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from . import config as C
from .geo import destination, haversine_m


def project_ais(ais: pd.DataFrame, t0: float, window_s: float = C.MATCH_WINDOW_S) -> pd.DataFrame:
    """One row per ship: its last-known report nearest to t0, dead-reckoned (speed x heading) to t0."""
    cols = ["mmsi", "lon", "lat", "sog", "dt_s"]
    if ais.empty:
        return pd.DataFrame(columns=cols)
    a = ais[(ais["time"] - t0).abs() <= window_s].copy()
    if a.empty:
        return pd.DataFrame(columns=cols)
    a["dt_s"] = t0 - a["time"]                                   # >0: report is older than the image
    a = a.loc[a.groupby("mmsi")["dt_s"].apply(lambda s: s.abs().idxmin())]
    a["lon"], a["lat"] = destination(a["lon"].values, a["lat"].values, a["cog"].values,
                                     a["sog"].values * C.KNOT_MS * a["dt_s"].values)
    return a[cols].reset_index(drop=True)


def match_radius_m(length_m, sog_kn, dt_s):
    """Gate grows with ship size, and with how far the ship may have drifted from the extrapolated track."""
    return C.MATCH_BASE_M + 0.5 * np.asarray(length_m) + C.MATCH_SPEED_FACTOR * np.asarray(sog_kn) * C.KNOT_MS * np.abs(dt_s)


def match(det: pd.DataFrame, proj: pd.DataFrame) -> dict[int, int]:
    """Hungarian assignment on distance / gate. Returns {detection index: ais row index} for accepted pairs."""
    if det.empty or proj.empty:
        return {}
    d = haversine_m(det["lon"].values[:, None], det["lat"].values[:, None],
                    proj["lon"].values[None, :], proj["lat"].values[None, :])
    gate = match_radius_m(det["length_m"].values[:, None], proj["sog"].values[None, :], proj["dt_s"].values[None, :])
    ratio = d / gate
    cost = np.where(ratio <= 1.0, ratio, 1e6)       # forbid pairs outside the gate
    rows, cols = linear_sum_assignment(cost)
    return {int(r): int(c) for r, c in zip(rows, cols) if cost[r, c] < 1e6}
