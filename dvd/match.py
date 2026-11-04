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
    """Gate grows with ship size, with how far the ship may have drifted from the extrapolated track, and with the
    radar's azimuth shift of moving ships (stationary ships keep the tight gate)."""
    speed = np.asarray(sog_kn) * C.KNOT_MS
    return (C.MATCH_BASE_M + 0.5 * np.asarray(length_m) + C.MATCH_SPEED_FACTOR * speed * np.abs(dt_s)
            + C.SAR_RANGE_OVER_VELOCITY_S * speed)


def add_ship_details(proj: pd.DataFrame, ships: dict) -> pd.DataFrame:
    """Attach each AIS ship's reported length and type, used to check that a match makes sense."""
    if proj.empty:
        return proj
    return proj.assign(ais_length_m=[ships.get(int(m), {}).get("length_m") for m in proj.mmsi],
                       ship_type=[ships.get(int(m), {}).get("ship_type") for m in proj.mmsi])


def plausible_size(det: pd.DataFrame, proj: pd.DataFrame) -> np.ndarray:
    """[detections x ships] True where the target's size and place fit the ship. At a quay a match needs a known length
    of at least 40 m: a large ship nearly always reports its length, and a small craft can't be told from the quay."""
    if "ais_length_m" not in proj or "shore_km" not in det:
        return np.ones((len(det), len(proj)), bool)
    radar = det["length_m"].values[:, None]
    at_quay = det["shore_km"].values[:, None] < C.QUAY_SHORE_KM
    ais = pd.to_numeric(proj["ais_length_m"], errors="coerce").values[None, :]
    too_long = radar > C.MATCH_MAX_LENGTH_RATIO * ais + C.MATCH_LENGTH_SLACK_M       # False where the AIS length is unknown (NaN)
    small = ~(ais >= C.QUAY_MIN_SHIP_M) | proj["ship_type"].isin(C.SMALL_CRAFT_TYPES).values[None, :]    # NaN counts as small
    return ~(too_long | (at_quay & small))


def match(det: pd.DataFrame, proj: pd.DataFrame) -> dict[int, int]:
    """Hungarian assignment on distance / gate. Returns {detection index: ais row index} for accepted pairs."""
    if det.empty or proj.empty:
        return {}
    d = haversine_m(det["lon"].values[:, None], det["lat"].values[:, None],
                    proj["lon"].values[None, :], proj["lat"].values[None, :])
    gate = match_radius_m(det["length_m"].values[:, None], proj["sog"].values[None, :], proj["dt_s"].values[None, :])
    ratio = d / gate
    cost = np.where((ratio <= 1.0) & plausible_size(det, proj), ratio, 1e6)       # forbid pairs outside the gate or of the wrong size
    rows, cols = linear_sum_assignment(cost)
    return {int(r): int(c) for r, c in zip(rows, cols) if cost[r, c] < 1e6}
