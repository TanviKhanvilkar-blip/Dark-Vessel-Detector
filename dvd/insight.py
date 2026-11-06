"""What can be said about a target beyond "there is something there": its surroundings, nearest ports and likely type.

Type estimates come from real traffic: the share of each ship type among ships of a given length in the Danish AIS
archive (data/ship_type_priors.json, built by `python -m dvd ship-priors`). They are estimates from length alone.
A single radar image cannot show where an unidentified ship came from, so for those we only report nearest ports;
real origins and destinations appear only for ships that broadcast them.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .geo import haversine_m

PRIORS_PATH = os.path.join("data", "ship_type_priors.json")
LENGTH_BINS = [(0, 40), (40, 80), (80, 150), (150, 250), (250, 10_000)]
IGNORED_TYPES = {"Other", "Reserved", "Spare 1", "Spare 2", "Not party to conflict", "Undefined"}
ANCHORAGE_RADIUS_M, ANCHORAGE_NEIGHBOURS = 1500.0, 3
CROWDED = ("Port area", "Anchorage or waiting area")    # a target here being near one from another pass proves little

# Main commercial ports near each area: (name, lat, lon), approximate harbour positions
PORTS = {
    "mumbai": [("Mumbai Port", 18.94, 72.85), ("Jawaharlal Nehru Port (Nhava Sheva)", 18.95, 72.95),
               ("Alibag", 18.64, 72.87), ("Dharamtar", 18.70, 73.02), ("Vasai", 19.33, 72.82)],
    "denmark": [("Copenhagen", 55.70, 12.60), ("Malmö", 55.62, 13.00), ("Landskrona", 55.87, 12.83),
                ("Helsingborg", 56.04, 12.69), ("Helsingør", 56.04, 12.61), ("Køge", 55.45, 12.20)],
}

# AIS "type of ship" codes, as sent in static messages by aisstream
AIS_TYPE_NAMES = [(20, 29, "Wing in ground"), (30, 30, "Fishing"), (31, 32, "Towing"), (33, 33, "Dredging"), (34, 34, "Diving"),
                  (35, 35, "Military"), (36, 36, "Sailing"), (37, 37, "Pleasure"), (40, 49, "High-speed craft"), (50, 50, "Pilot"),
                  (51, 51, "Search and rescue"), (52, 52, "Tug"), (53, 53, "Port tender"), (54, 54, "Anti-pollution"),
                  (55, 55, "Law enforcement"), (58, 58, "Medical"), (60, 69, "Passenger"), (70, 79, "Cargo"), (80, 89, "Tanker")]


def ais_type_name(code) -> str | None:
    try:
        c = int(code)
    except (TypeError, ValueError):
        return None
    return next((name for lo, hi, name in AIS_TYPE_NAMES if lo <= c <= hi), None)


def build_priors(ships: pd.DataFrame, top: int = 3, min_share: float = 0.05) -> dict:
    """{length bin: [(type, share), ...]} from ships with a known type and length."""
    s = ships.dropna(subset=["ship_type", "length_m"])
    s = s[(s.length_m > 0) & ~s.ship_type.isin(IGNORED_TYPES)]
    out = {"ships": int(len(s)), "bins": []}
    for lo, hi in LENGTH_BINS:
        b = s[(s.length_m >= lo) & (s.length_m < hi)]
        shares = b.ship_type.value_counts(normalize=True)
        out["bins"].append(dict(lo=lo, hi=hi, n=int(len(b)),
                                types=[[t, round(float(v), 3)] for t, v in shares.head(top).items() if v >= min_share]))
    return out


_priors = None


def priors() -> dict | None:
    global _priors
    if _priors is None and os.path.exists(PRIORS_PATH):
        _priors = json.load(open(PRIORS_PATH, encoding="utf-8"))
    return _priors


def likely_types(length_m: float, table: dict | None = None) -> list[list]:
    table = table or priors()
    if not table:
        return []
    return next((b["types"] for b in table["bins"] if b["lo"] <= length_m < b["hi"]), [])


def nearest_ports(region: str, lon: float, lat: float, n: int = 2) -> list[list]:
    """[[port name, km], ...], closest first."""
    ports = PORTS.get(region, [])
    d = [(name, float(haversine_m(lon, lat, plon, plat)) / 1000) for name, plat, plon in ports]
    return [[name, round(km, 1)] for name, km in sorted(d, key=lambda x: x[1])[:n]]


def setting(lon: float, lat: float, shore_km: float, others_lon: np.ndarray, others_lat: np.ndarray, port_km: float | None) -> str:
    """Plain description of where a target sits, from its surroundings."""
    near = int((haversine_m(lon, lat, others_lon, others_lat) < ANCHORAGE_RADIUS_M).sum()) - 1 if len(others_lon) else 0
    if port_km is not None and port_km < 3:
        return "Port area"
    if near >= ANCHORAGE_NEIGHBOURS:
        return "Anchorage or waiting area"
    return "Coastal waters" if shore_km < 3 else "Open water"
