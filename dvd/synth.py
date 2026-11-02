"""Synthetic Sentinel-1-like scenes and matching AIS, with known ground truth.

Lets the whole pipeline run and be tested without downloads or API keys. Real data replaces
`make_scene` / `make_ais` through the same interfaces (an intensity array + Grid, an AIS DataFrame).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage

from . import config as C
from .geo import Grid, destination


def coast_lon(lat):
    """Wavy north-south coastline; everything east of it is land."""
    return 72.68 + 0.015 * np.sin((np.asarray(lat) - C.AOI[1]) * 40)


def shore_km_approx(lon, lat):
    return (coast_lon(lat) - lon) * 111.32 * np.cos(np.radians(lat))


@dataclass
class Ship:
    ship_id: int
    lon: float
    lat: float
    length_m: float
    sog_kn: float
    cog_deg: float
    transponder: bool
    category: str  # 'ais' | 'dark' | 'out_of_range'
    mmsi: int = 0


@dataclass
class Scene:
    image: np.ndarray           # linear intensity, ~1.0 over calm sea
    land: np.ndarray            # bool, True on land
    grid: Grid
    time: float                 # epoch seconds
    ships: list[Ship] = field(default_factory=list)
    sea_state: float = 1.0


def _sea_mean(rng, h, w, sea_state):
    coarse = rng.normal(size=(24, 24))
    coarse = ndimage.gaussian_filter(coarse, 2.0)
    coarse /= max(np.abs(coarse).max(), 1e-6)
    field_ = ndimage.zoom(coarse, (h / 24, w / 24), order=1)[:h, :w]
    return (1.0 + 0.35 * sea_state * field_).astype(np.float32)


def _stamp_box(img, grid, lon, lat, length_m, width_m, cog_deg, amp, rng):
    r0, c0 = grid.rowcol(lon, lat)
    mx, my = grid.m_per_px(lat)
    half = int(max(length_m, width_m) / 2 / min(mx, my)) + 3
    ra, rb = max(int(r0) - half, 0), min(int(r0) + half + 1, grid.height)
    ca, cb = max(int(c0) - half, 0), min(int(c0) + half + 1, grid.width)
    if ra >= rb or ca >= cb:
        return
    rr, cc = np.mgrid[ra:rb, ca:cb]
    dx, dy = (cc - c0) * mx, -(rr - r0) * my
    s, c = np.sin(np.radians(cog_deg)), np.cos(np.radians(cog_deg))
    along, across = dx * s + dy * c, dx * c - dy * s
    inside = (np.abs(along) <= length_m / 2) & (np.abs(across) <= width_m / 2)
    n = int(inside.sum())
    if n:
        img[ra:rb, ca:cb][inside] += amp * rng.gamma(2.0, 0.5, size=n).astype(np.float32)


def make_scene(seed: int, t0: float = 1_780_000_000.0, n_ships: int = 28, sea_state: float = 1.0,
               grid: Grid | None = None) -> Scene:
    rng = np.random.default_rng(seed)
    grid = grid or Grid.from_aoi(C.AOI, C.RES_DEG)
    h, w = grid.height, grid.width
    lat_r = grid.north - (np.arange(h) + 0.5) * grid.res_deg
    lon_c = grid.west + (np.arange(w) + 0.5) * grid.res_deg
    land = lon_c[None, :] > coast_lon(lat_r)[:, None]

    mean = _sea_mean(rng, h, w, sea_state)
    img = rng.gamma(4.0, 0.25, size=(h, w)).astype(np.float32) * mean       # 4-look speckle
    land_texture = rng.gamma(1.5, 2.5 / 1.5, size=(h, w)).astype(np.float32)
    img = np.where(land, land_texture, img)

    for lon, lat in C.FIXED_STRUCTURES:
        _stamp_box(img, grid, lon, lat, 45, 45, 0, 60.0, rng)

    ships: list[Ship] = []
    south = C.AOI[1]
    while len(ships) < n_ships:
        lat = rng.uniform(south + 0.02, grid.north - 0.02)
        lon = rng.uniform(grid.west + 0.02, coast_lon(lat) - 0.02)
        if any(abs(s.lon - lon) < 0.01 and abs(s.lat - lat) < 0.01 for s in ships):
            continue
        if any(abs(sl - lon) < 0.005 and abs(sa - lat) < 0.005 for sl, sa in C.FIXED_STRUCTURES):
            continue
        length = float(np.clip(rng.lognormal(np.log(95), 0.55), 25, 300))
        sog = 0.1 if rng.random() < 0.12 else float(rng.uniform(2, 14))
        transponder = length >= 45 and rng.random() < 0.7
        category = "dark"
        if transponder:
            category = "ais" if shore_km_approx(lon, lat) <= C.RX_RANGE_KM else "out_of_range"
        ships.append(Ship(len(ships), lon, lat, length, sog, float(rng.uniform(0, 360)), transponder,
                          category, mmsi=419_000_000 + len(ships) if transponder else 0))

    for s in ships:
        _stamp_box(img, grid, s.lon, s.lat, s.length_m, max(s.length_m / 6, 15), s.cog_deg,
                   float(rng.uniform(15, 45)), rng)
    return Scene(img, land, grid, t0, ships, sea_state)


def make_ais(scene: Scene, seed: int, gap: tuple[float, float] | None = None) -> pd.DataFrame:
    """AIS messages a ground receiver would have logged around the image time.

    Only 'ais' ships produce messages; 'out_of_range' ships transmit but are never heard.
    `gap` = (start, end) epoch seconds of a simulated feed outage.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in scene.ships:
        if s.category != "ais":
            continue
        for dt in np.arange(-1800, 1801, 90) + rng.uniform(-20, 20, size=41):
            t = scene.time + dt
            if rng.random() < 0.2 or (gap and gap[0] <= t <= gap[1]):
                continue
            lon, lat = destination(s.lon, s.lat, s.cog_deg, s.sog_kn * C.KNOT_MS * dt)
            lon, lat = destination(lon, lat, rng.uniform(0, 360), abs(rng.normal(0, 12)))
            rows.append((s.mmsi, t, float(lon), float(lat), s.sog_kn + rng.normal(0, 0.2),
                         (s.cog_deg + rng.normal(0, 2)) % 360))
    return pd.DataFrame(rows, columns=["mmsi", "time", "lon", "lat", "sog", "cog"])
