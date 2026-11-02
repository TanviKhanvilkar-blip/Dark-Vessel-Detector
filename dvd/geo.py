"""Small geodesy helpers and the raster <-> lon/lat mapping for a north-up scene."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EARTH_R = 6371008.8


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def destination(lon, lat, bearing_deg, dist_m):
    """Point reached from (lon, lat) travelling dist_m along bearing_deg (clockwise from north).
    A negative distance walks backwards along the track."""
    lon, lat, brg = np.radians(lon), np.radians(lat), np.radians(bearing_deg)
    d = np.asarray(dist_m) / EARTH_R
    lat2 = np.arcsin(np.sin(lat) * np.cos(d) + np.cos(lat) * np.sin(d) * np.cos(brg))
    lon2 = lon + np.arctan2(np.sin(brg) * np.sin(d) * np.cos(lat), np.cos(d) - np.sin(lat) * np.sin(lat2))
    return np.degrees(lon2), np.degrees(lat2)


@dataclass(frozen=True)
class Grid:
    """North-up raster: row 0 is the northern edge, pixels are square in degrees."""
    west: float
    north: float
    res_deg: float
    height: int
    width: int

    @classmethod
    def from_aoi(cls, aoi, res_deg):
        west, south, east, north = aoi
        return cls(west, north, res_deg, int(round((north - south) / res_deg)), int(round((east - west) / res_deg)))

    def lonlat(self, row, col):
        """Pixel centre coordinates; accepts fractional row/col."""
        return self.west + (np.asarray(col) + 0.5) * self.res_deg, self.north - (np.asarray(row) + 0.5) * self.res_deg

    def rowcol(self, lon, lat):
        return (self.north - np.asarray(lat)) / self.res_deg - 0.5, (np.asarray(lon) - self.west) / self.res_deg - 0.5

    def m_per_px(self, lat):
        """(east-west, north-south) pixel size in metres at a latitude."""
        my = self.res_deg * np.pi / 180 * EARTH_R
        return my * np.cos(np.radians(lat)), my

    @property
    def centre_lat(self):
        return self.north - self.height * self.res_deg / 2
