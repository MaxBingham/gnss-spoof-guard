"""Lat/lon <-> local ENU conversion (ported from the V1 prototype, backend/app/geo.py).

Equirectangular approximation around a fixed origin (the center of the map bounding box).
Error is well below 0.1 m for areas < 2 km, which is all we need.
Convention: x = east [m], y = north [m], z = up [m].
"""

from __future__ import annotations

import math
from dataclasses import dataclass

EARTH_RADIUS_M = 6_378_137.0  # WGS84 equatorial radius


@dataclass(frozen=True)
class LatLon:
    lat: float
    lon: float


@dataclass(frozen=True)
class GeoOrigin:
    lat: float
    lon: float

    @property
    def m_per_deg_lat(self) -> float:
        return math.pi / 180.0 * EARTH_RADIUS_M

    @property
    def m_per_deg_lon(self) -> float:
        return math.pi / 180.0 * EARTH_RADIUS_M * math.cos(math.radians(self.lat))

    def to_enu(self, lat: float, lon: float) -> tuple[float, float]:
        """Return (east, north) in meters relative to the origin."""
        return (lon - self.lon) * self.m_per_deg_lon, (lat - self.lat) * self.m_per_deg_lat

    def to_latlon(self, x: float, y: float) -> tuple[float, float]:
        """Inverse of to_enu. Returns (lat, lon)."""
        return self.lat + y / self.m_per_deg_lat, self.lon + x / self.m_per_deg_lon


@dataclass(frozen=True)
class BBox:
    south: float
    west: float
    north: float
    east: float

    @property
    def center(self) -> GeoOrigin:
        return GeoOrigin((self.south + self.north) / 2.0, (self.west + self.east) / 2.0)


def bbox_around(a: LatLon, b: LatLon, margin_m: float = 150.0, min_size_m: float = 400.0) -> BBox:
    """Axis-aligned bbox around A and B plus a margin, at least min_size_m on each side."""
    origin = GeoOrigin((a.lat + b.lat) / 2.0, (a.lon + b.lon) / 2.0)
    ax, ay = origin.to_enu(a.lat, a.lon)
    bx, by = origin.to_enu(b.lat, b.lon)
    half_w = max(abs(ax - bx) / 2.0 + margin_m, min_size_m / 2.0)
    half_h = max(abs(ay - by) / 2.0 + margin_m, min_size_m / 2.0)
    cx, cy = (ax + bx) / 2.0, (ay + by) / 2.0
    south, west = origin.to_latlon(cx - half_w, cy - half_h)
    north, east = origin.to_latlon(cx + half_w, cy + half_h)
    return BBox(south=south, west=west, north=north, east=east)
