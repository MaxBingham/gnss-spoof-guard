"""Daytime sun exposure and sky view factor from a DSM (north-up raster, docs/MAP_PACKAGE.md convention).

exposure: share of the day's direct-beam energy a cell received (0 = in shadow all day, 1 = never shaded).
          A road in a building's afternoon shadow stored less heat and is cooler at night.
svf:      sky view factor of a horizontal cell, 1 - mean over azimuths of sin^2(horizon elevation).
          Street canyons radiate less to the cold sky and stay warmer at night.
Both by shifting the DSM along a direction and comparing heights (no per-cell loops).
"""

from __future__ import annotations

import math

import numpy as np

from .seb import solar


def _shifted(a: np.ndarray, dr: int, dc: int, fill: float) -> np.ndarray:
    """out[r, c] = a[r + dr, c + dc] (fill outside)."""
    out = np.full_like(a, fill)
    R, C = a.shape
    r0, r1 = max(0, -dr), min(R, R - dr)
    c0, c1 = max(0, -dc), min(C, C - dc)
    if r1 > r0 and c1 > c0:
        out[r0:r1, c0:c1] = a[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
    return out


def horizon_tan(dsm: np.ndarray, res: float, az: float, max_dist: float) -> np.ndarray:
    """max over distance of (dsm(p + d * dir) - dsm(p)) / d toward azimuth az (rad from north, clockwise)."""
    dx, dy = math.sin(az), math.cos(az)
    best = np.full(dsm.shape, -np.inf, np.float32)
    steps = int(max_dist / res)
    d = 1.0
    while d <= steps:
        dc, dr = int(round(dx * d)), int(round(-dy * d))  # north = row - 1
        dist = math.hypot(dc, dr) * res
        if dist > 0:
            np.maximum(best, (_shifted(dsm, dr, dc, -1e9) - dsm) / dist, out=best)
        d = d + 1.0 if d < 20 else d * 1.08  # fine near the cell, coarser far away
    return best


def sun_exposure(dsm: np.ndarray, res: float, lat_deg: float, doy: int, max_dist: float = 150.0) -> np.ndarray:
    hours = np.arange(5.0, 20.01, 0.5)
    direct, _, el, az = solar(hours, lat_deg, doy)
    lit = np.zeros(dsm.shape, np.float32)
    total = 0.0
    for e, a, s in zip(el, az, direct):
        if e <= math.radians(2.0) or s <= 1.0:
            continue
        tan_h = horizon_tan(dsm, res, float(a), min(max_dist, 60.0 / math.tan(e)))
        lit += s * (tan_h < math.tan(e))
        total += s
    return lit / max(total, 1e-9)


def sky_view(dsm: np.ndarray, res: float, n_dir: int = 16, max_dist: float = 80.0) -> np.ndarray:
    acc = np.zeros(dsm.shape, np.float32)
    for k in range(n_dir):
        t = np.maximum(horizon_tan(dsm, res, 2 * math.pi * k / n_dir, max_dist), 0.0)
        acc += t * t / (1 + t * t)  # sin^2(atan t)
    return 1.0 - acc / n_dir
