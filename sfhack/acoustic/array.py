"""UMA-16 v2 geometry: mic position per USB channel (from `geometry/uma16_positions.csv`, Acoular 26.8, BSD-3).

The channel order is NOT row by row; always map channels through `mic_positions()`.

Array frame: x, y in the board plane, z = board normal (right-handed, the side the mics face for a flat,
mics-up mounting). Directions in this frame: `array_az_deg` CCW from +x in the xy plane, `array_el_deg` above
the xy plane. `to_compass()` maps that to a compass bearing with the mounting config.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GEOMETRY_DIR = Path(__file__).parent / "geometry"
N_MICS = 16
FS = 48000
C_SOUND = 343.0


def mic_positions(path: Path | None = None) -> np.ndarray:
    """(16, 3) mic positions in m, row i = USB channel i."""
    path = path or GEOMETRY_DIR / "uma16_positions.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["channel"]))
    pos = np.array([[float(r["x_m"]), float(r["y_m"]), float(r["z_m"])] for r in rows])
    assert pos.shape == (N_MICS, 3)
    return pos


def unit_vector(array_az_deg, array_el_deg) -> np.ndarray:
    """Direction(s) towards the source in the array frame, shape (..., 3)."""
    az = np.radians(array_az_deg)
    el = np.radians(array_el_deg)
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)


@dataclass
class Mounting:
    """How the array sits on the table / mast.

    x_axis_bearing_deg: compass bearing (deg clockwise from north) the array's +x axis points to.
    mics_up: True = mics face up (array z = up, frame right-handed as ENU). False = mics face down: the
        frame is mirrored, so a CCW array azimuth turns clockwise on the compass.
    """
    x_axis_bearing_deg: float = 0.0
    mics_up: bool = True

    def to_compass(self, array_az_deg: float) -> float:
        sign = -1.0 if self.mics_up else 1.0
        return (self.x_axis_bearing_deg + sign * array_az_deg) % 360.0

    def from_compass(self, az_deg: float) -> float:
        sign = -1.0 if self.mics_up else 1.0
        return (sign * (az_deg - self.x_axis_bearing_deg)) % 360.0
