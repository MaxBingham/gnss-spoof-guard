"""Onboard reader of onboard/thermal_prior.npz (the drone's predicted night thermal map). No truth imports."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ThermalPrior:
    temp_K: np.ndarray  # predicted apparent temperature (rows, cols) float32
    sigma_K: np.ndarray  # 1-std uncertainty of temp_K
    cls: np.ndarray  # uint8 material class (meta["class_names"])
    roads: np.ndarray  # uint8 1 = paved highway visible from above
    road_kind: np.ndarray  # uint8 0 none, 1 vehicle, 2 pedestrian, 3 parking
    meta: dict[str, Any]
    roads_json: list[dict[str, Any]] = field(default_factory=list)  # OSM ways: highway, width_m, paved, pts_enu

    @property
    def res(self) -> float:
        return float(self.meta["res_m"])

    @property
    def min_x(self) -> float:
        return float(self.meta["extent_enu"]["min_x"])

    @property
    def max_y(self) -> float:
        return float(self.meta["extent_enu"]["max_y"])

    def to_px(self, xy: np.ndarray) -> np.ndarray:
        """ENU (..., 2) -> continuous (col, row), integer = cell centre."""
        xy = np.asarray(xy, float)
        return np.stack([(xy[..., 0] - self.min_x) / self.res - 0.5, (self.max_y - xy[..., 1]) / self.res - 0.5], -1)


def load_prior(path: str | Path) -> ThermalPrior:
    """`path`: the onboard (or drone) directory, or the npz itself."""
    p = Path(path)
    if p.is_dir():
        p = p / "thermal_prior.npz"
    with np.load(p) as z:
        return ThermalPrior(z["temp_K"], z["sigma_K"], z["cls"], z["roads"],
                            z["road_kind"] if "road_kind" in z else (z["roads"] > 0).astype(np.uint8),
                            json.loads(str(z["meta"])),
                            json.loads(str(z["roads_json"])) if "roads_json" in z else [])
