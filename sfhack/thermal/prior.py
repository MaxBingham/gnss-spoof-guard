"""Onboard reader of onboard/thermal_prior.npz (the drone's predicted night thermal map). No truth imports."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
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


    def ground_points(self, pos_xy: np.ndarray, agl_m: float, intrinsics: dict[str, float],
                      yaw: float | None = None, R: np.ndarray | None = None, step: int = 1) -> np.ndarray:
        """ENU (h // step, w // step, 2) where each camera pixel's ray meets flat ground agl_m below.
        Camera convention = sfhack.thermal.camera / MuJoCo "down": image top = forward, right = right.
        Either `yaw` (nadir camera, level drone) or `R` (3x3 camera-to-world, columns = camera x (right),
        y (image up), z (backward), the MuJoCo cam_xmat convention, e.g. from the AHRS attitude)."""
        k = intrinsics
        u = (np.arange(0, int(k["w"]), step) + 0.5 - k["cx"]) / k["fx"]
        v = (np.arange(0, int(k["h"]), step) + 0.5 - k["cy"]) / k["fy"]
        U, V = np.meshgrid(u, v)
        if R is None:
            c, s = np.cos(yaw or 0.0), np.sin(yaw or 0.0)
            R = np.array([[s, c, 0.0], [-c, s, 0.0], [0.0, 0.0, 1.0]])  # x = right, y = forward, z = up
        rays = np.stack([U, -V, -np.ones_like(U)], -1) @ np.asarray(R, float).T
        t = agl_m / np.maximum(-rays[..., 2], 1e-6)
        return np.asarray(pos_xy, float)[:2] + rays[..., :2] * t[..., None]

    def predict_view(self, pos_xy: np.ndarray, agl_m: float, intrinsics: dict[str, float], yaw: float | None = None,
                     R: np.ndarray | None = None, step: int = 2) -> tuple[np.ndarray, np.ndarray]:
        """The image the drone expects from this pose (flat-ground projection of the prior): (temp_K, sigma_K),
        each (h // step, w // step). Outside the map: nan."""
        uv = self.to_px(self.ground_points(pos_xy, agl_m, intrinsics, yaw, R, step)).astype(np.float32)
        t = cv2.remap(self.temp_K, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                      borderValue=float("nan"))
        sg = cv2.remap(self.sigma_K, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                       borderValue=float("nan"))
        return t, sg


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
