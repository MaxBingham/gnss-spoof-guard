"""Building geometry from the map package's OSM landmarks, shared by the simulated world (world/scene.py builds
MuJoCo boxes from it) and the onboard DSM (world/onboard_dsm.py rasterizes the same boxes). Pure geometry: no
MuJoCo, no truth, so onboard code may import it.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def footprint_obb(footprint: list[list[float]]) -> tuple[float, float, float, float, float]:
    """Oriented bounding box of a polygon: (cx, cy, half_len_x, half_len_y, angle_rad)."""
    pts = np.asarray(footprint, np.float32)
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
    return float(cx), float(cy), max(float(w), 0.5) / 2, max(float(h), 0.5) / 2, math.radians(float(ang))


def building_boxes(landmarks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Boxes (center, half sizes, yaw, height) for every landmark with a footprint, plus round towers."""
    boxes = []
    for lm in landmarks:
        if lm["type"] not in ("building", "tower"):
            continue
        if lm.get("footprint_enu"):
            cx, cy, hx, hy, ang = footprint_obb(lm["footprint_enu"])
        else:
            cx, cy = lm["centroid_enu"]
            hx = hy = float(lm.get("radius", 4.0))
            ang = 0.0
        boxes.append({"id": lm["id"], "type": lm["type"], "center": [cx, cy], "half": [hx, hy],
                      "yaw": ang, "height": float(lm["height"])})
    return boxes


def box_corners(box: dict[str, Any]) -> np.ndarray:
    """(4, 2) corners of a building box in ENU."""
    hx, hy = box["half"]
    c, s = math.cos(box["yaw"]), math.sin(box["yaw"])
    loc = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    return np.asarray(box["center"]) + loc @ np.array([[c, s], [-s, c]])
