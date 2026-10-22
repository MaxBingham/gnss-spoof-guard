"""The onboard map package as the drone sees it: orthophoto + landmarks + manifest.

Package layout (one directory):
  manifest.json   origin, bbox, extent in ENU, meters per pixel, A/B, sources and licenses
  landmarks.json  [{id, type, centroid_enu, footprint_enu?, height, radius, name?, osm_id}]
  ortho.jpg       north-up orthophoto covering extent_enu (row 0 = north edge)

This module has no MuJoCo dependency: onboard code (perception, planning) may import it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class OnboardMap:
    def __init__(self, pkg_dir: Path | str, ortho_file: str | None = None) -> None:
        self.dir = Path(pkg_dir)
        self.manifest: dict[str, Any] = json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))
        self.landmarks: list[dict[str, Any]] = json.loads((self.dir / "landmarks.json").read_text(encoding="utf-8"))
        ortho = self.manifest["ortho"]
        self.ortho_file = ortho_file or ortho["file"]
        self.ortho = np.asarray(Image.open(self.dir / self.ortho_file).convert("RGB"))
        self.gray = np.asarray(Image.fromarray(self.ortho).convert("L"))
        e = ortho["extent_enu"]
        self.min_x, self.min_y, self.max_x, self.max_y = e["min_x"], e["min_y"], e["max_x"], e["max_y"]
        h, w = self.gray.shape
        self.mpp_x = (self.max_x - self.min_x) / w
        self.mpp_y = (self.max_y - self.min_y) / h
        self.lm_xy = np.array([lm["centroid_enu"] for lm in self.landmarks], float).reshape(-1, 2)
        self.lm_height = np.array([lm.get("height", 0.0) for lm in self.landmarks], float)
        self.lm_ids = np.array([lm["id"] for lm in self.landmarks], int)
        self.lm_type = [lm["type"] for lm in self.landmarks]

    @property
    def m_per_px(self) -> float:
        return 0.5 * (self.mpp_x + self.mpp_y)

    @property
    def a_enu(self) -> np.ndarray:
        return np.array(self.manifest["a"]["enu"], float)

    @property
    def b_enu(self) -> np.ndarray:
        return np.array(self.manifest["b"]["enu"], float)

    @property
    def attribution(self) -> str:
        return str(self.manifest.get("attribution", ""))

    def enu_to_px(self, xy: np.ndarray) -> np.ndarray:
        """World (x, y) -> ortho pixel (u = column, v = row). Works on (2,) or (N, 2)."""
        xy = np.asarray(xy, float)
        u = (xy[..., 0] - self.min_x) / self.mpp_x
        v = (self.max_y - xy[..., 1]) / self.mpp_y
        return np.stack([u, v], axis=-1)

    def px_to_enu(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, float)
        x = self.min_x + uv[..., 0] * self.mpp_x
        y = self.max_y - uv[..., 1] * self.mpp_y
        return np.stack([x, y], axis=-1)

    def contains(self, xy: np.ndarray, margin: float = 0.0) -> bool:
        return bool(self.min_x + margin <= xy[0] <= self.max_x - margin and self.min_y + margin <= xy[1] <= self.max_y - margin)
