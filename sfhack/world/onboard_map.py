"""The onboard map package as the drone sees it: orthophoto + landmarks + manifest (+ terrain, mission).

Package layout (one directory, full spec in docs/MAP_PACKAGE.md):
  manifest.json   origin, bbox, extent in ENU, meters per pixel, A/B, sources and licenses
  landmarks.json  [{id, type, centroid_enu, footprint_enu?, height, radius, name?, osm_id}]
  ortho.jpg       north-up orthophoto covering extent_enu (row 0 = north edge)
  dsm.npz         (mission packages) north-up DTM + DSM rasters, z in the package's ENU datum
  mission.json    (mission packages) A, B, search radius, route

This module has no MuJoCo dependency: onboard code (perception, planning) may import it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..missionprep.geometry import Grid, sample_bilinear


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
        # Terrain (mission packages): without it the world is flat at z = 0, as in the sample package.
        self.dtm: np.ndarray | None = None
        self.dsm: np.ndarray | None = None
        self.dem_grid: Grid | None = None
        if (self.dir / "dsm.npz").exists():
            with np.load(self.dir / "dsm.npz") as z:
                self.dtm = np.asarray(z["dtm"], np.float32)
                self.dsm = np.asarray(z["dsm"], np.float32)
                self.dem_grid = Grid(float(z["min_x"]), float(z["max_y"]), float(z["res_m"]), *self.dtm.shape)
        mp = self.dir / "mission.json"
        self.mission: dict[str, Any] | None = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else None

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

    def ground_z(self, xy: np.ndarray) -> float:
        """Bare-earth height (map DTM) under (x, y); 0 without terrain."""
        if self.dtm is None or self.dem_grid is None:
            return 0.0
        return float(sample_bilinear(self.dtm, self.dem_grid, np.asarray(xy, float)[:2][None])[0])

    def surface_z(self, xy: np.ndarray) -> float:
        """Top surface height (map DSM: roofs, crowns) under (x, y); 0 without terrain."""
        if self.dsm is None or self.dem_grid is None:
            return 0.0
        return float(sample_bilinear(self.dsm, self.dem_grid, np.asarray(xy, float)[:2][None])[0])

    def surface_heights(self, xy: np.ndarray) -> np.ndarray:
        """Map DSM at many points (..., 2) -> (...); zeros without terrain."""
        xy = np.asarray(xy, float)
        if self.dsm is None or self.dem_grid is None:
            return np.zeros(xy.shape[:-1])
        return sample_bilinear(self.dsm, self.dem_grid, xy)

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
