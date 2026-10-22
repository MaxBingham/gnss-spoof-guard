"""The V3 onboard map as the drone carries it: surface model (DSM) + imagery + mission data.

Reads ONLY `manifest.json` and the onboard variant directory (`onboard/` or `onboard_heavy/`) of a V3 world
package (see sfhack.world.build_v3). The truth/ part of the package is the simulated reality and is never
opened here (tests/test_truth_separation.py checks the source).
No MuJoCo dependency: onboard code (perception, planning, autopilot) may import it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .scenarios import Scenario
from .surface import Grid

VARIANTS = ("onboard", "onboard_heavy")


class OnboardTerrainMap:
    def __init__(self, pkg_dir: Path | str, variant: str = "onboard") -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown onboard variant {variant!r} (have {VARIANTS})")
        self.dir = Path(pkg_dir)
        self.variant = variant
        self.manifest: dict[str, Any] = json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))
        self.scenario = Scenario.from_json(self.manifest["scenario"])
        info = self.manifest[variant]
        vdir = self.dir / variant
        self.dsm = Grid.load(vdir / info["dsm"])
        self.ortho: Grid | None = None  # grayscale imagery as a Grid (z = intensity 0..1), north-up
        self.ortho_rgb: np.ndarray | None = None
        if info.get("ortho"):
            o = info["ortho"]
            img = Image.open(vdir / o["file"]).convert("RGB")
            self.ortho_rgb = np.asarray(img)
            gray = np.asarray(img.convert("L"), np.float32) / 255.0
            x0, y0, x1, y1 = o["extent"]
            self.ortho = Grid(gray, x0, y1, (x1 - x0) / gray.shape[1])
        self.no_fly = [np.asarray(p, float) for p in self.manifest.get("no_fly_enu", [])]

    @property
    def name(self) -> str:
        return str(self.manifest["name"])

    @property
    def a_enu(self) -> np.ndarray:
        return np.array(self.manifest["a"]["enu"], float)

    @property
    def b_enu(self) -> np.ndarray:
        return np.array(self.manifest["b"]["enu"], float)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return self.dsm.extent

    def surface_z(self, xy: np.ndarray) -> np.ndarray:
        return self.dsm.sample(xy)

    def no_fly_mask(self, xy: np.ndarray) -> np.ndarray:
        """Boolean mask over points (..., 2): inside any no-fly polygon."""
        from matplotlib.path import Path as MplPath

        pts = np.asarray(xy, float)
        flat = pts.reshape(-1, 2)
        m = np.zeros(len(flat), bool)
        for p in self.no_fly:
            m |= MplPath(p).contains_points(flat)
        return m.reshape(pts.shape[:-1])
