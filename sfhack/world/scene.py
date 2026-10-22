"""MuJoCo worlds: decorate the Skydio X2 scene spec with ground, buildings and trees.

- `ArenaWorld`: the small random arena from fly_x2.py (no map package needed).
- `PackageWorld`: built from a map package: orthophoto-textured ground plane, buildings as
  collision boxes (oriented bounding box of the OSM footprint), trees as cylinder + sphere
  (visual only). The ground texture file is selectable, so the simulated "reality" can use
  different imagery than the onboard map (map-mismatch tests).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np


def _no_shadows(spec: mujoco.MjSpec) -> None:
    for light in spec.lights:
        light.castshadow = False


class ArenaWorld:
    """fly_x2.py's test arena: a few boxes ('buildings') and spheres ('trees') around a 15 m square."""

    def __init__(self, seed: int = 1, keep_free: list[tuple[float, float]] | None = None) -> None:
        self.seed = seed
        self.keep_free = keep_free or [(0, 0), (15, 0), (15, 15)]

    def add_to_spec(self, spec: mujoco.MjSpec) -> None:
        world = spec.worldbody
        rng = np.random.default_rng(self.seed)
        for i in range(12):
            x, y = rng.uniform(-5, 22, size=2)
            if min(math.hypot(x - fx, y - fy) for fx, fy in self.keep_free) < 3:
                continue
            h = rng.uniform(1.0, 3.5)
            world.add_geom(name=f"building{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                           size=[rng.uniform(0.6, 1.5), rng.uniform(0.6, 1.5), h / 2],
                           pos=[x, y, h / 2], rgba=[0.75, 0.75, 0.78, 1])
        for i in range(25):
            x, y = rng.uniform(-5, 22, size=2)
            world.add_geom(name=f"tree{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.5, 0, 0],
                           pos=[x, y, 0.5], rgba=[0.15, 0.55, 0.2, 1], contype=0, conaffinity=0)


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


def roof_image(ortho: np.ndarray, extent: dict[str, float], box: dict[str, Any], mpp: float = 0.5) -> np.ndarray:
    """Ortho crop under a building box, resampled into the box frame (row 0 = box +y edge, like the floor)."""
    hx, hy = box["half"]
    cx, cy = box["center"]
    W, H = max(2, int(round(2 * hx / mpp))), max(2, int(round(2 * hy / mpp)))
    a = -hx + (np.arange(W) + 0.5) * (2 * hx / W)
    b = hy - (np.arange(H) + 0.5) * (2 * hy / H)
    A, Bm = np.meshgrid(a, b)
    c, s = math.cos(box["yaw"]), math.sin(box["yaw"])
    X, Y = cx + A * c - Bm * s, cy + A * s + Bm * c
    h, w = ortho.shape[:2]
    u = (X - extent["min_x"]) / (extent["max_x"] - extent["min_x"]) * w - 0.5
    v = (extent["max_y"] - Y) / (extent["max_y"] - extent["min_y"]) * h - 0.5
    return np.ascontiguousarray(cv2.remap(ortho, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REPLICATE))


class PackageWorld:
    def __init__(self, pkg_dir: Path | str, texture_file: str | None = None,
                 render_trees: bool = True, roof_textures: bool = True) -> None:
        self.roof_textures = roof_textures
        self.dir = Path(pkg_dir).resolve()
        self.manifest: dict[str, Any] = json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))
        self.landmarks: list[dict[str, Any]] = json.loads((self.dir / "landmarks.json").read_text(encoding="utf-8"))
        self.texture_file = texture_file or self.manifest["ortho"]["file"]
        self.render_trees = render_trees

    @property
    def a_enu(self) -> np.ndarray:
        return np.array(self.manifest["a"]["enu"], float)

    @property
    def b_enu(self) -> np.ndarray:
        return np.array(self.manifest["b"]["enu"], float)

    def add_to_spec(self, spec: mujoco.MjSpec) -> None:
        _no_shadows(spec)
        e = self.manifest["ortho"]["extent_enu"]
        cx, cy = (e["min_x"] + e["max_x"]) / 2, (e["min_y"] + e["max_y"]) / 2
        hw, hh = (e["max_x"] - e["min_x"]) / 2, (e["max_y"] - e["min_y"]) / 2

        # MuJoCo only decodes PNG from files; pass the decoded pixels so JPEG packages work too.
        img = cv2.cvtColor(cv2.imread(str(self.dir / self.texture_file), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        tex = spec.add_texture(name="ortho", type=mujoco.mjtTexture.mjTEXTURE_2D,
                               width=img.shape[1], height=img.shape[0], nchannel=3)
        tex.data = img.tobytes()
        mat = spec.add_material(name="ortho", texrepeat=[1, 1], texuniform=False, reflectance=0.0)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
        floor = spec.geom("floor")
        floor.size = [hw, hh, 1.0]
        floor.pos = [cx, cy, 0.0]
        floor.material = "ortho"

        world = spec.worldbody
        mat_b = spec.add_material(name="building", rgba=[0.72, 0.71, 0.68, 1])
        mat_t = spec.add_material(name="tower", rgba=[0.85, 0.82, 0.74, 1])
        for box in building_boxes(self.landmarks):
            h = box["height"]
            quat = [math.cos(box["yaw"] / 2), 0, 0, math.sin(box["yaw"] / 2)]
            world.add_geom(name=f"lm{box['id']}", type=mujoco.mjtGeom.mjGEOM_BOX,
                           size=[box["half"][0], box["half"][1], h / 2],
                           pos=[box["center"][0], box["center"][1], h / 2], quat=quat,
                           material=mat_t.name if box["type"] == "tower" else mat_b.name)
            if self.roof_textures:
                # Roof = the orthophoto under the footprint (a real roof looks like its aerial photo; a blank
                # gray box top gave the down camera nothing to match over the whole campus core).
                rname = f"roof{box['id']}"
                rtex = spec.add_texture(name=rname, type=mujoco.mjtTexture.mjTEXTURE_2D)
                roof = roof_image(img, e, box)
                rtex.width, rtex.height, rtex.nchannel = roof.shape[1], roof.shape[0], 3
                rtex.data = roof.tobytes()
                rmat = spec.add_material(name=rname, texrepeat=[1, 1], texuniform=False, reflectance=0.0)
                rmat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = rname
                world.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[box["half"][0], box["half"][1], 1.0],
                               pos=[box["center"][0], box["center"][1], h + 0.02], quat=quat,
                               material=rname, contype=0, conaffinity=0)
        if self.render_trees:
            trunk = [0.36, 0.27, 0.18, 1]
            crown = [0.20, 0.42, 0.18, 1]
            for lm in self.landmarks:
                if lm["type"] != "tree":
                    continue
                x, y = lm["centroid_enu"]
                h, r = float(lm["height"]), float(lm.get("radius", 3.0))
                trunk_h = max(h - r, 1.0)
                world.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.25, trunk_h / 2, 0],
                               pos=[x, y, trunk_h / 2], rgba=trunk, contype=0, conaffinity=0, group=1)
                world.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[r, 0, 0],
                               pos=[x, y, trunk_h], rgba=crown, contype=0, conaffinity=0, group=1)
