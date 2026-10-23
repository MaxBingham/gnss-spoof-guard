"""V4 target landing, simulator side: the target scene in MuJoCo, plus the flight extension that wires the
feature into a `FlightSystem` and scores it on truth.

- `TargetWorld`: decorates a base world (Berkeley package or the plain field) with the six objects of a
  `TargetLayout` (sfhack/world/target_scene.py) and the landing pad with its ArUco marker texture.
- `TargetExtension`: the `FlightSystem(extension=...)` hook. It builds the scene, picks the spawn point, gives
  the autopilot a `TargetMission` (onboard: it gets only the mission brief, the onboard map and the bus), and
  scores the flight on the TRUTH layout (harness only: chosen object, landing object, pad error).

Truth stays here: onboard code never imports this module or the layout (tests/test_target_truth.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from ..world.target_scene import Part, SceneObject, TargetLayout, marker_texture


def _yaw_quat(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _quat_mul(a: list[float], b: list[float]) -> list[float]:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out.tolist()


class TargetWorld:
    """Base world + target scene. `base` is anything with add_to_spec (PackageWorld); its building and tree
    landmarks (if any) are available as keep-out circles via `keep_out_circles`."""

    def __init__(self, base: Any, layout: TargetLayout) -> None:
        self.base = base
        self.layout = layout

    def __getattr__(self, name: str) -> Any:  # a_enu, b_enu, landmarks, ... of the base world
        return getattr(self.base, name)

    def add_to_spec(self, spec: mujoco.MjSpec) -> None:
        if self.base is not None:
            self.base.add_to_spec(spec)
        lay = self.layout
        tex = marker_texture(lay.marker_dict, lay.marker_id, lay.pad_size, lay.marker_size)
        t = spec.add_texture(name="v4_marker", type=mujoco.mjtTexture.mjTEXTURE_2D)
        t.width, t.height, t.nchannel = tex.shape[1], tex.shape[0], 3
        t.data = tex.tobytes()
        mat = spec.add_material(name="v4_marker", texrepeat=[1, 1], texuniform=False, reflectance=0.0)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "v4_marker"
        for obj in lay.objects:
            for k, part in enumerate(obj.parts):
                self._add_part(spec, obj, k, part)

    def _add_part(self, spec: mujoco.MjSpec, obj: SceneObject, k: int, p: Part) -> None:
        world = spec.worldbody
        x, y = obj.to_world(np.array(p.pos[:2]))
        pos = [float(x), float(y), float(p.pos[2])]
        quat = _yaw_quat(obj.yaw)
        rgba = [*p.rgb, 1.0]
        name = f"v4_{obj.kind}_{obj.id}_{k}"
        col = {} if p.collide else {"contype": 0, "conaffinity": 0}
        G = mujoco.mjtGeom
        if p.shape == "box":
            world.add_geom(name=name, type=G.mjGEOM_BOX, size=list(p.size), pos=pos, quat=quat, rgba=rgba, **col)
        elif p.shape == "cylinder":
            q = quat if p.axis == "z" else _quat_mul(quat, [math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0])
            world.add_geom(name=name, type=G.mjGEOM_CYLINDER, size=[p.size[0], p.size[1], 0], pos=pos, quat=q,
                           rgba=rgba, **col)
        elif p.shape == "ellipsoid":
            world.add_geom(name=name, type=G.mjGEOM_ELLIPSOID, size=list(p.size), pos=pos, quat=quat, rgba=rgba, **col)
        elif p.shape in ("pyramid", "prism"):
            hx, hy, h = p.size
            base = [[hx, hy, 0], [-hx, hy, 0], [-hx, -hy, 0], [hx, -hy, 0]]
            top = [[0, 0, h]] if p.shape == "pyramid" else [[hx, 0, h], [-hx, 0, h]]
            mesh = spec.add_mesh(name=name)
            mesh.uservert = np.array(base + top, float).flatten().tolist()
            world.add_geom(name=name, type=G.mjGEOM_MESH, meshname=name, pos=pos, quat=quat, rgba=rgba, **col)
        elif p.shape == "pad":
            world.add_geom(name=name, type=G.mjGEOM_BOX, size=list(p.size), pos=pos, quat=quat, rgba=rgba)
            # marker plane on the pad top (visual only; the pad box carries the contact)
            world.add_geom(name=name + "_marker", type=G.mjGEOM_PLANE, size=[p.size[0], p.size[1], 1.0],
                           pos=[pos[0], pos[1], pos[2] + p.size[2] + 0.003], quat=quat, material="v4_marker",
                           contype=0, conaffinity=0)
        else:
            raise ValueError(f"unknown part shape {p.shape!r}")


def keep_out_circles(base: Any, margin: float = 4.0) -> list[tuple[float, float, float]]:
    """Buildings and mapped trees of a PackageWorld as (x, y, radius) circles (the objects must not overlap them)."""
    from ..world.scene import building_boxes
    out: list[tuple[float, float, float]] = []
    lms = getattr(base, "landmarks", None) or []
    for b in building_boxes(lms):
        out.append((b["center"][0], b["center"][1], math.hypot(*b["half"]) + margin))
    for lm in lms:
        if lm["type"] == "tree":
            out.append((lm["centroid_enu"][0], lm["centroid_enu"][1], float(lm.get("radius", 3.0)) + margin))
    return out
