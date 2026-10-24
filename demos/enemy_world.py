"""World for the enemy-drone tracker (harness / simulator side): a demos/50_base_3d.py world (farm by default) plus
the enemy drone, birds, a pool of drone variants (dataset only) and the gimbal camera of our drone.

Used by demos/52_track_enemy.py (live), demos/53_drone_dataset.py (synthetic training data) and
demos/56_track_eval.py (closed-loop evaluation). Nothing here runs onboard.
"""

from __future__ import annotations

import importlib

import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

DEMOS = Path(__file__).resolve().parent
ROOT = DEMOS.parent
for p in (str(DEMOS), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from sfhack.config import TrackerConfig  # noqa: E402
from sfhack.sim import enemy as en  # noqa: E402
from sfhack.sim.gimbal_camera import SegRenderer, boxes_from_seg, cam_quat, gimbal_forward  # noqa: E402,F401
from sfhack.sim.gimbal_camera import project as _project  # noqa: E402

CLASSES = ("drone", "bird")
TRK = TrackerConfig()
W, H = TRK.cam_w, TRK.cam_h
VFOV = TRK.vfov_deg()

# dataset drone variants (index 0 = the demo enemy); colours are re-drawn per sample
VARIANTS = [
    en.DroneStyle(),
    en.DroneStyle(motor_diag=0.42, prop_r=0.11, payload=False, body_half=(0.10, 0.05, 0.04)),
    en.DroneStyle(motor_diag=0.75, prop_r=0.17, payload=False, body_half=(0.12, 0.08, 0.05)),
    en.DroneStyle(motor_diag=1.05, prop_r=0.19, arms=6, payload=False, body_half=(0.15, 0.13, 0.07)),
    en.DroneStyle(motor_diag=1.6, prop_r=0.27, payload=True, body_half=(0.22, 0.15, 0.08)),
    en.DroneStyle(motor_diag=0.9, prop_r=0.16, arms=8, payload=True, body_half=(0.14, 0.14, 0.06)),
]
DATASET_BIRDS = [(f"bird{k}", kind) for k, kind in enumerate(k for k in en.BIRDS for _ in range(2))]


@dataclass
class EnemyWorld:
    model: mujoco.MjModel
    info: object  # 50_base_3d world info: name, ground(x, y), world (FarmWorld) ...
    mod: object  # the 50_base_3d module (PAD, CRUISE_AGL, UdpBearings, ...)
    birds: list[tuple[str, str]]
    drones: list[str]  # dataset drone bodies ("drone0" ...) or ["enemy"]
    name: str

    @property
    def ground(self):
        return self.info.ground


def build(world: str = "farm", n_birds: int = 16, seed: int = 0, dataset: bool = False) -> EnemyWorld:
    mod = importlib.import_module("50_base_3d")
    birds: list[tuple[str, str]] = []
    drones: list[str] = []

    def extra(spec: mujoco.MjSpec) -> None:
        nonlocal birds, drones
        if dataset:
            for k, st in enumerate(VARIANTS):
                en.add_drone(spec, f"drone{k}", st)
                drones.append(f"drone{k}")
            birds = list(DATASET_BIRDS)
            for name, kind in birds:
                en.add_bird(spec, name, en.BIRDS[kind])
        else:
            birds = en.add_scenario_bodies(spec, n_birds, seed)
            drones = ["enemy"]
        spec.worldbody.add_camera(name="gimbal", pos=[0, 0, -400], fovy=VFOV)

    model, info = mod.build_model(world, extra=extra)
    return EnemyWorld(model, info, mod, birds, drones, world)




def project(points: np.ndarray, cam_pos: np.ndarray, forward: np.ndarray, roll: float = 0.0) -> np.ndarray:
    """(N, 3) world points -> (N, 3) pixel u, v and depth, for the tracker camera (W x H, VFOV)."""
    return _project(points, cam_pos, forward, roll, W, H, VFOV)
