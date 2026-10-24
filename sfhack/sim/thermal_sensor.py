"""Downward thermal (LWIR) camera as a simulated sensor: renders the night scene from the TRUE thermal world
(<mission>/world/thermal_truth.npz) and publishes ThermalFrame messages on the `camera/thermal` topic.

SIMULATOR SIDE: it reads the MuJoCo state and the truth raster; onboard code only sees the published frames.

The renderer is night-thermal's `sfhack.thermal.camera.ThermalCamera` (MuJoCo depth + segmentation, roofs, walls,
NETD, fixed-pattern noise, drift) when that module is present. Without it a stub camera is used: every pixel
ray is intersected with the horizontal plane at the true ground height under the drone and samples the truth
raster there (no roofs / walls in 3-D, so roofs show at ground scale), plus NETD noise and a little blur.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np

from .. import bus as topics
from ..messages import ThermalFrame

HFOV_DEG = 50.0
WIDTH, HEIGHT = 640, 512


def pinhole(width: int, height: int, hfov_deg: float) -> dict[str, float]:
    fx = width / 2.0 / math.tan(math.radians(hfov_deg) / 2.0)
    return {"fx": fx, "fy": fx, "cx": width / 2.0, "cy": height / 2.0, "w": float(width), "h": float(height),
            "hfov_deg": hfov_deg, "vfov_deg": math.degrees(2 * math.atan(height / 2.0 / fx))}


class _TruthRaster:
    def __init__(self, path: Path) -> None:
        with np.load(path) as z:
            self.temp = np.ascontiguousarray(z["temp_K"], np.float32)
            meta = json.loads(str(z["meta"]))
        e = meta["extent_enu"]
        self.min_x, self.max_y, self.res = float(e["min_x"]), float(e["max_y"]), float(meta["res_m"])

    def sample(self, xy: np.ndarray) -> np.ndarray:
        u = ((xy[..., 0] - self.min_x) / self.res - 0.5).astype(np.float32)
        v = ((self.max_y - xy[..., 1]) / self.res - 0.5).astype(np.float32)
        return cv2.remap(self.temp, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


class StubThermalCamera:
    """Flat-ground thermal camera (fallback until the full renderer is merged)."""

    def __init__(self, model: mujoco.MjModel, ground_z: Any, name: str = "down", width: int = WIDTH,
                 height: int = HEIGHT, render_scale: int = 2, netd_K: float = 0.05, seed: int = 0) -> None:
        self.cam_id = model.camera(name).id
        self.ground_z = ground_z
        self.intrinsics = pinhole(width, height, HFOV_DEG)
        self.w, self.h, self.scale = width, height, render_scale
        k = pinhole(width // render_scale, height // render_scale, HFOV_DEG)
        u, v = np.meshgrid(np.arange(k["w"]) + 0.5, np.arange(k["h"]) + 0.5)
        # MuJoCo camera frame: x right, y up, looking along -z
        self.rays = np.stack([(u - k["cx"]) / k["fx"], -(v - k["cy"]) / k["fy"], -np.ones_like(u)], -1)
        self.netd = netd_K
        self.rng = np.random.default_rng(seed + 7000)

    def render(self, data: mujoco.MjData, truth: _TruthRaster) -> np.ndarray:
        p = data.cam_xpos[self.cam_id].copy()
        R = data.cam_xmat[self.cam_id].reshape(3, 3)
        d = self.rays @ R.T
        gz = self.ground_z(p[0], p[1])
        s = (gz - p[2]) / np.minimum(d[..., 2], -1e-3)
        xy = p[:2] + d[..., :2] * s[..., None]
        img = truth.sample(xy)
        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        img = cv2.GaussianBlur(img, (0, 0), 0.8)
        return (img + self.rng.normal(0.0, self.netd, img.shape)).astype(np.float32)


class ThermalSensor:
    """Publishes a ThermalFrame at `hz` from the simulator's current state (call `step()` after each sim step)."""

    def __init__(self, sim: Any, bus: Any, mission_dir: Path, world: Any, hz: float = 5.0, seed: int = 0,
                 renderer: str = "auto", contrast: float = 1.0, target_scene: Path | str | None = None) -> None:
        """`contrast`: scale the TRUE scene's temperatures around the air temperature (0.4 = a mild night: real
        ASTER night scenes over Berkeley suggest 1-3 K road contrast instead of the model's 5-10 K). The onboard
        prior is not changed, so this is also a prior-vs-reality contrast mismatch."""
        self.sim, self.bus, self.hz = sim, bus, hz
        self.next_t = 0.0
        self.seq = 0
        self.kind = "stub"
        truth_path = Path(mission_dir) / "world" / "thermal_truth.npz"
        if not truth_path.exists():
            raise FileNotFoundError(f"{truth_path} missing: build it with `python -m sfhack.thermal build`")
        cam = None
        self._make_full = None
        if renderer in ("auto", "full"):
            try:
                from ..thermal.camera import ThermalCamera, load_truth  # night-thermal's renderer

                self.truth = load_truth(truth_path, target_scene=target_scene)  # the scene the world was built with
                self.kind = "full"
                # Created at the first frame: it draws into the simulator's GL context (one context; switching
                # between two costs ~50 ms per frame on Windows). The camera's calibration error is per seed.
                self._make_full = lambda: ThermalCamera(sim.model, name="thermal", seed=seed, renderer=sim.renderer)
                cam = ThermalCamera(sim.model, name="thermal", seed=seed) if not hasattr(sim, "renderer") else None
            except ImportError:
                if renderer == "full":
                    raise
        if cam is None and self._make_full is None:
            cam = StubThermalCamera(sim.model, world.ground_z, seed=seed)
            self.truth = _TruthRaster(truth_path)
        if contrast != 1.0:
            if isinstance(self.truth, _TruthRaster):
                air = float(np.median(self.truth.temp))
                self.truth.temp = (air + contrast * (self.truth.temp - air)).astype(np.float32)
            else:
                air = float(self.truth.air_K)
                self.truth.temp_K = (air + contrast * (self.truth.temp_K - air)).astype(np.float32)
                self.truth.wall_K = air + contrast * (self.truth.wall_K - air)
        self.camera = cam
        self.intrinsics = {k: float(v) for k, v in cam.intrinsics.items()} if cam is not None else {}
        self.last: ThermalFrame | None = None

    def _ensure_camera(self) -> None:
        if self.camera is not None:
            return
        if self.sim.renderer is None:
            self.sim.render("down")  # creates the simulator's renderer (GL context)
        try:
            self.camera = self._make_full()
        except TypeError:  # older renderer without the shared-context argument
            from ..thermal.camera import ThermalCamera
            self.camera = ThermalCamera(self.sim.model, name="thermal")
        self.intrinsics = {k: float(v) for k, v in self.camera.intrinsics.items()}

    def step(self) -> None:
        t = self.sim.time
        if t + 1e-9 < self.next_t:
            return
        self.next_t = t + 1.0 / self.hz
        self.seq += 1
        self._ensure_camera()
        img = self.camera.render(self.sim.data, self.truth)
        self.last = ThermalFrame(t, self.seq, img, self.intrinsics)
        self.bus.publish(topics.THERMAL, self.last)
