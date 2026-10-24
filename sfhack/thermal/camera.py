"""Simulated downward LWIR camera (SIMULATOR SIDE: it reads world/thermal_truth.npz and the MuJoCo state).

    cam = ThermalCamera(sim.model)                 # camera "thermal" on the drone (falls back to "down")
    truth = load_truth(mission_dir)                # world/thermal_truth.npz
    img_K = cam.render(sim.data, truth)            # float32 (512, 640) apparent temperature, K
    img_8 = cam.to_8bit(img_K)                     # uint8 white-hot with AGC
    cam.intrinsics                                 # {fx, fy, cx, cy, w, h, hfov_deg, vfov_deg}

Image orientation = the MuJoCo "down" camera: image top = drone forward, image right = drone right.

How a frame is made: MuJoCo depth + segmentation render at render_scale (default 320 x 256) -> every pixel
back-projected to world xyz -> per geom kind: ground, roof slabs and tree crowns sample the truth raster at
(x, y) (roofs, canopy, cars, wet patches are in it), building walls and trunks get the wall temperature, placed
objects (world/target_scene.json, optional "temp_K") their own, the drone itself air temperature, sky ~air - 30 K;
a short LWIR air path (tau = exp(-range / 8 km)) -> upsample to 640 x 512 -> optics blur (diffraction spot of an
f/1 lens at 12 um pitch, sigma 0.8 px) -> vignetting (narcissus, -0.15 K at the corners) -> fixed-pattern
noise (per-pixel offset 80 mK, column 40 mK, gain 0.5 %) -> slow drift (Ornstein-Uhlenbeck offset, 0.25 K,
tau 90 s, reset by a flat-field correction every 180 s) -> temporal noise NETD 50 mK.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np

from ..missionprep.geometry import Grid

SURFACE, WALL, OBJECT, AIR = 0, 1, 2, 3


@dataclass
class ThermalTruth:
    temp_K: np.ndarray
    grid: Grid
    wall_K: float
    air_K: float
    meta: dict[str, Any] = field(default_factory=dict)
    object_K: dict[str, float] = field(default_factory=dict)  # geom name ("target<k>") -> K

    def sample(self, xy: np.ndarray) -> np.ndarray:
        """Bilinear lookup at ENU points (..., 2) with cv2.remap (fast for whole images)."""
        uv = self.grid.to_px(xy).astype(np.float32)
        return cv2.remap(self.temp_K, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def load_truth(path: str | Path) -> ThermalTruth:
    """Mission directory (or id) or the npz itself."""
    p = Path(path)
    if not p.exists() and not p.suffix:
        from ..missionprep.build import mission_dir

        p = mission_dir(str(path))
    if p.is_dir():
        p = p / "world" / "thermal_truth.npz"
    with np.load(p) as z:
        meta = json.loads(str(z["meta"]))
        temp = np.ascontiguousarray(z["temp_K"], np.float32)
        wall, air = float(z["wall_K"]), float(z["air_K"])
    e = meta["extent_enu"]
    grid = Grid(e["min_x"], e["max_y"], float(meta["res_m"]), temp.shape[0], temp.shape[1])
    objects: dict[str, float] = {}
    ts = p.parent / "target_scene.json"
    if ts.exists():
        try:
            for k, ob in enumerate(json.loads(ts.read_text(encoding="utf-8")).get("objects", [])):
                objects[f"target{k}"] = float(ob.get("temp_K", air - 1.0))
        except (ValueError, OSError):
            pass
    return ThermalTruth(temp, grid, wall, air, meta, objects)


class ThermalCamera:
    def __init__(self, model: mujoco.MjModel, name: str = "thermal", width: int = 640, height: int = 512,
                 hfov_deg: float = 50.0, render_scale: float = 0.5, netd_K: float = 0.05, seed: int = 0,
                 ffc_period_s: float = 180.0) -> None:
        self.model = model
        self.w, self.h = width, height
        self.hfov = hfov_deg
        self.vfov = math.degrees(2 * math.atan(math.tan(math.radians(hfov_deg) / 2) * height / width))
        try:
            self.cam_id = model.camera(name).id
            self.fovy_override = abs(float(model.cam_fovy[self.cam_id]) - self.vfov) > 0.05
        except KeyError:
            self.cam_id = model.camera("down").id  # same mount and orientation; fovy set while rendering
            self.fovy_override = True
        self.rw, self.rh = max(16, int(round(width * render_scale))), max(16, int(round(height * render_scale)))
        self.netd = netd_K
        self.rng = np.random.default_rng(seed + 4000)
        self.renderer: mujoco.Renderer | None = None
        self.kind, self.geom_offset = self._geom_kinds()
        self.geom_names = [model.geom(i).name for i in range(model.ngeom)]
        # static sensor non-uniformity
        r = self.rng
        self.fpn = (r.normal(0, 0.08, (height, width)) + r.normal(0, 0.04, (1, width))).astype(np.float32)
        self.gain = (1.0 + r.normal(0, 0.005, (height, width))).astype(np.float32)
        yy, xx = np.mgrid[0:height, 0:width]
        rr = ((xx - width / 2) ** 2 + (yy - height / 2) ** 2) / ((width / 2) ** 2 + (height / 2) ** 2)
        self.vignette = (-0.15 * rr).astype(np.float32)
        self.drift = 0.0
        self.drift_tau, self.drift_std = 90.0, 0.25
        self.ffc_period = ffc_period_s
        self.last_t: float | None = None
        self.last_ffc = 0.0
        self._agc: tuple[float, float] | None = None
        # rays of the internal render (camera frame: x right, y up, looking along -z)
        fy = (self.rh / 2) / math.tan(math.radians(self.vfov) / 2)
        u = (np.arange(self.rw) + 0.5 - self.rw / 2) / fy
        v = (np.arange(self.rh) + 0.5 - self.rh / 2) / fy
        U, V = np.meshgrid(u, v)
        self.rays = np.stack([U, -V, -np.ones_like(U)], -1).astype(np.float32)  # z-depth 1
        self.ray_len = np.linalg.norm(self.rays, axis=-1)

    # ------------------------------------------------------------------ public

    @property
    def intrinsics(self) -> dict[str, float]:
        fy = (self.h / 2) / math.tan(math.radians(self.vfov) / 2)
        fx = (self.w / 2) / math.tan(math.radians(self.hfov) / 2)
        return {"fx": fx, "fy": fy, "cx": self.w / 2, "cy": self.h / 2, "w": self.w, "h": self.h,
                "hfov_deg": self.hfov, "vfov_deg": self.vfov}

    def render(self, data: mujoco.MjData, thermal_truth: ThermalTruth | str | Path) -> np.ndarray:
        truth = thermal_truth if isinstance(thermal_truth, ThermalTruth) else self._truth_cached(thermal_truth)
        scene = self.scene_temperature(data, truth)
        img = cv2.resize(scene, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        img = cv2.GaussianBlur(img, (0, 0), 0.8)
        self._advance_drift(float(data.time))
        img = (img - 293.15) * self.gain + 293.15 + self.fpn + self.vignette + self.drift
        img += self.rng.normal(0, self.netd, img.shape).astype(np.float32)
        return img.astype(np.float32)

    def to_8bit(self, img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0, min_span_K: float = 4.0,
                smooth: float = 0.3) -> np.ndarray:
        """Linear AGC on the 1-99 % histogram range (at least min_span_K wide), smoothed over frames so the
        brightness does not flicker; white-hot."""
        lo, hi = np.percentile(img[::4, ::4], [lo_pct, hi_pct])
        if hi - lo < min_span_K:
            mid = (hi + lo) / 2
            lo, hi = mid - min_span_K / 2, mid + min_span_K / 2
        if self._agc is not None:
            lo = self._agc[0] + smooth * (lo - self._agc[0])
            hi = self._agc[1] + smooth * (hi - self._agc[1])
        self._agc = (float(lo), float(hi))
        return np.clip((img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    # ------------------------------------------------------------------ internals

    _truth_cache: dict[str, ThermalTruth] = {}

    def _truth_cached(self, p: str | Path) -> ThermalTruth:
        key = str(p)
        if key not in self._truth_cache:
            self._truth_cache[key] = load_truth(p)
        return self._truth_cache[key]

    def _geom_kinds(self) -> tuple[np.ndarray, np.ndarray]:
        m = self.model
        kind = np.full(m.ngeom + 1, SURFACE, np.int8)  # last slot: background (sky)
        off = np.zeros(m.ngeom + 1, np.float32)
        drone = m.body("x2").id if "x2" in [m.body(i).name for i in range(m.nbody)] else -1
        rng = np.random.default_rng(17)
        for i in range(m.ngeom):
            g = m.geom(i)
            body = int(m.geom_bodyid[i])
            root = int(m.body_rootid[body])
            if drone >= 0 and (body == drone or root == drone):
                kind[i] = AIR
                continue
            name = g.name
            mesh = m.mesh(int(m.geom_dataid[i])).name if int(m.geom_type[i]) == mujoco.mjtGeom.mjGEOM_MESH else ""
            if name.startswith("target"):
                kind[i] = OBJECT
            elif mesh.startswith("bw"):
                kind[i] = WALL
                off[i] = rng.normal(0, 1.0)
            elif int(m.geom_type[i]) == mujoco.mjtGeom.mjGEOM_CYLINDER and int(m.geom_group[i]) == 1:
                kind[i] = WALL  # tree trunk: about air temperature
                off[i] = -1.0
        return kind, off

    def _advance_drift(self, t: float) -> None:
        if self.last_t is None or t < self.last_t:
            self.last_t, self.last_ffc, self.drift = t, t, 0.0
            return
        dt = t - self.last_t
        self.last_t = t
        if t - self.last_ffc >= self.ffc_period:
            self.last_ffc, self.drift = t, 0.0  # flat-field correction (shutter)
            return
        a = math.exp(-dt / self.drift_tau)
        self.drift = a * self.drift + math.sqrt(max(1 - a * a, 0.0)) * self.drift_std * float(self.rng.normal())

    def scene_temperature(self, data: mujoco.MjData, truth: ThermalTruth) -> np.ndarray:
        """Noise-free apparent temperature at the internal resolution (rh, rw)."""
        m = self.model
        if self.renderer is None:
            self.renderer = mujoco.Renderer(m, self.rh, self.rw)
        r = self.renderer
        saved = float(m.cam_fovy[self.cam_id])
        if self.fovy_override:
            m.cam_fovy[self.cam_id] = self.vfov
        try:
            r.update_scene(data, camera=self.cam_id)
            r.enable_depth_rendering()
            depth = r.render().copy()
            r.disable_depth_rendering()
            r.enable_segmentation_rendering()
            seg = r.render().copy()
            r.disable_segmentation_rendering()
        finally:
            m.cam_fovy[self.cam_id] = saved
        gid = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
        gidx = np.where(gid >= 0, gid, m.ngeom)
        kind = self.kind[gidx]
        pos = data.cam_xpos[self.cam_id]
        R = data.cam_xmat[self.cam_id].reshape(3, 3)
        pts = pos + (self.rays * depth[..., None]) @ R.T
        out = truth.sample(pts[..., :2])
        out = np.where(kind == WALL, truth.wall_K + self.geom_offset[gidx], out)
        out = np.where(kind == AIR, truth.air_K, out)
        if (kind == OBJECT).any():
            obj_t = np.array([truth.object_K.get(n, truth.air_K - 1.0) for n in self.geom_names] + [truth.air_K],
                             np.float32)
            out = np.where(kind == OBJECT, obj_t[gidx], out)
        out = np.where(gid < 0, truth.air_K - 30.0, out)  # sky
        rng_m = depth * self.ray_len
        tau = np.exp(-rng_m / 8000.0)
        return (tau * out + (1 - tau) * truth.air_K).astype(np.float32)
