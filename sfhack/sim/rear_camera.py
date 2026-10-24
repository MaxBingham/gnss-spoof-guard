"""Rear-facing narrow-FOV near-infrared camera (SIMULATOR SIDE). Publishes `NirFrame` on `camera/rear`.

    sensor = RearCameraSensor(sim, bus, world_dir, seed=seed)   # after MujocoSim; call sensor.step() every sim step
    sensor.last                                                 # NirFrame (what the drone gets)
    sensor.last_truth                                           # per light: pixel, visible, on, range (display only)

The model (an 850 nm-filtered CMOS camera at night, 640 x 480, 20 deg HFOV, 4 ms exposure, 20 Hz):
- Lights are point sources, much smaller than a pixel at any range that matters (0.03 deg/px = 1 m at 2 km), so
  they are not rasterised by MuJoCo but projected with the TRUE camera pose: flux = I x beam lobe / r^2 x
  atmospheric transmission (exp(-3.0 r / V), V = visibility, 20 km clear night), spread by the optics PSF
  (Gaussian, sigma 0.9 px) plus a faint wide halo (lens scatter). Clipping at 255 makes the saturated core grow
  with brightness (bloom), as on a real sensor.
- Occlusion: a ray from the camera to each light against the scene (terrain, buildings, trees; not the base's own
  decoration in geom group 2): a light behind a hill or a house is gone.
- Blink: each light's code on the base clock (sim/base_lights.py), integrated over the exposure.
- Gimbal: one tilt axis, commanded onboard with CameraTiltCmd on `camera/rear/tilt` (60 deg/s, -10..70 deg below
  body -x); the frame carries the tilt and R_body_cam. A fixed mount saw the base only between ~0.8 and 2 km
  (at 80 m AGL the base is 18 deg down at 250 m, 2-4 deg down at 2 km; the FOV is 15 deg tall).
- Background: the MuJoCo scene through this camera, very dark (night NIR: faint terrain / sky glow), plus a dark
  offset, shot noise, read noise and a few hot pixels. Stray lights (lamps, a fire) appear as spots too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np

from .. import bus as topics
from ..messages import CameraTiltCmd, NirFrame
from .base_lights import TrueLights

CV_FROM_MJ = np.diag([1.0, -1.0, -1.0])  # MuJoCo camera (x right, y up, looks along -z) -> OpenCV (y down, z fwd)
TILT_RANGE = (-10.0, 70.0)  # deg below body -x
TILT_RATE = 60.0  # deg/s


def mount_quat(tilt_deg: float) -> np.ndarray:
    """MuJoCo camera quaternion in the body: optical axis body -x pitched tilt_deg down, image right = body +y."""
    p = math.radians(tilt_deg)
    x, y = np.array([0.0, 1.0, 0.0]), np.array([-math.sin(p), 0.0, math.cos(p)])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([x, y, np.cross(x, y)]).flatten())
    return q


@dataclass
class RearCameraModel:
    flux_1km: float = 4000.0  # DN x px of a fence lamp on its beam axis at 1 km, clear air
    visibility_km: float = 20.0
    psf_sigma_px: float = 0.9
    halo_frac: float = 0.02
    halo_sigma_px: float = 5.0
    exposure_s: float = 0.004
    dark_dn: float = 3.0
    read_noise_dn: float = 1.5
    shot_gain: float = 0.25  # DN per photo-electron (shot noise std = sqrt(signal * gain))
    background_gain: float = 0.05  # MuJoCo RGB -> night NIR DN (0 = no background render)
    background_every_s: float = 0.5  # the faint background is re-rendered this often (a full-scene render ~50 ms)
    hot_pixels: int = 25


class RearCamera:
    def __init__(self, model: mujoco.MjModel, lights: TrueLights, width: int = 640, height: int = 480,
                 hfov_deg: float = 20.0, name: str = "rear", params: RearCameraModel | None = None,
                 seed: int = 0, body: str = "x2") -> None:
        self.model, self.lights, self.name = model, lights, name
        self.p = params or RearCameraModel()
        self.w, self.h = width, height
        self.cam_id = model.camera(name).id
        self.body_id = model.body(body).id
        fx = width / 2 / math.tan(math.radians(hfov_deg) / 2)
        self.K = {"fx": fx, "fy": fx, "cx": width / 2, "cy": height / 2, "w": float(width), "h": float(height),
                  "hfov_deg": hfov_deg, "vfov_deg": math.degrees(2 * math.atan(height / 2 / fx))}
        # one-axis gimbal (tilt); camera axes in the body frame follow it
        self.tilt_deg = 0.0
        self.R_body_cam = np.eye(3)
        self.set_tilt(self._model_tilt())
        self.rng = np.random.default_rng(seed + 9600)
        self.hot = (self.rng.integers(0, height, self.p.hot_pixels), self.rng.integers(0, width, self.p.hot_pixels),
                    self.rng.uniform(20, 120, self.p.hot_pixels))
        self.group_mask = np.array([1, 1, 0, 1, 1, 1], np.uint8)  # rays / background: everything but group 2
        self.scene_opt = mujoco.MjvOption()
        self.scene_opt.geomgroup[:] = [1, 1, 0, 0, 0, 0]
        self.last_truth: list[dict[str, Any]] = []
        ys, xs = np.mgrid[0:height, 0:width]
        self._grid = (xs.astype(np.float32), ys.astype(np.float32))
        # temporal noise: a bank of unit-normal frames, picked and shifted at random (fresh draws cost ~8 ms)
        self.noise_bank = self.rng.normal(0.0, 1.0, (6, height, width)).astype(np.float32)
        self._bg: np.ndarray | None = None
        self._bg_t = -math.inf
        self._occ: dict[str, tuple[np.ndarray, bool]] = {}  # light id -> (camera position, occluded): rays ~0.6 ms

    def _model_tilt(self) -> float:
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.model.cam_quat[self.cam_id])
        y = R.reshape(3, 3)[:, 1]  # camera up axis in the body: (-sin p, 0, cos p)
        return math.degrees(math.atan2(-y[0], y[2]))

    def set_tilt(self, tilt_deg: float) -> None:
        """Move the gimbal (instantly; RearCameraSensor slews it). Changes the model camera (the view)."""
        self.tilt_deg = float(np.clip(tilt_deg, *TILT_RANGE))
        q = mount_quat(self.tilt_deg)
        self.model.cam_quat[self.cam_id] = q
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        self.R_body_cam = R.reshape(3, 3) @ CV_FROM_MJ

    @property
    def intrinsics(self) -> dict[str, float]:
        return dict(self.K)

    # ------------------------------------------------------------------ render

    def render(self, data: mujoco.MjData, renderer: mujoco.Renderer | None = None) -> np.ndarray:
        """uint8 (H, W) frame at data.time; fills self.last_truth."""
        p = self.p
        img = np.full((self.h, self.w), p.dark_dn, np.float32)
        mujoco.mj_camlight(self.model, data)  # the camera pose after a gimbal move (cam_xpos / cam_xmat)
        if renderer is not None and p.background_gain > 0:
            t = float(data.time)
            if self._bg is None or not (0.0 <= t - self._bg_t < p.background_every_s):
                self._bg, self._bg_t = self._background(data, renderer), t
            img += self._bg
        cam_pos = data.cam_xpos[self.cam_id].copy()
        R_wc = data.cam_xmat[self.cam_id].reshape(3, 3) @ CV_FROM_MJ  # columns: cam axes in world
        K = self.K
        self.last_truth = []
        for e in self.lights.emitters(float(data.time), cam_pos, p.exposure_s):
            d = e["xyz"] - cam_pos
            rng = float(np.linalg.norm(d))
            pc = R_wc.T @ d
            rec = {"id": e["id"], "kind": e["kind"], "nav": e["nav"], "range_m": rng, "u": None, "v": None,
                   "in_view": False, "occluded": False, "on": e["I"] > 0, "peak_dn": 0.0, "peak_on_dn": 0.0}
            self.last_truth.append(rec)
            if pc[2] <= 1.0:
                continue
            u, v = K["fx"] * pc[0] / pc[2] + K["cx"], K["fy"] * pc[1] / pc[2] + K["cy"]
            rec["u"], rec["v"] = float(u), float(v)
            if not (-10 <= u < self.w + 10 and -10 <= v < self.h + 10):
                continue
            rec["in_view"] = True
            if self._occluded(data, cam_pos, d, rng, e["id"]):
                rec["occluded"] = True
                continue
            flux_on = p.flux_1km * e["I_on"] / (rng / 1000.0) ** 2 * math.exp(-3.0 * rng / 1000.0 / p.visibility_km)
            rec["peak_on_dn"] = flux_on / (2 * math.pi * p.psf_sigma_px ** 2)  # this light when on (truth stats)
            if e["I"] <= 0:
                continue
            flux = flux_on * e["I"] / max(e["I_on"], 1e-12)
            rec["peak_dn"] = flux / (2 * math.pi * p.psf_sigma_px ** 2)
            self._splat(img, u, v, flux)
        # noise: read + shot (sqrt of the signal; exact in the spots, the dark background's level elsewhere)
        k = int(self.rng.integers(len(self.noise_bank)))
        nz = np.roll(self.noise_bank[k], (int(self.rng.integers(self.h)), int(self.rng.integers(self.w))), axis=(0, 1))
        floor = float(np.sqrt(max(float(img.mean()), 0.0) * p.shot_gain + p.read_noise_dn ** 2))
        bright = img > 4.0 * p.dark_dn + 20.0
        img = img + nz * floor
        if bright.any():
            img[bright] += nz[bright] * np.sqrt(img[bright] * p.shot_gain)
        img[self.hot[0], self.hot[1]] += self.hot[2]
        return np.clip(img, 0, 255).astype(np.uint8)

    def _splat(self, img: np.ndarray, u: float, v: float, flux: float) -> None:
        p = self.p
        for frac, s in ((1.0 - p.halo_frac, p.psf_sigma_px), (p.halo_frac, p.halo_sigma_px)):
            peak = frac * flux / (2 * math.pi * s * s)
            if peak < 0.3:
                continue
            r = int(math.ceil(s * math.sqrt(2 * math.log(max(peak / 0.3, 1.0001)))))
            x0, x1 = max(int(u) - r, 0), min(int(u) + r + 2, self.w)
            y0, y1 = max(int(v) - r, 0), min(int(v) + r + 2, self.h)
            if x0 >= x1 or y0 >= y1:
                continue
            xs, ys = self._grid[0][y0:y1, x0:x1], self._grid[1][y0:y1, x0:x1]
            img[y0:y1, x0:x1] += peak * np.exp(-((xs + 0.5 - u) ** 2 + (ys + 0.5 - v) ** 2) / (2 * s * s))

    def _occluded(self, data: mujoco.MjData, cam_pos: np.ndarray, d: np.ndarray, rng: float, key: str) -> bool:
        c = self._occ.get(key)
        if c is not None and float(np.linalg.norm(c[0] - cam_pos)) < 1.0:  # re-cast after 1 m of camera motion
            return c[1]
        geomid = np.zeros(1, np.int32)
        hit = mujoco.mj_ray(self.model, data, cam_pos, d / rng, self.group_mask, 1, self.body_id, geomid)
        occ = 0.0 <= hit < rng - 1.0
        self._occ[key] = (cam_pos.copy(), occ)
        return occ

    def _background(self, data: mujoco.MjData, r: mujoco.Renderer) -> np.ndarray:
        r.update_scene(data, camera=self.cam_id, scene_option=self.scene_opt)
        rgb = r.render()
        H, W = rgb.shape[:2]
        cw = int(round(H * self.w / self.h))  # MuJoCo keeps the vertical FOV: crop a wider renderer to our aspect
        if cw < W:
            rgb = rgb[:, (W - cw) // 2:(W - cw) // 2 + cw]
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        g = cv2.resize(g, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        return cv2.GaussianBlur(g, (0, 0), 1.5) * self.p.background_gain


class RearCameraSensor:
    """Publishes a NirFrame at `hz` (call `step()` after every sim step, like sim/thermal_sensor.py)."""

    def __init__(self, sim: Any, bus: Any, world_dir: Path | str, seed: int = 0, hz: float | None = None,
                 params: RearCameraModel | None = None, lights: TrueLights | None = None) -> None:
        cam = sim.cfg.camera
        self.sim, self.bus = sim, bus
        self.hz = hz or cam.rear_hz
        self.lights = lights or TrueLights.load(world_dir, seed)
        self.camera = RearCamera(sim.model, self.lights, cam.rear_width, cam.rear_height, cam.rear_hfov_deg,
                                 params=params, seed=seed)
        self.next_t = 0.0
        self.seq = 0
        self.last: NirFrame | None = None
        self.last_truth: list[dict[str, Any]] = []
        self.tilt_cmd = self.camera.tilt_deg
        self._last_t: float | None = None
        bus.subscribe(topics.CAMERA_REAR_TILT, self._on_tilt)

    def _on_tilt(self, msg: CameraTiltCmd) -> None:
        self.tilt_cmd = float(np.clip(msg.tilt_deg, *TILT_RANGE))

    def step(self) -> None:
        t = self.sim.time
        if t + 1e-9 < self.next_t:
            return
        self.next_t = t + 1.0 / self.hz
        dt = 0.0 if self._last_t is None else max(t - self._last_t, 0.0)
        self._last_t = t
        if self.tilt_cmd != self.camera.tilt_deg:  # gimbal slew
            step = float(np.clip(self.tilt_cmd - self.camera.tilt_deg, -TILT_RATE * dt, TILT_RATE * dt))
            self.camera.set_tilt(self.camera.tilt_deg + step)
        if self.sim.renderer is None:
            self.sim.render("down")  # creates the simulator's renderer (one GL context for every camera)
        img = self.camera.render(self.sim.data, self.sim.renderer)
        self.seq += 1
        self.last = NirFrame(t, self.seq, "rear", img, self.camera.intrinsics, self.camera.R_body_cam.copy(),
                             self.camera.p.exposure_s, self.camera.tilt_deg)
        self.last_truth = self.camera.last_truth
        self.bus.publish(topics.CAMERA_REAR, self.last)
