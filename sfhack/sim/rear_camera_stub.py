"""Rear NIR camera STUB + stub base lights (recon-nav, until recon-world's sim/rear_camera.py and world/base.py land).

SIMULATOR SIDE: reads the true drone pose from MuJoCo and the TRUE light positions; publishes gray CameraFrame
images (name "rear") on `camera/rear`. Onboard code only sees the frames and the surveyed light list.

Image model: dark NIR frame (background ~6 gray levels + noise), every light that is on, in its beam, in the
field of view and not hidden by the terrain (mj_ray) is a Gaussian spot whose brightness falls with range^2
(saturating near the base), plus centroid shimmer. Steady distractor lights (a village, a fire) blink no code.
The base clock is unknown to the drone: a per-seed offset shifts every code. The camera mount has a small
per-seed misalignment (calibration error), which the onboard heading learned from the lights inherits.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np

from .. import bus as topics
from ..config import RearCameraConfig
from ..messages import CameraFrame
from ..perception.light_localizer import BaseLight, intrinsics, rear_mount

LIGHT_HEIGHT_M = 3.0


def make_codes(n: int, bits: int = 10, min_ones: int = 5, max_ones: int = 7) -> list[str]:
    """n cyclic codes, no two equal under rotation, cyclic Hamming distance >= 2 between any rotations
    (one wrong sample cannot turn one light into another), never all on (a steady light matches none).
    8 bits give only 10 such codes with 4-6 ones (6 with 5-6): 10 bits (1 s at 10 Hz) give 22 with 5-7 ones."""
    def rots(c: str) -> list[str]:
        return [c[i:] + c[:i] for i in range(len(c))]

    out: list[str] = []
    for combo in itertools.product("01", repeat=bits):
        c = "".join(combo)
        if not (min_ones <= c.count("1") <= max_ones) or "000" in c + c[:2]:
            continue
        if any(min(sum(a != b for a, b in zip(r, o)) for r in rots(c)) < 2 for o in out):
            continue
        if any(r < c for r in rots(c)):  # canonical rotation only
            continue
        out.append(c)
        if len(out) == n:
            return out
    raise ValueError(f"only {len(out)} codes of {bits} bits")


def stub_base_lights(center: np.ndarray, ground_z: Callable[[float, float], float], half: float = 75.0,
                     n_fence: int = 12, n_outer: int = 0, outer_r: float = 400.0, bit_hz: float = 10.0,
                     beam_half_deg: float = 80.0) -> list[BaseLight]:
    """TRUE lights: n_fence on a square fence (2 * half wide) every 360 / n_fence deg around the centre, beams
    facing outward; optional outer ring of posts at outer_r. z = true ground + LIGHT_HEIGHT_M."""
    codes = make_codes(n_fence + n_outer)
    out = []
    for k in range(n_fence):
        a = 2 * math.pi * k / n_fence
        d = np.array([math.cos(a), math.sin(a)])
        s = half / max(abs(d[0]), abs(d[1]))
        xy = center[:2] + d * s
        out.append(BaseLight(f"F{k:02d}", np.array([xy[0], xy[1], ground_z(*xy) + LIGHT_HEIGHT_M]), codes[k], bit_hz,
                             "fence", math.degrees(a), beam_half_deg))
    for k in range(n_outer):
        a = 2 * math.pi * (k + 0.5) / n_outer
        xy = center[:2] + outer_r * np.array([math.cos(a), math.sin(a)])
        out.append(BaseLight(f"O{k:02d}", np.array([xy[0], xy[1], ground_z(*xy) + LIGHT_HEIGHT_M]), codes[n_fence + k],
                             bit_hz, "outer", math.degrees(a), beam_half_deg))
    return out


def surveyed(lights: list[BaseLight], std_m: float, seed: int) -> list[BaseLight]:
    """The onboard list: true positions + survey error."""
    r = np.random.default_rng(seed + 9100)
    return [BaseLight(lt.id, lt.pos + r.normal(0.0, std_m, 3), lt.code, lt.bit_hz, lt.ring, lt.facing_deg,
                      lt.beam_half_deg) for lt in lights]


def lights_to_json(lights: list[BaseLight], path: Path) -> None:
    path.write_text(json.dumps({"lights": [{"id": lt.id, "enu": [round(float(v), 3) for v in lt.pos], "code": lt.code,
                                            "bit_hz": lt.bit_hz, "ring": lt.ring, "facing_deg": lt.facing_deg,
                                            "beam_half_deg": lt.beam_half_deg} for lt in lights]}, indent=1),
                    encoding="utf-8")


class RearCameraStub:
    def __init__(self, sim: Any, bus: Any, lights: list[BaseLight], cam: RearCameraConfig | None = None, seed: int = 0,
                 distractors: list[np.ndarray] | None = None, brightness_2km: float = 120.0,
                 mount_err_deg: float = 0.05, shimmer_px: float = 0.15, psf_px: float = 0.8) -> None:
        self.sim, self.bus = sim, bus
        self.lights = lights
        self.cam = cam or RearCameraConfig()
        self.r = np.random.default_rng(seed + 9200)
        self.R_bc = rear_mount(self.cam.pitch_down_deg)
        err = np.radians(self.r.normal(0.0, mount_err_deg, 3))
        th = float(np.linalg.norm(err))
        K = np.array([[0, -err[2], err[1]], [err[2], 0, -err[0]], [-err[1], err[0], 0]])
        self.R_bc_true = (np.eye(3) + math.sin(th) / max(th, 1e-12) * K + (1 - math.cos(th)) / max(th, 1e-12) ** 2 * K @ K) @ self.R_bc
        self.f, self.cx, self.cy = intrinsics(self.cam)
        self.clock_offset = float(self.r.uniform(0.0, 10.0))
        self.distractors = [np.asarray(d, float) for d in (distractors or [])]
        self.b2k = brightness_2km
        self.shimmer, self.psf = shimmer_px, psf_px
        h, w = self.cam.height, self.cam.width
        self.noise = [np.clip(6.0 + self.r.normal(0.0, 2.0, (h, w)), 0, 255).astype(np.uint8) for _ in range(4)]
        self.next_t = 0.0
        self.seq = 0
        self.last: CameraFrame | None = None
        self.last_visible: list[str] = []  # harness / display: ids drawn in the last frame (on or off)
        self._geomid = np.zeros(1, np.int32)

    def _visible(self, cam_pos: np.ndarray, p: np.ndarray) -> bool:
        m, d = self.sim.model, self.sim.data
        v = p - cam_pos
        dist = float(np.linalg.norm(v))
        r = mujoco.mj_ray(m, d, cam_pos, v / dist, None, 1, self.sim.body_id, self._geomid)
        return r < 0 or r > dist - 2.0

    def _spot(self, img: np.ndarray, u: float, v: float, amp: float) -> None:
        sig = self.psf * (1.0 + max(0.0, amp / 255.0 - 1.0) ** 0.5)  # saturated lights bloom
        amp = min(amp, 255.0 * 4)
        r = int(math.ceil(3 * sig)) + 1
        h, w = img.shape
        x0, x1, y0, y1 = int(u) - r, int(u) + r + 1, int(v) - r, int(v) + r + 1
        if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
            return
        xs, ys = np.arange(max(x0, 0), min(x1, w)), np.arange(max(y0, 0), min(y1, h))
        g = amp * np.exp(-(((xs[None, :] + 0.5 - u) ** 2) + ((ys[:, None] + 0.5 - v) ** 2)) / (2 * sig * sig))
        sub = img[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1]
        img[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1] = np.minimum(sub + g, 255.0)

    def step(self) -> None:
        t = self.sim.time
        if t + 1e-9 < self.next_t:
            return
        self.next_t = t + 1.0 / self.cam.hz
        self.seq += 1
        d = self.sim.data
        bid = self.sim.body_id
        R_wb = d.xmat[bid].reshape(3, 3)
        R_wc = R_wb @ self.R_bc_true
        cam_pos = d.xpos[bid] + R_wb @ np.array([-0.15, 0.0, 0.0])
        img = self.noise[self.seq % 4].astype(np.float32)
        img = np.roll(img, int(self.r.integers(0, 50)), axis=1)
        tb = t + self.clock_offset
        drawn = []
        pts = [(lt.pos, lt) for lt in self.lights] + [(p, None) for p in self.distractors]
        for p, lt in pts:
            c = R_wc.T @ (p - cam_pos)
            if c[2] <= 1.0:
                continue
            u = self.cx + self.f * c[0] / c[2] + self.r.normal(0.0, self.shimmer)
            v = self.cy + self.f * c[1] / c[2] + self.r.normal(0.0, self.shimmer)
            if not (-5 <= u <= self.cam.width + 5 and -5 <= v <= self.cam.height + 5):
                continue
            if lt is not None and lt.facing_deg is not None:
                to_cam = cam_pos[:2] - p[:2]
                off = math.degrees(math.atan2(to_cam[1], to_cam[0])) - lt.facing_deg
                if abs((off + 180.0) % 360.0 - 180.0) > lt.beam_half_deg:
                    continue
            if not self._visible(cam_pos, p):
                continue
            if lt is not None:
                drawn.append(lt.id)
                if lt.code[int(math.floor(tb * lt.bit_hz)) % len(lt.code)] != "1":
                    continue
            dist = float(np.linalg.norm(p - cam_pos))
            self._spot(img, u, v, self.b2k * (2000.0 / dist) ** 2)
        self.last_visible = drawn
        self.last = CameraFrame(t, self.seq, "rear", img.astype(np.uint8), self._fovy())
        self.bus.publish(topics.CAMERA_REAR, self.last)

    def _fovy(self) -> float:
        return math.degrees(2 * math.atan(self.cam.height / 2.0 / self.f))
