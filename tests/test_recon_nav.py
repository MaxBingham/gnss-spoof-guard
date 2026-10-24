"""recon-nav: optical flow and the base-light localizer on synthetic data (no simulator)."""

from __future__ import annotations

import math

import cv2
import numpy as np

from sfhack.perception.optical_flow import R_BODY_CAM, FlowConfig, OpticalFlow
from sfhack.sim.thermal_sensor import pinhole


def _rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


class _Ground:
    """Smooth random temperature field on z = 0, 0.25 m cells, centred on the origin."""

    def __init__(self, seed: int = 0, n: int = 1600, res: float = 0.25, contrast: float = 2.0) -> None:
        r = np.random.default_rng(seed)
        f = cv2.GaussianBlur(r.normal(0, 1, (n, n)).astype(np.float32), (0, 0), 6.0)
        self.temp = (285.0 + contrast * f / f.std()).astype(np.float32)
        self.res, self.n = res, n

    def render(self, pos: np.ndarray, R: np.ndarray, intr: dict, scale: int = 2) -> np.ndarray:
        w, h = int(intr["w"]) // scale, int(intr["h"]) // scale
        u, v = np.meshgrid((np.arange(w) + 0.5) * scale, (np.arange(h) + 0.5) * scale)
        rays_c = np.stack([(u - intr["cx"]) / intr["fx"], -(v - intr["cy"]) / intr["fy"], -np.ones_like(u)], -1)
        d = rays_c @ (R @ R_BODY_CAM).T
        s = -pos[2] / np.minimum(d[..., 2], -1e-6)
        xy = pos[:2] + d[..., :2] * s[..., None]
        uu = (xy[..., 0] / self.res + self.n / 2 - 0.5).astype(np.float32)
        vv = (-xy[..., 1] / self.res + self.n / 2 - 0.5).astype(np.float32)
        img = cv2.remap(self.temp, uu, vv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return cv2.resize(img, (int(intr["w"]), int(intr["h"])), interpolation=cv2.INTER_LINEAR)


def test_flow_recovers_ground_velocity() -> None:
    g = _Ground()
    intr = pinhole(640, 512, 50.0)
    flow = OpticalFlow(FlowConfig(fpn_warmup_frames=1000))
    rng = np.random.default_rng(1)
    v_true = np.array([7.0, -4.0])
    errs = []
    pos = np.array([-40.0, 30.0, 80.0])
    for k in range(12):
        t = 0.2 * k
        R = _rot(rng.normal(0, 0.03), rng.normal(0, 0.03), 0.6 + rng.normal(0, 0.02))
        img = g.render(pos, R, intr) + rng.normal(0, 0.05, (512, 640)).astype(np.float32)
        m = flow.update(t, img, R, pos[2], intr)
        if m is not None:
            errs.append(m.vel - v_true)
        pos = pos + np.r_[v_true * 0.2, 0.0]
    errs = np.array(errs)
    assert len(errs) >= 10
    assert np.abs(errs.mean(0)).max() < 0.15, errs.mean(0)
    assert np.abs(errs).max() < 0.5


def test_flow_rejects_uniform_scene() -> None:
    intr = pinhole(640, 512, 50.0)
    flow = OpticalFlow()
    rng = np.random.default_rng(2)
    n = 0
    for k in range(6):
        img = (285.0 + rng.normal(0, 0.03, (512, 640))).astype(np.float32)
        n += flow.update(0.2 * k, img, np.eye(3), 80.0, intr) is not None
    assert n == 0
