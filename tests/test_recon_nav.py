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


def _project_lights(loc, lights, pos, R_wb, t, clock, rng, shimmer=0.15):
    """Synthetic rear frame: dark image with a Gaussian spot for every light that is on (truth = `pos`)."""
    W, H = int(2 * loc.cx), int(2 * loc.cy)
    img = np.clip(6 + rng.normal(0, 2, (H, W)), 0, 255).astype(np.float32)
    R_wc = R_wb @ loc.R_bc
    yy, xx = np.mgrid[0:H, 0:W]
    for lt in lights:
        c = R_wc.T @ (lt.pos - pos)
        if c[2] <= 1:
            continue
        u = loc.cx + loc.f * c[0] / c[2] + rng.normal(0, shimmer)
        v = loc.cy + loc.f * c[1] / c[2] + rng.normal(0, shimmer)
        if not (0 <= u < W and 0 <= v < H):
            continue
        if lt.facing_deg is not None:
            off = math.degrees(math.atan2(pos[1] - lt.pos[1], pos[0] - lt.pos[0])) - lt.facing_deg
            if abs((off + 180) % 360 - 180) > lt.beam_half_deg:
                continue
        if lt.code[int(math.floor((t + clock) * lt.bit_hz)) % len(lt.code)] == "1":
            amp = 120 * (2000 / np.linalg.norm(lt.pos - pos)) ** 2
            img += amp * np.exp(-((xx + 0.5 - u) ** 2 + (yy + 0.5 - v) ** 2) / (2 * 0.9**2))
    return np.clip(img, 0, 255).astype(np.uint8)


def test_light_localizer_ids_position_and_heading() -> None:
    from sfhack.config import CameraConfig, LightsConfig
    from sfhack.perception.light_localizer import LightLocalizer
    from sfhack.sim.rear_camera_stub import make_codes, stub_base_lights, surveyed

    codes = make_codes(12)
    assert len(set(codes)) == 12
    truth = stub_base_lights(np.zeros(2), lambda x, y: 0.0)
    loc = LightLocalizer(surveyed(truth, 0.15, 0), LightsConfig(), CameraConfig(rear_pitch_down_deg=3.0))  # gimbal on the base
    rng = np.random.default_rng(3)
    # 1.5 km out on a bearing 60 deg (ENU yaw 30 deg), 80 m up, tail to the base; AHRS heading 1.5 deg off
    d = np.array([math.cos(math.radians(30)), math.sin(math.radians(30))])
    yaw_true = math.radians(30)
    fixes = []
    for k in range(40):
        t = 0.1 * k
        pos = np.r_[d * (1500 + 8 * t), 80.0]
        R_true = _rot(rng.normal(0, 0.01), rng.normal(0, 0.01), yaw_true)
        R_ahrs = _rot(0.0, 0.0, math.radians(1.5)) @ R_true
        img = _project_lights(loc, truth, pos, R_true, t, 3.3, rng)
        fix = loc.update(t, img, R_ahrs, pos + np.array([30.0, -20.0, 0.5]), 1.0)
        if fix is not None:
            fixes.append((fix, pos))
    assert len(fixes) >= 10, loc.status()
    for fix, pos in fixes:
        assert set(fix.ids) <= {lt.id for lt in truth}
        assert fix.n_lights >= 2
        err = fix.xy - pos[:2]
        s = np.sqrt(np.linalg.eigvalsh(fix.cov).max())
        assert np.linalg.norm(err) < max(4 * s, 3.0), (err, s, fix.kind)
        assert abs(math.degrees(fix.yaw_corr) + 1.5) < max(4 * math.degrees(fix.yaw_std), 0.3)
    # outward-facing fence lights: only the near half is seen, the heading is free -> cross-track ~ range x yaw std
    cross = np.array([-d[1], d[0]])
    assert np.mean([abs((f.xy - p[:2]) @ d) for f, p in fixes]) < 5.0
    assert np.mean([abs((f.xy - p[:2]) @ cross) for f, p in fixes]) < 15.0


def test_steady_light_gets_no_id() -> None:
    from sfhack.perception.light_localizer import BaseLight, LightLocalizer
    from sfhack.sim.rear_camera_stub import make_codes

    codes = make_codes(3)
    lights = [BaseLight(f"L{i}", np.array([-10.0 * i, 0.0, 3.0]), c, 10.0) for i, c in enumerate(codes)]
    loc = LightLocalizer(lights)
    steady = [BaseLight("X", np.array([-10.0, 0.0, 3.0]), "1111111111", 10.0)]
    rng = np.random.default_rng(0)
    pos = np.array([800.0, 0.0, 60.0])
    for k in range(30):
        img = _project_lights(loc, steady, pos, _rot(0, 0, 0.0), 0.1 * k, 0.0, rng)
        loc.update(0.1 * k, img, _rot(0, 0, 0.0), pos, 1.0)
    assert loc.tracks and all(tr.light_id is None for tr in loc.tracks)
