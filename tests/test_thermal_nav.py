"""Night navigation (night-nav): thermal localizer on a synthetic prior, 60 m AGL route, keep-route health rule."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sfhack.config import HealthConfig, MissionConfig, ThermalConfig
from sfhack.messages import Health, PoseEstimate
from sfhack.perception.map_localizer import R_BODY_CAM
from sfhack.perception.thermal_localizer import ThermalLocalizer
from sfhack.planning.health import HealthMonitor
from sfhack.planning.mission import Waypoint, cruise_at_agl

EXT = 150.0  # m, half size of the synthetic map
RES = 0.5


def _synthetic_field() -> np.ndarray:
    """Warm roads on a cool lawn: a road grid with irregular spacing, a few warm roofs, smooth model noise."""
    n = int(2 * EXT / RES)
    rng = np.random.default_rng(3)
    t = np.full((n, n), 290.0, np.float32)
    for c in (37, 95, 180, 222, 301, 420, 515):  # north-south roads (columns)
        t[:, c:c + 14] = 299.0
    for r in (60, 151, 270, 333, 470, 560):  # east-west roads
        t[r:r + 12, :] = 298.5
    for _ in range(40):
        r, c = rng.integers(0, n - 30, 2)
        t[r:r + rng.integers(10, 30), c:c + rng.integers(10, 30)] = 294.0 + rng.normal(0, 2)
    import cv2
    t += cv2.GaussianBlur(rng.normal(0, 1.0, (n, n)).astype(np.float32), (0, 0), 6) * 3
    return t


def _package(tmp: Path, temp: np.ndarray) -> Path:
    d = tmp / "onboard"
    d.mkdir()
    n = temp.shape[0]
    ext = {"min_x": -EXT, "min_y": -EXT, "max_x": EXT, "max_y": EXT}
    (d / "manifest.json").write_text(json.dumps({"ortho": {"file": "ortho.jpg", "extent_enu": ext},
                                                 "a": {"enu": [0, 0]}, "b": {"enu": [10, 10]}}), encoding="utf-8")
    (d / "landmarks.json").write_text("[]", encoding="utf-8")
    Image.fromarray(np.zeros((n, n, 3), np.uint8)).save(d / "ortho.jpg")
    meta = {"extent_enu": ext, "res_m": RES, "rows": n, "cols": n, "class_names": ["x"], "hour": 23.0}
    roads = [{"highway": "residential", "kind": "vehicle", "paved": True,
              "pts_enu": [[-EXT + (c + 7) * RES, -EXT], [-EXT + (c + 7) * RES, EXT]]} for c in (37, 95, 180, 222, 301, 420, 515)]
    roads += [{"highway": "residential", "kind": "vehicle", "paved": True,
               "pts_enu": [[-EXT, EXT - (r + 6) * RES], [EXT, EXT - (r + 6) * RES]]} for r in (60, 151, 270, 333, 470, 560)]
    np.savez(d / "thermal_prior.npz", temp_K=temp, sigma_K=np.ones_like(temp), cls=np.zeros(temp.shape, np.uint8),
             roads=(temp > 297).astype(np.uint8), meta=json.dumps(meta), roads_json=json.dumps(roads))
    return d


def _render(temp: np.ndarray, cam_xy: np.ndarray, alt: float, yaw: float, R_wb: np.ndarray) -> tuple[np.ndarray, dict]:
    """Pinhole down camera over flat ground (same mount as the MuJoCo down camera), bilinear samples of `temp`."""
    import cv2
    w, h, hfov = 320, 256, 50.0
    fx = w / 2 / math.tan(math.radians(hfov) / 2)
    K = {"fx": fx, "fy": fx, "cx": w / 2, "cy": h / 2, "w": w, "h": h}
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    rays = np.stack([(u - w / 2) / fx, -(v - h / 2) / fx, -np.ones_like(u)], -1) @ (R_wb @ R_BODY_CAM).T
    s = -alt / rays[..., 2]
    xy = cam_xy + rays[..., :2] * s[..., None]
    cu = ((xy[..., 0] + EXT) / RES - 0.5).astype(np.float32)
    cv = ((EXT - xy[..., 1]) / RES - 0.5).astype(np.float32)
    img = cv2.remap(temp, cu, cv, cv2.INTER_LINEAR) + np.random.default_rng(0).normal(0, 0.05, (h, w)).astype(np.float32)
    return img + 3.0, K  # +3 K absolute offset: uncooled LWIR, only the pattern may matter


def _rot(yaw: float, pitch: float = 0.0) -> np.ndarray:
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return Rz @ Ry


@pytest.mark.parametrize("method", ["ncc", "grad", "road", "ncc+road"])
def test_thermal_localizer_recovers_offset(tmp_path: Path, method: str) -> None:
    temp = _synthetic_field()
    d = _package(tmp_path, temp)
    cfg = ThermalConfig(method=method)
    loc = ThermalLocalizer(d, cfg)
    true_xy = np.array([12.0, -7.0])
    R = _rot(math.radians(80), math.radians(4))  # flying north-ish, nose down 4 deg (cruise)
    img, K = _render(temp, true_xy, 60.0, math.radians(80), R)
    est = true_xy + np.array([6.0, -4.0])
    pose = PoseEstimate(0.0, np.array([est[0], est[1], 60.0]), np.zeros(3), math.radians(80),
                        np.diag([25.0, 25.0, 1.0, 1.0, 1.0, 1.0]), R, np.zeros(3))
    fix = loc.update(img, pose, 60.0, K)
    assert fix is not None, loc.last_reason
    err = fix.xy - true_xy
    if method == "road":  # 1-D where only one road is seen: the error must lie along the weak axis
        d2 = float(err @ np.linalg.solve(fix.cov, err))
        assert d2 < 9.21, (fix.xy, fix.cov)
    else:
        assert np.linalg.norm(err) < 1.5, (method, fix.xy, true_xy)


def test_thermal_localizer_rejects_uniform(tmp_path: Path) -> None:
    temp = np.full((int(2 * EXT / RES),) * 2, 291.0, np.float32)
    d = _package(tmp_path, temp)
    loc = ThermalLocalizer(d, ThermalConfig(method="ncc"))
    R = _rot(0.0)
    img, K = _render(temp, np.zeros(2), 60.0, 0.0, R)
    pose = PoseEstimate(0.0, np.array([0.0, 0.0, 60.0]), np.zeros(3), 0.0, np.eye(6) * 9.0, R, np.zeros(3))
    assert loc.update(img, pose, 60.0, K) is None
    assert loc.last_reason == "uniform"


def test_cruise_at_agl_follows_terrain() -> None:
    ground = lambda xy: 0.05 * float(xy[0])  # 5 % slope up to the east
    plan = [Waypoint(np.array([0.0, 0.0, 30.0]), 3.0, "TAKEOFF"), Waypoint(np.array([200.0, 0.0, 40.0]), 8.0, "CRUISE"),
            Waypoint(np.array([200.0, 0.0, 14.0]), 2.5, "DESCEND"), Waypoint(np.array([200.0, 0.0, 9.7]), 1.0, "LAND")]
    out = cruise_at_agl(plan, ground, 60.0, max_leg_m=50.0)
    cruise = [w for w in out if w.phase == "CRUISE"]
    assert len(cruise) == 4 and out[0].pos[2] == pytest.approx(60.0)
    for w in cruise:
        assert w.pos[2] == pytest.approx(ground(w.pos[:2]) + 60.0, abs=0.01)  # highest ground of each leg is its end
    assert [w.phase for w in out][-2:] == ["DESCEND", "LAND"]


def test_health_keeps_route_when_blind() -> None:
    hm = HealthMonitor(HealthConfig())
    hm.reset_timer(0.0)
    big = PoseEstimate(0.0, np.zeros(3), np.zeros(3), 0.0, np.eye(6) * 2500.0, np.eye(3), np.zeros(3))
    for t in np.arange(0.0, 200.0, 1.0):
        hs = hm.update(float(t), big)
        assert hs.state in (Health.NOMINAL, Health.DEGRADED) and hs.speed_scale == 1.0
    assert "BLIND, keep route" in hs.reason
    old = HealthMonitor(HealthConfig(keep_route_when_blind=False))
    old.reset_timer(0.0)
    assert any(old.update(float(t), big).state == Health.HOLD for t in range(30))
    assert MissionConfig().cruise_speed > 0
