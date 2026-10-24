"""recon-world: blink codes, onboard-safe base reader, simulated array, rear NIR camera."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from sfhack.acoustic.sim_array import ArrayLevers, simulate_bearing
from sfhack.world.base import BaseLight, canonical, decode_code, lyndon_codes

PKG = Path(__file__).resolve().parents[1] / "sfhack"


def test_codes_are_unique_under_rotation_and_decode_at_any_phase() -> None:
    codes = lyndon_codes()
    assert len(codes) == 22
    rots = [c[i:] + c[:i] for c in codes for i in range(8)]
    assert len(set(rots)) == len(rots)  # no rotation of one code equals a rotation of another (or itself)
    table = {c: f"L{i}" for i, c in enumerate(codes)}
    for i, c in enumerate(codes):
        for k in range(8):
            assert decode_code(c[k:] + c[:k], table) == f"L{i}"
    assert decode_code("11111111", table) is None and decode_code("0000000", table) is None


def test_blink_phase_staggers_lights() -> None:
    a = BaseLight("A", "fence", (0, 0, 0), "00010111", phase=0)
    b = BaseLight("B", "fence", (0, 0, 0), "00010111", phase=3)
    sa = "".join("1" if a.is_on(0.1 * k + 0.05) else "0" for k in range(8))
    sb = "".join("1" if b.is_on(0.1 * k + 0.05) else "0" for k in range(8))
    assert sa == "00010111" and sb != sa and canonical(sb) == canonical(sa)


def test_base_reader_is_onboard_safe() -> None:
    src = (PKG / "world" / "base.py").read_text(encoding="utf-8")
    assert not re.search(r"^\s*(from|import)\s+[\w.]*\b(sim|mujoco|degradation)\b", src, re.MULTILINE)
    assert "thermal_truth" not in src and "target_scene" not in src


def _tmp_mission(tmp: Path, dist: float, cls: str = "truck") -> Path:
    w = tmp / "m" / "world"
    w.mkdir(parents=True)
    x, y = dist * math.sin(math.radians(60)), dist * math.cos(math.radians(60))
    objs = [{"kind": "vehicle", "class": cls, "pos": [x, y, 0.0]}, {"kind": "person", "class": "person", "pos": [x, y, 0]}]
    (w / "target_scene.json").write_text(json.dumps({"objects": objs}), encoding="utf-8")
    (w / "base.json").write_text(json.dumps({"uma16_enu": [0, 0, 6], "uma16_x_axis_az_deg": 0.0}), encoding="utf-8")
    return tmp / "m"


def test_sim_array_error_matches_its_sigma(tmp_path: Path) -> None:
    m = _tmp_mission(tmp_path, 1800.0)
    errs, sig = [], []
    for s in range(40):
        est, info = simulate_bearing(m, seed=s)
        assert est is not None and est.label == "engine" and 0 <= est.az_deg < 360
        assert abs(info["true_az_deg"] - 60.0) < 0.5
        errs.append(info["az_error_deg"])
        sig.append(est.sigma_az_deg)
    assert 4.5 < np.mean(sig) < 10.0  # RECON_PLAN: 5-10 deg for low-frequency engine noise
    assert np.std(errs) < np.mean(sig) * 1.3 and abs(np.mean(errs)) < 3.0


def test_sim_array_levers(tmp_path: Path) -> None:
    m = _tmp_mission(tmp_path, 1800.0)
    est, info = simulate_bearing(m, seed=1, levers=ArrayLevers(bias_deg=12.0, sigma_deg=0.0))
    assert abs(info["az_error_deg"] - 12.0) < 1e-6
    calm = simulate_bearing(m, seed=1)[0]
    windy = simulate_bearing(m, seed=1, levers=ArrayLevers(wind_m_s=2.0))[0]
    assert windy.sigma_az_deg > calm.sigma_az_deg and windy.confidence <= calm.confidence
    far = _tmp_mission(tmp_path / "far", 9000.0, "car")
    est, info = simulate_bearing(far, seed=0)
    assert est is None and "not heard" in info["reason"]


UKR = None
try:
    from sfhack.missionprep.build import mission_dir

    UKR = mission_dir("ukraine-1")
except Exception:  # noqa: BLE001
    pass


@pytest.mark.skipif(UKR is None or not (UKR / "world" / "base_lights.json").exists(),
                    reason="ukraine-1 with a recon base (python -m sfhack.sim.recon_world) not available")
def test_rear_camera_sees_lights_at_their_projection() -> None:
    import mujoco

    from sfhack.bus import Bus
    from sfhack.config import Config
    from sfhack.sim.mujoco_sim import MujocoSim
    from sfhack.sim.rear_camera import RearCameraSensor
    from sfhack.world.scene import PackageWorld

    world = PackageWorld(UKR / "world")
    c = world.a_enu
    sim = MujocoSim(Config(), Bus(), world, spawn_xy=tuple(c))
    rear = RearCameraSensor(sim, sim.bus, UKR / "world", seed=0)
    az = 77.0  # out of the base, tail towards it, 600 m, 80 m above ground
    xy = c + 600.0 * np.array([math.sin(math.radians(az)), math.cos(math.radians(az))])
    z = world.ground_z(*xy) + 80.0
    yaw = math.atan2(xy[1] - c[1], xy[0] - c[0])
    sim.data.qpos[0:3] = [xy[0], xy[1], z]
    sim.data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    mujoco.mj_forward(sim.model, sim.data)
    rear.camera.set_tilt(math.degrees(math.atan2(z - 8.0, 600.0)))
    seen = 0
    for k in range(16):  # 0.8 s = one full code: every bright light is on in some frame
        sim.data.time = 5.0 + 0.05 * k
        rear.next_t = 0.0
        rear.step()
        img = rear.last.image
        assert img.shape == (480, 640) and rear.last.R_body_cam.shape == (3, 3)
        for r in rear.last_truth:
            if r["nav"] and r["on"] and not r["occluded"] and r["peak_dn"] > 60 and 3 < r["u"] < 637 and 3 < r["v"] < 477:
                u, v = int(r["u"]), int(r["v"])
                assert img[v - 2:v + 3, u - 2:u + 3].max() > 40
                seen += 1
    assert seen >= 8
    # the optical axis looks back along body -x, tilted down
    zc = rear.camera.R_body_cam[:, 2]
    assert zc[0] < -0.9 and zc[2] < 0
