"""Recon mission (onboard logic only, no simulator): search plan, thermal detector, state machine on a kinematic
drone (pose = setpoint) with a synthetic detector."""

from __future__ import annotations

import math

import numpy as np

from sfhack.config import default_config
from sfhack.messages import PoseEstimate, ThermalFrame
from sfhack.mission.recon import ReconMission, az_of, az_to_unit, cone_halfwidth, plan_search
from sfhack.perception.thermal_detect import Detection, DetectResult, ThermalDetector


def _pose(t: float, pos: np.ndarray, yaw: float = 0.0, sigma: float = 3.0) -> PoseEstimate:
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return PoseEstimate(t, np.asarray(pos, float).copy(), np.zeros(3), yaw, np.eye(6) * sigma ** 2, R, np.zeros(3))


def test_compass_convention() -> None:
    assert np.allclose(az_to_unit(90.0), [1.0, 0.0], atol=1e-12)  # east
    assert abs(az_of(np.array([0.0, 1.0])) - 0.0) < 1e-9
    assert abs(az_of(np.array([-1.0, 0.0])) - 270.0) < 1e-9


def test_search_plan_fits_budget_and_widens() -> None:
    c = default_config().recon
    base = np.zeros(2)
    for budget in (300.0, 600.0, 1200.0):
        p = plan_search(base, 64.0, 7.0, c, swath_m=60.0, time_budget_s=budget)
        assert p.est_time_s <= budget * 1.01 or p.step_m >= p.r_max - p.r_start - 1e-6
        u = az_to_unit(64.0)
        n = np.array([u[1], -u[0]])
        lat = np.abs((p.path - base) @ n)
        along = (p.path - base) @ u
        assert along.max() <= c.search_max_m + 1e-6
        far, near = lat[along > 0.8 * p.r_max], lat[(along > p.r_start) & (along < 0.5 * (p.r_start + p.r_max))]
        if len(far) and len(near):
            assert far.max() >= near.max()  # the cone widens with distance
    assert cone_halfwidth(2000.0, 7.0, c) > cone_halfwidth(1000.0, 7.0, c) > 0


def test_search_plan_stays_on_the_map() -> None:
    c = default_config().recon
    inside = lambda xy: abs(xy[0]) < 900 and abs(xy[1]) < 900  # noqa: E731
    p = plan_search(np.zeros(2), 45.0, 10.0, c, 60.0, 900.0, inside)
    assert all(inside(q) for q in p.path)
    assert p.r_max < 1300


def _synthetic_frame(agl: float, objects: list[tuple[float, float, float, float, float]], w: int = 640,
                     h: int = 512, hfov: float = 50.0, seed: int = 0) -> tuple[np.ndarray, dict[str, float]]:
    """Level camera, yaw 0 (image top = east). objects: (east, north, length_east, width_north, dT)."""
    fx = w / 2 / math.tan(math.radians(hfov) / 2)
    K = {"fx": fx, "fy": fx, "cx": w / 2, "cy": h / 2, "w": float(w), "h": float(h)}
    rng = np.random.default_rng(seed)
    img = 295.0 + rng.normal(0, 0.08, (h, w)).astype(np.float32)
    gsd = agl / fx
    v, u = np.mgrid[0:h, 0:w]
    east = -(v + 0.5 - h / 2) * gsd  # image up = forward = east (yaw 0)
    north = -(u + 0.5 - w / 2) * gsd  # image right = body right = south
    for e, n, le, wn, dT in objects:
        img[(np.abs(east - e) < le / 2) & (np.abs(north - n) < wn / 2)] += dT
    return img.astype(np.float32), K


def test_detector_finds_warm_car_not_road_or_hotspot() -> None:
    agl = 90.0
    objs = [(10.0, -5.0, 4.6, 1.9, 9.0),  # car, engine running
            (-15.0, 0.0, 200.0, 6.0, 3.5),  # warm road (east-west strip)
            (5.0, 15.0, 1.6, 1.4, 14.0),  # parked car's hot hood / HVAC unit
            (-8.0, -14.0, 4.6, 1.9, 1.5)]  # cooling car
    img, K = _synthetic_frame(agl, objs)
    det = ThermalDetector()
    r = det.detect(img, np.eye(3), agl, K)
    assert len(r.detections) == 1, (r.detections, r.rejected)
    d = r.detections[0]
    assert d.label == "vehicle"
    assert np.linalg.norm(d.xy_rel - [10.0, -5.0]) < 0.6
    assert 3.5 < d.size_m[0] < 6.0 and 1.2 < d.size_m[1] < 3.0


class _FakeDetector:
    """Returns a detection of every world object inside the (level) footprint; xy_rel from the pose."""

    def __init__(self, objects: list[np.ndarray], footprint: float = 30.0) -> None:
        self.objects, self.fp = objects, footprint

    def detect(self, img, R, agl, K, pos_xy=None, keep_anomaly=False) -> DetectResult:
        out = DetectResult(gsd_m=0.13)
        for o in self.objects:
            rel = o - np.asarray(pos_xy, float)
            if np.all(np.abs(rel) < self.fp):
                out.detections.append(Detection(rel.copy(), (4.6, 1.9), 9.0, 0.8, "vehicle", (320.0, 256.0), 8.7, 8.0,
                                                (310, 250, 20, 12)))
        return out


def _fly(m: ReconMission, dt: float = 0.05, t_max: float = 1500.0, cam_hz: float = 2.0) -> list[str]:
    m.start()
    t, phases = 0.0, []
    pos = m.setpoint.copy()
    img = np.full((512, 640), 295.0, np.float32)
    K = {"fx": 686.0, "fy": 686.0, "cx": 320.0, "cy": 256.0, "w": 640.0, "h": 512.0}
    next_cam = 0.0
    while t < t_max and m.phase != "REPORT":
        pose = _pose(t, pos, m.yaw)
        landing = m.phase == "LAND"
        agl = pos[2] - m.pad_z
        m.set_ground_sensing(agl if landing else None, 1.0 if landing and agl < 0.3 else 0.0)
        sp = m.update(t, pose, dt)
        pos = sp.pos.copy()  # kinematic drone
        if t >= next_cam:
            next_cam = t + 1.0 / cam_hz
            m.on_thermal(ThermalFrame(t, 0, img, K), pose)
        if not phases or phases[-1] != m.phase:
            phases.append(m.phase)
        t += dt
    return phases


def test_state_machine_find_observe_return_land_report() -> None:
    cfg = default_config()
    az = 64.0
    target = az_to_unit(az) * 1500.0 + np.array([20.0, -10.0])
    m = ReconMission(cfg.recon, cfg.mission, {"az_deg": az, "sigma_az_deg": 7.0}, np.zeros(2), 0.0,
                     detector=_FakeDetector([target]), mission_id="test")
    phases = _fly(m)
    assert phases[:4] == ["TAKEOFF", "OUTBOUND", "SEARCH", "OBSERVE"], phases
    assert phases[-4:] == ["HOME", "LAND", "LANDED", "REPORT"] or phases[-3:] == ["LAND", "LANDED", "REPORT"], phases
    rep = m.report
    assert rep is not None and rep["outcome"] == "FOUND"
    f = rep["finds"][0]
    assert np.linalg.norm(np.asarray(f["est_enu"]) - target) < 2.0
    assert abs(f["bearing_deg"] - az_of(target)) < 1.0
    assert rep["flight_time_s"] <= cfg.recon.battery_s
    assert len(m._finds()[0].snapshots) >= 3
    assert "vehicle" in rep["summary"]


def test_nothing_found_returns_on_battery_or_pattern_end() -> None:
    cfg = default_config()
    cfg.recon.battery_s = 600.0
    m = ReconMission(cfg.recon, cfg.mission, {"az_deg": 30.0, "sigma_az_deg": 8.0}, np.zeros(2), 0.0,
                     detector=_FakeDetector([]), mission_id="test")
    phases = _fly(m, t_max=900.0)
    assert "OBSERVE" not in phases and phases[-1] == "REPORT", phases
    rep = m.report
    assert rep["outcome"] == "NOTHING_FOUND" and "nothing found" in rep["summary"]
    assert rep["flight_time_s"] <= 600.0
    assert rep["searched"]["area_km2"] > 0.0 or rep["searched"]["thermal_frames"] > 0
