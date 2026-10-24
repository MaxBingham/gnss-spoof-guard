"""Recon-nav evaluation route: out along the bearing, a short orbit, back to the pad, tail always to the base.

The rear NIR camera looks backwards, so the quadcopter yaws to keep its tail on the base the whole flight:
outbound it flies forward, on the way home it flies tail first (a multirotor does not care). `TailToBaseMission`
is the waypoint mission with that yaw policy (heading from the base to the ESTIMATED position, rate-limited).
recon-mission's state machine can reuse `tail_to_base_yaw`.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from ..config import MissionConfig
from ..messages import Health, PoseEstimate, Setpoint
from .mission import GROUND_TARGET_Z, Mission, Waypoint


def bearing_to_enu_yaw(az_deg: float) -> float:
    """Compass azimuth (deg clockwise from north) -> ENU yaw (rad, 0 = east, CCW)."""
    return math.radians(90.0 - az_deg)


def tail_to_base_yaw(pos_xy: np.ndarray, base_xy: np.ndarray, fallback: float, min_dist: float = 25.0) -> float:
    d = np.asarray(pos_xy, float)[:2] - np.asarray(base_xy, float)[:2]
    return fallback if float(np.linalg.norm(d)) < min_dist else math.atan2(d[1], d[0])


def out_and_back_plan(a_xy: np.ndarray, az_deg: float, dist_m: float, agl: float, ground_z: Callable[[np.ndarray], float],
                      cfg: MissionConfig, orbit_r: float = 40.0, orbit_n: int = 12, leg_m: float = 50.0) -> list[Waypoint]:
    """TAKEOFF at A to `agl`, CRUISE out along `az_deg` to `dist_m` (legs of `leg_m`, each `agl` above the highest
    onboard ground along it), one orbit of radius `orbit_r` around the far point, CRUISE back, DESCEND / LAND on A."""
    a = np.asarray(a_xy, float)[:2]
    yaw = bearing_to_enu_yaw(az_deg)
    u = np.array([math.cos(yaw), math.sin(yaw)])
    far = a + u * dist_m

    def z_at(p0: np.ndarray, p1: np.ndarray) -> float:
        s = np.linspace(0.0, 1.0, max(2, int(np.linalg.norm(p1 - p0) / 5.0) + 1))
        return max(ground_z(p0 + (p1 - p0) * k) for k in s) + agl

    wps = [Waypoint(np.array([a[0], a[1], ground_z(a) + agl]), cfg.climb_speed, "TAKEOFF")]
    pts = [a + u * d for d in np.arange(leg_m, dist_m + 1e-6, leg_m)]
    if np.linalg.norm(pts[-1] - far) > 1.0:
        pts.append(far)
    ring = [far + orbit_r * np.array([math.cos(yaw + math.pi / 2 + 2 * math.pi * k / orbit_n),
                                      math.sin(yaw + math.pi / 2 + 2 * math.pi * k / orbit_n)]) for k in range(orbit_n + 1)]
    back = [far + (a - far) * k for k in np.linspace(0.0, 1.0, int(math.ceil(dist_m / leg_m)) + 1)[1:]]
    prev = a
    for p in pts + ring + [far] + back:
        wps.append(Waypoint(np.array([p[0], p[1], z_at(prev, p)]), cfg.cruise_speed, "CRUISE"))
        prev = p
    g = ground_z(a)
    wps.append(Waypoint(np.array([a[0], a[1], g + cfg.flare_alt]), cfg.descent_speed, "DESCEND"))
    wps.append(Waypoint(np.array([a[0], a[1], g + GROUND_TARGET_Z]), cfg.land_speed, "LAND"))
    return wps


class TailToBaseMission(Mission):
    """Waypoint mission whose yaw setpoint keeps the tail (rear camera) pointed at the base."""

    def __init__(self, *args, base_xy: np.ndarray, yaw_rate_max: float = math.radians(30.0), **kw) -> None:
        super().__init__(*args, **kw)
        self.base_xy = np.asarray(base_xy, float)[:2]
        self.yaw_rate_max = yaw_rate_max

    def update(self, t: float, pose: PoseEstimate, dt: float, health: Health = Health.NOMINAL, speed_scale: float = 1.0,
               last_fix_pos: np.ndarray | None = None) -> Setpoint:
        if self.phase not in ("PREFLIGHT", "LANDED", "DESCEND", "LAND"):
            want = tail_to_base_yaw(pose.pos[:2], self.base_xy, self.yaw)
            err = math.atan2(math.sin(want - self.yaw), math.cos(want - self.yaw))
            step = self.yaw_rate_max * max(dt, 0.0)
            self.yaw = math.atan2(math.sin(self.yaw + max(-step, min(step, err))), math.cos(self.yaw + max(-step, min(step, err))))
        return super().update(t, pose, dt, health=health, speed_scale=speed_scale, last_fix_pos=last_fix_pos)
