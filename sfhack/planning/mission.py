"""Waypoint mission (ported from fly_x2.py's Mission): a smooth setpoint moves along the waypoints.

Phases: PREFLIGHT -> TAKEOFF -> CRUISE -> DESCEND -> LAND -> LANDED (plus EMERGENCY_LAND,
RETURN_TO_LAST_FIX and HOLD driven by the health monitor). Waypoint-reached and touchdown checks
use the pose ESTIMATE, never the truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import MissionConfig
from ..messages import Health, PoseEstimate, Setpoint

GROUND_TARGET_Z = -0.3  # final landing target slightly below ground so the drone settles


@dataclass
class Waypoint:
    pos: np.ndarray  # (3,) world
    speed: float  # m/s, setpoint speed on the segment that ENDS at this waypoint
    phase: str


def a_to_b_plan(a_xy: np.ndarray, b_xy: np.ndarray, cfg: MissionConfig) -> list[Waypoint]:
    """A -> takeoff -> cruise at fixed altitude -> B -> descend -> slow final landing."""
    a, b = np.asarray(a_xy, float), np.asarray(b_xy, float)
    return [
        Waypoint(np.array([a[0], a[1], cfg.cruise_alt]), cfg.climb_speed, "TAKEOFF"),
        Waypoint(np.array([b[0], b[1], cfg.cruise_alt]), cfg.cruise_speed, "CRUISE"),
        Waypoint(np.array([b[0], b[1], cfg.flare_alt]), cfg.descent_speed, "DESCEND"),
        Waypoint(np.array([b[0], b[1], GROUND_TARGET_Z]), cfg.land_speed, "LAND"),
    ]


def fly_x2_plan(alt: float = 5.0, speed: float = 3.0, land_speed: float = 0.7) -> list[Waypoint]:
    """The original fly_x2.py square: take off, 15 m east, 15 m north, land."""
    return [
        Waypoint(np.array([0.0, 0.0, alt]), speed, "TAKEOFF"),
        Waypoint(np.array([15.0, 0.0, alt]), speed, "CRUISE"),
        Waypoint(np.array([15.0, 15.0, alt]), speed, "CRUISE"),
        Waypoint(np.array([15.0, 15.0, GROUND_TARGET_Z]), land_speed, "LAND"),
    ]


class Mission:
    def __init__(self, waypoints: list[Waypoint], start_pos: np.ndarray, yaw: float, cfg: MissionConfig) -> None:
        self.waypoints = waypoints
        self.cfg = cfg
        self.yaw = yaw
        self.setpoint = np.asarray(start_pos, float).copy()
        self.idx = 0
        self.phase = "PREFLIGHT"
        self.landed_timer = 0.0
        self.emergency = False
        self.wait_at_b_since: float | None = None
        self.recentering = False
        self.range_agl: float | None = None  # rangefinder height (autopilot), None = no reading
        self.imu_still_s = 0.0
        self.ground_contact = False  # rangefinder says we are down: the autopilot idles the motors
        self.events: list[tuple[float, str]] = []  # onboard decisions; the autopilot forwards them to the bus

    def log(self, t: float, text: str) -> None:
        self.events.append((t, text))

    @property
    def done(self) -> bool:
        return self.phase == "LANDED"

    @property
    def goal(self) -> np.ndarray:
        return self.waypoints[-1].pos

    def set_ground_sensing(self, range_agl: float | None, imu_still_s: float) -> None:
        self.range_agl = range_agl
        self.imu_still_s = imu_still_s

    def start(self) -> None:
        if self.phase == "PREFLIGHT":
            self.phase = self.waypoints[0].phase

    def _move_setpoint(self, target: np.ndarray, speed: float, dt: float) -> tuple[np.ndarray, bool]:
        delta = target - self.setpoint
        dist = float(np.linalg.norm(delta))
        step = speed * dt
        if dist > step and dist > 1e-9:
            vel = delta / dist * speed
            self.setpoint = self.setpoint + vel * dt
            return vel, False
        self.setpoint = target.copy()
        return np.zeros(3), True

    def update(
        self,
        t: float,
        pose: PoseEstimate,
        dt: float,
        health: Health = Health.NOMINAL,
        speed_scale: float = 1.0,
        last_fix_pos: np.ndarray | None = None,
    ) -> Setpoint:
        if self.phase in ("PREFLIGHT", "LANDED"):
            return Setpoint(t, self.setpoint.copy(), np.zeros(3), self.yaw, self.phase)

        # Health overrides apply while cruising (takeoff and the final landing are always completed).
        cruising = self.waypoints[self.idx].phase == "CRUISE" and not self.emergency
        if cruising and health == Health.HOLD:
            return Setpoint(t, self.setpoint.copy(), np.zeros(3), self.yaw, "HOLD")
        if cruising and health == Health.RETURN_TO_LAST_FIX and last_fix_pos is not None:
            target = np.array([last_fix_pos[0], last_fix_pos[1], self.setpoint[2]])
            vel, _ = self._move_setpoint(target, 0.5 * self.cfg.cruise_speed, dt)
            return Setpoint(t, self.setpoint.copy(), vel, self.yaw, "RETURN_TO_LAST_FIX")
        if cruising and health == Health.LAND:
            self.emergency = True
            self.log(t, "EMERGENCY_LAND: navigation health LAND")

        if self.emergency:
            target = np.array([self.setpoint[0], self.setpoint[1], GROUND_TARGET_Z])
            speed = self.cfg.descent_speed if pose.pos[2] > self.cfg.flare_alt else self.cfg.land_speed
            vel, reached = self._move_setpoint(target, speed, dt)
            self.phase = "EMERGENCY_LAND"
            self._check_touchdown(t, pose, dt)
            return Setpoint(t, self.setpoint.copy(), vel, self.yaw, self.phase)

        wp = self.waypoints[self.idx]
        self.phase = wp.phase
        scale = speed_scale if wp.phase == "CRUISE" else 1.0
        off = math.hypot(wp.pos[0] - pose.pos[0], wp.pos[1] - pose.pos[1])
        if wp.phase in ("DESCEND", "LAND") and off > self.cfg.descend_xy_tol:
            if not self.recentering:
                self.recentering = True
                self.log(t, f"RECENTER: {off:.1f} m off B horizontally, descent paused")
            self._check_touchdown(t, pose, dt)
            return Setpoint(t, self.setpoint.copy(), np.zeros(3), self.yaw, self.phase)
        self.recentering = False
        vel, reached = self._move_setpoint(wp.pos, wp.speed * scale, dt)
        last = self.idx == len(self.waypoints) - 1
        if reached and not last:
            err = wp.pos - pose.pos
            if math.hypot(err[0], err[1]) < self.cfg.wp_radius and abs(err[2]) < self.cfg.wp_radius:
                if self._ready_to_leave(t, wp, pose):
                    self.idx += 1
        if self.waypoints[self.idx].phase in ("DESCEND", "LAND"):
            self._check_touchdown(t, pose, dt)
        return Setpoint(t, self.setpoint.copy(), vel, self.yaw, self.phase)

    def _ready_to_leave(self, t: float, wp: Waypoint, pose: PoseEstimate) -> bool:
        """Above B (end of cruise): descend only when the estimate is precise, else hover and wait for a fix."""
        nxt = self.waypoints[self.idx + 1]
        if not (wp.phase == "CRUISE" and nxt.phase == "DESCEND"):
            return True
        s2 = pose.sigma2_horizontal()
        if self.wait_at_b_since is None:
            if s2 <= self.cfg.precision_sigma2:
                self.log(t, f"ABOVE_B: 2-sigma {s2:.1f} m, descending")
                return True
            self.wait_at_b_since = t
            self.log(t, f"WAIT_FOR_FIX above B: 2-sigma {s2:.1f} m > {self.cfg.precision_sigma2:.1f} m, hovering")
            return False
        waited = t - self.wait_at_b_since
        if s2 <= self.cfg.precision_sigma2 or waited >= self.cfg.precision_wait_s:
            why = "fix received" if s2 <= self.cfg.precision_sigma2 else "wait timeout"
            self.log(t, f"DESCEND after {waited:.1f} s hover ({why}, 2-sigma {s2:.1f} m)")
            return True
        return False

    def _check_touchdown(self, t: float, pose: PoseEstimate, dt: float) -> None:
        """Touchdown = rangefinder height < touchdown_range AND IMU stationary, both for touchdown_s
        (independent of the estimated height, whose baro drift left drones sliding on the ground with
        motors on). Without rangefinder readings: estimated height < landed_alt and |vz| small."""
        c = self.cfg
        if self.range_agl is not None:
            low = self.range_agl < c.touchdown_range
            if low and not self.ground_contact:
                self.log(t, f"GROUND_CONTACT: rangefinder {self.range_agl:.2f} m, motors to idle")
            # Idle (no position control) while down: chasing the estimate made the drone hop and skid,
            # so the IMU never looked stationary. Range rising again -> normal control resumes.
            self.ground_contact = low
            how = f"rangefinder {self.range_agl:.2f} m, IMU still {self.imu_still_s:.1f} s"
        else:
            low = pose.pos[2] < c.landed_alt and abs(pose.vel[2]) < c.touchdown_vz
            how = f"no rangefinder: estimated height {pose.pos[2]:.2f} m"
        self.landed_timer = self.landed_timer + dt if low else 0.0
        still = self.range_agl is None or self.imu_still_s >= c.touchdown_s
        if self.landed_timer >= c.touchdown_s and still:
            self.phase = "LANDED"
            self.log(t, f"TOUCHDOWN ({how}): motors off at estimated ({pose.pos[0]:.1f}, {pose.pos[1]:.1f})")
