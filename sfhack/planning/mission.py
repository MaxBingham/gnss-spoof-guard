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
        self.events: list[tuple[float, str]] = []  # onboard decisions; the autopilot forwards them to the bus

    def log(self, t: float, text: str) -> None:
        self.events.append((t, text))

    @property
    def done(self) -> bool:
        return self.phase == "LANDED"

    @property
    def goal(self) -> np.ndarray:
        return self.waypoints[-1].pos

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
        """Touchdown = estimated height below landed_alt and vertical speed ~0 for touchdown_s."""
        if pose.pos[2] < self.cfg.landed_alt and abs(pose.vel[2]) < self.cfg.touchdown_vz:
            self.landed_timer += dt
            if self.landed_timer >= self.cfg.touchdown_s:
                self.phase = "LANDED"
                self.log(t, f"TOUCHDOWN: motors off at estimated ({pose.pos[0]:.1f}, {pose.pos[1]:.1f})")
        else:
            self.landed_timer = 0.0
