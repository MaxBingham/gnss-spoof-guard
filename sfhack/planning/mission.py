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

        if self.emergency:
            target = np.array([self.setpoint[0], self.setpoint[1], GROUND_TARGET_Z])
            speed = self.cfg.descent_speed if pose.pos[2] > self.cfg.flare_alt else self.cfg.land_speed
            vel, reached = self._move_setpoint(target, speed, dt)
            self.phase = "EMERGENCY_LAND"
            self._check_touchdown(pose, reached, dt)
            return Setpoint(t, self.setpoint.copy(), vel, self.yaw, self.phase)

        wp = self.waypoints[self.idx]
        self.phase = wp.phase
        scale = speed_scale if wp.phase == "CRUISE" else 1.0
        vel, reached = self._move_setpoint(wp.pos, wp.speed * scale, dt)
        last = self.idx == len(self.waypoints) - 1
        if reached and not last:
            err = wp.pos - pose.pos
            if math.hypot(err[0], err[1]) < self.cfg.wp_radius and abs(err[2]) < self.cfg.wp_radius:
                self.idx += 1
        if last:
            self._check_touchdown(pose, reached, dt)
        return Setpoint(t, self.setpoint.copy(), vel, self.yaw, self.phase)

    def _check_touchdown(self, pose: PoseEstimate, setpoint_down: bool, dt: float) -> None:
        if setpoint_down and pose.pos[2] < self.cfg.landed_alt and abs(pose.vel[2]) < 0.3:
            self.landed_timer += dt
            if self.landed_timer > 1.0:
                self.phase = "LANDED"
        else:
            self.landed_timer = 0.0
