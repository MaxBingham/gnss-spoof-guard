"""Chase camera: follows the true drone from behind along its track, pulls back as the estimate drifts away
so the drone and its ghost stay in frame. The mouse adds an orbit / zoom offset on top."""

from __future__ import annotations

import math

import mujoco
import numpy as np


AZ_BIAS = 30.0  # deg: a three-quarter view reads the drone shape and both trails better than dead astern


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class ChaseCam:
    def __init__(self, heading_deg: float, start: np.ndarray) -> None:
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.az = heading_deg
        self.lookat = np.asarray(start, float).copy()
        self.dist = 30.0
        self.az_off = 0.0
        self.el_off = 0.0
        self.zoom = 1.0

    def reset_offsets(self) -> None:
        self.az_off = self.el_off = 0.0
        self.zoom = 1.0

    def update(self, dt: float, truth_pos: np.ndarray, truth_vel: np.ndarray, est_pos: np.ndarray) -> mujoco.MjvCamera:
        k = 1.0 - math.exp(-dt / 0.5) if dt > 0 else 1.0
        sep = float(np.linalg.norm(est_pos[:2] - truth_pos[:2]))
        target = truth_pos + 0.4 * (est_pos - truth_pos) * (sep > 2.0)
        target = np.array([target[0], target[1], max(target[2], 1.0)])
        self.lookat += (target - self.lookat) * k
        dist = min(34.0 + 1.25 * sep, 260.0)
        self.dist += (dist - self.dist) * k
        speed = float(np.hypot(truth_vel[0], truth_vel[1]))
        if speed > 1.5:
            heading = math.degrees(math.atan2(truth_vel[1], truth_vel[0]))
            self.az += _wrap(heading - self.az) * k * 0.6
        c = self.cam
        c.lookat[:] = self.lookat
        c.distance = self.dist * self.zoom
        c.azimuth = self.az + AZ_BIAS + self.az_off
        c.elevation = float(np.clip(-28.0 + self.el_off, -89.0, -3.0))
        return c
