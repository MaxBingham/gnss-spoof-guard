"""Chase camera: follows the true drone from behind along its track, pulls back as the estimate drifts away
so the drone and its ghost stay in frame. The mouse adds an orbit / zoom offset on top."""

from __future__ import annotations

import math

import mujoco
import numpy as np


AZ_BIAS = 30.0  # deg: a three-quarter view reads the drone shape and both trails better than dead astern


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _screen_bias(aim: tuple, distance: float, azimuth: float, elevation: float) -> np.ndarray:
    """World shift of lookat so the tracked point lands on pixel (sx, sy) instead of the frame center.

    MuJoCo free-camera forward (camera toward lookat) is (cos el cos az, cos el sin az, sin el),
    up is (-sin el cos az, -sin el sin az, cos el), right is forward × up. Moving lookat along
    +right shifts the subject left; moving it along +up shifts the subject down.
    """
    sx, sy, width, height, fovy = aim
    if height <= 0 or distance <= 0:
        return np.zeros(3)
    wpp = 2.0 * distance * math.tan(math.radians(fovy) / 2.0) / height
    dx = (sx - width / 2.0) * wpp
    dy = (sy - height / 2.0) * wpp
    az, el = math.radians(azimuth), math.radians(elevation)
    ca, sa, ce, se = math.cos(az), math.sin(az), math.cos(el), math.sin(el)
    forward = np.array([ce * ca, ce * sa, se])
    up = np.array([-se * ca, -se * sa, ce])
    right = np.cross(forward, up)
    n = float(np.linalg.norm(right))
    if n > 1e-8:
        right = right / n
    return -dx * right + dy * up


class ChaseCam:
    def __init__(self, heading_deg: float, start: np.ndarray) -> None:
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.az = heading_deg
        self.lookat = np.asarray(start, float).copy()
        self.dist = 30.0
        self.el = -28.0
        self.az_off = 0.0
        self.el_off = 0.0
        self.zoom = 1.0

    def reset_offsets(self) -> None:
        self.az_off = self.el_off = 0.0
        self.zoom = 1.0

    def update(self, dt: float, truth_pos: np.ndarray, truth_vel: np.ndarray, est_pos: np.ndarray,
               aim: tuple | None = None) -> mujoco.MjvCamera:
        """`aim` is (px_x, px_y, frame_w, frame_h, fovy_deg): bias lookat so the tracked midpoint
        projects there (the clear rectangle between the HUD panels) instead of at the frame center."""
        k = 1.0 - math.exp(-dt / 0.5) if dt > 0 else 1.0
        sep = float(np.linalg.norm(est_pos[:2] - truth_pos[:2]))
        target = truth_pos + 0.5 * (est_pos - truth_pos) * min(sep / 4.0, 1.0)
        target = np.array([target[0], target[1], max(target[2], 1.0)])
        self.lookat += (target - self.lookat) * k
        # Softer than 34 + 1.7*sep, still far enough that a ~120 m split plus its labels
        # fits the clear rectangle. Cap stays under 320.
        dist = min(30.0 + 1.5 * sep, 280.0)
        self.dist += (dist - self.dist) * k
        speed = float(np.hypot(truth_vel[0], truth_vel[1]))
        if speed > 1.5:
            heading = math.degrees(math.atan2(truth_vel[1], truth_vel[0]))
            self.az += _wrap(heading - self.az) * k * 0.6
        c = self.cam
        c.distance = self.dist * self.zoom
        c.azimuth = self.az + AZ_BIAS + self.az_off
        self.el += (-28.0 - min(0.3 * sep, 30.0) - self.el) * k  # steeper as they separate: both stay in frame
        c.elevation = float(np.clip(self.el + self.el_off, -89.0, -3.0))
        look = self.lookat
        if aim is not None:
            look = look + _screen_bias(aim, c.distance, c.azimuth, c.elevation)
        c.lookat[:] = look
        return c
