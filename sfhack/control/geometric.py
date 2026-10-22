"""Cascaded geometric controller, ported from fly_x2.py: position -> attitude -> motor thrusts.

Input is always a `PoseEstimate` (from the EKF, or from the TruthPoseSource when tuning),
never the simulator state directly.
"""

from __future__ import annotations

import numpy as np

from ..config import ControlGains
from ..messages import PoseEstimate, Setpoint
from .mixer import Mixer, VehicleParams


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


class GeometricController:
    def __init__(self, params: VehicleParams, gains: ControlGains) -> None:
        self.mass = params.mass
        self.g = params.g
        self.mixer = Mixer(params)
        self.kp_pos = np.array(gains.kp_pos)
        self.kd_pos = np.array(gains.kd_pos)
        self.kR = np.array(gains.kR)
        self.kw = np.array(gains.kw)
        self.max_acc = gains.max_acc
        self.drag_ff = gains.drag_ff
        self.drag_ff_quad = gains.drag_ff_quad
        self.last_acc_cmd = np.zeros(3)

    def __call__(self, pose: PoseEstimate, sp: Setpoint) -> np.ndarray:
        R, omega = pose.R, pose.omega
        # 1) position loop -> desired force vector in world frame
        acc = self.kp_pos * (sp.pos - pose.pos) + self.kd_pos * (sp.vel - pose.vel)
        acc = np.clip(acc, -self.max_acc, self.max_acc)
        self.last_acc_cmd = acc  # commanded acceleration (V3: the AHRS uses it to detect steady flight)
        # rotor-drag feedforward on the setpoint velocity (calm air): cruise needs a steady tilt against it
        vh = np.array([sp.vel[0], sp.vel[1], 0.0])
        ff = vh * (self.drag_ff + self.drag_ff_quad * float(np.linalg.norm(vh)))
        f_world = self.mass * (acc + ff + np.array([0.0, 0.0, self.g]))

        # 2) desired attitude from force direction and yaw
        z_d = f_world / np.linalg.norm(f_world)
        x_c = np.array([np.cos(sp.yaw), np.sin(sp.yaw), 0.0])
        y_d = _cross(z_d, x_c)
        y_d /= np.linalg.norm(y_d)
        x_d = _cross(y_d, z_d)
        R_d = np.column_stack([x_d, y_d, z_d])

        # 3) attitude loop -> body torques
        e_R_mat = 0.5 * (R_d.T @ R - R.T @ R_d)
        e_R = np.array([e_R_mat[2, 1], e_R_mat[0, 2], e_R_mat[1, 0]])
        torque = -self.kR * e_R - self.kw * omega
        thrust = float(f_world @ R[:, 2])

        # 4) mixer -> per-motor thrust, clipped to motor limits
        return self.mixer.wrench_to_motors(thrust, torque)
