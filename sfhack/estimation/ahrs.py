"""Attitude and heading reference (complementary filter on the onboard IMU + compass).

The controller's inner loop needs roll/pitch/yaw and body rates, and the EKF needs the attitude to
rotate accelerations into the world frame. Neither may come from the truth, so this small AHRS
provides them:
  - gyro integration (bias calibrated on the ground before takeoff),
  - roll/pitch correction from the accelerometer, only when |a| is close to 1 g and the body is
    not rotating (for a multirotor the accelerometer sees gravity only when NOT accelerating),
  - yaw correction from the compass.
Upgrade path: replace with a full error-state EKF (attitude + gyro/accel bias states) fused with
the visual fixes, which also makes tilt observable during accelerations.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import AhrsConfig
from ..messages import ImuMsg, MagMsg

G = 9.81


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _expm_so3(phi: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(phi))
    K = _skew(phi)
    if th < 1e-8:
        return np.eye(3) + K
    return np.eye(3) + math.sin(th) / th * K + (1 - math.cos(th)) / th**2 * (K @ K)


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class Ahrs:
    def __init__(self, cfg: AhrsConfig, initial_yaw: float = 0.0) -> None:
        self.cfg = cfg
        self.R = _rot_z(initial_yaw)
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        self.omega = np.zeros(3)
        self.last_t: float | None = None
        self.calibrating = True
        self._gyro_sum = np.zeros(3)
        self._acc_sum = np.zeros(3)
        self._n = 0
        self._yaw_initialized = False

    @property
    def yaw(self) -> float:
        return math.atan2(self.R[1, 0], self.R[0, 0])

    def finish_calibration(self) -> None:
        """End of the on-ground phase: gyro bias = mean rate; accel bias = mean - g (ground assumed level)."""
        if self._n > 0:
            self.gyro_bias = self._gyro_sum / self._n
            self.accel_bias = self._acc_sum / self._n - np.array([0.0, 0.0, G])
        self.R = _rot_z(self.yaw)
        self.calibrating = False

    def update_imu(self, msg: ImuMsg) -> None:
        dt = 0.0 if self.last_t is None else msg.t - self.last_t
        self.last_t = msg.t
        if self.calibrating:
            self._gyro_sum += msg.gyro
            self._acc_sum += msg.accel
            self._n += 1
            self.omega = msg.gyro - self._gyro_sum / self._n
            return
        self.omega = msg.gyro - self.gyro_bias
        if dt <= 0:
            return
        phi = self.omega * dt
        acc = msg.accel - self.accel_bias
        na = float(np.linalg.norm(acc))
        if abs(na - G) < self.cfg.accel_gate and float(np.linalg.norm(self.omega)) < self.cfg.rate_gate:
            up_meas = acc / na
            up_est = self.R.T[:, 2]
            phi = phi + self.cfg.tilt_gain * dt * np.cross(up_meas, up_est)
        self.R = self.R @ _expm_so3(phi)
        u, _, vt = np.linalg.svd(self.R)  # re-orthonormalize
        self.R = u @ vt

    def update_mag(self, msg: MagMsg, dt: float) -> None:
        if not self._yaw_initialized:
            self.R = _rot_z(msg.yaw) @ _rot_z(-self.yaw) @ self.R
            self._yaw_initialized = True
            return
        gain = 5.0 if self.calibrating else self.cfg.yaw_gain
        err = _wrap(msg.yaw - self.yaw)
        self.R = _rot_z(min(1.0, gain * dt) * err) @ self.R

    def correct_tilt_world(self, phi_world: np.ndarray) -> None:
        """Rotate the attitude by a small world-frame rotation (tilt error estimated by the EKF)."""
        self.R = _expm_so3(phi_world) @ self.R

    def rotate_accel(self, accel: np.ndarray) -> np.ndarray:
        """Bias-corrected specific force rotated to world, minus gravity = world acceleration."""
        return self.R @ (accel - self.accel_bias) - np.array([0.0, 0.0, G])
