"""Position/velocity EKF: state [x, y, z, vx, vy, vz] in the world (ENU) frame.

- Predict: IMU specific force rotated to world with the AHRS attitude, minus gravity.
- Update: barometer (z), visual fix (x, y) with Mahalanobis gating, zero-velocity on the ground.
- Yaw is not a state: it comes from the compass via the AHRS.

The model is linear, so this is a plain KF in practice; it is named EKF because the upgrade path
(attitude + IMU bias states, nonlinear camera measurement) makes it one. Covariance grows during
dead reckoning via the accelerometer process noise.
"""

from __future__ import annotations

import numpy as np

from ..config import EkfConfig
from ..messages import PoseEstimate, VisualFix

MAX_CONSECUTIVE_REJECTS = 10  # then the next fix is force-accepted (re-localization after divergence)


class Ekf:
    def __init__(self, cfg: EkfConfig, pos0: np.ndarray, vel0: np.ndarray | None = None) -> None:
        self.cfg = cfg
        self.x = np.zeros(6)
        self.x[:3] = pos0
        if vel0 is not None:
            self.x[3:] = vel0
        self.P = np.diag([cfg.init_pos_std**2] * 3 + [cfg.init_vel_std**2] * 3)
        self.t = 0.0
        self.consecutive_rejects = 0
        self.n_fix_accepted = 0
        self.n_fix_rejected = 0

    def predict(self, acc_world: np.ndarray, dt: float, t: float) -> None:
        self.t = t
        if dt <= 0:
            return
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        self.x = F @ self.x
        self.x[:3] += 0.5 * acc_world * dt * dt
        self.x[3:] += acc_world * dt
        q = self.cfg.accel_process_std**2
        Q = np.zeros((6, 6))
        Q[:3, :3] = np.eye(3) * q * dt**3 / 3
        Q[:3, 3:] = Q[3:, :3] = np.eye(3) * q * dt**2 / 2
        Q[3:, 3:] = np.eye(3) * q * dt
        self.P = F @ self.P @ F.T + Q

    def _update(self, H: np.ndarray, z: np.ndarray, R: np.ndarray) -> None:
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form

    def update_baro(self, alt: float) -> None:
        H = np.zeros((1, 6))
        H[0, 2] = 1.0
        self._update(H, np.array([alt]), np.array([[self.cfg.baro_std**2]]))

    def update_zero_velocity(self) -> None:
        H = np.zeros((3, 6))
        H[:, 3:] = np.eye(3)
        self._update(H, np.zeros(3), np.eye(3) * self.cfg.zupt_std**2)

    def mahalanobis2(self, fix: VisualFix) -> float:
        y = fix.pos - self.x[:2]
        S = self.P[:2, :2] + fix.cov
        return float(y @ np.linalg.solve(S, y))

    def update_fix(self, fix: VisualFix) -> bool:
        """Fuse a visual fix. Returns False (and sets fix.accepted) if gated out as an outlier."""
        H = np.zeros((2, 6))
        H[:, :2] = np.eye(2)
        d2 = self.mahalanobis2(fix)
        if d2 > self.cfg.gate_chi2 and self.consecutive_rejects < MAX_CONSECUTIVE_REJECTS:
            self.consecutive_rejects += 1
            self.n_fix_rejected += 1
            fix.accepted = False
            return False
        if d2 > self.cfg.gate_chi2:  # persistent disagreement: trust the camera, re-localize
            self.P[:2, :2] += np.eye(2) * d2 * float(np.trace(fix.cov))
        self._update(H, fix.pos, fix.cov)
        self.consecutive_rejects = 0
        self.n_fix_accepted += 1
        fix.accepted = True
        return True

    def pose(self, R: np.ndarray, omega: np.ndarray, yaw: float) -> PoseEstimate:
        return PoseEstimate(self.t, self.x[:3].copy(), self.x[3:].copy(), yaw, self.P.copy(), R.copy(), omega.copy(), "ekf")
