"""Position/velocity EKF: state [x, y, z, vx, vy, vz, bx, by] in the world (ENU) frame.

- Predict: IMU specific force rotated to world with the AHRS attitude, minus gravity, minus the
  horizontal acceleration bias (bx, by). The bias absorbs AHRS tilt error x g: flying on the EKF the
  tilt error is unobservable for the accelerometer (the controller holds the drone tilted by exactly
  the error), so only the visual fixes can reveal it. Without it, 1-2 deg of tilt error made blind
  stretches (near B, descent) drift by tens of meters. The autopilot moves the learned bias into the
  AHRS attitude every IMU step (`take_tilt_correction`, error-state reset), so the controller flies on
  the corrected attitude too.
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

N_STATES = 8  # x, y, z, vx, vy, vz, bx, by
MAX_CONSECUTIVE_REJECTS = 5  # then the next fix is accepted with inflated noise (re-localization)


class Ekf:
    def __init__(self, cfg: EkfConfig, pos0: np.ndarray, vel0: np.ndarray | None = None) -> None:
        self.cfg = cfg
        self.x = np.zeros(N_STATES)
        self.x[:3] = pos0
        if vel0 is not None:
            self.x[3:6] = vel0
        self.P = np.diag([cfg.init_pos_std**2] * 3 + [cfg.init_vel_std**2] * 3 + [cfg.init_acc_bias_std**2] * 2)
        self.t = 0.0
        self.consecutive_rejects = 0
        self.n_fix_accepted = 0
        self.n_fix_rejected = 0
        self.n_relocalized = 0
        self.n_gnss = 0
        self.last_fix_t: float | None = None
        self.nis_log: list[tuple[float, float, bool]] = []  # (t, NIS, accepted) for every visual fix
        self.last_decision = ""  # set when a fix is accepted by re-localization

    def predict(self, acc_world: np.ndarray, dt: float, t: float) -> None:
        self.t = t
        if dt <= 0:
            return
        F = np.eye(N_STATES)
        F[:3, 3:6] = np.eye(3) * dt
        F[0:2, 6:8] = -np.eye(2) * 0.5 * dt * dt
        F[3:5, 6:8] = -np.eye(2) * dt
        acc = acc_world.copy()
        acc[:2] -= self.x[6:8]
        self.x = F @ self.x
        self.x[:3] += 0.5 * acc * dt * dt
        self.x[3:6] += acc * dt
        q = self.cfg.accel_process_std**2
        Q = np.zeros((N_STATES, N_STATES))
        Q[:3, :3] = np.eye(3) * q * dt**3 / 3
        Q[:3, 3:6] = Q[3:6, :3] = np.eye(3) * q * dt**2 / 2
        Q[3:6, 3:6] = np.eye(3) * q * dt
        Q[6:8, 6:8] = np.eye(2) * self.cfg.acc_bias_rw**2 * dt
        self.P = F @ self.P @ F.T + Q

    def take_tilt_correction(self, g: float = 9.81) -> np.ndarray:
        """Error-state reset: hand the learned bias to the attitude as a world-frame tilt correction
        phi (gravity leak g (phi x z) = -b) and zero the bias. Returns phi for Ahrs.correct_tilt_world."""
        bx, by = self.x[6], self.x[7]
        self.x[6:8] = 0.0
        return np.array([by / g, -bx / g, 0.0])

    def _update(self, H: np.ndarray, z: np.ndarray, R: np.ndarray) -> None:
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(N_STATES) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form

    def update_baro(self, alt: float) -> None:
        H = np.zeros((1, N_STATES))
        H[0, 2] = 1.0
        self._update(H, np.array([alt]), np.array([[self.cfg.baro_std**2]]))

    def update_zero_velocity(self) -> None:
        H = np.zeros((3, N_STATES))
        H[:, 3:6] = np.eye(3)
        self._update(H, np.zeros(3), np.eye(3) * self.cfg.zupt_std**2)

    def mahalanobis2(self, fix: VisualFix) -> float:
        y = fix.pos - self.x[:2]
        S = self.P[:2, :2] + fix.cov
        return float(y @ np.linalg.solve(S, y))

    def update_fix(self, fix: VisualFix) -> bool:
        """Fuse a visual fix with a chi-square gate on the NIS (2 dof). Returns False if gated out.

        - Long dead-reckoning stretches: the KF covariance already grows with the process noise; a
          floor (`nofix_inflation`) keeps a correct fix from being rejected after many seconds blind.
        - After MAX_CONSECUTIVE_REJECTS the camera keeps disagreeing: accept the next fix with its
          noise inflated to make it consistent (re-localization) and report it via `last_decision`.
        """
        self.last_decision = ""
        if self.last_fix_t is not None and fix.t - self.last_fix_t > self.cfg.nofix_inflate_after_s:
            blind = fix.t - self.last_fix_t
            floor = (self.cfg.nofix_inflation * blind) ** 2
            for i in (0, 1):
                self.P[i, i] = max(self.P[i, i], floor)
        H = np.zeros((2, N_STATES))
        H[:, :2] = np.eye(2)
        d2 = self.mahalanobis2(fix)
        R = fix.cov
        if d2 > self.cfg.gate_chi2:
            if self.consecutive_rejects < MAX_CONSECUTIVE_REJECTS:
                self.consecutive_rejects += 1
                self.n_fix_rejected += 1
                self.nis_log.append((fix.t, d2, False))
                fix.accepted = False
                return False
            R = fix.cov * (d2 / self.cfg.gate_chi2)  # inflate the measurement noise until it is consistent
            self.n_relocalized += 1
            self.last_decision = (f"RELOCALIZE: accepted fix after {self.consecutive_rejects} rejections "
                                  f"(NIS {d2:.1f}, offset {np.linalg.norm(fix.pos - self.x[:2]):.1f} m)")
        self._update(H, fix.pos, R)
        self.nis_log.append((fix.t, d2, True))
        self.consecutive_rejects = 0
        self.n_fix_accepted += 1
        self.last_fix_t = fix.t
        fix.accepted = True
        return True

    def update_gnss(self, t: float, pos_xy: np.ndarray, cov: np.ndarray) -> None:
        """NAIVE baseline: fuse every GNSS fix as reported, no gate (a spoofed fix is believed)."""
        H = np.zeros((2, N_STATES))
        H[:, :2] = np.eye(2)
        self._update(H, np.asarray(pos_xy, float)[:2], np.asarray(cov, float))
        self.n_gnss += 1
        self.last_fix_t = t

    def pose(self, R: np.ndarray, omega: np.ndarray, yaw: float) -> PoseEstimate:
        return PoseEstimate(self.t, self.x[:3].copy(), self.x[3:6].copy(), yaw, self.P[:6, :6].copy(), R.copy(), omega.copy(), "ekf")

    @property
    def acc_bias(self) -> np.ndarray:
        return self.x[6:8].copy()
