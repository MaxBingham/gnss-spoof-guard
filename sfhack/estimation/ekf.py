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
- GNSS -> GNSS-free transition (V3.2): one update may move (bx, by) by at most `acc_bias_step_max` (a naive
  50 m GNSS jump otherwise became a 5 deg AHRS tilt). `gnss_lost_now` detects the end of GNSS fixes; the
  autopilot then either adopts its GNSS-free track (`adopt`) or calls `inflate_for_gnss_loss` (covariance
  floor; the first STRONG vision fix, quality >= `reloc_min_quality`, re-localizes, weaker fixes wait).

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
        self.last_gnss_t: float | None = None
        self.gnss_lost = False  # had GNSS, then none for gnss_loss_after_s
        self.relocalizing = False  # after GNSS loss, until the first strong fix is accepted
        self.n_weak_skipped = 0  # vision fixes not used while relocalizing (quality too low)
        self.nis_log: list[tuple[float, float, bool]] = []  # (t, NIS, accepted) for every visual fix
        self.last_decision = ""  # set when a fix is accepted by re-localization
        self.bias_handed = np.zeros(2)  # (bx, by) handed to the AHRS by the last take_tilt_correction
        self.vel_var_growth = 0.0  # accumulated horizontal velocity-variance growth (TRN particle random walk)
        self.n_vel_accepted = 0  # velocity measurements (optical flow)
        self.n_vel_rejected = 0
        self.vel_consecutive_rejects = 0
        self.vel_nis_log: list[tuple[float, float, bool]] = []

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
        v_old = self.P[3, 3] + self.P[4, 4]
        self.P = F @ self.P @ F.T + Q
        self.vel_var_growth += max(0.5 * (self.P[3, 3] + self.P[4, 4] - v_old), 0.0)

    def take_tilt_correction(self, g: float = 9.81) -> np.ndarray:
        """Error-state reset: hand the learned bias to the attitude as a world-frame tilt correction
        phi (gravity leak g (phi x z) = -b) and zero the bias. Returns phi for Ahrs.correct_tilt_world."""
        bx, by = self.x[6], self.x[7]
        self.bias_handed = np.array([bx, by])  # a parallel filter on the same AHRS subtracts this (see autopilot)
        self.x[6:8] = 0.0
        return np.array([by / g, -bx / g, 0.0])

    def _update(self, H: np.ndarray, z: np.ndarray, R: np.ndarray) -> None:
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y
        db = float(np.hypot(dx[6], dx[7]))
        if db > self.cfg.acc_bias_step_max:  # one fix cannot tilt the AHRS by degrees (see module doc)
            dx[6:8] *= self.cfg.acc_bias_step_max / db
        self.x = self.x + dx
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

    def update_fix(self, fix: VisualFix, allow_reloc: bool = True) -> bool:
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
        was_relocalizing = self.relocalizing
        if self.relocalizing and fix.quality < self.cfg.reloc_min_quality:
            self.n_weak_skipped += 1
            fix.accepted = False
            return False
        H = np.zeros((2, N_STATES))
        H[:, :2] = np.eye(2)
        d2 = self.mahalanobis2(fix)
        R = fix.cov
        if d2 > self.cfg.gate_chi2:
            if not allow_reloc:  # (TRN: a wrong terrain mode must not force its way in)
                self.n_fix_rejected += 1
                self.nis_log.append((fix.t, d2, False))
                fix.accepted = False
                return False
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
        self.relocalizing = False
        if was_relocalizing and not self.last_decision:  # (a forced accept above already logged itself)
            self.n_relocalized += 1
            self.last_decision = (f"RELOCALIZE: first strong vision fix after GNSS loss (quality {fix.quality:.2f}, "
                                  f"NIS {d2:.1f}, offset {np.linalg.norm(fix.pos - self.x[:2]):.1f} m)")
        self._update(H, fix.pos, R)
        self.nis_log.append((fix.t, d2, True))
        self.consecutive_rejects = 0
        self.n_fix_accepted += 1
        self.last_fix_t = fix.t
        fix.accepted = True
        return True

    def update_velocity(self, vel_xy: np.ndarray, cov: np.ndarray) -> bool:
        """Horizontal ground velocity (recon-nav: optical flow), chi-square gated (2 dof). It makes the accel-bias
        states (AHRS tilt error x g) observable without position fixes. Returns False if gated out."""
        H = np.zeros((2, N_STATES))
        H[:, 3:5] = np.eye(2)
        y = np.asarray(vel_xy, float) - self.x[3:5]
        S = self.P[3:5, 3:5] + cov
        d2 = float(y @ np.linalg.solve(S, y))
        R = np.asarray(cov, float)
        if d2 > self.cfg.gate_chi2:
            self.vel_consecutive_rejects += 1
            if self.vel_consecutive_rejects <= MAX_CONSECUTIVE_REJECTS:
                self.vel_nis_log.append((self.t, d2, False))
                self.n_vel_rejected += 1
                return False
            R = R * (d2 / self.cfg.gate_chi2)  # the flow keeps disagreeing: the INS velocity drifted (re-acquire)
        self.vel_nis_log.append((self.t, d2, True))
        self.vel_consecutive_rejects = 0
        self._update(H, np.asarray(vel_xy, float), R)
        self.n_vel_accepted += 1
        return True

    def update_gnss(self, t: float, pos_xy: np.ndarray, cov: np.ndarray) -> None:
        """NAIVE baseline: fuse every GNSS fix as reported, no gate (a spoofed fix is believed)."""
        H = np.zeros((2, N_STATES))
        H[:, :2] = np.eye(2)
        self._update(H, np.asarray(pos_xy, float)[:2], np.asarray(cov, float))
        self.n_gnss += 1
        self.last_fix_t = t
        self.last_gnss_t = t
        self.gnss_lost = False
        self.relocalizing = False

    def gnss_lost_now(self, t: float) -> bool:
        """True exactly once: on the first step `gnss_loss_after_s` after the last GNSS fix (until GNSS returns)."""
        if self.last_gnss_t is None or self.gnss_lost or t - self.last_gnss_t < self.cfg.gnss_loss_after_s:
            return False
        self.gnss_lost = True
        return True

    def inflate_for_gnss_loss(self) -> str:
        """Floor the position / velocity covariance (the GNSS fixes may have been spoofed: their tiny covariance
        is no promise) and wait for a strong vision fix to re-localize. Returns the decision text."""
        c = self.cfg
        self.relocalizing = True
        s_before = float(np.sqrt(max(self.P[0, 0], self.P[1, 1])))
        for i in (0, 1):
            self.P[i, i] = max(self.P[i, i], c.gnss_loss_pos_std**2)
        for i in (3, 4):
            self.P[i, i] = max(self.P[i, i], c.gnss_loss_vel_std**2)
        return (f"GNSS_LOST: no healthy GNSS-free track, position std {s_before:.1f} -> {c.gnss_loss_pos_std:.0f} m, "
                f"waiting for a strong vision fix")

    def adopt(self, other: "Ekf") -> float:
        """Take over another filter's state and covariance (GNSS-free track handover). Returns the jump (m)."""
        jump = float(np.linalg.norm(other.x[:2] - self.x[:2]))
        self.x = other.x.copy()
        self.P = other.P.copy()
        self.last_fix_t = other.last_fix_t
        self.consecutive_rejects = 0
        self.relocalizing = False
        return jump

    def sigma2_horizontal(self) -> float:
        return 2.0 * float(np.sqrt(max(np.linalg.eigvalsh(self.P[:2, :2]).max(), 0.0)))

    def pose(self, R: np.ndarray, omega: np.ndarray, yaw: float, *, report: bool = True) -> PoseEstimate:
        P = self.P[:6, :6].copy()
        if report:
            self._floor_horizontal(P)
        return PoseEstimate(self.t, self.x[:3].copy(), self.x[3:6].copy(), yaw, P, R.copy(), omega.copy(), "ekf")

    def _floor_horizontal(self, P: np.ndarray) -> None:
        """Raise the reported horizontal covariance so its major-axis 2σ is at least the honest-flight floor.

        Added as d·I on the position block (eigenvalues shift by d), which keeps the matrix PSD.
        The filter's internal P is left alone, so a spoof is still fused and the claim does not swell to cover it.
        """
        floor = self.cfg.horiz_std_floor ** 2
        ev = float(np.linalg.eigvalsh(0.5 * (P[:2, :2] + P[:2, :2].T)).max())
        short = floor - ev
        if short > 0.0:
            P[0, 0] += short
            P[1, 1] += short

    @property
    def acc_bias(self) -> np.ndarray:
        return self.x[6:8].copy()
