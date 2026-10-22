"""V3 navigation EKF: the inertial backbone that the map-referenced fixes (TRN, IBN) correct.

State (9): [x, y, z, vx, vy, vz, dpsi, bax, bay] in world ENU.
  dpsi      heading error of the AHRS (compass bias) [rad]: the IMU acceleration is rotated with a yaw that is
            off by dpsi, so the true acceleration is a_est - dpsi * (-a_y, a_x). Observable from position fixes
            whenever the vehicle accelerates (take-off, turns, speed changes).
  bax, bay  horizontal acceleration bias in world frame [m/s^2]: accelerometer bias residual + gravity leaking
            through the AHRS tilt error (the dominant dead-reckoning error, see V2 notes). First-order
            Gauss-Markov (time constant `bias_tau`): the tilt error builds up in manoeuvres and the AHRS removes
            it again in steady flight, so an estimate learned in cruise must fade (a random walk kept a stale
            bias through the hover above B and drifted 15 m).

Why: V2's 6-state filter had no bias states, so its covariance under-predicted the drift (V2 demo 06: 70 m after
60 s) and every fix was forgotten as soon as it was fused. With these states a TRN / IBN update zone also
calibrates the INS, so the drift between zones is smaller and the covariance is honest (health, planner, gating).

Measurements: barometer (z), position fix (x, y) with a NIS value for the integrity monitor, zero-velocity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..messages import PoseEstimate


@dataclass
class NavEkfConfig:
    accel_noise: float = 0.12  # m/s^2/sqrt(Hz), white acceleration noise (IMU noise + model error)
    heading_rw: float = 2e-4  # rad/sqrt(s), heading-error random walk
    # The 'bias' is mostly gravity leaking through the AHRS tilt error (1 deg = 0.17 m/s^2), which changes with
    # every manoeuvre: an honest model lets the map fixes learn it (0.03 m/s^2 random walk was overconfident and
    # left a 2-3 m/s velocity error that a position fix could not correct; 0.15 was so pessimistic that the
    # TRN cloud diffused faster than the terrain could focus it).
    bias_std: float = 0.08  # m/s^2, stationary std of the Gauss-Markov accel bias
    bias_tau: float = 30.0  # s, its correlation time
    init_pos_std: float = 1.0
    init_vel_std: float = 0.1
    init_heading_std: float = 0.03  # rad (~1.7 deg compass bias + AHRS settling)
    init_bias_std: float = 0.04  # m/s^2 (on the ground: calibrated, level pad)
    baro_std: float = 0.5
    zupt_std: float = 0.05


class NavEkf:
    N = 9

    def __init__(self, cfg: NavEkfConfig, pos0: np.ndarray) -> None:
        self.cfg = cfg
        self.x = np.zeros(self.N)
        self.x[:3] = pos0
        self.P = np.diag([cfg.init_pos_std**2] * 3 + [cfg.init_vel_std**2] * 3
                         + [cfg.init_heading_std**2] + [cfg.init_bias_std**2] * 2)
        self.t = 0.0
        self._I = np.eye(self.N)
        # Accumulated horizontal position variance growth from predictions only (per axis, m^2): the TRN
        # particle filter diffuses its cloud by exactly this, so it spreads as fast as the INS error grows.
        self.pos_var_growth = 0.0
        self.vel_var_growth = 0.0  # same for the horizontal velocity (TRN particles' velocity-error random walk)

    # ------------------------------------------------------------------ predict

    def predict(self, acc_world: np.ndarray, dt: float, t: float) -> None:
        self.t = t
        if dt <= 0:
            return
        c = self.cfg
        dpsi, ba = self.x[6], self.x[7:9]
        a = np.array(acc_world, float)
        # correct the measured acceleration with the current error estimates
        a_corr = a.copy()
        a_corr[0] += dpsi * a[1] - ba[0]
        a_corr[1] += -dpsi * a[0] - ba[1]
        x = self.x
        x[:3] += x[3:6] * dt + 0.5 * a_corr * dt * dt
        x[3:6] += a_corr * dt

        F = self._I.copy()
        F[0:3, 3:6] = np.eye(3) * dt
        # d(v)/d(dpsi) = (a_y, -a_x), d(v)/d(ba) = -I
        F[3, 6], F[4, 6] = a[1] * dt, -a[0] * dt
        F[3, 7], F[4, 8] = -dt, -dt
        F[0, 6], F[1, 6] = 0.5 * a[1] * dt * dt, -0.5 * a[0] * dt * dt
        F[0, 7], F[1, 8] = -0.5 * dt * dt, -0.5 * dt * dt
        decay = np.exp(-dt / c.bias_tau)
        F[7, 7] = F[8, 8] = decay
        self.x[7:9] *= decay
        q = c.accel_noise**2
        Q = np.zeros((self.N, self.N))
        Q[:3, :3] = np.eye(3) * q * dt**3 / 3
        Q[:3, 3:6] = Q[3:6, :3] = np.eye(3) * q * dt**2 / 2
        Q[3:6, 3:6] = np.eye(3) * q * dt
        Q[6, 6] = c.heading_rw**2 * dt
        Q[7, 7] = Q[8, 8] = c.bias_std**2 * (1.0 - decay * decay)
        p_old = self.P[0, 0] + self.P[1, 1]
        v_old = self.P[3, 3] + self.P[4, 4]
        self.P = F @ self.P @ F.T + Q
        self.pos_var_growth += max(0.5 * (self.P[0, 0] + self.P[1, 1] - p_old), 0.0)
        self.vel_var_growth += max(0.5 * (self.P[3, 3] + self.P[4, 4] - v_old), 0.0)

    # ------------------------------------------------------------------ updates

    def _update(self, H: np.ndarray, z: np.ndarray, R: np.ndarray) -> None:
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[6] = float(np.clip(self.x[6], -0.15, 0.15))  # physical bounds of the error states
        self.x[7:9] = np.clip(self.x[7:9], -0.6, 0.6)
        I_KH = self._I - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form
        self.P = 0.5 * (self.P + self.P.T)

    def update_baro(self, alt: float) -> None:
        H = np.zeros((1, self.N))
        H[0, 2] = 1.0
        self._update(H, np.array([alt]), np.array([[self.cfg.baro_std**2]]))

    def update_zero_velocity(self) -> None:
        H = np.zeros((3, self.N))
        H[:, 3:6] = np.eye(3)
        self._update(H, np.zeros(3), np.eye(3) * self.cfg.zupt_std**2)

    def nis_position(self, pos: np.ndarray, cov: np.ndarray) -> float:
        """Normalized innovation squared of a horizontal position fix (2 dof)."""
        y = np.asarray(pos, float)[:2] - self.x[:2]
        S = self.P[:2, :2] + cov
        return float(y @ np.linalg.solve(S, y))

    def update_position(self, pos: np.ndarray, cov: np.ndarray) -> None:
        H = np.zeros((2, self.N))
        H[:, :2] = np.eye(2)
        self._update(H, np.asarray(pos, float)[:2], np.asarray(cov, float))

    def reset_position(self, pos: np.ndarray, cov: np.ndarray) -> None:
        """Re-localization: replace the horizontal position (keep the rest), decorrelate it."""
        self.x[:2] = np.asarray(pos, float)[:2]
        self.P[:2, :] = 0.0
        self.P[:, :2] = 0.0
        self.P[:2, :2] = np.asarray(cov, float)

    # ------------------------------------------------------------------ output

    @property
    def sigma2_h(self) -> float:
        return 2.0 * float(np.sqrt(max(np.linalg.eigvalsh(self.P[:2, :2]).max(), 0.0)))

    def pose(self, R: np.ndarray, omega: np.ndarray, yaw: float) -> PoseEstimate:
        # The AHRS attitude is corrected by the estimated heading error (controller, camera geometry).
        c, s = np.cos(-self.x[6]), np.sin(-self.x[6])
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return PoseEstimate(self.t, self.x[:3].copy(), self.x[3:6].copy(), yaw - float(self.x[6]),
                            self.P[:6, :6].copy(), Rz @ R, omega.copy(), "ekf")
