"""TRN - terrain-referenced navigation: nadir lidar surface profile matched against the onboard DSM.

Ported from branch `death-v3` (perception/trn.py, there a single-beam altimeter on the V3 worlds); here the
measurement is the 8-beam nadir lidar fan (`lidar` topic, +-8 deg cross-track, 100 m range) and the DSM is
world/onboard_dsm.py (flat Berkeley terrain + OSM building footprints). Onboard only: no truth.

Measurement: each beam k gives a slant range r_k along a known body direction; with the AHRS attitude its world
direction d_k is known, so the measured SURFACE height where the beam hits is
    h_k = z_est + r_k * d_k,z          (z_est: EKF height, barometric, in the onboard map's datum)
at the horizontal offset r_k * d_k,xy from the vehicle (tilt handled, not assumed vertical).

Particle filter over (x, y, b, ex, ey):
  x, y      horizontal position
  b         vertical bias between the barometric height and the map datum (baro drift + map offset). It is
            LEARNED and kept: a restart (lost / re-initialization) starts around the learned value (in V3 a
            restart around 0 began 12 m off, matched the profile at the wrong place and dragged the INS 16 m).
  ex, ey    error of the INS velocity carried by this particle (coherent INS drift, not white noise)
Propagation: x += (v_ekf + e) dt; e random-walks with the EKF's own velocity-variance growth. Weight: robust
Student-t likelihood of h_k - (DSM(particle + r_k d_k,xy) + b) with an outlier floor (trees, OBB-vs-polygon
walls), the beams sharing `beam_weight` each, tempered by (distance since the last update / map-error
correlation length) since consecutive scans see the same map error. Systematic resampling.

Output: a position fix (mean + inflated covariance) only when the cloud is unimodal and compact, after enough
track has been matched, and only after `fix_travel` m of new track (hovering re-publishes the same information).
Over flat ground (lawns, plazas) the terrain carries no information: the cloud spreads with the INS error and
no fix comes out; `info` reports the terrain roughness along the recent track.
"""

from __future__ import annotations

import math
import time
from collections import deque

import numpy as np

from ..config import TrnConfig
from ..messages import LidarMsg, PoseEstimate, VisualFix
from ..world.surface import Grid


def beam_dirs_body(angles: np.ndarray) -> np.ndarray:
    """(n, 3) beam directions in the body frame (FLU) for fan angles from straight down, + = body right."""
    a = np.asarray(angles, float)
    return np.column_stack([np.zeros_like(a), -np.sin(a), -np.cos(a)])


class TrnFilter:
    name = "trn"

    def __init__(self, cfg: TrnConfig, dsm: Grid, seed: int = 0) -> None:
        self.cfg = cfg
        self.dsm = dsm
        self.rng = np.random.default_rng(seed + 2000)
        self.p: np.ndarray | None = None  # (N, 5) x, y, b, ex, ey
        self.w: np.ndarray | None = None
        self.t_last: float | None = None
        self.t_last_update = -math.inf
        self.t_last_fix = -math.inf
        self.dist_since_update = 0.0
        self.travel_since_init = 0.0
        self.last_good_lik_t = 0.0
        self.s_fix_ref: float | None = None  # travel_since_init at the last published fix
        self.proc_ms = 0.0  # cost of the last measurement update
        self.ms_total = 0.0
        self.n_timed = 0
        self.needs_reset = False
        self.n_updates = 0
        self.n_fixes = 0
        self.n_resets = 0
        self.status = "idle"
        self.info = 0.0  # m, terrain roughness along the recent track (information content)
        self.est: np.ndarray | None = None
        self.est_cov: np.ndarray | None = None
        self.bias = 0.0
        self.bias_known = False  # set once the cloud converged (a restart then keeps the bias)
        self.track: deque[tuple[float, float]] = deque()  # (s, map height at the estimate)
        self.s_total = 0.0
        self.last_pose_xy: np.ndarray | None = None
        self._var_growth_ref: float | None = None

    # ------------------------------------------------------------------ lifecycle

    @property
    def initialized(self) -> bool:
        return self.p is not None

    def reset(self, pose: PoseEstimate, t: float, inflate: float | None = None) -> None:
        c = self.cfg
        n = c.n_particles
        cov = pose.cov[:2, :2] * (inflate or c.init_inflate) ** 2
        evals, evecs = np.linalg.eigh(cov)
        std = np.clip(np.sqrt(np.maximum(evals, 0.0)), c.init_min_std, c.init_max_std)
        p = np.zeros((n, 5))
        p[:, :2] = pose.pos[:2] + (self.rng.normal(0, 1, (n, 2)) * std) @ evecs.T
        b0, bstd = (self.bias, c.bias_keep_std) if self.bias_known else (0.0, c.bias_init_std)
        p[:, 2] = self.rng.normal(b0, bstd, n)
        ve, vv = np.linalg.eigh(pose.cov[3:5, 3:5] * (inflate or c.init_inflate) ** 2)
        vstd = np.clip(np.sqrt(np.maximum(ve, 0.0)), c.vel_err_min_std, 5.0)
        p[:, 3:5] = (self.rng.normal(0, 1, (n, 2)) * vstd) @ vv.T
        self.p = p
        self.w = np.full(n, 1.0 / n)
        self.t_last = t
        self.travel_since_init = 0.0
        self.dist_since_update = 0.0
        self.last_good_lik_t = t
        self.needs_reset = False
        self.n_resets += 1
        self.status = "init"
        self.s_fix_ref = None

    # ------------------------------------------------------------------ filter steps

    def _predict(self, vel_xy: np.ndarray, dt: float, vel_var_inc: float) -> None:
        c = self.cfg
        p = self.p
        n = len(p)
        ds = float(np.hypot(*vel_xy)) * dt
        p[:, 0:2] += (vel_xy + p[:, 3:5]) * dt + self.rng.normal(0, c.pos_rw * math.sqrt(max(ds, 1e-6)), (n, 2))
        p[:, 3:5] += self.rng.normal(0, math.sqrt(max(vel_var_inc, 0.0) + c.vel_err_rw**2 * dt), (n, 2))
        p[:, 2] += self.rng.normal(0, c.bias_rw * math.sqrt(dt), n)
        self.dist_since_update += ds
        self.travel_since_init += ds

    def _update(self, h_meas: np.ndarray, hit_off: np.ndarray, t: float) -> float:
        """h_meas (k,), hit_off (k, 2): one scan of k valid beams."""
        c = self.cfg
        p, w = self.p, self.w
        xy = p[:, None, :2] + hit_off[None, :, :]  # (N, k, 2)
        e = h_meas[None, :] - (self.dsm.sample(xy) + p[:, 2:3])
        lik = (1.0 + e * e / (c.nu * c.sigma_meas**2)) ** (-(c.nu + 1) / 2)
        lik = (1 - c.outlier_floor) * lik + c.outlier_floor
        lik[~self.dsm.contains(xy)] = c.outlier_floor
        alpha = float(np.clip(self.dist_since_update / c.corr_length, 0.02, 1.0))
        loglik = np.log(lik).sum(axis=1) * c.beam_weight  # neighbouring beams share most of their map error
        mean_lik = float(w @ np.exp(np.log(lik).mean(axis=1)))
        logw = np.log(np.maximum(w, 1e-300)) + alpha * loglik
        logw -= logw.max()
        w = np.exp(logw)
        s = w.sum()
        self.w = w / s if np.isfinite(s) and s > 0 else np.full(len(w), 1.0 / len(w))
        self.dist_since_update = 0.0
        self.n_updates += 1
        if mean_lik > 3 * c.outlier_floor:
            self.last_good_lik_t = t
        elif t - self.last_good_lik_t > c.lost_after_s:
            self.needs_reset = True
            self.status = "lost"
        ess = 1.0 / float((self.w * self.w).sum())
        if ess < 0.5 * len(self.w):
            self._resample()
        return mean_lik

    def _resample(self) -> None:
        n = len(self.w)
        u = (self.rng.random() + np.arange(n)) / n
        idx = np.minimum(np.searchsorted(np.cumsum(self.w), u), n - 1)
        self.p = self.p[idx].copy()
        self.w = np.full(n, 1.0 / n)
        # roughening keeps diversity after duplicating particles
        self.p[:, 0:2] += self.rng.normal(0, 0.2 * self.dsm.mpp + 0.05, (n, 2))
        self.p[:, 2] += self.rng.normal(0, 0.02, n)

    # ------------------------------------------------------------------ estimate

    def _estimate(self, ins_std: float) -> tuple[np.ndarray, np.ndarray, float, bool, str]:
        c = self.cfg
        p, w = self.p, self.w
        mean = w @ p[:, :2]
        d = p[:, :2] - mean
        cov = (d * w[:, None]).T @ d
        bias = float(w @ p[:, 2])
        std_max = float(np.sqrt(max(np.linalg.eigvalsh(cov).max(), 0.0)))
        if self.travel_since_init < c.min_travel:
            return mean, cov, bias, False, f"collecting {self.travel_since_init:.0f}/{c.min_travel:.0f} m"
        limit = min(max(c.max_fix_std, c.rel_fix_std * ins_std), c.cap_fix_std)
        if std_max > limit:
            return mean, cov, bias, False, f"cloud too wide ({std_max:.1f} m)"
        # unimodal: (almost) all weight close to the mean and the densest bin near the mean
        r = np.hypot(d[:, 0], d[:, 1])
        if float(w[r < 3.0 * std_max + 1e-6].sum()) < 0.95:
            return mean, cov, bias, False, "multimodal"
        b = max(std_max / 2.0, self.dsm.mpp)
        ij = np.floor(d / b).astype(np.int64)
        keys, inv = np.unique(ij, axis=0, return_inverse=True)
        mass = np.bincount(inv.ravel(), weights=w, minlength=len(keys))
        mode = (keys[int(np.argmax(mass))] + 0.5) * b
        if float(np.hypot(*mode)) > max(1.5 * b, std_max):
            return mean, cov, bias, False, "multimodal"
        return mean, cov, bias, True, f"converged ({std_max:.1f} m)"

    # ------------------------------------------------------------------ main entry

    def step(self, msg: LidarMsg, pose: PoseEstimate, var_growth: float) -> VisualFix | None:
        """One lidar scan. `var_growth` = Ekf.vel_var_growth (accumulated). Returns a fix or None."""
        t0 = time.perf_counter()
        updated = self.n_updates
        try:
            return self._step(msg, pose, var_growth)
        finally:
            if self.n_updates != updated:  # a measurement update ran (not just a prediction)
                self.proc_ms = (time.perf_counter() - t0) * 1000.0
                self.ms_total += self.proc_ms
                self.n_timed += 1

    def _step(self, msg: LidarMsg, pose: PoseEstimate, var_growth: float) -> VisualFix | None:
        c = self.cfg
        t = msg.t
        if self.p is None:
            self.reset(pose, t)
        if self._var_growth_ref is None:
            self._var_growth_ref = var_growth
        var_inc, self._var_growth_ref = var_growth - self._var_growth_ref, var_growth
        dt = t - self.t_last if self.t_last is not None else 0.0
        self.t_last = t
        if dt > 0:
            self._predict(pose.vel[:2], dt, var_inc)
        r = np.asarray(msg.ranges, float)
        ok = np.isfinite(r) & (r > c.min_range)
        if not ok.any():
            return None
        if t - self.t_last_update < 1.0 / c.update_hz - 1e-6:
            return None
        self.t_last_update = t
        d = beam_dirs_body(np.asarray(msg.angles, float)[ok]) @ pose.R.T  # world beam directions
        h_meas = pose.pos[2] + r[ok] * d[:, 2]
        hit_off = r[ok, None] * d[:, :2]
        self._update(h_meas, hit_off, t)
        ins_std = 0.5 * pose.sigma2_horizontal()
        mean, cov, bias, good, why = self._estimate(ins_std)
        self.est, self.est_cov, self.bias = mean, cov, bias
        if good:
            self.bias_known = True
        self._track_info(mean)
        if self.status != "lost":
            self.status = why
        if not good or t - self.t_last_fix < c.fix_interval:
            return None
        if (self.s_fix_ref is not None and self.travel_since_init - self.s_fix_ref < c.fix_travel
                and t - self.t_last_fix < c.hover_fix_s):
            return None
        self.s_fix_ref = self.travel_since_init
        self.t_last_fix = t
        self.n_fixes += 1
        R = cov * c.fix_inflate**2 + np.eye(2) * c.min_fix_std**2
        std_max = float(np.sqrt(max(np.linalg.eigvalsh(cov).max(), 0.0)))
        strong = std_max < 0.5 * c.max_fix_std and self.travel_since_init > 2.0 * c.min_travel
        return VisualFix(t, mean.copy(), R, "trn", n_inliers=self.n_updates, quality=0.9 if strong else 0.5)

    def ins_corrected(self, dv: np.ndarray) -> None:
        """The EKF velocity the particles ride on just changed by dv (a fused fix): the particles' INS velocity
        errors lose the same amount, so the correction is not counted twice."""
        if self.p is not None:
            self.p[:, 3:5] -= dv

    def _track_info(self, xy: np.ndarray) -> None:
        if self.last_pose_xy is not None:
            self.s_total += float(np.hypot(*(xy - self.last_pose_xy)))
        self.last_pose_xy = xy.copy()
        self.track.append((self.s_total, float(self.dsm.sample(xy))))
        while self.track and self.s_total - self.track[0][0] > self.cfg.info_track_m:
            self.track.popleft()
        if len(self.track) >= 5:
            s, h = np.array(self.track).T
            A = np.column_stack([s, np.ones_like(s)])
            res = h - A @ np.linalg.lstsq(A, h, rcond=None)[0]
            self.info = float(res.std())

    def cloud(self, max_n: int = 600) -> tuple[np.ndarray, np.ndarray] | None:
        """Subsample of particles (xy) and weights for display."""
        if self.p is None:
            return None
        idx = np.linspace(0, len(self.p) - 1, min(max_n, len(self.p))).astype(int)
        return self.p[idx, :2].copy(), self.w[idx].copy()
