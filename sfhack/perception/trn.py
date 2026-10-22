"""TRN - terrain-referenced navigation: altimeter surface profile matched against the onboard DSM.

Measurement: the downward altimeter gives the slant range r along the body -z axis. With the AHRS attitude the
beam direction d (world) is known, so the measured SURFACE height under the beam is
    h_meas = z_est + r * d_z          (z_est: EKF height, barometric, in the onboard map's datum)
and the beam hits the ground at the horizontal offset r * d_xy from the vehicle (tilt handled here, not by
assuming a vertical beam).

Particle filter over (x, y, b, ex, ey):
  x, y      horizontal position
  b         vertical bias between the barometric height and the onboard map datum (baro drift + map offset)
  ex, ey    error of the INS velocity carried by this particle
Propagation: x += (v_ekf + e) dt; e random-walks with the EKF's own velocity-variance growth
(`NavEkf.vel_var_growth`) and starts from the EKF velocity covariance. The INS error is COHERENT (a slowly
changing velocity error from heading / accel bias), so each particle drifts like a possible INS error history
and the terrain selects consistent (position, velocity-error) pairs, like a batch profile correlator.
Lessons learned: a fixed small roughening made the cloud overconfident (it converged, then walked away with
the INS velocity error); white position diffusion sized to the INS variance growth was honest but too
diffuse to ever converge. Weight: robust Student-t likelihood of
h_meas - (DSM(particle + r d_xy) + b) with an outlier floor, tempered by (distance since last update /
map-error correlation length) because consecutive samples see the same map error. Systematic resampling.

Output: a position fix (mean + inflated covariance) only when the cloud is unimodal and compact, after enough
track has been matched; otherwise None. `info` reports the information content of the terrain along track.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from ..messages import AltimeterMsg, PoseEstimate, VisualFix
from ..world.scenarios import Scenario
from ..world.surface import Grid


@dataclass
class TrnConfig:
    n_particles: int = 3000
    sigma_meas: float = 1.0  # m, altimeter + map error (1 sigma core of the Student-t)
    nu: float = 4.0  # Student-t degrees of freedom (heavy tails: missing buildings, roof edges, trees)
    outlier_floor: float = 0.05  # likelihood floor (fraction of the peak)
    corr_length: float = 6.0  # m, map-error correlation length along track (likelihood tempering)
    update_hz: float = 10.0
    min_range: float = 0.5  # m
    init_inflate: float = 1.5  # initial cloud = EKF covariance x this (std), at least init_min_std
    init_min_std: float = 4.0
    init_max_std: float = 150.0
    bias_init_std: float = 3.0  # m
    bias_keep_std: float = 1.0  # m, bias spread on a restart once the bias has been learned
    bias_rw: float = 0.05  # m/sqrt(s)
    vel_err_min_std: float = 0.2  # m/s, initial spread of the per-particle INS velocity error (at least)
    vel_err_rw: float = 0.02  # m/s/sqrt(s), on top of the EKF's velocity-variance growth
    pos_rw: float = 0.05  # m/sqrt(m travelled), roughening
    min_travel: float = 60.0  # m of matched track after (re)initialization before the first fix
    max_fix_std: float = 6.0  # m, cloud 1-sigma (largest axis) below this always qualifies for a fix ...
    rel_fix_std: float = 0.4  # ... or below this x the INS 1-sigma (a fix that is 2.5x better than the INS) ...
    cap_fix_std: float = 25.0  # ... but never wider than this
    min_fix_std: float = 0.8  # m, floor of the published fix sigma
    fix_inflate: float = 2.0  # published sigma = cloud sigma x this (correlated map errors, shared velocity,
    # and consecutive fixes share most of their terrain information)
    fix_interval: float = 1.0  # s between published fixes
    info_track_m: float = 120.0  # along-track window for the information-content report
    lost_after_s: float = 8.0  # likelihood stays at the outlier floor this long -> request re-initialization
    fix_travel: float = 20.0  # m of new track between two published fixes (no fixes while hovering)


def trn_config_for(sc: Scenario) -> TrnConfig:
    if sc.kind == "urban":
        return TrnConfig(sigma_meas=2.0, nu=3.0, corr_length=3.0, init_min_std=4.0, bias_init_std=4.0,
                         min_travel=60.0, max_fix_std=5.0, min_fix_std=1.0, info_track_m=120.0, pos_rw=0.05)
    return TrnConfig(sigma_meas=3.5, corr_length=40.0, init_min_std=20.0, init_max_std=600.0, bias_init_std=10.0,
                     min_travel=500.0, max_fix_std=30.0, cap_fix_std=100.0, min_fix_std=5.0, info_track_m=1000.0,
                     pos_rw=0.1, n_particles=3000, fix_interval=1.0, bias_rw=0.1, fix_travel=30.0, fix_inflate=1.5)


class TrnFilter:
    name = "trn"

    def __init__(self, cfg: TrnConfig, dsm: Grid, seed: int = 0) -> None:
        self.cfg = cfg
        self.dsm = dsm
        self.rng = np.random.default_rng(seed + 2000)
        self.p: np.ndarray | None = None  # (N, 3) x, y, b
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
        self.status = "idle"
        self.info = 0.0  # m, terrain roughness along the recent track (information content)
        self.est: np.ndarray | None = None
        self.est_cov: np.ndarray | None = None
        self.bias = 0.0
        self.bias_known = False  # set once the cloud converged (a restart then keeps the bias)
        self.track: deque[tuple[float, float]] = deque()  # (s, map height at the estimate)
        self.s_total = 0.0
        self.profile: deque[tuple[float, float, float]] = deque(maxlen=400)  # (t, h_meas, h_map at estimate)
        self.last_pose_xy: np.ndarray | None = None
        self._var_growth_ref: float | None = None
        self.vel_err = np.zeros(2)  # weighted mean INS velocity error (diagnostics)

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
        z = self.rng.normal(0, 1, (n, 2)) * std
        p = np.zeros((n, 5))
        p[:, :2] = pose.pos[:2] + z @ evecs.T
        # the datum bias (baro drift + map offset) changes slowly: a restart keeps the learned value (a restart around
        # 0 began 12 m off on S1, matched the profile at the wrong place and dragged the INS 16 m)
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
        self.status = "init"
        self.s_fix_ref = None

    # ------------------------------------------------------------------ filter steps

    def _predict(self, vel_xy: np.ndarray, dt: float, vel_var_inc: float) -> None:
        c = self.cfg
        p = self.p
        n = len(p)
        speed = float(np.hypot(*vel_xy))
        ds = speed * dt
        p[:, 0:2] += (vel_xy + p[:, 3:5]) * dt + self.rng.normal(0, c.pos_rw * math.sqrt(max(ds, 1e-6)), (n, 2))
        p[:, 3:5] += self.rng.normal(0, math.sqrt(max(vel_var_inc, 0.0) + c.vel_err_rw**2 * dt), (n, 2))
        p[:, 2] += self.rng.normal(0, c.bias_rw * math.sqrt(dt), n)
        self.dist_since_update += ds
        self.travel_since_init += ds

    def _update(self, h_meas: float, hit_off: np.ndarray, t: float) -> float:
        c = self.cfg
        p, w = self.p, self.w
        xy = p[:, :2] + hit_off
        pred = self.dsm.sample(xy) + p[:, 2]
        e = h_meas - pred
        lik = (1.0 + e * e / (c.nu * c.sigma_meas**2)) ** (-(c.nu + 1) / 2)
        lik = (1 - c.outlier_floor) * lik + c.outlier_floor
        lik[~self.dsm.contains(xy)] = c.outlier_floor
        alpha = float(np.clip(self.dist_since_update / c.corr_length, 0.02, 1.0))
        mean_lik = float(w @ lik)
        w = w * lik**alpha
        s = w.sum()
        if not np.isfinite(s) or s <= 0:
            w = np.full(len(w), 1.0 / len(w))
        else:
            w = w / s
        self.w = w
        self.dist_since_update = 0.0
        self.n_updates += 1
        if mean_lik > 3 * c.outlier_floor:
            self.last_good_lik_t = t
        elif t - self.last_good_lik_t > c.lost_after_s:
            self.needs_reset = True
            self.status = "lost"
        ess = 1.0 / float((w * w).sum())
        if ess < 0.5 * len(w):
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
        self.vel_err = w @ p[:, 3:5]
        d = p[:, :2] - mean
        cov = (d * w[:, None]).T @ d
        bias = float(w @ p[:, 2])
        std_max = float(np.sqrt(max(np.linalg.eigvalsh(cov).max(), 0.0)))
        if self.travel_since_init < c.min_travel:
            return mean, cov, bias, False, f"collecting profile {self.travel_since_init:.0f}/{c.min_travel:.0f} m"
        limit = min(max(c.max_fix_std, c.rel_fix_std * ins_std), c.cap_fix_std)
        if std_max > limit:
            return mean, cov, bias, False, f"cloud too wide ({std_max:.1f} m > {limit:.1f} m)"
        # unimodal: (almost) all weight close to the mean and the densest bin near the mean
        r = np.hypot(d[:, 0], d[:, 1])
        if float(w[r < 3.0 * std_max + 1e-6].sum()) < 0.95:
            return mean, cov, bias, False, "multimodal (weight far from mean)"
        b = max(std_max / 2.0, self.dsm.mpp)
        ij = np.floor(d / b).astype(np.int64)
        keys, inv = np.unique(ij, axis=0, return_inverse=True)
        mass = np.bincount(inv.ravel(), weights=w, minlength=len(keys))
        mode = (keys[int(np.argmax(mass))] + 0.5) * b
        if float(np.hypot(*mode)) > max(1.5 * b, std_max):
            return mean, cov, bias, False, "multimodal (mode away from mean)"
        return mean, cov, bias, True, f"converged ({std_max:.1f} m)"

    # ------------------------------------------------------------------ main entry

    def step(self, msg: AltimeterMsg, pose: PoseEstimate, var_growth: float) -> VisualFix | None:
        """One altimeter sample. `var_growth` = NavEkf.vel_var_growth (accumulated). Returns a fix or None."""
        t0 = time.perf_counter()
        try:
            return self._step(msg, pose, var_growth)
        finally:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if self.t_last_update == msg.t:  # a measurement update ran (not just a prediction)
                self.proc_ms = dt_ms
                self.ms_total += dt_ms
                self.n_timed += 1

    def _step(self, msg: AltimeterMsg, pose: PoseEstimate, var_growth: float) -> VisualFix | None:
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
        if not (np.isfinite(msg.range) and msg.range > c.min_range):
            return None
        if t - self.t_last_update < 1.0 / c.update_hz - 1e-6:
            return None
        self.t_last_update = t
        d = -pose.R[:, 2]  # beam direction (body -z) in world
        h_meas = float(pose.pos[2] + msg.range * d[2])
        hit_off = msg.range * d[:2]
        self._update(h_meas, hit_off, t)
        ins_std = 0.5 * pose.sigma2_horizontal()
        mean, cov, bias, ok, why = self._estimate(ins_std)
        self.est, self.est_cov, self.bias = mean, cov, bias
        if ok:
            self.bias_known = True
        h_map = float(self.dsm.sample(mean + hit_off))
        self.profile.append((t, h_meas - bias, h_map))
        self._track_info(mean)
        if self.status != "lost":
            self.status = why
        if not ok or t - self.t_last_fix < c.fix_interval:
            return None
        # new terrain under the beam since the last fix: hovering re-publishes the same information, which the
        # EKF would fuse as independent (overconfident; in V3's first version a stale hover fix out-voted IBN)
        if self.s_fix_ref is not None and self.travel_since_init - self.s_fix_ref < c.fix_travel:
            return None
        self.s_fix_ref = self.travel_since_init
        self.t_last_fix = t
        R = cov * c.fix_inflate**2 + np.eye(2) * c.min_fix_std**2
        std_max = float(np.sqrt(max(np.linalg.eigvalsh(cov).max(), 0.0)))
        strong = std_max < 0.5 * c.max_fix_std and self.travel_since_init > 2.0 * c.min_travel
        return VisualFix(t, mean.copy(), R, "trn", n_inliers=self.n_updates, strong=strong)

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
