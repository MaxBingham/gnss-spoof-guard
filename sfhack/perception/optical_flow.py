"""Optical flow from the downward thermal camera: ground velocity without any map (recon-nav).

Like an optical mouse: each thermal frame is ortho-rectified onto a north-up ground grid centred on the point
straight below the camera (AHRS attitude, height above the surface from the EKF height minus the onboard DSM),
and consecutive grids are matched (NCC block matching). Both grids are camera-centred, so the content shift between them is
minus the camera's ground displacement in world ENU; divided by the frame interval it is the mean ground
velocity over that interval. Yaw comes from the AHRS (compass): a heading error rotates the measured velocity
exactly as it rotates the IMU's, so flow fixes the speed and the along-track drift, not the heading.

Night: the thermal scene is weak in places (uniform fields, forest canopy at air temperature). Before
rectification the camera's fixed-pattern noise is removed with a running mean in sensor space (the ground
moves through the image, the pattern does not; left in, it pulls the correlation peak to zero shift). A frame
pair gives no measurement when the rectified grid is too uniform (std below `min_std_K`) or the NCC
peak is weak (`min_response`).

Onboard: uses only the published ThermalFrame, the AHRS / EKF state and the onboard map.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass
class FlowConfig:
    grid_px: int = 128  # rectified grid size (cells per side)
    grid_half_frac: float = 0.24  # grid half size = this x height (inscribed in the 50 x 40 deg footprint at any yaw)
    min_agl: float = 8.0  # m above the surface: below this the footprint is small (use the rangefinder there)
    max_agl: float = 300.0
    min_std_K: float = 0.08  # rectified grid std below this = uniform, no measurement
    min_response: float = 0.5  # NCC peak below this = no reliable shift
    search_px: int = 24  # max shift per frame pair (cells): 7 m at 80 m AGL, 36 m/s at 5 Hz
    dog_lo: float = 1.0  # band-pass (grid cells) before matching
    dog_hi: float = 8.0
    max_dt: float = 0.5  # s, frame pairs further apart are not used (dropped frames)
    fpn_tau_s: float = 20.0  # running-mean time constant of the fixed-pattern estimate
    fpn_warmup_frames: int = 10  # frames before the fixed-pattern subtraction starts
    sigma_shift_px: float = 0.25  # 1-sigma shift error at a strong peak (grid cells) ...
    response_ref: float = 0.9  # ... grows as response_ref / response below this
    height_rel_std: float = 0.02  # relative height error (baro + DSM): scales the velocity
    sigma_floor: float = 0.08  # m/s


@dataclass
class FlowMeasurement:
    t: float  # time of the later frame
    vel: np.ndarray  # (2,) mean ground velocity over [t - dt, t], world ENU [m/s]
    cov: np.ndarray  # (2, 2)
    dt: float
    response: float  # NCC peak
    std_K: float  # rectified grid contrast
    agl: float
    accepted: bool = False  # set by the EKF (NIS gate)


@dataclass
class FlowStats:
    frames: int = 0
    measurements: int = 0
    no_height: int = 0
    uniform: int = 0
    weak_peak: int = 0
    big_shift: int = 0
    log: list[tuple[float, float, float, float]] = field(default_factory=list)  # t, response, std_K, |v|


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# MuJoCo "thermal"/"down" camera mount: camera x = body -y (image right = body right), camera y = body +x
# (image top = forward), camera looks along its -z = body down.
R_BODY_CAM = np.column_stack([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


class OpticalFlow:
    def __init__(self, cfg: FlowConfig | None = None) -> None:
        self.cfg = cfg or FlowConfig()
        self.prev: tuple[float, np.ndarray, np.ndarray, float] | None = None  # (t, image, attitude, agl)
        self.fpn: np.ndarray | None = None
        self.n_fpn = 0
        self._fpn_t: float | None = None
        self.stats = FlowStats()
        self.last: FlowMeasurement | None = None
        self.last_grid: np.ndarray | None = None  # for display
        n = self.cfg.grid_px
        c = (np.arange(n) - (n - 1) / 2.0)
        self._gx, self._gy = np.meshgrid(c, -c)  # column -> east, row 0 = north

    def reset(self) -> None:
        self.prev = None

    def _remove_fpn(self, img: np.ndarray, t: float) -> np.ndarray:
        img = img.astype(np.float32)
        if self.fpn is None:
            self.fpn = np.zeros_like(img)
        dt = 0.2 if self._fpn_t is None else max(t - self._fpn_t, 0.0)
        self._fpn_t = t
        frame = img - float(np.median(img))
        a = min(1.0, max(dt / self.cfg.fpn_tau_s, 1.0 / (self.n_fpn + 1)))
        self.fpn += a * (frame - self.fpn)
        self.n_fpn += 1
        if self.n_fpn <= self.cfg.fpn_warmup_frames:
            return img
        return img - self.fpn

    def rectify(self, img: np.ndarray, R_world_body: np.ndarray, agl: float, intr: dict[str, float]) -> tuple[np.ndarray, float]:
        """Ground grid (north-up, centred below the camera) sampled from the image. Returns (grid, cell m)."""
        cfg = self.cfg
        s = 2.0 * cfg.grid_half_frac * agl / cfg.grid_px
        v_world = np.stack([self._gx * s, self._gy * s, np.full_like(self._gx, -agl)], -1)
        R_wc = R_world_body @ R_BODY_CAM
        c = v_world @ R_wc  # rows: R_wc^T v (camera frame)
        zc = np.minimum(c[..., 2], -1e-6)
        u = (intr["cx"] + intr["fx"] * c[..., 0] / -zc - 0.5).astype(np.float32)
        v = (intr["cy"] - intr["fy"] * c[..., 1] / -zc - 0.5).astype(np.float32)
        px_m = agl / intr["fx"]  # ground size of one pixel at nadir
        sig = 0.5 * s / px_m  # anti-alias: the grid is coarser than the image (and kills the renderer's upsampling pattern)
        src = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sig) if sig > 0.5 else img.astype(np.float32)
        grid = cv2.remap(src, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return grid, s

    def _bandpass(self, g: np.ndarray) -> np.ndarray:
        """Difference of Gaussians: drops the cell-scale noise and the large-scale gradients (vignetting, slopes)."""
        g = g.astype(np.float64)
        return cv2.GaussianBlur(g, (0, 0), self.cfg.dog_lo) - cv2.GaussianBlur(g, (0, 0), self.cfg.dog_hi)

    def _match(self, a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
        """Shift (cols, rows) of the content from grid a to grid b: NCC of b's centre in a, sub-cell parabola fit.
        (cv2.phaseCorrelate locked onto the static parts, renderer upsampling and window edges: shifts 2-5x short.)"""
        m = self.cfg.search_px
        tpl = b[m:-m, m:-m].astype(np.float32)
        res = cv2.matchTemplate(a.astype(np.float32), tpl, cv2.TM_CCOEFF_NORMED)
        _, best, _, (px, py) = cv2.minMaxLoc(res)

        def sub(c: int, n: int, get) -> float:
            if 0 < c < n - 1:
                l, mid, r = get(c - 1), get(c), get(c + 1)
                den = l - 2 * mid + r
                return float(c + 0.5 * (l - r) / den) if den < 0 else float(c)
            return float(c)

        fx = sub(px, res.shape[1], lambda c: res[py, c])
        fy = sub(py, res.shape[0], lambda r: res[r, px])
        # b's centre sits at (fx - m, fy - m) in a: the content moved by minus that
        return -(fx - m), -(fy - m), float(best)

    def update(self, t: float, image_K: np.ndarray, R_world_body: np.ndarray, agl: float | None,
               intr: dict[str, Any]) -> FlowMeasurement | None:
        cfg, st = self.cfg, self.stats
        st.frames += 1
        img = self._remove_fpn(image_K, t)
        if agl is None or not (cfg.min_agl <= agl <= cfg.max_agl):
            st.no_height += 1
            self.prev = None
            return None
        prev, self.prev = self.prev, (t, img, R_world_body.copy(), float(agl))
        if prev is None:
            return None
        t0, img0, R0, agl0 = prev
        dt = t - t0
        if dt <= 0 or dt > cfg.max_dt:
            return None
        h = 0.5 * (agl + agl0)  # both frames on the same cell size (the ground moved under both by the same metres)
        g0, s = self.rectify(img0, R0, h, intr)
        grid, _ = self.rectify(img, R_world_body, h, intr)
        self.last_grid = grid
        std = float(min(grid.std(), g0.std()))
        if std < cfg.min_std_K:
            st.uniform += 1
            return None
        a, b = self._bandpass(g0), self._bandpass(grid)
        dx, dy, resp = self._match(a, b)
        st.log.append((t, float(resp), std, 0.0))
        if resp < cfg.min_response:
            st.weak_peak += 1
            return None
        if max(abs(dx), abs(dy)) > cfg.search_px - 1:
            st.big_shift += 1
            return None
        # content moved by (dx cols east, dy rows south) = the camera moved by (-dx east, +dy north)
        vel = np.array([-dx * s, dy * s]) / dt
        sig_px = cfg.sigma_shift_px * max(1.0, cfg.response_ref / max(resp, 1e-3))
        var = (sig_px * s / dt) ** 2
        var += (cfg.height_rel_std * float(np.linalg.norm(vel))) ** 2 + cfg.sigma_floor ** 2
        cov = np.eye(2) * var
        st.measurements += 1
        st.log[-1] = (t, float(resp), std, float(np.linalg.norm(vel)))
        self.last = FlowMeasurement(t, vel, cov, dt, float(resp), std, float(agl))
        return self.last
