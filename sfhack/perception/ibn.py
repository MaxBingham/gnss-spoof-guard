"""IBN - image-based navigation: down-camera frame matched against the onboard orthophoto (NCC).

Adapted from the mini demo's MapLocalizer (map_nav_demo.py), made onboard-only:
  1. Orthorectify: a north-up ground grid around the PRIOR position is projected into the camera with the
     estimated attitude (AHRS, heading-corrected), the camera extrinsics (calibration) and the camera height
     above the surface from the altimeter. Grid heights come from the onboard DSM, so roofs are rectified at
     roof height (true ortho) instead of being smeared by parallax.
  2. Match: normalized cross-correlation of the rectified template inside a search window of the onboard
     orthophoto (radius from the EKF covariance). Sub-pixel peak, uniqueness test (second peak).
  3. Refine: if the offset is large, re-rectify at the corrected position and match again in a small window.
Fix sigma grows with height above ground and shrinks with the correlation score.
No truth: the camera pose is the onboard estimate; the frame is the only simulator product used.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

from ..messages import CameraFrame, PoseEstimate, VisualFix
from ..world.surface import Grid

# Down camera extrinsics (body -> camera axes), see MujocoSim._build_model: camera x = body -y (right),
# camera y = body x (forward), camera looks along its -z = body down.
R_CAM_BODY = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
MIN_VALID = 0.85  # fraction of the template that must project into the image


@dataclass
class IbnConfig:
    match_mpp: float = 0.5  # m per pixel of the matching grid (onboard ortho is resampled to this)
    template_frac: float = 0.42  # template half size = this x height above surface (fits inside the image)
    min_agl: float = 5.0
    max_agl: float = 250.0
    search_k: float = 3.0  # search radius = k x 1-sigma of the prior (clipped below)
    min_search_m: float = 6.0
    max_search_m: float = 45.0
    ncc_min: float = 0.40
    peak_ratio_min: float = 1.12  # best NCC / best NCC outside the exclusion radius
    exclusion_m: float = 4.0
    sigma_ref: float = 0.8  # m at agl_ref with NCC 0.8
    agl_ref: float = 35.0
    hz: float = 2.0


class IbnLocalizer:
    name = "ibn"
    is_mock = False

    def __init__(self, cfg: IbnConfig, ortho: Grid, dsm: Grid) -> None:
        self.cfg = cfg
        self.map = ortho if abs(ortho.mpp - cfg.match_mpp) < 1e-6 else ortho.resampled(cfg.match_mpp)
        self.map_f = self.map.z.astype(np.float32)
        self.dsm = dsm
        self.t_last = -math.inf
        self.last_score = 0.0
        self.last_status = "idle"
        self.last_template: np.ndarray | None = None
        self.proc_ms = 0.0

    def covers(self, xy: np.ndarray, margin: float = 10.0) -> bool:
        return bool(self.map.contains(np.asarray(xy, float), margin))

    # ------------------------------------------------------------------ geometry

    def _rectify(self, gray: np.ndarray, frame: CameraFrame, R_cw: np.ndarray, cam_pos: np.ndarray,
                 center: np.ndarray, half: float) -> np.ndarray | None:
        mpp = self.cfg.match_mpp
        n = int(round(2 * half / mpp))
        off = (np.arange(n) - (n - 1) / 2.0) * mpp
        gx, gy = np.meshgrid(center[0] + off, center[1] - off)  # row 0 = north
        gz = self.dsm.sample(np.stack([gx, gy], -1))
        # never above the camera: towers next to the landing site would otherwise invalidate the frame
        gz = np.minimum(gz, cam_pos[2] - 3.0)
        pc = np.stack([gx - cam_pos[0], gy - cam_pos[1], gz - cam_pos[2]], -1) @ R_cw  # world -> camera
        depth = np.maximum(-pc[..., 2], 1e-3)
        fx, fy, cx, cy = frame.intrinsics()
        u = (cx + fx * pc[..., 0] / depth - 0.5).astype(np.float32)
        v = (cy - fy * pc[..., 1] / depth - 0.5).astype(np.float32)
        valid = (u >= 0) & (v >= 0) & (u <= frame.width - 1) & (v <= frame.height - 1)
        if valid.mean() < MIN_VALID:
            return None
        tpl = cv2.remap(gray, u, v, cv2.INTER_LINEAR)
        if not valid.all():  # fill the few pixels outside the image with the template mean (neutral for NCC)
            tpl[~valid] = float(tpl[valid].mean())
        return tpl

    def _match(self, tpl: np.ndarray, center: np.ndarray, radius_m: float) -> tuple[np.ndarray, float, float] | None:
        """NCC search of tpl around center. Returns (matched center xy, score, peak ratio)."""
        mpp = self.cfg.match_mpp
        th, tw = tpl.shape
        r = int(math.ceil(radius_m / mpp))
        row, col = self.map.xy_to_ij(center)
        top = int(round(row - (th - 1) / 2.0)) - r
        left = int(round(col - (tw - 1) / 2.0)) - r
        H, W = self.map_f.shape
        if top < 0 or left < 0 or top + th + 2 * r > H or left + tw + 2 * r > W:
            return None
        region = self.map_f[top:top + th + 2 * r, left:left + tw + 2 * r]
        if float(tpl.std()) < 1e-3 or float(region.std()) < 1e-3:
            return None
        ncc = cv2.matchTemplate(region, tpl, cv2.TM_CCOEFF_NORMED)
        _, best, _, (j, i) = cv2.minMaxLoc(ncc)
        # sub-pixel parabola fit
        di = dj = 0.0
        if 0 < i < ncc.shape[0] - 1:
            a, b, c = ncc[i - 1, j], ncc[i, j], ncc[i + 1, j]
            den = a - 2 * b + c
            di = 0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0
        if 0 < j < ncc.shape[1] - 1:
            a, b, c = ncc[i, j - 1], ncc[i, j], ncc[i, j + 1]
            den = a - 2 * b + c
            dj = 0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0
        ex = int(math.ceil(self.cfg.exclusion_m / mpp))
        masked = ncc.copy()
        masked[max(i - ex, 0):i + ex + 1, max(j - ex, 0):j + ex + 1] = -1.0
        second = float(masked.max()) if masked.size else -1.0
        ratio = best / max(second, 1e-3) if second > 0 else 10.0
        rc = top + i + di + (th - 1) / 2.0
        cc = left + j + dj + (tw - 1) / 2.0
        return self.map.ij_to_xy(rc, cc), float(best), float(ratio)

    # ------------------------------------------------------------------ main entry

    def update(self, frame: CameraFrame, prior: PoseEstimate, agl: float | None) -> VisualFix | None:
        c = self.cfg
        t0 = time.perf_counter()
        try:
            return self._update(frame, prior, agl)
        finally:
            self.proc_ms = (time.perf_counter() - t0) * 1000.0

    def _update(self, frame: CameraFrame, prior: PoseEstimate, agl: float | None) -> VisualFix | None:
        c = self.cfg
        if frame.t - self.t_last < 1.0 / c.hz - 1e-6:
            return None
        self.t_last = frame.t
        if not frame.image.any():
            self.last_status = "no image"
            return None
        prior_xy = prior.pos[:2]
        if agl is None or not (c.min_agl <= agl <= c.max_agl):
            self.last_status = "height above surface out of range"
            return None
        half = c.template_frac * agl
        sig = 0.5 * prior.sigma2_horizontal()
        radius = float(np.clip(c.search_k * sig, c.min_search_m, c.max_search_m))
        if not self.covers(prior_xy, half + radius + 2.0):
            self.last_status = "outside onboard imagery"
            return None
        gray = cv2.cvtColor(frame.image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        R_cw = prior.R @ R_CAM_BODY
        ground = float(self.dsm.sample(prior_xy))
        cam = np.array([prior_xy[0], prior_xy[1], ground + agl])
        center = prior_xy.copy()
        result = None
        for it in range(2):
            cam[:2] = center
            cam[2] = float(self.dsm.sample(center)) + agl
            tpl = self._rectify(gray, frame, R_cw, cam, center, half)
            if tpl is None:
                self.last_status = "template outside image (attitude)"
                return None
            m = self._match(tpl, center, radius)
            if m is None:
                self.last_status = "search window outside imagery / no texture"
                return None
            result = m
            shift = float(np.hypot(*(m[0] - center)))
            self.last_template = tpl
            if it == 0 and shift > 2.0:
                center = m[0]
                radius = c.min_search_m
                continue
            break
        xy, score, ratio = result
        self.last_score = score
        if score < c.ncc_min:
            self.last_status = f"weak match (NCC {score:.2f})"
            return None
        if ratio < c.peak_ratio_min:
            self.last_status = f"ambiguous match (peak ratio {ratio:.2f})"
            return None
        sigma = max(c.sigma_ref * agl / c.agl_ref, 0.3) * (0.8 / max(score, 0.3)) ** 2
        self.last_status = f"fix NCC {score:.2f} ratio {ratio:.2f}"
        return VisualFix(frame.t, xy, np.eye(2) * sigma * sigma, "ibn", n_inliers=int(score * 100),
                         frame_seq=frame.seq)
