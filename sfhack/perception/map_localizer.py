"""VISION localizer: match the ortho-rectified down-camera image against the onboard orthophoto (NCC).

Onboard only: camera frame + onboard map + the prior (EKF position/covariance, AHRS attitude). No truth.

1. Ortho-rectify: for every cell of a north-up grid centred under the camera, project the surface point
   (the onboard DSM height there: roofs and terrain; z = 0 for packages without terrain) into the image with
   the AHRS attitude and the EKF height, and sample the camera there (cv2.remap). Template size follows the
   footprint (~0.4 x height above the surface below), its resolution follows the template size (fixed pixel
   count), so it works from cruise down to ~5 m.
2. Resample the onboard ortho around the prior at the same resolution, search radius = 3 sigma of the
   prior (+ margin), and run normalized cross-correlation (cv2.matchTemplate, TM_CCOEFF_NORMED).
3. Peak -> camera position (sub-pixel parabola). Reject below `vision_min_score` or when a second peak
   elsewhere is almost as good (repetitive texture). Covariance from peak score and sharpness, plus the
   attitude error projected to the ground (height x tilt std).
"""

from __future__ import annotations

import math
import time

import cv2
import numpy as np

from ..config import LocalizerConfig
from ..messages import CameraFrame, PoseEstimate, VisualFix
from ..world.onboard_map import OnboardMap

# Down-camera mount in the body frame (columns = camera x, y, z axes; see MujocoSim._build_model):
# image right = body -y, image up = body forward, camera looks along -z = body down.
R_BODY_CAM = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
CAM_Z_OFFSET = -0.02  # m, camera below the body origin


class MapLocalizer:
    name = "vision"
    is_mock = False

    def __init__(self, cfg: LocalizerConfig) -> None:
        self.cfg = cfg
        self._map_id: int | None = None
        self._map_gray: np.ndarray | None = None
        self._map_blur: dict[int, np.ndarray] = {}
        # Last result, for the dashboard / demo panels
        self.last_score = 0.0
        self.last_template: np.ndarray | None = None
        self.last_reason = ""

    def _gray_map(self, om: OnboardMap, blur_px: float) -> np.ndarray:
        if self._map_id != id(om):
            self._map_id = id(om)
            self._map_gray = cv2.cvtColor(om.ortho, cv2.COLOR_RGB2GRAY).astype(np.float32)
            self._map_blur = {}
        key = int(round(blur_px * 4))  # quantized so the cache stays small
        if key not in self._map_blur:
            self._map_blur[key] = (cv2.GaussianBlur(self._map_gray, (0, 0), key / 4) if key > 1
                                   else self._map_gray)
        return self._map_blur[key]

    def _fail(self, why: str, score: float = 0.0) -> None:
        self.last_reason = why
        self.last_score = score
        return None

    def update(self, frame: CameraFrame, onboard_map: OnboardMap, prior: PoseEstimate) -> VisualFix | None:
        t0 = time.perf_counter()
        c = self.cfg
        if frame.name != "down" or int(frame.image.max()) < 8:  # black frame = camera failure
            return self._fail("no image")
        # Camera height above the mapped surface below it: onboard DSM (roofs, terrain) under the prior,
        # 0 for packages without terrain (flat world at z = 0).
        cam_z = float(prior.pos[2]) - CAM_Z_OFFSET
        alt = cam_z - onboard_map.surface_z(prior.pos[:2])
        if alt < c.vision_min_alt:
            return self._fail(f"below {c.vision_min_alt:.0f} m")

        # ---- 1. ortho-rectified template, north-up, centred under the camera
        n = c.vision_tpl_px
        half = float(np.clip(c.vision_tpl_frac * alt, c.vision_tpl_min_m, c.vision_tpl_max_m))
        mpp = 2.0 * half / n
        g = (np.arange(n) + 0.5) * mpp - half
        dx, dy = np.meshgrid(g, -g)  # row 0 = north
        # Each cell is projected at its mapped surface height (true ortho over buildings and slopes; over
        # terrain 15-30 m below the camera, with 20 m roofs, a single ground plane put roofs at 2-3x scale).
        surf = onboard_map.surface_heights(np.stack([prior.pos[0] + dx, prior.pos[1] + dy], axis=-1))
        rel = np.stack([dx, dy, surf - cam_z], axis=-1)  # surface point - camera, world
        R_wc = prior.R @ R_BODY_CAM
        pc = rel @ R_wc  # world -> camera frame
        depth = -pc[..., 2]
        fx, fy, cx, cy = frame.intrinsics()
        u = (cx + fx * pc[..., 0] / depth - 0.5).astype(np.float32)
        v = (cy - fy * pc[..., 1] / depth - 0.5).astype(np.float32)
        inside = (depth > 0) & (u >= 0) & (u <= frame.width - 1) & (v >= 0) & (v <= frame.height - 1)
        if inside.mean() < 0.97:
            return self._fail("footprint (tilt)")
        gray = cv2.cvtColor(frame.image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gsd_cam = alt / fy  # m per camera pixel at the image centre
        if mpp > 1.5 * gsd_cam:  # anti-alias before downsampling
            gray = cv2.GaussianBlur(gray, (0, 0), 0.5 * mpp / gsd_cam)
        tpl = cv2.remap(gray, u, v, cv2.INTER_LINEAR)
        if float(tpl.std()) < c.vision_min_std:
            return self._fail("no texture")

        # ---- 2. onboard ortho around the prior, same resolution
        sig_prior = math.sqrt(max(float(np.linalg.eigvalsh(prior.cov[:2, :2]).max()), 0.0))
        r_m = float(np.clip(3.0 * sig_prior + c.vision_search_margin_m, c.vision_search_min_m, c.vision_search_max_m))
        S = int(math.ceil(r_m / mpp))
        if S > c.vision_search_max_px:  # low and uncertain: the true position may lie outside any window
            return self._fail("prior too wide")  # we can afford; a capped window gives confident wrong fixes
        N = n + 2 * S
        px, py = float(prior.pos[0]), float(prior.pos[1])
        gr = (np.arange(N) + 0.5) * mpp
        X, Y = np.meshgrid(px - half - S * mpp + gr, py + half + S * mpp - gr)
        om = onboard_map
        mu = ((X - om.min_x) / om.mpp_x - 0.5).astype(np.float32)
        mv = ((om.max_y - Y) / om.mpp_y - 0.5).astype(np.float32)
        h, w = om.gray.shape
        if mu.min() < 0 or mv.min() < 0 or mu.max() > w - 1 or mv.max() > h - 1:
            return self._fail("off map")
        src = self._gray_map(om, 0.5 * mpp / om.m_per_px if mpp > 1.5 * om.m_per_px else 0.0)
        region = cv2.remap(src, mu, mv, cv2.INTER_LINEAR)

        # ---- 3. NCC search
        ncc = cv2.matchTemplate(region, tpl, cv2.TM_CCOEFF_NORMED)
        _, best, _, (j, i) = cv2.minMaxLoc(ncc)
        best = float(best)
        if best < c.vision_min_score:
            return self._fail("low score", best)
        # Second-best LOCAL maximum away from the best one (the shoulders of a broad true peak are not
        # local maxima; comparing against them rejected most good cruise matches as 'ambiguous').
        local_max = (ncc >= cv2.dilate(ncc, np.ones((5, 5), np.uint8))) & (ncc > -1.0)
        excl = max(3, int(round(c.vision_peak_excl_m / mpp)))
        local_max[max(0, i - excl):i + excl + 1, max(0, j - excl):j + excl + 1] = False
        second = float(ncc[local_max].max()) if local_max.any() else -1.0
        if best - second < c.vision_min_margin:
            return self._fail("ambiguous", best)

        def subpix(a: float, b: float, cc: float) -> tuple[float, float]:
            den = a - 2 * b + cc
            if den >= 0:  # not a maximum along this axis
                return 0.0, 1e-3
            return float(np.clip(0.5 * (a - cc) / den, -0.5, 0.5)), -den

        di, curv_i = (subpix(ncc[i - 1, j], ncc[i, j], ncc[i + 1, j]) if 0 < i < ncc.shape[0] - 1 else (0.0, 1e-3))
        dj, curv_j = (subpix(ncc[i, j - 1], ncc[i, j], ncc[i, j + 1]) if 0 < j < ncc.shape[1] - 1 else (0.0, 1e-3))
        cam_x = px - S * mpp + (j + dj) * mpp
        cam_y = py + S * mpp - (i + di) * mpp

        # ---- covariance: blunt peak / low score -> larger; plus attitude error projected to the ground
        sharp = math.sqrt(c.vision_curv_ref / max(min(curv_i, curv_j), 1e-3))
        sig_px = (1.0 + 4.0 * (1.0 - best)) * float(np.clip(sharp, 0.5, 4.0))
        sig = math.hypot(c.vision_sigma_floor, sig_px * mpp, alt * math.radians(c.vision_tilt_std_deg))
        self.last_score, self.last_template, self.last_reason = best, tpl, ""
        corners = [(float(u[a, b]) + 0.5, float(v[a, b]) + 0.5) for a, b in ((0, 0), (0, n - 1), (n - 1, n - 1), (n - 1, 0))]
        return VisualFix(frame.t, np.array([cam_x, cam_y]), np.eye(2) * sig * sig, "vision",
                         matched_px=corners, n_inliers=int(round(100 * best)),
                         proc_ms=(time.perf_counter() - t0) * 1000.0, frame_seq=frame.seq)
