"""IBN - image-based navigation: down-camera frame matched against the onboard orthophoto (NCC).

Merges V3's true-ortho rectification with V2's working vision localizer (perception/map_localizer.py in V2:
10/10 seeds within 3 m of B):
  1. Orthorectify: a north-up ground grid around the PRIOR position is projected into the camera with the
     estimated attitude (AHRS, heading-corrected), the camera extrinsics and the camera height above the surface
     from the altimeter. Grid heights come from the onboard DSM (roofs rectified at roof height). V2: the template
     has a fixed pixel count (`tpl_px`), its resolution follows the footprint, so it works from cruise down to
     ~4 m; anti-aliased before downsampling.
  2. Match: NCC (cv2.matchTemplate) of the template against the onboard ortho resampled to the same resolution
     inside a search window of radius k x sigma(prior). COARSE-TO-FINE: when that window would exceed
     `fine_search_px` template pixels, the search first runs on a coarser pyramid level over the whole window
     (search scales with the 2-sigma, never capped: V2's benchmark showed that a capped window produces
     confident fixes 40 m off), then refines at full resolution around the coarse peak. Beyond
     `max_search_m` there is no fix (the onboard estimate is too uncertain for one frame).
  3. Uniqueness (V2): the second-best LOCAL maximum outside an exclusion radius must be `min_margin` lower.
     Comparing against the masked maximum (the shoulders of a broad true peak) rejected every match above B.
Covariance: peak score and sharpness (sub-pixel curvature) + attitude error projected to the ground.
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
MIN_VALID = 0.97  # fraction of the template that must project into the image


@dataclass
class IbnConfig:
    tpl_px: int = 64  # template size in pixels (resolution follows the footprint)
    tpl_frac: float = 0.4  # template half size = this x height above surface (footprint half height ~0.7 x h)
    tpl_min_m: float = 2.0
    tpl_max_m: float = 20.0
    min_agl: float = 4.0
    max_agl: float = 250.0
    min_std: float = 0.015  # gray (0..1): a uniform template (roof, lawn) cannot match
    search_k: float = 3.0  # search radius = k x 1-sigma of the prior + margin ...
    search_margin_m: float = 4.0
    min_search_m: float = 6.0  # ... at least this ...
    max_search_m: float = 400.0  # ... beyond this: no fix (too uncertain for a single frame)
    fine_search_px: int = 64  # search radius in template pixels above which the coarse level runs first
    min_score: float = 0.5  # NCC peak below this -> no fix
    min_score_coarse: float = 0.45
    peak_excl_m: float = 3.0  # second local maximum is searched outside this radius around the best ...
    min_margin: float = 0.08  # ... and must be this much lower (else ambiguous -> no fix)
    curv_ref: float = 0.05  # NCC peak curvature [1/px^2] that counts as a sharp peak
    sigma_floor: float = 0.5  # m
    tilt_std_deg: float = 0.5  # AHRS attitude error projected to the ground: height x this
    hz: float = 2.0
    strong_score: float = 0.65  # 'strong' fix (may re-localize the INS after a long outage): NCC and ...
    strong_margin: float = 0.15  # ... margin to the second peak


class IbnLocalizer:
    name = "ibn"
    is_mock = False

    def __init__(self, cfg: IbnConfig, ortho: Grid, dsm: Grid) -> None:
        self.cfg = cfg
        self.map = ortho
        self.map_f = ortho.z.astype(np.float32)
        self._blur: dict[int, np.ndarray] = {}
        self.dsm = dsm
        self.t_last = -math.inf
        self.last_score = 0.0
        self.last_status = "idle"
        self.last_template: np.ndarray | None = None
        self.last_search_m = 0.0
        self.last_coarse = False
        self.proc_ms = 0.0

    def covers(self, xy: np.ndarray, margin: float = 10.0) -> bool:
        return bool(self.map.contains(np.asarray(xy, float), margin))

    def _map_blurred(self, mpp: float) -> np.ndarray:
        """Onboard ortho, low-passed for sampling at `mpp` (anti-aliasing, cached per quantized blur)."""
        key = int(round(4 * 0.5 * mpp / self.map.mpp)) if mpp > 1.5 * self.map.mpp else 0
        if key not in self._blur:
            self._blur[key] = cv2.GaussianBlur(self.map_f, (0, 0), key / 4) if key > 1 else self.map_f
        return self._blur[key]

    # ------------------------------------------------------------------ geometry

    def _rectify(self, gray: np.ndarray, frame: CameraFrame, R_cw: np.ndarray, cam_pos: np.ndarray,
                 center: np.ndarray, half: float, n: int) -> np.ndarray | None:
        mpp = 2.0 * half / n
        off = (np.arange(n) - (n - 1) / 2.0) * mpp
        gx, gy = np.meshgrid(center[0] + off, center[1] - off)  # row 0 = north
        gz = self.dsm.sample(np.stack([gx, gy], -1))
        # never above the camera: towers next to the landing site would otherwise invalidate the frame
        gz = np.minimum(gz, cam_pos[2] - 1.5)
        pc = np.stack([gx - cam_pos[0], gy - cam_pos[1], gz - cam_pos[2]], -1) @ R_cw  # world -> camera
        depth = np.maximum(-pc[..., 2], 1e-3)
        fx, fy, cx, cy = frame.intrinsics()
        u = (cx + fx * pc[..., 0] / depth - 0.5).astype(np.float32)
        v = (cy - fy * pc[..., 1] / depth - 0.5).astype(np.float32)
        valid = (u >= 0) & (v >= 0) & (u <= frame.width - 1) & (v <= frame.height - 1)
        if valid.mean() < MIN_VALID:
            return None
        gsd_cam = float(np.median(depth)) / fy  # m per camera pixel
        if mpp > 1.5 * gsd_cam:  # anti-alias before downsampling (V2)
            gray = cv2.GaussianBlur(gray, (0, 0), 0.5 * mpp / gsd_cam)
        tpl = cv2.remap(gray, u, v, cv2.INTER_LINEAR)
        if not valid.all():  # fill the few pixels outside the image with the template mean (neutral for NCC)
            tpl[~valid] = float(tpl[valid].mean())
        return tpl

    def _region(self, center: np.ndarray, half: float, mpp: float, S: int, n: int) -> np.ndarray | None:
        """Onboard ortho around center at resolution mpp: (n + 2S)^2 pixels, same grid as the template + margin."""
        N = n + 2 * S
        gr = (np.arange(N) - (N - 1) / 2.0) * mpp
        X, Y = np.meshgrid(center[0] + gr, center[1] - gr)
        row, col = self.map.xy_to_ij(np.stack([X, Y], -1))
        h, w = self.map_f.shape
        if row.min() < 0 or col.min() < 0 or row.max() > h - 1 or col.max() > w - 1:
            return None
        return cv2.remap(self._map_blurred(mpp), col.astype(np.float32), row.astype(np.float32), cv2.INTER_LINEAR)

    def _match(self, tpl: np.ndarray, center: np.ndarray, half: float, mpp: float, radius_m: float,
               min_score: float) -> tuple[np.ndarray, float, float, float] | str:
        """NCC search of tpl (resolution mpp) within radius_m of center. Returns (xy, score, margin, sigma_px)
        or the reason for no match."""
        c = self.cfg
        n = tpl.shape[0]
        S = int(math.ceil(radius_m / mpp))
        region = self._region(center, half, mpp, S, n)
        if region is None:
            return "search window outside imagery"
        if float(region.std()) < 1e-3:
            return "no texture in map"
        ncc = cv2.matchTemplate(region, tpl, cv2.TM_CCOEFF_NORMED)
        _, best, _, (j, i) = cv2.minMaxLoc(ncc)
        best = float(best)
        if best < min_score:
            return f"weak match (NCC {best:.2f})"
        # V2: second-best LOCAL maximum away from the best one (not the shoulders of a broad true peak)
        local_max = (ncc >= cv2.dilate(ncc, np.ones((5, 5), np.uint8))) & (ncc > -1.0)
        ex = max(3, int(round(c.peak_excl_m / mpp)))
        local_max[max(0, i - ex):i + ex + 1, max(0, j - ex):j + ex + 1] = False
        second = float(ncc[local_max].max()) if local_max.any() else -1.0
        if best - second < c.min_margin:
            return f"ambiguous match (NCC {best:.2f}, second peak {second:.2f})"

        def subpix(a: float, b: float, cc: float) -> tuple[float, float]:
            den = a - 2 * b + cc
            if den >= 0:  # not a maximum along this axis
                return 0.0, 1e-3
            return float(np.clip(0.5 * (a - cc) / den, -0.5, 0.5)), -den

        di, ci = subpix(ncc[i - 1, j], ncc[i, j], ncc[i + 1, j]) if 0 < i < ncc.shape[0] - 1 else (0.0, 1e-3)
        dj, cj = subpix(ncc[i, j - 1], ncc[i, j], ncc[i, j + 1]) if 0 < j < ncc.shape[1] - 1 else (0.0, 1e-3)
        xy = np.array([center[0] + (j + dj - S) * mpp, center[1] - (i + di - S) * mpp])
        sharp = math.sqrt(c.curv_ref / max(min(ci, cj), 1e-3))
        sig_px = (1.0 + 4.0 * (1.0 - best)) * float(np.clip(sharp, 0.5, 4.0))
        return xy, best, best - second, sig_px

    # ------------------------------------------------------------------ main entry

    def update(self, frame: CameraFrame, prior: PoseEstimate, agl: float | None) -> VisualFix | None:
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
        if int(frame.image.max()) < 8:  # black frame = camera failure / blackout
            self.last_status = "no image"
            return None
        prior_xy = prior.pos[:2].copy()
        if agl is None or not (c.min_agl <= agl <= c.max_agl):
            self.last_status = "height above surface out of range"
            return None
        n = c.tpl_px
        half = float(np.clip(c.tpl_frac * agl, c.tpl_min_m, c.tpl_max_m))
        mpp = 2.0 * half / n
        sig = 0.5 * prior.sigma2_horizontal()
        radius = max(c.search_k * sig + c.search_margin_m, c.min_search_m)
        self.last_search_m = radius
        if radius > c.max_search_m:
            self.last_status = f"prior too wide ({radius:.0f} m search)"
            return None
        if not self.covers(prior_xy, 0.0):
            self.last_status = "outside onboard imagery"
            return None
        gray = cv2.cvtColor(frame.image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        R_cw = prior.R @ R_CAM_BODY
        center = prior_xy
        self.last_coarse = False

        def rectify_at(xy: np.ndarray) -> np.ndarray | None:
            cam = np.array([xy[0], xy[1], float(self.dsm.sample(xy)) + agl])
            return self._rectify(gray, frame, R_cw, cam, xy, half, n)

        tpl = rectify_at(center)
        if tpl is None:
            self.last_status = "template outside image (attitude)"
            return None
        if float(tpl.std()) < c.min_std:
            self.last_status = "no texture under the camera"
            return None
        # coarse level: the window would be wider than fine_search_px template pixels
        if radius / mpp > c.fine_search_px:
            f = int(2 ** math.ceil(math.log2(radius / mpp / c.fine_search_px)))
            nc = max(n // f, 12)
            tpl_c = cv2.resize(tpl, (nc, nc), interpolation=cv2.INTER_AREA)
            m = self._match(tpl_c, center, half, 2.0 * half / nc, radius, c.min_score_coarse)
            if isinstance(m, str):
                self.last_status = f"coarse x{f}: {m}"
                return None
            self.last_coarse = True
            center = m[0]
            radius = max(3.0 * f * mpp * m[3], c.min_search_m)
            tpl = rectify_at(center)  # re-rectify with the DSM heights at the corrected position
            if tpl is None:
                self.last_status = "template outside image (attitude)"
                return None
        m = self._match(tpl, center, half, mpp, radius, c.min_score)
        if isinstance(m, str):
            self.last_status = m
            return None
        xy, score, margin, sig_px = m
        if float(np.hypot(*(xy - center))) > 2.0:  # moved: re-rectify at the match and confirm once more
            tpl2 = rectify_at(xy)
            if tpl2 is not None:
                m2 = self._match(tpl2, xy, half, mpp, c.min_search_m, c.min_score)
                if not isinstance(m2, str):
                    xy, score, margin, sig_px = m2
                    tpl = tpl2
        self.last_template = tpl
        self.last_score = score
        sigma = math.hypot(c.sigma_floor, sig_px * mpp, agl * math.radians(c.tilt_std_deg))
        self.last_status = (f"fix NCC {score:.2f} margin {margin:.2f}" + (" (coarse-to-fine)" if self.last_coarse else ""))
        return VisualFix(frame.t, xy, np.eye(2) * sigma * sigma, "ibn", n_inliers=int(score * 100),
                         frame_seq=frame.seq, strong=score >= c.strong_score and margin >= c.strong_margin)
