"""THERMAL localizer: match the downward night thermal image against the onboard thermal PRIOR (the temperature map
predicted from daytime imagery + OSM, onboard/thermal_prior.npz).

Onboard only: thermal frame + onboard map package + the prior pose (EKF position / covariance, AHRS attitude,
lidar height above the surface). No truth, no simulator imports.

Common step, ortho-rectification: every cell of a north-up grid (prior resolution, 0.5 m) centred under the
estimated camera position is projected into the image at its onboard-DSM height (roofs, terrain) with the AHRS
attitude; the camera height above the surface below comes from the nadir lidar (`agl_m`) when there is one,
else from the EKF height. The thermal image is low-pass filtered first (8.7 cm/px at 60 m vs 0.5 m cells).

Methods (each returns at most one ThermalFix; `method` selects one or a combination):
  ncc        whole-image NCC of the zero-mean rectified temperatures vs the prior temperatures (cv2.matchTemplate)
             within 3 sigma of the prior, small yaw search. Uncooled LWIR has a +-5 K absolute error and drifts,
             so only the normalized pattern is compared.
  grad       the same search on gradient-orientation channels (CFOG-like: |grad| split into 4 orientation bins
             mod 180 deg, so a roof that is warmer OR colder than the prior predicts still matches by its edges).
  road       road centerline matching: warm pixels segmented in the rectified image, thinned to centerlines,
             chamfer-matched (truncated distance transform) against the OSM road centerlines of the prior. The
             fix covariance is the spread of the chamfer likelihood over the search window, so one straight road
             gives a 1-D fix (cross-track only, along-road variance large) and a junction or curve a 2-D fix.
  junction   junctions of the observed centerlines (>= 3 branches) matched to the OSM road graph's junctions
             (nearest neighbour inside the gate, unique): an along-track fix where roads cross.
  combos     "ncc+road", "grad+road", "ncc+junction", "all": the candidates are computed and the one with the
             smallest covariance among those that agree with each other within the gate is returned; when
             two independent methods agree, their information-weighted mean.
No fix (None, `last_reason` says why) when: the rectified image is uniform, the score is low, a second peak is
almost as good (repetitive pattern), the prior is too wide for the search window, or the footprint leaves the map.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import ThermalConfig as ThermalLocalizerConfig
from ..messages import PoseEstimate
from ..world.onboard_map import OnboardMap
from .map_localizer import CAM_Z_OFFSET, R_BODY_CAM

METHODS = ("ncc", "grad", "road", "junction", "ncc+road", "grad+road", "ncc+junction", "ncc+grad", "all")
PAVED_ROAD_KINDS = ("vehicle", "parking")


@dataclass
class ThermalFix:
    xy: np.ndarray  # (2,) camera position, world ENU [m]
    cov: np.ndarray  # (2, 2) m^2
    yaw_corr: float  # rad, heading correction (true - estimated), 0 if not estimated
    quality: float  # 0..1 (NCC peak / chamfer agreement)
    method: str
    t: float = 0.0
    proc_ms: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)


def load_prior_npz(path: Path) -> dict[str, Any]:
    with np.load(path) as z:
        out = {k: z[k] for k in z.files}
    out["meta"] = json.loads(str(out["meta"]))
    out["roads_json"] = json.loads(str(out["roads_json"])) if "roads_json" in out else []
    return out


def _orient_channels(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """(H, W, 4) gradient magnitude in 4 orientation bins mod 180 deg (soft assignment), smoothed: CFOG-like."""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.mod(np.arctan2(gy, gx), np.pi)
    chans = []
    for k in range(4):
        c = k * np.pi / 4
        d = np.abs(np.angle(np.exp(2j * (ang - c)))) / 2  # angular distance mod 180, 0..pi/2
        wgt = np.clip(1.0 - d / (np.pi / 4), 0.0, 1.0)
        chans.append(cv2.GaussianBlur((mag * wgt).astype(np.float32), (0, 0), sigma))
    return np.stack(chans, -1)


def _skeleton(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning (vectorized per sub-iteration): 1-px, 8-connected centerlines."""
    img = np.pad((mask > 0).astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            p2, p3, p4 = img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:]
            p5, p6, p7 = img[2:, 2:], img[2:, 1:-1], img[2:, :-2]
            p8, p9 = img[1:-1, :-2], img[:-2, :-2]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9]
            B = sum(n.astype(np.int32) for n in nb)
            A = sum(((nb[k] == 0) & (nb[(k + 1) % 8] == 1)).astype(np.int32) for k in range(8))
            c = img[1:-1, 1:-1] == 1
            if step == 0:
                m = c & (B >= 2) & (B <= 6) & (A == 1) & ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
            else:
                m = c & (B >= 2) & (B <= 6) & (A == 1) & ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
            if m.any():
                img[1:-1, 1:-1][m] = 0
                changed = True
        if not changed:
            return img[1:-1, 1:-1] > 0


def _junctions_px(skel: np.ndarray, min_sep_px: float) -> list[tuple[float, float]]:
    """Skeleton pixels with >= 3 branches (crossing count of the 8-neighbourhood ring), clustered."""
    s = skel.astype(np.uint8)
    p = np.pad(s, 1)
    ring = [p[0:-2, 1:-1], p[0:-2, 2:], p[1:-1, 2:], p[2:, 2:], p[2:, 1:-1], p[2:, 0:-2], p[1:-1, 0:-2], p[0:-2, 0:-2]]
    trans = sum(((ring[k] == 0) & (ring[(k + 1) % 8] == 1)).astype(np.int32) for k in range(8))
    cand = np.argwhere((s == 1) & (trans >= 3))
    out: list[list[float]] = []
    for r, c in cand:
        for o in out:
            if math.hypot(o[0] / o[2] - r, o[1] / o[2] - c) < min_sep_px:
                o[0] += r
                o[1] += c
                o[2] += 1
                break
        else:
            out.append([float(r), float(c), 1.0])
    return [(o[0] / o[2], o[1] / o[2]) for o in out]


class ThermalLocalizer:
    name = "thermal"

    def __init__(self, onboard_dir: Path | str, cfg: ThermalLocalizerConfig | None = None,
                 onboard_map: OnboardMap | None = None, prior_path: Path | str | None = None) -> None:
        self.cfg = cfg or ThermalLocalizerConfig()
        d = Path(onboard_dir)
        self.map = onboard_map or OnboardMap(d)
        p = Path(prior_path) if prior_path is not None else d / "thermal_prior.npz"
        pr = load_prior_npz(p)
        self.prior_meta = pr["meta"]
        e = pr["meta"]["extent_enu"]
        self.min_x, self.max_y, self.res = float(e["min_x"]), float(e["max_y"]), float(pr["meta"]["res_m"])
        self.temp = np.ascontiguousarray(pr["temp_K"], np.float32)
        self.H, self.W = self.temp.shape
        self.roads_json = pr["roads_json"]
        self.roads_mask = (pr["roads"] > 0) if "roads" in pr else None
        self._blur_cache: dict[int, np.ndarray] = {}
        self._orient_cache: dict[int, np.ndarray] = {}
        self._build_road_layers()
        # last result, for panels / eval
        self.last_reason = ""
        self.last_score = 0.0
        self.last_tpl: np.ndarray | None = None
        self.last_skel_xy: np.ndarray | None = None  # observed centerline points (world, at the fix)
        self.last_junctions_xy: list[np.ndarray] = []
        self.last_candidates: dict[str, ThermalFix | None] = {}
        self.n_frames = 0
        self.n_fixes: dict[str, int] = {}

    # ------------------------------------------------------------------ prior layers

    def _build_road_layers(self) -> None:
        """Road centerline raster (1 px lines) and junctions. road_ref "mask": thinned road mask of the prior (the
        paved surface as predicted, so wide roads, plazas' edges and parking aisles sit where they are warm);
        "osm": the OSM way polylines (their nodes give the junctions; footway lines often run beside the warm
        band, and many warm roads have no way in the bbox)."""
        if self.cfg.road_ref == "mask" and self.roads_mask is not None:
            m = cv2.morphologyEx(self.roads_mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            sk = _skeleton(m)
            self.centerline = sk.astype(np.uint8)
            js = _junctions_px(sk, 4.0 / self.res)
            self.junctions = np.array([[self.min_x + (c + 0.5) * self.res, self.max_y - (r + 0.5) * self.res]
                                       for r, c in js], float).reshape(-1, 2)
            return
        cl = np.zeros((self.H, self.W), np.uint8)
        deg: dict[tuple[int, int], int] = {}
        for w in self.roads_json:
            if not w.get("paved", True):
                continue
            if w.get("kind") not in PAVED_ROAD_KINDS and not self.cfg.road_include_footways:
                continue
            if w.get("highway") == "steps":
                continue
            pts = np.asarray(w["pts_enu"], float)
            if len(pts) < 2:
                continue
            uv = self._to_px(pts)
            cv2.polylines(cl, [np.round(uv * 4).astype(np.int32)], False, 1, 1, cv2.LINE_8, shift=2)
            keys = [(int(round(x * 2)), int(round(y * 2))) for x, y in pts]
            for i, k in enumerate(keys):  # node degree: interior nodes count 2, ends 1
                deg[k] = deg.get(k, 0) + (1 if i in (0, len(keys) - 1) else 2)
        self.centerline = cl
        self.junctions = np.array([[k[0] / 2, k[1] / 2] for k, n in deg.items() if n >= 3], float).reshape(-1, 2)

    def _to_px(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, float)
        return np.stack([(xy[..., 0] - self.min_x) / self.res - 0.5, (self.max_y - xy[..., 1]) / self.res - 0.5], -1)

    def _region(self, src: np.ndarray, cx: float, cy: float, half: float, S: int, mpp: float,
                interp: int = cv2.INTER_LINEAR) -> np.ndarray | None:
        n = int(round(2 * half / mpp)) + 2 * S
        g = (np.arange(n) + 0.5) * mpp
        X, Y = np.meshgrid(cx - half - S * mpp + g, cy + half + S * mpp - g)
        uv = self._to_px(np.stack([X, Y], -1)).astype(np.float32)
        if uv[..., 0].min() < 0 or uv[..., 1].min() < 0 or uv[..., 0].max() > self.W - 1 or uv[..., 1].max() > self.H - 1:
            return None
        return cv2.remap(src, uv[..., 0], uv[..., 1], interp)

    # ------------------------------------------------------------------ main entry

    def _fail(self, why: str, score: float = 0.0) -> None:
        self.last_reason, self.last_score = why, score
        return None

    def update(self, img_K: np.ndarray, est_pose: PoseEstimate, agl_m: float | None,
               intrinsics: dict[str, float], t: float = 0.0) -> ThermalFix | None:
        t0 = time.perf_counter()
        c = self.cfg
        self.n_frames += 1
        self.last_candidates = {}
        rect = self.rectify(img_K, est_pose, agl_m, intrinsics)
        if rect is None:
            return None
        tpl, half, alt, off = rect
        sig_prior = math.sqrt(max(float(np.linalg.eigvalsh(est_pose.cov[:2, :2]).max()), 0.0))
        r_m = float(np.clip(3.0 * sig_prior + c.search_margin_m, c.search_min_m, c.search_max_m))
        if 3.0 * sig_prior + c.search_margin_m > c.search_max_m * 1.5:
            return self._fail("prior too wide")
        S = int(math.ceil(r_m / c.tpl_mpp))
        px, py = float(est_pose.pos[0] + off[0]), float(est_pose.pos[1] + off[1])  # template centre (optical axis)
        methods = {"all": ["ncc", "grad", "road", "junction"]}.get(c.method, c.method.split("+"))
        cands: list[ThermalFix] = []
        for m in methods:
            f = getattr(self, "_m_" + m)(tpl, px, py, half, S, alt)
            self.last_candidates[m] = f
            if f is not None:
                cands.append(f)
        if not cands:
            return None
        fix = self._combine(cands)
        fix.xy = fix.xy - off  # template centre -> camera position
        fix.t = t
        fix.proc_ms = (time.perf_counter() - t0) * 1000.0
        self.n_fixes[fix.method] = self.n_fixes.get(fix.method, 0) + 1
        self.last_reason = ""
        return fix

    def rectify(self, img_K: np.ndarray, est_pose: PoseEstimate, agl_m: float | None,
                intrinsics: dict[str, float]) -> tuple[np.ndarray, float, float, np.ndarray] | None:
        """North-up template centred on the ground point of the optical axis (pitch / roll in cruise move the
        footprint by height x tan(tilt): a template under the camera left the footprint at ~5 deg pitch).
        Returns (template, half size m, height above surface, centre - camera xy)."""
        c = self.cfg
        om = self.map
        pos = est_pose.pos
        s0 = om.surface_z(pos[:2])
        cam_z = s0 + agl_m if agl_m is not None and math.isfinite(agl_m) else float(pos[2]) - CAM_Z_OFFSET
        alt = cam_z - s0
        if alt < c.min_alt:
            return self._fail(f"below {c.min_alt:.0f} m")
        if not om.contains(pos[:2], margin=5.0):
            return self._fail("off map")
        mpp = c.tpl_mpp
        R_wc = est_pose.R @ R_BODY_CAM
        axis = -R_wc[:, 2]  # optical axis, world
        if axis[2] > -0.5:
            return self._fail("footprint (tilt)")
        off = axis[:2] * alt / -axis[2]
        half = float(np.clip(c.tpl_frac * alt, c.tpl_min_m, c.tpl_max_m))
        n = int(round(2 * half / mpp))
        g = (np.arange(n) + 0.5) * mpp - half
        dx, dy = np.meshgrid(g + off[0], -g + off[1])
        surf = om.surface_heights(np.stack([pos[0] + dx, pos[1] + dy], -1))
        rel = np.stack([dx, dy, surf - cam_z], -1)
        pc = rel @ R_wc
        depth = -pc[..., 2]
        fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
        h, w = img_K.shape
        u = (cx + fx * pc[..., 0] / np.maximum(depth, 1e-3) - 0.5).astype(np.float32)
        v = (cy - fy * pc[..., 1] / np.maximum(depth, 1e-3) - 0.5).astype(np.float32)
        inside = (depth > 0) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
        if inside.mean() < 0.97:
            return self._fail("footprint (tilt)")
        gsd = alt / fy
        img = np.asarray(img_K, np.float32)
        if mpp > 1.5 * gsd:  # anti-alias: area-average down to ~the cell size, then remap
            k = max(1, int(mpp / gsd / 1.5))
            if k > 1:
                img = cv2.resize(img, (w // k, h // k), interpolation=cv2.INTER_AREA)
                u, v = (u + 0.5) / k - 0.5, (v + 0.5) / k - 0.5
        tpl = cv2.remap(img, u, v, cv2.INTER_LINEAR)
        if float(tpl.std()) < c.min_std_K:
            return self._fail("uniform")
        self.last_tpl = tpl
        return tpl, half, alt, off

    # ------------------------------------------------------------------ peak search helper

    def _peak(self, score: np.ndarray, px: float, py: float, half: float, S: int, best_rot: float,
              min_score: float, alt: float, method: str) -> ThermalFix | None:
        c = self.cfg
        mpp = c.tpl_mpp
        _, best, _, (j, i) = cv2.minMaxLoc(score)
        best = float(best)
        if best < min_score:
            return self._fail(f"{method}: low score", best)
        # Peak region: the connected cells within peak_delta of the best score. Its shape is the information:
        # one straight edge or road gives a ridge (1-D fix), a corner or junction a compact peak. A round
        # covariance from the curvature on the two grid axes reported ridges as sharp 2-D fixes (wrong by 10-20 m
        # along the ridge, 25 % of the Berkeley fixes in the open-loop bench).
        reg = (score >= best - c.peak_delta).astype(np.uint8)
        _, lab = cv2.connectedComponents(reg, connectivity=8)
        comp = lab == lab[i, j]
        if comp[0, :].any() or comp[-1, :].any() or comp[:, 0].any() or comp[:, -1].any():
            return self._fail(f"{method}: ridge leaves the window", best)
        near = cv2.dilate(comp.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        local_max = (score >= cv2.dilate(score, np.ones((5, 5), np.uint8))) & ~near & (score > -1.0)
        second = float(score[local_max].max()) if local_max.any() else -1.0
        if best - second < c.min_margin:
            return self._fail(f"{method}: ambiguous", best)
        ii, jj = np.nonzero(comp)
        w = score[ii, jj] - (best - c.peak_delta) + 1e-3
        w = w / w.sum()
        mi, mj = float((w * ii).sum()), float((w * jj).sum())
        di_, dj_ = ii - mi, jj - mj
        cpx = np.array([[(w * dj_ * dj_).sum(), -(w * dj_ * di_).sum()],
                        [-(w * dj_ * di_).sum(), (w * di_ * di_).sum()]]) + np.eye(2) / 12.0  # (x, y) in cells^2
        x = px - S * mpp + mj * mpp
        y = py + S * mpp - mi * mpp
        k = c.peak_cov_scale * (1.0 + 2.0 * (1.0 - best)) ** 2
        cov = cpx * mpp * mpp * k + np.eye(2) * (c.sigma_floor ** 2 + (alt * math.radians(c.tilt_std_deg)) ** 2)
        self.last_score = best
        ev = np.linalg.eigvalsh(cov)
        return ThermalFix(np.array([x, y]), cov, best_rot, best, method,
                          info={"second": second, "margin": best - second, "sig_major": math.sqrt(ev[1]),
                                "sig_minor": math.sqrt(ev[0]), "peak_cells": int(comp.sum())})

    def _rotations(self, tpl: np.ndarray) -> list[tuple[float, np.ndarray]]:
        n = tpl.shape[0]
        out = []
        for a in self.cfg.yaw_search_deg:
            if a == 0:
                out.append((0.0, tpl))
                continue
            # template rotated by +a (CCW) best matching = the estimated heading is a too small: yaw_corr = +a
            M = cv2.getRotationMatrix2D(((n - 1) / 2, (n - 1) / 2), a, 1.0)
            out.append((math.radians(a), cv2.warpAffine(tpl, M, (n, n), flags=cv2.INTER_LINEAR,
                                                         borderMode=cv2.BORDER_REFLECT)))
        return out

    def _crop(self, a: np.ndarray, frac: float = 0.8) -> np.ndarray:
        """Central part of a rotated template (rotation leaves reflected corners)."""
        n = a.shape[0]
        k = int(n * (1 - frac) / 2)
        return a[k:n - k, k:n - k] if k > 0 else a

    def _search(self, region: np.ndarray, tpl: np.ndarray, S: int) -> tuple[np.ndarray, float]:
        """Best NCC map over the yaw candidates (cropped templates; offsets compensated)."""
        best_map, best_val, best_rot = None, -2.0, 0.0
        n = tpl.shape[0]
        vals = []
        for rot, t in self._rotations(tpl):
            tc = self._crop(t)
            k = (n - tc.shape[0]) // 2
            sc = cv2.matchTemplate(region, tc, cv2.TM_CCOEFF_NORMED)
            sc = sc[k:k + 2 * S + 1, k:k + 2 * S + 1] if sc.ndim == 2 else sc
            v = float(sc.max())
            vals.append((rot, v))
            if v > best_val:
                best_map, best_val, best_rot = sc, v, rot
        return best_map, self._refine_rot(vals, best_rot)  # type: ignore[return-value]

    @staticmethod
    def _refine_rot(vals: list[tuple[float, float]], best_rot: float) -> float:
        """Parabola through the best rotation's score and its neighbours (equal steps)."""
        vals = sorted(vals)
        i = [r for r, _ in vals].index(best_rot)
        if 0 < i < len(vals) - 1:
            (r0, a), (_, b), (r2, cc) = vals[i - 1], vals[i], vals[i + 1]
            den = a - 2 * b + cc
            if den < 0:
                return best_rot + float(np.clip(0.5 * (a - cc) / den, -0.5, 0.5)) * (r2 - r0) / 2
        return best_rot

    # ------------------------------------------------------------------ methods

    def _prior_blur(self, sigma_cells: float) -> np.ndarray:
        key = int(round(sigma_cells * 4))
        if key not in self._blur_cache:
            self._blur_cache[key] = cv2.GaussianBlur(self.temp, (0, 0), key / 4) if key > 0 else self.temp
        return self._blur_cache[key]

    def _m_ncc(self, tpl: np.ndarray, px: float, py: float, half: float, S: int, alt: float) -> ThermalFix | None:
        region = self._region(self._prior_blur(0.7), px, py, half, S, self.cfg.tpl_mpp)
        if region is None:
            return self._fail("off map")
        t = cv2.GaussianBlur(tpl, (0, 0), 0.7)
        sc, rot = self._search(region, t, S)
        return self._peak(sc, px, py, half, S, rot, self.cfg.min_score, alt, "ncc")

    def _m_grad(self, tpl: np.ndarray, px: float, py: float, half: float, S: int, alt: float) -> ThermalFix | None:
        if 0 not in self._orient_cache:
            self._orient_cache[0] = _orient_channels(self._prior_blur(0.7))
        pch = self._orient_cache[0]
        regs = [self._region(np.ascontiguousarray(pch[..., k]), px, py, half, S, self.cfg.tpl_mpp) for k in range(4)]
        if any(r is None for r in regs):
            return self._fail("off map")
        tch = _orient_channels(cv2.GaussianBlur(tpl, (0, 0), 0.7))
        best_map, best_val, best_rot = None, -2.0, 0.0
        n = tpl.shape[0]
        vals = []
        for rot, _ in self._rotations(tpl):
            M = cv2.getRotationMatrix2D(((n - 1) / 2, (n - 1) / 2), math.degrees(rot), 1.0)
            acc = None
            for k in range(4):
                ch = cv2.warpAffine(np.ascontiguousarray(tch[..., k]), M, (n, n), borderMode=cv2.BORDER_REFLECT) if rot else tch[..., k]
                tc = self._crop(np.ascontiguousarray(ch))
                kk = (n - tc.shape[0]) // 2
                sc = cv2.matchTemplate(regs[k], tc, cv2.TM_CCOEFF_NORMED)[kk:kk + 2 * S + 1, kk:kk + 2 * S + 1]
                acc = sc if acc is None else acc + sc
            acc = acc / 4.0
            v = float(acc.max())
            vals.append((rot, v))
            if v > best_val:
                best_map, best_val, best_rot = acc, v, rot
            # (orientation bins are not rotated with the template: +-3 deg is within one soft bin)
        return self._peak(best_map, px, py, half, S, self._refine_rot(vals, best_rot), self.cfg.grad_min_score, alt, "grad")

    def _observed_centerline(self, tpl: np.ndarray) -> np.ndarray:
        """Warm, elongated structures (roads) of the rectified image -> centerline mask."""
        t = cv2.GaussianBlur(tpl, (0, 0), 1.0)
        bg = cv2.GaussianBlur(t, (0, 0), 12.0)  # local background: roads are warmer than their surroundings
        d = t - bg
        thr = max(0.5, float(np.percentile(d, 70)))
        warm = (d > thr).astype(np.uint8)
        warm = cv2.morphologyEx(warm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(warm, 8)
        keep = np.zeros_like(warm)
        for k in range(1, n):  # elongated components only (roads), not compact blobs (roofs, cars)
            x, y, w, h, area = stats[k]
            if max(w, h) >= 16 and area / max(w * h, 1) < 0.6:
                keep[lab == k] = 1
        return _skeleton(keep)

    def _m_road(self, tpl: np.ndarray, px: float, py: float, half: float, S: int, alt: float) -> ThermalFix | None:
        c = self.cfg
        mpp = c.tpl_mpp
        skel = self._observed_centerline(tpl)
        npx = int(skel.sum())
        if npx < c.road_min_px:
            return self._fail("road: no centerline")
        reg = self._region(self.centerline.astype(np.float32), px, py, half, S, mpp, cv2.INTER_NEAREST)
        if reg is None:
            return self._fail("off map")
        if reg.max() <= 0:
            return self._fail("road: no mapped road")
        dt = cv2.distanceTransform((reg < 0.5).astype(np.uint8), cv2.DIST_L2, 3) * mpp
        dt = np.minimum(dt, c.road_trunc_m).astype(np.float32)
        best = None
        n = tpl.shape[0]
        for rot, _ in self._rotations(tpl):
            M = cv2.getRotationMatrix2D(((n - 1) / 2, (n - 1) / 2), math.degrees(rot), 1.0)
            sk = cv2.warpAffine(skel.astype(np.float32), M, (n, n), flags=cv2.INTER_NEAREST) if rot else skel.astype(np.float32)
            cnt = float(sk.sum())
            if cnt < c.road_min_px:
                continue
            cost = cv2.matchTemplate(dt, sk, cv2.TM_CCORR) / cnt  # mean truncated distance per shift
            v = float(cost.min())
            if best is None or v < best[0]:
                best = (v, cost, rot, sk)
        if best is None:
            return self._fail("road: no centerline")
        cmin, cost, rot, sk = best
        i, j = np.unravel_index(int(np.argmin(cost)), cost.shape)
        # agreement: share of centerline pixels within 1.5 m of a mapped road at the best shift
        win = dt[i:i + n, j:j + n]
        quality = float(((win < 1.5) & (sk > 0)).sum() / max(sk.sum(), 1))
        if quality < c.road_min_quality:
            return self._fail("road: poor agreement", quality)
        # likelihood over shifts -> mean and covariance (anisotropic: a straight road is 1-D)
        L = np.exp(-(cost - cmin) / c.road_temp_T_m)
        L /= L.sum()
        ii, jj = np.mgrid[0:cost.shape[0], 0:cost.shape[1]]
        xs = px - S * mpp + jj * mpp
        ys = py + S * mpp - ii * mpp
        mx, my = float((L * xs).sum()), float((L * ys).sum())
        dx, dy = xs - mx, ys - my
        cov = np.array([[(L * dx * dx).sum(), (L * dx * dy).sum()], [(L * dx * dy).sum(), (L * dy * dy).sum()]])
        # peak vs mean: a bimodal likelihood (parallel roads) has its mean between the modes
        bx = px - S * mpp + j * mpp
        by = py + S * mpp - i * mpp
        ev, evec = np.linalg.eigh(cov)
        ev = np.maximum(ev, c.road_sigma_floor ** 2)
        if math.sqrt(ev[0]) > c.road_max_minor_m:
            return self._fail("road: flat")
        if math.hypot(mx - bx, my - by) > 2.0 * math.sqrt(ev[1]) + 2.0:
            return self._fail("road: multimodal")
        ev[0] += (alt * math.radians(c.tilt_std_deg)) ** 2
        ev[1] += (alt * math.radians(c.tilt_std_deg)) ** 2
        cov = evec @ np.diag(ev) @ evec.T
        self.last_skel_xy = np.argwhere(sk > 0)
        return ThermalFix(np.array([bx, by]), cov, rot, quality, "road",
                          info={"cost": cmin, "sig_major": math.sqrt(ev[1]), "sig_minor": math.sqrt(ev[0])})

    def _m_junction(self, tpl: np.ndarray, px: float, py: float, half: float, S: int, alt: float) -> ThermalFix | None:
        c = self.cfg
        mpp = c.tpl_mpp
        if len(self.junctions) == 0:
            return self._fail("junction: none mapped")
        skel = self._observed_centerline(tpl)
        js = _junctions_px(skel, 4.0 / mpp)
        if not js:
            return self._fail("junction: none seen")
        n = tpl.shape[0]
        gate = min(3.0 * S * mpp / 3.0 + 2.0, c.junc_gate_m) if S else c.junc_gate_m
        offs = []
        self.last_junctions_xy = []
        for r, col in js:
            jx = px - half + (col + 0.5) * mpp
            jy = py + half - (r + 0.5) * mpp
            d = np.hypot(self.junctions[:, 0] - jx, self.junctions[:, 1] - jy)
            order = np.argsort(d)
            if d[order[0]] > gate:
                continue
            if len(order) > 1 and d[order[1]] < 1.5 * d[order[0]] + 3.0:
                continue  # not unique
            offs.append(self.junctions[order[0]] - np.array([jx, jy]))
            self.last_junctions_xy.append(self.junctions[order[0]])
        if not offs:
            return self._fail("junction: no unique match")
        off = np.mean(offs, axis=0)
        sig2 = c.junc_sigma ** 2 / len(offs) + (alt * math.radians(c.tilt_std_deg)) ** 2
        return ThermalFix(np.array([px, py]) + off, np.eye(2) * sig2, 0.0, min(1.0, 0.5 + 0.25 * len(offs)), "junction",
                          info={"n": len(offs)})

    # ------------------------------------------------------------------ combination

    def _combine(self, cands: list[ThermalFix]) -> ThermalFix:
        if len(cands) == 1:
            return cands[0]
        # pairwise agreement; fuse the largest agreeing set (information-weighted), prefer smaller covariance
        best_set: list[ThermalFix] = []
        for a in cands:
            group = [a]
            for b in cands:
                if b is a:
                    continue
                d = a.xy - b.xy
                if float(d @ np.linalg.solve(a.cov + b.cov, d)) < self.cfg.combo_agree_chi2:
                    group.append(b)
            if len(group) > len(best_set) or (len(group) == len(best_set) and
                                              np.trace(group[0].cov) < np.trace(best_set[0].cov)):
                best_set = group
        if len(best_set) == 1:
            return min(cands, key=lambda f: float(np.trace(f.cov)) / max(f.quality, 0.1))
        info = sum(np.linalg.inv(f.cov) for f in best_set)
        cov = np.linalg.inv(info)
        xy = cov @ sum(np.linalg.inv(f.cov) @ f.xy for f in best_set)
        yaw = [f.yaw_corr for f in best_set if f.method in ("ncc", "grad", "road")]
        return ThermalFix(xy, cov, float(np.mean(yaw)) if yaw else 0.0, max(f.quality for f in best_set),
                          "+".join(sorted(f.method for f in best_set)))
