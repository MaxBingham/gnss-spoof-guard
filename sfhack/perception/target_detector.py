"""V4 target perception (onboard, camera + range sensors only): find the six objects in the search area,
measure them, track them over frames and score what each one is. Plus the landing-station marker (ArUco).

Inputs: down-camera frames, the pose ESTIMATE (EKF position, AHRS attitude), the onboard map (ortho +
landmarks), lidar / rangefinder returns. No truth, no layout: the objects are not in the onboard map.

1. Change detection (primary segmentation). Ortho-rectify the frame onto a north-up ground grid in the EKF
   frame (flat ground, AHRS attitude, EKF height), sample the onboard ortho on the same grid, register the
   two (residual shift up to `det_register_m`, NCC), fit a per-channel gain + offset (lighting, haze) and
   compare in map albedo units with an alignment tolerance (per pixel: min color distance over +- a few
   grid pixels). Shadows, canopy, paving and dirt tracks are in the map, so they cancel; new objects do not.
   Known buildings and mapped trees (onboard landmarks) are masked out (their parallax is not in the ortho).
2. Blobs: threshold (robust noise of the difference), morphology, split vegetation-green pixels from the
   rest (a tree crown hanging over the truck must not merge with it), connected components. Per blob:
   metric footprint (min-area rectangle -> length, width, yaw), fill ratio, median top color (Lab, albedo),
   distance from nadir (oblique views get a lower weight: side walls and parallax inflate them).
3. Height (second, independent cue): lidar fan + rangefinder returns become 3D points in the EKF frame; a
   candidate's height is a high percentile of the points inside its footprint above the local ground.
4. Tracks: blobs associated by ground position over frames; size / color evidence is accumulated as
   weighted means (consistency = their spread). A blob that keeps changing while its track is in view but
   undetected is dropped as a false blob; one whose measured height is flat is rejected.
5. Class probabilities {white_car, brown_car, brown_truck, tent, hut, tree, unknown}: Gaussian class models
   on length, width, height, color and fill. Each class hypothesis corrects the footprint for its own roof
   height (a roof h above ground at camera height H projects h / (H - h) larger onto the ground plane).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import TargetConfig
from ..messages import CameraFrame, MarkerDetection, PoseEstimate, TargetCandidate
from ..world.onboard_map import OnboardMap
from .map_localizer import CAM_Z_OFFSET, R_BODY_CAM

CLASSES = ("white_car", "brown_car", "brown_truck", "tent", "hut", "tree", "unknown")
TARGET_CLASS = "brown_truck"
NAMES = {"white_car": "white car", "brown_car": "brown car", "brown_truck": "brown truck", "tent": "tent",
         "hut": "hut", "tree": "tree", "unknown": "unknown object"}


# ---------------------------------------------------------------------------------------- class models

@dataclass
class ClassModel:
    """Expected appearance from above. Sizes: footprint [m]; height: 80th percentile of the range returns
    over the footprint [m] (top height for flat roofs, lower for a pyramid); colors: Lab of the top
    surface in map albedo units (any of them); fill: blob area / its min-area rectangle."""

    length: tuple[float, float]
    width: tuple[float, float]
    height: tuple[float, float]
    colors: tuple[tuple[float, float, float], ...]
    color_sigma: tuple[float, float, float]
    fill: tuple[float, float] = (0.92, 0.08)
    prior: float = 1.0


BROWN = (51.0, 14.0, 32.0)
CLASS_MODELS: dict[str, ClassModel] = {
    "white_car": ClassModel((4.5, 0.45), (1.9, 0.3), (1.45, 0.35), ((92.0, -1.0, 2.0),), (8.0, 5.0, 6.0)),
    "brown_car": ClassModel((4.5, 0.45), (1.9, 0.3), (1.45, 0.35), (BROWN,), (8.0, 5.0, 6.0)),
    "brown_truck": ClassModel((8.0, 0.7), (2.55, 0.35), (3.15, 0.4), (BROWN,), (8.0, 5.0, 6.0)),
    "tent": ClassModel((3.0, 0.45), (3.0, 0.45), (1.2, 0.4), ((52.0, -42.0, 38.0), (67.0, 38.0, 70.0)), (10.0, 8.0, 9.0)),
    "hut": ClassModel((4.6, 0.5), (4.4, 0.5), (2.8, 0.4), ((64.0, -4.0, -5.0),), (9.0, 5.0, 6.0)),
    "tree": ClassModel((4.8, 0.9), (4.6, 0.9), (7.2, 1.6), ((44.0, -36.0, 30.0),), (9.0, 8.0, 8.0), fill=(0.80, 0.08)),
}
UNKNOWN_PRIOR = 0.05
# Broad ranges of the "unknown" class: uniform density over them
UNKNOWN_RANGE = {"length": 16.0, "width": 10.0, "height": 12.0, "L": 100.0, "a": 120.0, "b": 120.0, "fill": 1.0}
MEAS_SIGMA = {"length": 0.5, "width": 0.35, "fill": 0.08, "L": 6.0, "a": 4.0, "b": 4.0}  # one-observation noise


def color_name(lab: np.ndarray) -> str:
    L, a, b = (float(v) for v in lab)
    chroma = math.hypot(a, b)
    if L > 72 and chroma < 14:
        return "white"
    if a < -10 and b > 5:
        return "green"
    if a > 20 and b > 40 and L > 48:
        return "orange"
    if chroma < 11:
        return "gray"
    if 3 < a < 28 and 10 < b < 45 and L < 62:
        return "brown"
    return "other"


def _lab(rgb255: np.ndarray) -> np.ndarray:
    px = np.clip(np.asarray(rgb255, np.float32).reshape(1, 1, 3) / 255.0, 0, 1)
    return cv2.cvtColor(px, cv2.COLOR_RGB2Lab)[0, 0].astype(float)


# ---------------------------------------------------------------------------------------- blobs

@dataclass
class Blob:
    """One change blob in one frame (EKF world frame)."""

    xy: np.ndarray
    length: float
    width: float
    yaw: float
    area: float
    fill: float
    lab: np.ndarray
    color: str
    r_nadir: float  # m, horizontal distance from the camera's nadir point
    alt: float  # m, camera height above ground
    complete: bool  # not cut by the image / grid edge
    merged: bool  # implausible shape (two objects merged, misregistration): position only
    weight: float  # nadir weight in (0, 1]
    corners: np.ndarray  # (4, 2) world corners of the min-area rectangle
    green: bool


@dataclass
class FrameDetections:
    t: float
    seq: int
    blobs: list[Blob] = field(default_factory=list)
    ok: bool = False
    reason: str = ""
    contrast: float = float("nan")  # camera / map gain (1 = clear air, < 1 = haze)
    noise: float = float("nan")  # robust std of the background difference (albedo units)
    thresh: float = float("nan")
    shift_m: np.ndarray = field(default_factory=lambda: np.zeros(2))  # registration (map - EKF grid)
    registered: bool = False
    reg_score: tuple[float, float] = (float("nan"), float("nan"))  # NCC peak, NCC median
    grid: tuple[float, float, float, int, int] | None = None  # x0 (west), y0 (north), mpp, rows, cols
    valid: np.ndarray | None = None  # (rows, cols) bool: grid cells seen by the camera (and not masked)
    mask: np.ndarray | None = None  # (rows, cols) uint8 change mask (display)

    def in_view(self, xy: np.ndarray, margin_m: float = 2.0) -> bool:
        """True if the ground point is well inside the area the camera covered in this frame."""
        if self.grid is None or self.valid is None:
            return False
        x0, y0, mpp, rows, cols = self.grid
        c, r = (xy[0] - x0) / mpp, (y0 - xy[1]) / mpp
        m = int(math.ceil(margin_m / mpp))
        ri, ci = int(r), int(c)
        if ri - m < 0 or ci - m < 0 or ri + m >= rows or ci + m >= cols:
            return False
        return bool(self.valid[ri - m:ri + m + 1, ci - m:ci + m + 1].all())


def camera_pose(pose: PoseEstimate) -> tuple[np.ndarray, np.ndarray]:
    """(camera position, camera -> world rotation) from the pose estimate (MuJoCo camera axes: x right,
    y up, looking along -z)."""
    R_wc = pose.R @ R_BODY_CAM
    return pose.pos + pose.R @ np.array([0.0, 0.0, CAM_Z_OFFSET]), R_wc


def project_to_image(pts_world: np.ndarray, frame: CameraFrame, pose: PoseEstimate) -> np.ndarray:
    """(N, 3) world points -> (N, 2) pixel coordinates (u right, v down; nan behind the camera)."""
    cam, R_wc = camera_pose(pose)
    pc = (np.asarray(pts_world, float) - cam) @ R_wc
    depth = -pc[:, 2]
    fx, fy, cx, cy = frame.intrinsics()
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(depth > 0.1, cx + fx * pc[:, 0] / depth, np.nan)
        v = np.where(depth > 0.1, cy - fy * pc[:, 1] / depth, np.nan)
    return np.column_stack([u, v])


class ChangeDetector:
    def __init__(self, cfg: TargetConfig, onboard_map: OnboardMap) -> None:
        self.cfg = cfg
        self.om = onboard_map
        self.map_rgb = onboard_map.ortho.astype(np.float32)
        self.map_excl = self._exclusion_mask(onboard_map)

    @staticmethod
    def _exclusion_mask(om: OnboardMap) -> np.ndarray:
        """Known structures of the onboard map (buildings dilated for their parallax, mapped trees): their
        appearance from an oblique camera is not in the ortho, so changes there are not new objects."""
        h, w = om.gray.shape
        mask = np.zeros((h, w), np.uint8)
        for lm in om.landmarks:
            if lm["type"] in ("building", "tower") and lm.get("footprint_enu"):
                pts = om.enu_to_px(np.asarray(lm["footprint_enu"], float)).astype(np.int32)
                cv2.fillPoly(mask, [pts], 255)
            elif lm["type"] in ("building", "tower", "tree"):
                u, v = om.enu_to_px(np.asarray(lm["centroid_enu"], float))
                r = float(lm.get("radius", 3.0)) + 1.0
                cv2.circle(mask, (int(u), int(v)), int(math.ceil(r / om.m_per_px)), 255, -1)
        k = int(math.ceil(3.0 / om.m_per_px))
        if mask.any():
            bld = np.zeros_like(mask)
            for lm in om.landmarks:
                if lm["type"] in ("building", "tower") and lm.get("footprint_enu"):
                    cv2.fillPoly(bld, [om.enu_to_px(np.asarray(lm["footprint_enu"], float)).astype(np.int32)], 255)
            mask |= cv2.dilate(bld, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
        return mask

    def _sample_map(self, img: np.ndarray, X: np.ndarray, Y: np.ndarray, interp: int) -> np.ndarray:
        om = self.om
        mu = ((X - om.min_x) / om.mpp_x - 0.5).astype(np.float32)
        mv = ((om.max_y - Y) / om.mpp_y - 0.5).astype(np.float32)
        return cv2.remap(img, mu, mv, interp, borderMode=cv2.BORDER_REPLICATE)

    def detect(self, frame: CameraFrame, pose: PoseEstimate, area_center: np.ndarray, area_half: float) -> FrameDetections:
        c = self.cfg
        out = FrameDetections(frame.t, frame.seq)
        if frame.name != "down" or int(frame.image.max()) < 8:
            out.reason = "no image"
            return out
        cam, R_wc = camera_pose(pose)
        alt = float(cam[2])
        if alt < c.det_min_alt:
            out.reason = f"below {c.det_min_alt:.0f} m"
            return out
        fx, fy, cx, cy = frame.intrinsics()
        W, H = frame.width, frame.height

        # ---- ground footprint of the image (flat ground), clipped to the search area
        corners_px = np.array([[0, 0], [W, 0], [W, H], [0, H]], float)
        d = np.column_stack([(corners_px[:, 0] - cx) / fx, -(corners_px[:, 1] - cy) / fy, -np.ones(4)]) @ R_wc.T
        if np.any(d[:, 2] > -0.05):
            out.reason = "footprint (tilt)"
            return out
        s = -alt / d[:, 2]
        gx, gy = cam[0] + s * d[:, 0], cam[1] + s * d[:, 1]
        mpp = c.det_mpp
        margin = 4.0
        x_lo = max(gx.min(), area_center[0] - area_half - margin)
        x_hi = min(gx.max(), area_center[0] + area_half + margin)
        y_lo = max(gy.min(), area_center[1] - area_half - margin)
        y_hi = min(gy.max(), area_center[1] + area_half + margin)
        if x_hi - x_lo < 6 or y_hi - y_lo < 6:
            out.reason = "outside the area"
            return out
        x0 = math.floor(x_lo / mpp) * mpp
        y0 = math.ceil(y_hi / mpp) * mpp
        cols = int((x_hi - x0) / mpp)
        rows = int((y0 - y_lo) / mpp)
        S = int(math.ceil(c.det_register_m / mpp))
        k = c.det_align_px
        # extended grid (S margin) for the map: registration shift + alignment tolerance
        ge = (np.arange(-S, cols + S) + 0.5) * mpp
        gn = (np.arange(-S, rows + S) + 0.5) * mpp
        Xe, Ye = np.meshgrid(x0 + ge, y0 - gn)
        X, Y = Xe[S:S + rows, S:S + cols], Ye[S:S + rows, S:S + cols]

        # ---- camera -> grid
        rel = np.stack([X - cam[0], Y - cam[1], np.full_like(X, -alt)], axis=-1)
        pc = rel @ R_wc
        depth = -pc[..., 2]
        u = (cx + fx * pc[..., 0] / depth - 0.5).astype(np.float32)
        v = (cy - fy * pc[..., 1] / depth - 0.5).astype(np.float32)
        valid = (depth > 0) & (u >= 1) & (u <= W - 2) & (v >= 1) & (v <= H - 2)
        cam_rect = cv2.remap(frame.image.astype(np.float32), u, v, cv2.INTER_LINEAR)
        gsd = alt / fy
        if mpp > 1.3 * gsd:
            cam_rect = cv2.GaussianBlur(cam_rect, (0, 0), 0.4 * mpp / gsd)
        map_ext = self._sample_map(self.map_rgb, Xe, Ye, cv2.INTER_LINEAR)
        excl = self._sample_map(self.map_excl, X, Y, cv2.INTER_NEAREST) > 0
        use = valid & ~excl
        if use.mean() < 0.2:
            out.reason = "too little ground in view"
            return out

        # ---- registration: residual shift of the map against the camera (EKF error), NCC on gray
        cam_g = cv2.GaussianBlur(cv2.cvtColor(cam_rect, cv2.COLOR_RGB2GRAY), (0, 0), 1.0)
        map_g = cv2.GaussianBlur(cv2.cvtColor(map_ext, cv2.COLOR_RGB2GRAY), (0, 0), 1.0)
        dy = dx = 0
        rv, cv_ = np.where(valid.all(axis=1))[0], np.where(valid.all(axis=0))[0]
        if len(rv) > 40 and len(cv_) > 40:
            r0, r1, c0, c1 = rv[0] + 2, rv[-1] - 2, cv_[0] + 2, cv_[-1] - 2
            tpl = cam_g[r0:r1, c0:c1]
            if tpl.size > 1600 and float(tpl.std()) > 2.0:
                region = map_g[r0:r1 + 2 * S, c0:c1 + 2 * S]
                ncc = cv2.matchTemplate(region, tpl, cv2.TM_CCOEFF_NORMED)
                _, best, _, (j, i) = cv2.minMaxLoc(ncc)
                out.reg_score = (float(best), float(np.median(ncc)))
                if best > 0.35 and best - float(np.median(ncc)) > 0.15:
                    lim = S - k
                    dy, dx = int(np.clip(i - S, -lim, lim)), int(np.clip(j - S, -lim, lim))
                    out.registered = True
        out.shift_m = np.array([dx * mpp, -dy * mpp])

        # ---- photometric normalization: cam = gain * map + offset per channel (robust LSQ on the ground)
        map_al = map_ext[S + dy:S + dy + rows, S + dx:S + dx + cols]
        mvals, cvals = map_al[use], cam_rect[use]
        gain, off = np.ones(3), np.zeros(3)
        for _ in range(2):
            for ch in range(3):
                A = np.column_stack([mvals[:, ch], np.ones(len(mvals))])
                sol, *_ = np.linalg.lstsq(A, cvals[:, ch], rcond=None)
                gain[ch], off[ch] = sol
            res = np.linalg.norm(cvals - (mvals * gain + off), axis=1)
            keep = res < max(3.0 * 1.4826 * float(np.median(res)), 6.0)
            if keep.sum() < 200:
                break
            mvals, cvals = mvals[keep], cvals[keep]
        gain = np.maximum(gain, 0.05)
        out.contrast = float(np.mean(gain))
        cam_n = (cam_rect - off) / gain  # camera in map albedo units

        # ---- difference with alignment tolerance: min over +-k grid pixels of the map
        D = np.full((rows, cols), np.inf, np.float32)
        for ddy in range(-k, k + 1):
            for ddx in range(-k, k + 1):
                m = map_ext[S + dy + ddy:S + dy + ddy + rows, S + dx + ddx:S + dx + ddx + cols]
                D = np.minimum(D, np.linalg.norm(cam_n - m, axis=2))
        bg = D[use]
        med = float(np.median(bg))
        noise = 1.4826 * float(np.median(np.abs(bg - med)))
        thr = max(c.det_thresh_floor, med + c.det_thresh_k * noise)
        out.noise, out.thresh = noise, thr
        change = ((D > thr) & use).astype(np.uint8)
        change = cv2.morphologyEx(change, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        change = cv2.morphologyEx(change, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        out.mask = change * 255
        out.valid = use
        out.grid = (x0, y0, mpp, rows, cols)
        out.ok = True

        # ---- vegetation split, connected components, blob measurements
        R_, G_, B_ = cam_n[..., 0], cam_n[..., 1], cam_n[..., 2]
        green = (G_ > 1.12 * R_) & (G_ > 1.25 * B_)
        edge = cv2.dilate((~valid).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        edge[0, :] = edge[-1, :] = True
        edge[:, 0] = edge[:, -1] = True
        min_px = c.det_min_area / (mpp * mpp)
        for layer_is_green in (False, True):
            layer = (change > 0) & (green if layer_is_green else ~green)
            layer = cv2.morphologyEx(layer.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n, labels, stats, _ = cv2.connectedComponentsWithStats(layer, connectivity=8)
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] < min_px:
                    continue
                comp = labels == i
                b = self._measure(comp, cam_n, x0, y0, mpp, cam, alt, edge, layer_is_green)
                if b is not None:
                    out.blobs.append(b)
        return out

    def _measure(self, comp: np.ndarray, cam_n: np.ndarray, x0: float, y0: float, mpp: float, cam: np.ndarray,
                 alt: float, edge: np.ndarray, is_green: bool) -> Blob | None:
        c = self.cfg
        rr, cc = np.nonzero(comp)
        pts = np.column_stack([cc, rr]).astype(np.float32)
        (pcx, pcy), (w, h), ang = cv2.minAreaRect(pts)
        box = cv2.boxPoints(((pcx, pcy), (w + 1, h + 1), ang))
        corners = np.column_stack([x0 + (box[:, 0] + 0.5) * mpp, y0 - (box[:, 1] + 0.5) * mpp])
        bias = c.det_edge_bias  # blur halo + rectangle fit: blobs come out this much too long and wide
        L, Wd = max((max(w, h) + 1) * mpp - bias, mpp), max((min(w, h) + 1) * mpp - bias, mpp)
        e = box[1] - box[0] if np.linalg.norm(box[1] - box[0]) >= np.linalg.norm(box[2] - box[1]) else box[2] - box[1]
        yaw = math.atan2(-e[1], e[0])
        area = float(len(rr)) * mpp * mpp
        fill = min(1.0, area / max((L + bias) * (Wd + bias), 1e-6))
        xy = np.array([x0 + (cc.mean() + 0.5) * mpp, y0 - (rr.mean() + 0.5) * mpp])
        inner = cv2.erode(comp.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        src = inner if inner.sum() >= 12 else comp
        lab = _lab(np.median(cam_n[src], axis=0))
        r_nadir = float(np.hypot(*(xy - cam[:2])))
        complete = not bool(edge[comp].any())
        merged = area > c.det_max_area or L > 11.0 or (area > 12.0 and fill < 0.62) or Wd > 6.5
        weight = 1.0 / (1.0 + (r_nadir / max(alt, 1.0) / 0.35) ** 2)
        return Blob(xy, float(L), float(Wd), yaw, area, float(fill), lab, color_name(lab), r_nadir, alt, complete,
                    bool(merged), float(weight), corners, is_green)


# ---------------------------------------------------------------------------------------- height

class HeightMap:
    """Range returns (lidar fan + rangefinder) as 3D points in the EKF frame, inside the search area."""

    def __init__(self, area_center: np.ndarray, area_half: float, margin: float = 6.0) -> None:
        self.c = np.asarray(area_center, float)
        self.half = area_half + margin
        self.pts = np.zeros((0, 3))
        self._buf: list[np.ndarray] = []

    def add(self, pose: PoseEstimate, dirs_body: np.ndarray, ranges: np.ndarray) -> None:
        ok = np.isfinite(ranges)
        if not ok.any():
            return
        sensor = pose.pos + pose.R @ np.array([0.0, 0.0, 0.01])
        p = sensor + (dirs_body[ok] @ pose.R.T) * ranges[ok, None]
        inside = np.all(np.abs(p[:, :2] - self.c) <= self.half, axis=1)
        if inside.any():
            self._buf.append(p[inside])

    def _flush(self) -> None:
        if self._buf:
            self.pts = np.concatenate([self.pts, *self._buf])
            self._buf = []

    def footprint_height(self, xy: np.ndarray, length: float, width: float, yaw: float,
                         min_returns: int) -> tuple[float | None, int, float]:
        """(height above the local ground, returns inside the footprint, ground spread). Height = 80th
        percentile of the returns inside the (slightly shrunk) footprint minus the median ground level in a
        ring around it (0 if no ground returns there)."""
        self._flush()
        if len(self.pts) == 0:
            return None, 0, 0.0
        d = self.pts[:, :2] - np.asarray(xy, float)
        c, s = math.cos(yaw), math.sin(yaw)
        lx, ly = np.abs(c * d[:, 0] + s * d[:, 1]), np.abs(-s * d[:, 0] + c * d[:, 1])
        hl, hw = max(length / 2 - 0.3, 0.4), max(width / 2 - 0.3, 0.3)
        inside = (lx <= hl) & (ly <= hw)
        ring = (lx <= length / 2 + 4.0) & (ly <= width / 2 + 4.0) & ~((lx <= length / 2 + 1.0) & (ly <= width / 2 + 1.0))
        z_in = self.pts[inside, 2]
        z_ring = self.pts[ring, 2]
        z_ring = z_ring[z_ring < 0.8] if len(z_ring) else z_ring  # ground returns only (not other objects)
        ground = float(np.median(z_ring)) if len(z_ring) >= 5 else 0.0
        spread = 1.4826 * float(np.median(np.abs(z_ring - ground))) if len(z_ring) >= 5 else 0.0
        if len(z_in) < min_returns:
            return None, int(len(z_in)), spread
        return float(np.percentile(z_in, 80)) - ground, int(len(z_in)), spread


# ---------------------------------------------------------------------------------------- tracks

@dataclass
class Track:
    id: int
    xy: np.ndarray
    first_t: float
    last_t: float
    n_obs: int = 0
    n_nadir: int = 0
    n_miss: int = 0  # frames with the track well in view but no blob for it
    n_merged: int = 0
    pos_w: float = 0.0
    # weighted feature accumulators (complete, non-merged observations only)
    sw: float = 0.0
    sw2: float = 0.0
    swL: float = 0.0
    swL_alt: float = 0.0  # sum w * L / alt (roof-height correction per class hypothesis)
    swW: float = 0.0
    swW_alt: float = 0.0
    swL2: float = 0.0
    swW2: float = 0.0
    swF: float = 0.0
    swF2: float = 0.0
    swLab: np.ndarray = field(default_factory=lambda: np.zeros(3))
    swLab2: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0
    green_votes: float = 0.0
    height: float | None = None
    height_n: int = 0
    height_spread: float = 0.0
    contrast_sum: float = 0.0
    status: str = "tentative"  # tentative | candidate | rejected
    reject_reason: str = ""
    cues: list[str] = field(default_factory=list)
    probs: dict[str, float] = field(default_factory=dict)
    last_blob: Blob | None = None

    def add_cue(self, cue: str) -> bool:
        if cue not in self.cues:
            self.cues.append(cue)
            return True
        return False

    @property
    def n_eff(self) -> float:
        return self.sw * self.sw / self.sw2 if self.sw2 > 0 else 0.0

    def mean_lab(self) -> np.ndarray:
        return self.swLab / self.sw if self.sw > 0 else np.full(3, np.nan)

    def size(self, roof_h: float = 0.0) -> tuple[float, float]:
        """Footprint corrected for a roof at `roof_h` (projection onto the ground plane magnifies it)."""
        if self.sw <= 0:
            return float("nan"), float("nan")
        return (self.swL - roof_h * self.swL_alt) / self.sw, (self.swW - roof_h * self.swW_alt) / self.sw

    def spread(self, s: float, s2: float) -> float:
        if self.sw <= 0:
            return 0.0
        m = s / self.sw
        return math.sqrt(max(s2 / self.sw - m * m, 0.0))

    @property
    def color(self) -> str:
        return color_name(self.mean_lab()) if self.sw > 0 else "?"

    def observe(self, b: Blob) -> None:
        self.n_obs += 1
        self.n_miss = 0
        if b.weight > 0.5:
            self.n_nadir += 1
        # position: nadir-weighted, recent observations count more (slow EKF drift)
        w = b.weight
        self.pos_w = 0.9 * self.pos_w + w
        self.xy = self.xy + (w / self.pos_w) * (b.xy - self.xy)
        self.green_votes += (1.0 if b.green else -1.0) * w
        if b.merged:
            self.n_merged += 1
            return
        if not b.complete:
            return
        self.sw += w
        self.sw2 += w * w
        self.swL += w * b.length
        self.swL_alt += w * b.length / b.alt
        self.swW += w * b.width
        self.swW_alt += w * b.width / b.alt
        self.swL2 += w * b.length ** 2
        self.swW2 += w * b.width ** 2
        self.swF += w * b.fill
        self.swF2 += w * b.fill ** 2
        self.swLab += w * b.lab
        self.swLab2 += w * b.lab ** 2
        if w >= 0.5 or self.sw == w:
            self.yaw = b.yaw
        self.last_blob = b


def _lognorm(x: float, mu: float, sigma: float) -> float:
    return -0.5 * ((x - mu) / sigma) ** 2 - math.log(sigma) - 0.5 * math.log(2 * math.pi)


def class_log_likelihoods(tr: Track, contrast_factor: float = 1.0) -> dict[str, float]:
    """Log likelihood of the track's accumulated evidence under each class model. `contrast_factor` >= 1
    widens the color terms when the images are degraded (haze washes colors out)."""
    out: dict[str, float] = {}
    if tr.sw <= 0:
        return {k: 0.0 for k in CLASSES}
    n = max(tr.n_eff, 1.0)
    lab = tr.mean_lab()
    lab_sd = np.sqrt(np.maximum(tr.swLab2 / tr.sw - lab ** 2, 0.0))
    fill = tr.swF / tr.sw
    fill_sd = tr.spread(tr.swF, tr.swF2)
    h_meas_sd = math.hypot(0.25, tr.height_spread)
    for k, m in CLASS_MODELS.items():
        roof = tr.height if tr.height is not None else m.height[0]
        L, W = tr.size(max(roof, 0.0))
        Ls, Ws = tr.spread(tr.swL, tr.swL2), tr.spread(tr.swW, tr.swW2)
        ll = _lognorm(L, m.length[0], math.sqrt(m.length[1] ** 2 + (Ls ** 2 + MEAS_SIGMA["length"] ** 2) / n))
        ll += _lognorm(W, m.width[0], math.sqrt(m.width[1] ** 2 + (Ws ** 2 + MEAS_SIGMA["width"] ** 2) / n))
        ll += _lognorm(fill, m.fill[0], math.sqrt(m.fill[1] ** 2 + (fill_sd ** 2 + MEAS_SIGMA["fill"] ** 2) / n))
        if tr.height is not None:
            ll += _lognorm(tr.height, m.height[0], math.hypot(m.height[1], h_meas_sd))
        cl = []
        for col in m.colors:
            s = 0.0
            for j, key in enumerate("Lab"):
                sig = math.sqrt((m.color_sigma[j] * contrast_factor) ** 2 + (lab_sd[j] ** 2 + MEAS_SIGMA[key] ** 2) / n)
                s += _lognorm(float(lab[j]), col[j], sig)
            cl.append(s)
        ll += float(np.logaddexp.reduce(cl)) - math.log(len(cl))
        out[k] = ll
    u = -math.log(UNKNOWN_RANGE["length"]) - math.log(UNKNOWN_RANGE["width"]) - math.log(UNKNOWN_RANGE["fill"])
    u -= math.log(UNKNOWN_RANGE["L"]) + math.log(UNKNOWN_RANGE["a"]) + math.log(UNKNOWN_RANGE["b"])
    if tr.height is not None:
        u -= math.log(UNKNOWN_RANGE["height"])
    out["unknown"] = u
    return out


def class_probs(tr: Track, contrast_factor: float = 1.0) -> dict[str, float]:
    ll = class_log_likelihoods(tr, contrast_factor)
    known = (1.0 - UNKNOWN_PRIOR) / (len(CLASSES) - 1)
    lp = np.array([ll[k] + math.log(UNKNOWN_PRIOR if k == "unknown" else known) for k in CLASSES])
    p = np.exp(lp - lp.max())
    p /= p.sum()
    return {k: float(v) for k, v in zip(CLASSES, p)}


# ---------------------------------------------------------------------------------------- perception

class TargetPerception:
    """Change detection + height + tracking + class probabilities. Logs new / confirmed / rejected
    candidates through `log(t, text)` (the mission forwards them to the decision log)."""

    def __init__(self, cfg: TargetConfig, onboard_map: OnboardMap, area_center: np.ndarray, log=None) -> None:
        self.cfg = cfg
        self.area_center = np.asarray(area_center, float)
        self.detector = ChangeDetector(cfg, onboard_map)
        self.heights = HeightMap(self.area_center, cfg.area_half)
        self.tracks: list[Track] = []
        self._next_id = 1
        self.log = log or (lambda t, text: None)
        self.last: FrameDetections | None = None
        self.contrast_ema: float | None = None
        self.n_frames = 0

    # ---- range sensors
    def on_lidar(self, angles: np.ndarray, ranges: np.ndarray, pose: PoseEstimate) -> None:
        a = np.asarray(angles, float)
        dirs = np.column_stack([np.zeros_like(a), -np.sin(a), -np.cos(a)])  # FLU body: toward body-right
        self.heights.add(pose, dirs, np.asarray(ranges, float))

    def on_range(self, rng: float, pose: PoseEstimate) -> None:
        self.heights.add(pose, np.array([[0.0, 0.0, -1.0]]), np.array([rng]))

    # ---- camera
    @property
    def degraded(self) -> bool:
        return self.contrast_ema is not None and self.contrast_ema < self.cfg.degraded_contrast * self.cfg.nominal_contrast

    @property
    def contrast_factor(self) -> float:
        if self.contrast_ema is None:
            return 1.0
        return float(np.clip(self.cfg.nominal_contrast / max(self.contrast_ema, 0.05), 1.0, 3.0))

    def on_frame(self, frame: CameraFrame, pose: PoseEstimate) -> FrameDetections:
        fd = self.detector.detect(frame, pose, self.area_center, self.cfg.area_half)
        self.last = fd
        if not fd.ok:
            return fd
        self.n_frames += 1
        if math.isfinite(fd.contrast):
            self.contrast_ema = fd.contrast if self.contrast_ema is None else 0.8 * self.contrast_ema + 0.2 * fd.contrast
        self._associate(fd)
        return fd

    def _associate(self, fd: FrameDetections) -> None:
        c = self.cfg
        t = fd.t
        live = [tr for tr in self.tracks if tr.status != "rejected"]
        pairs = []
        for bi, b in enumerate(fd.blobs):
            for tr in live:
                d = float(np.linalg.norm(b.xy - tr.xy))
                gate = c.track_gate_m + 0.15 * b.length
                if d < gate:
                    pairs.append((d, bi, tr))
        pairs.sort(key=lambda p: p[0])
        used_b, used_t = set(), set()
        for d, bi, tr in pairs:
            if bi in used_b or tr.id in used_t:
                continue
            used_b.add(bi)
            used_t.add(tr.id)
            self._observe(tr, fd.blobs[bi], t)
        for bi, b in enumerate(fd.blobs):
            if bi in used_b or b.merged:
                continue
            if np.any(np.abs(b.xy - self.area_center) > c.area_half + 1.0):
                continue
            tr = Track(self._next_id, b.xy.copy(), t, t)
            self._next_id += 1
            self.tracks.append(tr)
            self._observe(tr, b, t)
        # in view, but nothing detected: a false blob fades out
        for tr in live:
            if tr.id in used_t:
                continue
            if fd.in_view(tr.xy, 2.5):
                tr.n_miss += 1
                limit = 3 if tr.status == "tentative" else 8
                if tr.n_miss >= limit:
                    if tr.status == "candidate":
                        self.log(t, f"CANDIDATE {tr.id} dropped: in view {tr.n_miss} frames without a change blob")
                    tr.status = "rejected"
                    tr.reject_reason = "vanished"
        # duplicates (a track split by an occluder or seen twice): keep the better one
        cands = [tr for tr in self.tracks if tr.status != "rejected"]
        for i, a in enumerate(cands):
            for b in cands[i + 1:]:
                if a.status == "rejected" or b.status == "rejected":
                    continue
                if np.linalg.norm(a.xy - b.xy) < 1.5:
                    loser = a if a.n_obs < b.n_obs else b
                    loser.status, loser.reject_reason = "rejected", "duplicate"

    def _observe(self, tr: Track, b: Blob, t: float) -> None:
        tr.observe(b)
        tr.last_t = t
        if tr.status == "tentative" and tr.n_obs >= self.cfg.track_min_obs:
            tr.status = "candidate"
            tr.add_cue("change")
            L, W = tr.size()
            self.log(t, f"CANDIDATE {tr.id}: new object at ({tr.xy[0]:.1f}, {tr.xy[1]:.1f}), {L:.1f} x {W:.1f} m, "
                        f"{tr.color} [cue: change vs onboard map, {tr.n_obs} frames]")

    # ---- evidence
    def update_evidence(self, t: float) -> None:
        """Heights from the range returns, cue bookkeeping and class probabilities for all candidates."""
        c = self.cfg
        for tr in self.tracks:
            if tr.status != "candidate":
                continue
            L, W = tr.size(tr.height or 0.0)
            if not math.isfinite(L):
                last = tr.last_blob
                L, W = (last.length, last.width) if last is not None else (3.0, 2.0)
            h, n, spread = self.heights.footprint_height(tr.xy, L, W, tr.yaw, c.height_min_returns)
            if h is not None:
                first = tr.height is None
                tr.height, tr.height_n, tr.height_spread = h, n, spread
                if h < c.object_min_height:
                    tr.status, tr.reject_reason = "rejected", "flat"
                    self.log(t, f"CANDIDATE {tr.id} rejected: height {h:.1f} m from {n} range returns - a flat "
                                f"change (false blob), not an object [cue: height]")
                    continue
                if first and tr.add_cue("height"):
                    self.log(t, f"CANDIDATE {tr.id}: height {h:.1f} m from {n} range returns [cue: height]")
            tr.probs = class_probs(tr, self.contrast_factor)
            if tr.n_nadir >= 1 and tr.sw > 0:
                tr.add_cue("size")
                tr.add_cue("color")

    def candidates(self) -> list[Track]:
        return [tr for tr in self.tracks if tr.status == "candidate"]

    def to_messages(self, p_target: dict[int, float], chosen: int | None, rejected: set[int]) -> list[TargetCandidate]:
        out = []
        for tr in self.tracks:
            if tr.status == "tentative" or (tr.status == "rejected" and tr.id not in rejected):
                continue
            L, W = tr.size(tr.height or 0.0)
            status = "chosen" if tr.id == chosen else ("rejected" if tr.id in rejected or tr.status == "rejected" else "tracking")
            out.append(TargetCandidate(tr.id, tr.xy.copy(), float(L), float(W), tr.yaw, tr.height, tr.color,
                                       dict(tr.probs), float(p_target.get(tr.id, 0.0)), tr.n_obs, tr.n_nadir,
                                       list(tr.cues), status))
        return out


# ---------------------------------------------------------------------------------------- marker

class MarkerDetector:
    """The landing-station marker (OpenCV ArUco) in the down camera -> marker center in the EKF frame:
    solvePnP gives camera -> marker, the AHRS attitude rotates it into the world (tilt-corrected)."""

    def __init__(self, cfg: TargetConfig) -> None:
        self.cfg = cfg
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.marker_dict))
        self.det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
        s = cfg.marker_size / 2
        self.obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)

    def detect(self, frame: CameraFrame, pose: PoseEstimate) -> MarkerDetection | None:
        if frame.name != "down" or int(frame.image.max()) < 8:
            return None
        gray = cv2.cvtColor(frame.image, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.det.detectMarkers(gray)
        if ids is None:
            return None
        hit = [i for i, v in enumerate(ids.flatten()) if int(v) == self.cfg.marker_id]
        if not hit:
            return None
        pts = corners[hit[0]].reshape(4, 2).astype(np.float32)
        fx, fy, cx, cy = frame.intrinsics()
        K = np.array([[fx, 0, cx - 0.5], [0, fy, cy - 0.5], [0, 0, 1]], np.float64)
        ok, _rvec, tvec = cv2.solvePnP(self.obj, pts, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return None
        t = tvec.flatten()
        cam, R_wc = camera_pose(pose)
        pos = cam + R_wc @ np.array([t[0], -t[1], -t[2]])  # OpenCV camera (x right, y down, z ahead) -> MuJoCo
        return MarkerDetection(frame.t, frame.seq, pts, pos, float(np.linalg.norm(t)))
