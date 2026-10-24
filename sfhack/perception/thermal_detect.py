"""Warm-object detection in the downward thermal (LWIR) frame (recon mission: find vehicles / people at night).

Onboard: thermal frame + AHRS attitude + estimated height above the ground in, `Detection`s out. It never reads
the truth; the onboard DSM (optional) only tells whether a blob sits on a roof.

How (per frame, ~10 ms at 640 x 512):
1. Local background = robust local mean over a square window wider than any vehicle (`bg_kernel_m`, in pixels
   from the ground sample distance): box mean, then again without the pixels more than `bg_clip_K` above it
   (the objects themselves). Anomaly = frame - background, in K above the surroundings. Uncooled radiometry is
   not absolute (offset +-2 K, gain 3 %); the anomaly is a difference and does not care. A grey opening
   (top-hat) was tried first: at night paved roads are ~5 K warmer than the grass beside them and narrower than
   the kernel, so every road became a "warm object" and swallowed the cars on it.
2. Hysteresis: pixels above `t_low` K, components that reach `t_high` K somewhere.
3. Each component -> ground metres (gsd at the estimated height), minAreaRect length / width, mean / peak anomaly
   -> class:
     vehicle  length 3-12 m, width 1.2-4.5 m, area 3.5-45 m^2, mean anomaly >= vehicle_min_mean_K (a running or
              just-stopped engine warms the whole body: parked cars at night are BELOW air temperature, only a
              recently driven one has a 2 m^2 warm hood)
     person   area 0.08-0.9 m^2, length <= 1.3 m, peak >= person_min_peak_K (8 K: warm rocks peak lower)
     fire     tiny and > 60 K: rejected
     rejected hot spot (HVAC unit, a parked car's hood, a moving car's engine bay), elongated / blocky warm
              ground (rocks, roads), roof-sized (barn roof), or at a building (onboard DSM > DTM + 2.5 m within
              5 m: roof edges and walls seen obliquely)
4. Pixel centroid -> ray (camera -> body -> world with the AHRS attitude) -> intersect the ground plane under the
   drone: `xy_rel` = ENU offset (east, north) of the object from the drone, in metres.

`conf` in 0..1 grows with the anomaly and with how well the size fits the class.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

# camera axes in the body frame (same as perception/map_localizer.py: image top = forward, right = body right)
R_BODY_CAM = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class Detection:
    xy_rel: np.ndarray  # (2,) ENU offset object - drone [m] (east, north)
    size_m: tuple[float, float]  # (length, width) on the ground [m]
    temp_K: float  # peak anomaly above the local background [K]
    conf: float  # 0..1
    label: str = "vehicle"  # vehicle | person
    uv: tuple[float, float] = (0.0, 0.0)  # pixel centroid (full frame)
    area_m2: float = 0.0
    mean_K: float = 0.0  # mean anomaly over the blob [K]
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h in pixels


@dataclass
class Rejected:
    uv: tuple[float, float]
    reason: str


@dataclass
class DetectorParams:
    bg_kernel_m: float = 24.0  # background window (much wider than the longest vehicle)
    bg_clip_K: float = 2.0
    smooth_px: float = 1.0
    t_low: float = 4.0  # K above background: blob pixels (a warm road is ~3-4 K above its surroundings)
    t_high: float = 5.5  # K: a blob must reach this somewhere
    # a 3 m tall truck seen 30 deg off nadir also shows a side: its footprint grows by ~h * tan(30 deg)
    vehicle_len_m: tuple[float, float] = (3.0, 12.0)
    vehicle_wid_m: tuple[float, float] = (1.2, 4.5)
    vehicle_area_m2: tuple[float, float] = (3.5, 45.0)
    vehicle_min_mean_K: float = 5.0
    # people (seen from above ~0.5 x 0.3 m, +8.5 K body / +11 K head over the air) vs warm rocks (<= 1.8 m,
    # +4..5.5 K over the air, i.e. <= ~7 K over grass): the peak separates them
    person_area_m2: tuple[float, float] = (0.08, 0.9)
    person_max_len_m: float = 1.3
    person_min_peak_K: float = 8.0
    fire_peak_K: float = 60.0  # a campfire (+300 K) or a burning wreck: hot spot, never a person / vehicle body
    roof_margin_m: float = 2.5  # DSM above DTM by more than this = on a roof ...
    roof_radius_m: float = 5.0  # ... anywhere within this distance of the blob
    min_agl_m: float = 15.0
    max_tilt_deg: float = 30.0
    border_px: int = 4  # blobs touching the frame border are cut: no class decision


@dataclass
class DetectResult:
    detections: list[Detection] = field(default_factory=list)
    rejected: list[Rejected] = field(default_factory=list)
    gsd_m: float = 0.0
    ms: float = 0.0
    anomaly: np.ndarray | None = None  # top-hat image [K] (for the viewer)


class ThermalDetector:
    def __init__(self, params: DetectorParams | None = None, onboard_map: Any | None = None) -> None:
        self.p = params or DetectorParams()
        self.map = onboard_map  # OnboardMap (optional): roof check with its DSM / DTM

    def detect(self, img_K: np.ndarray, R: np.ndarray, agl_m: float, intrinsics: dict[str, float],
               pos_xy: np.ndarray | None = None, keep_anomaly: bool = False) -> DetectResult:
        """`R`: body -> world attitude (AHRS). `agl_m`: estimated camera height above the ground. `pos_xy`: the
        estimated drone position (only for the roof check against the onboard map)."""
        import time

        t0 = time.perf_counter()
        p = self.p
        out = DetectResult()
        R_wc = np.asarray(R, float) @ R_BODY_CAM
        axis = -R_wc[:, 2]
        if agl_m is None or not math.isfinite(agl_m) or agl_m < p.min_agl_m:
            return out
        if axis[2] > -math.cos(math.radians(p.max_tilt_deg)):
            return out
        fx, fy, cx, cy = (float(intrinsics[k]) for k in ("fx", "fy", "cx", "cy"))
        img = np.asarray(img_K, np.float32)
        h, w = img.shape
        dist = agl_m / -axis[2]  # slant range along the optical axis
        gsd = dist / fy
        out.gsd_m = gsd
        k = int(np.clip(round(p.bg_kernel_m / gsd), 5, min(h, w) // 2)) | 1
        sm = cv2.GaussianBlur(img, (0, 0), p.smooth_px) if p.smooth_px > 0 else img
        bg = cv2.blur(sm, (k, k), borderType=cv2.BORDER_REFLECT)
        keep = (sm - bg < p.bg_clip_K).astype(np.float32)
        bg = cv2.blur(sm * keep, (k, k), borderType=cv2.BORDER_REFLECT) / np.maximum(
            cv2.blur(keep, (k, k), borderType=cv2.BORDER_REFLECT), 1e-3)
        an = sm - bg
        if keep_anomaly:
            out.anomaly = an
        mask = (an > p.t_low).astype(np.uint8)
        n, lab, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        px_area = gsd * gsd
        for i in range(1, n):
            x0, y0, bw, bh, npx = (int(v) for v in stats[i])
            area = npx * px_area
            if area < 0.5 * p.person_area_m2[0]:
                continue  # noise
            sub = lab[y0:y0 + bh, x0:x0 + bw] == i
            vals = an[y0:y0 + bh, x0:x0 + bw][sub]
            peak = float(vals.max())
            if peak < p.t_high:
                continue
            uv = (float(cents[i][0]), float(cents[i][1]))
            b = p.border_px
            if x0 <= b or y0 <= b or x0 + bw >= w - b or y0 + bh >= h - b:
                out.rejected.append(Rejected(uv, "border"))
                continue
            mean = float(vals.mean())
            ys, xs = np.nonzero(sub)
            (_, _), (rw, rh), _ = cv2.minAreaRect(np.stack([xs + x0, ys + y0], -1).astype(np.float32))
            length, width = max(rw, rh) * gsd + gsd, min(rw, rh) * gsd + gsd  # +1 px: rect through pixel centres
            label, conf, why = self._classify(area, length, width, mean, peak)
            if label is None:
                out.rejected.append(Rejected(uv, why))
                continue
            rel = self._ground_offset(uv, R_wc, agl_m, fx, fy, cx, cy)
            if rel is None:
                continue
            if self.map is not None and pos_xy is not None and self.map.dsm is not None:
                # on or right beside a building: roof edges, HVAC units and walls seen obliquely look like warm,
                # vehicle-sized blobs; the projected point of a wall lands just outside the footprint
                q = np.asarray(pos_xy, float)[:2] + rel
                r = p.roof_radius_m
                ring = q + np.array([[0, 0], [r, 0], [-r, 0], [0, r], [0, -r], [r, r], [-r, -r], [r, -r], [-r, r]]) * 0.75
                if float(np.max(self.map.surface_heights(ring) - np.array([self.map.ground_z(x) for x in ring]))) \
                        > p.roof_margin_m:
                    out.rejected.append(Rejected(uv, "at a building"))
                    continue
            out.detections.append(Detection(rel, (round(length, 2), round(width, 2)), round(peak, 2), round(conf, 3),
                                            label, uv, round(area, 2), round(mean, 2), (x0, y0, bw, bh)))
        out.ms = (time.perf_counter() - t0) * 1000.0
        return out

    def _classify(self, area: float, length: float, width: float, mean: float,
                  peak: float) -> tuple[str | None, float, str]:
        p = self.p
        if peak > p.fire_peak_K and area < p.vehicle_area_m2[0]:
            return None, 0.0, f"fire ({peak:.0f} K)"
        if p.person_area_m2[0] <= area <= p.person_area_m2[1] and length <= p.person_max_len_m:
            if peak < p.person_min_peak_K:
                return None, 0.0, "small, too cold"
            conf = _ramp(peak, p.person_min_peak_K, p.person_min_peak_K + 4.0) * 0.8 + 0.2 * _ramp(area, 0.12, 0.3)
            return "person", conf, ""
        if area < p.vehicle_area_m2[0]:
            return None, 0.0, f"hot spot {area:.1f} m2"
        if area > p.vehicle_area_m2[1] or length > p.vehicle_len_m[1]:
            return None, 0.0, f"too large {length:.0f} x {width:.0f} m"
        if not (p.vehicle_len_m[0] <= length and p.vehicle_wid_m[0] <= width <= p.vehicle_wid_m[1]):
            return None, 0.0, f"shape {length:.1f} x {width:.1f} m"
        if mean < p.vehicle_min_mean_K:
            return None, 0.0, f"vehicle-sized but cool ({mean:.1f} K)"
        fill = area / max(length * width, 1e-6)  # a vehicle fills its rectangle
        conf = (0.6 * _ramp(mean, p.vehicle_min_mean_K, p.vehicle_min_mean_K + 5.0) + 0.2 * _ramp(fill, 0.5, 0.8)
                + 0.2 * _ramp(length / max(width, 1e-6), 1.2, 2.0))
        return "vehicle", conf, ""

    @staticmethod
    def _ground_offset(uv: tuple[float, float], R_wc: np.ndarray, agl: float, fx: float, fy: float, cx: float,
                       cy: float) -> np.ndarray | None:
        pc = np.array([(uv[0] + 0.5 - cx) / fx, -(uv[1] + 0.5 - cy) / fy, -1.0])
        d = R_wc @ pc
        if d[2] > -0.1:
            return None
        return d[:2] * (agl / -d[2])


def _ramp(x: float, lo: float, hi: float) -> float:
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def to_8bit(img_K: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5, min_span_K: float = 4.0) -> np.ndarray:
    """White-hot 8-bit view of a thermal frame (percentile stretch, at least min_span_K wide)."""
    lo, hi = np.percentile(img_K[::4, ::4], [lo_pct, hi_pct])
    if hi - lo < min_span_K:
        mid = (hi + lo) / 2
        lo, hi = mid - min_span_K / 2, mid + min_span_K / 2
    return np.clip((img_K - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
