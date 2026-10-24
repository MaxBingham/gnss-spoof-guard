"""Enemy-drone detection and tracking on the gimbal camera (onboard).

- `DroneDetector`: YOLO nano (Ultralytics, AGPL-3.0, imported lazily) on one RGB frame -> boxes with class
  ("drone" / "bird") and confidence. Weights trained offline on synthetic farm-world renders (demos/53-55); this
  module only loads them from $SFHACK_DATA (`TrackerConfig.weights`).
- Geometry: a pixel -> world ray from the drone's HEADING ESTIMATE (AHRS) plus the gimbal encoder readings in the
  frame (pan relative to the heading, tilt up from the horizon, level horizon); range from the apparent size of the
  box and the nominal enemy size (`target_size_m`, the long side of the box spans ~`apparent_frac` of it).
  Enemy position = own position ESTIMATE + range * ray: with GNSS off it drifts with our own estimate.
- `EnemyTrack`: constant-velocity Kalman filter of the enemy position in the world (estimate frame), measurement
  covariance long along the ray (size-based range) and narrow across it (bearing).
- `EnemyTracker`: per frame picks the drone detection that fits the track (gate), keeps a hit window and says when
  the enemy is confirmed (`confirm_hits` of the last `confirm_window` frames).

Inputs at runtime: the frame, the pose estimate. No truth: tests/test_truth_separation.py.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import TrackerConfig, data_dir

CLASSES = ("drone", "bird")
APPARENT_FRAC = 0.93  # mean long side of the enemy's box / its tip-to-tip size (side and oblique views), from the
# dataset truth boxes (0.95 at 15-80 m, 0.87 at 80-150 m: edge pixels fade); demos/55_drone_eval.py checks the detector boxes


@dataclass
class Det:
    cls: str
    conf: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 px


@dataclass
class Measurement:
    det: Det
    ray: np.ndarray  # (3,) world unit vector from the camera
    range_m: float
    pos: np.ndarray  # (3,) world (estimate frame)
    cov: np.ndarray  # (3, 3)


def camera_ray(u: float, v: float, w: int, h: int, hfov_deg: float, yaw: float, pan_deg: float,
               tilt_deg: float) -> np.ndarray:
    """World unit vector through pixel (u, v): optical axis at heading yaw (rad, ENU) + pan, tilt up from the horizon,
    level horizon, square pixels, principal point at the centre."""
    a, t = yaw + math.radians(pan_deg), math.radians(tilt_deg)
    f = np.array([math.cos(a) * math.cos(t), math.sin(a) * math.cos(t), math.sin(t)])
    right = np.array([math.sin(a), -math.cos(a), 0.0])  # f x z, normalized (|f x z| = cos t)
    up = np.cross(right, f)
    fpx = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
    d = f + (u - w / 2) / fpx * right - (v - h / 2) / fpx * up
    return d / np.linalg.norm(d)


def pointing(rel: np.ndarray, yaw: float) -> tuple[float, float]:
    """Gimbal pan / tilt (deg) that put the world vector `rel` on the optical axis, with the heading `yaw`."""
    pan = math.degrees(math.atan2(rel[1], rel[0]) - yaw)
    pan = (pan + 180.0) % 360.0 - 180.0
    tilt = math.degrees(math.atan2(rel[2], math.hypot(rel[0], rel[1])))
    return pan, tilt


def range_from_box(box, cfg: TrackerConfig) -> float:
    side = max(box[2] - box[0], box[3] - box[1], 1.0)
    return cfg.focal_px() * cfg.target_size_m * APPARENT_FRAC / side


def measure(det: Det, cfg: TrackerConfig, cam_pos: np.ndarray, yaw: float, pan_deg: float,
            tilt_deg: float) -> Measurement:
    u, v = (det.box[0] + det.box[2]) / 2, (det.box[1] + det.box[3]) / 2
    ray = camera_ray(u, v, cfg.cam_w, cfg.cam_h, cfg.hfov_deg, yaw, pan_deg, tilt_deg)
    r = range_from_box(det.box, cfg)
    side = max(det.box[2] - det.box[0], det.box[3] - det.box[1], 1.0)
    s_r = r * math.hypot(cfg.size_sigma_frac, 1.0 / side)  # aspect + one pixel of box quantization
    s_c = max(r * math.radians(cfg.bearing_sigma_deg), 0.3)
    P = np.outer(ray, ray)
    cov = s_r ** 2 * P + s_c ** 2 * (np.eye(3) - P)
    return Measurement(det, ray, r, np.asarray(cam_pos, float) + r * ray, cov)


class DroneDetector:
    def __init__(self, cfg: TrackerConfig | None = None, weights: str | Path | None = None,
                 device: str | None = None) -> None:
        from ultralytics import YOLO  # lazy: AGPL-3.0 dependency, optional extra 'detector'

        self.cfg = cfg or TrackerConfig()
        w = Path(weights or self.cfg.weights)
        if not w.is_absolute():
            w = data_dir() / w
        if not w.exists():
            raise FileNotFoundError(f"detector weights not found: {w} (train them: demos/54_drone_train.py)")
        self.weights = w
        self.model = YOLO(str(w))
        if device is None:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        self.device = device
        names = self.model.names
        self.names = [names[i] for i in range(len(names))]
        self.last_ms = 0.0

    def detect(self, rgb: np.ndarray) -> list[Det]:
        t0 = time.perf_counter()
        bgr = np.ascontiguousarray(rgb[..., ::-1])  # ultralytics reads numpy frames as BGR
        r = self.model.predict(bgr, imgsz=self.cfg.imgsz, conf=self.cfg.conf_min, iou=self.cfg.iou_nms,
                               device=self.device, verbose=False)[0]
        self.last_ms = 1000.0 * (time.perf_counter() - t0)
        b = r.boxes
        boxes, conf, cls = b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)
        return [Det(self.names[c], float(p), tuple(float(x) for x in bb)) for bb, p, c in zip(boxes, conf, cls)]


class EnemyTrack:
    """Constant-velocity KF, state [x, y, z, vx, vy, vz] in the world (estimate frame)."""

    def __init__(self, m: Measurement, t: float, accel_sigma: float = 1.5) -> None:
        self.x = np.r_[m.pos, np.zeros(3)]
        self.P = np.zeros((6, 6))
        self.P[:3, :3] = m.cov
        self.P[3:, 3:] = np.eye(3) * 6.0 ** 2
        self.t = t
        self.q = accel_sigma ** 2
        self.n_updates = 1
        self.last_update_t = t

    def predict(self, t: float) -> None:
        dt = t - self.t
        if dt <= 0:
            return
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        G = np.r_[np.eye(3) * 0.5 * dt * dt, np.eye(3) * dt]
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.q * G @ G.T
        self.t = t

    def pos_at(self, t: float) -> np.ndarray:
        return self.x[:3] + self.x[3:] * max(t - self.t, 0.0)

    def nis(self, m: Measurement) -> float:
        S = self.P[:3, :3] + m.cov
        e = m.pos - self.x[:3]
        return float(e @ np.linalg.solve(S, e))

    def update(self, m: Measurement) -> None:
        H = np.zeros((3, 6))
        H[:, :3] = np.eye(3)
        S = self.P[:3, :3] + m.cov
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (m.pos - self.x[:3])
        self.P = (np.eye(6) - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        self.n_updates += 1
        self.last_update_t = self.t

    @property
    def pos(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def vel(self) -> np.ndarray:
        return self.x[3:].copy()

    def sigma_h(self) -> float:
        """1-sigma horizontal position uncertainty (m): sqrt of the larger eigenvalue of the xy block."""
        return float(math.sqrt(max(np.linalg.eigvalsh(self.P[:2, :2]).max(), 0.0)))


@dataclass
class FrameResult:
    t: float
    dets: list[Det]  # everything the network returned (>= conf_min)
    hit: Measurement | None  # the drone detection associated with the track this frame
    new_track: bool = False
    others: list[Measurement] = field(default_factory=list)  # drone detections outside the gate


class EnemyTracker:
    """Hit window + Kalman track. `process(frame dets, own pose)` every frame; `confirmed` once enough hits."""

    def __init__(self, cfg: TrackerConfig | None = None) -> None:
        self.cfg = cfg or TrackerConfig()
        self.track: EnemyTrack | None = None
        self.window: deque[int] = deque(maxlen=self.cfg.confirm_window)
        self.last_hit_t: float | None = None
        self.best_conf = 0.0
        self.last_conf = 0.0

    @property
    def hits(self) -> int:
        return int(sum(self.window))

    @property
    def confirmed(self) -> bool:
        return self.track is not None and self.hits >= self.cfg.confirm_hits

    def reset(self) -> None:
        self.track = None
        self.window.clear()
        self.last_hit_t = None
        self.best_conf = self.last_conf = 0.0

    def process(self, t: float, dets: list[Det], cam_pos: np.ndarray, yaw: float, pan_deg: float,
                tilt_deg: float, accept: Callable[[Measurement], bool] | None = None) -> FrameResult:
        """One frame. `accept(m)`: may a NEW track start from this measurement (e.g. the acoustic bearing gate)."""
        cfg = self.cfg
        ms = [measure(d, cfg, cam_pos, yaw, pan_deg, tilt_deg) for d in dets
              if d.cls == "drone" and d.conf >= cfg.drone_conf]
        res = FrameResult(t, dets, None)
        if self.track is not None:
            self.track.predict(t)
            pred = self.track.pos
            fits = [m for m in ms if np.linalg.norm(m.pos - pred) < cfg.gate_m + 3.0 * self.track.sigma_h()]
            if fits:
                best = min(fits, key=lambda m: self.track.nis(m) - 4.0 * m.det.conf)
                self.track.update(best)
                res.hit = best
            res.others = [m for m in ms if m is not res.hit]
        elif ms and (accept is None or any(accept(m) for m in ms)):
            best = max((m for m in ms if accept is None or accept(m)), key=lambda m: m.det.conf)
            self.track = EnemyTrack(best, t)
            self.window.clear()
            res.hit, res.new_track = best, True
            res.others = [m for m in ms if m is not best]
        else:
            res.others = ms
        self.window.append(1 if res.hit is not None else 0)
        if res.hit is not None:
            self.last_hit_t = t
            self.last_conf = res.hit.det.conf
            self.best_conf = max(self.best_conf, self.last_conf)
        return res

    def confidence(self) -> float:
        """Report confidence: detector confidence of the last hit times the hit rate of the window."""
        if self.track is None:
            return 0.0
        return float(self.last_conf * self.hits / max(len(self.window), 1))
