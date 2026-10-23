"""V4 learned truck detector (onboard): YOLO on one RGB down-camera frame -> boxes with class, calibrated
confidence and the ground position of each object relative to the drone. Plus `TruckSelector`, which
accumulates detections over a burst of frames and picks the truck with a margin rule.

Inputs at runtime: the RGB frame, the altitude ESTIMATE (EKF height above the flat ground) and the attitude
ESTIMATE (AHRS, body -> world). No truth, no labels, no layout: the weights were trained offline on synthetic
auto-labeled renders (demos/13-15); this module only loads them ($SFHACK_DATA, `DetectorConfig.weights`).

Ultralytics (AGPL-3.0) and torch are imported lazily, so the rest of the package does not depend on them.

Ground position: the box center is a ray through the camera (pinhole, square pixels, principal point at the
image center, `DetectorConfig.fovy_deg`), intersected with the plane at half the typical height of the
detected class (a roof h above ground at camera height H would otherwise project h / (H - h) too far out).
`ground_xy` is the (east, north) offset from the drone in meters; add the EKF position for world coordinates.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from ..config import DetectorConfig, data_dir
from .map_localizer import R_BODY_CAM


@dataclass
class Detection:
    cls: str  # class name (DetectorConfig.classes)
    conf: float  # calibrated confidence (temperature scaling)
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels of the input frame
    ground_xy: np.ndarray  # (2,) east, north offset of the object from the drone [m]
    raw_conf: float = 0.0  # network confidence before calibration


def calibrate(conf: np.ndarray | float, temperature: float) -> np.ndarray:
    """Temperature scaling of a sigmoid confidence: sigmoid(logit(p) / T)."""
    p = np.clip(np.asarray(conf, float), 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-np.log(p / (1 - p)) / temperature))


def ground_offsets(centers_px: np.ndarray, heights: np.ndarray, width: int, height: int, fovy_deg: float,
                   altitude: float, attitude: np.ndarray) -> np.ndarray:
    """(N, 2) east / north offsets from the camera of pixel rays hitting the planes z = heights (above ground)."""
    fy = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    c = np.asarray(centers_px, float).reshape(-1, 2)
    d_cam = np.column_stack([(c[:, 0] - width / 2.0) / fy, -(c[:, 1] - height / 2.0) / fy, -np.ones(len(c))])
    d_w = d_cam @ (np.asarray(attitude, float) @ R_BODY_CAM).T
    drop = np.maximum(altitude - np.asarray(heights, float), 0.1)
    t = drop / np.maximum(-d_w[:, 2], 1e-3)  # rays at or above the horizon: clipped far out
    return d_w[:, :2] * t[:, None]


class TruckDetector:
    def __init__(self, cfg: DetectorConfig | None = None, weights: str | Path | None = None,
                 device: str | None = None) -> None:
        from ultralytics import YOLO  # lazy: AGPL-3.0 dependency, optional extra 'detector'

        self.cfg = cfg or DetectorConfig()
        w = Path(weights or self.cfg.weights)
        if not w.is_absolute():
            w = data_dir() / w
        if not w.exists():
            raise FileNotFoundError(f"detector weights not found: {w} (train them: demos/14_det_train.py)")
        self.weights = w
        # Operating point chosen on the validation split (demos/15_det_eval.py) travels with the weights
        cal = w.parent.parent / "calibration.json"
        if cfg is None and cal.exists():
            c = json.loads(cal.read_text(encoding="utf-8"))
            self.cfg = replace(self.cfg, temperature=float(c["temperature"]), truck_threshold=float(c["truck_threshold"]))
        self.model = YOLO(str(w))
        if device is None:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        self.device = device
        names = self.model.names
        self.names = [names[i] for i in range(len(names))]
        self.last_ms = 0.0

    def predict_raw(self, rgb: np.ndarray, conf_min: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(boxes (N, 4) xyxy px, raw conf (N,), class index (N,)) for one RGB frame."""
        bgr = np.ascontiguousarray(rgb[..., ::-1])  # ultralytics reads numpy frames as BGR (OpenCV order)
        r = self.model.predict(bgr, imgsz=self.cfg.imgsz, conf=self.cfg.conf_min if conf_min is None else conf_min,
                               iou=self.cfg.iou_nms, device=self.device, verbose=False)[0]
        b = r.boxes
        return b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)

    def detect(self, rgb: np.ndarray, altitude: float, attitude: np.ndarray) -> list[Detection]:
        """One frame -> detections. altitude: camera height above the ground (estimate) [m]; attitude: (3, 3)
        body -> world rotation (estimate)."""
        t0 = time.perf_counter()
        boxes, conf, cls = self.predict_raw(rgb)
        self.last_ms = 1000.0 * (time.perf_counter() - t0)
        if len(conf) == 0:
            return []
        h, w = rgb.shape[:2]
        heights = np.array([0.5 * self.cfg.class_heights[c] if c < len(self.cfg.class_heights) else 1.0 for c in cls])
        centers = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2])
        gxy = ground_offsets(centers, heights, w, h, self.cfg.fovy_deg, float(altitude), attitude)
        cal = calibrate(conf, self.cfg.temperature)
        return [Detection(self.names[c], float(p), tuple(float(v) for v in b), g, float(r))
                for b, p, r, c, g in zip(boxes, cal, conf, cls, gxy)]


_default: TruckDetector | None = None


def detect(rgb: np.ndarray, altitude: float, attitude: np.ndarray) -> list[Detection]:
    """Module-level convenience: the default detector (weights from DetectorConfig), created on first use."""
    global _default
    if _default is None:
        _default = TruckDetector()
    return _default.detect(rgb, altitude, attitude)


# ---------------------------------------------------------------------------------------- selection

@dataclass
class Cluster:
    xy: np.ndarray  # (2,) world, running mean of the member detections
    n: int = 0  # detections (any class)
    frames: set[int] = field(default_factory=set)
    truck_sum: float = 0.0  # sum over frames of the best truck confidence in this cluster
    truck_n: int = 0  # frames in which this cluster had a truck-class detection
    classes: dict[str, float] = field(default_factory=dict)  # summed confidence per class

    @property
    def truck_score(self) -> float:
        """Mean truck evidence per frame the object was detected in (0 for frames it was another class)."""
        return self.truck_sum / max(len(self.frames), 1)


@dataclass
class Selection:
    xy: np.ndarray | None  # world position of the chosen object (None = no decision)
    score: float
    runner_up: float
    reason: str
    clusters: list[Cluster]


class TruckSelector:
    """Accumulate detections over a burst of frames (world ground positions), then choose the truck: the
    cluster with the best mean truck score, seen as truck in >= select_min_frames frames, beating the
    runner-up by select_margin. Otherwise no decision (the caller looks again / lower)."""

    def __init__(self, cfg: DetectorConfig | None = None) -> None:
        self.cfg = cfg or DetectorConfig()
        self.clusters: list[Cluster] = []
        self.frame = 0

    def add(self, detections: list[Detection], drone_xy: np.ndarray) -> None:
        self.frame += 1
        best_truck: dict[int, float] = {}
        for d in detections:
            p = np.asarray(drone_xy, float)[:2] + d.ground_xy
            k = self._assign(p)
            c = self.clusters[k]
            c.n += 1
            c.xy = c.xy + (p - c.xy) / c.n
            c.frames.add(self.frame)
            c.classes[d.cls] = c.classes.get(d.cls, 0.0) + d.conf
            if d.cls == self.cfg.target_class:
                best_truck[k] = max(best_truck.get(k, 0.0), d.conf)
        for k, s in best_truck.items():
            self.clusters[k].truck_sum += s
            self.clusters[k].truck_n += 1

    def _assign(self, p: np.ndarray) -> int:
        if self.clusters:
            d = [float(np.linalg.norm(c.xy - p)) for c in self.clusters]
            i = int(np.argmin(d))
            if d[i] < self.cfg.select_cluster_m:
                return i
        self.clusters.append(Cluster(p.copy()))
        return len(self.clusters) - 1

    def decide(self) -> Selection:
        ranked = sorted(self.clusters, key=lambda c: c.truck_score, reverse=True)
        cands = [c for c in ranked if c.truck_n >= self.cfg.select_min_frames]
        if not cands:
            return Selection(None, 0.0, 0.0, "no truck seen often enough", ranked)
        best = cands[0]
        second = max((c.truck_score for c in ranked if c is not best), default=0.0)
        if best.truck_score - second < self.cfg.select_margin:
            return Selection(None, best.truck_score, second, "margin too small", ranked)
        return Selection(best.xy.copy(), best.truck_score, second, "selected", ranked)


def describe(dets: list[Detection]) -> str:
    return ", ".join(f"{d.cls} {d.conf:.2f} @({d.ground_xy[0]:+.1f}, {d.ground_xy[1]:+.1f})" for d in dets)


__all__ = ["Detection", "TruckDetector", "detect", "TruckSelector", "Selection", "calibrate", "ground_offsets"]
