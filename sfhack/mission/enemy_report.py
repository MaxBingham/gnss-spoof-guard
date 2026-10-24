"""Reports from our drone to the base about the enemy drone (onboard side: built and buffered here).

The radio link is up only part of the time (~80 %); reports wait in the `ReportOutbox` while it is down and go out
in order when it returns (the link model is simulated reality: sfhack/sim/link.py, driven by the harness).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EnemyReport:
    t: float  # sim time the report was made (onboard clock)
    seq: int
    state: str  # tracker state: DETECTED / TRACKING / LOST
    pos: np.ndarray  # (3,) enemy position, world ENU (map point x east, y north from the UMA-16; z on the datum)
    agl: float  # m above the terrain map
    speed: float  # m/s horizontal
    heading_deg: float  # compass degrees of the enemy's horizontal velocity
    climb: float  # m/s
    confidence: float  # 0..1 (detector confidence x hit rate)
    sigma_m: float  # 1-sigma horizontal position uncertainty of the track
    range_m: float  # from our drone (estimate)
    own_pos: np.ndarray  # (3,) our drone's position estimate when the report was made
    snapshot: np.ndarray | None = None  # (s, s, 3) uint8 RGB crop around the detection
    snapshot_jpg: bytes = b""  # the same, JPEG (what would go over the radio)

    def bearing_from(self, xy=(0.0, 0.0)) -> float:
        d = self.pos[:2] - np.asarray(xy, float)
        return math.degrees(math.atan2(d[0], d[1])) % 360.0

    def text(self) -> str:
        return (f"#{self.seq} t {self.t:6.1f}s {self.state:8s} enemy at ({self.pos[0]:7.1f}, {self.pos[1]:7.1f}) "
                f"{self.agl:4.0f} m AGL  {self.speed:4.1f} m/s hdg {self.heading_deg:03.0f}  conf {self.confidence:.2f}"
                f"  +-{self.sigma_m:.1f} m")


def snapshot(rgb: np.ndarray, box, size: int = 96) -> tuple[np.ndarray, bytes]:
    """Square crop around the box (2x its long side, at least 32 px), resized to size x size, and its JPEG."""
    h, w = rgb.shape[:2]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    half = max(16.0, max(box[2] - box[0], box[3] - box[1]))
    x0, x1 = int(max(0, cx - half)), int(min(w, cx + half))
    y0, y1 = int(max(0, cy - half)), int(min(h, cy + half))
    crop = rgb[y0:y1, x0:x1]
    if crop.size == 0:
        crop = np.zeros((size, size, 3), np.uint8)
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_NEAREST if half < size / 2 else cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return crop, (jpg.tobytes() if ok else b"")


class ReportOutbox:
    """FIFO of reports not yet sent. The link drains it while it is up."""

    def __init__(self, max_len: int = 2000) -> None:
        self.queue: list[EnemyReport] = []
        self.max_len = max_len
        self.made = 0
        self.dropped = 0

    def put(self, r: EnemyReport) -> None:
        self.made += 1
        self.queue.append(r)
        if len(self.queue) > self.max_len:  # oldest first out: the base wants the latest picture
            self.queue.pop(0)
            self.dropped += 1

    def drain(self) -> list[EnemyReport]:
        out, self.queue = self.queue, []
        return out

    def __len__(self) -> int:
        return len(self.queue)
