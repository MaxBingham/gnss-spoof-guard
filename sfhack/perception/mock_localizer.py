"""MOCK localizer: true camera position + Gaussian noise when enough landmarks are in view.

# SIMULATION OF PERCEPTION - uses ground truth
This is the only perception module allowed to read the truth (see sfhack/sim/truth.py). It stands in
for a real map-matching localizer: it decides "would a real localizer see enough here?" from the
landmarks inside the true down-camera footprint, and if so returns a noisy fix.
"""

from __future__ import annotations

import numpy as np

from ..config import LocalizerConfig
from ..messages import CameraFrame, PoseEstimate, VisualFix
from ..sim.truth import TruthProvider  # SIMULATION OF PERCEPTION - uses ground truth
from ..world.onboard_map import OnboardMap


class MockLocalizer:
    name = "mock"
    is_mock = True

    def __init__(self, cfg: LocalizerConfig, truth: TruthProvider, seed: int = 0) -> None:
        self.cfg = cfg
        self._truth = truth
        self._rng = np.random.default_rng(seed + 1000)

    def update(self, frame: CameraFrame, onboard_map: OnboardMap, prior: PoseEstimate) -> VisualFix | None:
        if frame.name != "down" or not frame.image.any():  # black frame = camera failure
            return None
        pose = self._truth.camera_pose(frame.seq)  # SIMULATION OF PERCEPTION - uses ground truth
        if pose is None:
            return None
        cam_pos, cam_R = pose  # camera -> world; MuJoCo cameras look along -z, x right, y up
        fx, fy, cx, cy = frame.intrinsics()
        pts = np.column_stack([onboard_map.lm_xy, onboard_map.lm_height])
        pc = (pts - cam_pos) @ cam_R  # world -> camera frame
        depth = -pc[:, 2]
        ok = depth > 1.0
        u = cx + fx * pc[:, 0] / np.where(ok, depth, 1.0)
        v = cy - fy * pc[:, 1] / np.where(ok, depth, 1.0)
        ok &= (u >= 0) & (u < frame.width) & (v >= 0) & (v < frame.height)
        idx = np.flatnonzero(ok)
        if len(idx) < self.cfg.mock_min_landmarks:
            return None
        s = self.cfg.mock_sigma
        pos = cam_pos[:2] + self._rng.normal(0.0, s, 2)
        return VisualFix(frame.t, pos, np.eye(2) * s * s, "mock",
                         matched_ids=[int(onboard_map.lm_ids[i]) for i in idx],
                         matched_px=[(float(u[i]), float(v[i])) for i in idx],
                         n_inliers=len(idx), frame_seq=frame.seq)
