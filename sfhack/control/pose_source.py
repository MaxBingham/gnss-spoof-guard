"""Selectable pose source for the controller.

- `EkfPoseSource` (here): the onboard estimate (EKF position/velocity + AHRS attitude).
- `TruthPoseSource` (sfhack/sim/truth.py): TRUE state, for controller tuning and the dashboard's
  "truth" switch position. It lives in sim/ so that nothing in control/ can reach the truth.
"""

from __future__ import annotations

from typing import Protocol

from ..messages import PoseEstimate


class PoseSource(Protocol):
    name: str

    def get(self) -> PoseEstimate: ...


class EkfPoseSource:
    name = "ekf"

    def __init__(self, estimator: "HasPose") -> None:
        self._est = estimator

    def get(self) -> PoseEstimate:
        return self._est.pose()


class HasPose(Protocol):
    def pose(self) -> PoseEstimate: ...
