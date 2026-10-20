"""Localizer interface: down-camera frame + onboard map + prior -> VisualFix (or None).

Implementations:
  - MockLocalizer   (mock_localizer.py): MOCK, truth + noise when >= 3 landmarks are in view.
  - OrbMapLocalizer (orb_localizer.py):  real map matching (ORB + RANSAC homography).
They live in separate files so that only the mock module touches the truth.
"""

from __future__ import annotations

from typing import Protocol

from ..messages import CameraFrame, PoseEstimate, VisualFix
from ..world.onboard_map import OnboardMap


class Localizer(Protocol):
    name: str
    is_mock: bool

    def update(self, frame: CameraFrame, onboard_map: OnboardMap, prior: PoseEstimate) -> VisualFix | None: ...
