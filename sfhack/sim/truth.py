"""Ground truth access. THE ONLY door to the true state.

Allowed importers (enforced by tests/test_truth_separation.py):
  - sfhack/sim/*                        (the simulator itself)
  - sfhack/perception/mock_localizer.py (# SIMULATION OF PERCEPTION - uses ground truth)
  - harness code that only DISPLAYS or SCORES truth and never feeds it back:
    sfhack/runner.py, sfhack/server.py, sfhack/recorder.py, eval/, demos/, tests/

`TruthPoseSource` is the one sanctioned way to feed truth to the controller: it is the "truth"
position of the dashboard's pose-source switch and is used for controller tuning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..messages import PoseEstimate
from .mujoco_sim import MujocoSim, yaw_from_R


@dataclass
class TruthState:
    t: float
    pos: np.ndarray
    vel: np.ndarray  # world frame
    R: np.ndarray  # body -> world
    omega: np.ndarray  # body frame

    @property
    def yaw(self) -> float:
        return yaw_from_R(self.R)


class TruthProvider:
    def __init__(self, sim: MujocoSim) -> None:
        self._sim = sim

    def state(self) -> TruthState:
        return TruthState(*self._sim._truth_state())

    def camera_pose(self, frame_seq: int) -> tuple[np.ndarray, np.ndarray] | None:
        """True camera position and orientation (camera -> world) at the time a frame was rendered."""
        return self._sim._truth_camera_pose(frame_seq)

    def in_contact(self) -> bool:
        return self._sim._in_contact()


class TruthPoseSource:
    """TRUTH pose source: feeds the TRUE state to the controller (tuning / 'truth' switch only)."""

    name = "truth"

    def __init__(self, provider: TruthProvider) -> None:
        self._provider = provider

    def get(self) -> PoseEstimate:
        s = self._provider.state()
        return PoseEstimate(
            t=s.t, pos=s.pos, vel=s.vel, yaw=s.yaw, cov=np.eye(6) * 1e-4, R=s.R, omega=s.omega, source="truth"
        )
