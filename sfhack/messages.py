"""Typed messages exchanged over the bus. Every message carries `t` = simulation time [s].

Frames: world = local ENU (x east, y north, z up). Body = FLU (x forward, y left, z up).
Nothing in here contains ground truth; truth lives in `sfhack.sim.truth` only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


@dataclass
class ImuMsg:
    t: float
    gyro: np.ndarray  # (3,) rad/s, body frame
    accel: np.ndarray  # (3,) m/s^2, body frame specific force (reads +g on z at rest)


@dataclass
class BaroMsg:
    t: float
    alt: float  # m above the takeoff ground level


@dataclass
class RangeMsg:
    """Downward rangefinder: distance along the body -z axis to the first surface (ground, roof, tree)."""

    t: float
    range: float  # m; nan = nothing within the maximum range


@dataclass
class MagMsg:
    """Compass heading, already converted to an ENU yaw angle (0 = east, CCW positive)."""

    t: float
    yaw: float


@dataclass
class GnssMsg:
    """GNSS fix (only while the receiver is on, see sfhack.degradation). A jammed receiver publishes nothing."""

    t: float
    pos: np.ndarray  # (3,) ENU
    cov: np.ndarray = field(default_factory=lambda: np.eye(2) * 2.25)  # (2, 2) horizontal, receiver-reported


@dataclass
class LidarMsg:
    """Nadir multi-beam lidar: a cross-track fan of ranges to the first surface (ground, roof, wall)."""

    t: float
    angles: np.ndarray  # (n,) rad from straight down, positive toward body-right
    ranges: np.ndarray  # (n,) m; nan = no return


@dataclass
class CameraFrame:
    t: float
    seq: int
    name: str  # "down" or "front"
    image: np.ndarray  # (H, W, 3) uint8 RGB, row 0 = top
    fovy_deg: float

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    def intrinsics(self) -> tuple[float, float, float, float]:
        """(fx, fy, cx, cy) in pixels. Square pixels, principal point at the image center."""
        fy = (self.height / 2.0) / np.tan(np.radians(self.fovy_deg) / 2.0)
        return fy, fy, self.width / 2.0, self.height / 2.0


@dataclass
class ThermalFrame:
    """Downward LWIR frame. Orientation as the MuJoCo "down" camera: image top = forward, right = body right."""

    t: float
    seq: int
    image_K: np.ndarray  # (H, W) float32 apparent temperature [K], row 0 = top
    intrinsics: dict[str, float]  # pinhole: fx, fy, cx, cy, w, h (pixels)


@dataclass
class VisualFix:
    t: float  # timestamp of the camera frame the fix was computed from
    pos: np.ndarray  # (2,) x, y [m] world
    cov: np.ndarray  # (2, 2) m^2
    source: str  # "mock" or "orb"
    matched_ids: list[int] = field(default_factory=list)  # onboard-map landmark ids (mock) or []
    matched_px: list[tuple[float, float]] = field(default_factory=list)  # pixel positions in the frame
    n_inliers: int = 0
    proc_ms: float = 0.0
    frame_seq: int = -1
    accepted: bool = True  # set by the EKF after Mahalanobis gating
    quality: float = 1.0  # localizer confidence 0..1 (vision: NCC peak score); gates re-localization


@dataclass
class Decision:
    t: float
    text: str  # e.g. "RELOCALIZE: ...", "WAIT_FOR_FIX above B ...", "TOUCHDOWN ...", "MISSION_TIMEOUT ..."


@dataclass
class PoseEstimate:
    t: float
    pos: np.ndarray  # (3,)
    vel: np.ndarray  # (3,) world
    yaw: float
    cov: np.ndarray  # (6, 6) over [x, y, z, vx, vy, vz]
    R: np.ndarray  # (3, 3) body -> world attitude (from the AHRS)
    omega: np.ndarray  # (3,) body rates, bias-compensated gyro
    source: str = "ekf"  # "ekf" or "truth"

    def sigma2_horizontal(self) -> float:
        """2-sigma radius of the horizontal position uncertainty (largest axis)."""
        return 2.0 * float(np.sqrt(max(np.linalg.eigvalsh(self.cov[:2, :2]).max(), 0.0)))


@dataclass
class Setpoint:
    t: float
    pos: np.ndarray
    vel: np.ndarray
    yaw: float
    phase: str


@dataclass
class MotorCmd:
    t: float
    thrust: np.ndarray  # (4,) N per motor


class Health(str, Enum):
    NOMINAL = "NOMINAL"
    DEGRADED = "DEGRADED"
    HOLD = "HOLD"
    RETURN_TO_LAST_FIX = "RETURN_TO_LAST_FIX"
    LAND = "LAND"


@dataclass
class HealthState:
    t: float
    state: Health
    sigma2: float
    time_since_fix: float
    speed_scale: float
    reason: str = ""


@dataclass
class Disturbance:
    """kind: wind | camera_blackout | imu_bias_jump | comms_loss | clear. Legacy events: wind and
    camera_blackout map onto the degradation levers (sfhack.degradation.Conditions).

    wind: value = (vx, vy) wind speed [m/s]; camera_blackout / comms_loss: duration [s] (0 = until clear);
    imu_bias_jump: value = magnitude [m/s^2] (None = config default).
    """

    t: float
    kind: str
    value: Any = None
    duration: float = 0.0


def to_jsonable(obj: Any) -> Any:
    """Convert a message (dataclass with numpy fields) into plain JSON types. Images are dropped."""
    if isinstance(obj, np.ndarray):
        return np.round(obj, 5).tolist()
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_jsonable(getattr(obj, k)) for k in obj.__dataclass_fields__ if k != "image"}
    return obj
