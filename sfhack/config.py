"""All tunable parameters in one place: rates, noise levels, gains, camera, mission, health.

Units: SI (m, s, rad) unless the name says otherwise (`_deg`, `_hz`, `_px`).
Frames: world = local ENU (x east, y north, z up), origin = map center. Yaw 0 = east, CCW positive.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
X2_DIR = REPO_ROOT / "third_party" / "skydio_x2"
SAMPLE_WORLD_DIR = Path(__file__).resolve().parent / "world" / "sample_berkeley"


def data_dir() -> Path:
    """Root for large / generated data, outside the (OneDrive-synced) repo. Env var SFHACK_DATA."""
    p = Path(os.environ.get("SFHACK_DATA") or (Path.home() / "sf-hack-data"))
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class SimConfig:
    timestep: float = 0.002
    imu_hz: float = 200.0
    baro_hz: float = 20.0
    mag_hz: float = 20.0
    down_cam_hz: float = 5.0
    front_cam_hz: float = 5.0
    gnss_hz: float = 5.0
    gnss_enabled: bool = False  # GNSS exists but is DISABLED: this project is about flying without it
    max_time_s: float = 400.0


@dataclass
class SensorNoise:
    gyro_noise: float = 0.005  # rad/s, white, per sample
    gyro_bias_init: float = 0.01  # rad/s, turn-on bias std (removed by preflight calibration)
    gyro_bias_rw: float = 2e-5  # rad/s/sqrt(s), random walk
    accel_noise: float = 0.05  # m/s^2, white, per sample
    accel_bias_init: float = 0.1  # m/s^2, turn-on bias std (removed by preflight calibration)
    accel_bias_rw: float = 3e-4  # m/s^2/sqrt(s)
    baro_noise: float = 0.3  # m
    baro_bias_rw: float = 0.01  # m/sqrt(s)
    mag_noise_deg: float = 2.0
    mag_bias_deg: float = 1.0  # constant heading bias std (uncalibrated declination / hard iron)
    gnss_noise: float = 2.5  # m (unused while GNSS is disabled)


@dataclass
class CameraConfig:
    down_fovy_deg: float = 70.0
    down_width: int = 320
    down_height: int = 240
    front_fovy_deg: float = 75.0
    front_width: int = 320
    front_height: int = 240
    jpeg_quality: int = 80


@dataclass
class ControlGains:
    """Cascaded geometric controller, ported from fly_x2.py (tuned for the Skydio X2)."""

    kp_pos: tuple[float, float, float] = (2.0, 2.0, 4.0)
    kd_pos: tuple[float, float, float] = (2.5, 2.5, 3.5)
    kR: tuple[float, float, float] = (4.0, 4.0, 1.0)
    kw: tuple[float, float, float] = (0.6, 0.6, 0.3)
    max_acc: float = 4.0  # m/s^2, clip of the position-loop acceleration command


@dataclass
class MissionConfig:
    cruise_alt: float = 40.0  # m above ground; above everything on the Berkeley route
    cruise_speed: float = 8.0  # m/s
    climb_speed: float = 3.0
    descent_speed: float = 2.5
    flare_alt: float = 6.0  # below this, descend at land_speed
    land_speed: float = 0.7  # the slower final landing from fly_x2.py
    wp_radius: float = 1.0  # m, waypoint reached (horizontal, on the estimate)
    preflight_s: float = 2.0  # on the ground: IMU calibration + zero-velocity updates
    landed_alt: float = 0.35  # estimated z below which touchdown is checked
    success_radius: float = 3.0  # landed within this distance of B = success (evaluation)


@dataclass
class EkfConfig:
    accel_process_std: float = 0.15  # m/s^2/sqrt(Hz), continuous white accel noise (incl. model error)
    init_pos_std: float = 1.0
    init_vel_std: float = 0.1
    baro_std: float = 0.4
    zupt_std: float = 0.05  # zero-velocity update on the ground
    gate_chi2: float = 9.21  # 99 % for 2 dof: visual fixes beyond this Mahalanobis distance are rejected


@dataclass
class AhrsConfig:
    tilt_gain: float = 0.02  # 1/s, accel-based roll/pitch correction (only near 1 g)
    yaw_gain: float = 1.0  # 1/s, compass yaw correction
    accel_gate: float = 0.15  # m/s^2, use accel for tilt only when | |a| - g | < gate


@dataclass
class LocalizerConfig:
    mock_sigma: float = 2.0  # m, MOCK: noise added to the true position
    mock_min_landmarks: int = 3
    orb_features: int = 1500
    orb_min_inliers: int = 12
    orb_ratio: float = 0.8
    orb_ransac_px: float = 4.0
    orb_sigma_ref: float = 1.5  # m, fix sigma at orb_inliers_ref inliers
    orb_inliers_ref: int = 30
    orb_max_window_px: int = 700  # max search window (map px) around the prior
    orb_max_scale_err: float = 0.25
    orb_max_rot_deg: float = 20.0


@dataclass
class HealthConfig:
    """Thresholds on the 2-sigma horizontal radius and the time since the last accepted visual fix."""

    degraded_sigma2: float = 8.0
    degraded_no_fix_s: float = 4.0
    hold_sigma2: float = 20.0
    hold_no_fix_s: float = 15.0
    return_after_hold_s: float = 10.0  # HOLD this long without recovery -> RETURN_TO_LAST_FIX
    land_sigma2: float = 45.0
    land_no_fix_s: float = 60.0
    degraded_speed_scale: float = 0.5


@dataclass
class DisturbanceConfig:
    wind_drag: float = 0.15  # N per m/s of relative air speed (linear drag model for wind force)
    wind_gust_std: float = 0.3  # N, random gust force on top of the steady wind
    imu_bias_jump: float = 0.3  # m/s^2, magnitude of an accel bias jump


@dataclass
class Config:
    sim: SimConfig = field(default_factory=SimConfig)
    noise: SensorNoise = field(default_factory=SensorNoise)
    camera: CameraConfig = field(default_factory=CameraConfig)
    gains: ControlGains = field(default_factory=ControlGains)
    mission: MissionConfig = field(default_factory=MissionConfig)
    ekf: EkfConfig = field(default_factory=EkfConfig)
    ahrs: AhrsConfig = field(default_factory=AhrsConfig)
    localizer: LocalizerConfig = field(default_factory=LocalizerConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    disturbance: DisturbanceConfig = field(default_factory=DisturbanceConfig)


def default_config() -> Config:
    return Config()
