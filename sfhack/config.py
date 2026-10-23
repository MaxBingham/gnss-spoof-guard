"""All tunable parameters in one place: rates, noise levels, gains, camera, mission, health.

Units: SI (m, s, rad) unless the name says otherwise (`_deg`, `_hz`, `_px`).
Frames: world = local ENU (x east, y north, z up), origin = map center. Yaw 0 = east, CCW positive.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .degradation.conditions import Conditions

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
    range_hz: float = 20.0  # downward rangefinder
    mag_hz: float = 20.0
    down_cam_hz: float = 5.0
    front_cam_hz: float = 5.0
    gnss_hz: float = 5.0  # GNSS receiver (off by default: Config.conditions.gnss.mode = "off")
    lidar_hz: float = 10.0  # nadir multi-beam lidar
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
    range_noise: float = 0.02  # m, rangefinder white noise ...
    range_noise_rel: float = 0.01  # ... plus this fraction of the range
    range_max: float = 40.0  # m, beyond this the rangefinder reports nothing (nan)
    mag_noise_deg: float = 2.0
    mag_bias_deg: float = 1.0  # constant heading bias std (uncalibrated declination / hard iron)
    gnss_noise: float = 1.5  # m, white horizontal noise per GNSS fix (plus a slow Gauss-Markov error)
    lidar_range_max: float = 100.0  # m, nadir multi-beam lidar


@dataclass
class CameraConfig:
    down_fovy_deg: float = 70.0
    down_width: int = 320
    down_height: int = 240
    front_fovy_deg: float = 75.0
    front_width: int = 320
    front_height: int = 240
    jpeg_quality: int = 80
    # Optics of the down camera: Gaussian blur [px] (lens MTF + demosaicing of a small 320x240 camera).
    # 0 = the ideal sharp render (default, all existing flights). The V4 target extension uses 1.2: a 1.6 m
    # ArUco 6x6 marker then resolves below ~14 m and never from 16 m up (sharp renders decoded it from 25 m).
    down_blur_px: float = 0.0


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
    # Vision fixes stop at ~4-5 m (0.5 m/px ortho vs a tiny footprint): flare low and land briskly so the
    # final blind stretch is ~4 s (was 6 m at 0.7 m/s = 9 s, which drifted 4-10 m).
    flare_alt: float = 4.0  # below this, descend at land_speed
    land_speed: float = 1.0
    wp_radius: float = 1.0  # m, waypoint reached (horizontal, on the estimate)
    preflight_s: float = 4.0  # on the ground: IMU calibration + zero-velocity updates (gyro bias error ~ 1/sqrt(t))
    # Touchdown: rangefinder height (minus its at-rest reading, calibrated in preflight) below touchdown_range
    # AND IMU stationary, both for touchdown_s. Fallback without rangefinder: the estimated height rule.
    touchdown_range: float = 0.3  # m
    ground_idle: float = 0.3  # fraction of hover thrust, uniform on all motors, while in ground contact
    still_acc: float = 0.3  # m/s^2, IMU stationary: | |a| - g | below this ...
    still_rate: float = 0.2  # rad/s, ... and |omega| below this
    landed_alt: float = 0.35  # (fallback) estimated z below which touchdown is checked
    touchdown_vz: float = 0.25  # m/s, |estimated vertical speed| below this ...
    touchdown_s: float = 0.5  # ... for this long (while descending) = touchdown -> motors off, LANDED
    precision_sigma2: float = 3.0  # m, at B: descend only if the 2-sigma radius is below this ...
    precision_wait_s: float = 10.0  # ... or after hovering this long waiting for a fix
    descend_xy_tol: float = 1.5  # m, DESCEND / LAND: pause the descent while the estimate is farther than
    # this from B horizontally (a late fix moved it, or wind): re-center first, then continue down
    success_radius: float = 3.0  # landed within this distance of B = success (evaluation)


@dataclass
class EkfConfig:
    accel_process_std: float = 0.15  # m/s^2/sqrt(Hz), continuous white accel noise (incl. model error)
    init_pos_std: float = 1.0
    init_vel_std: float = 0.1
    # Horizontal accel bias states (world frame): absorb AHRS tilt error x g (1 deg = 0.17 m/s^2).
    # Sized to the measured dead reckoning (~15 m / 30 s): 0.05 / 0.02 put 2-sigma at 45 m (health LAND)
    # after 20 s without fixes.
    init_acc_bias_std: float = 0.03  # m/s^2 (the AHRS starts level-calibrated on the ground)
    acc_bias_rw: float = 0.01  # m/s^2/sqrt(s), random walk: tilt error drifts with the residual gyro bias
    baro_std: float = 0.4
    zupt_std: float = 0.05  # zero-velocity update on the ground
    gate_chi2: float = 9.21  # 99 % for 2 dof: visual fixes beyond this Mahalanobis distance are rejected
    nofix_inflate_after_s: float = 2.0  # without accepted fixes for this long, apply a covariance floor ...
    nofix_inflation: float = 0.5  # ... of (this * seconds blind) m std, so a correct fix is not gated out


@dataclass
class AhrsConfig:
    # Roll/pitch error x g leaks straight into the horizontal acceleration (1 deg = 0.17 m/s^2), so the
    # accel correction must beat the residual gyro bias: steady tilt error ~ bias / tilt_gain. 0.02 left
    # ~0.8 deg (-> 140 m drift in 60 s); 0.3 keeps it < 0.1 deg.
    tilt_gain: float = 0.3  # 1/s, accel-based roll/pitch correction (only in steady flight, see gates)
    yaw_gain: float = 1.0  # 1/s, compass yaw correction
    accel_gate: float = 0.05  # m/s^2, use accel for tilt only when | |a| - g | < gate (~1 m/s^2 horizontal)
    rate_gate: float = 0.1  # rad/s, ... and only while not rotating (steady thrust direction)


@dataclass
class LocalizerConfig:
    mock_sigma: float = 2.0  # m, MOCK: noise added to the true position
    # 2, not 3: at 40 m cruise the 70 deg footprint holds 3+ mapped landmarks on only ~15 % of the
    # Berkeley route (V1 needed the same relaxation); 2 is still a 'real localizer could match here'.
    mock_min_landmarks: int = 2
    # VISION localizer (perception/map_localizer.py): NCC of the ortho-rectified down camera vs the ortho
    vision_min_alt: float = 4.0  # m, below this the footprint is too small to be distinctive
    vision_tpl_px: int = 64  # template size in pixels (resolution follows the footprint)
    vision_tpl_frac: float = 0.4  # template half size = this x height (footprint half height ~0.7 x h)
    vision_tpl_min_m: float = 2.0
    vision_tpl_max_m: float = 20.0
    vision_min_std: float = 4.0  # gray levels: a uniform template (roof, blank ground) cannot match
    vision_search_margin_m: float = 4.0  # search radius = 3 sigma(prior) + margin ...
    vision_search_min_m: float = 6.0  # ... clipped to [min, max]
    vision_search_max_m: float = 40.0
    vision_search_max_px: int = 128  # wider than this many template pixels -> no fix (too uncertain for
    # the altitude: a capped window produced confident wrong fixes 40 m off in the benchmark)
    vision_min_score: float = 0.5  # NCC peak below this -> no fix
    vision_peak_excl_m: float = 3.0  # second peak is searched outside this radius around the best ...
    vision_min_margin: float = 0.08  # ... and must be this much lower (else ambiguous -> no fix)
    vision_curv_ref: float = 0.05  # NCC peak curvature [1/px^2] that counts as a sharp peak
    vision_sigma_floor: float = 0.5  # m
    vision_tilt_std_deg: float = 0.5  # AHRS attitude error projected to the ground: height x this
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
    imu_bias_jump: float = 0.3  # m/s^2, magnitude of an accel bias jump


@dataclass
class TargetConfig:
    """V4 target selection + precision landing (sfhack/planning/target_mission.py). Used only when a flight is
    built with the target extension (sfhack/sim/target_world.py); every other flight ignores this section.

    The mission brief the drone gets: land on the landing station on the brown truck inside a square search
    area around B. The station's marker (dictionary, id, size) is part of that brief, like a published spec."""

    area_half: float = 30.0  # m, search area = B +- this (square, axis-aligned)
    pad_size: float = 2.0  # m, landing pad side (on the truck's cargo roof)
    marker_dict: str = "DICT_6X6_50"  # OpenCV ArUco dictionary of the landing station marker
    marker_id: int = 7
    marker_size: float = 1.6  # m, black square side (inside a white quiet zone on the pad)
    # Flight
    search_alt: float = 30.0  # m above ground: the camera footprint (~42 x 56 m) covers most of the area
    search_alt_low: float = 20.0  # m, search / inspect altitude when the images look degraded (fog, rain)
    inspect_alt: float = 18.0  # m, fly over a candidate here: nadir look + lidar height across it
    marker_alt: float = 10.0  # m above ground: the marker resolves below ~12 m
    search_speed: float = 4.0  # m/s (5 Hz camera: a frame every 0.8 m)
    transit_speed: float = 6.0
    search_offset: float = 12.0  # m, search pattern = area center + the 4 corners of this square
    max_search_rounds: int = 2  # fewer than 6 objects after a round -> search again (lower)
    # Detection (perception/target_detector.py)
    det_mpp: float = 0.25  # m per pixel of the ortho-rectified detection grid
    det_min_alt: float = 8.0  # m, below this no area detection (objects fill the frame)
    det_align_px: int = 2  # alignment tolerance: per pixel, min difference over +- this many grid pixels
    det_register_m: float = 3.0  # m, residual camera <-> map shift searched before differencing
    det_thresh_k: float = 5.0  # change threshold = max(floor, k x robust noise of the difference)
    det_thresh_floor: float = 22.0  # color distance (0..255 albedo units)
    det_min_area: float = 2.0  # m^2, smaller change blobs are noise
    det_max_area: float = 60.0  # m^2, larger blobs: merged objects / misregistration, not a candidate
    nominal_contrast: float = 1.0  # camera / map contrast in clear air (radiometric calibration, measured)
    # Tracking / evidence
    track_gate_m: float = 3.5  # association radius (ground position)
    track_min_obs: int = 3  # observations before a blob counts as a candidate
    height_min_returns: int = 6  # lidar / rangefinder returns inside a footprint before its height counts
    object_min_height: float = 0.8  # m, a change blob lower than this is not an object (false blob)
    # Decision (planning/target_selector.py)
    p_decide: float = 0.90  # decide when P(candidate is THE truck) >= this ...
    p_margin: float = 0.50  # ... and it leads the runner-up by this much ...
    min_nadir_obs: int = 3  # ... and it was seen this often near nadir, with its height measured
    degraded_p_decide: float = 0.95  # same, when the images look degraded
    degraded_min_nadir_obs: int = 6
    degraded_contrast: float = 0.6  # image contrast below this x nominal = degraded
    unseen_odds: float = 0.5  # prior odds that the truck is not among the candidates seen so far
    unseen_odds_all_found: float = 0.02  # ... once 6 objects are tracked
    max_inspections: int = 8  # fly-overs before giving up on a clear decision
    # Approach and precision landing (planning/precision_landing.py)
    marker_timeout_s: float = 12.0  # no marker above a candidate within this -> TARGET_REJECTED
    land_xy_tol: float = 0.35  # m, pause the descent while farther than this from the marker
    land_hold_below: float = 1.8  # m above the pad: marker leaves the view, hold the last target
    land_speed: float = 0.7  # m/s
    max_reject: int = 3  # rejected candidates before TARGET_ABORT (hover, never land unconfirmed)


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
    target: TargetConfig = field(default_factory=TargetConfig)  # V4 target landing (off unless extension)
    # Initial degradation levers (simulated reality). Read by the simulator only, never onboard.
    conditions: Conditions = field(default_factory=Conditions)


def default_config() -> Config:
    return Config()
