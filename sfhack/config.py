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
    # Epoch jitter only (m). The rest of the ~1 m open-sky CEP is the slow Gauss-Markov
    # term in sfhack.degradation.models, not a fresh metre-scale draw every fix.
    gnss_noise: float = 0.4
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
    # downward thermal (LWIR) camera, same mount as "down": 640 x 512, 50 deg horizontal FOV (sfhack/thermal/camera.py)
    thermal_width: int = 640
    thermal_height: int = 512
    thermal_hfov_deg: float = 50.0


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
    # Reported horizontal 1σ floor (m). Formal P falls to ~0.3 m (2σ ~0.7 m) while an honest
    # seed-0 cruise actually flies ~2 m at the 95th percentile: GNSS multipath is correlated and
    # the vision update treats each fix as sharper than that error. The reported major axis is
    # held at this floor (2σ = 2.5 m). It does not grow with a spoof offset, so a drag still
    # claims a few metres against a hundred-metre error.
    horiz_std_floor: float = 1.25
    # V3.2 GNSS -> GNSS-free transition. A naive GNSS jump (50 m in one epoch) moved the accel-bias states by
    # ~1 m/s^2, which the error-state reset handed to the AHRS as a 5 deg tilt: after the jam the drone flew
    # off at 20 m/s with 2-sigma ~2 m. Normal per-update bias steps are <= 0.022 m/s^2 (vision + GNSS).
    acc_bias_step_max: float = 0.03  # m/s^2, max change of (bx, by) from one measurement update
    gnss_loss_after_s: float = 1.0  # no GNSS fix this long after having had them = GNSS lost
    # GNSS-free track: a second EKF (IMU + baro + vision, never GNSS) started at the first GNSS fix and run in
    # parallel. On GNSS loss the main filter adopts it when it is healthy (HANDOVER). A spoof the naive
    # main filter swallowed (lever jump: 50 m) is then gone at once; inflating the main covariance instead
    # cannot reach that far: a 60 m NCC window at 40 m altitude is ambiguous almost everywhere (no fixes for
    # 27 s, then a confident fix 65 m wrong).
    gnss_free_track: bool = True
    handover_max_fix_age_s: float = 5.0  # GNSS-free track is healthy: an accepted vision fix this recent ...
    handover_max_sigma2: float = 10.0  # ... and its 2-sigma below this (m)
    # Fallback (no healthy GNSS-free track, e.g. camera lost at the jam): floor the main covariance here, the
    # GNSS-made ~0.4 m is no promise once the receiver is gone, then re-localize on the first STRONG fix.
    gnss_loss_pos_std: float = 5.0  # m
    gnss_loss_vel_std: float = 1.0  # m/s
    reloc_min_quality: float = 0.6  # NCC peak score a fix needs to re-localize after GNSS loss


@dataclass
class AhrsConfig:
    # Roll/pitch error x g leaks straight into the horizontal acceleration (1 deg = 0.17 m/s^2), so the
    # accel correction must beat the residual gyro bias: steady tilt error ~ bias / tilt_gain. 0.02 left
    # ~0.8 deg (-> 140 m drift in 60 s); 0.3 keeps it < 0.1 deg.
    tilt_gain: float = 0.3  # 1/s, accel-based roll/pitch correction (only in steady flight, see gates)
    yaw_gain: float = 1.0  # 1/s, compass yaw correction
    accel_gate: float = 0.05  # m/s^2, use accel for tilt only when | |a| - g | < gate (~1 m/s^2 horizontal)
    rate_gate: float = 0.1  # rad/s, ... and only while not rotating (steady thrust direction)
    # Night-nav: integral term of the EKF's tilt corrections (correct_tilt_world) into the roll / pitch gyro bias,
    # 1/s. Without GNSS the residual gyro bias tilts the AHRS ~1 deg per 10-20 s, which only position fixes can
    # see; across a 30 s blind stretch (forest) the gravity leak then grew the error by tens of m. 0 = off.
    tilt_bias_ki: float = 0.0


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
class TrnConfig:
    """TRN (perception/trn.py): nadir lidar vs onboard DSM particle filter. Values = death-v3's urban set."""

    enabled: bool = True  # --trn on|off. Fused into the main filter only while no GNSS arrives (the naive
    # GNSS baseline and the spoof demo stay as they were); always into the GNSS-free track.
    dsm_mpp: float = 1.0  # m, onboard DSM raster
    n_particles: int = 3000
    sigma_meas: float = 2.0  # m, lidar + map error (1 sigma core of the Student-t)
    nu: float = 3.0  # Student-t dof (heavy tails: trees, footprint polygon vs simulated box walls)
    outlier_floor: float = 0.05  # likelihood floor (fraction of the peak)
    beam_weight: float = 0.35  # log-likelihood weight per beam (the 8 beams span ~11 m: correlated map error)
    corr_length: float = 3.0  # m, map-error correlation length along track (likelihood tempering)
    update_hz: float = 10.0
    min_range: float = 0.5  # m
    init_inflate: float = 1.5  # initial cloud = EKF covariance x this (std), at least init_min_std
    init_min_std: float = 4.0
    init_max_std: float = 150.0
    bias_init_std: float = 4.0  # m
    bias_keep_std: float = 1.0  # m, bias spread on a restart once the bias has been learned
    bias_rw: float = 0.05  # m/sqrt(s)
    vel_err_min_std: float = 0.2  # m/s
    vel_err_rw: float = 0.02  # m/s/sqrt(s), on top of the EKF's velocity-variance growth
    pos_rw: float = 0.05  # m/sqrt(m travelled), roughening
    min_travel: float = 60.0  # m of matched track after (re)initialization before the first fix
    max_fix_std: float = 5.0  # m, cloud 1-sigma below this always qualifies for a fix ...
    rel_fix_std: float = 0.4  # ... or below this x the INS 1-sigma ...
    cap_fix_std: float = 25.0  # ... but never wider than this
    min_fix_std: float = 1.0  # m, floor of the published fix sigma
    # Closed-loop dead reckoning drifts 2-3 m/s here (no airframe drag in calm air, so the accelerometer cannot
    # see velocity): with V3's sparse fixes (every 20 m, sigma x 2) the EKF ran 10-25 m off while the TRN cloud
    # itself stayed within 1-3 m, and hovering over B drifted 60 m. So fixes are frequent, and while hovering
    # one every hover_fix_s (the cloud then spreads with the INS velocity error, so its covariance stays honest).
    fix_inflate: float = 1.5  # published sigma = cloud sigma x this (correlated map errors)
    fix_interval: float = 0.5  # s between published fixes
    fix_travel: float = 6.0  # m of new track between two fixes ...
    hover_fix_s: float = 2.0  # ... or this long when (almost) hovering
    info_track_m: float = 120.0  # along-track window for the information-content report
    lost_after_s: float = 8.0  # likelihood at the outlier floor this long -> re-initialize (keeps the bias)


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
    # V3.2: sigma2 > land_sigma2 means HOLD (hover and keep searching), not LAND, while blind. LAND only
    # after land_no_fix_s without any fix. False = the old rule (sigma-based LAND).
    hold_when_blind: bool = True
    # V3.2 TRN: hovering gives terrain-referenced navigation no new terrain (no fixes while hovering), so while
    # TRN is tracking the drone keeps flying at the degraded speed across flat stretches, up to these limits.
    hold_sigma2_trn: float = 40.0
    hold_no_fix_s_trn: float = 30.0
    degraded_speed_scale: float = 0.5
    # Night-nav: no HOLD / RETURN / LAND mid-route when the uncertainty grows. Hovering gives TRN and the thermal
    # localizer nothing new and the INS drifts all the same, so the drone keeps flying the planned route
    # (state DEGRADED, reason "BLIND, keep route") and the HUD shows the growing 2-sigma. False = V3.2 rules.
    keep_route_when_blind: bool = True
    blind_speed_scale: float = 1.0  # cruise speed factor while DEGRADED under keep_route_when_blind


@dataclass
class ThermalConfig:
    """Night (night-nav): thermal localizer (perception/thermal_localizer.py) and the simulated thermal camera."""

    camera_hz: float = 5.0  # thermal frames per second fed to the localizer
    # Heading: the compass has a constant ~1 deg bias; every velocity change is then rotated into the wrong
    # direction (8 m/s x 1.5 deg = 0.2 m/s cross-track drift). With yaw_feedback the yaw corrections of accepted
    # thermal fixes (yaw search of the template) teach the autopilot the compass bias, slowly.
    yaw_feedback: bool = False
    # EKF accel-bias random walk while flying on thermal fixes (None = EkfConfig.acc_bias_rw, 0.01). With GNSS off
    # the AHRS tilt drifted ~1 deg in 10-20 s (0.2 m/s^2 of gravity leak); at 0.01 m/s^2/sqrt(s) the bias states
    # could not follow, the filter stayed confident (2-sigma 2.5 m) while the true error grew to 14 m, and then
    # rejected the correct thermal fixes (m-c804bb84 seed 0 ran away).
    ekf_acc_bias_rw: float | None = None
    # Thermal fixes arrive at 5 Hz with errors correlated over seconds (same map region, same prior bias); fused as
    # independent they collapse the EKF covariance (formal 2-sigma ~0.6 m vs ~2-3 m true). Scale their covariance.
    fix_cov_scale: float = 1.0
    # FlightSystem(night=True) sets ahrs.tilt_bias_ki to this unless it is set explicitly (0 = leave it off)
    reloc_min_span_s: float = 0.0  # forced re-localization from thermal only after rejections spanning this (0 = off)
    night_tilt_bias_ki: float = 0.05  # ukraine-1: runaways after a 30 s forest stretch gone (docs/NIGHT_NAV.md)
    confirm_jump_m: float = 5.0  # a fix moving the estimate more than this needs a second, consistent fix (0 = off)
    confirm_window_s: float = 2.0
    confirm_tol_m: float = 3.0
    yaw_feedback_gain: float = 0.05  # per accepted fix
    yaw_feedback_min_quality: float = 0.6
    yaw_feedback_max_deg: float = 5.0
    method: str = "grad"  # best in the open-loop bench and the closed-loop eval (docs/NIGHT_NAV.md)
    tpl_mpp: float = 0.5  # m per template cell (= prior resolution)
    tpl_frac: float = 0.26  # template half size = this x height above the surface: the north-up square must fit
    # in the 56 x 45 m (at 60 m) footprint at any heading (inscribed half ~0.26 x h at 45 deg)
    tpl_min_m: float = 6.0
    tpl_max_m: float = 18.0
    min_alt: float = 10.0  # m above the surface: below this the footprint is too small
    min_std_K: float = 0.4  # rectified image std below this = uniform (no fix)
    search_margin_m: float = 6.0  # search radius = 3 sigma(prior) + margin, clipped to [min, max]
    search_min_m: float = 8.0
    search_max_m: float = 60.0
    yaw_search_deg: tuple[float, ...] = (-3.0, -1.5, 0.0, 1.5, 3.0)
    min_score: float = 0.45  # NCC peak below this -> no fix
    min_margin: float = 0.06  # best - second local maximum below this -> ambiguous
    peak_delta: float = 0.05  # peak region = cells within this of the best score (its shape -> covariance)
    peak_cov_scale: float = 1.0  # covariance = this x weighted spread of the peak region (+ floor, tilt)
    sigma_floor: float = 1.0  # m
    tilt_std_deg: float = 0.5
    grad_min_score: float = 0.35
    # road centerlines
    road_trunc_m: float = 4.0  # chamfer distance truncation
    road_temp_T_m: float = 0.25  # chamfer likelihood temperature (m of mean distance)
    road_min_px: int = 30  # centerline pixels needed (15 m of road)
    road_min_quality: float = 0.55  # fraction of observed centerline within 1.5 m of a prior road
    road_sigma_floor: float = 1.0
    road_max_minor_m: float = 6.0  # cov minor axis 1-sigma above this = no fix
    road_include_footways: bool = True  # campus: footways are paved and warm too (road_ref "osm")
    road_ref: str = "mask"  # reference centerlines: "mask" (thinned prior road mask) | "osm" (way polylines)
    # junctions
    junc_gate_m: float = 12.0
    junc_sigma: float = 2.0
    combo_agree_chi2: float = 9.21


@dataclass
class DisturbanceConfig:
    wind_drag: float = 0.15  # N per m/s of relative air speed (linear drag model for wind force)
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
    trn: TrnConfig = field(default_factory=TrnConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    disturbance: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    # Initial degradation levers (simulated reality). Read by the simulator only, never onboard.
    conditions: Conditions = field(default_factory=Conditions)


def default_config() -> Config:
    return Config()


@dataclass
class AcousticConfig:
    """Base UMA-16 live path (`sfhack/acoustic`, `demos/40_uma16_live.py`). Not part of the flight system: the
    drone only gets the resulting BearingEstimate (bearing.json)."""
    device_match: str = "UMA16"          # sounddevice input name contains this (ASIO preferred)
    fs: int = 48000
    window_s: float = 0.5                # one DOA estimate per window
    learn_s: float = 10.0                # noise floor learned over the first seconds (keep the source off)
    detect_fmin: float = 50.0            # detection band (engine-like energy)
    detect_fmax: float = 2000.0
    detect_db: float = 6.0               # level over the learned floor (dB) to count as a detection:
    detect_bands: int = 3                # level = detect_bands-th largest 1/3-octave band excess (one tone is not enough)
    engine_score_db: float = 10.0        # harmonic comb score (dB) for the "engine" label (heuristic)
    doa_fmin: float = 100.0              # DOA band: the beam is very wide below ~500 Hz, but a far engine has little
                                         # else (demos/42: far engine -10 dB, 3.3 -> 2.0 deg rms vs 300 Hz);
    doa_fmax: float = 4000.0             # above ~4.1 kHz the 42 mm pitch aliases
    stable_s: float = 3.0                # source must be detected with a steady bearing this long
    stable_std_deg: float = 8.0          # max circular std of the azimuth over stable_s
    sigma_az_floor_deg: float = 5.0      # added in quadrature: mounting / north alignment, array bias,
    sigma_el_floor_deg: float = 10.0     # far low-frequency engines (elevation is poor near the horizon)
    # Mounting: compass bearing of the array's +x axis, and whether the mics face up (flat on the table).
    x_axis_bearing_deg: float = 0.0
    mics_up: bool = True
