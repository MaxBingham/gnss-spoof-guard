"""The onboard computer: sensor messages + onboard map in, motor commands out.

Wires AHRS + EKF (estimation), the localizer (perception), health + mission (planning) and the
geometric controller (control). It never imports the simulator. The only way truth can reach the
controller is a `TruthPoseSource` injected from outside (dashboard "truth" switch / tuning).
"""

from __future__ import annotations

import time

import numpy as np

from . import bus as topics
from .bus import Bus
from .config import Config
from .control.geometric import GeometricController
from .control.mixer import VehicleParams
from .control.pose_source import EkfPoseSource, PoseSource
from .estimation.ahrs import Ahrs
from .estimation.ekf import Ekf
from .messages import (
    BaroMsg, CameraFrame, Decision, GnssMsg, Health, HealthState, ImuMsg, MagMsg, MotorCmd, PoseEstimate, RangeMsg, Setpoint,
    VisualFix,
)
from .perception.localizer import Localizer
from .planning.health import HealthMonitor
from .planning.mission import Mission
from .world.onboard_map import OnboardMap

PUBLISH_HZ = 50.0  # pose / setpoint / health topics (the control loop itself runs at the IMU rate)


class Autopilot:
    def __init__(
        self,
        cfg: Config,
        bus: Bus,
        vehicle: VehicleParams,
        mission: Mission,
        start_pos: np.ndarray,
        start_yaw: float,
        onboard_map: OnboardMap | None = None,
        localizer: Localizer | None = None,
        truth_pose_source: PoseSource | None = None,
        pose_source: str = "ekf",
    ) -> None:
        self.cfg = cfg
        self.bus = bus
        self.mission = mission
        self.onboard_map = onboard_map
        self.localizer = localizer
        self.ahrs = Ahrs(cfg.ahrs, start_yaw)
        self.ekf = Ekf(cfg.ekf, np.asarray(start_pos, float))
        self.health = HealthMonitor(cfg.health)
        self.controller = GeometricController(vehicle, cfg.gains)
        self._sources: dict[str, PoseSource] = {"ekf": EkfPoseSource(self)}
        if truth_pose_source is not None:
            self._sources["truth"] = truth_pose_source
        self.pose_source = "ekf"
        self.set_pose_source(pose_source)

        self.preflight_until = cfg.mission.preflight_s
        self._last_imu_t: float | None = None
        self._last_mag_t: float | None = None
        self._baro_sum = 0.0
        self._baro_n = 0
        self.baro_offset = 0.0
        self._range_rest: list[float] = []  # preflight rangefinder readings (sensor height when standing)
        self.range_offset = 0.0
        self.range_agl: float | None = None  # rangefinder height above the surface below, None = no reading
        self._range_t = -1.0
        self.imu_still_s = 0.0  # how long the IMU has been stationary (touchdown check)
        self._next_pub = 0.0
        self.last_fix: VisualFix | None = None
        self.last_gnss: GnssMsg | None = None
        self.health_state = HealthState(0.0, Health.NOMINAL, 0.0, 0.0, 1.0)
        self.setpoint: Setpoint | None = None
        # Localizer statistics (for eval): frames seen, fixes returned, processing time
        self.loc_frames = 0
        self.loc_fixes = 0
        self.loc_ms_total = 0.0
        self.last_acc_world = np.zeros(3)  # diagnostics: EKF input after rotation + gravity removal
        self.decisions: list[tuple[float, str]] = []  # decision log (also published on topics.DECISION)

        bus.subscribe(topics.IMU, self.on_imu)
        bus.subscribe(topics.BARO, self.on_baro)
        bus.subscribe(topics.RANGE, self.on_range)
        bus.subscribe(topics.MAG, self.on_mag)
        bus.subscribe(topics.CAMERA_DOWN, self.on_camera)
        bus.subscribe(topics.GNSS, self.on_gnss)

    # ------------------------------------------------------------------ switches

    def set_pose_source(self, name: str) -> None:
        if name not in self._sources:
            raise ValueError(f"pose source {name!r} not available (have {list(self._sources)})")
        self.pose_source = name

    def set_localizer(self, localizer: Localizer | None) -> None:
        self.localizer = localizer

    def log_decision(self, t: float, text: str) -> None:
        self.decisions.append((t, text))
        self.bus.publish(topics.DECISION, Decision(t, text))

    # ------------------------------------------------------------------ estimate

    def pose(self) -> PoseEstimate:
        return self.ekf.pose(self.ahrs.R, self.ahrs.omega, self.ahrs.yaw)

    @property
    def in_preflight(self) -> bool:
        return self.mission.phase == "PREFLIGHT"

    # ------------------------------------------------------------------ sensor callbacks

    def on_imu(self, msg: ImuMsg) -> None:
        dt = 0.0 if self._last_imu_t is None else msg.t - self._last_imu_t
        self._last_imu_t = msg.t
        self.ahrs.update_imu(msg)
        mc = self.cfg.mission
        still = (abs(float(np.linalg.norm(msg.accel - self.ahrs.accel_bias)) - 9.81) < mc.still_acc
                 and float(np.linalg.norm(self.ahrs.omega)) < mc.still_rate)
        self.imu_still_s = self.imu_still_s + dt if still else 0.0
        if self.ahrs.calibrating:
            self.ekf.t = msg.t
            if msg.t >= self.preflight_until:  # end of the on-ground phase
                self.ahrs.finish_calibration()
                if self._baro_n:
                    self.baro_offset = self._baro_sum / self._baro_n
                self.mission.start()
        else:
            self.last_acc_world = self.ahrs.rotate_accel(msg.accel)
            self.ekf.predict(self.last_acc_world, dt, msg.t)
        if self.in_preflight or self.mission.done:
            self.ekf.update_zero_velocity()
        if not self.ahrs.calibrating:
            self.ahrs.correct_tilt_world(self.ekf.take_tilt_correction())
        self._control(msg.t, dt)

    def on_baro(self, msg: BaroMsg) -> None:
        if self.in_preflight:
            self._baro_sum += msg.alt
            self._baro_n += 1
            return
        self.ekf.update_baro(msg.alt - self.baro_offset)

    def on_range(self, msg: RangeMsg) -> None:
        if self.in_preflight:
            if np.isfinite(msg.range):
                self._range_rest.append(msg.range)
            return
        if self._range_rest and not self.range_offset:
            self.range_offset = float(np.median(self._range_rest))
        self.range_agl = msg.range - self.range_offset if np.isfinite(msg.range) else None
        self._range_t = msg.t

    def on_mag(self, msg: MagMsg) -> None:
        dt = 1.0 / self.cfg.sim.mag_hz if self._last_mag_t is None else msg.t - self._last_mag_t
        self._last_mag_t = msg.t
        self.ahrs.update_mag(msg, dt)

    def on_gnss(self, msg: GnssMsg) -> None:
        """NAIVE baseline: every GNSS fix is fused as reported (no spoof check)."""
        self.ekf.update_gnss(msg.t, msg.pos, msg.cov)
        self.health.on_gnss(msg.t, msg.pos)
        self.last_gnss = msg

    def on_camera(self, frame: CameraFrame) -> None:
        if self.localizer is None or self.onboard_map is None or self.in_preflight:
            return
        prior = self.pose()
        t0 = time.perf_counter()
        fix = self.localizer.update(frame, self.onboard_map, prior)
        self.loc_frames += 1
        self.loc_ms_total += (time.perf_counter() - t0) * 1000.0
        if fix is None:
            return
        self.loc_fixes += 1
        self.ekf.update_fix(fix)
        if self.ekf.last_decision:
            self.log_decision(frame.t, self.ekf.last_decision)
        self.health.on_fix(fix)
        self.last_fix = fix
        self.bus.publish(topics.FIX, fix)

    # ------------------------------------------------------------------ control loop (IMU rate)

    def _control(self, t: float, dt: float) -> None:
        est = self.pose()
        ctrl_pose = self._sources[self.pose_source].get()
        if self.mission.phase == "CRUISE":
            self.health.reset_timer(t)
        hs = self.health.update(t, est)
        self.health_state = hs
        fresh = self.range_agl is not None and t - self._range_t < 0.2
        self.mission.set_ground_sensing(self.range_agl if fresh else None, self.imu_still_s)
        # Health actions are enforced only when flying on the estimate (truth mode = tuning).
        enforce = self.pose_source == "ekf"
        sp = self.mission.update(
            t, ctrl_pose, dt,
            health=hs.state if enforce else Health.NOMINAL,
            speed_scale=hs.speed_scale if enforce else 1.0,
            last_fix_pos=self.health.last_fix_pos,
        )
        self.setpoint = sp
        while self.mission.events:
            self.log_decision(*self.mission.events.pop(0))
        if self.mission.phase in ("PREFLIGHT", "LANDED"):
            motors = np.zeros(4)
        elif self.mission.ground_contact and self.pose_source == "ekf":
            motors = np.full(4, self.cfg.mission.ground_idle * self.controller.mass * self.controller.g / 4)
        else:
            motors = self.controller(ctrl_pose, sp)
        self.bus.publish(topics.MOTOR, MotorCmd(t, motors))
        if t >= self._next_pub:
            self._next_pub = t + 1.0 / PUBLISH_HZ
            self.bus.publish(topics.POSE, est)
            self.bus.publish(topics.SETPOINT, sp)
            self.bus.publish(topics.HEALTH, hs)
