"""The onboard computer: sensor messages + onboard map in, motor commands out.

Wires AHRS + EKF (estimation), the localizer (perception), health + mission (planning) and the
geometric controller (control). It never imports the simulator. The only way truth can reach the
controller is a `TruthPoseSource` injected from outside (dashboard "truth" switch / tuning).
"""

from __future__ import annotations

import copy
import dataclasses
import math
import time
from typing import Any

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
    BaroMsg, CameraFrame, Decision, GnssMsg, Health, HealthState, ImuMsg, LidarMsg, MagMsg, MotorCmd, PoseEstimate,
    RangeMsg, Setpoint, ThermalFrame, VisualFix,
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
        seed: int = 0,
        thermal_localizer: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.bus = bus
        self.mission = mission
        self.onboard_map = onboard_map
        self.localizer = localizer
        self.ahrs = Ahrs(cfg.ahrs, start_yaw)
        self.ekf = Ekf(cfg.ekf, np.asarray(start_pos, float))
        # GNSS-free track (V3.2): IMU + baro + vision only, started at the first GNSS fix, adopted on GNSS loss
        self.ekf_free: Ekf | None = None
        self.n_handover = 0
        # TRN (V3.2): nadir lidar vs onboard DSM, a second GNSS-independent position source
        self.trn = None
        self.trn_fixes = 0  # fixes the TRN filter published
        self.trn_accepted = 0  # ... fused into the main filter
        self.last_trn_fix: VisualFix | None = None
        if onboard_map is not None and cfg.trn.enabled:
            from .perception.trn import TrnFilter
            from .world.onboard_dsm import build_dsm
            self.trn = TrnFilter(cfg.trn, build_dsm(onboard_map, cfg.trn.dsm_mpp), seed=seed)
        self.sources_used: set[str] = set()  # position sources the main filter fused (gnss / vision / trn)
        # Night (night-nav): downward thermal camera matched against the onboard thermal prior
        self.thermal = thermal_localizer
        if thermal_localizer is not None and cfg.thermal.ekf_acc_bias_rw is not None:
            # night: the AHRS tilt drifts faster than the day-tuned bias random walk lets the EKF follow (see
            # ThermalConfig.ekf_acc_bias_rw); a flight-local copy, the shared config stays as it is
            self.ekf.cfg = dataclasses.replace(self.ekf.cfg, acc_bias_rw=cfg.thermal.ekf_acc_bias_rw)
        self.thermal_frames = 0
        self.thermal_fixes = 0  # fixes the thermal localizer returned
        self.thermal_accepted = 0  # ... fused into the main filter
        self.thermal_ms_total = 0.0
        self.thermal_by_method: dict[str, list[int]] = {}  # method -> [returned, accepted]
        self.last_thermal_fix: VisualFix | None = None
        self.thermal_log: list[tuple[float, str, float, float, bool, float]] = []  # t, method, x, y, accepted, yaw_corr
        self.lidar_agl: float | None = None  # nadir lidar: camera height above the surface below
        self._thermal_pending: tuple[float, np.ndarray] | None = None  # (t, innovation) of a held big-jump fix
        self.thermal_held = 0
        self._thermal_rejects: list[tuple[float, np.ndarray]] = []  # (t, innovation) of consecutive NIS rejections  # big-jump fixes held for confirmation (and not confirmed)
        self.mag_bias_est = 0.0  # rad, compass heading bias learned from thermal yaw corrections (yaw_feedback)
        self._lidar_t = -1.0
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
        bus.subscribe(topics.LIDAR, self.on_lidar)
        bus.subscribe(topics.THERMAL, self.on_thermal)

    # ------------------------------------------------------------------ switches

    def set_pose_source(self, name: str) -> None:
        if name not in self._sources:
            raise ValueError(f"pose source {name!r} not available (have {list(self._sources)})")
        self.pose_source = name

    def set_localizer(self, localizer: Localizer | None) -> None:
        self.localizer = localizer

    def attach_mission(self, mission: Any) -> None:
        """Replace the waypoint Mission (before takeoff) with another one of the same interface, e.g. the recon
        mission (sfhack/mission/recon.py). A mission with `on_thermal(frame, pose)` gets every thermal frame."""
        self.mission = mission
        if hasattr(mission, "on_thermal"):
            self.bus.subscribe(topics.THERMAL, lambda f: mission.on_thermal(f, self.pose(report=False)))
        if hasattr(mission, "rear_tilt_deg") and hasattr(topics, "CAMERA_REAR_TILT"):
            from .messages import CameraTiltCmd

            nxt = [0.0]

            def point_rear_camera(est: PoseEstimate) -> None:  # 5 Hz: the rear gimbal looks at the base
                if getattr(self, "light_localizer", None) is not None:
                    return  # recon-nav's light localizer aims the gimbal itself (every rear frame)
                if est.t >= nxt[0]:
                    nxt[0] = est.t + 0.2
                    tilt = mission.rear_tilt_deg(est)
                    if tilt is not None:
                        self.bus.publish(topics.CAMERA_REAR_TILT, CameraTiltCmd(est.t, tilt))

            self.bus.subscribe(topics.POSE, point_rear_camera)

    def log_decision(self, t: float, text: str) -> None:
        self.decisions.append((t, text))
        self.bus.publish(topics.DECISION, Decision(t, text))

    # ------------------------------------------------------------------ estimate

    def pose(self, *, report: bool = True) -> PoseEstimate:
        return self.ekf.pose(self.ahrs.R, self.ahrs.omega, self.ahrs.yaw, report=report)

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
            if self.ekf_free is not None:
                self.ekf_free.predict(self.last_acc_world, dt, msg.t)
        if self.in_preflight or self.mission.done:
            self.ekf.update_zero_velocity()
            if self.ekf_free is not None:
                self.ekf_free.update_zero_velocity()
        if not self.ahrs.calibrating:
            self.ahrs.correct_tilt_world(self.ekf.take_tilt_correction())
            if self.ekf_free is not None:  # same AHRS: its bias states lose what the main filter handed over
                self.ekf_free.x[6:8] -= self.ekf.bias_handed
        if self.ekf.gnss_lost_now(msg.t):
            self._on_gnss_lost(msg.t)
        self._control(msg.t, dt)

    def on_baro(self, msg: BaroMsg) -> None:
        if self.in_preflight:
            self._baro_sum += msg.alt
            self._baro_n += 1
            return
        self.ekf.update_baro(msg.alt - self.baro_offset)
        if self.ekf_free is not None:
            self.ekf_free.update_baro(msg.alt - self.baro_offset)

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
        if self.mag_bias_est:
            msg = MagMsg(msg.t, math.atan2(math.sin(msg.yaw - self.mag_bias_est), math.cos(msg.yaw - self.mag_bias_est)))
        self.ahrs.update_mag(msg, dt)

    def on_gnss(self, msg: GnssMsg) -> None:
        """NAIVE baseline: every GNSS fix is fused as reported (no spoof check)."""
        if self.ekf_free is None and self.cfg.ekf.gnss_free_track and not self.in_preflight:
            self.ekf_free = copy.deepcopy(self.ekf)  # starts from the honest pre-flight / takeoff state
            self.ekf_free.nis_log = []
        self.ekf.update_gnss(msg.t, msg.pos, msg.cov)
        self.sources_used.add("gnss")
        self.health.on_gnss(msg.t, msg.pos)
        self.last_gnss = msg

    def on_camera(self, frame: CameraFrame) -> None:
        if self.localizer is None or self.onboard_map is None or self.in_preflight:
            return
        free = self.ekf_free
        if free is not None:  # GNSS-free track: own search window around its own prior. Called before the main
            # localization so the panels (localizer.last_*) show the main one.
            free_fix = self.localizer.update(frame, self.onboard_map,
                                             free.pose(self.ahrs.R, self.ahrs.omega, self.ahrs.yaw, report=False))
            if free_fix is not None:
                free.update_fix(free_fix)
        # Search on the filter covariance. The reporting floor would widen this window and can
        # hand a spoofed prior a look at the truth.
        prior = self.pose(report=False)
        t0 = time.perf_counter()
        fix = self.localizer.update(frame, self.onboard_map, prior)
        self.loc_frames += 1
        self.loc_ms_total += (time.perf_counter() - t0) * 1000.0
        if fix is None:
            return
        self.loc_fixes += 1
        if self.ekf.update_fix(fix):
            self.sources_used.add(fix.source)
        if self.ekf.last_decision:
            self.log_decision(frame.t, self.ekf.last_decision)
        self.health.on_fix(fix)
        self.last_fix = fix
        self.bus.publish(topics.FIX, fix)

    @property
    def trn_tracking(self) -> bool:
        """TRN has converged before and is not lost (it then needs motion: no new terrain while hovering)."""
        return self.trn is not None and self.trn.bias_known and not self.trn.needs_reset

    @property
    def receiving_gnss(self) -> bool:
        return self.ekf.last_gnss_t is not None and not self.ekf.gnss_lost

    def on_lidar(self, msg: LidarMsg) -> None:
        """TRN: one lidar scan into the particle filter. Its prior is the GNSS-free track when there is one (TRN
        must not inherit a spoof). Fixes go to the GNSS-free track always, to the main filter only while no GNSS
        arrives (the naive GNSS baseline stays as it was)."""
        self._lidar_height(msg)
        if self.trn is None or self.in_preflight or self.mission.done:
            return
        ref = self.ekf_free if self.ekf_free is not None else self.ekf
        pose = ref.pose(self.ahrs.R, self.ahrs.omega, self.ahrs.yaw)
        if self.trn.needs_reset:
            self.trn.reset(pose, msg.t, inflate=2.0)  # keeps the learned datum bias
        fix = self.trn.step(msg, pose, ref.vel_var_growth)
        if fix is None:
            return
        self.trn_fixes += 1
        self.last_trn_fix = fix
        v_ref = ref.x[3:5].copy()
        if self.ekf_free is not None:
            self.ekf_free.update_fix(copy.copy(fix), allow_reloc=False)
        if self.receiving_gnss:
            fix.accepted = False
            self.trn.ins_corrected(ref.x[3:5] - v_ref)
            return
        accepted = self.ekf.update_fix(fix, allow_reloc=False)
        self.trn.ins_corrected(ref.x[3:5] - v_ref)
        if accepted:
            self.trn_accepted += 1
            self.sources_used.add("trn")
            if self.ekf.last_decision:
                self.log_decision(msg.t, self.ekf.last_decision.replace("vision fix", "TRN fix"))
            self.health.on_fix(fix)
        self.bus.publish(topics.FIX, fix)

    def _lidar_height(self, msg: LidarMsg) -> None:
        """Height above the surface below from the near-nadir lidar beams (vertical component, AHRS tilt)."""
        R = self.ahrs.R
        hs = []
        for a, r in zip(msg.angles, msg.ranges):
            if abs(a) < 0.08 and np.isfinite(r):
                d = R @ np.array([0.0, -math.sin(a), -math.cos(a)])
                hs.append(-r * d[2])
        if hs:
            self.lidar_agl = float(np.median(hs))
            self._lidar_t = msg.t

    def on_thermal(self, frame: ThermalFrame) -> None:
        """Night: thermal fix vs the onboard thermal prior, fused like the vision fixes (NIS gate). The GNSS-free
        track gets its own fix around its own prior first (as on_camera)."""
        if self.thermal is None or self.in_preflight or self.mission.done:
            return
        agl = self.lidar_agl if self.lidar_agl is not None and frame.t - self._lidar_t < 0.5 else None
        free = self.ekf_free
        if free is not None and self.receiving_gnss:  # (after the loss both filters fuse the same fix, below)
            ff = self.thermal.update(frame.image_K, free.pose(self.ahrs.R, self.ahrs.omega, self.ahrs.yaw, report=False),
                                     agl, frame.intrinsics, t=frame.t)
            if ff is not None:
                free.update_fix(self._thermal_visual_fix(ff, frame))
        prior = self.pose(report=False)
        t0 = time.perf_counter()
        tf = self.thermal.update(frame.image_K, prior, agl, frame.intrinsics, t=frame.t)
        self.thermal_frames += 1
        self.thermal_ms_total += (time.perf_counter() - t0) * 1000.0
        if tf is None:
            return
        fix = self._thermal_visual_fix(tf, frame)
        if not self._thermal_confirmed(fix):
            return
        if free is not None and not self.receiving_gnss:
            free.update_fix(copy.copy(fix))
        self.thermal_fixes += 1
        stats = self.thermal_by_method.setdefault(tf.method, [0, 0])
        stats[0] += 1
        accepted = self.ekf.update_fix(fix, allow_reloc=self._thermal_reloc_allowed(fix))
        self._note_thermal_result(fix, accepted)
        if accepted:
            self.thermal_accepted += 1
            stats[1] += 1
            self.sources_used.add("thermal")
            tc = self.cfg.thermal
            if tc.yaw_feedback and tf.yaw_corr and tf.quality >= tc.yaw_feedback_min_quality:
                # yaw_corr = true - estimated heading = -(residual compass bias): learn it slowly
                lim = math.radians(tc.yaw_feedback_max_deg)
                self.mag_bias_est = float(np.clip(self.mag_bias_est - tc.yaw_feedback_gain * tf.yaw_corr, -lim, lim))
        self.thermal_log.append((frame.t, tf.method, float(tf.xy[0]), float(tf.xy[1]), accepted, tf.yaw_corr))
        if self.ekf.last_decision:
            self.log_decision(frame.t, self.ekf.last_decision.replace("vision fix", "thermal fix"))
        self.health.on_fix(fix)
        self.last_thermal_fix = fix
        self.bus.publish(topics.FIX, fix)

    def night_nav_source(self, t: float) -> tuple[str, str]:
        """(active source, detail) for a HUD: GNSS while it is received, else the freshest accepted position
        source among THERMAL / TRN (within 3 s), else INS (dead reckoning). Detail: age of each source's last
        accepted fix."""
        def age(ts: float | None) -> float | None:
            return None if ts is None else t - ts
        th = self.last_thermal_fix
        tr = self.last_trn_fix
        ages = {"GNSS": age(self.ekf.last_gnss_t) if self.receiving_gnss else None,
                "THERMAL": age(th.t) if th is not None and th.accepted else None,
                "TRN": age(tr.t) if tr is not None and tr.accepted else None}
        detail = "  ".join(f"{k} {'-' if a is None else ('%.1fs' % a if a < 60 else 'lost')}" for k, a in ages.items())
        if ages["GNSS"] is not None:
            return "GNSS", detail
        fresh = [(a, k) for k, a in ages.items() if k != "GNSS" and a is not None and a < 3.0]
        return (min(fresh)[1] if fresh else "INS"), detail

    def _thermal_reloc_allowed(self, fix: VisualFix) -> bool:
        """Forced re-localization (accepting a fix after repeated NIS rejections) from thermal fixes only when the
        rejections span reloc_min_span_s and the current innovation agrees with them (thermal.reloc_min_span_s,
        0 = the EKF's plain rule: 5 rejections, only 1 s at 5 Hz)."""
        tc = self.cfg.thermal
        if tc.reloc_min_span_s <= 0:
            return True
        rej = self._thermal_rejects
        if not rej or fix.t - rej[0][0] < tc.reloc_min_span_s:
            return False
        innov = fix.pos - self.ekf.x[:2]
        recent = np.array([r[1] for r in rej[-5:]])
        return bool(np.all(np.linalg.norm(recent - innov, axis=1) <= tc.confirm_tol_m * 2))

    def _note_thermal_result(self, fix: VisualFix, accepted: bool) -> None:
        if accepted:
            self._thermal_rejects = []
        else:
            self._thermal_rejects.append((fix.t, fix.pos - self.ekf.x[:2]))
            self._thermal_rejects = self._thermal_rejects[-50:]

    def _thermal_confirmed(self, fix: VisualFix) -> bool:
        """A thermal fix that would move the estimate by more than confirm_jump_m is held until a second fix
        within confirm_window_s shows the same innovation (within confirm_tol_m). A single wrong peak (repetitive
        footways, a roof whose polarity the prior got wrong) otherwise pulls the INS off and the accel-bias
        states with it; a true offset shows up again in the next frame."""
        tc = self.cfg.thermal
        if tc.confirm_jump_m <= 0:
            return True
        innov = fix.pos - self.ekf.x[:2]
        if float(np.linalg.norm(innov)) <= tc.confirm_jump_m:
            self._thermal_pending = None
            return True
        p = self._thermal_pending
        if p is not None and fix.t - p[0] <= tc.confirm_window_s and float(np.linalg.norm(innov - p[1])) <= tc.confirm_tol_m:
            self._thermal_pending = None
            return True
        if p is not None:
            self.thermal_held += 1
        self._thermal_pending = (fix.t, innov)
        return False

    def _thermal_visual_fix(self, tf: Any, frame: ThermalFrame) -> VisualFix:
        cov = np.asarray(tf.cov, float) * self.cfg.thermal.fix_cov_scale
        return VisualFix(frame.t, np.asarray(tf.xy, float).copy(), cov, f"thermal-{tf.method}",
                         n_inliers=int(round(100 * tf.quality)), proc_ms=tf.proc_ms, frame_seq=frame.seq,
                         quality=float(tf.quality))

    def _on_gnss_lost(self, t: float) -> None:
        """GNSS fixes stopped (jam / receiver off). Hand over to the GNSS-free track when it is healthy (it never
        saw GNSS, so a spoof the naive main filter swallowed is gone at once); else inflate and re-localize."""
        c = self.cfg.ekf
        free = self.ekf_free
        if (free is not None and free.last_fix_t is not None and t - free.last_fix_t < c.handover_max_fix_age_s
                and free.sigma2_horizontal() < c.handover_max_sigma2):
            jump = self.ekf.adopt(free)
            self.n_handover += 1
            self.log_decision(t, f"HANDOVER: GNSS lost, flying on the GNSS-free vision track (moved {jump:.1f} m, "
                                 f"2-sigma {free.sigma2_horizontal():.1f} m)")
        else:
            self.log_decision(t, self.ekf.inflate_for_gnss_loss())

    # ------------------------------------------------------------------ control loop (IMU rate)

    def _control(self, t: float, dt: float) -> None:
        est = self.pose()
        ctrl_pose = self._sources[self.pose_source].get()
        if self.mission.phase == "CRUISE":
            self.health.reset_timer(t)
        hs = self.health.update(t, est, trn_tracking=self.trn_tracking)
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
        if self.mission.phase == "PREFLIGHT" or self.mission.done:
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
