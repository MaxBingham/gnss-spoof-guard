"""V3 onboard computer: inertial backbone + map-referenced fixes (TRN, IBN) + integrity + route mission.

Sensor messages + the onboard map in, motor commands out. GNSS is disabled. Never imports the simulator:
  AHRS (V2) -> NavEkf (9 states) <- integrity monitor <- TRN particle filter (altimeter vs onboard DSM)
                                                      <- IBN (down camera vs onboard orthophoto)
  route planner (update zones) -> route mission (terrain following, precision landing) -> controller (V2)
Sources can be switched on/off at runtime (`set_source`), e.g. by the live view keys T / I.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from . import bus as topics
from .bus import Bus
from .config import Config
from .control.geometric import GeometricController
from .control.mixer import VehicleParams
from .control.pose_source import PoseSource
from .estimation.ahrs import Ahrs
from .estimation.integrity import IntegrityConfig, IntegrityMonitor
from .estimation.nav_ekf import NavEkf, NavEkfConfig
from .messages import (
    AltimeterMsg, BaroMsg, CameraControl, CameraFrame, Decision, Health, HealthState, ImuMsg, MagMsg, MotorCmd,
    PoseEstimate, Setpoint, VisualFix,
)
from .perception.ibn import IbnConfig, IbnLocalizer
from .perception.trn import TrnFilter, trn_config_for
from .planning.nav_health import NavHealthMonitor, nav_health_config_for
from .planning.route_mission import RouteMission, mission_config_for
from .planning.route_planner import RoutePlanner
from .world.onboard_terrain import OnboardTerrainMap

PUBLISH_HZ = 50.0
CONTROL_HZ = 100.0
ACC_LP_TAU = 0.3  # s, low-pass of the setpoint acceleration (planned manoeuvres)
FLYING = ("TAKEOFF", "CRUISE", "HOLD", "APPROACH", "DESCEND", "LAND", "EMERGENCY_LAND")


class NavAutopilot:
    def __init__(
        self,
        cfg: Config,
        bus: Bus,
        vehicle: VehicleParams,
        onboard_map: OnboardTerrainMap,
        sources: set[str],
        start_yaw: float,
        seed: int = 0,
        plan_informed: bool = True,
        truth_pose_source: PoseSource | None = None,
        pose_source: str = "ekf",
    ) -> None:
        self.cfg = cfg
        self.bus = bus
        self.map = om = onboard_map
        sc = om.scenario
        self.scenario = sc
        self.sources = {"trn": "trn" in sources, "ibn": "ibn" in sources and om.ortho is not None}
        a = om.a_enu
        self.z0 = float(om.surface_z(a))  # take-off height in the onboard map datum (the drone sits at A)
        self.ahrs = Ahrs(cfg.ahrs, start_yaw)
        self.ekf = NavEkf(NavEkfConfig(), np.array([a[0], a[1], self.z0]))
        self.trn = TrnFilter(trn_config_for(sc), om.dsm, seed)
        ibn_cfg = IbnConfig() if sc.kind == "urban" else IbnConfig(sigma_ref=0.6, agl_ref=40.0, max_search_m=60.0)
        self.ibn = IbnLocalizer(ibn_cfg, om.ortho, om.dsm) if om.ortho is not None else None
        urban = sc.kind == "urban"
        self.integrity = IntegrityMonitor(IntegrityConfig(relocalize_blind_s=30.0 if urban else 90.0,
                                                          exclude_s=30.0 if urban else 60.0))
        for name, on in self.sources.items():
            self.integrity.src[name].enabled = on
        self.health = NavHealthMonitor(nav_health_config_for(sc))
        self.planner = RoutePlanner(om)
        self.route = self.planner.plan({s for s, on in self.sources.items() if on}, informed=plan_informed)
        self.mission = RouteMission(self.route, np.array([a[0], a[1], self.z0]), start_yaw, mission_config_for(sc))
        self.controller = GeometricController(vehicle, cfg.gains)
        self._truth_source = truth_pose_source
        self.pose_source = pose_source

        self._last_imu_t: float | None = None
        self._next_ctrl = 0.0
        self._ctrl_dt = 0.0
        self._next_project = 0.0
        self.sp_acc = np.zeros(3)  # low-passed setpoint acceleration (planned manoeuvres)
        self._sp_vel_prev = np.zeros(3)
        self._last_mag_t: float | None = None
        self._baro_sum = 0.0
        self._baro_n = 0
        self.baro_offset = 0.0
        self._next_pub = 0.0
        self._next_cam_check = 0.0
        self.camera_on = False
        self.agl: float | None = None  # vertical height above the surface from the altimeter
        self.agl_t = -1.0
        self.last_ibn_fix: tuple[float, np.ndarray] | None = None
        self.last_fix: VisualFix | None = None
        self.s_est = 0.0
        self.health_state = HealthState(0.0, Health.NOMINAL, 0.0, 0.0, 1.0)
        self.setpoint: Setpoint | None = None
        self.decisions: list[tuple[float, str]] = []
        self.fixes: deque[VisualFix] = deque(maxlen=200)  # recent fixes (display)
        self.n_fix = {"trn": 0, "ibn": 0}
        self.ibn_frames = 0
        self.ibn_ms_total = 0.0
        self.max_sigma2_cruise = 0.0
        for line in self.route.log:
            self.log_decision(0.0, line)

        bus.subscribe(topics.IMU, self.on_imu)
        bus.subscribe(topics.BARO, self.on_baro)
        bus.subscribe(topics.MAG, self.on_mag)
        bus.subscribe(topics.ALTIMETER, self.on_altimeter)
        bus.subscribe(topics.CAMERA_DOWN, self.on_camera)

    # ------------------------------------------------------------------ switches / log

    def set_source(self, name: str, on: bool) -> None:
        if name == "ibn" and self.ibn is None:
            on = False
        self.sources[name] = on
        self.integrity.src[name].enabled = on
        if name == "trn" and on:
            self.trn.p = None  # restart around the current estimate
        self.log_decision(self.ekf.t, f"SOURCE_{'ENABLED' if on else 'DISABLED'} {name} (operator)")

    def log_decision(self, t: float, text: str) -> None:
        self.decisions.append((t, text))
        self.bus.publish(topics.DECISION, Decision(t, text))

    def pose(self) -> PoseEstimate:
        return self.ekf.pose(self.ahrs.R, self.ahrs.omega, self.ahrs.yaw)

    @property
    def phase(self) -> str:
        return self.mission.phase

    def active_sources(self) -> list[str]:
        return self.integrity.available(self.ekf.t)

    # ------------------------------------------------------------------ sensors

    def on_imu(self, msg: ImuMsg) -> None:
        dt = 0.0 if self._last_imu_t is None else msg.t - self._last_imu_t
        self._last_imu_t = msg.t
        # AHRS tilt correction only in steady flight: planned AND commanded horizontal acceleration ~0. Only
        # horizontal acceleration biases the gravity reference (vertical acceleration changes |f|, not its
        # direction when level), so the |f| ~ g gate is relaxed to 1 m/s^2 (terrain following climbs all the
        # time). Not tried again: removing the EKF acceleration estimate - unaided it contains the tilt error
        # itself (cancels every correction), aided the fix updates step the EKF velocity and the derivative
        # spikes mis-level the attitude. With fixes, the EKF accel-bias state absorbs the residual tilt.
        steady = (float(np.hypot(*self.sp_acc[:2])) < 0.15
                  and float(np.hypot(*self.controller.last_acc_cmd[:2])) < 0.3)
        if self.ahrs.calibrating:
            self.ahrs.update_imu(msg)
        else:
            self.ahrs.update_imu(msg, None, allow_tilt=steady, accel_gate=1.0)
        if self.ahrs.calibrating:
            self.ekf.t = msg.t
            if msg.t >= self.cfg.mission.preflight_s:
                self.ahrs.finish_calibration()
                if self._baro_n:
                    self.baro_offset = self._baro_sum / self._baro_n - self.z0
                self.mission.start()
                self.integrity.last_any_accept_t = msg.t
        else:
            self.ekf.predict(self.ahrs.rotate_accel(msg.accel), dt, msg.t)
        if self.mission.phase in ("PREFLIGHT", "LANDED"):
            self.ekf.update_zero_velocity()
        # guidance + control at CONTROL_HZ (motor commands hold in between); estimation runs at the IMU rate
        self._ctrl_dt += dt
        if msg.t + 1e-9 >= self._next_ctrl:
            self._next_ctrl = msg.t + 1.0 / CONTROL_HZ
            self._control(msg.t, self._ctrl_dt)
            self._ctrl_dt = 0.0

    def on_baro(self, msg: BaroMsg) -> None:
        if self.mission.phase == "PREFLIGHT":
            self._baro_sum += msg.alt
            self._baro_n += 1
            return
        self.ekf.update_baro(msg.alt - self.baro_offset)

    def on_mag(self, msg: MagMsg) -> None:
        dt = 1.0 / self.cfg.sim.mag_hz if self._last_mag_t is None else msg.t - self._last_mag_t
        self._last_mag_t = msg.t
        self.ahrs.update_mag(msg, dt)

    def on_altimeter(self, msg: AltimeterMsg) -> None:
        pose = self.pose()
        if np.isfinite(msg.range):
            self.agl = float(msg.range * pose.R[2, 2])  # tilt-compensated vertical distance
            self.agl_t = msg.t
        if not (self.sources["trn"] and self.mission.phase in ("CRUISE", "HOLD", "APPROACH")):
            return
        if "trn" in self.integrity.reset_requests:
            self.integrity.reset_requests.discard("trn")
            self.trn.reset(pose, msg.t, inflate=2.0)
            self.log_decision(msg.t, "TRN_RESET: particle cloud re-initialized around the INS")
        if self.trn.needs_reset:
            self.trn.reset(pose, msg.t, inflate=2.0)
            self.log_decision(msg.t, "TRN_RESET: profile no longer matches the map (lost), re-initialized")
        fix = self.trn.step(msg, pose, self.ekf.vel_var_growth)
        if fix is not None:
            self._fuse(fix)

    def on_camera(self, frame: CameraFrame) -> None:
        if self.ibn is None or not self.sources["ibn"] or frame.name != "down" or self.mission.phase not in FLYING:
            return
        agl = self.current_agl(frame.t)
        fix = self.ibn.update(frame, self.pose(), agl)
        self.ibn_frames += 1
        self.ibn_ms_total += self.ibn.proc_ms
        if fix is not None:
            self._fuse(fix)

    def current_agl(self, t: float) -> float | None:
        return self.agl if (self.agl is not None and t - self.agl_t < 0.5) else None

    def _fuse(self, fix: VisualFix) -> None:
        n_ev = len(self.integrity.events)
        ok = self.integrity.process(fix, self.ekf)
        for ev in self.integrity.events[n_ev:]:
            self.log_decision(*ev)
        self.n_fix[fix.source] += 1
        self.fixes.append(fix)
        self.last_fix = fix
        if ok and fix.source == "ibn":
            self.last_ibn_fix = (fix.t, fix.pos.copy())
            self._check_trn_against(fix)
        self.bus.publish(topics.FIX, fix)

    def _check_trn_against(self, fix: VisualFix) -> None:
        """After an accepted image fix: a TRN cloud that contradicts the corrected INS is stale (it drifted with
        the INS velocity before the fix) - re-center it instead of letting it pull the INS back later."""
        trn = self.trn
        if not self.sources["trn"] or trn.est is None or trn.est_cov is None:
            return
        d = trn.est - self.ekf.x[:2]
        S = trn.est_cov + self.ekf.P[:2, :2] + np.eye(2)
        d2 = float(d @ np.linalg.solve(S, d))
        if d2 > self.integrity.cfg.cross_chi2:
            trn.reset(self.pose(), fix.t, inflate=1.5)
            self.log_decision(fix.t, f"TRN_RESET: cloud {np.hypot(*d):.0f} m from the IBN-corrected INS, re-centered")

    # ------------------------------------------------------------------ control loop

    def _camera_wanted(self, pose: PoseEstimate) -> bool:
        if self.ibn is None or not self.sources["ibn"] or self.integrity.excluded("ibn", pose.t):
            return False
        ph = self.mission.phase
        if ph in ("APPROACH", "DESCEND", "LAND"):
            return True
        if ph not in ("CRUISE", "HOLD"):
            return False
        if not self.ibn.covers(pose.pos[:2], 20.0):
            return False
        return (ph == "HOLD" or self.route.zone_at(self.s_est, "ibn") or self.scenario.ibn_everywhere
                and self.health.state != Health.NOMINAL)

    def _control(self, t: float, dt: float) -> None:
        est = self.pose()
        ctrl_pose = est if (self.pose_source == "ekf" or self._truth_source is None) else self._truth_source.get()
        if self.mission.phase in ("CRUISE", "HOLD") and t >= self._next_project:
            self._next_project = t + 0.05
            self.s_est = self.route.project(est.pos[:2], self.s_est)
            self.max_sigma2_cruise = max(self.max_sigma2_cruise, est.sigma2_horizontal())
        if t >= self._next_cam_check:
            self._next_cam_check = t + 0.2
            want = self._camera_wanted(est)
            if want != self.camera_on:
                self.camera_on = want
                self.bus.publish(topics.CAMERA_CONTROL, CameraControl(t, want))
        avail = self.active_sources()
        ibn_possible = ("ibn" in avail and self.ibn is not None and self.ibn.covers(est.pos[:2], 20.0))
        active = self.mission.phase in ("CRUISE", "HOLD")
        prev = self.health.state
        hs = self.health.update(t, est, len(avail), ibn_possible, self.integrity.last_any_accept_t, active)
        if hs.state != prev:
            self.log_decision(t, f"HEALTH {hs.state.value}: {hs.reason}")
        self.health_state = hs
        enforce = self.pose_source == "ekf"
        sp = self.mission.update(t, ctrl_pose, dt, hs.state if enforce else Health.NOMINAL,
                                 hs.speed_scale if enforce else 1.0, self.current_agl(t), self.last_ibn_fix,
                                 ibn_available=self.ibn is not None and self.sources["ibn"]
                                 and self.ibn.covers(self.mission.b, 20.0))
        if dt > 0:
            k = min(dt / ACC_LP_TAU, 1.0)
            self.sp_acc += k * ((sp.vel - self._sp_vel_prev) / dt - self.sp_acc)
            self._sp_vel_prev = sp.vel.copy()
        self.setpoint = sp
        while self.mission.events:
            self.log_decision(*self.mission.events.pop(0))
        motors = np.zeros(4) if self.mission.phase in ("PREFLIGHT", "LANDED") else self.controller(ctrl_pose, sp)
        self.bus.publish(topics.MOTOR, MotorCmd(t, motors))
        if t >= self._next_pub:
            self._next_pub = t + 1.0 / PUBLISH_HZ
            self.bus.publish(topics.POSE, est)
            self.bus.publish(topics.SETPOINT, sp)
            self.bus.publish(topics.HEALTH, hs)

    # ------------------------------------------------------------------ display helpers

    def status_line(self) -> str:
        est = self.pose()
        src = "+".join(["INS"] + [s.upper() for s in self.active_sources()])
        return (f"{self.mission.phase:14s} {self.health_state.state.value:8s} {src:11s} 2s={est.sigma2_horizontal():6.1f} m "
                f"trn:{self.trn.status[:28]:28s} info {self.trn.info:4.1f} m")

    def expected_time(self) -> float:
        return self.route.length / max(self.scenario.cruise_speed, 1.0) + 120.0 + math.sqrt(self.route.length)
