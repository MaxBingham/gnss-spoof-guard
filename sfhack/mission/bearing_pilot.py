"""Bearing recon pilot (onboard): the UMA-16 at the base gives a bearing with the launch command; the drone takes off
from the pad, flies out along the bearing, orbits the target point, flies home and lands, all on its own estimate.

    WAIT -> TAKEOFF -> OUTBOUND -> ORBIT -> RETURN -> DESCEND -> LANDED

- Estimate: the repo's AHRS (gyro + accelerometer + compass) and EKF (IMU predict, barometer, GNSS while there
  is any, zero-velocity updates on the ground), wired as in autopilot.py: AHRS calibration on the ground for
  `preflight_s`, baro offset from the same window, accel-bias states handed to the AHRS every IMU step.
- Control: GeometricController on the EKF pose, at the IMU rate. When the GNSS fixes stop the pilot keeps flying
  on IMU + compass + baro dead reckoning; RETURN goes to where the estimate says the pad is and lands there.
- Inputs: sensor messages on the bus, and `launch(az_deg)` = the launch command from the base (the only thing the
  base ever gives; there is no data link). The pad and the array position are surveyed before the flight.
- Touchdown without a rangefinder: estimated height below `touchdown_alt`, or the descent stalls (setpoint well
  below, vertical speed ~0) near the ground for `stall_s`; then uniform idle thrust for `idle_s`, motors off.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .. import bus as topics
from ..bus import Bus
from ..config import Config
from ..control.geometric import GeometricController
from ..control.mixer import VehicleParams
from ..estimation.ahrs import Ahrs
from ..estimation.ekf import Ekf
from ..messages import BaroMsg, Decision, GnssMsg, ImuMsg, MagMsg, MotorCmd, PoseEstimate, Setpoint

STATES = ("WAIT", "TAKEOFF", "OUTBOUND", "ORBIT", "RETURN", "DESCEND", "LANDED")


@dataclass
class BearingPlan:
    range_m: float = 400.0  # how far out along the bearing the target point lies (from the array)
    cruise_agl: float = 30.0
    out_speed: float = 8.0
    climb: float = 2.5
    descent: float = 1.0
    orbit_r: float = 35.0
    orbit_s: float = 25.0
    bearing_sigma_deg: float = 6.0  # the array's 1-sigma, display only
    touchdown_alt: float = 0.35  # m, estimated height above the takeoff point
    stall_s: float = 0.5
    idle_s: float = 1.0


def az_to_dir(az_deg: float) -> np.ndarray:
    """Compass azimuth (deg clockwise from north) -> ENU unit vector."""
    a = math.radians(az_deg)
    return np.array([math.sin(a), math.cos(a)])


def orbit_entry(plan: BearingPlan, mast_xy: np.ndarray, az_deg: float) -> tuple[np.ndarray, float, float]:
    """Orbit centre (target point), the start angle (near side, toward the base) and the angular rate."""
    c = np.asarray(mast_xy, float) + az_to_dir(az_deg) * plan.range_m
    a0 = math.radians(90.0 - az_deg) + math.pi
    return c, a0, plan.out_speed * 0.6 / plan.orbit_r


def planned_route(plan: BearingPlan, pad_xy: np.ndarray, mast_xy: np.ndarray, az_deg: float,
                  z0: float = 0.0, n_orbit: int = 48) -> dict[str, np.ndarray]:
    """The ideal flight as polylines (takeoff, outbound, orbit, return, descent); z above the takeoff point + z0."""
    pad = np.asarray(pad_xy, float)
    c, a0, w = orbit_entry(plan, mast_xy, az_deg)
    zc = z0 + plan.cruise_agl
    ang = a0 + np.linspace(0.0, w * plan.orbit_s, n_orbit)
    orbit = np.column_stack([c[0] + plan.orbit_r * np.cos(ang), c[1] + plan.orbit_r * np.sin(ang), np.full(n_orbit, zc)])
    up = np.array([[pad[0], pad[1], z0], [pad[0], pad[1], zc]])
    return {"takeoff": up, "outbound": np.array([up[1], orbit[0]]), "orbit": orbit,
            "return": np.array([orbit[-1], up[1]]), "descent": up[::-1].copy(), "target": np.array([c[0], c[1], zc])}


class BearingPilot:
    def __init__(self, cfg: Config, bus: Bus, vehicle: VehicleParams, pad_xy, plan: BearingPlan | None = None,
                 mast_xy=(0.0, 0.0), start_yaw: float = 0.0) -> None:
        self.cfg, self.bus = cfg, bus
        self.plan = plan or BearingPlan()
        self.pad = np.asarray(pad_xy, float)[:2].copy()
        self.mast = np.asarray(mast_xy, float)[:2].copy()
        self.ahrs = Ahrs(cfg.ahrs, start_yaw)
        self.ekf = Ekf(cfg.ekf, np.array([self.pad[0], self.pad[1], 0.0]))
        self.controller = GeometricController(vehicle, cfg.gains)
        self.idle_thrust = cfg.mission.ground_idle * vehicle.mass * vehicle.g / 4
        self.state = "WAIT"
        self.t_state = 0.0
        self.bearing: float | None = None
        self.pending_launch: float | None = None  # launch command received while still calibrating
        self.sp = np.array([self.pad[0], self.pad[1], 0.0])
        self.v = np.zeros(3)
        self.yaw = start_yaw
        self.ground_contact_t: float | None = None
        self._stall_s = 0.0
        self._t0: float | None = None
        self._last_imu_t: float | None = None
        self._last_mag_t: float | None = None
        self._baro_sum, self._baro_n, self.baro_offset = 0.0, 0, 0.0
        self.motors = np.zeros(4)
        self.setpoint: Setpoint | None = None
        self.decisions: list[tuple[float, str]] = []
        bus.subscribe(topics.IMU, self.on_imu)
        bus.subscribe(topics.BARO, self.on_baro)
        bus.subscribe(topics.MAG, self.on_mag)
        bus.subscribe(topics.GNSS, self.on_gnss)

    # ------------------------------------------------------------------ interface

    @property
    def calibrating(self) -> bool:
        return self.ahrs.calibrating

    @property
    def on_ground(self) -> bool:
        return self.state in ("WAIT", "LANDED")

    def pose(self, *, report: bool = True) -> PoseEstimate:
        return self.ekf.pose(self.ahrs.R, self.ahrs.omega, self.ahrs.yaw, report=report)

    def log(self, t: float, text: str) -> None:
        self.decisions.append((t, text))
        self.bus.publish(topics.DECISION, Decision(t, text))

    def launch(self, az_deg: float, t: float) -> bool:
        """Launch command from the base (bearing in compass degrees). Accepted only on the ground: no data link."""
        if not self.on_ground:
            return False
        if self.calibrating:
            self.pending_launch = az_deg % 360.0
            self.log(t, f"LAUNCH {self.pending_launch:05.1f} deg received, waiting for the IMU calibration")
            return True
        self.bearing = az_deg % 360.0
        self.sp = np.array([*self.ekf.x[:2], max(float(self.ekf.x[2]), 0.0)])
        self.v = np.zeros(3)
        self._go("TAKEOFF", t, f"bearing {self.bearing:05.1f} deg")
        return True

    def retarget(self, az_deg: float, t: float) -> None:
        """Demo flag only (--retarget-in-flight): a bearing reaching the drone in the air would need a data link."""
        if self.on_ground:
            self.launch(az_deg, t)
            return
        self.bearing = az_deg % 360.0
        self._go("OUTBOUND", t, f"RETARGET {self.bearing:05.1f} deg")

    def target(self) -> np.ndarray | None:
        if self.bearing is None:
            return None
        c, _, _ = orbit_entry(self.plan, self.mast, self.bearing)
        return np.array([c[0], c[1], self.plan.cruise_agl])

    def route(self, z0: float = 0.0) -> dict[str, np.ndarray] | None:
        return None if self.bearing is None else planned_route(self.plan, self.pad, self.mast, self.bearing, z0)

    # ------------------------------------------------------------------ sensors

    def on_imu(self, msg: ImuMsg) -> None:
        if self._t0 is None:
            self._t0 = msg.t
        dt = 0.0 if self._last_imu_t is None else msg.t - self._last_imu_t
        self._last_imu_t = msg.t
        self.ahrs.update_imu(msg)
        if self.ahrs.calibrating:
            self.ekf.t = msg.t
            if msg.t - self._t0 >= self.cfg.mission.preflight_s:
                self.ahrs.finish_calibration()
                if self._baro_n:
                    self.baro_offset = self._baro_sum / self._baro_n
                self.log(msg.t, "PREFLIGHT done: IMU calibrated on the pad, baro zeroed")
                if self.pending_launch is not None:
                    az, self.pending_launch = self.pending_launch, None
                    self.launch(az, msg.t)
        else:
            self.ekf.predict(self.ahrs.rotate_accel(msg.accel), dt, msg.t)
        if self.on_ground or self.ground_contact_t is not None:
            self.ekf.update_zero_velocity()
        if not self.ahrs.calibrating:
            self.ahrs.correct_tilt_world(self.ekf.take_tilt_correction())
        if self.ekf.gnss_lost_now(msg.t):
            self.log(msg.t, "GNSS LOST: flying on IMU + compass + baro dead reckoning")
        self._control(msg.t, dt)

    def on_baro(self, msg: BaroMsg) -> None:
        if self.ahrs.calibrating:
            self._baro_sum += msg.alt
            self._baro_n += 1
            return
        self.ekf.update_baro(msg.alt - self.baro_offset)

    def on_mag(self, msg: MagMsg) -> None:
        dt = 1.0 / self.cfg.sim.mag_hz if self._last_mag_t is None else msg.t - self._last_mag_t
        self._last_mag_t = msg.t
        self.ahrs.update_mag(msg, dt)

    def on_gnss(self, msg: GnssMsg) -> None:
        self.ekf.update_gnss(msg.t, msg.pos, msg.cov)

    # ------------------------------------------------------------------ mission + control (IMU rate)

    def _go(self, s: str, t: float, why: str = "") -> None:
        self.state, self.t_state = s, t
        self.log(t, s + (f": {why}" if why else ""))

    def _move_to(self, goal: np.ndarray, speed: float, dt: float) -> bool:
        d = goal - self.sp
        n = float(np.linalg.norm(d))
        if n <= speed * dt:
            self.sp, self.v = goal.copy(), np.zeros(3)
            return True
        self.v = d / n * speed
        self.sp = self.sp + self.v * dt
        if np.linalg.norm(self.v[:2]) > 0.5:
            self.yaw = math.atan2(self.v[1], self.v[0])
        return False

    def _update_mission(self, est: PoseEstimate, t: float, dt: float) -> None:
        p, pl, s = est.pos, self.plan, self.state
        cruise = np.array([self.pad[0], self.pad[1], pl.cruise_agl])
        if s == "TAKEOFF":
            top = np.array([self.sp[0], self.sp[1], pl.cruise_agl])
            if self._move_to(top, pl.climb, dt) and abs(p[2] - pl.cruise_agl) < 1.0:
                self._go("OUTBOUND", t)
        elif s == "OUTBOUND":
            c, a0, _ = orbit_entry(pl, self.mast, self.bearing)
            entry = np.array([c[0] + pl.orbit_r * math.cos(a0), c[1] + pl.orbit_r * math.sin(a0), pl.cruise_agl])
            if self._move_to(entry, pl.out_speed, dt) and np.linalg.norm(p[:2] - entry[:2]) < 3.0:
                self._go("ORBIT", t, "at the target point (on the estimate)")
        elif s == "ORBIT":
            c, a0, w = orbit_entry(pl, self.mast, self.bearing)
            a = a0 + w * (t - self.t_state)
            self.sp = np.array([c[0] + pl.orbit_r * math.cos(a), c[1] + pl.orbit_r * math.sin(a), pl.cruise_agl])
            self.v = pl.orbit_r * w * np.array([-math.sin(a), math.cos(a), 0.0])
            self.yaw = math.atan2(c[1] - self.sp[1], c[0] - self.sp[0])  # look at the centre
            if t - self.t_state > pl.orbit_s:
                self._go("RETURN", t)
        elif s == "RETURN":
            if self._move_to(cruise, pl.out_speed, dt) and np.linalg.norm(p[:2] - self.pad) < 1.5:
                self._go("DESCEND", t, "over the pad (on the estimate)")
        elif s == "DESCEND":
            self._move_to(np.array([self.pad[0], self.pad[1], -1.0]), pl.descent, dt)
            if self.ground_contact_t is None:
                stalled = p[2] < 2.0 and self.sp[2] < p[2] - 0.4 and est.vel[2] > -0.2
                self._stall_s = self._stall_s + dt if stalled else 0.0
                if p[2] < pl.touchdown_alt or self._stall_s >= pl.stall_s:
                    self.ground_contact_t = t
                    self._pin_position()
                    self.log(t, f"GROUND CONTACT (est height {p[2]:.2f} m): motors to idle")
            elif t - self.ground_contact_t >= pl.idle_s:
                self.ground_contact_t = None
                self._stall_s = 0.0
                self._go("LANDED", t, "motors off")

    def _pin_position(self) -> None:
        """Touchdown: the zero-velocity updates that follow must stop the drone in the estimate, not rewrite where it
        flew. After a long blind stretch P couples position to velocity by ~ seconds blind, and the touchdown shock
        (and any skid) makes the IMU velocity meaningless, so a ZUPT then threw the position hundreds of m away."""
        P = self.ekf.P
        P[:3, 3:] = 0.0
        P[3:, :3] = 0.0
        self.ekf.x[3:6] = 0.0

    def _control(self, t: float, dt: float) -> None:
        est = self.pose()
        if not self.on_ground and dt > 0:
            self._update_mission(est, t, dt)
        self.setpoint = Setpoint(t, self.sp.copy(), self.v.copy(), self.yaw, self.state)
        if self.on_ground:
            self.motors = np.zeros(4)
        elif self.ground_contact_t is not None:
            self.motors = np.full(4, self.idle_thrust)
        else:
            self.motors = self.controller(est, self.setpoint)
        self.bus.publish(topics.MOTOR, MotorCmd(t, self.motors.copy()))
