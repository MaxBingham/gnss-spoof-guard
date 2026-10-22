"""V3 mission: follow the planned route (terrain following), then a precision landing at B.

Phases: PREFLIGHT -> TAKEOFF -> CRUISE (route, HOLD on health) -> APPROACH (hover above B until IBN confirms
the position) -> DESCEND -> LAND -> LANDED; EMERGENCY_LAND on health LAND.
- CRUISE: the setpoint advances along the route at the route's speed limit x health speed scale, never more
  than `max_lead` ahead of the estimate's projection on the route. Height = planned profile (onboard DSM +
  cruise height) + a terrain-clearance offset raised whenever the altimeter measures less than `min_clearance`.
- APPROACH: descend only when IBN confirms the position (an accepted IBN fix within `confirm_m` of the estimate,
  2-sigma <= `precision_sigma2`); otherwise after `precision_wait_s` (logged: no confirmation).
- DESCEND / LAND: heights relative to the ground measured by the altimeter, near the ground by the landing
  rangefinder; RECENTER pauses the descent while the estimate is > `descend_xy_tol` off B. Touchdown (V2 port):
  rangefinder < `touchdown_range` -> GROUND_CONTACT (motors idle), still low and IMU stationary for
  `touchdown_s` -> motors off, LANDED (the autopilot then applies zero-velocity updates).
All decisions use the estimate and onboard sensors only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..messages import Health, PoseEstimate, Setpoint
from ..world.scenarios import Scenario
from .route_planner import Route


@dataclass
class RouteMissionConfig:
    climb_speed: float = 3.0
    descent_speed: float = 2.0
    # V2: image fixes stop at ~4-5 m (the footprint gets too small): flare low and land briskly so the final
    # blind stretch is short (6 m at 0.7 m/s = 9 s drifted 4-10 m in V2)
    land_speed: float = 1.0
    flare_agl: float = 4.0
    descend_xy_tol: float = 1.5  # m, DESCEND / LAND pause while the estimate is farther than this from B
    max_lead: float = 12.0  # m, setpoint lead over the estimate along the route
    min_clearance: float = 15.0  # m, altimeter-measured height above the surface that triggers a climb
    clearance_climb: float = 3.0  # m/s
    decel: float = 1.5  # m/s^2 towards the end of the route
    accel: float = 1.5  # m/s^2, setpoint speed ramp (no step to cruise speed)
    precision_sigma2: float = 2.0  # m
    confirm_m: float = 2.0  # m, IBN fix vs estimate for 'confirmed'
    confirm_window_s: float = 2.0
    precision_wait_s: float = 30.0
    touchdown_range: float = 0.3  # m, landing rangefinder height (at-rest reading removed)
    touchdown_vz: float = 0.3  # (fallback without rangefinder)
    touchdown_s: float = 0.5
    ground_idle: float = 0.3  # fraction of hover thrust, uniform on all motors, while in ground contact
    still_acc: float = 0.3  # m/s^2, IMU stationary: | |a| - g | below this ...
    still_rate: float = 0.2  # rad/s, ... and |omega| below this
    yaw_rate: float = 0.5  # rad/s


def mission_config_for(sc: Scenario) -> RouteMissionConfig:
    if sc.kind == "urban":
        return RouteMissionConfig()
    return RouteMissionConfig(climb_speed=5.0, descent_speed=4.0, max_lead=40.0, min_clearance=50.0,
                              clearance_climb=5.0, precision_sigma2=3.0, confirm_m=3.0, precision_wait_s=40.0)


class RouteMission:
    def __init__(self, route: Route, start_pos: np.ndarray, yaw: float, cfg: RouteMissionConfig) -> None:
        self.route = route
        self.cfg = cfg
        self.phase = "PREFLIGHT"
        self.setpoint = np.asarray(start_pos, float).copy()
        self.yaw = yaw
        self.s = 0.0  # setpoint progress along the route
        self.z_offset = 0.0  # terrain-clearance offset added to the profile
        self.ground_est: float | None = None
        self.approach_since: float | None = None
        self.landed_timer = 0.0
        self.events: list[tuple[float, str]] = []
        self.arrival_t: float | None = None
        self.hold_logged = False
        self.v = 0.0  # current setpoint speed along the route
        self.range_agl: float | None = None  # landing rangefinder (set_ground_sensing)
        self.imu_still_s = 0.0
        self.ground_contact = False  # rangefinder says we are down: the autopilot idles the motors
        self.recentering = False

    @property
    def done(self) -> bool:
        return self.phase == "LANDED"

    @property
    def b(self) -> np.ndarray:
        return self.route.xy[-1]

    def log(self, t: float, text: str) -> None:
        self.events.append((t, text))

    def start(self) -> None:
        if self.phase == "PREFLIGHT":
            self.phase = "TAKEOFF"

    def _yaw_towards(self, target: float, dt: float) -> float:
        err = math.atan2(math.sin(target - self.yaw), math.cos(target - self.yaw))
        self.yaw += float(np.clip(err, -self.cfg.yaw_rate * dt, self.cfg.yaw_rate * dt))
        return self.yaw

    def _ground(self, pose: PoseEstimate, agl: float | None) -> None:
        if agl is not None:
            g = float(pose.pos[2] - agl)
            self.ground_est = g if self.ground_est is None else 0.8 * self.ground_est + 0.2 * g

    def update(self, t: float, pose: PoseEstimate, dt: float, health: Health, speed_scale: float,
               agl: float | None, ibn_fix: tuple[float, np.ndarray] | None, ibn_available: bool) -> Setpoint:
        c = self.cfg
        zero = np.zeros(3)
        if self.phase in ("PREFLIGHT", "LANDED"):
            return Setpoint(t, self.setpoint.copy(), zero, self.yaw, self.phase)
        self._ground(pose, agl)

        if health == Health.LAND and self.phase in ("TAKEOFF", "CRUISE", "HOLD", "APPROACH"):
            self.phase = "EMERGENCY_LAND"
            self.log(t, "EMERGENCY_LAND: navigation health LAND, landing in place")

        if self.phase == "TAKEOFF":
            xy0, z0, _, tan = self.route.at(0.0)
            target = np.array([self.setpoint[0], self.setpoint[1], z0])
            vel = self._move_vertical(target[2], c.climb_speed, dt)
            if pose.pos[2] > z0 - 2.0:
                self.phase = "CRUISE"
                self.log(t, f"CRUISE: route {self.route.length:.0f} m, {len(self.route.zones)} update zones")
            return Setpoint(t, self.setpoint.copy(), vel, self._yaw_towards(math.atan2(tan[1], tan[0]), dt), "TAKEOFF")

        if self.phase in ("CRUISE", "HOLD"):
            if health == Health.HOLD:
                if not self.hold_logged:
                    self.log(t, "HOLD: hovering, waiting for an image fix")
                    self.hold_logged = True
                self.phase = "HOLD"
                self.v = 0.0
                return Setpoint(t, self.setpoint.copy(), zero, self.yaw, "HOLD")
            if self.phase == "HOLD":
                self.phase = "CRUISE"
                self.hold_logged = False
            return self._cruise(t, pose, dt, speed_scale, agl)

        if self.phase == "APPROACH":
            return self._approach(t, pose, dt, ibn_fix, ibn_available)

        if self.phase in ("DESCEND", "LAND", "EMERGENCY_LAND"):
            return self._descend(t, pose, dt, agl)
        return Setpoint(t, self.setpoint.copy(), zero, self.yaw, self.phase)

    def _move_vertical(self, z_target: float, speed: float, dt: float) -> np.ndarray:
        dz = z_target - self.setpoint[2]
        step = speed * dt
        if abs(dz) > step:
            self.setpoint[2] += math.copysign(step, dz)
            return np.array([0.0, 0.0, math.copysign(speed, dz)])
        self.setpoint[2] = z_target
        return np.zeros(3)

    def _cruise(self, t: float, pose: PoseEstimate, dt: float, scale: float, agl: float | None) -> Setpoint:
        c = self.cfg
        r = self.route
        s_est = r.project(pose.pos[:2], self.s)
        _, _, vlim, _ = r.at(self.s)
        remaining = r.length - self.s
        v = min(vlim * scale, math.sqrt(2 * c.decel * max(remaining, 0.0)) + 0.3, self.v + c.accel * dt)
        if self.s - s_est > c.max_lead:
            v = max(self.v - c.decel * dt, 0.0)
        self.v = v
        self.s = min(self.s + v * dt, r.length)
        xy, z, _, tan = r.at(self.s)
        # terrain clearance from the altimeter (raise fast, relax slowly)
        if agl is not None and agl < c.min_clearance:
            if self.z_offset == 0.0:
                self.log(t, f"TERRAIN_CLEARANCE: altimeter {agl:.1f} m < {c.min_clearance:.0f} m, climbing")
            self.z_offset += c.clearance_climb * dt
        elif self.z_offset > 0 and (agl is None or agl > c.min_clearance * 1.5):
            self.z_offset = max(0.0, self.z_offset - 0.3 * dt)
        _, z_next, _, _ = r.at(self.s + 1.0)
        vz = (z_next - z) * v
        self.setpoint = np.array([xy[0], xy[1], z + self.z_offset])
        vel = np.array([tan[0] * v, tan[1] * v, vz])
        yaw = self._yaw_towards(math.atan2(tan[1], tan[0]), dt)
        if self.s >= r.length - 1e-6:
            self.phase = "APPROACH"
            self.approach_since = t
            self.arrival_t = t
            self.log(t, f"ARRIVED above B (estimate), 2-sigma {pose.sigma2_horizontal():.1f} m: precision check")
        return Setpoint(t, self.setpoint.copy(), vel, yaw, "CRUISE")

    def _approach(self, t: float, pose: PoseEstimate, dt: float, ibn_fix: tuple[float, np.ndarray] | None,
                  ibn_available: bool) -> Setpoint:
        c = self.cfg
        self.setpoint[:2] = self.b
        s2 = pose.sigma2_horizontal()
        waited = t - (self.approach_since or t)
        confirmed = (ibn_fix is not None and t - ibn_fix[0] <= c.confirm_window_s
                     and float(np.hypot(*(ibn_fix[1] - pose.pos[:2]))) <= c.confirm_m and s2 <= c.precision_sigma2)
        if confirmed:
            off = float(np.hypot(*(ibn_fix[1] - pose.pos[:2])))
            self.log(t, f"DESCEND: IBN confirms position (fix offset {off:.1f} m, 2-sigma {s2:.1f} m) after {waited:.1f} s")
            self.phase = "DESCEND"
        elif not ibn_available or waited >= c.precision_wait_s:
            why = "no IBN available" if not ibn_available else f"no IBN confirmation after {waited:.0f} s"
            self.log(t, f"DESCEND without confirmation ({why}), 2-sigma {s2:.1f} m")
            self.phase = "DESCEND"
        return Setpoint(t, self.setpoint.copy(), np.zeros(3), self.yaw, "APPROACH")

    def set_ground_sensing(self, range_agl: float | None, imu_still_s: float) -> None:
        """Landing rangefinder height (at-rest reading removed, None = no reading) and how long the IMU has been
        stationary; set by the autopilot every control step."""
        self.range_agl = range_agl
        self.imu_still_s = imu_still_s

    def _descend(self, t: float, pose: PoseEstimate, dt: float, agl: float | None) -> Setpoint:
        c = self.cfg
        if self.phase != "EMERGENCY_LAND":
            self.setpoint[:2] = self.b
        if self.range_agl is not None:  # precise near the ground (the radar altimeter is not)
            self.ground_est = float(pose.pos[2] - self.range_agl)
        ground = self.ground_est if self.ground_est is not None else pose.pos[2] - 30.0
        h = pose.pos[2] - ground
        off = float(np.hypot(*(self.b - pose.pos[:2])))
        if self.phase in ("DESCEND", "LAND") and off > c.descend_xy_tol and not self.ground_contact:
            # a late fix moved the estimate (or wind pushed the drone): re-center first, then continue down
            if not self.recentering:
                self.recentering = True
                self.log(t, f"RECENTER: {off:.1f} m off B horizontally, descent paused")
            vel = np.zeros(3)
        else:
            self.recentering = False
            if self.phase == "DESCEND" and h < c.flare_agl:
                self.phase = "LAND"
            speed = c.land_speed if (self.phase == "LAND" or h < c.flare_agl) else c.descent_speed
            vel = self._move_vertical(ground - 0.3, speed, dt)
        self._check_touchdown(t, pose, dt, agl, h)
        return Setpoint(t, self.setpoint.copy(), vel, self.yaw, self.phase)

    def _check_touchdown(self, t: float, pose: PoseEstimate, dt: float, agl: float | None, h: float) -> None:
        """V2 port. Touchdown = rangefinder height < touchdown_range AND IMU stationary, both for touchdown_s
        (independent of the estimated height). In ground contact the autopilot idles the motors: with position
        control on, the drone hopped and skidded chasing its estimate and the IMU never looked still. Without a
        rangefinder reading: altimeter / estimated height and |vz| small."""
        c = self.cfg
        if self.range_agl is not None:
            low = self.range_agl < c.touchdown_range
            if low and not self.ground_contact:
                self.log(t, f"GROUND_CONTACT: rangefinder {self.range_agl:.2f} m, motors to idle")
            self.ground_contact = low
            still = self.imu_still_s >= c.touchdown_s
            how = f"rangefinder {self.range_agl:.2f} m, IMU still {self.imu_still_s:.1f} s"
        else:
            self.ground_contact = False
            low = ((agl is not None and agl < c.touchdown_range) or (agl is None and h < c.touchdown_range)) \
                and abs(pose.vel[2]) < c.touchdown_vz
            still = True
            how = "no rangefinder: altimeter / estimated height"
        self.landed_timer = self.landed_timer + dt if low else 0.0
        if self.landed_timer >= c.touchdown_s and still:
            self.phase = "LANDED"
            self.ground_contact = False
            self.log(t, f"TOUCHDOWN ({how}): motors off at estimated ({pose.pos[0]:.1f}, {pose.pos[1]:.1f}), "
                        f"2-sigma {pose.sigma2_horizontal():.1f} m")
