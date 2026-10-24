"""Enemy-drone tracker pilot (onboard): the UMA-16 at the base hears an enemy drone; our drone takes off, finds it
with the gimballed front camera, follows it at a standoff and reports it to the base. Track and report only: it
never closes in (never nearer than `min_dist_m`), no interception.

    WAIT -> TAKEOFF -> SEARCH -> DETECTED -> TRACKING <-> LOST -> ... -> RETURN -> DESCEND -> LANDED

- SEARCH: the base's assumption: at the first sound the enemy was `assumed_range_m` out along the bearing, then it
  approaches at `assumed_speed`. Fly toward that (moving) point along the LATEST bearing, `search_agl` above the
  terrain map, the gimbal sweeping around it; once there, circle it. A new bearing (over the link) re-aims.
- DETECTED: a drone detection >= `drone_conf`: slow down, the gimbal locks on the track. `confirm_hits` hits in
  the last `confirm_window` frames -> TRACKING; none within `detected_timeout_s` -> back where it came from.
- TRACKING: hold `track_dist_m` horizontally from the enemy's track (on our side), `track_above_m` above it, the
  enemy centred by the gimbal, the track velocity fed forward; nearer than `min_dist_m` -> back off at once.
  Reports to the base every `report_every_s` (position, altitude, speed, heading, confidence, a cropped snapshot),
  buffered while the link is down. No hit for `lost_after_s` -> LOST.
  Plausibility: the base's UMA-16 bearing (~1 Hz while the link is up) must agree with the track; more than
  `suspect_sigma` sigma off for `suspect_s` -> SUSPECT (a bird / wrong object): drop the track, LOST, re-acquire.
- LOST: the predicted enemy position = the point on the LATEST base bearing at (track distance from the base when
  lost - assumed_speed * time since lost); until a bearing newer than the loss arrives (link down): constant-velocity
  prediction from the last track. Watch that point from `lost_standoff_m` on the base side, sweeping forward /
  backward along the bearing line (`lost_pattern_m`) and with the gimbal; a new detection (inside the bearing gate)
  -> DETECTED, `lost_search_s` -> SEARCH along the latest bearing.
- RETURN when the battery budget (`battery_s` flight time) is down to the way home + `reserve_s`, or on `recall`.

Everything here runs on the estimate (BearingPilot's AHRS + EKF, GNSS until the operator switches it off) and on the
detector output; enemy position = own estimate + camera ray * range from apparent size. No truth.
"""

from __future__ import annotations

import math
from typing import Callable, Protocol

import numpy as np

from .. import bus as topics
from ..bus import Bus
from ..config import Config, TrackerConfig
from ..control.mixer import VehicleParams
from ..messages import BearingUpdate, GimbalCmd, GimbalFrame, PoseEstimate
from ..perception.drone_detector import Det, EnemyTracker, FrameResult, pointing
from .bearing_pilot import BearingPilot, BearingPlan, az_to_dir, flat
from .enemy_report import EnemyReport, ReportOutbox, snapshot

TRACK_STATES = ("SEARCH", "DETECTED", "TRACKING", "LOST")
STATES = ("WAIT", "TAKEOFF", *TRACK_STATES, "RETURN", "DESCEND", "LANDED")


class Detector(Protocol):
    def detect(self, rgb: np.ndarray) -> list[Det]: ...


class TrackerPilot(BearingPilot):
    def __init__(self, cfg: Config, bus: Bus, vehicle: VehicleParams, pad_xy, tcfg: TrackerConfig | None = None,
                 mast_xy=(0.0, 0.0), terrain: Callable = flat, detector: Detector | None = None,
                 start_yaw: float = 0.0) -> None:
        tc = self.tcfg = tcfg or TrackerConfig()
        plan = BearingPlan(range_m=tc.assumed_range_m, cruise_agl=tc.search_agl, out_speed=tc.home_speed)
        super().__init__(cfg, bus, vehicle, pad_xy, plan, mast_xy, start_yaw, terrain)
        self.detector = detector
        self.tracker = EnemyTracker(tc)
        self.outbox = ReportOutbox()
        self.first_heard_t: float | None = None
        self.r_anchor = (tc.assumed_range_m, 0.0)  # (range from the array, time): the assumed enemy range model
        self.takeoff_t: float | None = None
        self.pan_cmd = self.tilt_cmd = 0.0
        self._next_gimbal = 0.0
        self._next_report = 0.0
        self.report_seq = 0
        self.resume_state = "SEARCH"  # DETECTED falls back here without confirmation
        self.lost_xy: np.ndarray | None = None
        self.lost_z = 0.0
        self.last_frame: GimbalFrame | None = None
        self.last_result: FrameResult | None = None
        self.last_snapshot: tuple[np.ndarray, bytes] | None = None
        self.detect_ms = 0.0
        self.n_frames = 0
        self.first_detect_t: float | None = None
        self.first_track_t: float | None = None
        self.recalled = False
        self.brg: tuple[float, float, float] | None = None  # latest base bearing (rx time, az deg, sigma deg)
        self.n_bearings = 0
        self.t_lost: float | None = None
        self.lost_pos = np.zeros(3)  # last track position / velocity when it was lost
        self.lost_vel = np.zeros(3)
        self.lost_range = 0.0  # its distance from the array
        self.lost_brg_after = 0.0  # bearings received after this count as "new" for the LOST prediction
        self.lost_mode = "track"  # "track" (constant velocity) or "bearing"
        self.suspect_since: float | None = None
        self.n_suspect = 0
        self.n_gated = 0  # new tracks refused by the bearing gate
        self.use_bearing = True  # False: ignore the base bearings in LOST / checks (evaluation baseline)
        bus.subscribe(topics.CAMERA_GIMBAL, self.on_frame)
        bus.subscribe(topics.UPLINK_BEARING, self.on_bearing)

    # ------------------------------------------------------------------ interface

    def launch(self, az_deg: float, t: float, first_heard_t: float | None = None) -> bool:
        """Launch command from the base: the UMA-16 bearing, and when the enemy was first heard."""
        if not self.on_ground:
            return False
        self.first_heard_t = t if first_heard_t is None else first_heard_t
        self.r_anchor = (self.tcfg.assumed_range_m, self.first_heard_t)
        return super().launch(az_deg, t)

    def on_bearing(self, msg: BearingUpdate) -> None:
        """The current UMA-16 bearing to the enemy reached the drone over the link (~1 Hz while it is up)."""
        if self.on_ground or not self.use_bearing:
            return
        if self.first_heard_t is None:
            self.first_heard_t = msg.first_heard_t
        t = self._last_imu_t or msg.t
        self.bearing = msg.az_deg % 360.0
        self.brg = (t, self.bearing, max(msg.sigma_deg, 0.5))
        self.n_bearings += 1
        if self.state == "LOST" and self.lost_mode == "track" and t >= self.lost_brg_after:
            self.lost_mode = "bearing"
            self.log(t, f"LOST: base bearing {self.bearing:05.1f} deg +-{msg.sigma_deg:.0f}: predicting on it")

    def fresh_bearing(self, t: float) -> tuple[float, float, float] | None:
        if self.brg is None or t - self.brg[0] > self.tcfg.bearing_fresh_s:
            return None
        return self.brg

    def bearing_offset(self, pos: np.ndarray, sigma_m: float, t: float) -> tuple[float, float] | None:
        """(angle of `pos` off the fresh base bearing, the combined 1-sigma), deg; None without a fresh bearing.
        Sigma: the bearing's own plus `sigma_m` (track + own position uncertainty) seen from the array."""
        b = self.fresh_bearing(t)
        if b is None:
            return None
        d = np.asarray(pos, float)[:2] - self.mast
        r = max(float(np.linalg.norm(d)), 1.0)
        off = (math.degrees(math.atan2(d[0], d[1])) - b[1] + 180.0) % 360.0 - 180.0
        return off, math.hypot(b[2], math.degrees(math.atan2(sigma_m, r)))

    def _own_sigma(self) -> float:
        return 0.5 * self.pose(report=False).sigma2_horizontal()

    def lost_prediction(self, t: float) -> np.ndarray:
        """Predicted enemy position while LOST (see the module doc)."""
        tc = self.tcfg
        dt = max(t - (self.t_lost or t), 0.0)
        if self.lost_mode == "bearing" and self.bearing is not None:
            r = max(tc.min_assumed_range_m, self.lost_range - tc.assumed_speed * dt)
            xy = self.mast + az_to_dir(self.bearing) * r
            return np.array([xy[0], xy[1], self.lost_pos[2]])
        v = np.clip(self.lost_vel, -10.0, 10.0)
        v[2] = 0.0
        return self.lost_pos + v * min(dt, tc.lost_cv_max_s)

    def lost_area(self, t: float) -> dict | None:
        """The LOST search area for the display: centre, mode and its size (m)."""
        if self.state != "LOST" or self.t_lost is None:
            return None
        c = self.lost_prediction(t)
        if self.lost_mode == "bearing":
            r = float(np.linalg.norm(c[:2] - self.mast))
            sig = self.brg[2] if self.brg is not None else 5.0
            return {"center": c, "mode": "bearing", "az": self.bearing, "along": self.tcfg.lost_pattern_m + 10.0,
                    "across": max(8.0, 2.0 * r * math.radians(sig))}
        return {"center": c, "mode": "track", "radius": 8.0 + 2.0 * (t - self.t_lost)}

    def recall(self, t: float) -> None:
        """RETURN ordered by the base (over the link)."""
        if not self.on_ground and self.state not in ("RETURN", "DESCEND"):
            self.recalled = True
            self._go("RETURN", t, "recalled by the base")

    def assumed_enemy(self, t: float) -> np.ndarray:
        """Where the base's assumption puts the enemy now (world, onboard z frame)."""
        tc = self.tcfg
        r0, t0 = self.r_anchor
        r = max(tc.min_assumed_range_m, r0 - tc.assumed_speed * max(t - t0, 0.0))
        xy = self.mast + az_to_dir(self.bearing or 0.0) * r
        return np.array([xy[0], xy[1], float(self.terrain(*xy)) + tc.assumed_agl])

    def battery_left(self, t: float) -> float:
        return self.tcfg.battery_s - (0.0 if self.takeoff_t is None else t - self.takeoff_t)

    # ------------------------------------------------------------------ camera

    def on_frame(self, fr: GimbalFrame) -> None:
        self.last_frame = fr
        if self.state not in TRACK_STATES:
            return
        t = fr.t
        self.n_frames += 1
        dets = self.detector.detect(fr.image) if self.detector is not None else []
        self.detect_ms = getattr(self.detector, "last_ms", 0.0)
        est = self.pose(report=False)
        trk = self.tracker
        if self.state == "SEARCH" and trk.track is not None and t - (trk.last_hit_t or t) > 1.0:
            trk.reset()  # an unconfirmed blip long gone: let a new detection start fresh
        own = self._own_sigma()

        def accept(m) -> bool:  # a new track must agree with a fresh base bearing (if there is one)
            if not (self.tcfg.gate_bearing and self.use_bearing):
                return True
            ob = self.bearing_offset(m.pos, math.sqrt(m.cov[0, 0] + m.cov[1, 1]) + own, t)
            ok = ob is None or abs(ob[0]) <= self.tcfg.suspect_sigma * ob[1]
            self.n_gated += 0 if ok else 1
            return ok

        res = trk.process(t, dets, est.pos, self.ahrs.yaw, fr.pan_deg, fr.tilt_deg, accept)
        if self.state == "LOST" and res.hit is None and res.others:  # a drone, just not where the old track said
            trk.reset()
            res = trk.process(t, dets, est.pos, self.ahrs.yaw, fr.pan_deg, fr.tilt_deg, accept)
        self.last_result = res
        if res.hit is not None:
            self.last_snapshot = snapshot(fr.image, res.hit.det.box, self.tcfg.snapshot_px)
        s = self.state
        if s in ("SEARCH", "LOST") and res.hit is not None:
            self.resume_state = s
            if self.first_detect_t is None:
                self.first_detect_t = t
            self._go("DETECTED", t, f"drone {res.hit.det.conf:.2f} at ~{res.hit.range_m:.0f} m")
        elif s == "DETECTED":
            if trk.confirmed:
                if self.first_track_t is None:
                    self.first_track_t = t
                self._go("TRACKING", t, f"confirmed ({trk.hits}/{len(trk.window)} frames), "
                                        f"{float(np.linalg.norm(trk.track.pos - est.pos)):.0f} m away")
                self._next_report = t
            elif t - self.t_state > self.tcfg.detected_timeout_s:
                trk.reset()
                self._go(self.resume_state, t, "not confirmed")
        elif s == "TRACKING" and t - (trk.last_hit_t or t) > self.tcfg.lost_after_s:
            self._enter_lost(t)
        elif s == "TRACKING" and self.use_bearing:
            ob = self.bearing_offset(trk.track.pos, trk.track.sigma_h() + own, t)
            if ob is not None and abs(ob[0]) > self.tcfg.suspect_sigma * ob[1]:
                self.suspect_since = t if self.suspect_since is None else self.suspect_since
                if t - self.suspect_since >= self.tcfg.suspect_s:
                    self.n_suspect += 1
                    self._enter_lost(t, f"SUSPECT: track {ob[0]:+.0f} deg off the UMA-16 bearing (3 sigma = "
                                        f"{self.tcfg.suspect_sigma * ob[1]:.0f} deg) for {self.tcfg.suspect_s:.0f} s: "
                                        "possibly a bird / wrong object; re-acquiring", suspect=True)
            else:
                self.suspect_since = None
        if self.state in ("TRACKING", "LOST") and trk.track is not None and t >= self._next_report:
            self._next_report = t + self.tcfg.report_every_s
            self._report(t, est)

    def _enter_lost(self, t: float, why: str = "", suspect: bool = False) -> None:
        trk = self.tracker.track
        t_hit = self.tracker.last_hit_t or t
        self.t_lost = t_hit  # the prediction runs from the last hit
        self.lost_pos, self.lost_vel = trk.pos_at(t_hit), trk.vel
        if suspect:  # the track was something else: its velocity says nothing about the enemy
            self.lost_vel = np.zeros(3)
        self.lost_range = float(np.linalg.norm(self.lost_pos[:2] - self.mast))
        # a suspect track was judged against bearings already: use them at once; otherwise wait for a newer one
        self.lost_brg_after = t - self.tcfg.bearing_fresh_s if suspect else t_hit
        b = self.brg
        self.lost_mode = "bearing" if (self.use_bearing and b is not None and b[0] >= self.lost_brg_after) else "track"
        self.suspect_since = None
        self.lost_xy, self.lost_z = self.lost_pos[:2].copy(), float(self.lost_pos[2])
        if suspect:
            self.tracker.reset()
        basis = "the base bearing" if self.lost_mode == "bearing" else "the last track (no newer bearing)"
        self._go("LOST", t, why or f"no detection for {self.tcfg.lost_after_s:.0f} s; predicting on {basis}")

    def _report(self, t: float, est: PoseEstimate) -> None:
        trk = self.tracker.track
        p, v, sig = trk.pos, trk.vel, trk.sigma_h()
        if self.state == "LOST":  # not seen: report the prediction (bearing / constant velocity), honestly wide
            p = self.lost_prediction(t)
            v = self.lost_vel.copy()
            a = self.lost_area(t)
            sig = a["across"] / 2 if a["mode"] == "bearing" else a["radius"]
        snap = self.last_snapshot
        self.report_seq += 1
        self.outbox.put(EnemyReport(
            t=t, seq=self.report_seq, state=self.state, pos=p, agl=float(p[2] - self.terrain(p[0], p[1])),
            speed=float(math.hypot(v[0], v[1])), heading_deg=math.degrees(math.atan2(v[0], v[1])) % 360.0,
            climb=float(v[2]), confidence=self.tracker.confidence() if self.state == "TRACKING" else 0.0,
            sigma_m=sig, range_m=float(np.linalg.norm(p - est.pos)), own_pos=est.pos.copy(),
            snapshot=None if snap is None else snap[0], snapshot_jpg=b"" if snap is None else snap[1]))

    # ------------------------------------------------------------------ mission (IMU rate)

    def _go(self, s: str, t: float, why: str = "") -> None:
        if s == "OUTBOUND":  # BearingPilot's TAKEOFF hands over here
            s = "SEARCH"
        elif s == "TAKEOFF":
            self.takeoff_t = t
        super()._go(s, t, why)

    def _update_mission(self, est: PoseEstimate, t: float, dt: float) -> None:
        s = self.state
        if s not in TRACK_STATES:
            super()._update_mission(est, t, dt)
            if s == "TAKEOFF":
                self._gimbal(t, 0.0, 0.0)
            elif s in ("RETURN", "DESCEND"):
                self._gimbal(t, 0.0, -20.0)
            return
        home = float(np.linalg.norm(est.pos[:2] - self.pad)) / self.tcfg.home_speed
        need = home + max(est.pos[2], 0.0) / self.plan.descent + self.tcfg.reserve_s
        if self.battery_left(t) < need:
            self._go("RETURN", t, f"battery: {self.battery_left(t):.0f} s left, {need:.0f} s needed to get home")
            return
        if s == "SEARCH":
            self._search(est, t, dt)
        elif s == "DETECTED":
            self._detected(est, t, dt)
        elif s == "TRACKING":
            self._tracking(est, t, dt)
        elif s == "LOST":
            self._lost(est, t, dt)

    def _gimbal(self, t: float, pan: float, tilt: float) -> None:
        tc = self.tcfg
        self.pan_cmd = float(np.clip(pan, -tc.pan_limit_deg, tc.pan_limit_deg))
        self.tilt_cmd = float(np.clip(tilt, tc.tilt_min_deg, tc.tilt_max_deg))
        if t >= self._next_gimbal:
            self._next_gimbal = t + 0.05
            self.bus.publish(topics.GIMBAL_CMD, GimbalCmd(t, self.pan_cmd, self.tilt_cmd))

    def _look_at(self, t: float, est: PoseEstimate, target: np.ndarray, sweep_pan: float = 0.0,
                 sweep_tilt: float = 0.0, period: float = 4.0) -> None:
        pan, tilt = pointing(np.asarray(target, float) - est.pos, self.ahrs.yaw)
        ph = 2 * math.pi * (t - self.t_state) / period
        self._gimbal(t, pan + sweep_pan * math.sin(ph), tilt + sweep_tilt * math.sin(0.61 * ph))

    def _circle(self, center_xy: np.ndarray, radius: float, speed: float, est: PoseEstimate, t: float,
                dt: float) -> None:
        """Setpoint on a circle around center_xy, entered from where the drone is."""
        rel = self.sp[:2] - center_xy
        a = math.atan2(rel[1], rel[0]) + speed / radius * dt
        goal = center_xy + radius * np.array([math.cos(a), math.sin(a)])
        self._move_xy(goal, speed * 1.5, dt)
        self.yaw = math.atan2(center_xy[1] - est.pos[1], center_xy[0] - est.pos[0])
        self._follow_terrain(dt)

    def _search(self, est: PoseEstimate, t: float, dt: float) -> None:
        tc = self.tcfg
        tgt = self.assumed_enemy(t)
        if float(np.linalg.norm(tgt[:2] - est.pos[:2])) > 40.0:
            self._move_xy(tgt, tc.search_speed, dt)
            self._follow_terrain(dt)
            self._look_at(t, est, tgt, tc.search_pan_deg, 4.0, tc.search_sweep_s)
        else:  # at the assumed point: circle it, looking all around (the enemy may already be past)
            self._circle(tgt[:2], 40.0, 6.0, est, t, dt)
            out = tgt + np.r_[az_to_dir(self.bearing) * 80.0, 0.0]
            self._look_at(t, est, out, 160.0, 5.0, 14.0)

    def _detected(self, est: PoseEstimate, t: float, dt: float) -> None:
        self.v[:2] *= math.exp(-dt / 1.5)  # slow down, hold the height, lock on
        self.sp[:2] += self.v[:2] * dt
        self.v[2] = 0.0
        trk = self.tracker.track
        if trk is not None:
            e = trk.pos_at(t)
            self.yaw = math.atan2(e[1] - est.pos[1], e[0] - est.pos[0])
            self._look_at(t, est, e)
            rel = est.pos - e
            d3 = float(np.linalg.norm(rel))
            if d3 < self.tcfg.min_dist_m:  # re-acquired too close: back off at once, as in TRACKING
                away = rel / max(d3, 1e-6)
                self.v = away * 4.0 * (self.tcfg.min_dist_m - d3 + 1.0)
                self.sp = est.pos + away * 2.0

    def _tracking(self, est: PoseEstimate, t: float, dt: float) -> None:
        tc = self.tcfg
        trk = self.tracker.track
        e = trk.pos_at(t)
        ev = trk.vel
        n = float(np.linalg.norm(ev[:2]))
        if n > 15.0:
            ev = ev * 15.0 / n
        rel = est.pos - e
        dh = float(np.linalg.norm(rel[:2]))
        u = rel[:2] / dh if dh > 1.0 else -ev[:2] / max(n, 1e-6) if n > 0.5 else np.array([1.0, 0.0])
        ground = float(self.terrain(*est.pos[:2]))
        des = np.array([*(e[:2] + u * tc.track_dist_m), max(e[2] + tc.track_above_m, ground + 12.0)])
        err = des - self.sp
        v = ev + 0.6 * err
        vh = float(np.linalg.norm(v[:2]))
        if vh > tc.track_speed_max:
            v[:2] *= tc.track_speed_max / vh
        v[2] = float(np.clip(v[2], -self.plan.sink, self.plan.climb))
        d3 = float(np.linalg.norm(rel))
        if d3 < tc.min_dist_m:  # too close: back off along the line from the enemy, now
            away = rel / max(d3, 1e-6)
            v += away * 4.0 * (tc.min_dist_m - d3 + 1.0)
            self.sp = est.pos + away * 2.0
        self.v = v
        self.sp = self.sp + v * dt
        leash = self.sp - est.pos  # the setpoint may not run away from a drone that cannot keep up
        L = float(np.linalg.norm(leash))
        if L > 8.0:
            self.sp = est.pos + leash * 8.0 / L
        self.yaw = math.atan2(e[1] - est.pos[1], e[0] - est.pos[0])
        self._look_at(t, est, e)

    def _lost(self, est: PoseEstimate, t: float, dt: float) -> None:
        tc = self.tcfg
        tgt = self.lost_prediction(t)
        self.lost_xy, self.lost_z = tgt[:2].copy(), float(tgt[2])
        # the line to search along: the bearing (radial from the array); constant velocity: where it was heading
        rad = tgt[:2] - self.mast
        u = rad / max(float(np.linalg.norm(rad)), 1e-6)
        if self.lost_mode == "track" and float(np.linalg.norm(self.lost_vel[:2])) > 0.5:
            u = -self.lost_vel[:2] / float(np.linalg.norm(self.lost_vel[:2]))
        # sweep forward / backward along the line, always ahead of the point (where the enemy is heading) and never
        # through it: the enemy is unseen here, so the standoff must hold whatever the prediction error
        sweep = 0.5 + 0.5 * math.sin(2 * math.pi * (t - self.t_state) / tc.lost_pattern_s)
        goal = tgt[:2] - u * (tc.lost_standoff_m + tc.lost_pattern_m * sweep)
        far = float(np.linalg.norm(goal - est.pos[:2])) > tc.lost_pattern_m + 10.0
        self._move_xy(goal, tc.track_speed_max if far else 6.0, dt)
        z_goal = max(tgt[2] + tc.track_above_m + 6.0, float(self.terrain(*self.sp[:2])) + 12.0)
        self.v[2] = float(np.clip(0.6 * (z_goal - self.sp[2]), -self.plan.sink, self.plan.climb))
        self.sp[2] += self.v[2] * dt
        self.yaw = math.atan2(tgt[1] - est.pos[1], tgt[0] - est.pos[0])
        self._look_at(t, est, tgt, 45.0 if far else 60.0, 6.0, 4.0)
        if t - self.t_state > tc.lost_search_s:
            self.r_anchor = (float(np.linalg.norm(tgt[:2] - self.mast)), t)
            self.tracker.reset()
            self._go("SEARCH", t, "lost too long: back to the UMA-16 bearing")
