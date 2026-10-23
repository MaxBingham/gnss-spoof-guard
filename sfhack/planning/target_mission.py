"""V4 target mission (ONBOARD): search the area, pick the brown truck with a clear margin, fly over it, find
the landing-station marker and land on the pad. Never touches down without a marker confirmation.

Phases: TAKEOFF -> SEARCH (center + 4 corners at search_alt) -> SELECT (decide, or INSPECT the leading
candidates at inspect_alt: nadir look + lidar height) -> APPROACH (above the chosen candidate at marker_alt)
-> MARKER_SEARCH (hover, marker_timeout_s) -> MARKER_LAND (descend on the marker) -> LANDED.
No marker in time -> TARGET_REJECTED, climb, next candidate. Out of candidates -> TARGET_ABORT: hover above A.

Inputs: the bus (down camera, lidar, rangefinder) and the pose ESTIMATE (`pose_fn`, the autopilot's EKF +
AHRS). No truth, no layout.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .. import bus as topics
from ..bus import Bus
from ..config import MissionConfig, TargetConfig
from ..messages import CameraFrame, Health, LidarMsg, MarkerDetection, PoseEstimate, RangeMsg, Setpoint, TargetCandidates
from ..perception.target_detector import NAMES, MarkerDetector, TargetPerception, Track
from ..world.onboard_map import OnboardMap
from .mission import Mission, Waypoint

MARKER_CONFIRM = 3  # marker detections before descending below marker_alt


class TargetMission(Mission):
    def __init__(self, mcfg: MissionConfig, tcfg: TargetConfig, bus: Bus, onboard_map: OnboardMap,
                 start_pos: np.ndarray, yaw: float, area_center: np.ndarray) -> None:
        start = np.asarray(start_pos, float)
        super().__init__([Waypoint(np.array([start[0], start[1], tcfg.search_alt]), mcfg.climb_speed, "TAKEOFF")],
                         start, yaw, mcfg)
        self.tc = tcfg
        self.bus = bus
        self.start_xy = start[:2].copy()
        self.center = np.asarray(area_center, float)
        self.pose_fn: Callable[[], PoseEstimate] | None = None  # set by the flight builder (autopilot.pose)
        self.perception = TargetPerception(tcfg, onboard_map, self.center, log=self.log)
        self.marker = MarkerDetector(tcfg)
        self.last_frame: CameraFrame | None = None
        self.last_marker: MarkerDetection | None = None
        self.marker_hits = 0
        self.route: list[tuple[np.ndarray, float, float]] = []  # (pos, speed, yaw) of the current leg
        self.chosen: Track | None = None
        self.rejected: set[int] = set()
        self.inspected: dict[int, int] = {}
        self.n_inspections = 0
        self.phase_t = 0.0
        self.p_target: dict[int, float] = {}
        self.target_xy: np.ndarray | None = None
        self.anchor: np.ndarray | None = None  # (3,) confirmed marker position, EKF frame (fixed)
        self._marker_pts: list[np.ndarray] = []
        # set by the flight builder: (t, xy, cov) -> EKF position update (the autopilot's EKF)
        self.position_fix: Callable[[float, np.ndarray, np.ndarray], None] | None = None
        bus.subscribe(topics.CAMERA_DOWN, self.on_camera)
        bus.subscribe(topics.LIDAR, self.on_lidar)
        bus.subscribe(topics.RANGE, self.on_range_msg)

    @property
    def hold_gnss(self) -> bool:
        """Read by the autopilot: no GNSS fusion during the marker landing; the marker anchor replaces it (every
        detection is a ~0.15 m position fix, `position_fix`). GNSS (1.5 m noise) left a ~0.6 m/s velocity error
        that walked the blind last 2 m of the descent off the 2 m pad."""
        return self.phase == "MARKER_LAND" and self.anchor is not None

    # ------------------------------------------------------------------ sensors

    def _pose(self) -> PoseEstimate | None:
        return self.pose_fn() if self.pose_fn is not None and self.phase not in ("PREFLIGHT", "LANDED") else None

    def on_lidar(self, msg: LidarMsg) -> None:
        pose = self._pose()
        if pose is not None:
            self.perception.on_lidar(msg.angles, msg.ranges, pose)

    def on_range_msg(self, msg: RangeMsg) -> None:
        pose = self._pose()
        if pose is not None and np.isfinite(msg.range):
            self.perception.on_range(msg.range, pose)

    def on_camera(self, frame: CameraFrame) -> None:
        pose = self._pose()
        if pose is None:
            return
        self.last_frame = frame
        agl = self.range_agl if self.range_agl is not None else pose.pos[2]
        if self.phase in ("SEARCH", "INSPECT", "SELECT", "APPROACH") and agl > self.tc.det_min_alt:
            self.perception.on_frame(frame, pose)
            self.perception.update_evidence(frame.t)
            self._update_p_target()
        if self.phase in ("APPROACH", "MARKER_SEARCH", "MARKER_LAND"):
            m = self.marker.detect(frame, pose)
            if m is not None and (self.target_xy is None or np.linalg.norm(m.pos[:2] - self.target_xy) < 6.0):
                if self.last_marker is None:
                    self.log(frame.t, f"MARKER: landing-station marker #{self.tc.marker_id} seen at "
                                      f"({m.pos[0]:.1f}, {m.pos[1]:.1f}), {m.distance:.1f} m away")
                self.last_marker = m
                self.marker_hits += 1
                if self.anchor is None:
                    self._marker_pts.append(m.pos.copy())
                elif self.phase == "MARKER_LAND" and self.position_fix is not None:
                    # marker = known landmark: drone position = anchor - (marker - drone) as seen now
                    self.position_fix(frame.t, pose.pos[:2] + (self.anchor[:2] - m.pos[:2]), np.eye(2) * 0.15 ** 2)
                self.bus.publish(topics.TARGET_MARKER, m)
        self.bus.publish(topics.TARGET_CANDIDATES, TargetCandidates(
            frame.t, self.perception.to_messages(self.p_target, self.chosen.id if self.chosen else None, self.rejected),
            self.chosen.id if self.chosen else None, self.phase, frame.seq))

    # ------------------------------------------------------------------ evidence

    def _update_p_target(self) -> None:
        """P(candidate i is THE truck): odds of each candidate's truck class probability, normalized over all
        candidates plus "the truck is not among them yet"."""
        cands = [tr for tr in self.perception.candidates() if tr.id not in self.rejected and tr.probs]
        odds = {tr.id: tr.probs.get("brown_truck", 0.0) / max(1e-6, 1.0 - tr.probs.get("brown_truck", 0.0)) for tr in cands}
        unseen = self.tc.unseen_odds_all_found if len(self.perception.candidates()) >= 6 else self.tc.unseen_odds
        tot = sum(odds.values()) + unseen
        self.p_target = {k: v / tot for k, v in odds.items()}

    def _ranked(self) -> list[Track]:
        cands = [tr for tr in self.perception.candidates() if tr.id not in self.rejected]
        return sorted(cands, key=lambda tr: -self.p_target.get(tr.id, 0.0))

    def _describe(self, tr: Track) -> str:
        L, W = tr.size(tr.height or 0.0)
        best = max(tr.probs, key=tr.probs.get) if tr.probs else "unknown"
        h = f" x {tr.height:.1f}" if tr.height is not None else ""
        return f"candidate {tr.id} = {NAMES.get(best, best)} p={self.p_target.get(tr.id, 0.0):.2f} ({L:.1f} x {W:.1f}{h} m, {tr.color})"

    # ------------------------------------------------------------------ flight

    def _set_route(self, t: float, phase: str, legs: list[tuple[np.ndarray, float, float]]) -> None:
        self.phase = phase
        self.phase_t = t
        self.route = list(legs)

    def _follow(self, pose: PoseEstimate, dt: float) -> tuple[np.ndarray, bool]:
        """Move the setpoint along the route; True when the last leg is reached (estimate within 1.5 m)."""
        if not self.route:
            return np.zeros(3), True
        pos, speed, yaw = self.route[0]
        self.yaw = yaw
        vel, reached = self._move_setpoint(pos, speed, dt)
        if reached and float(np.linalg.norm(pos - pose.pos)) < 1.5:
            self.route.pop(0)
        return vel, not self.route

    def update(self, t: float, pose: PoseEstimate, dt: float, health: Health = Health.NOMINAL,
               speed_scale: float = 1.0, last_fix_pos: np.ndarray | None = None) -> Setpoint:
        c = self.tc
        if self.phase in ("PREFLIGHT", "LANDED"):
            return Setpoint(t, self.setpoint.copy(), np.zeros(3), self.yaw, self.phase)
        if self.phase == "TAKEOFF":
            vel, reached = self._move_setpoint(self.waypoints[0].pos, self.cfg.climb_speed, dt)
            if reached and abs(pose.pos[2] - c.search_alt) < 1.5:
                o = c.search_offset
                pts = [self.center] + [self.center + o * np.array(d) for d in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
                self._set_route(t, "SEARCH", [(np.array([p[0], p[1], c.search_alt]), c.search_speed, self.yaw) for p in pts])
                self.log(t, f"SEARCH: pattern over the {2 * c.area_half:.0f} m area at {c.search_alt:.0f} m")
            return Setpoint(t, self.setpoint.copy(), vel, self.yaw, self.phase)

        if self.phase in ("SEARCH", "INSPECT"):
            vel, done = self._follow(pose, dt)
            if done:
                self._select(t)
            return Setpoint(t, self.setpoint.copy(), vel, self.yaw, self.phase)

        if self.phase == "APPROACH":
            vel, done = self._follow(pose, dt)
            if done:
                self.phase, self.phase_t = "MARKER_SEARCH", t
            return Setpoint(t, self.setpoint.copy(), vel, self.yaw, self.phase)

        if self.phase == "MARKER_SEARCH":
            if self.marker_hits >= MARKER_CONFIRM and self.last_marker is not None:
                self.anchor = np.median(np.array(self._marker_pts[-MARKER_CONFIRM * 2:]), axis=0)
                m = self.last_marker
                self.log(t, f"MARKER_CONFIRMED: {self.marker_hits} detections, pad at ({m.pos[0]:.1f}, {m.pos[1]:.1f}, "
                            f"{m.pos[2]:.1f}) - descending on the marker")
                self.phase, self.phase_t = "MARKER_LAND", t
            elif t - self.phase_t > c.marker_timeout_s:
                self._reject(t, f"no landing marker within {c.marker_timeout_s:.0f} s")
            return Setpoint(t, self.setpoint.copy(), np.zeros(3), self.yaw, self.phase)

        if self.phase == "MARKER_LAND":
            return self._marker_land(t, pose, dt)

        if self.phase == "TARGET_ABORT":  # hover above A for the operator (never land unconfirmed)
            vel, _ = self._follow(pose, dt)
            return Setpoint(t, self.setpoint.copy(), vel, self.yaw, self.phase)
        return Setpoint(t, self.setpoint.copy(), np.zeros(3), self.yaw, self.phase)

    def _marker_land(self, t: float, pose: PoseEstimate, dt: float) -> Setpoint:
        c = self.tc
        m = self.last_marker
        assert m is not None
        above = self.range_agl if self.range_agl is not None else pose.pos[2] - m.pos[2]
        stale = t - m.t
        if stale > 6.0 and above > c.land_hold_below + 1.0:
            self.marker_hits = 0
            self.last_marker, self.anchor, self._marker_pts = None, None, []
            self._set_route(t, "APPROACH", [(np.array([m.pos[0], m.pos[1], pose.pos[2] - above + c.marker_alt]), 1.5, self.yaw)])
            self.log(t, f"MARKER_LOST: {stale:.0f} s without the marker at {above:.1f} m, climbing back to {c.marker_alt:.0f} m")
            return Setpoint(t, self.setpoint.copy(), np.zeros(3), self.yaw, self.phase)
        a = self.anchor if self.anchor is not None else m.pos
        off = float(np.hypot(*(a[:2] - pose.pos[:2])))
        # Descend only while centered over the marker (below land_hold_below it leaves the view: the last
        # marker position in the EKF frame is held); off-center -> hold height and re-center.
        tol = max(c.land_xy_tol, 0.05 * above)
        target = np.array([a[0], a[1], self.setpoint[2]])
        if off <= tol:
            target[2] = a[2] - 0.5  # press onto the pad; touchdown detection idles the motors
        elif self.setpoint[2] < pose.pos[2] - 0.3:
            target[2] = pose.pos[2]  # stop the descent where we are
        vel, _ = self._move_setpoint(target, c.land_speed, dt)
        self._check_touchdown(t, pose, dt)
        if self.phase == "LANDED":
            self.log(t, f"LANDED on candidate {self.chosen.id if self.chosen else '?'} (marker-confirmed pad)")
        return Setpoint(t, self.setpoint.copy(), vel, self.yaw, "MARKER_LAND")

    # ------------------------------------------------------------------ decisions

    def _select(self, t: float) -> None:
        c = self.tc
        self._update_p_target()
        ranked = self._ranked()
        if not ranked:
            if self.n_inspections < c.max_inspections:
                self.n_inspections += 1
                o = c.search_offset * 1.6
                pts = [self.center + o * np.array(d) for d in ((1, 0), (0, 1), (-1, 0), (0, -1))]
                self._set_route(t, "SEARCH", [(np.array([p[0], p[1], c.search_alt_low]), c.search_speed, self.yaw) for p in pts])
                self.log(t, "SEARCH: no candidates yet, searching again lower")
            else:
                self._abort(t, "no candidates found")
            return
        top = ranked[0]
        p1 = self.p_target.get(top.id, 0.0)
        p2 = self.p_target.get(ranked[1].id, 0.0) if len(ranked) > 1 else 0.0
        runner = f"; runner-up {self._describe(ranked[1])}" if len(ranked) > 1 else ""
        clear = p1 >= c.p_decide and p1 - p2 >= c.p_margin and top.height is not None and top.n_nadir >= c.min_nadir_obs
        if clear:
            self.chosen = top
            self.log(t, f"TARGET: {self._describe(top)}{runner}")
            self.marker_hits, self.last_marker, self.anchor, self._marker_pts = 0, None, None, []
            self.target_xy = top.xy.copy()
            self._set_route(t, "APPROACH", [(np.array([top.xy[0], top.xy[1], c.search_alt]), c.transit_speed, top.yaw),
                                            (np.array([top.xy[0], top.xy[1], c.marker_alt]), 2.0, top.yaw)])
            return
        if self.n_inspections >= c.max_inspections:
            self._abort(t, f"no clear decision after {self.n_inspections} fly-overs ({self._describe(top)}{runner})")
            return
        # not clear: fly over the least-inspected of the leading candidates, lower (nadir size + lidar height)
        lead = [tr for tr in ranked[:3] if self.p_target.get(tr.id, 0.0) > 0.02] or ranked[:1]
        tr = min(lead, key=lambda tr: (self.inspected.get(tr.id, 0), -self.p_target.get(tr.id, 0.0)))
        self.inspected[tr.id] = self.inspected.get(tr.id, 0) + 1
        self.n_inspections += 1
        why = []
        if p1 < c.p_decide:
            why.append(f"p={p1:.2f} < {c.p_decide:.2f}")
        if p1 - p2 < c.p_margin:
            why.append(f"margin {p1 - p2:.2f} < {c.p_margin:.2f}")
        if top.height is None:
            why.append("height not measured")
        if top.n_nadir < c.min_nadir_obs:
            why.append(f"{top.n_nadir} nadir views")
        self.log(t, f"INSPECT: leader {self._describe(top)}{runner} - not clear ({', '.join(why)}); flying over candidate {tr.id}")
        u = np.array([math.cos(tr.yaw), math.sin(tr.yaw)])
        alt = c.inspect_alt
        a, b = tr.xy - 7.0 * u, tr.xy + 7.0 * u
        self._set_route(t, "INSPECT", [(np.array([a[0], a[1], alt]), c.transit_speed, tr.yaw),
                                       (np.array([b[0], b[1], alt]), 2.0, tr.yaw)])

    def _reject(self, t: float, why: str) -> None:
        tr = self.chosen
        if tr is not None:
            self.rejected.add(tr.id)
            self.log(t, f"TARGET_REJECTED: candidate {tr.id} - {why}; climbing, trying the next")
        self.chosen, self.target_xy, self.last_marker, self.marker_hits = None, None, None, 0
        self.anchor, self._marker_pts = None, []
        if len(self.rejected) >= self.tc.max_reject:
            self._abort(t, f"{len(self.rejected)} candidates rejected")
            return
        here = self.setpoint.copy()
        self._set_route(t, "INSPECT", [(np.array([here[0], here[1], self.tc.search_alt]), self.cfg.climb_speed, self.yaw)])

    def _abort(self, t: float, why: str) -> None:
        self.log(t, f"TARGET_ABORT: {why} - never landing unconfirmed; hovering above A for the operator")
        self._set_route(t, "TARGET_ABORT", [(np.array([self.start_xy[0], self.start_xy[1], self.tc.search_alt]),
                                             self.tc.transit_speed, self.yaw)])
