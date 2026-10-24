"""Enemy-drone tracking scenario (harness), shared by demos/52_track_enemy.py (live) and demos/56_track_eval.py.

Simulator side: the farm world with the enemy and birds (demos/enemy_world.py), the X2 with its sensor rig, the
gimbal camera, the enemy's motion model (sfhack/sim/enemy.py), the bird flock, the radio link (sfhack/sim/link.py).
Onboard: TrackerPilot (sfhack/mission/tracker_pilot.py) with the YOLO detector, fed only by bus messages.
Base: the UMA-16 bearing (keys / UDP / script) and what arrives over the link.

Truth is read here only to move the simulated world, to DISPLAY and to SCORE (frame-by-frame detection truth from
the camera's segmentation buffer when `score=True`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

import enemy_world as ew
from sfhack import bus as topics
from sfhack.bus import Bus
from sfhack.config import TrackerConfig, default_config
from sfhack.messages import BearingUpdate
from sfhack.mission.enemy_report import EnemyReport
from sfhack.mission.tracker_pilot import TrackerPilot
from sfhack.sim import enemy as en
from sfhack.sim.gimbal_camera import GimbalCamera, project
from sfhack.sim.base_acoustic import BaseAcoustic
from sfhack.sim.link import LinkModel
from sfhack.sim.sensor_rig import SensorRig

SETTLE_S = 1.0
WORLD_DT = 0.01  # enemy + birds move at 100 Hz (mocap)
TRAIL_DT = 0.25


@dataclass
class LossEvent:
    """A forced loss (evaluation): the enemy hidden from our camera for `occlusion_s` (as behind a tree row) while it
    turns sharply (its steering bearing jumps by `turn_deg`)."""

    t0: float
    t1: float  # visible again
    turn_deg: float
    dist0: float  # true distance our drone -> enemy at t0
    went_lost: bool = False
    reacq_t: float | None = None  # back in TRACKING on the true enemy
    end_t: float | None = None

    @property
    def reacquired(self) -> bool:
        return self.reacq_t is not None


@dataclass
class FrameTruth:
    """One camera frame, scored (harness)."""

    t: float
    state: str
    dist: float  # true distance camera -> enemy (m)
    in_fov: bool  # the enemy's centre projects into the image (may still be hidden)
    visible_px: int  # enemy pixels in the segmentation
    box: tuple | None  # true visible box
    detected: bool  # a drone detection >= drone_conf overlaps the true box
    best_conf: float  # best drone confidence on the enemy (any >= conf_min)
    fp: list[str] = field(default_factory=list)  # false drone detections >= drone_conf: "bird" / "other"


class TrackScenario:
    def __init__(self, world: ew.EnemyWorld, seed: int = 0, detector=None, tcfg: TrackerConfig | None = None,
                 enemy_speed: float | None = 4.0, availability: float = 0.8, score: bool = False,
                 birds_near_route: bool = True, send_bearings: bool = True, forced_losses: int = 0,
                 occlusion_s: float = 8.0, turn_deg: float = 30.0) -> None:
        self.world, self.seed, self.score = world, seed, score
        self.tcfg = tcfg or TrackerConfig()
        m = self.model = world.model
        d = self.data = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)
        for mid in range(m.nmocap):
            d.mocap_pos[mid] = en.HIDDEN
        while d.time < SETTLE_S:
            mujoco.mj_step(m, d)
        self.body = m.body("x2").id
        self.z0 = float(d.qpos[2])
        self.ground = world.ground
        W = world.mod
        self.pad = np.asarray(W.PAD, float)[:2].copy()
        self.mast = np.zeros(2)
        cfg = default_config()
        self.bus = Bus()
        self.rig = SensorRig(cfg, self.bus, m, d, seed=seed)
        self.cam = GimbalCamera(m, self.bus, self.tcfg)
        self.enemy = en.EnemyDrone(ground=self.ground, base_xy=self.mast, range0=self.tcfg.assumed_range_m,
                                   agl=self.tcfg.assumed_agl, speed=enemy_speed, seed=seed)
        self.birds = en.BirdFlock(m, self.ground, world.birds, seed=seed)
        self.birds_near_route = birds_near_route
        self.link = LinkModel(availability, seed=seed)
        if availability >= 1.0:
            self.link.forced = True
        elif availability <= 0.0:
            self.link.forced = False
        self.acoustic = BaseAcoustic(self.mast, seed=seed)
        self.send_bearings = send_bearings
        self.bearings_sent = 0
        # forced losses (evaluation)
        self.forced_losses, self.occlusion_s, self.turn_deg = forced_losses, occlusion_s, turn_deg
        self.losses: list[LossEvent] = []
        self.loss_rng = np.random.default_rng(seed + 8800)
        self.steer_offset = 0.0
        self.hidden = False
        self._track_since: float | None = None
        self.enemy_geoms = [g for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name == "enemy"]
        self.enemy_group = int(m.geom_group[self.enemy_geoms[0]]) if self.enemy_geoms else 1
        z0, ground = self.z0, self.ground
        self.pilot = TrackerPilot(cfg, self.bus, self.rig.vehicle_params(), self.pad, self.tcfg, self.mast,
                                  terrain=lambda x, y: ground(x, y) - z0, detector=detector)
        self.dt = float(m.opt.timestep)
        self._next_world = 0.0
        # base side
        self.bearing: float | None = None  # latest bearing from keys / UDP / script: the enemy's direction follows it
        self.first_heard_t: float | None = None
        self.launch_t: float | None = None
        self.recall_pending = False
        self.received: list[EnemyReport] = []
        self.rx_t: list[float] = []  # when each report reached the base
        # truth / display
        self.enemy_trail: list[np.ndarray] = []
        self.truth_trail: list[np.ndarray] = []
        self.est_trail: list[np.ndarray] = []
        self.track_trail: list[np.ndarray] = []
        self._next_trail = 0.0
        self.enemy_hist_t: list[float] = []
        self.enemy_hist_p: list[np.ndarray] = []
        self.frames: list[FrameTruth] = []
        self.min_dist = math.inf
        self.state_time: dict[str, float] = {}
        self.banner: tuple[float, str, tuple] | None = None
        self.prev_state = self.pilot.state
        self._read_truth()
        self.est = self.pilot.pose()

    @property
    def t(self) -> float:
        return float(self.data.time)

    def _read_truth(self) -> None:
        d = self.data
        self.true_pos = d.qpos[0:3].copy()
        self.true_vel = d.qvel[0:3].copy()
        self.true_R = d.xmat[self.body].reshape(3, 3).copy()

    # ------------------------------------------------------------------ base

    def hear(self, az: float, src: str = "") -> str:
        """A bearing from the keys / UDP / script. First one: the enemy appears 500 m out and the base launches the
        drone; later ones steer the enemy. What the drone gets is the base's ACOUSTIC bearing to the enemy (UMA-16
        model, ~1 Hz, with sigma), only while the link is up."""
        az %= 360.0
        t = self.t
        self.bearing = az
        if self.first_heard_t is None:
            self.first_heard_t = t
            if self.birds_near_route:  # some birds over the approach, so the detector meets them
                u = en.az_dir(az)
                self.birds.place_near(np.array([u * r for r in np.linspace(120, 480, 6)]), 0.5)
        if self.pilot.on_ground and self.pilot.state == "WAIT":
            if self.pilot.launch(az, t, self.first_heard_t):
                self.launch_t = t
                return f"enemy heard at {az:05.1f} deg ({src}): LAUNCH"
        return f"enemy steered to {az:05.1f} deg ({src})"

    def recall(self) -> None:
        self.recall_pending = True

    # ------------------------------------------------------------------ stepping

    def advance(self, n: int) -> None:
        d, pilot = self.data, self.pilot
        for _ in range(n):
            self.rig.step()
            t = self.t
            if t >= self._next_world:
                self._next_world = t + WORLD_DT
                steer = None if self.bearing is None else (self.bearing + self.steer_offset) % 360.0
                pose = self.enemy.step(t, steer)
                en.set_enemy(self.model, d, pose)
                self.birds.step(t, d)
            fr = self.cam.step(d, seg=self.score)
            if fr is not None and self.score and pilot.state in ("SEARCH", "DETECTED", "TRACKING", "LOST"):
                self._score_frame(fr.t)
            up = self.link.step(t)
            if self.enemy.active and self.acoustic.due(t):
                az, sig = self.acoustic.measure(t, self.enemy.pose.pos)
                if up and self.send_bearings and not pilot.on_ground:
                    self.bus.publish(topics.UPLINK_BEARING, BearingUpdate(t, az, self.first_heard_t, sig))
                    self.bearings_sent += 1
            if self.forced_losses:
                self._losses(t)
            if up:
                if len(pilot.outbox):
                    for r in pilot.outbox.drain():
                        self.received.append(r)
                        self.rx_t.append(t)
                if self.recall_pending:
                    pilot.recall(t)
                    self.recall_pending = False
            st = pilot.state
            self.state_time[st] = self.state_time.get(st, 0.0) + self.dt
            if st != self.prev_state:
                self.prev_state = st
                self.banner = (t, f"{st}", None)
            if self.enemy.active:
                dd = float(np.linalg.norm(d.qpos[0:3] - self.enemy.pose.pos))
                if not pilot.on_ground:
                    self.min_dist = min(self.min_dist, dd)
            if t >= self._next_trail:
                self._next_trail = t + TRAIL_DT
                self._read_truth()
                self.est = pilot.pose()
                if self.enemy.active:
                    self.enemy_trail.append(self.enemy.pose.pos.copy())
                    self.enemy_hist_t.append(t)
                    self.enemy_hist_p.append(self.enemy.pose.pos.copy())
                if not pilot.on_ground:
                    self.truth_trail.append(self.true_pos.copy())
                    self.est_trail.append(self.est.pos.copy())
                trk = pilot.tracker.track
                if trk is not None and pilot.state in ("TRACKING", "LOST", "DETECTED"):
                    self.track_trail.append(trk.pos + [0.0, 0.0, self.z0])
        self._read_truth()
        self.est = pilot.pose()

    def _hide(self, hidden: bool) -> None:
        self.hidden = hidden
        for g in self.enemy_geoms:  # group 4 is not rendered by the gimbal camera (nor in its segmentation)
            self.model.geom_group[g] = 4 if hidden else self.enemy_group

    def _losses(self, t: float) -> None:
        p = self.pilot
        ev = self.losses[-1] if self.losses else None
        if ev is not None and ev.end_t is None:
            if self.hidden and t >= ev.t1:
                self._hide(False)
            ev.went_lost = ev.went_lost or p.state == "LOST"
            if not self.hidden and p.state == "TRACKING" and p.tracker.track is not None:
                err = float(np.linalg.norm(p.tracker.track.pos + [0, 0, self.z0] - self.enemy.pose.pos))
                if err < 15.0:
                    ev.reacq_t, ev.end_t = t, t
            if ev.end_t is None and t > ev.t1 + 45.0:
                ev.end_t = t
            return
        self._track_since = (self._track_since or t) if p.state == "TRACKING" else None
        if (len(self.losses) < self.forced_losses and self._track_since is not None and t - self._track_since > 10.0
                and (ev is None or t - ev.end_t > 10.0)):
            turn = float(self.loss_rng.choice([-1.0, 1.0])) * self.turn_deg
            self.steer_offset += turn
            self._hide(True)
            d = float(np.linalg.norm(self.data.qpos[0:3] - self.enemy.pose.pos))
            self.losses.append(LossEvent(t, t + self.occlusion_s, turn, d))
            self.banner = (t, f"FORCED LOSS: enemy hidden {self.occlusion_s:.0f} s, turns {turn:+.0f} deg", None)

    def _score_frame(self, t: float) -> None:
        """Detection truth for the frame just published (segmentation of the same render)."""
        cam, res = self.cam, self.pilot.last_result
        seg = cam.last_seg
        pos, fwd = cam.last_pose
        e = self.enemy.pose.pos
        uv = project(e[None], pos, fwd, 0.0, self.tcfg.cam_w, self.tcfg.cam_h, self.tcfg.vfov_deg())[0]
        in_fov = uv[2] > 0 and 0 <= uv[0] < self.tcfg.cam_w and 0 <= uv[1] < self.tcfg.cam_h
        names = cam.sr.inst_names
        ei = names.index("enemy")
        mask = seg == ei
        npx = int(mask.sum())
        box = None
        if npx:
            ys, xs = np.nonzero(mask)
            box = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
        det, best, fps = False, 0.0, []
        dets = [] if res is None or abs(res.t - t) > 1e-6 else res.dets
        thr = self.tcfg.drone_conf
        for dt_ in dets:
            if dt_.cls != "drone":
                continue
            on_enemy = box is not None and _overlap(dt_.box, box, pad=3.0)
            if on_enemy:
                best = max(best, dt_.conf)
                det = det or dt_.conf >= thr
            elif dt_.conf >= thr:  # a false drone: on a bird, or on anything else (tree, building, ground, sky)
                x0, y0, x1, y1 = (int(v) for v in dt_.box)
                patch = seg[max(y0 - 2, 0):y1 + 2, max(x0 - 2, 0):x1 + 2]
                hit = [names[i] for i in np.unique(patch) if i >= 0]
                fps.append("bird" if any(h.startswith("bird") for h in hit) else "other")
        self.frames.append(FrameTruth(t, self.pilot.state, float(np.linalg.norm(e - pos)), bool(in_fov), npx, box,
                                      det, best, fps))

    # ------------------------------------------------------------------ scoring

    def enemy_at(self, t: float) -> np.ndarray | None:
        if not self.enemy_hist_t:
            return None
        ts = np.asarray(self.enemy_hist_t)
        P = np.asarray(self.enemy_hist_p)
        if t <= ts[0]:
            return P[0]
        return np.array([np.interp(t, ts, P[:, k]) for k in range(3)])

    def report_errors(self) -> np.ndarray:
        """(N, 2) horizontal and 3D error of every report the base received (vs the true enemy at report time)."""
        out = []
        for r in self.received:
            e = self.enemy_at(r.t)
            if e is None:
                continue
            p = r.pos + np.array([0.0, 0.0, self.z0])
            out.append((float(np.linalg.norm(p[:2] - e[:2])), float(np.linalg.norm(p - e))))
        return np.asarray(out).reshape(-1, 2)

    def own_err(self) -> float:
        return float(np.linalg.norm(self.est.pos[:2] - self.true_pos[:2]))

    def true_dist(self) -> float | None:
        if not self.enemy.active:
            return None
        return float(np.linalg.norm(self.true_pos - self.enemy.pose.pos))

    def agl(self) -> float:
        return max(0.0, float(self.true_pos[2] - self.ground(self.true_pos[0], self.true_pos[1])))


class OracleDetector:
    """SIMULATION OF PERCEPTION - uses ground truth: the true boxes from the camera's segmentation buffer, as a
    perfect detector (`--detector oracle`, to test the flight logic without the network). Needs score=True."""

    last_ms = 0.0

    def __init__(self, cam: GimbalCamera, min_px: int = 4) -> None:
        self.cam, self.min_px = cam, min_px

    def detect(self, rgb: np.ndarray):
        from sfhack.perception.drone_detector import Det
        from sfhack.sim.gimbal_camera import boxes_from_seg

        if self.cam.last_seg is None:
            return []
        kept, _ = boxes_from_seg(self.cam.last_seg, self.cam.sr, min_side=self.min_px)
        return [Det(ew.CLASSES[c], 0.9, box) for c, _i, box, _n in kept]


def _overlap(a, b, pad: float = 0.0) -> bool:
    return not (a[2] < b[0] - pad or a[0] > b[2] + pad or a[3] < b[1] - pad or a[1] > b[3] + pad)
