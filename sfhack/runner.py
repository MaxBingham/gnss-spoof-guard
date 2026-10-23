"""Composition root: builds simulator + autopilot for one flight and runs it.

This is harness code (like server.py, recorder.py, eval/): it may read the truth to SCORE and
DISPLAY a flight, but it never feeds truth into the autopilot, except through the two sanctioned
doors: TruthPoseSource (pose-source switch) and the MOCK localizer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .autopilot import Autopilot
from .bus import Bus
from .config import SAMPLE_WORLD_DIR, Config, default_config
from .degradation.conditions import Conditions, ConditionsMsg
from .messages import Disturbance
from .perception.localizer import Localizer
from .planning.mission import Mission, a_to_b_plan, fly_x2_plan
from .sim.mujoco_sim import MujocoSim
from .sim.truth import TruthPoseSource, TruthProvider
from .world.onboard_map import OnboardMap
from .world.scene import ArenaWorld, PackageWorld

ALT_TEXTURE = "ortho_alt.jpg"  # "different imagery" variant of the world ground texture


@dataclass
class FlightResult:
    landed: bool
    success: bool
    final_error: float  # m, horizontal distance truth <-> B at the end
    t_end: float
    est_err_mean: float  # m, horizontal |estimate - truth| over the flight
    est_err_max: float
    loc_frames: int
    loc_fixes: int
    loc_ms_mean: float
    fixes_rejected: int
    outcome: str = ""  # ARRIVED | MISSED | MISSION_TIMEOUT
    health_states: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.outcome} {'SUCCESS' if self.success else 'FAIL'}: landed={self.landed} final error to B "
                f"{self.final_error:.2f} m, t={self.t_end:.1f} s, est. error mean {self.est_err_mean:.2f} m / "
                f"max {self.est_err_max:.2f} m, localizer {self.loc_fixes}/{self.loc_frames} fixes, "
                f"{self.loc_ms_mean:.1f} ms/frame, rejected {self.fixes_rejected}, health {self.health_states}")


def make_localizer(name: str, cfg: Config, truth: TruthProvider, seed: int = 0) -> Localizer | None:
    if name == "off":
        return None
    if name == "mock":
        from .perception.mock_localizer import MockLocalizer
        return MockLocalizer(cfg.localizer, truth, seed=seed)
    if name == "vision":
        from .perception.map_localizer import MapLocalizer
        return MapLocalizer(cfg.localizer)
    raise ValueError(f"unknown localizer {name!r} (off | mock | vision)")


class FlightSystem:
    def __init__(
        self,
        cfg: Config | None = None,
        world_dir: Path | str | None = SAMPLE_WORLD_DIR,
        pose_source: str = "ekf",
        localizer: str = "mock",
        map_source: str = "same",
        seed: int = 0,
        render_front: bool = False,
        recorder: Any | None = None,
        reuse_sim: MujocoSim | None = None,
        conditions: Conditions | None = None,
        schedule: list[tuple[float, dict[str, Any]]] | None = None,
        extension: Any | None = None,
    ) -> None:
        """`conditions`: initial degradation levers (default `cfg.conditions`). `schedule`: lever changes
        at sim times, [(t, {'gnss': 'drag:1.5@90', 'wind.speed': 6}), ...] (see sfhack.degradation.cli)."""
        self.cfg = cfg or default_config()
        self.bus = Bus()
        self.seed = seed
        self.world_dir = Path(world_dir) if world_dir is not None else None
        self.map_source = map_source
        mc = self.cfg.mission

        if self.world_dir is None:  # fly_x2.py arena, no onboard map
            world: Any = ArenaWorld()
            self.onboard_map = None
            start, yaw = np.zeros(2), 0.0
            plan = fly_x2_plan()
            self.goal_xy = plan[-1].pos[:2]
        else:
            texture = None
            if map_source == "different":
                if not (self.world_dir / ALT_TEXTURE).exists():
                    raise FileNotFoundError(f"{ALT_TEXTURE} missing in {self.world_dir} (build it with the world builder)")
                texture = ALT_TEXTURE
            world = PackageWorld(self.world_dir, texture_file=texture)
            self.onboard_map = OnboardMap(self.world_dir)  # onboard map always uses the package's ortho.jpg
            start, self.goal_xy = world.a_enu, world.b_enu
            d = self.goal_xy - start
            yaw = math.atan2(d[1], d[0])
            plan = a_to_b_plan(start, self.goal_xy, mc)

        self.extension = extension
        if extension is not None:  # V4 target landing etc.: decorates the world, may move the spawn point
            world, start, yaw = extension.setup_world(self, world, start, yaw)
        reuse = (reuse_sim.model, reuse_sim.data) if reuse_sim is not None else None
        self.sim = MujocoSim(self.cfg, self.bus, world, spawn_xy=(start[0], start[1]), spawn_yaw=yaw,
                             seed=seed, render_front=render_front, reuse=reuse)
        if conditions is not None:
            self.sim.set_conditions(conditions)
        self.schedule = sorted(schedule or [], key=lambda e: e[0])
        self.conditions_log: list[tuple[float, str]] = [(0.0, self.sim.conditions.summary())]  # harness log
        self.truth = TruthProvider(self.sim)
        start3 = np.array([start[0], start[1], 0.0])
        self.mission = (extension.make_mission(self, start3, yaw) if extension is not None
                        else Mission(plan, start3, yaw, mc))
        self.autopilot = Autopilot(
            self.cfg, self.bus, self.sim.vehicle_params(), self.mission, start3, yaw,
            onboard_map=self.onboard_map,
            localizer=make_localizer(localizer, self.cfg, self.truth, seed) if self.onboard_map else None,
            truth_pose_source=TruthPoseSource(self.truth),
            pose_source=pose_source,
        )
        self.localizer_name = localizer if self.onboard_map else "off"
        if extension is not None:
            extension.attach(self)
        self.recorder = recorder
        if recorder is not None:
            recorder.attach(self)

        # Scoring log (truth vs estimate), 10 Hz
        self.log_t: list[float] = []
        self.log_truth: list[np.ndarray] = []
        self.log_est: list[np.ndarray] = []
        self.log_sigma2: list[float] = []
        self._next_log = 0.0
        self.health_seen: list[str] = []

    # ------------------------------------------------------------------ control

    @property
    def t(self) -> float:
        return self.sim.time

    @property
    def done(self) -> bool:
        return self.mission.done

    def set_pose_source(self, name: str) -> None:
        self.autopilot.set_pose_source(name)

    def set_localizer(self, name: str) -> None:
        if self.onboard_map is None:
            name = "off"
        self.autopilot.set_localizer(make_localizer(name, self.cfg, self.truth, self.seed))
        self.localizer_name = name

    @property
    def conditions(self) -> Conditions:
        """A copy of the levers in effect (change them with set_conditions / update_conditions)."""
        return self.sim.conditions.copy()

    def set_conditions(self, conditions: Conditions) -> None:
        """Replace all levers now (published on the `conditions` bus topic; the simulator adopts a copy)."""
        from . import bus as topics
        conditions.validate()
        self.bus.publish(topics.CONDITIONS, ConditionsMsg(self.t, conditions.copy()))
        self.conditions_log.append((self.t, conditions.summary()))

    def update_conditions(self, assignments: dict[str, Any] | str | None = None, **kw: Any) -> Conditions:
        """Change some levers now: update_conditions('gnss=drag:1.5@90,wind.speed=6') or
        update_conditions(gnss__mode='jam'). Returns the new levers."""
        c = self.conditions.update(assignments, **kw)
        self.set_conditions(c)
        return c

    def disturb(self, kind: str, value: Any = None, duration: float = 0.0) -> None:
        from . import bus as topics
        self.bus.publish(topics.DISTURBANCE, Disturbance(self.t, kind, value, duration))

    def step(self) -> None:
        while self.schedule and self.t >= self.schedule[0][0]:
            self.update_conditions(self.schedule.pop(0)[1])
        self.sim.step()
        if self.t >= self._next_log:
            self._next_log = self.t + 0.1
            est = self.autopilot.pose()
            self.log_t.append(self.t)
            self.log_truth.append(self.truth.state().pos)
            self.log_est.append(est.pos)
            self.log_sigma2.append(est.sigma2_horizontal())
            hs = self.autopilot.health_state.state.value
            if not self.health_seen or self.health_seen[-1] != hs:
                self.health_seen.append(hs)

    def run(self, max_t: float | None = None, settle_s: float = 1.0) -> FlightResult:
        """Run until landed (+ settle_s) or max_t of sim time."""
        max_t = max_t if max_t is not None else self.cfg.sim.max_time_s
        landed_at: float | None = None
        while self.t < max_t:
            self.step()
            if self.done:
                landed_at = landed_at if landed_at is not None else self.t
                if self.t - landed_at >= settle_s:
                    break
        r = self.result()
        self.autopilot.log_decision(self.t, f"{r.outcome}, final error {r.final_error:.1f} m")  # harness: scored on truth
        return r

    def result(self) -> FlightResult:
        truth = self.truth.state()
        final_err = float(np.linalg.norm(truth.pos[:2] - self.goal_xy))
        landed = self.done and truth.pos[2] < 0.5
        if self.log_t:
            errs = np.linalg.norm(np.array(self.log_truth)[:, :2] - np.array(self.log_est)[:, :2], axis=1)
            e_mean, e_max = float(errs.mean()), float(errs.max())
        else:
            e_mean = e_max = 0.0
        ap = self.autopilot
        outcome = "MISSION_TIMEOUT" if not self.done else ("ARRIVED" if landed and final_err <= self.cfg.mission.success_radius else "MISSED")
        return FlightResult(
            landed=landed,
            success=landed and final_err <= self.cfg.mission.success_radius,
            final_error=final_err,
            t_end=self.t,
            est_err_mean=e_mean,
            est_err_max=e_max,
            loc_frames=ap.loc_frames,
            loc_fixes=ap.loc_fixes,
            loc_ms_mean=ap.loc_ms_total / ap.loc_frames if ap.loc_frames else 0.0,
            fixes_rejected=ap.ekf.n_fix_rejected,
            health_states=list(self.health_seen),
            outcome=outcome,
        )

    def close(self) -> None:
        if self.recorder is not None:
            self.recorder.close()
        self.sim.close()
