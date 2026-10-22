"""V3 composition root (harness): world package + simulator + NavAutopilot for one flight, scored on truth.

Harness code like runner.py: it reads the truth to SCORE and DISPLAY (error, crash check, landing error) but
never feeds it to the autopilot (the only door is TruthPoseSource, pose_source="truth", for tuning).

Configurations (navigation sources; GNSS always disabled):
  ins             INS only (AHRS + EKF dead reckoning)
  ins_trn         + terrain-referenced navigation
  ins_ibn         + image-based navigation
  ins_trn_ibn     all sources
  mismatch_heavy  all sources with the heavily mismatched onboard map (onboard_heavy/)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .bus import Bus
from .config import Config, default_config
from .messages import Disturbance
from .nav_autopilot import NavAutopilot
from .sim.mujoco_sim import MujocoSim
from .sim.truth import TruthPoseSource, TruthProvider
from .world.build_v3 import worlds_dir
from .world.onboard_terrain import OnboardTerrainMap
from .world.scene import TerrainWorld

CONFIGS: dict[str, tuple[set[str], str]] = {
    "ins": (set(), "onboard"),
    "ins_trn": ({"trn"}, "onboard"),
    "ins_ibn": ({"ibn"}, "onboard"),
    "ins_trn_ibn": ({"trn", "ibn"}, "onboard"),
    "mismatch_heavy": ({"trn", "ibn"}, "onboard_heavy"),
}


@dataclass
class NavResult:
    world: str
    config: str
    seed: int
    outcome: str  # LANDED | EMERGENCY_LANDED | CRASH | TIMEOUT
    success: bool
    final_error: float  # m, truth landing position (or end position) to B
    arrival_error: float  # m, truth to B when the vehicle declared 'above B' (before the precision landing)
    est_err_mean: float
    est_err_max: float
    est_err_end: float
    max_sigma2: float  # m, max 2-sigma during cruise
    t_end: float
    time_degraded: float  # s in DEGRADED + HOLD during cruise
    exclusions: int
    resets: int
    fixes_trn: int
    fixes_ibn: int
    accepted_trn: int
    accepted_ibn: int
    route_m: float
    wall_s: float
    decisions: list[tuple[float, str]] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.world}/{self.config} seed {self.seed}: {self.outcome} {'SUCCESS' if self.success else 'FAIL'} "
                f"final {self.final_error:.1f} m, arrival {self.arrival_error:.1f} m, est err mean "
                f"{self.est_err_mean:.1f} / max {self.est_err_max:.1f} m, max 2-sigma {self.max_sigma2:.1f} m, "
                f"degraded {self.time_degraded:.0f} s, fixes trn {self.accepted_trn}/{self.fixes_trn} "
                f"ibn {self.accepted_ibn}/{self.fixes_ibn}, exclusions {self.exclusions}, t {self.t_end:.0f} s "
                f"(wall {self.wall_s:.0f} s)")


class NavFlightSystem:
    def __init__(self, world: str = "s1", config: str = "ins_trn_ibn", seed: int = 0, cfg: Config | None = None,
                 variant: str | None = None, plan_informed: bool = True, pose_source: str = "ekf",
                 world_dir: Path | str | None = None) -> None:
        self.cfg = cfg or default_config()
        self.bus = Bus()
        self.world_name = world
        self.config = config
        self.seed = seed
        sources, default_variant = CONFIGS[config]
        pkg = Path(world_dir) if world_dir else worlds_dir() / world
        self.world = TerrainWorld(pkg)
        self.onboard_map = OnboardTerrainMap(pkg, variant or default_variant)
        sc = self.onboard_map.scenario
        self.scenario = sc
        a, b = self.world.a_enu, self.world.b_enu
        yaw = math.atan2(b[1] - a[1], b[0] - a[0])
        self.sim = MujocoSim(self.cfg, self.bus, self.world, spawn_xy=(a[0], a[1]), spawn_yaw=yaw, seed=seed,
                             spawn_z=None, altimeter=sc.altimeter, down_cam_active=False)
        self.truth = TruthProvider(self.sim)
        self.autopilot = NavAutopilot(self.cfg, self.bus, self.sim.vehicle_params(), self.onboard_map, sources, yaw,
                                      seed=seed, plan_informed=plan_informed,
                                      truth_pose_source=TruthPoseSource(self.truth), pose_source=pose_source)
        self.goal_xy = b
        self.max_t = self.autopilot.expected_time() * 2.0 + 60.0
        self.log_t: list[float] = []
        self.log_truth: list[np.ndarray] = []
        self.log_est: list[np.ndarray] = []
        self.log_sigma2: list[float] = []
        self._next_log = 0.0
        self.crashed = False
        self.arrival_error = math.nan
        self._wall0 = time.time()

    @property
    def t(self) -> float:
        return self.sim.time

    @property
    def done(self) -> bool:
        return self.autopilot.mission.done or self.crashed

    def disturb(self, kind: str, value: Any = None, duration: float = 0.0) -> None:
        from . import bus as topics
        self.bus.publish(topics.DISTURBANCE, Disturbance(self.t, kind, value, duration))

    def step(self) -> None:
        self.sim.step()
        if self.t >= self._next_log:
            self._next_log = self.t + 0.1
            tr = self.truth.state()
            est = self.autopilot.pose()
            self.log_t.append(self.t)
            self.log_truth.append(tr.pos)
            self.log_est.append(est.pos)
            self.log_sigma2.append(est.sigma2_horizontal())
            m = self.autopilot.mission
            if m.arrival_t is not None and math.isnan(self.arrival_error):
                self.arrival_error = float(np.hypot(*(tr.pos[:2] - self.goal_xy)))
            # crash check (harness, truth): below the simulated surface (ray cast) or touching it while flying
            if m.phase not in ("PREFLIGHT", "LANDED", "LAND") and self.t > self.cfg.mission.preflight_s + 3:
                if tr.pos[2] < self.sim.surface_height(tr.pos[:2]) - 0.5 or (
                        self.truth.in_contact() and m.phase not in ("DESCEND", "LAND", "EMERGENCY_LAND")):
                    self.crashed = True
                    self.autopilot.log_decision(self.t, f"CRASH (harness, truth) at ({tr.pos[0]:.0f}, {tr.pos[1]:.0f})")

    def run(self, max_t: float | None = None, settle_s: float = 1.0) -> NavResult:
        max_t = max_t or self.max_t
        landed_at: float | None = None
        while self.t < max_t:
            self.step()
            if self.done:
                landed_at = landed_at if landed_at is not None else self.t
                if self.crashed or self.t - landed_at >= settle_s:
                    break
        r = self.result()
        self.autopilot.log_decision(self.t, f"{r.outcome} ({'success' if r.success else 'fail'}), final error "
                                            f"{r.final_error:.1f} m, arrival error {r.arrival_error:.1f} m (harness, truth)")
        r.decisions = list(self.autopilot.decisions)
        return r

    def result(self) -> NavResult:
        ap = self.autopilot
        tr = self.truth.state()
        final_err = float(np.hypot(*(tr.pos[:2] - self.goal_xy)))
        if self.log_t:
            errs = np.hypot(*(np.array(self.log_truth)[:, :2] - np.array(self.log_est)[:, :2]).T)
            e_mean, e_max, e_end = float(errs.mean()), float(errs.max()), float(errs[-1])
        else:
            e_mean = e_max = e_end = 0.0
        m = ap.mission
        if self.crashed:
            outcome = "CRASH"
        elif not m.done:
            outcome = "TIMEOUT"
        elif any(text.startswith("EMERGENCY_LAND") for _, text in m.events) or any(
                text.startswith("EMERGENCY_LAND") for _, text in ap.decisions):
            outcome = "EMERGENCY_LANDED"
        else:
            outcome = "LANDED"
        sc = self.scenario
        if sc.kind == "urban":
            success = outcome == "LANDED" and final_err <= sc.success_radius
        else:  # S2: arrival accuracy before the precision landing (landing patch IBN refines it)
            success = outcome == "LANDED" and not math.isnan(self.arrival_error) and self.arrival_error <= sc.success_radius
        st = ap.integrity.src
        ti = ap.health.time_in
        return NavResult(
            world=self.world_name, config=self.config, seed=self.seed, outcome=outcome, success=success,
            final_error=final_err, arrival_error=self.arrival_error, est_err_mean=e_mean, est_err_max=e_max,
            est_err_end=e_end, max_sigma2=ap.max_sigma2_cruise, t_end=self.t,
            time_degraded=ti.get("DEGRADED", 0.0) + ti.get("HOLD", 0.0),
            exclusions=sum(s.n_excluded_events for s in st.values()),
            resets=sum(1 for _, x in ap.decisions if x.startswith(("INS_RESET", "RELOCALIZE", "TRN_RESET"))),
            fixes_trn=st["trn"].n_fix, fixes_ibn=st["ibn"].n_fix, accepted_trn=st["trn"].n_accepted,
            accepted_ibn=st["ibn"].n_accepted, route_m=ap.route.length, wall_s=time.time() - self._wall0,
        )

    def close(self) -> None:
        self.sim.close()
