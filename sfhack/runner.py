"""Composition root: builds simulator + autopilot for one flight and runs it.

This is harness code (like server.py, recorder.py, eval/): it may read the truth to SCORE and
DISPLAY a flight, but it never feeds truth into the autopilot, except through the two sanctioned
doors: TruthPoseSource (pose-source switch) and the MOCK localizer.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .autopilot import Autopilot
from .bus import Bus
from .config import SAMPLE_WORLD_DIR, Config, data_dir, default_config
from .degradation.conditions import Conditions, ConditionsMsg
from .messages import Disturbance
from .perception.localizer import Localizer
from .planning.mission import Mission, a_to_b_plan, cruise_at_agl, fly_x2_plan, route_plan
from .sim.mujoco_sim import MujocoSim
from .sim.truth import TruthPoseSource, TruthProvider
from .world.onboard_map import OnboardMap
from .world.scene import ArenaWorld, PackageWorld

ALT_TEXTURE = "ortho_alt.jpg"
NAV_CONFIGS = ("imu", "flow", "lights")  # recon-nav: IMU + compass + baro; + optical flow; + flow + base lights  # "different imagery" variant of the world ground texture


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


def resolve_mission(mission: str | Path) -> Path:
    """Mission id (under $SFHACK_DATA/missions) or a mission directory -> the directory."""
    p = Path(mission)
    d = p if (p / "world").is_dir() else data_dir() / "missions" / str(mission)
    if not (d / "world" / "manifest.json").exists() or not (d / "onboard" / "manifest.json").exists():
        raise FileNotFoundError(f"mission {mission!r} not prepared ({d}); run: python -m sfhack.missionprep build")
    return d


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
        mission: str | Path | None = None,
        night: bool = False,
        cruise_agl: float | None = None,
        thermal_renderer: str = "auto",
        thermal_contrast: float = 1.0,
        night_route: bool = True,
        nav: str | None = None,
        plan: list | None = None,
        tail_to_base: bool = False,
        thermal_camera: bool | None = None,
    ) -> None:
        """`conditions`: initial degradation levers (default `cfg.conditions`). `schedule`: lever changes
        at sim times, [(t, {'gnss': 'drag:1.5@90', 'wind.speed': 6}), ...] (see sfhack.degradation.cli).
        `mission`: a mission prepared by sfhack.missionprep (id under $SFHACK_DATA/missions, or its directory):
        the simulated world (truth) comes from <mission>/world, the onboard map and route from <mission>/onboard.
        `night` (night-nav, mission packages with thermal files): the RGB down camera is useless in the dark, so the
        vision localizer is off; a downward thermal camera (sim/thermal_sensor.py, truth: world/thermal_truth.npz)
        feeds the onboard ThermalLocalizer (onboard/thermal_prior.npz). `cruise_agl`: fly the package route this
        many m above the onboard DTM (cruise_at_agl). `thermal_renderer`: auto | full | stub.
        `night_route` (night-demo, night only, needs onboard/thermal_prior.npz): re-plan on the onboard map with the
        route choice of sfhack/planning/night_route.py (straight while warm road is expected in the thermal footprint,
        else along the roads), at `cruise_agl` (default: the package's min_agl_m). Result in `self.night_route`.
        `nav` (recon-nav, mission packages): the recon navigation configs, NAV_CONFIGS: "imu" (IMU + compass + baro),
        "flow" (+ optical flow on the downward thermal camera), "lights" (+ base-light fixes from the rear NIR camera).
        No RGB vision, no thermal map matching, no lidar TRN. `thermal_camera`: render the downward thermal camera
        (default: when the config uses it; recon-mission's detection wants it always). `plan`: waypoints instead of
        the package route (e.g. planning/recon_route.out_and_back_plan); `tail_to_base`: yaw keeps the rear camera
        on the base (TailToBaseMission)."""
        self.cfg = cfg or default_config()
        self.night = night
        self.nav = nav
        if nav is not None:
            if nav not in NAV_CONFIGS:
                raise ValueError(f"nav {nav!r} not in {NAV_CONFIGS}")
            localizer = "off"
            c = self.cfg
            self.cfg = dataclasses.replace(
                c, trn=dataclasses.replace(c.trn, enabled=False),  # recon drone: no lidar
                ekf=dataclasses.replace(c.ekf, gnss_free_track=False),  # GNSS honest, then jammed: no spoof track
                sim=dataclasses.replace(c.sim, down_cam_hz=1e-6),  # RGB down camera: useless at night, not rendered
                ahrs=dataclasses.replace(c.ahrs, tilt_bias_ki=c.ahrs.tilt_bias_ki or c.thermal.night_tilt_bias_ki))
        self.night_route = None  # NightRouteResult when the night route choice re-planned the route
        if night:
            localizer = "off"
            if self.cfg.ahrs.tilt_bias_ki == 0.0 and self.cfg.thermal.night_tilt_bias_ki > 0.0:
                # night (GNSS loss expected): learn the roll / pitch gyro bias from the EKF's tilt corrections
                # (flight-local copy of the AHRS section; an explicit ahrs.tilt_bias_ki wins)
                self.cfg = dataclasses.replace(self.cfg, ahrs=dataclasses.replace(
                    self.cfg.ahrs, tilt_bias_ki=self.cfg.thermal.night_tilt_bias_ki))
        self.bus = Bus()
        self.seed = seed
        self.mission_dir = resolve_mission(mission) if mission is not None else None
        if self.mission_dir is not None:
            world_dir = self.mission_dir / "world"
        self.world_dir = Path(world_dir) if world_dir is not None else None
        self.map_source = map_source
        mc = self.cfg.mission
        plan_override = plan
        spawn_z = 0.0
        self.onboard_dir: Path | None = None

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
            # Onboard map: the mission package as uploaded to the drone (verified copy), else as built; the
            # sample world uses its own package for both.
            onboard_dir = self.world_dir
            if self.mission_dir is not None:
                drone = self.mission_dir / "drone"
                onboard_dir = drone if (drone / "upload_receipt.json").exists() else self.mission_dir / "onboard"
            self.onboard_dir = onboard_dir
            self.onboard_map = OnboardMap(onboard_dir)  # always the package's own ortho.jpg
            start, self.goal_xy = world.a_enu, world.b_enu
            d = self.goal_xy - start
            yaw = math.atan2(d[1], d[0])
            route = (self.onboard_map.mission or {}).get("route")
            plan = route_plan(route, mc) if route else a_to_b_plan(start, self.goal_xy, mc)
            if plan_override is not None:
                plan = list(plan_override)
                far = max(plan, key=lambda w: float(np.linalg.norm(w.pos[:2] - start)))
                yaw = math.atan2(far.pos[1] - start[1], far.pos[0] - start[0])
                self.goal_xy = plan[-1].pos[:2].copy()
            prior_dir = next((d for d in (onboard_dir, self.mission_dir / "onboard" if self.mission_dir else None)
                              if d is not None and (d / "thermal_prior.npz").exists()), None)
            if plan_override is None and night and night_route and prior_dir is not None and self.onboard_map.dsm is not None:
                from .planning.night_route import NightRouteParams, plan_night_route
                from .thermal.prior import load_prior

                agl = cruise_agl if cruise_agl is not None else float(
                    (self.onboard_map.mission or {}).get("cruise", {}).get("min_agl_m", 30.0))
                self.night_route = plan_night_route(self.onboard_map, load_prior(prior_dir),
                                                    NightRouteParams(cruise_agl_m=agl), mc)
                plan = route_plan(self.night_route.route, mc)
            elif plan_override is None and cruise_agl is not None:
                plan = cruise_at_agl(plan, self.onboard_map.ground_z, cruise_agl)
            spawn_z = world.ground_z(*start)  # harness: the true ground under A (z datum: 0 for mission worlds)

        self.world = world  # simulator side: true terrain for scoring and display
        reuse = (reuse_sim.model, reuse_sim.data) if reuse_sim is not None else None
        self.sim = MujocoSim(self.cfg, self.bus, world, spawn_xy=(start[0], start[1]), spawn_yaw=yaw,
                             seed=seed, render_front=render_front, reuse=reuse, spawn_z=spawn_z)
        if conditions is not None:
            self.sim.set_conditions(conditions)
        self.schedule = sorted(schedule or [], key=lambda e: e[0])
        self.conditions_log: list[tuple[float, str]] = [(0.0, self.sim.conditions.summary())]  # harness log
        self.truth = TruthProvider(self.sim)
        start3 = np.array([start[0], start[1], 0.0])
        self.thermal_sensor = None
        thermal_loc = None
        if night and self.cfg.thermal.method != "off":  # "off": IMU + baro + TRN only (baseline)
            if self.mission_dir is None or self.onboard_dir is None:
                raise ValueError("night mode needs a mission package (thermal_truth / thermal_prior)")
            from .perception.thermal_localizer import ThermalLocalizer
            from .sim.thermal_sensor import ThermalSensor
            self.thermal_sensor = ThermalSensor(self.sim, self.bus, self.mission_dir, world, hz=self.cfg.thermal.camera_hz,
                                                seed=seed, renderer=thermal_renderer, contrast=thermal_contrast)
            prior = self.onboard_dir / "thermal_prior.npz"
            if not prior.exists():  # uploaded before the thermal prior was built: use the built package's
                prior = self.mission_dir / "onboard" / "thermal_prior.npz"
            thermal_loc = ThermalLocalizer(self.onboard_dir, self.cfg.thermal, onboard_map=self.onboard_map,
                                           prior_path=prior)
        if nav is not None:
            thermal_loc = None  # recon: the thermal camera finds things and gives flow, it does not match a map
            if thermal_camera is None:
                thermal_camera = nav in ("flow", "lights")
            if thermal_camera and self.thermal_sensor is None:
                from .sim.thermal_sensor import ThermalSensor
                self.thermal_sensor = ThermalSensor(self.sim, self.bus, self.mission_dir, world, hz=self.cfg.thermal.camera_hz,
                                                    seed=seed, renderer=thermal_renderer, contrast=thermal_contrast)
        flow = light_loc = None
        self.rear_camera = None
        self.base_lights_truth = None
        if nav in ("flow", "lights"):
            from .perception.optical_flow import OpticalFlow
            flow = OpticalFlow(self.cfg.flow)
        if nav == "lights":
            from .perception.light_localizer import LightLocalizer
            from .sim.rear_camera_stub import RearCameraStub
            self.base_lights_truth, onboard_lights = self._base_lights(start, seed)
            self.rear_camera = RearCameraStub(self.sim, self.bus, self.base_lights_truth, self.cfg.rear_camera, seed=seed)
            light_loc = LightLocalizer(onboard_lights, self.cfg.lights, self.cfg.rear_camera)
        ground = self.onboard_map.ground_z if self.onboard_map is not None else None
        if tail_to_base:
            from .planning.recon_route import TailToBaseMission
            self.mission: Mission = TailToBaseMission(plan, start3, yaw, mc, ground_z=ground, base_xy=start)
        else:
            self.mission = Mission(plan, start3, yaw, mc, ground_z=ground)
        self.autopilot = Autopilot(
            self.cfg, self.bus, self.sim.vehicle_params(), self.mission, start3, yaw,
            onboard_map=self.onboard_map,
            localizer=make_localizer(localizer, self.cfg, self.truth, seed) if self.onboard_map else None,
            truth_pose_source=TruthPoseSource(self.truth),
            pose_source=pose_source,
            seed=seed,
            thermal_localizer=thermal_loc,
            flow=flow,
            light_localizer=light_loc,
        )
        self.localizer_name = localizer if self.onboard_map else "off"
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

    def _base_lights(self, a_xy: np.ndarray, seed: int) -> tuple[list, list]:
        """(TRUE lights for the simulated camera, onboard surveyed list). recon-world's files when the mission has
        them (world/base_lights.json = truth, onboard/base_lights.json = survey), else the stub fence at A."""
        from .perception.light_localizer import load_base_lights
        from .sim.rear_camera_stub import stub_base_lights, surveyed

        truth_f = self.world_dir / "base_lights.json" if self.world_dir is not None else None
        onb_f = self.onboard_dir / "base_lights.json" if self.onboard_dir is not None else None
        if truth_f is not None and truth_f.exists() and onb_f is not None and onb_f.exists():
            return load_base_lights(truth_f), load_base_lights(onb_f)
        truth = stub_base_lights(np.asarray(a_xy, float), self.world.ground_z if self.world_dir is not None else (lambda x, y: 0.0))
        return truth, surveyed(truth, self.cfg.lights.survey_std_m, seed)

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
        if self.thermal_sensor is not None:
            self.thermal_sensor.step()
        if self.rear_camera is not None:
            self.rear_camera.step()
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

    def true_ground_z(self, x: float, y: float) -> float:
        """Harness: the TRUE ground height (terrain worlds), for scoring and display only."""
        return self.world.ground_z(x, y) if isinstance(self.world, PackageWorld) else 0.0

    def result(self) -> FlightResult:
        truth = self.truth.state()
        final_err = float(np.linalg.norm(truth.pos[:2] - self.goal_xy))
        landed = self.done and truth.pos[2] - self.true_ground_z(truth.pos[0], truth.pos[1]) < 0.5
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
