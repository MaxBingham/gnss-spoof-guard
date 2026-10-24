"""MuJoCo physics + simulated sensors for the Skydio X2.

The simulator is the ONLY place that owns the true state. It publishes noisy sensor messages
on the bus and applies the latest `MotorCmd` it receives. Truth leaves this package only via
`sfhack.sim.truth` (for the dashboard, scoring, the MOCK localizer and the TruthPoseSource).
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Protocol

import mujoco
import numpy as np

from .. import bus as topics
from ..bus import Bus
from ..config import X2_DIR, Config
from ..control.mixer import VehicleParams
from ..degradation import models as lever
from ..degradation.conditions import Conditions, ConditionsMsg
from ..messages import BaroMsg, CameraFrame, Disturbance, GnssMsg, ImuMsg, LidarMsg, MagMsg, MotorCmd, RangeMsg

GRAVITY = 9.81


class WorldModel(Protocol):
    """Anything that can decorate the X2 scene spec with ground, buildings, trees."""

    def add_to_spec(self, spec: mujoco.MjSpec) -> None: ...


def _quat_from_axes(x: list[float], y: list[float]) -> np.ndarray:
    """Quaternion of a frame whose x and y axes (expressed in the parent frame) are given."""
    xa = np.asarray(x, float) / np.linalg.norm(x)
    ya = np.asarray(y, float) / np.linalg.norm(y)
    za = np.cross(xa, ya)
    mat = np.column_stack([xa, ya, za]).flatten()
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat)
    return q


def yaw_from_R(R: np.ndarray) -> float:
    return math.atan2(R[1, 0], R[0, 0])


def announce_scene_compile() -> None:
    """One line before the model compile, so the first launch is not a silent hang.

    Restart passes `reuse` and never reaches `spec.compile()`, so this stays quiet the second time.
    """
    if getattr(announce_scene_compile, "done", False):
        return
    announce_scene_compile.done = True  # type: ignore[attr-defined]
    print("compiling MuJoCo scene, once…", flush=True)


class _RateTimer:
    def __init__(self, hz: float, t0: float = 0.0) -> None:
        self.period = 1.0 / hz
        self.next_t = t0

    def due(self, t: float) -> bool:
        if t + 1e-9 >= self.next_t:
            self.next_t += self.period
            if self.next_t <= t:  # never fall behind by more than one period
                self.next_t = t + self.period
            return True
        return False


class MujocoSim:
    def __init__(
        self,
        cfg: Config,
        bus: Bus,
        world: WorldModel | None = None,
        spawn_xy: tuple[float, float] = (0.0, 0.0),
        spawn_yaw: float = 0.0,
        seed: int = 0,
        render_front: bool = False,
        reuse: tuple[mujoco.MjModel, mujoco.MjData] | None = None,
        spawn_z: float = 0.0,
    ) -> None:
        """`reuse` = (model, data) of an earlier flight in the same world: skips the model build and resets
        that data in place (the live demo restarts inside one open viewer window). `spawn_z` = ground height
        at the spawn point (terrain worlds)."""
        self.cfg = cfg
        self.bus = bus
        self.rng = np.random.default_rng(seed)
        self.range_rng = np.random.default_rng(seed + 2000)  # own stream: adding it left the others unchanged
        self.render_front = render_front
        # Degradation levers: own RNG streams, so nominal flights draw exactly what they drew before.
        self.conditions = cfg.conditions.copy()
        self.gnss = lever.GnssModel(np.random.default_rng(seed + 3000), cfg.noise.gnss_noise)
        self.wind_model = lever.WindModel(np.random.default_rng(seed + 3001))
        self.weather_rng = np.random.default_rng(seed + 3002)
        self.range_lever_rng = np.random.default_rng(seed + 3003)
        self.lidar_rng = np.random.default_rng(seed + 3004)
        self.weather = lever.weather_effect(self.conditions.weather)
        self.frames_dropped = 0
        if reuse is not None:
            self.model, self.data = reuse
            mujoco.mj_resetData(self.model, self.data)
        else:
            self.model = self._build_model(world)
            self.data = mujoco.MjData(self.model)
        m, d = self.model, self.data
        self.body_id = m.body("x2").id
        self.cam_ids = {"down": m.camera("down").id, "front": m.camera("front").id}

        # Spawn on the ground at A, motors off.
        d.qpos[0:3] = [spawn_xy[0], spawn_xy[1], spawn_z + 0.005]
        d.qpos[3:7] = [math.cos(spawn_yaw / 2), 0.0, 0.0, math.sin(spawn_yaw / 2)]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        self.ctrl = np.zeros(m.nu)

        self.renderer: mujoco.Renderer | None = None  # created lazily in the thread that renders
        self.frame_seq = 0
        self._cam_truth: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

        n = cfg.noise
        self.gyro_bias = self.rng.normal(0.0, n.gyro_bias_init, 3)
        self.accel_bias = self.rng.normal(0.0, n.accel_bias_init, 3)
        self.baro_bias = 0.0
        self.mag_bias = math.radians(self.rng.normal(0.0, n.mag_bias_deg))

        s = cfg.sim
        self.timers = {
            "imu": _RateTimer(s.imu_hz),
            "baro": _RateTimer(s.baro_hz),
            "range": _RateTimer(s.range_hz),
            "mag": _RateTimer(s.mag_hz),
            "down": _RateTimer(s.down_cam_hz),
            "front": _RateTimer(s.front_cam_hz),
            "gnss": _RateTimer(s.gnss_hz),
            "lidar": _RateTimer(s.lidar_hz),
        }
        self.lidar_dirs = lever.lidar_beam_dirs_body()

        # Disturbance state
        self.air_velocity = np.zeros(3)  # wind + gust (world), for display
        self.wind_force = np.zeros(3)
        self.camera_blackout_until = -1.0  # sim time; math.inf = until cleared (legacy camera_blackout)

        bus.subscribe(topics.MOTOR, self._on_motor)
        bus.subscribe(topics.DISTURBANCE, self.apply_disturbance)
        bus.subscribe(topics.CONDITIONS, self.set_conditions)

    # ------------------------------------------------------------------ model

    def _build_model(self, world: WorldModel | None) -> mujoco.MjModel:
        spec = mujoco.MjSpec.from_file(str(X2_DIR / "scene.xml"))
        spec.option.timestep = self.cfg.sim.timestep
        cam = self.cfg.camera
        drone = spec.body("x2")
        # MuJoCo cameras look along their -z axis. Down camera: image top = forward, image right = right.
        drone.add_camera(name="down", pos=[0, 0, -0.02], fovy=cam.down_fovy_deg,
                         quat=_quat_from_axes([0, -1, 0], [1, 0, 0]))
        # Thermal (LWIR) camera on the same mount; MuJoCo's fovy is the vertical FOV.
        thermal_fovy = math.degrees(2 * math.atan(math.tan(math.radians(cam.thermal_hfov_deg) / 2)
                                                  * cam.thermal_height / cam.thermal_width))
        drone.add_camera(name="thermal", pos=[0, 0, -0.02], fovy=thermal_fovy,
                         quat=_quat_from_axes([0, -1, 0], [1, 0, 0]))
        # Downward rangefinder: MuJoCo casts along the site's +z axis (flipped to point down), ignoring the X2.
        drone.add_site(name="rangefinder", pos=[0, 0, 0.01], quat=[0, 1, 0, 0])
        rf = spec.add_sensor(name="rangefinder", type=mujoco.mjtSensor.mjSENS_RANGEFINDER,
                             objtype=mujoco.mjtObj.mjOBJ_SITE, objname="rangefinder")
        rf.intprm[0] = 1  # output: distance only
        tilt = math.radians(15.0)  # front camera pitched 15 deg down
        drone.add_camera(name="front", pos=[0.12, 0, 0.03], fovy=cam.front_fovy_deg,
                         quat=_quat_from_axes([0, -1, 0], [math.sin(tilt), 0, math.cos(tilt)]))
        # Rear NIR camera (sim/rear_camera.py): optical axis body -x, pitched down; image right = body +y (left side).
        rp = math.radians(cam.rear_pitch_down_deg)
        rear_vfov = math.degrees(2 * math.atan(math.tan(math.radians(cam.rear_hfov_deg) / 2)
                                               * cam.rear_height / cam.rear_width))
        drone.add_camera(name="rear", pos=[-0.12, 0, 0.03], fovy=rear_vfov,
                         quat=_quat_from_axes([0, 1, 0], [-math.sin(rp), 0, math.cos(rp)]))
        spec.visual.global_.offwidth = max(cam.down_width, cam.front_width, 1280)  # 1280x720: demo video
        spec.visual.global_.offheight = max(cam.down_height, cam.front_height, 720)
        # Large outdoor scene: near/far planes are fractions of stat.extent.
        spec.stat.extent = 100.0
        spec.visual.map.znear = 0.001  # 0.1 m
        spec.visual.map.zfar = 30.0  # 3 km
        if world is not None:
            world.add_to_spec(spec)
        announce_scene_compile()
        return spec.compile()

    def vehicle_params(self) -> VehicleParams:
        """Static vehicle parameters (like a vehicle config file). Not state, so not truth."""
        m = self.model
        rotor_xy = np.array([m.site_pos[m.site(f"thrust{i + 1}").id][:2] for i in range(4)])
        return VehicleParams(
            mass=float(m.body_subtreemass[self.body_id]),
            g=GRAVITY,
            ctrl_max=float(m.actuator_ctrlrange[0, 1]),
            rotor_xy=rotor_xy,
            yaw_coeff=np.array([m.actuator_gear[i, 5] for i in range(4)]),
        )

    # ------------------------------------------------------------------ inputs

    def _on_motor(self, msg: MotorCmd) -> None:
        self.ctrl[:] = msg.thrust

    def set_conditions(self, conditions: Conditions | ConditionsMsg) -> None:
        """Adopt a copy of the degradation levers (also reachable via the `conditions` bus topic)."""
        if isinstance(conditions, ConditionsMsg):
            conditions = conditions.conditions
        conditions.validate()
        self.conditions = conditions.copy()
        self.weather = lever.weather_effect(self.conditions.weather)

    @property
    def camera_lost(self) -> bool:
        return self.conditions.camera.lost or self.time < self.camera_blackout_until

    def weather_visuals(self) -> lever.WeatherVisuals:
        return lever.weather_visuals(self.conditions.weather)

    def apply_disturbance(self, dist: Disturbance) -> None:
        t = self.data.time
        if dist.kind == "wind":
            vx, vy = (dist.value or (0.0, 0.0))[:2]
            speed, direction = lever.wind_from_vector(vx, vy)
            self.conditions.wind.speed = speed
            if speed > 0:
                self.conditions.wind.direction_deg = direction
        elif dist.kind == "camera_blackout":
            self.camera_blackout_until = t + dist.duration if dist.duration > 0 else math.inf
        elif dist.kind == "imu_bias_jump":
            mag = float(dist.value) if dist.value is not None else self.cfg.disturbance.imu_bias_jump
            ang = self.rng.uniform(0, 2 * math.pi)
            self.accel_bias += mag * np.array([math.cos(ang), math.sin(ang), 0.0])
        elif dist.kind == "clear":
            self.conditions.wind.speed = self.conditions.wind.gust = 0.0
            self.camera_blackout_until = -1.0
        # comms_loss is handled by the server (the drone keeps flying)

    # ------------------------------------------------------------------ stepping

    @property
    def time(self) -> float:
        return float(self.data.time)

    def step(self) -> None:
        m, d = self.model, self.data
        d.ctrl[:] = self.ctrl
        self._apply_wind()
        mujoco.mj_step(m, d)
        self._update_biases()
        self._publish_sensors()

    def _apply_wind(self) -> None:
        w = self.conditions.wind
        self.air_velocity = self.wind_model.air_velocity(w, self.cfg.sim.timestep)
        self.wind_force = self.wind_model.force(w, self.air_velocity, self.data.qvel[0:3], self.cfg.disturbance.wind_drag)
        self.data.xfrc_applied[self.body_id, 0:3] = self.wind_force

    def _update_biases(self) -> None:
        n, dt = self.cfg.noise, self.cfg.sim.timestep
        sq = math.sqrt(dt)
        self.gyro_bias += self.rng.normal(0, n.gyro_bias_rw * sq, 3)
        self.accel_bias += self.rng.normal(0, n.accel_bias_rw * sq, 3)
        self.baro_bias += self.rng.normal(0, n.baro_bias_rw * sq)

    def _publish_sensors(self) -> None:
        t = self.time
        d, n = self.data, self.cfg.noise
        if self.timers["imu"].due(t):
            gyro = d.sensor("body_gyro").data + self.gyro_bias + self.rng.normal(0, n.gyro_noise, 3)
            acc = d.sensor("body_linacc").data + self.accel_bias + self.rng.normal(0, n.accel_noise, 3)
            self.bus.publish(topics.IMU, ImuMsg(t, gyro.copy(), acc.copy()))
        if self.timers["baro"].due(t):
            alt = float(d.qpos[2]) + self.baro_bias + self.rng.normal(0, n.baro_noise)
            self.bus.publish(topics.BARO, BaroMsg(t, alt))
        c = self.conditions
        if self.timers["range"].due(t):
            r = float(d.sensor("rangefinder").data[0])  # -1 = no hit
            if 0.0 <= r <= n.range_max:
                r = max(0.0, r + self.range_rng.normal(0, n.range_noise + n.range_noise_rel * r))
            else:
                r = math.nan
            r = lever.degrade_range(r, self.weather, c.lidar.dropout, c.lidar.lost, self.range_lever_rng)
            self.bus.publish(topics.RANGE, RangeMsg(t, r))
        if self.timers["lidar"].due(t):
            self.bus.publish(topics.LIDAR, self._lidar_scan(t))
        if self.timers["mag"].due(t):
            yaw = yaw_from_R(d.xmat[self.body_id].reshape(3, 3)) + self.mag_bias
            yaw += math.radians(self.rng.normal(0, n.mag_noise_deg))
            self.bus.publish(topics.MAG, MagMsg(t, math.atan2(math.sin(yaw), math.cos(yaw))))
        if self.timers["gnss"].due(t):
            fix = self.gnss.sample(t, d.qpos[0:3], c.gnss)
            if fix is not None:
                self.bus.publish(topics.GNSS, GnssMsg(t, fix.pos, fix.cov))
        if self.timers["down"].due(t):
            if lever.camera_frame_kept(c, self.weather, self.weather_rng):
                self.bus.publish(topics.CAMERA_DOWN, self.render("down"))
            else:
                self.frames_dropped += 1
        if self.render_front and self.timers["front"].due(t):
            self.bus.publish(topics.CAMERA_FRONT, self.render("front"))

    # ------------------------------------------------------------------ cameras

    def render(self, name: str) -> CameraFrame:
        cam = self.cfg.camera
        w, h, fovy = (
            (cam.down_width, cam.down_height, cam.down_fovy_deg)
            if name == "down"
            else (cam.front_width, cam.front_height, cam.front_fovy_deg)
        )
        if self.renderer is None or (self.renderer.width, self.renderer.height) != (w, h):
            if self.renderer is not None:
                self.renderer.close()
            self.renderer = mujoco.Renderer(self.model, h, w)
        self.frame_seq += 1
        cid = self.cam_ids[name]
        self._cam_truth[self.frame_seq] = (self.data.cam_xpos[cid].copy(), self.data.cam_xmat[cid].reshape(3, 3).copy())
        while len(self._cam_truth) > 64:
            self._cam_truth.popitem(last=False)
        if name == "down" and self.camera_lost:
            img = np.zeros((h, w, 3), np.uint8)  # sensor failure: black frames
        else:
            self.renderer.update_scene(self.data, camera=name)
            img = self.renderer.render().copy()
            if self.weather.visibility_m < lever.CLEAR_VISIBILITY_M:
                img = lever.degrade_image(img, self.weather, self._view_distance(name), self.weather_rng)
        return CameraFrame(self.time, self.frame_seq, name, img, fovy)

    def _view_distance(self, name: str) -> float:
        """Air the camera looks through: the true distance to the surface below (down), ~60 m (front)."""
        if name != "down":
            return 60.0
        r = float(self.data.sensor("rangefinder").data[0])
        return r if r >= 0.0 else float(self.data.qpos[2])

    def _lidar_scan(self, t: float) -> LidarMsg:
        """Cast the nadir fan against the scene geometry (the X2 itself excluded), then apply the levers."""
        m, d, c = self.model, self.data, self.conditions
        origin = d.site_xpos[m.site("rangefinder").id].copy()
        R = d.xmat[self.body_id].reshape(3, 3)
        geomid = np.zeros(1, np.int32)
        rmax = self.cfg.noise.lidar_range_max
        ranges = np.full(len(self.lidar_dirs), math.nan)
        for i, vb in enumerate(self.lidar_dirs):
            r = mujoco.mj_ray(m, d, origin, R @ vb, None, 1, self.body_id, geomid)
            r = r if 0.0 <= r <= rmax else math.nan
            if math.isfinite(r):
                r = max(0.0, r + self.lidar_rng.normal(0, self.cfg.noise.range_noise))
            ranges[i] = lever.degrade_range(r, self.weather, c.lidar.dropout, c.lidar.lost, self.range_lever_rng)
        return LidarMsg(t, np.asarray(lever.LIDAR_BEAMS), ranges)

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    # ------------------------------------------------------------------ truth (package-private)
    # Accessed through sfhack.sim.truth only.

    def _truth_state(self) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        d = self.data
        return (
            self.time,
            d.qpos[0:3].copy(),
            d.qvel[0:3].copy(),
            d.xmat[self.body_id].reshape(3, 3).copy(),
            d.qvel[3:6].copy(),
        )

    def _truth_camera_pose(self, frame_seq: int) -> tuple[np.ndarray, np.ndarray] | None:
        return self._cam_truth.get(frame_seq)

    def _in_contact(self) -> bool:
        d = self.data
        for i in range(d.ncon):
            c = d.contact[i]
            if self.model.geom_bodyid[c.geom1] == self.body_id or self.model.geom_bodyid[c.geom2] == self.body_id:
                return True
        return False
