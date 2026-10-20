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
from ..messages import BaroMsg, CameraFrame, Disturbance, GnssMsg, ImuMsg, MagMsg, MotorCmd

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
    ) -> None:
        self.cfg = cfg
        self.bus = bus
        self.rng = np.random.default_rng(seed)
        self.render_front = render_front
        self.model = self._build_model(world)
        self.data = mujoco.MjData(self.model)
        m, d = self.model, self.data
        self.body_id = m.body("x2").id
        self.cam_ids = {"down": m.camera("down").id, "front": m.camera("front").id}

        # Spawn on the ground at A, motors off.
        d.qpos[0:3] = [spawn_xy[0], spawn_xy[1], 0.005]
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
            "mag": _RateTimer(s.mag_hz),
            "down": _RateTimer(s.down_cam_hz),
            "front": _RateTimer(s.front_cam_hz),
            "gnss": _RateTimer(s.gnss_hz),
        }

        # Disturbance state
        self.wind = np.zeros(3)
        self.gust = np.zeros(3)
        self.camera_blackout_until = -1.0  # sim time; math.inf = until cleared

        bus.subscribe(topics.MOTOR, self._on_motor)
        bus.subscribe(topics.DISTURBANCE, self.apply_disturbance)

    # ------------------------------------------------------------------ model

    def _build_model(self, world: WorldModel | None) -> mujoco.MjModel:
        spec = mujoco.MjSpec.from_file(str(X2_DIR / "scene.xml"))
        spec.option.timestep = self.cfg.sim.timestep
        cam = self.cfg.camera
        drone = spec.body("x2")
        # MuJoCo cameras look along their -z axis. Down camera: image top = forward, image right = right.
        drone.add_camera(name="down", pos=[0, 0, -0.02], fovy=cam.down_fovy_deg,
                         quat=_quat_from_axes([0, -1, 0], [1, 0, 0]))
        tilt = math.radians(15.0)  # front camera pitched 15 deg down
        drone.add_camera(name="front", pos=[0.12, 0, 0.03], fovy=cam.front_fovy_deg,
                         quat=_quat_from_axes([0, -1, 0], [math.sin(tilt), 0, math.cos(tilt)]))
        spec.visual.global_.offwidth = max(cam.down_width, cam.front_width, 640)
        spec.visual.global_.offheight = max(cam.down_height, cam.front_height, 480)
        # Large outdoor scene: near/far planes are fractions of stat.extent.
        spec.stat.extent = 100.0
        spec.visual.map.znear = 0.001  # 0.1 m
        spec.visual.map.zfar = 30.0  # 3 km
        if world is not None:
            world.add_to_spec(spec)
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

    def apply_disturbance(self, dist: Disturbance) -> None:
        t = self.data.time
        if dist.kind == "wind":
            vx, vy = (dist.value or (0.0, 0.0))[:2]
            self.wind = np.array([vx, vy, 0.0])
        elif dist.kind == "camera_blackout":
            self.camera_blackout_until = t + dist.duration if dist.duration > 0 else math.inf
        elif dist.kind == "imu_bias_jump":
            mag = float(dist.value) if dist.value is not None else self.cfg.disturbance.imu_bias_jump
            ang = self.rng.uniform(0, 2 * math.pi)
            self.accel_bias += mag * np.array([math.cos(ang), math.sin(ang), 0.0])
        elif dist.kind == "clear":
            self.wind[:] = 0.0
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
        dc = self.cfg.disturbance
        dt = self.cfg.sim.timestep
        if np.any(self.wind):
            # Ornstein-Uhlenbeck gust force with ~2 s correlation time
            self.gust += -self.gust * dt / 2.0 + self.rng.normal(0, dc.wind_gust_std * math.sqrt(dt), 3)
        else:
            self.gust[:] = 0.0
        rel = self.wind - self.data.qvel[0:3]
        force = dc.wind_drag * rel * (1.0 if np.any(self.wind) else 0.0) + self.gust
        self.data.xfrc_applied[self.body_id, 0:3] = force

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
        if self.timers["mag"].due(t):
            yaw = yaw_from_R(d.xmat[self.body_id].reshape(3, 3)) + self.mag_bias
            yaw += math.radians(self.rng.normal(0, n.mag_noise_deg))
            self.bus.publish(topics.MAG, MagMsg(t, math.atan2(math.sin(yaw), math.cos(yaw))))
        if self.cfg.sim.gnss_enabled and self.timers["gnss"].due(t):
            pos = d.qpos[0:3] + self.rng.normal(0, n.gnss_noise, 3)
            self.bus.publish(topics.GNSS, GnssMsg(t, pos))
        if self.timers["down"].due(t):
            self.bus.publish(topics.CAMERA_DOWN, self.render("down"))
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
        if self.time < self.camera_blackout_until:
            img = np.zeros((h, w, 3), np.uint8)  # sensor failure: black frames
        else:
            self.renderer.update_scene(self.data, camera=name)
            img = self.renderer.render().copy()
        return CameraFrame(self.time, self.frame_seq, name, img, fovy)

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
