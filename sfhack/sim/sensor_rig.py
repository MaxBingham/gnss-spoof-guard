"""IMU, barometer, compass and GNSS for the Skydio X2 in a MuJoCo model compiled elsewhere (simulator side).

demos/50_base_3d.py (and the farm world) build and compile their own scene, without MujocoSim's cameras and
rangefinder, so MujocoSim cannot wrap them. This rig puts the same sensor models (sfhack.config noise, rates,
turn-on biases and random walks, the GNSS receiver of sfhack.degradation) on the bus for such a model, and applies
the latest MotorCmd. Like MujocoSim it is the only owner of the true state: onboard code sees only the messages.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .. import bus as topics
from ..bus import Bus
from ..config import Config
from ..control.mixer import VehicleParams
from ..degradation import models as lever
from ..degradation.conditions import GnssConditions
from ..messages import BaroMsg, GnssMsg, ImuMsg, MagMsg, MotorCmd
from .mujoco_sim import GRAVITY, _RateTimer, yaw_from_R


class SensorRig:
    def __init__(self, cfg: Config, bus: Bus, model: mujoco.MjModel, data: mujoco.MjData, seed: int = 0) -> None:
        self.cfg, self.bus = cfg, bus
        self.model, self.data = model, data
        self.body_id = model.body("x2").id
        self.rng = np.random.default_rng(seed)
        self.gnss = lever.GnssModel(np.random.default_rng(seed + 3000), cfg.noise.gnss_noise)
        self.gnss_cond = GnssConditions(mode="ok")  # the harness switches it off (operator key N / --gnss-off-at)
        self.ctrl = np.zeros(model.nu)
        n = cfg.noise
        self.gyro_bias = self.rng.normal(0.0, n.gyro_bias_init, 3)
        self.accel_bias = self.rng.normal(0.0, n.accel_bias_init, 3)
        self.baro_bias = 0.0
        self.mag_bias = math.radians(self.rng.normal(0.0, n.mag_bias_deg))
        s = cfg.sim
        t0 = float(data.time)
        self.timers = {k: _RateTimer(hz, t0) for k, hz in
                       (("imu", s.imu_hz), ("baro", s.baro_hz), ("mag", s.mag_hz), ("gnss", s.gnss_hz))}
        bus.subscribe(topics.MOTOR, self._on_motor)

    @property
    def gnss_on(self) -> bool:
        return self.gnss_cond.mode != "off"

    def set_gnss(self, on: bool) -> None:
        self.gnss_cond.mode = "ok" if on else "off"

    def vehicle_params(self) -> VehicleParams:
        """Static vehicle parameters (a vehicle config file, not state)."""
        m = self.model
        return VehicleParams(
            mass=float(m.body_subtreemass[self.body_id]), g=GRAVITY, ctrl_max=float(m.actuator_ctrlrange[0, 1]),
            rotor_xy=np.array([m.site_pos[m.site(f"thrust{i + 1}").id][:2] for i in range(4)]),
            yaw_coeff=np.array([m.actuator_gear[i, 5] for i in range(4)]))

    def _on_motor(self, msg: MotorCmd) -> None:
        self.ctrl[:] = msg.thrust

    def step(self) -> None:
        d = self.data
        d.ctrl[:] = self.ctrl
        mujoco.mj_step(self.model, d)
        n, dt = self.cfg.noise, self.model.opt.timestep
        sq = math.sqrt(dt)
        self.gyro_bias += self.rng.normal(0, n.gyro_bias_rw * sq, 3)
        self.accel_bias += self.rng.normal(0, n.accel_bias_rw * sq, 3)
        self.baro_bias += self.rng.normal(0, n.baro_bias_rw * sq)
        self._publish(float(d.time))

    def _publish(self, t: float) -> None:
        d, n = self.data, self.cfg.noise
        if self.timers["imu"].due(t):
            gyro = d.sensor("body_gyro").data + self.gyro_bias + self.rng.normal(0, n.gyro_noise, 3)
            acc = d.sensor("body_linacc").data + self.accel_bias + self.rng.normal(0, n.accel_noise, 3)
            self.bus.publish(topics.IMU, ImuMsg(t, gyro.copy(), acc.copy()))
        if self.timers["baro"].due(t):
            self.bus.publish(topics.BARO, BaroMsg(t, float(d.qpos[2]) + self.baro_bias + self.rng.normal(0, n.baro_noise)))
        if self.timers["mag"].due(t):
            yaw = yaw_from_R(d.xmat[self.body_id].reshape(3, 3)) + self.mag_bias
            yaw += math.radians(self.rng.normal(0, n.mag_noise_deg))
            self.bus.publish(topics.MAG, MagMsg(t, math.atan2(math.sin(yaw), math.cos(yaw))))
        if self.timers["gnss"].due(t):
            fix = self.gnss.sample(t, d.qpos[0:3], self.gnss_cond)
            if fix is not None:
                self.bus.publish(topics.GNSS, GnssMsg(t, fix.pos, fix.cov))
