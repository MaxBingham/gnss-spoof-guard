"""Quadrotor mixer, ported from fly_x2.py: [total thrust, roll, pitch, yaw torque] <-> 4 motor thrusts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VehicleParams:
    """Static vehicle description (exported once by the simulator, like a vehicle config file)."""

    mass: float  # kg
    g: float  # m/s^2
    ctrl_max: float  # N, per-motor thrust limit
    rotor_xy: np.ndarray  # (4, 2) rotor positions in the body frame [m]
    yaw_coeff: np.ndarray  # (4,) reaction torque per N of thrust


class Mixer:
    def __init__(self, params: VehicleParams) -> None:
        self.ctrl_max = params.ctrl_max
        mix = np.zeros((4, 4))
        for i in range(4):
            x, y = params.rotor_xy[i]
            mix[:, i] = [1.0, y, -x, params.yaw_coeff[i]]
        self.mix = mix
        self.mix_inv = np.linalg.inv(mix)

    def motors_to_wrench(self, motors: np.ndarray) -> np.ndarray:
        """Per-motor thrusts -> [total thrust, roll torque, pitch torque, yaw torque]."""
        return self.mix @ motors

    def wrench_to_motors(self, thrust: float, torque: np.ndarray) -> np.ndarray:
        """Desired thrust + body torques -> per-motor thrusts, clipped to the motor limits."""
        motors = self.mix_inv @ np.array([thrust, *torque])
        return np.clip(motors, 0.0, self.ctrl_max)
