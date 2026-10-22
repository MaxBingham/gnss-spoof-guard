"""Visual-only geoms appended to an MjvScene (never part of the physics model)."""

from __future__ import annotations

import math

import mujoco
import numpy as np

EYE = np.eye(3).flatten()

# X2 rotor hubs in the body frame (FLU), from third_party/skydio_x2 (front rotors sit a bit higher).
ROTORS = np.array([[0.14, 0.18, 0.08], [0.14, -0.18, 0.08], [-0.14, 0.18, 0.05], [-0.14, -0.18, 0.05]])


def rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class Geoms:
    def __init__(self, scn: mujoco.MjvScene) -> None:
        self.scn = scn

    def _next(self) -> mujoco.MjvGeom | None:
        if self.scn.ngeom >= self.scn.maxgeom:
            return None
        g = self.scn.geoms[self.scn.ngeom]
        self.scn.ngeom += 1
        return g

    def _init(self, kind, size, pos, mat, rgb, alpha) -> mujoco.MjvGeom | None:
        g = self._next()
        if g is not None:
            mujoco.mjv_initGeom(g, kind, np.asarray(size, float), np.asarray(pos, float),
                                np.asarray(mat, float).flatten(), np.array([*rgb, alpha], np.float32))
            g.emission = 0.0
        return g

    def sphere(self, pos, r: float, rgb, alpha: float = 1.0, emission: float = 0.0) -> None:
        g = self._init(mujoco.mjtGeom.mjGEOM_SPHERE, [r, 0, 0], pos, EYE, rgb, alpha)
        if g is not None:
            g.emission = emission

    def ellipsoid(self, pos, radii, mat, rgb, alpha: float = 1.0) -> None:
        self._init(mujoco.mjtGeom.mjGEOM_ELLIPSOID, radii, pos, mat, rgb, alpha)

    def cylinder(self, pos, r: float, half_h: float, rgb, alpha: float = 1.0, mat=EYE, emission: float = 0.0) -> None:
        g = self._init(mujoco.mjtGeom.mjGEOM_CYLINDER, [r, half_h, 0], pos, mat, rgb, alpha)
        if g is not None:
            g.emission = emission

    def box(self, pos, half, mat, rgb, alpha: float = 1.0) -> None:
        self._init(mujoco.mjtGeom.mjGEOM_BOX, half, pos, mat, rgb, alpha)

    def _connector(self, kind, a, b, width: float, rgb, alpha: float, emission: float = 0.0) -> None:
        a, b = np.asarray(a, float), np.asarray(b, float)
        if np.linalg.norm(b - a) < 1e-6:
            return
        g = self._init(kind, [0, 0, 0], np.zeros(3), EYE, rgb, alpha)
        if g is not None:
            mujoco.mjv_connector(g, kind, width, a, b)
            g.emission = emission

    def capsule(self, a, b, r: float, rgb, alpha: float = 1.0, emission: float = 0.0) -> None:
        self._connector(mujoco.mjtGeom.mjGEOM_CAPSULE, a, b, r, rgb, alpha, emission)

    def arrow(self, a, b, r: float, rgb, alpha: float = 1.0, emission: float = 0.0) -> None:
        self._connector(mujoco.mjtGeom.mjGEOM_ARROW, a, b, r, rgb, alpha, emission)

    def line(self, a, b, rgb, width: float = 2.0, alpha: float = 1.0) -> None:
        self._connector(mujoco.mjtGeom.mjGEOM_LINE, a, b, width, rgb, alpha)

    def tube(self, pts: np.ndarray, r: float, rgb, alpha: float = 1.0, emission: float = 0.0) -> None:
        for a, b in zip(pts[:-1], pts[1:]):
            self.capsule(a, b, r, rgb, alpha, emission)

    def ring(self, center, radius: float, r: float, rgb, alpha: float = 1.0, n: int = 40) -> None:
        ang = np.linspace(0, 2 * np.pi, n + 1)
        pts = np.column_stack([center[0] + radius * np.cos(ang), center[1] + radius * np.sin(ang),
                               np.full(n + 1, center[2])])
        self.tube(pts, r, rgb, alpha)

    def dashed(self, a, b, r: float, rgb, alpha: float = 1.0, dash: float = 2.0) -> None:
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, int(np.linalg.norm(b - a) / dash))
        for i in range(0, n, 2):
            self.capsule(a + (b - a) * i / n, a + (b - a) * min(i + 1, n) / n, r, rgb, alpha)

    def drone(self, pos, R, scale: float, body_rgb, rotor_rgb, alpha: float = 1.0, nose_rgb=None) -> None:
        """Stylized X2 (body, four arms, rotor discs, nose marker) drawn `scale` times life size."""
        pos, R = np.asarray(pos, float), np.asarray(R, float)

        def w(p) -> np.ndarray:
            return pos + R @ (np.asarray(p, float) * scale)

        self.ellipsoid(w([0.0, 0.0, 0.05]), np.array([0.13, 0.07, 0.035]) * scale, R, body_rgb, alpha)
        for hub in ROTORS:
            self.capsule(w([0.0, 0.0, 0.05]), w(hub), 0.012 * scale, body_rgb, alpha)
            self.cylinder(w(hub + [0, 0, 0.012]), 0.085 * scale, 0.004 * scale, rotor_rgb, 0.75 * alpha, mat=R)
        self.sphere(w([0.15, 0.0, 0.05]), 0.025 * scale, nose_rgb or rotor_rgb, alpha, emission=0.4)
