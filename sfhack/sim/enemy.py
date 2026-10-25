"""The ENEMY drone and the bird distractors (simulated reality, simulator side).

PLACEHOLDER until Max delivers his enemy model: replace `EnemyDrone` (motion) and / or `add_enemy` (looks); the rest
of the code only uses `EnemyDrone.step(t, bearing_deg) -> EnemyPose` and the MuJoCo body named "enemy".

Scenario: the UMA-16 at the base hears the enemy and gives a bearing. At the first bearing the enemy appears
`range0` m out along it, `agl` m above the terrain, and approaches the base at `speed` with a little weaving. Its
direction follows the LATEST bearing: it steers toward the point on the new bearing ray at its current range, with a
limited lateral speed (a bearing jump of 30 deg at 400 m is a 200 m move, not a teleport; a bearing turning faster
than `lateral_max` / range leaves the enemy lagging behind it). It stops approaching at
`min_range` and loiters there.

Looks: a heavy quad, 1.2 m motor to motor (1.6 m over the props), matte charcoal body with olive arms and a payload
box: clearly bigger and darker than our 0.6 m textured Skydio X2. Birds: crow / pigeon / buzzard / stork-like bodies
with flapping wings (separate mocap bodies hinged at the shoulders).

Onboard code never imports this module (tests/test_truth_separation.py): the drone knows only the nominal enemy size
(`TrackerConfig.target_size_m`, checked against `ENEMY_SIZE` here by tests/test_enemy_tracker.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import mujoco
import numpy as np

HIDDEN = np.array([0.0, 0.0, -500.0])  # parked under the terrain until it exists

# demo enemy geometry (m)
ENEMY_MOTOR_DIAG = 1.2  # motor to motor across the diagonal
ENEMY_PROP_R = 0.2
ENEMY_SIZE = ENEMY_MOTOR_DIAG + 2 * ENEMY_PROP_R  # 1.6 m tip to tip: the size the tracker assumes
CHARCOAL = (0.10, 0.10, 0.11, 1.0)
OLIVE = (0.27, 0.30, 0.17, 1.0)
PROP_GRAY = (0.32, 0.32, 0.34, 1.0)


@dataclass
class DroneStyle:
    """One multirotor look (also the dataset's drone variants)."""

    motor_diag: float = ENEMY_MOTOR_DIAG
    prop_r: float = ENEMY_PROP_R
    arms: int = 4
    body: tuple = CHARCOAL
    arm: tuple = OLIVE
    prop: tuple = PROP_GRAY
    payload: bool = True
    body_half: tuple = (0.16, 0.11, 0.06)


@dataclass
class BirdStyle:
    span: float = 0.9  # wingspan, m
    body: tuple = (0.06, 0.06, 0.07, 1.0)  # crow black
    wing: tuple = (0.05, 0.05, 0.06, 1.0)
    flap_hz: float = 3.5
    flap_deg: float = 35.0
    glide: float = 0.0  # fraction of the time gliding (wings held out)


BIRDS = {
    "crow": BirdStyle(0.95, (0.06, 0.06, 0.07, 1), (0.05, 0.05, 0.06, 1), 3.8, 35.0, 0.2),
    "pigeon": BirdStyle(0.65, (0.45, 0.46, 0.50, 1), (0.40, 0.41, 0.45, 1), 6.0, 40.0, 0.1),
    "buzzard": BirdStyle(1.25, (0.36, 0.26, 0.17, 1), (0.30, 0.22, 0.14, 1), 2.2, 22.0, 0.75),
    "stork": BirdStyle(1.9, (0.93, 0.93, 0.92, 1), (0.10, 0.10, 0.10, 1), 1.8, 20.0, 0.6),
}


def _q_yaw(a: float) -> list[float]:
    return [math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]


def _deco(body, **kw):
    return body.add_geom(contype=0, conaffinity=0, group=1, **kw)


def add_drone(spec: mujoco.MjSpec, name: str, style: DroneStyle = DroneStyle()) -> mujoco.MjsBody:
    """A multirotor as a mocap body (visual only, no collisions), body frame FLU, origin at the centre."""
    b = spec.worldbody.add_body(name=name, mocap=True, pos=HIDDEN.tolist())
    G = mujoco.mjtGeom
    bx, by, bz = style.body_half
    _deco(b, name=f"{name}_body", type=G.mjGEOM_BOX, size=[bx, by, bz], rgba=list(style.body))
    _deco(b, name=f"{name}_dome", type=G.mjGEOM_ELLIPSOID, size=[bx * 0.8, by * 0.85, bz * 0.9], pos=[0, 0, bz * 0.8],
          rgba=list(style.body))
    R = style.motor_diag / 2
    for k in range(style.arms):
        a = math.pi / style.arms + 2 * math.pi * k / style.arms
        tip = np.array([R * math.cos(a), R * math.sin(a), 0.0])
        mid = tip / 2
        _deco(b, name=f"{name}_arm{k}", type=G.mjGEOM_BOX, size=[R / 2, 0.022, 0.018], pos=mid.tolist(),
              quat=_q_yaw(a), rgba=list(style.arm))
        _deco(b, name=f"{name}_motor{k}", type=G.mjGEOM_CYLINDER, size=[0.04, 0.035], pos=(tip + [0, 0, 0.03]).tolist(),
              rgba=list(style.body))
        _deco(b, name=f"{name}_prop{k}", type=G.mjGEOM_CYLINDER, size=[style.prop_r, 0.004],
              pos=(tip + [0, 0, 0.07]).tolist(), rgba=list(style.prop))
    if style.payload:
        _deco(b, name=f"{name}_payload", type=G.mjGEOM_BOX, size=[0.10, 0.06, 0.05], pos=[0, 0, -bz - 0.06],
              rgba=list(style.arm))
    for s in (-1, 1):  # landing skids
        _deco(b, name=f"{name}_skid{s}", type=G.mjGEOM_CAPSULE, size=[0.012, bx * 1.3], pos=[0, s * by * 1.4, -bz - 0.13],
              quat=[math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0], rgba=list(style.body))
    return b


def add_enemy(spec: mujoco.MjSpec, style: DroneStyle = DroneStyle()) -> mujoco.MjsBody:
    return add_drone(spec, "enemy", style)


def add_bird(spec: mujoco.MjSpec, name: str, style: BirdStyle) -> tuple[mujoco.MjsBody, ...]:
    """Bird = 3 mocap bodies: torso (head, tail) and two wings hinged at the shoulders (flapping = wing rotation)."""
    s = style.span
    G = mujoco.mjtGeom
    torso = spec.worldbody.add_body(name=name, mocap=True, pos=HIDDEN.tolist())
    L = 0.42 * s  # body length scales with the span
    _deco(torso, name=f"{name}_torso", type=G.mjGEOM_ELLIPSOID, size=[L / 2, 0.09 * L, 0.10 * L], rgba=list(style.body))
    _deco(torso, name=f"{name}_head", type=G.mjGEOM_SPHERE, size=[0.11 * L], pos=[0.52 * L, 0, 0.05 * L],
          rgba=list(style.body))
    _deco(torso, name=f"{name}_tail", type=G.mjGEOM_BOX, size=[0.16 * L, 0.10 * L, 0.01], pos=[-0.6 * L, 0, 0],
          rgba=list(style.wing))
    wings = []
    for side, tag in ((1, "l"), (-1, "r")):
        w = spec.worldbody.add_body(name=f"{name}_wing{tag}", mocap=True, pos=HIDDEN.tolist())
        half = 0.25 * s
        _deco(w, name=f"{name}_wing{tag}_g", type=G.mjGEOM_ELLIPSOID, size=[0.11 * s, half, 0.008],
              pos=[0, side * half, 0], rgba=list(style.wing))
        wings.append(w)
    return (torso, *wings)


# ---------------------------------------------------------------------------------------- poses

@dataclass
class EnemyPose:
    pos: np.ndarray  # (3,) world ENU, m
    vel: np.ndarray  # (3,) m/s
    yaw: float  # rad, ENU (0 = east, CCW)
    active: bool  # False before the first bearing (parked under the terrain)


def az_dir(az_deg: float) -> np.ndarray:
    a = math.radians(az_deg)
    return np.array([math.sin(a), math.cos(a)])


class EnemyDrone:
    """Motion model of the enemy (placeholder for Max's). `step(t, bearing_deg)` once per sim step (any rate)."""

    def __init__(self, ground: Callable = lambda x, y: 0.0, base_xy=(0.0, 0.0), range0: float = 500.0,
                 agl: float = 40.0, speed: float = 4.0, min_range: float = 80.0, lateral_max: float = 9.0,
                 weave_m: float = 6.0, weave_s: float = 17.0, seed: int = 0) -> None:
        rng = np.random.default_rng(seed + 7000)
        self.ground = ground
        self.base = np.asarray(base_xy, float)[:2].copy()
        self.range0, self.agl, self.min_range, self.lateral_max = range0, agl, min_range, lateral_max
        self.speed = float(speed if speed is not None else rng.uniform(3.0, 5.0))
        self.weave_m, self.weave_s = weave_m, weave_s * rng.uniform(0.8, 1.25)
        self.phase = rng.uniform(0, 2 * math.pi)
        self.bob = rng.uniform(0, 2 * math.pi)
        self.t_spawn: float | None = None
        self.r = range0  # range from the base along the ray (no weave)
        self.xy = np.zeros(2)  # position of the un-weaved track point
        self.pose = EnemyPose(HIDDEN.copy(), np.zeros(3), 0.0, False)
        self._t: float | None = None

    @property
    def active(self) -> bool:
        return self.t_spawn is not None

    def step(self, t: float, bearing_deg: float | None) -> EnemyPose:
        if self.t_spawn is None:
            if bearing_deg is None:
                self._t = t
                return self.pose
            self.t_spawn = t
            self.xy = self.base + az_dir(bearing_deg) * self.range0
        dt = 0.0 if self._t is None else max(t - self._t, 0.0)
        self._t = t
        prev = self.pose.pos.copy()
        if bearing_deg is not None and dt > 0:
            rel = self.xy - self.base
            r = float(np.linalg.norm(rel))
            r_new = max(self.min_range, r - self.speed * dt)
            # polar step: range closes at `speed`, the azimuth turns toward the bearing at <= lateral_max / r.
            # (A Cartesian step toward the goal point let the chord's radial part through unlimited: a bearing
            # turning steadily pulled the enemy from 500 m to 80 m in ~10 s.)
            az = math.degrees(math.atan2(rel[0], rel[1]))
            da = (bearing_deg - az + 180.0) % 360.0 - 180.0
            da_max = math.degrees(self.lateral_max * dt / max(r, 1e-6))
            az += max(-da_max, min(da_max, da))
            self.xy = self.base + az_dir(az) * r_new
        rel = self.xy - self.base
        r = float(np.linalg.norm(rel))
        u = rel / max(r, 1e-6)
        nrm = np.array([-u[1], u[0]])
        ts = t - self.t_spawn
        w = self.weave_m * math.sin(2 * math.pi * ts / self.weave_s + self.phase)
        xy = self.xy + nrm * w
        z = float(self.ground(*xy)) + self.agl + 1.5 * math.sin(2 * math.pi * ts / 23.0 + self.bob)
        pos = np.array([xy[0], xy[1], z])
        vel = (pos - prev) / dt if (dt > 0 and self.pose.active) else np.zeros(3)
        yaw = math.atan2(vel[1], vel[0]) if np.linalg.norm(vel[:2]) > 0.5 else self.pose.yaw
        self.pose = EnemyPose(pos, vel, yaw, True)
        return self.pose


def enemy_attitude(pose: EnemyPose) -> np.ndarray:
    """Quaternion (w, x, y, z): heading along the velocity, pitched forward a little with the speed."""
    sp = float(np.linalg.norm(pose.vel[:2]))
    pitch = math.radians(min(12.0, 2.0 * sp))
    qy = np.array([math.cos(pose.yaw / 2), 0, 0, math.sin(pose.yaw / 2)])
    qp = np.array([math.cos(pitch / 2), 0, math.sin(pitch / 2), 0])
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, qy, qp)
    return q


@dataclass
class Bird:
    kind: str
    center: np.ndarray  # (2,) circling / patrol centre
    radius: float
    agl: float
    speed: float
    phase: float
    cw: int
    flap_phase: float
    mocap: tuple[int, int, int] = (0, 0, 0)  # torso, left wing, right wing mocap ids
    pos: np.ndarray = field(default_factory=lambda: HIDDEN.copy())


class BirdFlock:
    """Birds circling or patrolling over the farmland (distractors for the detector). Simulated reality."""

    def __init__(self, model: mujoco.MjModel, ground: Callable, names: list[tuple[str, str]], seed: int = 0,
                 area_r: float = 700.0, base_xy=(0.0, 0.0)) -> None:
        rng = np.random.default_rng(seed + 9100)
        self.ground, self.birds = ground, []
        for name, kind in names:
            st = BIRDS[kind]
            a, r = rng.uniform(0, 2 * math.pi), area_r * math.sqrt(rng.uniform(0.05, 1.0))
            ids = tuple(int(model.body_mocapid[model.body(n).id]) for n in (name, f"{name}_wingl", f"{name}_wingr"))
            self.birds.append(Bird(kind, np.asarray(base_xy, float) + r * np.array([math.cos(a), math.sin(a)]),
                                   rng.uniform(15, 70), rng.uniform(8, 70), rng.uniform(8, 13), rng.uniform(0, 6.3),
                                   int(rng.choice([-1, 1])), rng.uniform(0, 6.3), ids))
        self.rng = rng

    def place_near(self, points: np.ndarray, frac: float = 0.5) -> None:
        """Move a fraction of the birds to circle near these points (e.g. along the enemy's approach)."""
        n = int(len(self.birds) * frac)
        for k, b in enumerate(self.birds[:n]):
            p = points[k % len(points)]
            b.center = np.asarray(p, float)[:2] + self.rng.normal(0, 40, 2)
            b.agl = float(self.rng.uniform(20, 60))

    def step(self, t: float, data: mujoco.MjData) -> None:
        for b in self.birds:
            st = BIRDS[b.kind]
            w = b.cw * b.speed / b.radius
            a = b.phase + w * t
            xy = b.center + b.radius * np.array([math.cos(a), math.sin(a)])
            z = float(self.ground(*xy)) + b.agl + 2.0 * math.sin(0.3 * t + b.phase)
            yaw = a + b.cw * math.pi / 2
            b.pos = np.array([xy[0], xy[1], z])
            bank = -b.cw * math.radians(15.0)
            cyc = (t * 0.15 + b.flap_phase) % 1.0
            gliding = cyc < st.glide
            flap = math.radians(8.0) if gliding else math.radians(st.flap_deg) * math.sin(
                2 * math.pi * st.flap_hz * t + b.flap_phase)
            set_bird(data, b.mocap, b.pos, yaw, bank, flap)


def _quat_mul(*qs) -> np.ndarray:
    q = np.array([1.0, 0, 0, 0])
    for p in qs:
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, q, np.asarray(p, float))
        q = out
    return q


def set_bird(data: mujoco.MjData, mocap: tuple[int, int, int], pos, yaw: float, bank: float, flap: float) -> None:
    """Torso at pos heading yaw, banked; wings rotated about the body x axis by +-flap at the shoulders."""
    qb = _quat_mul([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)], [math.cos(bank / 2), math.sin(bank / 2), 0, 0])
    t, l, r = mocap
    data.mocap_pos[t] = pos
    data.mocap_quat[t] = qb
    for mid, side in ((l, 1), (r, -1)):
        data.mocap_pos[mid] = pos
        a = side * flap
        data.mocap_quat[mid] = _quat_mul(qb, [math.cos(a / 2), math.sin(a / 2), 0, 0])


def bird_names(n: int, seed: int = 0) -> list[tuple[str, str]]:
    rng = np.random.default_rng(seed + 9000)
    kinds = list(BIRDS)
    p = np.array([0.4, 0.25, 0.25, 0.1])
    return [(f"bird{k}", kinds[int(rng.choice(len(kinds), p=p))]) for k in range(n)]


def add_scenario_bodies(spec: mujoco.MjSpec, n_birds: int = 16, seed: int = 0) -> list[tuple[str, str]]:
    """The enemy and `n_birds` birds into a world spec before it is compiled. Returns the bird (name, kind) list."""
    add_enemy(spec)
    names = bird_names(n_birds, seed)
    for name, kind in names:
        add_bird(spec, name, BIRDS[kind])
    return names


def set_enemy(model: mujoco.MjModel, data: mujoco.MjData, pose: EnemyPose) -> None:
    mid = int(model.body_mocapid[model.body("enemy").id])
    data.mocap_pos[mid] = pose.pos
    data.mocap_quat[mid] = enemy_attitude(pose)


def geom_classes(model: mujoco.MjModel, prefixes: dict[str, int]) -> np.ndarray:
    """(ngeom,) class index per geom from the BODY name prefix (e.g. {"enemy": 0, "bird": 1}), -1 = background."""
    out = np.full(model.ngeom, -1, np.int32)
    for g in range(model.ngeom):
        bname = model.body(int(model.geom_bodyid[g])).name
        for p, c in prefixes.items():
            if bname.startswith(p):
                out[g] = c
    return out
