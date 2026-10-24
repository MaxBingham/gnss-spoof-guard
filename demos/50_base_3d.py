"""3D view of the base in MuJoCo: the UMA-16 array in the middle, the drone takes off from the pad and flies
out along the bearing the array reports, looks around, comes back and lands.

    uv run python demos/50_base_3d.py                         # viewer; give a bearing with the keys or over UDP
    uv run python demos/50_base_3d.py --bearing 60            # take off right away toward 060 deg
    uv run python demos/50_base_3d.py --headless --bearing 60 # no window: fly once, print the log
    uv run python demos/50_base_3d.py --world simple          # the old flat world (120 m compound, 400 m range)
    uv run python demos/50_base_3d.py --shots docs/base3d     # offscreen renders + frame-rate measurement

Worlds (`--world`):
- farm (default): a FICTIONAL, procedurally generated rural-Ukraine area of 4 x 4 km (demos/base3d_farm.py): a storage
  hall with a concrete forecourt and the mast in its middle, HESCO wall, rolling hills, fields with windbreaks,
  3 farmsteads, a village along a paved road. The drone holds 30 m above the terrain; default range 1200 m.
- simple: flat ground, small compound, default range 400 m.

Bearing input (degrees clockwise from north, compass convention, same as BearingEstimate.az_deg):
- keys in the viewer: LEFT / RIGHT turn the pending bearing by 5 deg, ENTER launches toward it,
  F launches toward a random bearing, C switches overview / chase camera, 1 / 2 / 4 sim speed, BACKSPACE resets;
- UDP port 5005 (`--udp-port`): a JSON BearingEstimate ({"az_deg": 60, ...}) or a plain number "60". A new
  bearing while the drone is out retargets it.

Self-contained on purpose (own scene, simple cascaded controller on the true state, GNSS on): this is the MVP view
of "clap -> the drone flies that way". Ground truth state is used for control: no estimation here yet.
World frame: ENU, x east, y north, z up; the UMA-16 mast is at the origin, the pad at z = 0.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
X2_SCENE = ROOT / "third_party" / "skydio_x2" / "scene.xml"
OBJ = ROOT / "assets" / "objects" / "objects"

PAD = np.array([18.0, -10.0])   # landing pad, m (E, N) from the UMA-16 mast
WALL_HALF = 60                  # simple world: HESCO perimeter 120 m x 120 m square, gate in the south wall
CRUISE_AGL = 30.0
OUT_SPEED = 8.0
CLIMB = 2.5
SINK = 1.5                      # max sink rate while following the terrain, m/s
DESCENT = 1.0
ORBIT_R = 35.0
ORBIT_S = 25.0
SUN_DIR = [0.3, 0.55, -1.0]     # afternoon sun in the SSW (the farm texture paints its shadows the same way)

OLIVE = [0.33, 0.36, 0.22, 1]
SAND = [0.72, 0.64, 0.47, 1]
TAN = [0.55, 0.47, 0.33, 1]


# ----------------------------------------------------------------------------------------------- scene
def _mesh(spec, name: str, rel: str) -> str:
    spec.add_mesh(name=name, file=str(OBJ / rel))
    return name


def _deco(body, **kw):
    """Visual-only geom (no collisions): decoration never disturbs the flight."""
    return body.add_geom(contype=0, conaffinity=0, **kw)


def _box(body, pos, half, rgba, yaw=0.0):
    q = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    return _deco(body, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=half, rgba=rgba, quat=q)


def flat_ground(x, y):
    """Terrain height of the simple world (scalar or array, like FarmWorld.ground)."""
    return np.zeros(np.shape(x)) if np.ndim(x) else 0.0


def build_simple(spec, w) -> SimpleNamespace:
    """The original flat world: grass, a few field slabs, a 120 m compound, trees."""
    tex = spec.texture("groundplane")
    tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    tex.rgb1 = [0.30, 0.42, 0.20]
    tex.rgb2 = [0.27, 0.39, 0.18]
    tex.mark = mujoco.mjtMark.mjMARK_NONE
    spec.material("groundplane").texrepeat = [400, 400]
    spec.material("groundplane").reflectance = 0.0
    spec.geom("floor").size = [3000, 3000, 0.1]
    rng = random.Random(7)

    # fields and a dirt road to the gate (thin slabs on the ground)
    for i in range(40):
        x, y = rng.uniform(-900, 900), rng.uniform(-900, 900)
        if abs(x) < 120 and abs(y) < 120:
            continue
        c = rng.choice([[0.42, 0.45, 0.22, 1], [0.36, 0.30, 0.20, 1], [0.32, 0.46, 0.22, 1], [0.50, 0.47, 0.30, 1]])
        _box(w, [x, y, 0.01 + 0.001 * i], [rng.uniform(60, 160), rng.uniform(40, 120), 0.01], c, rng.uniform(0, 3.14))
    _box(w, [0, -560, 0.03], [3.0, 500, 0.02], [0.45, 0.38, 0.28, 1])
    _box(w, [0, 0, 0.025], [WALL_HALF - 2, WALL_HALF - 2, 0.02], [0.47, 0.42, 0.31, 1])  # packed earth inside

    # HESCO perimeter (sand-filled barriers), gate in the south wall, 4 watchtowers
    h = 1.1
    for side in range(4):
        for k in range(-WALL_HALF + 1, WALL_HALF, 2):
            if side == 3 and abs(k) < 6:
                continue  # gate
            p = {0: [k, WALL_HALF], 1: [WALL_HALF, k], 2: [-WALL_HALF, k], 3: [k, -WALL_HALF]}[side]
            _box(w, [p[0], p[1], h], [1.0, 1.0, h], SAND if (k // 2) % 2 else TAN)
    for cx, cy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        x, y = cx * (WALL_HALF - 4), cy * (WALL_HALF - 4)
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            _deco(w, type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[x + dx, y + dy, 3], size=[0.12, 3], rgba=[0.3, 0.25, 0.2, 1])
        _box(w, [x, y, 6.2], [1.6, 1.6, 0.2], [0.35, 0.3, 0.22, 1])
        _box(w, [x, y, 7.4], [1.8, 1.8, 0.1], OLIVE)

    # tents, huts, vehicles, camo nets inside; trees outside
    meshes = {
        "tent": _mesh(spec, "tent", "tent/ridge_green_a/ridge_green_a.stl"),
        "dome": _mesh(spec, "dome", "tent/dome_c/dome_c.stl"),
        "hut": _mesh(spec, "hut", "hut/cabin_a/cabin_a.stl"),
        "shed": _mesh(spec, "shed", "hut/shed_d/shed_d.stl"),
        "truck": _mesh(spec, "truck", "truck/box_truck_a/box_truck_a.stl"),
        "pickup": _mesh(spec, "pickup", "pickup/pickup_a/pickup_a.stl"),
        "suv": _mesh(spec, "suv", "suv/suv_a/suv_a.stl"),
        "oak": _mesh(spec, "oak", "tree/oak_a/oak_a.stl"),
        "conifer": _mesh(spec, "conifer", "tree/conifer_d/conifer_d.stl"),
    }

    def place(m, x, y, yaw, rgba):
        q = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        _deco(w, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=meshes[m], pos=[x, y, 0.03], quat=q, rgba=rgba)

    for i, (x, y) in enumerate([(-40, 35), (-30, 35), (-20, 35), (-40, 22), (-30, 22), (-20, 22)]):
        place("tent", x, y, 0.0, OLIVE if i % 2 else [0.38, 0.40, 0.25, 1])
    place("dome", 35, 38, 0.5, [0.40, 0.42, 0.28, 1])
    place("hut", -38, -30, 0.0, [0.45, 0.40, 0.30, 1])
    place("shed", -25, -35, 1.57, [0.45, 0.40, 0.30, 1])
    for i, (x, y) in enumerate([(30, -38), (40, -38), (50, -38)]):
        place("truck" if i == 0 else "pickup", x, y, 1.57, OLIVE)
    place("suv", 38, 10, 0.3, OLIVE)
    for x, y in [(36, 22), (-35, 0)]:   # camo nets over a position
        for dx, dy in [(-5, -4), (-5, 4), (5, -4), (5, 4)]:
            _deco(w, type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[x + dx, y + dy, 1.5], size=[0.06, 1.5], rgba=[0.3, 0.25, 0.2, 1])
        _box(w, [x, y, 3.05], [6, 5, 0.05], [0.30, 0.34, 0.20, 0.85])
    for i in range(160):
        a, r = rng.uniform(0, 6.283), rng.uniform(90, 900)
        x, y = r * math.cos(a), r * math.sin(a)
        if abs(x) < 8 and y < 0:
            continue  # keep the road free
        place(rng.choice(["oak", "conifer"]), x, y, rng.uniform(0, 6.28),
              [0.18, 0.40, 0.16, 1] if rng.random() < 0.5 else [0.12, 0.30, 0.14, 1])
    return SimpleNamespace(name="simple", ground=flat_ground, default_range=400.0,
                           overview=dict(lookat=[0, 0, 0], distance=170, elevation=-32, azimuth=-60))


def add_pad(w) -> None:
    """Pad markings on the dark pad slab: white border and a bold white H (reads from the overview)."""
    white = [0.97, 0.97, 0.97, 1]
    for dx in (-2.0, 2.0):
        _box(w, [PAD[0] + dx, PAD[1], 0.105], [0.6, 2.9, 0.006], white)
    _box(w, [PAD[0], PAD[1], 0.105], [1.45, 0.55, 0.006], white)
    for s in (-1, 1):
        _box(w, [PAD[0] + s * 4.55, PAD[1], 0.104], [0.2, 4.75, 0.005], white)
        _box(w, [PAD[0], PAD[1] + s * 4.55, 0.104], [4.75, 0.2, 0.005], white)


def add_mast(w) -> None:
    """UMA-16 mast in the middle: sandbag ring, concrete foot, pole, a white housing (stylised, larger than the
    132 x 202 mm board so it reads from far away) carrying the green board with its 16 mics, and a translucent
    "listening" disc. The floating "UMA-16" label is drawn per frame (draw_markers)."""
    for k in range(18):
        a = k * 2 * math.pi / 18
        _box(w, [4.0 * math.cos(a), 4.0 * math.sin(a), 0.35], [0.6, 0.3, 0.35], SAND, a + math.pi / 2)
    _deco(w, type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[0, 0, 0.15], size=[0.6, 0.15], rgba=[0.6, 0.6, 0.58, 1])
    _deco(w, type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[0, 0, 1.6], size=[0.07, 1.45], rgba=[0.25, 0.25, 0.27, 1])
    _box(w, [0, 0, 3.06], [0.40, 0.32, 0.10], [0.93, 0.93, 0.90, 1])           # housing
    _box(w, [0, 0, 3.155], [0.42, 0.34, 0.012], [0.95, 0.45, 0.10, 1])         # orange rim
    _box(w, [0, 0, 3.17], [0.101, 0.066, 0.009], [0.1, 0.45, 0.2, 1])          # the board
    for ix in range(4):
        for iy in range(4):
            _deco(w, type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=[-0.063 + 0.042 * ix, -0.063 + 0.042 * iy, 3.18],
                  size=[0.006], rgba=[0.85, 0.75, 0.2, 1])
    _deco(w, type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[0, 0, 0.07], size=[12, 0.01], rgba=[0.2, 0.8, 1.0, 0.35])


def build_model(world: str = "farm"):
    spec = mujoco.MjSpec.from_file(str(X2_SCENE))
    spec.option.timestep = 0.002
    # clip planes relative to extent: 1 m .. 6 km (a 400 000:1 range made thin slabs z-fight)
    spec.stat.extent = 200
    spec.visual.map.znear, spec.visual.map.zfar = 0.005, 30
    spec.visual.quality.numslices, spec.visual.quality.numstacks = 16, 10   # thousands of tree ellipsoids
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
    spec.visual.headlight.diffuse = [0.22, 0.22, 0.22]
    spec.visual.headlight.ambient = [0.30, 0.30, 0.30]
    for light in list(spec.lights):   # the X2 scene's overhead light: with the sun it washed everything out
        spec.delete(light)
    spec.visual.rgba.haze = [0.75, 0.82, 0.90, 1]
    spec.visual.rgba.fog = [0.74, 0.80, 0.88, 1]          # distance fog hides the flat edge of the 4 km map
    spec.visual.map.fogstart, spec.visual.map.fogend = 4.0, 15.0   # x extent: 800 m .. 3 km
    for t in spec.textures:
        if t.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            t.rgb1 = [0.55, 0.72, 0.92]
            t.rgb2 = [0.85, 0.90, 0.95]
    spec.worldbody.add_light(name="sun", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=[0, 0, 500], dir=SUN_DIR,
                             diffuse=[0.60, 0.58, 0.53], specular=[0.1, 0.1, 0.1], castshadow=0)
    if world == "farm":
        import base3d_farm  # noqa: PLC0415  (demos/ is on sys.path when this script runs)

        fw = base3d_farm.FarmWorld()
        deco = fw.build(spec)
        info = SimpleNamespace(name="farm", ground=fw.ground, default_range=1200.0, world=fw,
                               overview=dict(lookat=[30, 60, 0], distance=520, elevation=-24, azimuth=58))
    else:
        deco = spec.worldbody.add_body(name="deco")
        info = build_simple(spec, deco)
    # the pad slab is collision geometry (the drone stands on it), its markings are decoration
    spec.worldbody.add_geom(name="pad", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[PAD[0], PAD[1], 0.05],
                            size=[5, 5, 0.05], rgba=[0.16, 0.17, 0.18, 1])
    add_pad(deco)
    add_mast(deco)

    # drone: start on the pad
    spec.body("x2").pos = [PAD[0], PAD[1], 0.25]
    for k in list(spec.keys):
        spec.delete(k)
    return spec.compile(), info


# ------------------------------------------------------------------------------------------ controller
class Controller:
    """Cascaded geometric controller (position -> attitude -> motor thrusts), on the true state."""

    def __init__(self, m: mujoco.MjModel):
        body = m.body("x2").id
        self.mass = float(m.body_subtreemass[body])
        self.ctrl_max = float(m.actuator_ctrlrange[0, 1])
        mix = np.zeros((4, 4))
        for i in range(4):
            x, y = m.site_pos[m.site(f"thrust{i + 1}").id][:2]
            mix[:, i] = [1.0, y, -x, m.actuator_gear[i, 5]]
        self.mix_inv = np.linalg.inv(mix)
        self.kp, self.kd = np.array([1.6, 1.6, 4.0]), np.array([2.4, 2.4, 3.5])
        self.kR, self.kw = np.array([4.0, 4.0, 1.0]), np.array([0.6, 0.6, 0.3])

    def __call__(self, pos, vel, R, omega, p_ref, v_ref, yaw_ref):
        acc = self.kp * (p_ref - pos) + self.kd * (v_ref - vel)
        acc[:2] = np.clip(acc[:2], -4.0, 4.0)
        acc[2] = np.clip(acc[2], -3.0, 4.0)
        f = self.mass * (acc + [0, 0, 9.81])
        z = f / np.linalg.norm(f)
        xc = np.array([math.cos(yaw_ref), math.sin(yaw_ref), 0.0])
        y = np.cross(z, xc)
        y /= np.linalg.norm(y)
        Rd = np.column_stack([np.cross(y, z), y, z])
        e = 0.5 * (Rd.T @ R - R.T @ Rd)
        torque = -self.kR * np.array([e[2, 1], e[0, 2], e[1, 0]]) - self.kw * omega
        return np.clip(self.mix_inv @ np.array([float(f @ R[:, 2]), *torque]), 0.0, self.ctrl_max)


def az_to_dir(az_deg: float) -> np.ndarray:
    """Compass azimuth (deg clockwise from north) -> ENU unit vector."""
    a = math.radians(az_deg)
    return np.array([math.sin(a), math.cos(a)])


# --------------------------------------------------------------------------------------------- mission
class Mission:
    """WAIT -> TAKEOFF -> OUTBOUND -> ORBIT -> RETURN -> DESCEND -> LANDED, with a moving setpoint.

    The horizontal setpoint moves at a fixed speed; the height setpoint follows the terrain at CRUISE_AGL, looking
    10 s ahead along the setpoint velocity (climbs start before the hill), rate-limited (CLIMB up, SINK down)."""

    def __init__(self, range_m: float, ground=flat_ground):
        self.range_m = range_m
        self.ground = ground
        self.pad_z = float(ground(*PAD))
        self.state = "WAIT"
        self.bearing: float | None = None
        self.sp = np.array([PAD[0], PAD[1], self.pad_z + 0.25])
        self.v = np.zeros(3)
        self.yaw = math.pi / 2
        self.t_state = 0.0
        self.log: list[str] = []

    def target(self) -> np.ndarray:
        xy = az_to_dir(self.bearing) * self.range_m
        return np.array([*xy, self.ground(*xy) + CRUISE_AGL])

    def set_bearing(self, az: float, t: float) -> None:
        self.bearing = az % 360.0
        if self.state in ("WAIT", "LANDED"):
            self._go("TAKEOFF", t)
        elif self.state in ("ORBIT", "RETURN", "DESCEND"):
            self._go("OUTBOUND", t)
        self.log.append(f"t={t:6.1f}s  bearing {self.bearing:5.1f} deg -> {self.state}")

    def _go(self, s: str, t: float) -> None:
        self.state, self.t_state = s, t
        self.log.append(f"t={t:6.1f}s  {s}")

    def _move_xy(self, goal, speed: float, dt: float) -> bool:
        d = np.asarray(goal[:2], float) - self.sp[:2]
        n = float(np.linalg.norm(d))
        if n <= speed * dt:
            self.sp[:2], self.v[:2] = goal[:2], 0.0
            return True
        self.v[:2] = d / n * speed
        self.sp[:2] += self.v[:2] * dt
        if n > 2.0:
            self.yaw = math.atan2(d[1], d[0])
        return False

    def _move_z(self, z_goal: float, dt: float) -> None:
        self.v[2] = float(np.clip(0.6 * (z_goal - self.sp[2]), -SINK, CLIMB))
        self.sp[2] += self.v[2] * dt

    def _follow_terrain(self, dt: float) -> None:
        look = self.sp[:2] + np.outer(np.linspace(0.0, 10.0, 6), self.v[:2])
        self._move_z(float(np.max(self.ground(look[:, 0], look[:, 1]))) + CRUISE_AGL, dt)

    def update(self, pos: np.ndarray, t: float, dt: float) -> None:
        s = self.state
        if s == "TAKEOFF":
            self.v[:2] = 0.0
            self._move_z(self.pad_z + CRUISE_AGL, dt)
            if abs(pos[2] - self.pad_z - CRUISE_AGL) < 1.0:
                self._go("OUTBOUND", t)
        elif s == "OUTBOUND":
            tgt = self.target()
            if self._move_xy(tgt, OUT_SPEED, dt) and np.linalg.norm(pos[:2] - tgt[:2]) < 3:
                self._go("ORBIT", t)
            self._follow_terrain(dt)
        elif s == "ORBIT":
            c = self.target()
            w = OUT_SPEED * 0.6 / ORBIT_R
            a = w * (t - self.t_state) + math.radians(90 - self.bearing) + math.pi
            self.sp[:2] = c[:2] + ORBIT_R * np.array([math.cos(a), math.sin(a)])
            self.v[:2] = ORBIT_R * w * np.array([-math.sin(a), math.cos(a)])
            self.yaw = math.atan2(c[1] - self.sp[1], c[0] - self.sp[0])  # look at the centre
            self._follow_terrain(dt)
            if t - self.t_state > ORBIT_S:
                self._go("RETURN", t)
        elif s == "RETURN":
            if self._move_xy(PAD, OUT_SPEED, dt) and np.linalg.norm(pos[:2] - PAD) < 1.5:
                self._go("DESCEND", t)
            self._follow_terrain(dt)
        elif s == "DESCEND":
            self._move_xy(PAD, OUT_SPEED, dt)
            floor = self.pad_z + 0.15
            self.v[2] = -DESCENT if self.sp[2] > floor else 0.0
            self.sp[2] = max(floor, self.sp[2] - DESCENT * dt)
            if pos[2] < self.pad_z + 0.35:
                self._go("LANDED", t)


# ------------------------------------------------------------------------------------------- UDP input
class UdpBearings:
    def __init__(self, port: int):
        self.latest: float | None = None
        self.lock = threading.Lock()
        if port <= 0:
            return
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        threading.Thread(target=self._run, daemon=True).start()
        print(f"listening for bearings on UDP {port} (JSON with az_deg, or a plain number)")

    def _run(self) -> None:
        while True:
            data, addr = self.sock.recvfrom(4096)
            txt = data.decode("utf-8", "replace").strip()
            try:
                az = float(json.loads(txt)["az_deg"]) if txt.startswith("{") else float(txt.split()[0])
            except (ValueError, KeyError, IndexError, json.JSONDecodeError):
                continue
            with self.lock:
                self.latest = az
            print(f"UDP bearing {az:.1f} deg from {addr[0]}")

    def pop(self) -> float | None:
        with self.lock:
            az, self.latest = self.latest, None
        return az


# ----------------------------------------------------------------------------------------------- run
def state_of(d):
    return d.qpos[0:3].copy(), d.qvel[0:3].copy(), d.xmat[1].reshape(3, 3).copy(), d.qvel[3:6].copy()


def step(m, d, ctrl, mission, t_wall_dt):
    pos, vel, R, om = state_of(d)
    mission.update(pos, d.time, m.opt.timestep)
    if mission.state in ("WAIT", "LANDED"):
        d.ctrl[:] = 0.0
    else:
        d.ctrl[:] = ctrl(pos, vel, R, om, mission.sp, mission.v, mission.yaw)
    mujoco.mj_step(m, d)


def _new_geom(scn):
    if scn.ngeom >= scn.maxgeom:
        return None
    scn.ngeom += 1
    return scn.geoms[scn.ngeom - 1]


def add_line(scn, a, b, width, rgba):
    g = _new_geom(scn)
    if g is not None:
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.asarray(rgba, float))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float), np.asarray(b, float))


def add_sphere(scn, p, r, rgba):
    g = _new_geom(scn)
    if g is not None:
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]), np.asarray(p, float),
                            np.eye(3).ravel(), np.asarray(rgba, float))


def add_label(scn, p, text, rgba=(1, 1, 1, 1)):
    g = _new_geom(scn)
    if g is not None:
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LABEL, np.zeros(3), np.asarray(p, float), np.eye(3).ravel(),
                            np.asarray(rgba, float))
        g.label = text


def cam_pos(cam) -> np.ndarray:
    """World position of a free / tracking mjvCamera."""
    az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
    fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    return np.asarray(cam.lookat, float) - cam.distance * fwd


def terrain_path(ground, az, length, step_m, h) -> np.ndarray:
    """Points along a bearing from the mast, h m above the terrain."""
    s = np.arange(0.0, length + step_m, step_m)
    xy = np.outer(s, az_to_dir(az))
    return np.column_stack([xy, np.asarray(ground(xy[:, 0], xy[:, 1]), float) + h])


def draw_markers(scn, d, mission, pending, trail, eye):
    """Bearing line (4 m above the terrain), uncertainty wedge, target, trail, the drone pin and the labels."""
    ground = mission.ground
    az = mission.bearing if mission.bearing is not None else pending
    rgba = [1.0, 0.25, 0.15, 0.9] if mission.bearing is not None else [1.0, 0.9, 0.2, 0.6]
    u = az_to_dir(az)   # line width from the camera's distance to the line: thin up close, visible from far
    t = float(np.clip(eye[:2] @ u, 0.0, mission.range_m + 150))
    w = float(np.clip(0.0012 * math.hypot(np.linalg.norm(eye[:2] - t * u), eye[2] - ground(*(t * u)) - 4), 0.05, 0.6))
    for s, step_m, width, col in ((0, 25.0, w, rgba), (-6, 50.0, w / 3, [1.0, 0.6, 0.2, 0.35]),
                                  (6, 50.0, w / 3, [1.0, 0.6, 0.2, 0.35])):   # the line and a +-6 deg wedge
        pts = terrain_path(ground, az + s, mission.range_m + 150, step_m, 4.0)
        pts[0] = [0, 0, 3.3]
        for a, b in zip(pts[:-1], pts[1:]):
            add_line(scn, a, b, width, col)
    if mission.bearing is not None:
        add_sphere(scn, mission.target(), 3.0, [1.0, 0.2, 0.1, 0.5])
    for i in range(1, len(trail)):
        add_line(scn, trail[i - 1], trail[i], 0.25, [0.2, 0.9, 1.0, 0.8])
    # drone pin, sized with the camera distance: a ball above the drone, a stem down to it, a dashed drop line
    p = d.qpos[0:3].copy()
    dist = float(np.linalg.norm(eye - p))
    if dist > 40:
        r = min(0.016 * dist, 16.0)
        top = p + [0, 0, 1.0 + 3 * r]
        add_line(scn, p + [0, 0, 0.3], top, 0.05 * r, [1.0, 0.75, 0.0, 0.9])
        add_sphere(scn, top, r, [1.0, 0.75, 0.0, 0.85])
        add_label(scn, top + [0, 0, 1.4 * r], "X2")
        for z in np.arange(float(ground(p[0], p[1])), p[2] - 1.0, 3.0):
            add_line(scn, [p[0], p[1], z], [p[0], p[1], min(z + 1.5, p[2])], 0.03 * r, [1.0, 0.75, 0.0, 0.6])
    add_label(scn, [0, 0, 4.6], "UMA-16")


def draw(viewer, d, mission, pending, trail):
    scn = viewer.user_scn
    scn.ngeom = 0
    draw_markers(scn, d, mission, pending, trail, cam_pos(viewer.cam))
    p = d.qpos[0:3]
    try:
        info = (f"State: {mission.state}\nBearing: {('%.0f deg' % mission.bearing) if mission.bearing is not None else '-'}"
                f"   (pending {pending:.0f})\nDistance from base: {np.linalg.norm(p[:2]):.0f} m   "
                f"Alt: {p[2] - mission.ground(p[0], p[1]):.0f} m AGL")
        viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                          "UMA-16 -> recon drone", info))
    except Exception:  # noqa: BLE001  (older viewer without set_texts)
        pass


def set_overview(cam, info):
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    o = info.overview
    cam.lookat[:] = o["lookat"]
    cam.distance, cam.elevation, cam.azimuth = o["distance"], o["elevation"], o["azimuth"]


def set_chase(cam, m):
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = m.body("x2").id
    cam.distance, cam.elevation = 12, -20


def run_viewer(m, d, info, args):
    ctrl, udp = Controller(m), UdpBearings(args.udp_port)
    mission = Mission(args.range, info.ground)
    keys: list[int] = []
    pending = float(args.bearing) if args.bearing is not None else 0.0
    if args.bearing is not None:
        mission.set_bearing(args.bearing, d.time)
    speed = 1
    trail: list[np.ndarray] = []
    chase = False
    with mujoco.viewer.launch_passive(m, d, key_callback=keys.append) as viewer:
        set_overview(viewer.cam, info)
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = info.name == "farm"   # synced to the main scene
        last_print = ""
        while viewer.is_running():
            t0 = time.time()
            while keys:
                k = keys.pop(0)
                if k == 263:
                    pending = (pending - 5) % 360
                elif k == 262:
                    pending = (pending + 5) % 360
                elif k == 257:
                    mission.set_bearing(pending, d.time)
                elif k == ord("F"):
                    pending = random.uniform(0, 360)
                    mission.set_bearing(pending, d.time)
                elif k in (ord("1"), ord("2"), ord("4")):
                    speed = int(chr(k))
                elif k == ord("C"):
                    chase = not chase
                    set_chase(viewer.cam, m) if chase else set_overview(viewer.cam, info)
                elif k == 259:
                    mujoco.mj_resetData(m, d)
                    mission, trail = Mission(args.range, info.ground), []
            az = udp.pop()
            if az is not None:
                pending = az
                mission.set_bearing(az, d.time)
            with viewer.lock():
                for _ in range(int(speed / 60 / m.opt.timestep)):
                    step(m, d, ctrl, mission, 0)
                if not trail or np.linalg.norm(d.qpos[0:3] - trail[-1]) > 4:
                    trail.append(d.qpos[0:3].copy())
                    trail = trail[-400:]
                draw(viewer, d, mission, pending, trail)
            if mission.log and mission.log[-1] != last_print:
                last_print = mission.log[-1]
                print(last_print)
            viewer.sync()
            time.sleep(max(0.0, 1 / 60 - (time.time() - t0)))


def drone_contacts(m, d, drone_body: int) -> int:
    """Contacts between the drone and anything (terrain, pad, floor)."""
    return sum(drone_body in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]) for c in d.contact[:d.ncon])


def run_headless(m, d, info, args):
    ctrl, mission = Controller(m), Mission(args.range, info.ground)
    mission.set_bearing(args.bearing if args.bearing is not None else 60.0, d.time)
    drone = m.body("x2").id
    max_xtrack, min_agl, contact_steps = 0.0, math.inf, 0
    nxt = 0.0
    while d.time < args.max_t and mission.state != "LANDED":
        step(m, d, ctrl, mission, 0)
        p = d.qpos[0:3]
        agl = p[2] - float(info.ground(p[0], p[1]))
        if mission.state == "OUTBOUND":
            u = az_to_dir(mission.bearing)
            max_xtrack = max(max_xtrack, abs(u[0] * p[1] - u[1] * p[0]))
        if mission.state in ("OUTBOUND", "ORBIT", "RETURN"):
            min_agl = min(min_agl, agl)
        in_flight = mission.state in ("OUTBOUND", "ORBIT", "RETURN") or (
            mission.state in ("TAKEOFF", "DESCEND") and p[2] > mission.pad_z + 1.0)
        if in_flight and drone_contacts(m, d, drone):
            contact_steps += 1
        if d.time >= nxt:
            nxt += 10.0
            print(f"t={d.time:6.1f}  {mission.state:8s}  pos=({p[0]:7.1f},{p[1]:7.1f},{p[2]:5.1f})  "
                  f"AGL {agl:5.1f}  dist={np.linalg.norm(p[:2]):6.1f} m")
    p = d.qpos[0:3]
    err = float(np.linalg.norm(p[:2] - PAD))
    print("\n".join(mission.log))
    print(f"RESULT world={info.name} bearing={mission.bearing:.0f} state={mission.state} t={d.time:.1f}s  "
          f"landing error {err:.2f} m  max cross-track outbound {max_xtrack:.2f} m  min AGL in flight "
          f"{min_agl:.1f} m  ground-contact steps in flight {contact_steps}")
    return mission.state == "LANDED" and err < 1.0 and contact_steps == 0


# --------------------------------------------------------------------------------------------- renders
def run_shots(m, d, info, args):
    """Offscreen renders to args.shots (overview, mast close-up, village, drone 1 km out) + render frame rate."""
    import cv2  # noqa: PLC0415

    out = Path(args.shots)
    out.mkdir(parents=True, exist_ok=True)
    W, H = 1280, 720
    r = mujoco.Renderer(m, H, W, max_geom=20000)
    cam = mujoco.MjvCamera()
    show = Mission(args.range, info.ground)
    show.bearing = 60.0

    def render(mission):
        r.update_scene(d, cam)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = info.name == "farm"
        draw_markers(r.scene, d, mission, 60.0, [], cam_pos(cam))
        return r.render()

    def shot(name, lookat, dist, el, az, mission=show):
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:], cam.distance, cam.elevation, cam.azimuth = lookat, dist, el, az
        cv2.imwrite(str(out / f"{info.name}_{name}.jpg"), cv2.cvtColor(render(mission), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"saved {out / f'{info.name}_{name}.jpg'}")

    o = info.overview
    shot("overview", o["lookat"], o["distance"], o["elevation"], o["azimuth"])
    if info.name == "farm":
        shot("mast_forecourt", [7, -4, 2.0], 21, -11, 105)
        vx, vy = info.world.village_xy
        shot("village", [vx, vy, info.ground(vx, vy)], 380, -28, 115)

    # frame rate of the render path (scene update + markers + render), overview and chase
    for label, setup in (("overview", lambda: set_overview(cam, info)), ("chase", lambda: set_chase(cam, m))):
        setup()
        render(show)
        t0, n = time.time(), 60
        for _ in range(n):
            render(show)
        print(f"render fps ({label}, {W}x{H}, {r.scene.ngeom} scene geoms): {n / (time.time() - t0):.0f}")
    if info.name != "farm":
        return

    # fly out along 060 until 1 km from the base, then look at the drone over the fields from behind
    ctrl, fly = Controller(m), Mission(args.range, info.ground)
    fly.set_bearing(60.0, d.time)
    while np.linalg.norm(d.qpos[:2]) < min(1000.0, 0.85 * args.range) and d.time < args.max_t:
        step(m, d, ctrl, fly, 0)
    p = d.qpos[0:3].copy()
    shot("drone_1km", p + [0, 0, 1], 7, -12, 90 - 60 - 25, fly)          # from behind, a little to the side
    shot("drone_1km_wide", p, 160, -24, 90 - 60 - 40, fly)
    print(f"drone at ({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f}), {np.linalg.norm(p[:2]):.0f} m out, "
          f"{p[2] - info.ground(p[0], p[1]):.1f} m AGL")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", choices=["farm", "simple"], default="farm")
    ap.add_argument("--bearing", type=float, default=None, help="launch right away toward this bearing (deg)")
    ap.add_argument("--range", type=float, default=None,
                    help="how far out along the bearing (m); farm 1200, simple 400")
    ap.add_argument("--udp-port", type=int, default=5005, help="0 = off")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--shots", default=None, help="save offscreen renders to this directory and measure the fps")
    ap.add_argument("--max-t", type=float, default=900.0)
    args = ap.parse_args()
    t0 = time.time()
    m, info = build_model(args.world)
    if args.range is None:
        args.range = info.default_range
    trees = f", {info.world.n_trees_3d} 3D trees" if info.name == "farm" else ""
    print(f"world {info.name}: {m.ngeom} geoms, {m.nmesh} meshes{trees}, built in {time.time() - t0:.1f}s")
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    if args.shots:
        run_shots(m, d, info, args)
    elif args.headless:
        raise SystemExit(0 if run_headless(m, d, info, args) else 1)
    else:
        run_viewer(m, d, info, args)


if __name__ == "__main__":
    main()
