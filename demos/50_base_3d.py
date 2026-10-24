"""3D view of the base in MuJoCo: the UMA-16 array in the middle, the drone takes off from the pad and flies
out along the bearing the array reports, looks around, comes back and lands.

    uv run python demos/50_base_3d.py                         # viewer; give a bearing with the keys or over UDP
    uv run python demos/50_base_3d.py --bearing 60            # take off right away toward 060 deg
    uv run python demos/50_base_3d.py --headless --bearing 60 # no window: fly once, print the log

Bearing input (degrees clockwise from north, compass convention, same as BearingEstimate.az_deg):
- keys in the viewer: LEFT / RIGHT turn the pending bearing by 5 deg, ENTER launches toward it,
  F launches toward a random bearing, C switches overview / chase camera, 1 / 2 / 4 sim speed, BACKSPACE resets;
- UDP port 5005 (`--udp-port`): a JSON BearingEstimate ({"az_deg": 60, ...}) or a plain number "60". A new
  bearing while the drone is out retargets it.

Self-contained on purpose (own scene, simple cascaded controller on the true state, GNSS on): this is the MVP view
of "clap -> the drone flies that way". Ground truth state is used for control: no estimation here yet.
World frame: ENU, x east, y north, z up; the UMA-16 mast is at the origin.
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

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
X2_SCENE = ROOT / "third_party" / "skydio_x2" / "scene.xml"
OBJ = ROOT / "assets" / "objects" / "objects"

PAD = np.array([18.0, -10.0])   # landing pad, m (E, N) from the UMA-16 mast
WALL_HALF = 60                  # HESCO perimeter: 120 m x 120 m square, gate in the south wall
CRUISE_AGL = 30.0
OUT_SPEED = 8.0
CLIMB = 2.5
DESCENT = 1.0
ORBIT_R = 35.0
ORBIT_S = 25.0

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


def build_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(X2_SCENE))
    spec.option.timestep = 0.002
    spec.visual.map.zfar = 4000
    spec.stat.extent = 150
    w = spec.worldbody

    # ground: grass texture, sun, sky
    tex = spec.texture("groundplane")
    tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    tex.rgb1 = [0.30, 0.42, 0.20]
    tex.rgb2 = [0.27, 0.39, 0.18]
    tex.mark = mujoco.mjtMark.mjMARK_NONE
    spec.material("groundplane").texrepeat = [400, 400]
    spec.material("groundplane").reflectance = 0.0
    floor = spec.geom("floor")
    floor.size = [3000, 3000, 0.1]
    for t in spec.textures:
        if t.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            t.rgb1 = [0.55, 0.72, 0.92]
            t.rgb2 = [0.85, 0.90, 0.95]
    w.add_light(name="sun", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=[0, 0, 500], dir=[-0.3, -0.4, -1],
                diffuse=[0.8, 0.78, 0.7], castshadow=0)

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

    # landing pad with an H
    _box(w, [PAD[0], PAD[1], 0.05], [5, 5, 0.05], [0.55, 0.55, 0.55, 1])
    for dx in (-1.6, 1.6):
        _box(w, [PAD[0] + dx, PAD[1], 0.11], [0.35, 2.2, 0.01], [0.95, 0.95, 0.95, 1])
    _box(w, [PAD[0], PAD[1], 0.11], [1.6, 0.35, 0.01], [0.95, 0.95, 0.95, 1])

    # UMA-16 mast in the middle: sandbag ring, concrete foot, pole, the board (132 x 202 mm) with 16 mics,
    # and a translucent "listening" disc so it reads from far away
    for k in range(18):
        a = k * 2 * math.pi / 18
        _box(w, [4.0 * math.cos(a), 4.0 * math.sin(a), 0.35], [0.6, 0.3, 0.35], SAND, a + math.pi / 2)
    _deco(w, type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[0, 0, 0.15], size=[0.6, 0.15], rgba=[0.6, 0.6, 0.58, 1])
    _deco(w, type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[0, 0, 1.65], size=[0.05, 1.5], rgba=[0.25, 0.25, 0.27, 1])
    _box(w, [0, 0, 3.17], [0.101, 0.066, 0.009], [0.1, 0.45, 0.2, 1])
    for ix in range(4):
        for iy in range(4):
            _deco(w, type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=[-0.063 + 0.042 * ix, -0.063 + 0.042 * iy, 3.18],
                  size=[0.006], rgba=[0.85, 0.75, 0.2, 1])
    _deco(w, type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[0, 0, 0.06], size=[12, 0.01], rgba=[0.2, 0.8, 1.0, 0.35])

    # drone: start on the pad
    spec.body("x2").pos = [PAD[0], PAD[1], 0.25]
    for k in list(spec.keys):
        spec.delete(k)
    return spec.compile()


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
    """WAIT -> TAKEOFF -> OUTBOUND -> ORBIT -> RETURN -> DESCEND -> LANDED, with a moving setpoint."""

    def __init__(self, range_m: float):
        self.range_m = range_m
        self.state = "WAIT"
        self.bearing: float | None = None
        self.sp = np.array([PAD[0], PAD[1], 0.25])
        self.v = np.zeros(3)
        self.yaw = math.pi / 2
        self.t_state = 0.0
        self.log: list[str] = []

    def target(self) -> np.ndarray:
        d = az_to_dir(self.bearing)
        return np.array([*(d * self.range_m), CRUISE_AGL])

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

    def _move_to(self, goal: np.ndarray, speed: float, dt: float) -> bool:
        d = goal - self.sp
        n = float(np.linalg.norm(d))
        if n <= speed * dt:
            self.sp, self.v = goal.copy(), np.zeros(3)
            return True
        self.v = d / n * speed
        self.sp = self.sp + self.v * dt
        if np.linalg.norm(self.v[:2]) > 0.5:
            self.yaw = math.atan2(self.v[1], self.v[0])
        return False

    def update(self, pos: np.ndarray, t: float, dt: float) -> None:
        s = self.state
        if s == "TAKEOFF":
            if self._move_to(np.array([PAD[0], PAD[1], CRUISE_AGL]), CLIMB, dt) and abs(pos[2] - CRUISE_AGL) < 1.0:
                self._go("OUTBOUND", t)
        elif s == "OUTBOUND":
            if self._move_to(self.target(), OUT_SPEED, dt) and np.linalg.norm(pos[:2] - self.target()[:2]) < 3:
                self._go("ORBIT", t)
        elif s == "ORBIT":
            c = self.target()
            w = OUT_SPEED * 0.6 / ORBIT_R
            a = w * (t - self.t_state) + math.radians(90 - self.bearing) + math.pi
            self.sp = c + ORBIT_R * np.array([math.cos(a), math.sin(a), 0.0])
            self.v = ORBIT_R * w * np.array([-math.sin(a), math.cos(a), 0.0])
            self.yaw = math.atan2(c[1] - self.sp[1], c[0] - self.sp[0])  # look at the centre
            if t - self.t_state > ORBIT_S:
                self._go("RETURN", t)
        elif s == "RETURN":
            if self._move_to(np.array([PAD[0], PAD[1], CRUISE_AGL]), OUT_SPEED, dt) and \
                    np.linalg.norm(pos[:2] - PAD) < 1.5:
                self._go("DESCEND", t)
        elif s == "DESCEND":
            self._move_to(np.array([PAD[0], PAD[1], 0.15]), DESCENT, dt)
            if pos[2] < 0.35:
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


def add_line(scn, a, b, width, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(rgba, float))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float), np.asarray(b, float))
    scn.ngeom += 1


def add_sphere(scn, p, r, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]), np.asarray(p, float),
                        np.eye(3).ravel(), np.asarray(rgba, float))
    scn.ngeom += 1


def draw(viewer, d, mission, pending, trail):
    scn = viewer.user_scn
    scn.ngeom = 0
    mast = np.array([0, 0, 3.2])
    az = mission.bearing if mission.bearing is not None else pending
    end = np.array([*(az_to_dir(az) * (mission.range_m + 150)), 3.2])
    rgba = [1.0, 0.25, 0.15, 0.9] if mission.bearing is not None else [1.0, 0.9, 0.2, 0.6]
    add_line(scn, mast, end, 0.6, rgba)
    for s in (-6, 6):  # uncertainty wedge, +-6 deg
        e = np.array([*(az_to_dir(az + s) * (mission.range_m + 150)), 3.2])
        add_line(scn, mast, e, 0.2, [1.0, 0.6, 0.2, 0.35])
    if mission.bearing is not None:
        add_sphere(scn, mission.target(), 3.0, [1.0, 0.2, 0.1, 0.5])
    p = d.qpos[0:3]
    add_sphere(scn, p + [0, 0, 1.2], 1.0, [1.0, 1.0, 0.1, 0.5])  # halo so the small X2 reads from far away
    for i in range(1, len(trail)):
        add_line(scn, trail[i - 1], trail[i], 0.25, [0.2, 0.9, 1.0, 0.8])
    add_compass(scn)


def add_compass(scn, r0: float = 70.0, r1: float = 95.0) -> None:
    """Ticks at 0/90/180/270 deg around the mast (0 = north = the array's reference mark, white)."""
    for az in (0, 90, 180, 270):
        u = az_to_dir(az)
        n = scn.ngeom
        add_line(scn, [*(u * r0), 0.5], [*(u * r1), 0.5], 1.2 if az == 0 else 0.7,
                 [1.0, 1.0, 1.0, 0.95] if az == 0 else [0.3, 0.3, 0.3, 0.9])
        if scn.ngeom > n:
            scn.geoms[n].label = f"{az} deg" + (" (N)" if az == 0 else "")


def hud_text(d, mission, pending) -> str:
    p = d.qpos[0:3]
    b = f"{mission.bearing:.0f} deg" if mission.bearing is not None else "-"
    return (f"State: {mission.state}\nBearing: {b}   (pending {pending:.0f}, ENTER to launch)\n"
            f"Distance from base: {np.linalg.norm(p[:2]):.0f} m   Alt: {p[2]:.0f} m")


def run_viewer(m, d, args):
    ctrl, mission, udp = Controller(m), Mission(args.range), UdpBearings(args.udp_port)
    keys: list[int] = []
    pending = float(args.bearing) if args.bearing is not None else 0.0
    if args.bearing is not None:
        mission.set_bearing(args.bearing, d.time)
    speed = 1
    trail: list[np.ndarray] = []
    chase = False
    with mujoco.viewer.launch_passive(m, d, key_callback=keys.append) as viewer:
        viewer.cam.lookat[:] = [0, 0, 0]
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 200, -55, 90  # looking north: 0 deg at the top
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
                    if chase:
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                        viewer.cam.trackbodyid = m.body("x2").id
                        viewer.cam.distance = 12
                    else:
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                        viewer.cam.lookat[:] = [0, 0, 0]
                        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 200, -55, 90
                elif k == 259:
                    mujoco.mj_resetData(m, d)
                    mission, trail = Mission(args.range), []
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
                hud = hud_text(d, mission, pending)
            # set_texts takes the viewer's lock itself: calling it inside `with viewer.lock()` deadlocks
            try:
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                  "UMA-16 -> recon drone", hud))
            except AttributeError:  # older viewer without set_texts
                pass
            if mission.log and mission.log[-1] != last_print:
                last_print = mission.log[-1]
                print(last_print)
            viewer.sync()
            time.sleep(max(0.0, 1 / 60 - (time.time() - t0)))


def run_headless(m, d, args):
    ctrl, mission = Controller(m), Mission(args.range)
    mission.set_bearing(args.bearing if args.bearing is not None else 60.0, d.time)
    max_xtrack = 0.0
    nxt = 0.0
    while d.time < args.max_t and mission.state != "LANDED":
        step(m, d, ctrl, mission, 0)
        p = d.qpos[0:3]
        if mission.state == "OUTBOUND":
            u = az_to_dir(mission.bearing)
            max_xtrack = max(max_xtrack, abs(u[0] * p[1] - u[1] * p[0]))
        if d.time >= nxt:
            nxt += 5.0
            print(f"t={d.time:6.1f}  {mission.state:8s}  pos=({p[0]:7.1f},{p[1]:7.1f},{p[2]:5.1f})  "
                  f"dist={np.linalg.norm(p[:2]):6.1f} m")
    p = d.qpos[0:3]
    print("\n".join(mission.log))
    print(f"RESULT state={mission.state} t={d.time:.1f}s  landing error {np.linalg.norm(p[:2] - PAD):.2f} m  "
          f"max cross-track outbound {max_xtrack:.2f} m")
    return mission.state == "LANDED"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bearing", type=float, default=None, help="launch right away toward this bearing (deg)")
    ap.add_argument("--range", type=float, default=400.0, help="how far out along the bearing (m)")
    ap.add_argument("--udp-port", type=int, default=5005, help="0 = off")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--max-t", type=float, default=600.0)
    args = ap.parse_args()
    m = build_model()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    if args.headless:
        raise SystemExit(0 if run_headless(m, d, args) else 1)
    run_viewer(m, d, args)


if __name__ == "__main__":
    main()
