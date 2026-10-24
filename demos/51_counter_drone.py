"""Counter-drone demo: the UMA-16 hears a drone, our X2 flies out along the bearing and finds it with a camera.

    uv run python demos/51_counter_drone.py                        # viewer; bearings over UDP 5005 or the keys
    uv run python demos/51_counter_drone.py --headless --walk 0:0,40:0,130:90   # scripted walk, print the result
    uv run python scripts/fake_walk.py --from 0 --to 90 --rate 1   # a fake speaker walk over UDP, no array needed

Stage -> sim: a person walks around the array with a speaker playing drone sound. The INTRUDER (red quad) sits
`--range` m out (default 500) and `--intruder-agl` m above the terrain on the bearing of that sound, smoothed and
speed-limited (`--intruder-speed`, 20 m/s: at 500 m one degree of walking is 8.7 m, so walk about 1 deg/s).

Our drone knows only what the array sends (bearing + sigma, about 5 Hz), not the range:
  WAIT -> TAKEOFF (sound heard for 1.5 s) -> INTERCEPT: fly out along the audio ray at 15 m/s, leading the
  bearing rate, the gimbal camera looking along the ray -> TRACK: the camera confirmed the intruder (3 hits in
  2 s); hold station 40 m short of it and keep it centred -> LOST (no hit for 3 s): back to the audio ray.
  No sound and no visual for 15 s, or 1500 m out without a find -> RETURN -> DESCEND -> LANDED.

Camera (a simulated detector, not a neural net): 20 x 15 deg zoom on a gimbal, 5 frames/s. A frame detects the
intruder when it is inside the field of view, a ray from the camera reaches it (terrain, trees and buildings
occlude), and then with probability 0 at 3 px rising to 0.9 at 8 px of apparent size (1.2 m span: 8 px at
~280 m). Range from the apparent size (+-5 %), direction from the pixel (+-0.3 deg).

Keys: LEFT / RIGHT move a manual sound bearing by 5 deg and count as hearing it (no array needed), S stops the
manual sound, C overview / chase camera, V hides the camera picture, 1 / 2 / 4 sim speed, BACKSPACE resets.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import socket
import sys
import threading
import time
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
base = importlib.import_module("50_base_3d")
PAD, az_to_dir, add_line, add_sphere, add_label = base.PAD, base.az_to_dir, base.add_line, base.add_sphere, base.add_label

HUNT_AGL = 55.0            # our drone while searching (the intruder flies higher: the camera looks a little up)
HUNT_SPEED = 15.0
TRACK_SPEED = 18.0
STANDOFF = 40.0
MAX_OUT = 1500.0
LEAD_S = 4.0               # aim where the bearing will be in this many seconds
FOV_H, FOV_V = 20.0, 15.0
CAM_W, CAM_H = 640, 480
FRAME_DT = 0.2
SPAN = 1.2                 # intruder size for the pixel count (m)
PX_MIN, PX_FULL, P_MAX = 3.0, 8.0, 0.9
CONFIRM_HITS, CONFIRM_S, LOST_S = 3, 2.0, 3.0
HEAR_S = 1.5               # launch after hearing the sound this long
AUDIO_TIMEOUT = 3.0        # no bearing for this long = not hearing
GIVE_UP_S = 15.0
MSG_DT = 0.2               # manual / scripted bearings at 5 Hz


def adiff(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def blend_deg(a: float, b: float, k: float) -> float:
    return (a + k * adiff(b, a)) % 360.0


# ------------------------------------------------------------------------------------------- what we hear
class Audio:
    """The array's bearings as our drone sees them: smoothed bearing (0.5 s), bearing rate, sigma."""

    def __init__(self):
        self.az: float | None = None
        self.raw: float | None = None
        self.sigma, self.rate = 5.0, 0.0
        self.t_last, self.t_first = -1e9, 0.0
        self.history: deque[tuple[float, float]] = deque(maxlen=150)

    def update(self, az: float, sigma: float, t: float) -> None:
        if self.az is None or t - self.t_last > AUDIO_TIMEOUT:
            self.az, self.rate, self.t_first = az % 360.0, 0.0, t
        else:
            dt = max(t - self.t_last, 1e-3)
            new = blend_deg(self.az, az, 1 - math.exp(-dt / 0.5))
            self.rate += (1 - math.exp(-dt / 2.0)) * (adiff(new, self.az) / dt - self.rate)
            self.az = new
        self.raw, self.sigma, self.t_last = az % 360.0, max(float(sigma), 2.0), t
        self.history.append((t, self.raw))

    def hearing(self, t: float) -> bool:
        return self.az is not None and t - self.t_last <= AUDIO_TIMEOUT

    def heard_for(self, t: float) -> float:
        return t - self.t_first if self.hearing(t) else 0.0


# ---------------------------------------------------------------------------------------------- intruder
class Intruder:
    """Sits `range_m` out on the (smoothed, 1 s) sound bearing, `agl` m up; moves at most `speed` m/s."""

    def __init__(self, range_m: float, agl: float, speed: float, ground):
        self.range_m, self.agl, self.speed, self.ground = range_m, agl, speed, ground
        self.az: float | None = None
        self.t_msg = 0.0
        self.pos: np.ndarray | None = None
        self.yaw = 0.0

    def hear(self, az: float, t: float) -> None:
        if self.az is None:
            self.az = az % 360.0
        else:
            self.az = blend_deg(self.az, az, 1 - math.exp(-max(t - self.t_msg, 1e-3) / 1.0))
        self.t_msg = t
        if self.pos is None:
            xy = az_to_dir(self.az) * self.range_m
            self.pos = np.array([*xy, self.ground(*xy) + self.agl])

    def update(self, t: float, dt: float) -> None:
        if self.pos is None:
            return
        goal = az_to_dir(self.az) * self.range_m
        dxy = goal - self.pos[:2]
        n = float(np.linalg.norm(dxy))
        if n > 1e-6:
            self.pos[:2] += dxy / n * min(n, self.speed * dt)
            if n > 0.5:
                self.yaw = math.atan2(dxy[1], dxy[0])
        self.pos[2] = float(self.ground(*self.pos[:2])) + self.agl + 0.6 * math.sin(0.7 * t)

    def range_from_base(self) -> float:
        return float(np.linalg.norm(self.pos[:2])) if self.pos is not None else 0.0


# ------------------------------------------------------------------------------------------------ camera
def cam_axes(yaw: float, pitch: float):
    fwd = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
    right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    return fwd, right, np.cross(right, fwd)


class Camera:
    """Gimbal zoom camera on our drone with the simulated detector (see the module doc)."""

    def __init__(self, m, d, rng: np.random.Generator):
        self.m, self.d, self.rng = m, d, rng
        self.drone = m.body("x2").id
        self.yaw, self.pitch = math.pi / 2, 0.0
        self.last: dict = {}
        self._gid = np.zeros(1, np.int32)

    def pos(self) -> np.ndarray:
        return self.d.qpos[0:3] + [0, 0, -0.2]

    def frame(self, target: np.ndarray | None) -> dict:
        r = {"in_view": False, "occluded": False, "px": 0.0, "hit": False, "est": None}
        if target is not None:
            c = self.pos()
            v = target - c
            dist = float(np.linalg.norm(v))
            u = v / dist
            fwd, right, up = cam_axes(self.yaw, self.pitch)
            z = float(u @ fwd)
            ah, av = math.degrees(math.atan2(u @ right, z)), math.degrees(math.atan2(u @ up, z))
            r.update(dist=dist, ah=ah, av=av, px=SPAN / (dist * 2 * math.tan(math.radians(FOV_H / 2)) / CAM_W))
            r["in_view"] = z > 0 and abs(ah) <= FOV_H / 2 and abs(av) <= FOV_V / 2
            if r["in_view"]:
                g = mujoco.mj_ray(self.m, self.d, c, u, None, 1, self.drone, self._gid)
                r["occluded"] = 0.0 <= g < dist - 1.0
                p = P_MAX * float(np.clip((r["px"] - PX_MIN) / (PX_FULL - PX_MIN), 0.0, 1.0))
                r["hit"] = not r["occluded"] and self.rng.random() < p
            if r["hit"]:   # what the image gives: direction from the pixel, range from the apparent size
                yaw_m = math.atan2(u[1], u[0]) + math.radians(self.rng.normal(0, 0.3))
                pit_m = math.asin(float(np.clip(u[2], -1, 1))) + math.radians(self.rng.normal(0, 0.3))
                r["est"] = c + cam_axes(yaw_m, pit_m)[0] * dist * (1 + self.rng.normal(0, 0.05))
        self.last = r
        return r


# ------------------------------------------------------------------------------------------------- hunter
class Hunter(base.Mission):
    """Our X2: WAIT -> TAKEOFF -> INTERCEPT <-> TRACK / LOST -> RETURN -> DESCEND -> LANDED."""

    def __init__(self, ground, agl: float = HUNT_AGL):
        super().__init__(MAX_OUT, ground)
        self.agl = agl
        self.hits: deque[float] = deque()
        self.t_hit = -1e9
        self.est: np.ndarray | None = None
        self.t_launch: float | None = None
        self.t_found: float | None = None

    def _terrain(self, dt: float, agl: float) -> None:
        look = self.sp[:2] + np.outer(np.linspace(0.0, 10.0, 6), self.v[:2])
        self._move_z(float(np.max(self.ground(look[:, 0], look[:, 1]))) + agl, dt)

    def on_frame(self, r: dict, t: float) -> None:
        if not r["hit"]:
            return
        self.hits.append(t)
        while self.hits and t - self.hits[0] > CONFIRM_S:
            self.hits.popleft()
        self.t_hit, self.est = t, r["est"]
        if self.state in ("INTERCEPT", "LOST") and len(self.hits) >= CONFIRM_HITS:
            if self.t_found is None:
                self.t_found = t
            self._go("TRACK", t)

    def aim(self, cam: Camera, audio: Audio, t: float) -> None:
        c = cam.pos()
        if self.state == "TRACK" and self.est is not None:
            v = self.est - c
        elif audio.az is not None:
            r_look = float(np.linalg.norm(c[:2])) + 250.0
            p = az_to_dir(audio.az) * r_look
            v = np.array([*(p - c[:2]), float(self.ground(*p)) + self.agl + 20.0 - c[2]])
        else:
            return
        tri = lambda period: 2 * abs((t / period) % 1.0 - 0.5) * 2 - 1    # noqa: E731  (-1..1 triangle)
        sweep_y = 0.0 if self.state == "TRACK" else float(np.clip(2 * audio.sigma - FOV_H / 2, 0, 30)) * tri(3.0)
        sweep_p = 0.0 if self.state == "TRACK" else 5.0 * tri(4.0)
        cam.yaw = math.atan2(v[1], v[0]) + math.radians(sweep_y)
        cam.pitch = math.atan2(v[2], float(np.linalg.norm(v[:2]))) + math.radians(sweep_p)

    def update(self, pos, t, dt, audio: Audio | None = None) -> None:
        s = self.state
        if s == "WAIT":
            if audio is not None and audio.heard_for(t) >= HEAR_S:
                self.t_launch = t
                self._go("TAKEOFF", t)
        elif s == "TAKEOFF":
            self.v[:2] = 0.0
            self._move_z(self.pad_z + self.agl, dt)
            if pos[2] > self.pad_z + 15.0:
                self._go("INTERCEPT", t)
        elif s in ("INTERCEPT", "LOST"):
            r_own = float(np.linalg.norm(pos[:2]))
            if s == "LOST" and t - self.t_hit > LOST_S + 10.0:
                self._go("INTERCEPT", t)
            if (audio.az is None or t - audio.t_last > GIVE_UP_S) or r_own > MAX_OUT:
                self._go("RETURN", t)
                return
            az = audio.az + float(np.clip(audio.rate, -5, 5)) * LEAD_S
            self._move_xy(az_to_dir(az) * min(r_own + 80.0, MAX_OUT + 10.0), HUNT_SPEED, dt)
            self._terrain(dt, self.agl)
        elif s == "TRACK":
            if t - self.t_hit > LOST_S:
                self._go("LOST", t)
                return
            dxy = self.est[:2] - self.sp[:2]
            n = max(float(np.linalg.norm(dxy)), 1e-6)
            self._move_xy(self.est[:2] - dxy / n * STANDOFF, TRACK_SPEED, dt)
            self.yaw = math.atan2(dxy[1], dxy[0])
            floor = float(self.ground(*self.sp[:2])) + 20.0
            self._move_z(max(float(self.est[2]) - 10.0, floor), dt)
            if audio.az is None or t - audio.t_last > GIVE_UP_S:   # the sound stopped: the demo is over
                self._go("RETURN", t)
        else:
            super().update(pos, t, dt)


# ------------------------------------------------------------------------------------------------- input
class UdpTrack:
    """BearingEstimate JSON ({"az_deg", "sigma_az_deg", ...}; "detected": false = silence) or a plain number."""

    def __init__(self, port: int):
        self.msgs: list[tuple[float, float]] = []
        self.lock = threading.Lock()
        if port <= 0:
            return
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        threading.Thread(target=self._run, daemon=True).start()
        print(f"listening for bearings on UDP {port}")

    def _run(self) -> None:
        while True:
            txt = self.sock.recvfrom(4096)[0].decode("utf-8", "replace").strip()
            try:
                if txt.startswith("{"):
                    j = json.loads(txt)
                    if j.get("detected", True) is False:
                        continue
                    msg = (float(j["az_deg"]), float(j.get("sigma_az_deg", 5.0)))
                else:
                    msg = (float(txt.split()[0]), 5.0)
            except (ValueError, KeyError, IndexError, json.JSONDecodeError):
                continue
            with self.lock:
                self.msgs.append(msg)

    def pop_all(self) -> list[tuple[float, float]]:
        with self.lock:
            out, self.msgs = self.msgs, []
        return out


def parse_walk(s: str) -> np.ndarray:
    """'0:0,40:0,130:90' -> (n, 2) of (sim time s, bearing deg), piecewise linear; silence after the last."""
    return np.array([[float(v) for v in p.split(":")] for p in s.split(",")])


def walk_bearing(walk: np.ndarray, t: float) -> float | None:
    if t < walk[0, 0] or t > walk[-1, 0]:
        return None
    return float(np.interp(t, walk[:, 0], np.unwrap(walk[:, 1], period=360.0))) % 360.0


# -------------------------------------------------------------------------------------------------- world
class Demo:
    def __init__(self, m, d, info, args, seed: int = 0):
        self.m, self.d, self.info, self.args = m, d, info, args
        self.rng = np.random.default_rng(seed)
        self.ctrl = base.Controller(m)
        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)
        a, g = self.args, self.info.ground
        self.audio, self.hunter = Audio(), Hunter(g)
        self.intruder = Intruder(a.range, a.intruder_agl, a.intruder_speed, g)
        self.cam = Camera(self.m, self.d, self.rng)
        self.next_frame = 0.0
        self.trail: list[np.ndarray] = []
        self.track_err: list[float] = []

    def hear(self, az: float, sigma: float) -> None:
        t = self.d.time
        self.audio.update(az, sigma, t)
        self.intruder.hear(az, t)

    def step(self) -> None:
        m, d, h = self.m, self.d, self.hunter
        dt = m.opt.timestep
        self.intruder.update(d.time, dt)
        h.aim(self.cam, self.audio, d.time)
        if d.time >= self.next_frame:
            self.next_frame = d.time + FRAME_DT
            r = self.cam.frame(self.intruder.pos if h.state not in ("WAIT", "LANDED") else None)
            h.on_frame(r, d.time)
            if h.state == "TRACK" and self.intruder.pos is not None:
                self.track_err.append(float(np.linalg.norm(d.qpos[0:3] - self.intruder.pos)))
        pos, vel, R, om = base.state_of(d)
        h.update(pos, d.time, dt, self.audio)
        if h.state in ("WAIT", "LANDED"):
            d.ctrl[:] = 0.0
        else:
            d.ctrl[:] = self.ctrl(pos, vel, R, om, h.sp, h.v, h.yaw)
        mujoco.mj_step(m, d)
        if not self.trail or np.linalg.norm(d.qpos[0:3] - self.trail[-1]) > 4:
            self.trail = (self.trail + [d.qpos[0:3].copy()])[-500:]

    # -- drawing ----------------------------------------------------------------------------------------
    def draw(self, scn, eye) -> None:
        scn.ngeom = 0
        g, a, h, it = self.info.ground, self.audio, self.hunter, self.intruder
        t = self.d.time
        base.add_compass(scn, g)
        add_label(scn, [0, 0, 4.6], "UMA-16")
        if a.az is not None:
            hearing = a.hearing(t)
            length = max(it.range_m + 200.0, 700.0)
            w = float(np.clip(0.0012 * np.linalg.norm(eye), 0.05, 0.8))
            col = [1.0, 0.45, 0.1, 0.9] if hearing else [0.6, 0.6, 0.6, 0.5]
            for s, width, c in ((0.0, w, col), (-2 * a.sigma, w / 3, [1.0, 0.6, 0.2, 0.35]),
                                (2 * a.sigma, w / 3, [1.0, 0.6, 0.2, 0.35])):
                pts = base.terrain_path(g, a.az + s, length, 25.0 if s == 0 else 50.0, 4.0)
                pts[0] = [0, 0, 3.3]
                for p0, p1 in zip(pts[:-1], pts[1:]):
                    add_line(scn, p0, p1, width, c)
            for tk, az in a.history:   # raw bearings of the last 30 s, fading, on a 150 m ring
                k = float(np.clip(1 - (t - tk) / 30.0, 0, 1))
                if k > 0:
                    xy = az_to_dir(az) * 150.0
                    add_sphere(scn, [*xy, float(g(*xy)) + 2.0], 1.2, [1.0, 0.5, 0.1, 0.8 * k])
        if it.pos is not None:
            draw_quad(scn, it.pos, it.yaw, t)
            pin(scn, it.pos, eye, [0.9, 0.1, 0.1, 0.9], f"INTRUDER  {it.range_from_base():.0f} m (unknown to X2)", g)
        p = self.d.qpos[0:3].copy()
        pin(scn, p, eye, [1.0, 0.75, 0.0, 0.85], "X2", g)
        for p0, p1 in zip(self.trail[:-1], self.trail[1:]):
            add_line(scn, p0, p1, 0.25, [0.2, 0.9, 1.0, 0.8])
        if h.state not in ("WAIT", "LANDED"):
            locked = h.state == "TRACK"
            c = self.cam.pos()
            fwd, right, up = cam_axes(self.cam.yaw, self.cam.pitch)
            L = float(self.cam.last.get("dist", 300.0)) if locked else 300.0
            th, tv = math.tan(math.radians(FOV_H / 2)), math.tan(math.radians(FOV_V / 2))
            col = [0.2, 1.0, 0.3, 0.7] if locked else [1.0, 1.0, 1.0, 0.35]
            for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
                add_line(scn, c, c + L * (fwd + sx * th * right + sy * tv * up), 0.15, col)
            if locked and h.est is not None:
                add_sphere(scn, h.est, 3.0, [0.2, 1.0, 0.3, 0.4])

    def hud(self) -> str:
        a, h, it, r = self.audio, self.hunter, self.intruder, self.cam.last
        t = self.d.time
        hear = (f"{a.az:5.1f} deg +- {a.sigma:.1f}  ({a.rate:+.1f} deg/s)" if a.hearing(t)
                else "-" if a.az is None else f"lost {t - a.t_last:.0f} s ago (last {a.az:.0f} deg)")
        p = self.d.qpos[0:3]
        lines = [f"State: {h.state}", f"Sound bearing: {hear}",
                 f"X2: {np.linalg.norm(p[:2]):.0f} m from base, {p[2] - self.info.ground(p[0], p[1]):.0f} m AGL"]
        if r and h.state not in ("WAIT", "LANDED"):
            view = "OCCLUDED" if r["occluded"] else "in view" if r["in_view"] else "not in view"
            lines.append(f"Camera: intruder {view}, {r['px']:.1f} px, {r.get('dist', 0):.0f} m")
        if h.t_found is not None:
            lines.append(f"FOUND {h.t_found - h.t_launch:.0f} s after launch")
        if h.state == "TRACK" and h.est is not None:
            lines.append(f"Target estimate: {np.linalg.norm(h.est[:2]):.0f} m from base, "
                         f"bearing {(90 - math.degrees(math.atan2(h.est[1], h.est[0]))) % 360:.0f} deg")
        return "\n".join(lines)


def _geom(scn, gtype, size, pos, mat, rgba) -> None:
    g = base._new_geom(scn)
    if g is not None:
        mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float), np.asarray(mat, float).ravel(),
                            np.asarray(rgba, float))


def draw_quad(scn, p, yaw: float, t: float) -> None:
    """The intruder: body, two crossed arms, four rotor discs (1.2 m motor to motor)."""
    def rz(a):
        return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    _geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [0.16, 0.12, 0.06], p, rz(yaw), [0.55, 0.05, 0.05, 1])
    for k in range(4):
        a = yaw + math.pi / 4 + k * math.pi / 2
        u = np.array([math.cos(a), math.sin(a), 0.0])
        _geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [0.3, 0.025, 0.02], p + 0.3 * u, rz(a), [0.15, 0.15, 0.15, 1])
        _geom(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [0.17, 0.008, 0], p + 0.6 * u + [0, 0, 0.05], np.eye(3),
              [0.8, 0.1, 0.1, 0.55])


def pin(scn, p, eye, rgba, text, ground) -> None:
    """A ball on a stem above a far object, sized with the viewer's distance (same look as the X2 pin)."""
    dist = float(np.linalg.norm(eye - p))
    if dist <= 40:
        return
    r = min(0.016 * dist, 16.0)
    top = p + [0, 0, 1.0 + 3 * r]
    add_line(scn, p + [0, 0, 0.3], top, 0.05 * r, rgba)
    add_sphere(scn, top, r, rgba)
    add_label(scn, top + [0, 0, 1.4 * r], text)


# ---------------------------------------------------------------------------------------------- camera view
class CameraView:
    """Offscreen render of the gimbal camera (zoomed to FOV_V), box + text overlay, for the viewer corner."""

    def __init__(self, m, info):
        import cv2  # noqa: PLC0415
        self.cv2, self.info = cv2, info
        self.r = mujoco.Renderer(m, CAM_H, CAM_W, max_geom=20000)
        self.scale = math.tan(math.radians(FOV_V / 2)) / math.tan(math.radians(m.vis.global_.fovy / 2))
        self.vcam = mujoco.MjvCamera()
        self.vcam.type = mujoco.mjtCamera.mjCAMERA_FREE

    def render(self, demo: Demo) -> np.ndarray:
        cam, d, cv2 = demo.cam, demo.d, self.cv2
        fwd = cam_axes(cam.yaw, cam.pitch)[0]
        self.vcam.lookat[:] = cam.pos() + 10.0 * fwd
        self.vcam.distance = 10.0
        self.vcam.azimuth, self.vcam.elevation = math.degrees(cam.yaw), math.degrees(cam.pitch)
        self.r.update_scene(d, self.vcam)
        scn = self.r.scene
        scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = self.info.name == "farm"
        for gc in scn.camera:
            gc.frustum_top *= self.scale
            gc.frustum_bottom *= self.scale
            gc.frustum_width *= self.scale
        if demo.intruder.pos is not None:
            draw_quad(scn, demo.intruder.pos, demo.intruder.yaw, d.time)
        img = np.ascontiguousarray(self.r.render())
        r, h = cam.last, demo.hunter
        if r.get("in_view") and not r.get("occluded"):
            th, tv = math.tan(math.radians(FOV_H / 2)), math.tan(math.radians(FOV_V / 2))
            x = int(CAM_W / 2 + math.tan(math.radians(r["ah"])) / th * CAM_W / 2)
            y = int(CAM_H / 2 - math.tan(math.radians(r["av"])) / tv * CAM_H / 2)
            s = max(int(r["px"]), 8)
            col = (40, 255, 80) if h.state == "TRACK" else (255, 220, 0) if r["hit"] else (200, 200, 200)
            cv2.rectangle(img, (x - s, y - s), (x + s, y + s), col, 2 if h.state == "TRACK" else 1)
        cv2.drawMarker(img, (CAM_W // 2, CAM_H // 2), (255, 255, 255), cv2.MARKER_CROSS, 14, 1)
        label = "LOCK" if h.state == "TRACK" else "SEARCH" if h.state in ("INTERCEPT", "LOST") else h.state
        cv2.putText(img, f"X2 CAM {FOV_H:.0f} deg  {label}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (CAM_W - 1, CAM_H - 1), (255, 255, 255), 1)
        return img


# -------------------------------------------------------------------------------------------------- run
def run_viewer(m, d, info, args) -> None:
    demo, udp = Demo(m, d, info, args), UdpTrack(args.udp_port)
    keys: list[int] = []
    speed, chase, show_cam = 1, False, not args.no_camera_view
    manual_az, manual_on, manual_next = 0.0, False, 0.0
    view = None
    if show_cam:
        try:
            view = CameraView(m, info)
        except Exception as e:  # noqa: BLE001  (no offscreen GL next to the viewer: run without the picture)
            print(f"camera picture off: {e}")
    last_log, next_pic = "", 0.0
    with mujoco.viewer.launch_passive(m, d, key_callback=keys.append) as viewer:
        base.set_overview(viewer.cam, info)
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = info.name == "farm"
        while viewer.is_running():
            t0 = time.time()
            while keys:
                k = keys.pop(0)
                if k in (262, 263):
                    manual_az = (manual_az + (5 if k == 262 else -5)) % 360 if manual_on else manual_az
                    manual_on = True
                elif k == ord("S"):
                    manual_on = False
                elif k in (ord("1"), ord("2"), ord("4")):
                    speed = int(chr(k))
                elif k == ord("C"):
                    chase = not chase
                    base.set_chase(viewer.cam, m) if chase else base.set_overview(viewer.cam, info)
                elif k == ord("V"):
                    show_cam = not show_cam
                elif k == 259:
                    with viewer.lock():
                        demo.reset()
            with viewer.lock():
                for az, sigma in udp.pop_all():
                    demo.hear(az, sigma)
                for _ in range(int(speed / 60 / m.opt.timestep)):
                    if manual_on and d.time >= manual_next:
                        manual_next = d.time + MSG_DT
                        demo.hear(manual_az + demo.rng.normal(0, 2.0), 5.0)
                    demo.step()
                demo.draw(viewer.user_scn, base.cam_pos(viewer.cam))
                hud = demo.hud()
            # set_texts / set_images take the viewer's lock themselves: never inside `with viewer.lock()`
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                              "UMA-16 counter-drone", hud))
            if view is not None and time.time() >= next_pic:
                next_pic = time.time() + 0.1
                img = view.render(demo) if show_cam else np.zeros((1, 1, 3), np.uint8)
                vp = viewer.viewport
                rect = mujoco.MjrRect(vp.left + vp.width - CAM_W - 12, vp.bottom + 12, CAM_W, CAM_H)
                viewer.set_images([(rect, img)] if show_cam else [])
            log = demo.hunter.log[-1] if demo.hunter.log else ""
            if log != last_log:
                last_log = log
                print(log, flush=True)
            viewer.sync()
            time.sleep(max(0.0, 1 / 60 - (time.time() - t0)))


def run_headless(m, d, info, args, seed: int = 0, quiet: bool = False) -> dict:
    """Scripted walk (--walk) at 5 Hz with 2 deg noise; returns the result (time to find, tracking, landing)."""
    demo = Demo(m, d, info, args, seed)
    walk = parse_walk(args.walk)
    nxt_msg, nxt_print = 0.0, 0.0
    while d.time < args.max_t and demo.hunter.state != "LANDED":
        if d.time >= nxt_msg:
            nxt_msg = d.time + MSG_DT
            az = walk_bearing(walk, d.time)
            if az is not None:
                demo.hear(az + demo.rng.normal(0, 2.0), 5.0)
        demo.step()
        if not quiet and d.time >= nxt_print:
            nxt_print += 10.0
            p, it, r = d.qpos[0:3], demo.intruder, demo.cam.last
            gap = f"{np.linalg.norm(p - it.pos):6.0f}" if it.pos is not None else "     -"
            print(f"t={d.time:6.1f}  {demo.hunter.state:9s}  X2 {np.linalg.norm(p[:2]):6.0f} m out  "
                  f"gap to intruder {gap} m  cam in_view={r.get('in_view')} px={r.get('px', 0):.1f}")
    h = demo.hunter
    res = {"found": h.t_found is not None,
           "t_find_s": round(h.t_found - h.t_launch, 1) if h.t_found is not None else None,
           "track_dist_median_m": round(float(np.median(demo.track_err)), 1) if demo.track_err else None,
           "state": h.state, "t": round(d.time, 1),
           "landing_err_m": round(float(np.linalg.norm(d.qpos[:2] - PAD)), 2)}
    if not quiet:
        print("\n".join(h.log))
        print("RESULT", json.dumps(res))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", choices=["farm", "simple"], default="farm")
    ap.add_argument("--range", type=float, default=500.0, help="intruder distance from the base (m)")
    ap.add_argument("--intruder-agl", type=float, default=80.0, help="intruder height above the terrain (m)")
    ap.add_argument("--intruder-speed", type=float, default=20.0, help="max intruder speed (m/s)")
    ap.add_argument("--udp-port", type=int, default=5005, help="0 = off")
    ap.add_argument("--no-camera-view", action="store_true", help="no camera picture in the viewer corner")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--walk", default="0:0,40:0,130:90,200:90",
                    help="headless: t:bearing points (sim s : deg), linear in between, silence after the last")
    ap.add_argument("--max-t", type=float, default=600.0)
    args = ap.parse_args()
    base.CLIMB = 4.0
    t0 = time.time()
    m, info = base.build_model(args.world)
    print(f"world {info.name}: built in {time.time() - t0:.1f}s; intruder {args.range:.0f} m out, "
          f"{args.intruder_agl:.0f} m AGL")
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    if args.headless:
        res = run_headless(m, d, info, args)
        raise SystemExit(0 if res["found"] and res["state"] == "LANDED" else 1)
    run_viewer(m, d, info, args)


if __name__ == "__main__":
    main()
