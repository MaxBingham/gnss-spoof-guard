"""Counter-drone demo: the UMA-16 hears a drone, our X2 flies out along the bearing and finds it with a camera.

    uv run python demos/51_counter_drone.py                        # viewer; bearings over UDP 5005 or the keys
    uv run python demos/51_counter_drone.py --headless --walk 0:0,40:0,130:90   # scripted walk, print the result
    uv run python scripts/fake_walk.py --from 0 --to 90 --rate 1   # a fake speaker walk over UDP, no array needed
    uv run python demos/51_counter_drone.py --headless --walk-noise 6 --outliers 0.05 --dropouts 2:20   # messy
    uv run python demos/51_counter_drone.py --replay track.jsonl [--headless]   # a recorded listener / clap log

Stage -> sim: a person walks around the array with a speaker playing drone sound. The INTRUDER (red quad) sits
`--range` m out (default 500; `--range-from-level`: louder = closer, 150-1200 m) and `--intruder-agl` m above the
terrain, always exactly on the drawn sound line, and it closes on the base along that line
(`--approach`, 15 m/s, never inside `--r-min`). With `--intruder-free` it flies after the line instead,
speed-limited (`--intruder-speed`). At 500 m one degree of walking is 8.7 m, so walk about 1 deg/s.
Silent: it hovers where it is.

Audio in (UDP 5005, the 45 listener's contract): BearingEstimate JSON + az_rate_dps, level_db, seq, detected=true;
one {"detected": false} on loss (= silent at once); plain BearingEstimate JSON (the clap listener) and bare numbers
too. Out-of-order seq is dropped. Single bearings > max(3 sigma, 30 deg) off the smoothed bearing are gated unless
they persist 1 s (then the bearing jumps there).

Our drone knows only what the array sends (bearing + sigma, about 5 Hz), not the range:
  WAIT -> TAKEOFF (sound heard for 1.5 s) -> INTERCEPT: fly out along the audio ray at 15 m/s, leading the
  bearing rate, the gimbal camera looking along the ray. At ~100 mph. -> STRIKE: the camera confirmed the
  intruder (3 hits in 2 s). From here the run-in is the camera only: fly straight at the latest image
  estimate, and a silent array does not abort it. STRUCK inside 8 m, then home.
  LOST (no camera hit for 3 s): back onto the audio ray at the range the camera measured.
  Short dropouts: keep flying the last bearing + rate (extrapolated up to 3 s).
  No sound for 15 s while still searching, or 1500 m out (never further) -> RETURN -> DESCEND -> LANDED.

Camera (a simulated detector, not a neural net): 20 x 15 deg zoom on a gimbal, 5 frames/s. A frame detects the
intruder when it is inside the field of view, a ray from the camera reaches it (terrain, trees and buildings
occlude), and then with probability 0 at 3 px rising to 0.9 at 8 px of apparent size (1.2 m span: 8 px at
~280 m). Range from the apparent size (+-5 %), direction from the pixel (+-0.3 deg).

Keys: LEFT / RIGHT move a manual sound bearing by 5 deg and count as hearing it (no array needed), S silences the
manual sound, C overview / chase camera (during the intercept this lets go of the automatic close-up), V hides
the camera picture, 1 / 2 / 4 sim speed, BACKSPACE resets. On camera lock the view frames both drones until
a few seconds after the hit.
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
HUNT_SPEED = 45.0          # ~100 mph: a fast quad can do this; the run-in uses the same speed
STRIKE_M = 8.0             # camera-guided run-in counts as a strike inside this (m)
INTERCEPT_HOLD_S = 6.0     # keep the close-up this long after the hit
APPROACH = 15.0            # intruder closes on the base along the sound line (m/s)
R_MIN = 50.0               # it never comes closer than this
STANDOFF = 40.0
MAX_OUT = 1500.0
LEAD_S = 4.0               # aim where the bearing will be in this many seconds
FOV_H, FOV_V = 20.0, 15.0
CAM_W, CAM_H = 640, 480
PIC_W, PIC_H = 1280, 960     # the camera picture in the viewer corner (render size)
PIC_FRAC = 0.34            # ... drawn this fraction of the window width
BANNER_FRAC = 0.5          # the state banner on top: this fraction of the window width
FRAME_DT = 0.2
SPAN = 1.2                 # intruder size for the pixel count (m)
PX_MIN, PX_FULL, P_MAX = 3.0, 8.0, 0.9
CONFIRM_HITS, CONFIRM_S, LOST_S = 3, 2.0, 3.0
HEAR_S = 1.5               # launch after hearing the sound this long
AUDIO_TIMEOUT = 3.0        # no bearing for this long = not hearing
GIVE_UP_S = 15.0
MSG_DT = 0.2               # manual / scripted bearings at 5 Hz
OUTLIER_DEG = 30.0         # gate: a bearing > max(3 sigma, this) off the smoothed one is an outlier ...
OUTLIER_PERSIST_S = 1.0    # ... unless such bearings agree with each other for this long (then jump there)
RATE_MAX = 10.0            # deg/s, clip on the bearing rate
EXTRAP_S = 3.0             # in a dropout, extrapolate the bearing with the rate for at most this long
LINE_TAU = 0.6             # the drawn sound line glides after the smoothed bearing with this time constant
INTRUDER_ACC = 4.0         # m/s^2
INTRUDER_K = 0.2           # 1/s, intruder velocity = K * distance to its goal + bearing-rate feed-forward
INTRUDER_AB = (0.06, 0.0019)  # alpha-beta gains on the 5 Hz bearings (critically damped, ~3 s)


def adiff(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def blend_deg(a: float, b: float, k: float) -> float:
    return (a + k * adiff(b, a)) % 360.0


def circmean_deg(angles) -> float:
    r = np.radians(np.asarray(angles, float))
    return math.degrees(math.atan2(float(np.mean(np.sin(r))), float(np.mean(np.cos(r))))) % 360.0


def _num(v) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def parse_message(txt: str) -> dict | None:
    """One listener message -> {"detected": True, "az", "sigma", "rate", "level_db", "confidence", "el", "seq", "t"}
    or {"detected": False, "seq", "t"}; None if unusable. Accepts the 45 listener's contract, plain BearingEstimate
    JSON (clap listener), clap log lines ("compass_az") and a bare number."""
    txt = txt.strip()
    if not txt:
        return None
    if not txt.startswith("{"):
        az = _num(txt.split()[0])
        return None if az is None else {"detected": True, "az": az % 360.0, "sigma": 5.0, "rate": None,
                                        "level_db": None, "confidence": None, "el": None, "seq": None, "t": None}
    try:
        j = json.loads(txt)
    except json.JSONDecodeError:
        return None
    if isinstance(j, dict) and "msg" in j:   # a 45_uma16_track track.jsonl line: the sent message is nested
        j = j["msg"]
    if not isinstance(j, dict):
        return None
    seq = _num(j.get("seq"))
    seq = int(seq) if seq is not None else None
    if j.get("detected", True) in (False, "false", "False"):
        return {"detected": False, "seq": seq, "t": _num(j.get("t"))}
    az = _num(j.get("az_deg", j.get("compass_az")))
    if az is None:
        return None
    sigma = _num(j.get("sigma_az_deg"))
    return {"detected": True, "az": az % 360.0, "sigma": sigma if sigma is not None and sigma > 0 else 5.0,
            "rate": _num(j.get("az_rate_dps")), "level_db": _num(j.get("level_db")),
            "confidence": _num(j.get("confidence")), "el": _num(j.get("el_deg", j.get("el"))), "seq": seq,
            "t": _num(j.get("t"))}


# ------------------------------------------------------------------------------------------- what we hear
class Audio:
    """The array's bearings as our drone sees them: smoothed bearing (0.5-1.5 s, longer for a larger sigma),
    outlier gate, bearing rate (own differences blended with the listener's az_rate_dps), sigma, level.
    HEARING = a bearing within AUDIO_TIMEOUT and no loss message since; else SILENT."""

    def __init__(self):
        self.az: float | None = None
        self.line: float | None = None
        self.raw: float | None = None
        self.sigma, self.rate = 5.0, 0.0
        self.level_db: float | None = None
        self.confidence: float | None = None
        self.source = "-"
        self.t_last = -1e9          # last accepted bearing
        self.t_msg = -1e9           # last bearing message, accepted or gated
        self.t_first = 0.0          # start of the current hearing stretch
        self.lost_msg = False       # the listener said detected=false
        self.pending: list[tuple[float, float]] = []
        self.n_msgs = self.n_gated = self.n_jumps = self.n_lost = 0
        self.arrivals: deque[float] = deque(maxlen=100)
        self.history: deque[tuple[float, float, bool]] = deque(maxlen=150)

    def update(self, az: float, sigma: float, t: float, rate: float | None = None, level_db: float | None = None,
               confidence: float | None = None, source: str | None = None) -> str:
        """One bearing -> "new" (nothing heard for GIVE_UP_S), "resume" (back after a dropout, near the extrapolated
        bearing), "ok", "jump" (a persistent new bearing) or "gated" (outlier)."""
        az, sigma = az % 360.0, max(float(sigma), 2.0)
        self.n_msgs += 1
        self.arrivals.append(t)
        self.source = source or self.source
        self.raw = az
        if level_db is not None:
            self.level_db = level_db
        if confidence is not None:
            self.confidence = confidence
        thr = max(3.0 * sigma, OUTLIER_DEG)
        resumed = not self.hearing(t)
        msg_rate = float(np.clip(rate, -RATE_MAX, RATE_MAX)) if rate is not None else None
        if resumed and (self.az is None or t - self.t_last > GIVE_UP_S):
            self.az, self.sigma, self.t_first, self.lost_msg = az, sigma, t, False
            self.rate = msg_rate if msg_rate is not None else 0.0
            self.pending.clear()
            status = "new"
        elif abs(adiff(az, self.predict(t) if resumed else self.az)) > thr:
            self.pending = [p for p in self.pending if t - p[0] <= OUTLIER_PERSIST_S + 2.0] + [(t, az)]
            near = [p for p in self.pending if abs(adiff(p[1], az)) <= thr]
            if len(near) >= 2 and t - near[0][0] >= OUTLIER_PERSIST_S - 0.05:
                self.az = circmean_deg([p[1] for p in near[-5:]])
                self.rate = msg_rate if msg_rate is not None else 0.0
                self.pending.clear()
                self.n_jumps += 1
                if resumed:
                    self.t_first, self.lost_msg = t, False
                status = "jump"
            else:
                self.n_gated += 1
                status = "gated"
        elif resumed:
            self.az = blend_deg(self.predict(t), az, 0.5)
            self.t_first, self.lost_msg = t, False
            self.pending.clear()
            status = "resume"
        else:
            self.pending.clear()
            dt = max(t - self.t_last, 1e-3)
            new = blend_deg(self.az, az, 1 - math.exp(-dt / float(np.clip(0.1 * sigma, 0.5, 1.5))))
            self.rate += (1 - math.exp(-dt / 3.0)) * (adiff(new, self.az) / dt - self.rate)
            if rate is not None:
                self.rate += (1 - math.exp(-dt / 1.0)) * (float(np.clip(rate, -RATE_MAX, RATE_MAX)) - self.rate)
            self.rate = float(np.clip(self.rate, -RATE_MAX, RATE_MAX))
            self.az = new
            self.sigma += (1 - math.exp(-dt / 1.0)) * (sigma - self.sigma)
            status = "ok"
        self.t_msg = t
        if status != "gated":
            self.t_last = t
        self.history.append((t, az, status != "gated"))
        return status

    def lose(self, t: float, source: str | None = None) -> None:
        """The listener's loss message: silent now, not after AUDIO_TIMEOUT."""
        self.lost_msg = True
        self.n_lost += 1
        self.arrivals.append(t)
        self.source = source or self.source
        self.pending.clear()

    def hearing(self, t: float) -> bool:
        return self.az is not None and not self.lost_msg and t - self.t_last <= AUDIO_TIMEOUT

    def heard_for(self, t: float) -> float:
        return t - self.t_first if self.hearing(t) else 0.0

    def silent_for(self, t: float) -> float:
        """0 while hearing, else seconds since the last accepted bearing (inf if never heard)."""
        if self.hearing(t):
            return 0.0
        return t - self.t_last if self.az is not None else math.inf

    def advance(self, t: float, dt: float) -> None:
        """The drawn sound line (and where the intruder sits): the smoothed bearing glided every sim step, so it
        moves continuously instead of stepping with each 5 Hz message."""
        if self.az is None:
            return
        if self.line is None:
            self.line = self.az
        else:
            self.line = blend_deg(self.line, self.predict(t), 1 - math.exp(-dt / LINE_TAU))

    def predict(self, t: float) -> float:
        """Smoothed bearing carried forward with the rate over a dropout (at most EXTRAP_S)."""
        return (self.az + self.rate * float(np.clip(t - self.t_last, 0.0, EXTRAP_S))) % 360.0

    def msg_hz(self, t: float, window: float = 3.0) -> float:
        n = sum(1 for ta in self.arrivals if t - ta <= window)
        return n / window

    def status(self, t: float) -> str:
        if self.az is None:
            return "SILENT (nothing heard yet)"
        if self.hearing(t):
            return "HEARING"
        return f"SILENT {self.silent_for(t):.0f} s" + ("  (loss msg)" if self.lost_msg else "")


# ---------------------------------------------------------------------------------------------- intruder
class Intruder:
    """Flies `range_m` out on the (smoothed, 1.5 s) sound bearing, `agl` m up: velocity towards that goal,
    at most `speed` m/s, accelerations <= INTRUDER_ACC (no teleports, no jitter); hovers while silent. After a
    silence or a bearing jump it heads straight for the new bearing (max speed when far)."""

    def __init__(self, range_m: float, agl: float, speed: float, ground, range_from_level: bool = False,
                 approach: float = 0.0, r_min: float = R_MIN):
        self.range_m, self.agl, self.speed, self.ground = range_m, agl, speed, ground
        self.range_from_level = range_from_level
        self.approach, self.r_min = approach, r_min
        self.r_cur = range_m
        self.level_ref: float | None = None
        self.az: float | None = None
        self.rate = 0.0
        self.t_msg = 0.0
        self.pos: np.ndarray | None = None
        self.vel = np.zeros(2)
        self.v_max = 0.0
        self.yaw = 0.0

    def hear(self, az: float, t: float, fresh: bool = False, level_db: float | None = None) -> None:
        dt = float(np.clip(t - self.t_msg, 1e-3, EXTRAP_S))
        if self.az is None or fresh:
            self.az, self.rate = az % 360.0, 0.0
        else:   # alpha-beta filter on bearing + rate: smooth, without the lag of a plain low-pass while walking
            pred = self.az + self.rate * dt
            r = adiff(az, pred)
            self.az = (pred + INTRUDER_AB[0] * r) % 360.0
            self.rate = float(np.clip(self.rate + INTRUDER_AB[1] * r / dt, -RATE_MAX, RATE_MAX))
        if self.range_from_level and level_db is not None:
            if self.level_ref is None:   # the first level heard is the one at --range
                self.level_ref = level_db
            r = float(np.clip(self.range_m * 10 ** ((self.level_ref - level_db) / 20.0), 150.0, 1200.0))
            self.r_cur += (1 - math.exp(-dt / 3.0)) * (r - self.r_cur)
        self.t_msg = t
        if self.pos is None:
            xy = az_to_dir(self.az) * self.r_cur
            self.pos = np.array([*xy, self.ground(*xy) + self.agl])

    def update(self, t: float, dt: float, hearing: bool = True, line_az: float | None = None) -> None:
        """line_az given: sit exactly on the drawn sound line (the bearing our drone hears), no lag of its own."""
        if self.pos is None:
            return
        if hearing and self.approach > 0.0 and not self.range_from_level:
            self.r_cur = max(self.r_min, self.r_cur - self.approach * dt)
        if line_az is not None:
            xy = az_to_dir(line_az) * self.r_cur
            step = xy - self.pos[:2]
            sp = float(np.linalg.norm(step)) / dt
            self.vel = step / dt
            self.v_max = max(self.v_max, min(sp, 1e3))
            if sp > 0.5:
                self.yaw = math.atan2(step[1], step[0])
            self.pos[:2] = xy
            self.pos[2] = float(self.ground(*xy)) + self.agl + 0.6 * math.sin(0.7 * t)
            return
        v_des = np.zeros(2)
        if hearing:
            a = self.az + self.rate * float(np.clip(t - self.t_msg, 0.0, EXTRAP_S))
            tangent = np.array([math.cos(math.radians(a)), -math.sin(math.radians(a))])
            v_des = (INTRUDER_K * (az_to_dir(a) * self.r_cur - self.pos[:2])
                     + math.radians(self.rate) * self.r_cur * tangent)
            n = float(np.linalg.norm(v_des))
            if n > self.speed:
                v_des *= self.speed / n
        dv = v_des - self.vel
        n = float(np.linalg.norm(dv))
        if n > INTRUDER_ACC * dt:
            dv *= INTRUDER_ACC * dt / n
        self.vel += dv
        self.pos[:2] += self.vel * dt
        sp = float(np.linalg.norm(self.vel))
        self.v_max = max(self.v_max, sp)
        if sp > 0.5:
            self.yaw = math.atan2(self.vel[1], self.vel[0])
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
    """Our X2: WAIT -> TAKEOFF -> INTERCEPT -> STRIKE (camera lock) -> RETURN -> DESCEND -> LANDED.
    LOST falls back to the audio ray at the range the camera measured."""

    def __init__(self, ground, agl: float = HUNT_AGL):
        super().__init__(MAX_OUT, ground)
        self.agl = agl
        self.hits: deque[float] = deque()
        self.t_hit = -1e9
        self.est: np.ndarray | None = None
        self.r_est: float | None = None   # target range from the base, known once the camera has confirmed it
        self.struck = False
        self.t_struck: float | None = None
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
        if len(self.hits) >= CONFIRM_HITS:
            self.r_est = float(np.linalg.norm(r["est"][:2]))
        if self.state in ("INTERCEPT", "LOST") and len(self.hits) >= CONFIRM_HITS:
            if self.t_found is None:
                self.t_found = t
            self._go("STRIKE", t)

    def aim(self, cam: Camera, audio: Audio, t: float) -> None:
        c = cam.pos()
        if self.state == "STRIKE" and self.est is not None:
            v = self.est - c
        elif audio.az is not None and self.r_est is not None and self.est is not None:
            p = az_to_dir(audio.predict(t)) * self.r_est   # lost: on the sound line at the range we saw it at
            v = np.array([*(p - c[:2]), float(self.est[2]) - c[2]])
        elif audio.az is not None:
            r_look = float(np.linalg.norm(c[:2])) + 250.0
            p = az_to_dir(audio.predict(t)) * r_look
            v = np.array([*(p - c[:2]), float(self.ground(*p)) + self.agl + 20.0 - c[2]])
        else:
            return
        tri = lambda period: 2 * abs((t / period) % 1.0 - 0.5) * 2 - 1    # noqa: E731  (-1..1 triangle)
        sweep_y = 0.0 if self.state == "STRIKE" else float(np.clip(2 * audio.sigma - FOV_H / 2, 0, 30)) * tri(3.0)
        sweep_p = 0.0 if self.state == "STRIKE" else 5.0 * tri(4.0)
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
            r_sp = float(np.linalg.norm(self.sp[:2]))
            if s == "LOST" and t - self.t_hit > LOST_S + 10.0:
                self._go("INTERCEPT", t)
            if audio.silent_for(t) > GIVE_UP_S or r_sp >= MAX_OUT - 1.0:
                self._go("RETURN", t)
                return
            az = audio.predict(t) + float(np.clip(audio.rate, -5, 5)) * LEAD_S   # dropouts: last bearing + rate
            r_goal = min(r_sp + 250.0, MAX_OUT) if self.r_est is None else min(self.r_est, MAX_OUT)
            self._move_xy(az_to_dir(az) * r_goal, HUNT_SPEED, dt)
            self._terrain(dt, self.agl)
        elif s == "STRIKE":
            # The camera has the intruder: steer on the image estimate only. Sound does not abort this.
            if t - self.t_hit > LOST_S:
                self._go("LOST", t)
                return
            if float(np.linalg.norm(pos - self.est)) < STRIKE_M:
                self.struck = True
                self.t_struck = t
                self._go("RETURN", t)
                return
            self._move_xy(self.est[:2], HUNT_SPEED, dt)
            self.yaw = math.atan2(self.est[1] - self.sp[1], self.est[0] - self.sp[0])
            self._move_z(float(self.est[2]), dt)
        else:
            super().update(pos, t, dt)


# ------------------------------------------------------------------------------------------------- input
class UdpTrack:
    """Listener messages over UDP (see `parse_message`), in arrival order; out-of-order `seq` dropped (a seq far
    below the last one = the listener restarted, accepted). `push` is the socket-free entry (tests)."""

    SEQ_WINDOW = 50

    def __init__(self, port: int):
        self.msgs: list[dict] = []
        self.lock = threading.Lock()
        self.last_seq: int | None = None
        self.n_bad = self.n_out_of_order = 0
        if port <= 0:
            return
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(("0.0.0.0", port))
        except OSError as e:
            print(f"UDP {port} unavailable ({e}): no live bearings")
            return
        threading.Thread(target=self._run, daemon=True).start()
        print(f"listening for bearings on UDP {port}")

    def _run(self) -> None:
        while True:
            self.push(self.sock.recvfrom(4096)[0].decode("utf-8", "replace"))

    def push(self, txt: str) -> bool:
        msg = parse_message(txt)
        with self.lock:
            if msg is None:
                self.n_bad += 1
                return False
            seq = msg["seq"]
            if seq is not None and self.last_seq is not None and self.last_seq - self.SEQ_WINDOW < seq <= self.last_seq:
                self.n_out_of_order += 1
                return False
            if seq is not None:
                self.last_seq = seq
            self.msgs.append(msg)
        return True

    def pop_all(self) -> list[dict]:
        with self.lock:
            out, self.msgs = self.msgs, []
        return out


class Feed:
    """Timed messages [(sim t, parse_message dict)] handed out as the sim clock passes them (walk, replay)."""

    def __init__(self, items: list[tuple[float, dict]], source: str):
        self.items, self.source, self.i = sorted(items, key=lambda it: it[0]), source, 0
        self.t_end = self.items[-1][0] if self.items else 0.0

    def due(self, t: float) -> list[dict]:
        out = []
        while self.i < len(self.items) and self.items[self.i][0] <= t:
            out.append(self.items[self.i][1])
            self.i += 1
        return out

    def done(self) -> bool:
        return self.i >= len(self.items)

    def reset(self) -> None:
        self.i = 0


def load_replay(path: str | Path, start: float = 0.0) -> list[tuple[float, dict]]:
    """A recorded listener log (one message per line: the UDP contract, BearingEstimate JSON or a clap log line
    with "compass_az") -> timed messages, the first at sim time `start`. Lines without "t" follow 0.2 s apart."""
    msgs = [m for m in (parse_message(line) for line in Path(path).read_text(encoding="utf-8").splitlines()) if m]
    t0 = next((m["t"] for m in msgs if m["t"] is not None), 0.0)
    out, t_prev = [], t0 - MSG_DT
    for m in msgs:
        t_prev = m["t"] if m["t"] is not None else t_prev + MSG_DT
        out.append((start + t_prev - t0, m))
    return out


def parse_walk(s: str) -> np.ndarray | None:
    """'0:0,40:0,130:90' -> (n, 2) of (sim time s, bearing deg), piecewise linear; silence after the last.
    '' or 'none' = silence all the time."""
    if not s.strip() or s.strip().lower() == "none":
        return None
    return np.array([[float(v) for v in p.split(":")] for p in s.split(",")])


def walk_bearing(walk: np.ndarray, t: float) -> float | None:
    if t < walk[0, 0] or t > walk[-1, 0]:
        return None
    return float(np.interp(t, walk[:, 0], np.unwrap(walk[:, 1], period=360.0))) % 360.0


def walk_messages(args, rng: np.random.Generator) -> list[tuple[float, dict]]:
    """The scripted walk as the 45 listener would send it (contract JSON through `parse_message`), 5 Hz, with
    --walk-noise (deg), --outliers (fraction of uniformly random bearings) and --dropouts LEN:EVERY (no messages
    for LEN s every EVERY s; with --dropout-loss-msg a detected=false at each start). One loss message at the end."""
    walk = parse_walk(args.walk)
    if walk is None:
        return []
    d_len, d_every = (float(v) for v in args.dropouts.split(":")) if args.dropouts else (0.0, 0.0)
    out, seq, in_drop = [], 0, False
    t0, t1 = float(walk[0, 0]), float(walk[-1, 0])

    def send(t: float, j: dict) -> None:
        nonlocal seq
        seq += 1
        out.append((t, parse_message(json.dumps({**j, "seq": seq, "t": round(t, 3)}))))

    for t in np.arange(t0, t1 + 1e-9, MSG_DT):
        t = float(t)
        if d_every > 0 and t >= d_every and (t % d_every) < d_len:
            if not in_drop and args.dropout_loss_msg:
                send(t, {"detected": False})
            in_drop = True
            continue
        in_drop = False
        az_true = walk_bearing(walk, t)
        rate = adiff(walk_bearing(walk, min(t + 0.1, t1)), walk_bearing(walk, max(t - 0.1, t0))) / 0.2
        az = rng.uniform(0, 360) if rng.random() < args.outliers else az_true + rng.normal(0, args.walk_noise)
        send(t, {"az_deg": round(az % 360.0, 2), "el_deg": 10.0, "sigma_az_deg": max(5.0, args.walk_noise),
                 "sigma_el_deg": 15.0, "label": "drone", "confidence": 0.8, "array_enu": [0.0, 0.0, 3.0],
                 "az_rate_dps": round(rate + rng.normal(0, 0.5), 2), "level_db": round(-30 + rng.normal(0, 1), 1),
                 "detected": True})
    send(t1 + MSG_DT, {"detected": False})
    return out


# -------------------------------------------------------------------------------------------------- world
class HunterController(base.Controller):
    """The base demo clips horizontal acceleration at 4 m/s^2, which cannot reach ~100 mph in a short run."""

    def __call__(self, pos, vel, R, omega, p_ref, v_ref, yaw_ref):
        acc = self.kp * (p_ref - pos) + self.kd * (v_ref - vel)
        acc[:2] = np.clip(acc[:2], -25.0, 25.0)
        acc[2] = np.clip(acc[2], -15.0, 20.0)
        f = self.mass * (acc + [0, 0, 9.81])
        z = f / np.linalg.norm(f)
        xc = np.array([math.cos(yaw_ref), math.sin(yaw_ref), 0.0])
        y = np.cross(z, xc)
        y /= np.linalg.norm(y)
        Rd = np.column_stack([np.cross(y, z), y, z])
        e = 0.5 * (Rd.T @ R - R.T @ Rd)
        torque = -self.kR * np.array([e[2, 1], e[0, 2], e[1, 0]]) - self.kw * omega
        return np.clip(self.mix_inv @ np.array([float(f @ R[:, 2]), *torque]), 0.0, self.ctrl_max)


class Demo:
    def __init__(self, m, d, info, args, seed: int = 0):
        self.m, self.d, self.info, self.args = m, d, info, args
        self.rng = np.random.default_rng(seed)
        self.ctrl = HunterController(m)
        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)
        a, g = self.args, self.info.ground
        self.audio, self.hunter = Audio(), Hunter(g)
        self.intruder = Intruder(a.range, a.intruder_agl, a.intruder_speed, g, a.range_from_level,
                                 a.approach, a.r_min)
        self.cam = Camera(self.m, self.d, self.rng)
        self.next_frame = 0.0
        self.trail: list[np.ndarray] = []
        self.intruder_trail: list[np.ndarray] = []
        self.track_err: list[float] = []
        self.max_out = 0.0

    def handle(self, msg: dict, source: str) -> None:
        """One parsed listener message (see `parse_message`): our drone's audio and the intruder both hear it."""
        t = self.d.time
        if not msg.get("detected", True):
            self.audio.lose(t, source)
            return
        st = self.audio.update(msg["az"], msg.get("sigma", 5.0), t, msg.get("rate"), msg.get("level_db"),
                               msg.get("confidence"), source)
        if st != "gated":
            self.intruder.hear(msg["az"] if st in ("ok", "resume") else self.audio.az, t, st in ("new", "jump"),
                               msg.get("level_db"))

    def hear(self, az: float, sigma: float, source: str = "manual") -> None:
        self.handle({"detected": True, "az": az % 360.0, "sigma": sigma}, source)

    def step(self) -> None:
        m, d, h = self.m, self.d, self.hunter
        dt = m.opt.timestep
        self.audio.advance(d.time, dt)
        self.intruder.update(d.time, dt, self.audio.hearing(d.time),
                             None if self.args.intruder_free else self.audio.line)
        h.aim(self.cam, self.audio, d.time)
        if d.time >= self.next_frame:
            self.next_frame = d.time + FRAME_DT
            r = self.cam.frame(self.intruder.pos if h.state not in ("WAIT", "LANDED") else None)
            h.on_frame(r, d.time)
            if h.state == "STRIKE" and self.intruder.pos is not None:
                self.track_err.append(float(np.linalg.norm(d.qpos[0:3] - self.intruder.pos)))
        pos, vel, R, om = base.state_of(d)
        h.update(pos, d.time, dt, self.audio)
        if h.state in ("WAIT", "LANDED"):
            d.ctrl[:] = 0.0
        else:
            d.ctrl[:] = self.ctrl(pos, vel, R, om, h.sp, h.v, h.yaw)
        mujoco.mj_step(m, d)
        self.max_out = max(self.max_out, float(np.linalg.norm(d.qpos[0:2])))
        if not self.trail or np.linalg.norm(d.qpos[0:3] - self.trail[-1]) > 4:
            self.trail = (self.trail + [d.qpos[0:3].copy()])[-500:]
        ip = self.intruder.pos
        if ip is not None and (not self.intruder_trail or np.linalg.norm(ip - self.intruder_trail[-1]) > 4):
            self.intruder_trail = (self.intruder_trail + [ip.copy()])[-500:]

    # -- drawing ----------------------------------------------------------------------------------------
    def draw(self, scn, eye) -> None:
        scn.ngeom = 0
        g, a, h, it = self.info.ground, self.audio, self.hunter, self.intruder
        t = self.d.time
        base.add_compass(scn, g)
        add_label(scn, [0, 0, 4.6], "UMA-16")
        if a.az is not None:
            hearing = a.hearing(t)
            length = max(it.r_cur + 200.0, 700.0)
            w = float(np.clip(0.0012 * np.linalg.norm(eye), 0.05, 0.8))
            col = [1.0, 0.45, 0.1, 0.9] if hearing else [0.6, 0.6, 0.6, 0.5]
            for s, width, c in ((0.0, w, col), (-2 * a.sigma, w / 3, [1.0, 0.6, 0.2, 0.35]),
                                (2 * a.sigma, w / 3, [1.0, 0.6, 0.2, 0.35])):
                pts = base.terrain_path(g, (a.line if a.line is not None else a.az) + s, length,
                                        25.0 if s == 0 else 50.0, 4.0)
                pts[0] = [0, 0, 3.3]
                for p0, p1 in zip(pts[:-1], pts[1:]):
                    add_line(scn, p0, p1, width, c)
            for tk, az, ok in a.history:   # raw bearings of the last 30 s, fading, on a 150 m ring; gated grey
                k = float(np.clip(1 - (t - tk) / 30.0, 0, 1))
                if k > 0:
                    xy = az_to_dir(az) * 150.0
                    add_sphere(scn, [*xy, float(g(*xy)) + 2.0], 1.2,
                               [1.0, 0.5, 0.1, 0.8 * k] if ok else [0.6, 0.6, 0.6, 0.6 * k])
        if it.pos is not None:
            quad_marker(scn, it.pos, it.yaw, t, eye, RED_BODY, RED_ROTOR, f"INTRUDER  {it.range_from_base():.0f} m")
        p = self.d.qpos[0:3].copy()
        quad_marker(scn, p, quat_yaw(self.d.qpos[3:7]), t, eye, GREEN_BODY, GREEN_ROTOR, "X2", min_dist=25.0)
        self._draw_intercept(scn, p, it, h, t)
        for p0, p1 in zip(self.trail[:-1], self.trail[1:]):
            add_line(scn, p0, p1, 0.25, [0.2, 0.9, 1.0, 0.8])
        for p0, p1 in zip(self.intruder_trail[:-1], self.intruder_trail[1:]):
            add_line(scn, p0, p1, 0.2, [0.9, 0.15, 0.12, 0.75])
        if h.state not in ("WAIT", "LANDED"):
            locked = h.state == "STRIKE"
            c = self.cam.pos()
            fwd, right, up = cam_axes(self.cam.yaw, self.cam.pitch)
            L = float(self.cam.last.get("dist", 300.0)) if locked else 300.0
            th, tv = math.tan(math.radians(FOV_H / 2)), math.tan(math.radians(FOV_V / 2))
            col = [0.2, 1.0, 0.3, 0.7] if locked else [1.0, 1.0, 1.0, 0.35]
            for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
                add_line(scn, c, c + L * (fwd + sx * th * right + sy * tv * up), 0.15, col)
            if locked and h.est is not None:
                add_sphere(scn, h.est, 3.0, [0.2, 1.0, 0.3, 0.4])

    def showing_intercept(self) -> bool:
        """Camera lock, and a few seconds after the hit, while the close-up is worth keeping."""
        h, t = self.hunter, self.d.time
        return h.state == "STRIKE" or (h.t_struck is not None and t - h.t_struck < INTERCEPT_HOLD_S)

    def _draw_intercept(self, scn, p, it, h, t) -> None:
        """The run-in: X2 trail is cyan, intruder trail is red, and while locked a line joins them."""
        if it.pos is None or not self.showing_intercept():
            return
        gap = float(np.linalg.norm(it.pos - p))
        col = [0.25, 1.0, 0.35, 0.95] if h.struck else [1.0, 0.25, 0.15, 0.95]
        add_line(scn, p, it.pos, float(np.clip(0.004 * gap, 0.35, 2.5)), col)
        mid = 0.5 * (np.asarray(p, float) + it.pos)
        add_label(scn, mid + [0, 0, max(6.0, 0.04 * gap)], "HIT" if h.struck else f"gap {gap:.0f} m", col)
        if h.est is not None and not h.struck:
            add_line(scn, p, h.est, 0.2, [0.3, 1.0, 0.45, 0.55])
            add_sphere(scn, h.est, STRIKE_M, [0.2, 1.0, 0.35, 0.12])

    def banner(self) -> str:
        """The big state line on top of the viewer."""
        h = self.hunter
        if h.struck:
            return "TARGET HIT"
        return {"WAIT": "STANDBY", "TAKEOFF": "LAUNCH", "INTERCEPT": "INTERCEPT", "STRIKE": "STRIKE",
                "LOST": "INTERCEPT", "RETURN": "RETURN TO BASE", "DESCEND": "RETURN TO BASE",
                "LANDED": "STANDBY"}.get(h.state, h.state)

    def banner_color(self) -> tuple[int, int, int]:
        h = self.hunter
        if h.struck:
            return (30, 170, 60)
        return {"WAIT": (90, 90, 90), "TAKEOFF": (30, 130, 200), "INTERCEPT": (220, 140, 0), "STRIKE": (210, 25, 20),
                "LOST": (220, 140, 0), "LANDED": (90, 90, 90)}.get(h.state, (30, 130, 200))

    def hud_short(self) -> str:
        """One small line under the picture: bearing, ranges."""
        a, it = self.audio, self.intruder
        p = self.d.qpos[0:3]
        parts = ["no sound" if a.az is None else f"bearing {a.az:5.1f} deg" + ("" if a.hearing(self.d.time) else
                                                                               " (silent)")]
        if it.pos is not None:
            parts.append(f"intruder {it.range_from_base():.0f} m")
        parts.append(f"X2 {np.linalg.norm(p[:2]):.0f} m")
        if it.pos is not None and self.hunter.state not in ("WAIT", "LANDED"):
            parts.append(f"gap {np.linalg.norm(p - it.pos):.0f} m")
        return "   |   ".join(parts)

    def hud(self) -> str:
        a, h, it, r = self.audio, self.hunter, self.intruder, self.cam.last
        t = self.d.time
        p = self.d.qpos[0:3]
        lines = [f"State: {h.state}", f"Audio: {a.status(t)}   [{a.source}, {a.msg_hz(t):.1f} Hz]"]
        if a.az is not None:
            lvl = f"{a.level_db:.0f} dB" if a.level_db is not None else "-"
            last = "" if a.hearing(t) else "  (last)"
            lines.append(f"Bearing: {a.az:5.1f} +- {a.sigma:.1f} deg  rate {a.rate:+.1f} deg/s  level {lvl}{last}")
            if a.n_gated or a.n_jumps:
                lines.append(f"Gated outliers: {a.n_gated}, bearing jumps: {a.n_jumps}")
        lines.append(f"X2: {np.linalg.norm(p[:2]):.0f} m from base, {p[2] - self.info.ground(p[0], p[1]):.0f} m AGL")
        if r and h.state not in ("WAIT", "LANDED"):
            view = "OCCLUDED" if r["occluded"] else "in view" if r["in_view"] else "not in view"
            lines.append(f"Camera: intruder {view}, {r['px']:.1f} px, {r.get('dist', 0):.0f} m")
        if it.pos is not None:
            lines.append(f"Intruder: {it.range_from_base():.0f} m out, closing {it.approach:.0f} m/s along the sound line")
        if h.t_found is not None:
            lines.append(("STRUCK, " if h.struck else "") + f"camera lock {h.t_found - h.t_launch:.0f} s after launch")
        if h.state == "STRIKE" and h.est is not None:
            lines.append(f"Target estimate: {np.linalg.norm(h.est[:2]):.0f} m from base, "
                         f"bearing {(90 - math.degrees(math.atan2(h.est[1], h.est[0]))) % 360:.0f} deg")
        return "\n".join(lines)


def _geom(scn, gtype, size, pos, mat, rgba) -> None:
    g = base._new_geom(scn)
    if g is not None:
        mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float), np.asarray(mat, float).ravel(),
                            np.asarray(rgba, float))


RED_BODY, RED_ROTOR = (0.55, 0.05, 0.05, 1.0), (0.85, 0.1, 0.1, 0.6)
GREEN_BODY, GREEN_ROTOR = (0.05, 0.5, 0.12, 1.0), (0.15, 0.9, 0.3, 0.6)


def draw_quad(scn, p, yaw: float, t: float, scale: float = 1.0, body=RED_BODY, rotor=RED_ROTOR) -> None:
    """A quadcopter: body, two crossed arms, four rotor discs (1.2 m motor to motor at scale 1)."""
    def rz(a):
        return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    s = scale
    _geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [0.16 * s, 0.12 * s, 0.06 * s], p, rz(yaw), body)
    for k in range(4):
        a = yaw + math.pi / 4 + k * math.pi / 2
        u = np.array([math.cos(a), math.sin(a), 0.0])
        _geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [0.3 * s, 0.025 * s, 0.02 * s], p + 0.3 * s * u, rz(a),
              [0.15, 0.15, 0.15, 1])
        _geom(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [0.17 * s, 0.008 * s, 0], p + s * (0.6 * u + [0, 0, 0.05]),
              np.eye(3), rotor)


def quad_marker(scn, p, yaw: float, t: float, eye, body, rotor, text: str, min_dist: float = 0.0) -> None:
    """A quadcopter icon on the drone, scaled with the viewer's distance so it stays visible far out."""
    dist = float(np.linalg.norm(eye - p))
    if dist <= min_dist:
        return
    s = float(np.clip(0.035 * dist, 1.0, 50.0))
    draw_quad(scn, p, yaw, t, s, body, rotor)
    add_label(scn, p + [0, 0, 0.8 * s + 1.0], text)


def quat_yaw(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# ---------------------------------------------------------------------------------------------- camera view
class CameraView:
    """Offscreen render of the gimbal camera (zoomed to FOV_V), box + text overlay, for the viewer corner."""

    def __init__(self, m, info, w: int = PIC_W, h: int = PIC_H):
        import cv2  # noqa: PLC0415
        self.cv2, self.info, self.w, self.h = cv2, info, w, h
        g = m.vis.global_
        g.offwidth, g.offheight = max(g.offwidth, w), max(g.offheight, h)
        self.r = mujoco.Renderer(m, h, w, max_geom=20000)
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
        W, H = self.w, self.h
        k = W / CAM_W
        lw = max(1, round(k))
        if r.get("in_view") and not r.get("occluded"):
            th, tv = math.tan(math.radians(FOV_H / 2)), math.tan(math.radians(FOV_V / 2))
            x = int(W / 2 + math.tan(math.radians(r["ah"])) / th * W / 2)
            y = int(H / 2 - math.tan(math.radians(r["av"])) / tv * H / 2)
            s = int(max(r["px"], 8) * k)
            col = (40, 255, 80) if h.state == "STRIKE" else (255, 220, 0) if r["hit"] else (200, 200, 200)
            cv2.rectangle(img, (x - s, y - s), (x + s, y + s), col, 2 * lw if h.state == "STRIKE" else lw)
        cv2.drawMarker(img, (W // 2, H // 2), (255, 255, 255), cv2.MARKER_CROSS, int(14 * k), lw)
        label = "STRIKE" if h.state == "STRIKE" else "SEARCH" if h.state in ("INTERCEPT", "LOST") else h.state
        cv2.putText(img, f"X2 CAM {FOV_H:.0f} deg  {label}", (int(8 * k), int(22 * k)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55 * k, (255, 255, 255), lw, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (W - 1, H - 1), (255, 255, 255), lw)
        return img


def intercept_camera(a, b) -> tuple[np.ndarray, float, float, float]:
    """Frame the segment from `a` (our drone) to `b` (the intruder) from the side.

    Returns (lookat, distance, azimuth_deg, elevation_deg) for a free MuJoCo camera. Distance shrinks with
    the gap so the hit ends as a close-up, and the azimuth stays square to the closing line.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    gap = float(np.linalg.norm(b - a))
    heading = math.atan2(float(b[1] - a[1]), float(b[0] - a[0]))
    return (0.5 * (a + b), float(np.clip(1.6 * gap + 18.0, 22.0, 900.0)),
            math.degrees(heading + math.pi / 2.0) % 360.0, -18.0)


def apply_intercept_camera(cam, a, b) -> None:
    mid, dist, az, el = intercept_camera(a, b)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = mid
    cam.distance, cam.azimuth, cam.elevation = dist, az, el


def banner_image(text: str, color: tuple[int, int, int], w: int, h: int) -> np.ndarray:
    """The state banner: white bold text centred on a coloured bar (RGB)."""
    import cv2  # noqa: PLC0415
    img = np.empty((h, w, 3), np.uint8)
    img[:] = color
    font, thick = cv2.FONT_HERSHEY_DUPLEX, max(2, h // 25)
    scale = 1.0
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    scale = min(0.92 * w / max(tw, 1), 0.62 * h / max(th, 1))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.putText(img, text, ((w - tw) // 2, (h + th) // 2), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (255, 255, 255), max(1, h // 40))
    return img


# -------------------------------------------------------------------------------------------------- run
def run_viewer(m, d, info, args) -> None:
    demo, udp = Demo(m, d, info, args), UdpTrack(args.udp_port)
    feed = Feed(load_replay(args.replay), "replay") if args.replay else None
    keys: list[int] = []
    speed, chase, show_cam = 1, False, not args.no_camera_view
    cam_hold, framed = False, False   # C during the close-up keeps the manual view; framed = we took the camera
    manual_az, manual_on, manual_next = 0.0, False, 0.0
    view = None
    if show_cam:
        try:
            view = CameraView(m, info)
        except Exception as e:  # noqa: BLE001  (no offscreen GL next to the viewer: run without the picture)
            print(f"camera picture off: {e}")
    last_log, next_pic = "", 0.0
    with mujoco.viewer.launch_passive(m, d, key_callback=keys.append, show_left_ui=False,
                                      show_right_ui=False) as viewer:
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
                    if manual_on:
                        with viewer.lock():
                            demo.audio.lose(d.time, "manual")
                    manual_on = False
                elif k in (ord("1"), ord("2"), ord("4")):
                    speed = int(chr(k))
                elif k == ord("C"):
                    cam_hold = demo.showing_intercept()
                    chase = not chase
                    base.set_chase(viewer.cam, m) if chase else base.set_overview(viewer.cam, info)
                    framed = False
                elif k == ord("V"):
                    show_cam = not show_cam
                elif k == 259:
                    cam_hold, framed, chase = False, False, False
                    base.set_overview(viewer.cam, info)
                    with viewer.lock():
                        demo.reset()
                        if feed is not None:
                            feed.reset()
            with viewer.lock():
                for msg in udp.pop_all():
                    demo.handle(msg, "udp")
                for _ in range(int(speed / 60 / m.opt.timestep)):
                    if feed is not None:
                        for msg in feed.due(d.time):
                            demo.handle(msg, feed.source)
                    if manual_on and d.time >= manual_next:
                        manual_next = d.time + MSG_DT
                        demo.hear(manual_az + demo.rng.normal(0, 2.0), 5.0)
                    demo.step()
                want = demo.showing_intercept()
                target = demo.intruder.pos if demo.intruder.pos is not None else demo.hunter.est
                if want and not cam_hold and target is not None:
                    apply_intercept_camera(viewer.cam, d.qpos[0:3], target)
                    framed = True
                elif not want:
                    if framed:
                        base.set_chase(viewer.cam, m) if chase else base.set_overview(viewer.cam, info)
                    framed, cam_hold = False, False
                demo.draw(viewer.user_scn, base.cam_pos(viewer.cam))
                banner, small = demo.banner(), demo.hud_short()
            # set_texts / set_images take the viewer's lock themselves: never inside `with viewer.lock()`
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, small, ""))
            if time.time() >= next_pic:
                next_pic = time.time() + 0.1
                vp = viewer.viewport
                bw = int(vp.width * BANNER_FRAC)
                bh = max(bw // 7, 1)
                top = mujoco.MjrRect(vp.left + (vp.width - bw) // 2, vp.bottom + vp.height - bh - 12, bw, bh)
                gw, gh = bw * 3 // 5, max(bh * 2 // 5, 1)
                gnss = mujoco.MjrRect(vp.left + (vp.width - gw) // 2, top.bottom - gh - 8, gw, gh)
                images = [(top, banner_image(banner, demo.banner_color(), bw, bh)),
                          (gnss, banner_image("GNSS NOT AVAILABLE", (150, 20, 20), gw, gh))]
                if view is not None and show_cam:
                    img = view.render(demo)
                    dw = min(int(vp.width * PIC_FRAC), img.shape[1])
                    dh = dw * img.shape[0] // max(img.shape[1], 1)
                    if dw != img.shape[1]:
                        img = view.cv2.resize(img, (dw, dh), interpolation=view.cv2.INTER_AREA)
                    images.append((mujoco.MjrRect(vp.left + vp.width - dw - 12, vp.bottom + 12, dw, dh), img))
                viewer.set_images(images)
            log = demo.hunter.log[-1] if demo.hunter.log else ""
            if log != last_log:
                last_log = log
                print(log, flush=True)
            viewer.sync()
            time.sleep(max(0.0, 1 / 60 - (time.time() - t0)))


def run_headless(m, d, info, args, seed: int = 0, quiet: bool = False) -> dict:
    """Scripted walk (--walk, 5 Hz, --walk-noise / --outliers / --dropouts) or --replay; returns the result (time
    to find, tracking, landing, intruder top speed). Stops when landed, at --max-t, or when the input is over and
    the drone never launched."""
    demo = Demo(m, d, info, args, seed)
    feed = (Feed(load_replay(args.replay), "replay") if args.replay
            else Feed(walk_messages(args, np.random.default_rng(seed + 1000)), "walk"))
    nxt_print = 0.0
    while d.time < args.max_t and demo.hunter.state != "LANDED":
        for msg in feed.due(d.time):
            demo.handle(msg, feed.source)
        demo.step()
        if demo.hunter.state == "WAIT" and feed.done() and d.time > feed.t_end + AUDIO_TIMEOUT + HEAR_S:
            break
        if not quiet and d.time >= nxt_print:
            nxt_print += 10.0
            p, it, r = d.qpos[0:3], demo.intruder, demo.cam.last
            gap = f"{np.linalg.norm(p - it.pos):6.0f}" if it.pos is not None else "     -"
            print(f"t={d.time:6.1f}  {demo.hunter.state:9s}  X2 {np.linalg.norm(p[:2]):6.0f} m out  "
                  f"gap to intruder {gap} m  cam in_view={r.get('in_view')} px={r.get('px', 0):.1f}")
    h, a = demo.hunter, demo.audio
    res = {"launched": h.t_launch is not None, "found": h.t_found is not None,
           "t_find_s": round(h.t_found - h.t_launch, 1) if h.t_found is not None else None,
           "track_dist_median_m": round(float(np.median(demo.track_err)), 1) if demo.track_err else None,
           "struck": h.struck, "state": h.state, "t": round(d.time, 1),
           "landing_err_m": round(float(np.linalg.norm(d.qpos[:2] - PAD)), 2),
           "max_out_m": round(demo.max_out, 1), "intruder_max_speed_mps": round(demo.intruder.v_max, 2),
           "msgs": a.n_msgs, "gated": a.n_gated, "jumps": a.n_jumps, "loss_msgs": a.n_lost}
    if not quiet:
        print("\n".join(h.log))
        print("RESULT", json.dumps(res))
    return res


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", choices=["farm", "simple"], default="farm")
    ap.add_argument("--range", type=float, default=500.0, help="intruder start distance from the base (m)")
    ap.add_argument("--approach", type=float, default=APPROACH, help="intruder closes on the base at this speed (m/s)")
    ap.add_argument("--r-min", type=float, default=R_MIN, help="closest the intruder comes (m)")
    ap.add_argument("--intruder-agl", type=float, default=80.0, help="intruder height above the terrain (m)")
    ap.add_argument("--intruder-speed", type=float, default=20.0, help="max intruder speed with --intruder-free (m/s)")
    ap.add_argument("--intruder-free", action="store_true",
                    help="intruder flies after the sound speed-limited (default: always exactly on the sound line)")
    ap.add_argument("--udp-port", type=int, default=5005, help="0 = off")
    ap.add_argument("--no-camera-view", action="store_true", help="no camera picture in the viewer corner")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--range-from-level", action="store_true",
                    help="intruder range from level_db: the first level heard = --range, +6 dB = half, 150-1200 m")
    ap.add_argument("--replay", default=None, metavar="TRACK.JSONL",
                    help="replay a recorded listener log (t, az_deg / compass_az, sigma_az_deg ...) in sim time")
    ap.add_argument("--walk", default="0:0,40:0,130:90,200:90",
                    help="headless: t:bearing points (sim s : deg), linear in between, silence after the last; "
                         "'none' = silence")
    ap.add_argument("--walk-noise", type=float, default=2.0, help="headless walk: bearing noise (deg, 1 sigma)")
    ap.add_argument("--outliers", type=float, default=0.0, help="headless walk: fraction of random bearings")
    ap.add_argument("--dropouts", default="", metavar="LEN:EVERY",
                    help="headless walk: no bearings for LEN s every EVERY s (e.g. 2:20)")
    ap.add_argument("--dropout-loss-msg", action="store_true",
                    help="headless walk: each dropout starts with a detected=false message")
    ap.add_argument("--max-t", type=float, default=600.0)
    return ap


def build(world: str):
    base.CLIMB, base.SINK, base.OUT_SPEED = 20.0, 12.0, HUNT_SPEED
    return base.build_model(world)


def main() -> None:
    args = parser().parse_args()
    t0 = time.time()
    m, info = build(args.world)
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
