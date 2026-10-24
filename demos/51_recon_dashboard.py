"""Recon dashboard: the UMA-16 at the base gives a bearing, the drone takes off from the pad, flies out along it,
orbits, flies home and lands, in the live-demo style (OpenCV window over an offscreen MuJoCo render).

    uv run python demos/51_recon_dashboard.py                          # viewer; bearing from the keys or UDP 5005
    uv run python demos/51_recon_dashboard.py --bearing 60 --gnss-off-at 50   # launch at once, GNSS off after 50 m
    uv run python demos/51_recon_dashboard.py --headless --bearing 200 --gnss-off-at 50  # no window: fly, print, score
    uv run python demos/51_recon_dashboard.py --record out.mp4 --bearing 60 --gnss-off-at 50 --speed 3   # MP4

There is NEVER a data link between base and drone. The base knows the UMA-16 bearing and gives the launch command;
that is all. The drone flies on its own estimate: the repo's AHRS + EKF (sfhack/estimation) on the simulated IMU,
compass, barometer, and GNSS until it is switched off (key N, or --gnss-off-at <m of flight>). From then on it is
dead reckoning: the return goes to where the drone THINKS the pad is and it lands there, so the miss is visible.
Onboard: sfhack/mission/bearing_pilot.py (never sees the truth). Simulator side: sfhack/sim/sensor_rig.py.
This file is harness: it reads the truth to DISPLAY (white drone, trails, cameras, map) and SCORE (errors) only.

Bearing input (compass degrees, same as demos/50_base_3d.py):
  LEFT / RIGHT turn the pending bearing by 5 deg, ENTER launches toward it, F launches toward a random bearing,
  UDP port 5005 (--udp-port): JSON with "az_deg" or a plain number. A bearing while the drone is out is queued at
  the base (no link) and given with the next launch, after landing; --retarget-in-flight retargets it in the air
  as 50_base_3d does (that would need a link: the banner says so).
Keys: N GNSS off / on   V record on / off (MP4 under $SFHACK_DATA/demo)   C chase / overview camera
      1 2 4 sim speed   SPACE pause   BACKSPACE reset   O reset camera   Z screenshot   Tab fullscreen   Q quit
Mouse: drag to orbit, wheel to zoom.
World: --world base (demos/50_base_3d.py) or farm (demos/base3d_farm.py, default when it is there).
"""

from __future__ import annotations

import argparse
import importlib
import math
import random
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import mujoco
import numpy as np

from demo_live import WHEEL_ZOOM_SIGN, _mouse_wheel_delta, _VideoOut
from liveview.hud import AMBER, CYAN, GREEN, RED
from liveview.recon_dash import DashComposer
from sfhack.bus import Bus
from sfhack.config import data_dir, default_config
from sfhack.mission.bearing_pilot import BearingPilot, BearingPlan
from sfhack.sim.sensor_rig import SensorRig

DEMOS = Path(__file__).resolve().parent
WORLDS = {"base": "50_base_3d", "farm": "base3d_farm"}
DEFAULT_WORLD = "farm" if (DEMOS / "base3d_farm.py").exists() else "base"
SETTLE_S = 1.0  # the X2 drops onto the pad at spawn: settle before the IMU calibration starts
TRAIL_DT = 0.25
RELAUNCH_S = 3.0  # on the pad before a queued bearing launches
VIEW_SIZE, RECORD_SIZE = (1280, 720), (1920, 1080)
WINDOW = "UMA-16 recon dashboard"
KEYS = {2424832: "LEFT", 65361: "LEFT", 2555904: "RIGHT", 65363: "RIGHT", 13: "ENTER", 10: "ENTER",
        8: "BACKSPACE", 9: "TAB", 32: "SPACE", 27: "Q"}


def load_world(name: str):
    """The world builder module: build_model(), PAD, CRUISE_AGL, ... and UdpBearings (demos/50_base_3d.py API)."""
    return importlib.import_module(WORLDS[name])


# ---------------------------------------------------------------------------------------- the scenario (harness)

class ReconScenario:
    """World + sensor rig (simulator) + BearingPilot (onboard), and what the harness measures. The pilot gets sensor
    messages on the bus and the launch command; everything true here is for display and scoring."""

    def __init__(self, world, args: argparse.Namespace, model: mujoco.MjModel | None = None) -> None:
        self.world, self.args = world, args
        self.world_name = args.world
        self.model = model if model is not None else world.build_model()
        self.data = mujoco.MjData(self.model)
        m, d = self.model, self.data
        mujoco.mj_forward(m, d)
        while d.time < SETTLE_S:
            mujoco.mj_step(m, d)
        self.body = m.body("x2").id
        self.z0 = float(d.qpos[2])  # display datum: the X2 resting on the pad (onboard z = 0 there)
        self.pad = np.asarray(world.PAD, float)[:2].copy()
        self.mast = np.zeros(2)
        self.plan = BearingPlan(range_m=args.range, cruise_agl=world.CRUISE_AGL, out_speed=world.OUT_SPEED,
                                climb=world.CLIMB, descent=world.DESCENT, orbit_r=world.ORBIT_R, orbit_s=world.ORBIT_S)
        cfg = default_config()
        self.bus = Bus()
        self.rig = SensorRig(cfg, self.bus, m, d, seed=args.seed)
        self.pilot = BearingPilot(cfg, self.bus, self.rig.vehicle_params(), self.pad, self.plan, self.mast)
        self.dt = float(m.opt.timestep)
        # base side
        self.pending = float(args.bearing) if args.bearing is not None else 0.0
        self.bearing_src = "-"
        self.bearing: float | None = None  # the bearing the base launched the drone with
        self.launch_t: float | None = None
        self.queued: float | None = None
        self.n_sorties = 0
        # truth side (display + score)
        self.gnss_off_t: float | None = None
        self.gnss_off_flown = 0.0
        self.gnss_off_xy: np.ndarray | None = None
        self.gnss_off_idx: int | None = None
        self.banner: tuple[float, str, tuple] | None = None
        self._last_xy = d.qpos[0:2].copy()
        self._new_sortie()
        self.est = self.pilot.pose()
        self._read_truth()

    def _new_sortie(self) -> None:
        self.truth_trail: list[np.ndarray] = []
        self.est_trail: list[np.ndarray] = []
        self._next_trail = 0.0
        self.flown = 0.0
        self.max_err = 0.0
        self.err_target: float | None = None
        self.land_err: float | None = None
        self.land_claim: float | None = None
        self.landed_t: float | None = None
        self.touch_v: float | None = None  # true horizontal speed at ground contact (m/s)
        self.prev_state = self.pilot.state
        self.state_log: list[tuple[float, str]] = []
        if self.gnss_off_t is not None:  # GNSS stays off across sorties; the drift marks restart with the trail
            self.gnss_off_idx = 0

    @property
    def t(self) -> float:
        return float(self.data.time)

    def _read_truth(self) -> None:
        d = self.data
        self.true_pos = d.qpos[0:3].copy()
        self.true_vel = d.qvel[0:3].copy()
        self.true_R = d.xmat[self.body].reshape(3, 3).copy()

    # ------------------------------------------------------------------ base commands

    def base_bearing(self, az: float, src: str) -> str:
        """A bearing reached the base (keys / UDP / random). On the pad it goes out with the launch command."""
        az %= 360.0
        self.pending, self.bearing_src = az, src
        t = self.t
        if self.pilot.on_ground:
            if self.pilot.launch(az, t):
                self.launch_t, self.bearing = t, az
                self.queued = None
                self.n_sorties += 1
                self._new_sortie()
                self.banner = (t, f"LAUNCH toward {az:05.1f}°  (last thing the base tells the drone)", CYAN)
                return f"launch {az:.1f} deg ({src})"
            return "launch refused"
        if self.args.retarget_in_flight:
            self.pilot.retarget(az, t)
            self.bearing = az
            self.banner = (t, f"RETARGET {az:05.1f}° in flight (demo flag: would need a link)", AMBER)
            return f"retarget {az:.1f} deg (demo flag)"
        self.queued = az
        self.banner = (t, f"BASE: new bearing {az:05.1f}° queued, no link: goes with the next launch", AMBER)
        return f"queued {az:.1f} deg (no link while the drone is out)"

    def toggle_gnss(self) -> str:
        if self.rig.gnss_on:
            self.gnss_off("key N")
            return "GNSS OFF"
        self.rig.set_gnss(True)
        self.gnss_off_t = self.gnss_off_xy = self.gnss_off_idx = None
        self.banner = (self.t, "GNSS BACK ON", GREEN)
        return "GNSS ON"

    def gnss_off(self, why: str) -> None:
        if not self.rig.gnss_on:
            return
        self.rig.set_gnss(False)
        self.gnss_off_t, self.gnss_off_flown = self.t, self.flown
        self.gnss_off_xy = self.true_pos[:2].copy()
        self.gnss_off_idx = len(self.truth_trail)
        self.banner = (self.t, f"GNSS OFF ({why}, {self.flown:.0f} m flown): IMU + compass + baro only", RED)

    # ------------------------------------------------------------------ stepping

    def advance(self, n: int) -> None:
        d, pilot = self.data, self.pilot
        for _ in range(n):
            self.rig.step()
            xy = d.qpos[0:2]
            if not pilot.on_ground:
                self.flown += math.hypot(xy[0] - self._last_xy[0], xy[1] - self._last_xy[1])
            self._last_xy[:] = xy
            if self.args.gnss_off_at is not None and self.rig.gnss_on and self.flown >= self.args.gnss_off_at:
                self.gnss_off(f"--gnss-off-at {self.args.gnss_off_at:g}")
            if pilot.state != self.prev_state:
                self._on_state(pilot.state)
            if pilot.ground_contact_t is not None and self.touch_v is None:
                self.touch_v = float(math.hypot(d.qvel[0], d.qvel[1]))
            if self.t >= self._next_trail:
                self._next_trail = self.t + TRAIL_DT
                self._read_truth()
                self.est = pilot.pose()
                if not pilot.on_ground or pilot.state == "LANDED" and self.landed_t is not None and \
                        self.t - self.landed_t < TRAIL_DT * 1.5:
                    self.truth_trail.append(self.true_pos.copy())
                    self.est_trail.append(self.est.pos.copy())
                if self.launch_t is not None:
                    self.max_err = max(self.max_err, self.err_now())
            if pilot.state == "LANDED" and self.queued is not None and self.t - (self.landed_t or 0.0) >= RELAUNCH_S:
                self.base_bearing(self.queued, "queued")
        self._read_truth()
        self.est = pilot.pose()

    def _on_state(self, s: str) -> None:
        self.prev_state = s
        self._read_truth()
        self.est = self.pilot.pose()
        self.state_log.append((self.t, s))
        if s == "ORBIT" and self.err_target is None:
            self.err_target = self.err_now()
        elif s == "LANDED":
            self.landed_t = self.t
            self.land_err = float(np.linalg.norm(self.true_pos[:2] - self.pad))
            self.land_claim = float(np.linalg.norm(self.est.pos[:2] - self.pad))
            self.banner = (self.t, f"LANDED where it thinks the pad is: {self.land_err:.1f} m from the real pad",
                           GREEN if self.land_err < 5.0 else RED)

    def err_now(self) -> float:
        return float(np.linalg.norm(self.est.pos[:2] - self.true_pos[:2]))

    @property
    def done(self) -> bool:
        return self.pilot.state == "LANDED" and self.queued is None

    # ------------------------------------------------------------------ dashboard state

    def hud_state(self, speed: float, paused: bool, recording: bool) -> dict:
        p = self.pilot
        if self.queued is not None:
            pending = f"{self.queued:05.1f}° queued: no link, goes with the next launch"
        elif p.on_ground:
            pending = f"{self.pending:05.1f}°   ← → then ENTER (or F, or UDP {self.args.udp_port})"
        else:
            pending = f"{self.pending:05.1f}°   (drone out: cannot be told anything)"
        if self.gnss_off_t is None:
            since = "GNSS still on" if self.rig.gnss_on else "-"
        else:
            since = f"{self.t - self.gnss_off_t:.1f} s  ·  {self.flown - self.gnss_off_flown:.0f} m flown"
        dpad = float(np.linalg.norm(self.est.pos[:2] - self.pad))
        believes = (f"on the pad ({dpad:.1f} m off by its own estimate)" if p.state == "LANDED"
                    else f"{dpad:.0f} m from the pad, height {self.est.pos[2]:.0f} m")
        result = None
        if self.land_err is not None and p.state == "LANDED":
            col = GREEN if self.land_err < 5.0 else (AMBER if self.land_err < 15.0 else RED)
            result = (f"LANDED  {self.land_err:.1f} m from the pad  (the drone thinks {self.land_claim:.1f} m)", col)
        banner = None
        if self.banner is not None:
            banner = (self.banner[1], self.t - self.banner[0], self.banner[2])
        return {
            "world": self.world_name, "t": self.t, "recording": recording, "paused": paused, "speed": speed,
            "bearing": self.bearing, "sigma": self.plan.bearing_sigma_deg, "launch_t": self.launch_t,
            "source": f"from {self.bearing_src}" if self.launch_t is not None else "ENTER / F / UDP",
            "pending_line": pending, "queued": self.queued is not None,
            "state": p.state + (" (calibrating)" if p.calibrating else ""),
            "gnss_on": self.rig.gnss_on, "err": self.err_now(), "sigma2": self.est.sigma2_horizontal(),
            "max_err": self.max_err if self.launch_t is not None else None, "err_target": self.err_target,
            "land_err": self.land_err, "since_off": since,
            "from_base": f"{np.linalg.norm(self.true_pos[:2] - self.mast):.0f} m, height {self.true_pos[2] - self.z0:.0f} m",
            "believes": believes, "result": result, "banner": banner,
        }

    def summary(self) -> str:
        p = self.pilot
        off = ("GNSS on all the way" if self.gnss_off_t is None else
               f"GNSS off at {self.gnss_off_flown:.0f} m flown (t {self.gnss_off_t:.1f} s), blind for "
               f"{(self.landed_t or self.t) - self.gnss_off_t:.1f} s / {self.flown - self.gnss_off_flown:.0f} m")
        tgt = "-" if self.err_target is None else f"{self.err_target:.1f} m"
        land = "-" if self.land_err is None else f"{self.land_err:.2f} m (drone thinks {self.land_claim:.2f} m)"
        tilt = math.degrees(math.acos(float(np.clip(self.true_R[2, 2], -1.0, 1.0))))
        touch = "-" if self.touch_v is None else f"{self.touch_v:.1f} m/s"
        return (f"RESULT state={p.state} t={self.t:.1f}s bearing={self.bearing}  max error {self.max_err:.1f} m  "
                f"error at target {tgt}  landing error {land}  flown {self.flown:.0f} m  {off}\n"
                f"       touchdown horizontal speed {touch}, attitude on the ground {tilt:.0f} deg from level")


# ---------------------------------------------------------------------------------------- headless

def run_headless(world, args: argparse.Namespace) -> int:
    sc = ReconScenario(world, args)
    sc.base_bearing(args.bearing if args.bearing is not None else 60.0, "--bearing")
    wall, nxt = time.time(), 0.0
    while sc.t < args.max_t and not sc.done:
        sc.advance(50)
        if sc.t >= nxt:
            nxt += 10.0
            print(f"t={sc.t:6.1f}s {sc.pilot.state:9s} GNSS {'on ' if sc.rig.gnss_on else 'OFF'} "
                  f"true ({sc.true_pos[0]:7.1f},{sc.true_pos[1]:7.1f},{sc.true_pos[2] - sc.z0:5.1f})  "
                  f"err {sc.err_now():6.1f} m  2-sigma {sc.est.sigma2_horizontal():6.1f} m  flown {sc.flown:5.0f} m",
                  flush=True)
    for t, text in sc.pilot.decisions:
        print(f"   {t:6.1f}s {text}")
    print(sc.summary())
    print(f"wall {time.time() - wall:.1f} s  ({sc.t / max(time.time() - wall, 1e-6):.1f}x real time)")
    ok = sc.pilot.state == "LANDED" and (sc.gnss_off_t is not None or (sc.land_err or 1e9) < 5.0)
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------- record (offscreen)

def run_record(world, args: argparse.Namespace) -> int:
    out = Path(args.record)
    out.parent.mkdir(parents=True, exist_ok=True)
    W, H = args.size
    fps = 30
    sc = ReconScenario(world, args)
    comp = DashComposer(sc, W, H, map_half(args))
    comp.overview = args.overview
    sc.base_bearing(args.bearing if args.bearing is not None else 60.0, "--bearing")
    video = _VideoOut(out, W, H, fps)
    steps = int(round(args.speed / fps / sc.dt))
    t_end = args.until if args.until else args.max_t
    tail, n, wall = 0, 0, time.time()
    while tail < 3 * fps and sc.t < t_end:
        sc.advance(steps)
        video.write(comp.frame(sc, 1.0 / fps, args.speed, False, True))
        n += 1
        if n % (fps * 5) == 0:
            print(f"t={sc.t:6.1f}s {sc.pilot.state:9s} err {sc.err_now():6.1f} m  "
                  f"({n / max(time.time() - wall, 1e-6):.1f} frames/s rendered)", flush=True)
        if sc.done:
            tail += 1
    video.close()
    comp.close()
    print(sc.summary())
    print(f"video: {out}  ({n} frames, {n / fps:.1f} s)")
    return 0


# ---------------------------------------------------------------------------------------- viewer

def map_half(args: argparse.Namespace) -> float:
    return max(700.0, args.range + 400.0)


def key_of(code: int) -> str:
    if code in KEYS:
        return KEYS[code]
    c = code & 0xFF
    if c in KEYS:
        return KEYS[c]
    return chr(c).upper() if 32 < c < 127 else ""


def run_viewer(world, args: argparse.Namespace) -> None:
    W, H = args.size
    model = world.build_model()
    sc = ReconScenario(world, args, model)
    comp = DashComposer(sc, W, H, map_half(args))
    comp.overview = args.overview
    udp = world.UdpBearings(args.udp_port)
    if args.bearing is not None:
        print(f"t={sc.t:6.1f}s {sc.base_bearing(args.bearing, '--bearing')}")
    keys: deque[str] = deque()
    mouse = {"drag": None}

    def on_mouse(event, x, y, flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["drag"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and mouse["drag"] is not None and flags & cv2.EVENT_FLAG_LBUTTON:
            x0, y0 = mouse["drag"]
            comp.chase.az_off -= 0.3 * (x - x0)
            comp.chase.el_off -= 0.2 * (y - y0)
            mouse["drag"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            mouse["drag"] = None
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = WHEEL_ZOOM_SIGN * _mouse_wheel_delta(flags)
            if delta != 0:
                comp.chase.zoom = float(np.clip(comp.chase.zoom * (0.9 if delta > 0 else 1.1), 0.3, 6.0))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(WINDOW, W, H)
    cv2.setMouseCallback(WINDOW, on_mouse)
    fullscreen = args.fullscreen
    if fullscreen:
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    speed, paused = args.speed, False
    video: _VideoOut | None = None
    rec_t0, rec_n, rec_path = 0.0, 0, None
    wall0 = last_wall = time.perf_counter()
    budget = 0.0
    fps_t, fps_n, fps = time.perf_counter(), 0, 0.0
    acc = {"sim": 0.0, "view": 0.0, "cams": 0.0, "hud": 0.0}
    last_dec = 0
    snap = False
    while True:
        if args.quit_after and time.perf_counter() - wall0 > args.quit_after:
            break
        az = udp.pop()
        if az is not None:
            print(f"t={sc.t:6.1f}s UDP: {sc.base_bearing(az, f'UDP {args.udp_port}')}", flush=True)
        while keys:
            k = keys.popleft()
            if k == "LEFT":
                sc.pending = (sc.pending - 5.0) % 360.0
            elif k == "RIGHT":
                sc.pending = (sc.pending + 5.0) % 360.0
            elif k == "ENTER":
                print(f"t={sc.t:6.1f}s KEY ENTER: {sc.base_bearing(sc.pending, 'keys')}", flush=True)
            elif k == "F":
                print(f"t={sc.t:6.1f}s KEY F: {sc.base_bearing(random.uniform(0.0, 360.0), 'random (F)')}", flush=True)
            elif k == "N":
                print(f"t={sc.t:6.1f}s KEY N: {sc.toggle_gnss()}", flush=True)
            elif k == "C":
                comp.overview = not comp.overview
            elif k == "O":
                comp.chase.reset_offsets()
            elif k in ("1", "2", "4"):
                speed = float(k)
            elif k == "SPACE":
                paused = not paused
            elif k == "Z":
                snap = True
            elif k == "TAB":
                fullscreen = not fullscreen
                cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            elif k == "V":
                if video is None:
                    rec_path = data_dir() / "demo" / f"recon_dash_{datetime.now():%Y%m%d_%H%M%S}.mp4"
                    rec_path.parent.mkdir(parents=True, exist_ok=True)
                    video, rec_t0, rec_n = _VideoOut(rec_path, W, H, 30), time.perf_counter(), 0
                    print(f"t={sc.t:6.1f}s RECORDING -> {rec_path}", flush=True)
                else:
                    video.close()
                    video = None
                    print(f"t={sc.t:6.1f}s recording stopped: {rec_path} ({rec_n / 30:.1f} s)", flush=True)
            elif k == "BACKSPACE":
                print(f"t={sc.t:6.1f}s RESET", flush=True)
                sc = ReconScenario(world, args, model)
                comp.reset(sc)
                budget, last_dec = 0.0, 0
            elif k == "Q":
                if video is not None:
                    video.close()
                comp.close()
                cv2.destroyAllWindows()
                return
        now = time.perf_counter()
        wall_dt, last_wall = min(now - last_wall, 0.1), now  # cap: a stall must not cause a burst
        if not paused:
            budget += wall_dt * speed
            n = int(budget / sc.dt)
            budget -= n * sc.dt
            t_sim = time.perf_counter()
            sc.advance(n)
            acc["sim"] += time.perf_counter() - t_sim
        while last_dec < len(sc.pilot.decisions):
            t, text = sc.pilot.decisions[last_dec]
            last_dec += 1
            print(f"t={t:6.1f}s DRONE {text}", flush=True)
        frame = comp.frame(sc, wall_dt, speed, paused, video is not None)
        for k in ("view", "cams", "hud"):
            acc[k] += comp.times[k]
        if video is not None:  # wall-paced: the MP4 plays at the speed the window showed
            due = int((time.perf_counter() - rec_t0) * 30) + 1
            for _ in range(min(max(due - rec_n, 0), 4)):
                video.write(frame)
                rec_n += 1
            rec_n = max(rec_n, due - 4)
        if snap:
            snap = False
            p = data_dir() / "demo" / f"recon_dash_t{sc.t:05.1f}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(f"snapshot: {p}", flush=True)
        cv2.imshow(WINDOW, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        code = cv2.waitKeyEx(1)
        if code != -1:
            keys.append(key_of(code))
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        fps_n += 1
        t_fps = time.perf_counter()
        if t_fps - fps_t > 2.0:
            fps = fps_n / (t_fps - fps_t)
            ms = {k: 1000.0 * v / fps_n for k, v in acc.items()}
            print(f"t={sc.t:6.1f}s {sc.pilot.state:9s} {fps:4.1f} fps  sim {ms['sim']:5.1f}  view {ms['view']:5.1f}  "
                  f"cams {ms['cams']:5.1f}  hud {ms['hud']:5.1f} ms   err {sc.err_now():5.1f} m", flush=True)
            fps_t, fps_n = t_fps, 0
            acc = {k: 0.0 for k in acc}
    if video is not None:
        video.close()
    comp.close()
    cv2.destroyAllWindows()


def _size(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", choices=sorted(WORLDS), default=DEFAULT_WORLD)
    ap.add_argument("--bearing", type=float, default=None, help="launch right away toward this bearing (deg)")
    ap.add_argument("--range", type=float, default=400.0, help="how far out along the bearing (m)")
    ap.add_argument("--gnss-off-at", type=float, default=None, metavar="M",
                    help="switch GNSS off after this many metres of (true, horizontal) flight, e.g. 50")
    ap.add_argument("--retarget-in-flight", action="store_true",
                    help="a new bearing reaches the drone in the air (as 50_base_3d; would need a data link)")
    ap.add_argument("--udp-port", type=int, default=5005, help="0 = off")
    ap.add_argument("--seed", type=int, default=0, help="sensor noise seed")
    ap.add_argument("--speed", type=float, default=2.0, help="sim speed (viewer: 1 / 2 / 4 keys; record: any)")
    ap.add_argument("--headless", action="store_true", help="no window: fly once, print, score")
    ap.add_argument("--record", nargs="?", const=str(data_dir() / "demo" / "recon_dash.mp4"), default=None,
                    help="render offscreen to an MP4 (default $SFHACK_DATA/demo/recon_dash.mp4)")
    ap.add_argument("--until", type=float, default=0.0, help="with --record: stop at this sim time")
    ap.add_argument("--max-t", type=float, default=600.0)
    ap.add_argument("--size", type=_size, default=None, help="frame size WxH (viewer 1280x720, video 1920x1080)")
    ap.add_argument("--overview", action="store_true", help="start in the overview camera (key C toggles)")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--quit-after", type=float, default=0.0, help=argparse.SUPPRESS)  # wall s, viewer smoke test
    args = ap.parse_args()
    if args.size is None:
        args.size = RECORD_SIZE if args.record else VIEW_SIZE
    world = load_world(args.world)
    if args.headless:
        sys.exit(run_headless(world, args))
    if args.record:
        sys.exit(run_record(world, args))
    run_viewer(world, args)


if __name__ == "__main__":
    main()
