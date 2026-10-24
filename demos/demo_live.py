"""Live demo: the default Berkeley mission A -> B, flown on the onboard estimate (EKF + AHRS + vision fixes,
no truth in the control loop), with every degradation lever live.

    python demos/demo_live.py                         # viewer window (OpenCV), demo seed + vision localizer
    python demos/demo_live.py --fullscreen            # same, fullscreen (Tab toggles)
    python demos/demo_live.py --scenario-at 20        # play the spoof scenario at t = 20 s (also key P)
    python demos/demo_live.py --record out.mp4 --scenario-at 20   # same view, offscreen, to MP4
    python demos/demo_live.py --headless              # no window: run, print, score (verification)
    python demos/demo_live.py --gnss drag:2@90 --weather fog:0.5 --wind 6@270   # start with levers set
    python demos/demo_live.py --events "45:C,55:C"    # press keys at sim times
    python demos/demo_live.py --mission berkeley      # a mission from the planner (python -m sfhack.missionprep)

What you see:
  Solid white X2 = TRUTH (the simulator), drawn 9x life size or more. Cyan ghost = where the drone THINKS it is (EKF),
  cyan bubble = its claimed 2-sigma. White trail = true path, cyan trail = estimated path. A green, B red.
  Yellow frustum = down camera (live image bottom-left), red beams = rangefinder / lidar returns,
  orange pin = GNSS fix. A lost sensor goes dark. Fog / rain / dust and the wind arrow follow the levers.

Keys:  G GNSS ok/drag/jump/jam/off   F weather intensity   T weather kind   W wind   D wind direction
       U gusts   C camera loss   L lidar loss   V localizer on/off   P spoof scenario   X reset levers
       R restart   SPACE pause   1 / 2 / 4 sim speed   O reset camera   Z screenshot   Tab fullscreen   Q quit
Mouse: drag to orbit, wheel to zoom, click a lever row to press its key.

Truth is used here only to DISPLAY and SCORE (harness code); the autopilot never sees it, except through
the MOCK localizer when that one is selected (and labeled MOCK on screen).
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mujoco
import numpy as np

from liveview.chase import ChaseCam
from liveview.geoms import Geoms
from liveview.hud import Hud, HudState
from liveview.levers import LEVERS, LeverDeck
from liveview.render import SceneRenderer
from sfhack import bus as topics
from sfhack.config import Config, data_dir, default_config
from sfhack.degradation import Conditions
from sfhack.degradation.cli import add_cli_args, conditions_from_args, parse_schedule
from sfhack.degradation.models import gnss_sigma2_m
from sfhack.runner import FlightSystem
from sfhack.sim.mujoco_sim import announce_scene_compile

DEMO_SEED = 0
DEMO_LOCALIZER = "vision"  # 10/10 seeds within 3 m of B
NAV_LABEL = {
    "mock": "MOCK fixes (truth + 2 m noise)",
    "vision": "VISION fixes (camera vs onboard map)",
    "off": "DEAD RECKONING (localizer off)",
}
TRAIL_DT = 0.25  # s between trail points
DRONE_SCALE = 9.0  # the X2 is 0.4 m across; drawn 9x (more when the camera pulls back) to read across a room

CYAN = (0.0, 0.85, 1.0)
WHITE = (1.0, 1.0, 1.0)
GREEN = (0.15, 0.85, 0.2)
RED = (0.95, 0.15, 0.1)
ORANGE = (1.0, 0.55, 0.1)
YELLOW = (1.0, 0.85, 0.1)
LASER = (1.0, 0.18, 0.15)
DARK = (0.18, 0.18, 0.2)
WIND_RGB = (0.1, 0.45, 1.0)

VIEW_SIZE = (1280, 720)  # live window. --record stays 1920x1080.
RECORD_SIZE = (1920, 1080)
# Positive OpenCV wheel delta zooms in. Set to -1 if a trackpad's natural scrolling feels backwards.
WHEEL_ZOOM_SIGN = 1


def _rgb255(c) -> tuple[int, int, int]:
    return tuple(int(255 * v) for v in c[:3])  # type: ignore[return-value]


def _decimate(pts: list[np.ndarray], max_n: int) -> np.ndarray:
    a = np.array(pts)
    if len(a) > max_n:
        idx = np.unique(np.r_[np.linspace(0, len(a) - 1, max_n).astype(int), len(a) - 1])
        a = a[idx]
    return a


def _mouse_wheel_delta(flags: int) -> int:
    """Wheel delta from the high 16 bits of the mouse flags. `cv2.getMouseWheelDelta` is that read;
    some Python builds omit the inline helper."""
    fn = getattr(cv2, "getMouseWheelDelta", None)
    if fn is not None:
        return int(fn(int(flags)))
    return int(np.int16(np.uint16((int(flags) >> 16) & 0xFFFF)))


def _label(img: np.ndarray, text: str, org: tuple[int, int], scale: float, color=(255, 255, 255)) -> None:
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 6, y - th - 6), (x + tw + 6, y + base + 4), (10, 14, 22), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------------------- the demo

class LiveDemo:
    """One flight plus everything the jury sees. Truth: display + score only."""

    def __init__(self, seed: int, localizer: str, conditions: Conditions,
                 schedule: list | None = None, reuse: FlightSystem | None = None, mission: str | None = None,
                 cfg: Config | None = None) -> None:
        self.seed = seed
        self.mission = mission
        self.base_localizer = localizer if localizer != "off" else "mock"
        self.fs = FlightSystem(cfg=cfg, pose_source="ekf", localizer=localizer, seed=seed,
                               reuse_sim=reuse.sim if reuse is not None else None,
                               conditions=conditions, schedule=list(schedule or []), mission=mission)
        self.loc_on = localizer != "off"
        self.deck = LeverDeck(self.fs)
        self.truth_trail: list[np.ndarray] = []
        self.est_trail: list[np.ndarray] = []
        self._next_trail = 0.0
        self.cam_img = np.zeros((240, 320, 3), np.uint8)
        self.cam_t = -1.0
        self.range_m = math.nan
        self.range_t = -1.0
        self.lidar_ranges: np.ndarray | None = None
        self.lidar_dirs = np.zeros((0, 3))
        self.lidar_t = -1.0
        self.gnss_fixes: deque[np.ndarray] = deque(maxlen=30)
        self.gnss_t = -1.0
        self.done_at: float | None = None
        self.result = None
        bus = self.fs.bus
        bus.subscribe(topics.CAMERA_DOWN, self._on_frame)
        bus.subscribe(topics.RANGE, self._on_range)
        bus.subscribe(topics.GNSS, self._on_gnss)
        bus.subscribe(topics.LIDAR, self._on_lidar)
        self.goal = self.fs.goal_xy
        self.start = self.fs.onboard_map.a_enu if self.fs.onboard_map is not None else np.zeros(2)
        self.route = np.array([w.pos for w in self.fs.mission.waypoints])
        # display only: A / B pads and labels sit on the true terrain (0 in the flat sample world)
        self.start_z = self.fs.true_ground_z(*self.start)
        self.goal_z = self.fs.true_ground_z(*self.goal)
        self.cam_id = self.fs.sim.model.camera("down").id
        cc = self.fs.cfg.camera
        self.cam_tan = (math.tan(math.radians(cc.down_fovy_deg) / 2) * cc.down_width / cc.down_height,
                        math.tan(math.radians(cc.down_fovy_deg) / 2))
        rng = np.random.default_rng(7)
        self.wind_pts = rng.uniform(-1, 1, (24, 3)) * [45, 45, 18]
        self.rain_pts = rng.uniform(-1, 1, (80, 3)) * [30, 30, 16]
        self.draw_scale = DRONE_SCALE

    # ------------------------------------------------------------------ inputs (bus = what the sensors deliver)

    def _on_frame(self, frame) -> None:
        self.cam_img = frame.image
        self.cam_t = frame.t

    def _on_range(self, msg) -> None:
        self.range_m, self.range_t = msg.range, msg.t

    def _on_gnss(self, msg) -> None:
        self.gnss_fixes.append(np.asarray(msg.pos, float).copy())
        self.gnss_t = msg.t

    def _on_lidar(self, msg) -> None:
        a = np.asarray(msg.angles, float)
        self.lidar_dirs = np.column_stack([np.zeros_like(a), -np.sin(a), -np.cos(a)])  # body FLU, + = right
        self.lidar_ranges = np.asarray(msg.ranges, float)
        self.lidar_t = msg.t

    @property
    def t(self) -> float:
        return self.fs.t

    @property
    def c(self) -> Conditions:
        return self.deck.c

    def press(self, key: str) -> str:
        key = key.lower()
        if key == "v":
            self.loc_on = not self.loc_on
            self.fs.set_localizer(self.base_localizer if self.loc_on else "off")
            msg = f"localizer {'ON' if self.loc_on else 'OFF (dead reckoning)'}"
            self.deck.banner = (self.t, msg.upper())
            return msg
        return self.deck.press(key, self.t)

    # ------------------------------------------------------------------ stepping

    def advance(self, n_steps: int) -> None:
        fs = self.fs
        for _ in range(n_steps):
            if self.result is not None:
                return
            self.deck.tick(self.t)
            fs.step()
            if self.t >= self._next_trail:
                self._next_trail = self.t + TRAIL_DT
                self.truth_trail.append(fs.truth.state().pos.copy())
                self.est_trail.append(fs.autopilot.pose().pos.copy())
            if fs.done and self.done_at is None:
                self.done_at = self.t
            if (self.done_at is not None and self.t - self.done_at >= 1.0) or self.t >= fs.cfg.sim.max_time_s:
                self.result = fs.result()  # harness: scored on truth
                fs.autopilot.log_decision(self.t, f"{self.result.outcome}, final error {self.result.final_error:.1f} m")

    def close(self) -> None:
        self.fs.close()

    # ------------------------------------------------------------------ sensor display state

    def camera_dark(self) -> bool:
        return self.c.camera.lost or self.t - self.cam_t > 1.0 or not self.cam_img.any()

    def lidar_dark(self) -> bool:
        stale = self.t - max(self.range_t, self.lidar_t) > 0.5
        return stale or self.c.lidar.lost

    def gnss_state(self) -> str:
        """off (no receiver) | jam (no fix) | ok | spoof (fixes arrive, but the attacker moves them)."""
        mode = self.c.gnss.mode
        if mode == "off":
            return "off"
        if not self.gnss_fixes or self.t - self.gnss_t > 1.5 or mode == "jam":
            return "jam"
        return "spoof" if mode in ("drag", "jump") else "ok"

    # ------------------------------------------------------------------ 3D

    def update_particles(self, dt: float, center: np.ndarray, cam_dist: float = 36.0) -> None:
        w = self.c.wind
        ux, uy = math.sin(math.radians(w.direction_deg)), math.cos(math.radians(w.direction_deg))
        wind = np.array([w.speed * ux, w.speed * uy, 0.0])
        half = np.array([45.0, 45.0, 18.0])
        self.wind_pts += wind * dt * 1.5
        self.wind_pts = (self.wind_pts + half) % (2 * half) - half
        rhalf = np.array([30.0, 30.0, 16.0])
        self.rain_pts += (np.array([0.0, 0.0, -14.0]) + 0.5 * wind) * dt
        self.rain_pts = (self.rain_pts + rhalf) % (2 * rhalf) - rhalf
        self._center = center
        self.draw_scale = float(np.clip(cam_dist / 4.0, DRONE_SCALE, 30.0))

    def draw_geoms(self, scn: mujoco.MjvScene) -> None:
        g = Geoms(scn)
        fs, d = self.fs, self.fs.sim.data
        est, tr = fs.autopilot.pose(), fs.truth.state()
        # start / goal pads with light columns, planned route
        for xy, z0, rgb in ((self.start, self.start_z, GREEN), (self.goal, self.goal_z, RED)):
            g.cylinder([xy[0], xy[1], z0 + 0.08], 3.5, 0.06, rgb, 0.95, emission=0.3)
            g.cylinder([xy[0], xy[1], z0 + 30.0], 0.35, 30.0, rgb, 0.22, emission=0.6)
        if len(self.route) > 1:
            g.stroke(self.route, WHITE, 3.0, 0.55)  # one line per leg, not a capsule per dash
        # trails
        if len(self.truth_trail) > 1:
            g.stroke(_decimate(self.truth_trail, 40), WHITE, 4.0, 0.95)
            g.stroke(_decimate(self.est_trail, 40), CYAN, 4.0, 0.9)
        # the true drone and its ghost
        scale = self.draw_scale
        g.drone(tr.pos, tr.R, scale, (0.93, 0.93, 0.96), (0.12, 0.12, 0.14), 1.0, nose_rgb=ORANGE)
        g.drone(est.pos, est.R, scale, CYAN, CYAN, 0.38)
        cov = est.cov[:3, :3]
        evals, evecs = np.linalg.eigh(0.5 * (cov[:2, :2] + cov[:2, :2].T))
        # Same 2σ the HUD prints (filter posterior). No 0.8 m floor: a tight claim stays tight.
        radii = np.clip(2.0 * np.sqrt(np.maximum(evals, 0.0)), 0.05, 400.0)
        rz = float(np.clip(2.0 * math.sqrt(max(cov[2, 2], 0.0)), 0.05, 60.0))
        mat = np.eye(3)
        mat[:2, :2] = evecs
        if np.linalg.det(mat) < 0:
            mat[:, 0] *= -1
        g.ellipsoid(est.pos, [radii[0], radii[1], rz], mat, CYAN, 0.13)
        s2 = est.sigma2_horizontal()
        if s2 > 0.05:
            g.line_ring([est.pos[0], est.pos[1], 0.25], float(s2), CYAN, 0.85, width=4.0, n=24)
            # Above the scaled rotors, same radius: a 1 m claim reads on the body instead of inside it.
            halo = np.array([est.pos[0], est.pos[1], est.pos[2] + 0.22 * scale])
            g.line_ring(halo, float(s2), (0.75, 1.0, 1.0), 1.0, width=6.0, n=24)
        g.line(est.pos, [est.pos[0], est.pos[1], 0.25], CYAN, 2.0, 0.6)
        g.line(tr.pos, [tr.pos[0], tr.pos[1], 0.25], WHITE, 2.0, 0.5)
        if np.linalg.norm(est.pos - tr.pos) > 1.5:
            g.line(tr.pos, est.pos, RED, 3.0, 0.9)
        self._draw_camera(g, d)
        self._draw_lidar(g, tr)
        self._draw_gnss(g)
        self._draw_wind(g, tr.pos)
        self._draw_weather(g)

    def _draw_camera(self, g: Geoms, d: mujoco.MjData) -> None:
        p = d.cam_xpos[self.cam_id].copy()  # display: where the camera physically is (truth)
        R = d.cam_xmat[self.cam_id].reshape(3, 3)
        tx, ty = self.cam_tan
        corners = []
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            ray = R @ np.array([sx * tx, sy * ty, -1.0])
            t = min(-p[2] / ray[2], 250.0) if ray[2] < -1e-6 else 250.0
            corners.append(p + t * ray)
        dark = self.camera_dark()
        rgb, a, r = (DARK, 0.7, 0.05) if dark else (YELLOW, 0.9, 0.07)
        for c in corners:
            g.capsule(p, c, r, rgb, a, emission=0.0 if dark else 0.5)
        ring = np.array(corners + corners[:1])
        ring[:, 2] = np.maximum(ring[:, 2], 0.3)
        if not dark:
            g.tube(ring, 0.18, YELLOW, 0.95, emission=0.5)

    def _draw_lidar(self, g: Geoms, tr) -> None:
        if self.lidar_dark():
            g.capsule(tr.pos, tr.pos - np.array([0, 0, 2.5]), 0.08, DARK, 0.8)
            return
        down = tr.R @ np.array([0.0, 0.0, -1.0])
        beams: list[tuple[np.ndarray, float]] = [(down, self.range_m)]
        if self.lidar_ranges is not None and self.t - self.lidar_t < 0.5:
            dirs = (tr.R @ self.lidar_dirs.T).T
            beams += list(zip(dirs, self.lidar_ranges))
        max_r = self.fs.cfg.noise.range_max
        for u, rng_m in beams:
            if math.isfinite(rng_m):
                hit = tr.pos + u * rng_m
                g.capsule(tr.pos, hit, 0.07, LASER, 0.95, emission=0.9)
                g.sphere(hit, 0.45, LASER, 0.95, emission=0.9)
            else:
                g.dashed(tr.pos, tr.pos + u * max_r, 0.04, LASER, 0.35, dash=2.0)

    def _draw_gnss(self, g: Geoms) -> None:
        state = self.gnss_state()
        if not self.gnss_fixes or state == "off":
            return
        fixes = list(self.gnss_fixes)
        last = fixes[-1]
        rgb = DARK if state == "jam" else ORANGE
        tail = fixes[:-1][-6:]  # short faded tail; the pin and the ground ring stay
        n_tail = len(tail)
        for i, f in enumerate(tail):
            fade = (i + 1) / (n_tail + 1)
            g.sphere(f, 0.22, rgb, 0.12 + 0.40 * fade)
        g.sphere(last, 1.0, rgb, 0.95, emission=0.0 if state == "jam" else 0.6)
        # 2σ of the covariance the receiver reports, which is the noise the model injects.
        r2 = gnss_sigma2_m(self.fs.cfg.noise.gnss_noise)
        g.line_ring(last, r2, rgb, 0.9, width=5.0, n=28)
        g.line(last, [last[0], last[1], 0.3], rgb, 2.0, 0.6)
        g.line_ring([last[0], last[1], 0.3], r2, rgb, 0.75, width=4.0, n=28)

    def _draw_wind(self, g: Geoms, pos: np.ndarray) -> None:
        w = self.c.wind
        if w.speed <= 0 and w.gust <= 0:
            return
        u = np.array([math.sin(math.radians(w.direction_deg)), math.cos(math.radians(w.direction_deg)), 0.0])
        k = self.draw_scale / DRONE_SCALE
        L = (4.0 + 1.1 * w.speed) * k
        base = pos + np.array([0.0, 0.0, 8.0 * k])
        g.arrow(base - u * L / 2, base + u * L / 2, 0.9 * k, WIND_RGB, 0.95, emission=0.6)
        center = getattr(self, "_center", pos)
        streak = u * (0.6 + 0.35 * w.speed)
        for q in self.wind_pts:
            p = center + q
            if p[2] > 1.0:
                g.line(p, p + streak, (0.92, 0.96, 1.0), 2.0, 0.55)

    def _draw_weather(self, g: Geoms) -> None:
        wv = self.c.weather
        if wv.kind not in ("rain", "dust") or wv.intensity <= 0:
            return
        center = getattr(self, "_center", None)
        if center is None:
            return
        n = int(len(self.rain_pts) * wv.intensity)
        if wv.kind == "rain":
            w = self.c.wind
            drift = 0.04 * w.speed * np.array([math.sin(math.radians(w.direction_deg)), math.cos(math.radians(w.direction_deg)), 0])
            for q in self.rain_pts[:n]:
                p = center + q
                if p[2] > 0.5:
                    g.line(p, p + np.array([0, 0, 1.6]) - drift, (0.8, 0.85, 0.95), 1.5, 0.6)
        else:
            for q in self.rain_pts[:n]:
                p = center + q * [1, 1, 0.5]
                if p[2] > 0.3:
                    g.sphere(p, 0.18, (0.72, 0.6, 0.42), 0.5)

    def labels(self, project) -> list:
        est, tr = self.fs.autopilot.pose(), self.fs.truth.state()
        out = []
        pt = project(tr.pos + np.array([0, 0, 2.0]))
        pe = project(est.pos + np.array([0, 0, 2.0]))
        if pt is not None and pe is not None and math.dist(pt, pe) < 60:
            mid = ((pt[0] + pe[0]) // 2, (pt[1] + pe[1]) // 2)
            out.append((mid, "TRUE = ESTIMATE", (255, 255, 255, 255)))
        else:
            if pt is not None:
                out.append((pt, "TRUE", (255, 255, 255, 255)))
            if pe is not None:
                out.append((pe, "ESTIMATE", (0, 217, 255, 255)))
        gs = self.gnss_state()
        if self.gnss_fixes and gs not in ("off",):
            pg = project(self.gnss_fixes[-1] - np.array([0, 0, 9.0]))  # below the pin; screen nudge separates overlaps
            # A spoof always keeps the GNSS label. An honest fix colocated with the true drone does not.
            if pg is not None and (gs == "spoof" or pt is None or math.dist(pg, pt) > 40):
                txt = "GNSS (no fix)" if gs == "jam" else "GNSS FIX"
                out.append((pg, txt, (255, 140, 30, 255) if gs != "jam" else (140, 140, 150, 255)))
        for xy, z0, name, col in ((self.start, self.start_z, "A", (70, 215, 120, 255)),
                                  (self.goal, self.goal_z, "B", (255, 75, 65, 255))):
            pp = project(np.array([xy[0], xy[1], z0 + 62.0]))
            if pp is not None:
                out.append((pp, name, col))
        return out

    # ------------------------------------------------------------------ HUD

    def hud_state(self, speed: int, paused: bool, recording: bool, labels: list) -> HudState:
        fs, ap, c = self.fs, self.fs.autopilot, self.c
        est, tr = ap.pose(), fs.truth.state()
        err = float(np.linalg.norm(est.pos[:2] - tr.pos[:2]))
        hs = ap.health_state
        nav = NAV_LABEL["off"] if not self.loc_on else NAV_LABEL.get(self.base_localizer, self.base_localizer)
        fix_age = "never" if ap.health.last_fix_t is None else f"{self.t - ap.health.last_fix_t:.1f} s ago"
        gs = self.gnss_state()
        if gs == "off":
            gnss_line, gnss_lvl = "no receiver (GNSS-free flight)", 1
        elif gs == "jam":
            gnss_line, gnss_lvl = "NO FIX (jammed)" if c.gnss.mode == "jam" else "no fix", 2
        else:
            off = float(np.linalg.norm(self.gnss_fixes[-1][:2] - tr.pos[:2]))
            gnss_line = f"fix {off:.0f} m from truth" + ("  (SPOOFED)" if gs == "spoof" else "")
            gnss_lvl = 2 if gs == "spoof" else 0
        degr: list[tuple[str, int]] = []
        for lv in LEVERS:
            lvl = lv.level(c)
            if lvl and lv.key not in ("t", "d"):
                degr.append((f"{lv.name.upper()} {lv.value(c)}", lvl))
        if not self.loc_on:
            degr.append(("LOCALIZER OFF", 2))
        levers = [(lv.key.upper(), lv.name, lv.value(c), lv.level(c)) for lv in LEVERS]
        controls = [("P", "Play spoof scenario", "RUNNING" if self.deck.scenario_t0 is not None else ""),
                    ("V", "Localizer", "ON" if self.loc_on else "OFF"),
                    ("X", "Reset all levers", "")]
        cam_state = "lost" if self.camera_dark() else ("degraded" if c.weather.kind != "clear" and c.weather.intensity > 0 else "ok")
        lid_state = "lost" if self.lidar_dark() else ("degraded" if c.weather.kind != "clear" and c.weather.intensity > 0 else "ok")
        sensors = [("CAMERA", cam_state), ("LIDAR", lid_state),
                   ("GNSS", {"off": "off", "jam": "lost", "spoof": "degraded", "ok": "ok"}[gs]),
                   ("IMU", "ok"), ("BARO", "ok"), ("COMPASS", "ok")]
        w = c.wind
        weather = ("CLEAR" if c.weather.kind == "clear" or c.weather.intensity == 0
                   else f"{c.weather.kind.upper()} {c.weather.intensity:.0%}  vis {fs.sim.weather_visuals().visibility_m:.0f} m")
        banner = None
        if self.deck.banner is not None:
            banner = (self.deck.banner[1], self.t - self.deck.banner[0])
        result = None
        if self.result is not None:
            r = self.result
            result = f"{r.outcome} \u00b7 {r.final_error:.0f} m from B"
        return HudState(
            t=self.t, phase=fs.mission.phase, health=hs.state.value,
            health_reason=hs.reason or "", err=err, sigma2=est.sigma2_horizontal(), nav_source=nav, fix_age=fix_age,
            gnss_line=gnss_line, gnss_level=gnss_lvl, degradations=degr, levers=levers, controls=controls,
            sensors=sensors, wind=(w.speed, w.direction_deg, w.gust), weather=weather, banner=banner,
            scenario=self.deck.scenario_progress(), result=result, speed=speed, paused=paused,
            recording=recording, labels=labels, place=f"mission {self.mission}" if self.mission else "Berkeley",
            sources_line=self.sources_line())

    def sources_line(self) -> str:
        """Age of the last fix each position source delivered to the estimate (TRN: its status until it does)."""
        ap, t = self.fs.autopilot, self.t

        def age(ts: float | None) -> str:
            return "-" if ts is None else (f"{t - ts:.1f}s" if t - ts < 10 else "lost")

        gt = ap.ekf.last_gnss_t
        vt = ap.last_fix.t if ap.last_fix is not None and ap.last_fix.accepted else None
        parts = [f"GNSS {'off' if gt is None else age(gt)}", f"VISION {age(vt)}"]
        if ap.trn is None:
            parts.append("TRN off")
        else:
            f = ap.last_trn_fix
            if f is not None and f.accepted and t - f.t < 10:
                parts.append(f"TRN {age(f.t)}")
            else:
                parts.append(f"TRN {ap.trn.status.split(' (')[0]}")
        return "   ".join(parts)

    def map_panel(self, size: int) -> np.ndarray:
        om = self.fs.onboard_map
        est, tr = self.fs.autopilot.pose(), self.fs.truth.state()
        s2 = est.sigma2_horizontal()
        pts = [est.pos[:2], tr.pos[:2]]
        if self.gnss_fixes and self.gnss_state() != "off":
            pts.append(self.gnss_fixes[-1][:2])
        pts = np.array(pts)
        center = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
        span = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
        half = max(60.0, 0.6 * span + 25.0, 1.2 * s2)
        cu, cv = om.enu_to_px(center)
        half_px = half / om.m_per_px
        k = size / (2 * half_px)
        M = np.array([[k, 0, -(cu - half_px) * k], [0, k, -(cv - half_px) * k]])
        img = cv2.warpAffine(om.ortho, M, (size, size), flags=cv2.INTER_AREA, borderValue=(60, 60, 60))
        img = (img * 0.8).astype(np.uint8)
        lw = max(1, size // 150)

        def px(xy) -> tuple[int, int]:
            u, v = om.enu_to_px(np.asarray(xy, float)[:2])
            return int(round((u - (cu - half_px)) * k)), int(round((v - (cv - half_px)) * k))

        def trail(pts_, color) -> None:
            if len(pts_) > 1:
                p = np.array([px(q) for q in _decimate(pts_, 300)], np.int32)
                cv2.polylines(img, [p], False, color, lw + 1, cv2.LINE_AA)

        trail(self.truth_trail, (255, 255, 255))
        trail(self.est_trail, _rgb255(CYAN))
        for xy, col in ((self.start, GREEN), (self.goal, RED)):
            cv2.circle(img, px(xy), max(4, int(3.5 / om.m_per_px * k)), _rgb255(col), lw + 1, cv2.LINE_AA)
        fix = self.fs.autopilot.last_fix
        if fix is not None and self.t - fix.t < 1.0:
            cv2.drawMarker(img, px(fix.pos), (255, 220, 0) if fix.accepted else (255, 60, 60),
                           cv2.MARKER_TILTED_CROSS, 6 * lw + 6, lw + 1)
        if self.gnss_fixes and self.gnss_state() != "off":
            col = (255, 140, 30) if self.gnss_state() != "jam" else (120, 120, 120)
            cv2.circle(img, px(self.gnss_fixes[-1]), 3 * lw + 3, col, -1, cv2.LINE_AA)
        cv2.circle(img, px(est.pos), max(3, int(s2 / om.m_per_px * k)), _rgb255(CYAN), lw, cv2.LINE_AA)
        cv2.circle(img, px(tr.pos), 3 * lw + 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, px(est.pos), 2 * lw + 3, _rgb255(CYAN), -1, cv2.LINE_AA)
        bar_m = 20.0 if half < 100 else (50.0 if half < 250 else 100.0)
        bar_px = int(bar_m / om.m_per_px * k)
        fs_ = size / 400
        cv2.line(img, (10, size - 14), (10 + bar_px, size - 14), (255, 255, 255), lw + 1)
        _label(img, f"{bar_m:.0f} m", (16 + bar_px, size - 9), 0.5 * fs_ * 1.2)
        cv2.rectangle(img, (0, 0), (size, int(30 * fs_)), (10, 14, 22), -1)
        _label(img, "ONBOARD MAP  (orthophoto + OSM)", (10, int(21 * fs_)), 0.55 * fs_ * 1.2)
        cv2.rectangle(img, (0, 0), (size - 1, size - 1), (255, 255, 255), 1)
        return img

    def cam_panel(self, w: int, h: int) -> np.ndarray:
        img = self.cam_img
        sx, sy = w / img.shape[1], h / img.shape[0]
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        fs_ = h / 300
        dark = self.camera_dark()
        fix = self.fs.autopilot.last_fix
        fresh = fix is not None and self.t - fix.t < 0.5 and not dark
        if fresh and fix.matched_px:
            pts = np.array([(u * sx, v * sy) for u, v in fix.matched_px], np.int32)
            color = (255, 220, 0) if fix.accepted else (255, 60, 60)
            if fix.source == "vision":
                cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
            else:
                for p in pts:
                    cv2.circle(img, tuple(int(q) for q in p), 8, color, 2, cv2.LINE_AA)
        if dark:
            img = (img * 0.15).astype(np.uint8)
            txt = "CAMERA LOST"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 1.3 * fs_, 2)
            cv2.putText(img, txt, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_DUPLEX, 1.3 * fs_, (255, 70, 60), 2, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (w, int(30 * fs_)), (10, 14, 22), -1)
        _label(img, "DOWN CAMERA  (live)", (10, int(21 * fs_)), 0.55 * fs_ * 1.2)
        if fresh:
            what = f"MATCH  NCC {fix.n_inliers / 100:.2f}" if fix.source == "vision" else f"{fix.n_inliers} landmarks"
            _label(img, f"{what}  {'accepted' if fix.accepted else 'REJECTED'}", (10, h - 12), 0.55 * fs_ * 1.2,
                   (255, 220, 0) if fix.accepted else (255, 80, 80))
        elif not dark:
            _label(img, "no fix", (10, h - 12), 0.55 * fs_ * 1.2, (200, 200, 200))
        border = (255, 70, 60) if dark else (255, 215, 30)
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), border, max(2, int(3 * fs_)))
        return img


# ---------------------------------------------------------------------------------------- one frame

class Composer:
    """Renders the full demo frame (3D + weather + HUD + panels). The viewer and the recorder share it, so
    the MP4 is the same picture as the window."""

    def __init__(self, demo: LiveDemo, width: int, height: int) -> None:
        self.W, self.H = width, height
        self.sr = SceneRenderer(demo.fs.sim.model, width, height)
        self.hud = Hud(width, height)
        self.fovy = float(demo.fs.sim.model.vis.global_.fovy)
        self.chase = self._new_chase(demo)
        self.times = {"view": 0.0, "fog": 0.0, "hud": 0.0, "map": 0.0}

    def _clear_aim(self) -> tuple | None:
        """Pixel where the truth/estimate midpoint should sit: above the bottom-left block, left of the levers."""
        p = self.hud.px
        m = p(16)
        # Inset so the pair's labels stay off the lever column and the bottom-left block.
        x0 = m + p(Hud.COL_W) + p(24)
        x1 = self.W - m - p(360) - p(80)
        y0 = m + p(100)
        y1 = self.H - m - p(Hud.MAP) - p(90)
        if x1 - x0 < 80 or y1 - y0 < 80:
            return None
        return ((x0 + x1) * 0.5, (y0 + y1) * 0.5, self.W, self.H, self.fovy)

    @staticmethod
    def _new_chase(demo: LiveDemo) -> ChaseCam:
        h = demo.goal - demo.start
        return ChaseCam(math.degrees(math.atan2(h[1], h[0])), np.array([demo.start[0], demo.start[1], demo.start_z + 1.0]))

    def reset(self, demo: LiveDemo) -> None:
        old = self.chase
        self.chase = self._new_chase(demo)
        self.chase.az_off, self.chase.el_off, self.chase.zoom = old.az_off, old.el_off, old.zoom

    def frame(self, demo: LiveDemo, dt: float, speed: int, paused: bool, recording: bool) -> np.ndarray:
        tr, est = demo.fs.truth.state(), demo.fs.autopilot.pose()
        cam = self.chase.update(dt, tr.pos, tr.vel, est.pos, aim=self._clear_aim())
        demo.update_particles(0.0 if paused else dt * speed, np.array(cam.lookat), cam.distance)
        wv = demo.fs.sim.weather_visuals()
        vis = wv.visibility_m if wv.kind != "clear" and wv.intensity > 0 else None
        # Fog thickens from just in front of the drone and its ghost outward: both stay readable, the world
        # fades with the weather lever (the camera itself sits 30-150 m behind the drone, in the same fog).
        sep = float(np.linalg.norm(est.pos[:2] - tr.pos[:2]))
        fog_start = wv.fog_start_m + 0.85 * cam.distance + 0.8 * sep
        img = self.sr.render(demo.fs.sim.data, cam, demo.draw_geoms, vis, wv.fog_rgb, fog_start)
        st = demo.hud_state(speed, paused, recording, demo.labels(self.sr.project))
        p = self.hud.px
        t_map = time.perf_counter()
        map_img = demo.map_panel(p(Hud.MAP))
        t_cam = time.perf_counter()
        cam_img = demo.cam_panel(p(Hud.CAM_W), p(Hud.CAM_H))
        t_hud = time.perf_counter()
        out = self.hud.compose(img, st, map_img, cam_img)
        t_end = time.perf_counter()
        self.times = {"view": self.sr.t_view, "fog": self.sr.t_fog,
                      "map": t_cam - t_map, "hud": (t_hud - t_cam) + (t_end - t_hud)}
        return out

    def close(self) -> None:
        self.sr.close()


def parse_events(spec: str) -> list[tuple[float, str]]:
    """'45:C,70:C' -> [(45.0, 'C'), (70.0, 'C')], sorted by time."""
    out = []
    for item in filter(None, (s.strip() for s in spec.split(","))):
        t, k = item.split(":")
        out.append((float(t), k.strip().upper()))
    return sorted(out)


def _events(args) -> list[tuple[float, str]]:
    ev = parse_events(args.events)
    if args.scenario_at is not None:
        ev = sorted(ev + [(args.scenario_at, "P")])
    return ev


def _make_demo(args, reuse: LiveDemo | None = None) -> LiveDemo:
    cfg = default_config()
    cfg.trn.enabled = args.trn == "on"
    return LiveDemo(args.seed, args.localizer, args.conditions, parse_schedule(args.cond_at) if args.cond_at else None,
                    reuse=reuse.fs if reuse is not None else None, mission=args.mission, cfg=cfg)


# ---------------------------------------------------------------------------------------- viewer mode

WINDOW = "GNSS Spoof Guard - live demo"


def _timing_line(acc: dict[str, float], n: int) -> str:
    if n <= 0:
        return ""
    def ms(k: str) -> float:
        return 1000.0 * acc[k] / n
    return (f"sim {ms('sim'):5.1f}  view {ms('view'):5.1f}  fog {ms('fog'):5.1f}  "
            f"hud {ms('hud'):5.1f}  map {ms('map'):5.1f} ms")


def run_viewer(args: argparse.Namespace) -> None:
    W, H = args.size
    announce_scene_compile()
    demo = _make_demo(args)
    comp = Composer(demo, W, H)
    keys: deque[str] = deque()
    mouse = {"drag": None}

    def on_mouse(event, x, y, flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            for x0, y0, x1, y1, key in comp.hud.hits:
                if x0 <= x <= x1 and y0 <= y <= y1:
                    keys.append(key)
                    return
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
    events = _events(args)
    pending = list(events)
    wall0 = time.time()
    last_wall = time.perf_counter()
    sim_budget = 0.0
    next_log = 0.0
    reported = False
    fps_t, fps_n, fps = time.perf_counter(), 0, 0.0
    acc = {"sim": 0.0, "view": 0.0, "fog": 0.0, "hud": 0.0, "map": 0.0}
    acc_n = 0
    snap_dir = Path(args.snap_dir)
    snaps = sorted(float(x) for x in args.snap_at.split(",") if x.strip())
    while True:
        if args.quit_after and time.time() - wall0 > args.quit_after:
            break
        while pending and demo.t >= pending[0][0]:
            keys.append(pending.pop(0)[1])
        while keys:
            k = keys.popleft()
            if k in ("SPACE", " "):
                paused = not paused
            elif k in ("1", "2", "4"):
                speed = int(k)
            elif k.upper() == "R":
                print(f"t={demo.t:6.1f}s RESTART")
                old = demo
                demo = _make_demo(args, reuse=old)
                old.close()
                comp.reset(demo)
                reported, next_log, sim_budget = False, 0.0, 0.0
                pending = [e for e in events if e[1] != "R"]
            elif k.upper() == "O":
                comp.chase.reset_offsets()
            elif k.upper() == "Q":
                demo.close()
                comp.close()
                cv2.destroyAllWindows()
                return
            elif k == "TAB":
                fullscreen = not fullscreen
                cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            elif k.upper() == "Z":
                snaps.insert(0, -1.0)
            else:
                msg = demo.press(k)
                if msg:
                    print(f"t={demo.t:6.1f}s KEY {k.upper()}: {msg}", flush=True)
        now = time.perf_counter()
        wall_dt, last_wall = min(now - last_wall, 0.1), now  # cap: a stall must not cause a burst
        if not paused:
            sim_budget += wall_dt * speed
            n = int(sim_budget / demo.fs.sim.model.opt.timestep)
            sim_budget -= n * demo.fs.sim.model.opt.timestep
            t_sim = time.perf_counter()
            demo.advance(n)
            acc["sim"] += time.perf_counter() - t_sim
            if demo.t >= next_log:
                next_log = demo.t + 5.0
                print(f"t={demo.t:6.1f}s {demo.fs.mission.phase:14s} {demo.c.summary()}  view {fps:4.1f} fps", flush=True)
            if demo.result is not None and not reported:
                reported = True
                print(f"t={demo.t:6.1f}s {demo.result.summary()}")
        frame = comp.frame(demo, wall_dt, speed, paused, False)
        for k in ("view", "fog", "hud", "map"):
            acc[k] += comp.times[k]
        acc_n += 1
        if snaps and demo.t >= snaps[0]:
            snaps.pop(0)
            snap_dir.mkdir(parents=True, exist_ok=True)
            out = snap_dir / f"viewer_t{demo.t:05.1f}.png"
            cv2.imwrite(str(out), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(f"snapshot: {out}")
        cv2.imshow(WINDOW, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        code = cv2.waitKeyEx(1)
        if code != -1:
            keys.append(_key_name(code))
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        fps_n += 1
        t_fps = time.perf_counter()
        if t_fps - fps_t > 2.0:
            fps = fps_n / (t_fps - fps_t)
            print(f"t={demo.t:6.1f}s {demo.fs.mission.phase:14s}  {fps:4.1f} fps  {_timing_line(acc, acc_n)}",
                  flush=True)
            fps_t, fps_n = t_fps, 0
            acc = {k: 0.0 for k in acc}
            acc_n = 0
    demo.close()
    comp.close()
    cv2.destroyAllWindows()


def _key_name(code: int) -> str:
    c = code & 0xFF
    if c == 32:
        return "SPACE"
    if c == 9:
        return "TAB"
    if c == 27:
        return "Q"
    return chr(c).upper() if 32 < c < 127 else ""


# ---------------------------------------------------------------------------------------- headless / record

def run_headless(args: argparse.Namespace) -> int:
    demo = _make_demo(args)
    pending = _events(args)
    wall = time.time()
    next_print = 0.0
    while demo.result is None:
        while pending and demo.t >= pending[0][0]:
            t, k = pending.pop(0)
            print(f"t={demo.t:6.1f}s KEY {k}: {demo.press(k)}")
        demo.advance(50)
        if demo.t >= next_print:
            next_print += 10.0
            est, tr = demo.fs.autopilot.pose(), demo.fs.truth.state()
            err = float(np.linalg.norm(est.pos[:2] - tr.pos[:2]))
            print(f"t={demo.t:6.1f}s {demo.fs.mission.phase:14s} {demo.fs.autopilot.health_state.state.value:10s} "
                  f"2-sigma {est.sigma2_horizontal():6.1f} m  err {err:6.1f} m  levers: {demo.c.summary()}")
    for t, text in demo.fs.autopilot.decisions:
        print(f"   {t:6.1f}s {text}")
    r = demo.result
    print(r.summary())
    print(f"wall {time.time() - wall:.1f} s")
    demo.close()
    return 0 if r.success else 1


class _VideoOut:
    """H.264 through ffmpeg when it is installed (plays everywhere), else OpenCV's mp4v."""

    def __init__(self, path: Path, W: int, H: int, fps: int) -> None:
        self.proc = None
        self.writer = None
        if shutil.which("ffmpeg"):
            self.proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                 "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "21",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
        else:
            self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    def write(self, rgb: np.ndarray) -> None:
        if self.proc is not None:
            self.proc.stdin.write(np.ascontiguousarray(rgb).tobytes())
        else:
            self.writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.writer.release()


def run_record(args: argparse.Namespace) -> int:
    out = Path(args.record)
    out.parent.mkdir(parents=True, exist_ok=True)
    W, H = args.size
    fps = 30
    demo = _make_demo(args)
    comp = Composer(demo, W, H)
    pending = _events(args)
    video = None if args.snap_only else _VideoOut(out, W, H, fps)
    dt_frame = 1.0 / fps
    steps = int(round(args.speed * dt_frame / demo.fs.sim.model.opt.timestep))
    snaps = sorted(float(x) for x in args.snap_at.split(",") if x.strip())
    snap_dir = Path(args.snap_dir)
    t_end = args.until if args.until else math.inf
    tail = 0
    n_frames = 0
    wall = time.time()
    while tail < 3 * fps and demo.t < t_end:
        while pending and demo.t >= pending[0][0]:
            print(f"t={demo.t:6.1f}s KEY {pending[0][1]}: {demo.press(pending.pop(0)[1])}", flush=True)
        demo.advance(steps)
        if args.snap_only:
            if not snaps or demo.result is not None:
                break
            if demo.t < snaps[0] - 1.0:
                tr, est = demo.fs.truth.state(), demo.fs.autopilot.pose()
                comp.chase.update(dt_frame, tr.pos, tr.vel, est.pos)
                continue
        frame = comp.frame(demo, dt_frame, args.speed, False, False)
        if video is not None:
            video.write(frame)
        n_frames += 1
        if snaps and demo.t >= snaps[0]:
            snaps.pop(0)
            snap_dir.mkdir(parents=True, exist_ok=True)
            p = snap_dir / f"frame_t{demo.t:05.1f}.png"
            cv2.imwrite(str(p), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(f"snapshot: {p}", flush=True)
        if n_frames % (fps * 5) == 0:
            print(f"t={demo.t:6.1f}s {demo.fs.mission.phase:14s} {demo.c.summary()}  "
                  f"({n_frames / max(time.time() - wall, 1e-6):.1f} frames/s rendered)", flush=True)
        if demo.result is not None:
            tail += 1
    if video is not None:
        video.close()
    comp.close()
    if demo.result is not None:
        print(demo.result.summary())
    print(f"video: {out}  ({n_frames} frames, {n_frames / fps:.1f} s)")
    demo.close()
    return 0


def _size(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=DEMO_SEED)
    ap.add_argument("--mission", default=None, help="fly a mission prepared by sfhack.missionprep (id under "
                    "$SFHACK_DATA/missions); default: the built-in Berkeley sample")
    ap.add_argument("--localizer", choices=["vision", "mock", "off"], default=DEMO_LOCALIZER)
    ap.add_argument("--speed", type=int, choices=[1, 2, 4], default=2, help="sim speed (viewer and video)")
    ap.add_argument("--events", default="", help="key presses at sim times, e.g. '45:C,55:C,30:P'")
    ap.add_argument("--scenario-at", type=float, default=None, help="play the spoof scenario at this sim time")
    ap.add_argument("--headless", action="store_true", help="no window: fly, print, score")
    ap.add_argument("--trn", choices=["on", "off"], default="on",
                    help="terrain-referenced navigation (lidar vs onboard DSM) as a GNSS-free position source")
    ap.add_argument("--record", nargs="?", const=str(data_dir() / "demo" / "v2_demo.mp4"), default=None,
                    help="render offscreen to an mp4 (default $SFHACK_DATA/demo/v2_demo.mp4)")
    ap.add_argument("--until", type=float, default=0.0, help="with --record: stop at this sim time")
    ap.add_argument("--size", type=_size, default=None, help="frame size WxH (viewer 1280x720, video 1920x1080)")
    ap.add_argument("--fullscreen", action="store_true", help="viewer: start fullscreen (Tab toggles)")
    ap.add_argument("--snap-at", default="", help="sim times to save PNG frames, e.g. '30,60'")
    ap.add_argument("--snap-dir", default=str(data_dir() / "demo"), help="where --snap-at / Z write PNGs")
    ap.add_argument("--snap-only", action="store_true", help=argparse.SUPPRESS)  # with --record: PNGs, no video
    ap.add_argument("--quit-after", type=float, default=0.0, help=argparse.SUPPRESS)  # wall s, viewer smoke test
    add_cli_args(ap)
    args = ap.parse_args()
    if args.gnss is None:
        args.gnss = "ok"  # the demo starts with an honest receiver so the spoof has something to drag
    args.conditions = conditions_from_args(args)
    if args.size is None:
        args.size = RECORD_SIZE if args.record else VIEW_SIZE
    if args.headless:
        sys.exit(run_headless(args))
    if args.record:
        sys.exit(run_record(args))
    run_viewer(args)


if __name__ == "__main__":
    main()
