"""Live demo: the default Berkeley mission A -> B in the MuJoCo viewer, flown on the onboard estimate
(EKF + AHRS, no truth in the control loop, GPS disabled), with the localizer feeding visual fixes.

    python demos/demo_live.py                         # viewer, demo seed + demo localizer
    python demos/demo_live.py --localizer mock        # MOCK localizer (truth + noise, clearly labeled)
    python demos/demo_live.py --headless              # no window: run, print, score (verification)
    python demos/demo_live.py --record out.mp4        # offscreen backup video (chase cam + panels + text)
    python demos/demo_live.py --events "45:V,75:V"    # press keys at sim times (also in the viewer)

What you see:
  drone = TRUTH (the simulator). Cyan sphere = where the drone THINKS it is (EKF), cyan ring = 2-sigma.
  White trail = true path, cyan trail = estimated path. A green, B red.
  Bottom-left: the onboard map around the drone (white = truth, cyan = estimate + 2-sigma, yellow =
  last visual fix) and the live down-camera image.

Keys:  V localizer on/off (off -> dead reckoning, watch the estimate drift)   W wind gust (5 s)
       C camera blackout (10 s)   R restart   1 / 2 / 4 sim speed   SPACE pause

Truth is used here only to DISPLAY and SCORE (harness code); the autopilot never sees it, except
through the MOCK localizer when that one is selected (and labeled MOCK on screen).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mujoco
import numpy as np

from sfhack import bus as topics
from sfhack.config import data_dir
from sfhack.runner import FlightSystem

DEMO_SEED = 0
DEMO_LOCALIZER = "vision"  # 10/10 seeds within 3 m of B (mock: blind near B, see DEMO.md)
LOCALIZER_LABEL = {
    "mock": "MOCK (truth + 2 m noise)",
    "vision": "VISION (camera vs onboard ortho)",
    "off": "OFF -> dead reckoning",
}
GUST_SPEED = 8.0  # m/s
GUST_S = 5.0
BLACKOUT_S = 10.0
TRAIL_DT = 0.25  # s between trail points

CYAN = (0.0, 0.85, 1.0)
WHITE = (1.0, 1.0, 1.0)
GREEN = (0.15, 0.85, 0.2)
RED = (0.95, 0.15, 0.1)

MAP_PANEL = 300  # px, square
CAM_PANEL = (320, 240)  # w, h (native down-camera resolution)
PANEL_GAP = 10


def _rgb255(c: tuple[float, float, float]) -> tuple[int, int, int]:
    return tuple(int(255 * v) for v in c)  # type: ignore[return-value]


# ---------------------------------------------------------------------------------------- 3D overlay

class _Geoms:
    """Appends visual-only geoms to an MjvScene (viewer.user_scn or an offscreen renderer's scene)."""

    def __init__(self, scn: mujoco.MjvScene) -> None:
        self.scn = scn

    def _next(self) -> mujoco.MjvGeom | None:
        if self.scn.ngeom >= self.scn.maxgeom:
            return None
        g = self.scn.geoms[self.scn.ngeom]
        self.scn.ngeom += 1
        return g

    def sphere(self, pos, r: float, rgb, alpha: float = 1.0) -> None:
        g = self._next()
        if g is not None:
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0], float), np.asarray(pos, float),
                                np.eye(3).flatten(), np.array([*rgb, alpha], np.float32))

    def cylinder(self, pos, r: float, half_h: float, rgb, alpha: float = 1.0) -> None:
        g = self._next()
        if g is not None:
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([r, half_h, 0], float),
                                np.asarray(pos, float), np.eye(3).flatten(), np.array([*rgb, alpha], np.float32))

    def line(self, a, b, rgb, width: float = 3.0, alpha: float = 1.0) -> None:
        g = self._next()
        if g is not None:
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                                np.array([*rgb, alpha], np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, width, np.asarray(a, float), np.asarray(b, float))

    def polyline(self, pts: np.ndarray, rgb, width: float = 3.0, alpha: float = 1.0) -> None:
        for a, b in zip(pts[:-1], pts[1:]):
            self.line(a, b, rgb, width, alpha)

    def ring(self, center, radius: float, rgb, width: float = 3.0, n: int = 48) -> None:
        ang = np.linspace(0, 2 * np.pi, n + 1)
        pts = np.column_stack([center[0] + radius * np.cos(ang), center[1] + radius * np.sin(ang),
                               np.full(n + 1, center[2])])
        self.polyline(pts, rgb, width)


def _decimate(pts: list[np.ndarray], max_n: int) -> np.ndarray:
    a = np.array(pts)
    if len(a) > max_n:
        idx = np.unique(np.r_[np.linspace(0, len(a) - 1, max_n).astype(int), len(a) - 1])
        a = a[idx]
    return a


# ---------------------------------------------------------------------------------------- the demo

class LiveDemo:
    """One flight plus everything the jury sees (3D overlay, text, panels). Truth: display + score only."""

    def __init__(self, seed: int, localizer: str, reuse: FlightSystem | None = None) -> None:
        self.seed = seed
        self.base_localizer = localizer if localizer != "off" else "mock"
        self.fs = FlightSystem(pose_source="ekf", localizer=localizer, seed=seed,
                               reuse_sim=reuse.sim if reuse is not None else None)
        self.loc_on = localizer != "off"
        self.truth_trail: list[np.ndarray] = []
        self.est_trail: list[np.ndarray] = []
        self._next_trail = 0.0
        self.cam_img = np.zeros((CAM_PANEL[1], CAM_PANEL[0], 3), np.uint8)
        self.gust_until: float | None = None
        self.blackout_until = -1.0
        self.done_at: float | None = None
        self.result = None
        self.events: deque[tuple[float, str]] = deque(maxlen=3)  # operator actions (harness, not onboard)
        self.fs.bus.subscribe(topics.CAMERA_DOWN, self._on_frame)
        self.goal = self.fs.goal_xy
        self.start = self.fs.onboard_map.a_enu if self.fs.onboard_map is not None else np.zeros(2)

    # ------------------------------------------------------------------ inputs

    def _on_frame(self, frame) -> None:
        self.cam_img = frame.image

    @property
    def t(self) -> float:
        return self.fs.t

    def press(self, key: str) -> str:
        """Operator action by key letter. Returns a short description (also shown on screen)."""
        fs, t = self.fs, self.t
        if key == "V":
            self.loc_on = not self.loc_on
            fs.set_localizer(self.base_localizer if self.loc_on else "off")
            msg = f"localizer {'ON' if self.loc_on else 'OFF (dead reckoning)'}"
        elif key == "W":
            ang = np.random.default_rng(int(t * 1000)).uniform(0, 2 * np.pi)
            fs.disturb("wind", (GUST_SPEED * math.cos(ang), GUST_SPEED * math.sin(ang)))
            self.gust_until = t + GUST_S
            msg = f"wind gust {GUST_SPEED:.0f} m/s for {GUST_S:.0f} s"
        elif key == "C":
            fs.disturb("camera_blackout", duration=BLACKOUT_S)
            self.blackout_until = t + BLACKOUT_S
            msg = f"camera blackout {BLACKOUT_S:.0f} s"
        else:
            return ""
        self.events.append((t, msg))
        return msg

    # ------------------------------------------------------------------ stepping

    def advance(self, n_steps: int) -> None:
        fs = self.fs
        for _ in range(n_steps):
            if self.result is not None:
                return
            fs.step()
            if self.gust_until is not None and self.t >= self.gust_until:
                fs.disturb("wind", (0.0, 0.0))
                self.gust_until = None
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

    # ------------------------------------------------------------------ what the jury sees

    def draw_geoms(self, scn: mujoco.MjvScene) -> None:
        g = _Geoms(scn)
        est = self.fs.autopilot.pose()
        for xy, rgb in ((self.start, GREEN), (self.goal, RED)):
            g.cylinder([xy[0], xy[1], 0.06], 3.0, 0.05, rgb, 0.9)
            g.cylinder([xy[0], xy[1], 15.0], 0.35, 15.0, rgb, 0.45)
        if len(self.truth_trail) > 1:
            g.polyline(_decimate(self.truth_trail, 400), WHITE, 2.5, 0.9)
            g.polyline(_decimate(self.est_trail, 400), CYAN, 2.5, 0.9)
        g.sphere(est.pos, 0.9, CYAN, 0.45)
        g.ring(est.pos, max(est.sigma2_horizontal(), 0.3), CYAN, 3.0)
        g.ring([est.pos[0], est.pos[1], 0.15], max(est.sigma2_horizontal(), 0.3), CYAN, 2.0)  # shadow on ground
        g.line(est.pos, [est.pos[0], est.pos[1], 0.15], CYAN, 1.0, 0.5)

    def status(self) -> tuple[str, str]:
        fs, ap = self.fs, self.fs.autopilot
        est, tr = ap.pose(), fs.truth.state()
        err = float(np.linalg.norm(est.pos[:2] - tr.pos[:2]))
        hs = ap.health_state
        loc = LOCALIZER_LABEL["off"] if not self.loc_on else LOCALIZER_LABEL.get(self.base_localizer, self.base_localizer)
        fix_age = "-" if ap.health.last_fix_t is None else f"{self.t - ap.health.last_fix_t:.1f} s ago"
        rows = [
            ("Time", f"{self.t:6.1f} s"),
            ("Phase", fs.mission.phase),
            ("Health", hs.state.value + (f"  ({hs.reason})" if hs.reason and hs.state.value != "NOMINAL" else "")),
            ("Localizer", loc),
            ("Last visual fix", fix_age),
            ("2-sigma radius", f"{est.sigma2_horizontal():6.1f} m"),
            ("Position error", f"{err:6.1f} m   (eval only)"),
            ("GPS", "DISABLED"),
        ]
        return "\n".join(r[0] for r in rows), "\n".join(r[1] for r in rows)

    def decisions(self, n: int = 4) -> tuple[str, str]:
        rows = [("Decisions", "")] + [(f"{t:6.1f} s", text[:60]) for t, text in self.fs.autopilot.decisions[-n:]]
        return "\n".join(r[0] for r in rows), "\n".join(r[1] for r in rows)

    def console_line(self) -> str:
        ap, tr = self.fs.autopilot, self.fs.truth.state()
        err = float(np.linalg.norm(ap.pose().pos[:2] - tr.pos[:2]))
        return (f"t={self.t:6.1f}s {self.fs.mission.phase:14s} {ap.health_state.state.value:9s} "
                f"cam mean {self.cam_img.mean():5.1f}  frames {ap.loc_frames:4d} fixes {ap.loc_fixes:4d}  "
                f"2-sigma {ap.pose().sigma2_horizontal():5.1f} m  err {err:5.1f} m (eval)")

    def banner(self) -> str:
        if self.result is not None:
            r = self.result
            return f"{r.outcome}: landed {r.final_error:.1f} m from B  (press R to restart)"
        lines = []
        if self.blackout_until > self.t:
            lines.append(f"CAMERA BLACKOUT  {self.blackout_until - self.t:4.1f} s")
        if self.gust_until is not None:
            lines.append(f"WIND GUST  {self.gust_until - self.t:4.1f} s")
        if not self.loc_on:
            lines.append("LOCALIZER OFF - DEAD RECKONING")
        return "\n".join(lines)

    def map_panel(self, size: int = MAP_PANEL) -> np.ndarray:
        om = self.fs.onboard_map
        est, tr = self.fs.autopilot.pose(), self.fs.truth.state()
        s2 = est.sigma2_horizontal()
        sep = float(np.linalg.norm(est.pos[:2] - tr.pos[:2]))
        half = max(60.0, 0.6 * sep + 25.0, 1.2 * s2)  # m; zooms out so truth and estimate stay in view
        center = 0.5 * (est.pos[:2] + tr.pos[:2])
        cu, cv = om.enu_to_px(center)
        half_px = half / om.m_per_px
        k = size / (2 * half_px)
        M = np.array([[k, 0, -(cu - half_px) * k], [0, k, -(cv - half_px) * k]])
        img = cv2.warpAffine(om.ortho, M, (size, size), flags=cv2.INTER_AREA, borderValue=(60, 60, 60))

        def px(xy) -> tuple[int, int]:
            u, v = om.enu_to_px(np.asarray(xy, float)[:2])
            return int(round((u - (cu - half_px)) * k)), int(round((v - (cv - half_px)) * k))

        def trail(pts, color) -> None:
            if len(pts) > 1:
                p = np.array([px(q) for q in _decimate(pts, 300)], np.int32)
                cv2.polylines(img, [p], False, color, 1, cv2.LINE_AA)

        trail(self.truth_trail, (255, 255, 255))
        trail(self.est_trail, _rgb255(CYAN))
        for xy, c in ((self.start, GREEN), (self.goal, RED)):
            cv2.circle(img, px(xy), max(3, int(3.0 / om.m_per_px * k)), _rgb255(c), 2, cv2.LINE_AA)
        fix = self.fs.autopilot.last_fix
        if fix is not None and self.t - fix.t < 1.0:
            u, v = px(fix.pos)
            c = (255, 220, 0) if fix.accepted else (255, 60, 60)
            cv2.drawMarker(img, (u, v), c, cv2.MARKER_TILTED_CROSS, 10, 2)
        cv2.circle(img, px(est.pos), max(2, int(s2 / om.m_per_px * k)), _rgb255(CYAN), 1, cv2.LINE_AA)
        cv2.circle(img, px(tr.pos), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, px(tr.pos), 5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(img, px(est.pos), 4, _rgb255(CYAN), -1, cv2.LINE_AA)
        # scale bar + title
        bar_m = 20.0 if half < 100 else 50.0
        bar_px = int(bar_m / om.m_per_px * k)
        cv2.line(img, (8, size - 10), (8 + bar_px, size - 10), (255, 255, 255), 2)
        _label(img, f"{bar_m:.0f} m", (12 + bar_px, size - 6), 0.4)
        _label(img, "ONBOARD MAP", (6, 16), 0.45)
        _label(img, "white truth  cyan estimate", (6, 32), 0.35)
        cv2.rectangle(img, (0, 0), (size - 1, size - 1), (255, 255, 255), 1)
        return img

    def cam_panel(self, w: int = CAM_PANEL[0], h: int = CAM_PANEL[1]) -> np.ndarray:
        img = self.cam_img
        sx, sy = w / img.shape[1], h / img.shape[0]
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA) if img.shape[:2] != (h, w) else img.copy()
        fix = self.fs.autopilot.last_fix
        fresh = fix is not None and self.t - fix.t < 0.5
        if fresh and fix.matched_px:
            pts = np.array([(u * sx, v * sy) for u, v in fix.matched_px], np.int32)
            color = (255, 220, 0) if fix.accepted else (255, 60, 60)
            if fix.source == "vision":  # matched ground patch (ortho-rectified template footprint)
                cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
            else:  # mock: mapped landmarks inside the footprint
                for p in pts:
                    cv2.circle(img, tuple(int(q) for q in p), 6, color, 1, cv2.LINE_AA)
        if not img.any():
            _label(img, "NO SIGNAL", (w // 2 - 50, h // 2), 0.7, (255, 80, 80))
        _label(img, "DOWN CAMERA (live)", (6, 16), 0.45)
        if fresh:
            what = f"MATCH  NCC {fix.n_inliers / 100:.2f}" if fix.source == "vision" else f"{fix.n_inliers} landmarks"
            _label(img, f"{what}  {'accepted' if fix.accepted else 'REJECTED'}", (6, h - 10), 0.45,
                   (255, 220, 0) if fix.accepted else (255, 80, 80))
        else:
            _label(img, "no fix", (6, h - 10), 0.45, (200, 200, 200))
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (255, 255, 255), 1)
        return img


def _label(img: np.ndarray, text: str, org: tuple[int, int], scale: float, color=(255, 255, 255)) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _own_viewer_hwnd() -> int | None:
    """Windows: handle of THIS process's visible MuJoCo viewer window (never another app's window)."""
    import ctypes
    import os
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    proto = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    me, found = os.getpid(), []

    def enum(hwnd, _lp):
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value == me and user32.IsWindowVisible(hwnd):
            buf = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, buf, 256)
            if buf.value.startswith("MuJoCo"):
                found.append(hwnd)
        return True

    user32.EnumWindows(proto(enum), 0)
    return found[0] if found else None


def _snap_own_window(path: Path) -> None:
    """Windows: save a PNG of our own viewer window only (PrintWindow, so other apps never show up)."""
    import ctypes
    from ctypes import wintypes
    from PIL import Image
    hwnd = _own_viewer_hwnd()
    if hwnd is None:
        return
    user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
    r = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(r))
    w, h = r.right - r.left, r.bottom - r.top
    hdc = user32.GetDC(hwnd)
    mdc = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
    gdi32.SelectObject(mdc, bmp)
    user32.PrintWindow(hwnd, mdc, 3)  # PW_CLIENTONLY | PW_RENDERFULLCONTENT (OpenGL content)
    buf = ctypes.create_string_buffer(w * h * 4)
    hdr = (ctypes.c_uint32 * 10)(40, w, ctypes.c_uint32(-h).value, 1 | (32 << 16), 0, 0, 0, 0, 0, 0)
    gdi32.GetDIBits(mdc, bmp, 0, h, buf, hdr, 0)
    Image.frombuffer("RGBA", (w, h), buf, "raw", "BGRA", 0, 1).convert("RGB").save(path)
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(mdc)
    user32.ReleaseDC(hwnd, hdc)


def _maximize_window() -> None:
    """Windows: maximize the MuJoCo window (it opens taller than a laptop screen, hiding the panels)."""
    if sys.platform != "win32":
        return
    import ctypes
    import os
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    proto = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    me = os.getpid()
    for _ in range(50):
        found: list[int] = []

        def enum(hwnd, _lp):
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value != me:  # only OUR viewer window, never another MuJoCo app on the desktop
                return True
            buf = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, buf, 256)
            if buf.value.startswith("MuJoCo") and user32.IsWindowVisible(hwnd):
                found.append(hwnd)
            return True

        user32.EnumWindows(proto(enum), 0)
        if found:
            user32.ShowWindow(found[0], 3)  # SW_MAXIMIZE
            return
        time.sleep(0.1)


KEY_HELP = ("V\nW\nC\nR\n1 2 4\nSPACE",
            "localizer on/off\nwind gust\ncamera blackout 10 s\nrestart\nsim speed\npause")


def parse_events(spec: str) -> list[tuple[float, str]]:
    """'45:V,70:V' -> [(45.0, 'V'), (70.0, 'V')], sorted by time."""
    out = []
    for item in filter(None, (s.strip() for s in spec.split(","))):
        t, k = item.split(":")
        out.append((float(t), k.strip().upper()))
    return sorted(out)


# ---------------------------------------------------------------------------------------- viewer mode

def run_viewer(args: argparse.Namespace) -> None:
    import mujoco.viewer

    demo = LiveDemo(args.seed, args.localizer)
    m, d = demo.fs.sim.model, demo.fs.sim.data
    keys: deque[int] = deque()
    speed, paused = args.speed, False
    events = parse_events(args.events)

    with mujoco.viewer.launch_passive(m, d, key_callback=keys.append,
                                      show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = m.body("x2").id
        viewer.cam.distance, viewer.cam.elevation = 28.0, -22.0
        heading = demo.goal - demo.start
        viewer.cam.azimuth = math.degrees(math.atan2(heading[1], heading[0]))
        pending = list(events)
        wall0 = time.time()
        reported = False
        next_log = 0.0
        sim_budget = 0.0  # sim seconds owed to the wall clock (real-time pacing at `speed`)
        last_wall = time.perf_counter()
        next_panels = 0.0
        rtf_mark = (time.perf_counter(), 0.0)
        snaps = sorted(float(x) for x in args.snap_at.split(",") if x.strip())
        _maximize_window()
        while viewer.is_running():
            if args.quit_after and time.time() - wall0 > args.quit_after:
                break
            frame_start = time.perf_counter()
            while pending and demo.t >= pending[0][0]:
                key = pending.pop(0)[1]
                keys.append(32 if key == "SPACE" else ord(key[0]))
            while keys:
                code = keys.popleft()
                if code == 32:  # SPACE
                    paused = not paused
                    print(f"t={demo.t:6.1f}s {'PAUSED' if paused else 'RESUMED'}")
                elif code in (ord("1"), ord("2"), ord("4")):
                    speed = int(chr(code))
                    print(f"t={demo.t:6.1f}s speed {speed}x")
                elif code == ord("R"):
                    print(f"t={demo.t:6.1f}s RESTART")
                    with viewer.lock():
                        old = demo
                        demo = LiveDemo(args.seed, args.localizer, reuse=old)
                        old.close()
                    reported = False
                    next_log = 0.0
                    rtf_mark = (time.perf_counter(), 0.0)
                    pending = [e for e in events if e[1] != "R"]
                elif 0 <= code < 256:
                    msg = demo.press(chr(code))
                    if msg:
                        print(f"t={demo.t:6.1f}s KEY {chr(code)}: {msg}")
            now = time.perf_counter()
            wall_dt, last_wall = min(now - last_wall, 0.1), now  # cap: a stall must not cause a burst
            if not paused:
                # Advance by wall clock x speed (a fixed number of steps per frame ran at ~0.4x, since a
                # frame with panels costs ~35 ms).
                sim_budget += wall_dt * speed
                n = int(sim_budget / m.opt.timestep)
                sim_budget -= n * m.opt.timestep
                demo.advance(n)
                if demo.t >= next_log:
                    next_log = demo.t + 5.0
                    w0, s0 = rtf_mark
                    rtf = (demo.t - s0) / max(now - w0, 1e-6)
                    rtf_mark = (now, demo.t)
                    print(f"{demo.console_line()}  real-time x{rtf:.2f}", flush=True)
                if demo.result is not None and not reported:
                    reported = True
                    print(f"t={demo.t:6.1f}s {demo.result.summary()}")
            with viewer.lock():
                # The viewer also binds W (wireframe), C (contacts), R (reflection), 1/2/4 (geom groups):
                # undo those so our keys never change the picture.
                viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 0
                viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
                viewer.opt.geomgroup[:3] = 1
                viewer.opt.geomgroup[3:] = 0
                viewer.user_scn.ngeom = 0
                demo.draw_geoms(viewer.user_scn)
            left, right = demo.status()
            texts = [(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right),
                     (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                      KEY_HELP[0] + "\nspeed", KEY_HELP[1] + f"\n{speed}x" + ("  PAUSED" if paused else "")),
                     (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT, *demo.decisions())]
            if demo.banner():
                texts.append((mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOP, demo.banner(), None))
            viewer.set_texts(texts)
            vp = viewer.viewport
            if vp is not None and vp.height > 0 and now >= next_panels:
                next_panels = now + 1.0 / 15.0  # image panels at 15 Hz (texture upload is the costly part)
                side = max(160, int(0.36 * vp.height))  # panels scale with the (HiDPI) framebuffer
                cw, ch = side * 4 // 3, side
                # Drawn at base size (labels readable), then scaled to the HiDPI framebuffer.
                viewer.set_images([
                    (mujoco.MjrRect(PANEL_GAP, PANEL_GAP, side, side),
                     cv2.resize(demo.map_panel(), (side, side), interpolation=cv2.INTER_LINEAR)),
                    (mujoco.MjrRect(2 * PANEL_GAP + side, PANEL_GAP, cw, ch),
                     cv2.resize(demo.cam_panel(), (cw, ch), interpolation=cv2.INTER_LINEAR)),
                ])
            sys.stdout.flush()
            viewer.sync()
            if snaps and demo.t >= snaps[0] and sys.platform == "win32":
                snaps.pop(0)
                _snap_own_window(Path(args.snap_dir) / f"viewer_t{demo.t:05.1f}.png")
            time.sleep(max(0.0, 1.0 / 60.0 - (time.perf_counter() - frame_start)))
    demo.close()


# ---------------------------------------------------------------------------------------- headless / record

def run_headless(args: argparse.Namespace) -> int:
    demo = LiveDemo(args.seed, args.localizer)
    pending = parse_events(args.events)
    wall = time.time()
    next_print = 0.0
    while demo.result is None:
        while pending and demo.t >= pending[0][0]:
            t, k = pending.pop(0)
            print(f"t={demo.t:6.1f}s KEY {k}: {demo.press(k)}")
        demo.advance(50)
        if demo.t >= next_print:
            next_print += 10.0
            left, right = demo.status()
            row = dict(zip(left.split("\n"), right.split("\n")))
            print(f"t={demo.t:6.1f}s {row['Phase']:14s} {row['Health'][:10]:10s} 2-sigma {row['2-sigma radius'].strip():>8s}"
                  f"  err {row['Position error'].split('(')[0].strip():>8s}")
    for t, text in demo.fs.autopilot.decisions:
        print(f"   {t:6.1f}s {text}")
    r = demo.result
    print(r.summary())
    print(f"wall {time.time() - wall:.1f} s")
    demo.close()
    return 0 if r.success else 1


def compose_frame(demo: LiveDemo, renderer: mujoco.Renderer, cam: mujoco.MjvCamera, opt: mujoco.MjvOption,
                  speed: int) -> np.ndarray:
    renderer.update_scene(demo.fs.sim.data, camera=cam, scene_option=opt)
    demo.draw_geoms(renderer.scene)
    img = renderer.render().copy()
    H, W = img.shape[:2]
    mp, cp = demo.map_panel(), demo.cam_panel()
    img[H - PANEL_GAP - MAP_PANEL:H - PANEL_GAP, PANEL_GAP:PANEL_GAP + MAP_PANEL] = mp
    x0 = 2 * PANEL_GAP + MAP_PANEL
    img[H - PANEL_GAP - CAM_PANEL[1]:H - PANEL_GAP, x0:x0 + CAM_PANEL[0]] = cp

    def text_block(left: str, right: str, x: int, y: int, col2: int, scale: float = 0.5) -> None:
        ls, rs = left.split("\n"), right.split("\n")
        w = col2 + max(cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] for s in rs) + 16
        h = 20 * len(ls) + 10
        roi = img[y:y + h, x:x + w]
        roi[:] = (roi * 0.45).astype(np.uint8)
        for i, (a, b) in enumerate(zip(ls, rs)):
            _label(img, a, (x + 8, y + 20 * (i + 1)), scale)
            _label(img, b, (x + col2, y + 20 * (i + 1)), scale, (140, 230, 255))

    left, right = demo.status()
    text_block(left, right, 10, 10, 150)
    text_block(*KEY_HELP, W - 260, 10, 70, 0.45)
    dl, dr = demo.decisions()
    text_block(dl, dr, W - 620, H - 20 * len(dl.split("\n")) - 20, 80, 0.45)
    _label(img, f"sim speed {speed}x", (W - 250, 160), 0.45)
    ban = demo.banner()
    for i, line in enumerate(ban.split("\n") if ban else []):
        tw = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]
        cv2.putText(img, line, ((W - tw) // 2, 40 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, line, ((W - tw) // 2, 40 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 230, 120), 2, cv2.LINE_AA)
    return img


def run_record(args: argparse.Namespace) -> int:
    out = Path(args.record)
    out.parent.mkdir(parents=True, exist_ok=True)
    demo = LiveDemo(args.seed, args.localizer)
    m = demo.fs.sim.model
    W, H, fps = 1280, 720, 30
    renderer = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = m.body("x2").id
    cam.distance, cam.elevation = 28.0, -22.0
    heading = demo.goal - demo.start
    cam.azimuth = math.degrees(math.atan2(heading[1], heading[0]))
    opt = mujoco.MjvOption()
    pending = parse_events(args.events)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    gif_frames: list[np.ndarray] = []
    steps = int(round(args.speed / fps / m.opt.timestep))
    tail = 0
    while tail < 2 * fps:  # 2 s of video after the result
        while pending and demo.t >= pending[0][0]:
            demo.press(pending.pop(0)[1])
        demo.advance(steps)
        frame = compose_frame(demo, renderer, cam, opt, args.speed)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if len(gif_frames) < 100000 and (int(demo.t * fps / args.speed) % 6 == 0):
            gif_frames.append(cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA))
        if demo.result is not None:
            tail += 1
    writer.release()
    renderer.close()
    print(demo.result.summary())
    print(f"video: {out}")
    if args.gif:
        from PIL import Image
        gif = out.with_suffix(".gif")
        ims = [Image.fromarray(f) for f in gif_frames]
        ims[0].save(gif, save_all=True, append_images=ims[1:], duration=200, loop=0)
        print(f"gif:   {gif}")
    demo.close()
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=DEMO_SEED)
    ap.add_argument("--localizer", choices=["vision", "mock", "off"], default=DEMO_LOCALIZER)
    ap.add_argument("--speed", type=int, choices=[1, 2, 4], default=2, help="sim speed (viewer and video)")
    ap.add_argument("--events", default="", help="key presses at sim times, e.g. '45:V,75:V,60:C'")
    ap.add_argument("--headless", action="store_true", help="no window: fly, print, score")
    ap.add_argument("--record", nargs="?", const=str(data_dir() / "demo" / "v2_demo.mp4"), default=None,
                    help="render offscreen to an mp4 (default $SFHACK_DATA/demo/v2_demo.mp4)")
    ap.add_argument("--gif", action="store_true", help="with --record: also write a GIF next to the mp4")
    ap.add_argument("--quit-after", type=float, default=0.0, help=argparse.SUPPRESS)  # wall s, viewer smoke test
    ap.add_argument("--snap-at", default="", help=argparse.SUPPRESS)  # sim times: PNG of our own window
    ap.add_argument("--snap-dir", default=str(data_dir() / "demo"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.headless:
        sys.exit(run_headless(args))
    if args.record:
        sys.exit(run_record(args))
    run_viewer(args)


if __name__ == "__main__":
    main()
