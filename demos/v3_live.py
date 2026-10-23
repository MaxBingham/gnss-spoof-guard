"""V3 live demo: one flight A -> B in the MuJoCo viewer on the onboard estimate (GNSS disabled), INS backbone
+ terrain-referenced (TRN) + image-based (IBN) navigation, with the operator able to switch sources off.

    python demos/v3_live.py                                   # S1 San Francisco, all sources, demo seed
    python demos/v3_live.py --world s2 --speed 8              # S2 Carpathian foothills (8.4 km)
    python demos/v3_live.py --config ins                      # dead reckoning only (watch it drift)
    python demos/v3_live.py --events "60:T,90:T"              # press keys at sim times (also in the viewer)
    python demos/v3_live.py --headless                        # no window: fly, print, score
    python demos/v3_live.py --record                          # offscreen backup video ($SFHACK_DATA/v3/demo/)

What you see:
  drone = TRUTH (the simulator). Cyan sphere = where the drone THINKS it is (EKF), cyan ring = 2-sigma.
  White trail = true path, cyan trail = estimated path, yellow line = planned route. A green, B red.
  Bottom-left panels: the onboard map around the drone (TRN particle cloud orange, last fix: TRN orange /
  IBN yellow cross, red = rejected by the integrity monitor), the down camera (IBN), and the terrain profile
  the altimeter measures against the onboard surface model (TRN).

Keys:  T TRN on/off   I IBN on/off   W wind gust (8 m/s, 5 s)   C camera blackout (10 s)   A altimeter failure
       R restart   1 / 2 / 4 / 8 sim speed   SPACE pause

Harness code: truth is used only to DISPLAY and SCORE; the autopilot never sees it.
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

from sfhack.config import data_dir
from sfhack.nav_runner import BLACKOUT_S, CONFIGS, GUST, NavFlightSystem
from sfhack.world.surface import Grid

DEMO_SEED = 0
TRAIL_DT = 0.25  # s between trail points
NOISE = ("candidate", "UPDATE_ZONE")  # planner log lines, too verbose for the decision panel

CYAN = (0.0, 0.85, 1.0)
WHITE = (1.0, 1.0, 1.0)
GREEN = (0.15, 0.85, 0.2)
RED = (0.95, 0.15, 0.1)
YELLOW = (1.0, 0.85, 0.1)
ORANGE = (255, 140, 0)  # RGB 0..255 (panels)
FIX_COLOR = {"trn": (255, 140, 0), "ibn": (255, 220, 0)}

MAP_PANEL = 300  # px, square
CAM_PANEL = (320, 240)  # w, h (native down-camera resolution)
PROFILE_PANEL = (320, 160)
PANEL_GAP = 10


def _rgb255(c: tuple[float, float, float]) -> tuple[int, int, int]:
    return tuple(int(255 * v) for v in c)  # type: ignore[return-value]


def _label(img: np.ndarray, text: str, org: tuple[int, int], scale: float, color=(255, 255, 255)) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


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


def _decimate(pts, max_n: int) -> np.ndarray:
    a = np.asarray(pts)
    if len(a) > max_n:
        idx = np.unique(np.r_[np.linspace(0, len(a) - 1, max_n).astype(int), len(a) - 1])
        a = a[idx]
    return a


def _hillshade(dsm: Grid) -> np.ndarray:
    """RGB hillshade of the onboard surface model (map background where there is no onboard imagery)."""
    z = dsm.z.astype(np.float32)
    gy, gx = np.gradient(z, dsm.mpp)
    nx, ny = -gx, gy  # rows run north -> south
    shade = (nx * -0.5 + ny * 0.5 + 1.0) / np.sqrt(nx ** 2 + ny ** 2 + 1.0)
    shade = np.clip(0.35 + 0.55 * shade, 0, 1)
    zn = (z - z.min()) / max(float(np.ptp(z)), 1e-6)
    base = np.stack([0.45 + 0.25 * zn, 0.55 + 0.15 * zn, 0.35 + 0.1 * zn], -1)  # green lowland -> tan hills
    return (255 * np.clip(base * shade[..., None], 0, 1)).astype(np.uint8)


# ---------------------------------------------------------------------------------------- the demo

class LiveDemo:
    """One flight plus everything the jury sees (3D overlay, text, panels). Truth: display + score only."""

    def __init__(self, world: str, config: str, seed: int, events: str = "") -> None:
        self.fs = fs = NavFlightSystem(world, config, seed, events=events)
        self.ap = fs.autopilot
        self.urban = fs.scenario.kind == "urban"
        self.truth_trail: list[np.ndarray] = []
        self.est_trail: list[np.ndarray] = []
        self._next_trail = 0.0
        self.cam_img = np.zeros((CAM_PANEL[1], CAM_PANEL[0], 3), np.uint8)
        self.cam_t = -1e9
        self.gust_until = -1.0
        self.blackout_until = -1.0
        self.alt_failed = False
        self.done_at: float | None = None
        self.result = None
        self.events: deque[tuple[float, str]] = deque(maxlen=3)
        from sfhack import bus as topics
        fs.bus.subscribe(topics.CAMERA_DOWN, self._on_frame)
        self.start, self.goal = fs.world.a_enu, fs.world.b_enu
        self.start_z = float(fs.world.surface_z(self.start))
        self.goal_z = float(fs.world.surface_z(self.goal))
        om = self.ap.map
        self.bg = _hillshade(om.dsm)
        self.route_xyz = np.column_stack([self.ap.route.xy, self.ap.route.z])
        self.base_half = 60.0 if self.urban else 220.0  # m, map panel half width

    def _on_frame(self, frame) -> None:
        self.cam_img = frame.image
        self.cam_t = frame.t

    @property
    def t(self) -> float:
        return self.fs.t

    def press(self, key: str) -> str:
        """Operator key (T I W C A). Returns a short description (also shown on screen)."""
        msg = self.fs.press(key)
        if not msg:
            return ""
        if key == "W":
            self.gust_until = self.t + GUST[1]
        elif key == "C":
            self.blackout_until = self.t + BLACKOUT_S
        elif key == "A":
            self.alt_failed = True
        self.events.append((self.t, msg))
        return msg

    def advance(self, n_steps: int) -> None:
        fs = self.fs
        for _ in range(n_steps):
            if self.result is not None:
                return
            n_ev = len(fs.events)
            fs.step()  # also fires scripted --events / outage configurations
            if len(fs.events) < n_ev:
                self.events.append((self.t, self.ap.decisions[-1][1][:40]))
            if self.t >= self._next_trail:
                self._next_trail = self.t + TRAIL_DT
                self.truth_trail.append(fs.truth.state().pos.copy())
                self.est_trail.append(self.ap.pose().pos.copy())
            if fs.done and self.done_at is None:
                self.done_at = self.t
            if (self.done_at is not None and (fs.crashed or self.t - self.done_at >= 1.0)) or self.t >= fs.max_t:
                self.result = r = fs.result()  # harness: scored on truth
                self.ap.log_decision(self.t, f"{r.outcome} ({'success' if r.success else 'fail'}), final error "
                                             f"{r.final_error:.1f} m (harness, truth)")

    def close(self) -> None:
        self.fs.close()

    # ------------------------------------------------------------------ what the jury sees

    def draw_geoms(self, scn: mujoco.MjvScene) -> None:
        g = _Geoms(scn)
        est = self.ap.pose()
        s2 = max(est.sigma2_horizontal(), 0.3)
        tall = 15.0 if self.urban else 60.0
        for xy, z, rgb in ((self.start, self.start_z, GREEN), (self.goal, self.goal_z, RED)):
            g.cylinder([xy[0], xy[1], z + 0.06], 3.0, 0.05, rgb, 0.9)
            g.cylinder([xy[0], xy[1], z + tall], 0.35 if self.urban else 1.2, tall, rgb, 0.45)
        g.polyline(_decimate(self.route_xyz, 150), YELLOW, 1.5, 0.5)
        if len(self.truth_trail) > 1:
            g.polyline(_decimate(self.truth_trail, 350), WHITE, 2.5, 0.9)
            g.polyline(_decimate(self.est_trail, 350), CYAN, 2.5, 0.9)
        g.sphere(est.pos, 0.9 if self.urban else 2.0, CYAN, 0.45)
        g.ring(est.pos, s2, CYAN, 3.0)
        ground = float(self.fs.world.surface_z(est.pos[:2])) + 0.3
        g.ring([est.pos[0], est.pos[1], ground], s2, CYAN, 2.0)  # shadow on the ground
        g.line(est.pos, [est.pos[0], est.pos[1], ground], CYAN, 1.0, 0.5)

    def sources_text(self) -> str:
        ap, t = self.ap, self.t
        parts = ["INS"]
        for name in ("trn", "ibn"):
            if name == "ibn" and ap.ibn is None:
                continue
            if not ap.sources[name]:
                parts.append(f"{name.upper()} off")
            elif ap.integrity.excluded(name, t):
                parts.append(f"{name.upper()} EXCLUDED")
            else:
                parts.append(name.upper())
        return " + ".join(parts)

    def status(self) -> tuple[str, str]:
        fs, ap = self.fs, self.ap
        est, tr = ap.pose(), fs.truth.state()
        err = float(np.linalg.norm(est.pos[:2] - tr.pos[:2]))
        hs = ap.health_state
        st = ap.integrity.src
        ibn = ap.ibn.last_status if (ap.ibn is not None and ap.camera_on) else ("camera off" if ap.ibn else "-")
        rows = [
            ("Time", f"{self.t:6.1f} s"),
            ("Scenario", fs.scenario.name.upper() + f"  seed {fs.seed}  ({fs.config})"),
            ("Phase", ap.mission.phase),
            ("Health", hs.state.value + (f"  ({hs.reason[:38]})" if hs.reason and hs.state.value != "NOMINAL" else "")),
            ("Sources", self.sources_text()),
            ("TRN", f"{ap.trn.status[:30]}  fixes {st['trn'].n_accepted}/{st['trn'].n_fix}" if ap.sources["trn"]
             else "off"),
            ("IBN", f"{ibn[:30]}  fixes {st['ibn'].n_accepted}/{st['ibn'].n_fix}" if ap.ibn is not None else "no imagery"),
            ("2-sigma radius", f"{est.sigma2_horizontal():6.1f} m"),
            ("Position error", f"{err:6.1f} m   (eval only)"),
            ("GNSS", "DISABLED"),
        ]
        return "\n".join(r[0] for r in rows), "\n".join(r[1] for r in rows)

    def decisions(self, n: int = 5) -> tuple[str, str]:
        dec = [d for d in self.ap.decisions if not d[1].startswith(NOISE)][-n:]
        rows = [("Decisions", "")] + [(f"{t:6.1f} s", text[:58]) for t, text in dec]
        return "\n".join(r[0] for r in rows), "\n".join(r[1] for r in rows)

    def console_line(self) -> str:
        tr = self.fs.truth.state()
        err = float(np.linalg.norm(self.ap.pose().pos[:2] - tr.pos[:2]))
        return f"t={self.t:6.1f}s {self.ap.status_line()} err {err:6.1f} m (eval)"

    def banner(self) -> str:
        if self.result is not None:
            r = self.result
            where = (f"landed {r.final_error:.1f} m from B" if self.urban else
                     f"arrived {r.arrival_error:.1f} m from B, landed {r.final_error:.1f} m from B")
            return f"{r.outcome} {'SUCCESS' if r.success else 'FAIL'}: {where}"
        lines = []
        if self.blackout_until > self.t:
            lines.append(f"CAMERA BLACKOUT  {self.blackout_until - self.t:4.1f} s")
        if self.gust_until > self.t:
            lines.append(f"WIND GUST  {self.gust_until - self.t:4.1f} s")
        if self.alt_failed:
            lines.append("ALTIMETER FAILED")
        off = [n.upper() for n in ("trn", "ibn") if not self.ap.sources[n] and (n == "trn" or self.ap.ibn is not None)]
        if off and self.fs.config != "ins":
            lines.append(" + ".join(off) + " OFF")
        return "\n".join(lines)

    # ------------------------------------------------------------------ panels

    def _warp(self, img: np.ndarray, grid_min_x: float, grid_max_y: float, mpp: float, view_x0: float,
              view_y1: float, k: float, size: int, dst: np.ndarray | None) -> np.ndarray:
        """Warp a north-up raster into the panel (panel pixel = k px per m from (view_x0, view_y1))."""
        s = k * mpp  # panel px per raster px
        M = np.array([[s, 0, (grid_min_x - view_x0) * k], [0, s, (view_y1 - grid_max_y) * k]])
        if dst is None:
            return cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_AREA, borderValue=(60, 60, 60))
        return cv2.warpAffine(img, M, (size, size), dst=dst, flags=cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)

    def map_panel(self, size: int = MAP_PANEL) -> np.ndarray:
        ap = self.ap
        om = ap.map
        est, tr = ap.pose(), self.fs.truth.state()
        s2 = est.sigma2_horizontal()
        sep = float(np.linalg.norm(est.pos[:2] - tr.pos[:2]))
        half = max(self.base_half, 0.6 * sep + 25.0, 1.2 * s2)  # zooms out so truth and estimate stay in view
        center = 0.5 * (est.pos[:2] + tr.pos[:2])
        x0, y1 = center[0] - half, center[1] + half
        k = size / (2 * half)
        d = om.dsm
        img = self._warp(self.bg, d.min_x, d.max_y, d.mpp, x0, y1, k, size, None)
        if om.ortho_rgb is not None:
            o = om.ortho
            img = self._warp(om.ortho_rgb, o.min_x, o.max_y, o.mpp, x0, y1, k, size, img)

        def px(xy) -> tuple[int, int]:
            return int(round((xy[0] - x0) * k)), int(round((y1 - xy[1]) * k))

        def poly(pts, color, closed=False, w=1) -> None:
            if len(pts) > 1:
                p = np.array([px(q) for q in _decimate(pts, 300)], np.int32)
                cv2.polylines(img, [p], closed, color, w, cv2.LINE_AA)

        for nf in om.no_fly:
            poly(nf, (230, 40, 40), True, 2)
        poly(ap.route.xy, (255, 220, 0))
        poly(self.truth_trail, (255, 255, 255))
        poly(self.est_trail, _rgb255(CYAN))
        if ap.sources["trn"] and ap.trn.p is not None:
            pts = ap.trn.p[:: max(1, len(ap.trn.p) // 300), :2]
            for q in pts:
                cv2.circle(img, px(q), 1, ORANGE, -1)
        for xy, c in ((self.start, GREEN), (self.goal, RED)):
            cv2.circle(img, px(xy), max(3, int(3.0 * k)), _rgb255(c), 2, cv2.LINE_AA)
        fix = ap.last_fix
        if fix is not None and self.t - fix.t < 1.5:
            c = FIX_COLOR.get(fix.source, (255, 220, 0)) if fix.accepted else (255, 60, 60)
            cv2.drawMarker(img, px(fix.pos), c, cv2.MARKER_TILTED_CROSS, 12, 2)
        cv2.circle(img, px(est.pos), max(2, int(s2 * k)), _rgb255(CYAN), 1, cv2.LINE_AA)
        cv2.circle(img, px(tr.pos), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, px(tr.pos), 5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(img, px(est.pos), 4, _rgb255(CYAN), -1, cv2.LINE_AA)
        bar_m = next(b for b in (20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0) if b * k > size / 6)
        bar_px = int(bar_m * k)
        cv2.line(img, (8, size - 10), (8 + bar_px, size - 10), (255, 255, 255), 2)
        _label(img, f"{bar_m:.0f} m", (12 + bar_px, size - 6), 0.4)
        _label(img, "ONBOARD MAP", (6, 16), 0.45)
        _label(img, "white truth  cyan estimate", (6, 32), 0.35)
        _label(img, "orange TRN cloud  x last fix", (6, 46), 0.35)
        cv2.rectangle(img, (0, 0), (size - 1, size - 1), (255, 255, 255), 1)
        return img

    def cam_panel(self, w: int = CAM_PANEL[0], h: int = CAM_PANEL[1]) -> np.ndarray:
        ap = self.ap
        live = ap.camera_on and self.t - self.cam_t < 1.0
        if live:
            img = self.cam_img
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA) if img.shape[:2] != (h, w) else img.copy()
        else:
            img = np.full((h, w, 3), 30, np.uint8)
        _label(img, "DOWN CAMERA (IBN)", (6, 16), 0.45)
        if ap.ibn is None:
            _label(img, "no onboard imagery", (6, h // 2), 0.5, (200, 200, 200))
        elif not live:
            why = "IBN off" if not ap.sources["ibn"] else "off outside image zones"
            _label(img, f"camera {why}", (6, h // 2), 0.5, (200, 200, 200))
        elif not img.any():
            _label(img, "NO SIGNAL", (w // 2 - 50, h // 2), 0.7, (255, 80, 80))
        fix = ap.last_fix
        if live and ap.ibn is not None:
            fresh = fix is not None and fix.source == "ibn" and self.t - fix.t < 0.5
            if fresh:
                c = (255, 220, 0) if fix.accepted else (255, 80, 80)
                _label(img, f"MATCH NCC {ap.ibn.last_score:.2f}  {'accepted' if fix.accepted else 'REJECTED'}",
                       (6, h - 10), 0.45, c)
                cv2.rectangle(img, (2, 2), (w - 3, h - 3), c, 2)
            else:
                _label(img, ap.ibn.last_status[:40], (6, h - 10), 0.4, (200, 200, 200))
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (255, 255, 255), 1)
        return img

    def profile_panel(self, w: int = PROFILE_PANEL[0], h: int = PROFILE_PANEL[1]) -> np.ndarray:
        """TRN: surface height under the drone measured (baro - altimeter) vs the onboard map at the estimate."""
        img = np.full((h, w, 3), 30, np.uint8)
        prof = self.ap.trn.profile
        _label(img, "TERRAIN PROFILE (TRN)", (6, 16), 0.45)
        if len(prof) > 2 and self.ap.sources["trn"]:
            a = np.array(prof)[-300:]
            t, hm, hp = a[:, 0], a[:, 1], a[:, 2]
            lo, hi = float(min(hm.min(), hp.min())), float(max(hm.max(), hp.max()))
            pad = max(2.0, 0.1 * (hi - lo))
            lo, hi = lo - pad, hi + pad
            top, bot = 26, h - 22
            u = 4 + (t - t[0]) / max(t[-1] - t[0], 1e-6) * (w - 8)

            def v(z: np.ndarray) -> np.ndarray:
                return bot - (z - lo) / (hi - lo) * (bot - top)

            cv2.polylines(img, [np.column_stack([u, v(hp)]).astype(np.int32)], False, ORANGE, 2, cv2.LINE_AA)
            cv2.polylines(img, [np.column_stack([u, v(hm)]).astype(np.int32)], False, (255, 255, 255), 1, cv2.LINE_AA)
            _label(img, f"{hi - lo:.0f} m", (w - 50, 16), 0.35)
            _label(img, "white measured  orange onboard map", (6, h - 6), 0.35)
        else:
            _label(img, "TRN off" if not self.ap.sources["trn"] else "waiting for altimeter", (6, h // 2), 0.5,
                   (200, 200, 200))
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (255, 255, 255), 1)
        return img


# ---------------------------------------------------------------------------------------- Windows helpers

def _maximize_window() -> None:
    """Windows: maximize OUR MuJoCo viewer window (it opens taller than a laptop screen, hiding the panels)."""
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
            if pid.value != me:  # only our viewer window, never another MuJoCo app on the desktop
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


KEY_HELP = ("T\nI\nW\nC\nA\nR\n1 2 4 8\nSPACE",
            "TRN on/off\nIBN on/off\nwind gust\ncamera blackout\naltimeter failure\nrestart\nsim speed\npause")
SPEEDS = (1, 2, 4, 8)


def _camera_setup(cam: mujoco.MjvCamera, m: mujoco.MjModel, demo: LiveDemo) -> None:
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = m.body("x2").id
    cam.distance, cam.elevation = (28.0, -22.0) if demo.urban else (45.0, -25.0)
    heading = demo.goal - demo.start
    cam.azimuth = math.degrees(math.atan2(heading[1], heading[0]))


# ---------------------------------------------------------------------------------------- viewer mode

def run_viewer(args: argparse.Namespace) -> None:
    import mujoco.viewer

    speed = args.speed
    while True:  # one viewer window per flight; R closes it and starts a fresh flight
        demo = LiveDemo(args.world, args.config, args.seed, args.events)
        restart = _view_flight(demo, args, speed)
        demo.close()
        if restart is None:
            return
        speed = restart


def _view_flight(demo: LiveDemo, args: argparse.Namespace, speed: int) -> int | None:
    """Fly one flight in the viewer. Returns the sim speed to restart with (R), or None (window closed)."""
    import mujoco.viewer

    m, d = demo.fs.sim.model, demo.fs.sim.data
    keys: deque[int] = deque()
    paused = False
    restart: int | None = None
    print(f"{demo.fs.scenario.title}\nconfig {args.config} seed {args.seed} | GNSS DISABLED | "
          f"route {demo.ap.route.length:.0f} m", flush=True)
    with mujoco.viewer.launch_passive(m, d, key_callback=keys.append,
                                      show_left_ui=False, show_right_ui=False) as viewer:
        _camera_setup(viewer.cam, m, demo)
        wall0 = time.time()
        reported = False
        next_log = 0.0
        sim_budget = 0.0  # sim seconds owed to the wall clock (real-time pacing at `speed`)
        last_wall = time.perf_counter()
        next_panels = 0.0
        rtf_mark = (time.perf_counter(), 0.0)
        _maximize_window()
        while viewer.is_running():
            if args.quit_after and time.time() - wall0 > args.quit_after:
                break
            frame_start = time.perf_counter()
            while keys:
                code = keys.popleft()
                if code == 32:  # SPACE
                    paused = not paused
                    print(f"t={demo.t:6.1f}s {'PAUSED' if paused else 'RESUMED'}")
                elif 0 <= code < 256 and chr(code).isdigit() and int(chr(code)) in SPEEDS:
                    speed = int(chr(code))
                    print(f"t={demo.t:6.1f}s speed {speed}x")
                elif code == ord("R"):
                    print(f"t={demo.t:6.1f}s RESTART")
                    restart = speed
                elif 0 <= code < 256:
                    msg = demo.press(chr(code))
                    if msg:
                        print(f"t={demo.t:6.1f}s KEY {chr(code)}: {msg}")
            if restart is not None:
                break
            now = time.perf_counter()
            wall_dt, last_wall = min(now - last_wall, 0.1), now  # cap: a stall must not cause a burst
            if not paused:
                sim_budget += wall_dt * speed
                n = int(sim_budget / m.opt.timestep)
                sim_budget -= n * m.opt.timestep
                demo.advance(n)
                sim_budget = min(sim_budget, 0.2)  # slower than real time x speed: drop the debt, do not burst
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
                # The viewer binds T (transparent), I (inertia), A (auto connect), C (contacts), W (wireframe),
                # R (reflection), 1/2/4 (geom groups): undo those so our keys never change the picture.
                viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 0
                viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
                for f in (mujoco.mjtVisFlag.mjVIS_CONTACTPOINT, mujoco.mjtVisFlag.mjVIS_TRANSPARENT,
                          mujoco.mjtVisFlag.mjVIS_INERTIA, mujoco.mjtVisFlag.mjVIS_AUTOCONNECT):
                    viewer.opt.flags[f] = 0
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
                pw, ph = cw, side // 2
                viewer.set_images([
                    (mujoco.MjrRect(PANEL_GAP, PANEL_GAP, side, side),
                     cv2.resize(demo.map_panel(), (side, side), interpolation=cv2.INTER_LINEAR)),
                    (mujoco.MjrRect(2 * PANEL_GAP + side, PANEL_GAP, cw, ch),
                     cv2.resize(demo.cam_panel(), (cw, ch), interpolation=cv2.INTER_LINEAR)),
                    (mujoco.MjrRect(2 * PANEL_GAP + side, 2 * PANEL_GAP + ch, pw, ph),
                     cv2.resize(demo.profile_panel(), (pw, ph), interpolation=cv2.INTER_LINEAR)),
                ])
            sys.stdout.flush()
            viewer.sync()
            time.sleep(max(0.0, 1.0 / 60.0 - (time.perf_counter() - frame_start)))
    return restart


# ---------------------------------------------------------------------------------------- headless / record

def run_headless(args: argparse.Namespace) -> int:
    demo = LiveDemo(args.world, args.config, args.seed, args.events)
    wall = time.time()
    next_print = 0.0
    print(f"{demo.fs.scenario.title}\nconfig {args.config} seed {args.seed} | GNSS DISABLED")
    while demo.result is None:
        demo.advance(50)
        if demo.t >= next_print:
            next_print += 20.0
            print(demo.console_line(), flush=True)
    for t, text in demo.ap.decisions:
        if not text.startswith(NOISE):
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
    mp, cp, pp = demo.map_panel(), demo.cam_panel(), demo.profile_panel()
    img[H - PANEL_GAP - MAP_PANEL:H - PANEL_GAP, PANEL_GAP:PANEL_GAP + MAP_PANEL] = mp
    x0 = 2 * PANEL_GAP + MAP_PANEL
    img[H - PANEL_GAP - CAM_PANEL[1]:H - PANEL_GAP, x0:x0 + CAM_PANEL[0]] = cp
    y0 = H - 2 * PANEL_GAP - CAM_PANEL[1] - PROFILE_PANEL[1]
    img[y0:y0 + PROFILE_PANEL[1], x0:x0 + PROFILE_PANEL[0]] = pp

    def text_block(left: str, right: str, x: int, y: int, col2: int, scale: float = 0.5) -> None:
        ls, rs = left.split("\n"), right.split("\n")
        w = col2 + max(cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] for s in rs) + 16
        h = 20 * len(ls) + 10
        roi = img[y:y + h, x:min(x + w, W)]
        roi[:] = (roi * 0.45).astype(np.uint8)
        for i, (a, b) in enumerate(zip(ls, rs)):
            _label(img, a, (x + 8, y + 20 * (i + 1)), scale)
            _label(img, b, (x + col2, y + 20 * (i + 1)), scale, (140, 230, 255))

    left, right = demo.status()
    text_block(left, right, 10, 10, 150)
    text_block(*KEY_HELP, W - 260, 10, 80, 0.45)
    dl, dr = demo.decisions()
    text_block(dl, dr, W - 580, H - 20 * len(dl.split("\n")) - 20, 75, 0.42)
    _label(img, f"sim speed {speed}x", (W - 250, 200), 0.45)
    ban = demo.banner()
    for i, line in enumerate(ban.split("\n") if ban else []):
        tw = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]
        cv2.putText(img, line, ((W - tw) // 2, 40 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, line, ((W - tw) // 2, 40 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 230, 120), 2, cv2.LINE_AA)
    return img


def run_record(args: argparse.Namespace) -> int:
    out = Path(args.record or data_dir() / "v3" / "demo" / f"v3_{args.world}_{args.config}_seed{args.seed}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    demo = LiveDemo(args.world, args.config, args.seed, args.events)
    m = demo.fs.sim.model
    W, H, fps = 1280, 720, 30
    m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, W), max(m.vis.global_.offheight, H)
    renderer = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    _camera_setup(cam, m, demo)
    opt = mujoco.MjvOption()
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    gif_frames: list[np.ndarray] = []
    steps = int(round(args.speed / fps / m.opt.timestep))
    tail, n_frame, wall = 0, 0, time.time()
    while tail < 3 * fps:  # 3 s of video after the result
        demo.advance(steps)
        frame = compose_frame(demo, renderer, cam, opt, args.speed)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if args.gif and n_frame % 6 == 0:
            gif_frames.append(cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA))
        n_frame += 1
        if n_frame % (fps * 10) == 0:
            print(f"{demo.console_line()}  ({n_frame / fps:.0f} s video, wall {time.time() - wall:.0f} s)", flush=True)
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
    ap.add_argument("--world", default="s1")
    ap.add_argument("--config", choices=list(CONFIGS), default="ins_trn_ibn")
    ap.add_argument("--seed", type=int, default=DEMO_SEED)
    ap.add_argument("--speed", type=int, choices=SPEEDS, default=None,
                    help="sim speed (viewer and video); default 2 (S1) / 8 (S2)")
    ap.add_argument("--events", default="", help="key presses at sim times, e.g. '60:T,90:T,120:C'")
    ap.add_argument("--headless", action="store_true", help="no window: fly, print, score")
    ap.add_argument("--record", nargs="?", const="", default=None,
                    help="render offscreen to an mp4 (default $SFHACK_DATA/v3/demo/v3_<world>_<config>_seed<n>.mp4)")
    ap.add_argument("--gif", action="store_true", help="with --record: also write a GIF next to the mp4")
    ap.add_argument("--quit-after", type=float, default=0.0, help=argparse.SUPPRESS)  # wall s, viewer smoke test
    args = ap.parse_args()
    if args.speed is None:
        args.speed = 2 if args.world == "s1" else 8
    if args.headless:
        sys.exit(run_headless(args))
    if args.record is not None:
        sys.exit(run_record(args))
    run_viewer(args)


if __name__ == "__main__":
    main()
