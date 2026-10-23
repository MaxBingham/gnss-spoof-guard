"""V4 live demo: find the brown truck among six objects, decide with a clear margin, land on its marker pad.

    uv run python demos/10_target_landing.py --standalone --seed 3 --difficulty hard
    uv run python demos/10_target_landing.py --standalone --seed 3 --headless      # no window, print + score
    --nav gnss (default: honest GNSS, the subject is target selection) | vision | truth

What you see: the MuJoCo viewer following the drone. Bottom-left: the live down camera with the onboard
candidate boxes (cyan = tracking, yellow = CHOSEN target, red = rejected, green square = marker). In 3D: the
same candidate footprints on the ground. Bottom-right: the onboard decision log. Keys: SPACE pause,
1 / 2 / 4 sim speed. Truth is used only for the final score line (harness).
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_live import PANEL_GAP, _Geoms, _maximize_window, _snap_own_window  # noqa: E402

from sfhack import bus as topics  # noqa: E402
from sfhack.perception.target_detector import NAMES, project_to_image  # noqa: E402
from sfhack.sim.target_world import NAV_MODES, build_target_flight  # noqa: E402

CYAN, YELLOW, RED, GREEN = (0.0, 0.85, 1.0), (1.0, 0.85, 0.0), (0.95, 0.15, 0.1), (0.15, 0.9, 0.2)
STATUS_RGB = {"chosen": YELLOW, "rejected": RED, "tracking": CYAN}


class TargetDemo:
    def __init__(self, seed: int, difficulty: str, nav: str) -> None:
        self.fs, self.ext = build_target_flight(seed, difficulty, nav)
        self.seed, self.difficulty, self.nav = seed, difficulty, nav
        self.cands = None
        self.marker = None
        self.frame = None
        self.frame_pose = None
        self.score = None
        self.done_t: float | None = None
        b = self.fs.bus
        b.subscribe(topics.TARGET_CANDIDATES, self._on_cands)
        b.subscribe(topics.TARGET_MARKER, self._on_marker)
        b.subscribe(topics.CAMERA_DOWN, self._on_frame)

    def _on_cands(self, msg) -> None:
        self.cands = msg

    def _on_marker(self, msg) -> None:
        self.marker = msg

    def _on_frame(self, frame) -> None:
        self.frame, self.frame_pose = frame, self.fs.autopilot.pose()  # onboard pose = what the boxes use

    @property
    def t(self) -> float:
        return self.fs.t

    @property
    def phase(self) -> str:
        return self.fs.mission.phase

    def advance(self, n: int) -> None:
        for _ in range(n):
            if self.score is not None:
                return
            self.fs.step()
            if self.fs.done:
                self.done_t = self.done_t or self.t
            stop = (self.done_t is not None and self.t - self.done_t > 1.0) or self.t > 300.0
            if stop:
                self.score = self.ext.score(self.fs)
                print(f"t={self.t:6.1f}s SCORE (truth): {self.score.summary()}", flush=True)

    # ------------------------------------------------------------------ drawing

    def _footprint(self, c, z: float) -> np.ndarray:
        u = np.array([math.cos(c.yaw), math.sin(c.yaw)])
        v = np.array([-u[1], u[0]])
        L = c.length if math.isfinite(c.length) else 3.0
        W = c.width if math.isfinite(c.width) else 2.0
        pts = [c.xy + sx * L / 2 * u + sy * W / 2 * v for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1), (1, 1))]
        return np.array([[p[0], p[1], z] for p in pts])

    def draw_geoms(self, scn: mujoco.MjvScene) -> None:
        g = _Geoms(scn)
        if self.cands is None:
            return
        for c in self.cands.candidates:
            rgb = STATUS_RGB.get(c.status, CYAN)
            z = (c.height or 0.0) + 0.15
            g.polyline(self._footprint(c, z), rgb, 6.0 if c.status == "chosen" else 3.0)
            if c.status == "chosen":
                g.line([c.xy[0], c.xy[1], z], [c.xy[0], c.xy[1], z + 6.0], YELLOW, 4.0)

    def cam_panel(self, w: int = 400, h: int = 300) -> np.ndarray:
        if self.frame is None:
            return np.zeros((h, w, 3), np.uint8)
        img = self.frame.image.copy()
        if self.cands is not None and self.frame_pose is not None:
            for c in self.cands.candidates:
                z = c.height or 0.0
                uv = project_to_image(self._footprint(c, z), self.frame, self.frame_pose)
                if np.isnan(uv).any():
                    continue
                col = tuple(int(255 * v) for v in STATUS_RGB.get(c.status, CYAN))
                cv2.polylines(img, [uv.astype(np.int32)], True, col, 2 if c.status == "chosen" else 1, cv2.LINE_AA)
                best = max(c.probs, key=c.probs.get) if c.probs else "unknown"
                lab = f"{c.id} {NAMES.get(best, best)} {c.p_target:.2f}"
                p = uv.min(axis=0).astype(int)
                cv2.putText(img, lab, (int(p[0]), max(10, int(p[1]) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)
        if self.marker is not None and self.frame.t - self.marker.t < 0.5:
            cv2.polylines(img, [self.marker.corners_px.astype(np.int32)], True, (40, 230, 50), 2, cv2.LINE_AA)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        cv2.putText(img, f"DOWN CAMERA  {self.phase}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def status(self) -> tuple[str, str]:
        p = self.fs.autopilot.pose()
        left = "scenario\nnav\nphase\naltitude\ncandidates"
        n = len(self.cands.candidates) if self.cands else 0
        right = (f"seed {self.seed} {self.difficulty}\n{self.nav}\n{self.phase}\n{p.pos[2]:.1f} m\n{n}")
        if self.score is not None:
            left += "\nRESULT (truth)"
            s = self.score
            right += f"\n{'ON PAD' if s.on_pad else 'NOT ON PAD'} ({s.pad_error:.2f} m), chose {s.chosen_kind}"
        return left, right

    def decisions(self, n: int = 7) -> str:
        keep = ("TARGET", "INSPECT", "MARKER", "LANDED", "SEARCH", "TOUCHDOWN")
        lines = [f"{t:5.1f}  {txt}" for t, txt in self.fs.autopilot.decisions if txt.startswith(keep)]
        return "\n".join(x if len(x) <= 96 else x[:93] + "..." for x in lines[-n:])  # clear of the camera panel


def run_headless(a: argparse.Namespace) -> None:
    demo = TargetDemo(a.seed, a.difficulty, a.nav)
    t0 = time.time()
    while demo.score is None:
        demo.advance(500)
    for t, txt in demo.fs.autopilot.decisions:
        if not txt.startswith(("CANDIDATE", "RELOCALIZE")) or a.verbose:
            print(f"{t:6.1f} {txt}")
    print(f"wall {time.time() - t0:.0f} s")


def run_viewer(a: argparse.Namespace) -> None:
    import mujoco.viewer
    demo = TargetDemo(a.seed, a.difficulty, a.nav)
    m, d = demo.fs.sim.model, demo.fs.sim.data
    keys: deque[int] = deque()
    speed, paused = a.speed, False
    printed = 0
    with mujoco.viewer.launch_passive(m, d, key_callback=keys.append, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = m.body("x2").id
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 30.0, -35.0, 90.0
        _maximize_window()
        wall0 = last = time.perf_counter()
        budget, next_panel = 0.0, 0.0
        while viewer.is_running():
            if a.quit_after and time.perf_counter() - wall0 > a.quit_after:
                if a.snap:
                    _snap_own_window(Path(a.snap))
                break
            while keys:
                k = keys.popleft()
                if k == 32:
                    paused = not paused
                elif k in (ord("1"), ord("2"), ord("4")):
                    speed = int(chr(k))
            now = time.perf_counter()
            dt, last = min(now - last, 0.1), now
            if not paused:
                budget += dt * speed
                n = int(budget / m.opt.timestep)
                budget -= n * m.opt.timestep
                demo.advance(n)
            dec = demo.fs.autopilot.decisions
            for t, txt in dec[printed:]:
                if not txt.startswith("RELOCALIZE"):
                    print(f"{t:6.1f} {txt}", flush=True)
            printed = len(dec)
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                viewer.opt.geomgroup[:3] = 1
                demo.draw_geoms(viewer.user_scn)
            left, right = demo.status()
            viewer.set_texts([
                (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right),
                (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT, demo.decisions(), None),
                (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPRIGHT, "SPACE pause\n1/2/4 speed",
                 f"\n{speed}x" + ("  PAUSED" if paused else "")),
            ])
            vp = viewer.viewport
            if vp is not None and vp.height > 0 and now >= next_panel:
                next_panel = now + 1.0 / 15.0
                side = max(200, int(0.42 * vp.height))
                cw = side * 4 // 3
                viewer.set_images([(mujoco.MjrRect(PANEL_GAP, PANEL_GAP, cw, side),
                                    cv2.resize(demo.cam_panel(), (cw, side), interpolation=cv2.INTER_LINEAR))])
            viewer.sync()
            spare = 1.0 / 60.0 - (time.perf_counter() - now)
            if spare > 0:
                time.sleep(spare)
    demo.fs.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--standalone", action="store_true", help="plain-field world (the only mode so far; default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--difficulty", default="easy", choices=["easy", "hard"])
    ap.add_argument("--nav", default="gnss", choices=list(NAV_MODES))
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--speed", type=int, default=2)
    ap.add_argument("--quit-after", type=float, default=0.0, help="close the viewer after this many wall seconds")
    ap.add_argument("--snap", default="", help="with --quit-after: save a PNG of the viewer window before closing")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    (run_headless if a.headless else run_viewer)(a)


if __name__ == "__main__":
    main()
