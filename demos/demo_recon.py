"""Recon demo: acoustic bearing -> autonomous night recon flight -> thermal find -> home -> report.

    python demos/demo_recon.py --mission ukraine-1                       # viewer, bearing simulated from the targets
    python demos/demo_recon.py --mission ukraine-1 --bearing bearing.json --nav lights
    python demos/demo_recon.py --headless --seed 3                       # no window: run, print the score
    python demos/demo_recon.py --record out.mp4 --until 900              # offscreen video

Keys: N GNSS off now, L base lights on / off (when the world has them), SPACE pause, 1 / 2 / 4 / 8 sim speed,
R restart, O reset camera, TAB fullscreen, Q quit. Mouse: drag = orbit, wheel = zoom.
`--bearing` default: $SFHACK_DATA/acoustic/bearing.json when --live-bearing, else simulated by the harness.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from liveview.recon import ReconComposer, ReconView  # noqa: E402

from sfhack.config import data_dir, default_config  # noqa: E402
from sfhack.recon_runner import ReconFlight  # noqa: E402

WINDOW = "recon - acoustic cue -> night recon (no GNSS, no link)"


def make_flight(a: argparse.Namespace) -> ReconFlight:
    cfg = default_config()
    if a.battery_min:
        cfg.recon.battery_s = a.battery_min * 60.0
    bearing = a.bearing
    if bearing is None and a.live_bearing:
        bearing = str(data_dir() / "acoustic" / "bearing.json")
    if bearing is not None:
        print(f"bearing: {bearing} -> {json.loads(Path(bearing).read_text(encoding='utf-8'))}")
    return ReconFlight(mission=a.mission, seed=a.seed, bearing=bearing, nav=a.nav, cfg=cfg,
                       gnss_off_m=None if a.gnss_on else 50.0, thermal_hz=a.thermal_hz,
                       run_id=f"{time.strftime('%Y%m%d-%H%M%S')}_demo_s{a.seed}")


def press(view: ReconView, k: str) -> str:
    k = k.upper()
    if k == "N":
        view.rf.gnss_off()
        return "GNSS off"
    if k == "L":
        fs = view.fs
        for name in ("toggle_lights", "set_lights"):  # recon-world / recon-nav hook
            f = getattr(fs, name, None) or getattr(fs.sim, name, None)
            if f is not None:
                try:
                    return f"lights {f()}"
                except TypeError:
                    on = not getattr(fs.sim, "lights_on", True)
                    f(on)
                    return f"lights {'on' if on else 'off'}"
        return "no base lights in this world yet"
    return ""


def run_viewer(a: argparse.Namespace) -> None:
    W, H = a.size
    view = ReconView(make_flight(a))
    comp = ReconComposer(view, W, H)
    keys: deque[str] = deque()
    mouse = {"drag": None}

    def on_mouse(event, x, y, flags, _p) -> None:
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
            comp.chase.zoom = float(np.clip(comp.chase.zoom * (0.9 if flags > 0 else 1.1), 0.3, 8.0))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(WINDOW, W, H)
    cv2.setMouseCallback(WINDOW, on_mouse)
    fullscreen = False
    speed, paused = a.speed, False
    last = time.perf_counter()
    budget = 0.0
    next_log = 0.0
    finished = False
    while True:
        while keys:
            k = keys.popleft()
            if k == " ":
                paused = not paused
            elif k in "1248":
                speed = int(k)
            elif k.upper() == "R":
                view.rf.close()
                view = ReconView(make_flight(a))
                comp.reset(view)
                finished, next_log, budget = False, 0.0, 0.0
            elif k.upper() == "O":
                comp.chase.reset_offsets()
            elif k.upper() == "Q":
                view.rf.close()
                comp.close()
                cv2.destroyAllWindows()
                return
            elif k == "\t":
                fullscreen = not fullscreen
                cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            else:
                msg = press(view, k)
                if msg:
                    print(f"t={view.t:6.1f}s KEY {k.upper()}: {msg}", flush=True)
        now = time.perf_counter()
        wall_dt, last = min(now - last, 0.1), now
        if not paused and not view.rf.done:
            budget += wall_dt * speed
            n = int(budget / view.fs.sim.model.opt.timestep)
            budget -= n * view.fs.sim.model.opt.timestep
            view.advance(n)
            if view.t >= next_log:
                next_log = view.t + 10.0
                print(f"t={view.t:6.1f}s {view.m.phase:9s} finds {sum(t.verified for t in view.m.tracks)}", flush=True)
        if view.rf.done and not finished:
            finished = True
            s = view.rf.finish()
            print(f"REPORT: {view.m.report['summary']}\nscore: {json.dumps(s)}\nfiles: {view.rf.out_dir}", flush=True)
        frame = comp.frame(view, wall_dt, speed, paused)
        cv2.imshow(WINDOW, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        code = cv2.waitKey(1)
        if code >= 0:
            keys.append(chr(code & 0xFF))
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            view.rf.close()
            comp.close()
            return


def run_headless(a: argparse.Namespace) -> int:
    rf = make_flight(a)
    s = rf.run(progress_s=30.0)
    for t, text in rf.recon.decision_log:
        print(f"{t:7.1f}  {text}")
    print(json.dumps(s, indent=1))
    rf.close()
    return 0


def run_record(a: argparse.Namespace) -> int:
    W, H = a.size
    view = ReconView(make_flight(a))
    comp = ReconComposer(view, W, H)
    out = Path(a.record)
    out.parent.mkdir(parents=True, exist_ok=True)
    fps = 30
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    dt_frame = a.speed / fps
    hold = 0
    while view.t < (a.until or 1e9):
        if not view.rf.done:
            view.advance(int(round(dt_frame / view.fs.sim.model.opt.timestep)))
        else:
            if view.rf.score is None:
                view.rf.finish()
            hold += 1
            if hold > fps * 6:
                break
        frame = comp.frame(view, 1.0 / fps, a.speed, False)
        vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"video: {out}")
    view.rf.close()
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mission", default="ukraine-1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bearing", default=None, help="bearing.json (BearingEstimate); default: simulated")
    ap.add_argument("--live-bearing", action="store_true", help="use $SFHACK_DATA/acoustic/bearing.json (demos/40_uma16_live.py)")
    ap.add_argument("--nav", default=None, help="imu | flow | lights (recon-nav; default: the FlightSystem default)")
    ap.add_argument("--gnss-on", action="store_true", help="keep GNSS on (development stand-in for a working nav)")
    ap.add_argument("--thermal-hz", type=float, default=2.0)
    ap.add_argument("--battery-min", type=float, default=0.0)
    ap.add_argument("--speed", type=int, default=4)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--record", nargs="?", const=str(data_dir() / "demo" / "recon.mp4"), default=None)
    ap.add_argument("--until", type=float, default=0.0)
    a = ap.parse_args()
    a.size = (1920, 1080) if a.record else (1600, 900)
    if a.headless:
        sys.exit(run_headless(a))
    if a.record:
        sys.exit(run_record(a))
    run_viewer(a)


if __name__ == "__main__":
    main()
