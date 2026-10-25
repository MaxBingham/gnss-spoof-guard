"""Enemy-drone tracker: the UMA-16 at the base hears an ENEMY drone; our drone takes off, finds it with its gimballed
front camera (YOLO), follows it at a standoff and reports it to the base over the ~80 % link. Track and report only.

    uv run python demos/52_track_enemy.py                          # viewer; bearing from the keys or UDP 5005
    uv run python demos/52_track_enemy.py --bearing 60             # enemy heard at 060 at once
    uv run python demos/52_track_enemy.py --headless --bearing 60 --script-bearing   # no window: fly, print, score
    uv run python demos/52_track_enemy.py --record out.mp4 --bearing 60 --script-bearing --speed 2      # MP4
    uv run python demos/52_track_enemy.py --headless --detector oracle   # perfect detector (truth boxes): flight logic
    uv run python demos/52_track_enemy.py --headless --bearing 60 --comms inproc   # old in-process base <-> drone path

Base <-> drone: MAVLink 2 over UDP (signed; --comms mavlink, the default), base UDP 14560, drone UDP 14561, and the
base forwards the drone's packets to QGroundControl on UDP 14550 (docs/MAVLINK.md). --comms inproc: the in-process
path (same mission, no packets).

Acoustic re-acquisition: while the link is up the base sends the UMA-16 bearing to the enemy ~1 Hz (sigma grows with
the distance; the base tells our drone and the enemy apart by their rotor sound, ASSUMED to work in the sim:
sfhack/sim/base_acoustic.py). LOST predicts on the latest bearing, TRACKING checks the track against it.
--forced-losses N: N times the enemy is hidden from our camera for 8 s (as behind a tree row) and turns 30 deg.

Scenario: the first bearing = the enemy appears 500 m out along it, ~40 m above the terrain, and approaches the base
at ~4 m/s with a little weaving; its direction follows the LATEST bearing (live: someone walks around the UMA-16 with
a speaker playing drone sound and the bearings come over UDP 5005; --script-bearing: a slow scripted sweep). The
base launches our drone with the first bearing; later bearings reach it only while the link is up.
Onboard: sfhack/mission/tracker_pilot.py + sfhack/perception/drone_detector.py (never sees the truth). Simulator:
sfhack/sim/enemy.py (placeholder enemy, birds), sfhack/sim/gimbal_camera.py, sfhack/sim/link.py. This file is
harness: it reads the truth to DISPLAY and SCORE only.

Keys (as the recon dashboard): LEFT / RIGHT turn the pending bearing by 5 deg, ENTER sends it (first one = enemy heard
+ launch), F random bearing, N GNSS off / on, L link: model / forced down / forced up, R recall (RETURN, over the
link), V record on / off (MP4 under $SFHACK_DATA/demo), C chase / overview / enemy camera, 1 2 4 sim speed,
SPACE pause, BACKSPACE reset, O reset camera, Z screenshot, Tab fullscreen, Q quit. Mouse: drag orbits, wheel zooms.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

DEMOS = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMOS))
sys.path.insert(0, str(DEMOS.parent))

import enemy_world as ew  # noqa: E402
from track_scenario import OracleDetector, TrackScenario  # noqa: E402

from sfhack.config import TrackerConfig, data_dir  # noqa: E402

VIEW_SIZE, RECORD_SIZE = (1280, 720), (1920, 1080)
WINDOW = "Enemy-drone tracker"
KEYS = {2424832: "LEFT", 65361: "LEFT", 2555904: "RIGHT", 65363: "RIGHT", 13: "ENTER", 10: "ENTER",
        8: "BACKSPACE", 9: "TAB", 32: "SPACE", 27: "Q"}


class BearingScript:
    """Scripted UMA-16 bearings for headless runs / the evaluation: a slow sweep around the first bearing (someone
    walking around the array with the speaker), one update per second. With `rate` (deg/s, + = clockwise, i.e.
    increasing compass azimuth) the bearing turns at that constant rate instead. The enemy follows the bearing only
    within its lateral speed limit (`EnemyDrone.lateral_max`), so far out it lags the script."""

    def __init__(self, b0: float, seed: int, amp: float | None = None, period: float | None = None,
                 rate: float | None = None) -> None:
        self.rate = rate
        rng = np.random.default_rng(seed + 4200)
        self.b0 = b0
        self.amp = float(rng.uniform(5.0, 30.0) if amp is None else amp)
        self.period = float(rng.uniform(120.0, 240.0) if period is None else period)
        self.phase = float(rng.choice([-1.0, 1.0]))
        self.t0: float | None = None
        self.next_t = 0.0

    def __call__(self, t: float) -> float | None:
        if self.t0 is None:
            self.t0, self.next_t = t, t + 1.0
            return self.b0
        if t < self.next_t:
            return None
        self.next_t += 1.0
        if self.rate is not None:
            return (self.b0 + self.rate * (t - self.t0)) % 360.0
        return (self.b0 + self.phase * self.amp * math.sin(2 * math.pi * (t - self.t0) / self.period)) % 360.0


def make_script(b0: float, args) -> BearingScript | None:
    if args.bearing_rate is not None:
        return BearingScript(b0, args.seed, rate=args.bearing_rate)
    return BearingScript(b0, args.seed) if args.script_bearing else None


def make_detector(args, tcfg: TrackerConfig):
    if args.detector == "oracle":
        return None  # wired after the scenario exists (it reads the camera's segmentation)
    from sfhack.perception.drone_detector import DroneDetector

    det = DroneDetector(tcfg, weights=args.weights)
    print(f"detector: {det.weights} on {det.device}")
    return det


def new_scenario(world, args, detector, tcfg: TrackerConfig) -> TrackScenario:
    sc = TrackScenario(world, seed=args.seed, detector=detector, tcfg=tcfg, enemy_speed=args.enemy_speed,
                       availability=args.link, score=args.score or args.detector == "oracle",
                       send_bearings=not args.no_base_bearing, forced_losses=args.forced_losses, comms=args.comms,
                       qgc=not args.no_qgc)
    if args.detector == "oracle":
        sc.pilot.detector = OracleDetector(sc.cam)
    if args.gnss_off:
        sc.rig.set_gnss(False)
    return sc


def summary(sc: TrackScenario) -> str:
    p = sc.pilot
    err = sc.report_errors()
    tr = sc.state_time.get("TRACKING", 0.0)
    flight = sum(v for k, v in sc.state_time.items() if k not in ("WAIT", "LANDED"))
    e = "-" if not len(err) else f"{np.mean(err[:, 0]):.1f} m horizontal (median {np.median(err[:, 0]):.1f}), " \
                                  f"{np.mean(err[:, 1]):.1f} m 3D"
    fd = "-" if p.first_detect_t is None else f"{p.first_detect_t - (sc.launch_t or 0):.1f} s after launch"
    ft = "-" if p.first_track_t is None else f"{p.first_track_t - (sc.launch_t or 0):.1f} s"
    ls = sc.losses
    loss = "" if not ls else (f"; forced losses {len(ls)}: re-acquired {sum(e.reacquired for e in ls)} "
                              f"(times after reappearing: {[round(e.reacq_t - e.t1, 1) for e in ls if e.reacquired]} s)")
    return (f"RESULT state={p.state} t={sc.t:.1f}s  first detection {fd}, tracking from {ft}; tracking "
            f"{tr:.0f} s of {flight:.0f} s flight; reports made {p.outbox.made}, received {len(sc.received)} "
            f"(link up {100 * sc.link.measured_availability:.0f} %); report error {e}; closest approach "
            f"{sc.min_dist:.1f} m; own position error {sc.own_err():.1f} m; bearings sent {sc.bearings_sent}, "
            f"suspect tracks {p.n_suspect}, new tracks refused by the bearing gate {p.n_gated}{loss}")


# ---------------------------------------------------------------------------------------- headless

def run_headless(world, args, tcfg: TrackerConfig) -> int:
    sc = new_scenario(world, args, make_detector(args, tcfg), tcfg)
    b0 = args.bearing if args.bearing is not None else 60.0
    script = make_script(b0, args)
    print(sc.hear(b0, "--bearing"))
    wall, nxt, last_dec = time.time(), 0.0, 0
    while sc.t < args.max_t and not (sc.pilot.state == "LANDED" and sc.launch_t is not None):
        sc.advance(50)
        if script is not None:
            az = script(sc.t)
            if az is not None:
                sc.hear(az, "script")
        while last_dec < len(sc.pilot.decisions):
            t, text = sc.pilot.decisions[last_dec]
            last_dec += 1
            print(f"   {t:6.1f}s DRONE {text}", flush=True)
        if sc.t >= nxt:
            nxt += 10.0
            d = sc.true_dist()
            trk = sc.pilot.tracker.track
            te = "-" if trk is None or sc.enemy_at(sc.t) is None else \
                f"{np.linalg.norm(trk.pos[:2] - sc.enemy.pose.pos[:2]):5.1f}"
            print(f"t={sc.t:6.1f}s {sc.pilot.state:9s} link {'UP  ' if sc.link.is_up else 'DOWN'} "
                  f"dist to enemy {'-' if d is None else f'{d:6.1f}'} m  track err {te} m  AGL {sc.agl():4.0f}  "
                  f"rx {len(sc.received):3d}  det {sc.pilot.detect_ms:4.1f} ms", flush=True)
    print(summary(sc))
    if sc.mav is not None:
        print(sc.mav.stats())
    sc.close()
    print(f"wall {time.time() - wall:.1f} s ({sc.t / max(time.time() - wall, 1e-6):.2f}x real time)")
    return 0 if sc.pilot.first_track_t is not None else 1


# ---------------------------------------------------------------------------------------- record / viewer

def run_record(world, args, tcfg: TrackerConfig) -> int:
    from liveview.track_dash import TrackComposer
    from demo_live import _VideoOut

    out = Path(args.record)
    out.parent.mkdir(parents=True, exist_ok=True)
    W, H = args.size
    fps = 30
    sc = new_scenario(world, args, make_detector(args, tcfg), tcfg)
    comp = TrackComposer(sc, W, H)
    comp.mode = args.view
    b0 = args.bearing if args.bearing is not None else 60.0
    script = make_script(b0, args)
    sc.hear(b0, "--bearing")
    video = _VideoOut(out, W, H, fps)
    steps = int(round(args.speed / fps / sc.dt))
    t_end = args.until if args.until else args.max_t
    n, wall = 0, time.time()
    while sc.t < t_end and not (sc.pilot.state == "LANDED" and sc.launch_t is not None):
        sc.advance(steps)
        if script is not None:
            az = script(sc.t)
            if az is not None:
                sc.hear(az, "script")
        video.write(comp.frame(sc, 1.0 / fps, args.speed, False, True))
        n += 1
        if n % (fps * 5) == 0:
            print(f"t={sc.t:6.1f}s {sc.pilot.state:9s} ({n / max(time.time() - wall, 1e-6):.1f} frames/s)", flush=True)
    video.close()
    comp.close()
    print(summary(sc))
    if sc.mav is not None:
        print(sc.mav.stats())
    sc.close()
    print(f"video: {out}  ({n} frames, {n / fps:.1f} s)")
    return 0


def key_of(code: int) -> str:
    if code in KEYS:
        return KEYS[code]
    c = code & 0xFF
    if c in KEYS:
        return KEYS[c]
    return chr(c).upper() if 32 < c < 127 else ""


def run_viewer(world, args, tcfg: TrackerConfig) -> None:
    from demo_live import WHEEL_ZOOM_SIGN, _mouse_wheel_delta, _VideoOut
    from liveview.track_dash import TrackComposer

    W, H = args.size
    detector = make_detector(args, tcfg)
    sc = new_scenario(world, args, detector, tcfg)
    comp = TrackComposer(sc, W, H)
    comp.mode = args.view
    udp = world.mod.UdpBearings(args.udp_port)
    pending = float(args.bearing) if args.bearing is not None else 0.0
    if args.bearing is not None:
        print(f"t={sc.t:6.1f}s {sc.hear(args.bearing, '--bearing')}")
    script = make_script(pending, args)
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
    video = None
    rec_t0, rec_n, rec_path = 0.0, 0, None
    wall0 = last_wall = time.perf_counter()
    budget, last_dec, snap = 0.0, 0, False
    link_modes = [None, False, True]
    while True:
        if args.quit_after and time.perf_counter() - wall0 > args.quit_after:
            break
        az = udp.pop()
        if az is not None:
            pending = az % 360.0
            print(f"t={sc.t:6.1f}s UDP: {sc.hear(az, f'UDP {args.udp_port}')}", flush=True)
        if script is not None and sc.first_heard_t is not None and not paused:
            az = script(sc.t)
            if az is not None:
                pending = az
                sc.hear(az, "script")
        while keys:
            k = keys.popleft()
            if k == "LEFT":
                pending = (pending - 5.0) % 360.0
            elif k == "RIGHT":
                pending = (pending + 5.0) % 360.0
            elif k == "ENTER":
                print(f"t={sc.t:6.1f}s KEY ENTER: {sc.hear(pending, 'keys')}", flush=True)
            elif k == "F":
                pending = random.uniform(0.0, 360.0)
                print(f"t={sc.t:6.1f}s KEY F: {sc.hear(pending, 'random (F)')}", flush=True)
            elif k == "N":
                sc.rig.set_gnss(not sc.rig.gnss_on)
                sc.banner = (sc.t, "GNSS ON" if sc.rig.gnss_on else "GNSS OFF: IMU + compass + baro only", None)
                print(f"t={sc.t:6.1f}s KEY N: GNSS {'ON' if sc.rig.gnss_on else 'OFF'}", flush=True)
            elif k == "L":
                i = (link_modes.index(sc.link.forced) + 1) % 3
                sc.link.forced = link_modes[i]
                print(f"t={sc.t:6.1f}s KEY L: link {['model (~80 %)', 'forced DOWN', 'forced UP'][i]}", flush=True)
            elif k == "R":
                sc.recall()
                print(f"t={sc.t:6.1f}s KEY R: recall queued for the link", flush=True)
            elif k == "C":
                comp.cycle_mode()
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
                    rec_path = data_dir() / "demo" / f"track_enemy_{datetime.now():%Y%m%d_%H%M%S}.mp4"
                    rec_path.parent.mkdir(parents=True, exist_ok=True)
                    video, rec_t0, rec_n = _VideoOut(rec_path, W, H, 30), time.perf_counter(), 0
                    print(f"t={sc.t:6.1f}s RECORDING -> {rec_path}", flush=True)
                else:
                    video.close()
                    video = None
                    print(f"t={sc.t:6.1f}s recording stopped: {rec_path} ({rec_n / 30:.1f} s)", flush=True)
            elif k == "BACKSPACE":
                print(f"t={sc.t:6.1f}s RESET", flush=True)
                comp.close()
                sc.close()
                sc = new_scenario(world, args, detector, tcfg)
                comp = TrackComposer(sc, W, H)
                script = make_script(pending, args)
                budget, last_dec = 0.0, 0
            elif k == "Q":
                if video is not None:
                    video.close()
                comp.close()
                sc.close()
                cv2.destroyAllWindows()
                return
        now = time.perf_counter()
        wall_dt, last_wall = min(now - last_wall, 0.1), now
        if not paused:
            budget += wall_dt * speed
            n = int(budget / sc.dt)
            budget -= n * sc.dt
            sc.advance(n)
        while last_dec < len(sc.pilot.decisions):
            t, text = sc.pilot.decisions[last_dec]
            last_dec += 1
            print(f"t={t:6.1f}s DRONE {text}", flush=True)
        comp.pending = pending
        frame = comp.frame(sc, wall_dt, speed, paused, video is not None)
        if video is not None:
            due = int((time.perf_counter() - rec_t0) * 30) + 1
            for _ in range(min(max(due - rec_n, 0), 4)):
                video.write(frame)
                rec_n += 1
            rec_n = max(rec_n, due - 4)
        if snap:
            snap = False
            p = data_dir() / "demo" / f"track_enemy_t{sc.t:05.1f}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(f"snapshot: {p}", flush=True)
        cv2.imshow(WINDOW, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        code = cv2.waitKeyEx(1)
        if code != -1:
            keys.append(key_of(code))
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
    if video is not None:
        video.close()
    comp.close()
    sc.close()
    cv2.destroyAllWindows()


def _size(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", choices=("farm", "simple"), default="farm")
    ap.add_argument("--bearing", type=float, default=None, help="enemy heard at this bearing (deg) right away")
    ap.add_argument("--script-bearing", action="store_true", help="scripted slow bearing sweep (1 update / s)")
    ap.add_argument("--bearing-rate", type=float, default=None,
                    help="scripted bearing turning at this rate, deg/s (+ = clockwise), from --bearing (1 update / s)")
    ap.add_argument("--detector", choices=("yolo", "oracle"), default="yolo")
    ap.add_argument("--weights", default=None, help="YOLO weights (default TrackerConfig.weights under $SFHACK_DATA)")
    ap.add_argument("--enemy-speed", type=float, default=4.0, help="m/s (0 = random 3-5)")
    ap.add_argument("--link", type=float, default=0.8, help="link availability")
    ap.add_argument("--comms", choices=("mavlink", "inproc"), default="mavlink",
                    help="base <-> drone transport: MAVLink 2 over UDP (default) or the in-process path")
    ap.add_argument("--no-qgc", action="store_true", help="do not forward the drone's MAVLink to QGC (UDP 14550)")
    ap.add_argument("--gnss-off", action="store_true", help="GNSS off from the start (key N toggles)")
    ap.add_argument("--no-base-bearing", action="store_true", help="the base sends no acoustic bearings in flight")
    ap.add_argument("--forced-losses", type=int, default=0,
                    help="hide the enemy from the camera for 8 s while it turns 30 deg, this many times")
    ap.add_argument("--birds", type=int, default=16)
    ap.add_argument("--battery", type=float, default=None, help="flight time budget, s (TrackerConfig.battery_s)")
    ap.add_argument("--score", action="store_true", help="score every camera frame (segmentation truth, slower)")
    ap.add_argument("--udp-port", type=int, default=5005, help="0 = off")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0, help="sim speed (viewer: 1 / 2 / 4 keys; record: any)")
    ap.add_argument("--view", choices=("chase", "overview", "enemy"), default="chase")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--record", nargs="?", const=str(data_dir() / "demo" / "track_enemy.mp4"), default=None)
    ap.add_argument("--until", type=float, default=0.0, help="with --record: stop at this sim time")
    ap.add_argument("--max-t", type=float, default=400.0)
    ap.add_argument("--size", type=_size, default=None, help="frame size WxH (viewer 1280x720, video 1920x1080)")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--quit-after", type=float, default=0.0, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.enemy_speed == 0:
        args.enemy_speed = None
    if args.size is None:
        args.size = RECORD_SIZE if args.record else VIEW_SIZE
    tcfg = TrackerConfig()
    if args.battery is not None:
        tcfg.battery_s = args.battery
    world = ew.build(args.world, n_birds=args.birds, seed=args.seed)
    if args.headless:
        sys.exit(run_headless(world, args, tcfg))
    if args.record:
        sys.exit(run_record(world, args, tcfg))
    run_viewer(world, args, tcfg)


if __name__ == "__main__":
    main()
