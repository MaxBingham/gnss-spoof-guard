"""Fly around the base at night: the fence with its glowing, blinking navigation lights in 3D, and the rear NIR
camera (sfhack/sim/rear_camera.py) seeing them. Kinematic flight (no controller): the drone is placed on a scripted
path with its tail (rear camera) towards the base: climb at the pad -> orbit the base at 250 m -> out along the
recon bearing to the map edge -> back.

    uv run python -m sfhack.sim.recon_world --mission ukraine-1          # once: base, lights, targets
    uv run python demos/41_base_view.py [--mission ukraine-1] [--agl 80]
    uv run python demos/41_base_view.py --headless [--out DIR]           # PNG frames + light visibility table

Keys (window): SPACE pause, 1/2/4 speed, L lights on/off, C chase / overview camera, R restart, Q / ESC quit.
The labels on the rear camera panel are TRUTH (display only): light id; green = a decodable light (bright when on,
dark green = off in this frame of its code), gray = in view but behind terrain / a house, orange = stray light.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from liveview.night import apply_night_lighting  # noqa: E402

from sfhack import bus as topics  # noqa: E402
from sfhack.bus import Bus  # noqa: E402
from sfhack.config import Config  # noqa: E402
from sfhack.messages import CameraTiltCmd  # noqa: E402
from sfhack.missionprep.build import mission_dir  # noqa: E402
from sfhack.sim.base_lights import BlinkAnimator  # noqa: E402
from sfhack.sim.mujoco_sim import MujocoSim  # noqa: E402
from sfhack.sim.rear_camera import RearCameraSensor  # noqa: E402
from sfhack.world.scene import PackageWorld  # noqa: E402

PITCH_DEG = 6.0  # cruise nose-down pitch of the posed drone
DETECT_DN = 25.0  # peak above which a spot counts as detectable (background + noise ~ 5-10 DN)


def unit(az_deg: float) -> np.ndarray:
    return np.array([math.sin(math.radians(az_deg)), math.cos(math.radians(az_deg))])


class Path3:
    """Scripted path: (t) -> xy, z AGL, heading. Tail towards the base centre, always."""

    def __init__(self, world: PackageWorld, centre: np.ndarray, bearing: float, agl: float, speed: float = 12.0) -> None:
        self.world, self.c, self.agl, self.v = world, centre, agl, speed
        self.legs: list[tuple[float, object]] = []
        r_orbit = 250.0
        rmax = self._max_range(bearing) - 20.0
        climb = agl / 3.0
        a0 = bearing - 90.0
        self.segments = [
            ("climb", climb, lambda s: (centre, s * agl)),
            ("out to orbit", r_orbit / speed, lambda s: (centre + s * r_orbit * unit(a0), agl)),
            ("orbit", 2 * math.pi * r_orbit / speed, lambda s: (centre + r_orbit * unit(a0 + 360.0 * s), agl)),
            ("to bearing", (r_orbit * math.radians(90)) / speed,
             lambda s: (centre + r_orbit * unit(a0 + 90.0 * s), agl)),
            ("outbound", (rmax - r_orbit) / speed, lambda s: (centre + (r_orbit + s * (rmax - r_orbit)) * unit(bearing), agl)),
            ("return", (rmax - 40.0) / speed, lambda s: (centre + (rmax - s * (rmax - 40.0)) * unit(bearing), agl)),
        ]
        self.T = sum(d for _, d, _ in self.segments)

    def _max_range(self, az: float) -> float:
        g = self.world.terrain_grid
        r = 0.0
        while True:
            p = self.c + (r + 10.0) * unit(az)
            if not (g.min_x + 40 < p[0] < g.max_x - 40 and g.min_y + 40 < p[1] < g.max_y - 40):
                return r
            r += 10.0

    def at(self, t: float) -> tuple[str, np.ndarray, float, float]:
        t = t % self.T
        for name, dur, f in self.segments:
            if t <= dur:
                xy, agl = f(t / max(dur, 1e-6))
                break
            t -= dur
        xy = np.asarray(xy, float)
        d = self.c - xy
        heading = math.atan2(-d[1], -d[0]) if np.hypot(*d) > 1.0 else math.radians(90.0 - 45.0)  # nose away from base
        return name, xy, float(agl), heading


def place(sim: MujocoSim, xy: np.ndarray, z: float, yaw: float, pitch: float = math.radians(PITCH_DEG)) -> None:
    """Pose the drone (nose-down cruise pitch: tilts the rear camera up, as in real flight)."""
    d = sim.data
    q_yaw = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    q_pitch = np.array([math.cos(pitch / 2), 0, math.sin(pitch / 2), 0])
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, q_yaw, q_pitch)
    d.qpos[0:3] = [xy[0], xy[1], z]
    d.qpos[3:7] = q
    d.qvel[:] = 0.0
    mujoco.mj_forward(sim.model, d)


def rear_panel(frame: np.ndarray, truth: list[dict], scale: float = 1.0) -> np.ndarray:
    img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    n_ok = 0
    for r in truth:
        if not r["in_view"] or r["u"] is None:
            continue
        u, v = int(r["u"] * scale), int(r["v"] * scale)
        if r["occluded"]:
            col = (110, 110, 110)
        elif r["peak_on_dn"] >= DETECT_DN:
            col = ((80, 230, 80) if r["on"] else (40, 110, 40)) if r["nav"] else (60, 170, 255)
            n_ok += r["nav"]
        else:
            continue
        cv2.circle(img, (u, v), 9, col, 1, cv2.LINE_AA)
        cv2.putText(img, r["id"], (u + 10, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
    cv2.putText(img, f"REAR NIR 20 deg  lights decodable: {n_ok}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 255), 1, cv2.LINE_AA)
    return img


def chase_camera(xy: np.ndarray, z: float, heading: float, mode: str, centre: np.ndarray) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    if mode == "chase":  # behind the drone's nose-to-tail line, looking at the drone and the base beyond it
        cam.lookat[:] = [xy[0], xy[1], z]
        cam.distance = 40.0
        cam.azimuth = math.degrees(heading) + 20.0
        cam.elevation = -12.0
    else:  # overview: the base from above, drone somewhere around
        cam.lookat[:] = [centre[0], centre[1], 0.0]
        cam.distance = 300.0
        cam.azimuth = -60.0
        cam.elevation = -35.0
    return cam


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mission", default="ukraine-1")
    ap.add_argument("--agl", type=float, default=80.0)
    ap.add_argument("--speed", type=float, default=12.0, help="path speed (m/s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--every", type=float, default=10.0, help="headless: save a frame every N s of path time")
    ap.add_argument("--fixed-mount", action="store_true", help="do not move the rear camera gimbal")
    ap.add_argument("--camera", choices=("chase", "overview"), default="chase", help="initial 3D camera")
    a = ap.parse_args()

    mdir = mission_dir(a.mission)
    wdir = mdir / "world"
    if not (wdir / "base_lights.json").exists():
        sys.exit(f"no base in {wdir}: run  uv run python -m sfhack.sim.recon_world --mission {a.mission}")
    cfg = Config()
    world = PackageWorld(wdir)
    base = json.loads((wdir / "base.json").read_text(encoding="utf-8"))
    rt = json.loads((wdir / "recon_truth.json").read_text(encoding="utf-8")) if (wdir / "recon_truth.json").exists() else {}
    centre = np.array(base["center_enu"][:2])
    bearing = float(rt.get("bearing_deg", 60.0))
    sim = MujocoSim(cfg, Bus(), world, spawn_xy=tuple(centre), seed=a.seed, spawn_z=world.ground_z(*centre))
    apply_night_lighting(sim.model)
    W, H = 960, 540
    chase_r = mujoco.Renderer(sim.model, H, W)
    sim.renderer = chase_r  # one GL context: the rear camera's background renders through it too
    rear = RearCameraSensor(sim, sim.bus, wdir, seed=a.seed)
    anim = BlinkAnimator(sim.model, rear.lights)
    path = Path3(world, centre, bearing, a.agl, a.speed)
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
    print(f"base at {centre.round(1)}, {len(rear.lights.lights.lights)} lights, bearing {bearing:.1f} deg; "
          f"path {path.T:.0f} s (orbit 250 m, out to the map edge and back), {a.agl:.0f} m AGL")

    out = Path(a.out) if a.out else Path(mdir.parent.parent / "recon" / "base_view")
    if a.headless:
        out.mkdir(parents=True, exist_ok=True)
    t, speed, paused, mode = 0.0, 4.0, False, a.camera
    dt = 1.0 / cfg.camera.rear_hz
    next_save, rows, wall = 0.0, [], time.perf_counter()
    while True:
        name, xy, agl, heading = path.at(t)
        z = world.ground_z(*xy) + agl if name != "climb" else world.ground_z(*centre) + agl + 0.2
        place(sim, xy, z, heading)
        dist = float(np.hypot(*(xy - centre)))
        if not a.fixed_mount:  # aim the gimbal at the base centre (the drone would use its estimate)
            dep = math.degrees(math.atan2(z - world.ground_z(*centre) - 5.0, max(dist, 1.0)))
            sim.bus.publish(topics.CAMERA_REAR_TILT, CameraTiltCmd(t, dep + PITCH_DEG))
        sim.data.time = t
        anim.update(t)
        rear.next_t = 0.0
        rear.step()
        truth = rear.last_truth
        n_view = sum(r["nav"] and r["in_view"] for r in truth)
        n_occ = sum(r["nav"] and r["in_view"] and r["occluded"] for r in truth)
        n_det = sum(r["nav"] and not r["occluded"] and r["peak_on_dn"] >= DETECT_DN for r in truth)
        if a.headless and t >= next_save:
            next_save += a.every
            rows.append((t, name, dist, n_view, n_occ, n_det))
        chase_r.update_scene(sim.data, camera=chase_camera(xy, z, heading, mode, centre), scene_option=opt)
        rgb = chase_r.render()
        view = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        panel = rear_panel(rear.last.image, truth, 0.75)
        ph, pw = panel.shape[:2]
        view[H - ph - 10:H - 10, W - pw - 10:W - 10] = panel
        cv2.putText(view, f"t {t:6.1f} s  {name:12s}  {dist:5.0f} m from base  {agl:.0f} m AGL  "
                    f"lights in view {n_view} (behind terrain {n_occ}) decodable {n_det}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        if a.headless:
            if rows and rows[-1][0] == t:
                cv2.imwrite(str(out / f"base_view_{int(t):04d}.png"), view)
                cv2.imwrite(str(out / f"rear_{int(t):04d}.png"), rear.last.image)
            t += a.every / 4
            if t > path.T:
                break
            continue
        cv2.imshow("41 base view (night)", view)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord(" "):
            paused = not paused
        if k in (ord("1"), ord("2"), ord("4")):
            speed = float(chr(k))
        if k == ord("l"):
            rear.lights.enabled = not rear.lights.enabled
        if k == ord("c"):
            mode = "overview" if mode == "chase" else "chase"
        if k == ord("r"):
            t = 0.0
        now = time.perf_counter()
        if not paused:
            t += min(now - wall, 0.2) * speed if speed else dt
        wall = now
    if a.headless:
        print(f"{'t s':>6} {'segment':12s} {'dist m':>7} {'in view':>7} {'occluded':>8} {'decodable':>9}")
        for r in rows:
            print(f"{r[0]:6.0f} {r[1]:12s} {r[2]:7.0f} {r[3]:7d} {r[4]:8d} {r[5]:9d}")
        print(f"frames: {out}")
    chase_r.close()


if __name__ == "__main__":
    main()
