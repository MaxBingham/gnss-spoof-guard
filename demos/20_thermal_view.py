"""Thermal camera view: fly a mission route at night and show the simulated LWIR camera next to the map.

    uv run python -m sfhack.thermal build --mission berkeley        # once: world/thermal_truth.npz + prior
    uv run python demos/20_thermal_view.py                          # window (Esc quits, Space pauses)
    uv run python demos/20_thermal_view.py --record                 # mp4 -> $SFHACK_DATA/night/thermal/
    uv run python demos/20_thermal_view.py --snap 30,60,90 --headless   # PNG frames, no window

Panels: live thermal camera (640 x 512, white-hot, AGC) | daytime ortho around the drone with the camera
footprint | the onboard thermal PRIOR (what the drone predicts it should see) with the same footprint.
Harness / display code: it flies on the truth pose by default (--pose ekf to fly on the estimate) and reads
world/ for display; the drone's own code never sees any of this.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sfhack.config import data_dir  # noqa: E402
from sfhack.missionprep.geometry import Grid  # noqa: E402
from sfhack.runner import FlightSystem  # noqa: E402
from sfhack.thermal.camera import ThermalCamera, load_truth  # noqa: E402
from sfhack.thermal.prior import load_prior  # noqa: E402

PANEL = 512
VIEW_M = 140.0  # map panels: metres across


def footprint(cam: ThermalCamera, data, ground_z: float) -> np.ndarray:
    """ENU (4, 2) where the image corners hit the plane z = ground_z."""
    k = cam.intrinsics
    pos = data.cam_xpos[cam.cam_id]
    R = data.cam_xmat[cam.cam_id].reshape(3, 3)
    out = []
    for u, v in ((0, 0), (k["w"], 0), (k["w"], k["h"]), (0, k["h"])):
        ray = R @ np.array([(u - k["cx"]) / k["fx"], -(v - k["cy"]) / k["fy"], -1.0])
        t = (ground_z - pos[2]) / ray[2] if ray[2] < -1e-6 else 0.0
        out.append(pos[:2] + t * ray[:2])
    return np.array(out)


def map_panel(img: np.ndarray, grid: Grid, center: np.ndarray, fp: np.ndarray, track: list[np.ndarray]) -> np.ndarray:
    """North-up crop VIEW_M wide around `center` with the footprint polygon and the flown track."""
    s = PANEL / VIEW_M  # px per m
    # panel px = (x - cx) * s + PANEL/2, (cy - y) * s + PANEL/2 ; raster px = (x - min_x)/res - 0.5 ...
    a = s * grid.res
    M = np.array([[a, 0, PANEL / 2 - s * (center[0] - grid.min_x) + a * 0.5],
                  [0, a, PANEL / 2 - s * (grid.max_y - center[1]) + a * 0.5]], np.float32)
    out = cv2.warpAffine(img, M, (PANEL, PANEL), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

    def px(p: np.ndarray) -> np.ndarray:
        return np.round(np.column_stack([(p[:, 0] - center[0]) * s + PANEL / 2,
                                         (center[1] - p[:, 1]) * s + PANEL / 2])).astype(np.int32)

    if len(track) > 1:
        cv2.polylines(out, [px(np.array(track))], False, (60, 200, 255), 2, cv2.LINE_AA)
    cv2.polylines(out, [px(fp)], True, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(out, (PANEL // 2, PANEL // 2), 4, (0, 0, 255), -1, cv2.LINE_AA)
    return out


def colorize_prior(temp: np.ndarray, lo: float, hi: float) -> np.ndarray:
    g = np.clip((temp - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def label(img: np.ndarray, text: str, y: int = 24) -> None:
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    sub = img[y - h - 6:y + base + 2, 4:w + 16]
    sub[:] = (sub * 0.35).astype(np.uint8)  # dark box behind the text
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mission", default="berkeley")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pose", choices=["truth", "ekf"], default="truth")
    ap.add_argument("--localizer", choices=["off", "mock", "vision"], default="off")
    ap.add_argument("--hz", type=float, default=5.0, help="thermal frames per sim second")
    ap.add_argument("--until", type=float, default=0.0, help="stop at this sim time (0 = mission end)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--record", nargs="?", const="", default=None, help="mp4 path (default under $SFHACK_DATA/night/thermal)")
    ap.add_argument("--snap", default="", help="sim times to save PNGs, e.g. 30,60")
    ap.add_argument("--snap-dir", default=str(data_dir() / "night" / "thermal"))
    args = ap.parse_args()

    fs = FlightSystem(mission=args.mission, pose_source=args.pose, localizer=args.localizer, seed=args.seed)
    mdir = fs.mission_dir
    truth = load_truth(mdir)
    prior = load_prior(mdir / "onboard")
    fs.sim.render("down")  # creates the simulator's renderer; the thermal camera draws into it (one GL context)
    cam = ThermalCamera(fs.sim.model, seed=args.seed, renderer=fs.sim.renderer)
    ortho = cv2.imread(str(mdir / "world" / "ortho.jpg"), cv2.IMREAD_COLOR)
    grid = truth.grid
    lo, hi = np.percentile(prior.temp_K[::4, ::4], [1, 99])
    prior_img = colorize_prior(prior.temp_K, lo, hi)
    print(f"{args.mission}: thermal at hour {truth.meta.get('hour')} (air {truth.air_K - 273.15:.1f} C), "
          f"route {len(fs.mission.plan) if hasattr(fs.mission, 'plan') else '?'} legs, pose {args.pose}")

    out_dir = Path(args.snap_dir)
    video = None
    if args.record is not None:
        path = Path(args.record) if args.record else data_dir() / "night" / "thermal" / f"thermal_view_{args.mission}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        video = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), args.hz, (640 + 2 * PANEL, PANEL))
        print(f"recording {path}")
    snaps = sorted(float(x) for x in args.snap.split(",") if x.strip())
    show = not args.headless and video is None
    track: list[np.ndarray] = []
    next_frame, paused = 0.0, False
    render_ms: list[float] = []
    t_end = args.until or math.inf
    while not fs.done and fs.t < t_end:
        if paused:
            k = cv2.waitKey(30) & 0xFF
            paused = k != 32
            if k == 27:
                break
            continue
        fs.step()
        if fs.t < next_frame:
            continue
        next_frame = fs.t + 1.0 / args.hz
        st = fs.truth.state()  # harness: display
        track.append(st.pos[:2].copy())
        t0 = time.perf_counter()
        img = cam.render(fs.sim.data, truth)
        render_ms.append((time.perf_counter() - t0) * 1000)
        th = cv2.cvtColor(cam.to_8bit(img), cv2.COLOR_GRAY2BGR)
        th = cv2.resize(th, (640, PANEL)) if th.shape[0] != PANEL else th
        gz = fs.true_ground_z(st.pos[0], st.pos[1])
        fp = footprint(cam, fs.sim.data, gz)
        left = th
        label(left, f"LWIR 640x512 white-hot  t={fs.t:5.1f}s  AGL {st.pos[2] - gz:4.1f} m")
        label(left, f"scene {np.percentile(img, 1) - 273.15:4.1f}..{np.percentile(img, 99) - 273.15:4.1f} C  "
                    f"render {render_ms[-1]:3.0f} ms", 48)
        mid = map_panel(ortho, grid, st.pos[:2], fp, track)
        label(mid, f"daytime ortho  {VIEW_M:.0f} m  ({fs.mission.phase})")
        right = map_panel(prior_img, grid, st.pos[:2], fp, track)
        label(right, "onboard thermal prior (predicted)")
        frame = np.hstack([left, mid, right])
        if video is not None:
            video.write(frame)
        if snaps and fs.t >= snaps[0]:
            snaps.pop(0)
            out_dir.mkdir(parents=True, exist_ok=True)
            p = out_dir / f"thermal_{args.mission}_t{fs.t:05.1f}.png"
            cv2.imwrite(str(p), frame)
            print(f"snapshot {p}")
            if not snaps and args.headless and video is None:
                break
        if show:
            cv2.imshow("thermal view (Esc quit, Space pause)", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            paused = k == 32
    if video is not None:
        video.release()
    if render_ms:
        print(f"{len(render_ms)} thermal frames, render {np.median(render_ms):.0f} ms median "
              f"({1000 / np.median(render_ms):.1f} Hz possible)")
    cam.close()
    fs.close()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
