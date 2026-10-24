"""Is there enough thermal signal to localize? Offline check of the thermal model, not a localizer.

    uv run python demos/22_thermal_matchability.py [--missions berkeley m-c804bb84 ukraine-1] [--agl 60]

Along each mission route (every --step m, drone level at --agl above the true ground, yaw along the route),
render the simulated camera (truth, noise, traffic, absolute calibration error) and correlate it with the
onboard prior's predicted view (ThermalPrior.predict_view) over a +-15 m grid of position offsets (yaw known).
Score per frame: zero-mean NCC of the raw images and of gradient magnitudes; the peak's distance from the
true position; peak-to-sidelobe margin. A frame "localizes" when the NCC peak is within 2 m of the truth.
Harness code: places the drone with the truth, reads world/ for the camera, the prior for the prediction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sfhack.config import data_dir  # noqa: E402
from sfhack.runner import FlightSystem  # noqa: E402
from sfhack.thermal.camera import ThermalCamera, load_truth  # noqa: E402
from sfhack.thermal.prior import load_prior  # noqa: E402

SHIFT_M = 15
STEP_PX = 4  # image decimation for matching (640x512 -> 160x128)


def zncc(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100:
        return float("nan")
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = math.sqrt(float((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / d) if d > 0 else float("nan")


def grad(a: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(a, nan=float(np.nanmean(a)))
    gx, gy = cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1)
    return np.hypot(gx, gy)


def route_points(mission: dict, step: float) -> list[tuple[np.ndarray, float]]:
    pts = np.array([p["enu"][:2] for p in mission["route"] if p["phase"] in ("TAKEOFF", "CRUISE", "DESCEND")], float)
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        L = float(np.hypot(*(b - a)))
        if L < 1:
            continue
        yaw = math.atan2(b[1] - a[1], b[0] - a[0])
        for s in np.arange(0, L, step):
            out.append((a + (b - a) * s / L, yaw))
    return out


def run(mid: str, agl: float, step: float, contrast: float, seed: int) -> dict:
    fs = FlightSystem(mission=mid, pose_source="truth", localizer="off", seed=seed)
    m, d = fs.sim.model, fs.sim.data
    truth = load_truth(fs.mission_dir, contrast=contrast)
    prior = load_prior(fs.onboard_dir)
    fs.sim.render("down")
    cam = ThermalCamera(m, seed=seed, renderer=fs.sim.renderer)
    k = cam.intrinsics
    mission = json.loads((fs.onboard_dir / "mission.json").read_text(encoding="utf-8"))
    shifts = np.arange(-SHIFT_M, SHIFT_M + 1, 1.0)
    rows = []
    for i, (xy, yaw) in enumerate(route_points(mission, step)):
        gz = fs.true_ground_z(*xy)
        d.qpos[0:3] = [xy[0], xy[1], gz + agl]
        d.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        d.time = i * 2.0
        mujoco.mj_forward(m, d)
        img = cam.render(d, truth)[::STEP_PX, ::STEP_PX]
        cam_agl = float(d.cam_xpos[cam.cam_id][2] - gz)
        g_img = grad(img)
        best = {"raw": (-2.0, 0.0, 0.0), "grad": (-2.0, 0.0, 0.0)}
        surf = {"raw": np.full((len(shifts), len(shifts)), np.nan), "grad": np.full((len(shifts), len(shifts)), np.nan)}
        for a, dx in enumerate(shifts):
            for b, dy in enumerate(shifts):
                pred, _ = prior.predict_view(xy + [dx, dy], cam_agl, k, yaw=yaw, step=STEP_PX)
                for name, (u, v) in (("raw", (img, pred)), ("grad", (g_img, grad(pred)))):
                    c = zncc(u, v)
                    surf[name][a, b] = c
                    if c > best[name][0]:
                        best[name] = (c, dx, dy)
        row = {"i": i, "xy": [round(float(xy[0]), 1), round(float(xy[1]), 1)],
               "road_frac": float(np.mean(_roads_in_view(prior, xy, cam_agl, k, yaw)))}
        for name in ("raw", "grad"):
            c, dx, dy = best[name]
            s = surf[name]
            far = np.hypot(*np.meshgrid(shifts, shifts, indexing="ij")) > 4.0
            side = float(np.nanmax(np.where(far, s, np.nan)))
            row[name] = {"ncc": round(c, 3), "err_m": round(math.hypot(dx, dy), 1), "margin": round(c - side, 3),
                         "ncc_at_truth": round(float(s[SHIFT_M, SHIFT_M]), 3)}
        rows.append(row)
        print(f"{mid} {i:3d} ({xy[0]:7.1f},{xy[1]:7.1f}) roads {row['road_frac']:.2f}  "
              f"raw ncc {row['raw']['ncc']:.2f} err {row['raw']['err_m']:4.1f} m  "
              f"grad ncc {row['grad']['ncc']:.2f} err {row['grad']['err_m']:4.1f} m margin {row['grad']['margin']:.2f}",
              flush=True)
    fs.close()
    out = {"mission": mid, "agl": agl, "contrast": contrast, "frames": len(rows)}
    for name in ("raw", "grad"):
        errs = np.array([r[name]["err_m"] for r in rows])
        out[name] = {"within_2m": round(float(np.mean(errs <= 2.0)), 3), "within_5m": round(float(np.mean(errs <= 5.0)), 3),
                     "median_err_m": round(float(np.median(errs)), 1),
                     "median_ncc_at_truth": round(float(np.median([r[name]["ncc_at_truth"] for r in rows])), 3)}
    out["rows"] = rows
    return out


def _roads_in_view(prior, xy, agl, k, yaw) -> np.ndarray:
    uv = prior.to_px(prior.ground_points(xy, agl, k, yaw=yaw, step=8)).round().astype(int)
    r = np.clip(uv[..., 1], 0, prior.roads.shape[0] - 1)
    c = np.clip(uv[..., 0], 0, prior.roads.shape[1] - 1)
    return prior.roads[r, c] > 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--missions", nargs="+", default=["berkeley", "m-c804bb84", "ukraine-1"])
    ap.add_argument("--agl", type=float, default=60.0)
    ap.add_argument("--step", type=float, default=50.0, help="metres between test frames")
    ap.add_argument("--contrast", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(data_dir() / "night" / "thermal"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for mid in a.missions:
        r = run(mid, a.agl, a.step, a.contrast, a.seed)
        tag = f"{mid}_agl{a.agl:.0f}_c{a.contrast:.1f}"
        (out / f"matchability_{tag}.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        summary.append({k: v for k, v in r.items() if k != "rows"})
        print(json.dumps(summary[-1]), flush=True)
    print("| mission | AGL | contrast | frames | raw: within 2 m / 5 m, median err | grad: within 2 m / 5 m, median err | NCC at truth raw / grad |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for s in summary:
        print(f"| {s['mission']} | {s['agl']:.0f} m | {s['contrast']} | {s['frames']} | "
              f"{s['raw']['within_2m'] * 100:.0f} / {s['raw']['within_5m'] * 100:.0f} %, {s['raw']['median_err_m']} m | "
              f"{s['grad']['within_2m'] * 100:.0f} / {s['grad']['within_5m'] * 100:.0f} %, {s['grad']['median_err_m']} m | "
              f"{s['raw']['median_ncc_at_truth']} / {s['grad']['median_ncc_at_truth']} |")


if __name__ == "__main__":
    main()
