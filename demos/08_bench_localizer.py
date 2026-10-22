"""Demo 08: offline benchmark of the VISION localizer on a recorded Berkeley flight.

    python demos/08_bench_localizer.py                 # record seed 0 once (cached), then evaluate
    python demos/08_bench_localizer.py --seed 3 --rerecord

Recording: the default mission flown on the EKF with the MOCK localizer; every down-camera frame is
stored with the onboard prior (EKF position/covariance, AHRS attitude) and the TRUE camera position
(harness: scoring only) in $SFHACK_DATA/bench/. Evaluation replays each frame through MapLocalizer
twice: with the recorded EKF prior, and with a prior offset by --offset m (sigma --offset-sigma) to
test the search. Reports hit rate, error (fix vs true camera xy) and ms/frame by altitude band.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from sfhack import bus as topics
from sfhack.config import data_dir, default_config
from sfhack.messages import CameraFrame, PoseEstimate
from sfhack.perception.map_localizer import MapLocalizer
from sfhack.runner import FlightSystem

BANDS = [(30.0, 99.0), (10.0, 30.0), (4.0, 10.0)]


def record(seed: int, path) -> None:
    fs = FlightSystem(pose_source="ekf", localizer="mock", seed=seed)
    rec: dict[str, list] = {k: [] for k in ("t", "img", "pos", "cov", "R", "cam_xyz", "fovy")}

    def on_frame(fr: CameraFrame) -> None:
        est = fs.autopilot.pose()
        tp = fs.truth.camera_pose(fr.seq)  # harness: scoring only
        if tp is None or fs.autopilot.in_preflight:
            return
        rec["t"].append(fr.t)
        rec["img"].append(fr.image)
        rec["pos"].append(est.pos)
        rec["cov"].append(est.cov[:2, :2])
        rec["R"].append(est.R)
        rec["cam_xyz"].append(tp[0])
        rec["fovy"].append(fr.fovy_deg)

    fs.bus.subscribe(topics.CAMERA_DOWN, on_frame)
    r = fs.run()
    fs.close()
    np.savez_compressed(path, **{k: np.array(v) for k, v in rec.items()})
    print(f"recorded {len(rec['t'])} frames, flight: {r.summary()}")


def evaluate(path, offset: float, offset_sigma: float, seed: int) -> dict:
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}  # npz members decompress on every access: load once
    cfg = default_config()
    fs = FlightSystem(pose_source="ekf", localizer="off", seed=0)  # only for the onboard map
    om = fs.onboard_map
    fs.close()
    rng = np.random.default_rng(seed)
    out = {}
    for mode in ("ekf prior", f"prior +{offset:.0f} m"):
        loc = MapLocalizer(cfg.localizer)
        rows = []
        for k in range(len(d["t"])):
            fr = CameraFrame(float(d["t"][k]), k, "down", d["img"][k], float(d["fovy"][k]))
            pos = d["pos"][k].copy()
            cov = np.zeros((6, 6))
            cov[:2, :2] = d["cov"][k]
            if mode != "ekf prior":
                ang = rng.uniform(0, 2 * np.pi)
                pos[:2] = d["cam_xyz"][k][:2] + offset * np.array([np.cos(ang), np.sin(ang)])
                cov[:2, :2] = np.eye(2) * offset_sigma**2
            prior = PoseEstimate(fr.t, pos, np.zeros(3), 0.0, cov, d["R"][k], np.zeros(3))
            t0 = time.perf_counter()
            fix = loc.update(fr, om, prior)
            ms = (time.perf_counter() - t0) * 1000
            alt = float(d["cam_xyz"][k][2])
            err = float(np.linalg.norm(fix.pos - d["cam_xyz"][k][:2])) if fix is not None else np.nan
            sig = float(np.sqrt(fix.cov[0, 0])) if fix is not None else np.nan
            rows.append((alt, err, sig, ms, loc.last_reason))
        out[mode] = rows
        print(f"\n== {mode}: {len(rows)} frames")
        for lo, hi in BANDS:
            sel = [r for r in rows if lo <= r[0] < hi]
            if not sel:
                continue
            e = np.array([r[1] for r in sel])
            hit = ~np.isnan(e)
            ok = e[hit]
            reasons: dict[str, int] = {}
            for r in sel:
                if np.isnan(r[1]):
                    reasons[r[4]] = reasons.get(r[4], 0) + 1
            line = f"  alt {lo:4.0f}-{hi:3.0f} m: {len(sel):4d} frames, hit {100 * hit.mean():5.1f} %"
            if len(ok):
                within = 100 * np.mean(ok < 5.0)
                z = np.array([r[1] / r[2] for r in sel if not np.isnan(r[1])])
                line += (f" | err mean {ok.mean():5.2f} median {np.median(ok):5.2f} p95 {np.percentile(ok, 95):6.2f}"
                         f" max {ok.max():6.2f} m, <5 m {within:5.1f} % | err/sigma median {np.median(z):4.2f}")
            print(line)
            print(f"      no fix: {reasons}")
        ms = np.array([r[3] for r in rows])
        print(f"  {ms.mean():.1f} ms/frame mean, {np.percentile(ms, 95):.1f} ms p95")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rerecord", action="store_true")
    ap.add_argument("--offset", type=float, default=6.0)
    ap.add_argument("--offset-sigma", type=float, default=5.0)
    ap.add_argument("--headless", action="store_true", help="accepted for run_demo symmetry; always headless")
    args = ap.parse_args()
    out = data_dir() / "bench"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"flight_seed{args.seed}.npz"
    if args.rerecord or not path.exists():
        record(args.seed, path)
    evaluate(path, args.offset, args.offset_sigma, args.seed)


if __name__ == "__main__":
    main()
