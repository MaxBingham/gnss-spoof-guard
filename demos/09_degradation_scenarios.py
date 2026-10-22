"""Demo 09: one headless flight per degradation lever (sfhack.degradation), scored on truth vs estimate.

    python demos/09_degradation_scenarios.py                   # all scenarios, in parallel
    python demos/09_degradation_scenarios.py gnss_drag wind    # a subset
    python demos/09_degradation_scenarios.py --list

Columns: outcome and final distance to B; true / estimated max off-track distance from the A -> B line
(the drag spoof moves the TRUE drone while the estimate stays on the line); mean / max |estimate - truth|;
mean 2-sigma while the lever is active (a confident estimate has a small 2-sigma); visual / GNSS fixes.
Truth is used only to score (harness code).
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# name: (initial levers, schedule "T:assignments;...", lever-active window [t0, t1] for the 2-sigma column)
SCENARIOS: dict[str, tuple[str, str, tuple[float, float]]] = {
    "nominal": ("", "", (20, 70)),
    "gnss_ok": ("gnss=ok", "", (20, 70)),
    "gnss_jam": ("gnss=ok", "25:gnss=jam", (25, 70)),
    "gnss_drag": ("gnss=ok", "25:gnss=drag:1.5@90", (25, 70)),
    "gnss_jump": ("gnss=ok", "30:gnss=jump:40@90", (30, 70)),
    "rain": ("", "20:weather=rain:0.8;60:weather=clear", (20, 60)),
    "fog": ("", "20:weather=fog:0.8;60:weather=clear", (20, 60)),
    "dust": ("", "20:weather=dust:0.8;60:weather=clear", (20, 60)),
    "wind": ("", "20:wind=7@90,wind.gust=2", (20, 70)),
    "wind_rotating": ("", "20:wind=6@0,wind.rotation_deg_s=6,wind.gust=1", (20, 70)),
    "camera_lost": ("", "20:camera=lost;50:camera=ok", (20, 50)),
    "camera_drop": ("camera=0.6", "", (20, 70)),
    "lidar_lost": ("lidar=lost", "", (60, 200)),
    "lidar_dropout": ("lidar=0.7", "", (60, 200)),
}


def _off_track(xy: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = (b - a) / np.linalg.norm(b - a)
    rel = xy - a
    return np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0])


def fly(job: tuple[str, int, str]) -> dict:
    name, seed, localizer = job
    from sfhack.degradation import Conditions
    from sfhack.degradation.cli import parse_schedule
    from sfhack.runner import FlightSystem
    init, sched, (w0, w1) = SCENARIOS[name]
    wall = time.time()
    fs = FlightSystem(pose_source="ekf", localizer=localizer, seed=seed, conditions=Conditions().update(init),
                      schedule=parse_schedule(sched))
    r = fs.run()
    t = np.array(fs.log_t)
    tr, est, s2 = np.array(fs.log_truth)[:, :2], np.array(fs.log_est)[:, :2], np.array(fs.log_sigma2)
    a, b = fs.onboard_map.a_enu, fs.goal_xy
    win = (t >= w0) & (t <= w1)
    out = {
        "name": name, "r": r, "wall": time.time() - wall,
        "true_off": float(_off_track(tr, a, b).max()), "est_off": float(_off_track(est, a, b).max()),
        "s2_win": float(s2[win].mean()) if win.any() else float("nan"),
        "gnss": fs.autopilot.ekf.n_gnss, "dropped": fs.sim.frames_dropped,
        "decisions": list(fs.autopilot.decisions), "levers": list(fs.conditions_log),
    }
    fs.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenarios", nargs="*", help=f"subset of: {' '.join(SCENARIOS)}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--localizer", choices=["vision", "mock"], default="vision")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="print lever changes and onboard decisions")
    ap.add_argument("--headless", action="store_true", help="accepted for run_demo symmetry; always headless")
    args = ap.parse_args()
    if args.list:
        for k, (init, sched, _) in SCENARIOS.items():
            print(f"{k:14s} init '{init}'  schedule '{sched}'")
        return
    names = args.scenarios or list(SCENARIOS)
    unknown = set(names) - set(SCENARIOS)
    if unknown:
        raise SystemExit(f"unknown scenarios: {sorted(unknown)}")
    with ProcessPoolExecutor(args.workers) as ex:
        runs = list(ex.map(fly, [(n, args.seed, args.localizer) for n in names]))
    print(f"seed {args.seed}, localizer {args.localizer}\n")
    print(f"{'scenario':14s} {'outcome':16s} {'to B':>7s} | {'off-track true':>14s} {'est':>6s} | "
          f"{'|est-truth| mean':>16s} {'max':>6s} | {'2-sigma (lever)':>15s} | {'fixes vis/gnss':>14s} | wall")
    for x in runs:
        r = x["r"]
        print(f"{x['name']:14s} {r.outcome:16s} {r.final_error:6.1f}m | {x['true_off']:13.1f}m {x['est_off']:5.1f}m | "
              f"{r.est_err_mean:15.1f}m {r.est_err_max:5.1f}m | {x['s2_win']:14.1f}m | "
              f"{r.loc_fixes:6d}/{x['gnss']:<7d} | {x['wall']:4.0f}s")
        if args.verbose:
            for t, text in x["levers"]:
                print(f"      {t:6.1f}s LEVERS {text}")
            for t, text in x["decisions"]:
                print(f"      {t:6.1f}s {text}")


if __name__ == "__main__":
    main()
