"""Demo 07: run the default mission (EKF + mock localizer, no truth in the loop) over several seeds
in parallel and print success rate and error statistics.

    python demos/07_eval_seeds.py --seeds 10 [--localizer mock] [--workers 5]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def fly(args: tuple[int, str]) -> dict:
    seed, localizer = args
    from sfhack.runner import FlightSystem
    fs = FlightSystem(pose_source="ekf", localizer=localizer, seed=seed)
    r = fs.run()
    decisions = list(fs.autopilot.decisions)
    fs.close()
    return {"seed": seed, "r": r, "decisions": decisions}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--localizer", default="mock")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--headless", action="store_true", help="accepted for run_demo symmetry; always headless")
    args = ap.parse_args()
    with ProcessPoolExecutor(args.workers) as ex:
        runs = list(ex.map(fly, [(s, args.localizer) for s in range(args.seeds)]))
    for x in runs:
        r = x["r"]
        rej = 100 * r.fixes_rejected / max(r.loc_fixes, 1)
        print(f"seed {x['seed']:2d}: {r.outcome:15s} final {r.final_error:6.2f} m  est mean {r.est_err_mean:5.2f} / max "
              f"{r.est_err_max:6.2f} m  fixes {r.loc_fixes}/{r.loc_frames} rejected {rej:4.1f} %  t {r.t_end:5.1f} s")
        for t, text in x["decisions"]:
            print(f"           {t:6.1f}s {text}")
    rs = [x["r"] for x in runs]
    fe = np.array([r.final_error for r in rs])
    rej = sum(r.fixes_rejected for r in rs) / max(sum(r.loc_fixes for r in rs), 1)
    print(f"\nSUCCESS {sum(r.success for r in rs)}/{len(rs)} (landed within 3 m of B) | final error mean "
          f"{fe.mean():.2f} m, max {fe.max():.2f} m | est error mean {np.mean([r.est_err_mean for r in rs]):.2f} m | "
          f"rejected fixes {100 * rej:.1f} % | flight time mean {np.mean([r.t_end for r in rs]):.1f} s | "
          f"localizer {np.mean([r.loc_ms_mean for r in rs]):.2f} ms/frame, "
          f"hit rate {100 * sum(r.loc_fixes for r in rs) / max(sum(r.loc_frames for r in rs), 1):.0f} %")


if __name__ == "__main__":
    main()
