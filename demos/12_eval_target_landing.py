"""V4 evaluation (harness): headless target flights over seeds, scored on the truth layout.

    uv run python demos/12_eval_target_landing.py --seeds 20 --difficulty easy hard
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def one(args: tuple[int, str, str, float]) -> dict:
    seed, difficulty, nav, max_t = args
    from sfhack.sim.target_world import build_target_flight
    fs, ext = build_target_flight(seed, difficulty, nav)
    fs.run(max_t=max_t)
    s = ext.score(fs)
    fs.close()
    target_lines = [txt for _, txt in fs.autopilot.decisions if txt.startswith(("TARGET", "INSPECT", "MARKER_CONFIRMED"))]
    return {"seed": seed, "difficulty": difficulty, "score": s, "t": fs.t, "log": target_lines}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--difficulty", nargs="+", default=["easy", "hard"])
    ap.add_argument("--nav", default="gnss", choices=["gnss", "vision", "truth"])
    ap.add_argument("--max-t", type=float, default=240.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    jobs = [(s, d, a.nav, a.max_t) for d in a.difficulty for s in range(a.seed0, a.seed0 + a.seeds)]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, jobs))
    for d in a.difficulty:
        rs = [r for r in res if r["difficulty"] == d]
        for r in rs:
            print(f"{d:4s} seed {r['seed']:2d}: {r['score'].summary()} t={r['t']:.0f} s")
            if a.verbose:
                print("\n".join("      " + x for x in r["log"]))
        n = len(rs)
        correct = sum(r["score"].chosen_kind == "brown_truck" for r in rs)
        pad = sum(r["score"].on_pad for r in rs)
        wrong = sum(r["score"].landed and not r["score"].on_pad for r in rs)
        wrong_obj = sum(r["score"].landed and r["score"].landed_on not in ("brown_truck",) for r in rs)
        print(f"== {d} ({a.nav}): correct target {correct}/{n} ({100 * correct / n:.0f} %), landed on pad {pad}/{n} "
              f"({100 * pad / n:.0f} %), landed off the pad {wrong}/{n} (of which on a wrong object / ground: {wrong_obj})")


if __name__ == "__main__":
    main()
