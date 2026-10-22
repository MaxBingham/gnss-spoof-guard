"""Demo 02: the full loop on a map package. A -> B with the EKF as the controller's pose source
(no truth in the loop) and the selected localizer feeding visual fixes.

    python demos/02_mission_ekf.py                      # Berkeley sample, mock localizer
    python demos/02_mission_ekf.py --localizer off      # pure dead reckoning (expect drift)
    python demos/02_mission_ekf.py --seed 3 --world path/to/package
"""

from __future__ import annotations

import argparse
import time

from sfhack.config import SAMPLE_WORLD_DIR
from sfhack.runner import FlightSystem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default=str(SAMPLE_WORLD_DIR), help="map package directory")
    ap.add_argument("--localizer", choices=["mock", "off", "vision"], default="mock")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--headless", action="store_true", help="accepted for run_demo symmetry; always headless")
    args = ap.parse_args()

    fs = FlightSystem(world_dir=args.world, pose_source="ekf", localizer=args.localizer, seed=args.seed)
    print(f"A -> B {fs.goal_xy} | "
          f"pose source: EKF | localizer: {fs.localizer_name}")
    wall = time.time()
    last = -1.0
    while fs.t < fs.cfg.sim.max_time_s and not fs.done:
        fs.step()
        if fs.t - last >= 5.0:
            last = fs.t
            p, e = fs.truth.state().pos, fs.autopilot.pose().pos
            print(f"t={fs.t:5.1f}s phase {fs.mission.phase:8s} truth=({p[0]:7.1f},{p[1]:7.1f},{p[2]:5.1f}) "
                  f"est err={((p[:2] - e[:2]) ** 2).sum() ** 0.5:5.2f} m  fixes {fs.autopilot.loc_fixes}")
    r = fs.run(max_t=fs.t + 2.0)
    print(r.summary())
    print(f"wall time {time.time() - wall:.1f} s")
    fs.close()


if __name__ == "__main__":
    main()
