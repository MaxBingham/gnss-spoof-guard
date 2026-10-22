"""Demo 01: fly_x2.py's waypoint mission with the TruthPoseSource (controller tuning mode).

The controller sees the TRUE state here on purpose (this is the 'truth' switch position).
The whole stack still runs: sensors, AHRS and EKF run in the background, only the controller
input is truth.

    python demos/01_hover_truth.py              # MuJoCo viewer
    python demos/01_hover_truth.py --headless   # prints the flight log
    python demos/01_hover_truth.py --headless --world berkeley   # A -> B on the Berkeley map
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from sfhack.config import SAMPLE_WORLD_DIR
from sfhack.runner import FlightSystem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", help="run without a window")
    ap.add_argument("--world", choices=["arena", "berkeley"], default="arena")
    args = ap.parse_args()

    fs = FlightSystem(world_dir=None if args.world == "arena" else SAMPLE_WORLD_DIR,
                      pose_source="truth", localizer="off")
    print(f"Drone mass: {fs.autopilot.controller.mass:.2f} kg | waypoints: {len(fs.mission.waypoints)} | "
          f"pose source: TRUTH (tuning mode)")

    if args.headless:
        last_print = -1.0
        while fs.t < fs.cfg.sim.max_time_s and not fs.done:
            fs.step()
            if fs.t - last_print >= 1.0:
                last_print = fs.t
                p = fs.truth.state().pos
                print(f"t={fs.t:5.1f}s  pos=({p[0]:7.2f}, {p[1]:7.2f}, {p[2]:6.2f})  "
                      f"phase {fs.mission.phase:8s} waypoint {fs.mission.idx + 1}/{len(fs.mission.waypoints)}")
        r = fs.run(max_t=fs.t + 2.0)  # settle on the ground
        print(f"Mission {'COMPLETE' if fs.done else 'NOT finished'} at t={r.t_end:.1f}s, "
              f"final error to B: {r.final_error:.2f} m")
        fs.close()
        return

    m, d = fs.sim.model, fs.sim.data
    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = m.body("x2").id
        viewer.cam.distance = 6.0 if args.world == "arena" else 25.0
        viewer.cam.elevation = -25
        steps = int(1 / 60 / m.opt.timestep)
        while viewer.is_running():
            frame_start = time.time()
            for _ in range(steps):
                fs.step()
            viewer.sync()
            time.sleep(max(0.0, 1 / 60 - (time.time() - frame_start)))
            if fs.done and fs.t > 1.0 and not hasattr(main, "_reported"):
                main._reported = True  # type: ignore[attr-defined]
                print(f"Landed. final error to B: {np.linalg.norm(fs.truth.state().pos[:2] - fs.goal_xy):.2f} m")
    fs.close()


if __name__ == "__main__":
    main()
