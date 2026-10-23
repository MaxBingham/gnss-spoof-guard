"""V4 detector benchmark (harness): sweep the camera over the search area at a fixed altitude, run the onboard
target perception on each frame + lidar scan, and score the candidates against the truth layout.

The pose fed to the detector is the TRUE pose (this benchmarks perception, not estimation), like
demos/08_bench_localizer.py.

    uv run python demos/11_bench_target_detector.py --seeds 10 --difficulty easy --alt 30
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sfhack.bus import Bus  # noqa: E402
from sfhack.config import data_dir, default_config  # noqa: E402
from sfhack.messages import PoseEstimate  # noqa: E402
from sfhack.perception.target_detector import NAMES, TargetPerception  # noqa: E402
from sfhack.sim.mujoco_sim import MujocoSim  # noqa: E402
from sfhack.sim.target_world import TargetWorld  # noqa: E402
from sfhack.world.onboard_map import OnboardMap  # noqa: E402
from sfhack.world.scene import PackageWorld  # noqa: E402
from sfhack.world.target_scene import ensure_field_package, generate_layout  # noqa: E402


def truth_pose(sim: MujocoSim, t: float) -> PoseEstimate:
    d = sim.data
    R = d.xmat[sim.body_id].reshape(3, 3).copy()
    yaw = math.atan2(R[1, 0], R[0, 0])
    return PoseEstimate(t, d.qpos[0:3].copy(), np.zeros(3), yaw, np.eye(6) * 0.01, R, np.zeros(3), "truth")


def sweep(alt: float, half: float, lane: float = 12.0, step: float = 1.0) -> list[tuple[float, float, float]]:
    """Lawnmower lanes (x along, y across), yaw along the lane: (x, y, yaw)."""
    out = []
    for k, y in enumerate(np.arange(-lane, lane + 0.1, lane)):
        xs = np.arange(-half + 4, half - 4 + 0.1, step)
        if k % 2:
            xs = xs[::-1]
        yaw = 0.0 if k % 2 == 0 else math.pi
        out += [(float(x), float(y), yaw) for x in xs]
    return out


def run_seed(seed: int, difficulty: str, alt: float, verbose: bool) -> dict:
    cfg = default_config()
    cfg.camera.down_blur_px = 1.2
    tc = cfg.target
    lay = generate_layout(seed, difficulty, keep_out=[(0, 0, 5)], half=tc.area_half, marker_dict=tc.marker_dict,
                          marker_id=tc.marker_id, marker_size=tc.marker_size, pad_size=tc.pad_size)
    pkg = ensure_field_package(data_dir() / "v4")
    sim = MujocoSim(cfg, Bus(), TargetWorld(PackageWorld(pkg), lay))
    om = OnboardMap(pkg)
    log_lines: list[str] = []
    perc = TargetPerception(tc, om, np.zeros(2), log=lambda t, s: log_lines.append(f"{t:6.1f} {s}"))
    m, d = sim.model, sim.data
    t = 0.0
    t0 = time.perf_counter()
    for x, y, yaw in sweep(alt, tc.area_half):
        d.qpos[0:3] = [x, y, alt]
        d.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        mujoco.mj_forward(m, d)
        t += 0.25
        pose = truth_pose(sim, t)
        scan = sim._lidar_scan(t)
        perc.on_lidar(scan.angles, scan.ranges, pose)
        perc.on_range(float(d.sensor("rangefinder").data[0]), pose)
        frame = sim.render("down")
        frame.t = t
        perc.on_frame(frame, pose)
        perc.update_evidence(t)
    ms = (time.perf_counter() - t0) * 1000 / max(1, len(sweep(alt, tc.area_half)))
    sim.close()
    # score
    cands = perc.candidates()
    rows, matched = [], set()
    for tr in sorted(cands, key=lambda tr: -tr.probs.get("brown_truck", 0.0)):
        o, dist = lay.nearest(tr.xy)
        hit = o.kind if dist < 0.5 * o.length + 1.5 else "-"
        if hit != "-":
            matched.add(hit)
        L, W = tr.size(tr.height or 0.0)
        best = max(tr.probs, key=tr.probs.get) if tr.probs else "?"
        h = f"{tr.height:.1f}" if tr.height is not None else "?"
        rows.append(f"    cand {tr.id:2d} truth={hit:12s} {L:4.1f} x {W:4.1f} x {h:>4s} m {tr.color:10s} "
                    f"class={NAMES.get(best, best)} p={tr.probs.get(best, 0):.2f} p_truck={tr.probs.get('brown_truck', 0):.2f}")
    ranked = sorted(cands, key=lambda tr: -tr.probs.get("brown_truck", 0.0))
    top = lay.nearest(ranked[0].xy) if ranked else (None, 99)
    truck_first = bool(ranked) and top[0].kind == "brown_truck" and top[1] < 4.0
    res = {"seed": seed, "found": len(matched), "n_cand": len(cands), "truck_first": truck_first, "ms": ms}
    print(f"seed {seed:2d} {difficulty}: {len(matched)}/6 objects found ({sorted(matched)}), {len(cands)} candidates, "
          f"truck ranked first: {truck_first}, {ms:.0f} ms/frame")
    for r in rows:
        print(r)
    if verbose:
        print("\n".join("    | " + s for s in log_lines))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--alt", type=float, default=30.0)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    res = [run_seed(s, a.difficulty, a.alt, a.verbose) for s in range(a.seed0, a.seed0 + a.seeds)]
    n = len(res)
    print(f"\n{a.difficulty} @ {a.alt:.0f} m: all 6 found {sum(r['found'] == 6 for r in res)}/{n}, "
          f"mean found {np.mean([r['found'] for r in res]):.1f}, truck ranked first {sum(r['truck_first'] for r in res)}/{n}")


if __name__ == "__main__":
    main()
