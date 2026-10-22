"""V3 demo (headless): one flight A -> B on the onboard estimate, INS + map-referenced fixes, GNSS disabled.

    python demos/v3_fly.py --world s1 --config ins_trn_ibn --seed 0
    python demos/v3_fly.py --world s2 --config ins_trn          # Carpathian foothills, radar altimeter
    python demos/v3_fly.py --world s1 --config ins              # dead reckoning only (expect drift)
Configs: ins | ins_trn | ins_ibn | ins_trn_ibn | mismatch_heavy. Build worlds first: python -m sfhack.world.build_v3 s1 s2
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from sfhack.nav_runner import CONFIGS, NavFlightSystem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="s1")
    ap.add_argument("--config", choices=list(CONFIGS), default="ins_trn_ibn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--straight", action="store_true", help="shortest route, no information-aware planning")
    ap.add_argument("--every", type=float, default=10.0, help="status print period [s sim]")
    ap.add_argument("--headless", action="store_true", help="accepted for run_demo symmetry; always headless")
    args = ap.parse_args()

    wall = time.time()
    fs = NavFlightSystem(args.world, args.config, args.seed, plan_informed=not args.straight)
    ap_ = fs.autopilot
    print(f"{fs.scenario.title}\nconfig {args.config}: sources {sorted(s for s, on in ap_.sources.items() if on)} "
          f"| GNSS DISABLED | route {ap_.route.length:.0f} m | setup {time.time() - wall:.1f} s")
    for t, text in ap_.decisions:
        print(f"  {t:6.1f}s {text}")
    n_dec = len(ap_.decisions)
    last = -1e9
    while fs.t < fs.max_t and not fs.done:
        fs.step()
        for t, text in ap_.decisions[n_dec:]:
            print(f"  {t:6.1f}s {text}")
        n_dec = len(ap_.decisions)
        if fs.t - last >= args.every:
            last = fs.t
            p, e = fs.truth.state().pos, ap_.pose().pos
            print(f"t={fs.t:6.1f}s {ap_.status_line()} err={np.hypot(*(p[:2] - e[:2])):6.1f} m "
                  f"agl={ap_.agl if ap_.agl is not None else float('nan'):5.1f}")
    r = fs.run(max_t=fs.t + 2.0)
    for t, text in r.decisions[n_dec:]:
        print(f"  {t:6.1f}s {text}")
    print(r.summary())
    print(f"wall time {time.time() - wall:.1f} s")
    fs.close()


if __name__ == "__main__":
    main()
