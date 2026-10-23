"""Demo 11: the V3.2 evaluation table. Every scenario over N seeds (live-demo mission: vision localizer, GNSS ok
at start, GNSS jammed at 40 s), TRN on and off, printed as a Markdown table. Harness code: truth only scores.

    python demos/11_eval_table.py --seeds 10 [--workers 10] [--out table.md]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
_spec = importlib.util.spec_from_file_location("eval_levers", HERE / "10_eval_levers.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def fly(job: dict) -> dict:  # module-level wrapper: worker processes must be able to import it
    return ev.fly(job)


JAM = "40:gnss=jam"
SCENARIOS = [  # name, initial levers, schedule
    ("no jam (GNSS ok)", "gnss=ok", ""),
    ("jam at 40 s", "gnss=ok", JAM),
    ("jam + fog 0.65", "gnss=ok,weather=fog:0.65", JAM),
    ("jam + camera lost", "gnss=ok", "40:gnss=jam,camera=lost"),
    ("jam + wind 6 m/s", "gnss=ok,wind=6@90", JAM),
    ("G-key cycle: drag, jump, jam", "gnss=ok", "40:gnss=drag;42:gnss=jump;44:gnss=jam"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    jobs, keys = [], []
    for name, cond, sched in SCENARIOS:
        for trn in ("on", "off"):
            for s in range(args.seeds):
                jobs.append({"seed": s, "localizer": "vision", "cond": cond, "sched": sched,
                             "cfg": {"trn.enabled": trn == "on"}})
                keys.append((name, trn))
    wall = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        runs = list(ex.map(fly, jobs))
    lines = ["| Scenario | TRN | within 3 m | within 10 m | final error median / mean / max (m) | max 2-sigma (m) "
             "| max true est. error (m) | sources fused |", "|---|---|---|---|---|---|---|---|"]
    for name, _, _ in SCENARIOS:
        for trn in ("on", "off"):
            rs = [r for r, k in zip(runs, keys) if k == (name, trn)]
            fe = np.array([x["r"].final_error for x in rs])
            s = ev.summarize(rs)
            lines.append(f"| {name} | {trn} | {s['ok']}/{s['n']} | {int((fe <= 10).sum())}/{s['n']} | "
                         f"{np.median(fe):.1f} / {fe.mean():.1f} / {fe.max():.1f} | {s['sig2_max']:.1f} | "
                         f"{s['err_max']:.1f} | {', '.join(s['sources'])} |")
    table = "\n".join(lines)
    print(table)
    print(f"\n{len(jobs)} flights, wall {time.time() - wall:.0f} s")
    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
