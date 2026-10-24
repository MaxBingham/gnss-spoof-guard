"""Recon eval: the full mission (bearing -> search -> find -> orbit -> home -> land -> report) over N seeds.

    python demos/42_recon_eval.py --mission ukraine-1 --seeds 20 --navs lights
    python demos/42_recon_eval.py --mission ukraine-1-rmtest --seeds 6 --navs imu,flow,lights --workers 6
    python demos/42_recon_eval.py --mission ukraine-1-rmtest --seeds 4 --gnss-on    # mission logic with perfect-ish nav

Per flight (harness, truth-scored, sfhack/recon_runner.py): FOUND = a reported find within 50 m of a true target;
report position error (each true find: |reported position - nearest true target|); false finds; landing error
(|truth - pad| after touchdown); mission time (takeoff -> touchdown, battery 20 min); searched area. The bearing per
seed is simulated (recon-world's sim_array when merged, else truth + bias + noise, sigma 7 deg) unless --bearing.
Rates come with 95 % Wilson score intervals.

Output: $SFHACK_DATA/recon/eval/<stamp>/results.json, table.md (+ every flight's report under reports/).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def fly(job: dict) -> dict:
    from sfhack.config import default_config
    from sfhack.recon_runner import ReconFlight

    t0 = time.time()
    cfg = default_config()
    try:
        rf = ReconFlight(mission=job["mission"], seed=job["seed"], bearing=job.get("bearing"),
                         nav=None if job["nav"] == "default" else job["nav"], cfg=cfg,
                         gnss_off_m=None if job["gnss_on"] else 50.0, thermal_hz=job["thermal_hz"],
                         run_id=f"{job['stamp']}_{job['nav']}_s{job['seed']}")
        s = rf.run()
        s["decisions"] = [(round(t, 1), d) for t, d in rf.recon.decision_log]
        rf.close()
    except Exception as e:  # noqa: BLE001  (one broken flight must not kill the eval)
        import traceback

        s = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
    s.update({"nav": job["nav"], "seed": job["seed"], "mission": job["mission"], "wall_s": round(time.time() - t0, 1)})
    return s


def _fmt(xs: list[float]) -> str:
    return "-" if not xs else f"{np.mean(xs):.1f} / {np.median(xs):.1f} / {np.max(xs):.1f}"


def table(results: list[dict]) -> str:
    lines = ["| nav | n | found | find rate (95 % Wilson) | report error mean / median / max (m) | false finds / flight | "
             "landed | landing error mean / median / max (m) | mission time mean / max (s) | within 20 min | searched km2 |",
             "|---|---:|---:|---|---|---:|---:|---|---|---:|---:|"]
    for nav in sorted({r["nav"] for r in results}):
        rs = [r for r in results if r["nav"] == nav and "error" not in r]
        n = len(rs)
        if not n:
            continue
        k = sum(r["found"] for r in rs)
        lo, hi = wilson(k, n)
        errs = [r["first_find_err_m"] for r in rs if r["first_find_err_m"] is not None]
        land = [r["landing_err_m"] for r in rs if r["landed"]]
        mt = [r["mission_time_s"] for r in rs]
        lines.append(f"| {nav} | {n} | {k} | {k / n:.0%} [{lo:.0%}, {hi:.0%}] | {_fmt(errs)} | "
                     f"{np.mean([r['n_false_finds'] for r in rs]):.2f} | {sum(r['landed'] for r in rs)}/{n} | {_fmt(land)} | "
                     f"{np.mean(mt):.0f} / {np.max(mt):.0f} | {sum(r['within_battery'] for r in rs)}/{n} | "
                     f"{np.mean([r['searched_km2'] for r in rs]):.2f} |")
    bad = [r for r in results if "error" in r]
    if bad:
        lines.append(f"\n{len(bad)} flight(s) crashed: " + "; ".join(f"{r['nav']} s{r['seed']}: {r['error']}" for r in bad))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mission", default="ukraine-1")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--navs", default="default", help="comma list: imu,flow,lights (recon-nav) or default")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--thermal-hz", type=float, default=None)
    ap.add_argument("--gnss-on", action="store_true")
    ap.add_argument("--bearing", default=None)
    a = ap.parse_args()
    from sfhack.config import data_dir

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = data_dir() / "recon" / "eval" / stamp
    out.mkdir(parents=True, exist_ok=True)
    jobs = [{"mission": a.mission, "seed": s, "nav": nav, "gnss_on": a.gnss_on, "thermal_hz": a.thermal_hz,
             "bearing": a.bearing, "stamp": stamp}
            for s in range(a.seed0, a.seed0 + a.seeds) for nav in a.navs.split(",")]
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(a.workers, 6)) as ex:
        futs = [ex.submit(fly, j) for j in jobs]
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            if "error" in r:
                print(f"[{len(results)}/{len(jobs)}] {r['nav']} s{r['seed']}: ERROR {r['error']}", flush=True)
            else:
                print(f"[{len(results)}/{len(jobs)}] {r['nav']} s{r['seed']}: found={r['found']} err={r['first_find_err_m']} "
                      f"false={r['n_false_finds']} landed={r['landed']} land_err={r['landing_err_m']:.1f} m "
                      f"t={r['mission_time_s']:.0f} s  wall {r['wall_s']:.0f} s", flush=True)
            (out / "results.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    md = (f"# Recon eval {stamp}\n\nmission {a.mission}, seeds {a.seed0}-{a.seed0 + a.seeds - 1}, thermal {a.thermal_hz or 'cfg'} Hz"
          f"{', GNSS ON (stand-in)' if a.gnss_on else ', GNSS off at 50 m'}, wall {time.time() - t0:.0f} s\n\n{table(results)}\n")
    (out / "table.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
