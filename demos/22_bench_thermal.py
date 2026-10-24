"""Demo 22 (night-nav): open-loop benchmark of the thermal localizer methods.

The drone flies the mission at 60 m AGL at night with GNSS on the whole way (so the flight itself is unaffected);
for every n-th thermal frame each method gets a PERTURBED prior: the onboard estimate shifted by N(0, sigma) per
axis with covariance sigma^2, sigma from --sigmas. Scored against the true camera position (harness only).
Reports per method and sigma: fix rate, error median / p90 / max, share of fixes > 5 m off ("wrong"), NIS
consistency (share with Mahalanobis^2 < 9.21 against the truth), yaw-correction error, time per call.

    python demos/22_bench_thermal.py --mission berkeley --methods ncc,grad,road,junction,ncc+road --sigmas 3,10,20
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run(job: dict) -> list[dict]:
    import copy

    from sfhack import bus as topics
    from sfhack.config import default_config
    from sfhack.degradation import Conditions
    from sfhack.perception.thermal_localizer import ThermalLocalizer
    from sfhack.runner import FlightSystem

    cfg = default_config()
    for k, v in job.get("set", {}).items():
        sect, _, name = k.partition(".")
        cur = getattr(getattr(cfg, sect), name)
        setattr(getattr(cfg, sect), name, type(cur)(v))
    fs = FlightSystem(cfg=cfg, mission=job["mission"], seed=job["seed"], conditions=Conditions().update({"gnss.mode": "ok"}),
                      night=True, cruise_agl=job["agl"], thermal_renderer=job["renderer"])
    ap = fs.autopilot
    ap.thermal = None  # no fusion: the flight runs on GNSS, we only score the localizer
    locs = {}
    for m in job["methods"]:
        c = copy.deepcopy(cfg.thermal)
        c.method = m
        locs[m] = ThermalLocalizer(fs.onboard_dir, c, onboard_map=fs.onboard_map,
                                   prior_path=fs.mission_dir / "onboard" / "thermal_prior.npz")
    rng = np.random.default_rng(job["seed"] + 99)
    rows: list[dict] = []
    k = [0]

    def on_frame(fr) -> None:
        k[0] += 1
        if fs.mission.phase != "CRUISE" or k[0] % job["every"]:
            return
        tr = fs.truth.state()
        est = ap.pose(report=False)
        agl = ap.lidar_agl if ap.lidar_agl is not None and fr.t - ap._lidar_t < 0.5 else None
        yaw_err = math.atan2(math.sin(ap.ahrs.yaw - tr.yaw), math.cos(ap.ahrs.yaw - tr.yaw))
        for sig in job["sigmas"]:
            p = copy.copy(est)
            p.pos = est.pos.copy()
            p.pos[:2] = tr.pos[:2] + rng.normal(0, sig, 2)
            p.cov = est.cov.copy()
            p.cov[:2, :2] = np.eye(2) * sig * sig
            for m, loc in locs.items():
                t0 = time.perf_counter()
                f = loc.update(fr.image_K, p, agl, fr.intrinsics, t=fr.t)
                ms = (time.perf_counter() - t0) * 1000
                row = dict(method=m, sigma=sig, t=fr.t, ms=ms, ok=f is not None, reason=loc.last_reason,
                           x=float(tr.pos[0]), y=float(tr.pos[1]))
                if f is not None:
                    d = f.xy - tr.pos[:2]
                    row.update(err=float(np.linalg.norm(d)), nis=float(d @ np.linalg.solve(f.cov, d)),
                               yaw_corr_err=math.degrees(f.yaw_corr + yaw_err), yaw_err=math.degrees(yaw_err),
                               sig_major=math.sqrt(float(np.linalg.eigvalsh(f.cov).max())), q=f.quality, used=f.method)
                rows.append(row)

    fs.bus.subscribe(topics.THERMAL, on_frame)
    while fs.t < job["max_t"] and fs.mission.phase not in ("DESCEND", "LAND", "LANDED"):
        fs.step()
    fs.close()
    return rows


def main() -> None:
    from sfhack.config import data_dir

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mission", default="berkeley")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--methods", default="ncc,grad,road,junction")
    ap.add_argument("--sigmas", default="3,10,20")
    ap.add_argument("--every", type=int, default=3, help="score every n-th thermal frame")
    ap.add_argument("--agl", type=float, default=60.0)
    ap.add_argument("--renderer", default="auto")
    ap.add_argument("--max-t", type=float, default=300.0)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    sets = dict(s.split("=", 1) for s in args.set)
    jobs = [dict(mission=m, seed=s, methods=args.methods.split(","), sigmas=[float(x) for x in args.sigmas.split(",")],
                 every=args.every, agl=args.agl, renderer=args.renderer, max_t=args.max_t, set=sets)
            for m in args.mission.split(",") for s in range(args.seeds)]
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=min(12, len(jobs))) as ex:
        for job, rs in zip(jobs, ex.map(run, jobs)):
            for r in rs:
                r["mission"] = job["mission"]
            rows += rs
    print(f"{len(rows)} localizer calls in {time.time() - t0:.0f} s, settings {sets or 'defaults'}\n")
    print("| mission | method | prior sigma (m) | fix rate | error median / p90 / max (m) | wrong > 5 m | NIS < 9.21 | "
          "yaw-corr error median (deg) / AHRS yaw error (deg) | ms / call | top no-fix reasons |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for m in args.mission.split(","):
        for meth in args.methods.split(","):
            for sig in sorted({r["sigma"] for r in rows}):
                rs = [r for r in rows if r["mission"] == m and r["method"] == meth and r["sigma"] == sig]
                if not rs:
                    continue
                ok = [r for r in rs if r["ok"]]
                e = np.array([r["err"] for r in ok]) if ok else np.array([np.nan])
                reasons: dict[str, int] = {}
                for r in rs:
                    if not r["ok"]:
                        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
                top = ", ".join(f"{k} {v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])[:3])
                yc = np.median([abs(r["yaw_corr_err"]) for r in ok]) if ok else np.nan
                ye = np.median([abs(r["yaw_err"]) for r in ok]) if ok else np.nan
                print(f"| {m} | {meth} | {sig:.0f} | {len(ok)}/{len(rs)} ({100 * len(ok) / len(rs):.0f} %) | "
                      f"{np.nanmedian(e):.2f} / {np.nanpercentile(e, 90):.2f} / {np.nanmax(e):.1f} | "
                      f"{sum(r['err'] > 5 for r in ok)} | {np.mean([r['nis'] < 9.21 for r in ok]) * 100 if ok else 0:.0f} % | "
                      f"{yc:.2f} / {ye:.2f} | {np.mean([r['ms'] for r in rs]):.0f} | {top} |")
    out = Path(args.out) if args.out else data_dir() / "night" / "nav_bench" / f"bench_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows), encoding="utf-8")
    print(f"\nrows -> {out}")


if __name__ == "__main__":
    main()
