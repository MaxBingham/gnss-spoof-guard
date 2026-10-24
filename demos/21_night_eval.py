"""Demo 21 (night-nav): night flight A -> B with GNSS COMPLETELY off from 20 % of the route, over many seeds.

Scenario (NIGHT_PLAN): summer night 23:00, clear, no wind, 60 m AGL. GNSS ok at takeoff; when the TRUE position
passes 20 % of the route length the receiver goes off (gnss.mode "off": no message at all). From then on: IMU,
baro, nadir lidar (TRN on the onboard DSM) and the downward thermal camera vs the onboard thermal prior. The RGB
camera is off. Landing is not part of tonight's scenario: the flight is scored when the drone commits to B (end
of cruise -> start of the descent, after waiting up to precision_wait_s for a fix) and stopped there.

Harness code: truth only scores. Usage:
    python demos/21_night_eval.py --mission berkeley --seeds 20 --methods off,ncc,grad,road,junction,ncc+road
    python demos/21_night_eval.py --mission m-c804bb84 --seeds 3 --methods ncc --trace
Writes $SFHACK_DATA/night/nav_eval/<mission>/<method>/seed<k>.json and a markdown summary table on stdout.
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


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def route_progress(xy: np.ndarray, pts: np.ndarray, cum: np.ndarray) -> float:
    """Arc length of the projection of xy onto the polyline pts (harness)."""
    best, s_best = 1e18, 0.0
    for k in range(len(pts) - 1):
        a, b = pts[k], pts[k + 1]
        d = b - a
        L2 = float(d @ d)
        u = 0.0 if L2 < 1e-9 else float(np.clip((xy - a) @ d / L2, 0, 1))
        e = float(np.linalg.norm(xy - (a + u * d)))
        if e < best:
            best, s_best = e, cum[k] + u * math.sqrt(L2)
    return s_best


def _apply_sets(cfg, sets: dict[str, str]) -> None:
    for k, v in sets.items():
        sect, _, name = k.partition(".")
        cur = getattr(getattr(cfg, sect), name)
        if isinstance(cur, bool):
            val = str(v).lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, tuple):
            val = tuple(float(x) for x in str(v).split(":"))
        else:
            val = type(cur)(v)
        setattr(getattr(cfg, sect), name, val)


def fly(job: dict) -> dict:
    from sfhack.config import default_config
    from sfhack.degradation import Conditions
    from sfhack.runner import FlightSystem

    cfg = default_config()
    cfg.thermal.method = job["method"]
    _apply_sets(cfg, job.get("set", {}))
    cond = Conditions().update({"gnss.mode": "ok"})
    t_wall = time.time()
    fs = FlightSystem(cfg=cfg, mission=job["mission"], seed=job["seed"], conditions=cond, night=True,
                      cruise_agl=job["agl"], thermal_renderer=job["renderer"], thermal_contrast=job.get("contrast", 1.0))
    ap = fs.autopilot
    wps = [w for w in fs.mission.waypoints if w.phase in ("TAKEOFF", "CRUISE")]
    pts = np.array([w.pos[:2] for w in wps])
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    total = float(cum[-1])
    gnss_off_t = None
    arrive = None  # first end of cruise (ABOVE_B / WAIT_FOR_FIX)
    commit = None  # start of the descent
    trace = []
    errs = []
    next_row = 0.0
    n_dec = 0
    while fs.t < job["max_t"]:
        fs.step()
        tr = fs.truth.state()
        if gnss_off_t is None and fs.mission.phase == "CRUISE":
            if route_progress(tr.pos[:2], pts, cum) >= job["gnss_off_frac"] * total:
                fs.update_conditions({"gnss.mode": "off"})
                gnss_off_t = fs.t
        while n_dec < len(ap.decisions):
            text = ap.decisions[n_dec][1]
            n_dec += 1
            if arrive is None and (text.startswith("ABOVE_B") or text.startswith("WAIT_FOR_FIX")):
                est = ap.pose()
                arrive = dict(t=fs.t, err=float(np.linalg.norm(tr.pos[:2] - fs.goal_xy)),
                              est_err=float(np.linalg.norm(est.pos[:2] - tr.pos[:2])), sigma2=est.sigma2_horizontal())
        if fs.t >= next_row:
            next_row += 1.0
            est = ap.pose()
            yaw_err = math.degrees(math.atan2(math.sin(ap.ahrs.yaw - tr.yaw), math.cos(ap.ahrs.yaw - tr.yaw)))
            e = float(np.linalg.norm(est.pos[:2] - tr.pos[:2]))
            if gnss_off_t is not None:
                errs.append(e)
            trace.append([round(fs.t, 1), fs.mission.phase, fs.autopilot.health_state.state.value, round(e, 2),
                          round(est.sigma2_horizontal(), 2), round(yaw_err, 2), ap.thermal_accepted, ap.thermal_fixes,
                          ap.trn_accepted, round(float(tr.pos[2] - fs.true_ground_z(tr.pos[0], tr.pos[1])), 1)])
        if arrive is not None and fs.mission.phase in ("DESCEND", "LAND", "LANDED", "EMERGENCY_LAND"):
            est = ap.pose()
            commit = dict(t=fs.t, err=float(np.linalg.norm(tr.pos[:2] - fs.goal_xy)),
                          est_err=float(np.linalg.norm(est.pos[:2] - tr.pos[:2])), sigma2=est.sigma2_horizontal())
            break
    tr = fs.truth.state()
    res = dict(
        mission=job["mission"], method=job["method"], seed=job["seed"], set=job.get("set", {}), gnss_off_t=gnss_off_t,
        arrive=arrive, commit=commit, timeout=commit is None,
        final_err=commit["err"] if commit else float(np.linalg.norm(tr.pos[:2] - fs.goal_xy)),
        est_err_mean=float(np.mean(errs)) if errs else 0.0, est_err_max=float(np.max(errs)) if errs else 0.0,
        thermal_frames=ap.thermal_frames, thermal_fixes=ap.thermal_fixes, thermal_accepted=ap.thermal_accepted,
        thermal_ms=ap.thermal_ms_total / max(ap.thermal_frames, 1), by_method=ap.thermal_by_method,
        trn_fixes=ap.trn_fixes, trn_accepted=ap.trn_accepted, n_reloc=ap.ekf.n_relocalized,
        decisions=ap.decisions[:80], renderer=fs.thermal_sensor.kind if fs.thermal_sensor else "none",
        route_m=total, wall_s=time.time() - t_wall, trace=trace, thermal_log=ap.thermal_log,
    )
    fs.close()
    tag = job.get("tag") or job["method"].replace("+", "_")
    out = Path(job["out"]) / job["mission"] / tag / f"seed{job['seed']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


def summarize(rows: list[dict], radius: float) -> str:
    lines = [f"| mission | method | n | within {radius:.0f} m | 95 % Wilson CI | error at B mean / median / max (m) | "
             "est. error after GNSS off mean / max (m) | thermal fixes accepted / returned / frames | TRN accepted | timeouts |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    keys = sorted({(r["mission"], r.get("tag") or r["method"]) for r in rows}, key=lambda k: (k[0], k[1] != "off", k[1]))
    for m, meth in keys:
        rs = [r for r in rows if r["mission"] == m and (r.get("tag") or r["method"]) == meth]
        e = np.array([r["final_err"] for r in rs])
        k = int((e <= radius).sum())
        lo, hi = wilson(k, len(rs))
        name = "IMU + TRN only" if meth == "off" else meth
        lines.append(
            f"| {m} | {name} | {len(rs)} | {k}/{len(rs)} ({100 * k / len(rs):.0f} %) | {100 * lo:.0f}-{100 * hi:.0f} % | "
            f"{e.mean():.1f} / {np.median(e):.1f} / {e.max():.1f} | "
            f"{np.mean([r['est_err_mean'] for r in rs]):.1f} / {max(r['est_err_max'] for r in rs):.1f} | "
            f"{sum(r['thermal_accepted'] for r in rs)} / {sum(r['thermal_fixes'] for r in rs)} / "
            f"{sum(r['thermal_frames'] for r in rs)} | {sum(r['trn_accepted'] for r in rs)} | {sum(r['timeout'] for r in rs)} |")
    return "\n".join(lines)


def main() -> None:
    from sfhack.config import data_dir

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mission", default="berkeley", help="comma-separated mission ids")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--methods", default="ncc", help="comma-separated: off (IMU + TRN only), ncc, grad, road, junction, a+b, all")
    ap.add_argument("--agl", type=float, default=60.0)
    ap.add_argument("--gnss-off-frac", type=float, default=0.2)
    ap.add_argument("--renderer", default="auto", help="auto | full | stub")
    ap.add_argument("--contrast", type=float, default=1.0, help="scale the TRUE scene contrast around air (mild night)")
    ap.add_argument("--max-t", type=float, default=400.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--radius", type=float, default=5.0)
    ap.add_argument("--set", action="append", default=[], help="section.field=value config override (tuples: a:b:c)")
    ap.add_argument("--tag", default="", help="output folder / table name instead of the method")
    ap.add_argument("--out", default=str(data_dir() / "night" / "nav_eval"))
    ap.add_argument("--trace", action="store_true", help="print the per-second trace of every flight")
    args = ap.parse_args()
    sets = dict(s.split("=", 1) for s in args.set)
    jobs = [dict(mission=m, method=meth, seed=s, agl=args.agl, gnss_off_frac=args.gnss_off_frac, renderer=args.renderer,
                 max_t=args.max_t, out=args.out, set=sets, contrast=args.contrast, tag=(args.tag + "_" + meth.replace("+", "_")) if args.tag else "")
            for m in args.mission.split(",") for meth in args.methods.split(",")
            for s in range(args.seed0, args.seed0 + args.seeds)]
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
        for job, r in zip(jobs, ex.map(fly, jobs)):
            r["tag"] = job["tag"]
            rows.append(r)
            a = r["arrive"] or {}
            print(f"{r['mission']:>11} {r['tag'] or r['method']:>12} seed {r['seed']:2d}: error at B {r['final_err']:6.1f} m "
                  f"(arrival {a.get('err', float('nan')):.1f} m, claimed 2s {a.get('sigma2', float('nan')):.1f} m), "
                  f"est err mean/max {r['est_err_mean']:.1f}/{r['est_err_max']:.1f}, "
                  f"thermal {r['thermal_accepted']}/{r['thermal_fixes']}/{r['thermal_frames']} {r['thermal_ms']:.0f} ms, "
                  f"TRN {r['trn_accepted']}, GNSS off {r['gnss_off_t'] or 0:.0f} s, {r['renderer']}, "
                  f"{r['wall_s']:.0f} s wall{' TIMEOUT' if r['timeout'] else ''}", flush=True)
            if args.trace:
                print("   t phase health err 2sig yaw_err thermal_acc thermal_ret trn_acc agl")
                for row in r["trace"]:
                    print("  ", *row)
    print(f"\n{len(jobs)} flights in {time.time() - t0:.0f} s, settings: {sets or 'defaults'}, agl {args.agl}, "
          f"GNSS off at {args.gnss_off_frac:.0%} of the route\n")
    print(summarize(rows, args.radius))


if __name__ == "__main__":
    main()
