"""Demo 10: the live-demo mission (vision localizer, GNSS ok at start) over several seeds with levers and a
lever schedule, plus a per-second trace of the GNSS -> GNSS-free transition. Harness code: truth only scores.

    python demos/10_eval_levers.py --seeds 10 --cond-at "40:gnss=jam"
    python demos/10_eval_levers.py --seeds 5 --cond-at "40:gnss=jam" --trace          # per-second log
    python demos/10_eval_levers.py --seeds 10 --cond-at "40:gnss=jam" --weather fog:0.65
    python demos/10_eval_levers.py --seeds 10 --cond-at "40:gnss=jam;40:camera=lost" --trn on

Trace columns: phase, health, claimed 2-sigma, true |estimate - truth|, GNSS fixes fused, vision fixes
produced / accepted / rejected (max NIS), TRN fixes accepted, re-localizations (cumulative).
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def fly(job: dict) -> dict:
    from sfhack import bus as topics
    from sfhack.config import default_config
    from sfhack.degradation import Conditions
    from sfhack.degradation.cli import parse_schedule
    from sfhack.runner import FlightSystem

    cfg = default_config()
    for k, v in job["cfg"].items():
        sect, _, name = k.partition(".")
        setattr(getattr(cfg, sect), name, type(getattr(getattr(cfg, sect), name))(v))
    cond = Conditions().update(job["cond"])
    fs = FlightSystem(cfg=cfg, pose_source="ekf", localizer=job["localizer"], seed=job["seed"], conditions=cond,
                      schedule=parse_schedule(job["sched"]) if job["sched"] else None)
    ap, ekf = fs.autopilot, fs.autopilot.ekf
    n_gnss = [0]
    fs.bus.subscribe(topics.GNSS, lambda m: n_gnss.__setitem__(0, n_gnss[0] + 1))
    rows, sig2_max, err_max = [], 0.0, 0.0
    prev = dict(g=0, f=0, a=0, r=0, n=0, ta=0)
    next_row = 1.0
    landed_at = None
    max_t = cfg.sim.max_time_s
    while fs.t < max_t:
        fs.step()
        if fs.done:
            landed_at = landed_at if landed_at is not None else fs.t
            if fs.t - landed_at >= 1.0:
                break
        if fs.t >= next_row:
            next_row += 1.0
            est, tr = ap.pose(), fs.truth.state()
            err = float(np.linalg.norm(est.pos[:2] - tr.pos[:2]))
            s2 = est.sigma2_horizontal()
            if ap.mission.phase not in ("PREFLIGHT", "LANDED"):
                sig2_max, err_max = max(sig2_max, s2), max(err_max, err)
            nis = [x[1] for x in ekf.nis_log[prev["n"]:]]
            trn_acc = getattr(ap, "trn_accepted", 0)
            rows.append((fs.t, ap.mission.phase, ap.health_state.state.value, s2, err, n_gnss[0] - prev["g"],
                         ap.loc_fixes - prev["f"], ekf.n_fix_accepted - prev["a"], ekf.n_fix_rejected - prev["r"],
                         max(nis) if nis else float("nan"), trn_acc - prev["ta"], ekf.n_relocalized,
                         fs.conditions.summary()))
            prev = dict(g=n_gnss[0], f=ap.loc_fixes, a=ekf.n_fix_accepted, r=ekf.n_fix_rejected, n=len(ekf.nis_log),
                        ta=trn_acc)
    r = fs.result()
    sources = sorted(getattr(ap, "sources_used", set()))
    out = {"seed": job["seed"], "r": r, "decisions": list(ap.decisions), "rows": rows, "sig2_max": sig2_max,
           "err_max": err_max, "sources": sources}
    fs.close()
    return out


def add_eval_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--localizer", default="vision")
    ap.add_argument("--gnss", default="ok")
    ap.add_argument("--weather", default="")
    ap.add_argument("--wind", default="")
    ap.add_argument("--cond", default="", help="extra initial levers, e.g. 'camera=lost'")
    ap.add_argument("--cond-at", default="", help="lever schedule, e.g. '40:gnss=jam'")
    ap.add_argument("--set", action="append", default=[], help="config override, e.g. ekf.gate_chi2=9.21")
    ap.add_argument("--trn", choices=["on", "off"], default=None, help="terrain-referenced navigation")


def jobs_from_args(args: argparse.Namespace) -> list[dict]:
    cond = ",".join(filter(None, [f"gnss={args.gnss}" if args.gnss else "",
                                  f"weather={args.weather}" if args.weather else "",
                                  f"wind={args.wind}" if args.wind else "", args.cond]))
    cfg = dict(s.split("=", 1) for s in args.set)
    if args.trn is not None:
        cfg["trn.enabled"] = args.trn == "on"
    return [{"seed": s, "localizer": args.localizer, "cond": cond, "sched": args.cond_at, "cfg": cfg}
            for s in range(args.seed0, args.seed0 + args.seeds)]


def summarize(runs: list[dict], radius: float = 3.0) -> dict:
    rs = [x["r"] for x in runs]
    fe = np.array([r.final_error for r in rs])
    return {"n": len(rs), "ok": sum(r.success for r in rs), "fe_mean": float(fe.mean()), "fe_med": float(np.median(fe)),
            "fe_max": float(fe.max()), "sig2_max": max(x["sig2_max"] for x in runs),
            "err_max": max(x["err_max"] for x in runs),
            "sources": sorted(set().union(*[set(x["sources"]) for x in runs]))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_eval_args(ap)
    ap.add_argument("--trace", action="store_true", help="print the per-second log of every seed")
    ap.add_argument("--decisions", action="store_true", help="print the decision log of every seed")
    args = ap.parse_args()
    jobs = jobs_from_args(args)
    with ProcessPoolExecutor(args.workers) as ex:
        runs = list(ex.map(fly, jobs))
    for x in runs:
        r = x["r"]
        print(f"seed {x['seed']:2d}: {r.outcome:15s} final {r.final_error:6.2f} m  est mean {r.est_err_mean:5.2f} / "
              f"max {r.est_err_max:6.2f} m  max 2-sigma {x['sig2_max']:5.1f} m  vision {r.loc_fixes}/{r.loc_frames} "
              f"rejected {r.fixes_rejected}  t {r.t_end:5.1f} s  health {r.health_states}  sources {x['sources']}")
        if args.trace:
            print("      t  phase          health     2sig    err gnss vis acc rej  maxNIS trn reloc levers")
            for row in x["rows"]:
                t, ph, h, s2, e, g, f, a, rj, nis, ta, rl, lv = row
                print(f"  {t:5.0f}  {ph:13s}  {h:9s} {s2:5.1f} {e:6.1f} {g:4d} {f:3d} {a:3d} {rj:3d} {nis:7.1f} "
                      f"{ta:3d} {rl:5d} {lv}")
        if args.trace or args.decisions:
            for t, text in x["decisions"]:
                print(f"           {t:6.1f}s {text}")
    s = summarize(runs)
    print(f"\nSUCCESS {s['ok']}/{s['n']} within 3 m of B | final error mean {s['fe_mean']:.2f} m, median "
          f"{s['fe_med']:.2f} m, max {s['fe_max']:.2f} m | max 2-sigma {s['sig2_max']:.1f} m | max est error "
          f"{s['err_max']:.1f} m | sources {s['sources']}")


if __name__ == "__main__":
    main()
