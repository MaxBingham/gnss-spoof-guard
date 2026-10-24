"""Night eval: A -> B at night, GNSS completely off from 20 % of the route, N seeds per mission in parallel.

    python demos/30_night_eval.py --missions berkeley,m-c804bb84,ukraine-1 --seeds 20 --rows thermal,imu_trn
    python demos/30_night_eval.py --missions ukraine-1 --seeds 4 --workers 4 --rows thermal   # quick look

Rows (onboard configuration, everything else equal):
  thermal   FlightSystem(night=True): thermal localizer (cfg.thermal.method default) + TRN + IMU / baro
  imu_trn   same with cfg.thermal.method = "off" (baseline: IMU + baro + TRN only)
  straight  thermal, but the planner's straight route instead of the night route choice (night_route=False)
  low_contrast  thermal, but the TRUE scene's contrast is scaled by 0.4 (night-nav's thermal_contrast lever; else the harness edits the
            simulator's truth raster; the onboard prior still predicts the full contrast). night-research: ASTER
            night scenes over Berkeley show only ~1-3 K road contrast on mild nights vs the 5-10 K hot-day model.
  thermal_rw  thermal with cfg.ekf.acc_bias_rw = 0.04 (night-nav's candidate fix against overconfident runaways)

Scoring (harness, truth): ARRIVAL error = horizontal |truth - B| when the drone decides it is above B (first
ABOVE_B / WAIT_FOR_FIX decision, else the start of DESCEND). Success = arrival error <= 5 m. The landing itself is
not part of tonight's scenario; `--to-landing` flies on and also records the landed error. Rates come with a
95 % Wilson score interval. The operator's GNSS switch is simulated by the harness: when the TRUE along-route
progress passes `--gnss-off-frac` (0.2), gnss.mode -> "off" (no signal at all), never back on.

Output: $SFHACK_DATA/night/eval/<stamp>/results.json (per flight, incl. 1 Hz error / 2-sigma traces),
table.md, error_over_time.png.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime
from pathlib import Path

import numpy as np

SUCCESS_M = 5.0
ROUGH_COST = {"ukraine-1": 5.0, "m-c804bb84": 1.7, "berkeley": 1.0}  # relative wall time per flight (scheduling)
ROWS = {
    "thermal": {"night": True},
    "imu_trn": {"night": True, "thermal_method": "off"},
    "straight": {"night": True, "night_route": False},
    "low_contrast": {"night": True, "contrast": 0.4},
    "thermal_rw": {"night": True, "acc_bias_rw": 0.04},
}


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95 % Wilson score interval for k successes out of n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def _route_xy(fs) -> np.ndarray:
    wps = [w for w in fs.mission.waypoints if w.phase in ("TAKEOFF", "CRUISE")]
    return np.array([w.pos[:2] for w in wps], float)


def _progress(xy: np.ndarray, poly: np.ndarray, cum: np.ndarray) -> float:
    """Arc length along the polyline of the closest point to xy."""
    best, best_s = math.inf, 0.0
    for i in range(len(poly) - 1):
        a, b = poly[i], poly[i + 1]
        d = b - a
        L2 = float(d @ d)
        t = 0.0 if L2 < 1e-9 else float(np.clip((xy - a) @ d / L2, 0, 1))
        dist = float(np.linalg.norm(a + t * d - xy))
        if dist < best:
            best, best_s = dist, cum[i] + t * math.sqrt(L2)
    return best_s


def fly(job: dict) -> dict:
    from sfhack.config import default_config
    from sfhack.degradation import Conditions
    from sfhack.runner import FlightSystem

    t_wall = time.time()
    cfg = default_config()
    row = ROWS[job["row"]]
    params = inspect.signature(FlightSystem).parameters
    kw: dict = {}
    notes = []
    if row.get("night"):
        if "night" in params:
            kw["night"] = True
        else:
            notes.append("FlightSystem has no night mode on this branch: day localizer off")
    if "acc_bias_rw" in row:  # night-nav's candidate fix (bce976d): let the bias states track the tilt drift
        cfg.ekf.acc_bias_rw = row["acc_bias_rw"]
    if "thermal_method" in row:
        if hasattr(cfg, "thermal"):
            cfg.thermal.method = row["thermal_method"]
        else:
            notes.append("no cfg.thermal")
    if "night_route" in row and "night_route" in params:
        kw["night_route"] = row["night_route"]
    if "contrast" in row and "thermal_contrast" in params:  # night-nav's sim-side lever (same idea as below)
        kw["thermal_contrast"] = row["contrast"]
    if job.get("cruise_agl") and "cruise_agl" in params:
        kw["cruise_agl"] = job["cruise_agl"]
    cond = Conditions().update(gnss__mode="ok")
    localizer = "off" if row.get("night") else "vision"
    mission = str(Path(job["mission_root"]) / job["mission"]) if job.get("mission_root") else job["mission"]
    fs = FlightSystem(cfg=cfg, pose_source="ekf", localizer=localizer, seed=job["seed"], mission=mission,
                      conditions=cond, **kw)
    if "contrast" in row and "thermal_contrast" not in kw:  # simulated reality: a milder night than the prior predicts
        truth = getattr(getattr(fs, "thermal_sensor", None), "truth", None)
        temp = getattr(truth, "temp_K", getattr(truth, "temp", None))  # camera model | stub raster
        if temp is None:
            notes.append("no thermal truth raster to scale")
        else:
            ref = float(np.median(temp))
            temp[...] = ref + row["contrast"] * (temp - ref)
    poly = _route_xy(fs)
    cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(poly, axis=0).T))]
    route_len = float(cum[-1])
    goal = np.asarray(fs.goal_xy, float)
    gnss_off_t = None
    arrival = None
    n_dec = 0
    trace = []
    next_trace = 0.0
    landed_at = None
    next_div, diverged = 0.0, None
    max_t = cfg.sim.max_time_s
    while fs.t < max_t:
        fs.step()
        tr = fs.truth.state()
        if gnss_off_t is None and _progress(tr.pos[:2], poly, cum) >= job["gnss_off_frac"] * route_len:
            fs.update_conditions(gnss__mode="off")
            gnss_off_t = fs.t
        ap = fs.autopilot
        if arrival is None:
            new = ap.decisions[n_dec:]
            n_dec = len(ap.decisions)
            hit = any(text.startswith(("ABOVE_B", "WAIT_FOR_FIX")) for _, text in new) or \
                fs.mission.phase in ("DESCEND", "LAND", "LANDED")
            if hit:
                est = ap.pose()
                arrival = {"t": fs.t, "err": float(np.linalg.norm(tr.pos[:2] - goal)),
                           "est_err": float(np.linalg.norm(est.pos[:2] - tr.pos[:2])),
                           "sigma2": float(est.sigma2_horizontal()),
                           "est_to_b": float(np.linalg.norm(est.pos[:2] - goal))}
        if fs.t >= next_trace:
            next_trace = fs.t + 1.0
            est = ap.pose()
            trace.append([round(fs.t, 1), round(float(np.linalg.norm(est.pos[:2] - tr.pos[:2])), 2),
                          round(float(est.sigma2_horizontal()), 2)])
        if arrival is not None and not job["to_landing"] and fs.t - arrival["t"] >= 1.0:
            break
        if arrival is None and fs.t >= next_div:  # harness economy: an estimate this far off never comes back
            next_div = fs.t + 1.0
            if float(np.linalg.norm(ap.pose().pos[:2] - tr.pos[:2])) > job["abort_err"]:
                diverged = fs.t
                break
        if fs.done:
            landed_at = landed_at if landed_at is not None else fs.t
            if fs.t - landed_at >= 1.0:
                break
    r = fs.result()
    ap = fs.autopilot
    counters = {k: int(getattr(ap, k)) for k in ("thermal_frames", "thermal_fixes", "thermal_accepted", "loc_fixes")
                if hasattr(ap, k)}
    trn = getattr(ap, "trn", None)
    if trn is not None:
        for k in ("n_fixes", "n_accepted", "fixes", "accepted"):
            if hasattr(trn, k) and isinstance(getattr(trn, k), (int, float)):
                counters[f"trn_{k}"] = int(getattr(trn, k))
    fix_sources = dict(getattr(fs, "night_fix_counts", {}) or {})
    out = {**job, "arrival": arrival, "arrived": arrival is not None,
           "arrival_err": arrival["err"] if arrival else float(np.linalg.norm(fs.truth.state().pos[:2] - goal)),
           "final_err": r.final_error, "outcome": "DIVERGED" if diverged is not None else r.outcome,
           "diverged_t": diverged, "landed": r.landed, "t_end": fs.t,
           "gnss_off_t": gnss_off_t, "route_len_m": round(route_len, 1), "est_err_mean": r.est_err_mean,
           "est_err_max": r.est_err_max, "health": r.health_states, "counters": counters, "fix_sources": fix_sources,
           "decisions": [(round(t, 1), s) for t, s in ap.decisions][-40:], "trace": trace, "notes": notes,
           "wall_s": round(time.time() - t_wall, 1), "flightsystem_kwargs": {k: v for k, v in kw.items()}}
    fs.close()
    return out


def table(results: list[dict]) -> str:
    lines = ["| mission | row | n | within 5 m | rate (95 % Wilson) | arrival error mean / median / max (m) | "
             "est. error at arrival mean (m) | 2-sigma at arrival mean (m) | thermal fixes accepted / flight |",
             "|---|---|---:|---:|---|---|---:|---:|---:|"]
    keys = sorted({(r["mission"], r["row"]) for r in results}, key=lambda k: (k[0], list(ROWS).index(k[1])))  # noqa
    for m, row in keys:
        rs = [r for r in results if r["mission"] == m and r["row"] == row]
        e = np.array([r["arrival_err"] for r in rs])
        k = int((e <= SUCCESS_M).sum())
        lo, hi = wilson(k, len(rs))
        ea = [r["arrival"]["est_err"] for r in rs if r["arrival"]]
        sa = [r["arrival"]["sigma2"] for r in rs if r["arrival"]]
        th = [r["counters"].get("thermal_accepted", 0) for r in rs]
        lines.append(f"| {m} | {row} | {len(rs)} | {k} | {k / len(rs):.0%} [{lo:.0%}, {hi:.0%}] | "
                     f"{e.mean():.1f} / {np.median(e):.1f} / {e.max():.1f} | "
                     f"{np.mean(ea) if ea else float('nan'):.1f} | {np.mean(sa) if sa else float('nan'):.1f} | "
                     f"{np.mean(th):.0f} |")
    return "\n".join(lines)


def plot(results: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = sorted({(r["mission"], r["row"]) for r in results})
    missions = sorted({g[0] for g in groups})
    fig, axes = plt.subplots(1, len(missions), figsize=(6 * len(missions), 4.5), squeeze=False)
    for ax, m in zip(axes[0], missions):
        for (mm, row), col in zip([g for g in groups if g[0] == m], ("tab:cyan", "tab:orange", "tab:green")):
            rs = [r for r in results if r["mission"] == mm and r["row"] == row]
            # time since GNSS off, 1 s bins
            series = []
            for r in rs:
                tr = np.array(r["trace"])
                t0 = r["gnss_off_t"] or 0.0
                series.append((tr[:, 0] - t0, tr[:, 1]))
            if not series:
                continue
            T = np.arange(0, max(s[0].max() for s in series), 1.0)
            E = np.array([np.interp(T, s[0], s[1], right=np.nan) for s in series])
            med = np.nanmedian(E, axis=0)
            q1, q3 = np.nanpercentile(E, 25, axis=0), np.nanpercentile(E, 75, axis=0)
            ax.plot(T, med, color=col, label=f"{row} (median, n={len(rs)})")
            ax.fill_between(T, q1, q3, color=col, alpha=0.2)
        ax.set_title(m)
        ax.set_xlabel("time since GNSS off (s)")
        ax.set_ylabel("|estimate - truth| (m)")
        ax.axhline(SUCCESS_M, color="k", lw=0.8, ls="--")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=90)


def main() -> None:
    from sfhack.config import data_dir

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--missions", default="berkeley,m-c804bb84,ukraine-1")
    ap.add_argument("--rows", default="thermal,imu_trn")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gnss-off-frac", type=float, default=0.2)
    ap.add_argument("--cruise-agl", type=float, default=60.0)
    ap.add_argument("--to-landing", action="store_true", help="fly on after arrival and record the landed error")
    ap.add_argument("--abort-err", type=float, default=500.0, help="stop a flight once |estimate - truth| exceeds "
                    "this (m) before arrival: scored as a failure (DIVERGED)")
    ap.add_argument("--mission-root", default="", help="read missions from this folder (a frozen copy of the data)")
    ap.add_argument("--only", default="", help="restrict to mission:row pairs, comma separated")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else data_dir() / "night" / "eval" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    jobs = [{"mission": m, "row": row, "seed": s, "gnss_off_frac": args.gnss_off_frac, "cruise_agl": args.cruise_agl,
             "to_landing": args.to_landing, "abort_err": args.abort_err, "mission_root": args.mission_root}
            for m in args.missions.split(",") for row in args.rows.split(",") for s in range(args.seed0, args.seed0 + args.seeds)]
    if args.only:  # "mission:row" pairs, e.g. ukraine-1:straight
        keep = set(args.only.split(","))
        jobs = [j for j in jobs if f"{j['mission']}:{j['row']}" in keep]
    # headline rows first (if time runs out the main table is complete), longest flights first within them
    jobs.sort(key=lambda j: (j["row"] not in ("thermal", "imu_trn"), -ROUGH_COST.get(j["mission"], 1.0)))
    print(f"{len(jobs)} flights on {args.workers} workers -> {out}", flush=True)
    results, t0 = [], time.time()
    if (out / "results.json").exists():  # resume: keep finished flights, run the rest
        results = json.loads((out / "results.json").read_text(encoding="utf-8"))
        done = {(r["mission"], r["row"], r["seed"]) for r in results}
        jobs = [j for j in jobs if (j["mission"], j["row"], j["seed"]) not in done]
        print(f"resume: {len(results)} flights already in {out / 'results.json'}, {len(jobs)} to go", flush=True)
    n_total = len(results) + len(jobs)
    for attempt in range(6):  # a worker killed from outside (memory) breaks the whole pool: start a new one
        if not jobs:
            break
        broken = False
        with ProcessPoolExecutor(args.workers) as ex:
            futs = {ex.submit(fly, j): j for j in jobs}
            for f in as_completed(futs):
                j = futs[f]
                try:
                    r = f.result()
                except BrokenProcessPool as e:
                    broken = True
                    print(f"pool broken ({e}); restarting it for the remaining flights", flush=True)
                    break
                except Exception as e:  # noqa: BLE001 - one broken flight must not lose the others
                    print(f"FAILED {j}: {type(e).__name__}: {e}", flush=True)
                    jobs = [x for x in jobs if x is not j]
                    continue
                results.append(r)
                jobs = [x for x in jobs if x is not j]
                print(f"[{len(results):3d}/{n_total} {time.time() - t0:6.0f}s] {r['mission']:12s} {r['row']:9s} seed {r['seed']:2d}: "
                      f"arrival {r['arrival_err']:7.1f} m  ({r['outcome']}, gnss off {r['gnss_off_t']}, "
                      f"{r['counters']}) {' '.join(r['notes'])}", flush=True)
                (out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
            if broken:
                for f in futs:
                    f.cancel()
        if not broken:
            break
    md = table(results)
    (out / "table.md").write_text(md + "\n", encoding="utf-8")
    print("\n" + md)
    try:
        plot(results, out / "error_over_time.png")
        print(f"\nplot: {out / 'error_over_time.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"plot failed: {e}")
    print(f"results: {out / 'results.json'}")


if __name__ == "__main__":
    main()
