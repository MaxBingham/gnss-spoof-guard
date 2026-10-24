"""Recon-nav eval: out along a bearing and back on ukraine-1 at night, three nav configs, N seeds.

GNSS on for the first 50 m from the pad (truth distance, harness), then off for good. The drone flies out along
the bearing at `--agl` above the onboard terrain, orbits the far point once, flies back tail first and lands on A,
always with its tail (rear NIR camera) toward the base. Configs (FlightSystem nav=): imu (IMU + compass + baro),
flow (+ optical flow on the downward thermal camera), lights (+ flow + base-light fixes).

ukraine-1 is 2.0 x 1.0 km with A near its SW corner: the longest leg over mapped ground is ~2.04 km toward the NE
corner, so the default bearing points there and the turn is at `--dist` 1950 m (map limit; past the edge there is
no terrain and no thermal scene).

Per flight (results.json): a 1 Hz trace (distance from the base, |estimate - truth|, along / cross-track error,
heading error, 2-sigma, light frames / fixes / accepted, flow measurements), the error at the turn and at fixed
distances out and back, and the landing error on A. Plots: error vs distance from the base (outbound and return),
heading error, light-fix rate vs distance. Wilson 95 % CI for "within 50 m at the turn" (and at 1 / 1.5 km).

  uv run python demos/42_recon_nav_eval.py --seeds 20 --workers 6
  uv run python demos/42_recon_nav_eval.py --seeds 2 --navs lights --dist 800        # quick check
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

CHECK_KM = (0.5, 1.0, 1.5, 2.0)


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95 % Wilson score interval for k successes out of n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def default_bearing(mission: str, margin: float = 40.0) -> tuple[float, float]:
    """(compass az deg, ground distance to the far corner) toward the mapped corner farthest from A."""
    from sfhack.runner import resolve_mission

    man = json.loads((resolve_mission(mission) / "onboard" / "manifest.json").read_text(encoding="utf-8"))
    a = np.array(man["a"]["enu"], float)
    e = man["ortho"]["extent_enu"]
    corners = [np.array([x, y]) for x in (e["min_x"] + margin, e["max_x"] - margin) for y in (e["min_y"] + margin, e["max_y"] - margin)]
    c = max(corners, key=lambda p: float(np.linalg.norm(p - a)))
    d = c - a
    return (math.degrees(math.atan2(d[0], d[1])) % 360.0), float(np.linalg.norm(d))


def fly(job: dict) -> dict:
    from sfhack.config import default_config
    from sfhack.degradation import Conditions
    from sfhack.planning.recon_route import out_and_back_plan
    from sfhack.runner import FlightSystem
    from sfhack.world.onboard_map import OnboardMap
    from sfhack.runner import resolve_mission

    t_wall = time.time()
    cfg = default_config()
    cfg.mission.cruise_speed = job["speed"]
    cfg.sim.max_time_s = job["max_t"]
    mdir = resolve_mission(job["mission"])
    om = OnboardMap(mdir / "onboard")
    a = om.a_enu
    plan = out_and_back_plan(a, job["az"], job["dist"], job["agl"], om.ground_z, cfg.mission)
    fs = FlightSystem(cfg=cfg, mission=job["mission"], nav=job["nav"], plan=plan, tail_to_base=True, seed=job["seed"],
                      conditions=Conditions().update(gnss__mode="ok"), thermal_renderer=job["renderer"])
    ap = fs.autopilot
    u = np.array([math.sin(math.radians(job["az"])), math.cos(math.radians(job["az"]))])  # along the bearing (ENU)
    nrm = np.array([-u[1], u[0]])
    gnss_off_t = None
    trace = []
    next_trace = 0.0
    max_d, turn = 0.0, None
    leg = "out"
    marks: dict[str, dict] = {}
    diverged = None
    landed_at = None
    while fs.t < job["max_t"]:
        fs.step()
        tr = fs.truth.state()
        rel = tr.pos[:2] - a
        d = float(np.linalg.norm(rel))
        if gnss_off_t is None and d >= job["gnss_off_m"]:
            fs.update_conditions(gnss__mode="off")
            gnss_off_t = fs.t
        if fs.t < next_trace:
            if fs.done:
                landed_at = landed_at if landed_at is not None else fs.t
                if fs.t - landed_at >= 1.0:
                    break
            continue
        next_trace = fs.t + 1.0
        est = ap.pose()
        e = est.pos[:2] - tr.pos[:2]
        err = float(np.linalg.norm(e))
        yaw_err = math.degrees(math.atan2(math.sin(ap.ahrs.yaw - tr.yaw), math.cos(ap.ahrs.yaw - tr.yaw)))
        if leg == "out" and d > max_d:
            max_d = d
        if leg == "out" and fs.mission.idx >= job["turn_idx"]:
            leg = "orbit"
            turn = {"t": fs.t, "d": d, "err": err, "along": float(e @ u), "cross": float(e @ nrm)}
        if leg == "orbit" and fs.mission.idx > job["back_idx"]:
            leg = "back"
        for km in CHECK_KM:
            key = f"{leg}_{km}"
            if leg in ("out", "back") and key not in marks and ((leg == "out" and d >= km * 1000) or
                                                                 (leg == "back" and d <= km * 1000 and turn and turn["d"] > km * 1000)):
                marks[key] = {"t": fs.t, "err": err, "along": float(e @ u), "cross": float(e @ nrm)}
        src, _ = ap.recon_nav_source(fs.t)
        trace.append([round(fs.t, 1), leg, round(d, 1), round(err, 2), round(float(e @ u), 2), round(float(e @ nrm), 2),
                      round(yaw_err, 3), round(float(est.sigma2_horizontal()), 2), ap.light_frames, ap.light_fixes,
                      ap.light_accepted, ap.flow_meas, ap.flow_accepted, src, round(math.degrees(ap.mag_bias_est), 3)])
        if err > job["abort_err"]:
            diverged = fs.t
            break
        if fs.done:
            landed_at = landed_at if landed_at is not None else fs.t
            if fs.t - landed_at >= 1.0:
                break
    tr = fs.truth.state()
    land_err = float(np.linalg.norm(tr.pos[:2] - a))
    out = {**job, "gnss_off_t": gnss_off_t, "turn": turn, "max_d": max_d, "marks": marks,
           "landed": bool(fs.done), "land_err": land_err, "outcome": "DIVERGED" if diverged else ("LANDED" if fs.done else "TIMEOUT"),
           "diverged_t": diverged, "t_end": fs.t, "mag_bias_true_deg": math.degrees(fs.sim.mag_bias),
           "mag_bias_est_deg": math.degrees(ap.mag_bias_est),
           "counters": {"light_frames": ap.light_frames, "light_fixes": ap.light_fixes, "light_accepted": ap.light_accepted,
                        "flow_frames": ap.flow_frames, "flow_meas": ap.flow_meas, "flow_accepted": ap.flow_accepted,
                        "vel_rejected": ap.ekf.n_vel_rejected, "fix_rejected": ap.ekf.n_fix_rejected},
           "flow_stats": {k: getattr(ap.flow.stats, k) for k in ("frames", "measurements", "no_height", "uniform", "weak_peak", "big_shift")} if ap.flow else None,
           "thermal_kind": fs.thermal_sensor.kind if fs.thermal_sensor is not None else None,
           "decisions": [(round(t, 1), s) for t, s in ap.decisions][-30:], "trace": trace,
           "wall_s": round(time.time() - t_wall, 1)}
    fs.close()
    return out


def table(results: list[dict]) -> str:
    navs = [n for n in ("imu", "flow", "lights") if any(r["nav"] == n for r in results)]
    lines = ["| nav | n | err at turn: median / p90 / max (m) | within 50 m at turn [95 % CI] | out 1.0 km median | "
             "back 1.0 km median | landing on A: median / max (m) | |heading err| at turn median (deg) |",
             "|---|---|---|---|---|---|---|---|"]
    for nav in navs:
        rs = [r for r in results if r["nav"] == nav]
        te = np.array([r["turn"]["err"] if r["turn"] else np.inf for r in rs])
        k = int(np.sum(te <= 50.0))
        lo, hi = wilson(k, len(rs))
        def med(key: str) -> str:
            v = [r["marks"][key]["err"] for r in rs if key in r["marks"]]
            return f"{np.median(v):.1f}" if v else "-"
        le = np.array([r["land_err"] for r in rs])
        he = []
        for r in rs:
            if r["turn"]:
                row = min(r["trace"], key=lambda x: abs(x[0] - r["turn"]["t"]))
                he.append(abs(row[6]))
        fin = te[np.isfinite(te)]
        lines.append(f"| {nav} | {len(rs)} | {np.median(fin):.1f} / {np.percentile(fin, 90):.1f} / {fin.max():.1f} | "
                     f"{k}/{len(rs)} [{100 * lo:.0f}-{100 * hi:.0f} %] | {med('out_1.0')} | {med('back_1.0')} | "
                     f"{np.median(le):.1f} / {le.max():.1f} | {np.median(he) if he else float('nan'):.2f} |")
    return "\n".join(lines)


def plot(results: list[dict], out: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"imu": "#c0392b", "flow": "#e67e22", "lights": "#2471a3"}
    paths = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, leg in zip(axes, ("out", "back")):
        for nav, col in colors.items():
            rs = [r for r in results if r["nav"] == nav]
            if not rs:
                continue
            bins = np.arange(0, 2101, 100)
            per = [[] for _ in bins]
            for r in rs:
                rows = [x for x in r["trace"] if x[1] == leg]
                if rows:
                    ax.plot([x[2] for x in rows], [x[3] for x in rows], color=col, alpha=0.18, lw=0.8)
                for x in rows:
                    per[min(int(x[2] // 100), len(bins) - 1)].append(x[3])
            m = [(b + 50, np.median(v)) for b, v in zip(bins, per) if len(v) >= 3]
            if m:
                ax.plot(*zip(*m), color=col, lw=2.4, label=f"{nav} (median of {len(rs)})")
        ax.set_yscale("log")
        ax.set_xlabel("true distance from the base (m)")
        ax.set_title("outbound (GNSS off after 50 m)" if leg == "out" else "return (tail first, homing on the lights)")
        ax.axhline(50, color="0.5", ls="--", lw=0.8)
        ax.grid(alpha=0.3, which="both")
        if leg == "back":
            ax.invert_xaxis()
    axes[0].set_ylabel("|estimate - truth| (m)")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("ukraine-1, night, no GNSS after 50 m: position error vs distance from the base")
    fig.tight_layout()
    p = out / "error_vs_distance.png"
    fig.savefig(p, dpi=130)
    paths.append(p)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for nav, col in colors.items():
        for r in [r for r in results if r["nav"] == nav]:
            axes[0].plot([x[0] for x in r["trace"]], [x[6] for x in r["trace"]], color=col, alpha=0.35, lw=0.8)
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("AHRS heading - true heading (deg)")
    axes[0].set_title("heading error (lights learn the compass bias)")
    axes[0].grid(alpha=0.3)
    rs = [r for r in results if r["nav"] == "lights"]
    if rs:
        bins = np.arange(0, 2101, 100)
        frames = np.zeros(len(bins))
        fixes = np.zeros(len(bins))
        for r in rs:
            tr = r["trace"]
            for x0, x1 in zip(tr[:-1], tr[1:]):
                b = min(int(x1[2] // 100), len(bins) - 1)
                frames[b] += x1[8] - x0[8]
                fixes[b] += x1[10] - x0[10]
        ok = frames > 0
        axes[1].bar(bins[ok] + 50, fixes[ok] / frames[ok], width=90, color=colors["lights"])
        axes[1].set_xlabel("true distance from the base (m)")
        axes[1].set_ylabel("accepted light fixes per rear frame")
        axes[1].set_title("light-fix rate vs distance (inside ~350 m the lights leave the FOV)")
        axes[1].set_ylim(0, 1)
        axes[1].grid(alpha=0.3)
    fig.tight_layout()
    p = out / "heading_and_light_rate.png"
    fig.savefig(p, dpi=130)
    paths.append(p)
    plt.close(fig)
    return paths


def main() -> None:
    from sfhack.config import data_dir

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mission", default="ukraine-1")
    ap.add_argument("--navs", default="imu,flow,lights")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--az", type=float, default=None, help="bearing, deg clockwise from north (default: toward the far map corner)")
    ap.add_argument("--dist", type=float, default=1950.0)
    ap.add_argument("--agl", type=float, default=80.0)
    ap.add_argument("--speed", type=float, default=10.0)
    ap.add_argument("--gnss-off-m", type=float, default=50.0)
    ap.add_argument("--renderer", default="auto", help="thermal camera renderer: auto | full | stub")
    ap.add_argument("--abort-err", type=float, default=3000.0)
    ap.add_argument("--max-t", type=float, default=900.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out) if args.out else data_dir() / "recon" / "nav_eval" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    if (out / "results.json").exists():
        results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    if not args.plot_only:
        az, d_max = default_bearing(args.mission)
        az = args.az if args.az is not None else az
        dist = min(args.dist, d_max - 30.0) if args.az is None else args.dist
        n_out = int(math.ceil(dist / 50.0))
        turn_idx = 1 + n_out  # first orbit waypoint (TAKEOFF = 0, out legs 1..n_out)
        back_idx = turn_idx + 13  # after the 13 orbit points + the far point
        jobs = [{"mission": args.mission, "nav": nav, "seed": s, "az": round(az, 2), "dist": dist, "agl": args.agl,
                 "speed": args.speed, "gnss_off_m": args.gnss_off_m, "renderer": args.renderer, "abort_err": args.abort_err,
                 "max_t": args.max_t, "turn_idx": turn_idx, "back_idx": back_idx}
                for nav in args.navs.split(",") for s in range(args.seed0, args.seed0 + args.seeds)]
        done = {(r["nav"], r["seed"]) for r in results}
        jobs = [j for j in jobs if (j["nav"], j["seed"]) not in done]
        jobs.sort(key=lambda j: {"lights": 0, "flow": 1, "imu": 2}[j["nav"]])
        print(f"{len(jobs)} flights on {args.workers} workers, az {az:.1f} deg, out {dist:.0f} m -> {out}", flush=True)
        t0 = time.time()
        with ProcessPoolExecutor(args.workers) as ex:
            futs = {ex.submit(fly, j): j for j in jobs}
            for f in as_completed(futs):
                j = futs[f]
                try:
                    r = f.result()
                except Exception as e:  # noqa: BLE001 - one broken flight must not lose the others
                    print(f"FAILED {j['nav']} seed {j['seed']}: {type(e).__name__}: {e}", flush=True)
                    continue
                results.append(r)
                tn = r["turn"] or {}
                print(f"[{len(results):3d} {time.time() - t0:6.0f}s] {r['nav']:6s} seed {r['seed']:2d}: turn err "
                      f"{tn.get('err', float('nan')):8.1f} m (along {tn.get('along', float('nan')):7.1f} cross "
                      f"{tn.get('cross', float('nan')):7.1f}) land {r['land_err']:7.1f} m {r['outcome']} "
                      f"compass bias {r['mag_bias_true_deg']:+.2f} est {r['mag_bias_est_deg']:+.2f} {r['counters']} "
                      f"wall {r['wall_s']:.0f}s", flush=True)
                (out / "results.json").write_text(json.dumps(results), encoding="utf-8")
    md = table(results)
    (out / "table.md").write_text(md + "\n", encoding="utf-8")
    print("\n" + md)
    for p in plot(results, out):
        print(f"plot: {p}")
    print(f"results: {out / 'results.json'}")


if __name__ == "__main__":
    main()
