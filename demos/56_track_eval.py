"""Enemy-drone tracker, closed-loop evaluation in the farm world (harness: truth for SCORING only).

    uv run python demos/56_track_eval.py                       # A: 20 seeds natural + B: forced losses, 10 seeds x 6
    uv run python demos/56_track_eval.py --only A --seeds 20   # just the natural flights
    uv run python demos/56_track_eval.py --only B --seeds 10   # just the re-acquisition grid
    uv run python demos/56_track_eval.py --report              # re-aggregate the saved runs

A (natural): enemy heard at a random bearing, scripted slow bearing sweep (someone walking around the UMA-16), birds
over the approach, link ~80 %, base bearings on, GNSS on. Every camera frame is scored against the segmentation:
detection rate vs distance (frames with the enemy visible), false drone detections per minute on birds / on
everything else, time to first detection / to TRACKING, % of time TRACKING (first SEARCH -> end), error of the
reports the base received (TRACKING and LOST reports separately), closest approach.
B (forced losses): 3 times per flight, after >= 10 s of TRACKING, the enemy is hidden from our camera for 8 s (as
behind a tree row) while it turns sharply (steering bearing +-30 deg). Re-acquired = back in TRACKING on the true
enemy (track within 15 m) within 45 s of reappearing. Conditions: base bearing on / off x link 100 / 80 / 0 %.

CIs (95 %): Wilson for proportions (frames are correlated within a flight: per-frame CIs are optimistic; the
per-flight spread is given too), bootstrap over flights for times and errors, exact Poisson for rates.
Output: $SFHACK_DATA/tracker/eval/{runs/*.json, RESULTS.md, detection_vs_distance.png}.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

DEMOS = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMOS))
sys.path.insert(0, str(DEMOS.parent))

from sfhack.config import TrackerConfig, data_dir  # noqa: E402

OUT = data_dir() / "tracker" / "eval"
BINS = [0, 25, 50, 75, 100, 150, 200, 300]
CONDS_B = [("bearing", 1.0), ("bearing", 0.8), ("bearing", 0.0), ("none", 1.0), ("none", 0.8), ("none", 0.0)]

_world = None
_det = None


def _init(weights: str | None) -> None:
    global _world, _det
    import enemy_world as ew
    from sfhack.perception.drone_detector import DroneDetector

    _world = ew.build("farm", n_birds=16, seed=0)
    _det = DroneDetector(TrackerConfig(), weights=weights)


def run_one(job: dict) -> dict:
    from track_scenario import TrackScenario

    m52 = importlib.import_module("52_track_enemy")
    seed, kind = job["seed"], job["kind"]
    rng = np.random.default_rng(seed + 31)
    b0 = float(rng.uniform(0, 360))
    tcfg = TrackerConfig()
    if kind == "A":
        sc = TrackScenario(_world, seed=seed, detector=_det, tcfg=tcfg, enemy_speed=None, availability=0.8,
                           score=True)
        script = m52.BearingScript(b0, seed)
        max_t = job.get("max_t", 260.0)
    else:
        sc = TrackScenario(_world, seed=seed, detector=_det, tcfg=tcfg, enemy_speed=None, availability=job["link"],
                           score=False, send_bearings=job["bearing"] == "bearing", forced_losses=3)
        script = None
        max_t = job.get("max_t", 380.0)
    t0 = time.time()
    sc.hear(b0, "eval")
    while sc.t < max_t and not (sc.pilot.state == "LANDED" and sc.launch_t is not None):
        sc.advance(50)
        if script is not None:
            az = script(sc.t)
            if az is not None:
                sc.hear(az, "script")
        if kind == "B" and len(sc.losses) >= 3 and sc.losses[-1].end_t is not None:
            break
    p = sc.pilot
    err_tr, err_lost = [], []
    for r in sc.received:
        e = sc.enemy_at(r.t)
        if e is None:
            continue
        d = float(np.linalg.norm(r.pos[:2] - e[:2]))
        (err_tr if r.state == "TRACKING" else err_lost).append(d)
    t_search = next((t for t, s in p.decisions if s.startswith("SEARCH")), None)
    out = {"job": job, "b0": b0, "t_end": sc.t, "wall_s": time.time() - t0, "launch_t": sc.launch_t,
           "t_search": t_search, "first_detect_t": p.first_detect_t, "first_track_t": p.first_track_t,
           "state_time": sc.state_time, "min_dist": sc.min_dist, "err_tracking": err_tr, "err_lost": err_lost,
           "reports_made": p.outbox.made, "reports_rx": len(sc.received), "link_avail": sc.link.measured_availability,
           "bearings_sent": sc.bearings_sent, "n_suspect": p.n_suspect, "n_gated": p.n_gated,
           "decisions": p.decisions[-60:], "own_err": sc.own_err()}
    if kind == "A":
        out["frames"] = [{"t": f.t, "s": f.state, "d": round(f.dist, 1), "fov": f.in_fov, "px": f.visible_px,
                          "det": f.detected, "c": round(f.best_conf, 3), "fp": f.fp} for f in sc.frames]
    else:
        out["losses"] = [{"t0": e.t0, "t1": e.t1, "turn": e.turn_deg, "dist0": e.dist0, "went_lost": e.went_lost,
                          "reacq_t": e.reacq_t, "end_t": e.end_t} for e in sc.losses]
    return out


def _work(job: dict) -> str:
    path = OUT / "runs" / f"{job['name']}.json"
    try:
        res = run_one(job)
    except Exception as exc:  # keep the batch going; the report lists the failure
        res = {"job": job, "error": repr(exc)}
    path.write_text(json.dumps(res), encoding="utf-8")
    return f"{job['name']}: " + ("ERROR " + res["error"] if "error" in res else
                                 f"t {res['t_end']:.0f} s, wall {res['wall_s']:.0f} s, first track "
                                 f"{res['first_track_t']}")


# ---------------------------------------------------------------------------------------- statistics

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def boot(x, f=np.mean, n: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    s = np.array([f(x[rng.integers(0, len(x), len(x))]) for _ in range(n)])
    return float(f(x)), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def poisson_ci(k: int, exposure: float) -> tuple[float, float, float]:
    from scipy.stats import chi2

    lo = 0.0 if k == 0 else chi2.ppf(0.025, 2 * k) / 2
    hi = chi2.ppf(0.975, 2 * k + 2) / 2
    return k / exposure, lo / exposure, hi / exposure


def fmt_p(t) -> str:
    return f"{100 * t[0]:.1f} % [{100 * t[1]:.1f}, {100 * t[2]:.1f}]"


def fmt_v(t, unit: str = "", nd: int = 1) -> str:
    return f"{t[0]:.{nd}f}{unit} [{t[1]:.{nd}f}, {t[2]:.{nd}f}]"


def report() -> str:
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((OUT / "runs").glob("*.json"))]
    bad = [r for r in runs if "error" in r]
    A = [r for r in runs if "error" not in r and r["job"]["kind"] == "A"]
    B = [r for r in runs if "error" not in r and r["job"]["kind"] == "B"]
    tc = TrackerConfig()
    md = ["# Enemy-drone tracker - closed-loop evaluation (farm world)", "",
          f"Generated by `demos/56_track_eval.py`. Detector `{tc.weights}`, drone threshold {tc.drone_conf}, "
          f"confirmation {tc.confirm_hits}/{tc.confirm_window} frames, camera {tc.cam_w}x{tc.cam_h} "
          f"{tc.hfov_deg:.0f} deg at {tc.cam_hz:.0f} Hz. 95 % CIs: Wilson (proportions), bootstrap over flights "
          "(means), exact Poisson (rates).", ""]
    if bad:
        md += [f"**{len(bad)} runs failed**: " + ", ".join(f"{r['job']['name']} ({r['error'][:80]})" for r in bad), ""]
    if A:
        frames = [f for r in A for f in r["frames"]]
        md += [f"## A. Natural flights: {len(A)} seeds", "",
               "Enemy heard at a random bearing, 500 m out, 3-5 m/s, scripted bearing sweep (5-30 deg), 16 birds (half "
               "over the approach), link ~80 %, GNSS on.", "",
               "### Detection rate vs true distance (frames with the enemy visible in the camera)", "",
               "| distance m | frames | detected (>= threshold) | rate [95 % CI] | flights with frames |",
               "|---|---|---|---|---|"]
        rows = []
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            fs = [f for f in frames if f["px"] > 0 and lo <= f["d"] < hi]
            k = sum(f["det"] for f in fs)
            nfl = sum(1 for r in A if any(f["px"] > 0 and lo <= f["d"] < hi for f in r["frames"]))
            w = wilson(k, len(fs))
            rows.append((lo, hi, w, len(fs)))
            md.append(f"| {lo}-{hi} | {len(fs)} | {k} | {fmt_p(w) if fs else '-'} | {nfl} |")
        cam_min = sum(len(r["frames"]) for r in A) / tc.cam_hz / 60.0
        fpb = sum(f["fp"].count("bird") for f in frames)
        fpo = sum(f["fp"].count("other") for f in frames)
        md += ["", "### False drone detections (>= threshold, not on the enemy), per minute of camera time", "",
               f"Camera time {cam_min:.1f} min ({len(frames)} frames).", "",
               "| on | detections | per minute [95 % CI] |", "|---|---|---|",
               f"| birds | {fpb} | {fmt_v(poisson_ci(fpb, cam_min), '', 2)} |",
               f"| everything else (trees, buildings, ground, sky) | {fpo} | {fmt_v(poisson_ci(fpo, cam_min), '', 2)} |",
               ""]
        susp = sum(r["n_suspect"] for r in A)
        gated = sum(r["n_gated"] for r in A)
        md += [f"Tracks dropped as SUSPECT by the bearing check: {susp}; new tracks refused by the bearing gate: "
               f"{gated} (detections).", ""]
        fd = [r["first_detect_t"] - r["launch_t"] for r in A if r["first_detect_t"] is not None]
        ft = [r["first_track_t"] - r["launch_t"] for r in A if r["first_track_t"] is not None]
        ntr = sum(r["first_track_t"] is not None for r in A)
        frac = []
        for r in A:
            if r["t_search"] is None:
                continue
            span = r["t_end"] - r["t_search"]
            frac.append(r["state_time"].get("TRACKING", 0.0) / max(span, 1e-6))
        tr_time = sum(r["state_time"].get("TRACKING", 0.0) for r in A)
        span_all = sum(r["t_end"] - r["t_search"] for r in A if r["t_search"] is not None)
        et = [e for r in A for e in r["err_tracking"]]
        el = [e for r in A for e in r["err_lost"]]
        et_fl = [np.mean(r["err_tracking"]) for r in A if r["err_tracking"]]
        md += ["### Mission metrics", "", "| metric | value [95 % CI] | n |", "|---|---|---|",
               f"| flights that reached TRACKING | {fmt_p(wilson(ntr, len(A)))} | {len(A)} |",
               f"| time to first detection after launch, s | {fmt_v(boot(fd), ' s')} | {len(fd)} |",
               f"| time to TRACKING after launch, s | {fmt_v(boot(ft), ' s')} | {len(ft)} |",
               f"| % of time TRACKING (first SEARCH -> end), per flight | {fmt_p(boot(frac))} | {len(frac)} |",
               f"| % of time TRACKING, pooled (Wilson on 1 s slots, optimistic) | "
               f"{fmt_p(wilson(int(tr_time), int(span_all)))} | {int(span_all)} s |",
               f"| report error at the base, TRACKING reports, mean of flight means, m | {fmt_v(boot(et_fl), ' m')} | "
               f"{len(et_fl)} flights |",
               f"| ... all TRACKING reports: mean / median / 95th pct, m | {np.mean(et):.1f} / {np.median(et):.1f} / "
               f"{np.percentile(et, 95):.1f} | {len(et)} |" if et else "| TRACKING reports | - | 0 |",
               f"| ... LOST reports (prediction): mean / median, m | {np.mean(el):.1f} / {np.median(el):.1f} | {len(el)} |"
               if el else "| LOST reports | - | 0 |",
               f"| reports received / made | {sum(r['reports_rx'] for r in A)} / {sum(r['reports_made'] for r in A)} "
               f"| link up {100 * np.mean([r['link_avail'] for r in A]):.0f} % |",
               f"| closest approach to the enemy (true), min over flights / median, m | "
               f"{min(r['min_dist'] for r in A):.1f} / {np.median([r['min_dist'] for r in A]):.1f} | {len(A)} |",
               f"| flights closer than 10 m at any time | {fmt_p(wilson(sum(r['min_dist'] < 10 for r in A), len(A)))} "
               f"| {len(A)} |", ""]
        _plot(rows)
        md += ["![detection vs distance](detection_vs_distance.png)", ""]
    if B:
        md += [f"## B. Forced losses: enemy hidden 8 s (tree-row occlusion) + sharp 30 deg turn, 3 per flight", "",
               "Re-acquired = back in TRACKING with the track within 15 m of the true enemy within 45 s of reappearing. "
               "Time = from reappearing (median and mean, bootstrap CI of the mean).", "",
               "| base bearing | link | flights | losses | re-acquired [95 % CI] | time to re-acquire, median / mean "
               "[95 % CI] s | closest approach min, m |", "|---|---|---|---|---|---|---|"]
        for cb, cl in CONDS_B:
            rs = [r for r in B if r["job"]["bearing"] == cb and abs(r["job"]["link"] - cl) < 1e-6]
            ev = [e for r in rs for e in r["losses"]]
            k = sum(e["reacq_t"] is not None for e in ev)
            tt = [e["reacq_t"] - e["t1"] for e in ev if e["reacq_t"] is not None]
            med = f"{np.median(tt):.1f}" if tt else "-"
            mind = f"{min(r['min_dist'] for r in rs):.1f}" if rs else "-"
            md.append(f"| {'on' if cb == 'bearing' else 'off'} | {100 * cl:.0f} % | {len(rs)} | {len(ev)} | "
                      f"{fmt_p(wilson(k, len(ev))) if ev else '-'} | {med} / {fmt_v(boot(tt), '') if tt else '-'} | "
                      f"{mind} |")
        md.append("")
    text = "\n".join(md)
    (OUT / "RESULTS.md").write_text(text, encoding="utf-8")
    return text


def _plot(rows) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    x = [(lo + hi) / 2 for lo, hi, _w, n in rows if n]
    p = np.array([w for _lo, _hi, w, n in rows if n])
    ax.errorbar(x, 100 * p[:, 0], yerr=[100 * (p[:, 0] - p[:, 1]), 100 * (p[:, 2] - p[:, 0])], fmt="o-", capsize=4)
    ax.set_xlabel("true distance to the enemy [m]")
    ax.set_ylabel("frames detected [%] (Wilson 95 %)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.set_title("Enemy-drone detection rate vs distance (visible frames)")
    fig.tight_layout()
    fig.savefig(OUT / "detection_vs_distance.png", dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=("A", "B"), default=None)
    ap.add_argument("--seeds", type=int, default=None, help="A: 20, B: 10 per condition")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--report", action="store_true", help="only aggregate the saved runs")
    ap.add_argument("--redo", action="store_true", help="re-run jobs that already have a result")
    a = ap.parse_args()
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    if not a.report:
        jobs = []
        if a.only in (None, "A"):
            jobs += [{"kind": "A", "seed": s, "name": f"A_s{s:02d}"} for s in range(a.seeds or 20)]
        if a.only in (None, "B"):
            jobs += [{"kind": "B", "seed": 100 + s, "bearing": cb, "link": cl,
                      "name": f"B_{cb}_{int(100 * cl):03d}_s{s:02d}"}
                     for cb, cl in CONDS_B for s in range(a.seeds or 10)]
        if not a.redo:
            jobs = [j for j in jobs if not (OUT / "runs" / f"{j['name']}.json").exists()]
        print(f"{len(jobs)} runs on {a.workers} workers", flush=True)
        t0 = time.time()
        with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.weights,)) as pool:
            for k, msg in enumerate(pool.imap_unordered(_work, jobs)):
                print(f"[{k + 1}/{len(jobs)} {time.time() - t0:.0f} s] {msg}", flush=True)
    print(report())


if __name__ == "__main__":
    main()
