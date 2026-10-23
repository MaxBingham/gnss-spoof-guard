"""V3 evaluation (the pitch slide): Monte Carlo over seeds x navigation configurations x scenarios.

    python demos/v3_eval.py --seeds 20                              # s1 + s2, all configs
    python demos/v3_eval.py --worlds s1 --configs ins,ins_trn --seeds 5 --tag quick

Per flight (truth-scored, harness): final error at B, arrival error (above B, before the precision landing),
max 2-sigma, success (S1: landed within 3 m of B; S2: arrived within 10 m of B before the precision landing),
time in DEGRADED/HOLD, source exclusions. Results: $SFHACK_DATA/v3/eval/<tag>/ (runs.jsonl, table.md, *.png).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

# One thread per flight: 18 workers x (BLAS + OpenCV pools over all cores) ran ~10x slower than one flight alone.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

CONFIG_ORDER = ["ins", "ins_trn", "ins_ibn", "ins_trn_ibn", "mismatch_heavy", "trn_off30", "ibn_off30"]
LABEL = {"ins": "INS only", "ins_trn": "INS+TRN", "ins_ibn": "INS+IBN", "ins_trn_ibn": "INS+TRN+IBN",
         "mismatch_heavy": "all, map mismatch heavy", "trn_off30": "INS+TRN, TRN off 30 s",
         "ibn_off30": "IBN off 30 s"}


def fly(job: tuple[str, str, int]) -> dict:
    world, config, seed = job
    import cv2

    from sfhack.nav_runner import NavFlightSystem

    cv2.setNumThreads(1)

    fs = NavFlightSystem(world, config, seed)
    try:
        r = fs.run()
    finally:
        fs.close()
    d = asdict(r)
    d["decisions"] = [(round(t, 1), x) for t, x in r.decisions if not x.startswith(("candidate", "UPDATE_ZONE"))]
    return d


def summarize(runs: list[dict], world: str, config: str) -> dict | None:
    rs = [r for r in runs if r["world"] == world and r["config"] == config]
    if not rs:
        return None
    fe = np.array([r["final_error"] for r in rs])
    ae = np.array([r["arrival_error"] if not math.isnan(r["arrival_error"]) else np.inf for r in rs])
    return {
        "world": world, "config": config, "n": len(rs),
        "success": sum(r["success"] for r in rs),
        "final_median": float(np.median(fe)), "final_p90": float(np.percentile(fe, 90)),
        "arrival_median": float(np.median(ae)), "arrived": int(np.isfinite(ae).sum()),
        "max_sigma2_median": float(np.median([r["max_sigma2"] for r in rs])),
        "degraded_mean_s": float(np.mean([r["time_degraded"] for r in rs])),
        "exclusions": int(sum(r["exclusions"] for r in rs)),
        "crashes": sum(r["outcome"] == "CRASH" for r in rs),
        "emergency": sum(r["outcome"] == "EMERGENCY_LANDED" for r in rs),
        "est_err_mean": float(np.mean([r["est_err_mean"] for r in rs])),
    }


def fmt(x: float) -> str:
    return "inf" if not np.isfinite(x) else (f"{x:.1f}" if x < 100 else f"{x:.0f}")


def table(rows: list[dict]) -> str:
    out = ["| scenario | configuration | success | final error at B median / p90 [m] | arrival error median [m] "
           "| max 2-sigma median [m] | time DEGRADED+HOLD mean [s] | exclusions | crash / emergency land |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['world'].upper()} | {LABEL.get(r['config'], r['config'])} | {r['success']}/{r['n']} | "
                   f"{fmt(r['final_median'])} / {fmt(r['final_p90'])} | {fmt(r['arrival_median'])} | "
                   f"{fmt(r['max_sigma2_median'])} | {r['degraded_mean_s']:.0f} | {r['exclusions']} | "
                   f"{r['crashes']} / {r['emergency']} |")
    return "\n".join(out)


def charts(runs: list[dict], rows: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    worlds = sorted({r["world"] for r in rows})
    fig, axes = plt.subplots(1, len(worlds), figsize=(7 * len(worlds), 5), squeeze=False)
    for ax, w in zip(axes[0], worlds):
        cfgs = [c for c in CONFIG_ORDER if any(r["world"] == w and r["config"] == c for r in rows)]
        data = [np.clip([r["final_error"] for r in runs if r["world"] == w and r["config"] == c], 0.3, 5000)
                for c in cfgs]
        bp = ax.boxplot(data, tick_labels=[LABEL[c].replace(", ", "\n") for c in cfgs], showfliers=True,
                        patch_artist=True)
        colors = ["#888888", "#2a9d8f", "#e9c46a", "#264653", "#e76f51", "#8ab17d", "#b56576"]
        for patch, c in zip(bp["boxes"], cfgs):
            patch.set_facecolor(colors[CONFIG_ORDER.index(c)])
        for i, c in enumerate(cfgs):
            r = next(x for x in rows if x["world"] == w and x["config"] == c)
            ax.text(i + 1, 3500, f"{r['success']}/{r['n']}", ha="center", fontsize=10, fontweight="bold")
        sc_r = 3.0 if w == "s1" else 10.0
        ax.axhline(sc_r, color="green", ls="--", lw=1, label=f"success radius {sc_r:.0f} m")
        ax.set_yscale("log")
        ax.set_ylim(0.3, 6000)
        ax.set_ylabel("final error at B [m] (log)")
        ax.set_title(f"{w.upper()}: landing error at B, {len(data[0]) if data else 0} seeds per configuration")
        ax.grid(True, axis="y", which="both", alpha=0.3)
        ax.legend(loc="lower left")
    fig.suptitle("GNSS disabled - INS backbone + terrain-referenced (TRN) + image-based (IBN) navigation")
    fig.tight_layout()
    fig.savefig(out / "final_error.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(worlds), figsize=(7 * len(worlds), 4.5), squeeze=False)
    for ax, w in zip(axes[0], worlds):
        rs = [r for r in rows if r["world"] == w]
        x = np.arange(len(rs))
        ax.bar(x - 0.2, [r["max_sigma2_median"] for r in rs], 0.4, label="max 2-sigma (median) [m]", color="#264653")
        ax.bar(x + 0.2, [r["degraded_mean_s"] for r in rs], 0.4, label="time DEGRADED/HOLD (mean) [s]", color="#e9c46a")
        ax.set_xticks(x, [LABEL[r["config"]].replace(", ", "\n") for r in rs])
        ax.set_yscale("log")
        ax.set_title(f"{w.upper()}: integrity / health")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "health.png", dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", default="s1,s2")
    ap.add_argument("--configs", default=",".join(CONFIG_ORDER))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--resume", action="store_true", help="keep the flights already in runs.jsonl, fly the rest")
    args = ap.parse_args()
    from sfhack.world.build_v3 import v3_dir

    out = v3_dir() / "eval" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    worlds, configs = args.worlds.split(","), args.configs.split(",")
    jobs = [(w, c, s) for w in worlds for c in configs for s in range(args.seed0, args.seed0 + args.seeds)]
    # long natural-terrain flights first so the pool stays busy
    jobs.sort(key=lambda j: (j[0] == "s1", j[1], j[2]))
    runs = []
    if args.resume and (out / "runs.jsonl").exists():
        runs = [json.loads(line) for line in (out / "runs.jsonl").read_text(encoding="utf-8").splitlines() if line]
        done = {(r["world"], r["config"], r["seed"]) for r in runs}
        jobs = [j for j in jobs if j not in done]
        print(f"resume: {len(runs)} flights kept, {len(jobs)} to fly")
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex, open(out / "runs.jsonl", "a" if args.resume else "w",
                                                       encoding="utf-8") as f:
        futs = {ex.submit(fly, j): j for j in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            j = futs[fut]
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001 - keep the batch going, report the failure
                print(f"[{k}/{len(jobs)}] {j} FAILED: {e!r}")
                continue
            runs.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"[{k}/{len(jobs)} {time.time() - t0:5.0f}s] {r['world']}/{r['config']:15s} seed {r['seed']:2d}: "
                  f"{r['outcome']:16s} {'OK ' if r['success'] else 'FAIL'} final {r['final_error']:7.1f} m "
                  f"arrival {r['arrival_error']:7.1f} m max2s {r['max_sigma2']:6.1f} excl {r['exclusions']}", flush=True)
    rows = [s for w in worlds for c in configs if (s := summarize(runs, w, c)) is not None]
    md = table(rows)
    (out / "table.md").write_text(md + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    charts(runs, rows, out)
    print("\n" + md)
    print(f"\n{len(runs)} flights in {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
