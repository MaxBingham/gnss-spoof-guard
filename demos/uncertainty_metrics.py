"""Horizontal position error for five uncertainty settings, one fixed seed.

    python demos/uncertainty_metrics.py

Nominal is an honest receiver in clear air. Each other row turns one lever on at
t=20 s and leaves it on: GNSS drag 1.5 m/s east, GNSS jam, fog 0.65, camera lost.
Vision localizer, seed 0, flown to touchdown. RMS and the 95th percentile are
|estimate - truth| horizontal from the moment the lever is on. Not a sweep.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

SEED = 0
LEVER_ON_S = 20.0
# name, initial levers, schedule. One lever each, so a row is that uncertainty.
CASES: tuple[tuple[str, str, str], ...] = (
    ("nominal", "gnss=ok", ""),
    ("gnss_drag", "gnss=ok", "20:gnss=drag:1.5@90"),
    ("jam", "gnss=ok", "20:gnss=jam"),
    ("fog", "gnss=ok", "20:weather=fog:0.65"),
    ("camera_loss", "gnss=ok", "20:camera=lost"),
)


def horizontal_stats(err: np.ndarray) -> tuple[float, float]:
    """RMS and 95th percentile of a horizontal error series (metres)."""
    e = np.asarray(err, float)
    if e.size == 0:
        return float("nan"), float("nan")
    return float(np.sqrt(np.mean(e**2))), float(np.percentile(e, 95))


def fly_case(name: str, init: str, schedule: str, seed: int = SEED, localizer: str = "vision",
             max_t: float | None = None, window_t: float = LEVER_ON_S) -> dict:
    """One flight. Truth is used only to score (harness)."""
    from sfhack.degradation import Conditions
    from sfhack.degradation.cli import parse_schedule
    from sfhack.runner import FlightSystem

    fs = FlightSystem(pose_source="ekf", localizer=localizer, seed=seed,
                      conditions=Conditions().update(init), schedule=parse_schedule(schedule))
    try:
        result = fs.run(max_t=max_t)
        t = np.asarray(fs.log_t, float)
        if t.size:
            err = np.linalg.norm(np.asarray(fs.log_est)[:, :2] - np.asarray(fs.log_truth)[:, :2], axis=1)
            err = err[t >= window_t]
        else:
            err = np.zeros(0)
        rms, p95 = horizontal_stats(err)
        return {"name": name, "rms": rms, "p95": p95, "outcome": result.outcome,
                "final_m": result.final_error, "n": int(err.size)}
    finally:
        fs.close()


def _job(job: tuple) -> dict:
    return fly_case(*job)


def run_cases(seed: int = SEED, localizer: str = "vision", workers: int = 3,
              max_t: float | None = None) -> list[dict]:
    jobs = [(name, init, sched, seed, localizer, max_t, LEVER_ON_S) for name, init, sched in CASES]
    if workers <= 1:
        return [_job(j) for j in jobs]
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(_job, jobs))


def format_table(rows: list[dict]) -> str:
    lines = [f"{'setting':<14} {'RMS m':>8} {'p95 m':>8} {'outcome':<16} {'to B m':>8} {'n':>5}"]
    for r in rows:
        lines.append(f"{r['name']:<14} {r['rms']:8.2f} {r['p95']:8.2f} {r['outcome']:<16} "
                     f"{r['final_m']:8.1f} {r['n']:5d}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-t", type=float, default=0.0, help="stop at this sim time (0 = fly to touchdown)")
    args = ap.parse_args()
    rows = run_cases(seed=args.seed, workers=args.workers, max_t=args.max_t or None)
    print(f"seed {args.seed}, vision localizer, horizontal |est-truth| from t={LEVER_ON_S:.0f} s")
    print(format_table(rows))


if __name__ == "__main__":
    main()
