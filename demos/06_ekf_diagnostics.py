"""Demo 06: dead-reckoning diagnostics. The controller flies on TRUTH (TruthPoseSource), the
localizer is OFF, and the EKF runs on IMU + baro + compass only. Logs truth vs estimate and
plots the error over time to $SFHACK_DATA/diagnostics/.

    python demos/06_ekf_diagnostics.py                 # default Berkeley mission, seed 0
    python demos/06_ekf_diagnostics.py --seed 3 --t 90
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from sfhack.config import SAMPLE_WORLD_DIR, data_dir
from sfhack.runner import FlightSystem


def run(seed: int = 0, t_max: float = 90.0, world: str | None = None) -> dict[str, np.ndarray]:
    """Fly t_max seconds of the default mission on truth, EKF without visual fixes. Returns logs."""
    fs = FlightSystem(world_dir=world or SAMPLE_WORLD_DIR, pose_source="truth", localizer="off", seed=seed)
    log: dict[str, list] = {k: [] for k in ("t", "truth", "est", "sigma2", "att_err_deg", "acc_w")}
    ap = fs.autopilot
    next_log = 0.0
    while fs.t < t_max and not fs.done:
        fs.step()
        if fs.t >= next_log:
            next_log = fs.t + 0.1
            tr, est = fs.truth.state(), ap.pose()
            dR = tr.R.T @ ap.ahrs.R
            log["t"].append(fs.t)
            log["truth"].append(tr.pos)
            log["est"].append(est.pos)
            log["sigma2"].append(est.sigma2_horizontal())
            log["att_err_deg"].append(math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2)))))
            log["acc_w"].append(ap.last_acc_world.copy())
    fs.close()
    out = {k: np.array(v) for k, v in log.items()}
    out["takeoff_t"] = np.array(fs.cfg.mission.preflight_s)
    return out


def horizontal_error(log: dict[str, np.ndarray]) -> np.ndarray:
    return np.linalg.norm(log["truth"][:, :2] - log["est"][:, :2], axis=1)


def error_at(log: dict[str, np.ndarray], flight_s: float) -> float:
    """Horizontal error flight_s seconds after takeoff (end of preflight)."""
    t0 = float(log["takeoff_t"])
    i = int(np.searchsorted(log["t"], t0 + flight_s))
    return float(horizontal_error(log)[min(i, len(log["t"]) - 1)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t", type=float, default=90.0, help="sim seconds to fly")
    ap.add_argument("--headless", action="store_true", help="accepted for run_demo symmetry; always headless")
    args = ap.parse_args()

    log = run(args.seed, args.t)
    err = horizontal_error(log)
    hover = (log["t"] > 0.5) & (log["t"] < float(log["takeoff_t"]))
    print(f"preflight |acc_world| mean {np.linalg.norm(log['acc_w'][hover], axis=1).mean():.3f} m/s^2 (should be ~0)")
    print(f"attitude error mean {log['att_err_deg'].mean():.2f} deg, max {log['att_err_deg'].max():.2f} deg")
    for s in (30, 60, 80):
        print(f"horizontal drift {s:3d} s after takeoff: {error_at(log, s):7.2f} m")
    inside = np.mean(err <= log["sigma2"])
    print(f"truth inside 2-sigma radius {100 * inside:.0f} % of the time")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = data_dir() / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axs[0].plot(log["t"], err, label="|est - truth| horizontal")
    axs[0].plot(log["t"], log["sigma2"], "--", label="2-sigma radius")
    axs[0].set_ylabel("m"); axs[0].legend(); axs[0].grid(alpha=0.3)
    axs[1].plot(log["t"], log["att_err_deg"]); axs[1].set_ylabel("AHRS attitude error [deg]"); axs[1].grid(alpha=0.3)
    axs[2].plot(log["t"], log["acc_w"]); axs[2].set_ylabel("EKF world accel [m/s^2]"); axs[2].grid(alpha=0.3)
    axs[2].set_xlabel("sim time [s]")
    fig.suptitle(f"Dead reckoning (no visual fixes), seed {args.seed}")
    path = out / f"ekf_dead_reckoning_seed{args.seed}.png"
    fig.savefig(path, dpi=110)
    print(f"plot: {path}")


if __name__ == "__main__":
    main()
