"""The uncertainty table: RMS, 95th percentile, and an outcome. One fixed seed, not a sweep."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demos"))
from uncertainty_metrics import fly_case, horizontal_stats  # noqa: E402


def test_horizontal_stats_rms_and_percentile() -> None:
    rms, p95 = horizontal_stats(np.array([0.0, 0.0, 0.0, 10.0]))
    assert rms == np.sqrt(25.0)
    assert p95 == np.percentile([0.0, 0.0, 0.0, 10.0], 95)


def test_nominal_claim_covers_cruise_p95() -> None:
    """Honest seed 0: the reported 2σ stays at or above the cruise error's 95th percentile."""
    from sfhack.degradation import Conditions
    from sfhack.runner import FlightSystem

    fs = FlightSystem(pose_source="ekf", localizer="vision", seed=0, conditions=Conditions().update("gnss=ok"))
    sig, err = [], []
    next_log = 20.0
    try:
        while fs.t < 70.0 and not fs.done:
            fs.step()
            if fs.t + 1e-9 < next_log:
                continue
            next_log = fs.t + 0.1
            if fs.mission.phase != "CRUISE":
                continue
            est, tr = fs.autopilot.pose(), fs.truth.state().pos
            sig.append(est.sigma2_horizontal())
            err.append(float(np.linalg.norm(est.pos[:2] - tr[:2])))
    finally:
        fs.close()
    assert len(err) > 80  # a real stretch of cruise, not a handful of samples
    p95 = float(np.percentile(err, 95))
    covered = float(np.mean(np.asarray(sig) >= p95 - 1e-9))
    assert covered >= 0.8
    assert float(np.median(sig)) < 5.0  # still a tight claim, not a spoof-sized bubble


def test_drag_increases_horizontal_error() -> None:
    """Seed 0, vision, a short cruise. Drag walks the estimate; the row carries RMS, p95, outcome."""
    nominal = fly_case("nominal", "gnss=ok", "", seed=0, localizer="vision", max_t=18.0, window_t=8.0)
    drag = fly_case("gnss_drag", "gnss=ok", "8:gnss=drag:1.5@90", seed=0, localizer="vision",
                    max_t=18.0, window_t=8.0)
    for row in (nominal, drag):
        assert row["n"] > 20
        assert np.isfinite(row["rms"]) and np.isfinite(row["p95"])
        assert row["p95"] >= 0.0 and row["rms"] >= 0.0
        assert row["outcome"] in {"ARRIVED", "MISSED", "MISSION_TIMEOUT"}
    assert drag["rms"] > nominal["rms"] + 1.0
