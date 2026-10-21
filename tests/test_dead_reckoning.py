"""Dead reckoning without visual fixes (IMU + baro + compass only) must stay within bounds.

Regression test for the AHRS tilt bug: with tilt_gain 0.02 the residual gyro bias left ~0.8 deg of
roll/pitch error, which leaked 0.14 m/s^2 of gravity into the EKF (142 m drift after 60 s)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "ekf_diagnostics", Path(__file__).resolve().parents[1] / "demos" / "06_ekf_diagnostics.py")
diag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diag)  # type: ignore[union-attr]


def test_dead_reckoning_drift_bounds() -> None:
    log = diag.run(seed=0, t_max=66.0)
    d30, d60 = diag.error_at(log, 30.0), diag.error_at(log, 60.0)
    assert d30 < 20.0, f"drift after 30 s: {d30:.1f} m"
    assert d60 < 80.0, f"drift after 60 s: {d60:.1f} m"
