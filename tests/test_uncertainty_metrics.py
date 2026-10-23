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
