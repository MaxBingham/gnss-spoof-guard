"""V3.2 GNSS -> GNSS-free transition: bias-step cap, handover to the GNSS-free track, fallback inflation."""

from __future__ import annotations

import numpy as np

from sfhack.config import default_config
from sfhack.degradation import Conditions
from sfhack.degradation.cli import parse_schedule
from sfhack.estimation.ekf import Ekf
from sfhack.messages import VisualFix
from sfhack.runner import FlightSystem


def _ekf_in_flight() -> Ekf:
    cfg = default_config().ekf
    ekf = Ekf(cfg, np.zeros(3))
    for k in range(200):  # 1 s of hover with GNSS: tight covariance, like the naive filter before a jam
        ekf.predict(np.zeros(3), 0.005, k * 0.005)
        if k % 40 == 0:
            ekf.update_gnss(k * 0.005, np.zeros(2), np.eye(2) * 0.65)
    return ekf


def test_one_gnss_jump_cannot_tilt_the_ahrs() -> None:
    """A 50 m jump fused naively moves the position, but the bias states (= AHRS tilt) only by the cap."""
    ekf = _ekf_in_flight()
    ekf.update_gnss(1.0, np.array([50.0, 0.0]), np.eye(2) * 0.65)
    assert ekf.x[0] > 5.0  # the position moves ...
    assert np.hypot(*ekf.x[6:8]) <= ekf.cfg.acc_bias_step_max + 1e-9  # ... the bias only by the cap


def test_gnss_loss_without_free_track_inflates_and_needs_a_strong_fix() -> None:
    ekf = _ekf_in_flight()
    t = ekf.last_gnss_t + ekf.cfg.gnss_loss_after_s + 0.01
    assert ekf.gnss_lost_now(t) and not ekf.gnss_lost_now(t + 1.0)  # fires once
    ekf.inflate_for_gnss_loss()
    assert ekf.P[0, 0] >= ekf.cfg.gnss_loss_pos_std**2 - 1e-9
    weak = VisualFix(t, np.array([3.0, 0.0]), np.eye(2), "vision", quality=0.55)
    strong = VisualFix(t, np.array([3.0, 0.0]), np.eye(2), "vision", quality=0.8)
    assert not ekf.update_fix(weak)
    assert ekf.update_fix(strong) and ekf.last_decision.startswith("RELOCALIZE")
    assert not ekf.relocalizing


def test_jam_after_gnss_jump_hands_over_and_lands_at_b() -> None:
    """The live viewer's G key goes ok -> drag -> jump -> jam. The naive main filter swallows the 50 m jump;
    on the jam the autopilot hands over to its GNSS-free vision track and still lands within 3 m of B
    (before V3.2: 5 deg AHRS tilt from the jump, km-level divergence, emergency landing)."""
    fs = FlightSystem(localizer="vision", seed=0, conditions=Conditions().update("gnss=ok"),
                      schedule=parse_schedule("40:gnss=drag;42:gnss=jump;44:gnss=jam"))
    r = fs.run()
    decisions = [text for _, text in fs.autopilot.decisions]
    fs.close()
    assert any(d.startswith("HANDOVER") for d in decisions), decisions
    assert r.success, r.summary()
