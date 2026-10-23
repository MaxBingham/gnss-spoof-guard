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


# ---------------------------------------------------------------------------------------- TRN


def test_onboard_dsm_has_the_campus_buildings() -> None:
    from sfhack.config import SAMPLE_WORLD_DIR
    from sfhack.world.onboard_dsm import build_dsm
    from sfhack.world.onboard_map import OnboardMap

    om = OnboardMap(SAMPLE_WORLD_DIR)
    dsm = build_dsm(om)
    campanile = next(lm for lm in om.landmarks if lm["type"] == "tower")
    assert float(dsm.sample(np.asarray(campanile["centroid_enu"]))) > 80.0
    assert float(dsm.sample(om.a_enu)) < 1.0 and 0.2 < float((dsm.z > 0.5).mean()) < 0.6


def test_trn_converges_on_a_building_profile() -> None:
    """Ideal lidar ranges from the DSM along the Berkeley route: from a 6 m-off prior the cloud converges near
    the true track and publishes fixes (the filter itself; closed-loop numbers are in MERGE_NOTES_V32.md)."""
    from sfhack.config import SAMPLE_WORLD_DIR
    from sfhack.messages import LidarMsg, PoseEstimate
    from sfhack.perception.trn import TrnFilter, beam_dirs_body
    from sfhack.world.onboard_dsm import build_dsm
    from sfhack.world.onboard_map import OnboardMap

    om = OnboardMap(SAMPLE_WORLD_DIR)
    cfg = default_config()
    dsm = build_dsm(om)
    trn = TrnFilter(cfg.trn, dsm, seed=0)
    angles = np.array([-0.14, -0.10, -0.06, -0.02, 0.02, 0.06, 0.10, 0.14])
    dirs = beam_dirs_body(angles)
    vel = np.array([0.0, 8.0])
    true = np.array([-15.0, -230.0])
    offset = np.array([6.0, -3.0])  # the INS prior is off by this
    fixes = []
    for k in range(150):  # 15 s at 10 Hz = 120 m over the southern campus buildings
        t = k * 0.1
        true = true + vel * 0.1
        r = np.array([(40.0 - float(dsm.sample(true + 40.0 * d[:2] / -d[2]))) / -d[2] for d in dirs])
        cov = np.diag([9.0, 9.0, 1.0, 0.04, 0.04, 0.04])
        pose = PoseEstimate(t, np.array([*(true + offset), 40.0]), np.array([*vel, 0.0]), 0.0, cov, np.eye(3),
                            np.zeros(3), "ekf")
        fix = trn.step(LidarMsg(t, angles, r), pose, 0.0)
        if fix is not None:
            fixes.append(np.linalg.norm(fix.pos - true))
    assert fixes, trn.status
    assert np.median(fixes) < 2.5, fixes
