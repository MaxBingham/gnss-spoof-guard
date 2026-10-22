"""Fast end-to-end smoke tests (a few seconds of sim time each)."""

from __future__ import annotations

from sfhack.runner import FlightSystem


def test_arena_truth_mission_lands_at_b() -> None:
    fs = FlightSystem(world_dir=None, pose_source="truth", localizer="off")
    r = fs.run(max_t=40.0)
    fs.close()
    assert r.success, r.summary()


def test_berkeley_ekf_loop_takes_off_and_gets_fixes() -> None:
    fs = FlightSystem(pose_source="ekf", localizer="mock")
    fs.run(max_t=25.0)
    ap = fs.autopilot
    err = float(((fs.truth.state().pos[:2] - ap.pose().pos[:2]) ** 2).sum() ** 0.5)
    fs.close()
    assert fs.truth.state().pos[2] > 30.0
    assert ap.loc_fixes > 0
    assert err < 10.0


def test_berkeley_vision_mission_lands_at_b() -> None:
    """Full default mission on the EKF with the VISION localizer (no truth in the loop) lands within 3 m
    of B, and touchdown comes from the rangefinder."""
    fs = FlightSystem(pose_source="ekf", localizer="vision", seed=0)
    r = fs.run()
    decisions = [text for _, text in fs.autopilot.decisions]
    fs.close()
    assert r.success, r.summary()
    assert any(d.startswith("TOUCHDOWN (rangefinder") for d in decisions), decisions
