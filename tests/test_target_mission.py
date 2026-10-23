"""V4 closed loop: one hard layout, the drone picks the brown truck and lands on its marker pad."""

from __future__ import annotations

from sfhack.sim.target_world import build_target_flight


def test_hard_layout_lands_on_the_pad() -> None:
    fs, ext = build_target_flight(3, "hard", "gnss")
    fs.run(max_t=200)
    s = ext.score(fs)
    fs.close()
    texts = [txt for _, txt in fs.autopilot.decisions]
    assert any(t.startswith("TARGET: candidate") and "brown truck" in t for t in texts), texts
    assert any(t.startswith("MARKER_CONFIRMED") for t in texts)
    assert s.chosen_kind == "brown_truck" and s.on_pad and s.pad_error < 0.6, s.summary()
