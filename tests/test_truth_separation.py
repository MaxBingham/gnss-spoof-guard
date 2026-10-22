"""TRUTH SEPARATION: onboard code (estimation, control, planning, perception, world, autopilot)
must never reach the true state. Only the modules listed in sfhack/sim/truth.py may import it."""

from __future__ import annotations

import re
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "sfhack"
ALLOWED = {
    "perception/mock_localizer.py",  # SIMULATION OF PERCEPTION - uses ground truth
    "runner.py", "server.py", "recorder.py",  # harness: display / scoring only
}
TRUTH_PATTERNS = [  # imports of the simulator / truth, and calls of its private truth accessors
    r"^\s*(from|import)\s+[\w.]*\bsim\b",
    r"^\s*import\s+mujoco\b",
    r"\bTruthProvider\(", r"\._truth_state\(", r"\._truth_camera_pose\(",
]


def test_onboard_code_never_touches_truth() -> None:
    offenders = []
    for f in PKG.rglob("*.py"):
        rel = f.relative_to(PKG).as_posix()
        if rel.startswith("sim/") or rel in ALLOWED:
            continue
        if rel.startswith("world/scene.py"):  # builds the MuJoCo world (simulator side), never read onboard
            continue
        src = f.read_text(encoding="utf-8")
        hits = [p for p in TRUTH_PATTERNS if re.search(p, src, re.MULTILINE)]
        if hits:
            offenders.append(f"{rel}: {hits}")
    assert not offenders, "truth access from onboard code:\n" + "\n".join(offenders)


ONBOARD = ("autopilot.py", "estimation/", "control/", "planning/", "perception/", "world/onboard_map.py")
LEVER_PATTERNS = [  # the degradation levers are simulated reality: the drone must not know it is spoofed
    r"\bdegradation\b", r"\bConditions(Msg)?\b", r"\bCONDITIONS\b", r"\.conditions\b",
]


def test_onboard_code_never_reads_degradation_levers() -> None:
    offenders = []
    for f in PKG.rglob("*.py"):
        rel = f.relative_to(PKG).as_posix()
        if not rel.startswith(ONBOARD):
            continue
        src = f.read_text(encoding="utf-8")
        hits = [p for p in LEVER_PATTERNS if re.search(p, src)]
        if hits:
            offenders.append(f"{rel}: {hits}")
    assert not offenders, "degradation levers read by onboard code:\n" + "\n".join(offenders)


def test_mock_localizer_is_marked() -> None:
    src = (PKG / "perception" / "mock_localizer.py").read_text(encoding="utf-8")
    assert "SIMULATION OF PERCEPTION - uses ground truth" in src
