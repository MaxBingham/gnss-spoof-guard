"""TRUTH SEPARATION: onboard code (estimation, control, planning, perception, world, autopilots) must never
reach the true state. Only the modules listed in sfhack/sim/truth.py may import it.

V3 adds a second kind of truth: the world package's truth/ part (the simulated reality: true terrain,
true building heights, world texture). Onboard code may read only manifest.json and onboard*/ - checked
here on the source, and on the data: the onboard DSM is never the truth array."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1] / "sfhack"
ALLOWED = {
    "perception/mock_localizer.py",  # SIMULATION OF PERCEPTION - uses ground truth
    "runner.py", "server.py", "recorder.py",  # harness: display / scoring only
    "nav_runner.py",  # V3 harness: display / scoring only
}
SIM_SIDE = {  # build or display the simulated world itself; never imported by onboard code
    "world/scene.py",  # builds the MuJoCo world
    "world/build_v3.py",  # offline world builder: writes truth/ and onboard/
}
TRUTH_PATTERNS = [  # imports of the simulator / truth, and calls of its private truth accessors
    r"^\s*(from|import)\s+[\w.]*\bsim\b",
    r"^\s*import\s+mujoco\b",
    r"\bTruthProvider\(", r"\._truth_state\(", r"\._truth_camera_pose\(",
]
TRUTH_DATA_PATTERNS = [  # V3: data access to the truth half of a world package (code, not prose)
    r"[\"']truth/",  # a path string into truth/
    r"/\s*[\"']truth[\"']",  # Path(...) / "truth"
    r"manifest\w*\[[\"']truth[\"']\]",  # manifest["truth"]
    r"\bTerrainWorld\(",  # the simulator-side world
    r"^\s*(from|import)\s+[\w.]*build_v3",  # the builder (writes truth/)
]
ONBOARD = [f for f in PKG.rglob("*.py")
           if not (f.relative_to(PKG).as_posix().startswith("sim/") or f.relative_to(PKG).as_posix() in ALLOWED
                   or f.relative_to(PKG).as_posix() in SIM_SIDE)]


def test_onboard_code_never_touches_truth() -> None:
    offenders = []
    for f in ONBOARD:
        src = f.read_text(encoding="utf-8")
        hits = [p for p in TRUTH_PATTERNS if re.search(p, src, re.MULTILINE)]
        if hits:
            offenders.append(f"{f.relative_to(PKG).as_posix()}: {hits}")
    assert not offenders, "truth access from onboard code:\n" + "\n".join(offenders)


def test_onboard_code_never_reads_world_truth_data() -> None:
    offenders = []
    for f in ONBOARD:
        src = f.read_text(encoding="utf-8")
        hits = [p for p in TRUTH_DATA_PATTERNS if re.search(p, src)]
        if hits:
            offenders.append(f"{f.relative_to(PKG).as_posix()}: {hits}")
    assert not offenders, "onboard code references the truth part of the world package:\n" + "\n".join(offenders)


def test_v3_onboard_modules_are_checked() -> None:
    names = {f.relative_to(PKG).as_posix() for f in ONBOARD}
    for m in ("perception/trn.py", "perception/ibn.py", "estimation/nav_ekf.py", "estimation/integrity.py",
              "planning/route_planner.py", "planning/route_mission.py", "planning/nav_health.py",
              "world/onboard_terrain.py", "world/surface.py", "nav_autopilot.py"):
        assert m in names, m


def test_mock_localizer_is_marked() -> None:
    src = (PKG / "perception" / "mock_localizer.py").read_text(encoding="utf-8")
    assert "SIMULATION OF PERCEPTION - uses ground truth" in src


@pytest.mark.parametrize("world", ["s1", "s2"])
def test_onboard_map_is_never_the_truth_array(world: str) -> None:
    from sfhack.world.build_v3 import worlds_dir
    from sfhack.world.surface import Grid

    d = worlds_dir() / world
    if not (d / "manifest.json").exists():
        pytest.skip(f"world {world} not built (python -m sfhack.world.build_v3 {world})")
    truth = Grid.load(d / "truth" / "dsm.npz")
    for variant in ("onboard", "onboard_heavy"):
        ob = Grid.load(d / variant / "dsm.npz")
        assert ob.shape == truth.shape
        diff = np.abs(ob.z - truth.z)
        assert float(np.mean(diff < 1e-3)) < 0.01, f"{variant} DSM is (almost) the truth array"
        assert float(np.sqrt(np.mean(diff**2))) > 0.5
