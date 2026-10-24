"""Farm world of demos/50_base_3d.py: flat base, gentle hills, texture lined up with the 3D objects."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demos"))

import base3d_farm  # noqa: E402


def test_terrain_flat_at_base_and_gentle():
    w = base3d_farm.FarmWorld()
    a = np.linspace(0, 2 * np.pi, 64)
    for r in (0.0, 50.0, 150.0, 200.0):
        assert np.all(np.abs(w.ground(r * np.cos(a), r * np.sin(a))) < 1e-9)
    assert w.ground(18.0, -10.0) == 0.0          # the pad
    assert 8.0 < np.ptp(w.Z) < 20.0


def test_texture_lines_up_with_3d(tmp_path):
    """Marker painted into the hfield texture vs a 3D post at the same world point, rendered top-down."""
    assert base3d_farm.check_texture(tmp_path / "check.png") < 3.0
