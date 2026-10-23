"""V4 target scene (simulator side): seeded layouts, difficulty levels, the marker only resolvable low."""

from __future__ import annotations

import math

import cv2
import mujoco
import numpy as np
import pytest

from sfhack.bus import Bus
from sfhack.config import data_dir, default_config
from sfhack.sim.mujoco_sim import MujocoSim
from sfhack.sim.target_world import TargetWorld
from sfhack.world.scene import PackageWorld
from sfhack.world.target_scene import KINDS, ensure_field_package, generate_layout


def _gap(a, b) -> float:
    """Smallest distance between two footprint rectangles (sampled edges)."""
    g = np.linspace(-0.5, 0.5, 41)
    def edge(o):
        L, W = o.length, o.width
        pts = np.concatenate([np.column_stack([g * L, np.full_like(g, W / 2)]), np.column_stack([g * L, np.full_like(g, -W / 2)]),
                              np.column_stack([np.full_like(g, L / 2), g * W]), np.column_stack([np.full_like(g, -L / 2), g * W])])
        return o.to_world(pts)
    pa, pb = edge(a), edge(b)
    return float(np.min(np.linalg.norm(pa[:, None] - pb[None], axis=2)))


@pytest.mark.parametrize("difficulty", ["easy", "hard"])
def test_layouts_are_seeded_complete_and_inside(difficulty: str) -> None:
    keep = [(0.0, 0.0, 5.0)]
    for seed in range(30):
        lay = generate_layout(seed, difficulty, center=(10.0, -5.0), keep_out=keep)
        again = generate_layout(seed, difficulty, center=(10.0, -5.0), keep_out=keep)
        assert [o.xy.tolist() for o in lay.objects] == [o.xy.tolist() for o in again.objects]
        assert sorted(o.kind for o in lay.objects) == sorted(KINDS)
        for o in lay.objects:
            corners = o.footprint()
            assert np.all(np.abs(corners - lay.center) <= lay.half), (seed, o.kind)
            assert math.hypot(*o.xy) >= 5.0 + o.radius - 1e-6, "keep-out (spawn) violated"
        truck = lay.truck
        assert truck.contains(lay.pad_xy) and abs(lay.pad_z - truck.height - 0.06) < 1e-6
        assert 7.5 <= truck.length <= 8.5 and 3.0 <= truck.height <= 3.4
        car = lay.by_kind("brown_car")
        assert 4.0 <= car.length <= 5.0 and car.height <= 1.7
        if difficulty == "hard":
            assert 1.8 <= _gap(truck, car) <= 4.0, (seed, _gap(truck, car))
            tree = lay.by_kind("tree")
            r = tree.length / 2
            local = truck.to_local(tree.xy)
            assert abs(local[1]) - r < truck.width / 2, "crown should hang over the truck"
            assert np.linalg.norm(tree.xy - lay.pad_xy) > r + 0.5 * lay.pad_size, "crown must not cover the pad"
        else:
            others = [o for o in lay.objects if o is not truck]
            assert min(_gap(truck, o) for o in others) > 3.0


def test_brown_shades_overlap_between_car_and_truck() -> None:
    cars = np.array([generate_layout(s).by_kind("brown_car").color for s in range(60)])
    trucks = np.array([generate_layout(s).truck.color for s in range(60)])
    for k in range(3):  # same distribution: each channel's ranges overlap substantially
        lo, hi = max(cars[:, k].min(), trucks[:, k].min()), min(cars[:, k].max(), trucks[:, k].max())
        assert hi - lo > 0.5 * (trucks[:, k].max() - trucks[:, k].min())


def test_marker_resolves_only_at_low_altitude_and_rangefinder_sees_the_roof() -> None:
    cfg = default_config()
    cfg.camera.down_blur_px = 1.2  # the target extension's camera optics
    lay = generate_layout(3, "easy", keep_out=[(0, 0, 5)], marker_dict=cfg.target.marker_dict,
                          marker_id=cfg.target.marker_id, marker_size=cfg.target.marker_size)
    pkg = ensure_field_package(data_dir() / "v4")
    sim = MujocoSim(cfg, Bus(), TargetWorld(PackageWorld(pkg), lay))
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.target.marker_dict)),
                                  cv2.aruco.DetectorParameters())
    m, d = sim.model, sim.data
    seen = {}
    for alt in (10.0, 25.0):
        hits = 0
        for yaw in (0.0, 1.0, 2.0):
            d.qpos[0:3] = [lay.pad_xy[0], lay.pad_xy[1], alt]
            d.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            mujoco.mj_forward(m, d)
            _, ids, _ = det.detectMarkers(cv2.cvtColor(sim.render("down").image, cv2.COLOR_RGB2GRAY))
            hits += int(ids is not None and cfg.target.marker_id in ids.flatten())
        seen[alt] = hits
        rng = float(d.sensor("rangefinder").data[0])
        assert abs(rng - (alt - lay.pad_z)) < 0.1, (alt, rng)
    sim.close()
    assert seen[10.0] == 3 and seen[25.0] == 0, seen
