"""V4 detector scene randomizer (simulator side): library layouts, held-out splits, auto-labels, hidden pad."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sfhack.config import data_dir
from sfhack.sim import detector_scene as ds

lib = None
if ds.OBJECTS_DIR.joinpath("catalog.json").exists():
    lib = ds.load_library()
needs_lib = pytest.mark.skipif(lib is None, reason=f"object library not found at {ds.OBJECTS_DIR} (SFHACK_OBJECTS)")


@needs_lib
@pytest.mark.parametrize("split", ds.SPLITS)
def test_layouts_complete_seeded_inside_and_held_out(split: str) -> None:
    for seed in range(40):
        lay = ds.sample_layout(lib, seed, split)
        again = ds.sample_layout(lib, seed, split)
        assert lay.to_json() == again.to_json()
        roles = [o.role for o in lay.objects]
        for r in ("target", "brown_car", "white_car", "tent", "hut", "tree"):
            assert roles.count(r) == 1, (seed, roles)
        assert sum(o.key == "truck" for o in lay.objects) == 1
        assert roles.count("distractor") <= 3
        for o in lay.objects:
            if o.role not in ("background", "occluder"):  # trees, may stand just outside
                assert math.hypot(*o.xy) <= ds.AREA_R, (o.key, o.xy)
            held = ds.HELDOUT.get(o.key, ())
            if split in ("train", "val"):
                assert o.variant not in held
            elif o.key in ds.TEST_ONLY_HELDOUT:
                assert o.variant in held
        # min spacing between footprint circles of the placed (non-occluder) objects
        placed = [o for o in lay.objects if o.role != "occluder"]
        for i, a in enumerate(placed):
            for b in placed[i + 1:]:
                d = math.hypot(a.xy[0] - b.xy[0], a.xy[1] - b.xy[1])
                assert d >= ds.footprint_radius(lib[a.key][a.variant]) + ds.footprint_radius(lib[b.key][b.variant]) + 0.5
        # occluder trunks stay off every footprint
        for o in lay.objects:
            if o.role == "occluder":
                for v in lay.objects:
                    if v.key in ("truck", "car", "pickup", "suv"):
                        assert math.hypot(o.xy[0] - v.xy[0], o.xy[1] - v.xy[1]) > 0.5 * lib[v.key][v.variant]["size_m"][1]


@needs_lib
def test_brown_shared_between_truck_and_brown_car() -> None:
    """Brown car / pickup draw from the truck's color distribution: color alone cannot pick the truck."""
    truck, car = [], []
    for seed in range(60):
        lay = ds.sample_layout(lib, seed, "train")
        truck.append(next(iter(lay.truck.rgba.values()))[:3])
        car.append(next(o for o in lay.objects if o.role == "brown_car").rgba["body"][:3])
    t, c = np.array(truck), np.array(car)
    assert np.all(np.abs(t.mean(0) - c.mean(0)) < 0.02)
    assert t[:, 0].min() < c[:, 0].max() and c[:, 0].min() < t[:, 0].max()


@pytest.fixture(scope="module")
def scene():
    if lib is None:
        pytest.skip("object library not found")
    sc = ds.DetectorScene(lib, ds.ensure_ground_textures(data_dir() / "v4" / "detector"))
    yield sc
    sc.close()


def _nadir_over_truck(lay: ds.Layout, alt: float = 15.0) -> ds.View:
    t = lay.truck
    return ds.View(np.array([t.xy[0], t.xy[1], alt]), ds.body_attitude(0.0, 0.0, 0.0), 0.0, 0.5, 1.2, 0.8, 0.3, 0,
                   (0.0, 0.0), 0.0, "clear", 0.0, 0.0, 0.0, 0.5)


@needs_lib
def test_render_labels_truck_and_hidden_pad(scene) -> None:
    lay = ds.sample_layout(lib, 3, "val")
    view = _nadir_over_truck(lay)
    out = scene.render(lay, view, np.random.default_rng(0), keep_seg=True)
    assert out.rgb.shape == (ds.H, ds.W, 3)
    trucks = [lab for lab in out.labels if lab.key == "truck"]
    assert len(trucks) == 1 and trucks[0].cls == ds.CLASSES.index("truck")
    x0, y0, x1, y1 = trucks[0].box
    assert 0 <= x0 < x1 <= ds.W and 0 <= y0 < y1 <= ds.H
    assert (x0 - 2) <= ds.W / 2 <= (x1 + 2) and (y0 - 2) <= ds.H / 2 <= (y1 + 2)  # truck under the camera
    for lab in out.labels:
        assert lab.visible_frac >= 0.2 and min(lab.box[2] - lab.box[0], lab.box[3] - lab.box[1]) >= 8
    # Pad + marker: dark marker cells around the image center; hidden pad -> they vanish, the box stays the truck
    c = out.rgb[ds.H // 2 - 25:ds.H // 2 + 25, ds.W // 2 - 25:ds.W // 2 + 25].astype(float).mean(2)
    lay.pad = False
    out2 = scene.render(lay, view, np.random.default_rng(0))
    c2 = out2.rgb[ds.H // 2 - 25:ds.H // 2 + 25, ds.W // 2 - 25:ds.W // 2 + 25].astype(float).mean(2)
    assert c.std() > 2.0 * c2.std()
    assert [lab.key for lab in out2.labels if lab.key == "truck"] == ["truck"]


@needs_lib
def test_tree_occlusion_is_measured(scene) -> None:
    """A crown over a vehicle lowers its visible fraction; > 80 % hidden objects are dropped, not labeled."""
    seen = False
    for seed in range(40):
        lay = ds.sample_layout(lib, seed, "train")
        occ = [o for o in lay.objects if o.role == "occluder"]
        if not occ:
            continue
        rng = np.random.default_rng(seed)
        view = _nadir_over_truck(lay, 30.0)
        out = scene.render(lay, view, rng)
        for lab in out.labels + out.dropped:
            if lab.key != "tree" and lab.visible_frac < 0.95:
                seen = True
        for lab in out.dropped:
            assert lab.visible_frac < 0.2 or min(lab.box[2] - lab.box[0], lab.box[3] - lab.box[1]) < 8
        if seen:
            break
    assert seen
