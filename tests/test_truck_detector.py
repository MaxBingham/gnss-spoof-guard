"""V4 learned truck detector (onboard): truth separation, ground projection, selector, runtime on a rendered frame.

Truth separation: ground-truth labels exist only in offline data generation (sim/detector_scene.py,
demos/13). At runtime the detector gets RGB + altitude + attitude estimates and nothing else.
"""

from __future__ import annotations

import ast
import inspect
import math
import time
from pathlib import Path

import numpy as np
import pytest

from sfhack.config import DetectorConfig, data_dir
from sfhack.perception import truck_detector as td

SRC = Path(td.__file__)


def test_runtime_detector_never_sees_truth() -> None:
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(("." * node.level) + (node.module or ""))
    banned = ("sim", "mujoco", "detector_scene", "degradation", "truth", "target_scene", "target_world")
    hits = [m for m in imported for b in banned if b in m.split(".") or m.endswith(b)]
    assert not hits, f"truck_detector imports simulator / truth modules: {hits}"
    # Only the runtime inputs: an RGB frame, the altitude estimate, the attitude estimate
    assert list(inspect.signature(td.detect).parameters) == ["rgb", "altitude", "attitude"]
    assert list(inspect.signature(td.TruckDetector.detect).parameters) == ["self", "rgb", "altitude", "attitude"]
    src = SRC.read_text(encoding="utf-8")
    for word in ("labels/", "segmentation", "Layout", "meta/", ".jsonl"):
        assert word not in src, f"runtime detector references offline label data ({word!r})"


def test_ground_offsets_nadir_and_tilted() -> None:
    R = np.eye(3)  # level, heading east: image top = east (body x), image right = south (body -y)
    w, h, fovy = 640, 480, 70.0
    fy = (h / 2) / math.tan(math.radians(fovy) / 2)
    g = td.ground_offsets(np.array([[320, 240], [320, 240 - fy], [320 + fy, 240]]), np.zeros(3), w, h, fovy, 30.0, R)
    assert np.allclose(g[0], [0, 0], atol=1e-9)
    assert np.allclose(g[1], [30, 0], atol=1e-6)  # one focal length up = 45 deg forward
    assert np.allclose(g[2], [0, -30], atol=1e-6)  # right = south
    # an object 3 m tall projects at its half height: the image-center ray still hits right below
    assert np.allclose(td.ground_offsets(np.array([[320, 240]]), np.array([1.5]), w, h, fovy, 30.0, R), 0, atol=1e-9)
    # 20 deg nose-down pitch: the image center looks forward by h * tan(20 deg)
    a = math.radians(20)
    Rp = np.array([[math.cos(a), 0, -math.sin(a)], [0, 1, 0], [math.sin(a), 0, math.cos(a)]]).T  # pitch about body y
    gc = td.ground_offsets(np.array([[320, 240]]), np.zeros(1), w, h, fovy, 30.0, Rp)[0]
    assert abs(abs(gc[0]) - 30 * math.tan(a)) < 1e-6 and abs(gc[1]) < 1e-9


def test_calibration_monotone_and_identity() -> None:
    p = np.linspace(0.01, 0.99, 50)
    assert np.allclose(td.calibrate(p, 1.0), p)
    q = td.calibrate(p, 2.0)
    assert np.all(np.diff(q) > 0) and q[0] > p[0] and q[-1] < p[-1]


def test_selector_margin_rule() -> None:
    cfg = DetectorConfig()
    sel = td.TruckSelector(cfg)
    for k in range(10):
        dets = [td.Detection("truck", 0.85, (0, 0, 1, 1), np.array([5.0, 1.0]) + 0.2 * np.sin(k)),
                td.Detection("car", 0.8, (0, 0, 1, 1), np.array([-6.0, 3.0])),
                td.Detection("truck", 0.3 if k % 3 == 0 else 0.0, (0, 0, 1, 1), np.array([-6.2, 3.1]))]
        sel.add([d for d in dets if d.conf > 0], drone_xy=np.array([100.0, 50.0]))
    s = sel.decide()
    assert s.xy is not None and np.linalg.norm(s.xy - [105, 51]) < 0.5, s
    # two equally truck-like objects -> no decision
    sel = td.TruckSelector(cfg)
    for _ in range(10):
        sel.add([td.Detection("truck", 0.7, (0, 0, 1, 1), np.array([5.0, 0.0])),
                 td.Detection("truck", 0.68, (0, 0, 1, 1), np.array([-5.0, 0.0]))], np.zeros(2))
    assert sel.decide().xy is None


WEIGHTS = data_dir() / DetectorConfig().weights


@pytest.mark.skipif(not WEIGHTS.exists(), reason="no trained weights (demos/14_det_train.py)")
def test_detector_on_rendered_frame_and_speed() -> None:
    """Harness test: render a held-out scene, run the onboard detector on the RGB only, compare with truth."""
    pytest.importorskip("ultralytics")
    import torch

    from sfhack.sim import detector_scene as ds

    lib = ds.load_library()
    sc = ds.DetectorScene(lib, ds.ensure_ground_textures(data_dir() / "v4" / "detector"))
    lay = ds.sample_layout(lib, 424242, "test")
    t = lay.truck
    view = ds.View(np.array([t.xy[0] + 3.0, t.xy[1] - 2.0, 30.0]), ds.body_attitude(0.7, math.radians(4), 1.0), 4.0,
                   1.0, 1.1, 0.8, 0.3, 0, (0.0, 0.0), 0.0, "clear", 0.0, 0.5, 1.0, 0.5)
    rgb = sc.render(lay, view, np.random.default_rng(0)).rgb
    sc.close()
    timings = {}
    devices = (["0"] if torch.cuda.is_available() else []) + ["cpu"]
    for dev in devices:
        det = td.TruckDetector(device=dev)
        dets = det.detect(rgb, view.pos[2], view.R)  # warm-up
        ms = []
        for _ in range(10):
            t0 = time.perf_counter()
            dets = det.detect(rgb, view.pos[2], view.R)
            ms.append(1000 * (time.perf_counter() - t0))
        timings[dev] = float(np.median(ms))
        trucks = [d for d in dets if d.cls == "truck" and d.conf >= det.cfg.truck_threshold]
        assert trucks, td.describe(dets)
        best = max(trucks, key=lambda d: d.conf)
        err = np.linalg.norm(view.pos[:2] + best.ground_xy - np.asarray(t.xy))
        assert err < 2.5, f"truck ground position off by {err:.2f} m"
    print("\nms per 640x480 frame (median of 10):", {k: round(v, 1) for k, v in timings.items()})
    assert timings["cpu"] < 2000
