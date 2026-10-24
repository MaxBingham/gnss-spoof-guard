"""Mission prep pipeline, offline: synthetic OSM / ortho / DEM in, world + onboard package out, uploaded and flown.

LiDAR "fails" here, so the synthetic map cloud fallback is what gets built. Real downloads are exercised by
`python -m sfhack.missionprep build` (see docs/MAP_PACKAGE.md), never by the test suite.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from sfhack.geo import BBox, GeoOrigin, LatLon
from sfhack.missionprep import build
from sfhack.missionprep.geometry import Grid, clean_ring, prism_mesh, sample_bilinear, triangulate
from sfhack.missionprep.lidar import read_ply
from sfhack.missionprep.sources import FetchError, Fetcher
from sfhack.missionprep.upload import upload

A = LatLon(37.8000, -122.3000)
B = LatLon(37.8027, -122.3000)  # 300 m north
TOWER_H = 80.0  # on the straight line A -> B: the route must go around it


def _origin() -> GeoOrigin:
    return GeoOrigin((A.lat + B.lat) / 2, (A.lon + B.lon) / 2)


def fake_osm(f: Fetcher, bbox: BBox) -> tuple[dict, dict]:
    o = _origin()
    els, nid = [], 1

    def ll(x: float, y: float) -> dict[str, float]:
        lat, lon = o.to_latlon(x, y)
        return {"lat": lat, "lon": lon}

    for i, (x, y) in enumerate((x, y) for x in (-120, -80, -45, 45, 80, 120) for y in (-200, -100, 0, 100, 200)):
        ring = [ll(x - 6, y - 6), ll(x + 6, y - 6), ll(x + 6, y + 6), ll(x - 6, y + 6), ll(x - 6, y - 6)]
        els.append({"type": "way", "id": 1000 + i, "geometry": ring, "tags": {"building": "yes", "building:levels": "3"}})
    els.append({"type": "way", "id": 5000, "tags": {"man_made": "tower", "height": str(TOWER_H)},
                "geometry": [ll(-5, -5), ll(5, -5), ll(5, 5), ll(-5, 5), ll(-5, -5)]})
    for x in range(-180, 181, 20):
        for y in range(-280, 281, 20):
            if (x, y) != (0, 0):
                els.append({"type": "node", "id": nid, **ll(x + 3, y + 3), "tags": {"natural": "tree"}})
                nid += 1
    return {"elements": els}, {"endpoint": "test"}


def fake_ortho(f: Fetcher, bbox: BBox, grid: Grid) -> tuple[np.ndarray, dict]:
    import cv2

    rng = np.random.default_rng(1)
    img = sum(cv2.resize(rng.uniform(0, 255, (grid.rows // s + 1, grid.cols // s + 1)).astype(np.float32),
                         (grid.cols, grid.rows)) for s in (4, 16, 64)) / 3
    rgb = np.repeat(np.clip(img, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
    return rgb, {"name": "synthetic test ortho", "license": "CC0", "attribution": "test", "url": ""}


def fake_dem(f: Fetcher, bbox: BBox, grid: Grid) -> tuple[np.ndarray, dict]:
    X, Y = grid.centers()
    dem = 100.0 + 0.04 * Y + 3.0 * np.sin(X / 40.0)  # 24 m rise to the north + ripples
    return dem.astype(np.float32), {"name": "synthetic DEM", "license": "CC0", "attribution": "test", "url": "",
                                    "product": "synthetic test DEM", "vertical_datum": "test"}


def no_lidar(*_a, **_k):
    raise FetchError("offline test: no LiDAR")


@pytest.fixture(scope="module")
def mission(tmp_path_factory: pytest.TempPathFactory) -> str:
    mp = pytest.MonkeyPatch()
    mp.setenv("SFHACK_DATA", str(tmp_path_factory.mktemp("sfhack-data")))
    rep = build.prepare_mission(A, B, 25.0, "test-mission", log=lambda s: None,
                                downloads={"osm": fake_osm, "ortho": fake_ortho, "terrain": fake_dem, "lidar": no_lidar})
    assert rep["layers"]["lidar"]["kind"] == "synthetic"
    yield rep["mission_id"]
    mp.undo()


def test_geometry_triangulates_concave_footprints() -> None:
    L = clean_ring([[0, 0], [10, 0], [10, 10], [6, 10], [6, 4], [4, 4], [4, 10], [0, 10], [0, 0]])  # U shape
    tris = triangulate(L)
    assert tris is not None and len(tris) == len(L) - 2
    def tri_area(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        (ux, uy), (vx, vy) = q - p, r - p
        return abs(ux * vy - uy * vx) / 2

    area = sum(tri_area(L[a], L[b], L[c]) for a, b, c in tris)
    assert area == pytest.approx(100 - 12)
    v, f = prism_mesh(L, tris, 0.0, 5.0)
    vol = sum(np.dot(v[a], np.cross(v[b], v[c])) for a, b, c in f) / 6  # outward faces: positive volume
    assert vol == pytest.approx(88 * 5)


def test_package_layout_checksums_and_reuse(mission: str) -> None:
    d = build.mission_dir(mission)
    for p in ("world/manifest.json", "world/landmarks.json", "world/ortho.jpg", "world/terrain.npz",
              "onboard/manifest.json", "onboard/mission.json", "onboard/ortho.jpg", "onboard/landmarks.json",
              "onboard/dsm.npz", "onboard/map_cloud.ply", "build_report.json"):
        assert (d / p).exists(), p
    man = json.loads((d / "onboard" / "manifest.json").read_text())
    for layer in man["layers"].values():
        assert build.sha256(d / "onboard" / layer["file"]) == layer["sha256"]
        assert (d / "onboard" / layer["file"]).stat().st_size == layer["bytes"]
    again = build.prepare_mission(A, B, 25.0, mission, log=lambda s: None)  # same inputs: no rebuild
    assert again.get("cached")
    with pytest.raises(ValueError):
        build.mission_dir("../escape")


def test_onboard_is_an_imperfect_copy(mission: str) -> None:
    d = build.mission_dir(mission)
    truth = {lm["id"]: lm for lm in json.loads((d / "world" / "landmarks.json").read_text())}
    onb = {lm["id"]: lm for lm in json.loads((d / "onboard" / "landmarks.json").read_text())}
    assert set(onb) < set(truth)  # some landmarks removed, none invented
    ratios = [onb[i]["height"] / truth[i]["height"] for i in onb if truth[i]["type"] in ("building", "tower")]
    assert max(abs(r - 1) for r in ratios) > 0.02 and all(0.79 <= r <= 1.21 for r in ratios)
    with np.load(d / "world" / "terrain.npz") as w, np.load(d / "onboard" / "dsm.npz") as o:
        wg = Grid(float(w["min_x"]), float(w["max_y"]), float(w["res_m"]), *w["z"].shape)
        og = Grid(float(o["min_x"]), float(o["max_y"]), float(o["res_m"]), *o["dtm"].shape)
        X, Y = wg.centers()
        diff = sample_bilinear(o["dtm"], og, np.stack([X, Y], -1)) - w["z"]
        assert 0.01 < float(np.abs(diff).mean()) < 0.5  # same terrain, never the same array
        assert float((o["dsm"] - o["dtm"]).max()) > 60  # the tower is in the DSM
    xyz, cls = read_ply(d / "onboard" / "map_cloud.ply")
    e = json.loads((d / "onboard" / "manifest.json").read_text())["ortho"]["extent_enu"]
    assert len(xyz) > 10000 and set(np.unique(cls)) <= {2, 5, 6}
    assert e["min_x"] - 5 <= xyz[:, 0].min() and xyz[:, 0].max() <= e["max_x"] + 5


def test_route_goes_around_the_tower_and_clears_the_map(mission: str) -> None:
    d = build.mission_dir(mission)
    m = json.loads((d / "onboard" / "mission.json").read_text())
    cruise = [w for w in m["route"] if w["phase"] in ("TAKEOFF", "CRUISE")]
    assert max(w["enu"][2] for w in cruise) < TOWER_H  # went around, not over
    assert m["route"][-1]["phase"] == "LAND" and m["route"][-1]["enu"][2] == pytest.approx(m["b"]["ground_z"] - 0.3)
    assert m["b"]["ground_z"] > 10  # B is uphill: z datum = ground at A
    assert m["target_scene"]["status"] == "pending" and m["target_scene"]["radius_m"] == 25.0
    pts = [np.array(w["enu"][:2]) for w in cruise]
    for p, q in zip(pts, pts[1:]):  # tower centre (0, 0) stays at least the obstacle margin away
        t = np.clip(np.dot(-p, q - p) / max(np.dot(q - p, q - p), 1e-9), 0, 1)
        assert np.hypot(*(p + t * (q - p))) > 5 + build.OBSTACLE_MARGIN_M - 4


def test_upload_verifies_checksums(mission: str) -> None:
    events: list[dict] = []
    assert not upload(mission, mbps=500, on_event=events.append, corrupt="dsm")
    assert not (build.mission_dir(mission) / "drone").exists()
    assert any(e["type"] == "layer_done" and not e["ok"] for e in events)
    assert upload(mission, mbps=500)
    assert (build.mission_dir(mission) / "drone" / "upload_receipt.json").exists()


def test_mission_flies_on_terrain(mission: str) -> None:
    """FlightSystem(mission=...) spawns on the terrain, follows the stepped route and lands on B's slope."""
    from sfhack.runner import FlightSystem

    test_upload_verifies_checksums(mission)
    from sfhack.config import default_config

    cfg = default_config()
    # This tests the package pipeline, not TRN. With TRN (V3.2 merge) seed 0 lands 3.04 m off instead of 2.58 m:
    # the mock localizer is blind near B on this synthetic map, so the final descent is dead reckoning either way.
    cfg.trn.enabled = False
    fs = FlightSystem(cfg=cfg, mission=mission, localizer="mock", seed=0)
    assert fs.onboard_dir is not None and fs.onboard_dir.name == "drone"
    assert len(fs.mission.waypoints) >= 4 and fs.mission.waypoints[-1].pos[2] > 10
    r = fs.run()
    fs.close()
    assert r.success, r.summary()
    assert math.isfinite(r.final_error)


def test_planner_api(mission: str) -> None:
    from fastapi.testclient import TestClient

    from sfhack.missionprep.server import app

    c = TestClient(app)
    assert "Mission Planner" in c.get("/").text
    assert c.get("/api/defaults").json()["radius_m"] == build.DEFAULT_RADIUS_M
    assert any(m["mission_id"] == mission for m in c.get("/api/missions").json())
    view = c.get(f"/api/missions/{mission}").json()
    assert view["route"][0]["phase"] == "TAKEOFF" and len(view["route"][0]["latlon"]) == 2
    assert {lay["file"] for lay in view["layers"]} >= {"ortho.jpg", "dsm.npz", "map_cloud.ply", "mission.json"}
    assert c.get("/api/missions/..%2Fescape").status_code in (400, 404)
    assert c.post("/api/upload", json={"mission_id": "../x"}).status_code == 400


class FakeViewer:
    """Stands in for subprocess.Popen in the /api/fly test: records the call, exits when told to."""

    instances: list["FakeViewer"] = []

    def __init__(self, args, **kw) -> None:
        self.args, self.kw, self.returncode, self.terminated = args, kw, None, False
        self.pid = 4242 + len(FakeViewer.instances)
        FakeViewer.instances.append(self)

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminated, self.returncode = True, 1

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def test_fly_endpoint_launches_one_viewer(mission: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess
    import sys

    from fastapi.testclient import TestClient

    from sfhack.config import REPO_ROOT
    from sfhack.missionprep import server

    monkeypatch.setattr(server.subprocess, "Popen", FakeViewer)
    FakeViewer.instances.clear()
    c = TestClient(server.app)
    assert c.post("/api/fly", json={"mission_id": "../x"}).status_code == 400
    assert c.post("/api/fly", json={"mission_id": "never-built"}).status_code == 404

    r = c.post("/api/fly", json={"mission_id": mission}).json()
    v = FakeViewer.instances[0]
    assert r["pid"] == v.pid and r["command"].endswith(f"--mission {mission}")
    assert v.args == [sys.executable, str(REPO_ROOT / "demos" / "demo_live.py"), "--mission", mission]
    assert v.kw["cwd"] == str(REPO_ROOT) and v.kw["stderr"] == subprocess.STDOUT
    if sys.platform == "win32":
        assert v.kw["creationflags"] & subprocess.CREATE_NEW_PROCESS_GROUP
    assert c.get("/api/fly/status").json()["state"] == "running"

    c.post("/api/fly", json={"mission_id": mission})  # one viewer at a time: the first one is stopped
    assert v.terminated and len(FakeViewer.instances) == 2

    w = FakeViewer.instances[1]  # crash: status shows the tail of viewer.log
    w.kw["stdout"].write(b"Traceback (most recent call last):\nRuntimeError: no OpenGL context\n")
    w.kw["stdout"].flush()
    w.returncode = 1
    s = c.get("/api/fly/status").json()
    assert s["state"] == "exited" and s["exit_code"] == 1 and s["log_tail"][-1] == "RuntimeError: no OpenGL context"
    assert c.post("/api/fly/stop").json()["state"] == "stopped"

    c.post("/api/fly", json={"mission_id": mission})  # a user stop is not reported as a crash
    c.post("/api/fly/stop")
    s = c.get("/api/fly/status").json()
    assert s["state"] == "stopped" and "log_tail" not in s

    r = c.post("/api/fly", json={"mission_id": mission, "night": True}).json()  # planner "Night" checkbox
    assert FakeViewer.instances[-1].args[-1] == "--night" and r["night"] and r["command"].endswith("--night")
    c.post("/api/fly/stop")


def test_utm_projection_basics() -> None:
    from sfhack.missionprep.sentinel import latlon_to_utm, utm_zone

    assert utm_zone(23.32) == 34 and utm_zone(-122.26) == 10
    e, n = latlon_to_utm(np.array([0.0, 49.0, 49.0]), np.array([21.0, 21.0, 22.0]), 34)
    assert e[0] == pytest.approx(500000.0) and n[0] == pytest.approx(0.0, abs=1e-6)
    assert e[1] == pytest.approx(500000.0)
    assert n[1] == pytest.approx(5427455.78, abs=0.01)  # 0.9996 x meridian arc to 49 deg (5429627.63 m, quadrature)
    assert e[2] - 500000.0 == pytest.approx(73.13e3, rel=0.01)  # 1 deg of longitude at 49 N, x 0.9996


def test_global_region_writes_nir_ndvi_and_raw_osm(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SFHACK_DATA", str(tmp_path))

    def fake_s2(f: Fetcher, bbox: BBox, grid: Grid) -> tuple[np.ndarray, dict]:
        rgb, src = fake_ortho(f, bbox, grid)
        nir = np.full(grid.shape, 128, np.uint8)
        return rgb, {**src, "scene": "TEST", "date": "2025-08-28", "_nir": nir, "_ndvi": nir}

    rep = build.prepare_mission(A, B, 25.0, "test-global", log=lambda s: None, region="global", min_agl_m=60.0,
                                downloads={"osm": fake_osm, "ortho": fake_s2, "terrain": fake_dem})
    d = build.mission_dir(rep["mission_id"])
    assert rep["region"] == "global" and rep["inputs"]["region"] == "global"
    assert rep["layers"]["lidar"]["kind"] == "synthetic"
    for sub in ("world", "onboard"):
        assert (d / sub / "nir.png").exists() and (d / sub / "ndvi.png").exists()
    assert json.loads((d / "world" / "osm.json").read_text(encoding="utf-8"))["elements"]
    om = json.loads((d / "onboard" / "manifest.json").read_text(encoding="utf-8"))
    assert "nir" in om["layers"] and om["region"] == "global"
    mission = json.loads((d / "onboard" / "mission.json").read_text(encoding="utf-8"))
    assert mission["cruise"]["min_agl_m"] == 60.0
