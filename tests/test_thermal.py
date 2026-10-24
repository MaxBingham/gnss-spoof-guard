"""Thermal world model and camera: physics sanity, classification, raster alignment, camera geometry."""

from __future__ import annotations

import json
import math

import mujoco
import numpy as np
import pytest

from sfhack.geo import GeoOrigin
from sfhack.missionprep.geometry import Grid
from sfhack.thermal.camera import ThermalCamera, ThermalTruth
from sfhack.thermal.classify import classify
from sfhack.thermal.materials import CLS, SEASONS
from sfhack.thermal.model import solar_hour, temperature_map
from sfhack.thermal.seb import air_temp, simulate


@pytest.fixture(scope="module")
def curves():
    return simulate(37.87, SEASONS["summer"], days=3, dt=300.0)


def test_night_contrast_roads_warm_vegetation_cool(curves) -> None:
    t = curves.at(21.85)[:, -1, -1]  # full sun, open sky
    air = curves.air_at(21.85)
    assert t[CLS["asphalt"]] > air + 1.0
    assert t[CLS["grass"]] < air - 1.0
    assert t[CLS["asphalt"]] - t[CLS["grass"]] > 4.0
    assert t[CLS["roof_metal"]] < t[CLS["roof_dark"]]  # low emissivity looks cold
    # daytime: asphalt far above air, water near air
    d = curves.at(14.0)[:, -1, -1]
    assert d[CLS["asphalt"]] > curves.air_at(14.0) + 15
    assert abs(d[CLS["water"]] - curves.air_at(14.0)) < 6


def test_shaded_canyon_road_differs(curves) -> None:
    tab = curves.at(21.85)[CLS["asphalt"]]  # (exposure, svf)
    assert tab[0, -1] < tab[-1, -1]  # shaded all day -> cooler at night
    assert tab[-1, 0] > tab[-1, -1]  # low sky view -> warmer at night


def test_air_temperature_cycle() -> None:
    c = SEASONS["summer"]
    assert abs(air_temp(np.array([c.t_max_h]), c)[0] - (c.tmax_c + 273.15)) < 1e-6
    assert abs(air_temp(np.array([c.t_min_h]), c)[0] - (c.tmin_c + 273.15)) < 1e-6


def test_solar_hour_berkeley_and_ukraine() -> None:
    assert solar_hour(23.0, -122.26, "summer")[0] == pytest.approx(21.85, abs=0.02)
    assert solar_hour(23.0, 25.0, "summer")[0] == pytest.approx(21.67, abs=0.02)


def _synthetic_scene():
    g = Grid(-50.0, 50.0, 0.5, 200, 200)
    rgb = np.full((200, 200, 3), (95, 125, 80), np.uint8)  # lawn
    nir = np.full((200, 200), 170, np.uint8)
    origin = GeoOrigin(37.0, -122.0)
    lat0, lon0 = origin.to_latlon(-50, 0)
    lat1, lon1 = origin.to_latlon(50, 0)
    road = {"type": "way", "id": 1, "tags": {"highway": "residential", "surface": "asphalt"},
            "geometry": [{"lat": lat0, "lon": lon0}, {"lat": lat1, "lon": lon1}]}
    # the image shows the road (grey, no NIR) where OSM puts it
    rgb[:, 93:107] = (90, 90, 92)
    nir[:, 93:107] = 85
    return g, rgb, nir, origin, {"elements": [road]}


def test_classify_road_and_lawn() -> None:
    g, rgb, nir, origin, osm = _synthetic_scene()
    cm = classify(rgb, nir, g, osm, origin, [])
    assert cm.cls[100, 100] == CLS["asphalt"] and cm.roads[100, 100] == 1
    assert cm.cls[100, 20] == CLS["grass"] and cm.roads[100, 20] == 0
    assert cm.roads_json and cm.roads_json[0]["paved"]
    # a tree crown landmark hides the road
    cm2 = classify(rgb, nir, g, osm, origin, [{"type": "tree", "centroid_enu": [0.0, 0.0], "radius": 4.0}])
    assert cm2.cls[100, 100] == CLS["tree"] and cm2.roads[100, 100] == 0


def test_no_nir_fallback_still_finds_vegetation() -> None:
    g, rgb, _, origin, osm = _synthetic_scene()
    cm = classify(rgb, None, g, osm, origin, [])
    assert cm.cls[100, 20] == CLS["grass"] and cm.cls[100, 100] == CLS["asphalt"]


def test_temperature_map_interpolates(curves) -> None:
    tab = curves.at(21.85)
    cls = np.full((4, 4), CLS["asphalt"], np.uint8)
    full = temperature_map(cls, np.ones((4, 4)), np.ones((4, 4)), tab, curves.exposures, curves.svfs)
    half = temperature_map(cls, np.full((4, 4), 0.25), np.ones((4, 4)), tab, curves.exposures, curves.svfs)
    assert full.shape == (4, 4) and full.dtype == np.float32
    assert np.allclose(full, tab[CLS["asphalt"], -1, -1], atol=1e-3)
    assert tab[CLS["asphalt"], 0, -1] < half[0, 0] < full[0, 0]


def _camera_world(x: float, y: float, z: float, yaw: float = 0.0):
    """Flat floor, a 'drone' body with a down-looking thermal camera mounted like MujocoSim's."""
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[200, 200, 1])
    body = spec.worldbody.add_body(name="x2", pos=[x, y, z], quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.02])
    # MujocoSim: image top = body forward (+x), image right = body right (-y)
    xa, ya = np.array([0, -1, 0.0]), np.array([1, 0, 0.0])
    za = np.cross(xa, ya)
    R = np.column_stack([xa, ya, za])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    vfov = math.degrees(2 * math.atan(math.tan(math.radians(25)) * 512 / 640))
    body.add_camera(name="thermal", pos=[0, 0, -0.05], quat=q.tolist(), fovy=vfov)
    spec.visual.global_.offwidth = 640
    spec.visual.global_.offheight = 512
    spec.stat.extent = 100.0
    spec.visual.map.znear = 0.001
    spec.visual.map.zfar = 30.0
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def _hot_spot_truth() -> ThermalTruth:
    g = Grid(-100.0, 100.0, 0.5, 400, 400)
    t = np.full(g.shape, 290.0, np.float32)
    uv = g.to_px(np.array([10.25, -0.25]))  # hot square centred on cell (219, 200): 10.25 m east, 0.25 m south
    c, r = int(round(uv[0])), int(round(uv[1]))
    t[r - 4:r + 5, c - 4:c + 5] = 310.0
    return ThermalTruth(t, g, 291.0, 289.0, {})


def test_camera_geometry_and_orientation() -> None:
    m, d = _camera_world(0.0, 0.0, 60.0, yaw=0.0)  # facing east: image top = east
    cam = ThermalCamera(m, netd_K=0.0)
    cam.fpn[:] = 0
    cam.vignette[:] = 0
    img = cam.render(d, _hot_spot_truth())
    assert img.shape == (512, 640) and img.dtype == np.float32
    k = cam.intrinsics
    assert k["w"] == 640 and k["h"] == 512 and abs(k["hfov_deg"] - 50) < 1e-6
    rr, cc = np.nonzero(img > 300.0)
    r, c = rr.mean() + 0.5, cc.mean() + 0.5  # centroid of the hot square (pixel centres at +0.5)
    # facing east at 60 m: east = image up, south = image right
    assert abs((c - k["cx"]) - 0.25 / 60 * k["fx"]) < 1.0
    assert abs((k["cy"] - r) - 10.25 / 60 * k["fy"]) < 1.0
    assert img.max() > 305 and abs(np.median(img) - 290) < 0.3
    cam.close()


def test_camera_noise_and_8bit() -> None:
    m, d = _camera_world(0.0, 0.0, 60.0)
    cam = ThermalCamera(m)
    truth = _hot_spot_truth()
    a = cam.render(d, truth)
    d.time += 0.1
    b = cam.render(d, truth)
    temporal = float(np.std(a[100:200, 100:200] - b[100:200, 100:200]) / math.sqrt(2))
    assert 0.03 < temporal < 0.08  # NETD 50 mK
    u8 = cam.to_8bit(a)
    assert u8.dtype == np.uint8 and u8.shape == a.shape and u8.max() == 255  # white-hot: the hot spot saturates
    cam.close()


def _mission(mid: str):
    from sfhack.config import data_dir

    d = data_dir() / "missions" / mid
    if not (d / "world" / "thermal_truth.npz").exists():
        pytest.skip(f"{mid}: run `python -m sfhack.thermal build --mission {mid}` first")
    return d


@pytest.mark.parametrize("mid", ["berkeley", "m-c804bb84"])
def test_mission_rasters_aligned(mid: str) -> None:
    from sfhack.thermal.prior import load_prior

    d = _mission(mid)
    man = json.loads((d / "world" / "manifest.json").read_text(encoding="utf-8"))
    shape = (man["ortho"]["height_px"], man["ortho"]["width_px"])
    p = load_prior(d / "onboard")
    with np.load(d / "world" / "thermal_truth.npz") as z:
        truth = z["temp_K"]
    assert truth.shape == shape and p.temp_K.shape == shape and p.sigma_K.shape == shape
    assert p.cls.dtype == np.uint8 and p.roads.dtype == np.uint8
    assert p.meta["extent_enu"] == man["ortho"]["extent_enu"] and p.meta["res_m"] == 0.5
    assert (p.sigma_K > 0).all()
    # roads are warmer than vegetation in the truth, and prior and truth agree on where
    roads = p.roads > 0
    veg = np.isin(p.cls, [CLS["grass"], CLS["tree"]])
    assert np.median(truth[roads]) > np.median(truth[veg]) + 3.0
    assert np.corrcoef(p.temp_K.ravel(), truth.ravel())[0, 1] > 0.4
