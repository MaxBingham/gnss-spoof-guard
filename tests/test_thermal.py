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
    cam = ThermalCamera(m, netd_K=0.0, abs_error_K=0.0, gain_error=0.0)
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


def test_load_truth_contrast_scales_around_air(tmp_path) -> None:
    from sfhack.thermal.camera import load_truth

    t = np.full((40, 60), 290.0, np.float32)
    t[10:30, 20:40] = 300.0
    meta = {"extent_enu": {"min_x": -15.0, "min_y": -10.0, "max_x": 15.0, "max_y": 10.0}, "res_m": 0.5}
    np.savez(tmp_path / "thermal_truth.npz", temp_K=t, wall_K=np.float32(296.0), air_K=np.float32(295.0),
             meta=json.dumps(meta))
    full = load_truth(tmp_path / "thermal_truth.npz", soften_m=0.0)
    half = load_truth(tmp_path / "thermal_truth.npz", contrast=0.5, soften_m=0.0)
    assert full.grid.shape == (40, 60) and full.grid.min_x == -15.0
    assert half.temp_K[20, 30] == pytest.approx(297.5) and half.temp_K[0, 0] == pytest.approx(292.5)
    assert half.wall_K == pytest.approx(295.5)
    soft = load_truth(tmp_path / "thermal_truth.npz")
    assert 290.0 < soft.temp_K[10, 30] < 300.0  # edge softened, interior kept
    assert soft.temp_K[20, 30] == pytest.approx(300.0, abs=1e-3)


@pytest.mark.parametrize("yaw", [0.0, 1.1, -2.4])
def test_prior_predict_view_matches_camera(yaw: float) -> None:
    """The onboard prediction (prior.predict_view) and the simulated camera agree on geometry."""
    from sfhack.thermal.prior import ThermalPrior

    truth = _hot_spot_truth()
    t = truth.temp_K.copy()
    t[150:160, 230:300] = 305.0  # an asymmetric bar so a wrong rotation shows
    truth.temp_K = t
    m, d = _camera_world(3.0, -2.0, 60.0, yaw=yaw)
    cam = ThermalCamera(m, netd_K=0.0, abs_error_K=0.0, gain_error=0.0)
    cam.fpn[:] = 0
    cam.vignette[:] = 0
    cam.texture[:] = 0
    img = cam.render(d, truth)[::2, ::2]
    g = truth.grid
    meta = {"extent_enu": {"min_x": g.min_x, "max_y": g.max_y, "max_x": g.max_x, "min_y": g.min_y}, "res_m": g.res}
    prior = ThermalPrior(t, np.ones_like(t), np.zeros(t.shape, np.uint8), np.zeros(t.shape, np.uint8),
                         np.zeros(t.shape, np.uint8), meta)
    pred, sig = prior.predict_view(np.array([3.0, -2.0]), 60.0 - 0.05, cam.intrinsics, yaw=yaw)  # camera 5 cm below the body
    assert pred.shape == img.shape and np.isfinite(pred).all()
    assert np.corrcoef(pred.ravel(), img.ravel())[0, 1] > 0.95
    cam.close()


def test_night_cooling_through_the_night(curves) -> None:
    """Roads stay warmer than grass all night (no crossover) while everything cools from 21 to 03 h."""
    tab = {h: curves.at(h)[:, -1, -1] for h in (19.85, 21.85, 1.85)}  # 21, 23, 03 h PDT in solar time
    a, g = CLS["asphalt"], CLS["grass"]
    assert tab[19.85][a] > tab[21.85][a] > tab[1.85][a]
    for v in tab.values():
        assert v[a] - v[g] > 3.0


def test_untagged_minor_roads_abroad_are_uncertain() -> None:
    g, rgb, nir, origin, osm = _synthetic_scene()
    road = osm["elements"][0]
    road["tags"] = {"highway": "residential"}  # no surface tag
    prior = classify(rgb, nir, g, osm, origin, [], untagged_paved_p=0.4)
    assert prior.cls[100, 100] == CLS["dirt_road"] and prior.roads[100, 100] == 0 and prior.uncertain[100, 100]
    assert prior.roads_json[0]["paved"] is None
    truth = classify(rgb, nir, g, osm, origin, [], untagged_paved_p=0.4, truth=True)
    assert truth.cls[100, 100] in (CLS["dirt_road"], CLS["asphalt_minor"])
    us = classify(rgb, nir, g, osm, origin, [])  # US default: untagged residential is asphalt
    assert us.cls[100, 100] == CLS["asphalt"] and us.roads[100, 100] == 1


def test_underground_building_is_not_a_roof() -> None:
    g, rgb, nir, origin, osm = _synthetic_scene()
    lat0, lon0 = origin.to_latlon(-20, -20)
    lat1, lon1 = origin.to_latlon(-10, -10)
    ring = [{"lat": lat0, "lon": lon0}, {"lat": lat0, "lon": lon1}, {"lat": lat1, "lon": lon1},
            {"lat": lat1, "lon": lon0}, {"lat": lat0, "lon": lon0}]
    fp = [[-20.0, -20.0], [-10.0, -20.0], [-10.0, -10.0], [-20.0, -10.0]]
    for tags, roof in (({"building": "yes"}, True), ({"building": "garages", "layer": "-1"}, False),
                       ({"building": "underground"}, False)):
        el = {"type": "way", "id": 7, "tags": tags, "geometry": ring}
        lm = {"id": 0, "type": "building", "osm_id": "way/7", "footprint_enu": fp, "height": 8.0}
        cm = classify(rgb, nir, g, {"elements": osm["elements"] + [el]}, origin, [lm])
        r, c = g.to_px(np.array([-15.0, -15.0])).round().astype(int)[::-1]
        assert (cm.cls[r, c] in (CLS["roof_dark"], CLS["roof_light"])) == roof, tags


def test_traffic_moves_along_paved_roads() -> None:
    g = Grid(-100.0, 100.0, 0.5, 400, 400)
    t = np.full(g.shape, 290.0, np.float32)
    cls = np.full(g.shape, CLS["grass"], np.uint8)
    cls[190:210, :] = CLS["asphalt"]  # an east-west road through y = 0
    truth = ThermalTruth(t, g, 291.0, 289.0, {}, cls=cls,
                         roads=[{"pts_enu": [[-100.0, 0.0], [100.0, 0.0]], "width_m": 8.0}])
    m, d = _camera_world(0.0, 0.0, 60.0)
    cam = ThermalCamera(m, netd_K=0.0, abs_error_K=0.0, gain_error=0.0, traffic=40, seed=3)
    a = cam.scene_temperature(d, truth)
    assert (a > 289.0 + 10).any()  # hot engine hoods in view
    d.time += 2.0
    b = cam.scene_temperature(d, truth)
    assert np.abs(a - b).max() > 5  # they moved
    # off the pavement nothing is painted: an all-grass map shows no cars
    truth.cls = np.full(g.shape, CLS["grass"], np.uint8)
    cam2 = ThermalCamera(m, netd_K=0.0, abs_error_K=0.0, gain_error=0.0, traffic=40, seed=3)
    assert cam2.scene_temperature(d, truth).max() < 291.0
    cam.close()
    cam2.close()
