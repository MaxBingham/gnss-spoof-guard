"""Prepare a mission: download the open data around A and B, build the simulated world (truth) and the
onboard map package (what is uploaded to the drone). Format: docs/MAP_PACKAGE.md.

    $SFHACK_DATA/missions/<id>/
      world/     truth for MuJoCo: manifest.json, landmarks.json, ortho.jpg, terrain.npz
      onboard/   the drone's package: manifest.json, mission.json, ortho.jpg, landmarks.json, dsm.npz, map_cloud.ply
      build_report.json

This is harness / tooling code: it sees the truth (it makes it). The onboard package it writes is a
deliberately imperfect copy, and onboard code reads only that directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image

from ..config import MissionConfig, data_dir
from ..geo import BBox, GeoOrigin, LatLon
from . import lidar as L
from .geometry import Grid, polygon_mask, sample_bilinear
from .osm import extract_landmarks, render_osm_ground
from .sources import SRC_OSM, FetchError, Fetcher, fetch_dem, fetch_ortho, fetch_osm

FORMAT = "sfhack-mission/1"
DEFAULT_A = LatLon(37.8703, -122.2595)  # near Sather Gate (same A/B as sfhack/world/sample_berkeley)
DEFAULT_B = LatLon(37.8754, -122.2590)  # Hearst Ave in front of Etcheverry Hall
DEFAULT_RADIUS_M = 30.0
MARGIN_M = 150.0
MIN_SIZE_M = 400.0
ORTHO_RES = 0.5  # m/px, world and onboard ortho
DEM_RES = 1.0  # m, onboard dsm.npz (dtm + dsm)
HFIELD_RES = 2.0  # m, world terrain.npz (MuJoCo height field)
CLOUD_VOXEL = 1.0  # m, map_cloud.ply (0.75 m gave 2.2 M points / 28 MB on the Berkeley default)
LIDAR_SPACING = 0.5  # m, EPT depth to read
LIDAR_TIMEBOX_S = 90.0  # per prepare; past this the synthetic cloud is used
# Route (planned on the ONBOARD map, the drone has nothing else). Required altitude per 4 m cell =
# max(DSM within OBSTACLE_MARGIN_M + CLEARANCE_M, DTM within the margin + MIN_AGL_M). Dijkstra finds the
# path A -> B where every metre costs 1 + (altitude it forces above what A and B need) / CLIMB_PENALTY_M,
# so the route goes around a tower instead of 100 m over it (vision fixes need < ~60 m). Legs of at most
# LEG_MAX_M, each flown at the highest requirement along it and the next one. MIN_AGL also bounds which
# buildings the onboard map may "forget" (ONBOARD_DROP_FRAC): a missing building is never in the way.
ROUTE_CELL_M = 4.0
OBSTACLE_MARGIN_M = 15.0
CLIMB_PENALTY_M = 5.0
LEG_MAX_M = 120.0
CLEARANCE_M = 15.0
MIN_AGL_M = 30.0
# Onboard map = imperfect copy of the truth (docs/MAP_PACKAGE.md, "Onboard vs truth")
ONBOARD_HEIGHT_ERR = 0.20  # building / tower heights x U(1 - e, 1 + e)
ONBOARD_DROP_FRAC = 0.10  # buildings (lower than MIN_AGL_M) and trees missing from the onboard map
ONBOARD_TREE_JITTER_M = 0.5
ONBOARD_DTM_NOISE_M = 0.15  # smooth (~20 m correlation) error added to the onboard DTM
CLOUD_NOISE_M = 0.05  # synthetic cloud range noise; real LiDAR keeps its own errors
# Launch pad: the IMU is calibrated on the ground assuming it stands level (preflight). On a 2.5 deg slope
# that put 0.4 m/s^2 of gravity into the accel bias and the drone drifted off downhill right after takeoff,
# invisible to the estimate. Real crews take off from a level pad: the terrain under A is flattened.
PAD_RADIUS_M = 4.0
PAD_BLEND_M = 10.0  # linear blend back to the DEM between PAD_RADIUS_M and this radius


def missions_dir() -> Path:
    p = data_dir() / "missions"
    p.mkdir(parents=True, exist_ok=True)
    return p


MISSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def mission_dir(mission_id: str) -> Path:
    """<missions>/<id>; ids are plain names (letters, digits, - and _) so they never leave that folder."""
    if not MISSION_ID.match(mission_id):
        raise ValueError(f"invalid mission id {mission_id!r} (letters, digits, - and _, max 64)")
    return missions_dir() / mission_id


def default_mission_id(a: LatLon, b: LatLon, radius_m: float) -> str:
    key = f"{a.lat:.6f},{a.lon:.6f},{b.lat:.6f},{b.lon:.6f},{radius_m:.1f}"
    return "m-" + hashlib.sha1(key.encode()).hexdigest()[:8]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_for(a: LatLon, b: LatLon) -> tuple[GeoOrigin, dict[str, float], BBox]:
    """ENU origin = midpoint of A and B; extent = A/B + margin, symmetric, snapped to 2 m so every raster
    (0.5 / 1 / 2 m) tiles it exactly; bbox = that extent in lat/lon (what is downloaded)."""
    origin = GeoOrigin((a.lat + b.lat) / 2, (a.lon + b.lon) / 2)
    ax, ay = origin.to_enu(a.lat, a.lon)
    hw = max(abs(ax) + MARGIN_M, MIN_SIZE_M / 2)
    hh = max(abs(ay) + MARGIN_M, MIN_SIZE_M / 2)
    hw, hh = 2.0 * math.ceil(hw / 2.0), 2.0 * math.ceil(hh / 2.0)
    extent = {"min_x": -hw, "min_y": -hh, "max_x": hw, "max_y": hh}
    south, west = origin.to_latlon(-hw, -hh)
    north, east = origin.to_latlon(hw, hh)
    return origin, extent, BBox(south, west, north, east)


@dataclass
class Layer:
    name: str
    seconds: float = 0.0
    source: str = ""
    note: str = ""


def _timed(fn: Callable[..., Any], *args: Any, **kw: Any) -> tuple[Any, float, str]:
    t0 = time.time()
    try:
        return fn(*args, **kw), time.time() - t0, ""
    except Exception as e:  # noqa: BLE001 - every layer has a fallback; the error goes to the log
        return None, time.time() - t0, f"{type(e).__name__}: {e}"


def _fetch_lidar(f: Fetcher, bbox: BBox, deadline: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    res = L.find_resources(f, bbox)
    if not res:
        raise FetchError("no 3DEP LiDAR collection covers this area")
    errors = []
    for r in res[:2]:
        try:
            ll, cls, info = L.read_ept(f, r["url"], bbox, LIDAR_SPACING, max_nodes=300, deadline=deadline)
            if len(ll) < 1000:
                raise FetchError(f"{r['name']}: only {len(ll)} points in the bbox")
            return ll, cls, {**info, "collection_name": r["name"], "year": r["year"]}
        except Exception as e:  # noqa: BLE001
            errors.append(f"{r['name']}: {e}")
    raise FetchError(" | ".join(errors))


# ------------------------------------------------------------------------------------------- geometry helpers


def _footprint_base(lm: dict[str, Any], dtm: np.ndarray, g: Grid) -> float:
    if lm.get("footprint_enu"):
        pts = np.vstack([np.asarray(lm["footprint_enu"], float), [lm["centroid_enu"]]])
    else:
        pts = np.asarray([lm["centroid_enu"]], float)
    return float(sample_bilinear(dtm, g, pts).min())


def _local_mask(g: Grid, ring: np.ndarray, pad: int = 1) -> tuple[np.ndarray, slice, slice] | None:
    """Polygon mask on the sub-window of the grid that holds the ring."""
    uv = g.to_px(ring)
    c0, r0 = (np.floor(uv.min(axis=0)) - pad).astype(int)
    c1, r1 = (np.ceil(uv.max(axis=0)) + pad + 1).astype(int)
    c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1, g.cols), min(r1, g.rows)
    if c1 <= c0 or r1 <= r0:
        return None
    sub = Grid(g.min_x + c0 * g.res, g.max_y - r0 * g.res, g.res, r1 - r0, c1 - c0)
    return polygon_mask(sub, [ring]), slice(r0, r1), slice(c0, c1)


def burn_dsm(dtm: np.ndarray, g: Grid, landmarks: list[dict[str, Any]]) -> np.ndarray:
    """DSM = DTM raised to roof level inside footprints (building / tower / fountain) and to crown top under trees."""
    dsm = dtm.copy()
    for lm in landmarks:
        top = float(lm["base_z"]) + float(lm["height"])
        if lm.get("footprint_enu"):
            m = _local_mask(g, np.asarray(lm["footprint_enu"], float))
        else:
            cx, cy = lm["centroid_enu"]
            r = float(lm.get("radius", 3.0))
            ang = np.linspace(0, 2 * math.pi, 16, endpoint=False)
            m = _local_mask(g, np.column_stack([cx + r * np.cos(ang), cy + r * np.sin(ang)]))
        if m is None:
            continue
        mask, rs, cs = m
        sub = dsm[rs, cs]
        np.maximum(sub, np.where(mask > 0, top, -np.inf), out=sub)
    return dsm


def lidar_dsm(xyz: np.ndarray, g: Grid) -> np.ndarray:
    """Max z of LiDAR returns per cell (nan where empty)."""
    uv = np.round(g.to_px(xyz[:, :2])).astype(int)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < g.cols) & (uv[:, 1] >= 0) & (uv[:, 1] < g.rows)
    out = np.full(g.rows * g.cols, -np.inf)
    np.maximum.at(out, uv[ok, 1] * g.cols + uv[ok, 0], xyz[ok, 2])
    out[~np.isfinite(out)] = np.nan
    return out.reshape(g.shape)


def required_z(dsm: np.ndarray, dtm: np.ndarray, g: Grid) -> tuple[np.ndarray, Grid]:
    """Altitude the drone needs over each ROUTE_CELL_M cell (obstacles within OBSTACLE_MARGIN_M count)."""
    k = max(1, int(round(ROUTE_CELL_M / g.res)))
    R, C = g.rows // k, g.cols // k

    def pool(a: np.ndarray) -> np.ndarray:  # block max: k x k cells -> one
        return a[:R * k, :C * k].reshape(R, k, C, k).max(axis=(1, 3))

    r = int(math.ceil(OBSTACLE_MARGIN_M / (k * g.res)))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    surf = cv2.dilate(pool(dsm), disk)
    ground = cv2.dilate(pool(dtm), disk)
    return np.maximum(surf + CLEARANCE_M, ground + MIN_AGL_M), Grid(g.min_x, g.max_y, k * g.res, R, C)


def plan_route(a: np.ndarray, b: np.ndarray, a_z: float, b_z: float, dsm: np.ndarray, dtm: np.ndarray, g: Grid,
               mc: MissionConfig) -> tuple[list[dict[str, Any]], list[dict[str, float]]]:
    """Waypoints A -> B on the onboard DSM: takeoff, stepped cruise legs, descend, land (ENU, z absolute)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    req, g4 = required_z(dsm, dtm, g)
    R, C = g4.shape

    def cell(p: np.ndarray) -> tuple[int, int]:
        c, r = np.round(g4.to_px(p)).astype(int)
        return int(np.clip(r, 0, R - 1)), int(np.clip(c, 0, C - 1))

    def seg_req(p: np.ndarray, q: np.ndarray) -> float:
        s = np.linspace(0, 1, max(2, int(np.hypot(*(q - p)) / 2.0) + 1))[:, None]
        cr = np.round(g4.to_px(p + s * (q - p))).astype(int)
        return float(req[np.clip(cr[:, 1], 0, R - 1), np.clip(cr[:, 0], 0, C - 1)].max())

    ra, rb = cell(a), cell(b)
    z_ref = max(req[ra], req[rb])
    cost = 1.0 + np.maximum(req - z_ref, 0.0) / CLIMB_PENALTY_M
    idx = np.arange(R * C).reshape(R, C)
    src, dst, w = [], [], []
    pairs = ((idx[:, :-1], idx[:, 1:], 1.0), (idx[:-1, :], idx[1:, :], 1.0),  # 8-connected grid
             (idx[:-1, :-1], idx[1:, 1:], math.sqrt(2)), (idx[:-1, 1:], idx[1:, :-1], math.sqrt(2)))
    for a_, b_, k in pairs:
        src.append(a_.ravel())
        dst.append(b_.ravel())
        w.append(ROUTE_CELL_M * k * 0.5 * (cost.ravel()[a_.ravel()] + cost.ravel()[b_.ravel()]))
    G = coo_matrix((np.concatenate(w), (np.concatenate(src), np.concatenate(dst))), shape=(R * C, R * C)).tocsr()
    _, pred = dijkstra(G, directed=False, indices=int(idx[ra]), return_predecessors=True)
    path = [int(idx[rb])]
    while path[-1] != int(idx[ra]) and pred[path[-1]] >= 0:
        path.append(int(pred[path[-1]]))
    X, Y = g4.centers()
    pts = [a] + [np.array([X.flat[k], Y.flat[k]]) for k in path[::-1][1:-1]] + [b]
    # string pulling: skip ahead while the straight line needs no more altitude than the path it replaces
    keep, i = [pts[0]], 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1:
            along = max(seg_req(pts[k], pts[k + 1]) for k in range(i, j))
            if seg_req(pts[i], pts[j]) <= along + 0.5:
                break
            j -= 1
        keep.append(pts[j])
        i = j
    wps = [keep[0]]
    for p in keep[1:]:  # split long legs
        n = max(1, math.ceil(float(np.hypot(*(p - wps[-1]))) / LEG_MAX_M))
        q0 = wps[-1]
        wps += [q0 + (p - q0) * (k + 1) / n for k in range(n)]
    n = len(wps) - 1
    leg_req = [seg_req(wps[i], wps[i + 1]) for i in range(n)]
    z = [math.ceil(max(leg_req[i], leg_req[i + 1] if i + 1 < n else -np.inf)) for i in range(n)]
    route = [{"phase": "TAKEOFF", "enu": [a[0], a[1], math.ceil(leg_req[0])], "speed_m_s": mc.climb_speed}]
    for i in range(n):
        p = wps[i + 1]
        route.append({"phase": "CRUISE", "enu": [float(p[0]), float(p[1]), float(z[i])], "speed_m_s": mc.cruise_speed})
    route.append({"phase": "DESCEND", "enu": [b[0], b[1], b_z + mc.flare_alt], "speed_m_s": mc.descent_speed})
    route.append({"phase": "LAND", "enu": [b[0], b[1], b_z - 0.3], "speed_m_s": mc.land_speed})
    for wp in route:
        wp["enu"] = [round(float(v), 2) for v in wp["enu"]]
    legs = [{"leg": i + 1, "required_z": round(leg_req[i], 1), "z": float(z[i])} for i in range(n)]
    return route, legs


# ------------------------------------------------------------------------------------------- onboard copy


def degrade_ortho(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Same imagery, different 'processing': gamma, contrast, saturation, a slight blur."""
    img = rgb.astype(np.float32) / 255.0
    img = img ** rng.uniform(0.9, 1.1)
    gray = img.mean(axis=2, keepdims=True)
    img = gray + 0.88 * (img - gray)  # saturation
    img = 0.5 + rng.uniform(0.88, 0.95) * (img - 0.5) + rng.uniform(-0.02, 0.02)
    img = cv2.GaussianBlur(img, (0, 0), 0.5)
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def smooth_noise(shape: tuple[int, int], corr_px: float, std: float, rng: np.random.Generator) -> np.ndarray:
    n = cv2.GaussianBlur(rng.normal(0.0, 1.0, shape).astype(np.float32), (0, 0), corr_px)
    return (n / max(float(n.std()), 1e-9) * std).astype(np.float32)


def onboard_landmarks(truth: list[dict[str, Any]], rng: np.random.Generator) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out, dropped, scaled = [], {"building": [], "tree": []}, 0
    for lm in truth:
        lm = json.loads(json.dumps(lm))
        t = lm["type"]
        if t == "building" and lm["height"] < MIN_AGL_M and rng.uniform() < ONBOARD_DROP_FRAC:
            dropped["building"].append(lm["id"])
            continue
        if t == "tree" and rng.uniform() < ONBOARD_DROP_FRAC:
            dropped["tree"].append(lm["id"])
            continue
        if t in ("building", "tower"):
            lm["height"] = round(lm["height"] * rng.uniform(1 - ONBOARD_HEIGHT_ERR, 1 + ONBOARD_HEIGHT_ERR), 2)
            scaled += 1
        if t == "tree":
            lm["centroid_enu"] = [round(v + rng.normal(0, ONBOARD_TREE_JITTER_M), 2) for v in lm["centroid_enu"]]
            lm["height"] = round(lm["height"] * rng.uniform(0.8, 1.2), 2)
        lm.pop("base_z", None)  # recomputed on the onboard DTM
        out.append(lm)
    return out, {"buildings_removed": dropped["building"], "trees_removed": dropped["tree"], "heights_scaled": scaled}


# ------------------------------------------------------------------------------------------- main entry


def _is_complete(mdir: Path, inputs: dict[str, Any]) -> bool:
    rep = mdir / "build_report.json"
    if not (rep.exists() and (mdir / "onboard" / "manifest.json").exists() and (mdir / "world" / "manifest.json").exists()):
        return False
    try:
        return json.loads(rep.read_text(encoding="utf-8")).get("inputs") == inputs
    except ValueError:
        return False


def prepare_mission(a: LatLon = DEFAULT_A, b: LatLon = DEFAULT_B, radius_m: float = DEFAULT_RADIUS_M,
                    mission_id: str | None = None, *, force: bool = False, offline: bool = False,
                    no_lidar: bool = False, log: Callable[[str], None] = print,
                    downloads: dict[str, Callable[..., Any]] | None = None,
                    progress: Callable[[float, str], None] | None = None) -> dict[str, Any]:
    """Build (or reuse) $SFHACK_DATA/missions/<id>. Returns the build report. `downloads` replaces layer
    fetchers by name (osm, ortho, terrain, lidar; same signatures as the defaults): offline tests."""
    prog = progress or (lambda frac, stage: None)
    t_start = time.time()
    mid = mission_id or default_mission_id(a, b, radius_m)
    mdir = mission_dir(mid)
    inputs = {"format": FORMAT, "a": [a.lat, a.lon], "b": [b.lat, b.lon], "radius_m": float(radius_m),
              "no_lidar": no_lidar}
    if not force and _is_complete(mdir, inputs):
        rep = json.loads((mdir / "build_report.json").read_text(encoding="utf-8"))
        log(f"mission {mid}: already prepared ({rep['total_seconds']:.1f} s build), reusing {mdir}")
        prog(1.0, "cached")
        return {**rep, "cached": True}

    origin, extent, bbox = frame_for(a, b)
    g_ortho, g_dem, g_hf = Grid.covering(extent, ORTHO_RES), Grid.covering(extent, DEM_RES), Grid.covering(extent, HFIELD_RES)
    a_xy, b_xy = np.array(origin.to_enu(a.lat, a.lon)), np.array(origin.to_enu(b.lat, b.lon))
    log(f"mission {mid}: A ({a.lat:.5f}, {a.lon:.5f}) -> B ({b.lat:.5f}, {b.lon:.5f}), "
        f"{float(np.hypot(*(b_xy - a_xy))):.0f} m, search radius {radius_m:.0f} m")
    log(f"bbox {extent['max_x'] - extent['min_x']:.0f} x {extent['max_y'] - extent['min_y']:.0f} m "
        f"(A/B + {MARGIN_M:.0f} m): S {bbox.south:.5f} W {bbox.west:.5f} N {bbox.north:.5f} E {bbox.east:.5f}")
    prog(0.02, "download")

    f = Fetcher(offline=offline, log=log)
    layers: dict[str, Layer] = {k: Layer(k) for k in ("osm", "ortho", "terrain", "lidar")}
    results: dict[str, Any] = {}
    deadline = time.time() + LIDAR_TIMEBOX_S
    with ThreadPoolExecutor(4) as ex:
        dl = {"osm": fetch_osm, "ortho": fetch_ortho, "terrain": fetch_dem, "lidar": _fetch_lidar, **(downloads or {})}
        jobs = {ex.submit(_timed, dl["osm"], f, bbox): "osm",
                ex.submit(_timed, dl["ortho"], f, bbox, g_ortho): "ortho",
                ex.submit(_timed, dl["terrain"], f, bbox, g_dem): "terrain"}
        if not no_lidar:
            jobs[ex.submit(_timed, dl["lidar"], f, bbox, deadline)] = "lidar"
        log("downloading OSM (Overpass), orthophoto (NAIP), terrain (3DEP), LiDAR (3DEP EPT) in parallel ...")
        done = 0
        for fut in as_completed(jobs):
            name = jobs[fut]
            val, secs, err = fut.result()
            layers[name].seconds = secs
            results[name] = val
            done += 1
            if err:
                layers[name].note = err
                log(f"  {name}: FAILED after {secs:.1f} s ({err[:200]})")
            else:
                log(f"  {name}: ok in {secs:.1f} s")
            prog(0.02 + 0.6 * done / len(jobs), f"download {name}")
    log(f"downloaded {f.bytes_net / 1e6:.1f} MB, {f.bytes_cache / 1e6:.1f} MB from cache")
    f.close()

    # ---- OSM (required)
    if results.get("osm") is None:
        raise FetchError(f"OSM data unavailable: {layers['osm'].note}")
    osm, osm_src = results["osm"]
    layers["osm"].source = osm_src.get("endpoint", "")
    sources: list[dict[str, Any]] = [{k: v for k, v in SRC_OSM.items()}]

    # ---- terrain: z datum = ground at A
    t0 = time.time()
    if results.get("terrain") is not None:
        dem, dem_src = results["terrain"]
        sources.append({k: v for k, v in dem_src.items() if k in ("name", "license", "attribution", "url")})
        layers["terrain"].source = dem_src["product"]
    else:
        dem, dem_src = np.zeros(g_dem.shape, np.float32), {"product": "flat (no DEM available)", "native_res_m": None,
                                                           "vertical_datum": "n/a"}
        layers["terrain"].source = dem_src["product"]
    z0 = float(sample_bilinear(dem, g_dem, a_xy[None])[0])
    dtm = (dem - z0).astype(np.float32)
    X, Y = g_dem.centers()
    w = np.clip((np.hypot(X - a_xy[0], Y - a_xy[1]) - PAD_RADIUS_M) / (PAD_BLEND_M - PAD_RADIUS_M), 0.0, 1.0)
    dtm = (w * dtm).astype(np.float32)  # level launch pad at z = 0 around A
    log(f"terrain: {dem_src['product']}, elevation at A {z0:.1f} m ({dem_src['vertical_datum']}) = z 0, "
        f"relief {float(dtm.min()):+.1f} .. {float(dtm.max()):+.1f} m")

    # ---- LiDAR -> ENU, vertical check against the DEM
    cloud_xyz = cloud_cls = None
    lidar_info: dict[str, Any] = {}
    if results.get("lidar") is not None:
        ll, cls, lidar_info = results["lidar"]
        cloud_xyz = L.to_enu(ll, origin, z0)
        ground = cls == L.GROUND
        if ground.sum() > 100:
            off = float(np.median(cloud_xyz[ground, 2] - sample_bilinear(dtm, g_dem, cloud_xyz[ground, :2])))
            lidar_info["vertical_offset_to_dem_m"] = round(off, 2)
            if abs(off) > 0.5:
                cloud_xyz[:, 2] -= off
                log(f"lidar: shifted {-off:+.2f} m vertically to the DEM datum")
        cloud_cls = cls
        layers["lidar"].source = f"real 3DEP LiDAR: {lidar_info['collection_name']} (depth {lidar_info['depth']}, " \
                                 f"{lidar_info['spacing_m']} m)"
        sources.append({k: v for k, v in L.SRC_LIDAR.items()})
        log(f"lidar: {len(cloud_xyz):,} points from {lidar_info['collection_name']} ({lidar_info['nodes']} EPT nodes)")
    else:
        layers["lidar"].source = "synthetic (sampled from the onboard building meshes + terrain)"
        layers["lidar"].note = layers["lidar"].note or "disabled (--no-lidar)"
    prog(0.7, "landmarks")

    # ---- truth landmarks: base on the terrain, untagged heights from the LiDAR where there is one
    landmarks = extract_landmarks(osm, origin, bbox)
    nd = lidar_dsm(cloud_xyz, g_dem) if cloud_xyz is not None else None
    n_lidar_h = 0
    for lm in landmarks:
        lm["base_z"] = round(_footprint_base(lm, dtm, g_dem), 2)
        if nd is not None and lm["type"] == "building" and lm.get("height_source") == "default" and lm.get("footprint_enu"):
            m = _local_mask(g_dem, np.asarray(lm["footprint_enu"], float), pad=0)
            if m is not None:
                mask, rs, cs = m
                mask = cv2.erode(mask, np.ones((3, 3), np.uint8)) if mask.sum() > 25 else mask
                vals = nd[rs, cs][mask > 0]
                vals = vals[np.isfinite(vals)]
                if len(vals) >= 4:
                    h = float(np.percentile(vals, 90)) - lm["base_z"]
                    if 2.5 <= h <= 120.0:
                        lm["height"], lm["height_source"] = round(h, 2), "lidar"
                        n_lidar_h += 1
    counts = {k: sum(1 for lm in landmarks if lm["type"] == k) for k in ("building", "tower", "fountain", "tree")}
    hs = {k: sum(1 for lm in landmarks if lm["type"] == "building" and lm.get("height_source") == k)
          for k in ("tag", "levels", "lidar", "default")}
    log(f"osm: {counts}; building heights from {hs}")

    # ---- ortho
    if results.get("ortho") is not None:
        ortho, ortho_src = results["ortho"]
        sources.append({k: v for k, v in ortho_src.items() if k in ("name", "license", "attribution", "url")})
        layers["ortho"].source = ortho_src["name"]
    else:
        ortho = render_osm_ground(osm, origin, extent, (g_ortho.cols, g_ortho.rows))
        ortho_src = {"name": "OSM-rendered (no imagery available)"}
        layers["ortho"].source = ortho_src["name"]
        log("ortho: no imagery, rendered a stylized ground from OSM (vision localizer will be weak)")
    layers["terrain"].seconds += time.time() - t0

    # ---- write the truth world
    prog(0.75, "world")
    t0 = time.time()
    rng = np.random.default_rng(int(hashlib.sha1(mid.encode()).hexdigest()[:8], 16))
    if mdir.exists():
        shutil.rmtree(mdir)
    wdir, odir = mdir / "world", mdir / "onboard"
    wdir.mkdir(parents=True)
    odir.mkdir(parents=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    b_ground_truth = float(sample_bilinear(dtm, g_dem, b_xy[None])[0])
    attribution = " · ".join(s["attribution"] for s in sources)
    Image.fromarray(ortho).save(wdir / "ortho.jpg", quality=92)
    terrain_hf = cv2.resize(dtm, (g_hf.cols, g_hf.rows), interpolation=cv2.INTER_AREA)
    np.savez_compressed(wdir / "terrain.npz", z=terrain_hf.astype(np.float32), **{k: np.float64(v) for k, v in g_hf.meta().items()})
    (wdir / "landmarks.json").write_text(json.dumps(landmarks, separators=(",", ":")), encoding="utf-8")
    target = {"status": "pending", "center_enu": [round(float(v), 2) for v in b_xy], "radius_m": float(radius_m),
              "ground_z": round(b_ground_truth, 2), "objects": [],
              "object_format": {"mesh": "STL path, absolute or relative to world/", "pos": "[x, y, z] ENU m (z optional: "
                                "true ground)", "yaw_deg": "about +z, 0 = +x (east), CCW", "scale": 1.0, "rgba": [0.85, 0.85, 0.85, 1]}}
    (wdir / "target_scene.json").write_text(json.dumps(target, indent=2), encoding="utf-8")
    common = {
        "bbox": {"south": bbox.south, "west": bbox.west, "north": bbox.north, "east": bbox.east},
        "origin": {"lat": origin.lat, "lon": origin.lon},
        "coordinate_frame": "local ENU (x=east, y=north, z=up) in meters, equirectangular around origin",
        "z_datum": f"z = 0 at the ground under A = {z0:.2f} m {dem_src['vertical_datum']}",
        "z0_elevation_m": round(z0, 3),
    }
    ortho_meta = {"file": "ortho.jpg", "source": ortho_src["name"], "width_px": g_ortho.cols, "height_px": g_ortho.rows,
                  "extent_enu": {k: extent[k] for k in ("min_x", "min_y", "max_x", "max_y")},
                  "m_per_px": [ORTHO_RES, ORTHO_RES]}
    world_manifest = {
        "version": 2, "format": FORMAT, "kind": "world (simulation truth)", "package_id": f"{mid}-world",
        "created_at": now, **common,
        "a": {"lat": a.lat, "lon": a.lon, "enu": [round(float(v), 2) for v in a_xy], "ground_z": 0.0},
        "b": {"lat": b.lat, "lon": b.lon, "enu": [round(float(v), 2) for v in b_xy], "ground_z": round(b_ground_truth, 2)},
        "terrain": {"file": "terrain.npz", "source": dem_src["product"], **g_hf.meta()},
        "buildings": "extruded",
        "ortho": ortho_meta,
        "sources": sources, "attribution": attribution, "landmark_counts": counts, "hand_made": False,
        "files": ["manifest.json", "landmarks.json", "ortho.jpg", "terrain.npz", "target_scene.json"],
    }
    (wdir / "manifest.json").write_text(json.dumps(world_manifest, indent=2), encoding="utf-8")
    t_world = time.time() - t0

    # ---- onboard package
    prog(0.82, "onboard")
    t0 = time.time()
    ob_lms, diff = onboard_landmarks(landmarks, rng)
    ob_dtm = (dtm + smooth_noise(dtm.shape, 20.0 / DEM_RES, ONBOARD_DTM_NOISE_M, rng)).astype(np.float32)
    for lm in ob_lms:
        lm["base_z"] = round(_footprint_base(lm, ob_dtm, g_dem), 2)
    ob_dsm = burn_dsm(ob_dtm, g_dem, ob_lms).astype(np.float32)
    Image.fromarray(degrade_ortho(ortho, rng)).save(odir / "ortho.jpg", quality=80)
    (odir / "landmarks.json").write_text(json.dumps(ob_lms, separators=(",", ":")), encoding="utf-8")
    np.savez_compressed(odir / "dsm.npz", dsm=ob_dsm, dtm=ob_dtm, **{k: np.float64(v) for k, v in g_dem.meta().items()})
    t_dsm = time.time() - t0

    # map_cloud.ply
    t0 = time.time()
    if cloud_xyz is not None:
        cxyz, ccls = L.voxel_downsample(cloud_xyz, CLOUD_VOXEL, cloud_cls)
        cloud_kind = "real"
    else:
        footprinted = [lm for lm in ob_lms if lm.get("footprint_enu")]
        trees = [lm for lm in ob_lms if lm["type"] == "tree"]
        raw, rcls = L.synthetic_cloud(ob_dtm, g_dem, footprinted, trees, CLOUD_VOXEL * 0.7, rng, CLOUD_NOISE_M)
        cxyz, ccls = L.voxel_downsample(raw, CLOUD_VOXEL, rcls)
        cloud_kind = "synthetic"
    L.write_ply(odir / "map_cloud.ply", cxyz, ccls,
                [f"sfhack map_cloud {cloud_kind}", "frame local ENU meters, see manifest.json origin / z_datum",
                 f"voxel {CLOUD_VOXEL} m"])
    log(f"map_cloud.ply: {cloud_kind}, {len(cxyz):,} points after {CLOUD_VOXEL} m voxel filter")
    t_cloud = time.time() - t0 + layers["lidar"].seconds

    # mission.json (planned on the onboard map only)
    mc = MissionConfig()
    b_ground = float(sample_bilinear(ob_dtm, g_dem, b_xy[None])[0])
    b_surface = float(sample_bilinear(ob_dsm, g_dem, b_xy[None])[0])
    warnings = []
    if b_surface - b_ground > 1.5:
        warnings.append(f"B is under/on a mapped object ({b_surface - b_ground:.1f} m above ground): pick open ground")
    route, legs = plan_route(a_xy, b_xy, 0.0, b_ground, ob_dsm, ob_dtm, g_dem, mc)
    mission = {
        "version": 1, "format": FORMAT, "mission_id": mid, "created_at": now,
        "frame": "local ENU (x=east, y=north, z=up) m; origin and z datum in manifest.json",
        "a": {"lat": a.lat, "lon": a.lon, "enu": [round(float(v), 2) for v in a_xy], "ground_z": 0.0},
        "b": {"lat": b.lat, "lon": b.lon, "enu": [round(float(v), 2) for v in b_xy], "ground_z": round(b_ground, 2),
              "search_radius_m": float(radius_m)},
        "cruise": {"speed_m_s": mc.cruise_speed, "clearance_m": CLEARANCE_M, "min_agl_m": MIN_AGL_M,
                   "obstacle_margin_m": OBSTACLE_MARGIN_M, "max_leg_m": LEG_MAX_M,
                   "max_z": max(w["enu"][2] for w in route), "legs": legs},
        "route": route,
        "target_scene": {
            "status": "pending",
            "center_enu": [round(float(v), 2) for v in b_xy], "radius_m": float(radius_m),
            "ground_z": round(b_ground, 2),
            "objects": [],
            "note": "Search area around B (the drone knows only this). The object placer puts the objects into "
                    "world/target_scene.json (truth, loaded by the simulator); `objects` here stays empty. Nothing "
                    "is placed yet.",
        },
        "warnings": warnings,
    }
    (odir / "mission.json").write_text(json.dumps(mission, indent=2), encoding="utf-8")
    for w in warnings:
        log(f"WARNING: {w}")
    log("route: " + " -> ".join(f"{w['phase']}({w['enu'][2]:.0f})" for w in route))

    files = {
        "ortho": ("ortho.jpg", "orthophoto, north-up, RGB JPEG", {"res_m": ORTHO_RES}),
        "landmarks": ("landmarks.json", "OSM landmarks (onboard copy)", {}),
        "dsm": ("dsm.npz", "DTM + DSM rasters for TRN, float32", {"res_m": DEM_RES}),
        "map_cloud": ("map_cloud.ply", f"3D point cloud ({cloud_kind}) for LiDAR scan-to-map", {"voxel_m": CLOUD_VOXEL,
                                                                                               "points": int(len(cxyz))}),
        "mission": ("mission.json", "A, B, search radius, route, target_scene", {}),
    }
    layer_meta = {}
    for key, (fn, desc, extra) in files.items():
        p = odir / fn
        layer_meta[key] = {"file": fn, "description": desc, "bytes": p.stat().st_size, "sha256": sha256(p), **extra}
    onboard_manifest = {
        "version": 2, "format": FORMAT, "kind": "onboard map package", "package_id": mid, "created_at": now, **common,
        "a": mission["a"], "b": mission["b"],
        "ortho": {**ortho_meta, "processing": "photometric copy of the world ortho (gamma, contrast, saturation, "
                                              "0.5 px blur, JPEG q80)"},
        "dsm": {"file": "dsm.npz", **g_dem.meta()},
        "map_cloud": {"file": "map_cloud.ply", "kind": cloud_kind, "voxel_m": CLOUD_VOXEL, "points": int(len(cxyz)),
                      **({"lidar": lidar_info} if cloud_kind == "real" else {})},
        "layers": layer_meta,
        "differences_from_truth": {
            "building_height_scale": [1 - ONBOARD_HEIGHT_ERR, 1 + ONBOARD_HEIGHT_ERR],
            "buildings_removed": diff["buildings_removed"], "trees_removed": diff["trees_removed"],
            "tree_position_jitter_m": ONBOARD_TREE_JITTER_M, "dtm_noise_std_m": ONBOARD_DTM_NOISE_M,
            "ortho": "photometric changes + blur + JPEG q80 (geometry identical)",
            "map_cloud": ("real 3DEP LiDAR: a different source than the OSM-extruded world" if cloud_kind == "real"
                          else "sampled from the onboard (perturbed) meshes + DTM with noise, voxel filtered"),
        },
        "sources": sources, "attribution": attribution, "landmark_counts": {k: sum(1 for lm in ob_lms if lm["type"] == k)
                                                                            for k in counts},
        "files": ["manifest.json"] + [v[0] for v in files.values()],
    }
    (odir / "manifest.json").write_text(json.dumps(onboard_manifest, indent=2), encoding="utf-8")
    t_onboard = time.time() - t0

    total = time.time() - t_start
    report = {
        "mission_id": mid, "dir": str(mdir), "inputs": inputs, "created_at": now, "total_seconds": round(total, 1),
        "downloaded_mb": round(f.bytes_net / 1e6, 1), "cache_mb": round(f.bytes_cache / 1e6, 1),
        "layers": {
            "osm": {"seconds": round(layers["osm"].seconds, 1), "source": layers["osm"].source, "note": layers["osm"].note},
            "ortho": {"seconds": round(layers["ortho"].seconds, 1), "source": layers["ortho"].source, "note": layers["ortho"].note},
            "terrain": {"seconds": round(layers["terrain"].seconds, 1), "source": layers["terrain"].source,
                        "note": layers["terrain"].note},
            "lidar": {"seconds": round(t_cloud, 1), "source": layers["lidar"].source, "note": layers["lidar"].note,
                      "kind": cloud_kind},
            "world_build": {"seconds": round(t_world, 1)},
            "onboard_dsm": {"seconds": round(t_dsm, 1)},
            "onboard_total": {"seconds": round(t_onboard + t_dsm, 1)},
        },
        "sizes_bytes": {**{f"world/{p.name}": p.stat().st_size for p in sorted(wdir.iterdir())},
                        **{f"onboard/{p.name}": p.stat().st_size for p in sorted(odir.iterdir())}},
        "warnings": warnings,
    }
    (mdir / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"mission {mid} prepared in {total:.1f} s -> {mdir}")
    prog(1.0, "done")
    return report


def summarize(report: dict[str, Any]) -> str:
    lines = [f"mission {report['mission_id']}: {report['total_seconds']} s, {report['downloaded_mb']} MB downloaded"]
    for k, v in report["layers"].items():
        lines.append(f"  {k:14s} {v.get('seconds', 0):6.1f} s  {v.get('source', '')}  {v.get('note', '')[:80]}")
    for k, v in report["sizes_bytes"].items():
        lines.append(f"  {k:26s} {v / 1e6:8.2f} MB")
    return "\n".join(lines)

