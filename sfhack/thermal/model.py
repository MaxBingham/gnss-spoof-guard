"""Thermal world model: what a downward LWIR camera sees at night, as truth (simulator) and prior (drone).

    build_thermal(mission_dir, hour=23.0, season="summer", seed=0) writes
      world/thermal_truth.npz    TRUTH, simulator only: temp_K (apparent K, 0.5 m, ortho grid), wall_K, air_K, meta
      onboard/thermal_prior.npz  the drone's prediction: temp_K, sigma_K, cls, roads, road_kind, roads_json, meta
      (+ drone/thermal_prior.npz when the mission has a verified upload in drone/)

Pipeline: material classes (classify.py) from ortho + NIR + OSM + landmarks -> per-pixel sun exposure and sky
view factor from a DSM (shading.py) -> 1-D surface energy balance per material (seb.py), evaluated at the local
solar time of `hour` -> apparent temperature.

Truth and prior run the same physics on different inputs: the truth on world/ (original ortho, true landmarks,
true terrain) plus what no map knows (roof construction, parked and recently driven cars, wet and shaded road
patches, rooftop HVAC, per-road and per-class model error, sensor-scale texture). The prior on onboard/ (the
degraded ortho, perturbed landmarks, onboard DSM, OSM with some minor ways missing) with a per-pixel sigma.

This is mission-preparation tooling (it makes the truth). Onboard code reads only the prior, via prior.py.
"""

from __future__ import annotations

import io
import json
import math
import shutil
import time
import zlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ..geo import BBox, GeoOrigin
from ..missionprep.build import burn_dsm, smooth_noise
from ..missionprep.geometry import Grid, resample
from .classify import ROAD_KIND, ClassMap, classify
from .materials import CLASS_NAMES, CLS, MATERIALS, ROOF_CLASSES, SEASONS, Climate
from .seb import Curves, simulate
from .shading import sky_view, sun_exposure

RES = 0.5
FORMAT = "sfhack-thermal/1"
UNTAGGED_PAVED_P = 0.4  # outside the US: share of minor roads without a surface tag that are paved (truth draw)


# ------------------------------------------------------------------------------------------------ inputs


def _read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _read_rgb(p: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _naip_nir(bbox: BBox, grid: Grid, log: Any) -> tuple[np.ndarray | None, dict[str, Any]]:
    """NAIP band 4 (NIR) on the ortho grid (cached download). None outside NAIP coverage."""
    from ..missionprep.sources import MAX_EXPORT_PX, NAIP_EXPORT, FetchError, Fetcher

    s = min(1.0, MAX_EXPORT_PX / max(grid.cols, grid.rows))
    rw, rh = max(1, round(grid.cols * s)), max(1, round(grid.rows * s))
    params = {"bbox": f"{bbox.west},{bbox.south},{bbox.east},{bbox.north}", "bboxSR": 4326, "imageSR": 4326,
              "size": f"{rw},{rh}", "format": "png", "bandIds": "3", "pixelType": "U8",
              "interpolation": "RSP_BilinearInterpolation", "f": "image"}
    f = Fetcher()
    try:
        img = Image.open(io.BytesIO(f.get(NAIP_EXPORT, params, timeout=90.0))).convert("L")
    except (FetchError, OSError) as e:
        log(f"  NIR: NAIP band 4 not available ({str(e)[:100]})")
        return None, {}
    finally:
        f.close()
    a = np.asarray(img)
    if a.shape != (rh, rw) or float(a.std()) < 2.0:
        return None, {}
    if a.shape != grid.shape:
        a = cv2.resize(a, (grid.cols, grid.rows), interpolation=cv2.INTER_LINEAR)
    return a, {"file": "nir.png", "source": "USDA NAIP band 4 (NIR) via USGS NAIP ImageServer", "native_res_m": 0.6}


def load_nir(mdir: Path, which: str, bbox: BBox, grid: Grid, log: Any) -> tuple[np.ndarray | None, str]:
    """<mission>/<which>/nir.png (missionprep writes it for new missions); for older missions fetch NAIP band 4
    once and ADD world/nir.png + onboard/nir.png (+ drone/) in the same format. Missing NIR -> None."""
    p = mdir / which / "nir.png"
    if p.exists():
        return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), "file"
    if which != "world" and not (mdir / "world" / "nir.png").exists():
        return None, "none"
    world_p = mdir / "world" / "nir.png"
    if not world_p.exists():
        man = _read_json(mdir / "world" / "manifest.json")
        if "NAIP" not in man.get("ortho", {}).get("source", ""):
            return None, "none"
        nir, _ = _naip_nir(bbox, grid, log)
        if nir is None:
            return None, "none"
        cv2.imwrite(str(world_p), nir)
        # onboard copy: same pixels, slightly blurred and re-gained (like the onboard ortho)
        rng = np.random.default_rng(zlib.crc32(mdir.name.encode()) + 77)
        nb = np.clip(cv2.GaussianBlur(nir.astype(np.float32), (0, 0), 0.5) * rng.uniform(0.92, 0.98) + 2, 0, 255)
        for d in ("onboard", "drone"):
            if (mdir / d).is_dir() and not (mdir / d / "nir.png").exists():
                cv2.imwrite(str(mdir / d / "nir.png"), nb.astype(np.uint8))
        log("  NIR: fetched NAIP band 4, added world/nir.png + onboard/nir.png")
    nir = cv2.imread(str(mdir / which / "nir.png"), cv2.IMREAD_GRAYSCALE) if (mdir / which / "nir.png").exists() else None
    return nir, "naip" if nir is not None else "none"


def load_osm(manifest: dict[str, Any], mdir: Path, log: Any) -> dict[str, Any] | None:
    """Raw OSM elements: world/osm.json if the mission carries it, else the planner's cached download."""
    for p in (mdir / "world" / "osm.json", mdir / "osm.json"):
        if p.exists():
            return _read_json(p)
    from ..missionprep.sources import FetchError, Fetcher, fetch_osm

    b = manifest["bbox"]
    bbox = BBox(b["south"], b["west"], b["north"], b["east"])
    for offline in (True, False):
        f = Fetcher(offline=offline)
        try:
            osm, _ = fetch_osm(f, bbox)
            return osm
        except FetchError as e:
            log(f"  OSM ({'cache' if offline else 'network'}) failed: {str(e)[:120]}")
        finally:
            f.close()
    return None


def truth_dsm(mdir: Path, g1: Grid, landmarks: list[dict[str, Any]]) -> np.ndarray:
    with np.load(mdir / "world" / "terrain.npz") as z:
        t = np.asarray(z["z"], np.float32)
        tg = Grid(float(z["min_x"]), float(z["max_y"]), float(z["res_m"]), *t.shape)
    dtm = resample(t, tg, g1)
    return burn_dsm(dtm, g1, [lm for lm in landmarks if "base_z" in lm])


def canopy_height(mdir: Path, g1: Grid, grid: Grid) -> np.ndarray | None:
    """Canopy height model on the 0.5 m grid from onboard/map_cloud.ply when it is REAL LiDAR: max return per
    1 m cell minus the onboard DTM. Building returns stay in (roofs are overwritten by footprints later; an
    unmapped building is not vegetation in NDVI). None for a synthetic cloud (it only knows the OSM trees)."""
    from ..missionprep.lidar import read_ply

    p = mdir / "onboard" / "map_cloud.ply"
    if not p.exists():
        return None
    man = _read_json(mdir / "onboard" / "manifest.json")
    if man.get("map_cloud", {}).get("kind") != "real":
        return None
    xyz, _ = read_ply(p)
    uv = np.round(g1.to_px(xyz[:, :2])).astype(int)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < g1.cols) & (uv[:, 1] >= 0) & (uv[:, 1] < g1.rows)
    top = np.full(g1.rows * g1.cols, -np.inf)
    np.maximum.at(top, uv[ok, 1] * g1.cols + uv[ok, 0], xyz[ok, 2])
    top = top.reshape(g1.shape)
    with np.load(mdir / "onboard" / "dsm.npz") as z:
        dtm = np.asarray(z["dtm"], np.float32)
        dg = Grid(float(z["min_x"]), float(z["max_y"]), float(z["res_m"]), *dtm.shape)
    dtm = dtm if dg == g1 else resample(dtm, dg, g1)
    chm = np.where(np.isfinite(top), top - dtm, 0.0).astype(np.float32)
    chm = np.maximum(chm, 0.0)
    # fill single empty cells (voxel filter gaps) from the 3x3 max
    filled = cv2.dilate(chm, np.ones((3, 3), np.uint8))
    chm = np.where(np.isfinite(top), chm, filled)
    return _up2(chm, grid.shape)


def onboard_dsm(mdir: Path, g1: Grid) -> np.ndarray:
    with np.load(mdir / "onboard" / "dsm.npz") as z:
        d = np.asarray(z["dsm"], np.float32)
        dg = Grid(float(z["min_x"]), float(z["max_y"]), float(z["res_m"]), *d.shape)
    return d if dg == g1 else resample(d, dg, g1)


# ------------------------------------------------------------------------------------------------ physics


def solar_hour(clock_hour: float, lon: float, season: str) -> tuple[float, float]:
    """Local clock time -> local solar time. Time zone = round(lon / 15), +1 h daylight saving in summer
    (Berkeley PDT -7, Ukraine EEST +3). Equation of time ignored (< 0.25 h)."""
    tz = round(lon / 15.0) + (1 if season == "summer" else 0)
    return (clock_hour + lon / 15.0 - tz) % 24.0, float(tz)


def temperature_map(cls: np.ndarray, exposure: np.ndarray, svf: np.ndarray, table: np.ndarray,
                    exps: np.ndarray, svfs: np.ndarray) -> np.ndarray:
    """Per pixel: bilinear in (exposure, svf) of the class's (n_exp, n_svf) table (float32, per class)."""
    e = np.clip(exposure, exps[0], exps[-1]).astype(np.float32)
    s = np.clip(svf, svfs[0], svfs[-1]).astype(np.float32)
    ie = np.clip(np.searchsorted(exps, e, side="right") - 1, 0, len(exps) - 2)
    is_ = np.clip(np.searchsorted(svfs, s, side="right") - 1, 0, len(svfs) - 2)
    fe = ((e - exps[ie]) / (exps[ie + 1] - exps[ie])).astype(np.float32)
    fs = ((s - svfs[is_]) / (svfs[is_ + 1] - svfs[is_])).astype(np.float32)
    out = np.zeros(cls.shape, np.float32)
    for k in np.unique(cls):
        m = cls == k
        t = table[int(k)].astype(np.float32)
        i, j, a, b = ie[m], is_[m], fe[m], fs[m]
        out[m] = (1 - a) * ((1 - b) * t[i, j] + b * t[i, j + 1]) + a * ((1 - b) * t[i + 1, j] + b * t[i + 1, j + 1])
    return out


def _up2(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return cv2.resize(a.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


# ------------------------------------------------------------------------------------------------ truth extras


def _rect(cx: float, cy: float, length: float, width: float, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    pts = np.array([[length / 2, width / 2], [-length / 2, width / 2], [-length / 2, -width / 2], [length / 2, -width / 2]])
    return pts @ np.array([[c, s], [-s, c]]) + [cx, cy]


def _fill(img: np.ndarray, grid: Grid, poly: np.ndarray, value: float) -> None:
    pts = np.round((grid.to_px(poly) + 0.5) * 16).astype(np.int32)
    cv2.fillPoly(img, [pts], float(value), lineType=cv2.LINE_AA, shift=4)


def place_cars(cm: ClassMap, grid: Grid, rng: np.random.Generator, t_air: float) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Parked cars along vehicle roads and in parking lots. Body: radiatively cooled below air (glass and roof see
    the cold sky); 15 % recently driven: warm hood. Returns (cars, temp raster, coverage 0..1)."""
    temp = np.zeros(grid.shape, np.float32)
    cover = np.zeros(grid.shape, np.float32)
    cars: list[dict[str, Any]] = []
    occ = {"residential": 0.35, "unclassified": 0.3, "tertiary": 0.25, "service": 0.2, "living_street": 0.35,
           "secondary": 0.12, "primary": 0.08}

    def add(x: float, y: float, yaw: float) -> None:
        uv = grid.to_px(np.array([x, y]))
        if not (0 <= uv[0] < grid.cols and 0 <= uv[1] < grid.rows):
            return
        r, c = int(uv[1]), int(uv[0])
        if cm.roads[r, c] == 0 and cm.road_kind[r, c] != ROAD_KIND["parking"]:
            return  # under a tree / on a roof / off the pavement
        body = t_air - 2.0 + rng.normal(0, 0.7)
        hot = rng.uniform() < 0.15
        poly = _rect(x, y, 4.4, 1.8, yaw)
        m = np.zeros(grid.shape, np.float32)
        _fill(m, grid, poly, 1.0)
        sel = m > 0
        temp[sel] = body * m[sel] + temp[sel] * (1 - m[sel])
        cover[sel] = np.maximum(cover[sel], m[sel])
        if hot:
            hood = _rect(x + 1.5 * math.cos(yaw), y + 1.5 * math.sin(yaw), 1.3, 1.6, yaw)
            hm = np.zeros(grid.shape, np.float32)
            _fill(hm, grid, hood, 1.0)
            hs = hm > 0
            temp[hs] = t_air + rng.uniform(6, 14)
        cars.append({"xy": [round(x, 2), round(y, 2)], "yaw": round(yaw, 3), "hot": bool(hot)})

    for w in cm.roads_json:
        p = occ.get(w["highway"])
        if p is None or not w["paved"]:
            continue
        pts = np.asarray(w["pts_enu"], float)
        for a, b in zip(pts[:-1], pts[1:]):
            d = b - a
            L = float(np.hypot(*d))
            if L < 6:
                continue
            u = d / L
            n = np.array([-u[1], u[0]])
            yaw = math.atan2(u[1], u[0])
            off = w["width_m"] / 2 - 1.1
            for s in np.arange(4.0, L - 4.0, 6.0):
                for side in (-1, 1):
                    if rng.uniform() < p:
                        x, y = a + u * s + n * side * off
                        add(float(x), float(y), yaw + (math.pi if rng.uniform() < 0.5 else 0.0))
    # parking lots: stalls at random positions over parking pixels
    pk = np.argwhere(cm.road_kind == ROAD_KIND["parking"])
    if len(pk):
        n = int(len(pk) * RES * RES / 30.0 * 0.5)
        for r, c in pk[rng.choice(len(pk), size=min(n, len(pk)), replace=False)]:
            x = grid.min_x + (c + 0.5) * grid.res
            y = grid.max_y - (r + 0.5) * grid.res
            add(x, y, rng.uniform(0, math.pi))
    return cars, temp, cover


def _blobs(grid: Grid, centers: np.ndarray, radii: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.zeros(grid.shape, np.float32)
    for (x, y), (ra, rb), v in zip(centers, radii, values):
        uv = grid.to_px(np.array([x, y]))
        cv2.ellipse(out, (int(uv[0]), int(uv[1])), (max(1, int(ra / grid.res)), max(1, int(rb / grid.res))),
                    float(np.degrees(np.arctan2(rb, ra) * 7) % 180), 0, 360, float(v), -1)
    return cv2.GaussianBlur(out, (0, 0), 2.0)


# ------------------------------------------------------------------------------------------------ main


def build_thermal(mission_dir: str | Path, hour: float = 23.0, season: str = "summer", seed: int = 0,
                  climate: Climate | None = None, log: Any = print) -> dict[str, Any]:
    t0 = time.time()
    mdir = Path(mission_dir)
    if not (mdir / "world" / "manifest.json").exists():
        from ..missionprep.build import mission_dir as _mdir

        mdir = _mdir(str(mission_dir))
    wman = _read_json(mdir / "world" / "manifest.json")
    e = wman["ortho"]["extent_enu"]
    grid = Grid.covering(e, RES)
    g1 = Grid.covering(e, 1.0)
    origin = GeoOrigin(wman["origin"]["lat"], wman["origin"]["lon"])
    b = wman["bbox"]
    bbox = BBox(b["south"], b["west"], b["north"], b["east"])
    climate = climate or SEASONS[season]
    rng = np.random.default_rng(zlib.crc32(mdir.name.encode()) * 1000 + seed)
    info: dict[str, Any] = {"mission": mdir.name, "hour": hour, "season": season, "seed": seed}

    osm = load_osm(wman, mdir, log)
    info["osm_elements"] = len(osm["elements"]) if osm else 0
    w_rgb = _read_rgb(mdir / "world" / "ortho.jpg")
    o_rgb = _read_rgb(mdir / "onboard" / "ortho.jpg")
    w_nir, nir_src = load_nir(mdir, "world", bbox, grid, log)
    o_nir, _ = load_nir(mdir, "onboard", bbox, grid, log)
    for name, a in (("world ortho", w_rgb), ("onboard ortho", o_rgb), ("world nir", w_nir), ("onboard nir", o_nir)):
        if a is not None and a.shape[:2] != grid.shape:
            raise ValueError(f"{name} {a.shape[:2]} != ortho grid {grid.shape}")
    info["nir"] = nir_src
    w_lm = _read_json(mdir / "world" / "landmarks.json")
    o_lm = _read_json(mdir / "onboard" / "landmarks.json")

    # material classes: truth from world/, prior from onboard/ (5 % of minor OSM ways unknown onboard)
    chm = canopy_height(mdir, g1, grid)
    info["canopy"] = "lidar" if chm is not None else "spectral"
    us = any(k in wman["ortho"].get("source", "") for k in ("NAIP", "USGS"))
    native = float((wman.get("nir") or {}).get("native_res_m", 0.6 if us else 10.0))
    region = {"untagged_paved_p": None if us else UNTAGGED_PAVED_P, "coarse": native > 2.0}
    info.update({"region": "us" if us else "intl", "imagery_native_res_m": native})
    cm_t = classify(w_rgb, w_nir, grid, osm, origin, w_lm, rng=np.random.default_rng(1), chm=chm, truth=True, **region)
    cm_p = classify(o_rgb, o_nir, grid, osm, origin, o_lm, drop_minor_frac=0.05, rng=np.random.default_rng(seed + 2),
                    chm=chm, **region)
    log(f"  classes: truth {cm_t.stats['class_frac']}")

    # shading on 1 m DSMs
    lat = origin.lat
    dsm_t = truth_dsm(mdir, g1, w_lm)
    dsm_p = onboard_dsm(mdir, g1)
    exp_t, svf_t = _up2(sun_exposure(dsm_t, 1.0, lat, climate.doy), grid.shape), _up2(sky_view(dsm_t, 1.0), grid.shape)
    exp_p, svf_p = _up2(sun_exposure(dsm_p, 1.0, lat, climate.doy), grid.shape), _up2(sky_view(dsm_p, 1.0), grid.shape)

    # energy balance curves -> table at the solar hour
    h_sol, tz = solar_hour(hour, origin.lon, season)
    curves: Curves = simulate(lat, climate)
    table = curves.at(h_sol)  # (n_mat, n_exp, n_svf) apparent K
    t_air = curves.air_at(h_sol)
    info.update({"hour_solar": round(h_sol, 3), "utc_offset_h": tz, "air_K": round(t_air, 2)})

    prior = temperature_map(cm_p.cls, exp_p, svf_p, table, curves.exposures, curves.svfs)
    truth_cls = cm_t.cls.copy()

    # ---- truth deviations the prior does not know
    # roof construction per building (prior guesses dark / light from the ortho only)
    for bid in np.unique(cm_t.bldg_id[cm_t.bldg_id >= 0]):
        m = cm_t.bldg_id == bid
        light = cm_t.cls[m][0] == CLS["roof_light"]
        p = [0.10, 0.65, 0.15, 0.10] if light else [0.50, 0.10, 0.35, 0.05]  # dark, light, gravel, metal
        truth_cls[m] = ROOF_CLASSES[int(rng.choice(4, p=p))]
    # irrigated vs dry grass per connected patch (watering is not in any map)
    grass = (cm_t.cls == CLS["grass"]).astype(np.uint8)
    n_lab, lab = cv2.connectedComponents(grass, connectivity=8)
    if n_lab > 1:
        flip = rng.uniform(size=n_lab) < 0.12
        flip[0] = False
        truth_cls[flip[lab] & (grass > 0)] = CLS["dry_grass"]
    truth = temperature_map(truth_cls, exp_t, svf_t, table, curves.exposures, curves.svfs)
    # per-building and per-road offsets, per-class calibration error, smooth model error, texture
    nb = int(cm_t.bldg_id.max()) + 2
    truth += np.where(cm_t.bldg_id >= 0, rng.normal(0, 1.0, nb)[cm_t.bldg_id], 0.0).astype(np.float32)
    ns = max(len(cm_t.roads_json), 1) + 1
    truth += np.where(cm_t.seg_id >= 0, rng.normal(0, 1.0, ns)[cm_t.seg_id], 0.0).astype(np.float32)
    truth += rng.normal(0, 0.5, len(MATERIALS)).astype(np.float32)[truth_cls]
    truth += smooth_noise(grid.shape, 40.0, 0.6, rng)
    truth += smooth_noise(grid.shape, 2.0, 0.3, rng) + rng.normal(0, 0.2, grid.shape).astype(np.float32)
    # wet patches (sprinklers next to lawns) and shade from unmapped street trees on paved surfaces
    paved = np.isin(truth_cls, [CLS[n] for n in ("asphalt", "asphalt_minor", "concrete", "pavers", "parking")])
    area_km2 = grid.rows * grid.cols * RES * RES / 1e6
    ys, xs = np.nonzero(paved)
    if len(ys):
        def centers(n: int) -> np.ndarray:
            k = rng.choice(len(ys), size=n)
            return np.column_stack([grid.min_x + (xs[k] + 0.5) * RES, grid.max_y - (ys[k] + 0.5) * RES])
        n_wet = max(3, int(10 * area_km2))
        wet = _blobs(grid, centers(n_wet), rng.uniform(3, 10, (n_wet, 2)), -rng.uniform(2, 4, n_wet))
        n_sh = max(5, int(40 * area_km2))
        shade = _blobs(grid, centers(n_sh), rng.uniform(4, 12, (n_sh, 2)), -rng.uniform(2.0, 4.5, n_sh))
        truth += np.where(paved, wet + shade, 0.0).astype(np.float32)
        info.update({"wet_patches": n_wet, "shade_patches": n_sh})
    # rooftop HVAC units (warm)
    n_hvac = 0
    for bid in np.unique(cm_t.bldg_id[cm_t.bldg_id >= 0]):
        rr, cc = np.nonzero(cm_t.bldg_id == bid)
        if len(rr) * RES * RES < 300 or rng.uniform() > 0.4:
            continue
        for _ in range(int(rng.integers(1, 5))):
            k = int(rng.integers(len(rr)))
            r0, c0 = rr[k], cc[k]
            truth[max(r0 - 2, 0):r0 + 2, max(c0 - 2, 0):c0 + 2] = t_air + rng.uniform(8, 20)
            n_hvac += 1
    # parked cars (on top of everything on the pavement)
    cars, car_t, car_cov = place_cars(cm_t, grid, rng, t_air)
    truth = truth * (1 - car_cov) + car_t * car_cov
    info.update({"cars": len(cars), "cars_hot": sum(c["hot"] for c in cars), "hvac_units": n_hvac})

    # ---- prior sigma: class sigma, more at class edges (misregistration) and in ortho shadows
    sig = np.array([m.prior_sigma_K for m in MATERIALS], np.float32)[cm_p.cls]
    edge = cv2.morphologyEx(cm_p.cls, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    edge = cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    sig = np.sqrt(sig ** 2 + np.where(edge, 1.5 ** 2, 0) + np.where(cm_p.uncertain, 2.0 ** 2, 0)).astype(np.float32)
    sig = np.sqrt(sig ** 2 + 0.6 ** 2)  # smooth model error the truth carries

    meta = {"format": FORMAT, "bbox": wman["bbox"], "extent_enu": {k: e[k] for k in ("min_x", "min_y", "max_x", "max_y")},
            "res_m": RES, "rows": grid.rows, "cols": grid.cols,
            "raster": "north-up, row 0 = north edge, cell centre x = min_x + (col + 0.5) * res, y = max_y - (row + 0.5) * res",
            "class_names": CLASS_NAMES, "road_kinds": ROAD_KIND, "hour": hour, "hour_solar": round(h_sol, 3),
            "season": season, "sky": "clear", "wind": "calm", "air_K": round(t_air, 2),
            "units": "apparent (radiometric, emissivity 1) temperature in K, LWIR",
            "model": "1-D surface energy balance per material x sun exposure x sky view factor (sfhack.thermal)"}
    world_meta = {**meta, "kind": "thermal truth (simulator only)", "seed": seed, "deviations": info}
    onboard_meta = {**meta, "kind": "thermal prior (onboard)", "classifier": cm_p.stats}
    np.savez_compressed(mdir / "world" / "thermal_truth.npz", temp_K=truth.astype(np.float32),
                        wall_K=np.float32(t_air + 1.5), air_K=np.float32(t_air), cls=truth_cls,
                        cars=json.dumps(cars), meta=json.dumps(world_meta))
    pr_path = mdir / "onboard" / "thermal_prior.npz"
    np.savez_compressed(pr_path, temp_K=prior.astype(np.float32), sigma_K=sig.astype(np.float32),
                        cls=cm_p.cls, roads=cm_p.roads, road_kind=cm_p.road_kind,
                        roads_json=json.dumps(cm_p.roads_json), meta=json.dumps(onboard_meta))
    if (mdir / "drone").is_dir():
        shutil.copyfile(pr_path, mdir / "drone" / "thermal_prior.npz")
    road = cm_t.roads > 0
    veg = np.isin(truth_cls, [CLS["grass"], CLS["dry_grass"], CLS["tree"]])
    info.update({
        "seconds": round(time.time() - t0, 1),
        "truth_road_K": round(float(np.median(truth[road])), 2) if road.any() else None,
        "truth_veg_K": round(float(np.median(truth[veg])), 2) if veg.any() else None,
        "prior_vs_truth_rmse_K": round(float(np.sqrt(np.mean((prior - truth) ** 2))), 2),
        "prior_vs_truth_corr": round(float(np.corrcoef(prior.ravel(), truth.ravel())[0, 1]), 3),
        "roads_frac_prior": cm_p.stats["road_frac"],
    })
    (mdir / "world" / "thermal_build.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    log(f"thermal {mdir.name}: hour {hour} (solar {h_sol:.2f}), air {t_air - 273.15:.1f} C, "
        f"road {info['truth_road_K'] and info['truth_road_K'] - 273.15:.1f} C, veg {info['truth_veg_K'] and info['truth_veg_K'] - 273.15:.1f} C, "
        f"prior rmse {info['prior_vs_truth_rmse_K']} K, corr {info['prior_vs_truth_corr']}, {info['seconds']} s")
    return info
