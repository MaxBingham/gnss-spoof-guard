"""Recon world (SIMULATOR SIDE: it writes the truth): the base at A with its fence lights, stray lights, and the
targets + distractors out in the field. RECON_PLAN sections "Base" and "Targets".

    python -m sfhack.sim.recon_world --mission ukraine-1 [--seed 0] [--bearing 62] [--dist 1800]

writes
    world/base.json, world/base_lights.json      TRUTH: base layout and light positions (scene.py renders them)
    world/stray_lights.json                      TRUTH: village lamps + a fire (not navigation lights: distractors
                                                 for the light localizer; steady or flickering, never a valid code)
    world/target_scene.json                      TRUTH: vehicles, people, distractors with thermal parts
                                                 (scene.py + thermal/camera.py render them). The first run keeps the
                                                 original as target_scene.orig.json.
    world/recon_truth.json                       TRUTH for scoring: true bearing / distance of the group, seed
    world/recon/seed_<k>/target_scene.json       the same scene per seed (FlightSystem / eval: pass the path)
    onboard/base_lights.json, onboard/base.json  SURVEYED copies (survey error `survey_sigma_m`, seeded): the only
                                                 base knowledge the drone has

Coordinates: mission ENU (z datum = ground under A). The base centre and the pad are at A (flattened launch pad).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..missionprep.geometry import Grid, sample_bilinear
from ..world.base import BaseInfo, BaseLight, BaseLights, az_deg, lyndon_codes


# ---------------------------------------------------------------------------------------------- terrain access

class Terrain:
    """TRUE ground height and extent of a mission world (world/terrain.npz, world/manifest.json)."""

    def __init__(self, world_dir: Path) -> None:
        self.dir = world_dir
        self.manifest = json.loads((world_dir / "manifest.json").read_text(encoding="utf-8"))
        with np.load(world_dir / self.manifest["terrain"]["file"]) as z:
            self.z = np.asarray(z["z"], np.float32)
            self.grid = Grid(float(z["min_x"]), float(z["max_y"]), float(z["res_m"]), *self.z.shape)
        self.a = np.array(self.manifest["a"]["enu"], float)

    def ground(self, xy: np.ndarray | list[float]) -> np.ndarray:
        return sample_bilinear(self.z, self.grid, np.atleast_2d(np.asarray(xy, float)))

    def inside(self, xy: np.ndarray, margin: float = 0.0) -> bool:
        g = self.grid
        xy = np.atleast_2d(xy)
        return bool(np.all((xy[:, 0] > g.min_x + margin) & (xy[:, 0] < g.max_x - margin)
                           & (xy[:, 1] > g.min_y + margin) & (xy[:, 1] < g.max_y - margin)))

    def max_range(self, az: float, margin: float = 0.0, step: float = 10.0) -> float:
        """Distance from A along azimuth `az` (deg, cw from north) to the map edge (minus margin)."""
        u = np.array([math.sin(math.radians(az)), math.cos(math.radians(az))])
        r = 0.0
        while self.inside(self.a + (r + step) * u, margin):
            r += step
        return r


def _unit(az: float) -> np.ndarray:
    return np.array([math.sin(math.radians(az)), math.cos(math.radians(az))])


# ---------------------------------------------------------------------------------------------- base

@dataclass
class BaseDesign:
    fence_half_m: float = 75.0  # square fence, 150 m wide, axis-aligned, centred on A
    n_fence_lights: int = 12  # one every 30 deg (rays from the centre hit the square fence)
    fence_light_h: float = 4.0  # emitter height above ground (pole on the fence)
    outer_r_m: float = 400.0  # optional outer ring
    n_outer: int = 6
    outer_light_h: float = 5.0
    edge_margin_m: float = 40.0  # outer posts only where the map has terrain (A is near the map corner)
    fence_h: float = 2.2
    pad_size_m: float = 5.0
    pad_yaw_deg: float = 0.0  # takeoff heading, cw from north
    uma16_offset: tuple[float, float] = (-12.0, 8.0)  # mast position relative to A (m E, m N)
    uma16_h: float = 6.0
    uma16_x_axis_az_deg: float = 0.0  # array x axis points north (= AcousticConfig.x_axis_bearing_deg default)
    survey_sigma_m: float = 0.05  # RTK survey of the posts (1 sigma per axis)
    survey_yaw_sigma_deg: float = 0.2


def fence_point(half: float, az: float) -> np.ndarray:
    """Point where the ray from the centre at azimuth `az` hits the square fence of half width `half`."""
    u = _unit(az)
    return u * (half / max(abs(u[0]), abs(u[1])))


def design_base(terrain: Terrain, d: BaseDesign = BaseDesign()) -> tuple[BaseLights, BaseInfo, dict[str, Any]]:
    """TRUE base layout around A. Returns (lights, base info, fence geometry for the scene)."""
    a = terrain.a
    codes = lyndon_codes()
    # spread codes: fence lights take every other code of the sorted list, outer ones the rest
    order = codes[::2] + codes[1::2]
    lights: list[BaseLight] = []
    for i in range(d.n_fence_lights):
        az = i * 360.0 / d.n_fence_lights  # F00 = north, then clockwise
        p = a + fence_point(d.fence_half_m, az)
        z = float(terrain.ground(p)[0]) + d.fence_light_h
        lights.append(BaseLight(f"F{i:02d}", "fence", (p[0], p[1], z), order[i], aim_az_deg=az, phase=(3 * i) % 8))
    outer_az = _outer_azimuths(terrain, d)
    for j, az in enumerate(outer_az):
        p = a + d.outer_r_m * _unit(az)
        z = float(terrain.ground(p)[0]) + d.outer_light_h
        lights.append(BaseLight(f"O{j:02d}", "outer", (p[0], p[1], z), order[d.n_fence_lights + j], aim_az_deg=az,
                                beam_h_deg=60.0, intensity=2.0, phase=(3 * j + 1) % 8))
    pad_z = float(terrain.ground(a)[0])
    um = a + np.asarray(d.uma16_offset)
    info = BaseInfo(center_enu=(a[0], a[1], pad_z), pad_enu=(a[0], a[1], pad_z + 0.01), pad_yaw_deg=d.pad_yaw_deg,
                    pad_size_m=d.pad_size_m, uma16_enu=(um[0], um[1], float(terrain.ground(um)[0]) + d.uma16_h),
                    uma16_x_axis_az_deg=d.uma16_x_axis_az_deg, fence_half_m=d.fence_half_m,
                    meta={"fence": "square, axis-aligned, centred on A", "fence_lights": "F00 north, clockwise, every "
                          f"{360 / d.n_fence_lights:.0f} deg", "outer_ring": f"{len(outer_az)} posts at "
                          f"{d.outer_r_m:.0f} m, only azimuths with terrain (A is near the map corner)"})
    fence = {"half_m": d.fence_half_m, "height_m": d.fence_h, "pad_size_m": d.pad_size_m,
             "pad_yaw_deg": d.pad_yaw_deg, "uma16_mast_h": d.uma16_h}
    return BaseLights(lights, meta={"kind": "truth"}), info, fence


def _outer_azimuths(terrain: Terrain, d: BaseDesign) -> list[float]:
    """n_outer azimuths evenly over the arc where a post at outer_r_m lies inside the map (full circle if all do)."""
    if d.n_outer <= 0:
        return []
    ok = [az for az in range(0, 360) if terrain.inside(terrain.a + d.outer_r_m * _unit(az), d.edge_margin_m)]
    if len(ok) == 360:
        return [k * 360.0 / d.n_outer for k in range(d.n_outer)]
    if not ok:
        return []
    # the valid arc: the longest run of consecutive valid degrees (wrapping)
    s = set(ok)
    start = next(az for az in ok if (az - 1) % 360 not in s)
    run = [start]
    while (run[-1] + 1) % 360 in s and len(run) < 360:
        run.append((run[-1] + 1) % 360)
    lo, span = run[0], len(run) - 1
    return [float((lo + span * k / max(d.n_outer - 1, 1)) % 360) for k in range(d.n_outer)]


def survey(lights: BaseLights, info: BaseInfo, d: BaseDesign, rng: np.random.Generator) -> tuple[BaseLights, BaseInfo]:
    """Onboard copies: positions with survey error."""
    s = d.survey_sigma_m

    def jit(p: tuple[float, float, float]) -> tuple[float, float, float]:
        q = np.asarray(p) + rng.normal(0.0, s, 3)
        return (float(q[0]), float(q[1]), float(q[2]))

    ol = BaseLights([replace(lt, enu=jit(lt.enu)) for lt in lights.lights], lights.bit_s, lights.code_bits, s,
                    {"kind": "onboard survey", "survey": f"RTK, {s * 100:.0f} cm 1-sigma per axis"})
    oi = replace(info, center_enu=jit(info.center_enu), pad_enu=jit(info.pad_enu), uma16_enu=jit(info.uma16_enu),
                 pad_yaw_deg=info.pad_yaw_deg + float(rng.normal(0.0, d.survey_yaw_sigma_deg)), survey_sigma_m=s,
                 meta={**info.meta, "kind": "onboard survey"})
    return ol, oi


def stray_lights(terrain: Terrain, rng: np.random.Generator, bearing_deg: float) -> list[dict[str, Any]]:
    """Visible lights that are NOT navigation lights: lamps on village houses near the base (steady), and a camp
    fire out along the bearing (flickering). TRUTH only."""
    out: list[dict[str, Any]] = []
    lm = json.loads((terrain.dir / "landmarks.json").read_text(encoding="utf-8"))
    houses = [x for x in lm if x["type"] == "building" and 90.0 < np.hypot(*(np.array(x["centroid_enu"]) - terrain.a)) < 450.0]
    for h in [houses[i] for i in rng.permutation(len(houses))[:6]]:
        c = np.asarray(h["centroid_enu"], float)
        ring = np.asarray(h.get("footprint_enu") or [c], float)
        v = ring[int(np.argmin(np.hypot(*(ring - terrain.a).T)))]  # corner towards the base
        u = (v - c) / max(float(np.hypot(*(v - c))), 1e-6)
        x, y = v + 2.0 * u  # yard lamp on a pole 2 m outside that corner
        z = float(terrain.ground([x, y])[0]) + 4.0
        out.append({"id": f"S{len(out):02d}", "kind": "lamp", "enu": [round(x, 2), round(y, 2), round(z, 2)],
                    "intensity": float(rng.uniform(0.3, 1.2)), "mode": "steady"})
    r = min(800.0, terrain.max_range(bearing_deg, 80.0))
    p = terrain.a + r * _unit(bearing_deg + float(rng.normal(0, 8))) + rng.normal(0, 30, 2)
    if terrain.inside(p, 20.0):
        out.append({"id": f"S{len(out):02d}", "kind": "fire", "enu": [round(p[0], 2), round(p[1], 2),
                    round(float(terrain.ground(p)[0]) + 0.5, 2)], "intensity": 1.5, "mode": "flicker"})
    return out


# ---------------------------------------------------------------------------------------------- targets
# Parts in the object frame (x forward, y left, z up, origin on the ground): (type, half-size, centre, dT, rgba).
# dT = apparent temperature above the air (K). Running engines 15-30 K above air on the hood / engine deck; a
# vehicle that has been driving / idling for a while is heat-soaked: body 6-8 K above air (engine bay, exhaust,
# occupants), so the whole silhouette stands out (recon-mission's detector wants >= air + 6 K over the body).
# The windscreen reflects the cold clear sky (below air); clothed people ~8-10 K.

OLIVE, DARK, GLASS = [0.29, 0.33, 0.22, 1], [0.12, 0.12, 0.12, 1], [0.15, 0.2, 0.25, 1]
VEHICLES: dict[str, list[tuple[str, list[float], list[float], float, list[float]]]] = {
    "truck": [("box", [3.0, 1.1, 0.45], [0.0, 0.0, 0.45], 9.0, DARK),       # wheels / chassis (warm brakes)
              ("box", [2.2, 1.2, 1.1], [-0.9, 0.0, 1.9], 6.5, OLIVE),       # canvas cargo box (troops inside)
              ("box", [0.8, 1.2, 1.0], [2.3, 0.0, 1.6], 8.0, OLIVE),        # cab
              ("box", [0.45, 1.05, 0.4], [3.0, 0.0, 0.95], 24.0, OLIVE),    # engine / grille
              ("cylinder", [0.07, 0.5], [1.6, -1.15, 2.2], 40.0, DARK)],    # exhaust stack
    "car": [("box", [2.25, 0.9, 0.4], [0.0, 0.0, 0.6], 7.5, OLIVE),
            ("box", [0.6, 0.85, 0.06], [1.6, 0.0, 1.05], 18.0, OLIVE),      # hood
            ("box", [0.35, 0.8, 0.25], [0.75, 0.0, 1.2], -4.0, GLASS),      # windscreen
            ("box", [0.9, 0.8, 0.25], [-0.45, 0.0, 1.25], 6.5, OLIVE)],     # cabin roof
    "apc": [("box", [3.3, 1.55, 0.4], [0.0, 0.0, 0.4], 10.0, DARK),         # tracks / wheels
            ("box", [3.4, 1.5, 0.75], [0.0, 0.0, 1.45], 7.0, OLIVE),        # hull
            ("box", [1.0, 1.3, 0.06], [-2.2, 0.0, 2.25], 20.0, OLIVE),      # engine deck (rear)
            ("cylinder", [0.7, 0.3], [0.6, 0.0, 2.5], 6.0, OLIVE)],         # turret
}
PERSON = [("cylinder", [0.25, 0.75], [0.0, 0.0, 0.75], 8.5, DARK), ("sphere", [0.13], [0.0, 0.0, 1.62], 11.0, DARK)]
DISTRACTORS: dict[str, list[tuple[str, list[float], list[float], float, list[float]]]] = {
    # a parked car that cooled for hours: engine barely warm, body radiating to the sky
    "cooling_car": [("box", [2.25, 0.9, 0.4], [0.0, 0.0, 0.6], -1.0, [0.5, 0.5, 0.52, 1]),
                    ("box", [0.6, 0.85, 0.06], [1.6, 0.0, 1.05], 3.0, [0.5, 0.5, 0.52, 1]),
                    ("box", [0.35, 0.8, 0.25], [0.75, 0.0, 1.2], -5.0, GLASS),
                    ("box", [0.9, 0.8, 0.25], [-0.45, 0.0, 1.25], -2.0, [0.5, 0.5, 0.52, 1])],
    # a barn: large, static, warm roof (livestock, dark shingles) - too big for a vehicle
    "barn": [("box", [7.0, 4.5, 2.0], [0.0, 0.0, 2.0], 1.0, [0.45, 0.25, 0.18, 1]),
             ("box", [7.2, 4.7, 0.15], [0.0, 0.0, 4.15], 4.0, [0.3, 0.2, 0.15, 1])],
    # warm rocks: sun-heated boulders still a few K above air
    "warm_rocks": [("sphere", [0.7], [0.0, 0.0, 0.3], 5.0, [0.45, 0.44, 0.42, 1]),
                   ("sphere", [0.5], [1.6, 0.8, 0.2], 5.5, [0.45, 0.44, 0.42, 1]),
                   ("box", [0.6, 0.45, 0.35], [-1.2, 1.1, 0.35], 4.5, [0.45, 0.44, 0.42, 1]),
                   ("sphere", [0.9], [0.4, -1.7, 0.4], 4.0, [0.45, 0.44, 0.42, 1])],
    # a cow: warm, vehicle-ish temperature, but small and alone
    "animal": [("box", [1.1, 0.35, 0.4], [0.0, 0.0, 1.05], 10.0, [0.35, 0.22, 0.12, 1]),
               ("box", [0.25, 0.15, 0.2], [1.3, 0.0, 1.35], 11.0, [0.35, 0.22, 0.12, 1])],
    # a small camp fire: very hot, tiny
    "campfire": [("cylinder", [0.5, 0.1], [0.0, 0.0, 0.1], 300.0, [0.9, 0.4, 0.1, 1])],
}


@dataclass
class TargetDesign:
    dist_m: tuple[float, float] = (1500.0, 3000.0)  # group distance from the base (clipped to the map)
    n_vehicles: tuple[int, int] = (2, 3)
    n_people: tuple[int, int] = (0, 4)
    n_distractors: tuple[int, int] = (3, 5)
    edge_margin_m: float = 80.0
    road_snap_m: float = 150.0  # vehicles stand on a road if one is this close to the chosen spot


def _roads(world_dir: Path, terrain: Terrain) -> list[np.ndarray]:
    p = world_dir / "thermal_truth.npz"
    if not p.exists():
        return []
    with np.load(p) as z:
        roads = json.loads(str(z["roads_json"])) if "roads_json" in z else []
    out = []
    for w in roads:
        if w.get("kind") != "vehicle" and w.get("highway") != "track":
            continue
        pts = np.asarray(w["pts_enu"], float)
        # densify to 5 m
        dense = [pts[0]]
        for q0, q1 in zip(pts[:-1], pts[1:]):
            n = max(1, int(np.hypot(*(q1 - q0)) / 5.0))
            dense += [q0 + (q1 - q0) * (k / n) for k in range(1, n + 1)]
        dense = np.array(dense)
        dense = dense[[terrain.inside(q, 40.0) for q in dense]]
        if len(dense) >= 4:
            out.append(dense)
    return out


def choose_bearing(terrain: Terrain, rng: np.random.Generator, d: TargetDesign,
                   bearing_deg: float | None = None, dist_m: float | None = None) -> tuple[float, float]:
    """True bearing (deg cw from north) and distance of the group from the base, inside the map."""
    if bearing_deg is None:
        cands = [(az, terrain.max_range(az, d.edge_margin_m)) for az in range(0, 360, 2)]
        far = [(az, r) for az, r in cands if r >= d.dist_m[0]]
        if not far:  # map too small: the farthest direction
            az, r = max(cands, key=lambda c: c[1])
            far = [(az, r)]
        az, r = far[int(rng.integers(len(far)))]
        bearing_deg = float(az + rng.uniform(-1.0, 1.0))
    rmax = terrain.max_range(bearing_deg, d.edge_margin_m)
    if dist_m is None:
        lo, hi = min(d.dist_m[0], rmax), min(d.dist_m[1], rmax)
        dist_m = float(rng.uniform(lo, hi))
    return bearing_deg % 360.0, float(min(dist_m, rmax))


def _obj(kind: str, cls: str, label: str, parts: list, pos: np.ndarray, yaw: float, terrain: Terrain,
         air_K: float) -> dict[str, Any]:
    gz = float(terrain.ground(pos)[0])
    ps = [{"type": t, "size": s, "pos": c, "temp_K": round(air_K + dT, 2), "rgba": rgba} for t, s, c, dT, rgba in parts]
    return {"kind": kind, "role": "distractor" if kind == "distractor" else "target", "class": cls, "name": label,
            "label": label, "pos": [round(float(pos[0]), 2), round(float(pos[1]), 2),
            round(gz, 2)], "yaw_deg": round(float(yaw), 1), "temp_K": max(p["temp_K"] for p in ps), "parts": ps}


def place_targets(terrain: Terrain, rng: np.random.Generator, d: TargetDesign = TargetDesign(),
                  bearing_deg: float | None = None, dist_m: float | None = None,
                  air_K: float = 295.8) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Vehicles (+ people) as one group at (bearing, dist) from the base, on a road when one is near; distractors
    scattered over the search cone. Returns (objects, truth summary)."""
    bearing_deg, dist_m = choose_bearing(terrain, rng, d, bearing_deg, dist_m)
    a = terrain.a
    centre = a + dist_m * _unit(bearing_deg)
    roads = _roads(terrain.dir, terrain)
    nv = int(rng.integers(d.n_vehicles[0], d.n_vehicles[1] + 1))
    kinds = list(rng.choice(["truck", "car", "apc"], size=nv, p=[0.45, 0.3, 0.25]))
    objs: list[dict[str, Any]] = []
    placed: list[np.ndarray] = []
    road, k0 = None, 0
    if roads:
        best = min(((np.min(np.hypot(*(r - centre).T)), i) for i, r in enumerate(roads)), key=lambda c: c[0])
        if best[0] < d.road_snap_m:
            road = roads[best[1]]
            k0 = int(np.argmin(np.hypot(*(road - centre).T)))
    for i, kind in enumerate(kinds):
        if road is not None:  # a column on the road, 12-20 m apart (5 m road samples)
            k = int(np.clip(k0 + i * int(rng.integers(3, 5)), 1, len(road) - 2))
            p, tangent = road[k], road[k + 1] - road[k - 1]
            yaw = math.degrees(math.atan2(tangent[1], tangent[0]))
            p = p + rng.normal(0, 0.7, 2)
        else:
            p = centre + rng.normal(0, 12.0, 2)
            yaw = float(rng.uniform(0, 360))
        objs.append(_obj("vehicle", kind, f"{kind} (engine running)", VEHICLES[kind], p, yaw, terrain, air_K))
        placed.append(p)
    group = np.mean(placed, axis=0)
    for i in range(int(rng.integers(d.n_people[0], d.n_people[1] + 1))):
        p = placed[int(rng.integers(len(placed)))] + rng.normal(0, 6.0, 2)
        objs.append(_obj("person", "person", "person", PERSON, p, rng.uniform(0, 360), terrain, air_K))
    nd = int(rng.integers(d.n_distractors[0], d.n_distractors[1] + 1))
    dkinds = ["cooling_car", "barn", "warm_rocks"] + list(rng.choice(["animal", "campfire", "warm_rocks", "cooling_car"],
                                                                    size=max(0, nd - 3)))
    for kind in dkinds[:nd]:
        for _ in range(50):  # inside the cone (+-15 deg, 0.6-1.1 x distance), away from the group, inside the map
            p = a + dist_m * rng.uniform(0.6, 1.1) * _unit(bearing_deg + rng.uniform(-15.0, 15.0))
            if terrain.inside(p, 40.0) and np.hypot(*(p - group)) > 80.0:
                break
        objs.append(_obj("distractor", kind, kind.replace("_", " "), DISTRACTORS[kind], p, rng.uniform(0, 360),
                         terrain, air_K))
    gd = group - a
    truth = {"bearing_deg": round(az_deg(*gd), 3), "dist_m": round(float(np.hypot(*gd)), 1),
             "group_enu": [round(float(group[0]), 2), round(float(group[1]), 2)],
             "chosen_bearing_deg": round(bearing_deg, 3), "chosen_dist_m": round(dist_m, 1),
             "on_road": road is not None, "n_vehicles": nv,
             "n_people": sum(o["kind"] == "person" for o in objs),
             "n_distractors": sum(o["kind"] == "distractor" for o in objs),
             "map_max_range_m": round(terrain.max_range(bearing_deg, d.edge_margin_m), 0)}
    return objs, truth


# ---------------------------------------------------------------------------------------------- writing

def target_scene_dict(objs: list[dict[str, Any]], truth: dict[str, Any], seed: int) -> dict[str, Any]:
    return {"status": "recon", "seed": seed, "objects": objs, "recon_truth": truth,
            "object_format": {"parts": "list of {type: box|cylinder|sphere, size: half extents (box) | [r, half_h] "
                              "(cylinder) | [r] (sphere), pos: object frame (x forward, y left, z up, origin on the "
                              "ground), temp_K: apparent LWIR temperature, rgba}",
                              "pos": "[x, y, ground_z] ENU m", "yaw_deg": "about +z, 0 = +x (east), CCW",
                              "kind": "vehicle | person | distractor", "mesh": "(alternative to parts) STL path"}}


def build(mission: str, seed: int = 0, bearing_deg: float | None = None, dist_m: float | None = None,
          base: BaseDesign = BaseDesign(), targets: TargetDesign = TargetDesign(), write_default: bool = True,
          log: Any = print) -> dict[str, Any]:
    from ..missionprep.build import mission_dir

    mdir = mission_dir(mission) if not Path(mission).is_dir() else Path(mission)
    wdir, odir = mdir / "world", mdir / "onboard"
    terrain = Terrain(wdir)
    rng = np.random.default_rng(seed + 9100)
    lights, info, fence = design_base(terrain, base)
    air_K = 295.8
    tp = wdir / "thermal_truth.npz"
    if tp.exists():
        with np.load(tp) as z:
            air_K = float(z["air_K"])
    objs, truth = place_targets(terrain, rng, targets, bearing_deg, dist_m, air_K)
    strays = stray_lights(terrain, np.random.default_rng(seed + 9200), truth["bearing_deg"])
    o_lights, o_info = survey(lights, info, base, np.random.default_rng(seed + 9300))

    lights.save(wdir / "base_lights.json")
    (wdir / "base.json").write_text(json.dumps({**info.to_dict(), "fence": fence}, indent=2), encoding="utf-8")
    (wdir / "stray_lights.json").write_text(json.dumps({"format": "sfhack-stray-lights/1", "lights": strays}, indent=2),
                                            encoding="utf-8")
    o_lights.save(odir / "base_lights.json")
    o_info.save(odir / "base.json")
    scene = target_scene_dict(objs, truth, seed)
    sdir = wdir / "recon" / f"seed_{seed}"
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "target_scene.json").write_text(json.dumps(scene, indent=2), encoding="utf-8")
    rt = {"mission": mdir.name, "seed": seed, "base_center_enu": list(info.center_enu),
          "uma16_enu": list(info.uma16_enu), **truth}
    (sdir / "recon_truth.json").write_text(json.dumps(rt, indent=2), encoding="utf-8")
    if write_default:
        ts = wdir / "target_scene.json"
        if ts.exists() and not (wdir / "target_scene.orig.json").exists():
            shutil.copy2(ts, wdir / "target_scene.orig.json")
        ts.write_text(json.dumps(scene, indent=2), encoding="utf-8")
        (wdir / "recon_truth.json").write_text(json.dumps(rt, indent=2), encoding="utf-8")
    log(f"recon world {mdir.name} seed {seed}: {len(lights.ring('fence'))} fence + {len(lights.ring('outer'))} outer "
        f"lights, {len(strays)} stray lights; group at {truth['bearing_deg']:.1f} deg, {truth['dist_m']:.0f} m "
        f"({truth['n_vehicles']} vehicles, {truth['n_people']} people, {truth['n_distractors']} distractors, "
        f"on road: {truth['on_road']}); map edge along the bearing at {truth['map_max_range_m']:.0f} m")
    return rt


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m sfhack.sim.recon_world", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mission", default="ukraine-1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=None, help="write world/recon/seed_<k> for k < SEEDS (default "
                    "scene = --seed)")
    ap.add_argument("--bearing", type=float, default=None, help="true group bearing, deg cw from north")
    ap.add_argument("--dist", type=float, default=None, help="true group distance from the base (m)")
    ap.add_argument("--no-outer", action="store_true", help="fence ring only")
    a = ap.parse_args()
    base = BaseDesign(n_outer=0) if a.no_outer else BaseDesign()
    for k in range(a.seeds or 0):
        if k != a.seed:
            build(a.mission, k, a.bearing, a.dist, base, write_default=False)
    build(a.mission, a.seed, a.bearing, a.dist, base)


if __name__ == "__main__":
    main()
