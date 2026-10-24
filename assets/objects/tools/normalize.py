"""Re-runnable asset pipeline: source GLB / procedural -> normalized STL library.

Frame of every output: meters, Z up, +X forward (length axis),
origin at the bottom center of the footprint (object stands on z = 0).

Usage (from the project root):
    C:\\Users\\jean-\\.venvs\\sf-hack-assets\\Scripts\\python.exe -I tools/normalize.py [--only key/variant ...]

Outputs objects/<key>/<variant>/{<variant>.stl, parts/*.stl, meta.json, preview.png},
catalog.json and ATTRIBUTION.md.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "sources"
OUT = ROOT / "objects"
MAX_TRIS = 20000

CC0 = ("CC0 1.0", "https://creativecommons.org/publicdomain/zero/1.0/")
CCBY3 = ("CC-BY 3.0", "https://creativecommons.org/licenses/by/3.0/")

# ---------------------------------------------------------------- colors
RGBA = {
    "car_white": [0.92, 0.92, 0.92, 1], "car_brown": [0.40, 0.26, 0.15, 1],
    "truck_brown": [0.42, 0.28, 0.16, 1],
    "glass": [0.12, 0.16, 0.22, 1], "tire": [0.07, 0.07, 0.07, 1],
    "trim": [0.30, 0.30, 0.32, 1], "light_front": [0.95, 0.93, 0.80, 1],
    "light_rear": [0.70, 0.05, 0.05, 1],
    "tent_green": [0.22, 0.45, 0.20, 1], "tent_green_light": [0.35, 0.58, 0.28, 1],
    "tent_orange": [0.92, 0.45, 0.10, 1], "tent_orange_dark": [0.70, 0.30, 0.06, 1],
    "pole": [0.25, 0.20, 0.15, 1], "dark": [0.10, 0.10, 0.10, 1],
    "wood": [0.52, 0.36, 0.22, 1], "wood_light": [0.64, 0.48, 0.32, 1],
    "roof_gray": [0.38, 0.38, 0.40, 1], "stone": [0.50, 0.50, 0.50, 1],
    "trunk": [0.36, 0.25, 0.16, 1], "crown": [0.22, 0.48, 0.18, 1],
    "crown_conifer": [0.12, 0.34, 0.16, 1],
    # landing pad: neutral gray so the truck cannot be found by color from altitude
    # (the placer adds the ArUco marker on top)
    "landing_pad": [0.55, 0.55, 0.55, 1],
}
CAR_BODY_OPTIONS = {"white": RGBA["car_white"], "brown": RGBA["car_brown"]}

# Target real-world ranges per key (L, W, H) used for the sanity check.
SPEC = {
    "car": [(4.3, 4.8), (1.7, 2.1), (1.3, 1.6)],  # width incl. mirrors
    "truck": [(7.0, 9.0), (2.4, 2.6), (3.0, 3.5)],
    "tent": [(2.5, 3.5), (2.5, 3.5), (1.5, 2.2)],
    "hut": [(3.0, 5.0), (3.0, 5.0), (2.8, 3.5)],
    "tree": [(2.5, 8.0), (2.5, 8.0), (6.0, 12.0)],  # deciduous crown 4-8 m, conifers narrower
    "suv": [(4.4, 5.0), (1.8, 2.2), (1.6, 1.9)],
    "pickup": [(5.0, 5.6), (1.9, 2.3), (1.7, 2.0)],
}

QUAT_CAR_PARTS = {"Blue": "body", "LightBlue": "body", "White": "body", "Main": "body",
                  "Windows": "windows", "Black": "tires", "Grey": "trim",
                  "Material.007": "trim", "Headlights": "lights_front",
                  "TailLights": "lights_rear", "BrakeLight": "lights_rear"}
CAR_COLORS = {"body": RGBA["car_white"], "windows": RGBA["glass"], "tires": RGBA["tire"],
              "trim": RGBA["trim"], "lights_front": RGBA["light_front"],
              "lights_rear": RGBA["light_rear"]}
TREE_PARTS = {"woodBark": "trunk", "woodBarkDark": "trunk", "_defaultMat": "trunk",
              "leafsGreen": "crown", "leafsDark": "crown"}


PP_GLB = {  # direct public download links (static.poly.pizza), fetched 2026-10-23
    "Cz6yDaUcM9": "59a67a6c-490e-472e-bae6-5a4d2541f1c7",
    "unqqkULtRU": "d35173c8-6078-4367-8f87-b1f2599f0bb7",
    "xsMtZhBkxL": "e5fbf2ee-5c9e-47d5-8ab6-80cacd463baa",
    "qn4grQgHm8": "4e925a01-dbb8-4aab-848b-221306b835ea",
    "Fy3WI3uXNQ": "4e798498-be05-47ac-988a-ad953cf46236",
    "5Q7qIrfDxA": "fc8d560d-91b5-439c-88bf-4cbddb7eadbb",
    "HTx7PZt6Zm": "9f69cf1b-e9f0-495a-9f68-0dab4bffd6d2",
    "et0OmFeZVkb": "0647436a-ac3f-4e34-abdd-01f57e43cc2f",
    "4MJWbyd6vw": "36d65045-d2ff-4689-a34d-a4acbe1873cb",
}


def pp(model_id, author, lic=CC0):
    return dict(source_url=f"https://poly.pizza/m/{model_id}", author=author,
                download_url=f"https://static.poly.pizza/{PP_GLB[model_id]}.glb",
                license=lic[0], license_url=lic[1])


KENNEY_NATURE = dict(source_url="https://kenney.nl/assets/nature-kit", author="Kenney (kenney.nl)",
                     download_url="https://kenney.nl/media/pages/assets/nature-kit/37ac38a37b-1677698939/"
                                  "kenney_nature-kit.zip",
                     license=CC0[0], license_url=CC0[1])
PROC = dict(source_url="procedural (tools/normalize.py)", author="generated",
            license=CC0[0], license_url=CC0[1])

# fwd: axis of the source file (glTF, Y up) that points to the vehicle front, or "auto"
# (longest horizontal extent becomes +X). scale_to: (dimension, meters) for the uniform scale.
ASSETS = [
    # ---- cars
    dict(key="car", variant="sedan_a", src="polypizza/Cz6yDaUcM9.glb", fwd="+z",
         scale_to=("length", 4.7), parts=QUAT_CAR_PARTS, colors=CAR_COLORS,
         note="Quaternius 'Car' (low-poly sedan)", **pp("Cz6yDaUcM9", "Quaternius")),
    dict(key="car", variant="hatch_b", src="polypizza/unqqkULtRU.glb", fwd="+z",
         scale_to=("length", 4.3), parts=QUAT_CAR_PARTS, colors=CAR_COLORS,
         note="Quaternius 'Car' (compact/hatchback)", **pp("unqqkULtRU", "Quaternius")),
    # ---- extras
    dict(key="suv", variant="suv_a", src="polypizza/xsMtZhBkxL.glb", fwd="+z",
         scale_to=("length", 4.7), parts=QUAT_CAR_PARTS, colors=CAR_COLORS,
         note="Quaternius 'SUV'", **pp("xsMtZhBkxL", "Quaternius")),
    dict(key="pickup", variant="pickup_a", src="polypizza/qn4grQgHm8.glb", fwd="+z",
         scale_to=("length", 5.3), parts={"*": "body", "Headlights": "lights_front",
                                           "BrakeLight": "lights_rear"},
         colors=CAR_COLORS, note="Quaternius 'Pickup Truck' (texture atlas, single body part)",
         **pp("qn4grQgHm8", "Quaternius")),
    # ---- trucks
    dict(key="truck", variant="box_truck_a", proc="truck",
         proc_args=dict(L=8.0, W=2.5, H=3.3, cab_len=2.0), **PROC),
    dict(key="truck", variant="box_truck_b", proc="truck",
         proc_args=dict(L=7.2, W=2.45, H=3.1, cab_len=1.8), **PROC),
    dict(key="truck", variant="cargo_truck_c", src="polypizza/Fy3WI3uXNQ.glb", fwd="+z",
         scale_to=("height", 3.1),
         parts={"truck_base_colour": "body", "truck_bc_colour2": "body",
                "truck_metal_colour": "cargo_box", "truck_metal_light": "trim",
                "truck_black_colour": "tires", "truck_windows": "windows",
                "truck_headlight": "lights_front", "truck_backlight": "lights_rear"},
         colors={"body": RGBA["truck_brown"], "cargo_box": RGBA["truck_brown"],
                 "trim": RGBA["trim"], "tires": RGBA["tire"], "windows": RGBA["glass"],
                 "lights_front": RGBA["light_front"], "lights_rear": RGBA["light_rear"],
                 "roof_plate": RGBA["landing_pad"]},
         title="Cargo Truck", note="J-Toastie 'Cargo Truck' (semi-truck; flat roof plate added)", **pp("Fy3WI3uXNQ", "J-Toastie", CCBY3)),
    # ---- tents
    dict(key="tent", variant="ridge_green_a", src="polypizza/5Q7qIrfDxA.glb", fwd="auto",
         scale_to=("height", 1.6),
         parts={"Green": "fabric", "LightGreen": "fabric_accent", "DarkWood": "poles",
                "Black": "entrance"},
         colors={"fabric": RGBA["tent_green"], "fabric_accent": RGBA["tent_green_light"],
                 "poles": RGBA["pole"], "entrance": RGBA["dark"]},
         note="Quaternius 'Tent'", **pp("5Q7qIrfDxA", "Quaternius")),
    dict(key="tent", variant="ridge_orange_b", src="kenney_nature-kit/tent_detailedClosed.glb",
         fwd="auto", scale_to=("length", 3.2),
         parts={"colorRed": "fabric", "colorRedDark": "fabric_accent", "wood": "poles",
                "_defaultMat": "poles"},
         colors={"fabric": RGBA["tent_orange"], "fabric_accent": RGBA["tent_orange_dark"],
                 "poles": RGBA["pole"]},
         note="Kenney Nature Kit tent_detailedClosed", **KENNEY_NATURE),
    dict(key="tent", variant="dome_c", proc="tent", proc_args=dict(D=2.8, H=1.6), **PROC),
    # ---- huts
    dict(key="hut", variant="cabin_a", proc="hut",
         proc_args=dict(L=4.2, W=3.4, wall_h=2.2, ridge_h=3.2), **PROC),
    dict(key="hut", variant="shed_d", proc="hut",
         proc_args=dict(L=3.4, W=3.0, wall_h=2.1, ridge_h=2.95, overhang=0.25), **PROC),
    dict(key="hut", variant="cabin_shed_b", src="polypizza/HTx7PZt6Zm.glb", fwd="auto",
         scale_to=("height", 3.4),
         parts={"__colors__": [((84, 84, 60), "walls"), ((84, 60, 60), "trim"),
                               ((36, 84, 36), "roof"), ((132, 132, 132), "stone"),
                               ((180, 204, 228), "windows")]},
         colors={"walls": RGBA["wood"], "trim": RGBA["wood_light"], "roof": RGBA["roof_gray"],
                 "stone": RGBA["stone"], "windows": RGBA["glass"]},
         note="CreativeTrio 'Cabin Shed' (single texture; parts split by texture color)",
         **pp("HTx7PZt6Zm", "CreativeTrio")),
    dict(key="hut", variant="log_cabin_e", src="polypizza/et0OmFeZVkb.glb", fwd="auto",
         scale_to=("height", 3.5), parts={"mat18": "walls", "mat20": "trim"}, roof_split=0.6,
         colors={"walls": RGBA["wood"], "trim": RGBA["wood_light"], "roof": RGBA["roof_gray"]},
         title="Log Cabin", note="Jarlan Perez 'Log Cabin' (roof split by face normal)",
         **pp("et0OmFeZVkb", "Jarlan Perez", CCBY3)),
    dict(key="hut", variant="hut_c", src="polypizza/4MJWbyd6vw.glb", fwd="auto",
         scale_to=("height", 3.2), parts={"Wood": "walls", "Stone": "base"},
         colors={"walls": RGBA["wood"], "base": RGBA["stone"]},
         note="Quaternius 'Hut'", **pp("4MJWbyd6vw", "Quaternius")),
    # ---- trees
    dict(key="tree", variant="oak_a", src="kenney_nature-kit/tree_oak.glb", fwd="auto",
         scale_to=("height", 9.0), parts=TREE_PARTS,
         colors={"trunk": RGBA["trunk"], "crown": RGBA["crown"]},
         note="Kenney Nature Kit tree_oak", **KENNEY_NATURE),
    dict(key="tree", variant="deciduous_b", src="kenney_nature-kit/tree_detailed.glb", fwd="auto",
         scale_to=("height", 10.0), parts=TREE_PARTS,
         colors={"trunk": RGBA["trunk"], "crown": RGBA["crown"]},
         note="Kenney Nature Kit tree_detailed", **KENNEY_NATURE),
    dict(key="tree", variant="deciduous_c", src="kenney_nature-kit/tree_default.glb", fwd="auto",
         scale_to=("height", 11.0), parts=TREE_PARTS,
         colors={"trunk": RGBA["trunk"], "crown": RGBA["crown"]},
         note="Kenney Nature Kit tree_default", **KENNEY_NATURE),
    dict(key="tree", variant="conifer_d", src="kenney_nature-kit/tree_pineTallA_detailed.glb",
         fwd="auto", scale_to=("height", 12.0), parts=TREE_PARTS,
         colors={"trunk": RGBA["trunk"], "crown": RGBA["crown_conifer"]},
         note="Kenney Nature Kit tree_pineTallA_detailed", **KENNEY_NATURE),
    dict(key="tree", variant="conifer_e", src="kenney_nature-kit/tree_pineRoundC.glb",
         fwd="auto", scale_to=("height", 10.0), parts=TREE_PARTS,
         colors={"trunk": RGBA["trunk"], "crown": RGBA["crown_conifer"]},
         note="Kenney Nature Kit tree_pineRoundC", **KENNEY_NATURE),
]


# ---------------------------------------------------------------- loading
def load_parts(path, mat_map):
    """Load a glTF scene, bake node transforms, group faces into named parts by material."""
    scene = trimesh.load(path, force="scene")
    groups = {}
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        g = scene.geometry[gname]
        if not isinstance(g, trimesh.Trimesh) or len(g.faces) == 0:
            continue
        m = trimesh.Trimesh(vertices=g.vertices.copy(), faces=g.faces.copy(), process=False)
        m.apply_transform(T)
        if "__colors__" in mat_map:  # single texture atlas: nearest prototype color per face
            protos = mat_map["__colors__"]
            vc = g.visual.to_color().vertex_colors[:, :3].astype(float)
            fc = vc[g.faces].mean(1)
            P = np.array([c for c, _ in protos], float)
            lab = np.argmin(((fc[:, None, :] - P[None]) ** 2).sum(-1), 1)
            for i, (_, part) in enumerate(protos):
                idx = np.nonzero(lab == i)[0]
                if len(idx):
                    groups.setdefault(part, []).append(m.submesh([idx], append=True))
            continue
        mat = getattr(getattr(g.visual, "material", None), "name", None)
        part = mat_map.get(mat, mat_map.get("*", "body"))
        groups.setdefault(part, []).append(m)
    return {k: trimesh.util.concatenate(v) for k, v in groups.items()}


def axis_vec(s):
    v = np.zeros(3)
    v["xyz".index(s[1])] = 1.0 if s[0] == "+" else -1.0
    return v


def orient(parts, fwd):
    """glTF Y-up -> Z-up, then yaw so the forward / length axis is +X."""
    R_up = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])  # +Y -> +Z
    for m in parts.values():
        m.apply_transform(R_up)
    if fwd == "auto":
        allv = np.vstack([m.vertices for m in parts.values()])
        ext = allv.max(0) - allv.min(0)
        f = np.array([1.0, 0, 0]) if ext[0] >= ext[1] else np.array([0, 1.0, 0])
    else:
        f = R_up[:3, :3] @ axis_vec(fwd)
    yaw = -np.arctan2(f[1], f[0])
    R_yaw = trimesh.transformations.rotation_matrix(yaw, [0, 0, 1])
    for m in parts.values():
        m.apply_transform(R_yaw)


def bounds_of(parts):
    allv = np.vstack([m.vertices for m in parts.values()])
    return allv.min(0), allv.max(0)


def recenter_and_scale(parts, scale_to):
    lo, hi = bounds_of(parts)
    ext = hi - lo
    dim, target = scale_to
    ref = {"length": ext[0], "width": ext[1], "height": ext[2],
           "crown": max(ext[0], ext[1])}[dim]
    s = target / ref
    shift = -np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    for m in parts.values():
        m.apply_translation(shift)
        m.apply_scale(s)
    return float(s)


def decimate(parts, max_tris=MAX_TRIS):
    total = sum(len(m.faces) for m in parts.values())
    if total <= max_tris:
        return parts, False
    ratio = max_tris / total * 0.98
    out = {}
    for k, m in parts.items():
        n = len(m.faces)
        if n < 200:
            out[k] = m
            continue
        out[k] = m.simplify_quadric_decimation(face_count=max(100, int(n * ratio)))
    return out, True


# ---------------------------------------------------------------- procedural
def box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    b = trimesh.creation.box(extents=hi - lo)
    b.apply_translation((lo + hi) / 2)
    return b


def hull(points):
    return trimesh.convex.convex_hull(np.asarray(points, float))


def wheel(x, y, r, w):
    c = trimesh.creation.cylinder(radius=r, height=w, sections=20)
    c.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    c.apply_translation([x, y, r])
    return c


def proc_truck(L=8.0, W=2.5, H=3.3, cab_len=2.0):
    """Box truck: cargo box with flat roof at z=H, cab in front (+X), 3 axles."""
    r, tw = 0.5, 0.36
    x0, x1 = -L / 2, L / 2
    bumper = 0.12
    cab_x0 = x1 - bumper - cab_len
    cargo_x1 = cab_x0 - 0.08
    cab_w, cab_h = W - 0.08, H * 0.82
    parts = {}
    parts["cargo_box"] = box([x0, -W / 2, 1.05], [cargo_x1, W / 2, H])
    parts["chassis"] = box([x0 + 0.3, -0.9, 0.62], [x1 - bumper, 0.9, 1.05])
    slope = 0.45
    parts["cab"] = hull([[cab_x0, s * cab_w / 2, z] for s in (-1, 1) for z in (0.75, cab_h)]
                        + [[x1 - bumper, s * cab_w / 2, z] for s in (-1, 1)
                           for z in (0.75, cab_h * 0.62)]
                        + [[x1 - bumper - slope, s * cab_w / 2, cab_h] for s in (-1, 1)])
    # windshield: thin plate just outside the sloped front face
    ws = []
    for s in (-1, 1):
        for t in (0.68, 0.97):
            z = cab_h * 0.62 + t * (cab_h - cab_h * 0.62)
            x = (x1 - bumper) - t * slope + 0.02
            ws.append([x, s * (cab_w / 2 - 0.1), z])
            ws.append([x - 0.03, s * (cab_w / 2 - 0.1), z])
    windows = [hull(ws)]
    for s in (-1, 1):  # side windows
        y = s * (cab_w / 2 + 0.01)
        windows.append(box([cab_x0 + 0.25, min(y, y + s * 0.02), cab_h * 0.66],
                           [x1 - bumper - slope - 0.05, max(y, y + s * 0.02), cab_h * 0.95]))
    parts["windows"] = trimesh.util.concatenate(windows)
    parts["trim"] = trimesh.util.concatenate([
        box([x1 - bumper, -cab_w / 2, 0.45], [x1, cab_w / 2, 0.8]),           # front bumper
        box([x0 - 0.02, -W / 2 + 0.05, 0.9], [x0, W / 2 - 0.05, H - 0.05]),  # rear door frame
    ])
    parts["lights_front"] = trimesh.util.concatenate(
        [box([x1 - 0.02, s * cab_w / 2 - (0.35 if s > 0 else 0), 0.85],
             [x1 - bumper + 0.11, s * cab_w / 2 + (0 if s > 0 else 0.35), 1.0]) for s in (-1, 1)])
    parts["lights_rear"] = trimesh.util.concatenate(
        [box([x0 - 0.04, s * W / 2 - (0.3 if s > 0 else 0), 1.1],
             [x0 - 0.01, s * W / 2 + (0 if s > 0 else 0.3), 1.3]) for s in (-1, 1)])
    yw = W / 2 - tw / 2 - 0.02
    axles = [x1 - bumper - cab_len * 0.55, x0 + 1.9, x0 + 0.85] if L > 7.5 else \
            [x1 - bumper - cab_len * 0.55, x0 + 1.3]
    parts["tires"] = trimesh.util.concatenate([wheel(ax, s * yw, r, tw) for ax in axles for s in (-1, 1)])
    colors = {"cargo_box": RGBA["truck_brown"], "cab": RGBA["truck_brown"],
              "chassis": RGBA["dark"], "windows": RGBA["glass"], "trim": RGBA["trim"],
              "lights_front": RGBA["light_front"], "lights_rear": RGBA["light_rear"],
              "tires": RGBA["tire"]}
    return parts, colors


def proc_hut(L=4.2, W=3.4, wall_h=2.2, ridge_h=3.2, overhang=0.3, t=0.12):
    """Cabin: walls with gable ends (ridge along X), two roof slabs, door + windows."""
    parts = {}
    zr = ridge_h - t - 0.01  # gable peak, just below the roof slabs
    gable = [[x, y, z] for x in (-L / 2, L / 2) for (y, z) in
             ((-W / 2, 0), (W / 2, 0), (-W / 2, wall_h), (W / 2, wall_h), (0, zr))]
    parts["walls"] = hull(gable)
    slabs = []
    hw = W / 2 + overhang
    slope = (zr - wall_h) / (W / 2)
    zb_ridge, zb_edge = zr + 0.01, zr + 0.01 - slope * hw  # slab underside follows the walls
    for s in (-1, 1):
        pts = []
        for x in (-L / 2 - overhang, L / 2 + overhang):
            pts += [[x, 0, zb_ridge + t], [x, s * hw, zb_edge + t], [x, 0, zb_ridge], [x, s * hw, zb_edge]]
        slabs.append(hull(pts))
    parts["roof"] = trimesh.util.concatenate(slabs)
    parts["door"] = box([L / 2, -0.45, 0.0], [L / 2 + 0.04, 0.45, 1.95])
    wins = []
    for s in (-1, 1):
        wins.append(box([-0.5, (W / 2 if s > 0 else -W / 2 - 0.04), 1.0],
                        [0.5, (W / 2 + 0.04 if s > 0 else -W / 2), 1.7]))
    parts["windows"] = trimesh.util.concatenate(wins)
    parts["base"] = box([-L / 2 - 0.08, -W / 2 - 0.08, 0], [L / 2 + 0.08, W / 2 + 0.08, 0.15])
    colors = {"walls": RGBA["wood"], "roof": RGBA["roof_gray"], "door": RGBA["wood_light"],
              "windows": RGBA["glass"], "base": RGBA["stone"]}
    return parts, colors


def dome(rx, ry, rz, subdiv):
    """Upper half of an ellipsoid (convex, closed by its flat base)."""
    v = trimesh.creation.icosphere(subdivisions=subdiv).vertices
    v = v[v[:, 2] >= -1e-9]
    return hull(v * [rx, ry, rz])


def proc_tent(D=2.8, H=1.6):
    """Dome tent with a small vestibule on +X and a dark entrance patch."""
    s = dome(D / 2, D / 2, H, 3)
    v = dome(0.7, 0.75, H * 0.72, 2)
    v.apply_translation([D / 2 - 0.2, 0, 0])
    door = hull([[D / 2 + 0.47, y, z] for y in (-0.35, 0.35) for z in (0.02, 0.85)]
                + [[D / 2 + 0.50, 0, 0.95]] + [[D / 2 + 0.44, y, z] for y in (-0.35, 0.35)
                                                for z in (0.02, 0.85)])
    poles = []
    for ang in (np.pi / 4, -np.pi / 4):  # two crossing arcs
        th = np.linspace(0, np.pi, 24)
        pts = np.stack([np.cos(th) * D / 2 * 1.005, np.zeros_like(th), np.sin(th) * H * 1.005], 1)
        R = trimesh.transformations.rotation_matrix(ang, [0, 0, 1])[:3, :3]
        pts = pts @ R.T
        for a, b in zip(pts[:-1], pts[1:]):
            poles.append(trimesh.creation.cylinder(radius=0.02, segment=[a, b], sections=6))
    parts = {"fabric": s, "fabric_accent": v, "entrance": door,
             "poles": trimesh.util.concatenate(poles)}
    colors = {"fabric": RGBA["tent_orange"], "fabric_accent": RGBA["tent_orange_dark"],
              "entrance": RGBA["dark"], "poles": RGBA["dark"]}
    return parts, colors


PROCEDURAL = {"truck": proc_truck, "hut": proc_hut, "tent": proc_tent}


def split_roof(parts, frac, min_tilt=0.3):
    """Move faces above frac*height that are tilted (|n_z| > min_tilt) into a 'roof' part."""
    H = bounds_of(parts)[1][2]
    roof = []
    for k in list(parts):
        m = parts[k]
        sel = (m.triangles_center[:, 2] > frac * H) & (np.abs(m.face_normals[:, 2]) > min_tilt)
        if sel.any():
            roof.append(m.submesh([np.nonzero(sel)[0]], append=True))
            parts[k] = m.submesh([np.nonzero(~sel)[0]], append=True)
    if roof:
        parts["roof"] = trimesh.util.concatenate(roof)


# ---------------------------------------------------------------- analysis
def landing_area(mesh, min_size=2.0):
    """Highest horizontal up-facing region; returns center/size of its bounding rectangle."""
    n = mesh.face_normals
    up = n[:, 2] > 0.995
    if not up.any():
        return None
    zc = mesh.triangles_center[:, 2]
    area = mesh.area_faces
    levels = {}
    for i in np.nonzero(up)[0]:
        levels.setdefault(round(zc[i], 2), []).append(i)
    for z in sorted(levels, reverse=True):
        idx = levels[z]
        a = area[idx].sum()
        if a < min_size * min_size:
            continue
        tri = mesh.triangles[idx].reshape(-1, 3)
        lo, hi = tri.min(0), tri.max(0)
        size = hi[:2] - lo[:2]
        fill = a / max(size[0] * size[1], 1e-9)
        if size.min() >= min_size and fill > 0.9:
            return {"center": [round(float((lo[0] + hi[0]) / 2), 3),
                               round(float((lo[1] + hi[1]) / 2), 3), round(float(z), 3)],
                    "size": [round(float(size[0]), 3), round(float(size[1]), 3)],
                    "normal": [0, 0, 1], "fill_ratio": round(float(fill), 3)}
    return None


def add_roof_plate(parts, colors, size=(2.6, 2.4), t=0.04):
    lo, hi = bounds_of(parts)
    L = hi[0] - lo[0]
    cx = lo[0] + 0.35 * L  # over the cargo area (rear part of the vehicle)
    z = hi[2]
    parts["roof_plate"] = box([cx - size[0] / 2, -size[1] / 2, z], [cx + size[0] / 2, size[1] / 2, z + t])
    colors.setdefault("roof_plate", RGBA["landing_pad"])


# ---------------------------------------------------------------- preview
def shade(parts, colors, light=(0.4, -0.6, 0.8)):
    polys, cols = [], []
    l = np.asarray(light) / np.linalg.norm(light)
    for k, m in parts.items():
        c = np.array(colors.get(k, [0.6, 0.6, 0.6, 1]))[:3]
        f = 0.45 + 0.55 * np.clip(m.face_normals @ l, 0, 1)
        polys.append(m.triangles)
        cols.append(np.clip(c[None, :] * f[:, None], 0, 1))
    return np.concatenate(polys), np.concatenate(cols)


def preview(parts, colors, path, title):
    lo, hi = bounds_of(parts)
    edge = (hi - lo).max() / 12
    fine = {}
    for k, m in parts.items():
        try:
            v, f = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=edge, max_iter=6)
            fine[k] = trimesh.Trimesh(v, f, process=False)
        except Exception:
            fine[k] = m
    tris, cols = shade(fine, colors)
    c = (lo + hi) / 2
    r = (hi - lo).max() / 2 * 1.05
    fig = plt.figure(figsize=(12, 6), dpi=90)
    for i, (el, az, name) in enumerate([(90, -90, "top view (+X right)"), (22, -55, "3/4 view")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        pc = Poly3DCollection(tris, facecolors=cols, edgecolors="none")
        ax.add_collection3d(pc)
        ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(0, 2 * r)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=el, azim=az)
        if el == 90:
            ax.set_proj_type("ortho")
            ax.quiver(c[0], c[1], hi[2] + 0.05, r * 0.6, 0, 0, color="red")  # +X arrow
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(name)
    ext = hi - lo
    fig.suptitle(f"{title}   L x W x H = {ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f} m")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- main
def rnd(v, n=3):
    return [round(float(x), n) for x in v]


def process(a):
    key, var = a["key"], a["variant"]
    d = OUT / key / var
    (d / "parts").mkdir(parents=True, exist_ok=True)
    for old in (d / "parts").glob("*.stl"):
        old.unlink()
    if "proc" in a:
        parts, colors = PROCEDURAL[a["proc"]](**a.get("proc_args", {}))
        lo, hi = bounds_of(parts)
        shift = -np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
        for m in parts.values():
            m.apply_translation(shift)
        scale = 1.0
    else:
        parts = load_parts(SRC / a["src"], a["parts"])
        colors = dict(a["colors"])
        orient(parts, a["fwd"])
        scale = recenter_and_scale(parts, a["scale_to"])
    for m in parts.values():
        m.merge_vertices()
    if a.get("roof_split"):
        split_roof(parts, a["roof_split"])
    parts, decimated = decimate(parts)

    land = None
    if key == "truck":
        whole0 = trimesh.util.concatenate(list(parts.values()))
        land = landing_area(whole0)
        if land is None:
            add_roof_plate(parts, colors)
            land = landing_area(trimesh.util.concatenate(list(parts.values())))
            land["added_roof_plate"] = True
        if land is not None:
            land["rgba"] = RGBA["landing_pad"]  # recommended pad color (neutral gray)

    whole = trimesh.util.concatenate(list(parts.values()))
    lo, hi = whole.bounds
    ext = hi - lo
    whole.export(d / f"{var}.stl", file_type="stl")
    main_part = max(parts, key=lambda k: parts[k].area)
    files = {"whole": {"path": f"objects/{key}/{var}/{var}.stl",
                       "rgba": colors.get(main_part, [0.6, 0.6, 0.6, 1])},
             "parts": []}
    for name, m in parts.items():
        m.export(d / "parts" / f"{name}.stl", file_type="stl")
        entry = {"name": name, "path": f"objects/{key}/{var}/parts/{name}.stl",
                 "rgba": colors.get(name, [0.6, 0.6, 0.6, 1]), "triangles": int(len(m.faces))}
        if key == "car" and name == "body":
            entry["rgba_options"] = CAR_BODY_OPTIONS
        files["parts"].append(entry)

    meta = {
        "key": key, "variant": var, "files": files,
        "size_m": rnd(ext), "scale_factor": round(scale, 5), "triangles": int(len(whole.faces)),
        "decimated": decimated, "origin": "bottom_center", "forward_axis": "+x", "up_axis": "+z",
        "bounds_m": {"min": rnd(lo), "max": rnd(hi)},
        "footprint_m": rnd(ext[:2]),
        "height_m": round(float(ext[2]), 3),
        "collision_box": {"center": rnd((lo + hi) / 2), "half_size": rnd(ext / 2)},
    }
    if key == "tree" and "trunk" in parts and "crown" in parts:
        tl, th = parts["trunk"].bounds
        cl, ch = parts["crown"].bounds
        meta["collision_boxes_detailed"] = {
            "trunk": {"center": rnd((tl + th) / 2), "half_size": rnd((th - tl) / 2)},
            "crown": {"center": rnd((cl + ch) / 2), "half_size": rnd((ch - cl) / 2)}}
        meta["crown_diameter_m"] = round(float(max(ch[0] - cl[0], ch[1] - cl[1])), 3)
    if land is not None:
        meta["landing_area"] = land
    meta.update({k: a[k] for k in ("source_url", "author", "license", "license_url")})
    meta["download_url"] = a.get("download_url")
    meta["title"] = a.get("title")
    meta["source_file"] = f"sources/{a['src']}" if "src" in a else None
    meta["procedural"] = "proc" in a
    meta["note"] = a.get("note", "procedural low-poly mesh generated by tools/normalize.py"
                         if "proc" in a else "")
    spec = SPEC.get(key)
    if spec:
        meta["size_check"] = {ax: ("ok" if lo_ - 1e-6 <= v <= hi_ + 1e-6 else
                                   f"{v:.2f} outside {lo_}-{hi_}")
                              for ax, v, (lo_, hi_) in zip("LWH", ext, spec)}
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    preview(parts, colors, d / "preview.png", f"{key}/{var}")
    return meta


def write_catalog(metas):
    cat = {"units": "m", "frame": {"up": "+z", "forward": "+x", "origin": "bottom_center"},
           "notes": "Each variant: whole STL + colored parts. Place by setting geom pos at the "
                    "desired ground point (z = ground height) and yaw about +z.",
           "objects": {}}
    for m in metas:
        cat["objects"].setdefault(m["key"], []).append(m)
    (ROOT / "catalog.json").write_text(json.dumps(cat, indent=2))

    lines = ["# Attribution", "",
             "All meshes were converted, re-oriented, scaled and (where needed) decimated by "
             "`tools/normalize.py`. Kenney and Quaternius assets are CC0 (credit given anyway). "
             "CC-BY assets require the credit below to be kept.", "",
             "| object/variant | files | source | download | author | license |",
             "|---|---|---|---|---|---|"]
    for m in metas:
        lic = f"[{m['license']}]({m['license_url']})"
        src = m["source_url"] if m["procedural"] else f"[{m['source_url']}]({m['source_url']})"
        dl = f"[link]({m['download_url']})" if m.get("download_url") else "-"
        lines.append(f"| {m['key']}/{m['variant']} | `objects/{m['key']}/{m['variant']}/` | {src} "
                     f"| {dl} | {m['author']} | {lic} |")
    lines += ["", "CC-BY 3.0 credits:", ""]
    for m in metas:
        if m["license"].startswith("CC-BY"):
            lines.append(f"- \"{m['title']}\" by {m['author']} ({m['source_url']}), licensed "
                         f"under {m['license']} ({m['license_url']}); modified (re-oriented, "
                         f"scaled, split into parts, converted to STL).")
    (ROOT / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="key/variant filters")
    args = ap.parse_args()
    metas = []
    for a in ASSETS:
        kv = f"{a['key']}/{a['variant']}"
        if args.only and kv not in args.only:
            # keep previous meta in the catalog
            mp = OUT / a["key"] / a["variant"] / "meta.json"
            if mp.exists():
                metas.append(json.loads(mp.read_text()))
            continue
        m = process(a)
        metas.append(m)
        print(f"{kv:28s} size={m['size_m']} tris={m['triangles']:6d} scale={m['scale_factor']:.4f} "
              f"check={m.get('size_check')} land={m.get('landing_area')}")
    write_catalog(metas)


if __name__ == "__main__":
    sys.exit(main())
