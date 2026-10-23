"""V4 detector data (simulator side): randomized scenes around B built from the object library (real-scale STL
models, catalog.json), rendered by MuJoCo, auto-labeled from MuJoCo's segmentation buffer.

SIMULATOR / HARNESS SIDE. This module holds the truth (which object is where, which class every pixel belongs
to). It feeds offline dataset generation and the harness evaluation (demos/13-17) only; the onboard detector
(`perception/truck_detector.py`) sees RGB frames and never imports this module (tests/test_truck_detector.py).

Scene: B = world origin, flat ground z = 0. Exactly one brown truck (it carries the landing station: a neutral
gray pad with the ArUco marker), one brown car, one white car, a tent, a hut, a tree, 0-3 distractors (brown
pickup, SUV, brown hut), trees that partly hang over vehicles, background trees. All objects stand within
`AREA_R` of B with random yaw and a minimum spacing. The brown car, pickup, SUV and hut draw their brown from the
same distribution as the truck: color alone cannot separate them; shape and size must.

Rendering: one MuJoCo model per process holds a pool of instances of every library variant; a sample moves the
instances it uses into place (static bodies: `body_pos`, `body_quat`) and parks the rest under the ground, so
nothing is recompiled. Ground = a 2048 px texture (crop of the Berkeley USGS ortho or a procedural surface),
re-uploaded per sample. One directional sun with a 4096 px shadow map clipped to the scene (no shadow acne).
Camera effects after the render: optics blur, sensor noise, fog / rain from the degradation levers.

Splits: train / val never contain the held-out variants (`HELDOUT`) or the held-out ground textures; test uses
ONLY the held-out truck and car variants plus held-out grounds (north part of the ortho, dry grass, concrete).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np

from ..config import REPO_ROOT, SAMPLE_WORLD_DIR
from ..degradation import models as lever
from ..degradation.conditions import WeatherConditions
from ..perception.map_localizer import R_BODY_CAM

OBJECTS_DIR = Path(os.environ.get("SFHACK_OBJECTS") or (REPO_ROOT.parent / "7 SF Hack Objects"))
CLASSES = ("truck", "car", "pickup", "suv", "tent", "hut", "tree")
AREA_R = 20.0  # m, objects stand within this radius of B
SPLITS = ("train", "val", "test")
# Never rendered in train / val. truck + car: the test split uses ONLY these (every test truck is unseen).
HELDOUT = {"truck": ("box_truck_b",), "car": ("hatch_b",), "tent": ("dome_c",), "hut": ("log_cabin_e",),
           "tree": ("conifer_e",)}
TEST_ONLY_HELDOUT = ("truck", "car")
POOL = {"truck": 1, "car": 2, "pickup": 1, "suv": 1, "tent": 1, "hut": 2, "tree": 4}  # instances per variant
BROWN = (0.41, 0.27, 0.155)
WHITE = (0.92, 0.92, 0.92)
SUV_COLORS = {"white": WHITE, "gray": (0.45, 0.46, 0.48), "black": (0.08, 0.08, 0.09), "blue": (0.15, 0.22, 0.42)}
BROWN_PARTS = {"truck": ("cargo_box", "cab", "body", "roof_plate"), "car": ("body",), "pickup": ("body",),
               "suv": ("body",)}

W, H, FOVY = 640, 480, 70.0
GROUND_HALF = 110.0  # m, textured ground plane (the rest is a tiled far plane)
TEX_PX = 2048
PAD_SIZE, MARKER_SIZE, MARKER_QUIET = 2.4, 1.6, 0.15  # m
MARKER_DICT, MARKER_ID = "DICT_6X6_50", 7
HIDDEN_Z = -300.0


# ---------------------------------------------------------------------------------------- library

def load_library(root: Path = OBJECTS_DIR) -> dict[str, dict[str, dict]]:
    """{key: {variant: meta}} from catalog.json (paths made absolute)."""
    cat = json.loads((Path(root) / "catalog.json").read_text(encoding="utf-8"))
    out: dict[str, dict[str, dict]] = {}
    for key, metas in cat["objects"].items():
        out[key] = {}
        for m in metas:
            m = json.loads(json.dumps(m))
            for p in m["files"]["parts"]:
                p["abspath"] = str(Path(root) / p["path"])
            out[key][m["variant"]] = m
    return out


def variant_pool(lib: dict, key: str, split: str) -> list[str]:
    allv = sorted(lib[key])
    held = HELDOUT.get(key, ())
    if split in ("train", "val"):
        return [v for v in allv if v not in held]
    if key in TEST_ONLY_HELDOUT:
        return [v for v in allv if v in held]
    return allv


def footprint_radius(meta: dict) -> float:
    if "crown_diameter_m" in meta:
        return 0.5 * float(meta["crown_diameter_m"])
    L, Wd = meta["size_m"][:2]
    return 0.5 * math.hypot(L, Wd)


# ---------------------------------------------------------------------------------------- layout

@dataclass
class Placed:
    key: str  # library key = detection class
    variant: str
    xy: tuple[float, float]
    yaw: float
    role: str  # target, brown_car, white_car, tent, hut, tree, distractor, occluder, background
    color: str  # brown, white, default, ...
    rgba: dict[str, list[float]] = field(default_factory=dict)  # part name -> rgba override
    inst: int = -1  # pool instance, set when rendered


@dataclass
class Layout:
    seed: int
    split: str
    objects: list[Placed]
    pad: bool = True  # landing pad + marker on the truck (train hides it in 20 %: learn the shape, not the marker)

    @property
    def truck(self) -> Placed:
        return next(o for o in self.objects if o.role == "target")

    def to_json(self) -> dict:
        return {"seed": self.seed, "split": self.split, "pad": self.pad,
                "objects": [{"key": o.key, "variant": o.variant, "xy": [round(o.xy[0], 3), round(o.xy[1], 3)],
                             "yaw": round(o.yaw, 4), "role": o.role, "color": o.color} for o in self.objects]}


def _shade(rgb, rng: np.random.Generator, amount: float = 0.10, tint: float = 0.03) -> list[float]:
    """+-amount brightness, +-tint per channel (relative)."""
    s = rng.uniform(1 - amount, 1 + amount)
    return [float(np.clip(c * s * rng.uniform(1 - tint, 1 + tint), 0, 1)) for c in rgb] + [1.0]


def _recolor(meta: dict, key: str, rgb, rng: np.random.Generator) -> dict[str, list[float]]:
    out = {}
    for p in meta["files"]["parts"]:
        if p["name"] in BROWN_PARTS.get(key, ()):
            out[p["name"]] = _shade(rgb, rng, 0.0, 0.015) if out else _shade(rgb, rng)
    if out:  # all body parts share one shade (+ a tiny per-part tint)
        base = next(iter(out.values()))
        out = {k: [float(np.clip(c * rng.uniform(0.97, 1.03), 0, 1)) for c in base[:3]] + [1.0] for k in out}
    return out


def _default_jitter(meta: dict, rng: np.random.Generator, names: tuple[str, ...]) -> dict[str, list[float]]:
    s = rng.uniform(0.9, 1.1)
    return {p["name"]: [float(np.clip(c * s, 0, 1)) for c in p["rgba"][:3]] + [1.0]
            for p in meta["files"]["parts"] if p["name"] in names}


def sample_layout(lib: dict, seed: int, split: str) -> Layout:
    """Seeded random layout around B (origin). Always one brown truck, brown car, white car, tent, hut, tree."""
    rng = np.random.default_rng([seed, SPLITS.index(split), 17])
    for _attempt in range(100):
        objs: list[Placed] = []
        circles: list[tuple[float, float, float, int]] = []  # x, y, r, index
        gap = rng.uniform(0.6, 2.5)
        used: dict[tuple[str, str], int] = {}

        def pick(key: str) -> str | None:
            vs = [v for v in variant_pool(lib, key, split) if used.get((key, v), 0) < POOL[key]]
            if not vs:
                return None
            v = str(rng.choice(vs))
            used[(key, v)] = used.get((key, v), 0) + 1
            return v

        def place(key: str, role: str, color: str, rgba_fn, max_r: float = AREA_R) -> Placed | None:
            v = pick(key)
            if v is None:
                return None
            meta = lib[key][v]
            r = footprint_radius(meta)
            for _ in range(300):
                rho = max_r * math.sqrt(rng.random())
                ang = rng.uniform(-math.pi, math.pi)
                xy = (rho * math.cos(ang), rho * math.sin(ang))
                if rho + 0.4 * r > max_r:
                    continue
                if all(math.hypot(xy[0] - x, xy[1] - y) >= r + rr + gap for x, y, rr, _ in circles):
                    o = Placed(key, v, xy, float(rng.uniform(-math.pi, math.pi)), role, color, rgba_fn(meta))
                    circles.append((xy[0], xy[1], r, len(objs)))
                    objs.append(o)
                    return o
            used[(key, v)] -= 1
            return None

        brown = lambda key: (lambda meta: _recolor(meta, key, BROWN, rng))  # noqa: E731
        truck = place("truck", "target", "brown", brown("truck"))
        bcar = place("car", "brown_car", "brown", brown("car"))
        wcar = place("car", "white_car", "white", lambda meta: {"body": _shade(WHITE, rng, 0.08, 0.01)})
        tent = place("tent", "tent", "default", lambda meta: _default_jitter(meta, rng, ("fabric", "fabric_accent")))
        hut = place("hut", "hut", "default", lambda meta: _default_jitter(meta, rng, ("walls", "roof", "trim")))
        tree = place("tree", "tree", "default", lambda meta: _default_jitter(meta, rng, ("crown",)))
        if any(o is None for o in (truck, bcar, wcar, tent, hut, tree)):
            continue
        n_dis = int(rng.choice([0, 1, 2, 3], p=[0.25, 0.35, 0.25, 0.15]))
        for kind in rng.permutation(["pickup", "suv", "brown_hut"])[:n_dis]:
            if kind == "pickup":
                place("pickup", "distractor", "brown", brown("pickup"))
            elif kind == "suv":
                if rng.random() < 0.5:
                    place("suv", "distractor", "brown", brown("suv"))
                else:
                    cname = str(rng.choice(list(SUV_COLORS)))
                    place("suv", "distractor", cname, lambda meta, c=cname: {"body": _shade(SUV_COLORS[c], rng, 0.08)})
            else:
                place("hut", "distractor", "brown", lambda meta: {
                    "roof": _shade((0.36, 0.23, 0.13), rng), "walls": _shade(BROWN, rng)})
        # Trees hanging over vehicles (partial occlusion), trunk clear of the footprint
        n_occ = int(rng.choice([0, 1, 2], p=[0.5, 0.35, 0.15]))
        vehicles = [o for o in objs if o.key in ("truck", "car", "pickup", "suv")]
        for _ in range(n_occ):
            tgt = truck if rng.random() < 0.5 else vehicles[int(rng.integers(len(vehicles)))]
            v = pick("tree")
            if v is None:
                break
            meta = lib["tree"][v]
            cr = footprint_radius(meta)
            L, Wd = lib[tgt.key][tgt.variant]["size_m"][:2]
            side = rng.choice([-1.0, 1.0])
            lx = rng.uniform(-0.45, 0.45) * L
            ly = side * (Wd / 2 + 0.5 + cr * rng.uniform(0.15, 0.8))
            c, s = math.cos(tgt.yaw), math.sin(tgt.yaw)
            xy = (tgt.xy[0] + c * lx - s * ly, tgt.xy[1] + s * lx + c * ly)
            ok = all(math.hypot(xy[0] - x, xy[1] - y) >= rr + 0.8 for x, y, rr, i in circles if objs[i] is not tgt)
            if ok:
                circles.append((xy[0], xy[1], 0.6, len(objs)))
                objs.append(Placed("tree", v, xy, float(rng.uniform(-math.pi, math.pi)), "occluder", "default",
                                   _default_jitter(meta, rng, ("crown",))))
            else:
                used[("tree", v)] -= 1
        for _ in range(int(rng.integers(0, 3))):
            place("tree", "background", "default", lambda meta: _default_jitter(meta, rng, ("crown",)), max_r=AREA_R + 8)
        return Layout(seed, split, objs, pad=split != "train" or bool(rng.random() < 0.8))
    raise RuntimeError(f"could not place the detector scene (seed {seed}, {split})")


# ---------------------------------------------------------------------------------------- view

def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_axis(axis: np.ndarray, a: float) -> np.ndarray:
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def body_attitude(yaw: float, tilt: float, tilt_dir: float) -> np.ndarray:
    """Body -> world: heading `yaw`, then tilted by `tilt` about the horizontal axis at angle `tilt_dir`."""
    return rot_axis(np.array([math.cos(tilt_dir), math.sin(tilt_dir), 0.0]), tilt) @ rot_z(yaw)


@dataclass
class View:
    pos: np.ndarray  # (3,) camera position, world (z = height above the flat ground)
    R: np.ndarray  # (3, 3) body -> world attitude (camera = R @ R_BODY_CAM)
    tilt_deg: float
    sun_az: float  # rad
    sun_el: float  # rad above the horizon
    sun: float  # diffuse intensity
    ambient: float
    ground: int  # index into the split's ground texture list
    ground_xy: tuple[float, float]  # ground plane offset
    ground_yaw: float
    weather: str  # clear, fog, rain
    weather_i: float
    blur_px: float
    noise: float  # sensor noise, gray levels
    pad_gray: float

    def to_json(self) -> dict:
        return {"pos": [round(float(v), 3) for v in self.pos], "R": np.round(self.R, 5).tolist(),
                "tilt_deg": round(self.tilt_deg, 2), "sun_az_deg": round(math.degrees(self.sun_az), 1),
                "sun_el_deg": round(math.degrees(self.sun_el), 1), "sun": round(self.sun, 3),
                "ambient": round(self.ambient, 3), "ground": self.ground, "weather": self.weather,
                "weather_i": round(self.weather_i, 3), "blur_px": round(self.blur_px, 2), "noise": round(self.noise, 2)}


def sample_view(rng: np.random.Generator, n_grounds: int, alt: tuple[float, float] = (12.0, 45.0),
                tilt_max_deg: float = 25.0, offset_max: float = 6.0, weather_p: tuple[float, float] = (0.15, 0.15),
                layout: Layout | None = None, lib: dict | None = None) -> View:
    """Camera somewhere above B: altitude, tilt, heading, offset; sun, ground, weather, optics."""
    h = float(rng.uniform(*alt))
    for _ in range(50):
        rho = offset_max * math.sqrt(rng.random())
        ang = rng.uniform(-math.pi, math.pi)
        xy = np.array([rho * math.cos(ang), rho * math.sin(ang)])
        if layout is None or lib is None or _clear_of_trees(xy, h, layout, lib):
            break
    tilt = math.radians(float(rng.uniform(0.0, tilt_max_deg)))
    R = body_attitude(float(rng.uniform(-math.pi, math.pi)), tilt, float(rng.uniform(-math.pi, math.pi)))
    u = rng.random()
    weather, wi = "clear", 0.0
    if u < weather_p[0]:
        weather, wi = "fog", float(rng.uniform(0.2, 0.75))
    elif u < weather_p[0] + weather_p[1]:
        weather, wi = "rain", float(rng.uniform(0.2, 0.8))
    sun_el = math.radians(float(rng.uniform(25.0, 85.0)))
    overcast = weather != "clear"
    return View(np.array([xy[0], xy[1], h]), R, math.degrees(tilt), float(rng.uniform(-math.pi, math.pi)), sun_el,
                float(rng.uniform(0.25, 0.5) if overcast else rng.uniform(0.5, 0.95)),
                float(rng.uniform(0.35, 0.55) if overcast else rng.uniform(0.18, 0.4)),
                int(rng.integers(n_grounds)), (float(rng.uniform(-10, 10)), float(rng.uniform(-10, 10))),
                float(rng.uniform(-math.pi, math.pi)), weather, wi, float(rng.uniform(0.0, 1.0)),
                float(rng.uniform(0.0, 4.0)), float(rng.uniform(0.38, 0.6)))


def _clear_of_trees(xy: np.ndarray, h: float, layout: Layout, lib: dict) -> bool:
    """A low camera must not sit inside a crown."""
    for o in layout.objects:
        if o.key != "tree":
            continue
        meta = lib["tree"][o.variant]
        if h < meta["size_m"][2] + 3.0 and math.hypot(xy[0] - o.xy[0], xy[1] - o.xy[1]) < footprint_radius(meta) + 1.5:
            return False
    return True


# ---------------------------------------------------------------------------------------- ground textures

def _fbm(rng: np.random.Generator, n: int, scales: list[tuple[float, float]]) -> np.ndarray:
    out = np.zeros((n, n), np.float32)
    for sigma, amp in scales:
        z = rng.normal(0, 1, (n, n)).astype(np.float32)
        z = cv2.GaussianBlur(z, (0, 0), sigma)
        out += amp * z / max(float(z.std()), 1e-6)
    return out


def _detail(img: np.ndarray, rng: np.random.Generator, amp: float = 5.0) -> np.ndarray:
    n = img.shape[0]
    d = _fbm(rng, n, [(0.8, 0.6), (2.5, 0.4)]) * amp
    return np.clip(img.astype(np.float32) + d[..., None], 0, 255).astype(np.uint8)


def _grass(rng: np.random.Generator, n: int, dry: bool = False) -> np.ndarray:
    if dry:
        base = np.array([rng.uniform(150, 185), rng.uniform(135, 160), rng.uniform(80, 105)], np.float32)
        tint = np.array([1.0, 0.9, 0.55], np.float32)
    else:
        g = rng.uniform(95, 135)
        base = np.array([g * rng.uniform(0.62, 0.85), g, g * rng.uniform(0.4, 0.55)], np.float32)
        tint = np.array([0.8, 1.0, 0.5], np.float32)
    tone = _fbm(rng, n, [(120, 9.0), (35, 7.0), (8, 5.0), (1.5, 6.0)])
    img = base + tone[..., None] * tint
    if not dry and rng.random() < 0.4:  # mowing stripes
        w = int(rng.integers(20, 40))
        stripes = ((np.arange(n) // w) % 2).astype(np.float32) * rng.uniform(5, 10)
        img += (stripes[None, :, None] if rng.random() < 0.5 else stripes[:, None, None])
    for _ in range(int(rng.integers(3, 12))):  # bare / dirt patches
        c = rng.integers(0, n, 2)
        ax = rng.integers(10, 60, 2)
        m = np.zeros((n, n), np.float32)
        cv2.ellipse(m, (int(c[0]), int(c[1])), (int(ax[0]), int(ax[1])), float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), 8)[..., None]
        img = img * (1 - 0.6 * m) + np.array([140, 120, 90], np.float32) * 0.6 * m
    return np.clip(img, 0, 255).astype(np.uint8)


def _asphalt(rng: np.random.Generator, n: int, concrete: bool = False) -> np.ndarray:
    g = rng.uniform(160, 192) if concrete else rng.uniform(55, 95)
    tone = _fbm(rng, n, [(90, 5.0), (20, 4.0), (1.0, 7.0 if not concrete else 4.0)])
    img = np.full((n, n, 3), g, np.float32) + tone[..., None] * np.array([1.0, 1.0, 1.04], np.float32)
    mpp = 2 * GROUND_HALF / n
    if concrete:  # slab joints every 4-6 m
        step = int(rng.uniform(4, 6) / mpp)
        for k in range(0, n, step):
            img[:, k:k + 2] *= 0.7
            img[k:k + 2, :] *= 0.7
    else:
        for _ in range(int(rng.integers(5, 25))):  # cracks (sealed, dark)
            p = rng.uniform(0, n, 2)
            pts = [p]
            a = rng.uniform(0, 2 * math.pi)
            for _ in range(int(rng.integers(4, 12))):
                a += rng.normal(0, 0.6)
                pts.append(pts[-1] + 12 * np.array([math.cos(a), math.sin(a)]))
            cv2.polylines(img, [np.array(pts, np.int32)], False, (30, 30, 32), 1, cv2.LINE_AA)
        if rng.random() < 0.6:  # parking bay lines
            col = (225, 225, 220) if rng.random() < 0.7 else (215, 185, 60)
            step, L = int(2.6 / mpp), int(5.0 / mpp)
            for row in range(int(rng.integers(2, 6))):
                y0 = int(rng.integers(0, n - L))
                x0 = int(rng.integers(0, n // 2))
                for k in range(int(rng.integers(8, 30))):
                    x = x0 + k * step
                    if x < n:
                        cv2.line(img, (x, y0), (x, y0 + L), col, 1, cv2.LINE_AA)
    for _ in range(int(rng.integers(3, 15))):  # stains
        c = rng.integers(0, n, 2)
        m = np.zeros((n, n), np.float32)
        cv2.circle(m, (int(c[0]), int(c[1])), int(rng.integers(5, 30)), 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), 6)[..., None]
        img = img * (1 - 0.25 * m)
    return np.clip(img, 0, 255).astype(np.uint8)


def _naip(rng: np.random.Generator, n: int, ortho: np.ndarray, extent: dict, y_range: tuple[float, float]) -> np.ndarray:
    """A 2 * GROUND_HALF m square crop of the Berkeley ortho whose center lies in the y band `y_range`."""
    mpp = (extent["max_x"] - extent["min_x"]) / ortho.shape[1]
    side = 2 * GROUND_HALF
    half_px = int(round(side / 2 / mpp))
    cx = rng.uniform(extent["min_x"] + side / 2, extent["max_x"] - side / 2)
    cy = rng.uniform(*y_range)
    col = int((cx - extent["min_x"]) / mpp)
    row = int((extent["max_y"] - cy) / mpp)
    crop = ortho[max(row - half_px, 0):row + half_px, max(col - half_px, 0):col + half_px]
    crop = cv2.resize(crop, (n, n), interpolation=cv2.INTER_CUBIC)
    crop = np.rot90(crop, int(rng.integers(4)))
    if rng.random() < 0.5:
        crop = crop[:, ::-1]
    crop = cv2.convertScaleAbs(np.ascontiguousarray(crop), alpha=float(rng.uniform(0.85, 1.15)), beta=float(rng.uniform(-12, 12)))
    return _detail(crop, rng, 6.0)


# (kind, count) per split group. "test" grounds never appear in train / val.
GROUNDS = {
    "trainval": [("naip_south", 28), ("grass", 14), ("asphalt", 14)],
    "test": [("naip_north", 10), ("dry_grass", 6), ("concrete", 6), ("grass", 3), ("asphalt", 3)],
}
NAIP_Y = {"naip_south": (-433.86 + GROUND_HALF, 60.0), "naip_north": (170.0, 433.86 - GROUND_HALF)}
GROUND_VERSION = 1


def ensure_ground_textures(root: Path) -> dict[str, list[dict]]:
    """Write the ground texture pools once (JPEG, 2048 px over 220 m) and return {group: [{file, kind}]}."""
    root = Path(root) / f"grounds_v{GROUND_VERSION}"
    idx_file = root / "index.json"
    if idx_file.exists():
        return json.loads(idx_file.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    man = json.loads((SAMPLE_WORLD_DIR / "manifest.json").read_text(encoding="utf-8"))
    ortho = cv2.cvtColor(cv2.imread(str(SAMPLE_WORLD_DIR / man["ortho"]["file"])), cv2.COLOR_BGR2RGB)
    ext = man["ortho"]["extent_enu"]
    index: dict[str, list[dict]] = {}
    for gi, (group, kinds) in enumerate(GROUNDS.items()):
        rng = np.random.default_rng([GROUND_VERSION, gi, 99])
        index[group] = []
        for kind, count in kinds:
            for k in range(count):
                if kind.startswith("naip"):
                    img = _naip(rng, TEX_PX, ortho, ext, NAIP_Y[kind])
                elif kind in ("grass", "dry_grass"):
                    img = _grass(rng, TEX_PX, dry=kind == "dry_grass")
                else:
                    img = _asphalt(rng, TEX_PX, concrete=kind == "concrete")
                f = f"{group}_{kind}_{k:02d}.jpg"
                cv2.imwrite(str(root / f), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
                index[group].append({"file": str(root / f), "kind": kind})
    idx_file.write_text(json.dumps(index, indent=1), encoding="utf-8")
    return index


def ground_group(split: str) -> str:
    return "test" if split == "test" else "trainval"


# ---------------------------------------------------------------------------------------- MuJoCo scene

def pad_texture(px_per_m: int = 200) -> np.ndarray:
    """Marker plane: white quiet zone + the ArUco marker (the pad around it is a plain gray box)."""
    n = int(round((MARKER_SIZE + 2 * MARKER_QUIET) * px_per_m))
    m = int(round(MARKER_SIZE * px_per_m))
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, MARKER_DICT))
    marker = cv2.aruco.generateImageMarker(d, MARKER_ID, m)
    img = np.full((n, n), 225, np.uint8)
    o = (n - m) // 2
    img[o:o + m, o:o + m] = np.where(marker > 127, 225, 25).astype(np.uint8)
    return np.ascontiguousarray(np.repeat(img[..., None], 3, axis=2))


def _quat_from_R(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, float).flatten())
    return q


@dataclass
class Instance:
    key: str
    variant: str
    body: int
    geoms: dict[str, int]  # part name -> geom id
    default_rgba: dict[str, np.ndarray]
    pad_geoms: tuple[int, ...] = ()


@dataclass
class Label:
    cls: int
    key: str
    role: str
    color: str
    variant: str
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 px (visible pixels)
    visible_frac: float  # visible / unoccluded-by-trees pixels
    pixels: int
    xy: tuple[float, float]


@dataclass
class RenderOut:
    rgb: np.ndarray  # (H, W, 3) uint8, after camera effects
    labels: list[Label]  # kept boxes (>= min size, <= max occlusion)
    dropped: list[Label]  # visible but dropped (too small / too occluded)
    seg: np.ndarray | None = None  # (H, W) instance index (-1 = background) if requested


class DetectorScene:
    """One MuJoCo model with a pool of all library variants; `render(layout, view)` -> image + labels."""

    def __init__(self, lib: dict, grounds: dict[str, list[dict]], width: int = W, height: int = H,
                 fovy: float = FOVY, shadow: tuple[int, float] = (4096, 0.4)) -> None:
        self.lib = lib
        self.shadow = shadow  # (shadow map px, shadowclip); MuJoCo defaults (1024, 1.0) speckle
        self.grounds = grounds
        self.width, self.height, self.fovy = width, height, fovy
        self.instances: list[Instance] = []
        self.model = self._build()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.cam_id = m.camera("cam").id
        self.sun_id = m.light("sun").id
        self.tex_id = m.texture("ground").id
        self.plane_id = m.geom("ground").id
        self.pad_z = {g: float(m.geom_pos[g][2]) for inst in self.instances for g in inst.pad_geoms}
        self.geom_inst = np.full(m.ngeom, -1, np.int32)
        for i, inst in enumerate(self.instances):
            for g in list(inst.geoms.values()) + list(inst.pad_geoms):
                self.geom_inst[g] = i
        self.renderer = mujoco.Renderer(m, height, width)
        self._loaded_ground: str | None = None

    # -------------------------------------------------------------- build

    def _build(self) -> mujoco.MjModel:
        spec = mujoco.MjSpec()
        spec.stat.extent = 100.0
        spec.stat.center = [0, 0, 0]
        vis = spec.visual
        vis.map.znear = 0.001  # 0.1 m
        vis.map.zfar = 30.0  # 3 km
        # Shadow acne fix: a 4096 px map over a +-40 m box around the scene (the default clip of a 100 m extent
        # spreads 2048 px over 200 m -> 10 cm texels and self-shadow speckle on every flat surface).
        vis.quality.shadowsize, vis.map.shadowclip = self.shadow
        vis.quality.offsamples = 4
        vis.global_.offwidth = max(self.width, 640)
        vis.global_.offheight = max(self.height, 480)
        vis.headlight.diffuse = [0.0, 0.0, 0.0]
        vis.headlight.specular = [0.0, 0.0, 0.0]
        vis.headlight.ambient = [0.3, 0.3, 0.3]
        vis.rgba.haze = [0.75, 0.8, 0.88, 1.0]
        tex = spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                               rgb1=[0.6, 0.72, 0.85], rgb2=[0.85, 0.9, 0.95], width=64, height=64)
        del tex
        g = spec.add_texture(name="ground", type=mujoco.mjtTexture.mjTEXTURE_2D)
        g.width, g.height, g.nchannel = TEX_PX, TEX_PX, 3
        g.data = np.full((TEX_PX, TEX_PX, 3), 100, np.uint8).tobytes()
        for name, rep in (("ground", [1, 1]), ("ground_far", [18, 18])):
            mat = spec.add_material(name=name, texrepeat=rep, texuniform=False, reflectance=0.0, specular=0.05)
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "ground"
        wb = spec.worldbody
        wb.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[GROUND_HALF, GROUND_HALF, 1.0],
                    material="ground")
        wb.add_geom(name="ground_far", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2000, 2000, 1.0],
                    pos=[0, 0, -0.08], material="ground_far", rgba=[0.85, 0.85, 0.85, 1])
        wb.add_light(name="sun", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=[0, 0, 60], dir=[0, 0, -1],
                     castshadow=True, diffuse=[0.8, 0.8, 0.8], ambient=[0.1, 0.1, 0.1], specular=[0.15, 0.15, 0.15])
        wb.add_camera(name="cam", pos=[0, 0, 30], fovy=self.fovy)
        ptex = pad_texture()
        t = spec.add_texture(name="marker", type=mujoco.mjtTexture.mjTEXTURE_2D)
        t.width, t.height, t.nchannel = ptex.shape[1], ptex.shape[0], 3
        t.data = ptex.tobytes()
        mm = spec.add_material(name="marker", texrepeat=[1, 1], texuniform=False, reflectance=0.0)
        mm.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "marker"

        meshes: set[str] = set()
        pending: list[tuple[Instance, Any, list]] = []
        for key in CLASSES:
            for variant, meta in sorted(self.lib[key].items()):
                for k in range(POOL[key]):
                    name = f"{key}_{variant}_{k}"
                    body = wb.add_body(name=name, pos=[0, 0, HIDDEN_Z])
                    parts = []
                    for p in meta["files"]["parts"]:
                        mname = f"{variant}_{p['name']}"
                        if mname not in meshes:
                            mesh = spec.add_mesh(name=mname, file=p["abspath"])
                            mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
                            meshes.add(mname)
                        body.add_geom(name=f"{name}_{p['name']}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mname,
                                      rgba=list(p["rgba"]), contype=0, conaffinity=0, mass=1.0)
                        parts.append(p)
                    pads = []
                    if key == "truck":
                        la = meta["landing_area"]
                        cx, cy, cz = la["center"]
                        body.add_geom(name=f"{name}_pad", type=mujoco.mjtGeom.mjGEOM_BOX,
                                      size=[PAD_SIZE / 2, PAD_SIZE / 2, 0.015], pos=[cx, cy, cz + 0.015],
                                      rgba=[0.5, 0.5, 0.5, 1], contype=0, conaffinity=0, mass=1.0)
                        s = MARKER_SIZE / 2 + MARKER_QUIET
                        body.add_geom(name=f"{name}_marker", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[s, s, 1.0],
                                      pos=[cx, cy, cz + 0.033], material="marker", contype=0, conaffinity=0)
                        pads = [f"{name}_pad", f"{name}_marker"]
                    pending.append((Instance(key, variant, -1, {}, {}), name, [parts, pads]))
        model = spec.compile()
        for inst, name, (parts, pads) in pending:
            inst.body = model.body(name).id
            inst.geoms = {p["name"]: model.geom(f"{name}_{p['name']}").id for p in parts}
            inst.default_rgba = {p["name"]: np.array(p["rgba"], float) for p in parts}
            inst.pad_geoms = tuple(model.geom(n).id for n in pads)
            self.instances.append(inst)
        return model

    # -------------------------------------------------------------- per sample

    def _set_ground(self, split: str, view: View) -> str:
        pool = self.grounds[ground_group(split)]
        entry = pool[view.ground % len(pool)]
        if self._loaded_ground != entry["file"]:
            img = cv2.cvtColor(cv2.imread(entry["file"]), cv2.COLOR_BGR2RGB)
            m = self.model
            adr = m.tex_adr[self.tex_id]
            m.tex_data[adr:adr + img.size] = img.reshape(-1)
            self.renderer._gl_context.make_current()
            mujoco.mjr_uploadTexture(m, self.renderer._mjr_context, self.tex_id)
            self._loaded_ground = entry["file"]
        self.model.geom_pos[self.plane_id] = [view.ground_xy[0], view.ground_xy[1], 0.0]
        self.model.geom_quat[self.plane_id] = [math.cos(view.ground_yaw / 2), 0, 0, math.sin(view.ground_yaw / 2)]
        return entry["kind"]

    def apply_layout(self, layout: Layout) -> None:
        m = self.model
        for inst in self.instances:
            m.body_pos[inst.body] = [0.0, 0.0, HIDDEN_Z]
        free: dict[tuple[str, str], list[int]] = {}
        for i, inst in enumerate(self.instances):
            free.setdefault((inst.key, inst.variant), []).append(i)
        for g, z in self.pad_z.items():  # hidden pad: sunk 1 m into the cargo box
            m.geom_pos[g][2] = z if layout.pad else z - 1.0
        for o in layout.objects:
            i = free[(o.key, o.variant)].pop(0)
            o.inst = i
            inst = self.instances[i]
            m.body_pos[inst.body] = [o.xy[0], o.xy[1], 0.0]
            m.body_quat[inst.body] = [math.cos(o.yaw / 2), 0, 0, math.sin(o.yaw / 2)]
            for part, gid in inst.geoms.items():
                m.geom_rgba[gid] = o.rgba.get(part, inst.default_rgba[part])

    def _set_view(self, view: View) -> None:
        m = self.model
        R_wc = view.R @ R_BODY_CAM
        m.cam_pos[self.cam_id] = view.pos
        m.cam_quat[self.cam_id] = _quat_from_R(R_wc)
        m.cam_fovy[self.cam_id] = self.fovy
        d = np.array([-math.cos(view.sun_el) * math.cos(view.sun_az), -math.cos(view.sun_el) * math.sin(view.sun_az),
                      -math.sin(view.sun_el)])
        target = np.array([view.pos[0], view.pos[1], 0.0])
        m.light_dir[self.sun_id] = d
        m.light_pos[self.sun_id] = target - 60.0 * d
        m.light_diffuse[self.sun_id] = [view.sun * 1.0, view.sun * 0.98, view.sun * 0.93]
        m.light_ambient[self.sun_id] = [0.0, 0.0, 0.0]
        m.vis.headlight.ambient[:] = [view.ambient, view.ambient, view.ambient * 1.04]
        for inst in self.instances:
            if inst.pad_geoms:
                m.geom_rgba[inst.pad_geoms[0]] = [view.pad_gray, view.pad_gray, view.pad_gray, 1.0]

    def _seg(self) -> np.ndarray:
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=self.cam_id)
        s = self.renderer.render()
        self.renderer.disable_segmentation_rendering()
        ids = s[..., 0]
        geom = (s[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (ids >= 0)
        out = np.full(ids.shape, -1, np.int32)
        out[geom] = self.geom_inst[ids[geom]]
        return out

    def render(self, layout: Layout, view: View, rng: np.random.Generator, min_box_px: int = 8,
               max_occlusion: float = 0.8, keep_seg: bool = False, effects: bool = True,
               layout_applied: bool = False) -> RenderOut:
        if not layout_applied:
            self.apply_layout(layout)
        self._set_ground(layout.split, view)
        self._set_view(view)
        m, d = self.model, self.data
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)
        self.renderer.update_scene(d, camera=self.cam_id)
        rgb = self.renderer.render().copy()
        seg = self._seg()
        # Same view without the trees: unoccluded pixel count of every other object
        trees = [o for o in layout.objects if o.key == "tree"]
        for o in trees:
            m.body_pos[self.instances[o.inst].body][2] = HIDDEN_Z
        mujoco.mj_kinematics(m, d)
        seg_free = self._seg()
        for o in trees:
            m.body_pos[self.instances[o.inst].body][2] = 0.0
        mujoco.mj_kinematics(m, d)
        if effects:
            rgb = camera_effects(rgb, view, rng)
        labels, dropped = [], []
        for o in layout.objects:
            mask = seg == o.inst
            n = int(mask.sum())
            if n == 0:
                continue
            n_free = n if o.key == "tree" else max(int((seg_free == o.inst).sum()), n)
            ys, xs = np.nonzero(mask)
            box = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
            lab = Label(CLASSES.index(o.key), o.key, o.role, o.color, o.variant, box, n / n_free, n, o.xy)
            small = (box[2] - box[0]) < min_box_px or (box[3] - box[1]) < min_box_px
            (dropped if small or lab.visible_frac < 1.0 - max_occlusion else labels).append(lab)
        return RenderOut(rgb, labels, dropped, seg if keep_seg else None)

    def close(self) -> None:
        self.renderer.close()


def camera_effects(rgb: np.ndarray, view: View, rng: np.random.Generator) -> np.ndarray:
    """Optics blur, weather through the slant range (degradation levers), sensor noise."""
    img = rgb
    if view.blur_px > 0.05:
        img = cv2.GaussianBlur(img, (0, 0), view.blur_px)
    if view.weather != "clear":
        eff = lever.weather_effect(WeatherConditions(view.weather, view.weather_i))
        cos_t = max(math.cos(math.radians(view.tilt_deg)), 0.3)
        img = lever.degrade_image(img, eff, float(view.pos[2]) / cos_t, rng)
    if view.noise > 0.05:
        img = np.clip(img.astype(np.float32) + rng.normal(0, view.noise, img.shape), 0, 255).astype(np.uint8)
    return img


def draw_labels(rgb: np.ndarray, labels: list[Label], names: tuple[str, ...] = CLASSES) -> np.ndarray:
    colors = [(255, 60, 60), (60, 160, 255), (255, 200, 0), (200, 80, 255), (255, 128, 0), (0, 220, 220), (60, 220, 60)]
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for lab in labels:
        c = colors[lab.cls % len(colors)]
        x0, y0, x1, y1 = (int(round(v)) for v in lab.box)
        cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), c, 1)
        txt = names[lab.cls] + (f" {lab.color}" if lab.color in ("brown", "white") else "")
        cv2.putText(out, txt, (x0, max(y0 - 3, 9)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
