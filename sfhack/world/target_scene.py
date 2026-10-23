"""Randomized target scene (V4): six objects in a search area, the landing station on the brown truck.

SIMULATOR SIDE. This module holds the TRUTH of the layout (which object is where, which one is the truck).
`sfhack/sim/target_world.py` builds it into MuJoCo and the harness scores against it; onboard code never
imports it (tests/test_target_truth.py). The drone only gets the mission brief (`config.TargetConfig`).

Objects (realistic sizes, meters): white car and brown car ~4.5 x 1.8 x 1.5 (body + cabin), brown truck
~8 x 2.5 x 3.2 (cab + cargo box, landing pad with an ArUco marker on the cargo roof), tent ~3 x 3 x 2
(pyramid), hut ~4 x 4 x 3 (pitched roof), tree 6-10 m (trunk + crown). Brown shades of car and truck come
from the same distribution, so color alone cannot tell them apart.

Difficulty: `easy` = well separated; `hard` = the brown car parked right next to the truck and a tree whose
crown hangs over the truck's cab (partial occlusion; the pad stays visible).

Also builds the plain-field map package for `--standalone` flights (ortho + manifest, $SFHACK_DATA/v4):
the simulator renders it as the ground, the drone loads it as its onboard map (no objects in it).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

KINDS = ("white_car", "brown_car", "brown_truck", "tent", "hut", "tree")
DIFFICULTIES = ("easy", "hard")
FIELD_PKG_VERSION = 1


@dataclass
class Part:
    """One primitive of an object, in the object frame (x = long axis, z up, origin on the ground).

    box: size = half sizes; cylinder: (radius, half height); ellipsoid: radii; pyramid: (half x, half y,
    height), base at pos z; prism: (half x, half y, height), ridge along x, base at pos z.
    """

    shape: str
    pos: tuple[float, float, float]
    size: tuple[float, float, float]
    rgb: tuple[float, float, float]
    collide: bool = True
    axis: str = "z"  # cylinder axis in the object frame: "z" (trunk) or "y" (wheel)


@dataclass
class SceneObject:
    id: int
    kind: str
    xy: np.ndarray  # (2,) footprint center, world ENU
    yaw: float  # rad, direction of the long axis
    length: float  # overall footprint, m
    width: float
    height: float  # top, m above ground
    color: tuple[float, float, float]  # dominant top color (albedo 0..1)
    parts: list[Part] = field(default_factory=list)

    @property
    def radius(self) -> float:
        """Half diagonal of the footprint (placement spacing)."""
        return 0.5 * math.hypot(self.length, self.width)

    def to_world(self, local_xy: np.ndarray) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        p = np.asarray(local_xy, float)
        return self.xy + np.stack([c * p[..., 0] - s * p[..., 1], s * p[..., 0] + c * p[..., 1]], axis=-1)

    def to_local(self, xy: np.ndarray) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        d = np.asarray(xy, float) - self.xy
        return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], axis=-1)

    def footprint(self) -> np.ndarray:
        """(4, 2) world corners of the footprint rectangle."""
        hl, hw = self.length / 2, self.width / 2
        return self.to_world(np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]]))

    def contains(self, xy: np.ndarray, margin: float = 0.0) -> bool:
        p = self.to_local(xy)
        return bool(abs(p[0]) <= self.length / 2 + margin and abs(p[1]) <= self.width / 2 + margin)


@dataclass
class TargetLayout:
    seed: int
    difficulty: str
    center: np.ndarray  # (2,) search area center (= B)
    half: float  # search area half size
    objects: list[SceneObject]
    pad_xy: np.ndarray  # (2,) landing pad center
    pad_z: float  # top surface of the pad, m above ground
    pad_yaw: float
    pad_size: float
    marker_dict: str
    marker_id: int
    marker_size: float

    @property
    def truck(self) -> SceneObject:
        return next(o for o in self.objects if o.kind == "brown_truck")

    def by_kind(self, kind: str) -> SceneObject:
        return next(o for o in self.objects if o.kind == kind)

    def object_at(self, xy: np.ndarray, margin: float = 0.3) -> SceneObject | None:
        """The object whose footprint contains `xy` (tree: within the crown radius of the trunk)."""
        for o in self.objects:
            if o.contains(xy, margin):
                return o
        return None

    def nearest(self, xy: np.ndarray) -> tuple[SceneObject, float]:
        d = [float(np.linalg.norm(o.xy - np.asarray(xy, float)[:2])) for o in self.objects]
        i = int(np.argmin(d))
        return self.objects[i], d[i]

    def on_pad(self, xy: np.ndarray, margin: float = 0.0) -> bool:
        p = np.asarray(xy, float)[:2] - self.pad_xy
        c, s = math.cos(self.pad_yaw), math.sin(self.pad_yaw)
        lx, ly = c * p[0] + s * p[1], -s * p[0] + c * p[1]
        h = self.pad_size / 2 + margin
        return abs(lx) <= h and abs(ly) <= h

    def describe(self) -> str:
        rows = [f"layout seed {self.seed} ({self.difficulty}), area {2 * self.half:.0f} m around "
                f"({self.center[0]:.1f}, {self.center[1]:.1f})"]
        for o in self.objects:
            rgb = ",".join(f"{v:.2f}" for v in o.color)
            rows.append(f"  {o.id} {o.kind:12s} at ({o.xy[0]:7.1f}, {o.xy[1]:7.1f}) yaw {math.degrees(o.yaw):5.0f} deg "
                        f"{o.length:.1f} x {o.width:.1f} x {o.height:.1f} m  rgb {rgb}")
        rows.append(f"  pad at ({self.pad_xy[0]:.1f}, {self.pad_xy[1]:.1f}), top {self.pad_z:.2f} m, marker "
                    f"{self.marker_dict} #{self.marker_id}")
        return "\n".join(rows)


# ---------------------------------------------------------------------------------------- colors

def brown(rng: np.random.Generator) -> tuple[float, float, float]:
    """Shared brown distribution of the brown car and the truck (overlapping on purpose)."""
    r = rng.uniform(0.36, 0.50)
    return (r, r * rng.uniform(0.62, 0.72), r * rng.uniform(0.32, 0.45))


def _jitter(rgb: Iterable[float], rng: np.random.Generator, amount: float = 0.04) -> tuple[float, float, float]:
    return tuple(float(np.clip(v + rng.uniform(-amount, amount), 0.0, 1.0)) for v in rgb)  # type: ignore[return-value]


# ---------------------------------------------------------------------------------------- object models

def _car(rng: np.random.Generator, rgb: tuple[float, float, float]) -> tuple[float, float, float, list[Part]]:
    L, W, H = rng.uniform(4.2, 4.8), rng.uniform(1.75, 1.9), rng.uniform(1.4, 1.6)
    body_top = 0.62 * H
    cab_len = 0.5 * L
    dark = (0.06, 0.06, 0.07)
    parts = [
        Part("box", (0.0, 0.0, (0.18 + body_top) / 2), (L / 2, W / 2, (body_top - 0.18) / 2), rgb),
        Part("box", (-0.06 * L, 0.0, (body_top + H) / 2), (cab_len / 2, 0.44 * W, (H - body_top) / 2), rgb),
        # windshield band in front of the cabin roof (dark glass, seen obliquely from the air)
        Part("box", (-0.06 * L + cab_len / 2 + 0.05, 0.0, (body_top + H) / 2 - 0.05),
             (0.05, 0.42 * W, (H - body_top) / 2 - 0.05), dark),
    ]
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(Part("cylinder", (sx * 0.32 * L, sy * (W / 2 - 0.12), 0.33), (0.33, 0.12, 0), dark, axis="y"))
    return L, W, H, parts


def _truck(rng: np.random.Generator, rgb: tuple[float, float, float], pad_size: float) -> tuple[float, float, float, list[Part], float]:
    """Returns (L, W, H, parts, pad local x). Cab at +x (front), cargo box behind it."""
    L, W, H = rng.uniform(7.6, 8.4), rng.uniform(2.4, 2.6), rng.uniform(3.1, 3.3)
    cab_len, gap = 2.2, 0.2
    cargo_len = L - cab_len - gap
    x_cargo = -L / 2 + cargo_len / 2
    x_cab = L / 2 - cab_len / 2
    cab_rgb = _jitter(rgb, rng, 0.03)
    dark = (0.08, 0.08, 0.09)
    parts = [
        Part("box", (0.0, 0.0, 0.55), (L / 2 - 0.1, W / 2 - 0.2, 0.2), dark),  # chassis
        Part("box", (x_cargo, 0.0, (0.9 + H) / 2), (cargo_len / 2, W / 2, (H - 0.9) / 2), rgb),
        Part("box", (x_cab, 0.0, (0.45 + 0.84 * H) / 2), (cab_len / 2, W / 2, (0.84 * H - 0.45) / 2), cab_rgb),
        Part("box", (L / 2 + 0.01, 0.0, 0.62 * H), (0.02, W / 2 - 0.15, 0.13 * H), dark),  # windshield
    ]
    for x in (x_cab, x_cargo - 0.25 * cargo_len, x_cargo + 0.25 * cargo_len):
        for sy in (-1, 1):
            parts.append(Part("cylinder", (x, sy * (W / 2 - 0.25), 0.5), (0.5, 0.2, 0), dark, axis="y"))
    # Landing pad on the cargo roof (the marker plane is added by the MuJoCo builder)
    parts.append(Part("pad", (x_cargo, 0.0, H + 0.03), (pad_size / 2, pad_size / 2, 0.03), (0.85, 0.85, 0.85)))
    return L, W, H, parts, x_cargo


def _tent(rng: np.random.Generator) -> tuple[float, float, float, list[Part], tuple[float, float, float]]:
    s, H = rng.uniform(2.7, 3.2), rng.uniform(1.8, 2.2)
    rgb = _jitter((0.16, 0.48, 0.18) if rng.random() < 0.5 else (0.92, 0.48, 0.10), rng, 0.05)
    return s, s, H, [Part("pyramid", (0.0, 0.0, 0.0), (s / 2, s / 2, H), rgb)], rgb


def _hut(rng: np.random.Generator) -> tuple[float, float, float, list[Part], tuple[float, float, float]]:
    L, W = rng.uniform(3.7, 4.3), rng.uniform(3.7, 4.3)
    wall_h, roof_h, over = 2.0, rng.uniform(0.9, 1.1), 0.25
    walls = _jitter((0.78, 0.68, 0.52), rng, 0.04)
    g = rng.uniform(0.30, 0.40)
    roof = _jitter((g, g + 0.02, g + 0.06), rng, 0.02)  # slate gray
    parts = [
        Part("box", (0.0, 0.0, wall_h / 2), (L / 2, W / 2, wall_h / 2), walls),
        Part("prism", (0.0, 0.0, wall_h), (L / 2 + over, W / 2 + over, roof_h), roof),
    ]
    return L + 2 * over, W + 2 * over, wall_h + roof_h, parts, roof


def _tree(rng: np.random.Generator) -> tuple[float, float, float, list[Part], tuple[float, float, float]]:
    H = rng.uniform(6.0, 10.0)
    r = rng.uniform(2.0, 2.8)
    rz = 1.1 * r
    crown = _jitter((0.18, 0.40, 0.15), rng, 0.04)
    zc = H - rz
    parts = [
        Part("cylinder", (0.0, 0.0, zc / 2), (0.25, zc / 2, 0), (0.36, 0.27, 0.18)),
        Part("ellipsoid", (0.0, 0.0, zc), (r, r, rz), crown, collide=False),
    ]
    return 2 * r, 2 * r, H, parts, crown


# ---------------------------------------------------------------------------------------- layout

def generate_layout(
    seed: int,
    difficulty: str = "easy",
    center: Iterable[float] = (0.0, 0.0),
    half: float = 30.0,
    pad_size: float = 2.0,
    marker_dict: str = "DICT_6X6_50",
    marker_id: int = 7,
    marker_size: float = 1.6,
    keep_out: Iterable[tuple[float, float, float]] = (),
) -> TargetLayout:
    """Seeded random layout. `keep_out`: (x, y, radius) circles no object may touch (spawn point, buildings)."""
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty {difficulty!r} not in {DIFFICULTIES}")
    rng = np.random.default_rng(seed)
    c = np.asarray(list(center), float)
    keep = [(float(x), float(y), float(r)) for x, y, r in keep_out]
    hard = difficulty == "hard"
    gap = 3.0 if hard else 5.0  # minimum clearance between footprint circles of unrelated objects

    for _attempt in range(200):
        objs: list[SceneObject] = []
        ok = True

        def free(xy: np.ndarray, radius: float, ignore: tuple[int, ...] = ()) -> bool:
            if np.any(np.abs(xy - c) > half - radius - 1.0):
                return False
            if any(math.hypot(xy[0] - x, xy[1] - y) < radius + r for x, y, r in keep):
                return False
            return all(float(np.linalg.norm(xy - o.xy)) >= radius + o.radius + gap
                       for o in objs if o.id not in ignore)

        def place(kind: str, dims: tuple, parts: list[Part], rgb, at: np.ndarray | None = None,
                  yaw: float | None = None, ignore: tuple[int, ...] = ()) -> SceneObject | None:
            L, W, H = dims
            radius = 0.5 * math.hypot(L, W)
            for _ in range(300):
                xy = at if at is not None else c + rng.uniform(-half, half, 2)
                if free(xy, radius, ignore):
                    o = SceneObject(len(objs), kind, xy.copy(), float(yaw if yaw is not None else rng.uniform(-math.pi, math.pi)),
                                    float(L), float(W), float(H), tuple(float(v) for v in rgb), parts)
                    objs.append(o)
                    return o
                if at is not None:
                    return None
            return None

        # 1. truck (with the pad)
        t_rgb = brown(rng)
        L, W, H, parts, pad_x = _truck(rng, t_rgb, pad_size)
        truck = place("brown_truck", (L, W, H), parts, t_rgb)
        if truck is None:
            continue
        # 2. brown car: hard = parked alongside the truck (2.3-3.3 m between the sides)
        b_rgb = brown(rng)
        cL, cW, cH, cparts = _car(rng, b_rgb)
        if hard:
            side = rng.choice([-1.0, 1.0])
            off = np.array([rng.uniform(-2.0, 2.0), side * (W / 2 + rng.uniform(2.3, 3.3) + cW / 2)])
            at = truck.to_world(off)
            yaw = truck.yaw + (math.pi if rng.random() < 0.5 else 0.0) + rng.uniform(-0.15, 0.15)
            car = place("brown_car", (cL, cW, cH), cparts, b_rgb, at=at, yaw=yaw, ignore=(truck.id,))
        else:
            car = place("brown_car", (cL, cW, cH), cparts, b_rgb)
        # 3. tree: hard = crown over the truck's cab (the pad on the cargo roof stays visible)
        tL, tW, tH, tparts, t_crown = _tree(rng)
        if hard and car is not None:
            r = tL / 2
            tree_side = -float(np.sign(truck.to_local(car.xy)[1]) or 1.0)  # the side away from the car
            local = np.array([L / 2 - rng.uniform(0.3, 1.0), tree_side * (W / 2 + rng.uniform(0.9, 1.5))])
            # crown must stay clear of the pad: pad spans x <= pad_x + pad_size / 2 in truck coordinates
            if local[0] - r < pad_x + pad_size / 2 + 0.4:
                local[0] = pad_x + pad_size / 2 + 0.4 + r
            tree = place("tree", (tL, tW, tH), tparts, t_crown, at=truck.to_world(local), yaw=0.0, ignore=(truck.id,))
        else:
            tree = place("tree", (tL, tW, tH), tparts, t_crown)
        hL, hW, hH, hparts, h_roof = _hut(rng)
        hut = place("hut", (hL, hW, hH), hparts, h_roof)
        sL, sW, sH, sparts, s_rgb = _tent(rng)
        tent = place("tent", (sL, sW, sH), sparts, s_rgb)
        w_rgb = _jitter((0.9, 0.9, 0.9), rng, 0.04)
        wL, wW, wH, wparts = _car(rng, w_rgb)
        white = place("white_car", (wL, wW, wH), wparts, w_rgb)
        ok = all(o is not None for o in (car, tree, hut, tent, white))
        if not ok:
            continue
        # truck-relative objects were placed ignoring the truck: check they do not intersect it
        if car is not None and _overlap(truck, car, 0.8):
            continue
        if tree is not None and _trunk_hits(truck, tree):
            continue
        pad_xy = truck.to_world(np.array([pad_x, 0.0]))
        return TargetLayout(seed, difficulty, c, float(half), objs, pad_xy, float(truck.height + 0.06),
                            float(truck.yaw), float(pad_size), marker_dict, int(marker_id), float(marker_size))
    raise RuntimeError(f"could not place the target scene (seed {seed}, {difficulty}); keep-out too large?")


def _overlap(a: SceneObject, b: SceneObject, clearance: float) -> bool:
    """True if the footprints come closer than `clearance` (sampled corners/edges, conservative enough here)."""
    for p, q in ((a, b), (b, a)):
        hl, hw = q.length / 2, q.width / 2
        g = np.linspace(-1, 1, 9)
        edge = np.concatenate([np.column_stack([g * hl, np.full(9, hw)]), np.column_stack([g * hl, np.full(9, -hw)]),
                               np.column_stack([np.full(9, hl), g * hw]), np.column_stack([np.full(9, -hl), g * hw])])
        for pt in q.to_world(edge):
            if p.contains(pt, clearance):
                return True
    return False


def _trunk_hits(truck: SceneObject, tree: SceneObject) -> bool:
    return truck.contains(tree.xy, 0.6)


# ---------------------------------------------------------------------------------------- marker texture

def marker_texture(marker_dict: str, marker_id: int, pad_size: float, marker_size: float, px_per_m: int = 160) -> np.ndarray:
    """RGB texture of the pad: white with the ArUco marker centered (white quiet zone around it)."""
    n = int(round(pad_size * px_per_m))
    m = int(round(marker_size * px_per_m))
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, marker_dict))
    marker = cv2.aruco.generateImageMarker(d, marker_id, m)
    img = np.full((n, n), 235, np.uint8)
    o = (n - m) // 2
    img[o:o + m, o:o + m] = np.where(marker > 127, 235, 20).astype(np.uint8)
    return np.ascontiguousarray(np.repeat(img[..., None], 3, axis=2))


# ---------------------------------------------------------------------------------------- field package

def _fbm(rng: np.random.Generator, shape: tuple[int, int], scales: Iterable[tuple[float, float]]) -> np.ndarray:
    out = np.zeros(shape, np.float32)
    for sigma, amp in scales:
        n = rng.normal(0, 1, shape).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), sigma)
        n /= max(float(n.std()), 1e-6)
        out += amp * n
    return out


def field_ortho(size_m: float = 400.0, mpp: float = 0.5, seed: int = 4242) -> np.ndarray:
    """A plain grass field with mown patches, a few dirt tracks and bushes (texture for the localizer)."""
    rng = np.random.default_rng(seed)
    n = int(round(size_m / mpp))
    base = np.array([92, 118, 58], np.float32)  # grass
    tone = _fbm(rng, (n, n), [(40, 10.0), (12, 7.0), (3, 5.0), (1, 4.0)])
    img = base[None, None, :] + tone[..., None] * np.array([0.9, 1.0, 0.6], np.float32)
    # dirt tracks (light brown, in the map: the change detector must ignore them)
    track = np.zeros((n, n), np.uint8)
    for _ in range(7):
        pts = [rng.uniform(0, n, 2)]
        ang = rng.uniform(0, 2 * math.pi)
        for _ in range(12):
            ang += rng.normal(0, 0.35)
            pts.append(pts[-1] + 30 * np.array([math.cos(ang), math.sin(ang)]))
        cv2.polylines(track, [np.array(pts, np.int32)], False, 255, int(rng.integers(3, 6)), cv2.LINE_AA)
    t = cv2.GaussianBlur(track.astype(np.float32) / 255.0, (0, 0), 1.0)[..., None]
    img = img * (1 - t) + (np.array([150, 128, 95], np.float32) + tone[..., None] * 3) * t
    # dark bushes / patches
    for _ in range(int(n * n / 4000)):
        x, y = rng.integers(0, n, 2)
        cv2.circle(img, (int(x), int(y)), int(rng.integers(2, 6)), (55, 78, 40), -1, cv2.LINE_AA)
    return np.clip(img, 0, 255).astype(np.uint8)


def ensure_field_package(root: Path, size_m: float = 400.0, mpp: float = 0.5) -> Path:
    """Write the plain-field map package once (manifest + landmarks + ortho.png) and return its directory.
    Spawn (A) and the search area center (B) are both the field center: a standalone flight takes off
    inside the search area (the layout keeps objects clear of the spawn point)."""
    d = Path(root) / f"field_pkg_v{FIELD_PKG_VERSION}"
    if (d / "manifest.json").exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    img = field_ortho(size_m, mpp)
    cv2.imwrite(str(d / "ortho.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    h = size_m / 2
    manifest = {
        "version": 1, "package_id": "v4-field", "terrain": "flat (z=0)", "hand_made": True,
        "coordinate_frame": "local ENU (x=east, y=north, z=up) in meters",
        "a": {"enu": [0.0, 0.0]}, "b": {"enu": [0.0, 0.0]},
        "attribution": "procedural grass field (sfhack.world.target_scene)",
        "files": ["manifest.json", "landmarks.json", "ortho.png"],
        "ortho": {"file": "ortho.png", "source": "procedural", "width_px": img.shape[1], "height_px": img.shape[0],
                  "extent_enu": {"min_x": -h, "min_y": -h, "max_x": h, "max_y": h}, "m_per_px": [mpp, mpp]},
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (d / "landmarks.json").write_text("[]", encoding="utf-8")
    return d
