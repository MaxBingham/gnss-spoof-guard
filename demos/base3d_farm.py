"""Fictional rural-Ukraine world for demos/50_base_3d.py (`--world farm`): procedurally generated, no real place.

About 4 x 4 km of gently rolling farmland around the base (ENU metres, the UMA-16 mast at the origin):
- terrain: seeded smooth hills (~15 m relief) as a MuJoCo height field, flat within 220 m of the base (the pad is
  at z = 0) and towards the map edge; `FarmWorld.ground(x, y)` gives the height for the mission code;
- ground texture (generated PNG, cached under $SFHACK_DATA/base3d, not in git): a slightly rotated patchwork of large
  fields in late-summer colours with crop rows, windbreak tree rows on most field edges, a paved road, dirt tracks
  from the base gate to the road and to the farms, a village with gardens, a pond;
- 3D: the base (storage hall, concrete forecourt, HESCO wall with gate and 4 watchtowers, office, containers, tents,
  vehicles, a camo net), the windbreak trees near the base (one ellipsoid each), 3 farmsteads, the village with a
  domed hall. Every 3D tree / building has its footprint and its sun shadow painted into the texture too.

All decoration is visual only (contype 0) and sits in its own static body "deco", so the broadphase never pairs it
with the drone; the height field is the only collision geometry added here (the pad comes from 50_base_3d.py).

Texture <-> world (checked with a marker render, `python demos/base3d_farm.py --check`): MuJoCo drapes a 2D texture
over a height field north-up, image row 0 = north edge, column 0 = west edge, spanning the whole field; the height
data starts with the SOUTH row.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OBJ = ROOT / "assets" / "objects" / "objects"

HALF = 2000.0               # map half size, m
HF_N = 401                  # height field vertices per side (10 m spacing)
CELL = 2 * HALF / (HF_N - 1)
TEX_N = 4096                # texture pixels per side (~1 m / px)
RES = 2 * HALF / TEX_N
TEX_VERSION = 5             # bump when the painting changes (cache key)
RELIEF = 16.0               # peak-to-peak hill height before the flattening, m
FLAT_R, BLEND_R = 220.0, 260.0
FIELD_ANGLE = math.radians(11.0)
TREES_3D_R = 1800.0         # windbreak trees closer than this get a 3D canopy ...
MAX_TREES_3D = 2400         # ... nearest first, at most this many (viewer frame rate)
SUN_DIR = [0.3, 0.55, -1.0]          # direction the sunlight travels (afternoon sun in the SSW, ~60 deg up)
SUN = np.array(SUN_DIR[:2]) / -SUN_DIR[2]   # shadow offset per metre of height

# base compound (the mast at the origin, the pad in 50_base_3d.py at (18, -10))
WALL = (-95.0, 95.0, -70.0, 85.0)     # HESCO perimeter x0, x1, y0, y1; gate in the south wall at x = 0
GATE_HALF = 7.0
FORECOURT = (-50.0, 50.0, -32.5, 32.5)
HALL = dict(x=0.0, y=50.5, yaw=0.0, L=90.0, W=30.0, eave=7.0, ridge=9.0)
OFFICE = dict(x=-76.0, y=2.0, yaw=math.pi / 2, L=22.0, W=11.0, h=7.0)

ROAD_PTS = [(-2150, -380), (-1500, -440), (-900, -560), (-300, -540), (300, -500), (800, -470), (1150, -530),
            (1450, -560), (1800, -500), (2150, -420)]
VILLAGE = np.array([1400.0, -555.0])
FARMS = [(-980.0, 640.0, FIELD_ANGLE), (880.0, 760.0, FIELD_ANGLE + math.pi / 2), (-1160.0, -1080.0, FIELD_ANGLE)]

OLIVE = [0.33, 0.36, 0.22, 1]
SAND = [0.72, 0.64, 0.47, 1]
TAN = [0.55, 0.47, 0.33, 1]
TRUNK = [0.30, 0.24, 0.18, 1]
TREE_GREENS = [(0.16, 0.27, 0.12), (0.19, 0.31, 0.13), (0.15, 0.30, 0.15), (0.22, 0.35, 0.15), (0.25, 0.39, 0.17)]

# late-summer crops: name, rgb (0-1), stripe period m, stripe amplitude, tramline every m (0 = none), probability
CROPS = [
    ("stubble", (0.78, 0.70, 0.47), 6.0, 0.07, 0.0, 0.27),
    ("wheat", (0.81, 0.69, 0.40), 3.0, 0.03, 24.0, 0.10),
    ("sunflower", (0.47, 0.43, 0.20), 3.0, 0.08, 0.0, 0.20),
    ("ploughed", (0.47, 0.36, 0.26), 2.5, 0.14, 0.0, 0.16),
    ("green", (0.38, 0.46, 0.22), 3.0, 0.10, 0.0, 0.15),
    ("meadow", (0.50, 0.55, 0.30), 0.0, 0.00, 0.0, 0.12),
]


def data_dir() -> Path:
    p = Path(os.environ.get("SFHACK_DATA") or (Path.home() / "sf-hack-data")) / "base3d"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------------------------------- geometry
def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _spline(pts, step=10.0) -> np.ndarray:
    """Catmull-Rom through the control points, sampled every ~`step` m."""
    p = np.asarray(pts, float)
    p = np.vstack([p[0], p, p[-1]])
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1:i + 3]
        t = np.linspace(0, 1, max(2, int(np.linalg.norm(p2 - p1) / step)), endpoint=False)[:, None]
        out.append(0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3))
    out.append(p[-2][None])
    return np.vstack(out)


def _dist_poly(pts, poly) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(pts, float))
    d = np.full(len(pts), np.inf)
    for a, b in zip(poly[:-1], poly[1:]):
        ab = b - a
        t = np.clip((pts - a) @ ab / max(float(ab @ ab), 1e-9), 0.0, 1.0)
        d = np.minimum(d, np.linalg.norm(pts - (a + t[:, None] * ab), axis=1))
    return d


def _arclen(poly) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))])


def _at(poly, s_cum, s):
    """Point and unit direction on a polyline at arclength s."""
    i = int(np.clip(np.searchsorted(s_cum, s) - 1, 0, len(poly) - 2))
    d = poly[i + 1] - poly[i]
    n = float(np.linalg.norm(d))
    return poly[i] + d * ((s - s_cum[i]) / max(n, 1e-9)), d / max(n, 1e-9)


def _rect(cx, cy, L, W, yaw) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    loc = np.array([[-L / 2, -W / 2], [L / 2, -W / 2], [L / 2, W / 2], [-L / 2, W / 2]])
    return loc @ np.array([[c, s], [-s, c]]) + [cx, cy]


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def _quat(R) -> list:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q.tolist()


def _outward(V, F) -> np.ndarray:
    """Wind every triangle of a convex shape centred at the origin counter-clockwise seen from outside."""
    F = np.array(F)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    flip = np.einsum("ij,ij->i", np.cross(b - a, c - a), a + b + c) < 0
    F[flip] = F[flip][:, [0, 2, 1]]
    return F


def _unit_prims() -> dict:
    """Low-poly unit box (half size 1), sphere (radius 1) and cylinder (radius 1, half height 1)."""
    V, F = [], []
    for ax in range(3):
        for s in (-1, 1):
            u, v = [i for i in range(3) if i != ax]
            k = len(V)
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = [0.0, 0.0, 0.0]
                p[ax], p[u], p[v] = s, a, b
                V.append(p)
            F += [[k, k + 1, k + 2], [k, k + 2, k + 3]]
    box = (np.array(V), _outward(np.array(V), F))
    ns, nt = 12, 7
    V = [[0, 0, -1.0]] + [[math.cos(2 * math.pi * j / ns) * math.cos(la), math.sin(2 * math.pi * j / ns) * math.cos(la),
                          math.sin(la)] for i in range(1, nt) for la in [math.pi * i / nt - math.pi / 2]
                         for j in range(ns)] + [[0, 0, 1.0]]
    F = [[0, 1 + j, 1 + (j + 1) % ns] for j in range(ns)]
    for i in range(nt - 2):
        for j in range(ns):
            a, b = 1 + i * ns + j, 1 + i * ns + (j + 1) % ns
            F += [[a, b, b + ns], [a, b + ns, a + ns]]
    top = len(V) - 1
    F += [[top, top - ns + j, top - ns + (j + 1) % ns] for j in range(ns)]
    sphere = (np.array(V), _outward(np.array(V), F))
    ns = 10
    ring = [[math.cos(2 * math.pi * j / ns), math.sin(2 * math.pi * j / ns)] for j in range(ns)]
    V = [[x, y, z] for z in (-1.0, 1.0) for x, y in ring]                    # side (smooth)
    V += [[x, y, z] for z in (-1.0, 1.0) for x, y in ring] + [[0, 0, -1.0], [0, 0, 1.0]]   # caps (flat)
    F = []
    for j in range(ns):
        a, b = j, (j + 1) % ns
        F += [[a, b, b + ns], [a, b + ns, a + ns]]
        F += [[4 * ns, 2 * ns + a, 2 * ns + b], [4 * ns + 1, 3 * ns + a, 3 * ns + b]]
    cyl = (np.array(V), _outward(np.array(V), F))
    return {mujoco.mjtGeom.mjGEOM_BOX: box, mujoco.mjtGeom.mjGEOM_ELLIPSOID: sphere,
            mujoco.mjtGeom.mjGEOM_CYLINDER: cyl}


_PRIMS = _unit_prims()


# ----------------------------------------------------------------------------------------------- world
class FarmWorld:
    def __init__(self, seed: int = 11):
        self.seed = seed
        self._terrain()
        self._layout()

    # ---------------------------------------------------------------------------------------- terrain
    def _terrain(self) -> None:
        rng = np.random.default_rng(self.seed + 1)
        n = HF_N
        h = cv2.resize(rng.normal(size=(7, 7)).astype(np.float32), (n, n), interpolation=cv2.INTER_CUBIC)
        h += 0.35 * cv2.resize(rng.normal(size=(19, 19)).astype(np.float32), (n, n), interpolation=cv2.INTER_CUBIC)
        h = (h - h.mean()) / np.ptp(h) * RELIEF
        xs = np.linspace(-HALF, HALF, n)
        X, Y = np.meshgrid(xs, xs)          # row 0 = SOUTH (MuJoCo height field order)
        w = _smoothstep((np.hypot(X, Y) - FLAT_R) / BLEND_R)
        w *= _smoothstep((HALF - np.maximum(abs(X), abs(Y))) / 300.0)
        self.Z = (h * w).astype(np.float64)

    def ground(self, x, y):
        """Terrain height at (x, y), m (bilinear on the height-field grid; the base area is exactly 0)."""
        scalar = np.isscalar(x) and np.isscalar(y)
        fx = np.clip((np.asarray(x, float) + HALF) / CELL, 0, HF_N - 1.000001)
        fy = np.clip((np.asarray(y, float) + HALF) / CELL, 0, HF_N - 1.000001)
        j, i = np.floor(fx).astype(int), np.floor(fy).astype(int)
        tx, ty = fx - j, fy - i
        Z = self.Z
        z = (Z[i, j] * (1 - tx) * (1 - ty) + Z[i, j + 1] * tx * (1 - ty)
             + Z[i + 1, j] * (1 - tx) * ty + Z[i + 1, j + 1] * tx * ty)
        return float(z) if scalar else z

    # ----------------------------------------------------------------------------------------- layout
    def _layout(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.road = _spline(ROAD_PTS)
        self.road_s = _arclen(self.road)
        gate = (0.0, WALL[2] - 2)
        self.tracks = [
            _spline([gate, (6, -200), (-4, -380), (0, -528)]),
            _spline([gate, (-120, -95), (-150, 60), (-200, 200), (-560, 430), FARMS[0][:2]]),
            _spline([(-200, 200), (120, 300), (520, 560), FARMS[1][:2]]),
            _spline([(-1070, -540), (-1110, -800), FARMS[2][:2]]),
        ]
        self.buildings: list[dict] = []     # gable / flat buildings: footprint + shadow in the texture, 3D in build()
        self.trees: list[tuple] = []        # (x, y, r, rz, trunk, rgb, windbreak?)
        self.plots: list[tuple] = []        # (polygon, rgb, stripe period) gardens / yards painted on the ground
        self.yards: list[np.ndarray] = []
        self.extras: list[tuple] = []       # (kind, params) silos, dome, ...
        self.keepout: list[tuple] = [(0.0, 7.0, 175.0)]   # (x, y, r): no windbreak trees here
        self._village(rng)
        for fx, fy, yaw in FARMS:
            self._farm(rng, fx, fy, yaw)
        self._fields(rng)

    def _house(self, rng, x, y, yaw, L=None, W=None, wall=None, roof=None):
        L = L or rng.uniform(8.5, 11.5)
        W = W or rng.uniform(6.5, 8.0)
        wall = wall or [(0.93, 0.91, 0.85), (0.94, 0.87, 0.64), (0.72, 0.81, 0.88), (0.80, 0.87, 0.74),
                        (0.90, 0.83, 0.74)][rng.integers(5)]
        roof = roof or [(0.58, 0.22, 0.16), (0.22, 0.45, 0.32), (0.22, 0.33, 0.58), (0.47, 0.49, 0.52),
                        (0.45, 0.30, 0.19), (0.62, 0.30, 0.18)][rng.integers(6)]
        self.buildings.append(dict(x=x, y=y, yaw=yaw, L=L, W=W, eave=rng.uniform(2.7, 3.2),
                                   ridge=rng.uniform(5.6, 6.6), wall=wall, roof=roof, overhang=0.5))

    def _village(self, rng) -> None:
        road, s_cum = self.road, self.road_s
        sv = float(s_cum[np.argmin(np.linalg.norm(road - VILLAGE, axis=1))])
        self.village_s = sv
        self.village_xy = tuple(_at(road, s_cum, sv)[0])
        p, d = _at(road, s_cum, sv - 150)
        nrm = np.array([-d[1], d[0]])
        self.pond = (*(p - nrm * 115), 78.0, 46.0, math.atan2(d[1], d[0]) + 0.3)
        self.keepout.append((self.pond[0], self.pond[1], 95.0))
        p, d = _at(road, s_cum, sv + 40)
        nrm = np.array([-d[1], d[0]])
        self.street = _spline([p, p + nrm * 140 + d * 15, p + nrm * 330 + d * 5])
        church = _at(road, s_cum, sv - 25)
        cyaw = math.atan2(church[1][1], church[1][0])
        cxy = church[0] + np.array([-church[1][1], church[1][0]]) * 34
        self.buildings.append(dict(x=cxy[0], y=cxy[1], yaw=cyaw, L=20.0, W=11.0, eave=6.0, ridge=9.5,
                                   wall=(0.95, 0.94, 0.90), roof=(0.20, 0.45, 0.36), overhang=0.4))
        self.extras.append(("dome", cxy[0], cxy[1], cyaw))
        self.plots.append((_rect(*cxy, 46, 50, cyaw), (0.42, 0.50, 0.28), 0.0))
        for k in range(6):
            a = 2 * math.pi * k / 6 + 0.4
            self.trees.append((cxy[0] + 21 * math.cos(a), cxy[1] + 21 * math.sin(a), 3.2, 3.5, 1.5,
                               (0.20, 0.33, 0.15), False))

        def lot(p, d, side, offset, depth, step):
            n = np.array([-d[1], d[0]]) * side
            hxy = p + n * offset
            yaw = math.atan2(d[1], d[0])
            self._house(rng, *hxy, yaw)
            plot_c = p + n * (offset - 6 + depth / 2)
            col = [(0.40, 0.48, 0.24), (0.45, 0.42, 0.26), (0.36, 0.44, 0.22), (0.50, 0.52, 0.30)][rng.integers(4)]
            self.plots.append((_rect(*plot_c, step - 2.5, depth, yaw), col, rng.choice([0.0, 1.6, 2.2])))
            for _ in range(rng.integers(1, 4)):
                t = p + n * (offset + rng.uniform(12, depth - 12)) + d * rng.uniform(-step / 3, step / 3)
                self.trees.append((t[0], t[1], rng.uniform(2.0, 3.0), rng.uniform(2.0, 2.6), 1.2,
                                   TREE_GREENS[rng.integers(2, 5)], False))
            self.keepout.append((*plot_c, depth / 2 + 25))

        s = sv - 300
        while s < sv + 320:
            step = rng.uniform(26, 33)
            p, d = _at(road, s_cum, s)
            for side in (1, -1):
                near_church = side == 1 and abs(s - (sv - 25)) < 35
                near_street = side == 1 and abs(s - (sv + 40)) < 20
                near_pond = side == -1 and abs(s - (sv - 150)) < 75
                if near_church or near_street or near_pond or rng.random() < 0.1:
                    continue
                lot(p, d, side, 17.0, 70.0, step)
            s += step
        st, st_cum = self.street, _arclen(self.street)
        s = 40.0
        while s < st_cum[-1] - 15:
            step = rng.uniform(28, 34)
            p, d = _at(st, st_cum, s)
            for side in (1, -1):
                if rng.random() < 0.85:
                    lot(p, d, side, 14.0, 55.0, step)
            s += step

    def _farm(self, rng, fx, fy, yaw) -> None:
        R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])

        def to_w(u, v):
            return np.array([fx, fy]) + R @ [u, v]

        self.yards.append(_rect(fx, fy, 200, 130, yaw))
        self.keepout.append((fx, fy, 150.0))
        nb = int(rng.integers(2, 4))
        for k in range(nb):
            L, W = rng.uniform(58, 78), rng.uniform(14, 18)
            c = to_w(-15 + rng.uniform(-6, 6), -40 + 30 * k)
            wall = [(0.80, 0.79, 0.74), (0.62, 0.36, 0.27), (0.70, 0.69, 0.66)][rng.integers(3)]
            roof = [(0.62, 0.62, 0.60), (0.50, 0.33, 0.22), (0.56, 0.58, 0.60)][rng.integers(3)]
            self.buildings.append(dict(x=c[0], y=c[1], yaw=yaw, L=L, W=W, eave=rng.uniform(4.2, 5.0),
                                       ridge=rng.uniform(7.5, 8.6), wall=wall, roof=roof, overhang=0.6))
        for k in range(int(rng.integers(1, 3)) + 1):
            c = to_w(62 + 9 * k, -40)
            self.extras.append(("silo", c[0], c[1], 3.6, rng.uniform(12, 15)))
        h = to_w(70, 38)
        self._house(rng, *h, yaw)
        self.plots.append((_rect(*to_w(70, 50), 34, 34, yaw), (0.40, 0.47, 0.24), 1.8))
        for k in range(12):
            t = to_w(-95 + 17 * k, 72 + rng.uniform(-2, 2))
            self.trees.append((t[0], t[1], rng.uniform(3.5, 4.8), rng.uniform(3.5, 4.5), 2.0,
                               (0.18, 0.31, 0.14), False))
        for k in range(4):
            t = to_w(50 + rng.uniform(-6, 30), 55 + rng.uniform(-10, 8))
            self.trees.append((t[0], t[1], rng.uniform(2.2, 3.2), 2.4, 1.2, (0.24, 0.38, 0.17), False))

    def _fields(self, rng) -> None:
        """Brick patchwork of fields in a frame rotated by FIELD_ANGLE; windbreak trees on most edges."""
        ca, sa = math.cos(FIELD_ANGLE), math.sin(FIELD_ANGLE)

        def xy(u, v):
            return np.array([u * ca - v * sa, u * sa + v * ca])

        probs = np.array([c[5] for c in CROPS])
        self.fields = []    # (polygon xy, crop index, stripes along v?, colour jitter)
        edges = []
        ext = 2900.0
        v = -ext
        while v < ext:
            v1 = v + rng.uniform(230, 380)
            u = -ext - rng.uniform(0, 500)
            while u < ext:
                u1 = u + rng.uniform(380, 650)
                parts = [(u, u1)] if rng.random() > 0.25 else [(u, (u + u1) / 2), ((u + u1) / 2, u1)]
                for a, b in parts:
                    poly = np.array([xy(a + 2.5, v + 2.5), xy(b - 2.5, v + 2.5), xy(b - 2.5, v1 - 2.5),
                                     xy(a + 2.5, v1 - 2.5)])
                    self.fields.append((poly, int(rng.choice(len(CROPS), p=probs / probs.sum())),
                                        bool(rng.random() < 0.75),
                                        rng.uniform(0.90, 1.08) * rng.uniform(0.98, 1.02, 3)))
                if rng.random() < 0.62:
                    edges.append((xy(u, v), xy(u1, v)))
                if rng.random() < 0.42:
                    edges.append((xy(u, v), xy(u, v1)))
                u = u1
            v = v1
        # windbreak trees: poplars (tall, narrow) and broad trees (acacia / oak / maple), ~11 m apart
        pts, props, edge_of = [], [], []
        for e, (a, b) in enumerate(edges):
            L = float(np.linalg.norm(b - a))
            d = (b - a) / L
            nrm = np.array([-d[1], d[0]])
            poplar_row = rng.random() < 0.3
            for t in np.arange(5.5, L, 11.0):
                if rng.random() < 0.07:
                    continue
                edge_of.append(e)
                pts.append(a + d * (t + rng.uniform(-2, 2)) + nrm * rng.uniform(-1.2, 1.2))
                if poplar_row or rng.random() < 0.1:
                    props.append((rng.uniform(2.4, 3.0), rng.uniform(6.5, 8.5), 1.0))
                else:
                    props.append((rng.uniform(4.0, 5.6), rng.uniform(3.4, 5.0), 2.0))
        pts = np.array(pts)
        keep = (abs(pts[:, 0]) < HALF - 5) & (abs(pts[:, 1]) < HALF - 5)
        keep &= _dist_poly(pts, self.road) > 14
        keep &= _dist_poly(pts, self.street) > 12
        for t in self.tracks:
            keep &= _dist_poly(pts, t) > 9
        for x, y, r in self.keepout:
            keep &= np.hypot(pts[:, 0] - x, pts[:, 1] - y) > r
        for p, (r, rz, trunk) in zip(pts[keep], np.array(props)[keep]):
            self.trees.append((p[0], p[1], r, rz, trunk, TREE_GREENS[rng.integers(0, 4)], True))
        # undergrowth belts in the texture: join neighbouring kept trees of the same edge
        self.belts = []
        edge_of = np.array(edge_of)
        for e in np.unique(edge_of[keep]):
            run = pts[keep & (edge_of == e)]
            cut = np.nonzero(np.linalg.norm(np.diff(run, axis=0), axis=1) > 25)[0] + 1
            self.belts += [r for r in np.split(run, cut) if len(r) > 1]

    # ---------------------------------------------------------------------------------------- texture
    def texture(self) -> np.ndarray:
        """RGB uint8, TEX_N x TEX_N, north up; cached under $SFHACK_DATA/base3d."""
        path = data_dir() / f"farm_s{self.seed}_v{TEX_VERSION}_{TEX_N}.png"
        if path.exists():
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is not None and img.shape[:2] == (TEX_N, TEX_N):
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self._paint()
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return img

    @staticmethod
    def _px(pts) -> np.ndarray:
        """World xy -> cv2 pixel coordinates with 3 fractional bits (shift=3)."""
        p = np.atleast_2d(np.asarray(pts, float))
        return np.round(np.column_stack([(p[:, 0] + HALF) / RES - 0.5, (HALF - p[:, 1]) / RES - 0.5]) * 8).astype(
            np.int32)

    def _paint(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed + 2)
        n = TEX_N

        def noise(cells):
            return cv2.resize(rng.random((cells, cells)).astype(np.float32), (n, n), interpolation=cv2.INTER_CUBIC)

        low, mid, fine = noise(14), noise(110), noise(900)
        grain = rng.random((n, n), dtype=np.float32)
        var = ((1 + 0.16 * (low - 0.5)) * (1 + 0.10 * (mid - 0.5)) * (1 + 0.08 * (fine - 0.5))
               * (1 + 0.06 * (grain - 0.5)))
        del low, fine

        # fields: id image, then per-pixel crop colour with rows in the field's own direction
        fid = np.zeros((n, n), np.uint16)
        for k, (poly, *_rest) in enumerate(self.fields):
            cv2.fillPoly(fid, [self._px(poly)], k + 1, lineType=cv2.LINE_8, shift=3)
        nf = len(self.fields) + 1
        col = np.zeros((nf, 3), np.float32)
        col[0] = (0.47, 0.50, 0.30)                      # grass verges between fields
        period = np.zeros(nf, np.float32)
        amp = np.zeros(nf, np.float32)
        tram = np.zeros(nf, np.float32)
        along_v = np.zeros(nf, bool)
        for k, (_p, crop, av, jit) in enumerate(self.fields, start=1):
            _name, rgb, per, a, tr, _pr = CROPS[crop]
            col[k], period[k], amp[k], tram[k], along_v[k] = np.array(rgb) * jit, per, a, tr, av
        ca, sa = math.cos(FIELD_ANGLE), math.sin(FIELD_ANGLE)
        xs = (np.arange(n, dtype=np.float32) + 0.5) * RES - HALF
        img = np.empty((n, n, 3), np.float32)
        for r0 in range(0, n, 512):
            ys = HALF - (np.arange(r0, r0 + 512, dtype=np.float32) + 0.5) * RES
            X, Y = np.meshgrid(xs, ys)
            U, V = X * ca + Y * sa, -X * sa + Y * ca
            f = fid[r0:r0 + 512]
            coord = np.where(along_v[f], V, U)
            per = np.maximum(period[f], 1e-3)
            stripe = 1 + amp[f] * np.sin(2 * np.pi * coord / per)
            tr = tram[f]
            stripe *= np.where((tr > 0) & (np.mod(coord, np.maximum(tr, 1)) < 0.9), 0.86, 1.0)
            img[r0:r0 + 512] = col[f] * (stripe * var[r0:r0 + 512])[..., None]
        del fid

        def blend(mask, rgb, v=None, alpha=1.0):
            idx = mask > 0
            a = (mask[idx].astype(np.float32) / 255 * alpha)[:, None]
            c = np.asarray(rgb, np.float32)
            if c.ndim == 3:
                c = c[idx]
            c = c * (var[idx][:, None] if v is None else v[idx][:, None])
            img[idx] = img[idx] * (1 - a) + c * a

        def shade(mask, k):
            idx = mask > 0
            img[idx] *= (1 - (1 - k) * mask[idx].astype(np.float32) / 255)[:, None]

        def lines(poly, width_m):
            m = np.zeros((n, n), np.uint8)
            cv2.polylines(m, [self._px(poly)], False, 255, max(1, round(width_m / RES)), cv2.LINE_AA, shift=3)
            return m

        def polys(ps):
            m = np.zeros((n, n), np.uint8)
            cv2.fillPoly(m, [self._px(p) for p in ps], 255, cv2.LINE_AA, shift=3)
            return m

        sh = 1 / RES * 8            # metres -> shifted pixels
        ang = math.degrees(math.atan2(-SUN[1], SUN[0]))

        # pond (before the gardens: its reed belt runs under nothing else)
        px, py, a, b, th = self.pond
        c = self._px([(px, py)])[0]
        m = np.zeros((n, n), np.uint8)
        cv2.ellipse(m, tuple(c), (round((a + 7) * sh), round((b + 7) * sh)), -math.degrees(th), 0, 360, 255, -1,
                    cv2.LINE_AA, 3)
        blend(m, (0.30, 0.40, 0.20))
        m[:] = 0
        cv2.ellipse(m, tuple(c), (round(a * sh), round(b * sh)), -math.degrees(th), 0, 360, 255, -1, cv2.LINE_AA, 3)
        blend(m, (0.17, 0.27, 0.27), v=1 + 0.08 * (mid - 0.5))

        # gardens, churchyard, farm yards
        plot_layer = np.zeros((n, n, 3), np.float32)
        m = np.zeros((n, n), np.uint8)
        for poly, rgb, stripe in self.plots:
            pm = np.zeros_like(m)
            cv2.fillPoly(pm, [self._px(poly)], 255, cv2.LINE_AA, shift=3)
            idx = pm > 0
            cc = np.asarray(rgb, np.float32) * rng.uniform(0.92, 1.08)
            if stripe:
                ys, xs_ = np.nonzero(idx)
                e = poly[1] - poly[0]
                e /= np.linalg.norm(e)
                wx = (xs_ + 0.5) * RES - HALF
                wy = HALF - (ys + 0.5) * RES
                s = 1 + 0.18 * np.sign(np.sin(2 * np.pi * (wx * e[0] + wy * e[1]) / stripe))
                plot_layer[ys, xs_] = cc * s[:, None].astype(np.float32)
            else:
                plot_layer[idx] = cc
            m = np.maximum(m, pm)
        blend(m, plot_layer)
        del plot_layer
        blend(polys(self.yards), (0.58, 0.53, 0.44))

        # base: cleared ring outside the wall, packed earth inside, the concrete forecourt
        x0, x1, y0, y1 = WALL
        blend(polys([_rect((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0 + 50, y1 - y0 + 50, 0)]), (0.56, 0.54, 0.36))
        blend(polys([_rect((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, 0)]), (0.56, 0.51, 0.41))
        fx0, fx1, fy0, fy1 = FORECOURT
        blend(polys([_rect((fx0 + fx1) / 2, (fy0 + fy1) / 2, fx1 - fx0, fy1 - fy0, 0)]), (0.68, 0.68, 0.65),
              v=np.ones_like(mid))

        # dirt tracks, village street, the paved road
        for t in self.tracks:
            blend(lines(t, 5.0), (0.60, 0.51, 0.38))
        blend(lines(self.street, 8.0), (0.55, 0.53, 0.47))
        blend(lines(self.road, 11.0), (0.52, 0.50, 0.44))
        blend(lines(self.road, 7.0), (0.25, 0.25, 0.26), v=1 + 0.06 * (mid - 0.5))

        # windbreak undergrowth, shadows of everything 3D, tree canopies and roofs on top
        m = np.zeros((n, n), np.uint8)
        for run in self.belts:
            cv2.polylines(m, [self._px(run)], False, 255, round(6.5 / RES), cv2.LINE_AA, shift=3)
        blend(m, (0.22, 0.30, 0.14), v=0.8 + 0.4 * mid)
        m[:] = 0
        for b in self._shadow_casters():
            fp, h = b
            hull = cv2.convexHull(np.vstack([fp, fp + SUN * h]).astype(np.float32))[:, 0]
            cv2.fillPoly(m, [self._px(hull)], 255, cv2.LINE_AA, shift=3)
        for x, y, r, rz, trunk, _rgb, _wb in self.trees:
            c = self._px([np.array([x, y]) + SUN * (trunk + rz)])[0]
            cv2.ellipse(m, tuple(c), (round(math.hypot(r, 0.6 * rz) * sh), round(r * sh)), ang, 0, 360, 255, -1,
                        cv2.LINE_AA, 3)
        shade(cv2.GaussianBlur(m, (5, 5), 0), 0.55)
        canopy = np.zeros((n, n, 3), np.float32)
        m[:] = 0
        for x, y, r, rz, trunk, rgb, _wb in self.trees:
            c = tuple(self._px([(x, y)])[0])
            cv2.circle(m, c, round(r * sh), 255, -1, cv2.LINE_AA, 3)
            cv2.circle(canopy, c, round(r * sh), tuple(float(v) * 0.6 for v in rgb), -1, cv2.LINE_AA, 3)
        blend(m, canopy, v=0.8 + 0.4 * mid)
        del canopy
        roofs = np.zeros((n, n, 3), np.float32)
        m[:] = 0
        for b in self.buildings:
            fp = _rect(b["x"], b["y"], b["L"], b["W"], b["yaw"])
            cv2.fillPoly(m, [self._px(fp)], 255, cv2.LINE_AA, shift=3)
            cv2.fillPoly(roofs, [self._px(fp)], tuple(float(v) for v in b["roof"]), cv2.LINE_AA, shift=3)
        blend(m, roofs, v=np.ones((n, n), np.float32))
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)

    def _shadow_casters(self):
        for b in self.buildings + self._base_buildings():
            yield _rect(b["x"], b["y"], b["L"], b["W"], b["yaw"]), b.get("ridge", b.get("h", 5.0))
        for kind, *p in self.extras:
            if kind == "silo":
                x, y, r, h = p
                yield np.array([[x + r * math.cos(a), y + r * math.sin(a)] for a in np.linspace(0, 6.28, 12)]), h

    @staticmethod
    def _base_buildings():
        return [dict(HALL), dict(OFFICE, ridge=OFFICE["h"])]

    # ------------------------------------------------------------------------------------------ build
    def build(self, spec: mujoco.MjSpec) -> mujoco.MjsBody:
        """Height field + texture, and every 3D object in a static body "deco". Returns that body."""
        img = self.texture()
        z = self.Z
        zmin, zmax = float(z.min()), float(z.max())
        span = max(zmax - zmin, 0.1)
        tex = spec.add_texture(name="farm", type=mujoco.mjtTexture.mjTEXTURE_2D, width=TEX_N, height=TEX_N,
                               nchannel=3)
        tex.data = np.ascontiguousarray(img).tobytes()
        mat = spec.add_material(name="farm", texrepeat=[1, 1], texuniform=False, reflectance=0.0, specular=0.0,
                                shininess=0.0)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "farm"
        hf = spec.add_hfield(name="terrain", size=[HALF, HALF, span, 1.0], nrow=HF_N, ncol=HF_N)
        hf.userdata = ((z - zmin) / span).ravel().tolist()
        spec.worldbody.add_geom(name="terrain", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="terrain",
                                pos=[0, 0, zmin], material="farm")
        # no infinite ground plane (it would show through the hollows); flat surroundings beyond the map
        spec.delete(spec.geom("floor"))
        self.body = spec.worldbody.add_body(name="deco")
        self.spec = spec
        self.matte = spec.add_material(name="matte", specular=0.05, shininess=0.0, reflectance=0.0).name
        self._batches: dict = {}
        self._mesh_names: dict[str, str] = {}
        for x, y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            sx, sy = (2000, HALF + 4000) if x else (HALF, 2000)
            self._box([x * (HALF + 2000), y * (HALF + 2000), -0.6], [sx, sy, 0.5], [0.58, 0.56, 0.36, 1])
        self._base()
        for b in self.buildings:
            self._gable(b)
        for kind, *p in self.extras:
            if kind == "silo":
                x, y, r, h = p
                g = self.ground(x, y)
                self._geom(mujoco.mjtGeom.mjGEOM_CYLINDER, [x, y, g + h / 2 - 0.5], [r, h / 2 + 0.5, 0],
                           [0.74, 0.75, 0.77, 1])
                self._geom(mujoco.mjtGeom.mjGEOM_ELLIPSOID, [x, y, g + h], [r, r, 1.6], [0.66, 0.67, 0.70, 1])
            elif kind == "dome":
                x, y, yaw = p
                b = next(b for b in self.buildings if b["x"] == x and b["y"] == y)
                top = self.ground(x, y) + b["ridge"]
                self._geom(mujoco.mjtGeom.mjGEOM_CYLINDER, [x, y, top + 1.0], [1.9, 2.0, 0], [0.95, 0.94, 0.90, 1])
                self._geom(mujoco.mjtGeom.mjGEOM_ELLIPSOID, [x, y, top + 3.4], [2.2, 2.2, 2.6], [0.86, 0.70, 0.25, 1])
                self._box([x, y, top + 6.7], [0.08, 0.08, 0.9], [0.86, 0.70, 0.25, 1])
                self._box([x, y, top + 7.0], [0.08, 0.5, 0.08], [0.86, 0.70, 0.25, 1], yaw)
        self._trees()
        self._flush()
        return self.body

    # 3D helpers. MuJoCo draws every geom with its own GL state changes (~35 us each on an Intel Arc laptop GPU:
    # 3200 geoms = 8 fps), so opaque boxes / ellipsoids / cylinders / building meshes are merged into one mesh per
    # colour (_flush); only textured, transparent and asset-mesh geoms stay separate geoms.
    def _geom(self, gtype, pos, size, rgba, quat=None, **kw):
        quat = quat or [1, 0, 0, 0]
        if gtype in _PRIMS and not kw and len(rgba) == 4 and rgba[3] >= 1.0:
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, np.asarray(quat, float))
            V, F = _PRIMS[gtype]
            scale = [size[0], size[0], size[1]] if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER else size
            self._add_tris(rgba, (V * scale) @ R.reshape(3, 3).T + pos, F)
            return None
        kw.setdefault("material", self.matte)
        return self.body.add_geom(type=gtype, pos=list(pos), size=list(size), rgba=list(rgba),
                                  quat=quat, contype=0, conaffinity=0, **kw)

    def _add_tris(self, rgba, V, F) -> None:
        key = tuple(round(float(c), 3) for c in rgba[:3])
        vs, fs, n = self._batches.setdefault(key, ([], [], [0]))
        vs.append(np.asarray(V, float))
        fs.append(np.asarray(F, int) + n[0])
        n[0] += len(V)

    def _flush(self) -> None:
        for k, (rgb, (vs, fs, _n)) in enumerate(self._batches.items()):
            name = f"deco{k}"
            self.spec.add_mesh(name=name, uservert=np.vstack(vs).ravel().tolist(),
                               userface=np.vstack(fs).ravel().tolist())
            self.body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, rgba=[*rgb, 1], material=self.matte,
                               contype=0, conaffinity=0)
        self._batches = {}

    def _box(self, pos, half, rgba, yaw=0.0):
        return self._geom(mujoco.mjtGeom.mjGEOM_BOX, pos, half, rgba, [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])

    def _mesh(self, key: str, rel: str) -> str:
        if key not in self._mesh_names:
            self.spec.add_mesh(name=key, file=str(OBJ / rel))
            self._mesh_names[key] = key
        return key

    def _place(self, key, rel, x, y, yaw, rgba):
        self._geom(mujoco.mjtGeom.mjGEOM_MESH, [x, y, self.ground(x, y) + 0.03], [0, 0, 0], rgba,
                   [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)], meshname=self._mesh(key, rel))

    def _gable(self, b: dict) -> None:
        """Walls with gables (one pentagonal prism mesh) + two roof slabs, on the ground at (x, y)."""
        x, y, yaw, L, W, he, hr = b["x"], b["y"], b["yaw"], b["L"], b["W"], b["eave"], b["ridge"]
        corners = _rect(x, y, L, W, yaw)
        zg = self.ground(x, y)
        down = zg - float(min(self.ground(cx, cy) for cx, cy in corners)) + 0.4
        prof = [(-W / 2, -down), (W / 2, -down), (W / 2, he), (0.0, hr), (-W / 2, he)]
        verts, faces = [], []

        def tri(a, b_, c):
            a, b_, c = (np.asarray(v, float) for v in (a, b_, c))
            nrm = np.cross(b_ - a, c - a)
            cen = (a + b_ + c) / 3 - [0, 0, (hr - down) / 2]
            if nrm @ cen < 0:
                b_, c = c, b_
            k = len(verts)
            verts.extend([a, b_, c])
            faces.append([k, k + 1, k + 2])

        e0 = [(-L / 2, py, pz) for py, pz in prof]
        e1 = [(L / 2, py, pz) for py, pz in prof]
        for i in range(1, 4):
            tri(e0[0], e0[i], e0[i + 1])
            tri(e1[0], e1[i], e1[i + 1])
        for i in range(5):
            j = (i + 1) % 5
            tri(e0[i], e0[j], e1[j])
            tri(e0[i], e1[j], e1[i])
        Rz = _rz(yaw)
        self._add_tris(b["wall"], np.array(verts) @ Rz.T + [x, y, zg], faces)
        s, r, o, t = W / 2, hr - he, b.get("overhang", 0.4), 0.07
        th = math.atan2(r, s)
        ln = math.hypot(s, r)
        for side in (1, -1):
            nrm = np.array([0, side * math.sin(th), math.cos(th)])
            loc = np.array([0, side * (s + o * math.cos(th)) / 2, he + r / 2 - o * math.sin(th) / 2]) + nrm * t
            Rl = Rz @ _rx(-side * th)
            self._geom(mujoco.mjtGeom.mjGEOM_BOX, np.array([x, y, zg]) + Rz @ loc, [L / 2 + o, (ln + o) / 2, t],
                       [*b["roof"], 1], _quat(Rl))

    def _trees(self) -> None:
        trees = np.array([(x, y, r, rz, tr, wb) for x, y, r, rz, tr, _c, wb in self.trees])
        dist = np.hypot(trees[:, 0], trees[:, 1])
        wb = trees[:, 5] > 0
        order = np.argsort(dist + np.where(wb, 0, -1e9))      # village / farm trees always, then nearest windbreaks
        order = [i for i in order if not wb[i] or dist[i] < TREES_3D_R]
        n_extra = int((~wb).sum())
        self.n_trees_3d = min(len(order), MAX_TREES_3D + n_extra)
        for i in order[:self.n_trees_3d]:
            x, y, r, rz, trunk, rgb, _ = self.trees[i]
            g = self.ground(x, y)
            self._geom(mujoco.mjtGeom.mjGEOM_ELLIPSOID, [x, y, g + trunk + rz], [r, r, rz], [*rgb, 1])
            self._geom(mujoco.mjtGeom.mjGEOM_CYLINDER, [x, y, g + trunk / 2 + 0.2], [0.18 + 0.04 * r, trunk / 2 + 0.6],
                       TRUNK)

    # ------------------------------------------------------------------------------------------- base
    def _base(self) -> None:
        x0, x1, y0, y1 = WALL
        # concrete forecourt with slab joints (the mast stands in its middle)
        fx0, fx1, fy0, fy1 = FORECOURT
        tex = self.spec.add_texture(name="slabs", type=mujoco.mjtTexture.mjTEXTURE_2D,
                                    builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, mark=mujoco.mjtMark.mjMARK_EDGE,
                                    rgb1=[0.71, 0.71, 0.69], rgb2=[0.68, 0.68, 0.66], markrgb=[0.60, 0.60, 0.58],
                                    width=256, height=256)
        m = self.spec.add_material(name="slabs", texuniform=True, texrepeat=[1 / 12, 1 / 12], specular=0.05,
                                   shininess=0.0, reflectance=0.0)
        m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
        self._geom(mujoco.mjtGeom.mjGEOM_BOX, [(fx0 + fx1) / 2, (fy0 + fy1) / 2, 0.0],
                   [(fx1 - fx0) / 2, (fy1 - fy0) / 2, 0.04], [1, 1, 1, 1], material="slabs")

        # storage hall: pale walls, pitched pale metal roof, 4 roll-up doors facing the forecourt
        self._gable(dict(HALL, wall=(0.80, 0.80, 0.77), roof=(0.80, 0.82, 0.84), overhang=0.4))
        hx, hy, hw = HALL["x"], HALL["y"], HALL["W"]
        for dx in (-33, -11, 11, 33):
            fy = hy - hw / 2 - 0.06
            self._box([hx + dx, fy, 2.85], [3.3, 0.06, 2.85], [0.47, 0.52, 0.57, 1])
            for k in range(1, 9):
                self._box([hx + dx, fy - 0.05, 0.62 * k], [3.3, 0.02, 0.04], [0.38, 0.42, 0.46, 1])
            self._box([hx + dx, fy - 0.03, 5.85], [3.6, 0.06, 0.15], [0.62, 0.55, 0.20, 1])
        self._box([hx - 44.95, hy, 1.1], [0.06, 0.55, 1.1], [0.35, 0.37, 0.40, 1])   # side door

        # office: two storeys, flat roof, windows facing the forecourt
        ox, oy, oL, oW, oh = OFFICE["x"], OFFICE["y"], OFFICE["L"], OFFICE["W"], OFFICE["h"]
        self._box([ox, oy, oh / 2], [oW / 2, oL / 2, oh / 2], [0.84, 0.79, 0.66, 1])
        self._box([ox, oy, oh + 0.3], [oW / 2 + 0.15, oL / 2 + 0.15, 0.3], [0.40, 0.40, 0.40, 1])
        for fl in (1.6, 4.8):
            for k in range(6):
                self._box([ox + oW / 2 + 0.03, oy - 8.5 + 3.4 * k, fl], [0.05, 0.75, 0.7], [0.20, 0.25, 0.30, 1])
        self._box([ox + oW / 2 + 0.03, oy + 0.2, 1.2], [0.06, 0.7, 1.2], [0.30, 0.25, 0.20, 1])

        # containers (20 ft), two stacked
        cols = [OLIVE, [0.55, 0.25, 0.18, 1], [0.25, 0.35, 0.50, 1], SAND, [0.42, 0.44, 0.40, 1]]
        for k, (cx, cy, cz) in enumerate([(72, 30, 0), (72, 22, 0), (72, 22, 2.59), (72, 14, 0), (84, 30, 0),
                                          (84, 22, 0), (84, 2, 0), (72, -6, 0), (84, -6, 0), (84, -6, 2.59)]):
            self._box([cx, cy, cz + 1.295], [3.03, 1.22, 1.295], cols[k % len(cols)])

        # tents (south-west), vehicles under a camo net (south-east), a few more by the hall
        for i, (x, y) in enumerate([(-82, -58), (-70, -58), (-58, -58), (-82, -44), (-70, -44)]):
            self._place("tent", "tent/ridge_green_a/ridge_green_a.stl", x, y, math.pi / 2,
                        OLIVE if i % 2 else [0.38, 0.40, 0.25, 1])
        self._place("dome", "tent/dome_c/dome_c.stl", -58, -44, 0.5, [0.40, 0.42, 0.28, 1])
        for i, (x, y) in enumerate([(58, -52), (66, -52), (74, -52), (82, -52)]):
            self._place("truck" if i < 2 else "pickup", "truck/box_truck_a/box_truck_a.stl" if i < 2 else
                        "pickup/pickup_a/pickup_a.stl", x, y, math.pi / 2, OLIVE)
        self._place("suv", "suv/suv_a/suv_a.stl", -30, -48, 0.2, OLIVE)
        self._place("pickup", "pickup/pickup_a/pickup_a.stl", 58, 70, 0.0, OLIVE)
        self._place("truck", "truck/box_truck_a/box_truck_a.stl", -58, 72, 0.0, OLIVE)
        cx, cy = 70, -52
        for dx, dy in [(-15, -6), (-15, 6), (15, -6), (15, 6)]:
            self._geom(mujoco.mjtGeom.mjGEOM_CYLINDER, [cx + dx, cy + dy, 2.0], [0.07, 2.0, 0], [0.3, 0.25, 0.2, 1])
        self._box([cx, cy, 4.05], [16, 7, 0.05], [0.30, 0.34, 0.20, 0.9])

        # HESCO perimeter, gate in the south wall (posts + barrier), 4 corner watchtowers
        h = 1.1
        for side in range(4):
            lo, hi = (x0, x1) if side in (0, 3) else (y0, y1)
            for k in np.arange(lo + 1, hi, 2.0):
                if side == 3 and abs(k) < GATE_HALF:
                    continue
                p = {0: [k, y1], 1: [x1, k], 2: [x0, k], 3: [k, y0]}[side]
                self._box([p[0], p[1], h], [1.0, 1.0, h], SAND if int(k // 2) % 2 else TAN)
        for gx in (-GATE_HALF, GATE_HALF):
            self._box([gx, y0, 1.6], [0.25, 0.25, 1.6], [0.25, 0.25, 0.25, 1])
        for k in range(7):
            self._box([-GATE_HALF + 0.3 + 2 * k + 0.5, y0, 1.1], [0.5, 0.08, 0.08],
                      [0.85, 0.15, 0.12, 1] if k % 2 else [0.95, 0.95, 0.95, 1])
        for cx_, cy_ in [(x0 + 4, y0 + 4), (x0 + 4, y1 - 4), (x1 - 4, y0 + 4), (x1 - 4, y1 - 4)]:
            for dx, dy in [(-1.3, -1.3), (-1.3, 1.3), (1.3, -1.3), (1.3, 1.3)]:
                self._geom(mujoco.mjtGeom.mjGEOM_CYLINDER, [cx_ + dx, cy_ + dy, 3.5], [0.14, 3.5, 0],
                           [0.30, 0.25, 0.20, 1])
            self._box([cx_, cy_, 7.2], [1.9, 1.9, 0.2], [0.35, 0.30, 0.22, 1])
            for dx, dy, hx_, hy_ in [(0, 1.8, 1.9, 0.1), (0, -1.8, 1.9, 0.1), (1.8, 0, 0.1, 1.9), (-1.8, 0, 0.1, 1.9)]:
                self._box([cx_ + dx, cy_ + dy, 7.9], [hx_, hy_, 0.5], SAND)
            self._box([cx_, cy_, 9.4], [2.2, 2.2, 0.12], OLIVE)
            for dx, dy in [(-1.7, -1.7), (-1.7, 1.7), (1.7, -1.7), (1.7, 1.7)]:
                self._geom(mujoco.mjtGeom.mjGEOM_CYLINDER, [cx_ + dx, cy_ + dy, 8.8], [0.06, 0.6, 0],
                           [0.30, 0.25, 0.20, 1])


# ------------------------------------------------------------------------------------- texture check
def check_texture(out: Path) -> float:
    """Marker test: paint a red square into the texture at a known world point near the NE corner and a blue one
    near the SW corner, put 3D posts on the same points, render top-down; return the pixel offset between them."""
    spec = mujoco.MjSpec()
    spec.visual.headlight.ambient = [0.7] * 3
    spec.visual.global_.offwidth = spec.visual.global_.offheight = 800
    spec.visual.map.znear, spec.visual.map.zfar = 0.001, 3
    spec.stat.extent = 2000
    w = FarmWorld()
    img = np.full((TEX_N, TEX_N, 3), 200, np.uint8)
    marks = [(1500.0, 1200.0, (255, 0, 0)), (-1700.0, -900.0, (0, 0, 255))]
    for x, y, rgb in marks:
        c = FarmWorld._px([(x, y)])[0]
        cv2.circle(img, tuple(c), round(40 / RES * 8), rgb, -1, cv2.LINE_8, 3)
    tex = spec.add_texture(name="t", type=mujoco.mjtTexture.mjTEXTURE_2D, width=TEX_N, height=TEX_N, nchannel=3)
    tex.data = img.tobytes()
    mat = spec.add_material(name="t", texrepeat=[1, 1], texuniform=False)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "t"
    z = w.Z
    zmin, span = float(z.min()), float(np.ptp(z))
    hf = spec.add_hfield(name="h", size=[HALF, HALF, span, 1.0], nrow=HF_N, ncol=HF_N)
    hf.userdata = ((z - zmin) / span).ravel().tolist()
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="h", material="t", pos=[0, 0, zmin])
    for x, y, _rgb in marks:   # thin black posts standing on the terrain at the same points
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[x, y, w.ground(x, y) + 1],
                                size=[12, 1, 0], rgba=[0, 0, 0, 1])
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 800, 800)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0, 0, 0]
    cam.distance, cam.elevation, cam.azimuth = 4900, -90, 90
    r.update_scene(d, cam)
    rgb = r.render()
    cv2.imwrite(str(out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    worst = 0.0
    for (x, y, col) in marks:
        colour = np.all(np.abs(rgb.astype(int) - col) < 60, axis=2)
        black = np.all(rgb < 25, axis=2)
        # look only around the expected screen point (north up, centred)
        f = 800 / 2 / math.tan(math.radians(m.vis.global_.fovy) / 2) / (4900 - w.ground(x, y))
        u, v = 400 + x * f, 400 - y * f
        win = np.zeros_like(colour)
        win[int(v) - 60:int(v) + 60, int(u) - 60:int(u) + 60] = True
        cy, cx = np.argwhere(colour & win).mean(axis=0)
        by, bx = np.argwhere(black & win).mean(axis=0)
        worst = max(worst, math.hypot(cx - bx, cy - by))
        print(f"marker ({x:.0f}, {y:.0f}): texture at px ({cx:.1f}, {cy:.1f}), 3D post at ({bx:.1f}, {by:.1f})")
    return worst


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        out = Path(sys.argv[-1]) if sys.argv[-1].endswith(".png") else data_dir() / "texture_check.png"
        off = check_texture(out)
        print(f"texture vs 3D offset {off:.1f} px  ({out})")
        raise SystemExit(0 if off < 3 else 1)
    import time

    t0 = time.time()
    w = FarmWorld()
    print(f"layout {time.time() - t0:.1f}s: {len(w.fields)} fields, {len(w.trees)} trees, "
          f"{len(w.buildings)} buildings, relief {np.ptp(w.Z):.1f} m")
    t0 = time.time()
    w.texture()
    print(f"texture {time.time() - t0:.1f}s")
