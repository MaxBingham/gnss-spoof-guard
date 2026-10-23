"""3D point clouds: USGS 3DEP LiDAR from the public Entwine Point Tile (EPT) octrees, or a synthetic cloud.

EPT (https://entwine.io/en/latest/entwine-point-tile.html): ept.json gives a cube in EPSG:3857, every
octree node D-X-Y-Z is a LAZ file under ept-data/, ept-hierarchy/*.json lists point counts per node
(-1 = the subtree lives in its own hierarchy file). Depth d holds points at roughly cube / 2^d / span
spacing, so reading depths 0..d gives a cloud of about that resolution. No PDAL needed: laspy[lazrs]
decodes the nodes, httpx fetches them in parallel (cached like every other download).
"""

from __future__ import annotations

import io
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from ..geo import BBox, GeoOrigin
from .sources import SRC_LIDAR, FetchError, Fetcher

EPT_INDEX = "https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/boundaries/resources.geojson"
R_MERC = 6_378_137.0
NOISE_CLASSES = (7, 18)  # ASPRS low / high noise
GROUND, BUILDING = 2, 6


def merc(lat: np.ndarray | float, lon: np.ndarray | float) -> tuple[Any, Any]:
    return R_MERC * np.radians(lon), R_MERC * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def inv_merc(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.degrees(2 * np.arctan(np.exp(y / R_MERC)) - np.pi / 2), np.degrees(x / R_MERC)


def _inside(poly: list[list[float]], x: float, y: float) -> bool:
    c = False
    for i in range(len(poly)):
        x1, y1 = poly[i][:2]
        x2, y2 = poly[i - 1][:2]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            c = not c
    return c


def find_resources(f: Fetcher, bbox: BBox) -> list[dict[str, Any]]:
    """EPT collections whose boundary contains the bbox centre, newest (by year in the name) first."""
    index = json.loads(f.get(EPT_INDEX, timeout=60.0))
    lat, lon = (bbox.south + bbox.north) / 2, (bbox.west + bbox.east) / 2
    hits = []
    for feat in index["features"]:
        g = feat["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        if any(_inside(p[0], lon, lat) for p in polys):
            p = feat["properties"]
            years = [int(y) for y in re.findall(r"(?<!\d)(20\d\d|19\d\d)(?!\d)", p["name"])]
            hits.append({"name": p["name"], "url": p["url"], "points": p.get("count"), "year": max(years) if years else 0})
    return sorted(hits, key=lambda h: (-h["year"], h["name"]))


def read_ept(f: Fetcher, ept_url: str, bbox: BBox, target_spacing_m: float = 0.5, max_nodes: int = 400,
             deadline: float | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Points inside bbox: (lat, lon, z) float64 (N, 3) and ASPRS class (N,) uint8, noise removed."""
    import laspy

    base = ept_url.rsplit("/", 1)[0] + "/"
    ept = json.loads(f.get(ept_url, timeout=30.0))
    if ept.get("srs", {}).get("horizontal") != "3857":
        raise FetchError(f"EPT srs {ept.get('srs', {}).get('horizontal')} (only EPSG:3857 supported)")
    bx0, by0, _, bx1, _, _ = ept["bounds"]
    span = int(ept.get("span", 128))
    cube = bx1 - bx0
    qx0, qy0 = merc(bbox.south, bbox.west)
    qx1, qy1 = merc(bbox.north, bbox.east)
    k = math.cos(math.radians((bbox.south + bbox.north) / 2))  # mercator m -> ground m
    depth = max(0, math.ceil(math.log2(cube * k / (span * target_spacing_m))))

    def hits(key: str) -> bool:
        d, x, y, _ = (int(v) for v in key.split("-"))
        size = cube / 2 ** d
        nx0, ny0 = bx0 + x * size, by0 + y * size
        return nx0 < qx1 and nx0 + size > qx0 and ny0 < qy1 and ny0 + size > qy0

    nodes: dict[str, int] = {}
    todo = ["0-0-0-0"]
    while todo:
        h = json.loads(f.get(base + f"ept-hierarchy/{todo.pop()}.json", timeout=30.0))
        for key, n in h.items():
            if int(key.split("-")[0]) > depth or not hits(key):
                continue
            if n == -1:
                todo.append(key)
            elif n > 0:
                nodes[key] = n
    while len(nodes) > max_nodes and depth > 0:  # too large an area for this resolution: coarser
        depth -= 1
        nodes = {k_: n for k_, n in nodes.items() if int(k_.split("-")[0]) <= depth}

    def node(key: str) -> tuple[np.ndarray, np.ndarray]:
        if deadline is not None and time.time() > deadline:
            raise FetchError("LiDAR time box exceeded")
        las = laspy.read(io.BytesIO(f.get(base + f"ept-data/{key}.laz", timeout=60.0)))
        x, y = np.asarray(las.x), np.asarray(las.y)
        m = (x >= qx0) & (x <= qx1) & (y >= qy0) & (y <= qy1)
        cls = np.asarray(las.classification, np.uint8)[m]
        return np.column_stack([x[m], y[m], np.asarray(las.z)[m]]), cls

    with ThreadPoolExecutor(12) as ex:
        parts = list(ex.map(node, sorted(nodes)))
    xyz = np.vstack([p[0] for p in parts]) if parts else np.zeros((0, 3))
    cls = np.concatenate([p[1] for p in parts]) if parts else np.zeros(0, np.uint8)
    keep = ~np.isin(cls, NOISE_CLASSES)
    xyz, cls = xyz[keep], cls[keep]
    lat, lon = inv_merc(xyz[:, 0], xyz[:, 1])
    info = {**SRC_LIDAR, "collection": ept_url, "depth": depth, "nodes": len(nodes),
            "spacing_m": round(cube * k / span / 2 ** depth, 2), "points_raw": int(len(xyz))}
    return np.column_stack([lat, lon, xyz[:, 2]]), cls, info


def to_enu(latlonz: np.ndarray, origin: GeoOrigin, z0: float) -> np.ndarray:
    x = (latlonz[:, 1] - origin.lon) * origin.m_per_deg_lon
    y = (latlonz[:, 0] - origin.lat) * origin.m_per_deg_lat
    return np.column_stack([x, y, latlonz[:, 2] - z0])


def voxel_downsample(xyz: np.ndarray, voxel: float, cls: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Centroid per occupied voxel (class = class of the first point in the voxel)."""
    if len(xyz) == 0:
        return xyz, cls
    q = np.floor((xyz - xyz.min(axis=0)) / voxel).astype(np.int64)
    key = (q[:, 0] << 42) ^ (q[:, 1] << 21) ^ q[:, 2]
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    cnt = np.bincount(inv).astype(float)
    out = np.column_stack([np.bincount(inv, xyz[:, i]) / cnt for i in range(3)])
    return out, (cls[first] if cls is not None else None)


def write_ply(path: Any, xyz: np.ndarray, cls: np.ndarray, comments: list[str]) -> None:
    """Binary little-endian PLY: float32 x, y, z (ENU m) + uchar classification (ASPRS)."""
    rec = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("classification", "u1")])
    rec["x"], rec["y"], rec["z"], rec["classification"] = xyz[:, 0], xyz[:, 1], xyz[:, 2], cls
    head = ["ply", "format binary_little_endian 1.0", *[f"comment {c}" for c in comments],
            f"element vertex {len(xyz)}", "property float x", "property float y", "property float z",
            "property uchar classification", "end_header"]
    with open(path, "wb") as fh:
        fh.write(("\n".join(head) + "\n").encode("ascii"))
        fh.write(rec.tobytes())


def read_ply(path: Any) -> tuple[np.ndarray, np.ndarray]:
    """Reader for the files write_ply produces: (xyz float32 (N, 3), classification uint8 (N,))."""
    with open(path, "rb") as fh:
        n = 0
        while True:
            line = fh.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            if line == "end_header":
                break
        rec = np.frombuffer(fh.read(), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("classification", "u1")], count=n)
    return np.column_stack([rec["x"], rec["y"], rec["z"]]), rec["classification"].copy()


def synthetic_cloud(dtm: np.ndarray, grid: Any, buildings: list[dict[str, Any]], trees: list[dict[str, Any]],
                    spacing: float, rng: np.random.Generator, noise_std: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Airborne-LiDAR-like cloud sampled from terrain + extruded footprints + tree crowns (+ range noise).

    Ground everywhere except under roofs, roofs on a `spacing` lattice, walls on a spacing x spacing
    lattice, crowns as the upper half of a sphere. Classes: 2 ground, 6 building, 5 high vegetation."""
    from .geometry import polygon_mask

    step = max(1, int(round(spacing / grid.res)))
    X, Y = grid.centers()
    roof = polygon_mask(grid, [np.asarray(b["footprint_enu"]) for b in buildings]) if buildings else np.zeros(grid.shape, np.uint8)
    sel = (roof[::step, ::step] == 0)
    pts = [np.column_stack([X[::step, ::step][sel], Y[::step, ::step][sel], dtm[::step, ::step][sel]])]
    cls = [np.full(int(sel.sum()), GROUND, np.uint8)]
    for b in buildings:
        ring = np.asarray(b["footprint_enu"], float)
        z0, z1 = float(b["base_z"]), float(b["base_z"]) + float(b["height"])
        lo, hi = ring.min(axis=0), ring.max(axis=0)
        gx, gy = np.meshgrid(np.arange(lo[0], hi[0], spacing), np.arange(lo[1], hi[1], spacing))
        cand = np.column_stack([gx.ravel(), gy.ravel()])
        if len(cand):
            from matplotlib.path import Path as MplPath

            inside = MplPath(ring).contains_points(cand)
            pts.append(np.column_stack([cand[inside], np.full(int(inside.sum()), z1)]))
            cls.append(np.full(int(inside.sum()), BUILDING, np.uint8))
        for i in range(len(ring)):
            a, c = ring[i], ring[(i + 1) % len(ring)]
            L = float(np.hypot(*(c - a)))
            ts = np.arange(0.0, L, spacing) / max(L, 1e-9)
            zs = np.arange(z0 + spacing, z1, spacing)
            if len(ts) and len(zs):
                T, Z = np.meshgrid(ts, zs)
                xy = a + T.ravel()[:, None] * (c - a)
                pts.append(np.column_stack([xy, Z.ravel()]))
                cls.append(np.full(len(xy), BUILDING, np.uint8))
    for t in trees:
        r, h = float(t.get("radius", 3.0)), float(t["height"])
        cx, cy = t["centroid_enu"]
        zc = float(t["base_z"]) + max(h - r, 1.0)
        n = max(8, int(2 * math.pi * r * r / spacing ** 2))
        u, v = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
        th, ph = 2 * math.pi * u, np.arccos(v)  # upper hemisphere (what an airborne scanner sees)
        pts.append(np.column_stack([cx + r * np.sin(ph) * np.cos(th), cy + r * np.sin(ph) * np.sin(th), zc + r * np.cos(ph)]))
        cls.append(np.full(n, 5, np.uint8))
    xyz = np.vstack(pts)
    xyz = xyz + rng.normal(0.0, noise_std, xyz.shape)
    return xyz, np.concatenate(cls)
