"""Polygons and north-up rasters in the local ENU frame.

Raster convention (shared by ortho.jpg, terrain.npz, dsm.npz): array[row, col], row 0 = NORTH edge,
cell (row, col) centre at x = min_x + (col + 0.5) * res, y = max_y - (row + 0.5) * res.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Grid:
    """A north-up raster frame: origin corner (min_x, max_y), cell size res, shape (rows, cols)."""

    min_x: float
    max_y: float
    res: float
    rows: int
    cols: int

    @classmethod
    def covering(cls, extent: dict[str, float], res: float) -> "Grid":
        cols = int(math.ceil((extent["max_x"] - extent["min_x"]) / res))
        rows = int(math.ceil((extent["max_y"] - extent["min_y"]) / res))
        return cls(extent["min_x"], extent["max_y"], res, rows, cols)

    @property
    def max_x(self) -> float:
        return self.min_x + self.cols * self.res

    @property
    def min_y(self) -> float:
        return self.max_y - self.rows * self.res

    @property
    def shape(self) -> tuple[int, int]:
        return self.rows, self.cols

    def centers(self) -> tuple[np.ndarray, np.ndarray]:
        """(X, Y) cell-centre coordinates, each (rows, cols)."""
        x = self.min_x + (np.arange(self.cols) + 0.5) * self.res
        y = self.max_y - (np.arange(self.rows) + 0.5) * self.res
        return np.meshgrid(x, y)

    def to_px(self, xy: np.ndarray) -> np.ndarray:
        """ENU (..., 2) -> continuous pixel coords (col, row) of the cell centres' frame (centre = integer)."""
        xy = np.asarray(xy, float)
        return np.stack([(xy[..., 0] - self.min_x) / self.res - 0.5, (self.max_y - xy[..., 1]) / self.res - 0.5], -1)

    def meta(self) -> dict[str, float | int]:
        return {"min_x": round(self.min_x, 3), "max_y": round(self.max_y, 3), "max_x": round(self.max_x, 3),
                "min_y": round(self.min_y, 3), "res_m": self.res, "rows": self.rows, "cols": self.cols}


def sample_bilinear(arr: np.ndarray, grid: Grid, xy: np.ndarray) -> np.ndarray:
    """Bilinear lookup of a north-up raster at ENU points (..., 2); clamps at the edges."""
    uv = grid.to_px(xy)
    u = np.clip(uv[..., 0], 0, grid.cols - 1.0)
    v = np.clip(uv[..., 1], 0, grid.rows - 1.0)
    u0 = np.minimum(np.floor(u).astype(int), grid.cols - 2) if grid.cols > 1 else np.zeros_like(u, int)
    v0 = np.minimum(np.floor(v).astype(int), grid.rows - 2) if grid.rows > 1 else np.zeros_like(v, int)
    fu, fv = u - u0, v - v0
    u1, v1 = np.minimum(u0 + 1, grid.cols - 1), np.minimum(v0 + 1, grid.rows - 1)
    return ((1 - fv) * ((1 - fu) * arr[v0, u0] + fu * arr[v0, u1]) + fv * ((1 - fu) * arr[v1, u0] + fu * arr[v1, u1]))


def resample(arr: np.ndarray, src: Grid, dst: Grid) -> np.ndarray:
    """Bilinear resample of a raster from one north-up grid onto another (cv2.remap, edge replicate)."""
    X, Y = dst.centers()
    uv = src.to_px(np.stack([X, Y], -1)).astype(np.float32)
    return cv2.remap(arr.astype(np.float32), uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def polygon_mask(grid: Grid, polys: list[np.ndarray]) -> np.ndarray:
    """uint8 mask (1 inside) of ENU polygons on the grid (cv2.fillPoly, 1/16 px precision)."""
    mask = np.zeros(grid.shape, np.uint8)
    pts = [np.round((grid.to_px(np.asarray(p, float)) + 0.5) * 16).astype(np.int32) for p in polys]
    cv2.fillPoly(mask, pts, 1, lineType=cv2.LINE_8, shift=4)
    return mask


def signed_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def clean_ring(pts: list[list[float]] | np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Drop the closing duplicate, repeated and collinear vertices; return a CCW ring (n, 2)."""
    p = np.asarray(pts, float)
    if len(p) > 1 and np.allclose(p[0], p[-1]):
        p = p[:-1]
    keep = [q for i, q in enumerate(p) if i == 0 or np.hypot(*(q - p[i - 1])) > eps]
    p = np.asarray(keep, float)
    if len(p) > 1 and np.hypot(*(p[0] - p[-1])) <= eps:
        p = p[:-1]
    changed = True
    while changed and len(p) > 3:
        changed = False
        for i in range(len(p)):
            a, b, c = p[i - 1], p[i], p[(i + 1) % len(p)]
            cr = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            if abs(cr) < eps * max(np.hypot(*(c - a)), 1e-9):
                p = np.delete(p, i, axis=0)
                changed = True
                break
    if len(p) >= 3 and signed_area(p) < 0:
        p = p[::-1].copy()
    return p


def _in_tri(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    d1 = (p[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[1] - b[1])
    d2 = (p[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (p[1] - c[1])
    d3 = (p[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[1] - a[1])
    neg = d1 < 0 or d2 < 0 or d3 < 0
    pos = d1 > 0 or d2 > 0 or d3 > 0
    return not (neg and pos)


def triangulate(ring: np.ndarray) -> list[tuple[int, int, int]] | None:
    """Ear clipping of a simple CCW polygon. Returns CCW index triangles, None if the ring is not simple."""
    n = len(ring)
    if n < 3:
        return None
    idx = list(range(n))
    tris: list[tuple[int, int, int]] = []
    while len(idx) > 3:
        m = len(idx)
        for k in range(m):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % m]
            a, b, c = ring[i0], ring[i1], ring[i2]
            if (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) <= 1e-9:
                continue  # reflex or flat corner: not an ear
            if any(_in_tri(ring[j], a, b, c) for j in idx if j not in (i0, i1, i2)
                   and not (np.allclose(ring[j], a) or np.allclose(ring[j], b) or np.allclose(ring[j], c))):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            return None  # no ear: self-intersecting or degenerate footprint
    tris.append((idx[0], idx[1], idx[2]))
    return tris


def prism_mesh(ring: np.ndarray, tris: list[tuple[int, int, int]], z0: float, z1: float) -> tuple[np.ndarray, np.ndarray]:
    """Closed extrusion of a CCW ring from z0 to z1 with outward faces: (vertices (2n, 3), faces (k, 3))."""
    n = len(ring)
    v = np.vstack([np.column_stack([ring, np.full(n, z0)]), np.column_stack([ring, np.full(n, z1)])])
    faces = [(n + a, n + b, n + c) for a, b, c in tris]  # roof, normal up
    faces += [(a, c, b) for a, b, c in tris]  # floor, normal down
    for i in range(n):
        j = (i + 1) % n
        faces += [(i, j, n + j), (i, n + j, n + i)]  # wall, normal to the right of the CCW edge = outward
    return v, np.asarray(faces, np.int32)


def obb(ring: np.ndarray) -> tuple[float, float, float, float, float]:
    """Oriented bounding box: (cx, cy, half_x, half_y, yaw_rad)."""
    (cx, cy), (w, h), ang = cv2.minAreaRect(np.asarray(ring, np.float32))
    return float(cx), float(cy), max(float(w), 0.5) / 2, max(float(h), 0.5) / 2, math.radians(float(ang))
