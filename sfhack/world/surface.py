"""Regular north-up raster in local ENU: surface models (DSM / terrain) and ortho imagery grids.

`Grid` stores a 2D array with row 0 = north edge, column 0 = west edge, square cells of `mpp` meters,
cell centers at (min_x + (j + 0.5) * mpp, max_y - (i + 0.5) * mpp). Bilinear lookups are vectorized
because the TRN particle filter queries thousands of points per altimeter sample.

No MuJoCo and no truth here: onboard code (perception, planning) may import it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Grid:
    z: np.ndarray  # (H, W) float32
    min_x: float
    max_y: float
    mpp: float  # meters per cell

    @property
    def shape(self) -> tuple[int, int]:
        return self.z.shape  # type: ignore[return-value]

    @property
    def max_x(self) -> float:
        return self.min_x + self.z.shape[1] * self.mpp

    @property
    def min_y(self) -> float:
        return self.max_y - self.z.shape[0] * self.mpp

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(min_x, min_y, max_x, max_y)."""
        return self.min_x, self.min_y, self.max_x, self.max_y

    def xy_to_ij(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Continuous (row, col) of world points, cell centers at integer coordinates."""
        xy = np.asarray(xy, float)
        col = (xy[..., 0] - self.min_x) / self.mpp - 0.5
        row = (self.max_y - xy[..., 1]) / self.mpp - 0.5
        return row, col

    def ij_to_xy(self, row: np.ndarray, col: np.ndarray) -> np.ndarray:
        x = self.min_x + (np.asarray(col, float) + 0.5) * self.mpp
        y = self.max_y - (np.asarray(row, float) + 0.5) * self.mpp
        return np.stack([x, y], axis=-1)

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """(X, Y) arrays of the cell centers, same shape as z."""
        h, w = self.z.shape
        x = self.min_x + (np.arange(w) + 0.5) * self.mpp
        y = self.max_y - (np.arange(h) + 0.5) * self.mpp
        return np.meshgrid(x, y)

    def contains(self, xy: np.ndarray, margin: float = 0.0) -> np.ndarray | bool:
        xy = np.asarray(xy, float)
        ok = ((xy[..., 0] >= self.min_x + margin) & (xy[..., 0] <= self.max_x - margin)
              & (xy[..., 1] >= self.min_y + margin) & (xy[..., 1] <= self.max_y - margin))
        return bool(ok) if ok.ndim == 0 else ok

    def sample(self, xy: np.ndarray) -> np.ndarray:
        """Bilinear interpolation at world points (..., 2); clamped at the edges."""
        row, col = self.xy_to_ij(xy)
        h, w = self.z.shape
        row = np.clip(row, 0.0, h - 1.000001)
        col = np.clip(col, 0.0, w - 1.000001)
        i0 = np.floor(row).astype(np.int64)
        j0 = np.floor(col).astype(np.int64)
        fr, fc = row - i0, col - j0
        z = self.z
        a = z[i0, j0] * (1 - fc) + z[i0, j0 + 1] * fc
        b = z[i0 + 1, j0] * (1 - fc) + z[i0 + 1, j0 + 1] * fc
        return a * (1 - fr) + b * fr

    def crop(self, center_xy: np.ndarray, half_m: float) -> "Grid":
        """Sub-grid of +-half_m around a point (clipped to the grid)."""
        row, col = self.xy_to_ij(np.asarray(center_xy, float))
        r = int(np.ceil(half_m / self.mpp))
        h, w = self.z.shape
        i0, i1 = max(int(row) - r, 0), min(int(row) + r + 1, h)
        j0, j1 = max(int(col) - r, 0), min(int(col) + r + 1, w)
        return Grid(self.z[i0:i1, j0:j1], self.min_x + j0 * self.mpp, self.max_y - i0 * self.mpp, self.mpp)

    def resampled(self, mpp: float) -> "Grid":
        """Same extent at another resolution (area average when coarsening, bilinear when refining)."""
        import cv2

        h, w = self.z.shape
        nw, nh = max(int(round(w * self.mpp / mpp)), 2), max(int(round(h * self.mpp / mpp)), 2)
        interp = cv2.INTER_AREA if mpp > self.mpp else cv2.INTER_LINEAR
        z = cv2.resize(self.z.astype(np.float32), (nw, nh), interpolation=interp)
        return Grid(z, self.min_x, self.max_y, (w * self.mpp) / nw)

    def save(self, path: Path | str, **meta: float) -> None:
        np.savez_compressed(path, z=self.z.astype(np.float32), min_x=self.min_x, max_y=self.max_y, mpp=self.mpp, **meta)

    @classmethod
    def load(cls, path: Path | str) -> "Grid":
        with np.load(path) as f:
            return cls(f["z"].astype(np.float32), float(f["min_x"]), float(f["max_y"]), float(f["mpp"]))


def roughness(grid: Grid, window_m: float) -> np.ndarray:
    """Local standard deviation of the surface inside a square window (TRN information proxy)."""
    import cv2

    k = max(int(round(window_m / grid.mpp)) | 1, 3)
    z = grid.z.astype(np.float64)
    m = cv2.blur(z, (k, k))
    m2 = cv2.blur(z * z, (k, k))
    return np.sqrt(np.maximum(m2 - m * m, 0.0)).astype(np.float32)
