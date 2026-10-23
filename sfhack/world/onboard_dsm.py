"""Onboard surface model (DSM) for TRN, rasterized from the onboard map package's OSM landmarks: flat terrain
(z = 0, see the manifest), buildings / towers as the oriented boxes of their footprints (world/buildings.py, the
same OSM-derived geometry the simulated world is built from) and OSM trees as crown domes.

Map error (the DSM is the drone's map, not the simulated reality): each building's height is off by a fixed
N(0, `height_err_std`) draw (OSM heights are estimates), the raster is `mpp` m, and the simulator's wall/roof
details (roof planes 2 cm above the box, tree trunks) are not modelled. A first version rasterized the true
footprint POLYGONS instead of the boxes: L-shapes and courtyards then disagreed with the simulated world by
whole storeys and the TRN cloud never converged below 6-11 m.

No MuJoCo dependency: onboard code may import it.
"""

from __future__ import annotations

import cv2
import numpy as np

from .buildings import box_corners, building_boxes
from .onboard_map import OnboardMap
from .surface import Grid

MAP_ERROR_SEED = 1234  # the map's own error is fixed (a property of the map, not of the flight)


def build_dsm(om: OnboardMap, mpp: float = 1.0, height_err_std: float = 0.5) -> Grid:
    w = int(np.ceil((om.max_x - om.min_x) / mpp))
    h = int(np.ceil((om.max_y - om.min_y) / mpp))
    z = np.zeros((h, w), np.float32)
    grid = Grid(z, om.min_x, om.max_y, mpp)
    rng = np.random.default_rng(MAP_ERROR_SEED)

    # trees first (buildings overwrite them): crown sphere of radius r on a trunk, as OSM tree nodes give them
    X, Y = grid.cell_centers()
    for lm in om.landmarks:
        if lm["type"] != "tree":
            continue
        (x, y), ht, r = lm["centroid_enu"], float(lm["height"]), float(lm.get("radius", 3.0))
        trunk_h = max(ht - r, 1.0)
        i0, j0 = grid.xy_to_ij(np.array([x - r, y + r]))
        i1, j1 = grid.xy_to_ij(np.array([x + r, y - r]))
        rs = slice(max(int(i0), 0), min(int(i1) + 2, h))
        cs = slice(max(int(j0), 0), min(int(j1) + 2, w))
        rho2 = (X[rs, cs] - x) ** 2 + (Y[rs, cs] - y) ** 2
        dome = np.where(rho2 < r * r, trunk_h + np.sqrt(np.maximum(r * r - rho2, 0.0)), 0.0)
        z[rs, cs] = np.maximum(z[rs, cs], dome)

    boxes = sorted(building_boxes(om.landmarks), key=lambda b: b["height"])
    for box in boxes:  # ascending height: a taller box drawn later wins where boxes overlap
        height = max(box["height"] + float(rng.normal(0.0, height_err_std)), 0.5)
        row, col = grid.xy_to_ij(box_corners(box))
        pts = np.round(np.stack([col, row], axis=-1) * 4).astype(np.int32)  # 2 bits sub-pixel
        cv2.fillPoly(z, [pts], height, lineType=cv2.LINE_8, shift=2)
    return grid
