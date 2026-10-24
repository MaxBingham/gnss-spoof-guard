"""Recon test scene (HARNESS, simulated truth): a copy of a mission with warm targets + distractors along a bearing.

    python demos/43_recon_testscene.py                      # ukraine-1 -> ukraine-1-rmtest, true bearing 64 deg
    python demos/43_recon_testscene.py --az 60 --dist 1700 --seed 3

Stand-in until recon-world's target placement lands (same file: world/target_scene.json, objects with "temp_K",
rendered by the thermal camera as uniform-temperature meshes). Objects (temperatures relative to the night air):
  targets      2 cars (4.6 x 1.9 x 1.5 m, +9 K: engines running), 1 truck (7.5 x 2.5 x 3 m, +7 K), 1 person (+7 K)
  distractors  warm rock (1.8 x 1.6 m, +6 K), barn roof (14 x 9 m, +4 K), cooling car (+1.5 K, engine off long ago)
The onboard package is copied unchanged (the drone has no map of any of this).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
from pathlib import Path

import numpy as np

from sfhack.config import data_dir
from sfhack.world.onboard_map import OnboardMap

ROOT = Path(__file__).resolve().parents[1]


def box_stl(path: Path, lx: float, ly: float, lz: float) -> None:
    """Binary STL of a box centred on x / y, standing on z = 0 (length along x)."""
    x, y = lx / 2, ly / 2
    v = np.array([[-x, -y, 0], [x, -y, 0], [x, y, 0], [-x, y, 0], [-x, -y, lz], [x, -y, lz], [x, y, lz], [-x, y, lz]], float)
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7),
             (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    with open(path, "wb") as f:
        f.write(b"recon box".ljust(80, b" "))
        f.write(struct.pack("<I", len(faces)))
        for a, b, c in faces:
            n = np.cross(v[b] - v[a], v[c] - v[a])
            n = n / np.linalg.norm(n)
            f.write(struct.pack("<12fH", *n, *v[a], *v[b], *v[c], 0))


OBJECTS = [  # name, kind, (l, w, h), dT [K], along [m], across [m] (relative to the cluster point on the bearing)
    ("car1", "target", (4.6, 1.9, 1.5), 9.0, 0.0, 0.0),
    ("car2", "target", (4.6, 1.9, 1.5), 9.0, 14.0, 5.0),
    ("truck", "target", (7.5, 2.5, 3.0), 7.0, -12.0, -6.0),
    ("person", "target", (0.5, 0.4, 1.7), 7.0, 6.0, 9.0),
    ("rock", "distractor", (1.8, 1.6, 0.8), 6.0, -300.0, 60.0),
    ("barn", "distractor", (14.0, 9.0, 5.0), 4.0, -150.0, -120.0),
    ("cooling_car", "distractor", (4.6, 1.9, 1.5), 1.5, -80.0, 140.0),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="ukraine-1")
    ap.add_argument("--dst", default="ukraine-1-rmtest")
    ap.add_argument("--az", type=float, default=64.0, help="true bearing of the target cluster from A [deg from north]")
    ap.add_argument("--dist", type=float, default=1700.0, help="distance of the cluster from A [m]")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    src, dst = data_dir() / "missions" / a.src, data_dir() / "missions" / a.dst
    if not dst.exists():
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("drone"))
    wdir = dst / "world"
    om = OnboardMap(dst / "onboard")
    air = float(json.loads((wdir / "thermal_build.json").read_text(encoding="utf-8"))["air_K"])
    A = om.a_enu
    az = math.radians(a.az)
    u = np.array([math.sin(az), math.cos(az)])  # along the bearing (ENU)
    n = np.array([u[1], -u[0]])  # to the right
    rng = np.random.default_rng(a.seed)
    (wdir / "meshes").mkdir(exist_ok=True)
    objs = []
    for name, kind, (lx, ly, lz), dT, along, across in OBJECTS:
        xy = A + u * (a.dist + along) + n * across
        for _ in range(40):  # not on a roof / in a tree: shift until the onboard DSM is bare ground
            if om.surface_z(xy) - om.ground_z(xy) < 1.0:
                break
            xy = xy + rng.normal(0, 8.0, 2)
        box_stl(wdir / "meshes" / f"{name}.stl", lx, ly, lz)
        objs.append({"name": name, "kind": kind, "mesh": f"meshes/{name}.stl", "pos": [round(float(xy[0]), 2),
                     round(float(xy[1]), 2)], "yaw_deg": round(float(math.degrees(math.atan2(u[1], u[0])) + rng.normal(0, 20)), 1),
                     "scale": 1.0, "temp_K": round(air + dT, 2), "dims_m": [lx, ly, lz],
                     "rgba": [0.35, 0.38, 0.3, 1] if kind == "target" else [0.5, 0.45, 0.4, 1]})
    ts = json.loads((wdir / "target_scene.json").read_text(encoding="utf-8"))
    ts.update({"status": "placed (demos/43_recon_testscene.py)", "objects": objs, "true_bearing_deg": a.az,
               "cluster_enu": [round(float(v), 2) for v in A + u * a.dist], "cluster_dist_m": a.dist})
    (wdir / "target_scene.json").write_text(json.dumps(ts, indent=2), encoding="utf-8")
    print(f"{dst}: {len(objs)} objects, cluster at {ts['cluster_enu']} ({a.dist:.0f} m @ {a.az:.0f} deg from A)")
    for o in objs:
        print(f"  {o['name']:12s} {o['kind']:10s} {o['pos']}  {o['temp_K'] - air:+.1f} K")


if __name__ == "__main__":
    main()
