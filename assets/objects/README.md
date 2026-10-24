# Scene objects around point B (MuJoCo, Skydio X2 sim)

A library of 3D objects (binary STL) at real-world scale, ready to place around point B.
All meshes share one frame: **meters, Z up, +X = forward (length axis), origin at the bottom
center of the footprint**. An object placed at `pos = (x, y, ground_z)` stands on the ground.
Rotate it by yaw about +Z only.

## Layout

```
catalog.json                    all objects and variants (one entry per variant = its meta.json)
objects/<key>/<variant>/
    <variant>.stl               whole object (one mesh)
    parts/*.stl                 the same object split into parts (same frame), one rgba each
    meta.json                   size, scale, triangles, collision box, landing area, source, license
    preview.png                 top view and 3/4 view (matplotlib)
sources/                        original downloaded GLB files (input of the pipeline; not in the team repo)
tools/normalize.py              re-runnable pipeline: sources/procedural -> objects/ + catalog
tools/verify_mujoco.py          loads everything into one MuJoCo scene, checks z, renders lineup
lineup.png, lineup_top.png      verification renders (oblique; top view from 30 m, fovy 60)
lineup_scene.xml                MJCF of the verification scene (absolute mesh paths; regenerated, not in the team repo)
ATTRIBUTION.md                  source, author and license for every variant
```

Keys: `car` (white or brown car), `truck` (brown truck that carries the landing station),
`tent`, `hut`, `tree`. Extra clutter: `suv`, `pickup`.

## Objects

| key | variant | L x W x H (m) | tris | license | source | size check |
|---|---|---|---|---|---|---|
| car | sedan_a | 4.70 x 2.01 x 1.31 | 2954 | CC0 | Quaternius (poly.pizza/m/Cz6yDaUcM9) | ok |
| car | hatch_b | 4.30 x 2.13 x 1.49 | 3124 | CC0 | Quaternius (poly.pizza/m/unqqkULtRU) | W 2.13 incl. mirrors |
| truck | box_truck_a | 8.04 x 2.50 x 3.30 | 628 | CC0 | procedural | ok |
| truck | box_truck_b | 7.24 x 2.45 x 3.10 | 468 | CC0 | procedural | ok |
| truck | cargo_truck_c | 8.55 x 2.40 x 3.14 | 4299 | CC-BY 3.0 | J-Toastie (poly.pizza/m/Fy3WI3uXNQ) | ok (roof plate added) |
| tent | ridge_green_a | 3.85 x 2.26 x 1.60 | 784 | CC0 | Quaternius (poly.pizza/m/5Q7qIrfDxA) | L includes guy lines |
| tent | ridge_orange_b | 3.20 x 2.33 x 2.05 | 228 | CC0 | Kenney Nature Kit | W 2.33 |
| tent | dome_c | 3.30 x 2.80 x 1.62 | 1962 | CC0 | procedural | ok |
| hut | cabin_a | 4.80 x 4.00 x 3.20 | 88 | CC0 | procedural | ok |
| hut | shed_d | 3.90 x 3.50 x 2.95 | 88 | CC0 | procedural | ok |
| hut | cabin_shed_b | 4.24 x 3.00 x 3.40 | 2745 | CC0 | CreativeTrio (poly.pizza/m/HTx7PZt6Zm) | ok |
| hut | log_cabin_e | 3.67 x 2.91 x 3.50 | 2640 | CC-BY 3.0 | Jarlan Perez (poly.pizza/m/et0OmFeZVkb) | W 2.91 |
| hut | hut_c | 4.91 x 3.64 x 3.20 | 632 | CC0 | Quaternius (poly.pizza/m/4MJWbyd6vw) | open-sided shelter |
| tree | oak_a | 5.43 x 4.70 x 9.00 | 196 | CC0 | Kenney Nature Kit | ok |
| tree | deciduous_b | 6.36 x 5.72 x 10.00 | 402 | CC0 | Kenney Nature Kit | ok |
| tree | deciduous_c | 4.86 x 4.21 x 11.00 | 114 | CC0 | Kenney Nature Kit | ok |
| tree | conifer_d | 3.08 x 3.03 x 12.00 | 134 | CC0 | Kenney Nature Kit | conifer (narrow crown) |
| tree | conifer_e | 4.43 x 3.83 x 10.00 | 206 | CC0 | Kenney Nature Kit | conifer |
| suv | suv_a | 4.70 x 2.36 x 1.71 | 3294 | CC0 | Quaternius (poly.pizza/m/xsMtZhBkxL) | W incl. mirrors |
| pickup | pickup_a | 5.30 x 2.37 x 1.89 | 6432 | CC0 | Quaternius (poly.pizza/m/qn4grQgHm8) | W incl. mirrors |

Low-poly models are wider than real vehicles at a uniform scale, because the width includes the mirrors.
No object needed decimation: the largest has 6.4k triangles, below the 20k limit.

## Using catalog.json in MuJoCo

```python
import json, mujoco, numpy as np
from pathlib import Path
CATALOG = Path("assets/objects/catalog.json")    # from the repo root; adjust to where it lives
ROOT = CATALOG.parent                            # every path in the catalog is relative to it
cat = json.loads(CATALOG.read_text())

def place(spec, meta, name, pos, yaw, body_rgba=None):
    """Add one object: visual part meshes + one collision box."""
    q = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    body = spec.worldbody.add_body(name=name, pos=pos, quat=q)
    for p in meta["files"]["parts"]:
        mesh = spec.add_mesh(name=f"{name}_{p['name']}", file=str(ROOT / p["path"]))
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL   # meshes may be open shells
        rgba = body_rgba if (body_rgba and p["name"] == "body") else p["rgba"]
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name, rgba=rgba,
                      contype=0, conaffinity=0)                    # visual only
    cb = meta["collision_box"]
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=cb["center"], size=cb["half_size"],
                  group=3, rgba=[0, 0, 0, 0])                      # collision proxy
    return body

spec = mujoco.MjSpec()
car = cat["objects"]["car"][0]
place(spec, car, "white_car", [10, 3, 0], 0.3)
place(spec, cat["objects"]["car"][1], "brown_car", [14, -2, 0], 1.2,
      body_rgba=next(p for p in car["files"]["parts"] if p["name"] == "body")["rgba_options"]["brown"])
truck = cat["objects"]["truck"][0]
tb = place(spec, truck, "truck", [0, 0, 0], 0.0)
la = truck["landing_area"]                       # truck-local frame: center xyz + size [x, y]
tb.add_body(name="landing_station", pos=la["center"])   # put the station / pad here
# recommended pad color: la["rgba"], neutral gray 0.55 0.55 0.55 1; the placer adds the ArUco marker
model = spec.compile()
```

Notes:
- Car body colors: the `body` part has `rgba` (white 0.92 0.92 0.92 1) and `rgba_options`
  {white, brown (0.40 0.26 0.15 1)}. Truck parts are brown (0.42 0.28 0.16 1). Each part has a
  recommended rgba, so use the parts rather than the whole STL when color matters.
- `collision_box` is an axis-aligned box in the object frame (center + half sizes). Trees also
  have `collision_boxes_detailed` (trunk, crown) and `crown_diameter_m`.
- `landing_area` (trucks only) is the flat top surface: `center` is in the object frame (z = roof
  height) and `size` is [length along X, width along Y]. Every truck has at least 2.4 x 2.4 m.
  On `cargo_truck_c` a flat `roof_plate` part was added (`added_roof_plate: true`). The pad and
  `roof_plate` are neutral gray (`landing_area.rgba` = 0.55 0.55 0.55 1), so the truck cannot be
  found by color from altitude; the ArUco marker is added later by the placer. Transform
  the center with the truck's pose: `world = truck_pos + Rz(yaw) @ center`.
- MuJoCo keeps mesh geoms at their file coordinates. The verification checks that every
  object's lowest vertex is at z = 0.000 after compile.
- Note the CC-BY credits in ATTRIBUTION.md if `cargo_truck_c` or `log_cabin_e` is used.

## Re-running

```
uv venv <venv> --python 3.12      # keep the venv outside OneDrive / the repo
uv pip install --python <venv>/Scripts/python.exe trimesh numpy scipy mujoco pillow matplotlib fast-simplification networkx
<venv>/Scripts/python.exe -I tools/normalize.py          # all, or --only truck/box_truck_a (needs sources/)
<venv>/Scripts/python.exe -I tools/verify_mujoco.py
```

The `ASSETS` list in `tools/normalize.py` holds every variant: the source file or the
procedural generator, the forward axis, the scale target, the material-to-part mapping, the
colors and the license. Change it and re-run to add or retune objects.
