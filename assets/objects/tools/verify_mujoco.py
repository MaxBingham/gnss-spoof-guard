"""Load every catalog object into one MuJoCo scene, check placement, render a lineup.

Usage (from the project root):
    C:\\Users\\jean-\\.venvs\\sf-hack-assets\\Scripts\\python.exe -I tools/verify_mujoco.py

Writes lineup.png (oblique) and lineup_top.png (top view from 30 m) to the project root,
plus lineup_scene.xml (the generated MJCF, absolute mesh paths).
Every object is placed with its origin (bottom center) on z = 0 and yaw 0 (+X forward).
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ROW_ORDER = ["tent", "car", "suv", "pickup", "truck", "hut", "tree"]  # trees farthest from the oblique camera
GAP = 1.5


def layout(catalog):
    """Rows by object key (cars/suv/pickup share a row); returns [(meta, x, y)]."""
    rows = {}
    for key in ROW_ORDER + [k for k in catalog["objects"] if k not in ROW_ORDER]:
        row = "vehicles" if key in ("car", "suv", "pickup") else key
        rows.setdefault(row, []).extend(catalog["objects"].get(key, []))
    placed, y = [], 0.0
    for row, metas in rows.items():
        if not metas:
            continue
        depth = max(m["size_m"][1] for m in metas)
        widths = [m["size_m"][0] for m in metas]
        total = sum(widths) + GAP * (len(widths) - 1)
        x = -total / 2
        y -= depth / 2
        for m, w in zip(metas, widths):
            placed.append((m, x + w / 2, y))
            x += w + GAP
        y -= depth / 2 + GAP * 1.5
    # center the whole block on the origin
    ys = [p[2] for p in placed]
    off = (max(ys) + min(ys)) / 2
    return [(m, x, yy - off) for m, x, yy in placed]


def build(catalog):
    spec = mujoco.MjSpec()
    spec.modelname = "asset_lineup"
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    spec.visual.global_.fovy = 60
    spec.visual.quality.shadowsize = 4096
    spec.visual.map.shadowclip = 0.6
    spec.visual.headlight.ambient = [0.25, 0.25, 0.25]
    spec.visual.headlight.diffuse = [0.25, 0.25, 0.25]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    wb = spec.worldbody
    tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                           builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           rgb1=[0.42, 0.50, 0.36], rgb2=[0.38, 0.46, 0.33], width=512, height=512)
    mat = spec.add_material(name="ground")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    mat.texrepeat = [40, 30]
    wb.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[40, 30, 0.1],
                material="ground")
    wb.add_light(name="sun", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=[0, 0, 50], dir=[0.3, 0.4, -1],
                 diffuse=[0.55, 0.55, 0.55], specular=[0.05, 0.05, 0.05], castshadow=1)

    entries = []
    for i, (m, x, y) in enumerate(layout(catalog)):
        name = f"{m['key']}__{m['variant']}"
        body = wb.add_body(name=name, pos=[x, y, 0])
        geoms = []
        for p in m["files"]["parts"]:
            mname = f"{name}__{p['name']}"
            mesh = spec.add_mesh(name=mname, file=str(ROOT / p["path"]))
            mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
            body.add_geom(name=mname, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mname,
                          rgba=p["rgba"], contype=0, conaffinity=0, group=0)
            geoms.append(mname)
        cb = m["collision_box"]
        body.add_geom(name=f"{name}__collision", type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=cb["center"], size=cb["half_size"], group=3, rgba=[1, 0, 0, 0.2])
        if "landing_area" in m:  # visualize where the landing station goes (2 x 2 m pad)
            la = m["landing_area"]
            body.add_geom(name=f"{name}__pad", type=mujoco.mjtGeom.mjGEOM_BOX,
                          pos=[la["center"][0], la["center"][1], la["center"][2] + 0.02],
                          size=[1.0, 1.0, 0.02], rgba=la.get("rgba", [0.55, 0.55, 0.55, 1]),
                          contype=0, conaffinity=0)
        entries.append((m, name, geoms, (x, y)))
    return spec, entries


def world_bounds(model, data, geoms):
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in geoms:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        mid = model.geom_dataid[gid]
        a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        v = model.mesh_vert[a:a + n] @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        lo, hi = np.minimum(lo, v.min(0)), np.maximum(hi, v.max(0))
    return lo, hi


def render(model, data, path, lookat, distance, azimuth, elevation):
    r = mujoco.Renderer(model, height=1080, width=1920)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    opt = mujoco.MjvOption()
    r.update_scene(data, camera=cam, scene_option=opt)
    Image.fromarray(r.render()).save(path)
    r.close()


def main():
    catalog = json.loads((ROOT / "catalog.json").read_text())
    spec, entries = build(catalog)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    (ROOT / "lineup_scene.xml").write_text(spec.to_xml())

    print(f"{'object':8s} {'variant':16s} {'size L x W x H (m)':22s} {'tris':>6s} "
          f"{'min z':>7s} {'check':8s} license")
    ok_all = True
    for m, name, geoms, (x, y) in entries:
        lo, hi = world_bounds(model, data, geoms)
        size = hi - lo
        zmin = lo[2]
        status = "ok"
        if abs(zmin) > 0.01:
            status = "FLOATS" if zmin > 0 else "SINKS"
        if np.abs(size - np.array(m["size_m"])).max() > 0.02 + 0.01 * max(m["size_m"]) and \
                "landing_area" not in m:
            status = "SIZE?"
        ok_all &= status == "ok"
        print(f"{m['key']:8s} {m['variant']:16s} {size[0]:5.2f} x {size[1]:5.2f} x {size[2]:5.2f}"
              f"      {m['triangles']:6d} {zmin:7.3f} {status:8s} {m['license']}"
              f"{' (procedural)' if m['procedural'] else ''}")

    xs = [e[3][0] for e in entries]
    ys = [e[3][1] for e in entries]
    c = [(max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, 0]
    render(model, data, ROOT / "lineup.png", [c[0], c[1], 1.5], 42, -65, -30)
    render(model, data, ROOT / "lineup_top.png", c, 30, -90, -90)
    print(f"\nrendered {ROOT / 'lineup.png'} and {ROOT / 'lineup_top.png'} "
          f"(top view: camera 30 m above ground, fovy {model.vis.global_.fovy:.0f} deg)")
    print("ALL OK" if ok_all else "SOME CHECKS FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
