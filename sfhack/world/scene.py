"""MuJoCo worlds: decorate the Skydio X2 scene spec with ground, buildings and trees.

- `ArenaWorld`: the small random arena from fly_x2.py (no map package needed).
- `PackageWorld`: built from a map package: orthophoto-textured ground plane, buildings as
  collision boxes (oriented bounding box of the OSM footprint), trees as cylinder + sphere
  (visual only). The ground texture file is selectable, so the simulated "reality" can use
  different imagery than the onboard map (map-mismatch tests).
  Mission worlds (sfhack.missionprep, docs/MAP_PACKAGE.md) add terrain (`terrain.npz` -> height
  field, ortho draped on it) and `"buildings": "extruded"`: footprint prisms (visual mesh, roof slab
  texture-mapped into the ortho, convex triangle prisms in hidden group 3 for collision).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np

from ..missionprep.geometry import Grid, clean_ring, obb, prism_mesh, sample_bilinear, triangulate
from .buildings import building_boxes, footprint_obb  # noqa: F401  (moved there: the onboard DSM uses them too)


def _no_shadows(spec: mujoco.MjSpec) -> None:
    for light in spec.lights:
        light.castshadow = False


class ArenaWorld:
    """fly_x2.py's test arena: a few boxes ('buildings') and spheres ('trees') around a 15 m square."""

    def __init__(self, seed: int = 1, keep_free: list[tuple[float, float]] | None = None) -> None:
        self.seed = seed
        self.keep_free = keep_free or [(0, 0), (15, 0), (15, 15)]

    def add_to_spec(self, spec: mujoco.MjSpec) -> None:
        world = spec.worldbody
        rng = np.random.default_rng(self.seed)
        for i in range(12):
            x, y = rng.uniform(-5, 22, size=2)
            if min(math.hypot(x - fx, y - fy) for fx, fy in self.keep_free) < 3:
                continue
            h = rng.uniform(1.0, 3.5)
            world.add_geom(name=f"building{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                           size=[rng.uniform(0.6, 1.5), rng.uniform(0.6, 1.5), h / 2],
                           pos=[x, y, h / 2], rgba=[0.75, 0.75, 0.78, 1])
        for i in range(25):
            x, y = rng.uniform(-5, 22, size=2)
            world.add_geom(name=f"tree{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.5, 0, 0],
                           pos=[x, y, 0.5], rgba=[0.15, 0.55, 0.2, 1], contype=0, conaffinity=0)


def roof_image(ortho: np.ndarray, extent: dict[str, float], box: dict[str, Any], mpp: float = 0.5) -> np.ndarray:
    """Ortho crop under a building box, resampled into the box frame (row 0 = box +y edge, like the floor)."""
    hx, hy = box["half"]
    cx, cy = box["center"]
    W, H = max(2, int(round(2 * hx / mpp))), max(2, int(round(2 * hy / mpp)))
    a = -hx + (np.arange(W) + 0.5) * (2 * hx / W)
    b = hy - (np.arange(H) + 0.5) * (2 * hy / H)
    A, Bm = np.meshgrid(a, b)
    c, s = math.cos(box["yaw"]), math.sin(box["yaw"])
    X, Y = cx + A * c - Bm * s, cy + A * s + Bm * c
    h, w = ortho.shape[:2]
    u = (X - extent["min_x"]) / (extent["max_x"] - extent["min_x"]) * w - 0.5
    v = (extent["max_y"] - Y) / (extent["max_y"] - extent["min_y"]) * h - 0.5
    return np.ascontiguousarray(cv2.remap(ortho, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REPLICATE))


class PackageWorld:
    def __init__(self, pkg_dir: Path | str, texture_file: str | None = None,
                 render_trees: bool = True, roof_textures: bool = True) -> None:
        self.roof_textures = roof_textures
        self.dir = Path(pkg_dir).resolve()
        self.manifest: dict[str, Any] = json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))
        self.landmarks: list[dict[str, Any]] = json.loads((self.dir / "landmarks.json").read_text(encoding="utf-8"))
        self.texture_file = texture_file or self.manifest["ortho"]["file"]
        self.render_trees = render_trees
        self.terrain_z: np.ndarray | None = None  # north-up, row 0 = north (docs/MAP_PACKAGE.md)
        self.terrain_grid: Grid | None = None
        t = self.manifest.get("terrain")
        if isinstance(t, dict) and t.get("file"):
            with np.load(self.dir / t["file"]) as z:
                self.terrain_z = np.asarray(z["z"], np.float32)
                self.terrain_grid = Grid(float(z["min_x"]), float(z["max_y"]), float(z["res_m"]), *self.terrain_z.shape)

    def ground_z(self, x: float, y: float) -> float:
        """TRUE ground height (simulator side: spawn height, display). 0 for flat worlds."""
        if self.terrain_z is None or self.terrain_grid is None:
            return 0.0
        return float(sample_bilinear(self.terrain_z, self.terrain_grid, np.array([[x, y]]))[0])

    @property
    def a_enu(self) -> np.ndarray:
        return np.array(self.manifest["a"]["enu"], float)

    @property
    def b_enu(self) -> np.ndarray:
        return np.array(self.manifest["b"]["enu"], float)

    def add_to_spec(self, spec: mujoco.MjSpec) -> None:
        if self.manifest.get("buildings") == "extruded":
            self._add_mission_world(spec)
            return
        _no_shadows(spec)
        e = self.manifest["ortho"]["extent_enu"]
        cx, cy = (e["min_x"] + e["max_x"]) / 2, (e["min_y"] + e["max_y"]) / 2
        hw, hh = (e["max_x"] - e["min_x"]) / 2, (e["max_y"] - e["min_y"]) / 2

        # MuJoCo only decodes PNG from files; pass the decoded pixels so JPEG packages work too.
        img = cv2.cvtColor(cv2.imread(str(self.dir / self.texture_file), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        tex = spec.add_texture(name="ortho", type=mujoco.mjtTexture.mjTEXTURE_2D,
                               width=img.shape[1], height=img.shape[0], nchannel=3)
        tex.data = img.tobytes()
        mat = spec.add_material(name="ortho", texrepeat=[1, 1], texuniform=False, reflectance=0.0)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
        floor = spec.geom("floor")
        floor.size = [hw, hh, 1.0]
        floor.pos = [cx, cy, 0.0]
        floor.material = "ortho"

        world = spec.worldbody
        mat_b = spec.add_material(name="building", rgba=[0.72, 0.71, 0.68, 1])
        mat_t = spec.add_material(name="tower", rgba=[0.85, 0.82, 0.74, 1])
        for box in building_boxes(self.landmarks):
            h = box["height"]
            quat = [math.cos(box["yaw"] / 2), 0, 0, math.sin(box["yaw"] / 2)]
            world.add_geom(name=f"lm{box['id']}", type=mujoco.mjtGeom.mjGEOM_BOX,
                           size=[box["half"][0], box["half"][1], h / 2],
                           pos=[box["center"][0], box["center"][1], h / 2], quat=quat,
                           material=mat_t.name if box["type"] == "tower" else mat_b.name)
            if self.roof_textures:
                # Roof = the orthophoto under the footprint (a real roof looks like its aerial photo; a blank
                # gray box top gave the down camera nothing to match over the whole campus core).
                rname = f"roof{box['id']}"
                rtex = spec.add_texture(name=rname, type=mujoco.mjtTexture.mjTEXTURE_2D)
                roof = roof_image(img, e, box)
                rtex.width, rtex.height, rtex.nchannel = roof.shape[1], roof.shape[0], 3
                rtex.data = roof.tobytes()
                rmat = spec.add_material(name=rname, texrepeat=[1, 1], texuniform=False, reflectance=0.0)
                rmat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = rname
                world.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[box["half"][0], box["half"][1], 1.0],
                               pos=[box["center"][0], box["center"][1], h + 0.02], quat=quat,
                               material=rname, contype=0, conaffinity=0)
        if self.render_trees:
            trunk = [0.36, 0.27, 0.18, 1]
            crown = [0.20, 0.42, 0.18, 1]
            for lm in self.landmarks:
                if lm["type"] != "tree":
                    continue
                x, y = lm["centroid_enu"]
                h, r = float(lm["height"]), float(lm.get("radius", 3.0))
                trunk_h = max(h - r, 1.0)
                world.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.25, trunk_h / 2, 0],
                               pos=[x, y, trunk_h / 2], rgba=trunk, contype=0, conaffinity=0, group=1)
                world.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[r, 0, 0],
                               pos=[x, y, trunk_h], rgba=crown, contype=0, conaffinity=0, group=1)

    # ------------------------------------------------------------------ mission worlds (terrain + extruded)

    def _terrain_pad(self, x: float, y: float, half: float, half_z: float) -> dict[str, list[float]]:
        """pos / quat of a box whose top face lies in the least-squares terrain plane around (x, y), 1 cm proud."""
        s = np.linspace(-half, half, 9)
        dx, dy = (v.ravel() for v in np.meshgrid(s, s))
        z = np.array([self.ground_z(x + a, y + b) for a, b in zip(dx, dy)])
        (gx, gy, z0), *_ = np.linalg.lstsq(np.column_stack([dx, dy, np.ones_like(dx)]), z, rcond=None)
        n = np.array([-gx, -gy, 1.0]) / math.sqrt(gx * gx + gy * gy + 1.0)
        ang = math.acos(float(n[2]))
        axis = np.cross([0.0, 0.0, 1.0], n)
        axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 1e-9 else np.array([1.0, 0.0, 0.0])
        quat = [math.cos(ang / 2), *(axis * math.sin(ang / 2))]
        pos = np.array([x, y, z0]) + n * (0.01 - half_z)
        return {"pos": pos.tolist(), "quat": quat}

    def _add_mission_world(self, spec: mujoco.MjSpec) -> None:
        _no_shadows(spec)
        e = self.manifest["ortho"]["extent_enu"]
        img = cv2.cvtColor(cv2.imread(str(self.dir / self.texture_file), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        world = spec.worldbody

        def texture(name: str, rgb: np.ndarray) -> str:
            rgb = np.ascontiguousarray(rgb)
            tex = spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D,
                                   width=rgb.shape[1], height=rgb.shape[0], nchannel=3)
            tex.data = rgb.tobytes()
            mat = spec.add_material(name=name, texrepeat=[1, 1], texuniform=False, reflectance=0.0)
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
            return name

        floor = spec.geom("floor")
        if self.terrain_z is not None and self.terrain_grid is not None:
            g, z = self.terrain_grid, self.terrain_z
            zmin, zmax = float(z.min()), float(z.max())
            span = max(zmax - zmin, 0.1)
            # MuJoCo height field: vertices on the grid points (= our cell centres) and data row 0 = SOUTH
            # edge, while a 2D texture is draped north-up over the field. Crop the ortho to that span.
            rx, ry = (g.cols - 1) * g.res / 2, (g.rows - 1) * g.res / 2
            cx, cy = g.min_x + g.cols * g.res / 2, g.max_y - g.rows * g.res / 2
            hf = spec.add_hfield(name="terrain", size=[rx, ry, span, 1.0], nrow=g.rows, ncol=g.cols)
            hf.userdata = ((z[::-1] - zmin) / span).astype(np.float64).ravel().tolist()
            h, w = img.shape[:2]
            px = round((g.res / 2) / ((e["max_x"] - e["min_x"]) / w))
            py = round((g.res / 2) / ((e["max_y"] - e["min_y"]) / h))
            world.add_geom(name="terrain", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="terrain",
                           pos=[cx, cy, zmin], material=texture("ortho_hf", img[py:h - py, px:w - px]))
            floor.pos = [cx, cy, zmin - 1.0]
            floor.rgba = [0.42, 0.42, 0.40, 1]
            floor.material = ""
            # Contact pads at A and B: invisible boxes in the local terrain plane. On the height field alone
            # the X2 (a small box + rotor ellipsoids underneath) rested 9 deg tilted at A, which the preflight
            # IMU calibration took for level, and rocked at B so the IMU never looked still after touchdown.
            # (The builder also levels the terrain under A: a launch pad.)
            for name, (px_, py_), half in (("pad_a", self.a_enu, 2.5), ("pad_b", self.b_enu, 4.0)):
                world.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=[half, half, 0.25], group=3,
                               rgba=[0, 0, 0, 0], **self._terrain_pad(px_, py_, half, 0.25))
        else:
            floor.size = [(e["max_x"] - e["min_x"]) / 2, (e["max_y"] - e["min_y"]) / 2, 1.0]
            floor.pos = [(e["min_x"] + e["max_x"]) / 2, (e["min_y"] + e["max_y"]) / 2, 0.0]
            floor.material = texture("ortho_hf", img)
        roof_mat = texture("ortho_roof", img)  # roofs index the full ortho through texcoords
        mat_b = spec.add_material(name="building", rgba=[0.72, 0.71, 0.68, 1])
        mat_t = spec.add_material(name="tower", rgba=[0.85, 0.82, 0.74, 1])

        def uv(xy: np.ndarray) -> np.ndarray:
            return np.column_stack([(xy[:, 0] - e["min_x"]) / (e["max_x"] - e["min_x"]),
                                    (e["max_y"] - xy[:, 1]) / (e["max_y"] - e["min_y"])])

        self.n_collision_meshes = 0
        for lm in self.landmarks:
            if lm["type"] not in ("building", "tower", "fountain"):
                continue
            base = float(lm.get("base_z", 0.0))
            top = base + float(lm["height"])
            mat = mat_t.name if lm["type"] == "tower" else mat_b.name
            ring = clean_ring(lm["footprint_enu"]) if lm.get("footprint_enu") else None
            tris = triangulate(ring) if ring is not None and len(ring) >= 3 else None
            if tris is None:  # node landmark or a self-intersecting footprint: box / cylinder stand-in
                if ring is not None and len(ring) >= 3:
                    cx_, cy_, hx, hy, ang = obb(ring)
                    world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, (top - base + 1) / 2],
                                   pos=[cx_, cy_, (top + base - 1) / 2],
                                   quat=[math.cos(ang / 2), 0, 0, math.sin(ang / 2)], material=mat)
                else:
                    x, y = lm["centroid_enu"]
                    world.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[x, y, (top + base - 1) / 2],
                                   size=[float(lm.get("radius", 3.0)), (top - base + 1) / 2, 0], material=mat)
                continue
            i = lm["id"]
            v, f = prism_mesh(ring, tris, base - 1.0, top)  # sunk 1 m: footprints sit on slopes
            spec.add_mesh(name=f"bw{i}", uservert=v.ravel().tolist(), userface=f.ravel().tolist())
            world.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"bw{i}", material=mat, contype=0, conaffinity=0)
            v, f = prism_mesh(ring, tris, top, top + 0.05)  # roof slab showing the ortho under the footprint
            spec.add_mesh(name=f"br{i}", uservert=v.ravel().tolist(), userface=f.ravel().tolist(),
                          usertexcoord=np.vstack([uv(ring), uv(ring)]).ravel().tolist())
            world.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"br{i}", material=roof_mat,
                           contype=0, conaffinity=0)
            for k, tri in enumerate(tris):  # collision: convex pieces (MuJoCo collides a mesh as its convex hull)
                tr = ring[list(tri)]
                (ux, uy), (vx, vy) = tr[1] - tr[0], tr[2] - tr[0]
                if abs(ux * vy - uy * vx) < 0.02:  # sliver: no volume to collide with
                    continue
                v, f = prism_mesh(tr, [(0, 1, 2)], base - 1.0, top + 0.05)
                spec.add_mesh(name=f"bc{i}_{k}", uservert=v.ravel().tolist(), userface=f.ravel().tolist())
                world.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"bc{i}_{k}", group=3, rgba=[1, 0, 0, 0])
                self.n_collision_meshes += 1

        if self.render_trees:
            trunk = [0.36, 0.27, 0.18, 1]
            crown = [0.20, 0.42, 0.18, 1]
            for lm in self.landmarks:
                if lm["type"] != "tree":
                    continue
                x, y = lm["centroid_enu"]
                z0 = float(lm.get("base_z", 0.0))
                h, r = float(lm["height"]), float(lm.get("radius", 3.0))
                trunk_h = max(h - r, 1.0)
                world.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.25, trunk_h / 2, 0],
                               pos=[x, y, z0 + trunk_h / 2], rgba=trunk, contype=0, conaffinity=0, group=1)
                world.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[r, 0, 0],
                               pos=[x, y, z0 + trunk_h], rgba=crown, contype=0, conaffinity=0, group=1)
        self._add_target_scene(spec)

    def _add_target_scene(self, spec: mujoco.MjSpec) -> None:
        """Objects the object placer put into world/target_scene.json (TRUTH: the drone has no map of them).
        Each: {"mesh": STL path (absolute or relative to world/), "pos": [x, y, z], "yaw_deg", "scale", "rgba"}."""
        p = self.dir / "target_scene.json"
        if not p.exists():
            return
        for k, ob in enumerate(json.loads(p.read_text(encoding="utf-8")).get("objects", [])):
            mesh = Path(ob["mesh"]) if Path(ob["mesh"]).is_absolute() else self.dir / ob["mesh"]
            spec.add_mesh(name=f"target{k}", file=str(mesh), scale=[float(ob.get("scale", 1.0))] * 3)
            x, y = ob["pos"][:2]
            z = float(ob["pos"][2]) if len(ob["pos"]) > 2 else self.ground_z(x, y)
            yaw = math.radians(float(ob.get("yaw_deg", 0.0)))
            spec.worldbody.add_geom(name=f"target{k}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"target{k}",
                                    pos=[x, y, z], quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)],
                                    rgba=ob.get("rgba", [0.85, 0.85, 0.85, 1]))
