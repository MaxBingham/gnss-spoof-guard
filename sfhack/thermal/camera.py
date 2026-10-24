"""Simulated downward LWIR camera (SIMULATOR SIDE: it reads world/thermal_truth.npz and the MuJoCo state).

    cam = ThermalCamera(sim.model)                 # camera "thermal" on the drone (falls back to "down")
    truth = load_truth(mission_dir)                # world/thermal_truth.npz
    img_K = cam.render(sim.data, truth)            # float32 (512, 640) apparent temperature, K
    img_8 = cam.to_8bit(img_K)                     # uint8 white-hot with AGC
    cam.intrinsics                                 # {fx, fy, cx, cy, w, h, hfov_deg, vfov_deg}

Image orientation = the MuJoCo "down" camera: image top = drone forward, image right = drone right.

How a frame is made: MuJoCo depth + segmentation render at render_scale (default 320 x 256) -> every pixel
back-projected to world xyz -> per geom kind: ground, roof slabs and tree crowns sample the truth raster at
(x, y) (roofs, canopy, cars, wet patches are in it), building walls and trunks get the wall temperature, placed
objects (world/target_scene.json, optional "temp_K") their own, the drone itself air temperature, sky ~air - 30 K;
a short LWIR air path (tau = exp(-range / 8 km)) -> upsample to 640 x 512 -> optics blur (diffraction spot of an
f/1 lens at 12 um pitch, sigma 0.8 px) -> vignetting (narcissus, -0.15 K at the corners) -> fixed-pattern
noise (per-pixel offset 80 mK, column 40 mK, gain 0.5 %) -> absolute calibration error per camera (offset
2 K, gain 3 %: uncooled radiometry is not absolute) -> slow drift (Ornstein-Uhlenbeck offset, 0.25 K,
tau 90 s, reset by a flat-field correction every 180 s) -> temporal noise NETD 50 mK.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np

from ..missionprep.geometry import Grid
from .materials import CLS

SURFACE, WALL, OBJECT, AIR = 0, 1, 2, 3


@dataclass
class ThermalTruth:
    temp_K: np.ndarray
    grid: Grid
    wall_K: float
    air_K: float
    meta: dict[str, Any] = field(default_factory=dict)
    object_K: dict[str, float] = field(default_factory=dict)  # geom name ("target<k>") -> K
    cls: np.ndarray | None = None  # truth material class (uint8), same grid
    roads: list[dict[str, Any]] = field(default_factory=list)  # paved vehicle roads (traffic)

    def sample(self, xy: np.ndarray) -> np.ndarray:
        """Bilinear lookup at ENU points (..., 2) with cv2.remap (fast for whole images)."""
        uv = self.grid.to_px(xy).astype(np.float32)
        return cv2.remap(self.temp_K, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def load_truth(path: str | Path, contrast: float = 1.0, soften_m: float = 0.3,
               target_scene: str | Path | None = None) -> ThermalTruth:
    """Mission directory (or id) or the npz itself. `contrast` < 1 scales the scene around the air temperature
    (a milder night than the model: robustness runs without rebuilding; the onboard prior is unchanged).
    `soften_m`: Gaussian sigma applied to the 0.5 m raster, so class edges look like edges and not like 0.5 m
    stair steps at the camera's 6-9 cm ground resolution. `target_scene`: objects from another target_scene.json
    (default: the one next to the npz; use the file the MuJoCo world was built with)."""
    p = Path(path)
    if not p.exists() and not p.suffix:
        from ..missionprep.build import mission_dir

        p = mission_dir(str(path))
    if p.is_dir():
        p = p / "world" / "thermal_truth.npz"
    with np.load(p) as z:
        meta = json.loads(str(z["meta"]))
        temp = np.ascontiguousarray(z["temp_K"], np.float32)
        wall, air = float(z["wall_K"]), float(z["air_K"])
        cls = np.asarray(z["cls"], np.uint8) if "cls" in z else None
        roads = json.loads(str(z["roads_json"])) if "roads_json" in z else []
    if soften_m > 0:
        temp = cv2.GaussianBlur(temp, (0, 0), soften_m / float(meta["res_m"]))
    if contrast != 1.0:
        temp = (air + contrast * (temp - air)).astype(np.float32)
        wall = air + contrast * (wall - air)
        meta = {**meta, "contrast_at_load": contrast}
    e = meta["extent_enu"]
    grid = Grid(e["min_x"], e["max_y"], float(meta["res_m"]), temp.shape[0], temp.shape[1])
    objects: dict[str, float] = {}
    ts = Path(target_scene) if target_scene is not None else p.parent / "target_scene.json"
    if ts.exists():
        try:
            for k, ob in enumerate(json.loads(ts.read_text(encoding="utf-8")).get("objects", [])):
                objects[f"target{k}"] = float(ob.get("temp_K", air - 1.0))
                for j, part in enumerate(ob.get("parts", [])):  # recon objects: one geom per part
                    objects[f"target{k}_{j}"] = float(part.get("temp_K", ob.get("temp_K", air - 1.0)))
        except (ValueError, OSError):
            pass
    return ThermalTruth(temp, grid, wall, air, meta, objects, cls, roads)


class ThermalCamera:
    def __init__(self, model: mujoco.MjModel, name: str = "thermal", width: int = 640, height: int = 512,
                 hfov_deg: float = 50.0, render_scale: float = 0.5, netd_K: float = 0.05, seed: int = 0,
                 ffc_period_s: float = 180.0, renderer: mujoco.Renderer | None = None,
                 abs_error_K: float = 2.0, gain_error: float = 0.03, traffic: int = 6) -> None:
        """`renderer`: an existing mujoco.Renderer of the same model to draw into (e.g. MujocoSim.renderer, at
        least 300 x 240): one GL context instead of two; switching contexts costs ~50 ms per frame on Windows.
        The internal render then uses a viewport of that renderer."""
        self.model = model
        self.traffic = traffic
        self.vehicles: list[dict[str, Any]] | None = None  # created on the first frame (needs the truth roads)
        self.shared = renderer is not None
        if renderer is not None:
            render_scale = min(renderer.width / width, renderer.height / height)
        self.w, self.h = width, height
        self.hfov = hfov_deg
        self.vfov = math.degrees(2 * math.atan(math.tan(math.radians(hfov_deg) / 2) * height / width))
        try:
            self.cam_id = model.camera(name).id
            self.fovy_override = abs(float(model.cam_fovy[self.cam_id]) - self.vfov) > 0.05
        except KeyError:
            self.cam_id = model.camera("down").id  # same mount and orientation; fovy set while rendering
            self.fovy_override = True
        self.rw, self.rh = max(16, int(round(width * render_scale))), max(16, int(round(height * render_scale)))
        self.netd = netd_K
        self.rng = np.random.default_rng(seed + 4000)
        self.renderer: mujoco.Renderer | None = renderer
        self.kind, self.geom_offset = self._geom_kinds()
        self.geom_names = [model.geom(i).name for i in range(model.ngeom)]
        # static sensor non-uniformity
        r = self.rng
        self.fpn = (r.normal(0, 0.08, (height, width)) + r.normal(0, 0.04, (1, width))).astype(np.float32)
        self.gain = (1.0 + r.normal(0, 0.005, (height, width))).astype(np.float32)
        # uncooled radiometry: absolute accuracy is poor (+-5 K or 5 % in datasheets); per camera a constant
        # offset and a global gain error, so matching must use normalized images, not Kelvin
        self.abs_offset = float(r.normal(0, abs_error_K))
        self.abs_gain = float(1.0 + r.normal(0, gain_error))
        yy, xx = np.mgrid[0:height, 0:width]
        rr = ((xx - width / 2) ** 2 + (yy - height / 2) ** 2) / ((width / 2) ** 2 + (height / 2) ** 2)
        self.vignette = (-0.15 * rr).astype(np.float32)
        # temporal noise: a bank of NETD frames, picked and shifted at random (drawing 330k normals per frame
        # cost ~8 ms)
        self.noise_bank = r.normal(0, netd_K, (8, height, width)).astype(np.float32) if netd_K > 0 else None
        # sub-raster surface texture fixed in the world (aggregate, cracks, grass tufts): 0.25 m cells, 0.2 K,
        # tiled every 64 m. The truth raster is 0.5 m; the camera sees 8.7 cm at 60 m.
        tex = cv2.GaussianBlur(r.normal(0, 1, (256, 256)).astype(np.float32), (0, 0), 0.8)
        self.texture = (tex / max(float(tex.std()), 1e-6) * 0.2).astype(np.float32)
        self.texture_res = 0.25
        self.drift = 0.0
        self.drift_tau, self.drift_std = 90.0, 0.25
        self.ffc_period = ffc_period_s
        self.last_t: float | None = None
        self.last_ffc = 0.0
        self._agc: tuple[float, float] | None = None
        # rays of the internal render (camera frame: x right, y up, looking along -z)
        fy = (self.rh / 2) / math.tan(math.radians(self.vfov) / 2)
        u = (np.arange(self.rw) + 0.5 - self.rw / 2) / fy
        v = (np.arange(self.rh) + 0.5 - self.rh / 2) / fy
        U, V = np.meshgrid(u, v)
        self.rays = np.stack([U, -V, -np.ones_like(U)], -1).astype(np.float32)  # z-depth 1
        self.ray_len = np.linalg.norm(self.rays, axis=-1)

    # ------------------------------------------------------------------ public

    @property
    def intrinsics(self) -> dict[str, float]:
        fy = (self.h / 2) / math.tan(math.radians(self.vfov) / 2)
        fx = (self.w / 2) / math.tan(math.radians(self.hfov) / 2)
        return {"fx": fx, "fy": fy, "cx": self.w / 2, "cy": self.h / 2, "w": self.w, "h": self.h,
                "hfov_deg": self.hfov, "vfov_deg": self.vfov}

    def render(self, data: mujoco.MjData, thermal_truth: ThermalTruth | str | Path) -> np.ndarray:
        truth = thermal_truth if isinstance(thermal_truth, ThermalTruth) else self._truth_cached(thermal_truth)
        scene = self.scene_temperature(data, truth)
        img = cv2.resize(scene, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        img = cv2.GaussianBlur(img, (0, 0), 0.8)
        self._advance_drift(float(data.time))
        img = (img - 293.15) * (self.gain * self.abs_gain) + 293.15 + self.abs_offset + self.fpn + self.vignette + self.drift
        if self.noise_bank is not None:
            k = int(self.rng.integers(len(self.noise_bank)))
            dy, dx = int(self.rng.integers(self.h)), int(self.rng.integers(self.w))
            img += np.roll(self.noise_bank[k], (dy, dx), axis=(0, 1))
        return img.astype(np.float32)

    def to_8bit(self, img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0, min_span_K: float = 4.0,
                smooth: float = 0.3) -> np.ndarray:
        """Linear AGC on the 1-99 % histogram range (at least min_span_K wide), smoothed over frames so the
        brightness does not flicker; white-hot."""
        lo, hi = np.percentile(img[::4, ::4], [lo_pct, hi_pct])
        if hi - lo < min_span_K:
            mid = (hi + lo) / 2
            lo, hi = mid - min_span_K / 2, mid + min_span_K / 2
        if self._agc is not None:
            lo = self._agc[0] + smooth * (lo - self._agc[0])
            hi = self._agc[1] + smooth * (hi - self._agc[1])
        self._agc = (float(lo), float(hi))
        return np.clip((img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)

    def close(self) -> None:
        if self.renderer is not None and not self.shared:
            self.renderer.close()
            self.renderer = None

    # ------------------------------------------------------------------ internals

    _truth_cache: dict[str, ThermalTruth] = {}

    def _truth_cached(self, p: str | Path) -> ThermalTruth:
        key = str(p)
        if key not in self._truth_cache:
            self._truth_cache[key] = load_truth(p)
        return self._truth_cache[key]

    def _geom_kinds(self) -> tuple[np.ndarray, np.ndarray]:
        m = self.model
        kind = np.full(m.ngeom + 1, SURFACE, np.int8)  # last slot: background (sky)
        off = np.zeros(m.ngeom + 1, np.float32)
        drone = m.body("x2").id if "x2" in [m.body(i).name for i in range(m.nbody)] else -1
        rng = np.random.default_rng(17)
        for i in range(m.ngeom):
            g = m.geom(i)
            body = int(m.geom_bodyid[i])
            root = int(m.body_rootid[body])
            if drone >= 0 and (body == drone or root == drone):
                kind[i] = AIR
                continue
            name = g.name
            mesh = m.mesh(int(m.geom_dataid[i])).name if int(m.geom_type[i]) == mujoco.mjtGeom.mjGEOM_MESH else ""
            if name.startswith("target"):
                kind[i] = OBJECT
            elif mesh.startswith("bw"):
                kind[i] = WALL
                off[i] = rng.normal(0, 1.0)
            elif int(m.geom_type[i]) == mujoco.mjtGeom.mjGEOM_CYLINDER and int(m.geom_group[i]) == 1:
                kind[i] = WALL  # tree trunk: about air temperature
                off[i] = -1.0
        return kind, off

    # ------------------------------------------------------------------ traffic (per flight, not in any map)

    _PAVED = tuple(CLS[n] for n in ("asphalt", "asphalt_minor", "concrete", "pavers", "parking"))

    def _spawn_vehicles(self, truth: ThermalTruth) -> None:
        r = np.random.default_rng(int(self.rng.integers(1 << 31)))
        segs = []
        for w in truth.roads:
            p = np.asarray(w["pts_enu"], float)
            if len(p) >= 2:
                L = float(np.hypot(*np.diff(p, axis=0).T).sum())
                if L > 30:
                    segs.append((p, L, float(w["width_m"])))
        self.vehicles = []
        if not segs:
            return
        lens = np.array([s[1] for s in segs])
        for _ in range(self.traffic):
            p, L, width = segs[int(r.choice(len(segs), p=lens / lens.sum()))]
            self.vehicles.append({"pts": p, "cum": np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))]),
                                  "s": float(r.uniform(0, L)), "dir": float(r.choice([-1.0, 1.0])),
                                  "speed": float(r.uniform(6, 12)), "lane": max(width / 4, 1.0),
                                  "hood_K": float(r.uniform(14, 25)), "body_K": float(r.normal(-1.0, 0.7)), "t": None})

    def _paint_traffic(self, out: np.ndarray, pts: np.ndarray, kind: np.ndarray, truth: ThermalTruth,
                       t: float) -> np.ndarray:
        """Moving cars on paved vehicle roads: warm body, hot engine hood, cold windscreen (reflects the sky)."""
        if self.vehicles is None:
            self._spawn_vehicles(truth)
        on_paved = None
        cam_xy = pts[pts.shape[0] // 2, pts.shape[1] // 2, :2]
        for v in self.vehicles or []:
            dt = 0.0 if v["t"] is None else max(t - v["t"], 0.0)
            v["t"] = t
            v["s"] += v["dir"] * v["speed"] * dt
            if not 0.0 <= v["s"] <= v["cum"][-1]:  # end of the road: turn around
                v["s"] = float(np.clip(v["s"], 0.0, v["cum"][-1]))
                v["dir"] = -v["dir"]
            i = int(np.clip(np.searchsorted(v["cum"], v["s"]) - 1, 0, len(v["pts"]) - 2))
            a, b = v["pts"][i], v["pts"][i + 1]
            seg = b - a
            u = seg / max(float(np.hypot(*seg)), 1e-6) * v["dir"]
            n = np.array([u[1], -u[0]])  # right of the driving direction
            c = a + (b - a) * ((v["s"] - v["cum"][i]) / max(v["cum"][i + 1] - v["cum"][i], 1e-6)) + n * v["lane"]
            if np.hypot(*(c - cam_xy)) > 80.0:
                continue
            d = pts[..., :2] - c
            lx, ly = d @ u, d @ n
            car = (np.abs(lx) < 2.25) & (np.abs(ly) < 0.9) & (kind == SURFACE)
            if not car.any():
                continue
            if on_paved is None and truth.cls is not None:
                uv = truth.grid.to_px(pts[..., :2]).astype(np.float32)
                cl = cv2.remap(truth.cls, uv[..., 0], uv[..., 1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
                on_paved = np.isin(cl, self._PAVED)
            if on_paved is not None:
                car &= on_paved  # hidden under crowns and bridges
            temp = np.full(out.shape, truth.air_K + v["body_K"], np.float32)
            temp = np.where(lx > 1.2, truth.air_K + v["hood_K"], temp)  # engine bay
            temp = np.where((lx > 0.5) & (lx <= 1.2), truth.air_K - 4.0, temp)  # windscreen: cold sky
            out = np.where(car, temp, out)
        return out

    def _advance_drift(self, t: float) -> None:
        if self.last_t is None or t < self.last_t:
            self.last_t, self.last_ffc, self.drift = t, t, 0.0
            return
        dt = t - self.last_t
        self.last_t = t
        if t - self.last_ffc >= self.ffc_period:
            self.last_ffc, self.drift = t, 0.0  # flat-field correction (shutter)
            return
        a = math.exp(-dt / self.drift_tau)
        self.drift = a * self.drift + math.sqrt(max(1 - a * a, 0.0)) * self.drift_std * float(self.rng.normal())

    def _render_fused(self, r: mujoco.Renderer) -> tuple[np.ndarray, np.ndarray]:
        """Depth AND segmentation from ONE GL pass (mujoco.Renderer needs two: the render plus a GL context
        switch is the frame cost). Same maths as mujoco.Renderer.render: reverse-Z perspective depth, segid
        colours -> (objid, objtype)."""
        scene, ctx = r._scene, r._mjr_context
        rect = mujoco.MjrRect(0, 0, self.rw, self.rh)
        flags = scene.flags.copy()
        scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = True
        scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = True
        if r._gl_context:
            r._gl_context.make_current()
        rgb = np.empty((self.rh, self.rw, 3), np.uint8)
        dep = np.empty((self.rh, self.rw), np.float32)
        try:
            mujoco.mjr_render(rect, scene, ctx)
            mujoco.mjr_readPixels(rgb, dep, rect, ctx)
        finally:
            np.copyto(scene.flags, flags)
        m = self.model
        far = np.float32(m.vis.map.zfar * m.stat.extent)
        near = np.float32(m.vis.map.znear * m.stat.extent)
        c = -(far + near) / (far - near)
        d = -(np.float32(2) * far * near) / (far - near)
        c, d = np.float32(-0.5) * c - np.float32(0.5), np.float32(-0.5) * d
        depth = (d / (dep.astype(np.float64) + c)).astype(np.float32)
        segimg = rgb[..., 0].astype(np.int64) + rgb[..., 1].astype(np.int64) * 256 + rgb[..., 2].astype(np.int64) * 65536
        n = scene.ngeom
        lut = np.full((n + 2, 2), -1, np.int32)
        for i in range(n):
            g = scene.geoms[i]
            if g.segid != -1:
                lut[g.segid + 1] = (g.objid, g.objtype)
        seg = lut[np.clip(segimg, 0, n + 1)]
        if r._gl_context:
            depth, seg = np.flipud(depth), np.flipud(seg)
        return np.ascontiguousarray(depth), np.ascontiguousarray(seg)

    def scene_temperature(self, data: mujoco.MjData, truth: ThermalTruth) -> np.ndarray:
        """Noise-free apparent temperature at the internal resolution (rh, rw)."""
        m = self.model
        if self.renderer is None:
            self.renderer = mujoco.Renderer(m, self.rh, self.rw)
        r = self.renderer
        saved = float(m.cam_fovy[self.cam_id])
        if self.fovy_override:
            m.cam_fovy[self.cam_id] = self.vfov
        try:
            r.update_scene(data, camera=self.cam_id)
            try:
                depth, seg = self._render_fused(r)
            except AttributeError:  # Renderer internals changed: two passes
                r.enable_depth_rendering()
                depth = r.render().copy()
                r.disable_depth_rendering()
                r.enable_segmentation_rendering()
                seg = r.render().copy()
                r.disable_segmentation_rendering()
        finally:
            m.cam_fovy[self.cam_id] = saved
        gid = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
        gidx = np.where(gid >= 0, gid, m.ngeom)
        kind = self.kind[gidx]
        pos = data.cam_xpos[self.cam_id]
        R = data.cam_xmat[self.cam_id].reshape(3, 3)
        pts = pos + (self.rays * depth[..., None]) @ R.T
        out = truth.sample(pts[..., :2])
        tu = (pts[..., 0] / self.texture_res).astype(np.float32) % 256
        tv = (-pts[..., 1] / self.texture_res).astype(np.float32) % 256
        out = out + cv2.remap(self.texture, tu, tv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        out = np.where(kind == WALL, truth.wall_K + self.geom_offset[gidx], out)
        out = np.where(kind == AIR, truth.air_K, out)
        if (kind == OBJECT).any():
            obj_t = np.array([truth.object_K.get(n, truth.air_K - 1.0) for n in self.geom_names] + [truth.air_K],
                             np.float32)
            out = np.where(kind == OBJECT, obj_t[gidx], out)
        out = np.where(gid < 0, truth.air_K - 30.0, out)  # sky
        if self.traffic and truth.roads:
            out = self._paint_traffic(out, pts, kind, truth, float(data.time))
        rng_m = depth * self.ray_len
        tau = np.exp(-rng_m / 8000.0)
        return (tau * out + (1 - tau) * truth.air_K).astype(np.float32)
