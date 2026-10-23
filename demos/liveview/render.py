"""Offscreen MuJoCo render of the chase view with a painted sky and depth-based weather fog.

The fog is applied in post from the depth buffer (Koschmieder: contrast falls to 5 % at the visibility), so
it tracks the weather lever every frame. MuJoCo's own fog is fixed when the render context is created.
"""

from __future__ import annotations

import time
from typing import Callable

import cv2
import mujoco
import numpy as np

SKY_TOP = np.array([0.32, 0.50, 0.75], np.float32)
SKY_HORIZON = np.array([0.80, 0.86, 0.92], np.float32)
# Chase-view only. Sensor haze and weather_effect stay on the visibility curve; this floor keeps
# fog 0.65 as haze so building edges survive (transmission never falls below this).
FOG_TRANSMISSION_FLOOR = np.float32(0.35)


class SceneRenderer:
    def __init__(self, model: mujoco.MjModel, width: int, height: int) -> None:
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        model.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
        model.vis.headlight.diffuse[:] = [0.55, 0.55, 0.55]
        self.model = model
        self.W, self.H = width, height
        self.r = mujoco.Renderer(model, height, width, max_geom=20000)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        self.opt.geomgroup[:3] = 1  # world, trees, X2 visual mesh (3 = X2 collision shapes)
        self._rgb = np.zeros((height, width, 3), np.uint8)
        self._depth = np.zeros((height, width), np.float32)
        ext = model.stat.extent
        near, far = model.vis.map.znear * ext, model.vis.map.zfar * ext
        c = -(far + near) / (far - near)
        d = -(2.0 * far * near) / (far - near)
        self._c, self._d = -0.5 * c - 0.5, -0.5 * d  # reverse-Z depth -> metric distance: d / (z + c)
        self._sky = self._sky_gradient()
        self.gl_cam: tuple | None = None
        self.t_view = 0.0
        self.t_fog = 0.0

    def _sky_gradient(self) -> np.ndarray:
        t = np.linspace(0.0, 1.0, self.H, dtype=np.float32)[:, None] ** 0.6
        col = SKY_TOP[None, :] * (1 - t) + SKY_HORIZON[None, :] * t
        return np.repeat((col * 255)[:, None, :], self.W, axis=1)

    def render(self, data: mujoco.MjData, cam: mujoco.MjvCamera, draw: Callable[[mujoco.MjvScene], None],
               visibility_m: float | None, fog_rgb: tuple[float, float, float], fog_start_m: float = 0.0) -> np.ndarray:
        t0 = time.perf_counter()
        r = self.r
        r.update_scene(data, camera=cam, scene_option=self.opt)
        scn = r.scene
        scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        scn.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 0
        draw(scn)
        if r._gl_context:
            r._gl_context.make_current()
        mujoco.mjr_render(r._rect, scn, r._mjr_context)
        g0, g1 = scn.camera[0], scn.camera[1]
        width = g0.frustum_width or 0.5 * (g0.frustum_top - g0.frustum_bottom) * self.W / self.H
        self.gl_cam = (0.5 * (np.array(g0.pos) + np.array(g1.pos)), np.array(g0.forward), np.array(g0.up),
                       g0.frustum_center, width, g0.frustum_bottom, g0.frustum_top, g0.frustum_near)
        mujoco.mjr_readPixels(self._rgb, self._depth, r._rect, r._mjr_context)
        rgb = np.flipud(self._rgb)
        z = np.flipud(self._depth)
        sky = z <= 1e-9
        out = rgb.astype(np.float32)
        out[sky] = self._sky[sky]
        t1 = time.perf_counter()
        # Clear frames skip the distance buffer (visibility is None). Fog math stays float32.
        # The exp is ~2 ms; the full-frame blend is the rest of this slice. Half-res transmission
        # was timed and did not move the frame, so the composite stays full resolution.
        if visibility_m is not None and visibility_m < 5000.0:
            zf = np.asarray(z, np.float32)
            dist = np.float32(self._d) / (zf + np.float32(self._c))
            dist[sky] = np.float32(1.0e6)
            travel = np.maximum(dist - np.float32(fog_start_m), np.float32(0.0))
            tr = np.exp(np.float32(-3.0) * travel / np.float32(visibility_m))
            tr = np.maximum(tr, FOG_TRANSMISSION_FLOOR)[..., None]
            fog = np.asarray(fog_rgb, np.float32) * np.float32(255.0)
            out = out * tr + fog * (np.float32(1.0) - tr)
        self.t_view = t1 - t0
        self.t_fog = time.perf_counter() - t1
        return np.clip(out, 0, 255).astype(np.uint8)

    def project(self, p) -> tuple[int, int] | None:
        """World point -> pixel (x right, y down) of the last rendered frame, None if behind the camera."""
        if self.gl_cam is None:
            return None
        pos, fwd, up, center, width, bottom, top, near = self.gl_cam
        v = np.asarray(p, float) - pos
        zc = float(v @ fwd)
        if zc <= near:
            return None
        right = np.cross(fwd, up)
        xn, yn = float(v @ right) * near / zc, float(v @ up) * near / zc
        u = (xn - (center - width)) / (2 * width) * self.W
        vv = (top - yn) / (top - bottom) * self.H
        return int(round(u)), int(round(vv))

    def close(self) -> None:
        self.r.close()


def blend_rgba(dst: np.ndarray, rgba: np.ndarray, x: int, y: int) -> None:
    """Alpha-composite an RGBA patch onto an RGB frame at (x, y), clipped to the frame."""
    H, W = dst.shape[:2]
    h, w = rgba.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    patch = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    a = patch[..., 3:4].astype(np.float32) / 255.0
    roi = dst[y0:y1, x0:x1]
    roi[:] = (roi * (1 - a) + patch[..., :3] * a).astype(np.uint8)


def paste(dst: np.ndarray, img: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    dst[y:y + h, x:x + w] = img
