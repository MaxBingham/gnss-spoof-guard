"""Recon dashboard (demos/51_recon_dashboard.py): the bearing recon flight in the live-demo style.

    3D      chase view: white X2 = TRUE (simulator), cyan ghost + 2-sigma bubble = ESTIMATE (EKF), white / cyan trails,
            red dashes = error between them; amber dashed = IDEAL plan (route + orbit), orange ray = UMA-16 bearing
    left    BASE KNOWS (bearing, launch command, LINK: NONE)  |  SIM TRUTH (state, errors, GNSS, drift)  |  map
    right   front FPV camera, downward camera (onboard renders), legend
    top     LINK: NONE banner, event banners (GNSS off, landed)

Harness code: reads the truth to DISPLAY and SCORE only. The flight itself runs on BearingPilot (onboard).
"""

from __future__ import annotations

import math
import time
from typing import Any

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .chase import ChaseCam
from .geoms import Geoms
from .hud import AMBER, CYAN, GREEN, MUTED, ORANGE, RED, TEXT, VIOLET, WHITE, Hud, mix
from .render import SceneRenderer, paste
from sfhack.mission.bearing_pilot import planned_route

CYAN3 = (0.0, 0.85, 1.0)
WHITE3 = (1.0, 1.0, 1.0)
RED3 = (0.95, 0.15, 0.1)
IDEAL3 = (1.0, 0.78, 0.15)
BEAR3 = (1.0, 0.35, 0.12)
GREEN3 = (0.15, 0.85, 0.2)
DRONE_SCALE = 9.0  # the X2 is 0.4 m across; drawn 9x (more when the camera pulls back)
ERR_EVERY = 6  # trail points between two red error dashes (map)
ERR_EVERY_3D = 12  # ... in 3D (one GL line each)
STATE_COLOR = {"WAIT": MUTED, "TAKEOFF": CYAN, "OUTBOUND": CYAN, "ORBIT": AMBER, "RETURN": GREEN, "DESCEND": GREEN,
               "LANDED": WHITE}


def rgb255(c) -> tuple[int, int, int]:
    return tuple(int(255 * v) for v in c[:3])  # type: ignore[return-value]


def decimate(pts, max_n: int) -> np.ndarray:
    a = np.asarray(pts, float)
    if len(a) > max_n:
        idx = np.unique(np.r_[np.linspace(0, len(a) - 1, max_n).astype(int), len(a) - 1])
        a = a[idx]
    return a


def blend(dst: np.ndarray, rgba, x: int, y: int) -> None:
    """render.blend_rgba in integer math (the float32 version was ~8 ms per frame for the dashboard's cards)."""
    rgba = np.asarray(rgba)
    H, W = dst.shape[:2]
    h, w = rgba.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    patch = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    a = cv2.cvtColor(np.ascontiguousarray(patch[..., 3]), cv2.COLOR_GRAY2RGB)
    roi = dst[y0:y1, x0:x1]
    roi[:] = cv2.add(cv2.multiply(roi, cv2.bitwise_not(a), scale=1.0 / 255.0),
                     cv2.multiply(np.ascontiguousarray(patch[..., :3]), a, scale=1.0 / 255.0))


def dash_segments(pts: np.ndarray, dash: float) -> list[tuple[np.ndarray, np.ndarray]]:
    pts = np.asarray(pts, float)
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(float(np.linalg.norm(b - a)) / dash))
        out += [(a + (b - a) * i / n, a + (b - a) * min(i + 1, n) / n) for i in range(0, n, 2)]
    return out


def dashed_stroke(g: Geoms, pts: np.ndarray, rgb, width: float, alpha: float, dash: float) -> None:
    """Screen-space dashed polyline (GL lines: one geom per dash, no capsules)."""
    pts = np.asarray(pts, float)
    for a, b in zip(pts[:-1], pts[1:]):
        L = float(np.linalg.norm(b - a))
        n = max(1, int(L / dash))
        for i in range(0, n, 2):
            g.line(a + (b - a) * i / n, a + (b - a) * min(i + 1, n) / n, rgb, width, alpha)


# ---------------------------------------------------------------------------------------- 3D renderer

class BaseSceneRenderer(SceneRenderer):
    """SceneRenderer without the depth readback and the painted sky: the base worlds have their own skybox and no
    weather fog, and reading the depth buffer back was a third of the frame. project() is inherited."""

    def render(self, data: mujoco.MjData, cam: mujoco.MjvCamera, draw, *_unused) -> np.ndarray:
        t0 = time.perf_counter()
        r = self.r
        r.update_scene(data, camera=cam, scene_option=self.opt)
        scn = r.scene
        scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        draw(scn)
        if r._gl_context:
            r._gl_context.make_current()
        mujoco.mjr_render(r._rect, scn, r._mjr_context)
        g0, g1 = scn.camera[0], scn.camera[1]
        width = g0.frustum_width or 0.5 * (g0.frustum_top - g0.frustum_bottom) * self.W / self.H
        self.gl_cam = (0.5 * (np.array(g0.pos) + np.array(g1.pos)), np.array(g0.forward), np.array(g0.up),
                       g0.frustum_center, width, g0.frustum_bottom, g0.frustum_top, g0.frustum_near)
        mujoco.mjr_readPixels(self._rgb, None, r._rect, r._mjr_context)
        frame = cv2.flip(self._rgb, 0)
        if frame.shape[1] != self.out_w or frame.shape[0] != self.out_h:
            frame = cv2.resize(frame, (self.out_w, self.out_h), interpolation=cv2.INTER_LINEAR)
        self.t_view, self.t_fog = time.perf_counter() - t0, 0.0
        return frame


# ---------------------------------------------------------------------------------------- onboard cameras

class OnboardCams:
    """Front FPV (gimbal-level, 15 deg down) and downward camera, rendered from free cameras at the drone's true pose
    (the base world compiles its own scene without the X2's camera mounts). Display only."""

    FPV_PITCH = math.radians(15.0)
    MOUNT_Z = 0.15  # m above the body origin (= the ground at rest): clear of the pad slab, which has no collision

    def __init__(self, model: mujoco.MjModel, w: int = 320, h: int = 240) -> None:
        self.model = model
        self.r = mujoco.Renderer(model, h, w)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[2] = self.opt.geomgroup[3] = 0  # the X2's own visual / collision geoms
        self.img = {"fpv": np.zeros((h, w, 3), np.uint8), "down": np.zeros((h, w, 3), np.uint8)}
        self.t_ms = 0.0

    def render(self, data: mujoco.MjData, kind: str, pos: np.ndarray, R: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        heading = math.atan2(R[1, 0], R[0, 0])
        if kind == "fpv":
            c, s = math.cos(heading), math.sin(heading)
            f = np.array([c * math.cos(self.FPV_PITCH), s * math.cos(self.FPV_PITCH), -math.sin(self.FPV_PITCH)])
            p = pos + np.array([0.3 * c, 0.3 * s, self.MOUNT_Z])
            az, el, fovy = math.degrees(heading), -15.0, 75.0
        else:
            f = np.array([0.0, 0.0, -1.0])
            p = pos + np.array([0.0, 0.0, self.MOUNT_Z])
            az, el, fovy = math.degrees(heading), -90.0, 70.0
        cam = self.cam
        cam.lookat[:] = p + f
        cam.distance, cam.azimuth, cam.elevation = 1.0, az, el
        g = self.model.vis.global_
        old, g.fovy = g.fovy, fovy
        self.r.update_scene(data, camera=cam, scene_option=self.opt)
        g.fovy = old
        scn = self.r.scene
        scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        self.img[kind] = self.r.render().copy()
        self.t_ms = 1000.0 * (time.perf_counter() - t0)
        return self.img[kind]

    def close(self) -> None:
        self.r.close()


# ---------------------------------------------------------------------------------------- static top-down map

class TopDownMap:
    """The world rendered once from straight above (north up), cropped and annotated every frame."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, half_m: float, px: int = 1536) -> None:
        self.half, self.n = half_m, px
        self.mpp = 2.0 * half_m / px
        D = 8000.0
        ext = float(model.stat.extent)
        model.vis.map.zfar = max(float(model.vis.map.zfar), 4.0 * D / ext)
        r = mujoco.Renderer(model, px, px)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.0, 0.0, 0.0]
        cam.distance, cam.azimuth, cam.elevation = D, 90.0, -90.0
        g = model.vis.global_
        old, g.fovy = g.fovy, math.degrees(2.0 * math.atan(half_m / D))
        opt = mujoco.MjvOption()
        opt.geomgroup[3] = 0
        r.update_scene(data, camera=cam, scene_option=opt)
        g.fovy = old
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self.img = (r.render().astype(np.float32) * 0.85).astype(np.uint8)
        r.close()

    def to_px(self, xy) -> tuple[float, float]:
        return (float(xy[0]) + self.half) / self.mpp, (self.half - float(xy[1])) / self.mpp

    def panel(self, sc: Any, view: "DashView", size: int) -> np.ndarray:
        est, tp = sc.est, sc.true_pos
        pts = [sc.pad, sc.mast, est.pos[:2], tp[:2]]
        tgt = view.route["target"] if view.route is not None else None
        if tgt is not None:
            r = sc.plan.orbit_r
            pts += [tgt[:2] + [r, r], tgt[:2] - [r, r]]
        pts = np.array(pts)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        center = 0.5 * (lo + hi)
        half = max(70.0, 0.5 * float(np.max(hi - lo)) * 1.12 + 25.0)
        cu, cv = self.to_px(center)
        hp = half / self.mpp
        k = size / (2 * hp)
        M = np.array([[k, 0, -(cu - hp) * k], [0, k, -(cv - hp) * k]])
        img = cv2.warpAffine(self.img, M, (size, size), flags=cv2.INTER_AREA, borderValue=(40, 44, 40))
        lw = max(1, size // 160)

        def px(xy) -> tuple[int, int]:
            u, v = self.to_px(xy)
            return int(round((u - (cu - hp)) * k)), int(round((v - (cv - hp)) * k))

        def m2px(m: float) -> int:
            return max(1, int(round(m / self.mpp * k)))

        def dashed(a, b, col, w, dash_px=8) -> None:
            pa, pb = np.array(px(a), float), np.array(px(b), float)
            L = float(np.linalg.norm(pb - pa))
            n = max(1, int(L / dash_px))
            for i in range(0, n, 2):
                q0 = pa + (pb - pa) * i / n
                q1 = pa + (pb - pa) * min(i + 1, n) / n
                cv2.line(img, tuple(int(x) for x in q0), tuple(int(x) for x in q1), col, w, cv2.LINE_AA)

        az = sc.bearing if sc.bearing is not None else sc.pending
        L = sc.plan.range_m + 150.0
        for dz, w, col in ((0.0, lw + 1, rgb255(BEAR3)),
                           (-sc.plan.bearing_sigma_deg, lw, (200, 110, 60)), (sc.plan.bearing_sigma_deg, lw, (200, 110, 60))):
            a = math.radians(az + dz)
            end = sc.mast + L * np.array([math.sin(a), math.cos(a)])
            if dz == 0.0 and sc.bearing is not None:
                cv2.line(img, px(sc.mast), px(end), col, w, cv2.LINE_AA)
            else:
                dashed(sc.mast, end, col, w, 6)
        route = view.route
        if route is not None:
            for key in ("outbound", "orbit", "return"):
                pl = route[key]
                for a, b in zip(pl[:-1], pl[1:]):
                    dashed(a, b, rgb255(IDEAL3), lw, 7)
            cv2.circle(img, px(route["target"]), m2px(4.0) + 3, rgb255(IDEAL3), lw, cv2.LINE_AA)
        tt, et = sc.truth_trail, sc.est_trail
        if len(tt) > 1:
            if sc.gnss_off_idx is not None:
                for i in range(sc.gnss_off_idx, len(tt), ERR_EVERY):
                    cv2.line(img, px(tt[i]), px(et[i]), (255, 70, 60), max(1, lw - 1), cv2.LINE_AA)
            for trail, col in ((tt, (255, 255, 255)), (et, rgb255(CYAN3))):
                p = np.array([px(q) for q in decimate(trail, 400)], np.int32)
                cv2.polylines(img, [p], False, col, lw + 1, cv2.LINE_AA)
        cv2.rectangle(img, px(sc.pad - 5), px(sc.pad + 5), rgb255(GREEN3), lw + 1, cv2.LINE_AA)
        cv2.drawMarker(img, px(sc.mast), (255, 230, 120), cv2.MARKER_DIAMOND, 6 * lw + 6, lw + 1)
        if sc.gnss_off_xy is not None:
            cv2.drawMarker(img, px(sc.gnss_off_xy), (255, 70, 60), cv2.MARKER_TILTED_CROSS, 6 * lw + 6, lw + 1)
        s2 = est.sigma2_horizontal()
        if np.linalg.norm(est.pos[:2] - tp[:2]) > 1.0:
            cv2.line(img, px(tp), px(est.pos), (255, 70, 60), lw + 1, cv2.LINE_AA)
        cv2.circle(img, px(est.pos), max(3, m2px(s2)), rgb255(CYAN3), lw, cv2.LINE_AA)
        cv2.circle(img, px(tp), 3 * lw + 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, px(est.pos), 2 * lw + 3, rgb255(CYAN3), -1, cv2.LINE_AA)
        bar_m = next(b for b in (20, 50, 100, 200, 500, 1000, 2000) if b / (2 * half) > 0.12)
        bar_px = m2px(bar_m)
        fs_ = size / 400
        cv2.line(img, (10, size - 14), (10 + bar_px, size - 14), (255, 255, 255), lw + 1)
        cv2.putText(img, f"{bar_m} m", (16 + bar_px, size - 9), cv2.FONT_HERSHEY_DUPLEX, 0.55 * fs_ * 1.2,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "N", (size - int(22 * fs_), int(56 * fs_)), cv2.FONT_HERSHEY_DUPLEX, 0.7 * fs_ * 1.2,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return img


# ---------------------------------------------------------------------------------------- 3D scene

class DashView:
    """Everything drawn into the 3D scene and the labels, for one scenario (`sc`, duck-typed)."""

    def __init__(self, sc: Any) -> None:
        self.sc = sc
        self.draw_scale = DRONE_SCALE
        self.route: dict[str, np.ndarray] | None = None
        self.route_dashes: list[tuple[np.ndarray, np.ndarray]] = []
        self._route_key: tuple | None = None

    def refresh(self, cam_dist: float) -> None:
        sc = self.sc
        key = (sc.bearing, sc.z0)
        if key != self._route_key:
            self._route_key = key
            self.route = None if sc.bearing is None else planned_route(sc.plan, sc.pad, sc.mast, sc.bearing, sc.z0)
            self.route_dashes = [] if self.route is None else [
                seg for key in ("takeoff", "outbound", "orbit", "return")
                for seg in dash_segments(self.route[key], 3.0 if key == "orbit" else 8.0)]
        self.draw_scale = float(np.clip(cam_dist / 4.0, DRONE_SCALE, 30.0))

    def draw_geoms(self, scn: mujoco.MjvScene) -> None:
        g = Geoms(scn)
        sc = self.sc
        est, z0 = sc.est, sc.z0
        epos = est.pos + np.array([0.0, 0.0, z0])  # display: onboard z (above the takeoff point) on the true datum
        tp, tR = sc.true_pos, sc.true_R
        # UMA-16 bearing ray (+- 1 sigma wedge): orange once launched, pending bearing faint
        mast = np.array([sc.mast[0], sc.mast[1], 3.2])
        az = sc.bearing if sc.bearing is not None else sc.pending
        L = sc.plan.range_m + 150.0
        u = np.array([math.sin(math.radians(az)), math.cos(math.radians(az)), 0.0])
        if sc.bearing is not None:
            g.capsule(mast, mast + L * u, 0.35, BEAR3, 0.85, emission=0.5)
        else:
            g.capsule(mast, mast + L * u, 0.3, (1.0, 0.9, 0.2), 0.55, emission=0.4)
        for s in (-1, 1):
            a = math.radians(az + s * sc.plan.bearing_sigma_deg)
            g.line(mast, mast + L * np.array([math.sin(a), math.cos(a), 0.0]), (1.0, 0.6, 0.2), 2.0, 0.6)
        # pad column (green) so the pad reads from far away
        g.cylinder([sc.pad[0], sc.pad[1], 15.0], 0.3, 15.0, GREEN3, 0.22, emission=0.6)
        # IDEAL plan
        if self.route is not None:
            r = self.route
            for a, b in self.route_dashes:
                g.line(a, b, IDEAL3, 3.0, 0.85)
            t = r["target"]
            g.sphere(t, 2.5, IDEAL3, 0.45, emission=0.4)
            g.cylinder([t[0], t[1], 0.5 * t[2]], 0.25, 0.5 * t[2], IDEAL3, 0.3, emission=0.4)
        # trails + error dashes
        tt, et = sc.truth_trail, sc.est_trail
        if len(tt) > 1:
            g.stroke(decimate(tt, 150), WHITE3, 4.0, 0.95)
            g.stroke(decimate(et, 150) + [0, 0, z0], CYAN3, 4.0, 0.9)
            if sc.gnss_off_idx is not None:
                for i in range(sc.gnss_off_idx, len(tt), ERR_EVERY_3D):
                    if np.linalg.norm(tt[i][:2] - et[i][:2]) > 1.0:
                        g.line(tt[i], et[i] + [0, 0, z0], RED3, 2.0, 0.7)
        # true drone + ghost with its 2-sigma bubble
        scale = self.draw_scale
        g.drone(tp, tR, scale, (0.93, 0.93, 0.96), (0.12, 0.12, 0.14), 1.0, nose_rgb=(1.0, 0.55, 0.1))
        g.drone(epos, est.R, scale, CYAN3, CYAN3, 0.38)
        cov = est.cov[:3, :3]
        evals, evecs = np.linalg.eigh(0.5 * (cov[:2, :2] + cov[:2, :2].T))
        radii = np.clip(2.0 * np.sqrt(np.maximum(evals, 0.0)), 0.05, 400.0)
        rz = float(np.clip(2.0 * math.sqrt(max(cov[2, 2], 0.0)), 0.05, 60.0))
        mat = np.eye(3)
        mat[:2, :2] = evecs
        if np.linalg.det(mat) < 0:
            mat[:, 0] *= -1
        g.ellipsoid(epos, [radii[0], radii[1], rz], mat, CYAN3, 0.13)
        s2 = est.sigma2_horizontal()
        g.line_ring([epos[0], epos[1], 0.25], float(s2), CYAN3, 0.85, width=4.0, n=32)
        g.line(epos, [epos[0], epos[1], 0.25], CYAN3, 2.0, 0.6)
        g.line(tp, [tp[0], tp[1], 0.25], WHITE3, 2.0, 0.5)
        if np.linalg.norm(epos - tp) > 1.0:
            dashed_stroke(g, np.array([tp, epos]), RED3, 4.0, 0.95, 1.5)

    def labels(self, project) -> list:
        sc = self.sc
        est, tp = sc.est, sc.true_pos
        epos = est.pos + np.array([0.0, 0.0, sc.z0])
        out = []
        pt, pe = project(tp + [0, 0, 2.0]), project(epos + [0, 0, 2.0])
        if pt is not None and pe is not None and math.dist(pt, pe) < 60:
            out.append((((pt[0] + pe[0]) // 2, (pt[1] + pe[1]) // 2), "TRUE = ESTIMATE", (255, 255, 255, 255)))
        else:
            if pt is not None:
                out.append((pt, "TRUE", (255, 255, 255, 255)))
            if pe is not None:
                out.append((pe, "ESTIMATE", (0, 217, 255, 255)))
        for xyz, text, col in ((np.array([sc.pad[0], sc.pad[1], 31.0]), "PAD", (70, 215, 120, 255)),
                               (np.array([sc.mast[0], sc.mast[1], 9.0]), "UMA-16", (255, 230, 120, 255))):
            p = project(xyz)
            if p is not None:
                out.append((p, text, col))
        if self.route is not None:
            p = project(self.route["target"] + [0, 0, 6.0])
            if p is not None:
                out.append((p, "TARGET (ideal)", (255, 200, 40, 255)))
        return out


# ---------------------------------------------------------------------------------------- HUD

class DashHud(Hud):
    COL_W = 400
    RIGHT_W = 380
    CAM_H = 285
    MAP = 280

    def pill(self, text: str, color, size: float = 22, alpha: float = 1.0, max_w: int | None = None) -> Image.Image | None:
        if alpha <= 0:
            return None
        probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        fnt = self.f(size, True)
        while max_w is not None and size > 12 and probe.textlength(text, font=fnt) + self.px(40) > max_w:
            size -= 1
            fnt = self.f(size, True)
        w, h = int(probe.textlength(text, font=fnt)) + self.px(40), self.px(size + 22)
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=h // 2, fill=(10, 14, 22, int(228 * alpha)),
                            outline=color[:3] + (int(255 * alpha),), width=self.px(3))
        d.text((w // 2, h // 2), text, font=fnt, fill=color[:3] + (int(255 * alpha),), anchor="mm")
        return im

    def title_card(self, st: dict) -> Image.Image:
        w, h = self.px(self.COL_W), self.px(58)
        im, d = self._card(w, h, CYAN)
        d.text((self.px(14), self.px(6)), "UMA-16 RECON", font=self.f(22, True), fill=TEXT)
        sub = f"Skydio X2 · {st['world']} · bearing from the base array · MuJoCo live"
        d.text((self.px(14), self.px(32)), sub, font=self._fit(sub, w - self.px(110), 12, False, 9), fill=MUTED)
        rec = "REC  " if st["recording"] else ""
        d.text((w - self.px(12), self.px(10)), f"{rec}t {st['t']:5.1f} s", font=self.f(15, True),
               fill=RED if st["recording"] else TEXT, anchor="ra")
        sp = "PAUSED" if st["paused"] else f"{st['speed']:g}x"
        d.text((w - self.px(12), self.px(34)), sp, font=self.f(13, True), fill=AMBER if st["paused"] else MUTED, anchor="ra")
        return im

    def base_card(self, st: dict) -> Image.Image:
        w, h = self.px(self.COL_W), self.px(176)
        im, d = self._card(w, h, GREEN)
        x0 = self.px(16)
        d.text((x0, self.px(9)), "BASE KNOWS", font=self.f(14, True), fill=GREEN)
        cw = self._chip_w("LINK: NONE", 12)
        self._chip(d, w - self.px(12) - cw, self.px(6), "LINK: NONE", RED, 12)
        y = self.px(34)
        d.text((x0, y), "UMA-16 BEARING", font=self.f(11, True), fill=MUTED)
        d.text((x0 + self.px(190), y), "LAUNCH COMMAND", font=self.f(11, True), fill=MUTED)
        b = st["bearing"]
        d.text((x0, y + self.px(14)), "—" if b is None else f"{b:05.1f}°", font=self.f(30, True),
               fill=TEXT if b is not None else MUTED)
        if b is not None:
            d.text((x0 + self.px(118), y + self.px(28)), f"±{st['sigma']:.0f}°", font=self.f(14, True), fill=MUTED)
        lt = st["launch_t"]
        d.text((x0 + self.px(190), y + self.px(16)), "not given" if lt is None else f"t {lt:.1f} s",
               font=self.f(20, True), fill=TEXT if lt is not None else MUTED)
        d.text((x0 + self.px(190), y + self.px(42)), st["source"], font=self._fit(st["source"], w - x0 - self.px(200), 12),
               fill=MUTED)
        ry = y + self.px(70)
        rows = [("Pending", st["pending_line"], AMBER if st["queued"] else TEXT)]
        for k, v, c in rows:
            d.text((x0, ry), k, font=self.f(13), fill=MUTED)
            d.text((x0 + self.px(80), ry), v, font=self._fit(v, w - x0 - self.px(96), 13, True, 10), fill=c)
            ry += self.px(19)
        note = "After launch the base hears nothing: the drone reports only once it is back."
        d.text((x0, ry + self.px(4)), note, font=self._fit(note, w - 2 * x0, 12, False, 9), fill=MUTED)
        return im

    def truth_card(self, st: dict) -> Image.Image:
        w, h = self.px(self.COL_W), self.px(262)
        im, d = self._card(w, h, VIOLET)
        x0 = self.px(16)
        d.text((x0, self.px(9)), "SIM TRUTH", font=self.f(14, True), fill=VIOLET)
        tag = "not available to the base"
        cw = self._chip_w(tag, 11)
        self._chip(d, w - self.px(12) - cw, self.px(6), tag, VIOLET, 11)
        y = self.px(32)
        state = st["state"]
        d.text((x0, y), "DRONE STATE", font=self.f(11, True), fill=MUTED)
        d.text((x0, y + self.px(13)), state, font=self.f(26, True), fill=STATE_COLOR.get(state, TEXT))
        gon = st["gnss_on"]
        txt = "GNSS ON" if gon else "GNSS OFF"
        cw = self._chip_w(txt, 15)
        self._chip(d, w - self.px(14) - cw, y + self.px(14), txt, GREEN if gon else RED, 15)
        y += self.px(50)
        col_w = (w - 2 * x0) // 2
        d.text((x0, y), "POSITION ERROR NOW", font=self.f(11, True), fill=MUTED)
        d.text((x0 + col_w, y), "CLAIMED 2σ", font=self.f(11, True), fill=MUTED)
        over = st["err"] > st["sigma2"]
        d.text((x0, y + self.px(13)), f"{st['err']:.1f} m", font=self.f(32, True), fill=RED if over else WHITE)
        d.text((x0 + col_w, y + self.px(13)), f"{st['sigma2']:.1f} m", font=self.f(32, True), fill=CYAN)
        y += self.px(56)
        tile_w = (w - 2 * x0 - 2 * self.px(8)) // 3
        for i, (k, v) in enumerate((("MAX ERROR", st["max_err"]), ("AT TARGET", st["err_target"]),
                                    ("LANDING ERROR", st["land_err"]))):
            tx = x0 + i * (tile_w + self.px(8))
            d.rounded_rectangle((tx, y, tx + tile_w, y + self.px(50)), radius=self.px(6), fill=mix(WHITE, 0.07))
            d.text((tx + self.px(8), y + self.px(5)), k, font=self.f(10, True), fill=MUTED)
            d.text((tx + self.px(8), y + self.px(20)), "—" if v is None else f"{v:.1f} m", font=self.f(20, True),
                   fill=TEXT if v is not None else MUTED)
        y += self.px(60)
        rows = [("Since GNSS off", st["since_off"], RED if not gon else MUTED),
                ("From the base", st["from_base"], TEXT),
                ("Drone believes", st["believes"], CYAN)]
        for k, v, c in rows:
            d.text((x0, y), k, font=self.f(13), fill=MUTED)
            d.text((x0 + self.px(118), y), v, font=self._fit(v, w - x0 - self.px(130), 13, True, 10), fill=c)
            y += self.px(19)
        return im

    def __init__(self, width: int, height: int) -> None:
        super().__init__(width, height)
        self._static: dict[tuple, Image.Image] = {}
        self._base_key: tuple | None = None
        self._base_im: Image.Image | None = None

    def _cached(self, key: tuple, make) -> Image.Image:
        if key not in self._static:
            self._static[key] = make()
        return self._static[key]

    def legend_card(self) -> Image.Image:
        w, h = self.px(self.RIGHT_W), self.px(150)
        im, d = self._card(w, h)
        items = [((255, 255, 255, 255), "TRUE drone + path (simulator)", False),
                 (CYAN, "ESTIMATE: where it thinks it is, 2σ", False),
                 ((255, 200, 40, 255), "IDEAL: planned route + orbit (dashed)", True),
                 ((255, 70, 60, 255), "error: estimate ↔ true (dashes)", True),
                 (ORANGE, "UMA-16 bearing, ±1σ wedge", False)]
        y = self.px(10)
        for c, text, dash in items:
            if dash:
                for k in range(3):
                    x = self.px(16 + 8 * k)
                    d.rectangle((x, y + self.px(7), x + self.px(5), y + self.px(11)), fill=c)
            else:
                d.rounded_rectangle((self.px(16), y + self.px(3), self.px(38), y + self.px(15)), radius=self.px(3), fill=c)
            d.text((self.px(48), y), text, font=self.f(13), fill=TEXT)
            y += self.px(26)
        return im

    def keys_card(self, w: int) -> Image.Image:
        h = self.px(30)
        im, d = self._card(w, h)
        txt = ("← → bearing   ENTER launch   F random   N GNSS off/on   V record   C camera   "
               "1 2 4 speed   SPACE pause   Z snap   Q quit")
        d.text((w // 2, h // 2), txt, font=self._fit(txt, w - self.px(16), 12, False, 8), fill=MUTED, anchor="mm")
        return im

    def panel_header(self, img: np.ndarray, title: str, tag: str, color) -> None:
        """Caption strip on a camera / map panel (in place)."""
        strip = self._cached(("header", img.shape[1], title, tag), lambda: self._header(img.shape[1], title, tag, color))
        blend(img, strip, 0, 0)

    def _header(self, W: int, title: str, tag: str, color) -> np.ndarray:
        hh = self.px(26)
        strip = Image.new("RGBA", (W, hh), (10, 14, 22, 215))
        d = ImageDraw.Draw(strip)
        d.text((self.px(8), hh // 2), title, font=self._fit(title, W - self.px(150), 13, True, 9), fill=TEXT, anchor="lm")
        tw = self._chip_w(tag, 10)
        self._chip(d, W - tw - self.px(6), self.px(3), tag, color, 10)
        return np.asarray(strip)

    def compose(self, frame: np.ndarray, st: dict, map_img: np.ndarray, fpv: np.ndarray, down: np.ndarray) -> np.ndarray:
        W, H, p = self.W, self.H, self.px
        m = p(16)
        for lab, x, y in self._layout_labels(st["labels"]):
            blend(frame, np.asarray(lab), x, y)
        title = self.title_card(st)
        blend(frame, np.asarray(title), m, m)
        y = m + title.height + p(8)
        bkey = ("base", st["bearing"], st["launch_t"], st["source"], st["pending_line"], st["queued"])
        if bkey != self._base_key:
            self._base_key, self._base_im = bkey, self.base_card(st)
        base = self._base_im
        blend(frame, np.asarray(base), m, y)
        y += base.height + p(8)
        blend(frame, np.asarray(self.truth_card(st)), m, y)
        side = p(self.MAP)
        paste(frame, map_img, m, H - m - side, side, side)
        rw, ch = p(self.RIGHT_W), p(self.CAM_H)
        rx = W - m - rw
        paste(frame, fpv, rx, m, rw, ch)
        paste(frame, down, rx, m + ch + p(8), rw, ch)
        legend = self._cached(("legend",), lambda: np.asarray(self.legend_card()))
        blend(frame, legend, rx, H - m - legend.shape[0])
        cx0, cx1 = m + p(self.COL_W) + p(10), rx - p(10)
        kx = m + side + p(10)
        keys = self._cached(("keys", rx - p(10) - kx), lambda: np.asarray(self.keys_card(rx - p(10) - kx)))
        blend(frame, keys, kx, H - m - keys.shape[0])
        link = self.pill("LINK: NONE  ·  no data link between base and drone", RED, 18, max_w=cx1 - cx0)
        if link is not None:
            blend(frame, np.asarray(link), (cx0 + cx1 - link.width) // 2, m)
            yb = m + link.height + p(8)
        else:
            yb = m
        if st["result"]:
            res = self.pill(st["result"][0], st["result"][1], 24, max_w=cx1 - cx0)
            blend(frame, np.asarray(res), (cx0 + cx1 - res.width) // 2, yb)
            yb += res.height + p(6)
        if st["banner"]:
            text, age, col = st["banner"]
            a = 1.0 if age < 3.0 else max(0.0, 1.0 - (age - 3.0))
            ban = self.pill(text, col, 20, a, max_w=cx1 - cx0)
            if ban is not None:
                blend(frame, np.asarray(ban), (cx0 + cx1 - ban.width) // 2, yb)
        return frame

    def _keep_labels_clear(self, placed: list) -> None:
        p = self.px
        m = p(16)
        col_r = m + p(self.COL_W)
        right_l = self.W - m - p(self.RIGHT_W)
        for item in placed:
            lab, x, y = item
            if x + lab.width > right_l:
                item[1] = float(right_l - lab.width - p(6))
            if x < col_r:
                item[1] = float(col_r + p(6))


# ---------------------------------------------------------------------------------------- one frame

class DashComposer:
    """Renders the full dashboard frame; the viewer and the recorder share it (the MP4 is the window picture)."""

    def __init__(self, sc: Any, width: int, height: int, map_half_m: float) -> None:
        self.W, self.H = width, height
        model = sc.model
        g = model.vis.global_
        g.offwidth, g.offheight = max(g.offwidth, width, 1536), max(g.offheight, height, 1536)
        model.vis.map.znear = min(float(model.vis.map.znear), 0.02 / float(model.stat.extent))  # cameras near the ground
        self.view = DashView(sc)
        self.map = TopDownMap(model, sc.data, map_half_m)
        self.sr = BaseSceneRenderer(model, width, height)
        # SceneRenderer brightens the headlight for the Berkeley ortho; the base world has its own sun
        model.vis.headlight.ambient[:] = [0.22, 0.22, 0.22]
        model.vis.headlight.diffuse[:] = [0.42, 0.42, 0.42]
        self.hud = DashHud(width, height)
        self.cams = OnboardCams(model)
        self.fovy = float(g.fovy)
        self.chase = self._new_chase(sc)
        self.overview = False
        self.ov_cam = mujoco.MjvCamera()
        self.ov_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam_every = 4  # frames between two renders of the same onboard camera (one cam every other frame)
        self.n = 0
        self.times = {"view": 0.0, "cams": 0.0, "hud": 0.0}

    @staticmethod
    def _new_chase(sc: Any) -> ChaseCam:
        return ChaseCam(150.0, np.array([sc.pad[0], sc.pad[1], 1.0]))

    def reset(self, sc: Any) -> None:
        old = self.chase
        self.view = DashView(sc)
        self.chase = self._new_chase(sc)
        self.chase.az_off, self.chase.el_off, self.chase.zoom = old.az_off, old.el_off, old.zoom

    def _aim(self) -> tuple | None:
        p = self.hud.px
        m = p(16)
        x0, x1 = m + p(DashHud.COL_W) + p(20), self.W - m - p(DashHud.RIGHT_W) - p(20)
        y0, y1 = m + p(110), self.H - m - p(60)
        if x1 - x0 < 80 or y1 - y0 < 80:
            return None
        return ((x0 + x1) * 0.5, (y0 + y1) * 0.5, self.W, self.H, self.fovy)

    def _overview(self, sc: Any) -> mujoco.MjvCamera:
        c = self.ov_cam
        az = sc.bearing if sc.bearing is not None else sc.pending
        u = np.array([math.sin(math.radians(az)), math.cos(math.radians(az))])
        mid = 0.5 * (sc.mast + u * sc.plan.range_m)
        c.lookat[:] = [mid[0], mid[1], 0.0]
        c.distance = (1.1 * sc.plan.range_m + 160.0) * self.chase.zoom
        c.azimuth = math.degrees(math.atan2(u[1], u[0])) + 35.0 + self.chase.az_off
        c.elevation = float(np.clip(-48.0 + self.chase.el_off, -89.0, -5.0))
        return c

    def frame(self, sc: Any, dt: float, speed: float, paused: bool, recording: bool) -> np.ndarray:
        tp, est = sc.true_pos, sc.est
        epos = est.pos + np.array([0.0, 0.0, sc.z0])
        cam = self.chase.update(dt, tp, sc.true_vel, epos, aim=self._aim())
        if self.overview:
            cam = self._overview(sc)
        self.view.refresh(cam.distance if not self.overview else 60.0)
        self.model_fovy_reset(sc)
        img = self.sr.render(sc.data, cam, self.view.draw_geoms)
        t1 = time.perf_counter()
        # onboard cameras: alternate, each at half the frame rate (a fresh render per frame of both costs ~2x)
        self.n += 1
        ph = self.n % self.cam_every
        kinds = ("fpv", "down") if self.n <= 2 else (("fpv",) if ph == 0 else ("down",) if ph == 2 else ())
        for k in kinds:
            self.cams.render(sc.data, k, tp, sc.true_R)
        t2 = time.perf_counter()
        hud = self.hud
        side, rw, ch = hud.px(DashHud.MAP), hud.px(DashHud.RIGHT_W), hud.px(DashHud.CAM_H)
        map_img = self.map.panel(sc, self.view, side)
        hud.panel_header(map_img, "MAP  (true + estimated + ideal)", "SIM TRUTH", VIOLET)
        fpv = cv2.resize(self.cams.img["fpv"], (rw, ch), interpolation=cv2.INTER_LINEAR)
        down = cv2.resize(self.cams.img["down"], (rw, ch), interpolation=cv2.INTER_LINEAR)
        self._cam_overlay(down, sc, crosshair=True)
        self._cam_overlay(fpv, sc, crosshair=False)
        hud.panel_header(fpv, "FRONT FPV CAMERA  (onboard)", "NOT SENT: NO LINK", VIOLET)
        hud.panel_header(down, "DOWNWARD CAMERA  (onboard)", "NOT SENT: NO LINK", VIOLET)
        st = sc.hud_state(speed, paused, recording)
        st["labels"] = self.view.labels(self.sr.project)
        out = hud.compose(img, st, map_img, fpv, down)
        self.times = {"view": self.sr.t_view, "cams": t2 - t1, "hud": time.perf_counter() - t2}
        return out

    def model_fovy_reset(self, sc: Any) -> None:
        sc.model.vis.global_.fovy = self.fovy

    def _cam_overlay(self, img: np.ndarray, sc: Any, crosshair: bool) -> None:
        h, w = img.shape[:2]
        fs_ = h / 300
        if crosshair:
            c = (w // 2, h // 2)
            cv2.drawMarker(img, c, (255, 230, 60), cv2.MARKER_CROSS, int(26 * fs_), 1, cv2.LINE_AA)
        hdg = (90.0 - math.degrees(math.atan2(sc.true_R[1, 0], sc.true_R[0, 0]))) % 360.0
        txt = f"alt {sc.true_pos[2] - sc.z0:4.0f} m   hdg {hdg:03.0f}"
        cv2.putText(img, txt, (int(8 * fs_), h - int(10 * fs_)), cv2.FONT_HERSHEY_DUPLEX, 0.5 * fs_ * 1.2,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (175, 125, 255), max(2, int(2 * fs_)))

    def close(self) -> None:
        self.sr.close()
        self.cams.close()
