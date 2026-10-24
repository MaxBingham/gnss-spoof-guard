"""Enemy-drone tracker dashboard (demos/52_track_enemy.py), built on the recon dashboard's components
(liveview/recon_dash.py: renderer, top-down map image, cards; liveview/chase.py, geoms.py, hud.py).

    3D      white X2 = our drone (TRUE), cyan ghost = its estimate; RED quad + red trail = the enemy (TRUE), orange
            trail = the enemy as our drone tracks it (onboard), green dots = the reports the base has RECEIVED;
            orange ray = UMA-16 bearing, amber = the assumed enemy point (SEARCH), yellow = the camera's view cone
    left    BASE (bearing, link up / down, the last report received + its snapshot)  |  SIM TRUTH (state, distances,
            errors, GNSS)
    right   gimbal camera with the detector's boxes (class, confidence), the map, legend
Harness code: reads the truth to DISPLAY only.
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
from .recon_dash import BaseSceneRenderer, TopDownMap, blend, decimate, rgb255
from .render import paste

CYAN3 = (0.0, 0.85, 1.0)
WHITE3 = (1.0, 1.0, 1.0)
RED3 = (0.95, 0.12, 0.08)
TRACK3 = (1.0, 0.55, 0.05)
RX3 = (0.2, 0.95, 0.35)
BEAR3 = (1.0, 0.35, 0.12)
AMBER3 = (1.0, 0.78, 0.15)
CONE3 = (1.0, 0.95, 0.3)
ACOU3 = (0.75, 0.45, 1.0)  # the acoustic bearing the drone received
LOST3 = (1.0, 0.2, 0.85)  # LOST: predicted point + search area
DRONE_SCALE = 9.0
STATE_COLOR = {"WAIT": MUTED, "TAKEOFF": CYAN, "SEARCH": AMBER, "DETECTED": ORANGE, "TRACKING": GREEN, "STRIKE": RED, "LOST": VIOLET,
               "RETURN": CYAN, "DESCEND": CYAN, "LANDED": WHITE}


class TrackView:
    """3D geoms + labels for one TrackScenario (`sc`)."""

    def __init__(self, sc: Any) -> None:
        self.sc = sc
        self.scale = DRONE_SCALE

    def draw_geoms(self, scn: mujoco.MjvScene) -> None:
        g = Geoms(scn)
        sc = self.sc
        z0 = sc.z0
        p = sc.pilot
        mast = np.array([sc.mast[0], sc.mast[1], 3.2])
        if sc.bearing is not None:
            u = np.array([math.sin(math.radians(sc.bearing)), math.cos(math.radians(sc.bearing)), 0.0])
            g.capsule(mast, mast + 650.0 * u, 0.35, BEAR3, 0.8, emission=0.5)
        g.cylinder([sc.pad[0], sc.pad[1], sc.ground(*sc.pad) + 15.0], 0.3, 15.0, (0.15, 0.85, 0.2), 0.22, emission=0.6)
        if p.state == "SEARCH" and p.bearing is not None:
            a = p.assumed_enemy(sc.t) + [0, 0, z0]
            g.sphere(a, 3.0, AMBER3, 0.45, emission=0.4)
        if p.brg is not None and not p.on_ground:  # the base's acoustic bearing (what the drone got last), +-1 sigma
            for k, wd, al in ((0.0, 3.0, 0.9), (-1.0, 1.5, 0.5), (1.0, 1.5, 0.5)):
                a = math.radians(p.brg[1] + k * p.brg[2])
                g.line(mast, mast + 650.0 * np.array([math.sin(a), math.cos(a), 0.0]), ACOU3, wd, al)
        area = p.lost_area(sc.t)
        if area is not None:  # LOST: the predicted enemy point and the search area
            c = area["center"] + [0, 0, z0]
            g.sphere(c, 2.5, LOST3, 0.6, emission=0.6)
            gz = float(sc.ground(c[0], c[1])) + 0.5
            if area["mode"] == "bearing":
                u = np.array([math.sin(math.radians(area["az"])), math.cos(math.radians(area["az"])), 0.0])
                nrm = np.array([-u[1], u[0], 0.0])
                L, Wd = area["along"], area["across"] / 2
                q = [c + u * L + nrm * Wd, c + u * L - nrm * Wd, c - u * L - nrm * Wd, c - u * L + nrm * Wd]
                for k in range(4):
                    g.line(q[k], q[(k + 1) % 4], LOST3, 3.0, 0.9)
                g.line(c - u * L, c + u * L, LOST3, 2.0, 0.7)
            else:
                g.line_ring([c[0], c[1], c[2]], float(area["radius"]), LOST3, 0.9, width=3.0, n=40)
            g.line(c, [c[0], c[1], gz], LOST3, 1.5, 0.6)
        # trails
        if len(sc.truth_trail) > 1:
            g.stroke(decimate(sc.truth_trail, 150), WHITE3, 3.0, 0.9)
        if len(sc.enemy_trail) > 1:
            g.stroke(decimate(sc.enemy_trail, 150), RED3, 4.0, 0.95)
        if len(sc.track_trail) > 1:
            g.stroke(decimate(sc.track_trail, 150), TRACK3, 3.0, 0.9)
        for r in sc.received[-40:]:
            g.sphere(r.pos + [0, 0, z0], 0.6 + 0.02 * self.scale, RX3, 0.8, emission=0.5)
        # our drone (true + estimate)
        tp, tR = sc.true_pos, sc.true_R
        epos = sc.est.pos + np.array([0.0, 0.0, z0])
        g.drone(tp, tR, self.scale, (0.93, 0.93, 0.96), (0.12, 0.12, 0.14), 1.0, nose_rgb=(1.0, 0.55, 0.1))
        g.drone(epos, sc.est.R, self.scale, CYAN3, CYAN3, 0.35)
        g.line(tp, [tp[0], tp[1], float(sc.ground(tp[0], tp[1])) + 0.3], WHITE3, 2.0, 0.5)
        # the camera's view cone (truth pose of the gimbal)
        if sc.cam.last_pose is not None and not p.on_ground:
            cpos, f = sc.cam.last_pose
            from sfhack.sim.gimbal_camera import cam_axes

            f, right, up = cam_axes(f)
            L = 45.0
            hx = math.tan(math.radians(sc.tcfg.hfov_deg) / 2) * L
            hy = hx * sc.tcfg.cam_h / sc.tcfg.cam_w
            corners = [cpos + L * f + sx * hx * right + sy * hy * up for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            for k, c in enumerate(corners):
                g.line(cpos, c, CONE3, 1.5, 0.55)
                g.line(c, corners[(k + 1) % 4], CONE3, 1.5, 0.55)
        # the enemy (true) and the onboard track of it
        if sc.enemy.active:
            e = sc.enemy.pose
            from sfhack.sim.enemy import enemy_attitude

            q = enemy_attitude(e)
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, q)
            g.drone(e.pos, R.reshape(3, 3), self.scale * 1.3, (0.9, 0.15, 0.1), (0.95, 0.45, 0.4), 0.95,
                    nose_rgb=(1.0, 0.9, 0.2))
            g.line(e.pos, [e.pos[0], e.pos[1], float(sc.ground(e.pos[0], e.pos[1])) + 0.3], RED3, 2.0, 0.5)
            if not p.on_ground:
                g.line(tp, e.pos, (1.0, 0.4, 0.4), 1.5, 0.5)
        trk = p.tracker.track
        if trk is not None and p.state in ("DETECTED", "TRACKING", "STRIKE", "LOST"):
            tpos = trk.pos + [0, 0, z0]
            s = float(np.clip(2.0 * trk.sigma_h(), 0.5, 80.0))
            g.sphere(tpos, max(s, 0.08 * self.scale), TRACK3, 0.25)
            g.sphere(tpos, 0.05 * self.scale + 0.3, TRACK3, 0.9, emission=0.6)

    def labels(self, project) -> list:
        sc, out = self.sc, []
        tp = sc.true_pos
        q = project(tp + [0, 0, 2.0])
        if q is not None:
            out.append((q, "OUR DRONE", (255, 255, 255, 255)))
        if sc.enemy.active:
            q = project(sc.enemy.pose.pos + [0, 0, 3.0])
            if q is not None:
                d = sc.true_dist()
                tag = "  · HIDDEN from the camera" if getattr(sc, "hidden", False) else ""
                out.append((q, f"ENEMY (true) {d:.0f} m{tag}", (255, 90, 80, 255)))
        trk = sc.pilot.tracker.track
        if trk is not None and sc.pilot.state in ("DETECTED", "TRACKING", "STRIKE", "LOST"):
            q = project(trk.pos + [0, 0, sc.z0 - 3.0])
            if q is not None:
                out.append((q, "TRACK (onboard)", (255, 160, 40, 255)))
        area = sc.pilot.lost_area(sc.t)
        if area is not None:
            q = project(area["center"] + [0, 0, sc.z0 + 4.0])
            if q is not None:
                out.append((q, f"PREDICTED ({'base bearing' if area['mode'] == 'bearing' else 'last track'})",
                            (255, 80, 220, 255)))
        if sc.received:
            q = project(sc.received[-1].pos + [0, 0, sc.z0 - 6.0])
            if q is not None:
                out.append((q, "BASE RECEIVED", (70, 230, 110, 255)))
        for xyz, text, col in ((np.array([sc.pad[0], sc.pad[1], sc.ground(*sc.pad) + 31.0]), "PAD", (70, 215, 120, 255)),
                               (np.array([sc.mast[0], sc.mast[1], 9.0]), "UMA-16", (255, 230, 120, 255))):
            q = project(xyz)
            if q is not None:
                out.append((q, text, col))
        return out


class TrackHud(Hud):
    COL_W = 400
    RIGHT_W = 420
    CAM_H = 315
    MAP = 300

    def __init__(self, width: int, height: int) -> None:
        super().__init__(width, height)
        self._cache: dict = {}

    def pill(self, text: str, color, size: float = 20, alpha: float = 1.0, max_w: int | None = None):
        if alpha <= 0:
            return None
        probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        fnt = self.f(size, True)
        while max_w is not None and size > 12 and probe.textlength(text, font=fnt) + self.px(40) > max_w:
            size -= 1
            fnt = self.f(size, True)
        w, h = int(probe.textlength(text, font=fnt)) + self.px(40), self.px(size + 20)
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=h // 2, fill=(10, 14, 22, int(228 * alpha)),
                            outline=color[:3] + (int(255 * alpha),), width=self.px(3))
        d.text((w // 2, h // 2), text, font=fnt, fill=color[:3] + (int(255 * alpha),), anchor="mm")
        return im

    def title(self, st: dict) -> Image.Image:
        w, h = self.px(self.COL_W), self.px(58)
        im, d = self._card(w, h, RED)
        d.text((self.px(14), self.px(6)), "ENEMY TRACKER", font=self.f(22, True), fill=TEXT)
        sub = f"UMA-16 bearing → X2 gimbal camera + YOLO → follow + report · {st['world']} · MuJoCo live"
        d.text((self.px(14), self.px(32)), sub, font=self._fit(sub, w - self.px(110), 12, False, 8), fill=MUTED)
        rec = "REC  " if st["recording"] else ""
        d.text((w - self.px(12), self.px(10)), f"{rec}t {st['t']:5.1f} s", font=self.f(15, True),
               fill=RED if st["recording"] else TEXT, anchor="ra")
        sp = "PAUSED" if st["paused"] else f"{st['speed']:g}x"
        d.text((w - self.px(12), self.px(34)), sp, font=self.f(13, True), fill=AMBER if st["paused"] else MUTED,
               anchor="ra")
        return im

    def base_card(self, st: dict) -> Image.Image:
        w, h = self.px(self.COL_W), self.px(292)
        im, d = self._card(w, h, GREEN)
        x0 = self.px(16)
        d.text((x0, self.px(9)), "BASE", font=self.f(14, True), fill=GREEN)
        up = st["link_up"]
        txt = f"LINK {'UP' if up else 'DOWN'}{st['link_tag']}"
        cw = self._chip_w(txt, 12)
        self._chip(d, w - self.px(12) - cw, self.px(6), txt, GREEN if up else RED, 12)
        y = self.px(34)
        d.text((x0, y), "UMA-16 BEARING", font=self.f(11, True), fill=MUTED)
        d.text((x0 + self.px(190), y), "PENDING (← → ENTER)", font=self.f(11, True), fill=MUTED)
        b = st["bearing"]
        d.text((x0, y + self.px(14)), "—" if b is None else f"{b:05.1f}°", font=self.f(28, True),
               fill=TEXT if b is not None else MUTED)
        d.text((x0 + self.px(190), y + self.px(16)), f"{st['pending']:05.1f}°", font=self.f(20, True), fill=MUTED)
        y += self.px(54)
        d.text((x0, y), f"Reports received {st['rx']}  ·  buffered on the drone {st['buffered']}  ·  "
                        f"link up {st['avail']:.0f} %", font=self._fit("x" * 60, w - 2 * x0, 12, False, 9), fill=MUTED)
        y += self.px(22)
        d.text((x0, y), "LAST REPORT RECEIVED", font=self.f(11, True), fill=MUTED)
        r = st["last_rx"]
        snap_s = self.px(96)
        sx = w - self.px(14) - snap_s
        if r is None:
            d.text((x0, y + self.px(18)), "none yet", font=self.f(18, True), fill=MUTED)
        else:
            col = GREEN if st["rx_age"] < 3.0 else AMBER if st["rx_age"] < 10.0 else RED
            d.text((x0, y + self.px(14)), f"{st['rx_age']:.1f} s old  ({r.state})", font=self.f(16, True), fill=col)
            rows = [("Map point", f"E {r.pos[0]:+.0f}  N {r.pos[1]:+.0f} m"),
                    ("From base", f"{r.bearing_from():05.1f}°  {np.linalg.norm(r.pos[:2]):.0f} m"),
                    ("Altitude", f"{r.agl:.0f} m AGL"),
                    ("Speed / hdg", f"{r.speed:.1f} m/s  {r.heading_deg:03.0f}°"),
                    ("Confidence", f"{r.confidence:.2f}   ±{r.sigma_m:.1f} m")]
            yy = y + self.px(38)
            for k, v in rows:
                d.text((x0, yy), k, font=self.f(12), fill=MUTED)
                d.text((x0 + self.px(92), yy), v, font=self._fit(v, sx - x0 - self.px(100), 13, True, 9), fill=TEXT)
                yy += self.px(18)
            if r.snapshot is not None:
                snap = Image.fromarray(cv2.resize(r.snapshot, (snap_s, snap_s), interpolation=cv2.INTER_NEAREST))
                im.paste(snap, (sx, y + self.px(18)))
                d.rectangle((sx, y + self.px(18), sx + snap_s - 1, y + self.px(18) + snap_s - 1), outline=GREEN, width=2)
                d.text((sx + snap_s // 2, y + self.px(20) + snap_s), "snapshot", font=self.f(10), fill=MUTED,
                       anchor="ma")
        return im

    def truth_card(self, st: dict) -> Image.Image:
        w, h = self.px(self.COL_W), self.px(234)
        im, d = self._card(w, h, VIOLET)
        x0 = self.px(16)
        d.text((x0, self.px(9)), "SIM TRUTH", font=self.f(14, True), fill=VIOLET)
        tag = "not available to the base"
        cw = self._chip_w(tag, 11)
        self._chip(d, w - self.px(12) - cw, self.px(6), tag, VIOLET, 11)
        y = self.px(32)
        d.text((x0, y), "TRACKER STATE", font=self.f(11, True), fill=MUTED)
        d.text((x0, y + self.px(13)), st["state"], font=self.f(26, True), fill=STATE_COLOR.get(st["state"], TEXT))
        txt = "GNSS ON" if st["gnss_on"] else "GNSS OFF"
        cw = self._chip_w(txt, 14)
        self._chip(d, w - self.px(14) - cw, y + self.px(14), txt, GREEN if st["gnss_on"] else RED, 14)
        y += self.px(50)
        col_w = (w - 2 * x0) // 2
        d.text((x0, y), "DISTANCE TO ENEMY", font=self.f(11, True), fill=MUTED)
        d.text((x0 + col_w, y), "TRACK ERROR NOW", font=self.f(11, True), fill=MUTED)
        dist = st["dist"]
        dcol = RED if dist is not None and dist < 10.0 else WHITE
        d.text((x0, y + self.px(13)), "—" if dist is None else f"{dist:.0f} m", font=self.f(28, True), fill=dcol)
        te = st["track_err"]
        d.text((x0 + col_w, y + self.px(13)), "—" if te is None else f"{te:.1f} m", font=self.f(28, True), fill=ORANGE)
        y += self.px(50)
        rows = [("Base bearing", st["acoustic"], (200, 150, 255, 255)),
                ("Report error", st["rx_err"], TEXT), ("Closest so far", st["closest"], TEXT),
                ("Own pos error", st["own_err"], CYAN), ("Detector", st["det"], MUTED)]
        for k, v, c in rows:
            d.text((x0, y), k, font=self.f(13), fill=MUTED)
            d.text((x0 + self.px(118), y), v, font=self._fit(v, w - x0 - self.px(130), 13, True, 9), fill=c)
            y += self.px(19)
        return im

    def legend(self) -> Image.Image:
        w, h = self.px(self.RIGHT_W), self.px(180)
        im, d = self._card(w, h)
        items = [((255, 255, 255, 255), "our drone (true) · cyan ghost = its estimate"),
                 ((255, 70, 60, 255), "ENEMY (true) + its path"),
                 ((255, 150, 20, 255), "enemy as our drone tracks it (onboard)"),
                 ((60, 240, 100, 255), "reports the base has RECEIVED"),
                 (ORANGE, "UMA-16 bearing · amber = assumed enemy point"),
                 ((190, 115, 255, 255), "acoustic bearing received ±1σ"),
                 ((255, 50, 215, 255), "LOST: predicted point + search area")]
        y = self.px(9)
        for c, text in items:
            d.rounded_rectangle((self.px(16), y + self.px(3), self.px(38), y + self.px(14)), radius=self.px(3), fill=c)
            d.text((self.px(48), y), text, font=self.f(12), fill=TEXT)
            y += self.px(24)
        return im

    def keys(self, w: int) -> Image.Image:
        h = self.px(30)
        im, d = self._card(w, h)
        txt = ("← → bearing  ENTER send  F random  N GNSS  L link  R recall  V record  C camera  1 2 4 speed  "
               "SPACE pause  Z snap  Q quit")
        d.text((w // 2, h // 2), txt, font=self._fit(txt, w - self.px(16), 12, False, 7), fill=MUTED, anchor="mm")
        return im

    def header(self, img: np.ndarray, title: str, tag: str, color) -> None:
        key = ("hdr", img.shape[1], title, tag)
        if key not in self._cache:
            W = img.shape[1]
            hh = self.px(26)
            strip = Image.new("RGBA", (W, hh), (10, 14, 22, 215))
            d = ImageDraw.Draw(strip)
            d.text((self.px(8), hh // 2), title, font=self._fit(title, W - self.px(150), 13, True, 9), fill=TEXT,
                   anchor="lm")
            tw = self._chip_w(tag, 10)
            self._chip(d, W - tw - self.px(6), self.px(3), tag, color, 10)
            self._cache[key] = np.asarray(strip)
        blend(img, self._cache[key], 0, 0)

    def _keep_labels_clear(self, placed: list) -> None:
        p = self.px
        m = p(16)
        col_r = m + p(self.COL_W)
        right_l = self.W - m - p(self.RIGHT_W)
        for item in placed:
            lab, x, _y = item
            if x + lab.width > right_l:
                item[1] = float(right_l - lab.width - p(6))
            if x < col_r:
                item[1] = float(col_r + p(6))

    def compose(self, frame: np.ndarray, st: dict, cam: np.ndarray, map_img: np.ndarray) -> np.ndarray:
        W, H, p = self.W, self.H, self.px
        m = p(16)
        for lab, x, y in self._layout_labels(st["labels"]):
            blend(frame, np.asarray(lab), x, y)
        t = self.title(st)
        blend(frame, np.asarray(t), m, m)
        y = m + t.height + p(8)
        b = self.base_card(st)
        blend(frame, np.asarray(b), m, y)
        y += b.height + p(8)
        blend(frame, np.asarray(self.truth_card(st)), m, y)
        rw, ch = p(self.RIGHT_W), p(self.CAM_H)
        rx = W - m - rw
        paste(frame, cam, rx, m, rw, ch)
        ms = p(self.MAP)
        paste(frame, map_img, rx, m + ch + p(8), rw, ms)
        if "legend" not in self._cache:
            self._cache["legend"] = np.asarray(self.legend())
        leg = self._cache["legend"]
        if m + ch + p(8) + ms + p(8) + leg.shape[0] <= H - m:
            blend(frame, leg, rx, H - m - leg.shape[0])
        cx0, cx1 = m + p(self.COL_W) + p(10), rx - p(10)
        kk = ("keys", cx1 - cx0)
        if kk not in self._cache:
            self._cache[kk] = np.asarray(self.keys(cx1 - cx0))
        blend(frame, self._cache[kk], cx0, H - m - self._cache[kk].shape[0])
        yb = m
        link = self.pill(f"LINK {'UP' if st['link_up'] else 'DOWN: reports buffered on the drone'}",
                         GREEN if st["link_up"] else RED, 18, max_w=cx1 - cx0)
        blend(frame, np.asarray(link), (cx0 + cx1 - link.width) // 2, yb)
        yb += link.height + p(6)
        if st["banner"]:
            text, age, col = st["banner"]
            a = 1.0 if age < 3.0 else max(0.0, 1.0 - (age - 3.0))
            ban = self.pill(text, col, 20, a, max_w=cx1 - cx0)
            if ban is not None:
                blend(frame, np.asarray(ban), (cx0 + cx1 - ban.width) // 2, yb)
        return frame


class TrackComposer:
    """Full dashboard frame (viewer and recorder share it)."""

    MODES = ("chase", "overview", "enemy")

    def __init__(self, sc: Any, width: int, height: int) -> None:
        self.W, self.H = width, height
        model = sc.model
        g = model.vis.global_
        g.offwidth, g.offheight = max(g.offwidth, width, 1536), max(g.offheight, height, 1536)
        self.view = TrackView(sc)
        self.map = TopDownMap(model, sc.data, 900.0)
        hl = model.vis.headlight
        amb, dif = hl.ambient.copy(), hl.diffuse.copy()
        self.sr = BaseSceneRenderer(model, width, height)
        self.sr.opt.geomgroup[1] = 1  # the enemy and the birds
        hl.ambient[:], hl.diffuse[:] = amb, dif
        self.sr.fog = sc.world.name == "farm"
        self.hud = TrackHud(width, height)
        self.fovy = float(g.fovy)
        self.chase = ChaseCam(150.0, np.array([sc.pad[0], sc.pad[1], 1.0]))
        self.ov = mujoco.MjvCamera()
        self.ov.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.mode = "chase"
        self.pending = 0.0
        self.times = {"view": 0.0, "hud": 0.0}

    def cycle_mode(self) -> None:
        self.mode = self.MODES[(self.MODES.index(self.mode) + 1) % len(self.MODES)]

    def _camera(self, sc: Any, dt: float) -> mujoco.MjvCamera:
        tp = sc.true_pos
        e = sc.enemy.pose.pos if sc.enemy.active else None
        if self.mode == "overview":
            c = self.ov
            az = sc.bearing if sc.bearing is not None else self.pending
            u = np.array([math.sin(math.radians(az)), math.cos(math.radians(az))])
            mid = 0.5 * (sc.mast + u * 450.0)
            c.lookat[:] = [mid[0], mid[1], 0.0]
            c.distance = 700.0 * self.chase.zoom
            c.azimuth = math.degrees(math.atan2(u[1], u[0])) + 35.0 + self.chase.az_off
            c.elevation = float(np.clip(-45.0 + self.chase.el_off, -89.0, -5.0))
            return c
        if self.mode == "enemy" and e is not None:
            return self.chase.update(dt, e, sc.enemy.pose.vel, e, aim=self._aim())
        other = tp if e is None or sc.pilot.on_ground or np.linalg.norm(e - tp) > 120.0 else e
        return self.chase.update(dt, tp, sc.true_vel, other, aim=self._aim())

    def _aim(self) -> tuple | None:
        p = self.hud.px
        m = p(16)
        x0, x1 = m + p(TrackHud.COL_W) + p(20), self.W - m - p(TrackHud.RIGHT_W) - p(20)
        y0, y1 = m + p(110), self.H - m - p(60)
        if x1 - x0 < 80 or y1 - y0 < 80:
            return None
        return ((x0 + x1) * 0.5, (y0 + y1) * 0.5, self.W, self.H, self.fovy)

    def frame(self, sc: Any, dt: float, speed: float, paused: bool, recording: bool) -> np.ndarray:
        cam = self._camera(sc, dt)
        self.view.scale = float(np.clip(cam.distance / 4.0, DRONE_SCALE, 30.0))
        sc.model.vis.global_.fovy = self.fovy
        img = self.sr.render(sc.data, cam, self.view.draw_geoms)
        t1 = time.perf_counter()
        hud = self.hud
        rw, ch, ms = hud.px(TrackHud.RIGHT_W), hud.px(TrackHud.CAM_H), hud.px(TrackHud.MAP)
        camimg = self._cam_panel(sc, rw, ch)
        mp = self._map_panel(sc, ms)
        st = self._state(sc, speed, paused, recording)
        st["labels"] = self.view.labels(self.sr.project)
        out = hud.compose(img, st, camimg, mp)
        self.times = {"view": self.sr.t_view, "hud": time.perf_counter() - t1}
        return out

    # ------------------------------------------------------------------ panels

    def _cam_panel(self, sc: Any, w: int, h: int) -> np.ndarray:
        p = sc.pilot
        fr = p.last_frame if p.last_frame is not None else sc.cam.last
        if fr is None:
            img = np.zeros((h, w, 3), np.uint8)
        else:
            img = cv2.resize(fr.image, (w, h), interpolation=cv2.INTER_AREA)
            kx, ky = w / fr.image.shape[1], h / fr.image.shape[0]
            res = p.last_result
            fs = h / 300
            if res is not None and abs(res.t - fr.t) < 1e-6:
                hit_box = None if res.hit is None else res.hit.det.box
                for d in res.dets:
                    x0, y0, x1, y1 = d.box
                    pad = 4
                    a = (int(x0 * kx) - pad, int(y0 * ky) - pad)
                    b = (int(x1 * kx) + pad, int(y1 * ky) + pad)
                    strong = d.cls == "bird" or d.conf >= sc.tcfg.drone_conf
                    col = (255, 60, 50) if d.cls == "drone" else (80, 170, 255)
                    is_hit = hit_box is not None and d.box == hit_box
                    cv2.rectangle(img, a, b, col, 3 if is_hit else (2 if strong else 1), cv2.LINE_AA)
                    lab = f"{'ENEMY' if is_hit else d.cls} {d.conf:.2f}"
                    if is_hit:
                        lab += f"  ~{res.hit.range_m:.0f} m"
                    cv2.putText(img, lab, (a[0], max(a[1] - 5, 12)), cv2.FONT_HERSHEY_DUPLEX, 0.45 * fs, col, 1,
                                cv2.LINE_AA)
            cv2.drawMarker(img, (w // 2, h // 2), (255, 230, 60), cv2.MARKER_CROSS, int(22 * fs), 1, cv2.LINE_AA)
            txt = f"pan {fr.pan_deg:+5.0f}  tilt {fr.tilt_deg:+4.0f}   {p.state}   YOLO {p.detect_ms:4.0f} ms"
            cv2.putText(img, txt, (int(8 * fs), h - int(10 * fs)), cv2.FONT_HERSHEY_DUPLEX, 0.45 * fs,
                        (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (175, 125, 255), 2)
        tag = "SNAPSHOTS SENT" if sc.link.is_up else "BUFFERED: LINK DOWN"
        self.hud.header(img, "GIMBAL CAMERA  (onboard)  ·  YOLO drone / bird", tag, GREEN if sc.link.is_up else RED)
        return img

    def _map_panel(self, sc: Any, size: int) -> np.ndarray:
        tm = self.map
        pts = [sc.mast, sc.pad, sc.true_pos[:2]]
        if sc.enemy.active:
            pts.append(sc.enemy.pose.pos[:2])
        if sc.received:
            pts.append(sc.received[-1].pos[:2])
        pts = np.array(pts)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        center = 0.5 * (lo + hi)
        half = max(80.0, 0.5 * float(np.max(hi - lo)) * 1.15 + 30.0)
        cu, cv = tm.to_px(center)
        hp = half / tm.mpp
        k = size / (2 * hp)
        M = np.array([[k, 0, -(cu - hp) * k], [0, k, -(cv - hp) * k]])
        img = cv2.warpAffine(tm.img, M, (size, size), flags=cv2.INTER_AREA, borderValue=(40, 44, 40))
        lw = max(1, size // 160)

        def px(xy) -> tuple[int, int]:
            u, v = tm.to_px(xy)
            return int(round((u - (cu - hp)) * k)), int(round((v - (cv - hp)) * k))

        def m2px(mm: float) -> int:
            return max(1, int(round(mm / tm.mpp * k)))

        if sc.bearing is not None:
            a = math.radians(sc.bearing)
            cv2.line(img, px(sc.mast), px(sc.mast + 650 * np.array([math.sin(a), math.cos(a)])), rgb255(BEAR3), lw,
                     cv2.LINE_AA)
        if sc.pilot.brg is not None and not sc.pilot.on_ground:
            for k in (-1.0, 1.0):
                a = math.radians(sc.pilot.brg[1] + k * sc.pilot.brg[2])
                cv2.line(img, px(sc.mast), px(sc.mast + 650 * np.array([math.sin(a), math.cos(a)])), rgb255(ACOU3), 1,
                         cv2.LINE_AA)
        area = sc.pilot.lost_area(sc.t)
        if area is not None:
            c = area["center"][:2]
            if area["mode"] == "bearing":
                u = np.array([math.sin(math.radians(area["az"])), math.cos(math.radians(area["az"]))])
                nrm = np.array([-u[1], u[0]])
                L, Wd = area["along"], area["across"] / 2
                q = np.array([px(c + u * L + nrm * Wd), px(c + u * L - nrm * Wd), px(c - u * L - nrm * Wd),
                              px(c - u * L + nrm * Wd)], np.int32)
                cv2.polylines(img, [q], True, rgb255(LOST3), lw + 1, cv2.LINE_AA)
            else:
                cv2.circle(img, px(c), m2px(area["radius"]), rgb255(LOST3), lw + 1, cv2.LINE_AA)
            cv2.drawMarker(img, px(c), rgb255(LOST3), cv2.MARKER_TILTED_CROSS, 6 * lw + 6, lw + 1)
        if sc.pilot.state == "SEARCH" and sc.pilot.bearing is not None:
            cv2.drawMarker(img, px(sc.pilot.assumed_enemy(sc.t)[:2]), rgb255(AMBER3), cv2.MARKER_TILTED_CROSS,
                           6 * lw + 6, lw + 1)
        for trail, col, wd in ((sc.truth_trail, (255, 255, 255), lw), (sc.enemy_trail, (255, 60, 50), lw + 1),
                               (sc.track_trail, rgb255(TRACK3), lw)):
            if len(trail) > 1:
                q = np.array([px(x) for x in decimate(trail, 400)], np.int32)
                cv2.polylines(img, [q], False, col, wd, cv2.LINE_AA)
        for r in sc.received[-120:]:
            cv2.circle(img, px(r.pos[:2]), lw + 1, rgb255(RX3), -1, cv2.LINE_AA)
        if sc.received:
            r = sc.received[-1]
            cv2.circle(img, px(r.pos[:2]), max(4, m2px(2 * r.sigma_m)), rgb255(RX3), lw, cv2.LINE_AA)
        cv2.rectangle(img, px(sc.pad - 5), px(sc.pad + 5), (40, 215, 50), lw + 1, cv2.LINE_AA)
        cv2.drawMarker(img, px(sc.mast), (255, 230, 120), cv2.MARKER_DIAMOND, 6 * lw + 6, lw + 1)
        cv2.circle(img, px(sc.true_pos[:2]), 2 * lw + 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, px(sc.est.pos[:2]), 2 * lw + 2, rgb255(CYAN3), lw, cv2.LINE_AA)
        if sc.enemy.active:
            cv2.circle(img, px(sc.enemy.pose.pos[:2]), 2 * lw + 3, (255, 50, 40), -1, cv2.LINE_AA)
        bar = next(b for b in (20, 50, 100, 200, 500, 1000) if b / (2 * half) > 0.12)
        cv2.line(img, (10, size - 14), (10 + m2px(bar), size - 14), (255, 255, 255), lw + 1)
        cv2.putText(img, f"{bar} m", (16 + m2px(bar), size - 9), cv2.FONT_HERSHEY_DUPLEX, 0.45 * size / 300,
                    (255, 255, 255), 1, cv2.LINE_AA)
        self.hud.header(img, "MAP  (true enemy · track · received)", "SIM TRUTH", VIOLET)
        return img

    # ------------------------------------------------------------------ state

    def _state(self, sc: Any, speed: float, paused: bool, recording: bool) -> dict:
        p = sc.pilot
        trk = p.tracker.track
        te = None
        if trk is not None and sc.enemy.active and p.state in ("DETECTED", "TRACKING", "STRIKE", "LOST"):
            te = float(np.linalg.norm(trk.pos + [0, 0, sc.z0] - sc.enemy.pose.pos))
        err = sc.report_errors()
        rx_err = "—" if not len(err) else f"{err[-1, 1]:.1f} m last · {np.mean(err[:, 1]):.1f} m mean ({len(err)})"
        last = sc.received[-1] if sc.received else None
        banner = None
        if sc.banner is not None:
            t0, text, col = sc.banner
            col = col or STATE_COLOR.get(text, AMBER)
            banner = (text if text not in STATE_COLOR else f"TRACKER: {text}", sc.t - t0, col)
        forced = sc.link.forced
        b = p.brg
        if b is None:
            acoustic = "none received" if sc.acoustic.last is None else "heard, not received (link)"
        else:
            acoustic = f"{b[1]:05.1f}° ±{b[2]:.0f}°  ({sc.t - b[0]:.0f} s old)"
            if p.suspect_since is not None:
                acoustic += "  · TRACK SUSPECT"
        return {
            "world": sc.world.name, "t": sc.t, "recording": recording, "paused": paused, "speed": speed,
            "bearing": sc.bearing, "pending": self.pending, "link_up": sc.link.is_up,
            "link_tag": "" if forced is None else " (forced)", "avail": 100 * sc.link.measured_availability,
            "rx": len(sc.received), "buffered": len(p.outbox), "last_rx": last,
            "rx_age": 0.0 if last is None else sc.t - last.t, "state": p.state + (" (calibrating)" if p.calibrating else ""),
            "gnss_on": sc.rig.gnss_on, "dist": sc.true_dist(), "track_err": te, "rx_err": rx_err,
            "closest": "—" if not math.isfinite(sc.min_dist) else f"{sc.min_dist:.1f} m (limit 10 m)",
            "own_err": f"{sc.own_err():.1f} m (estimate vs true)",
            "det": f"YOLO {p.detect_ms:.0f} ms · {p.n_frames} frames · hits {p.tracker.hits}/{len(p.tracker.window)}",
            "banner": banner, "acoustic": acoustic,
        }

    def close(self) -> None:
        self.sr.close()
