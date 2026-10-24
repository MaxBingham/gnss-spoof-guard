"""Recon viewer (demos/demo_recon.py): night chase view of the recon flight with its panels.

    3D      dark scene (the base lights glow when recon-world's base is in the world), pad, bearing line, search
            cone, planned pattern, estimated track (blue) + true track (white), finds (red pillars)
    right   thermal camera (white-hot) with the detector's boxes: vehicle / person (red = part of a confirmed find,
            amber = new), rejected warm spots (grey x); below it the rear NIR camera (lights with their ids)
    left    HUD: state, battery, GNSS, nav sources, 2-sigma, distance / bearing from the base, finds, decisions
    bottom  map: searched area, cone, pattern, tracks, finds (+ the true targets, harness view)
    REPORT  full-screen report card after landing: summary, snapshots, map

Harness code: it reads the truth to DISPLAY (true track, true targets, true error), never to fly.
"""

from __future__ import annotations

import math
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from sfhack.mission.recon import az_of, az_to_unit, cone_halfwidth
from sfhack.mission.report import draw_map
from sfhack.recon_runner import ReconFlight, true_targets

from .chase import ChaseCam
from .geoms import Geoms
from .hud import AMBER, CARD_BG, CYAN, GREEN, MUTED, RED, TEXT, WHITE, font
from .night import agc_8bit, apply_night_lighting, night_sky
from .render import SceneRenderer, blend_rgba, paste

BLUE = (0.35, 0.75, 1.0)
WHITE3 = (1.0, 1.0, 1.0)
RED3 = (1.0, 0.25, 0.2)
AMBER3 = (1.0, 0.7, 0.2)
GREEN3 = (0.3, 0.9, 0.45)
PHASE_COLOR = {"TAKEOFF": CYAN, "OUTBOUND": CYAN, "SEARCH": AMBER, "OBSERVE": RED, "RETURN": GREEN, "HOME": GREEN,
               "LAND": GREEN, "LANDED": WHITE, "REPORT": WHITE, "PREFLIGHT": MUTED}


class ReconView:
    """Holds the flight + display state (trails, rear camera) for one run."""

    def __init__(self, rf: ReconFlight) -> None:
        self.rf = rf
        self.fs = rf.fs
        self.m = rf.recon
        self.truth_trail: list[np.ndarray] = []
        self.est_trail: list[np.ndarray] = []
        self._next_trail = 0.0
        self.targets = true_targets(self.fs.mission_dir, rf.scene)
        self.pad = self.m.pad
        self.pad_z = float(self.fs.true_ground_z(*self.pad))
        self.rear = rf.rear  # recon-world's RearCameraSensor (shared with the light localizer when nav uses it)
        self.blink = None
        if self.rear is not None:
            from sfhack.sim.base_lights import BlinkAnimator

            self.blink = BlinkAnimator(self.fs.sim.model, self.rear.lights)  # the 3D lights blink their codes
        self.map_img: np.ndarray | None = None
        self._map_t = -1.0
        self.decisions_seen = 0

    @property
    def t(self) -> float:
        return self.rf.t

    def advance(self, n: int) -> None:
        for _ in range(n):
            if self.rf.done:
                return
            self.rf.step()
        if self.t >= self._next_trail:
            self._next_trail = self.t + 0.5
            self.truth_trail.append(self.fs.truth.state().pos.copy())
            self.est_trail.append(self.fs.autopilot.pose(report=False).pos.copy())

    def rear_frame(self) -> np.ndarray | None:
        last = getattr(self.rear, "last", None)
        return None if last is None else last.image

    def toggle_lights(self) -> str:
        if self.rear is None:
            return "no base lights in this world"
        self.rear.lights.enabled = not getattr(self.rear.lights, "enabled", True)
        return "on" if self.rear.lights.enabled else "OFF"

    def draw_geoms(self, scn: Any) -> None:
        g = Geoms(scn)
        m = self.m
        z0 = self.pad_z
        g.cylinder([self.pad[0], self.pad[1], z0 + 0.08], 2.5, 0.06, GREEN3, 0.95, emission=0.4)
        g.cylinder([self.pad[0], self.pad[1], z0 + 40.0], 0.5, 40.0, GREEN3, 0.18, emission=0.6)
        u = az_to_unit(m.az)
        n = np.array([u[1], -u[0]])
        far = m.plan.r_max if m.plan is not None else m.c.search_max_m
        zl = z0 + 2.0
        g.line([*self.pad, zl], [*(self.pad + u * far * 1.1), zl], AMBER3, 3.0, 0.9)
        for side in (-1, 1):
            pts = [np.array([*(self.pad + u * r + side * n * cone_halfwidth(r, m.sigma_az, m.c)), zl])
                   for r in np.linspace(0, far * 1.1, 12)]
            g.stroke(np.array(pts), (1.0, 0.5, 0.1), 2.0, 0.7)
        if m.plan is not None:
            zc = z0 + m.c.cruise_agl
            g.stroke(np.array([[p[0], p[1], zc] for p in m.plan.path]), (0.6, 0.6, 0.65), 1.5, 0.45)
        if len(self.truth_trail) > 1:
            g.stroke(np.array(self.truth_trail[-600:]), WHITE3, 3.0, 0.9)
        if len(self.est_trail) > 1:
            g.stroke(np.array(self.est_trail[-600:]), BLUE, 3.0, 0.9)
        est = self.fs.autopilot.pose(report=False)
        r2 = est.sigma2_horizontal()
        g.line_ring([est.pos[0], est.pos[1], est.pos[2]], r2, BLUE, 0.8, 2.0)
        g.drone(est.pos, est.R, 12.0, BLUE, BLUE, 0.45)
        tr = self.fs.truth.state()
        g.drone(tr.pos, tr.R if hasattr(tr, "R") else np.eye(3), 12.0, (0.85, 0.85, 0.9), (0.95, 0.95, 1.0), 1.0)
        for t_ in m.tracks:
            if not t_.verified and (t_.confirmed_t is None or t_.rejected):
                continue
            gz = float(m.ground_z(t_.xy))
            g.cylinder([t_.xy[0], t_.xy[1], gz + 25.0], 0.8, 25.0, RED3, 0.35, emission=0.8)
            g.ring([t_.xy[0], t_.xy[1], gz + 1.0], 6.0, 0.4, RED3, 0.9)


class ReconComposer:
    """Full frame: chase view + thermal (right top) + rear camera (right bottom) + HUD (left top) + map (left bottom)."""

    def __init__(self, view: ReconView, width: int, height: int) -> None:
        self.W, self.H = width, height
        self.s = height / 900.0
        model = view.fs.sim.model
        self.sr = SceneRenderer(model, width, height)
        apply_night_lighting(model)
        self.sr._sky = night_sky(self.sr.H, self.sr.W)
        self.fovy = float(model.vis.global_.fovy)
        self.chase = self._new_chase(view)
        self.times: dict[str, float] = {}

    def px(self, v: float) -> int:
        return int(round(v * self.s))

    @staticmethod
    def _new_chase(view: ReconView) -> ChaseCam:
        u = az_to_unit(view.m.az)
        return ChaseCam(math.degrees(math.atan2(u[1], u[0])), np.array([view.pad[0], view.pad[1], view.pad_z + 1.0]))

    def reset(self, view: ReconView) -> None:
        old = self.chase
        self.chase = self._new_chase(view)
        self.chase.az_off, self.chase.el_off, self.chase.zoom = old.az_off, old.el_off, old.zoom

    def close(self) -> None:
        self.sr.close()

    def _right_w(self) -> int:
        return int(0.36 * self.W)

    def frame(self, view: ReconView, dt: float, speed: int, paused: bool) -> np.ndarray:
        p = self.px
        m = p(16)
        rw = self._right_w()
        tr, est = view.fs.truth.state(), view.fs.autopilot.pose(report=False)
        aim = ((p(16) + p(380) + self.W - rw - m) / 2, (p(60) + self.H - p(330)) / 2, self.W, self.H, self.fovy)
        cam = self.chase.update(0.0 if paused else dt * speed, tr.pos, tr.vel, est.pos, aim=aim)  # sim time
        if view.blink is not None:
            view.blink.update(view.t)
        img = self.sr.render(view.fs.sim.data, cam, view.draw_geoms, None, (0, 0, 0), 0.0)
        t0 = time.perf_counter()
        th = int(rw * 512 / 640)
        paste(img, self.thermal_panel(view, rw, th), self.W - m - rw, m, rw, th)
        rh = min(int(rw * 0.42), self.H - 3 * m - th - p(26))
        if rh > p(60):
            paste(img, self.rear_panel(view, rw, rh), self.W - m - rw, 2 * m + th, rw, rh)
        side = p(330)
        paste(img, self.map_panel(view, side), m, self.H - m - side, side, side)
        blend_rgba(img, np.asarray(self.hud_card(view, speed, paused)), m, m)
        if view.m.phase == "REPORT" and view.m.report is not None:
            blend_rgba(img, np.asarray(self.report_card(view)), (self.W - p(1100)) // 2, p(70))
        self.times = {"view": self.sr.t_view, "panels": time.perf_counter() - t0}
        return img

    # ------------------------------------------------------------------ panels

    def thermal_panel(self, view: ReconView, w: int, h: int) -> np.ndarray:
        ts = view.fs.thermal_sensor
        last = getattr(ts, "last", None)
        if last is None:
            img = np.zeros((h, w, 3), np.uint8)
        else:
            img = cv2.cvtColor(cv2.resize(agc_8bit(last.image_K), (w, h), interpolation=cv2.INTER_LINEAR), cv2.COLOR_GRAY2RGB)
        sx, sy = w / 640.0, h / 512.0
        res = view.m.last_result
        fs_ = h / 480
        if res is not None and last is not None and abs(view.m.last_frame_t - last.t) < 0.5:
            for rj in res.rejected:
                if rj.reason == "border":
                    continue
                q = (int(rj.uv[0] * sx), int(rj.uv[1] * sy))
                cv2.drawMarker(img, q, (150, 150, 150), cv2.MARKER_TILTED_CROSS, int(10 * fs_), 1, cv2.LINE_AA)
            confirmed = [t_ for t_ in view.m.tracks if t_.confirmed_t is not None or t_.verified]
            pose = view.fs.autopilot.pose(report=False)
            for d in res.detections:
                xy = pose.pos[:2] + d.xy_rel
                hit = any(np.linalg.norm(t_.xy - xy) < view.m.c.assoc_radius_m for t_ in confirmed)
                col = (255, 60, 50) if hit else (255, 190, 40)
                x0, y0, bw, bh = d.bbox
                pad = 6
                cv2.rectangle(img, (int((x0 - pad) * sx), int((y0 - pad) * sy)),
                              (int((x0 + bw + pad) * sx), int((y0 + bh + pad) * sy)), col, max(2, int(2 * fs_)), cv2.LINE_AA)
                cv2.putText(img, f"{d.label} {d.conf:.2f} +{d.temp_K:.0f}K", (int((x0 - pad) * sx), int((y0 - pad) * sy) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45 * fs_, col, 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (120, 130, 150), 2)
        agl = view.fs.autopilot.pose(report=False).pos[2] - view.m.ground_z(view.fs.autopilot.pose(report=False).pos[:2])
        cv2.rectangle(img, (2, h - int(30 * fs_)), (w - 3, h - 3), (10, 14, 22), -1)
        cv2.putText(img, f"THERMAL (down)  AGL {agl:.0f} m  detector {view.m.det_ms / max(view.m.frames, 1):.0f} ms",
                    (8, h - int(10 * fs_)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * fs_, (0, 220, 255), 1, cv2.LINE_AA)
        return img

    def rear_panel(self, view: ReconView, w: int, h: int) -> np.ndarray:
        rimg = view.rear_frame()
        if rimg is None:
            img = np.full((h, w, 3), 12, np.uint8)
            cv2.putText(img, "REAR NIR camera: waiting for recon-world", (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (150, 160, 175), 1, cv2.LINE_AA)
        else:
            g = rimg if rimg.ndim == 3 else cv2.cvtColor(rimg.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            iw = min(w, int(h * g.shape[1] / g.shape[0]))  # keep the camera's aspect (letterbox)
            x0 = (w - iw) // 2
            img = np.full((h, w, 3), 8, np.uint8)
            img[:, x0:x0 + iw] = cv2.resize(g, (iw, h), interpolation=cv2.INTER_AREA)
            sx, sy = iw / g.shape[1], h / g.shape[0]
            for lt in getattr(view.rear, "last_truth", []) or []:  # truth labels (harness display), grey
                if isinstance(lt, dict) and lt.get("in_view") and not lt.get("occluded"):
                    q = (x0 + int(lt["u"] * sx), int(lt["v"] * sy))
                    cv2.putText(img, str(lt.get("id", "")), (q[0] + 6, q[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                                (140, 140, 150), 1, cv2.LINE_AA)
            ll = getattr(view.fs.autopilot, "light_localizer", None)  # recon-nav: lights it decoded, green
            for b in getattr(ll, "last_blobs", []) or []:
                try:
                    u, v, lid = b[0], b[1], b[2]
                    q = (x0 + int(u * sx), int(v * sy))
                    cv2.circle(img, q, 8, (0, 255, 120), 1, cv2.LINE_AA)
                    cv2.putText(img, str(lid), (q[0] + 9, q[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1)
                except (IndexError, TypeError):
                    break
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (120, 130, 150), 2)
        cv2.putText(img, "REAR NIR (base lights)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
        return img

    def map_panel(self, view: ReconView, size: int) -> np.ndarray:
        if view.map_img is None or view.t - view._map_t > 1.0 or view.map_img.shape[0] != size:
            view._map_t = view.t
            tt = np.array([p[:2] for p in view.truth_trail]) if view.truth_trail else None
            bgr = draw_map(view.m, size=size, truth_track=tt,
                           truth_targets=[(t["xy"], t["kind"]) for t in view.targets], title="MAP (truth in green)")
            tr = view.fs.truth.state()
            est = view.fs.autopilot.pose(report=False)
            view.map_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            view._map_drone = (tr.pos[:2].copy(), est.pos[:2].copy())
        return view.map_img

    def hud_card(self, view: ReconView, speed: int, paused: bool) -> Image.Image:
        p = self.px
        W, H = p(380), p(470)
        im = Image.new("RGBA", (W, H), CARD_BG)
        d = ImageDraw.Draw(im)
        m, fs = view.m, view.fs
        f1, f2, f3 = font(p(26), True), font(p(16)), font(p(14))
        y = p(10)
        d.text((p(14), y), m.phase, font=f1, fill=PHASE_COLOR.get(m.phase, TEXT))
        d.text((W - p(14), y + p(6)), f"{m.clock(view.t) if m.t_takeoff is not None else '23:00'}  x{speed}"
               + ("  PAUSED" if paused else ""), font=f2, fill=MUTED, anchor="ra")
        y += p(40)
        left = max(m.battery_left(view.t), 0.0)
        frac = left / m.c.battery_s
        d.text((p(14), y), f"battery {left / 60:4.1f} min", font=f2, fill=TEXT)
        d.rectangle([p(170), y + p(4), W - p(14), y + p(16)], outline=MUTED)
        d.rectangle([p(170), y + p(4), p(170) + int((W - p(184)) * frac), y + p(16)],
                    fill=GREEN if frac > 0.35 else AMBER if frac > 0.15 else RED)
        y += p(26)
        gnss = view.rf.gnss_off_t is None and view.rf.gnss_off_m is not None
        if view.rf.gnss_off_m is None:
            gnss = True
        d.text((p(14), y), "GNSS", font=f2, fill=TEXT)
        d.text((p(70), y), "ON" if gnss else "OFF (jammed)", font=f2, fill=GREEN if gnss else RED)
        nav = "GNSS stand-in" if view.rf.gnss_off_m is None else (view.rf.nav or "imu+compass+baro")
        d.text((W - p(14), y), f"nav: {nav}", font=f2, fill=CYAN, anchor="ra")
        y += p(24)
        est = fs.autopilot.pose(report=False)
        tr = fs.truth.state()
        srcs = ", ".join(sorted(fs.autopilot.sources_used)) or "-"
        d.text((p(14), y), f"fused: {srcs}", font=f3, fill=MUTED)
        y += p(22)
        s2 = est.sigma2_horizontal()
        err = float(np.linalg.norm(est.pos[:2] - tr.pos[:2]))
        d.text((p(14), y), f"2-sigma {s2:6.0f} m", font=f2, fill=TEXT)
        d.text((W - p(14), y), f"true error {err:5.0f} m", font=f2, fill=AMBER if err > 50 else GREEN, anchor="ra")
        y += p(24)
        rel = est.pos[:2] - m.pad
        d.text((p(14), y), f"from base {np.linalg.norm(rel):5.0f} m @ {az_of(rel):03.0f} deg", font=f2, fill=TEXT)
        y += p(24)
        d.text((p(14), y), f"bearing in {m.az:03.0f} +- {m.sigma_az:.0f} deg   searched {m.searched_area_km2():.2f} km2",
               font=f3, fill=MUTED)
        y += p(22)
        finds = [t_ for t_ in m.tracks if t_.verified]
        d.text((p(14), y), f"finds {len(finds)}", font=f2, fill=RED if finds else TEXT)
        if finds:
            f0 = finds[0]
            dd = f0.xy - m.pad
            d.text((p(90), y), f"{f0.label} #{f0.id}: {np.linalg.norm(dd):.0f} m @ {az_of(dd):03.0f} deg", font=f2, fill=RED)
        y += p(28)
        import textwrap

        lines: list[str] = []
        for tt, text in m.decision_log[-5:]:
            wrapped = textwrap.wrap(f"{tt:4.0f}s {text}", 50)
            lines += wrapped[:1] + ["       " + s for s in wrapped[1:2]]
        for line in lines[-9:]:
            d.text((p(14), y), line, font=f3, fill=MUTED)
            y += p(19)
        d.text((p(14), H - p(22)), "N GNSS off  L lights  SPACE pause  1/2/4/8 speed  R restart  Q quit", font=f3, fill=MUTED)
        return im

    def report_card(self, view: ReconView) -> Image.Image:
        p = self.px
        W, H = p(1100), p(700)
        im = Image.new("RGBA", (W, H), (8, 12, 20, 240))
        d = ImageDraw.Draw(im)
        rep = view.m.report or {}
        d.text((p(24), p(18)), "RECON REPORT (handed over on the pad)", font=font(p(28), True), fill=WHITE)
        d.text((p(24), p(60)), rep.get("summary", ""), font=font(p(22), True), fill=RED if rep.get("finds") else AMBER)
        y = p(100)
        d.text((p(24), y), f"takeoff {rep.get('takeoff_clock')}  landed {rep.get('landing_clock')}  flight "
                           f"{rep.get('flight_time_s', 0) / 60:.1f} min  searched {rep['searched']['area_km2']:.2f} km2 "
                           f"out to {rep['searched']['max_distance_m'] / 1000:.1f} km", font=font(p(16)), fill=TEXT)
        y += p(28)
        for f in rep.get("finds", [])[:4]:
            d.text((p(24), y), f"#{f['id']} {f['label']}: {f['distance_m']:.0f} m @ {f['bearing_deg']:03.0f} deg, {f['time']}, "
                               f"{f['size_m'][0]:.1f} x {f['size_m'][1]:.1f} m, +{f['anomaly_K']:.0f} K, conf {f['conf']:.2f}",
                   font=font(p(16)), fill=TEXT)
            y += p(22)
        if view.rf.score is not None:
            s = view.rf.score
            d.text((p(24), y + p(4)), f"scored (sim truth): found {s['found']}, find error {s['find_err_m']} m, "
                                      f"landing error {s['landing_err_m']:.1f} m", font=font(p(15)), fill=GREEN)
            y += p(26)
        arr = np.array(im)
        x = p(24)
        y0 = y + p(10)
        snaps = [s for t_ in view.m.tracks if t_.verified for s in t_.snapshots[:2]][:4]
        for s in snaps:
            c = cv2.cvtColor(cv2.resize(s["crop"], (p(200), p(200)), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2RGBA)
            if y0 + c.shape[0] <= H and x + c.shape[1] <= W - p(420):
                arr[y0:y0 + c.shape[0], x:x + c.shape[1]] = c
            x += p(210)
        mp = view.map_img
        if mp is not None:
            side = p(400)
            mm = cv2.cvtColor(cv2.resize(mp, (side, side), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2RGBA)
            arr[H - side - p(16):H - p(16), W - side - p(16):W - p(16)] = mm
        return Image.fromarray(arr)
