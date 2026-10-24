"""Night mode of the live demo (`demo_live.py --night`): dark 3D scene, a large white-hot thermal panel, a map
panel with the estimate, the truth, the 2-sigma ellipse and the matched roads, and a night HUD (GNSS ON / OFF,
active nav source, uncertainty in metres over time).

Harness code: it may read the truth (world/*) to DISPLAY. The thermal image shown is, in order of preference:
  1. frames the simulator publishes on the bus (`camera/thermal`, what the onboard thermal localizer sees),
  2. sfhack.thermal.camera.ThermalCamera rendered here on world/thermal_truth.npz (night-thermal's model),
  3. a stub: a nadir warp of a temperature raster (world/thermal_truth.npz, else warm = low NDVI / greenness),
     labelled STUB on screen.
"""

from __future__ import annotations

import math
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .hud import AMBER, CARD_BG, CYAN, GREEN, MUTED, RED, TEXT, WHITE, font, mix
from .render import SceneRenderer, blend_rgba, paste

THERMAL_TOPIC = "camera/thermal"
THERMAL_W, THERMAL_H = 640, 512  # sensor (display aspect); the stub renders half size and upsamples
THERMAL_HFOV_DEG = 50.0
NIGHT_SKY_TOP = np.array([0.01, 0.015, 0.04], np.float32)
NIGHT_SKY_HORIZON = np.array([0.06, 0.08, 0.14], np.float32)
FIX_FRESH_S = 3.0  # a source counts as "active" this long after its last accepted fix
SOURCE_LABEL = {"thermal": "THERMAL", "trn": "TRN", "vision": "VISION", "mock": "MOCK", "gnss": "GNSS",
                "lidar": "LIDAR"}


# ---------------------------------------------------------------------------------------- lighting

def apply_night_lighting(model: mujoco.MjModel) -> None:
    """Moonless-night look for the chase view (the simulated RGB down camera goes dark too: realistic)."""
    model.light_diffuse[:] *= 0.18
    model.light_ambient[:] *= 0.15
    model.light_specular[:] *= 0.0
    model.vis.headlight.ambient[:] = [0.10, 0.11, 0.14]
    model.vis.headlight.diffuse[:] = [0.16, 0.17, 0.22]


def night_sky(h: int, w: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None] ** 0.8
    col = NIGHT_SKY_TOP[None, :] * (1 - t) + NIGHT_SKY_HORIZON[None, :] * t
    return np.repeat((col * 255)[:, None, :], w, axis=1)


# ---------------------------------------------------------------------------------------- thermal

def agc_8bit(img_k: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """Automatic gain control: percentile stretch to 0..255, white-hot."""
    lo, hi = np.percentile(img_k, [lo_pct, hi_pct])
    return np.clip((img_k - lo) / max(hi - lo, 0.05) * 255.0, 0, 255).astype(np.uint8)


def _load_raster(path: Path, key: str) -> tuple[np.ndarray, dict[str, float]] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        if key not in z:
            return None
        arr = np.asarray(z[key], np.float32)
        meta = {}
        if "meta" in z:
            import json
            meta = json.loads(str(z["meta"]))
        for k in ("min_x", "max_y", "res_m"):
            if k in z:
                meta[k] = float(z[k])
    return arr, meta


class ThermalFeed:
    """What the thermal panel shows (see the module docstring for the order)."""

    def __init__(self, fs: Any, mission_dir: Path | None) -> None:
        self.fs = fs
        self.img8: np.ndarray | None = None
        self.img_t = -1.0
        self.kind = "none"
        self.bus_frames = 0
        fs.bus.subscribe(THERMAL_TOPIC, self._on_frame)
        self.camera = None
        self.truth = None
        self.raster: np.ndarray | None = None
        self.grid: tuple[float, float, float] | None = None  # min_x, max_y, res
        self._next = 0.0
        self.period = 0.1  # s sim time between harness renders (10 Hz)
        if mission_dir is None:
            return
        wdir = mission_dir / "world"
        try:  # night-thermal's camera model, if it is on this branch and the mission has its truth raster
            from sfhack.thermal.camera import ThermalCamera, load_truth  # noqa: PLC0415

            if (wdir / "thermal_truth.npz").exists():
                self.camera = ThermalCamera(fs.sim.model)
                self.truth = load_truth(mission_dir)
                self.kind = "camera"
        except Exception as e:  # noqa: BLE001 - optional dependency, any failure falls back to the stub
            print(f"night: ThermalCamera unavailable ({type(e).__name__}: {e}); stub thermal view")
            self.camera = None
        self._init_stub(wdir)

    def _init_stub(self, wdir: Path) -> None:
        import json

        man = json.loads((wdir / "manifest.json").read_text(encoding="utf-8"))
        e = man["ortho"]["extent_enu"]
        res = float(man["ortho"]["m_per_px"][0])
        loaded = _load_raster(wdir / "thermal_truth.npz", "temp_K")
        if loaded is None:
            loaded = _load_raster(wdir / "thermal_truth.npz", "T")
        if loaded is not None:
            arr, meta = loaded
            self.raster = arr
            self.grid = (meta.get("min_x", e["min_x"]), meta.get("max_y", e["max_y"]),
                         meta.get("res_m", (e["max_x"] - e["min_x"]) / arr.shape[1]))
            self.stub_src = "thermal_truth.npz"
        else:
            if (wdir / "ndvi.png").exists():
                ndvi = np.asarray(Image.open(wdir / "ndvi.png"), np.float32) / 127.5 - 1.0
                veg = np.clip((ndvi - 0.1) / 0.6, 0, 1)
            else:
                rgb = np.asarray(Image.open(wdir / "ortho.jpg").convert("RGB"), np.float32)
                veg = np.clip((rgb[..., 1] - 0.5 * (rgb[..., 0] + rgb[..., 2])) / 25.0, 0, 1)
            # warm asphalt / roofs ~ 300 K, cool vegetation ~ 288 K (23 h, clear summer night)
            self.raster = (300.0 - 12.0 * cv2.GaussianBlur(veg, (0, 0), 1.0)).astype(np.float32)
            self.grid = (e["min_x"], e["max_y"], res)
            self.stub_src = "NDVI" if (wdir / "ndvi.png").exists() else "ortho greenness"
        if self.kind == "none":
            self.kind = "stub"

    def _on_frame(self, msg: Any) -> None:
        img = getattr(msg, "image_K", None)
        if img is None:
            img = getattr(msg, "image", None)
        if img is None:
            return
        img = np.asarray(img)
        self.img8 = img if img.dtype == np.uint8 else agc_8bit(img.astype(np.float32))
        if self.img8.ndim == 3:
            self.img8 = cv2.cvtColor(self.img8, cv2.COLOR_RGB2GRAY)
        self.img_t = float(getattr(msg, "t", self.fs.t))
        self.bus_frames += 1
        self.kind = "bus"

    def update(self) -> None:
        """Harness-side render when nothing arrives on the bus."""
        fs = self.fs
        if self.kind == "bus" and fs.t - self.img_t < 1.0:
            return
        if fs.t < self._next:
            return
        self._next = fs.t + self.period
        if self.camera is not None:
            try:
                img = self.camera.render(fs.sim.data, self.truth)
                self.img8 = self.camera.to_8bit(img) if hasattr(self.camera, "to_8bit") else agc_8bit(img)
                self.img_t, self.kind = fs.t, "camera"
                return
            except Exception as e:  # noqa: BLE001
                print(f"night: ThermalCamera.render failed ({type(e).__name__}: {e}); stub thermal view")
                self.camera = None
        if self.raster is not None:
            self.img8 = agc_8bit(self._stub_render())
            self.img_t, self.kind = fs.t, "stub"

    def _stub_render(self) -> np.ndarray:
        """Nadir warp of the temperature raster under the true camera (yaw only; display stub)."""
        fs = self.fs
        tr = fs.truth.state()
        h = max(2.0, float(tr.pos[2] - fs.true_ground_z(tr.pos[0], tr.pos[1])))
        w, hh = THERMAL_W // 2, THERMAL_H // 2
        s = 2.0 * h * math.tan(math.radians(THERMAL_HFOV_DEG) / 2) / w  # m per image px
        fwd = tr.R[:2, 0] / max(float(np.linalg.norm(tr.R[:2, 0])), 1e-6)
        right = np.array([fwd[1], -fwd[0]])
        min_x, max_y, res = self.grid
        # image (u, v) -> world: c + right * (u - cu) s + fwd * (cv - v) s -> raster (col, row)
        cu, cv = (w - 1) / 2, (hh - 1) / 2
        A = np.array([[right[0] * s, -fwd[0] * s], [right[1] * s, -fwd[1] * s]])
        b = tr.pos[:2] - A @ np.array([cu, cv])
        M = np.array([[1 / res, 0], [0, -1 / res]]) @ np.c_[A, b]
        M[:, 2] += [-min_x / res - 0.5, max_y / res - 0.5]
        img = cv2.warpAffine(self.raster, M, (w, hh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REPLICATE)
        rng = np.random.default_rng(int(fs.t * 10))
        img = img + rng.normal(0, 0.05, img.shape).astype(np.float32)  # ~NETD 50 mK
        return cv2.resize(img, (THERMAL_W, THERMAL_H), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------------------- state

def _src(fix: Any) -> str:
    """Fix source family: night-nav tags thermal fixes "thermal-<method>" (thermal-ncc, thermal-road, ...)."""
    s = str(getattr(fix, "source", "?")).lower()
    return "thermal" if s.startswith("thermal") else s


class NightState:
    """Per-flight night bookkeeping: fixes per source, GNSS switch, uncertainty history."""

    def __init__(self, demo: Any) -> None:
        self.demo = demo
        fs = demo.fs
        self.feed = ThermalFeed(fs, getattr(fs, "mission_dir", None))
        self.last_fix_t: dict[str, float] = {}
        self.n_fix: dict[str, int] = {}
        self.n_rej: dict[str, int] = {}
        self.last_fix: Any = None
        self.hist: deque[tuple[float, float, float]] = deque(maxlen=2400)  # t, 2-sigma, true error
        self._next_hist = 0.0
        self.gnss_off_at: float | None = None
        fs.bus.subscribe("fix", self._on_fix)
        self.prior_temp, self.prior_roads, self.prior_grid = self._load_prior(getattr(fs, "onboard_dir", None))

    @staticmethod
    def _load_prior(onboard_dir: Path | None) -> tuple[np.ndarray | None, np.ndarray | None, tuple | None]:
        if onboard_dir is None or not (Path(onboard_dir) / "thermal_prior.npz").exists():
            return None, None, None
        import json

        with np.load(Path(onboard_dir) / "thermal_prior.npz", allow_pickle=False) as z:
            temp = np.asarray(z["temp_K"], np.float32) if "temp_K" in z else None
            roads = np.asarray(z["roads"], np.uint8) if "roads" in z else None
            meta = json.loads(str(z["meta"])) if "meta" in z else {}
        bbox = meta.get("bbox_enu") or meta.get("extent_enu") or meta.get("bbox") or {}
        grid = None
        if "min_x" in bbox and "max_y" in bbox:
            grid = (float(bbox["min_x"]), float(bbox["max_y"]), float(meta.get("res_m", 0.5)))
        return temp, roads, grid

    def _on_fix(self, fix: Any) -> None:
        src = _src(fix)
        if getattr(fix, "accepted", True):
            self.last_fix_t[src] = float(getattr(fix, "t", self.demo.t))
            self.n_fix[src] = self.n_fix.get(src, 0) + 1
        else:
            self.n_rej[src] = self.n_rej.get(src, 0) + 1
        self.last_fix = fix

    def tick(self) -> None:
        d = self.demo
        self.feed.update()
        if d.t >= self._next_hist:
            self._next_hist = d.t + 0.25
            est, tr = d.fs.autopilot.pose(), d.fs.truth.state()
            self.hist.append((d.t, float(est.sigma2_horizontal()), float(np.linalg.norm(est.pos[:2] - tr.pos[:2]))))

    def gnss_on(self) -> bool:
        return self.demo.c.gnss.mode != "off"

    def nav_sources(self) -> list[str]:
        d = self.demo
        out = []
        if self.gnss_on() and d.gnss_state() in ("ok", "spoof"):
            out.append("GNSS")
        for src, t in sorted(self.last_fix_t.items()):
            if d.t - t < FIX_FRESH_S and src != "gnss":
                out.append(SOURCE_LABEL.get(src, src.upper()))
        return out


# ---------------------------------------------------------------------------------------- composer

class NightComposer:
    """Full night frame: dark chase view + thermal panel (right) + map (bottom left) + HUD (top left)."""

    def __init__(self, demo: Any, width: int, height: int) -> None:
        from liveview.chase import ChaseCam  # noqa: F401  (same chase camera as the day view)

        self.W, self.H = width, height
        self.s = height / 900.0
        self.sr = SceneRenderer(demo.fs.sim.model, width, height)
        apply_night_lighting(demo.fs.sim.model)
        self.sr._sky = night_sky(self.sr.H, self.sr.W)
        self.fovy = float(demo.fs.sim.model.vis.global_.fovy)
        self.chase = self._new_chase(demo)
        self.times = {"view": 0.0, "fog": 0.0, "hud": 0.0, "map": 0.0}
        self.hits: list = []
        self.hud = self  # demo_live's viewer reads comp.hud.hits for clickable rows

    def px(self, v: float) -> int:
        return int(round(v * self.s))

    @staticmethod
    def _new_chase(demo: Any):
        from liveview.chase import ChaseCam

        h = demo.goal - demo.start
        return ChaseCam(math.degrees(math.atan2(h[1], h[0])), np.array([demo.start[0], demo.start[1], demo.start_z + 1.0]))

    def reset(self, demo: Any) -> None:
        old = self.chase
        self.chase = self._new_chase(demo)
        self.chase.az_off, self.chase.el_off, self.chase.zoom = old.az_off, old.el_off, old.zoom

    def close(self) -> None:
        self.sr.close()

    # ------------------------------------------------------------------ frame

    def _aim(self) -> tuple:
        p = self.px
        x0, x1 = p(16) + p(400), self.W - self._thermal_w() - p(32)
        y0, y1 = p(80), self.H - p(16) - p(300)
        return ((x0 + x1) * 0.5, (y0 + y1) * 0.5, self.W, self.H, self.fovy)

    def _thermal_w(self) -> int:
        return int(min(0.46 * self.W, (self.H - self.px(16) * 2 - self.px(70)) * THERMAL_W / THERMAL_H))

    def frame(self, demo: Any, dt: float, speed: int, paused: bool, recording: bool) -> np.ndarray:
        ns: NightState = demo.night
        ns.tick()
        tr, est = demo.fs.truth.state(), demo.fs.autopilot.pose()
        cam = self.chase.update(dt, tr.pos, tr.vel, est.pos, aim=self._aim())
        demo.update_particles(0.0 if paused else dt * speed, np.array(cam.lookat), cam.distance)
        img = self.sr.render(demo.fs.sim.data, cam, demo.draw_geoms, None, (0, 0, 0), 0.0)
        t0 = time.perf_counter()
        p = self.px
        m = p(16)
        tw = self._thermal_w()
        th = int(tw * THERMAL_H / THERMAL_W)
        paste(img, self.thermal_panel(demo, tw, th), self.W - m - tw, m, tw, th)
        blend_rgba(img, np.asarray(self.thermal_caption(demo, tw)), self.W - m - tw, m + th + p(6))
        t1 = time.perf_counter()
        side = p(300)
        paste(img, self.map_panel(demo, side), m, self.H - m - side, side, side)
        t2 = time.perf_counter()
        card = self.hud_card(demo, speed, paused, recording)
        blend_rgba(img, np.asarray(card), m, m)
        if demo.result is not None:
            r = demo.result
            ban = self._banner(f"{r.outcome} · {r.final_error:.1f} m from B", RED if not r.success else GREEN)
            blend_rgba(img, np.asarray(ban), (m + p(400) + self.W - m - tw - ban.width) // 2, m)
        elif demo.deck.banner is not None and demo.t - demo.deck.banner[0] < 4.0:
            ban = self._banner(demo.deck.banner[1], AMBER)
            blend_rgba(img, np.asarray(ban), (m + p(400) + self.W - m - tw - ban.width) // 2, m)
        self.times = {"view": self.sr.t_view, "fog": 0.0, "map": t2 - t1, "hud": (t1 - t0) + (time.perf_counter() - t2)}
        return img

    # ------------------------------------------------------------------ panels

    def thermal_panel(self, demo: Any, w: int, h: int) -> np.ndarray:
        ns: NightState = demo.night
        feed = ns.feed
        if feed.img8 is None:
            img = np.zeros((h, w, 3), np.uint8)
        else:
            img = cv2.cvtColor(cv2.resize(feed.img8, (w, h), interpolation=cv2.INTER_LINEAR), cv2.COLOR_GRAY2RGB)
        fs_ = h / 480
        fix = ns.last_fix
        fresh = fix is not None and _src(fix) == "thermal" and demo.t - float(fix.t) < 0.6
        if fresh and getattr(fix, "matched_px", None):
            pts = np.array([(u * w / THERMAL_W, v * h / THERMAL_H) for u, v in fix.matched_px], np.int32)
            col = (255, 220, 0) if getattr(fix, "accepted", True) else (255, 60, 60)
            cv2.polylines(img, [pts], False, col, max(2, int(3 * fs_)), cv2.LINE_AA)
        # reticle
        cx, cy = w // 2, h // 2
        L = int(18 * fs_)
        for a, b in (((cx - L, cy), (cx - L // 3, cy)), ((cx + L // 3, cy), (cx + L, cy)),
                     ((cx, cy - L), (cx, cy - L // 3)), ((cx, cy + L // 3), (cx, cy + L))):
            cv2.line(img, a, b, (0, 230, 255), max(1, int(2 * fs_)), cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 200, 255) if fresh else (120, 130, 150), max(2, int(3 * fs_)))
        return img

    def thermal_caption(self, demo: Any, w: int) -> Image.Image:
        ns: NightState = demo.night
        p = self.px
        h = p(56)
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=p(8), fill=CARD_BG, outline=mix(WHITE, 0.16))
        kind = {"bus": "onboard sensor feed", "camera": "thermal camera model",
                "stub": f"STUB view ({getattr(ns.feed, 'stub_src', '')})", "none": "no thermal"}[ns.feed.kind]
        d.text((p(12), p(8)), f"THERMAL CAMERA  640x512 · 50° · white-hot   [{kind}]",
               font=font(max(8, p(15)), True), fill=TEXT)
        fix = ns.last_fix
        if fix is not None and demo.t - float(getattr(fix, "t", -99)) < 1.5:
            src = _src(fix).upper()
            acc = getattr(fix, "accepted", True)
            q = getattr(fix, "quality", None)
            meth = getattr(fix, "method", "") or str(getattr(fix, "source", "")).partition("-")[2]
            txt = f"{src} FIX {'accepted' if acc else 'REJECTED'}" + (f" · {meth}" if meth else "") + \
                  (f" · quality {q:.2f}" if isinstance(q, (int, float)) else "")
            d.text((p(12), p(31)), txt, font=font(max(8, p(15))), fill=(255, 220, 0, 255) if acc else RED)
        else:
            n = ns.n_fix.get("thermal", 0)
            d.text((p(12), p(31)), f"no fix right now · thermal fixes so far: {n}", font=font(max(8, p(15))), fill=MUTED)
        return im

    def map_panel(self, demo: Any, size: int) -> np.ndarray:
        ns: NightState = demo.night
        om = demo.fs.onboard_map
        est, tr = demo.fs.autopilot.pose(), demo.fs.truth.state()
        s2 = float(est.sigma2_horizontal())
        pts = np.array([est.pos[:2], tr.pos[:2]])
        center = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
        span = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
        half = max(90.0, 0.6 * span + 30.0, 1.15 * s2 + 0.5 * span + 10.0)
        k = size / (2 * half)  # px per metre

        def px(xy) -> tuple[int, int]:
            xy = np.asarray(xy, float)
            return int(round((xy[0] - center[0] + half) * k)), int(round((center[1] + half - xy[1]) * k))

        def warp(raster: np.ndarray, grid: tuple, interp=cv2.INTER_AREA, border=0) -> np.ndarray:
            min_x, max_y, res = grid
            # map px (u, v) -> world (center - half + u / k, center + half - v / k) -> raster (col, row)
            M = np.array([[1 / (k * res), 0, (center[0] - half - min_x) / res - 0.5],
                          [0, 1 / (k * res), (max_y - center[1] - half) / res - 0.5]])
            return cv2.warpAffine(raster, M, (size, size), flags=interp | cv2.WARP_INVERSE_MAP, borderValue=border)

        ogrid = (om.min_x, om.max_y, om.mpp_x)
        if ns.prior_temp is not None and ns.prior_grid is not None:
            t = warp(ns.prior_temp, ns.prior_grid, border=float(np.median(ns.prior_temp)))
            lo, hi = np.percentile(ns.prior_temp, [2, 98])
            g8 = np.clip((t - lo) / max(hi - lo, 0.1) * 255, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(cv2.applyColorMap(g8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
            title = "ONBOARD MAP  (predicted night temperature)"
        else:
            img = (warp(om.ortho, ogrid, border=(40, 40, 40)) * 0.45).astype(np.uint8)
            title = "ONBOARD MAP  (orthophoto)"
        if ns.prior_roads is not None and ns.prior_grid is not None:
            roads = warp(ns.prior_roads, ns.prior_grid, cv2.INTER_NEAREST)
            img[roads > 0] = (0.55 * img[roads > 0] + 0.45 * np.array([255, 150, 40])).astype(np.uint8)
            fix = ns.last_fix
            if fix is not None and _src(fix) == "thermal" and demo.t - float(fix.t) < 1.0:
                # matched roads: the prior road pixels inside the camera footprint around the fix
                agl = max(5.0, float(est.pos[2] - om.ground_z(est.pos[:2])))
                r_fp = agl * math.tan(math.radians(THERMAL_HFOV_DEG) / 2)
                fp = np.zeros((size, size), np.uint8)
                cv2.circle(fp, px(fix.pos), int(r_fp * k), 1, -1)
                hit = (roads > 0) & (fp > 0)
                img[hit] = (255, 235, 60) if getattr(fix, "accepted", True) else (255, 70, 60)
        lw = max(1, size // 160)
        route = getattr(demo, "route", None)
        if route is not None and len(route) > 1:
            cv2.polylines(img, [np.array([px(q) for q in route[:, :2]], np.int32)], False, (200, 200, 210), lw, cv2.LINE_AA)
        for trail, col in ((demo.truth_trail, (255, 255, 255)), (demo.est_trail, (0, 217, 255))):
            if len(trail) > 1:
                a = np.array(trail)[-400:, :2]
                cv2.polylines(img, [np.array([px(q) for q in a], np.int32)], False, col, lw + 1, cv2.LINE_AA)
        for xy, col in ((demo.start, (70, 215, 120)), (demo.goal, (255, 75, 65))):
            cv2.circle(img, px(xy), max(4, int(4 * k)), col, lw + 1, cv2.LINE_AA)
        # 2-sigma ellipse of the filter covariance
        cov = np.asarray(est.cov)[:2, :2]
        ev, evec = np.linalg.eigh(0.5 * (cov + cov.T))
        ax = 2.0 * np.sqrt(np.maximum(ev, 0.0)) * k
        ang = math.degrees(math.atan2(-evec[1, 1], evec[0, 1]))
        cv2.ellipse(img, px(est.pos), (max(2, int(ax[1])), max(2, int(ax[0]))), ang, 0, 360, (0, 217, 255), lw + 1, cv2.LINE_AA)
        fix = ns.last_fix
        if fix is not None and demo.t - float(getattr(fix, "t", -99)) < 1.0:
            cv2.drawMarker(img, px(fix.pos), (255, 220, 0) if getattr(fix, "accepted", True) else (255, 60, 60),
                           cv2.MARKER_TILTED_CROSS, 6 * lw + 8, lw + 1)
        cv2.circle(img, px(tr.pos), 3 * lw + 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, px(est.pos), 2 * lw + 3, (0, 217, 255), -1, cv2.LINE_AA)
        f = size / 400
        bar_m = 20.0 if half < 100 else (50.0 if half < 250 else 100.0)
        cv2.line(img, (10, size - 14), (10 + int(bar_m * k), size - 14), (255, 255, 255), lw + 1)
        cv2.putText(img, f"{bar_m:.0f} m", (16 + int(bar_m * k), size - 9), cv2.FONT_HERSHEY_DUPLEX, 0.5 * f, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (size, int(28 * f)), (10, 14, 22), -1)
        cv2.putText(img, title, (8, int(19 * f)), cv2.FONT_HERSHEY_DUPLEX, 0.5 * f, (235, 240, 245), 1, cv2.LINE_AA)
        cv2.putText(img, "white = true  cyan = estimate + 2-sigma", (8, int(46 * f)), cv2.FONT_HERSHEY_DUPLEX, 0.42 * f,
                    (235, 240, 245), 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (size - 1, size - 1), (255, 255, 255), 1)
        return img

    # ------------------------------------------------------------------ HUD

    def _banner(self, text: str, color) -> Image.Image:
        p = self.px
        fnt = font(max(8, p(26)), True)
        tw = int(ImageDraw.Draw(Image.new("RGBA", (1, 1))).textlength(text, font=fnt))
        im = Image.new("RGBA", (tw + p(36), p(50)), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle((0, 0, im.width - 1, im.height - 1), radius=p(10), fill=mix(color, 0.25), outline=color, width=p(2))
        d.text((im.width // 2, im.height // 2), text, font=fnt, fill=color, anchor="mm")
        return im

    def hud_card(self, demo: Any, speed: int, paused: bool, recording: bool) -> Image.Image:
        ns: NightState = demo.night
        p = self.px
        w, h = p(400), p(410)
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=p(10), fill=CARD_BG, outline=mix(WHITE, 0.16))
        x, y = p(16), p(12)
        d.text((x, y), "NIGHT FLIGHT · 23:00", font=font(max(8, p(22)), True), fill=TEXT)
        d.text((w - p(16), y + p(4)), f"t {demo.t:5.0f} s", font=font(max(8, p(16))), fill=MUTED, anchor="ra")
        y += p(30)
        place = f"mission {demo.mission}" if demo.mission else "Berkeley"
        d.text((x, y), f"Skydio X2 · {place}", font=font(max(8, p(15))), fill=MUTED)
        d.text((w - p(16), y), demo.fs.mission.phase, font=font(max(8, p(15)), True), fill=TEXT, anchor="ra")
        y += p(32)
        on = ns.gnss_on()
        col = GREEN if on else RED
        chip = f"GNSS: {'ON' if on else 'OFF'}"
        fnt = font(max(8, p(30)), True)
        cw = int(d.textlength(chip, font=fnt)) + p(30)
        d.rounded_rectangle((x, y, x + cw, y + p(46)), radius=p(23), fill=mix(col, 0.25), outline=col, width=p(2))
        d.text((x + cw // 2, y + p(23)), chip, font=fnt, fill=col, anchor="mm")
        sub = "key N: GNSS off" if on else (f"no signal since {ns.gnss_off_at:.0f} s" if ns.gnss_off_at is not None
                                           else "no signal")
        d.text((x + cw + p(12), y + p(14)), sub, font=font(max(8, p(15))), fill=MUTED)
        y += p(62)
        d.text((x, y), "NAV SOURCE", font=font(max(8, p(14)), True), fill=MUTED)
        y += p(20)
        src = ns.nav_sources()
        txt = " + ".join(src + ["IMU", "BARO"]) if src else "IMU + BARO (dead reckoning)"
        d.text((x, y), txt, font=font(max(8, p(19)), True), fill=CYAN if src else AMBER)
        y += p(34)
        est, tr = demo.fs.autopilot.pose(), demo.fs.truth.state()
        s2 = float(est.sigma2_horizontal())
        err = float(np.linalg.norm(est.pos[:2] - tr.pos[:2]))
        ucol = GREEN if s2 < 5 else (AMBER if s2 < 15 else RED)
        d.text((x, y), "UNCERTAINTY (2σ)", font=font(max(8, p(14)), True), fill=MUTED)
        d.text((x + p(200), y), "TRUE ERROR (eval)", font=font(max(8, p(14)), True), fill=MUTED)
        y += p(20)
        d.text((x, y), f"{s2:5.1f} m", font=font(max(8, p(34)), True), fill=ucol)
        d.text((x + p(200), y), f"{err:5.1f} m", font=font(max(8, p(34)), True), fill=WHITE)
        y += p(50)
        self._sparkline(d, ns, x, y, w - 2 * x, p(90))
        y += p(100)
        fixes = ", ".join(f"{SOURCE_LABEL.get(k, k.upper())} {v}" for k, v in sorted(ns.n_fix.items())) or "none yet"
        d.text((x, y), f"fixes accepted: {fixes}", font=font(max(8, p(14))), fill=MUTED)
        y += p(22)
        state = ("PAUSED" if paused else f"{speed}x") + ("  ● REC" if recording else "")
        d.text((x, y), f"N GNSS · SPACE pause · 1/2/4 speed ({state}) · R restart · Q quit",
               font=font(max(8, p(12))), fill=MUTED)
        return im

    def _sparkline(self, d: ImageDraw.ImageDraw, ns: NightState, x: int, y: int, w: int, h: int) -> None:
        d.rectangle((x, y, x + w, y + h), outline=mix(WHITE, 0.2))
        if len(ns.hist) < 2:
            return
        a = np.array(ns.hist)
        t1 = a[-1, 0]
        a = a[a[:, 0] >= t1 - 180.0]
        t0 = max(a[0, 0], t1 - 180.0)
        vmax = max(10.0, float(np.max(a[:, 1:])) * 1.15)

        def pt(t: float, v: float) -> tuple[float, float]:
            return x + (t - t0) / max(t1 - t0, 1.0) * w, y + h - min(v, vmax) / vmax * h

        if ns.gnss_off_at is not None and ns.gnss_off_at >= t0:
            gx = pt(ns.gnss_off_at, 0)[0]
            d.line((gx, y, gx, y + h), fill=RED, width=max(1, self.px(1.5)))
            d.text((gx + self.px(4), y + self.px(2)), "GNSS off", font=font(max(8, self.px(11))), fill=RED)
        d.line([pt(t, v) for t, v in a[:, [0, 2]]], fill=WHITE, width=max(1, self.px(1.5)))
        d.line([pt(t, v) for t, v in a[:, [0, 1]]], fill=CYAN, width=max(1, self.px(2.5)))
        d.text((x + w - self.px(4), y + self.px(2)), f"{vmax:.0f} m", font=font(max(8, self.px(11))), fill=MUTED, anchor="ra")
        d.text((x + self.px(4), y + h - self.px(16)), "cyan 2σ  white true error  (last 3 min)",
               font=font(max(8, self.px(11))), fill=MUTED)
