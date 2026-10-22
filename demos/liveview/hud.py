"""Heads-up display: cards drawn with PIL (anti-aliased TrueType text, readable from across a room) and
alpha-composited onto the rendered frame. Everything is laid out for 1600x900 and scaled with the height.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .render import blend_rgba, paste

CARD_BG = (10, 14, 22, 228)
TEXT = (235, 240, 245, 255)
MUTED = (150, 160, 175, 255)
CYAN = (0, 217, 255, 255)
WHITE = (255, 255, 255, 255)
GREEN = (70, 215, 120, 255)
AMBER = (255, 185, 50, 255)
RED = (255, 75, 65, 255)
ORANGE = (255, 140, 30, 255)
VIOLET = (175, 125, 255, 255)
LEVEL_COLOR = (GREEN, AMBER, RED)
HEALTH_COLOR = {"NOMINAL": GREEN, "DEGRADED": AMBER, "HOLD": RED, "RETURN_TO_LAST_FIX": RED, "LAND": RED}


def mix(color, a: float) -> tuple[int, int, int, int]:
    """`color` at opacity `a` over the card background, as an opaque card-alpha fill (PIL draws replace
    pixels instead of compositing them)."""
    return tuple(int(c * a + b * (1 - a)) for c, b in zip(color[:3], CARD_BG[:3])) + (CARD_BG[3],)


SENSOR_COLOR = {"ok": GREEN, "degraded": AMBER, "lost": RED, "off": (90, 95, 105, 255)}


@lru_cache(maxsize=None)
def _font_path(bold: bool) -> str | None:
    try:
        import matplotlib
        p = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
        return str(p) if p.exists() else None
    except ImportError:
        return None


@lru_cache(maxsize=None)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = _font_path(bold)
    return ImageFont.truetype(path, size) if path else ImageFont.load_default(size)


@dataclass
class HudState:
    t: float
    phase: str
    health: str
    health_reason: str
    err: float  # m, |estimate - truth| horizontal (eval only)
    sigma2: float  # m, claimed 2-sigma radius
    nav_source: str
    fix_age: str
    gnss_line: str
    gnss_level: int
    degradations: list[tuple[str, int]]
    levers: list[tuple[str, str, str, int]]  # key, name, value, level
    controls: list[tuple[str, str, str]]  # key, label, value
    sensors: list[tuple[str, str]]  # name, ok | degraded | lost | off
    wind: tuple[float, float, float]  # speed m/s, compass deg toward, gust m/s
    weather: str
    banner: tuple[str, float] | None  # text, age s
    scenario: str
    result: str | None
    speed: int
    paused: bool
    recording: bool
    labels: list[tuple[tuple[int, int], str, tuple[int, int, int, int]]] = field(default_factory=list)


class Hud:
    def __init__(self, width: int, height: int) -> None:
        self.W, self.H = width, height
        self.s = height / 900.0
        self.hits: list[tuple[int, int, int, int, str]] = []  # clickable rects (x0, y0, x1, y1, key)

    def px(self, v: float) -> int:
        return int(round(v * self.s))

    def f(self, size: float, bold: bool = False) -> ImageFont.FreeTypeFont:
        return font(max(8, self.px(size)), bold)

    def _card(self, w: int, h: int, accent=None) -> tuple[Image.Image, ImageDraw.ImageDraw]:
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=self.px(10), fill=CARD_BG, outline=mix(WHITE, 0.16), width=1)
        if accent is not None:
            d.rounded_rectangle((0, 0, self.px(5), h - 1), radius=self.px(3), fill=accent)
        return im, d

    def _chip(self, d: ImageDraw.ImageDraw, x: int, y: int, text: str, color, size: float = 14,
              filled: bool = True) -> int:
        fnt = self.f(size, True)
        tw = int(d.textlength(text, font=fnt))
        pad, h = self.px(8), self.px(size + 10)
        bg = mix(color, 0.22) if filled else CARD_BG
        d.rounded_rectangle((x, y, x + tw + 2 * pad, y + h), radius=h // 2, fill=bg, outline=color, width=max(1, self.px(1.5)))
        d.text((x + pad, y + h // 2), text, font=fnt, fill=color, anchor="lm")
        return x + tw + 2 * pad

    # ------------------------------------------------------------------ cards

    def _title(self, st: HudState) -> Image.Image:
        w, h = self.px(470), self.px(66)
        im, d = self._card(w, h, CYAN)
        d.text((self.px(18), self.px(8)), "GNSS SPOOF GUARD", font=self.f(25, True), fill=TEXT)
        d.text((self.px(18), self.px(40)), "Skydio X2 over Berkeley, flying A to B  |  MuJoCo live", font=self.f(14), fill=MUTED)
        rec = "REC  " if st.recording else ""
        d.text((w - self.px(14), self.px(14)), f"{rec}t {st.t:5.1f} s", font=self.f(16, True),
               fill=RED if st.recording else TEXT, anchor="ra")
        sp = "PAUSED" if st.paused else f"{st.speed}x"
        d.text((w - self.px(14), self.px(40)), sp, font=self.f(14, True), fill=AMBER if st.paused else MUTED, anchor="ra")
        return im

    def _mission(self, st: HudState) -> Image.Image:
        w, h = self.px(470), self.px(292)
        im, d = self._card(w, h)
        x0 = self.px(18)
        d.text((x0, self.px(12)), "MISSION", font=self.f(13, True), fill=MUTED)
        d.text((x0, self.px(30)), st.phase.replace("_", " "), font=self.f(30, True), fill=TEXT)
        hc = HEALTH_COLOR.get(st.health, AMBER)
        self._chip(d, self.px(250), self.px(36), st.health, hc, 15)
        # error vs claimed 2-sigma
        y = self.px(84)
        col_w = (w - 2 * x0) // 2
        over = st.err > st.sigma2
        d.text((x0, y), "TRUE ERROR", font=self.f(13, True), fill=MUTED)
        d.text((x0 + col_w, y), "CLAIMED 2\u03c3", font=self.f(13, True), fill=MUTED)
        d.text((x0, y + self.px(16)), f"{st.err:.1f} m", font=self.f(42, True), fill=RED if over else WHITE)
        d.text((x0 + col_w, y + self.px(16)), f"{st.sigma2:.1f} m", font=self.f(42, True), fill=CYAN)
        # bar: the 2-sigma band vs the true error
        by, bh = y + self.px(78), self.px(14)
        bx0, bx1 = x0, w - x0
        scale = max(st.err, st.sigma2, 5.0) * 1.15
        d.rounded_rectangle((bx0, by, bx1, by + bh), radius=bh // 2, fill=mix(WHITE, 0.12))
        sx = bx0 + int((bx1 - bx0) * min(st.sigma2 / scale, 1.0))
        d.rounded_rectangle((bx0, by, max(sx, bx0 + bh), by + bh), radius=bh // 2, fill=mix(CYAN, 0.5))
        ex = bx0 + int((bx1 - bx0) * min(st.err / scale, 1.0))
        d.rectangle((ex - self.px(2), by - self.px(5), ex + self.px(2), by + bh + self.px(5)), fill=RED if over else WHITE)
        verdict = ("OVERCONFIDENT: true error outside the claimed 2\u03c3" if over
                   else "CONSISTENT: true error inside the claimed 2\u03c3")
        d.text((x0, by + bh + self.px(10)), verdict, font=self.f(14, True), fill=RED if over else GREEN)
        # navigation rows
        rows = [("Navigating on", st.nav_source, TEXT), ("Last visual fix", st.fix_age, TEXT),
                ("GNSS", st.gnss_line, LEVEL_COLOR[st.gnss_level])]
        ry = by + bh + self.px(38)
        for k, v, c in rows:
            d.text((x0, ry), k, font=self.f(14), fill=MUTED)
            d.text((x0 + self.px(130), ry), v, font=self.f(14, True), fill=c)
            ry += self.px(22)
        d.text((x0, h - self.px(20)), "true error is scored on simulator truth (display only, never onboard)",
               font=self.f(11), fill=MUTED)
        return im

    def _degradations(self, st: HudState) -> Image.Image:
        w, h = self.px(470), self.px(74)
        im, d = self._card(w, h)
        d.text((self.px(18), self.px(8)), "ACTIVE DEGRADATIONS", font=self.f(13, True), fill=MUTED)
        x, y = self.px(18), self.px(34)
        items = st.degradations or [("ALL NOMINAL", 0)]
        for text, lvl in items:
            fnt = self.f(14, True)
            need = int(ImageDraw.Draw(im).textlength(text, font=fnt)) + self.px(16)
            if x + need > w - self.px(10):
                break
            x = self._chip(d, x, y, text, LEVEL_COLOR[lvl], 14) + self.px(8)
        return im

    def _levers(self, st: HudState, ox: int, oy: int) -> Image.Image:
        w = self.px(360)
        row_h = self.px(25)
        n = len(st.levers) + len(st.controls)
        h = self.px(48) + row_h * n + self.px(40)
        im, d = self._card(w, h, AMBER)
        d.text((self.px(18), self.px(12)), "LEVERS", font=self.f(15, True), fill=TEXT)
        d.text((w - self.px(14), self.px(14)), "press key or click", font=self.f(12), fill=MUTED, anchor="ra")
        y = self.px(46)
        self.hits = []
        for key, name, value, lvl in st.levers:
            self._lever_row(d, y, row_h, w, key, name, value, LEVEL_COLOR[lvl])
            self.hits.append((ox, oy + y, ox + w, oy + y + row_h, key))
            y += row_h
        d.line((self.px(14), y + self.px(3), w - self.px(14), y + self.px(3)), fill=mix(WHITE, 0.16), width=1)
        y += self.px(6)
        for key, label, value in st.controls:
            self._lever_row(d, y, row_h, w, key, label, value, CYAN if key == "P" else TEXT, dot=False)
            self.hits.append((ox, oy + y, ox + w, oy + y + row_h, key))
            y += row_h
        d.text((w // 2, h - self.px(16)), "R restart   SPACE pause   1 2 4 speed   O camera   Q quit",
               font=self.f(12), fill=MUTED, anchor="mm")
        return im

    def _lever_row(self, d, y, row_h, w, key, name, value, color, dot=True) -> None:
        x0 = self.px(16)
        kf = self.f(13, True)
        kw = max(self.px(24), int(d.textlength(key, font=kf)) + self.px(12))
        d.rounded_rectangle((x0, y + self.px(3), x0 + kw, y + row_h - self.px(4)), radius=self.px(4),
                            fill=mix(WHITE, 0.14), outline=mix(WHITE, 0.4))
        d.text((x0 + kw // 2, y + row_h // 2), key, font=kf, fill=TEXT, anchor="mm")
        d.text((x0 + kw + self.px(10), y + row_h // 2), name, font=self.f(15), fill=TEXT, anchor="lm")
        vx = w - self.px(34) if dot else w - self.px(16)
        d.text((vx, y + row_h // 2), value, font=self.f(15, True), fill=color, anchor="rm")
        if dot:
            cx, cy, r = w - self.px(20), y + row_h // 2, self.px(6)
            d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)

    def _saturday(self) -> Image.Image:
        w, h = self.px(360), self.px(214)
        im, d = self._card(w, h, VIOLET)
        d.text((self.px(18), self.px(12)), "DETECTION  \u00b7  TRUST  \u00b7  DECISIONS", font=self.f(15, True), fill=TEXT)
        self._chip(d, self.px(18), self.px(38), "BUILT AT HACKATHON", VIOLET, 12)
        y = self.px(72)
        for title in ("Detection", "Trust ranking", "Decision log"):
            bh = self.px(40)
            box = (self.px(16), y, w - self.px(16), y + bh)
            _dashed_rect(d, box, mix(VIOLET, 0.6), self.px(8), max(1, self.px(1.5)))
            d.text((box[0] + self.px(12), y + bh // 2), title, font=self.f(15, True), fill=TEXT, anchor="lm")
            d.text((box[2] - self.px(12), y + bh // 2), "reserved for Saturday", font=self.f(12), fill=MUTED, anchor="rm")
            y += bh + self.px(8)
        return im

    def _sensors(self, st: HudState) -> Image.Image:
        w, h = self.px(716), self.px(40)
        im, d = self._card(w, h)
        d.text((self.px(14), h // 2), "SENSORS", font=self.f(13, True), fill=MUTED, anchor="lm")
        x = self.px(96)
        for name, state in st.sensors:
            c = SENSOR_COLOR[state]
            r = self.px(6)
            d.ellipse((x, h // 2 - r, x + 2 * r, h // 2 + r), fill=c)
            d.text((x + 2 * r + self.px(6), h // 2), name, font=self.f(14, True),
                   fill=TEXT if state != "off" else MUTED, anchor="lm")
            x += 2 * r + self.px(14) + int(d.textlength(name, font=self.f(14, True))) + self.px(12)
        return im

    def _weather(self, st: HudState) -> Image.Image:
        w, h = self.px(360), self.px(150)
        im, d = self._card(w, h)
        d.text((self.px(18), self.px(12)), "WIND & WEATHER", font=self.f(13, True), fill=MUTED)
        spd, deg, gust = st.wind
        cx, cy, r = self.px(70), self.px(88), self.px(44)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=mix(WHITE, 0.45), width=max(1, self.px(2)))
        for lbl, a in (("N", 0), ("E", 90), ("S", 180), ("W", 270)):
            ux, uy = np.sin(np.radians(a)), -np.cos(np.radians(a))
            d.text((cx + ux * (r + self.px(10)), cy + uy * (r + self.px(10))), lbl, font=self.f(11, True), fill=MUTED, anchor="mm")
        if spd > 0:
            ux, uy = np.sin(np.radians(deg)), -np.cos(np.radians(deg))
            L = r * min(1.0, 0.35 + spd / 14)
            tip = (cx + ux * L, cy + uy * L)
            tail = (cx - ux * L * 0.8, cy - uy * L * 0.8)
            d.line((tail, tip), fill=(120, 200, 255, 255), width=self.px(5))
            px_, py_ = -uy, ux
            hs = self.px(12)
            d.polygon([tip, (tip[0] - ux * hs + px_ * hs * 0.6, tip[1] - uy * hs + py_ * hs * 0.6),
                       (tip[0] - ux * hs - px_ * hs * 0.6, tip[1] - uy * hs - py_ * hs * 0.6)], fill=(120, 200, 255, 255))
        else:
            d.text((cx, cy), "CALM", font=self.f(12, True), fill=MUTED, anchor="mm")
        x = self.px(140)
        d.text((x, self.px(40)), f"{spd:.0f} m/s" if spd else "calm", font=self.f(26, True), fill=TEXT)
        d.text((x, self.px(74)), f"toward {deg:03.0f} deg" + (f", gusts {gust:g}" if gust else ""), font=self.f(13), fill=MUTED)
        d.text((x, self.px(100)), st.weather, font=self.f(16, True), fill=AMBER if st.weather != "CLEAR" else GREEN)
        return im

    def _legend(self) -> Image.Image:
        w, h = self.px(360), self.px(112)
        im, d = self._card(w, h)
        items = [((255, 255, 255, 255), "true drone + true path (simulator)"),
                 (CYAN, "ghost: where it thinks it is, 2\u03c3 bubble"),
                 (ORANGE, "GNSS fix (what the receiver reports)"),
                 ((255, 70, 70, 255), "lidar / rangefinder returns  |  yellow: camera")]
        y = self.px(12)
        for c, text in items:
            d.rounded_rectangle((self.px(16), y + self.px(3), self.px(36), y + self.px(15)), radius=self.px(3), fill=c)
            d.text((self.px(46), y), text, font=self.f(13), fill=TEXT)
            y += self.px(24)
        return im

    def _banner(self, text: str, age: float, max_w: int, big: bool = False) -> Image.Image | None:
        alpha = 1.0 if age < 3.0 else max(0.0, 1.0 - (age - 3.0) / 1.0)
        if alpha <= 0:
            return None
        size = 30 if big else 26
        probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        fnt = self.f(size, True)
        while size > 14 and probe.textlength(text, font=fnt) + self.px(48) > max_w:
            size -= 1
            fnt = self.f(size, True)
        tw = int(probe.textlength(text, font=fnt))
        w, h = tw + self.px(48), self.px(56)
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        col = RED if any(k in text for k in ("SPOOF", "LOST", "JAM", "MISSED")) else (
            GREEN if any(k in text for k in ("ARRIVED", "BACK", "HONEST", "NOMINAL")) else AMBER)
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=h // 2, fill=(10, 14, 22, int(225 * alpha)),
                            outline=col[:3] + (int(255 * alpha),), width=self.px(3))
        d.text((w // 2, h // 2), text, font=fnt, fill=col[:3] + (int(255 * alpha),), anchor="mm")
        return im

    def _label(self, text: str, color) -> Image.Image:
        fnt = self.f(14, True)
        tw = int(ImageDraw.Draw(Image.new("RGBA", (1, 1))).textlength(text, font=fnt))
        w, h = tw + self.px(16), self.px(24)
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=self.px(6), fill=(10, 14, 22, 190), outline=color, width=1)
        d.text((w // 2, h // 2), text, font=fnt, fill=color, anchor="mm")
        return im

    # ------------------------------------------------------------------ compose

    def compose(self, frame: np.ndarray, st: HudState, map_img: np.ndarray, cam_img: np.ndarray) -> np.ndarray:
        W, H, p = self.W, self.H, self.px
        m = p(16)
        for (u, v), text, color in st.labels:
            lab = self._label(text, color)
            blend_rgba(frame, np.asarray(lab), u - lab.width // 2, v - lab.height - p(10))
        blend_rgba(frame, np.asarray(self._title(st)), m, m)
        blend_rgba(frame, np.asarray(self._mission(st)), m, m + p(76))
        blend_rgba(frame, np.asarray(self._degradations(st)), m, m + p(378))
        rx = W - m - p(360)
        lev = self._levers(st, rx, m)
        blend_rgba(frame, np.asarray(lev), rx, m)
        y = m + lev.height + p(10)
        sat = self._saturday()
        blend_rgba(frame, np.asarray(sat), rx, y)
        y += sat.height + p(10)
        if y + p(150) <= H - m - p(122):
            blend_rgba(frame, np.asarray(self._weather(st)), rx, y)
        blend_rgba(frame, np.asarray(self._legend()), rx, H - m - p(112))
        # bottom-left: onboard map + live down camera + sensor strip
        side = p(300)
        cw = p(400)
        paste(frame, map_img, m, H - m - side, side, side)
        paste(frame, cam_img, m + side + p(16), H - m - side, cw, side)
        blend_rgba(frame, np.asarray(self._sensors(st)), m, H - m - side - p(50))
        cx0, cx1 = m + p(470), rx  # free strip between the left and right columns
        if st.result:
            ban = self._banner(st.result, 0.0, cx1 - cx0 - 2 * m, big=True)
        elif st.banner:
            ban = self._banner(st.banner[0], st.banner[1], cx1 - cx0 - 2 * m)
        else:
            ban = None
        if ban is not None:
            blend_rgba(frame, np.asarray(ban), (cx0 + cx1 - ban.width) // 2, m)
        if st.scenario:
            sc = self._label(st.scenario, CYAN)
            blend_rgba(frame, np.asarray(sc), (cx0 + cx1 - sc.width) // 2, m + p(64))
        return frame


def _dashed_rect(d: ImageDraw.ImageDraw, box, color, dash: int, width: int) -> None:
    x0, y0, x1, y1 = box
    for xa in range(x0, x1, 2 * dash):
        d.line((xa, y0, min(xa + dash, x1), y0), fill=color, width=width)
        d.line((xa, y1, min(xa + dash, x1), y1), fill=color, width=width)
    for ya in range(y0, y1, 2 * dash):
        d.line((x0, ya, x0, min(ya + dash, y1)), fill=color, width=width)
        d.line((x1, ya, x1, min(ya + dash, y1)), fill=color, width=width)
