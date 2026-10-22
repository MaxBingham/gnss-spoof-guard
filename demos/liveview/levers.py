"""Live lever controls on top of `sfhack.degradation.Conditions`: key bindings, the one-key spoof scenario,
and the adapter that hands the levers to the flight.

The adapter is the only place that knows how the simulator consumes the levers. Once the simulator reads
`Conditions` (`FlightSystem.set_conditions`), every lever is real. Until then only wind and camera loss
reach the physics (through the legacy disturbance topic); the other levers are drawn but flagged
"visual only" on screen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sfhack import bus as topics
from sfhack.degradation import Conditions, ConditionsMsg, compass_unit

GNSS_CYCLE = ("ok", "drag", "jump", "jam", "off")
WEATHER_STEPS = (0.0, 0.35, 0.65, 1.0)
WEATHER_KIND_CYCLE = ("fog", "rain", "dust")
WIND_STEPS = (0.0, 4.0, 8.0, 12.0)
GUST = 2.5  # m/s, gust std when gusts are on

# The spoof scenario: (seconds after the key press, lever assignments, banner).
SCENARIO: list[tuple[float, dict[str, Any], str]] = [
    (0.0, {"gnss": "drag:2.0@90"}, "SPOOF ATTACK: GNSS fix dragged east at 2 m/s"),
    (14.0, {"weather": "fog:0.65", "wind": "7@270", "wind.gust": GUST}, "WEATHER: fog rolls in, 7 m/s crosswind"),
    (24.0, {"camera": "lost"}, "CAMERA LOST: vision localizer is blind"),
    (34.0, {"camera": "ok", "weather": "clear", "wind": "0", "wind.gust": 0.0}, "CAMERA BACK, weather clears"),
    (40.0, {"gnss": "ok"}, "SPOOFER OFF: GNSS honest again"),
]


def _cycle(seq, cur):
    return seq[(seq.index(cur) + 1) % len(seq)] if cur in seq else seq[0]


def _next_step(steps, cur: float) -> float:
    return next((s for s in steps if s > cur + 1e-6), steps[0])


@dataclass
class Lever:
    key: str
    name: str
    path: str  # Conditions group (for the "visual only" flag)
    act: Callable[[Conditions], None]
    value: Callable[[Conditions], str]
    level: Callable[[Conditions], int]  # 0 nominal, 1 degraded, 2 lost / under attack


def _gnss_value(c: Conditions) -> str:
    g = c.gnss
    return {"drag": f"DRAG {g.drag_rate:g} m/s", "jump": f"JUMP {g.jump_m:g} m"}.get(g.mode, g.mode.upper())


def _weather(c: Conditions) -> None:
    w = c.weather
    i = _next_step(WEATHER_STEPS, w.intensity if w.kind != "clear" else 0.0)
    c.set("weather.kind", "clear" if i == 0 else (w.kind if w.kind != "clear" else "fog"))
    c.set("weather.intensity", i)


def _weather_kind(c: Conditions) -> None:
    w = c.weather
    c.set("weather.kind", _cycle(WEATHER_KIND_CYCLE, w.kind))
    if w.intensity == 0:
        c.set("weather.intensity", WEATHER_STEPS[2])


LEVERS: list[Lever] = [
    Lever("g", "GNSS", "gnss", lambda c: c.set("gnss.mode", _cycle(GNSS_CYCLE, c.gnss.mode)), _gnss_value,
          lambda c: {"ok": 0, "off": 1, "jam": 2, "drag": 2, "jump": 2}[c.gnss.mode]),
    Lever("f", "Weather", "weather", _weather,
          lambda c: "CLEAR" if c.weather.kind == "clear" or c.weather.intensity == 0
          else f"{c.weather.kind.upper()} {c.weather.intensity:.2f}",
          lambda c: 0 if c.weather.kind == "clear" or c.weather.intensity == 0 else (2 if c.weather.intensity > 0.6 else 1)),
    Lever("t", "Weather kind", "weather", _weather_kind, lambda c: c.weather.kind.upper(), lambda c: 0),
    Lever("w", "Wind", "wind", lambda c: c.set("wind.speed", _next_step(WIND_STEPS, c.wind.speed)),
          lambda c: f"{c.wind.speed:g} m/s" if c.wind.speed else "CALM",
          lambda c: 0 if c.wind.speed == 0 else (2 if c.wind.speed >= 10 else 1)),
    Lever("d", "Wind direction", "wind", lambda c: c.set("wind.direction_deg", (c.wind.direction_deg + 45) % 360),
          lambda c: f"toward {c.wind.direction_deg:03.0f} deg", lambda c: 0),
    Lever("u", "Gusts", "wind", lambda c: c.set("wind.gust", 0.0 if c.wind.gust else GUST),
          lambda c: f"{c.wind.gust:g} m/s" if c.wind.gust else "OFF", lambda c: 1 if c.wind.gust else 0),
    Lever("c", "Camera", "camera", lambda c: c.set("camera.lost", not c.camera.lost),
          lambda c: "LOST" if c.camera.lost else "OK", lambda c: 2 if c.camera.lost else 0),
    Lever("l", "Lidar", "lidar", lambda c: c.set("lidar.lost", not c.lidar.lost),
          lambda c: "LOST" if c.lidar.lost else "OK", lambda c: 2 if c.lidar.lost else 0),
]
LEVER_BY_KEY = {lv.key: lv for lv in LEVERS}


class LeverDeck:
    """Current levers + scenario playback + the adapter to the flight. Harness code (never onboard)."""

    def __init__(self, initial: Conditions, schedule: list[tuple[float, dict[str, str]]] | None = None) -> None:
        self.initial = initial.copy()
        self.c = initial.copy()
        self.schedule = list(schedule or [])
        self.scenario_t0: float | None = None
        self.scenario_step = 0
        self.banner: tuple[float, str] | None = None  # (sim time, text)
        self._legacy_cam_lost = False
        self.dirty = True

    # ------------------------------------------------------------------ operator input

    def press(self, key: str, t: float) -> str:
        key = key.lower()
        if key == "x":
            self.c = self.initial.copy()
            self.scenario_t0 = None
            msg = "all levers back to nominal"
        elif key == "p":
            self.scenario_t0, self.scenario_step = t, 0
            msg = "scenario: spoof attack"
        elif key in LEVER_BY_KEY:
            lv = LEVER_BY_KEY[key]
            lv.act(self.c)
            msg = f"{lv.name}: {lv.value(self.c)}"
        else:
            return ""
        self.dirty = True
        self.banner = (t, msg.upper() if key != "p" else SCENARIO[0][2])
        return msg

    def tick(self, t: float) -> None:
        while self.schedule and t >= self.schedule[0][0]:
            self.c.update(self.schedule.pop(0)[1])
            self.dirty = True
            self.banner = (t, self.c.summary().upper())
        if self.scenario_t0 is not None:
            while self.scenario_step < len(SCENARIO) and t >= self.scenario_t0 + SCENARIO[self.scenario_step][0]:
                _, assign, text = SCENARIO[self.scenario_step]
                self.c.update(dict(assign))
                self.banner = (t, text)
                self.scenario_step += 1
                self.dirty = True
            if self.scenario_step >= len(SCENARIO):
                self.scenario_t0 = None

    def scenario_progress(self) -> str:
        if self.scenario_t0 is None:
            return ""
        return f"SCENARIO {self.scenario_step}/{len(SCENARIO)}"

    # ------------------------------------------------------------------ adapter

    @staticmethod
    def integrated(fs) -> bool:
        """True once the simulator consumes `Conditions` (then every lever acts on the sensors)."""
        return hasattr(fs, "set_conditions")

    def sim_effect(self, fs, group: str) -> bool:
        return self.integrated(fs) or group in ("wind", "camera")

    def apply(self, fs) -> None:
        if not self.dirty:
            return
        self.dirty = False
        c = self.c.copy()
        if self.integrated(fs):
            fs.set_conditions(c)
            return
        fs.bus.publish(topics.CONDITIONS, ConditionsMsg(fs.t, c))
        if c.camera.lost != self._legacy_cam_lost:
            self._legacy_cam_lost = c.camera.lost
            if c.camera.lost:
                fs.disturb("camera_blackout", duration=0.0)  # until cleared
            else:
                fs.disturb("clear")  # also zeroes the wind; re-applied just below
        ux, uy = compass_unit(c.wind.direction_deg)
        fs.disturb("wind", (c.wind.speed * ux, c.wind.speed * uy))
