"""`Conditions`: every degradation lever for one moment of a flight, in one dataclass.

This is the simulated REALITY (what the world does to the sensors and the airframe). The simulator reads
it; onboard code (autopilot, estimation, control, planning, perception) never does
(tests/test_truth_separation.py), so the drone cannot "know" that it is being spoofed or jammed.

Directions are compass degrees: 0 = north (+y), 90 = east (+x), clockwise. Wind and spoof directions
are the direction the force / false position moves TOWARD.

Set it from config (`Config.conditions`), from the CLI (`sfhack.degradation.cli`), or at runtime
(`FlightSystem.set_conditions` / `update_conditions`, or a `ConditionsMsg` on the `conditions` bus topic).
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any

GNSS_MODES = ("off", "ok", "jam", "drag", "jump")
WEATHER_KINDS = ("clear", "rain", "fog", "dust")


@dataclass
class GnssConditions:
    """off: no receiver (the GNSS-free default). ok: honest fixes. jam: no fix.
    drag: the reported position slides away at `drag_rate` m/s toward `drag_heading_deg` (grows while in drag).
    jump: the reported position is offset by `jump_m` toward `jump_heading_deg` at once.
    Leaving drag / jump clears the false offset."""

    mode: str = "off"
    drag_rate: float = 1.0  # m/s
    drag_heading_deg: float = 90.0
    jump_m: float = 50.0
    jump_heading_deg: float = 90.0


@dataclass
class WeatherConditions:
    """One intensity in [0, 1] plus a kind. Degrades the down camera (contrast / visibility, pixel noise,
    frame dropouts) and the range sensors (range noise, return dropouts). `clear` ignores the intensity."""

    kind: str = "clear"
    intensity: float = 0.0


@dataclass
class WindConditions:
    """Steady wind `speed` m/s blowing toward `direction_deg`, turning at `rotation_deg_s`, plus gusts
    (Ornstein-Uhlenbeck air-speed fluctuation with std `gust` m/s). Applied as a drag force in MuJoCo."""

    speed: float = 0.0
    direction_deg: float = 0.0
    rotation_deg_s: float = 0.0
    gust: float = 0.0


@dataclass
class CameraConditions:
    """lost: the down camera delivers black frames. drop_rate: fraction of frames never delivered."""

    lost: bool = False
    drop_rate: float = 0.0


@dataclass
class LidarConditions:
    """Applies to the downward rangefinder and the nadir multi-beam lidar.
    lost: no returns at all. dropout: fraction of individual returns missing."""

    lost: bool = False
    dropout: float = 0.0


@dataclass
class Conditions:
    gnss: GnssConditions = field(default_factory=GnssConditions)
    weather: WeatherConditions = field(default_factory=WeatherConditions)
    wind: WindConditions = field(default_factory=WindConditions)
    camera: CameraConditions = field(default_factory=CameraConditions)
    lidar: LidarConditions = field(default_factory=LidarConditions)

    def __post_init__(self) -> None:
        self.validate()

    def copy(self) -> Conditions:
        return copy.deepcopy(self)

    def validate(self) -> None:
        if self.gnss.mode not in GNSS_MODES:
            raise ValueError(f"gnss.mode {self.gnss.mode!r} not in {GNSS_MODES}")
        if self.weather.kind not in WEATHER_KINDS:
            raise ValueError(f"weather.kind {self.weather.kind!r} not in {WEATHER_KINDS}")
        for name, v in (("weather.intensity", self.weather.intensity), ("camera.drop_rate", self.camera.drop_rate),
                        ("lidar.dropout", self.lidar.dropout)):
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} = {v} must be in [0, 1]")
        for name, v in (("wind.speed", self.wind.speed), ("wind.gust", self.wind.gust),
                        ("gnss.drag_rate", self.gnss.drag_rate), ("gnss.jump_m", self.gnss.jump_m)):
            if v < 0 or not math.isfinite(v):
                raise ValueError(f"{name} = {v} must be finite and >= 0")

    # ------------------------------------------------------------------ dotted access

    def get(self, path: str) -> Any:
        group, key = _split(path)
        return getattr(getattr(self, group), key)

    def set(self, path: str, value: Any) -> Conditions:
        """Set one lever by dotted path ('gnss.mode', 'wind.speed', ...). Strings are coerced to the field
        type, so CLI text works. Validates, and returns self for chaining."""
        group, key = _split(path)
        sub = getattr(self, group)
        types = {f.name: f.type for f in fields(sub)}
        old = getattr(sub, key)
        setattr(sub, key, _coerce(value, types[key], path))
        try:
            self.validate()
        except ValueError:
            setattr(sub, key, old)
            raise
        return self

    def update(self, assignments: dict[str, Any] | str | None = None, **kw: Any) -> Conditions:
        """Apply several levers: a dict / keyword args of dotted paths (use '__' for '.' in keywords:
        `update(gnss__mode='drag')`), or an assignment string 'gnss.mode=drag,wind.speed=6'."""
        items: dict[str, Any] = {}
        if isinstance(assignments, str):
            items.update(parse_assignments(assignments))
        elif assignments:
            items.update(assignments)
        items.update({k.replace("__", "."): v for k, v in kw.items()})
        for path, value in items.items():
            if "." in path:
                self.set(path, value)
            else:
                for p, v in expand_shorthand(path, str(value)).items():
                    self.set(p, v)
        return self

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, dict[str, Any]]) -> Conditions:
        c = cls()
        for group, sub in d.items():
            for key, value in sub.items():
                c.set(f"{group}.{key}", value)
        return c

    def summary(self) -> str:
        """Short human-readable list of the levers that are not at their default (for logs / banners)."""
        parts = []
        g = self.gnss
        if g.mode == "drag":
            parts.append(f"GNSS DRAG {g.drag_rate:g} m/s @ {g.drag_heading_deg:.0f} deg")
        elif g.mode == "jump":
            parts.append(f"GNSS JUMP {g.jump_m:g} m @ {g.jump_heading_deg:.0f} deg")
        elif g.mode != "off":
            parts.append(f"GNSS {g.mode.upper()}")
        if self.weather.kind != "clear" and self.weather.intensity > 0:
            parts.append(f"{self.weather.kind.upper()} {self.weather.intensity:.2f}")
        w = self.wind
        if w.speed > 0 or w.gust > 0:
            rot = f" rot {w.rotation_deg_s:g} deg/s" if w.rotation_deg_s else ""
            gust = f" gust {w.gust:g}" if w.gust else ""
            parts.append(f"WIND {w.speed:g} m/s @ {w.direction_deg:.0f} deg{rot}{gust}")
        if self.camera.lost:
            parts.append("CAMERA LOST")
        elif self.camera.drop_rate > 0:
            parts.append(f"CAMERA DROP {self.camera.drop_rate:.0%}")
        if self.lidar.lost:
            parts.append("LIDAR LOST")
        elif self.lidar.dropout > 0:
            parts.append(f"LIDAR DROPOUT {self.lidar.dropout:.0%}")
        return ", ".join(parts) or "nominal"


@dataclass
class ConditionsMsg:
    """Bus message on topic `conditions` (sfhack.bus.CONDITIONS): the simulator adopts a copy."""

    t: float
    conditions: Conditions


def compass_unit(direction_deg: float) -> tuple[float, float]:
    """Compass degrees (0 north, 90 east) -> ENU unit vector (x east, y north)."""
    a = math.radians(direction_deg)
    return math.sin(a), math.cos(a)


def parse_assignments(spec: str) -> dict[str, str]:
    """'gnss.mode=drag, wind.speed=6' -> {'gnss.mode': 'drag', 'wind.speed': '6'}."""
    out: dict[str, str] = {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        if "=" not in item:
            raise ValueError(f"expected path=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def expand_shorthand(group: str, value: str) -> dict[str, str]:
    """Group-level shorthands (CLI / schedules):
    gnss=off|ok|jam|drag[:RATE[@HEADING]]|jump[:METERS[@HEADING]]   weather=KIND[:INTENSITY]
    wind=SPEED[@DIRECTION]   camera=ok|lost|DROP_RATE   lidar=ok|lost|DROPOUT"""
    v = value.strip().lower()
    head, _, rest = v.partition(":")
    a, _, b = rest.partition("@")
    if group == "gnss":
        out = {"gnss.mode": head}
        if head in ("drag", "jump"):
            names = ("drag_rate", "drag_heading_deg") if head == "drag" else ("jump_m", "jump_heading_deg")
            if a:
                out[f"gnss.{names[0]}"] = a
            if b:
                out[f"gnss.{names[1]}"] = b
        return out
    if group == "weather":
        return {"weather.kind": head, "weather.intensity": a or ("0" if head == "clear" else "0.5")}
    if group == "wind":
        speed, _, direction = v.partition("@")
        return {"wind.speed": speed, **({"wind.direction_deg": direction} if direction else {})}
    if group in ("camera", "lidar"):
        rate = "drop_rate" if group == "camera" else "dropout"
        if v in ("lost", "ok"):
            return {f"{group}.lost": str(v == "lost"), **({f"{group}.{rate}": "0"} if v == "ok" else {})}
        return {f"{group}.lost": "false", f"{group}.{rate}": v}
    raise KeyError(f"no shorthand for {group!r}")


def _split(path: str) -> tuple[str, str]:
    parts = path.split(".")
    groups = {f.name for f in fields(Conditions)}
    if len(parts) != 2 or parts[0] not in groups:
        raise KeyError(f"unknown lever {path!r} (groups: {sorted(groups)})")
    sub_fields = {f.name for f in fields(Conditions.__dataclass_fields__[parts[0]].default_factory())}
    if parts[1] not in sub_fields:
        raise KeyError(f"unknown lever {path!r} ({parts[0]} has {sorted(sub_fields)})")
    return parts[0], parts[1]


def _coerce(value: Any, typ: Any, path: str) -> Any:
    typ = typ if isinstance(typ, str) else getattr(typ, "__name__", str(typ))
    if typ == "bool":
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ("1", "true", "yes", "on"):
                return True
            if v in ("0", "false", "no", "off"):
                return False
            raise ValueError(f"{path}: not a boolean: {value!r}")
        return bool(value)
    if typ == "float":
        return float(value)
    if typ == "str":
        return str(value).strip().lower()
    raise TypeError(f"{path}: unsupported field type {typ}")
