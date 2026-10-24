"""The base (RECON_PLAN "Base"): fence, navigation lights with blink codes, landing pad, UMA-16 mast.

ONBOARD-SAFE: types, the onboard package readers and the blink-code logic only. The truth layout is designed and
written by the simulator side (`sfhack/sim/recon_world.py`); onboard code reads only the SURVEYED copy:

    lights = load_base_lights(onboard_dir)        # onboard/base_lights.json: surveyed positions + codes
    base = load_base(onboard_dir)                 # onboard/base.json: pad position / heading, UMA-16, fence
    light.is_on(t_base)                           # blink state on the BASE clock (the drone does not know the
                                                  # offset between its clock and the base clock)
    decode_code(bits)                             # 8 observed bits (any phase) -> canonical code -> light id

Blink codes: every light repeats an 8-bit pattern, one bit per `bit_s` (0.1 s = 10 Hz bit rate). The drone
samples at an unknown phase, so codes are aperiodic necklaces (Lyndon words) of weight 3-5: each cyclic rotation
of one code differs from every rotation of every other code, and a code is never all on / all off. Canonical form
= the lexicographically smallest rotation; `decode_code` rotates the observation into it. A camera at >= 20 Hz
sees every bit at least twice.

Frame: mission ENU (x east, y north, z up, z datum = ground under A). Azimuths: degrees clockwise from north.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CODE_BITS = 8
BIT_S = 0.1  # 10 Hz bit rate


def _rotations(code: str) -> list[str]:
    return [code[i:] + code[:i] for i in range(len(code))]


def canonical(code: str) -> str:
    """Smallest cyclic rotation (the drone sees a code at an arbitrary phase)."""
    return min(_rotations(code))


def lyndon_codes(n: int = CODE_BITS, min_ones: int = 3, max_ones: int = 5) -> list[str]:
    """All aperiodic binary necklaces of length n with min_ones..max_ones ones, canonical form, sorted.
    n = 8, weight 3-5: 22 codes (enough for 12 fence + 8 outer lights)."""
    out = []
    for v in range(1 << n):
        s = format(v, f"0{n}b")
        if not min_ones <= s.count("1") <= max_ones or canonical(s) != s:
            continue
        if any(s == s[k:] + s[:k] for k in range(1, n)):  # periodic (e.g. 01010101): ambiguous phase
            continue
        out.append(s)
    return out


def decode_code(bits: str | list[int], codes: dict[str, str]) -> str | None:
    """Observed `CODE_BITS` bits (any starting phase) -> light id, or None. `codes`: canonical code -> id."""
    s = "".join(str(int(b)) for b in bits) if not isinstance(bits, str) else bits
    if len(s) != CODE_BITS:
        return None
    return codes.get(canonical(s))


@dataclass
class BaseLight:
    id: str  # "F00".."F11" fence ring (F00 north, clockwise), "O00".. outer ring (clockwise)
    ring: str  # "fence" | "outer"
    enu: tuple[float, float, float]  # emitter position (m, mission ENU; z absolute, datum = ground under A)
    code: str  # canonical 8-bit blink code, "1" = on
    aim_az_deg: float = 0.0  # beam centre, clockwise from north (aimed outward from the base centre)
    aim_el_deg: float = 3.0  # beam centre above the horizon ("aimed low")
    beam_h_deg: float = 70.0  # horizontal half width (1/e^2 of the intensity)
    beam_v_deg: float = 25.0  # vertical half width (1/e^2)
    intensity: float = 1.0  # relative radiant intensity on the beam axis (1 = a 10 W 850 nm LED lamp)
    phase: int = 0  # the code starts this many bits late on the base clock (lights are staggered: canonical codes
    #                 all begin with 0s, so in phase every light would be dark at the same time)

    def is_on(self, t_base: float, bit_s: float = BIT_S) -> bool:
        k = (int(math.floor(t_base / bit_s)) - self.phase) % len(self.code)
        return self.code[k] == "1"

    def lobe(self, az_deg: float, el_deg: float) -> float:
        """Relative intensity towards a direction (azimuth / elevation seen FROM the light). Gaussian beam."""
        da = (az_deg - self.aim_az_deg + 180.0) % 360.0 - 180.0
        de = el_deg - self.aim_el_deg
        if abs(da) >= 120.0:  # behind the lamp housing
            return 0.0
        return self.intensity * math.exp(-2.0 * (da / self.beam_h_deg) ** 2 - 2.0 * (de / self.beam_v_deg) ** 2)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["enu"] = [round(float(v), 3) for v in self.enu]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BaseLight":
        kw = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        kw["enu"] = tuple(float(v) for v in kw["enu"])
        return cls(**kw)


@dataclass
class BaseLights:
    lights: list[BaseLight]
    bit_s: float = BIT_S
    code_bits: int = CODE_BITS
    survey_sigma_m: float = 0.0  # 1-sigma survey error of the positions in this file (0 for the truth file)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def codes(self) -> dict[str, str]:
        """canonical code -> light id (the decoder table)."""
        return {canonical(lt.code): lt.id for lt in self.lights}

    def by_id(self, light_id: str) -> BaseLight:
        return next(lt for lt in self.lights if lt.id == light_id)

    def ring(self, name: str) -> list[BaseLight]:
        return [lt for lt in self.lights if lt.ring == name]

    def to_dict(self) -> dict[str, Any]:
        return {"format": "sfhack-base-lights/1",
                "frame": "mission ENU (x east, y north, z up), z datum = ground under A; az clockwise from north",
                "bit_s": self.bit_s, "code_bits": self.code_bits, "survey_sigma_m": self.survey_sigma_m,
                "code_note": "canonical (smallest-rotation) cyclic 8-bit code, '1' = on, one bit per bit_s on the "
                             "base clock; the drone's clock offset to the base clock is unknown",
                "meta": self.meta, "lights": [lt.to_dict() for lt in self.lights]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BaseLights":
        return cls([BaseLight.from_dict(x) for x in d["lights"]], float(d.get("bit_s", BIT_S)),
                   int(d.get("code_bits", CODE_BITS)), float(d.get("survey_sigma_m", 0.0)), d.get("meta", {}))

    def save(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p


@dataclass
class BaseInfo:
    """onboard/base.json: what the drone knows about its base (surveyed)."""
    center_enu: tuple[float, float, float]
    pad_enu: tuple[float, float, float]  # pad centre on the pad surface
    pad_yaw_deg: float  # pad axis (the drone's takeoff heading), clockwise from north
    pad_size_m: float
    uma16_enu: tuple[float, float, float]  # array centre
    uma16_x_axis_az_deg: float  # azimuth of the array's x axis (live path: array az 0 -> this compass az)
    fence_half_m: float
    survey_sigma_m: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("center_enu", "pad_enu", "uma16_enu"):
            d[k] = [round(float(v), 3) for v in d[k]]
        return {"format": "sfhack-base/1", "frame": "mission ENU, z datum = ground under A; az clockwise from north",
                **d}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BaseInfo":
        kw = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        for k in ("center_enu", "pad_enu", "uma16_enu"):
            kw[k] = tuple(float(v) for v in kw[k])
        return cls(**kw)

    def save(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p


def load_base_lights(onboard_dir: Path | str) -> BaseLights:
    return BaseLights.from_dict(json.loads((Path(onboard_dir) / "base_lights.json").read_text(encoding="utf-8")))


def load_base(onboard_dir: Path | str) -> BaseInfo:
    return BaseInfo.from_dict(json.loads((Path(onboard_dir) / "base.json").read_text(encoding="utf-8")))


def az_deg(dx: float, dy: float) -> float:
    """ENU vector -> azimuth clockwise from north, [0, 360)."""
    return math.degrees(math.atan2(dx, dy)) % 360.0
