"""Stub base lights for tests and missions without recon-world's base (SIMULATOR SIDE: makes TRUE positions).

The simulated rear NIR camera is recon-world's sim/rear_camera.py; this module keeps the light layout helpers
recon-nav's tests use: cyclic blink codes, a square fence ring of outward-facing lights, a surveyed copy.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

from typing import Callable

import numpy as np

from ..perception.light_localizer import BaseLight

LIGHT_HEIGHT_M = 3.0


def make_codes(n: int, bits: int = 10, min_ones: int = 5, max_ones: int = 7) -> list[str]:
    """n cyclic codes, no two equal under rotation, cyclic Hamming distance >= 2 between any rotations
    (one wrong sample cannot turn one light into another), never all on (a steady light matches none).
    8 bits give only 10 such codes with 4-6 ones (6 with 5-6): 10 bits (1 s at 10 Hz) give 22 with 5-7 ones."""
    def rots(c: str) -> list[str]:
        return [c[i:] + c[:i] for i in range(len(c))]

    out: list[str] = []
    for combo in itertools.product("01", repeat=bits):
        c = "".join(combo)
        if not (min_ones <= c.count("1") <= max_ones) or "000" in c + c[:2]:
            continue
        if any(min(sum(a != b for a, b in zip(r, o)) for r in rots(c)) < 2 for o in out):
            continue
        if any(r < c for r in rots(c)):  # canonical rotation only
            continue
        out.append(c)
        if len(out) == n:
            return out
    raise ValueError(f"only {len(out)} codes of {bits} bits")


def stub_base_lights(center: np.ndarray, ground_z: Callable[[float, float], float], half: float = 75.0,
                     n_fence: int = 12, n_outer: int = 0, outer_r: float = 400.0, bit_hz: float = 10.0,
                     beam_half_deg: float = 80.0) -> list[BaseLight]:
    """TRUE lights: n_fence on a square fence (2 * half wide) every 360 / n_fence deg around the centre, beams
    facing outward; optional outer ring of posts at outer_r. z = true ground + LIGHT_HEIGHT_M."""
    codes = make_codes(n_fence + n_outer)
    out = []
    for k in range(n_fence):
        a = 2 * math.pi * k / n_fence
        d = np.array([math.cos(a), math.sin(a)])
        s = half / max(abs(d[0]), abs(d[1]))
        xy = center[:2] + d * s
        out.append(BaseLight(f"F{k:02d}", np.array([xy[0], xy[1], ground_z(*xy) + LIGHT_HEIGHT_M]), codes[k], bit_hz,
                             "fence", math.degrees(a), beam_half_deg))
    for k in range(n_outer):
        a = 2 * math.pi * (k + 0.5) / n_outer
        xy = center[:2] + outer_r * np.array([math.cos(a), math.sin(a)])
        out.append(BaseLight(f"O{k:02d}", np.array([xy[0], xy[1], ground_z(*xy) + LIGHT_HEIGHT_M]), codes[n_fence + k],
                             bit_hz, "outer", math.degrees(a), beam_half_deg))
    return out


def surveyed(lights: list[BaseLight], std_m: float, seed: int) -> list[BaseLight]:
    """The onboard list: true positions + survey error."""
    r = np.random.default_rng(seed + 9100)
    return [BaseLight(lt.id, lt.pos + r.normal(0.0, std_m, 3), lt.code, lt.bit_hz, lt.ring, lt.facing_deg,
                      lt.beam_half_deg) for lt in lights]


def lights_to_json(lights: list[BaseLight], path: Path) -> None:
    path.write_text(json.dumps({"lights": [{"id": lt.id, "enu": [round(float(v), 3) for v in lt.pos], "code": lt.code,
                                            "bit_hz": lt.bit_hz, "ring": lt.ring, "facing_deg": lt.facing_deg,
                                            "beam_half_deg": lt.beam_half_deg} for lt in lights]}, indent=1),
                    encoding="utf-8")
