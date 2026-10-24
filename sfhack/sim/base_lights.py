"""The base's navigation lights as the simulated REALITY (SIMULATOR SIDE: truth positions, the base clock).

    lights = TrueLights.load(world_dir, seed)     # world/base_lights.json + world/stray_lights.json
    lights.emitters(t_sim)                        # [(id, xyz, intensity towards ... , on_fraction)] for a camera
    anim = BlinkAnimator(model, lights)           # 3D viewer: the emissive geoms blink their codes
    anim.update(t_sim)

The lights run on the BASE clock: t_base = t_sim + clock_offset_s (seeded, unknown to the drone). Stray lights
(world/stray_lights.json): village lamps (steady) and a fire (flickering) - visible, but never a valid code.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ..world.base import BaseLight, BaseLights, az_deg

ON_RGBA = np.array([1.0, 0.25, 0.2, 1.0])
OFF_RGBA = np.array([0.18, 0.06, 0.05, 1.0])
GLOW_RGBA = np.array([1.0, 0.2, 0.15, 0.25])


@dataclass
class StrayLight:
    id: str
    kind: str  # lamp | fire
    enu: np.ndarray
    intensity: float
    mode: str  # steady | flicker


class TrueLights:
    def __init__(self, lights: BaseLights, strays: list[StrayLight], clock_offset_s: float = 0.0,
                 seed: int = 0) -> None:
        self.lights = lights
        self.strays = strays
        self.clock_offset_s = clock_offset_s
        self.pos = np.array([lt.enu for lt in lights.lights], float).reshape(-1, 3)
        self.rng = np.random.default_rng(seed + 9500)
        self.enabled = True  # L key: lights on / off (the base switches them on for a mission)

    @classmethod
    def load(cls, world_dir: Path | str, seed: int = 0) -> "TrueLights":
        w = Path(world_dir)
        lights = BaseLights.from_dict(json.loads((w / "base_lights.json").read_text(encoding="utf-8")))
        strays = []
        sp = w / "stray_lights.json"
        if sp.exists():
            for s in json.loads(sp.read_text(encoding="utf-8"))["lights"]:
                strays.append(StrayLight(s["id"], s["kind"], np.asarray(s["enu"], float), float(s["intensity"]),
                                         s["mode"]))
        offset = float(np.random.default_rng(seed + 9400).uniform(0.0, lights.bit_s * lights.code_bits))
        return cls(lights, strays, offset, seed)

    def t_base(self, t_sim: float) -> float:
        return t_sim + self.clock_offset_s

    def on_fraction(self, lt: BaseLight, t_sim: float, exposure_s: float = 0.0) -> float:
        """Fraction of the exposure [t - e/2, t + e/2] the light is on."""
        if not self.enabled:
            return 0.0
        tb = self.t_base(t_sim)
        if exposure_s <= 0.0:
            return float(lt.is_on(tb, self.lights.bit_s))
        ts = tb + exposure_s * (np.arange(4) + 0.5) / 4 - exposure_s / 2
        return float(np.mean([lt.is_on(t, self.lights.bit_s) for t in ts]))

    def emitters(self, t_sim: float, eye: np.ndarray, exposure_s: float = 0.0) -> list[dict[str, Any]]:
        """Every light with its radiant intensity towards `eye` (relative units; 1 = a fence lamp on its axis)."""
        out = []
        for lt, p in zip(self.lights.lights, self.pos):
            d = eye - p
            horiz = float(np.hypot(d[0], d[1]))
            gain = lt.lobe(az_deg(d[0], d[1]), math.degrees(math.atan2(d[2], max(horiz, 1e-6))))
            out.append({"id": lt.id, "kind": lt.ring, "xyz": p, "I": gain * self.on_fraction(lt, t_sim, exposure_s),
                        "I_on": gain if self.enabled else 0.0, "nav": True})
        for s in self.strays:
            f = 1.0 if s.mode == "steady" else float(np.clip(0.65 + 0.35 * self.rng.normal(), 0.15, 1.3))
            out.append({"id": s.id, "kind": s.kind, "xyz": s.enu, "I": s.intensity * f, "I_on": s.intensity * f,
                        "nav": False})
        return out


class BlinkAnimator:
    """Blinks the emissive light geoms of the MuJoCo world (world/scene.py `_add_base`) on the base clock."""

    def __init__(self, model: mujoco.MjModel, lights: TrueLights) -> None:
        self.model, self.lights = model, lights
        self.ids = []
        for lt in lights.lights.lights:
            try:
                self.ids.append((lt, model.geom(f"baselight_{lt.id}").id, model.geom(f"baseglow_{lt.id}").id))
            except KeyError:
                continue

    def update(self, t_sim: float) -> None:
        for lt, gid, hid in self.ids:
            on = self.lights.on_fraction(lt, t_sim) > 0.5
            self.model.geom_rgba[gid] = ON_RGBA if on else OFF_RGBA
            self.model.geom_rgba[hid] = GLOW_RGBA if on else 0.0
