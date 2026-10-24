"""BearingEstimate: the only thing the drone gets from the base microphone array (RECON_PLAN interface contract).

Live (real UMA-16, `demos/40_uma16_live.py`) and simulated (`sfhack/acoustic/sim_array.py`) bearings use the
same type. JSON file form: `bearing.json` with the same keys.

Conventions:
- `az_deg`: degrees clockwise from north (compass convention), [0, 360).
- `el_deg`: degrees above the horizon.
- `sigma_az_deg`, `sigma_el_deg`: 1-sigma uncertainty (degrees) the drone uses to size its search cone.
- `array_enu`: array position in the mission's local ENU frame (m); None if unknown (the live path writes the
  configured value or None; the mission puts the base UMA-16 position from `onboard/base.json` there).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class BearingEstimate:
    t: float                    # unix time of the estimate (s)
    az_deg: float               # clockwise from north
    el_deg: float               # above horizon
    sigma_az_deg: float
    sigma_el_deg: float
    label: str = "unknown"      # e.g. "engine", "unknown"
    confidence: float = 0.0     # 0..1
    array_enu: tuple[float, float, float] | None = None

    def __post_init__(self):
        self.az_deg = float(self.az_deg) % 360.0
        if self.array_enu is not None:
            self.array_enu = tuple(float(v) for v in self.array_enu)

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["array_enu"] is not None:
            d["array_enu"] = list(d["array_enu"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BearingEstimate":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BearingEstimate":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
