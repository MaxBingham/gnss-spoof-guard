"""The base's UMA-16 bearing to the enemy drone (simulated reality, harness side).

Azimuth from the array to the TRUE enemy position plus Gaussian noise with sigma growing with the distance:
sigma = `sigma0_deg` + `per_100m_deg` * d / 100 m, capped at `max_deg` (a farther engine is quieter against the
background, so the DOA scatters more; demos/42_doa_eval.py measured ~2-3 deg rms close in). Each measurement carries
its sigma, as the live path's BearingEstimate does.

ASSUMPTION (sim only): the base tells our drone and the enemy apart by their different rotor sound (blade-pass
frequency / harmonics), so this bearing is always the enemy's, never our drone's, even when both are close together.
The real UMA-16 path (sfhack/acoustic) does not do that separation yet.
"""

from __future__ import annotations

import math

import numpy as np


class BaseAcoustic:
    def __init__(self, mast_xy=(0.0, 0.0), sigma0_deg: float = 2.0, per_100m_deg: float = 1.5, max_deg: float = 12.0,
                 rate_hz: float = 1.0, seed: int = 0) -> None:
        self.mast = np.asarray(mast_xy, float)[:2].copy()
        self.sigma0, self.per_100m, self.max = sigma0_deg, per_100m_deg, max_deg
        self.period = 1.0 / rate_hz
        self.rng = np.random.default_rng(seed + 6600)
        self.t_next: float | None = None
        self.last: tuple[float, float, float] | None = None  # (t, az_deg, sigma_deg)

    def sigma(self, dist: float) -> float:
        return min(self.max, self.sigma0 + self.per_100m * dist / 100.0)

    def measure(self, t: float, enemy_xy) -> tuple[float, float]:
        d = np.asarray(enemy_xy, float)[:2] - self.mast
        s = self.sigma(float(np.linalg.norm(d)))
        az = (math.degrees(math.atan2(d[0], d[1])) + self.rng.normal(0.0, s)) % 360.0
        self.last = (t, az, s)
        return az, s

    def due(self, t: float) -> bool:
        if self.t_next is None:
            self.t_next = t
        if t + 1e-9 < self.t_next:
            return False
        self.t_next += self.period
        return True
