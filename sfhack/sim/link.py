"""Radio link between the base and our drone (simulated reality, simulator / harness side).

Two-state Markov model: up / down periods with exponential lengths, mean down `mean_down_s`, mean up chosen so the
long-run availability is `availability` (~80 %, README "Current idea"). While up, the drone's ReportOutbox drains to
the base and the base's queued messages (new UMA-16 bearings, recall) reach the drone; while down both wait.

STUB with the same idea as the dashboard's link (no link code on origin/jean when this was written): swap the class
for the shared model once it lands; the harness only calls `step(t) -> bool` and reads `up`.
"""

from __future__ import annotations

import numpy as np


class LinkModel:
    def __init__(self, availability: float = 0.8, mean_down_s: float = 6.0, seed: int = 0, start_up: bool = True) -> None:
        self.availability = availability
        self.mean_down = mean_down_s
        self.mean_up = mean_down_s * availability / max(1.0 - availability, 1e-6)
        self.rng = np.random.default_rng(seed + 5500)
        self.up = start_up
        self.t_switch: float | None = None
        self.forced: bool | None = None  # operator override (demo key L): True up, False down, None = model
        self.up_s = self.total_s = 0.0
        self._t: float | None = None

    def step(self, t: float) -> bool:
        if self._t is not None:
            dt = t - self._t
            self.total_s += dt
            self.up_s += dt if self.is_up else 0.0
        self._t = t
        if self.t_switch is None:
            self.t_switch = t + self._draw()
        while t >= self.t_switch:
            self.up = not self.up
            self.t_switch += self._draw()
        return self.is_up

    def _draw(self) -> float:
        return float(self.rng.exponential(self.mean_up if self.up else self.mean_down))

    @property
    def is_up(self) -> bool:
        return self.up if self.forced is None else self.forced

    @property
    def measured_availability(self) -> float:
        return self.up_s / self.total_s if self.total_s > 0 else 1.0
