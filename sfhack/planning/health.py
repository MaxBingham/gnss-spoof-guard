"""Navigation health state machine.

NOMINAL -> DEGRADED (slow down) -> HOLD (hover) -> RETURN_TO_LAST_FIX -> LAND,
driven by the 2-sigma horizontal radius of the estimate and the time since the last accepted
visual fix. Thresholds live in config.HealthConfig. HOLD / RETURN recover to NOMINAL as soon as
fixes come back; LAND is terminal.
"""

from __future__ import annotations

import numpy as np

from ..config import HealthConfig
from ..messages import Health, HealthState, PoseEstimate, VisualFix


class HealthMonitor:
    def __init__(self, cfg: HealthConfig) -> None:
        self.cfg = cfg
        self.state = Health.NOMINAL
        self.last_fix_t: float | None = None
        self.last_fix_pos: np.ndarray | None = None
        self.active_since: float | None = None
        self.hold_since = 0.0
        self.reason = ""

    def on_fix(self, fix: VisualFix) -> None:
        if fix.accepted:
            self.last_fix_t = fix.t
            self.last_fix_pos = fix.pos.copy()

    def reset_timer(self, t: float) -> None:
        """Start counting 'time since fix' from now (called when monitoring becomes relevant)."""
        if self.active_since is None:
            self.active_since = t

    def update(self, t: float, pose: PoseEstimate) -> HealthState:
        c = self.cfg
        ref = self.last_fix_t if self.last_fix_t is not None else (self.active_since if self.active_since is not None else t)
        tsf = max(0.0, t - ref)
        s2 = pose.sigma2_horizontal()
        good = s2 < c.degraded_sigma2 and tsf < c.degraded_no_fix_s
        prev = self.state

        if self.state == Health.LAND:
            pass
        elif s2 > c.land_sigma2 or tsf > c.land_no_fix_s:
            self._set(Health.LAND, f"sigma2 {s2:.0f} m / no fix {tsf:.0f} s", t)
        elif self.state in (Health.HOLD, Health.RETURN_TO_LAST_FIX):
            if good:
                self._set(Health.NOMINAL, "fix recovered", t)
            elif self.state == Health.HOLD and t - self.hold_since > c.return_after_hold_s and self.last_fix_pos is not None:
                self._set(Health.RETURN_TO_LAST_FIX, f"no recovery after {c.return_after_hold_s:.0f} s hold", t)
        elif s2 > c.hold_sigma2 or tsf > c.hold_no_fix_s:
            self._set(Health.HOLD, f"sigma2 {s2:.1f} m / no fix {tsf:.1f} s", t)
        elif not good:
            self._set(Health.DEGRADED, f"sigma2 {s2:.1f} m / no fix {tsf:.1f} s", t)
        else:
            self._set(Health.NOMINAL, "", t)
        if self.state == Health.HOLD and prev != Health.HOLD:
            self.hold_since = t

        scale = {Health.NOMINAL: 1.0, Health.DEGRADED: c.degraded_speed_scale}.get(self.state, 0.0)
        return HealthState(t, self.state, s2, tsf, scale, self.reason)

    def _set(self, state: Health, reason: str, t: float) -> None:
        if state != self.state or reason:
            self.reason = reason
        self.state = state
