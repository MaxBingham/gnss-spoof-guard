"""V3 navigation health: NOMINAL / DEGRADED / HOLD / LAND from the covariance and the source availability.

  NOMINAL   2-sigma below `degraded_sigma2` and at least one map-referenced source available
  DEGRADED  2-sigma above `degraded_sigma2`, or every aiding source excluded / disabled -> fly slower
  HOLD      2-sigma above `hold_sigma2` while image-based navigation is possible here: hover, camera on, wait
            for an IBN fix (hovering does not help TRN, which needs motion) up to `hold_max_s`, then continue
            DEGRADED on the INS
  LAND      2-sigma above `land_sigma2`: navigation lost, land in place (terminal)
Recovery uses hysteresis (`recover_frac`). Thresholds come from the scenario (urban: meters, natural: tens).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..messages import Health, HealthState, PoseEstimate
from ..world.scenarios import Scenario


@dataclass
class NavHealthConfig:
    degraded_sigma2: float = 10.0
    hold_sigma2: float = 30.0
    land_sigma2: float = 120.0
    hold_max_s: float = 20.0
    recover_frac: float = 0.8
    degraded_speed_scale: float = 0.6


def nav_health_config_for(sc: Scenario) -> NavHealthConfig:
    if sc.kind == "urban":
        return NavHealthConfig(degraded_sigma2=25.0, hold_sigma2=60.0, land_sigma2=150.0)
    return NavHealthConfig(degraded_sigma2=40.0, hold_sigma2=150.0, land_sigma2=600.0)


class NavHealthMonitor:
    def __init__(self, cfg: NavHealthConfig) -> None:
        self.cfg = cfg
        self.state = Health.NOMINAL
        self.reason = ""
        self.hold_since: float | None = None
        self.hold_done = False  # one HOLD per degradation episode
        self.last_fix_t = 0.0
        self.time_in: dict[str, float] = {h.value: 0.0 for h in Health}
        self._last_t: float | None = None

    def update(self, t: float, pose: PoseEstimate, sources_available: int, ibn_possible: bool,
               last_fix_t: float, active: bool) -> HealthState:
        """`active`: in flight on the route (health only acts during cruise)."""
        c = self.cfg
        s2 = pose.sigma2_horizontal()
        self.last_fix_t = last_fix_t
        if self._last_t is not None and active:
            self.time_in[self.state.value] += t - self._last_t
        self._last_t = t
        prev = self.state
        if not active:
            pass
        elif self.state == Health.LAND:
            pass
        elif s2 > c.land_sigma2:
            self._set(Health.LAND, f"2-sigma {s2:.0f} m > {c.land_sigma2:.0f} m: navigation lost")
        elif self.state == Health.HOLD:
            if s2 < c.recover_frac * c.hold_sigma2:
                self._set(Health.DEGRADED if s2 > c.degraded_sigma2 else Health.NOMINAL, f"fix recovered, 2-sigma {s2:.1f} m")
            elif self.hold_since is not None and t - self.hold_since > c.hold_max_s:
                self._set(Health.DEGRADED, f"HOLD timeout ({c.hold_max_s:.0f} s), continuing on INS, 2-sigma {s2:.0f} m")
                self.hold_done = True
        elif s2 > c.hold_sigma2 and ibn_possible and not self.hold_done:
            self._set(Health.HOLD, f"2-sigma {s2:.0f} m > {c.hold_sigma2:.0f} m, hovering for an image fix")
            self.hold_since = t
        elif s2 > c.degraded_sigma2 or sources_available == 0:
            why = f"2-sigma {s2:.1f} m" if s2 > c.degraded_sigma2 else "no aiding source available"
            if self.state != Health.DEGRADED or not self.reason:
                self._set(Health.DEGRADED, why)
        elif self.state != Health.DEGRADED or s2 < c.recover_frac * c.degraded_sigma2:
            if self.state != Health.NOMINAL:
                self._set(Health.NOMINAL, f"2-sigma {s2:.1f} m")
            if s2 < c.recover_frac * c.degraded_sigma2:
                self.hold_done = False
        changed = self.state != prev
        scale = {Health.NOMINAL: 1.0, Health.DEGRADED: c.degraded_speed_scale}.get(self.state, 0.0)
        hs = HealthState(t, self.state, s2, t - last_fix_t, scale, self.reason)
        hs.changed = changed  # type: ignore[attr-defined]
        return hs

    def _set(self, state: Health, reason: str) -> None:
        self.state = state
        self.reason = reason
