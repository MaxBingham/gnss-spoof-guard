"""V3 navigation health: NOMINAL / DEGRADED / HOLD from the covariance and the source availability.

  NOMINAL   2-sigma below `degraded_sigma2` and at least one map-referenced source available
  DEGRADED  2-sigma above `degraded_sigma2`, or every aiding source excluded / disabled -> fly slower
  HOLD      2-sigma above `hold_sigma2` while image-based navigation is possible here: hover, camera on, IBN
            searches a window that grows with the 2-sigma (coarse-to-fine), up to `hold_max_s`; then continue
            DEGRADED (TRN needs motion, hovering does not help it) and HOLD again after `rehold_after_s`
  LOST      2-sigma above `lost_sigma2`: logged, behaves like DEGRADED / HOLD (never terminal)
Lesson from V2: health LAND (land in place when blind) turned every long outage into a failed mission, although
a fix usually came back seconds later. Every state is recoverable; only the final touchdown at B is terminal.
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
    lost_sigma2: float = 120.0
    hold_max_s: float = 20.0
    rehold_after_s: float = 30.0  # DEGRADED flight between two HOLD attempts (TRN gets motion meanwhile)
    recover_frac: float = 0.8
    degraded_speed_scale: float = 0.6


def nav_health_config_for(sc: Scenario) -> NavHealthConfig:
    if sc.kind == "urban":
        return NavHealthConfig(degraded_sigma2=25.0, hold_sigma2=60.0, lost_sigma2=150.0)
    return NavHealthConfig(degraded_sigma2=40.0, hold_sigma2=150.0, lost_sigma2=600.0, hold_max_s=30.0,
                           rehold_after_s=60.0)


class NavHealthMonitor:
    def __init__(self, cfg: NavHealthConfig) -> None:
        self.cfg = cfg
        self.state = Health.NOMINAL
        self.reason = ""
        self.hold_since: float | None = None
        self.hold_ended: float | None = None  # end of the last HOLD episode (re-HOLD after rehold_after_s)
        self.lost = False
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
        lost_now = s2 > c.lost_sigma2
        lost_msg = f", navigation LOST (2-sigma > {c.lost_sigma2:.0f} m), recoverable" if lost_now and not self.lost else ""
        self.lost = lost_now if lost_now or s2 < c.recover_frac * c.lost_sigma2 else self.lost
        can_hold = ibn_possible and (self.hold_ended is None or t - self.hold_ended > c.rehold_after_s)
        if not active:
            pass
        elif self.state == Health.HOLD:
            if s2 < c.recover_frac * c.hold_sigma2:
                self._set(Health.DEGRADED if s2 > c.degraded_sigma2 else Health.NOMINAL,
                          f"fix recovered, 2-sigma {s2:.1f} m")
                self.hold_ended = t
            elif not ibn_possible or (self.hold_since is not None and t - self.hold_since > c.hold_max_s):
                why = "no image fix possible here" if not ibn_possible else f"no fix within {c.hold_max_s:.0f} s"
                self._set(Health.DEGRADED, f"HOLD ended ({why}), continuing slowly on INS + TRN, 2-sigma {s2:.0f} m")
                self.hold_ended = t
        elif s2 > c.hold_sigma2 and can_hold:
            self._set(Health.HOLD, f"2-sigma {s2:.0f} m > {c.hold_sigma2:.0f} m, hovering for an image fix{lost_msg}")
            self.hold_since = t
        elif s2 > c.degraded_sigma2 or sources_available == 0:
            why = f"2-sigma {s2:.1f} m" if s2 > c.degraded_sigma2 else "no aiding source available"
            if self.state != Health.DEGRADED or lost_msg:
                self._set(Health.DEGRADED, why + lost_msg)
        elif self.state != Health.DEGRADED or s2 < c.recover_frac * c.degraded_sigma2:
            if self.state != Health.NOMINAL:
                self._set(Health.NOMINAL, f"2-sigma {s2:.1f} m")
        changed = self.state != prev
        scale = {Health.NOMINAL: 1.0, Health.DEGRADED: c.degraded_speed_scale}.get(self.state, 0.0)
        hs = HealthState(t, self.state, s2, t - last_fix_t, scale, self.reason)
        hs.changed = changed  # type: ignore[attr-defined]
        return hs

    def _set(self, state: Health, reason: str) -> None:
        self.state = state
        self.reason = reason
