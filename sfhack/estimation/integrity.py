"""Integrity monitoring: cross-check the position sources (INS, TRN, IBN) before a fix reaches the EKF.

Every fix is tested against the INS prediction (NIS, chi-square 2 dof) and against the most recent fix of the
other map-referenced source (motion-compensated with the EKF velocity). Decisions:
  - consistent with the INS                         -> fuse
  - inconsistent, and the other source + INS agree  -> SOURCE_EXCLUDED <src> for `exclude_s` (it is the outlier)
  - repeatedly inconsistent, other source agrees    -> INS_RESET (the INS drifted; both map sources agree)
  - repeatedly inconsistent, no other source:
        INS blind for a long time  -> RELOCALIZE on this source (inflated noise, V2 behaviour)
        INS recently confirmed     -> SOURCE_EXCLUDED (trust the recently confirmed INS)
Excluded sources keep being evaluated; three consecutive consistent fixes reinstate them early.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..messages import VisualFix
from .nav_ekf import NavEkf


@dataclass
class IntegrityConfig:
    gate_chi2: float = 9.21  # 99 %, 2 dof: fix vs INS
    cross_chi2: float = 13.8  # 99.9 %, 2 dof: source vs source
    cross_window_s: float = 10.0  # the other source's fix must be this recent for a cross-check
    exclude_s: float = 30.0
    max_rejects: int = 5
    exclude_after_rejects: int = 2  # with a contradicting (INS-confirmed) other source
    relocalize_blind_s: float = 30.0  # single-source re-localization only after this long without any fix
    reinstate_after: int = 3


@dataclass
class SourceState:
    name: str
    excluded_until: float = -1.0
    rejects: int = 0
    consistent_while_excluded: int = 0
    last_fix: VisualFix | None = None  # last fix (any outcome) for cross-checks
    last_accepted_t: float = -1e9
    n_fix: int = 0
    n_accepted: int = 0
    n_rejected: int = 0
    n_excluded_events: int = 0
    enabled: bool = True  # user / configuration switch (e.g. live view key T / I)
    nis_log: list[tuple[float, float, bool]] = field(default_factory=list)


class IntegrityMonitor:
    def __init__(self, cfg: IntegrityConfig, sources: tuple[str, ...] = ("trn", "ibn")) -> None:
        self.cfg = cfg
        self.src = {s: SourceState(s) for s in sources}
        self.last_any_accept_t = 0.0
        self.events: list[tuple[float, str]] = []
        self.reset_requests: set[str] = set()  # sources whose estimator should restart around the INS

    def excluded(self, name: str, t: float) -> bool:
        return t < self.src[name].excluded_until

    def available(self, t: float) -> list[str]:
        return [s for s, st in self.src.items() if st.enabled and not self.excluded(s, t)]

    def _cross(self, fix: VisualFix, other: VisualFix, ekf: NavEkf) -> float:
        dt = fix.t - other.t
        pred = other.pos + ekf.x[3:5] * dt  # move the other fix forward with the INS velocity
        motion = np.eye(2) * (0.3 * abs(dt)) ** 2
        d = fix.pos - pred
        return float(d @ np.linalg.solve(fix.cov + other.cov + motion, d))

    def process(self, fix: VisualFix, ekf: NavEkf) -> bool:
        """Decide what to do with a fix; applies it to the EKF when accepted. Returns True if fused."""
        c = self.cfg
        t = fix.t
        st = self.src[fix.source]
        st.n_fix += 1
        nis = ekf.nis_position(fix.pos, fix.cov)
        offset = float(np.hypot(*(fix.pos - ekf.x[:2])))
        fix.accepted = False
        others = [o for name, o in self.src.items()
                  if name != fix.source and o.enabled and not self.excluded(name, t) and o.last_fix is not None
                  and t - o.last_fix.t <= c.cross_window_s]
        prev_last, st.last_fix = st.last_fix, fix

        if self.excluded(fix.source, t):
            st.consistent_while_excluded = st.consistent_while_excluded + 1 if nis <= c.gate_chi2 else 0
            if st.consistent_while_excluded >= c.reinstate_after:
                st.excluded_until = -1.0
                st.rejects = 0
                self.events.append((t, f"SOURCE_REINSTATED {fix.source}: consistent with the INS again "
                                       f"({c.reinstate_after} fixes, offset {offset:.1f} m)"))
            st.nis_log.append((t, nis, False))
            return False

        if nis <= c.gate_chi2:
            ekf.update_position(fix.pos, fix.cov)
            self._accepted(st, fix, nis)
            return True

        st.rejects += 1
        st.n_rejected += 1
        st.nis_log.append((t, nis, False))
        confirmed_others = [o for o in others if o.last_fix.accepted]
        contradicted = [o for o in confirmed_others if self._cross(fix, o.last_fix, ekf) > c.cross_chi2]
        agreeing = [o for o in others if self._cross(fix, o.last_fix, ekf) <= c.cross_chi2]

        if contradicted and st.rejects >= c.exclude_after_rejects:
            o = contradicted[0]
            dist = float(np.hypot(*(fix.pos - o.last_fix.pos)))
            self._exclude(st, t, f"inconsistent with {o.name}+ins by {dist:.0f} m (NIS {nis:.0f})")
            return False
        if st.rejects < c.max_rejects:
            return False
        if agreeing:  # this source and another one agree: the INS is the outlier
            o = agreeing[0]
            W1, W2 = np.linalg.inv(fix.cov), np.linalg.inv(o.last_fix.cov)
            cov = np.linalg.inv(W1 + W2)
            pos = cov @ (W1 @ fix.pos + W2 @ (o.last_fix.pos + ekf.x[3:5] * (t - o.last_fix.t)))
            ekf.reset_position(pos, cov * 2.0)
            self.events.append((t, f"INS_RESET: {fix.source}+{o.name} agree, INS off by {offset:.0f} m"))
            self._accepted(st, fix, nis)
            o.rejects = 0
            return True
        blind = t - self.last_any_accept_t
        if blind >= c.relocalize_blind_s:
            R = fix.cov * (nis / c.gate_chi2)
            ekf.update_position(fix.pos, R)
            self.events.append((t, f"RELOCALIZE on {fix.source}: {st.rejects} rejections, INS blind "
                                   f"{blind:.0f} s, offset {offset:.0f} m (NIS {nis:.0f})"))
            self._accepted(st, fix, nis)
            return True
        self._exclude(st, t, f"inconsistent with ins by {offset:.0f} m, INS confirmed {blind:.0f} s ago (NIS {nis:.0f})")
        return False

    def _accepted(self, st: SourceState, fix: VisualFix, nis: float) -> None:
        fix.accepted = True
        st.rejects = 0
        st.n_accepted += 1
        st.last_accepted_t = fix.t
        st.nis_log.append((fix.t, nis, True))
        self.last_any_accept_t = fix.t

    def _exclude(self, st: SourceState, t: float, why: str) -> None:
        st.excluded_until = t + self.cfg.exclude_s
        st.rejects = 0
        st.consistent_while_excluded = 0
        st.n_excluded_events += 1
        self.reset_requests.add(st.name)
        self.events.append((t, f"SOURCE_EXCLUDED {st.name}: {why}, for {self.cfg.exclude_s:.0f} s"))
