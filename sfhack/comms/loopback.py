"""Base + drone MAVLink endpoints in one process over UDP loopback, in lockstep with the simulation (harness helper).

The harness hands in the onboard pilot and bus (the drone side) and `gate`, a callable that says whether the
simulated radio link carries a packet right now (sfhack/sim/link.py); neither endpoint sees the link model itself.
`step(t)` once per sim step: periodic sends on both sides, then every datagram in flight is delivered in that step.
Datagrams from outside (QGroundControl, a forger: sfhack/comms/signing_check.py) are read `external_hz` times a second.
"""

from __future__ import annotations

import math

from ..config import CommsConfig
from .base_link import BaseLink
from .drone_link import DroneLink
from .mavlink_io import exchange, pair


class LoopbackPair:
    def __init__(self, pilot, bus, gate=None, cfg: CommsConfig | None = None, fixed_ports: bool = True,
                 qgc: bool | None = None, external_hz: float = 10.0) -> None:
        cfg = cfg or CommsConfig()
        self.base = BaseLink(cfg, gate=gate, port=None if fixed_ports else 0, qgc=qgc)
        self.drone = DroneLink(pilot, bus, cfg, gate=gate, port=None if fixed_ports else 0)
        pair(self.base, self.drone)
        self.ext_dt = 1.0 / external_hz
        self._next_ext = -math.inf

    def step(self, t: float) -> None:
        self.drone.update(t)
        self.base.update(t)
        self.exchange(t)
        if t >= self._next_ext:
            self._next_ext = t + self.ext_dt
            self.base.poll(t)
            self.drone.poll(t)

    def exchange(self, t: float) -> None:
        exchange(self.base, self.drone, t)

    def stats(self) -> str:
        d, rx = self.drone, self.base.received
        jpg = [len(r.snapshot_jpg) for r in rx if r.snapshot_jpg]
        snap = f", snapshots {sum(jpg) / len(jpg) / 1024:.1f} kB on average" if jpg else ""
        return (f"MAVLink {self.base.stats()}; {d.stats()}; reports sent {d.reports_sent} (resent {d.reports_resent}, "
                f"acked {d.reports_acked}){snap}, bearings received {d.bearings_rx}")

    def close(self) -> None:
        self.base.close()
        self.drone.close()
