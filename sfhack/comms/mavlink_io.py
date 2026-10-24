"""MAVLink 2 over UDP, shared by the base and the drone endpoints (sfhack/comms/base_link.py, drone_link.py).

One `MavEndpoint` = one UDP socket + one pymavlink encoder / parser of the sfhack dialect, MAVLink 2 signing on
(unsigned or wrongly signed packets are dropped and counted), one datagram per packet. The same class would sit on
a serial radio: only `write` / `_recv` touch the socket.

Simulated link: `gate()` (from the harness, the link model) decides whether an outgoing packet reaches the air;
while it is down the packet is lost (the MAVLink sequence number still advances, as on a real radio). The endpoint
never sees the link model, only the callable.

Determinism in the sim: both endpoints live in one process and the harness calls `exchange(base, drone)` once per
step, which reads until each side has every datagram the other side put on the air (`tx_peer` vs `rx_peer`), so a
packet is handled in the sim step it was sent in, as in the in-process path. A real deployment just calls `poll`
in its loop.
"""

from __future__ import annotations

import select
import socket
import time
from typing import Callable

import numpy as np

from ..geo import GeoOrigin
from .dialect import load

MAVLINK_EPOCH = 1420070400  # 2015-01-01, the signing timestamp's epoch (10 us units)
# HEARTBEAT.custom_mode = index of the onboard mission state (BearingPilot / TrackerPilot); the base decodes it back
FLIGHT_STATES = ("WAIT", "TAKEOFF", "OUTBOUND", "ORBIT", "SEARCH", "DETECTED", "TRACKING", "LOST", "RETURN", "DESCEND",
                 "LANDED")
ON_GROUND = ("WAIT", "LANDED")
TRACKER_STATES = {"SEARCH": 1, "DETECTED": 2, "TRACKING": 3, "LOST": 4}  # SFHACK_TRACKER_STATE


class Frame:
    """World ENU (x east, y north, z up on the onboard datum) <-> WGS84 (degE7, AMSL m) around the surveyed base."""

    def __init__(self, lat: float, lon: float, alt_m: float) -> None:
        self.geo = GeoOrigin(lat, lon)
        self.alt0 = alt_m

    def to_global(self, p) -> tuple[int, int, float]:
        lat, lon = self.geo.to_latlon(float(p[0]), float(p[1]))
        return int(round(lat * 1e7)), int(round(lon * 1e7)), self.alt0 + float(p[2])

    def to_world(self, lat_e7: int, lon_e7: int, alt: float) -> np.ndarray:
        x, y = self.geo.to_enu(lat_e7 * 1e-7, lon_e7 * 1e-7)
        return np.array([x, y, float(alt) - self.alt0])


def enu_to_ned(v) -> tuple[float, float, float]:
    return float(v[1]), float(v[0]), -float(v[2])


def ned_to_enu(n: float, e: float, d: float) -> np.ndarray:
    return np.array([e, n, -d], float)


def usec(t: float) -> int:
    return int(round(t * 1e6))


class MavEndpoint:
    """UDP socket + MAVLink 2 (sfhack dialect), signed. Subclasses implement `handle(msg, t)`."""

    def __init__(self, sysid: int, compid: int, bind: tuple[str, int], peer: tuple[str, int] | None = None,
                 key: bytes | None = None, link_id: int = 0, gate: Callable[[], bool] | None = None,
                 name: str = "") -> None:
        self.mod = load()
        self.name = name or f"sys{sysid}"
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        try:
            self.sock.bind(bind)
        except OSError:
            if bind[1] == 0:
                raise
            print(f"[mavlink] {self.name}: UDP {bind[0]}:{bind[1]} busy, using an ephemeral port", flush=True)
            self.sock.bind((bind[0], 0))
        self.sock.setblocking(False)
        self.addr = self.sock.getsockname()
        self.peer = peer
        self.gate = gate
        self.mav = self.mod.MAVLink(self, srcSystem=sysid, srcComponent=compid)
        self.mav.robust_parsing = True
        if key is not None:
            s = self.mav.signing
            s.secret_key, s.link_id, s.sign_outgoing = key, link_id, True
            s.timestamp = int((time.time() - MAVLINK_EPOCH) * 1e5)
            s.allow_unsigned_callback = None  # unsigned -> dropped
        self.taps: list[tuple[str, int]] = []  # extra destinations of everything received from the peer (QGC)
        self.now = 0.0
        # counters
        self.tx_peer = 0  # datagrams that reached the air towards the peer
        self.tx_lost = 0  # dropped by the link (gate down)
        self.tx_bytes = 0  # bytes that reached the air (MAVLink packets incl. header + signature)
        self.rx_peer = 0  # datagrams received from the peer (any content)
        self.rx_msgs = 0
        self.rx_other = 0  # datagrams from anyone else (QGC, a forger)
        self.rejected: dict[str, int] = {}  # reason -> count (unsigned, bad signature, bad data)
        self.rx_by_type: dict[str, int] = {}
        self._peer_seq: int | None = None
        self.peer_seq_lost = 0  # gaps in the peer's MAVLink sequence numbers (packets lost on the link)

    # ------------------------------------------------------------------ sending

    def write(self, buf: bytes) -> None:
        """pymavlink's output 'file': one packet = one datagram, through the link gate."""
        if self.peer is None:
            return
        if self.gate is not None and not self.gate():
            self.tx_lost += 1
            return
        self.sock.sendto(bytes(buf), self.peer)
        self.tx_peer += 1
        self.tx_bytes += len(buf)

    def send(self, msg) -> None:
        self.mav.send(msg)

    # ------------------------------------------------------------------ receiving

    def poll(self, t: float | None = None, expect_peer: int | None = None, timeout: float = 1.0) -> int:
        """Read and handle everything waiting; with `expect_peer`, also wait (up to `timeout` s) until that many
        datagrams from the peer have arrived (lockstep in the sim). Returns the number of datagrams read."""
        if t is not None:
            self.now = t
        n = 0
        deadline = None
        while True:
            try:
                data, addr = self.sock.recvfrom(65535)
            except (BlockingIOError, InterruptedError):
                if expect_peer is None or self.rx_peer >= expect_peer:
                    return n
                if deadline is None:
                    deadline = time.perf_counter() + timeout
                left = deadline - time.perf_counter()
                if left <= 0:  # lost on loopback (should not happen): count it lost, keep the sim running
                    print(f"[mavlink] {self.name}: {expect_peer - self.rx_peer} datagram(s) from the peer did not "
                          f"arrive within {timeout:.1f} s", flush=True)
                    self.rx_peer = expect_peer
                    return n
                select.select([self.sock], [], [], left)
                continue
            except ConnectionResetError:
                continue
            n += 1
            from_peer = self.peer is not None and addr[1] == self.peer[1] and addr[0] == self.peer[0]
            if from_peer:
                self.rx_peer += 1
            else:
                self.rx_other += 1
            self._datagram(data, addr, from_peer)

    def _datagram(self, data: bytes, addr, from_peer: bool) -> None:
        mav = self.mav
        try:
            msgs = mav.parse_buffer(data) or []
        except self.mod.MAVError as e:
            msgs = [self.mod.MAVLink_bad_data(bytearray(data), str(e))]
        if mav.buf_len():  # a partial / broken packet: start the next datagram clean
            mav.buf, mav.buf_index, mav.expected_length = bytearray(), 0, self.mod.HEADER_LEN_V1 + 2
        for m in msgs:
            if m.get_type() == "BAD_DATA":
                why = str(getattr(m, "reason", ""))
                signed = len(data) > 2 and data[0] == 0xFD and data[2] & 0x01
                key = ("bad signature" if signed else "unsigned") if "signature" in why.lower() else "bad data"
                self.rejected[key] = self.rejected.get(key, 0) + 1
                continue
            if from_peer:
                seq = m.get_seq()
                if self._peer_seq is not None:
                    self.peer_seq_lost += (seq - self._peer_seq - 1) % 256
                self._peer_seq = seq
                for tap in self.taps:
                    try:
                        self.sock.sendto(data, tap)
                    except OSError:
                        pass
            self.rx_msgs += 1
            typ = m.get_type()
            self.rx_by_type[typ] = self.rx_by_type.get(typ, 0) + 1
            self.handle(m, self.now, from_peer)

    def handle(self, msg, t: float, from_peer: bool) -> None:  # pragma: no cover - overridden
        pass

    def close(self) -> None:
        self.sock.close()

    def stats(self) -> str:
        rej = ", ".join(f"{k} {v}" for k, v in sorted(self.rejected.items())) or "none"
        return (f"{self.name}: tx {self.tx_peer} ({self.tx_bytes / 1024:.0f} kB; lost on the link {self.tx_lost}), rx "
                f"{self.rx_peer} from the peer + {self.rx_other} other, rejected: {rej}")


def pair(a: MavEndpoint, b: MavEndpoint) -> None:
    """Point two endpoints at each other's bound address (ephemeral-port fallback)."""
    a.peer, b.peer = b.addr, a.addr


def exchange(a: MavEndpoint, b: MavEndpoint, t: float, max_rounds: int = 8) -> None:
    """Lockstep (sim): deliver everything in flight between a and b, including the replies it causes."""
    for _ in range(max_rounds):
        if a.rx_peer >= b.tx_peer and b.rx_peer >= a.tx_peer:
            return
        b.poll(t, a.tx_peer)
        a.poll(t, b.tx_peer)
