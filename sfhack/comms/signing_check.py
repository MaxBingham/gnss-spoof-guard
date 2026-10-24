"""MAVLink 2 signing check: forged commands do not reach the drone.

    uv run python -m sfhack.comms.signing_check                    # self-contained: base + drone + attacker
    uv run python -m sfhack.comms.signing_check --target 127.0.0.1:14561   # attack a running demo's drone port

Self-contained: a drone endpoint (stand-in pilot) and the real base on loopback, plus an attacker socket. A RETURN
(MAV_CMD_NAV_RETURN_TO_LAUNCH) from the base with the shared key is executed and acknowledged; the same command
unsigned, signed with a wrong key, or replayed byte for byte (captured off the air) is dropped by the drone's parser:
no COMMAND_ACK, no recall, counted in `rejected`. --target: only sends an unsigned and a wrong-key RETURN to a
running drone (e.g. demos/52_track_enemy.py); its MAVLink stats line at the end shows them under "rejected".
"""

from __future__ import annotations

import argparse
import hashlib
import socket
import time

from ..config import CommsConfig
from .base_link import BaseLink
from .dialect import load
from .drone_link import DroneLink
from .mavlink_io import MAVLINK_EPOCH, pair


class StandInPilot:
    """Just enough of a pilot for the drone endpoint to execute a recall."""

    state = "SEARCH"
    on_ground = False

    def __init__(self) -> None:
        self.recalls: list[float] = []

    def recall(self, t: float) -> None:
        self.recalls.append(t)
        self.state = "RETURN"


class _Bus:
    def subscribe(self, *_a) -> None:
        pass


def forged_packets(cfg: CommsConfig) -> dict[str, bytes]:
    """A RETURN command as an attacker would build it: unsigned, and signed with a wrong key."""
    m = load()

    class Sink:
        buf = b""

        def write(self, b) -> None:
            Sink.buf = bytes(b)

    out = {}
    for name, key in (("unsigned", None), ("wrong key", hashlib.sha256(b"guessed-passphrase").digest())):
        mav = m.MAVLink(Sink(), srcSystem=cfg.base_sysid, srcComponent=cfg.base_compid)  # spoofs the base's ids
        if key is not None:
            mav.signing.secret_key, mav.signing.sign_outgoing = key, True
            mav.signing.timestamp = int((time.time() - MAVLINK_EPOCH) * 1e5)
        mav.command_long_send(cfg.drone_sysid, cfg.drone_compid, m.MAV_CMD_NAV_RETURN_TO_LAUNCH, 0, 0, 0, 0, 0, 0, 0, 0)
        out[name] = Sink.buf
    return out


def run(cfg: CommsConfig | None = None, verbose: bool = True) -> dict[str, dict]:
    cfg = cfg or CommsConfig()
    m = load()
    pilot = StandInPilot()
    drone = DroneLink(pilot, _Bus(), cfg, port=0)
    base = BaseLink(cfg, port=0, qgc=False)
    pair(base, drone)
    captured: list[bytes] = []
    send = base.sock.sendto

    def tap(data, addr):  # an eavesdropper records the base's signed packets
        captured.append(bytes(data))
        return send(data, addr)

    base.sock = _Tapped(base.sock, tap)
    attacker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    results: dict[str, dict] = {}

    def attempt(name: str, fire) -> None:
        n_rx, n_rej, n_rec = drone.rx_peer + drone.rx_other, sum(drone.rejected.values()), len(pilot.recalls)
        base.acks.clear()
        pilot.state = "SEARCH"
        fire()
        deadline = time.perf_counter() + 1.0
        while drone.rx_peer + drone.rx_other == n_rx and time.perf_counter() < deadline:
            drone.poll(0.0)
            time.sleep(0.005)
        time.sleep(0.05)
        base.poll(0.0)
        acked = m.MAV_CMD_NAV_RETURN_TO_LAUNCH in base.acks
        r = {"executed": len(pilot.recalls) > n_rec, "acked": acked, "rejected": sum(drone.rejected.values()) - n_rej}
        results[name] = r
        if verbose:
            print(f"{name:22s} -> {'EXECUTED + COMMAND_ACK' if r['executed'] else 'DROPPED by the drone'}"
                  f"  (rejected {r['rejected']}, drone rejected so far: {dict(drone.rejected)})")

    def signed() -> None:
        base.recall(0.0)
        base.pending.clear()  # one transmission, no retries

    attempt("signed (shared key)", signed)
    replay = captured[-1]
    forged = forged_packets(cfg)
    attempt("unsigned", lambda: attacker.sendto(forged["unsigned"], drone.addr))
    attempt("signed, wrong key", lambda: attacker.sendto(forged["wrong key"], drone.addr))
    attempt("replayed (captured)", lambda: attacker.sendto(replay, drone.addr))
    attacker.close()
    drone.close()
    base.close()
    return results


class _Tapped:
    """Socket wrapper: sendto goes through `tap`, everything else to the socket."""

    def __init__(self, sock, tap) -> None:
        self._sock, self._tap = sock, tap

    def sendto(self, data, addr):
        return self._tap(data, addr)

    def __getattr__(self, name):
        return getattr(self._sock, name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default=None, help="host:port of a running drone endpoint (demo: 127.0.0.1:14561)")
    args = ap.parse_args()
    cfg = CommsConfig()
    if args.target is None:
        run(cfg)
        return
    host, port = args.target.rsplit(":", 1)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for name, pkt in forged_packets(cfg).items():
        s.sendto(pkt, (host, int(port)))
        print(f"sent a {name} RETURN command to {host}:{port} ({len(pkt)} bytes)")
    s.close()


if __name__ == "__main__":
    main()
