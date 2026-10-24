"""Minimal ground-station listener on UDP 14550: what QGroundControl sees of our drone, without QGroundControl.

    uv run python -m sfhack.comms.gcs_listen            # then run demos/52_track_enemy.py (default --comms mavlink)

Parses with pymavlink's plain common.xml dialect and no signing key, as QGC does: the base forwards the drone's
packets still signed, and a receiver without a key accepts signed MAVLink 2 packets (it just cannot check them).
Prints the vehicle once a second: state (HEARTBEAT.custom_mode), position, altitude, heading, battery, GPS fix,
plus the STATUSTEXT messages. Packets of the sfhack dialect show up as unknown ids (54401 / 54402), as in QGC.
"""

from __future__ import annotations

import argparse
import socket
import time

from .mavlink_io import FLIGHT_STATES


def listen(port: int = 14550, seconds: float = 0.0, quiet: bool = False) -> dict:
    from pymavlink.dialects.v20 import common as m

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(0.2)
    mav = m.MAVLink(None)
    mav.robust_parsing = True
    seen: dict = {"types": {}, "unknown": {}, "texts": [], "last": {}}
    parts: dict[int, list[str]] = {}
    t0 = last_print = time.time()
    while not seconds or time.time() - t0 < seconds:
        try:
            data, _ = sock.recvfrom(65535)
        except (TimeoutError, socket.timeout, ConnectionResetError):
            data = b""
        for msg in (mav.parse_buffer(data) or []) if data else []:
            typ = msg.get_type()
            if typ.startswith("UNKNOWN"):  # "UNKNOWN_54401": not in common.xml
                mid = int(typ.split("_")[-1]) if typ.split("_")[-1].isdigit() else typ
                seen["unknown"][mid] = seen["unknown"].get(mid, 0) + 1
                continue
            seen["types"][typ] = seen["types"].get(typ, 0) + 1
            seen["last"][typ] = msg
            if typ == "STATUSTEXT":  # MAVLink 2: chunks of one text share an id, the last one is < 50 chars
                parts.setdefault(msg.id, []).append(msg.text)
                if not msg.id or len(msg.text) < 50:
                    text = "".join(parts.pop(msg.id))
                    seen["texts"].append(text)
                    if not quiet:
                        print(f"   STATUSTEXT: {text}")
        if not quiet and time.time() - last_print >= 1.0:
            last_print = time.time()
            print(summary(seen))
    sock.close()
    return seen


def summary(seen: dict) -> str:
    last = seen["last"]
    hb, g, s, gps = (last.get(k) for k in ("HEARTBEAT", "GLOBAL_POSITION_INT", "SYS_STATUS", "GPS_RAW_INT"))
    if hb is None:
        return "waiting for a HEARTBEAT on UDP 14550 ..."
    st = FLIGHT_STATES[hb.custom_mode] if hb.custom_mode < len(FLIGHT_STATES) else "?"
    pos = "-" if g is None else (f"{g.lat * 1e-7:.6f}, {g.lon * 1e-7:.6f}  alt {g.alt / 1000:6.1f} m AMSL "
                                 f"({g.relative_alt / 1000:5.1f} m rel)  hdg {g.hdg / 100:5.1f}")
    batt = "-" if s is None else f"{s.battery_remaining} %"
    fix = "-" if gps is None else ("3D fix" if gps.fix_type >= 3 else "NO FIX")
    return f"sys {hb.get_srcSystem()} {st:9s} {pos}  battery {batt}  GPS {fix}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=14550)
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0 = until Ctrl-C)")
    args = ap.parse_args()
    try:
        seen = listen(args.port, args.seconds)
    except KeyboardInterrupt:
        return
    print("message types:", dict(sorted(seen["types"].items())), " unknown ids:", seen["unknown"])


if __name__ == "__main__":
    main()
