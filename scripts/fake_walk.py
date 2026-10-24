"""Fake speaker walk around the array: bearings over UDP like the live listener, for demos/51_counter_drone.py.

    python scripts/fake_walk.py --from 0 --to 90 --rate 1          # hold 10 s at 0, walk to 90 at 1 deg/s, hold 30 s
    python scripts/fake_walk.py --from 300 --to 30 --rate 2 --hold 5
"""
from __future__ import annotations

import argparse
import json
import random
import socket
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--from", dest="az0", type=float, default=0.0)
    ap.add_argument("--to", dest="az1", type=float, default=90.0)
    ap.add_argument("--rate", type=float, default=1.0, help="walking speed (deg/s), the short way round")
    ap.add_argument("--hold", type=float, default=10.0, help="seconds at the start bearing before walking")
    ap.add_argument("--hold-end", type=float, default=30.0, help="seconds at the end bearing before going silent")
    ap.add_argument("--noise", type=float, default=2.0, help="bearing noise (deg, 1 sigma)")
    ap.add_argument("--sigma", type=float, default=5.0, help="sigma_az_deg sent with every bearing")
    ap.add_argument("--hz", type=float, default=5.0)
    a = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    span = (a.az1 - a.az0 + 180.0) % 360.0 - 180.0
    walk_s = abs(span) / max(a.rate, 1e-6)
    total = a.hold + walk_s + a.hold_end
    print(f"{a.az0:.0f} -> {a.az1:.0f} deg: hold {a.hold:.0f} s, walk {walk_s:.0f} s, hold {a.hold_end:.0f} s "
          f"-> UDP {a.host}:{a.port}")
    t0 = time.time()
    while (t := time.time() - t0) < total:
        k = min(max((t - a.hold) / walk_s, 0.0), 1.0) if walk_s > 0 else 1.0
        az = (a.az0 + k * span + random.gauss(0.0, a.noise)) % 360.0
        msg = {"t": time.time(), "az_deg": round(az, 2), "el_deg": 10.0, "sigma_az_deg": a.sigma,
               "sigma_el_deg": 15.0, "label": "drone", "confidence": 0.8, "detected": True}
        sock.sendto(json.dumps(msg).encode("utf-8"), (a.host, a.port))
        print(f"\rt={t:5.1f}s  az {az:5.1f}", end="", flush=True)
        time.sleep(1.0 / a.hz)
    sock.sendto(json.dumps({"detected": False}).encode("utf-8"), (a.host, a.port))
    print("\nsilent")


if __name__ == "__main__":
    main()
