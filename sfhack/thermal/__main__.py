"""CLI:  python -m sfhack.thermal build --mission <id> [--hour 23] [--season summer] [--seed 0]"""

from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m sfhack.thermal")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write world/thermal_truth.npz and onboard/thermal_prior.npz")
    b.add_argument("--mission", required=True, nargs="+", help="mission id(s) or directory")
    b.add_argument("--hour", type=float, default=23.0, help="local clock time (default 23.0)")
    b.add_argument("--season", default="summer", choices=["summer", "spring", "autumn", "winter"])
    b.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        from .model import build_thermal

        for m in a.mission:
            info = build_thermal(m, hour=a.hour, season=a.season, seed=a.seed)
            print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
