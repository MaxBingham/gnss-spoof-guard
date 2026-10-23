"""Mission planner CLI.

    python -m sfhack.missionprep serve [--port 8765]               # browser page: pick A / B, prepare, upload
    python -m sfhack.missionprep build                              # default Berkeley A / B
    python -m sfhack.missionprep build --a 37.8703 -122.2595 --b 37.8754 -122.2590 --radius 30 --id berkeley
    python -m sfhack.missionprep upload <id> [--mbps 2]            # simulated link upload + checksum check
    python -m sfhack.missionprep list
"""

from __future__ import annotations

import argparse
import json
import sys

from ..geo import LatLon
from .build import DEFAULT_A, DEFAULT_B, DEFAULT_RADIUS_M, missions_dir, prepare_mission, summarize


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m sfhack.missionprep", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="mission planner web page")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    b = sub.add_parser("build", help="prepare a mission without the browser")
    b.add_argument("--a", type=float, nargs=2, metavar=("LAT", "LON"), default=[DEFAULT_A.lat, DEFAULT_A.lon])
    b.add_argument("--b", type=float, nargs=2, metavar=("LAT", "LON"), default=[DEFAULT_B.lat, DEFAULT_B.lon])
    b.add_argument("--radius", type=float, default=DEFAULT_RADIUS_M, help="search radius around B (m)")
    b.add_argument("--id", default=None, help="mission id (default: hash of A, B, radius)")
    b.add_argument("--force", action="store_true", help="rebuild even if the mission exists (downloads stay cached)")
    b.add_argument("--offline", action="store_true", help="cache only, no network")
    b.add_argument("--no-lidar", action="store_true", help="skip 3DEP LiDAR, use the synthetic cloud")
    u = sub.add_parser("upload", help="simulated upload of the onboard package to the drone")
    u.add_argument("id")
    u.add_argument("--mbps", type=float, default=2.0, help="link bandwidth in MB/s")
    sub.add_parser("list", help="prepared missions")
    args = ap.parse_args()

    if args.cmd == "serve":
        import uvicorn

        from .server import app

        print(f"mission planner: http://{args.host}:{args.port}/")
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    elif args.cmd == "build":
        rep = prepare_mission(LatLon(*args.a), LatLon(*args.b), args.radius, args.id, force=args.force,
                              offline=args.offline, no_lidar=args.no_lidar)
        print(summarize(rep))
        print(f"\nfly it:  uv run python demos/demo_live.py --mission {rep['mission_id']}")
    elif args.cmd == "upload":
        from .upload import upload

        ok = upload(args.id, args.mbps, on_event=lambda e: print(e["text"]) if "text" in e else None)
        sys.exit(0 if ok else 1)
    elif args.cmd == "list":
        for d in sorted(missions_dir().iterdir()):
            rep = d / "build_report.json"
            if rep.exists():
                r = json.loads(rep.read_text(encoding="utf-8"))
                print(f"{d.name:20s} A {r['inputs']['a']} B {r['inputs']['b']} r {r['inputs']['radius_m']} m  {r['created_at']}")


if __name__ == "__main__":
    main()
