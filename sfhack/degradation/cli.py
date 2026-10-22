"""Command-line levers shared by the demos.

    --gnss drag:1.5@90            GNSS mode (off | ok | jam | drag[:RATE[@HEADING]] | jump[:METERS[@HEADING]])
    --weather fog:0.6             weather KIND[:INTENSITY] (clear | rain | fog | dust)
    --wind 6@90                   steady wind SPEED[@DIRECTION]; add gusts / rotation with --cond
    --camera-lost / --camera-drop 0.3
    --lidar-lost / --lidar-dropout 0.5
    --cond "wind.gust=2,wind.rotation_deg_s=3"        any lever by dotted path (repeatable)
    --cond-at "30:gnss=drag:1.5@90;90:gnss=ok"         levers changed at sim times (repeatable)
"""

from __future__ import annotations

import argparse

from .conditions import Conditions, parse_assignments

Schedule = list[tuple[float, dict[str, str]]]


def add_cli_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("degradation levers (sfhack.degradation)")
    g.add_argument("--gnss", help="off | ok | jam | drag[:RATE[@HEADING]] | jump[:METERS[@HEADING]]")
    g.add_argument("--weather", help="KIND[:INTENSITY], KIND = clear | rain | fog | dust")
    g.add_argument("--wind", help="SPEED[@DIRECTION] (m/s, compass deg the wind blows toward)")
    g.add_argument("--camera-lost", action="store_true", help="down camera delivers black frames")
    g.add_argument("--camera-drop", type=float, help="fraction of down-camera frames dropped")
    g.add_argument("--lidar-lost", action="store_true", help="no rangefinder / lidar returns")
    g.add_argument("--lidar-dropout", type=float, help="fraction of range returns dropped")
    g.add_argument("--cond", action="append", default=[], help="'path=value,...', e.g. 'wind.gust=2'")
    g.add_argument("--cond-at", action="append", default=[], help="'T:path=value,...;T2:...' at sim times")


def conditions_from_args(args: argparse.Namespace, base: Conditions | None = None) -> Conditions:
    c = base.copy() if base is not None else Conditions()
    shorthand = {"gnss": args.gnss, "weather": args.weather, "wind": args.wind}
    c.update({k: v for k, v in shorthand.items() if v})
    if args.camera_lost:
        c.set("camera.lost", True)
    if args.camera_drop is not None:
        c.set("camera.drop_rate", args.camera_drop)
    if args.lidar_lost:
        c.set("lidar.lost", True)
    if args.lidar_dropout is not None:
        c.set("lidar.dropout", args.lidar_dropout)
    for spec in args.cond:
        c.update(spec)
    return c


def parse_schedule(specs: list[str] | str) -> Schedule:
    """'30:gnss=drag:1.5@90;90:gnss=ok,wind=0' -> [(30.0, {...}), (90.0, {...})], sorted by time."""
    out: Schedule = []
    for spec in [specs] if isinstance(specs, str) else specs:
        for item in filter(None, (s.strip() for s in spec.split(";"))):
            t, _, assign = item.partition(":")
            out.append((float(t), parse_assignments(assign)))
    out.sort(key=lambda e: e[0])
    for _, assign in out:  # fail at startup, not mid-flight
        Conditions().update(assign)
    return out
