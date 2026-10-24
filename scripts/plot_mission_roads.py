"""Figure + paved / unpaved statistics of a prepared mission's road network (OSM `surface` tag).

    uv run python scripts/plot_mission_roads.py ukraine-1 [--out docs/ukraine-1_roads.png]

Reads world/osm.json (harness-side: the raw download) and onboard/mission.json (route), writes a PNG and
prints a markdown table of road length per highway class and surface group.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from sfhack.geo import GeoOrigin  # noqa: E402
from sfhack.missionprep.build import mission_dir  # noqa: E402

PAVED = {"asphalt", "paved", "concrete", "concrete:plates", "concrete:lanes", "paving_stones", "sett", "cobblestone",
         "chipseal", "metal", "wood"}
UNPAVED = {"unpaved", "gravel", "fine_gravel", "compacted", "dirt", "earth", "ground", "grass", "mud", "sand",
           "pebblestone", "rock", "woodchips", "grass_paver"}
ROADS = ("motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "living_street",
         "service", "track")


def surface_group(tags: dict[str, str]) -> str:
    s = tags.get("surface", "")
    if s in PAVED:
        return "paved"
    if s in UNPAVED or tags.get("highway") == "track":
        return "unpaved"
    return "untagged"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mission")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = mission_dir(args.mission)
    wm = json.loads((d / "world" / "manifest.json").read_text(encoding="utf-8"))
    osm = json.loads((d / "world" / "osm.json").read_text(encoding="utf-8"))
    mission = json.loads((d / "onboard" / "mission.json").read_text(encoding="utf-8"))
    o = GeoOrigin(wm["origin"]["lat"], wm["origin"]["lon"])
    ext = wm["ortho"]["extent_enu"]
    lengths: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    fig, ax = plt.subplots(figsize=(16, 16 * (ext["max_y"] - ext["min_y"]) / (ext["max_x"] - ext["min_x"]) + 1))
    ax.imshow(np.asarray(Image.open(d / "world" / "ortho.jpg")), extent=(ext["min_x"], ext["max_x"], ext["min_y"], ext["max_y"]))
    color = {"paved": "#ff3b30", "unpaved": "#c69c6d", "untagged": "#bbbbbb"}
    for el in osm["elements"]:
        t = el.get("tags", {})
        if el.get("type") != "way" or t.get("highway") not in ROADS:
            continue
        xy = np.array([o.to_enu(p["lat"], p["lon"]) for p in el["geometry"]])
        inside = (xy[:, 0] >= ext["min_x"]) & (xy[:, 0] <= ext["max_x"]) & (xy[:, 1] >= ext["min_y"]) & (xy[:, 1] <= ext["max_y"])
        seg = np.hypot(*np.diff(xy, axis=0).T)
        lengths[t["highway"]][surface_group(t)] += float((seg * (inside[1:] & inside[:-1])).sum())
        g = surface_group(t)
        ax.plot(xy[:, 0], xy[:, 1], color=color[g], lw=2.5 if g == "paved" else 1.2)
    r = np.array([w["enu"][:2] for w in mission["route"]])
    ax.plot(r[:, 0], r[:, 1], "c-", lw=2, label="planned route")
    a, b = mission["a"]["enu"], mission["b"]["enu"]
    ax.plot(*a, "go", ms=12)
    ax.plot(*b, "ro", ms=12)
    ax.text(a[0] + 15, a[1] + 15, "A", color="w", fontsize=16, weight="bold")
    ax.text(b[0] + 15, b[1] + 15, "B", color="w", fontsize=16, weight="bold")
    for g, c in color.items():
        ax.plot([], [], color=c, lw=3, label=f"road: {g}")
    ax.legend(loc="lower right")
    ax.set_xlim(ext["min_x"], ext["max_x"])
    ax.set_ylim(ext["min_y"], ext["max_y"])
    ax.set_title(f"{args.mission}: roads by OSM surface tag, {wm['ortho']['source']}")
    out = Path(args.out) if args.out else d / "roads.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=80, bbox_inches="tight")
    print(f"figure: {out}")
    print("\n| highway | paved km | unpaved km | untagged km |\n|---|---:|---:|---:|")
    tot = defaultdict(float)
    for h in ROADS:
        if h in lengths:
            v = lengths[h]
            for k in color:
                tot[k] += v[k]
            print(f"| {h} | {v['paved'] / 1e3:.2f} | {v['unpaved'] / 1e3:.2f} | {v['untagged'] / 1e3:.2f} |")
    print(f"| **all** | **{tot['paved'] / 1e3:.2f}** | **{tot['unpaved'] / 1e3:.2f}** | **{tot['untagged'] / 1e3:.2f}** |")
    s = sum(tot.values())
    if s:
        print(f"\nshares: paved {tot['paved'] / s:.0%}, unpaved {tot['unpaved'] / s:.0%}, untagged {tot['untagged'] / s:.0%}")


if __name__ == "__main__":
    main()
