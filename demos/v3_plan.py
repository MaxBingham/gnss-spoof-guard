"""V3 demo: information map + planned routes (shortest vs information-aware) with update zones and the
predicted 2-sigma along each route. Saves $SFHACK_DATA/v3/plan_<world>.png.

    python demos/v3_plan.py --world s2 --sources trn,ibn
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sfhack.planning.route_planner import RoutePlanner  # noqa: E402
from sfhack.world.build_v3 import v3_dir, worlds_dir  # noqa: E402
from sfhack.world.onboard_terrain import OnboardTerrainMap  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="s2")
    ap.add_argument("--sources", default="trn,ibn")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()
    sources = {s for s in args.sources.split(",") if s}
    om = OnboardTerrainMap(worlds_dir() / args.world)
    pl = RoutePlanner(om)
    best = pl.plan(sources)
    base = pl.plan(sources, informed=False)
    for line in best.log:
        print(line)

    ext = pl.grid.extent
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    for ax, arr, title, cmap in ((axes[0], pl.u_trn, "TRN usefulness (DSM roughness + slope vs noise)", "viridis"),
                                 (axes[1], pl.u_ibn, "IBN usefulness (image texture)", "magma")):
        ax.imshow(arr, extent=(ext[0], ext[2], ext[1], ext[3]), cmap=cmap, vmin=0, vmax=1)
        ax.imshow(np.ma.masked_where(~pl.blocked, pl.blocked), extent=(ext[0], ext[2], ext[1], ext[3]),
                  cmap="Reds", alpha=0.5, vmin=0, vmax=1)
        for poly in om.no_fly:
            ax.fill(*np.vstack([poly, poly[:1]]).T, color="red", alpha=0.6, label="no-fly")
        ax.plot(*base.xy.T, "w--", lw=1.5, label=f"shortest {base.length:.0f} m")
        ax.plot(*best.xy.T, "c-", lw=2, label=f"chosen {best.length:.0f} m")
        for z in best.zones:
            m = (best.s >= z.s0) & (best.s <= z.s1)
            ax.plot(*best.xy[m].T, "-", lw=5, alpha=0.6, color="lime" if z.kind == "trn" else "orange")
        ax.plot(*om.a_enu, "go", ms=9)
        ax.plot(*om.b_enu, "r*", ms=14)
        ax.set_title(title)
        ax.set_xlabel("east [m]")
        ax.legend(loc="lower left", fontsize=8)
    axes[2].plot(base.s, base.sigma2_pred, "k--", label="shortest route")
    axes[2].plot(best.s, best.sigma2_pred, "c-", label="chosen route")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("distance along route [m]")
    axes[2].set_ylabel("predicted 2-sigma [m]")
    axes[2].set_title(f"sources: INS + {', '.join(sorted(sources)).upper() or 'none'}")
    axes[2].grid(True, which="both", alpha=0.3)
    axes[2].legend()
    fig.suptitle(om.scenario.title)
    fig.tight_layout()
    out = v3_dir() / f"plan_{args.world}.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
