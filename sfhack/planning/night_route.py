"""Night route choice: fly the straight (planned) route while the thermal camera can expect warm road in its
footprint, else detour along the roads.

Onboard planning code: reads only the onboard package (dsm.npz via OnboardMap, thermal_prior.npz via
sfhack.thermal.prior). At night the thermal localizer gets its fixes from warm asphalt; over forest or fields
it sees nothing and the estimate drifts on IMU + TRN alone. So:

1. Visibility per 4 m route cell: paved-road area (prior `roads` mask) inside the camera footprint at the
   cruise height (disc of radius FOOTPRINT_FRAC x half the ground width; 640 x 512 px, 50 deg HFOV at 60 m AGL
   -> 56 m wide) >= `min_road_area_m2`.
2. The planned route (straight line around obstacles) is sampled every 4 m. If its longest stretch without a
   visible road ("blind gap") is <= `max_blind_gap_m`, it is kept: STRAIGHT.
3. Else the route is re-planned (same Dijkstra on the onboard DSM as the mission planner) with every metre over
   a blind cell costing `blind_cost` extra: ROADS. The detour is kept only if it shortens the longest blind
   gap and is at most `max_detour_frac` longer; else STRAIGHT (with the warning in the report).

Defaults and why (docs/NIGHT_DEMO.md, "Route choice"): 150 m^2 of road ~ a 5 m wide road crossing 30 m of the
56 m footprint; 250 m blind ~ 30 s at 8 m/s, about what IMU + TRN hold to a few metres.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np

from ..config import MissionConfig
from ..missionprep.geometry import Grid
from ..world.onboard_map import OnboardMap

THERMAL_HFOV_DEG = 50.0
ROUTE_CELL_M = 4.0  # same grid as the mission planner's route raster


@dataclass
class NightRouteParams:
    cruise_agl_m: float = 60.0
    min_road_area_m2: float = 150.0  # paved road inside the footprint for a cell to count as "visible"
    max_blind_gap_m: float = 250.0  # longest blind stretch the straight route may have
    blind_cost: float = 3.0  # extra cost per metre over a blind cell when detouring
    max_detour_frac: float = 0.6  # detour at most 60 % longer than the straight route
    footprint_frac: float = 0.9  # disc radius = this x half the footprint width (inscribed, any heading)


@dataclass
class RouteScore:
    length_m: float
    visible_frac: float
    longest_blind_m: float
    blind_m: float


@dataclass
class NightRouteResult:
    choice: str  # STRAIGHT | ROADS
    route: list[dict[str, Any]]
    straight: RouteScore
    roads: RouteScore | None
    params: NightRouteParams
    reason: str
    visible: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 0), bool))
    grid: Grid | None = field(repr=False, default=None)

    def summary(self) -> str:
        s = self.straight
        out = (f"night route: {self.choice} ({self.reason}); straight {s.length_m:.0f} m, road visible "
               f"{s.visible_frac:.0%}, longest blind {s.longest_blind_m:.0f} m")
        if self.roads is not None:
            r = self.roads
            out += f"; roads {r.length_m:.0f} m, visible {r.visible_frac:.0%}, longest blind {r.longest_blind_m:.0f} m"
        return out

    def report(self) -> dict[str, Any]:
        return {"choice": self.choice, "reason": self.reason, "params": asdict(self.params),
                "straight": asdict(self.straight), "roads": asdict(self.roads) if self.roads else None}


def road_visibility(roads: np.ndarray, roads_grid: Grid, g4: Grid, p: NightRouteParams) -> np.ndarray:
    """bool (rows, cols) on the 4 m route grid: enough paved road inside the footprint centred there."""
    from ..missionprep.geometry import resample

    k = max(1, int(round(ROUTE_CELL_M / roads_grid.res)))
    R, C = roads.shape[0] // k, roads.shape[1] // k
    area = roads[:R * k, :C * k].reshape(R, k, C, k).astype(np.float32).sum(axis=(1, 3)) * roads_grid.res ** 2
    coarse = Grid(roads_grid.min_x, roads_grid.max_y, k * roads_grid.res, R, C)
    area = resample(area, coarse, g4) if coarse != g4 else area  # m^2 of road per 4 m cell
    half_w = p.cruise_agl_m * math.tan(math.radians(THERMAL_HFOV_DEG) / 2)
    r = max(1, int(round(p.footprint_frac * half_w / g4.res)))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)).astype(np.float32)
    in_fp = cv2.filter2D(area, -1, disk, borderType=cv2.BORDER_CONSTANT)
    return in_fp >= p.min_road_area_m2


def score_route(route: list[dict[str, Any]], visible: np.ndarray, g4: Grid) -> RouteScore:
    """Sample the cruise part (TAKEOFF .. last CRUISE waypoint) every 2 m."""
    xy = np.array([w["enu"][:2] for w in route if w["phase"] in ("TAKEOFF", "CRUISE")], float)
    pts = [xy[0]]
    for a, b in zip(xy[:-1], xy[1:]):
        n = max(1, int(math.ceil(float(np.hypot(*(b - a))) / 2.0)))
        pts += [a + (b - a) * (i + 1) / n for i in range(n)]
    pts = np.array(pts)
    step = np.r_[0.0, np.hypot(*np.diff(pts, axis=0).T)]
    cr = np.round(g4.to_px(pts)).astype(int)
    vis = visible[np.clip(cr[:, 1], 0, g4.rows - 1), np.clip(cr[:, 0], 0, g4.cols - 1)]
    longest = run = blind = 0.0
    for v, d in zip(vis, step):
        if v:
            run = 0.0
        else:
            run += d
            blind += d
            longest = max(longest, run)
    L = float(step.sum())
    return RouteScore(round(L, 1), round(1.0 - blind / max(L, 1e-6), 3), round(longest, 1), round(blind, 1))


def plan_night_route(om: OnboardMap, prior: Any, p: NightRouteParams | None = None,
                     mc: MissionConfig | None = None, base_route: list[dict[str, Any]] | None = None) -> NightRouteResult:
    """`prior`: sfhack.thermal.prior.ThermalPrior (onboard). `base_route`: the straight candidate (default: re-planned
    on the onboard DSM at the cruise AGL, i.e. the mission planner's route at that height)."""
    from ..missionprep.build import plan_route, required_z

    p = p or NightRouteParams()
    mc = mc or MissionConfig()
    if om.dsm is None or om.dtm is None or om.dem_grid is None:
        raise ValueError("night route needs the onboard dsm.npz")
    g = om.dem_grid
    a, b = om.a_enu, om.b_enu
    b_z = om.ground_z(b)
    _, g4 = required_z(om.dsm, om.dtm, g, p.cruise_agl_m)
    rg = Grid(prior.min_x, prior.max_y, prior.res, *prior.roads.shape)
    visible = road_visibility(np.asarray(prior.roads > 0, np.uint8), rg, g4, p)
    if base_route is None:
        base_route, _ = plan_route(a, b, 0.0, b_z, om.dsm, om.dtm, g, mc, p.cruise_agl_m)
    s_straight = score_route(base_route, visible, g4)
    if s_straight.longest_blind_m <= p.max_blind_gap_m:
        return NightRouteResult("STRAIGHT", base_route, s_straight, None, p,
                                f"longest blind stretch {s_straight.longest_blind_m:.0f} m <= {p.max_blind_gap_m:.0f} m",
                                visible, g4)
    extra = np.where(visible, 0.0, p.blind_cost).astype(np.float64)
    road_route, _ = plan_route(a, b, 0.0, b_z, om.dsm, om.dtm, g, mc, p.cruise_agl_m, extra_cost=extra)
    s_roads = score_route(road_route, visible, g4)
    longer = s_roads.length_m / max(s_straight.length_m, 1.0) - 1.0
    if s_roads.longest_blind_m < s_straight.longest_blind_m and longer <= p.max_detour_frac:
        return NightRouteResult("ROADS", road_route, s_straight, s_roads, p,
                                f"straight blind for {s_straight.longest_blind_m:.0f} m > {p.max_blind_gap_m:.0f} m; "
                                f"along roads {longer:+.0%} length, blind {s_roads.longest_blind_m:.0f} m", visible, g4)
    return NightRouteResult("STRAIGHT", base_route, s_straight, s_roads, p,
                            f"detour does not help enough ({longer:+.0%} length, blind {s_roads.longest_blind_m:.0f} m)",
                            visible, g4)


def main() -> None:
    """python -m sfhack.planning.night_route <mission> [--agl 60] [--min-road 150] [--max-gap 250] [--png out.png]"""
    import argparse
    import json
    from pathlib import Path

    from ..config import data_dir
    from ..thermal.prior import load_prior

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mission")
    ap.add_argument("--agl", type=float, default=NightRouteParams.cruise_agl_m)
    ap.add_argument("--min-road", type=float, default=NightRouteParams.min_road_area_m2, help="m^2 in the footprint")
    ap.add_argument("--max-gap", type=float, default=NightRouteParams.max_blind_gap_m, help="m blind allowed")
    ap.add_argument("--png", default=None)
    args = ap.parse_args()
    mdir = data_dir() / "missions" / args.mission
    odir = mdir / "drone" if (mdir / "drone" / "upload_receipt.json").exists() else mdir / "onboard"
    om = OnboardMap(odir)
    res = plan_night_route(om, load_prior(odir), NightRouteParams(cruise_agl_m=args.agl, min_road_area_m2=args.min_road,
                                                                  max_blind_gap_m=args.max_gap))
    print(res.summary())
    print(json.dumps(res.report(), indent=1))
    if args.png:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        g4 = res.grid
        fig, ax = plt.subplots(figsize=(14, 14 * g4.rows / g4.cols + 1))
        ax.imshow(om.ortho, extent=(om.min_x, om.max_x, om.min_y, om.max_y))
        vis = np.zeros((*res.visible.shape, 4))
        vis[res.visible] = (1.0, 0.6, 0.1, 0.35)
        ax.imshow(vis, extent=(g4.min_x, g4.max_x, g4.min_y, g4.max_y))
        base = plan_night_route(om, load_prior(odir), NightRouteParams(cruise_agl_m=args.agl, max_blind_gap_m=1e9))
        for route, col, lab in ((base.route, "w", "straight (planned)"), (res.route, "c", f"night choice: {res.choice}")):
            xy = np.array([w["enu"][:2] for w in route])
            ax.plot(xy[:, 0], xy[:, 1], col + "-", lw=2.5, label=lab)
        ax.plot(*om.a_enu, "go", ms=10)
        ax.plot(*om.b_enu, "ro", ms=10)
        ax.set_title(f"{args.mission}: orange = warm road expected in the thermal footprint at {args.agl:.0f} m AGL "
                     f"(>= {args.min_road:.0f} m^2)")
        ax.legend(loc="lower right")
        Path(args.png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.png, dpi=80, bbox_inches="tight")
        print(f"figure: {args.png}")


if __name__ == "__main__":
    main()
