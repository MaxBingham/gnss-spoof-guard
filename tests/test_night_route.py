"""Night route choice (sfhack/planning/night_route.py) on a synthetic flat map: straight when the straight
line has warm road under it, along the roads when it would be blind."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sfhack.missionprep.geometry import Grid
from sfhack.planning.night_route import NightRouteParams, plan_night_route


def _map(road_on_line: bool):
    g = Grid(-300.0, 300.0, 1.0, 600, 600)  # 600 x 600 m, flat
    flat = np.zeros(g.shape, np.float32)
    om = SimpleNamespace(dsm=flat, dtm=flat, dem_grid=g, a_enu=np.array([-250.0, -250.0]),
                         b_enu=np.array([250.0, 250.0]), ground_z=lambda xy: 0.0)
    res = 0.5
    roads = np.zeros((1200, 1200), np.uint8)
    col = lambda x: int((x + 300) / res)  # noqa: E731
    row = lambda y: int((300 - y) / res)  # noqa: E731
    if road_on_line:  # 6 m wide road along the diagonal A -> B
        for t in np.linspace(-260, 260, 3000):
            roads[max(row(t) - 6, 0):row(t) + 6, max(col(t) - 6, 0):col(t) + 6] = 1
    else:  # L-shaped road: east along y = -250, then north along x = 250
        roads[row(-247):row(-253), col(-260):col(256)] = 1
        roads[row(260):row(-256), col(247):col(253)] = 1
    prior = SimpleNamespace(min_x=-300.0, max_y=300.0, res=res, roads=roads)
    return om, prior


def test_straight_when_the_line_sees_road() -> None:
    om, prior = _map(road_on_line=True)
    r = plan_night_route(om, prior, NightRouteParams())
    assert r.choice == "STRAIGHT" and r.straight.longest_blind_m <= 250.0


def test_detour_along_roads_when_the_line_is_blind() -> None:
    om, prior = _map(road_on_line=False)
    r = plan_night_route(om, prior, NightRouteParams(max_detour_frac=0.6))
    assert r.straight.longest_blind_m > 250.0
    assert r.choice == "ROADS", r.summary()
    assert r.roads is not None and r.roads.longest_blind_m < 60.0 and r.roads.visible_frac > 0.9
    # a stricter detour budget keeps the straight line (the L is ~41 % longer than the diagonal)
    assert plan_night_route(om, prior, NightRouteParams(max_detour_frac=0.2)).choice == "STRAIGHT"
