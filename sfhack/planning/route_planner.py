"""Route planning with update zones: trade distance against expected localization quality.

1. Information map (onboard DSM + onboard imagery only):
   - TRN usefulness per cell and flight direction: surface roughness inside the altimeter track window
     (residual std) plus the slope along the direction of flight, as a signal-to-noise ratio against the
     TRN measurement noise (altimeter + map error): u = s^2 / (s^2 + sigma_meas^2).
   - IBN usefulness: image texture (local std of the orthophoto) inside the template footprint.
2. A* on a grid (8-connected): cost per move = length x (1 + w x (1 - usefulness)); no-fly polygons (with a
   margin) are blocked. Several weights w give candidate routes, from the shortest feasible (w = 0) to strongly
   information-seeking ones.
3. Each candidate is scored with a covariance model of the INS + fixes along the route (1D position /
   velocity / accel-bias Kalman filter: drift between zones, fixes inside zones). Choose the route minimising
   length + lambda x (2-sigma at B), and explain the choice in the log.
4. Update zones = route stretches where TRN / IBN are expected to work (marked for the autopilot and the view).
5. Altitude profile: terrain following over the onboard DSM (max over a lateral buffer and along-track window)
   + cruise height above the surface, slope limited.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..world.onboard_terrain import OnboardTerrainMap
from ..world.surface import Grid, roughness

DIRS = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]  # (drow, dcol); row grows south


@dataclass
class PlannerConfig:
    cell_m: float = 10.0
    trn_window_m: float = 20.0
    trn_sigma_meas: float = 2.5  # effective: altimeter + map error, inflated for along-track correlation
    ibn_window_m: float = 30.0
    ibn_texture_ref: float = 0.10  # gray std giving full IBN usefulness
    weights: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0)
    lambda_sigma: float = 20.0  # m of extra route accepted per m of 2-sigma at B
    lambda_max: float = 2.0  # ... and per m of the largest 2-sigma along the route (mid-route risk)
    no_fly_margin_m: float = 20.0
    zone_thr: float = 0.5
    min_zone_m: float = 30.0
    speed: float = 10.0
    # covariance model (per axis): white accel noise, accel-bias random walk, initial bias
    accel_noise: float = 0.12
    bias_std0: float = 0.03
    bias_rw: float = 2e-3
    trn_fix_sigma: float = 2.0  # m, expected TRN fix sigma in fully useful terrain
    trn_converge_m: float = 60.0  # m of useful terrain before the first TRN fix
    ibn_fix_sigma: float = 1.0
    ibn_rate_hz: float = 1.0
    landing_zone_m: float = 60.0  # IBN zone before B (precision landing)
    # altitude profile
    cruise_agl: float = 35.0
    lateral_buffer_m: float = 15.0
    along_window_m: float = 30.0
    max_gradient: float = 0.5
    approach_agl: float = 25.0
    roof_clearance: float = 12.0  # urban: minimum height above roofs (cruise height is over the terrain)
    ground_window_m: float = 0.0  # urban: terrain estimate = morphological opening of the DSM (0 = DSM itself)
    tall_penalty_m: float = 5.0  # urban: cost x (1 + excess / this) where buildings force a climb
    min_corner_speed: float = 3.0
    max_lat_acc: float = 2.0


def planner_config_for(om: OnboardTerrainMap) -> PlannerConfig:
    sc = om.scenario
    if sc.kind == "urban":
        return PlannerConfig(cell_m=sc.planner_cell_m, speed=sc.cruise_speed, cruise_agl=sc.cruise_agl,
                             lateral_buffer_m=6.0, along_window_m=10.0, ground_window_m=61.0)
    return PlannerConfig(cell_m=sc.planner_cell_m, trn_window_m=200.0, trn_sigma_meas=7.0, ibn_window_m=40.0,
                         lambda_sigma=30.0, no_fly_margin_m=100.0, min_zone_m=300.0, speed=sc.cruise_speed,
                         trn_fix_sigma=10.0, trn_converge_m=500.0, ibn_fix_sigma=1.5,
                         landing_zone_m=sc.landing_patch_m / 2, cruise_agl=sc.cruise_agl, lateral_buffer_m=60.0,
                         along_window_m=200.0, max_gradient=0.35, approach_agl=40.0, max_lat_acc=2.5)


@dataclass
class Zone:
    kind: str  # trn | ibn
    s0: float
    s1: float


@dataclass
class Route:
    xy: np.ndarray  # (N, 2) resampled polyline
    s: np.ndarray  # (N,) arc length
    z: np.ndarray  # (N,) altitude setpoint profile (onboard map datum)
    speed: np.ndarray  # (N,) speed limit (corners)
    u_trn: np.ndarray  # (N,) expected TRN usefulness along the route
    u_ibn: np.ndarray
    zones: list[Zone] = field(default_factory=list)
    sigma2_pred: np.ndarray | None = None  # predicted 2-sigma along the route
    log: list[str] = field(default_factory=list)

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def at(self, s: float) -> tuple[np.ndarray, float, float, np.ndarray]:
        """(xy, z, speed limit, unit tangent) at arc length s."""
        s = float(np.clip(s, 0.0, self.s[-1]))
        i = int(np.clip(np.searchsorted(self.s, s) - 1, 0, len(self.s) - 2))
        f = (s - self.s[i]) / max(self.s[i + 1] - self.s[i], 1e-9)
        xy = self.xy[i] * (1 - f) + self.xy[i + 1] * f
        z = float(self.z[i] * (1 - f) + self.z[i + 1] * f)
        tan = self.xy[i + 1] - self.xy[i]
        tan = tan / max(float(np.hypot(*tan)), 1e-9)
        return xy, z, float(min(self.speed[i], self.speed[i + 1])), tan

    def project(self, xy: np.ndarray, s_hint: float, window: float = 200.0) -> float:
        """Arc length of the route point closest to xy (searched around s_hint)."""
        i0, i1 = np.searchsorted(self.s, [s_hint - window, s_hint + window])
        if i1 <= i0:
            return s_hint
        seg = self.xy[i0:i1]
        d2 = (seg[:, 0] - xy[0]) ** 2 + (seg[:, 1] - xy[1]) ** 2
        return float(self.s[i0 + int(np.argmin(d2))])

    def zone_at(self, s: float, kind: str) -> bool:
        return any(z.kind == kind and z.s0 <= s <= z.s1 for z in self.zones)


class RoutePlanner:
    def __init__(self, om: OnboardTerrainMap, cfg: PlannerConfig | None = None) -> None:
        self.om = om
        self.cfg = cfg or planner_config_for(om)
        c = self.cfg
        dsm = om.dsm
        x0, y0, x1, y1 = dsm.extent
        self.grid = Grid(np.zeros((int((y1 - y0) // c.cell_m), int((x1 - x0) // c.cell_m)), np.float32),
                         x0, y1, c.cell_m)
        X, Y = self.grid.cell_centers()
        self.cell_xy = np.stack([X, Y], -1)
        self._ground_model()
        self._info_maps()
        margin = c.no_fly_margin_m
        self.blocked = om.no_fly_mask(self.cell_xy)
        if margin > 0 and self.blocked.any():
            k = 2 * int(math.ceil(margin / c.cell_m)) + 1
            self.blocked = cv2.dilate(self.blocked.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        edge = int(math.ceil(60.0 / c.cell_m))
        self.blocked[:edge, :] = self.blocked[-edge:, :] = True
        self.blocked[:, :edge] = self.blocked[:, -edge:] = True

    # ------------------------------------------------------------------ surface / terrain

    def _ground_model(self) -> None:
        """Terrain estimate (urban: buildings removed by a morphological opening), roof heights, and the
        per-cell 'excess' height of buildings that would force a climb above the cruise height."""
        c = self.cfg
        dsm = self.om.dsm
        if c.ground_window_m > 0:
            k = max(int(round(c.ground_window_m / dsm.mpp)) | 1, 3)
            ker = np.ones((k, k), np.uint8)
            ground = cv2.GaussianBlur(cv2.dilate(cv2.erode(dsm.z, ker), ker), (0, 0), k / 4)
        else:
            ground = dsm.z.copy()
        self.ground = Grid(ground.astype(np.float32), dsm.min_x, dsm.max_y, dsm.mpp)
        kb = max(int(round(2 * c.lateral_buffer_m / dsm.mpp)) | 1, 3)
        self.roofs = Grid(cv2.dilate(dsm.z, np.ones((kb, kb), np.uint8)), dsm.min_x, dsm.max_y, dsm.mpp)
        if c.ground_window_m > 0:
            excess = np.maximum(self.roofs.z + c.roof_clearance - (ground + c.cruise_agl), 0.0)
            self.excess = Grid(excess.astype(np.float32), dsm.min_x, dsm.max_y, dsm.mpp).sample(self.cell_xy)
        else:
            self.excess = np.zeros(self.grid.shape, np.float32)

    # ------------------------------------------------------------------ information maps

    def _info_maps(self) -> None:
        c = self.cfg
        dsm = self.om.dsm
        sig_t = roughness(dsm, c.trn_window_m)
        z = dsm.z.astype(np.float64)
        gy, gx = np.gradient(z, dsm.mpp)
        gy = -gy  # north positive
        k = max(int(round(c.trn_window_m / dsm.mpp)) | 1, 3)
        gxx, gyy, gxy = (cv2.blur(a, (k, k)) for a in (gx * gx, gy * gy, gx * gy))
        pts = self.cell_xy
        S = Grid(sig_t, dsm.min_x, dsm.max_y, dsm.mpp).sample(pts)
        Gxx = Grid(gxx.astype(np.float32), dsm.min_x, dsm.max_y, dsm.mpp).sample(pts)
        Gyy = Grid(gyy.astype(np.float32), dsm.min_x, dsm.max_y, dsm.mpp).sample(pts)
        Gxy = Grid(gxy.astype(np.float32), dsm.min_x, dsm.max_y, dsm.mpp).sample(pts)
        L = c.trn_window_m
        self.u_trn_dir = []
        for dr, dc in DIRS[:4]:
            ux, uy = dc, -dr
            n = math.hypot(ux, uy)
            ux, uy = ux / n, uy / n
            slope2 = np.maximum(ux * ux * Gxx + 2 * ux * uy * Gxy + uy * uy * Gyy, 0.0)
            s2 = S * S + slope2 * L * L / 48.0
            self.u_trn_dir.append(s2 / (s2 + c.trn_sigma_meas**2))
        self.u_trn = np.max(self.u_trn_dir, axis=0)
        self.u_ibn = np.zeros(self.grid.shape, np.float32)
        if self.om.ortho is not None:
            o = self.om.ortho
            tex = roughness(o, c.ibn_window_m)
            inside = o.contains(pts, c.ibn_window_m)
            self.u_ibn = np.where(inside, np.clip(Grid(tex, o.min_x, o.max_y, o.mpp).sample(pts) / c.ibn_texture_ref,
                                                  0, 1), 0.0).astype(np.float32)
            if self.om.scenario.kind == "urban":
                # parallax and occlusion between tall buildings hurt image matching at low height
                self.u_ibn *= np.clip(1.0 - S / 15.0, 0.3, 1.0)

    def usefulness(self, xy: np.ndarray, sources: set[str]) -> tuple[np.ndarray, np.ndarray]:
        ut = self.grid_sample(self.u_trn, xy) if "trn" in sources else np.zeros(len(xy))
        ui = self.grid_sample(self.u_ibn, xy) if "ibn" in sources else np.zeros(len(xy))
        return ut, ui

    def grid_sample(self, arr: np.ndarray, xy: np.ndarray) -> np.ndarray:
        return Grid(arr.astype(np.float32), self.grid.min_x, self.grid.max_y, self.grid.mpp).sample(xy)

    # ------------------------------------------------------------------ A*

    def _astar(self, start: np.ndarray, goal: np.ndarray, w: float, sources: set[str]) -> np.ndarray | None:
        g = self.grid
        H, W = g.shape
        r0, c0 = (int(round(v)) for v in g.xy_to_ij(start))
        r1, c1 = (int(round(v)) for v in g.xy_to_ij(goal))
        blocked = self.blocked.copy()
        for r, cc in ((r0, c0), (r1, c1)):  # A and B are always reachable
            blocked[max(r - 1, 0):r + 2, max(cc - 1, 0):cc + 2] = False
        ui = self.u_ibn if "ibn" in sources else np.zeros_like(self.u_ibn)
        cost_dir = []
        for k in range(4):
            ut = self.u_trn_dir[k] if "trn" in sources else np.zeros_like(self.u_trn)
            cost_dir.append((1.0 + w * (1.0 - np.maximum(ut, ui))) * (1.0 + self.excess / self.cfg.tall_penalty_m))
        step = [math.hypot(dr, dc) * g.mpp for dr, dc in DIRS]
        INF = math.inf
        dist = np.full((H, W), INF)
        prev = np.full((H, W, 2), -1, np.int32)
        dist[r0, c0] = 0.0
        pq = [(0.0, 0.0, r0, c0)]
        while pq:
            f, d, r, cc = heapq.heappop(pq)
            if d > dist[r, cc]:
                continue
            if (r, cc) == (r1, c1):
                break
            for k, (dr, dc) in enumerate(DIRS):
                nr, nc = r + dr, cc + dc
                if not (0 <= nr < H and 0 <= nc < W) or blocked[nr, nc]:
                    continue
                cm = 0.5 * (cost_dir[k % 4][r, cc] + cost_dir[k % 4][nr, nc])
                nd = d + step[k] * cm
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    prev[nr, nc] = (r, cc)
                    h = math.hypot(nr - r1, nc - c1) * g.mpp
                    heapq.heappush(pq, (nd + h, nd, nr, nc))
        if not np.isfinite(dist[r1, c1]):
            return None
        path = [(r1, c1)]
        while path[-1] != (r0, c0):
            path.append(tuple(prev[path[-1]]))
        rc = np.array(path[::-1], float)
        xy = g.ij_to_xy(rc[:, 0], rc[:, 1])
        xy[0], xy[-1] = start, goal
        return xy

    def _shortcut(self, xy: np.ndarray, w: float, sources: set[str]) -> np.ndarray:
        """String pulling: replace path pieces by straight segments that are free and not more costly."""
        out = [xy[0]]
        i = 0
        n = len(xy)
        while i < n - 1:
            j_best = i + 1
            for j in range(n - 1, i + 1, -1):
                seg = np.linspace(xy[i], xy[j], max(int(np.hypot(*(xy[j] - xy[i])) / (self.grid.mpp / 2)), 2))
                if self._blocked_any(seg):
                    continue
                if self._cost(seg, w, sources) <= 1.02 * self._cost(xy[i:j + 1], w, sources):
                    j_best = j
                    break
            out.append(xy[j_best])
            i = j_best
        return np.array(out)

    def _blocked_any(self, pts: np.ndarray) -> bool:
        r, c = self.grid.xy_to_ij(pts)
        r = np.clip(np.round(r).astype(int), 0, self.grid.shape[0] - 1)
        c = np.clip(np.round(c).astype(int), 0, self.grid.shape[1] - 1)
        inner = self.blocked[r, c]
        inner[0] = inner[-1] = False
        return bool(inner.any())

    def _cost(self, pts: np.ndarray, w: float, sources: set[str]) -> float:
        d = np.diff(pts, axis=0)
        seg = np.hypot(d[:, 0], d[:, 1])
        mid = 0.5 * (pts[1:] + pts[:-1])
        ut, ui = self.usefulness(mid, sources)
        ex = self.grid_sample(self.excess, mid)
        return float((seg * (1.0 + w * (1.0 - np.maximum(ut, ui))) * (1.0 + ex / self.cfg.tall_penalty_m)).sum())

    # ------------------------------------------------------------------ evaluation

    def predict_sigma(self, xy: np.ndarray, s: np.ndarray, sources: set[str]) -> np.ndarray:
        """Predicted 2-sigma (horizontal, per axis) along a resampled route: INS drift + zone fixes."""
        c = self.cfg
        ut, ui = self.usefulness(xy, sources)
        P = np.diag([1.0, 0.01, c.bias_std0**2])
        out = np.zeros(len(s))
        useful_run = 0.0
        t_since_ibn = 0.0
        for k in range(len(s)):
            ds = s[k] - s[k - 1] if k else 0.0
            dt = ds / c.speed
            if dt > 0:
                F = np.array([[1, dt, -0.5 * dt * dt], [0, 1, -dt], [0, 0, 1]])
                q = c.accel_noise**2
                Q = np.array([[q * dt**3 / 3, q * dt**2 / 2, 0], [q * dt**2 / 2, q * dt, 0], [0, 0, c.bias_rw**2 * dt]])
                P = F @ P @ F.T + Q
            useful_run = useful_run + ds if ut[k] > c.zone_thr else 0.0
            fixes: list[float] = []
            if useful_run >= c.trn_converge_m and dt > 0:
                fixes += [c.trn_fix_sigma / math.sqrt(ut[k])] * max(int(round(dt)), 1)
            t_since_ibn += dt
            if ui[k] > c.zone_thr and t_since_ibn >= 1.0 / c.ibn_rate_hz:
                fixes.append(c.ibn_fix_sigma / math.sqrt(ui[k]))
                t_since_ibn = 0.0
            for sf in fixes:
                Hm = np.array([[1.0, 0.0, 0.0]])
                S_ = Hm @ P @ Hm.T + sf * sf
                K = P @ Hm.T / S_
                P = (np.eye(3) - K @ Hm) @ P
            out[k] = 2.0 * math.sqrt(P[0, 0])
        return out

    # ------------------------------------------------------------------ public

    def plan(self, sources: set[str], informed: bool = True) -> Route:
        c = self.cfg
        a, b = self.om.a_enu, self.om.b_enu
        log: list[str] = []
        weights = c.weights if (informed and sources) else (0.0,)
        cands = []
        for w in weights:
            path = self._astar(a, b, w, sources)
            if path is None:
                continue
            path = self._shortcut(path, w, sources)
            r = self._finish(path, sources)
            length = r.length
            s2b = float(r.sigma2_pred[-1])
            cands.append((length + c.lambda_sigma * s2b + c.lambda_max * float(r.sigma2_pred.max()), w, r))
            log.append(f"candidate w={w:.1f}: {length:.0f} m, expected 2-sigma at B {s2b:.1f} m, "
                       f"max {float(r.sigma2_pred.max()):.1f} m")
        if not cands:
            raise RuntimeError("no feasible route (no-fly zones block A -> B)")
        cands.sort(key=lambda x: x[0])
        _, w_best, best = cands[0]
        base = next(r for _, w, r in cands if w == 0.0)
        extra = best.length - base.length
        if w_best == 0.0 or extra < 1.0:
            why = (f"PLAN: shortest feasible route {best.length:.0f} m"
                   f"{' (avoids no-fly zones)' if self.om.no_fly else ''}, expected 2-sigma at B "
                   f"{best.sigma2_pred[-1]:.1f} m")
        else:
            gain = "TRN" if "trn" in sources else "IBN"
            what = "rough terrain / buildings" if self.om.scenario.kind == "urban" else "ridges"
            why = (f"PLAN: detour {extra:+.0f} m over {what} for {gain} updates, expected 2-sigma at B: "
                   f"{best.sigma2_pred[-1]:.1f} m vs {base.sigma2_pred[-1]:.1f} m, max along route "
                   f"{best.sigma2_pred.max():.0f} m vs {base.sigma2_pred.max():.0f} m on the shortest route")
        best.log = log + [why] + [f"UPDATE_ZONE {z.kind.upper()} {z.s0:.0f}-{z.s1:.0f} m ({z.s1 - z.s0:.0f} m)"
                                  for z in best.zones]
        return best

    def _finish(self, corners: np.ndarray, sources: set[str]) -> Route:
        c = self.cfg
        pts = corners
        for _ in range(3):  # Chaikin corner cutting
            q = 0.75 * pts[:-1] + 0.25 * pts[1:]
            r = 0.25 * pts[:-1] + 0.75 * pts[1:]
            mid = np.empty((2 * len(q), 2))
            mid[0::2], mid[1::2] = q, r
            pts = np.vstack([pts[:1], mid, pts[-1:]])
        d = np.hypot(*np.diff(pts, axis=0).T)
        s = np.concatenate([[0.0], np.cumsum(d)])
        step = c.cell_m / 2
        ss = np.arange(0.0, s[-1], step)
        ss = np.append(ss, s[-1])
        xy = np.column_stack([np.interp(ss, s, pts[:, 0]), np.interp(ss, s, pts[:, 1])])
        ut, ui = self.usefulness(xy, sources)
        z = self._altitude_profile(xy, ss)
        speed = self._speed_profile(xy, ss)
        route = Route(xy, ss, z, speed, ut, ui)
        route.zones = self._zones(ss, ut, "trn") + self._zones(ss, ui, "ibn")
        if "ibn" in sources and self.om.ortho is not None and self.om.ortho.contains(self.om.b_enu):
            route.zones.append(Zone("ibn", max(ss[-1] - c.landing_zone_m, 0.0), ss[-1]))
        route.sigma2_pred = self.predict_sigma(xy, ss, sources)
        return route

    def _zones(self, s: np.ndarray, u: np.ndarray, kind: str) -> list[Zone]:
        c = self.cfg
        good = u > c.zone_thr
        zones = []
        i = 0
        while i < len(s):
            if good[i]:
                j = i
                while j + 1 < len(s) and good[j + 1]:
                    j += 1
                if s[j] - s[i] >= c.min_zone_m:
                    zones.append(Zone(kind, float(s[i]), float(s[j])))
                i = j + 1
            else:
                i += 1
        return zones

    def _altitude_profile(self, xy: np.ndarray, s: np.ndarray) -> np.ndarray:
        c = self.cfg
        dsm = self.om.dsm
        n_w = max(int(round(c.along_window_m / max(s[1] - s[0], 1e-6))), 1)

        def window_max(h: np.ndarray) -> np.ndarray:
            return np.array([h[max(i - n_w, 0):i + n_w + 1].max() for i in range(len(h))])

        roofs = window_max(self.roofs.sample(xy))
        if c.ground_window_m > 0:  # urban: cruise height over the terrain, roof clearance over buildings
            z = np.maximum(window_max(self.ground.sample(xy)) + c.cruise_agl, roofs + c.roof_clearance)
        else:  # natural: cruise height over the highest surface around the track
            z = roofs + c.cruise_agl
        z[-1] = max(z[-1], float(dsm.sample(xy[-1])) + c.approach_agl)
        # slope limit: climb early enough before obstacles, descend gently after them
        for i in range(len(z) - 2, -1, -1):
            z[i] = max(z[i], z[i + 1] - c.max_gradient * (s[i + 1] - s[i]))
        for i in range(1, len(z)):
            z[i] = max(z[i], z[i - 1] - c.max_gradient * (s[i] - s[i - 1]))
        return z

    def _speed_profile(self, xy: np.ndarray, s: np.ndarray) -> np.ndarray:
        c = self.cfg
        v = np.full(len(s), c.speed)
        if len(xy) >= 3:
            a = np.arctan2(*np.diff(xy, axis=0).T[::-1])
            dth = np.abs(np.angle(np.exp(1j * np.diff(a))))
            ds = np.maximum(np.diff(s)[1:], 1e-6)
            kappa = dth / ds
            vc = np.sqrt(c.max_lat_acc / np.maximum(kappa, 1e-6))
            v[1:-1] = np.minimum(v[1:-1], np.maximum(vc, c.min_corner_speed))
            for i in range(len(v) - 2, -1, -1):  # decelerate before corners (2 m/s^2)
                v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * 2.0 * (s[i + 1] - s[i])))
        return v
