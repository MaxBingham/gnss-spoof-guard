"""Recon mission (onboard): acoustic bearing in, search cone, thermal find, orbit, home on the base, land, report.

    PREFLIGHT -> TAKEOFF -> OUTBOUND -> SEARCH -> OBSERVE -> RETURN -> HOME -> LAND -> LANDED -> REPORT

A drop-in for planning.mission.Mission (same interface: phase, events, start, update, set_ground_sensing,
ground_contact, done), attached to the Autopilot with `Autopilot.attach_mission`. Everything here runs on the
onboard estimate (EKF pose + AHRS attitude), the onboard map (DTM for terrain following, map extent) and the
bearing loaded before takeoff. It never reads the truth: there is no data link and no GNSS after the first 50 m.

- Outbound along the bearing, then a zigzag across it whose half-width follows the cone (bearing sigma + own
  drift) and whose along-track step is chosen so that the pattern, the way home and the landing fit the battery.
- The tail (rear NIR camera) always points at the base (yaw = direction base -> drone): outbound it looks back
  at the lights, and home it flies tail-first, so the lights stay in the rear camera all the way.
- Thermal detections (perception/thermal_detect.py) are projected into the estimate frame and grouped into
  tracks; a static track seen `confirm_hits` times is a find -> OBSERVE: orbit it (centre = the track, updated by
  each re-detection, so the orbit is relative to the target even while the estimate drifts), store snapshots.
- RETURN when the find is observed, the pattern is done, or the battery says so (time to get home + landing +
  reserve). HOME: slow down and descend near the base; then the waypoint Mission's precision landing on the pad.
- REPORT: what was found, where (bearing + distance from the base), when, snapshots, track, searched area.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import cv2
import numpy as np

from ..config import MissionConfig, ReconConfig
from ..messages import Health, PoseEstimate, Setpoint, ThermalFrame
from ..perception.thermal_detect import Detection, ThermalDetector, to_8bit
from ..planning.mission import GROUND_TARGET_Z, Mission, Waypoint

FLYING = ("TAKEOFF", "OUTBOUND", "SEARCH", "OBSERVE", "RETURN", "HOME", "LAND")


# ---------------------------------------------------------------------------------------------- inputs


def load_bearing(src: Any) -> Any:
    """bearing.json path, dict or BearingEstimate -> an object with az_deg, sigma_az_deg (+ the contract's other
    fields when present). az_deg: degrees clockwise from north."""
    if isinstance(src, (str, Path)):
        src = json.loads(Path(src).read_text(encoding="utf-8"))
    if isinstance(src, dict):
        try:
            from ..acoustic.bearing import BearingEstimate  # the contract type (recon-acoustic / recon-world)

            names = {f.name for f in dataclasses.fields(BearingEstimate)}
            return BearingEstimate(**{k: v for k, v in src.items() if k in names})
        except (ImportError, TypeError):
            return SimpleNamespace(**{"sigma_az_deg": 7.0, "label": "", "confidence": 1.0, **src})
    return src


def bearing_dict(b: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(b):
        d = dataclasses.asdict(b)
    else:
        d = dict(vars(b))
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


def load_base(onboard_dir: Path | None, fallback_pad: np.ndarray) -> dict[str, Any]:
    """Pad position (+ yaw) from the onboard package's base.json (recon-world), else mission point A."""
    pad = np.asarray(fallback_pad, float)[:2]
    info: dict[str, Any] = {"pad_enu": pad.tolist(), "source": "mission point A"}
    p = Path(onboard_dir) / "base.json" if onboard_dir is not None else None
    if p is not None and p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            pd = d.get("pad", d)
            xy = pd.get("enu") or pd.get("pos_enu") or pd.get("pad_enu") or pd.get("center_enu")
            if xy is not None:
                info = {**d, "pad_enu": [float(xy[0]), float(xy[1])], "source": str(p.name)}
        except (ValueError, OSError, AttributeError):
            pass
    return info


def az_to_unit(az_deg: float) -> np.ndarray:
    a = math.radians(az_deg)
    return np.array([math.sin(a), math.cos(a)])


def az_of(v: np.ndarray) -> float:
    """Compass azimuth (deg clockwise from north, 0..360) of an ENU vector."""
    return math.degrees(math.atan2(v[0], v[1])) % 360.0


# ---------------------------------------------------------------------------------------------- search pattern


@dataclass
class SearchPlan:
    path: np.ndarray  # (N, 2) ENU, from the pad out along the bearing, zigzag in the cone
    r_start: float
    r_max: float
    step_m: float  # along-track advance per crossing
    swath_m: float
    est_time_s: float
    halfwidth: list[tuple[float, float]] = field(default_factory=list)  # (r, cone half-width) samples


def cone_halfwidth(r: float, sigma_az_deg: float, c: ReconConfig) -> float:
    return c.cone_k * math.hypot(r * math.tan(math.radians(sigma_az_deg)), c.pos_drift_frac * r)


def plan_search(base: np.ndarray, az_deg: float, sigma_az_deg: float, c: ReconConfig, swath_m: float,
                time_budget_s: float, inside: Callable[[np.ndarray], bool] | None = None) -> SearchPlan:
    """Pad -> straight out to r_start -> zigzag to r_max. `inside(xy)`: point within the usable map."""
    inside = inside or (lambda xy: True)
    u = az_to_unit(az_deg)
    n = np.array([u[1], -u[0]])  # right of the bearing
    r_max = c.search_max_m
    for r in np.arange(c.search_start_m, c.search_max_m + 1.0, 10.0):  # map edge along the bearing
        if not inside(base + u * r):
            r_max = max(float(r) - 30.0, c.search_start_m)
            break
    r_start = min(c.search_start_m, r_max)
    v = c.search_speed

    def pattern(step: float) -> np.ndarray:
        pts = [base.copy(), base + u * r_start]
        side, r = 1.0, r_start
        while r < r_max - 1e-6:
            r = min(r + step, r_max)
            a = max(0.0, cone_halfwidth(r, sigma_az_deg, c) - swath_m / 2)
            p = base + u * r + n * side * a
            k = 0
            while not inside(p) and k < 20:  # shrink toward the bearing line until inside the map
                a *= 0.8
                p = base + u * r + n * side * a
                k += 1
            if not inside(p):
                p = base + u * r
            pts.append(p)
            side = -side
        return np.array(pts)

    def time_of(path: np.ndarray) -> float:
        return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) / v

    lo, hi = 0.8 * swath_m, max(r_max - r_start, swath_m)
    path = pattern(hi)
    if time_of(pattern(lo)) <= time_budget_s:
        hi = lo
    else:
        for _ in range(30):  # bisection on the along-track step
            mid = 0.5 * (lo + hi)
            if time_of(pattern(mid)) <= time_budget_s:
                hi = mid
            else:
                lo = mid
    step = hi
    path = pattern(step)
    hw = [(float(r), cone_halfwidth(float(r), sigma_az_deg, c)) for r in np.linspace(0, r_max, 12)]
    return SearchPlan(path, r_start, r_max, step, swath_m, time_of(path), hw)


# ---------------------------------------------------------------------------------------------- finds


@dataclass
class Track:
    id: int
    label: str
    xy: np.ndarray  # mean position, estimate frame
    hits: list[tuple[float, np.ndarray, float]] = field(default_factory=list)  # (t, xy, conf)
    sigma2_at_first: float = 0.0  # 2-sigma of the drone estimate at the first detection
    size_m: tuple[float, float] = (0.0, 0.0)
    temp_K: float = 0.0
    confirmed_t: float | None = None  # candidate: seen confirm_hits times on one pass
    verified: bool = False  # seen again from the orbit (other viewing angles): a find
    rejected: bool = False  # candidate not seen again from the orbit (oblique wall, glint, ...)
    moving: bool = False
    labels: dict[str, int] = field(default_factory=dict)
    snapshots: list[dict[str, Any]] = field(default_factory=list)  # {t, image (uint8), crop (uint8), ...}

    @property
    def n(self) -> int:
        return len(self.hits)

    @property
    def conf(self) -> float:
        return float(np.mean([h[2] for h in self.hits[-10:]])) if self.hits else 0.0

    def speed(self) -> float:
        if len(self.hits) < 3:
            return 0.0
        t = np.array([h[0] for h in self.hits[-15:]])
        if t[-1] - t[0] < 1.0:
            return 0.0
        xy = np.array([h[1] for h in self.hits[-15:]])
        vx = np.polyfit(t - t[0], xy[:, 0], 1)[0]
        vy = np.polyfit(t - t[0], xy[:, 1], 1)[0]
        return float(math.hypot(vx, vy))

    def add(self, t: float, xy: np.ndarray, d: Detection) -> None:
        self.hits.append((t, xy.copy(), d.conf))
        self.hits = self.hits[-60:]
        self.labels[d.label] = self.labels.get(d.label, 0) + 1
        self.label = max(self.labels, key=self.labels.get)
        w = 1.0 / min(len(self.hits), 20)
        self.xy = (1 - w) * self.xy + w * xy
        self.size_m = d.size_m if d.conf >= self.conf else self.size_m
        self.temp_K = max(self.temp_K, d.temp_K)


# ---------------------------------------------------------------------------------------------- mission


class ReconMission:
    def __init__(self, cfg: ReconConfig, mission_cfg: MissionConfig, bearing: Any, pad_xy: np.ndarray,
                 start_yaw: float, ground_z: Callable[[np.ndarray], float] | None = None,
                 map_inside: Callable[[np.ndarray, float], bool] | None = None, onboard_map: Any | None = None,
                 detector: ThermalDetector | None = None, mission_id: str = "", thermal_hfov_deg: float = 50.0,
                 thermal_aspect: float = 512 / 640) -> None:
        self.c = cfg
        self.mcfg = mission_cfg
        self.bearing = load_bearing(bearing)
        self.az = float(self.bearing.az_deg) % 360.0
        self.sigma_az = float(getattr(self.bearing, "sigma_az_deg", 7.0))
        self.pad = np.asarray(pad_xy, float)[:2].copy()
        self.ground_z = ground_z or (lambda xy: 0.0)
        self.pad_z = float(self.ground_z(self.pad))
        self.map_inside = map_inside or (lambda xy, margin: True)
        self.detector = detector or ThermalDetector(onboard_map=onboard_map)
        self.mission_id = mission_id
        self.yaw = start_yaw
        self.setpoint = np.array([self.pad[0], self.pad[1], self.pad_z])
        self.phase = "PREFLIGHT"
        self.events: list[tuple[float, str]] = []
        self.decision_log: list[tuple[float, str]] = []
        self.range_agl: float | None = None
        self.imu_still_s = 0.0
        self.t_takeoff: float | None = None
        self.t_landed: float | None = None
        self.return_reason = ""
        # thermal footprint at cruise height (across track = image width, the drone flies nose-out / tail-in)
        half = math.tan(math.radians(thermal_hfov_deg) / 2)
        self.footprint = (2 * cfg.cruise_agl * half, 2 * cfg.cruise_agl * half * thermal_aspect)
        self.swath = cfg.swath_frac * self.footprint[0]
        self.plan: SearchPlan | None = None
        self.path: np.ndarray | None = None
        self.path_cum: np.ndarray | None = None
        self.path_s = 0.0
        self.path_speed = cfg.search_speed
        self.tracks: list[Track] = []
        self.observe_track: Track | None = None
        self.observe_t0: float | None = None
        self.orbit_phi = 0.0
        self.last_snap_t = -1e9
        self.inner: Mission | None = None  # waypoint Mission for the final approach + precision landing
        self.track_log: list[tuple[float, float, float, float, float, str]] = []  # t, x, y, z, sigma2, phase (1 Hz)
        self._next_track_t = 0.0
        self.max_r = 0.0
        self.frames = 0
        self.det_ms = 0.0
        self.last_result: Any = None  # last DetectResult (viewer)
        self.last_frame_t = -1.0
        # searched area: thermal footprints marked on a coarse grid (estimate frame)
        self.cov_cell = cfg.coverage_cell_m
        self.cov_origin = self.pad - cfg.search_max_m - 200.0
        nc = int(2 * (cfg.search_max_m + 200.0) / self.cov_cell)
        self.coverage = np.zeros((nc, nc), np.uint8)
        self.report: dict[str, Any] | None = None

    # ------------------------------------------------------------------ Mission interface

    def log(self, t: float, text: str) -> None:
        self.events.append((t, text))  # the autopilot pops these onto the decision topic
        self.decision_log.append((t, text))

    @property
    def done(self) -> bool:
        return self.phase in ("LANDED", "REPORT")

    @property
    def goal(self) -> np.ndarray:
        return np.array([self.pad[0], self.pad[1], self.pad_z])

    @property
    def waypoints(self) -> list[Waypoint]:  # (viewer / eval helpers expect a waypoint list)
        pts = self.path if self.path is not None else np.array([self.pad])
        z = self.pad_z + self.c.cruise_agl
        return [Waypoint(np.array([p[0], p[1], z]), self.c.search_speed, "SEARCH") for p in pts]

    @property
    def ground_contact(self) -> bool:
        return bool(self.inner is not None and self.inner.ground_contact)

    def set_ground_sensing(self, range_agl: float | None, imu_still_s: float) -> None:
        self.range_agl = range_agl
        self.imu_still_s = imu_still_s
        if self.inner is not None:
            self.inner.set_ground_sensing(range_agl, imu_still_s)

    def start(self) -> None:
        if self.phase == "PREFLIGHT":
            self.phase = "TAKEOFF"

    def elapsed(self, t: float) -> float:
        return 0.0 if self.t_takeoff is None else t - self.t_takeoff

    def clock(self, t: float) -> str:
        h = (self.c.clock_start_h + self.elapsed(t) / 3600.0) % 24.0
        return f"{int(h):02d}:{int(round((h % 1) * 60)) % 60:02d}"

    def time_to_home(self, pose: PoseEstimate) -> float:
        d = float(np.linalg.norm(pose.pos[:2] - self.pad))
        return d / self.c.return_speed + self.c.land_allow_s

    def battery_left(self, t: float) -> float:
        return self.c.battery_s - self.elapsed(t)

    # ------------------------------------------------------------------ update (IMU rate)

    def update(self, t: float, pose: PoseEstimate, dt: float, health: Health = Health.NOMINAL,
               speed_scale: float = 1.0, last_fix_pos: np.ndarray | None = None) -> Setpoint:
        if self.phase == "PREFLIGHT" or self.done:
            if self.phase == "LANDED" and self.report is None:
                self.t_landed = self.t_landed or t
                self.report = self.build_report(t)
                self.phase = "REPORT"
                self.log(t, f"REPORT: {self.report['summary']}")
            return Setpoint(t, self.setpoint.copy(), np.zeros(3), self.yaw, self.phase)
        if self.t_takeoff is None:
            self.t_takeoff = t
            self.log(t, f"TAKEOFF: bearing {self.az:.0f} deg +- {self.sigma_az:.0f} deg, battery {self.c.battery_s / 60:.0f} min")
        if t >= self._next_track_t:
            self._next_track_t = t + 1.0
            self.track_log.append((t, float(pose.pos[0]), float(pose.pos[1]), float(pose.pos[2]),
                                   pose.sigma2_horizontal(), self.phase))
        self.max_r = max(self.max_r, float(np.linalg.norm(pose.pos[:2] - self.pad)))
        vel = np.zeros(3)
        ph = self.phase
        if ph == "TAKEOFF":
            target = np.array([self.pad[0], self.pad[1], self.pad_z + self.c.takeoff_agl])
            vel = self._move_z_only(target[2], self.c.climb_speed, dt)
            if pose.pos[2] > target[2] - 2.0:
                self._start_outbound(t, pose)
        elif ph in ("OUTBOUND", "SEARCH"):
            vel = self._follow_path(t, pose, dt)
            r_sp = float(np.dot(self.setpoint[:2] - self.pad, az_to_unit(self.az)))
            if ph == "OUTBOUND" and self.plan is not None and r_sp >= self.plan.r_start - 1.0:
                self.phase = "SEARCH"
                self.log(t, f"SEARCH: {self.plan.r_start:.0f}-{self.plan.r_max:.0f} m, zigzag step "
                            f"{self.plan.step_m:.0f} m, swath {self.plan.swath_m:.0f} m, est. {self.plan.est_time_s:.0f} s")
            if self.path_cum is not None and self.path_s >= self.path_cum[-1] - 1e-6:
                self._start_return(t, pose, "search pattern complete, nothing confirmed" if not self._finds()
                                   else "search pattern complete")
            elif self.elapsed(t) + self.time_to_home(pose) + self.c.reserve_s >= self.c.battery_s:
                self._start_return(t, pose, f"BATTERY: {self.battery_left(t):.0f} s left, "
                                            f"{self.time_to_home(pose):.0f} s needed to get home and land")
        elif ph == "OBSERVE":
            vel = self._orbit(t, pose, dt)
            if self.phase == "OBSERVE" and self.elapsed(t) + self.time_to_home(pose) + self.c.reserve_s >= self.c.battery_s:
                tr = self.observe_track
                if tr is not None and not tr.verified and sum(1 for h in tr.hits if h[0] > (self.observe_t0 or t)) > 0:
                    tr.verified = True  # seen again, just not as often as wanted: keep it
                self._start_return(t, pose, f"BATTERY during the orbit: {self.battery_left(t):.0f} s left")
        elif ph == "RETURN":
            vel = self._follow_path(t, pose, dt)
            if float(np.linalg.norm(pose.pos[:2] - self.pad)) < self.c.home_radius_m:
                self.phase = "HOME"
                self._set_path(t, [pose.pos[:2], self.pad], self.c.home_speed, reset=True)
                self.log(t, f"HOME: {np.linalg.norm(pose.pos[:2] - self.pad):.0f} m from the pad, descending to "
                            f"{self.c.home_agl:.0f} m, 2-sigma {pose.sigma2_horizontal():.1f} m")
        elif ph == "HOME":
            vel = self._follow_path(t, pose, dt, agl=self.c.home_agl, terrain=False)
            off = float(np.linalg.norm(pose.pos[:2] - self.pad))
            if self.path_cum is not None and self.path_s >= self.path_cum[-1] - 1e-6 and off < 3.0 \
                    and abs(pose.pos[2] - (self.pad_z + self.c.home_agl)) < 3.0:
                self._start_landing(t)
        if self.phase == "LAND" and self.inner is not None:
            sp = self.inner.update(t, pose, dt)
            self.setpoint = self.inner.setpoint.copy()
            while self.inner.events:
                self.log(*self.inner.events.pop(0))
            if self.inner.done:
                self.phase = "LANDED"
                self.t_landed = t
            return Setpoint(t, sp.pos, sp.vel, self.yaw, "LAND")
        self._update_yaw(pose, dt)
        return Setpoint(t, self.setpoint.copy(), vel, self.yaw, self.phase)

    # ------------------------------------------------------------------ phases

    def _start_outbound(self, t: float, pose: PoseEstimate) -> None:
        budget = (self.c.battery_s - self.elapsed(t) - self.c.reserve_s - self.c.land_allow_s
                  - self.c.observe_s - 30.0)
        inside = lambda xy: self.map_inside(xy, self.c.map_margin_m)  # noqa: E731
        # time home from the far end must fit too: plan with the budget minus the way back from r_max
        plan = plan_search(self.pad, self.az, self.sigma_az, self.c, self.swath, budget, inside)
        for _ in range(3):
            back = plan.r_max / self.c.return_speed
            plan = plan_search(self.pad, self.az, self.sigma_az, self.c, self.swath, budget - back, inside)
        self.plan = plan
        self.phase = "OUTBOUND"
        self._set_path(t, list(plan.path), self.c.search_speed, reset=True)
        self.path_s = 0.0
        self.log(t, f"OUTBOUND: along {self.az:.0f} deg, search {plan.r_start:.0f}-{plan.r_max:.0f} m, cone half-width "
                    f"{cone_halfwidth(plan.r_max, self.sigma_az, self.c):.0f} m at {plan.r_max:.0f} m, pattern "
                    f"{plan.est_time_s:.0f} s of {budget:.0f} s budget")

    def _start_return(self, t: float, pose: PoseEstimate, why: str) -> None:
        self.phase = "RETURN"
        self.return_reason = why
        self._set_path(t, [pose.pos[:2], self.pad], self.c.return_speed, reset=True)
        self.log(t, f"RETURN: {why}; {np.linalg.norm(pose.pos[:2] - self.pad):.0f} m to the pad, "
                    f"2-sigma {pose.sigma2_horizontal():.0f} m, {len(self._finds())} find(s)")

    def _start_landing(self, t: float) -> None:
        mc = dataclasses.replace(self.mcfg, precision_sigma2=self.c.precision_sigma2,
                                 precision_wait_s=self.c.precision_wait_s)
        z_home = self.pad_z + self.c.home_agl
        wps = [Waypoint(np.array([self.pad[0], self.pad[1], z_home]), 2.0, "CRUISE"),
               Waypoint(np.array([self.pad[0], self.pad[1], self.pad_z + mc.flare_alt]), mc.descent_speed, "DESCEND"),
               Waypoint(np.array([self.pad[0], self.pad[1], self.pad_z + GROUND_TARGET_Z]), mc.land_speed, "LAND")]
        self.inner = Mission(wps, self.setpoint.copy(), self.yaw, mc, ground_z=self.ground_z)
        self.inner.start()
        self.inner.set_ground_sensing(self.range_agl, self.imu_still_s)
        self.phase = "LAND"
        self.log(t, "LAND: above the pad, precision landing")

    def _start_observe(self, t: float, pose: PoseEstimate, tr: Track) -> None:
        self.phase = "OBSERVE"
        self.observe_track = tr
        self.observe_t0 = t
        rel = pose.pos[:2] - tr.xy
        self.orbit_phi = math.atan2(rel[1], rel[0])
        d = tr.xy - self.pad
        self.log(t, f"OBSERVE: {tr.label} #{tr.id} confirmed ({tr.n} hits, conf {tr.conf:.2f}) at "
                    f"{np.linalg.norm(d):.0f} m / {az_of(d):.0f} deg from the base, orbit {self.c.orbit_radius_m:.0f} m "
                    f"for {self.c.observe_s:.0f} s")

    # ------------------------------------------------------------------ guidance

    def _set_path(self, t: float, pts: list[np.ndarray], speed: float, reset: bool) -> None:
        p = np.array([np.asarray(q, float)[:2] for q in pts])
        if reset:
            p[0] = self.setpoint[:2]  # start where the setpoint is (no jump)
        self.path = p
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1) if len(p) > 1 else np.zeros(0)
        self.path_cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.path_s = 0.0
        self.path_speed = speed

    def _path_point(self, s: float) -> tuple[np.ndarray, np.ndarray]:
        assert self.path is not None and self.path_cum is not None
        cum = self.path_cum
        if len(cum) < 2 or cum[-1] <= 0:
            return self.path[-1].copy(), np.zeros(2)
        s = float(np.clip(s, 0.0, cum[-1]))
        i = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(cum) - 2))
        L = max(cum[i + 1] - cum[i], 1e-9)
        a, b = self.path[i], self.path[i + 1]
        return a + (b - a) * (s - cum[i]) / L, (b - a) / L

    def _target_z(self, xy: np.ndarray, direction: np.ndarray, agl: float, terrain: bool = True) -> float:
        if not terrain:  # HOME: a constant height above the pad
            return self.pad_z + agl
        zs = [self.ground_z(xy + direction * d) for d in np.linspace(0.0, self.c.terrain_lookahead_m, 7)]
        return float(max(zs)) + agl

    def _move_z_only(self, z_target: float, rate: float, dt: float) -> np.ndarray:
        dz = z_target - self.setpoint[2]
        step = float(np.clip(dz, -rate * dt, rate * dt))
        self.setpoint[2] += step
        return np.array([0.0, 0.0, step / dt if dt > 0 else 0.0])

    def _follow_path(self, t: float, pose: PoseEstimate, dt: float, agl: float | None = None,
                     terrain: bool = True) -> np.ndarray:
        """Carrot along the path at path_speed; it waits while the drone lags more than lead_max_m behind."""
        lag = float(np.linalg.norm(self.setpoint[:2] - pose.pos[:2]))
        v = self.path_speed if lag < self.c.lead_max_m else 0.0
        self.path_s += v * dt
        xy, tangent = self._path_point(self.path_s)
        self.setpoint[0], self.setpoint[1] = xy
        agl = self.c.cruise_agl if agl is None else agl
        z_t = self._target_z(xy, tangent, agl, terrain)
        rate = self.c.climb_speed if z_t > self.setpoint[2] else self.c.descent_speed
        vz = self._move_z_only(z_t, rate, dt)
        return np.array([tangent[0] * v, tangent[1] * v, vz[2]])

    def _orbit(self, t: float, pose: PoseEstimate, dt: float) -> np.ndarray:
        tr = self.observe_track
        assert tr is not None and self.observe_t0 is not None
        r = self.c.orbit_radius_m
        centre = tr.xy
        target = centre + r * np.array([math.cos(self.orbit_phi), math.sin(self.orbit_phi)])
        d = target - self.setpoint[:2]
        dist = float(np.linalg.norm(d))
        step = self.c.orbit_speed * dt
        if dist > 5.0:  # approach the circle
            self.setpoint[:2] += d / dist * min(step * 1.5, dist)
            v2 = d / dist * self.c.orbit_speed
        else:
            self.orbit_phi += self.c.orbit_speed / r * dt  # counter-clockwise
            new = centre + r * np.array([math.cos(self.orbit_phi), math.sin(self.orbit_phi)])
            v2 = (new - self.setpoint[:2]) / max(dt, 1e-6)
            self.setpoint[:2] = new
        z_t = float(self.ground_z(centre)) + self.c.cruise_agl
        vz = self._move_z_only(z_t, self.c.climb_speed if z_t > self.setpoint[2] else self.c.descent_speed, dt)
        re_hits = sum(1 for h in tr.hits if h[0] > self.observe_t0 + 0.5)
        if not tr.verified and re_hits >= self.c.verify_hits:
            tr.verified = True
            dd = tr.xy - self.pad
            self.log(t, f"FIND: {tr.label} #{tr.id} verified from the orbit ({re_hits} re-detections) at "
                        f"{np.linalg.norm(dd):.0f} m / {az_of(dd):.0f} deg from the base")
        elapsed = t - self.observe_t0
        if not tr.verified and (elapsed >= self.c.observe_s or (elapsed >= self.c.verify_timeout_s and re_hits == 0)):
            tr.rejected, tr.confirmed_t = True, None
            tr.snapshots = []
            self.log(t, f"REJECTED: #{tr.id} not seen again from the orbit ({re_hits} re-detections in {elapsed:.0f} s)")
            self._resume_search(t, pose, f"after rejecting #{tr.id}")
        elif elapsed >= self.c.observe_s:
            n_snap = len(tr.snapshots)
            if self.c.continue_after_find and self.path_cum is not None and self.path_s < self.path_cum[-1] \
                    and self.elapsed(t) + self.time_to_home(pose) + self.c.reserve_s + 60.0 < self.c.battery_s:
                self._resume_search(t, pose, f"after observing #{tr.id} ({n_snap} snapshots)")
            else:
                self._start_return(t, pose, f"{tr.label} #{tr.id} observed for {self.c.observe_s:.0f} s, "
                                            f"{n_snap} snapshots")
        return np.array([v2[0], v2[1], vz[2]])

    def _resume_search(self, t: float, pose: PoseEstimate, why: str) -> None:
        """Back to the search pattern where it was left (or home if it is done)."""
        if self.path is None or self.path_cum is None or self.path_s >= self.path_cum[-1] - 1e-6:
            self._start_return(t, pose, f"search pattern complete ({why})")
            return
        rest = self.path[int(np.searchsorted(self.path_cum, self.path_s, side="right")):]
        self._set_path(t, [self.setpoint[:2], *rest], self.c.search_speed, reset=True)
        self.phase = "SEARCH"
        self.observe_track = None
        self.log(t, f"SEARCH: resumed {why}")

    def _update_yaw(self, pose: PoseEstimate, dt: float) -> None:
        """Tail to the base: yaw = direction base -> drone (the nose looks away, the rear camera at the lights)."""
        d = pose.pos[:2] - self.pad
        if np.linalg.norm(d) > 25.0:
            want = math.atan2(d[1], d[0])
        elif self.phase == "TAKEOFF":
            u = az_to_unit(self.az)
            want = math.atan2(u[1], u[0])
        else:
            return
        err = math.atan2(math.sin(want - self.yaw), math.cos(want - self.yaw))
        lim = math.radians(45.0) * dt
        self.yaw = math.atan2(math.sin(self.yaw + np.clip(err, -lim, lim)), math.cos(self.yaw + np.clip(err, -lim, lim)))

    def rear_tilt_deg(self, pose: PoseEstimate) -> float | None:
        """Rear NIR gimbal tilt (deg below body -x) that puts the base centre on the optical axis: the depression
        angle to the base plus the body's own pitch (nose-down cruise tilts the tail camera up)."""
        if self.phase in ("PREFLIGHT", "LANDED", "REPORT"):
            return None
        d = pose.pos[:2] - self.pad
        dist = float(np.linalg.norm(d))
        if dist < 20.0:
            return None
        dep = math.degrees(math.atan2(pose.pos[2] - (self.pad_z + 3.0), dist))  # lights ~3 m up the fence posts
        back = -pose.R[:, 0]  # body -x in the world
        up = math.degrees(math.asin(float(np.clip(back[2], -1.0, 1.0))))
        return float(np.clip(dep + up, -10.0, 70.0))

    # ------------------------------------------------------------------ thermal (camera rate)

    def on_thermal(self, frame: ThermalFrame, pose: PoseEstimate) -> None:
        if self.phase not in ("OUTBOUND", "SEARCH", "OBSERVE", "RETURN"):
            return
        agl = float(pose.pos[2] - self.ground_z(pose.pos[:2]))
        res = self.detector.detect(frame.image_K, pose.R, agl, frame.intrinsics, pos_xy=pose.pos[:2])
        self.frames += 1
        self.det_ms += res.ms
        self.last_result = res
        self.last_frame_t = frame.t
        if res.gsd_m <= 0:
            return
        self._mark_coverage(pose, agl, frame.intrinsics)
        hit_tracks: list[tuple[Track, Detection]] = []
        for d in res.detections:
            xy = pose.pos[:2] + d.xy_rel
            tr = min(self.tracks, key=lambda q: float(np.linalg.norm(q.xy - xy)), default=None)
            if tr is None or np.linalg.norm(tr.xy - xy) > self.c.assoc_radius_m:
                tr = Track(len(self.tracks) + 1, d.label, xy.copy(), sigma2_at_first=pose.sigma2_horizontal())
                self.tracks.append(tr)
            tr.add(frame.t, xy, d)
            hit_tracks.append((tr, d))
        for tr, d in hit_tracks:
            if tr.confirmed_t is None and not tr.rejected:
                recent = [h for h in tr.hits if frame.t - h[0] <= self.c.confirm_window_s]
                if len(recent) >= self.c.confirm_hits and np.mean([h[2] for h in recent]) >= self.c.confirm_min_conf:
                    if tr.speed() > self.c.moving_speed:
                        if not tr.moving:
                            tr.moving = True
                            self.log(frame.t, f"SIGHTING: moving {tr.label} #{tr.id} ({tr.speed():.1f} m/s), not orbited")
                    else:
                        tr.confirmed_t = frame.t
                        dd = tr.xy - self.pad
                        self.log(frame.t, f"CANDIDATE: {tr.label} #{tr.id} at {np.linalg.norm(dd):.0f} m / {az_of(dd):.0f} deg "
                                          f"from the base, {tr.size_m[0]:.1f} x {tr.size_m[1]:.1f} m, +{tr.temp_K:.0f} K")
                        if self.phase in ("OUTBOUND", "SEARCH"):
                            self._start_observe(frame.t, pose, tr)
            if tr.confirmed_t is not None and not tr.rejected and len(tr.snapshots) < self.c.max_snapshots and (
                    frame.t - max((s["t"] for s in tr.snapshots), default=-1e9) >= self.c.snapshot_every_s):
                tr.snapshots.append(self._snapshot(frame, d, pose))

    def _snapshot(self, frame: ThermalFrame, d: Detection, pose: PoseEstimate) -> dict[str, Any]:
        img8 = to_8bit(frame.image_K)
        h, w = img8.shape
        cu, cv = int(d.uv[0]), int(d.uv[1])
        half = 80
        crop = img8[max(cv - half, 0):min(cv + half, h), max(cu - half, 0):min(cu + half, w)].copy()
        full = cv2.resize(img8, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        full = cv2.cvtColor(full, cv2.COLOR_GRAY2BGR)
        x0, y0, bw, bh = (v // 2 for v in d.bbox)
        cv2.rectangle(full, (x0 - 3, y0 - 3), (x0 + bw + 3, y0 + bh + 3), (0, 0, 255), 1)
        return {"t": frame.t, "clock": self.clock(frame.t), "crop": crop, "frame": full, "conf": d.conf,
                "label": d.label, "drone_est": [round(float(v), 1) for v in pose.pos],
                "agl_m": round(float(pose.pos[2] - self.ground_z(pose.pos[:2])), 1)}

    def _mark_coverage(self, pose: PoseEstimate, agl: float, K: dict[str, float]) -> None:
        R_wc = pose.R @ np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        corners = []
        for u, v in ((0, 0), (K["w"], 0), (K["w"], K["h"]), (0, K["h"])):
            pc = np.array([(u - K["cx"]) / K["fx"], -(v - K["cy"]) / K["fy"], -1.0])
            d = R_wc @ pc
            if d[2] > -0.1:
                return
            corners.append(pose.pos[:2] + d[:2] * (agl / -d[2]))
        px = ((np.array(corners) - self.cov_origin) / self.cov_cell)
        px[:, 1] = self.coverage.shape[0] - px[:, 1]
        cv2.fillConvexPoly(self.coverage, np.round(px).astype(np.int32), 1)

    def _finds(self) -> list[Track]:
        """Verified finds: confirmed on the pass AND seen again from the orbit."""
        return [tr for tr in self.tracks if tr.verified]

    # ------------------------------------------------------------------ report

    def searched_area_km2(self) -> float:
        return float(self.coverage.sum()) * self.cov_cell ** 2 / 1e6

    def build_report(self, t: float) -> dict[str, Any]:
        finds = []
        for tr in self._finds():
            d = tr.xy - self.pad
            finds.append({
                "id": tr.id, "label": tr.label, "est_enu": [round(float(v), 1) for v in tr.xy],
                "bearing_deg": round(az_of(d), 1), "distance_m": round(float(np.linalg.norm(d)), 1),
                "time": self.clock(tr.confirmed_t or tr.hits[0][0]), "t_s": round(float(tr.confirmed_t or 0.0), 1),
                "hits": tr.n, "conf": round(tr.conf, 2), "size_m": [round(float(v), 1) for v in tr.size_m],
                "anomaly_K": round(float(tr.temp_K), 1),
                "drone_2sigma_m": round(float(tr.sigma2_at_first), 1),
                "snapshots": [f"find{tr.id}_{k}.png" for k in range(len(tr.snapshots))],
            })
        sightings = [{"id": tr.id, "label": tr.label, "est_enu": [round(float(v), 1) for v in tr.xy],
                      "speed_m_s": round(tr.speed(), 1), "hits": tr.n} for tr in self.tracks if tr.moving]
        groups: dict[str, int] = {}
        for f in finds:
            groups[f["label"]] = groups.get(f["label"], 0) + 1
        if finds:
            c = np.mean([f["est_enu"] for f in finds], axis=0) - self.pad
            what = ", ".join(f"{n} {lab}{'s' if n > 1 else ''}" for lab, n in sorted(groups.items()))
            summary = f"{what}, {np.linalg.norm(c) / 1000:.1f} km at {az_of(c):03.0f} deg, {finds[0]['time']}"
        else:
            summary = (f"nothing found in the searched area ({self.searched_area_km2():.2f} km^2, out to "
                       f"{self.max_r / 1000:.1f} km along {self.az:03.0f} deg)")
        return {
            "format": "sfhack-recon-report/1",
            "mission": self.mission_id,
            "summary": summary,
            "outcome": "FOUND" if finds else "NOTHING_FOUND",
            "bearing_in": bearing_dict(self.bearing),
            "base_pad_enu": [round(float(v), 2) for v in self.pad],
            "takeoff_clock": self.clock(self.t_takeoff or 0.0) if self.t_takeoff is not None else None,
            "landing_clock": self.clock(t),
            "flight_time_s": round(self.elapsed(t), 1),
            "return_reason": self.return_reason,
            "finds": finds,
            "moving_sightings": sightings,
            "searched": {
                "area_km2": round(self.searched_area_km2(), 3),
                "max_distance_m": round(self.max_r, 0),
                "pattern": None if self.plan is None else {
                    "r_start_m": round(self.plan.r_start), "r_max_m": round(self.plan.r_max),
                    "step_m": round(self.plan.step_m, 1), "swath_m": round(self.plan.swath_m, 1),
                    "cone_halfwidth": [(round(r), round(w)) for r, w in self.plan.halfwidth],
                },
                "thermal_frames": self.frames,
                "detector_ms_mean": round(self.det_ms / max(self.frames, 1), 1),
            },
            "decisions": [(round(tt, 1), s) for tt, s in self.decision_log],
        }
