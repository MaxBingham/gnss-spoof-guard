"""Recon flight (HARNESS, like runner.py): builds the FlightSystem on a mission, swaps in the onboard ReconMission,
switches GNSS off after the first 50 m of flight, runs to the report, and scores it against the truth.

    rf = ReconFlight(mission="ukraine-1", seed=0, bearing="bearing.json", nav="lights")
    score = rf.run()          # report -> $SFHACK_DATA/reports/<mission>/<run>/ (+ score.json, map_truth.png)

Truth is used only to (a) simulate the array's bearing when none is given, (b) switch GNSS off at 50 m of TRUE
flight distance (the operator's key N), (c) score. The drone gets the bearing and its own sensors only.
"""

from __future__ import annotations

import inspect
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config, data_dir, default_config
from .degradation.conditions import Conditions
from .mission.recon import ReconMission, az_of, load_base, load_bearing
from .mission.report import draw_map, write_report
from .perception.thermal_detect import ThermalDetector
from .runner import FlightSystem


def scene_path(mission_dir: Path, scene_seed: int | None) -> Path | None:
    """recon-world's per-seed scene world/recon/seed_<k>/target_scene.json (None: world/target_scene.json)."""
    if scene_seed is None:
        return None
    p = Path(mission_dir) / "world" / "recon" / f"seed_{scene_seed}" / "target_scene.json"
    return p if p.exists() else None


def true_targets(mission_dir: Path, scene: Path | None = None) -> list[dict[str, Any]]:
    """target_scene.json objects with their kind (target | distractor); harness only."""
    p = scene or Path(mission_dir) / "world" / "target_scene.json"
    if not p.exists():
        return []
    objs = json.loads(p.read_text(encoding="utf-8")).get("objects", [])
    out = []
    for k, o in enumerate(objs):
        kind = o.get("role") or o.get("kind") or ("distractor" if o.get("distractor") else "target")
        out.append({"name": o.get("name", f"target{k}"), "kind": "target" if kind in ("target", "vehicle", "person",
                    "people", "truck", "car") else kind, "xy": np.asarray(o["pos"][:2], float),
                    "class": o.get("class", o.get("label", o.get("name", "")))})
    return out


def simulate_bearing(mission_dir: Path, base_xy: np.ndarray, seed: int, sigma_deg: float = 7.0,
                     bias_std_deg: float = 4.0, scene: Path | None = None) -> dict[str, Any]:
    """Bearing to the target cluster as the base array would report it: recon-world's sim_array when present,
    else true azimuth + a per-seed bias + noise, sigma `sigma_deg` (engine noise, 1.5-3 km)."""
    pre = Path(mission_dir) / "world" / "recon" / f"seed_{seed}" / "bearing.json"  # recon-world's per-seed array
    if scene is not None and pre.exists() and pre.parent == scene.parent:
        return json.loads(pre.read_text(encoding="utf-8"))
    try:
        from .acoustic import sim_array  # recon-world: (BearingEstimate | None, info)

        out = sim_array.simulate_bearing(mission_dir, seed)
        b = out[0] if isinstance(out, tuple) else out
        if b is not None:
            return b
    except (ImportError, AttributeError, TypeError, FileNotFoundError):
        pass
    tg = [t["xy"] for t in true_targets(mission_dir, scene) if t["kind"] == "target"]
    if not tg:
        raise ValueError(f"no targets in {mission_dir}/world/target_scene.json: give a bearing")
    c = np.mean(tg, axis=0) - base_xy
    rng = np.random.default_rng(seed + 9100)
    az = az_of(c) + rng.normal(0.0, bias_std_deg) + rng.normal(0.0, 2.0)
    return {"t": 0.0, "az_deg": round(float(az) % 360.0, 2), "el_deg": round(math.degrees(math.atan2(0.0, float(np.linalg.norm(c)))), 2),
            "sigma_az_deg": sigma_deg, "sigma_el_deg": 5.0, "label": "engine", "confidence": 0.8,
            "array_enu": [float(base_xy[0]), float(base_xy[1]), 3.0], "source": "harness fallback (truth + noise)"}


class ReconFlight:
    def __init__(self, mission: str = "ukraine-1", seed: int = 0, bearing: Any = None, nav: str | None = None,
                 cfg: Config | None = None, gnss_off_m: float | None = 50.0, thermal_hz: float | None = None,
                 thermal_renderer: str = "auto", run_id: str | None = None, out_root: Path | None = None,
                 render_front: bool = False, scene_seed: int | str | None = "seed", rear_camera: bool = False) -> None:
        """`scene_seed`: recon-world's per-seed target scene (world/recon/seed_<k>/); "seed" = the flight seed,
        None = world/target_scene.json."""
        self.cfg = cfg or default_config()
        self.cfg.thermal.method = "off"  # no map-matching on the thermal prior: the thermal camera FINDS things here
        if thermal_hz is not None:  # optical flow (nav flow / lights) needs >= 5 Hz
            self.cfg.thermal.camera_hz = max(thermal_hz, 5.0) if nav in ("flow", "lights") else thermal_hz
        params = inspect.signature(FlightSystem).parameters
        kw: dict[str, Any] = {}
        if nav is not None:
            if "nav" in params:
                kw["nav"] = nav  # recon-nav: imu | flow | lights
            else:
                print(f"[recon] FlightSystem has no nav= on this branch; nav {nav!r} ignored (IMU + compass + baro)")
        if "night_route" in params:
            kw["night_route"] = False
        self.nav = nav
        self.seed = seed
        cond = Conditions().update(gnss__mode="ok")
        from .runner import resolve_mission

        self.scene = scene_path(resolve_mission(mission), seed if scene_seed == "seed" else scene_seed)
        if self.scene is not None and "target_scene" in params:
            kw["target_scene"] = self.scene
        self.fs = fs = FlightSystem(cfg=self.cfg, mission=mission, night=True, localizer="off", seed=seed,
                                    conditions=cond, render_front=render_front, thermal_renderer=thermal_renderer, **kw)
        if fs.thermal_sensor is None:
            from .sim.thermal_sensor import ThermalSensor

            fs.thermal_sensor = ThermalSensor(fs.sim, fs.bus, fs.mission_dir, fs.world, hz=self.cfg.thermal.camera_hz,
                                              seed=seed, renderer=thermal_renderer, target_scene=self.scene)
        # rear NIR camera (recon-world): reuse the one the nav stack made (light localizer), else make one for the
        # viewer when the world has base lights
        self.rear = next((v for v in vars(fs).values() if type(v).__name__ == "RearCameraSensor"), None)
        self._own_rear = False
        if self.rear is None and rear_camera and fs.world_dir is not None and (fs.world_dir / "base_lights.json").exists():
            from .sim.rear_camera import RearCameraSensor

            self.rear = RearCameraSensor(fs.sim, fs.bus, fs.world_dir, seed=seed)
            self._own_rear = True
        om = fs.onboard_map
        assert om is not None and fs.mission_dir is not None
        self.mission_id = fs.mission_dir.name
        self.base = load_base(fs.onboard_dir, om.a_enu)
        pad = np.asarray(self.base["pad_enu"], float)
        if bearing is None:
            bearing = simulate_bearing(fs.mission_dir, pad, seed, scene=self.scene)
        self.bearing = load_bearing(bearing)
        self.recon = ReconMission(self.cfg.recon, self.cfg.mission, self.bearing, pad, fs.mission.yaw,
                                  ground_z=om.ground_z, map_inside=lambda xy, m: om.contains(xy, m), onboard_map=om,
                                  detector=ThermalDetector(onboard_map=om), mission_id=self.mission_id,
                                  thermal_hfov_deg=self.cfg.camera.thermal_hfov_deg if hasattr(self.cfg.camera, "thermal_hfov_deg") else 50.0)
        fs.mission = self.recon
        fs.autopilot.attach_mission(self.recon)
        self.spawn = fs.truth.state().pos.copy()
        self.gnss_off_m = gnss_off_m
        self.gnss_off_t: float | None = None
        self.run_id = run_id or f"{datetime.now():%Y%m%d-%H%M%S}_s{seed}"
        self.out_dir = (out_root or data_dir() / "reports" / self.mission_id) / self.run_id
        self.truth_track: list[tuple[float, float, float, float]] = []
        self._next_tt = 0.0
        self.t_wall0 = time.time()
        self.score: dict[str, Any] | None = None

    @property
    def t(self) -> float:
        return self.fs.t

    def gnss_off(self) -> None:
        """Operator's key N (also the harness at 50 m)."""
        if self.gnss_off_t is None:
            self.fs.update_conditions(gnss__mode="off")
            self.gnss_off_t = self.t
            self.fs.autopilot.log_decision(self.t, "GNSS OFF (jammed): from here IMU + compass + baro + own sensors only")

    def step(self) -> None:
        self.fs.step()
        if self._own_rear:
            self.rear.step()
        tr = self.fs.truth.state()
        if self.gnss_off_t is None and self.gnss_off_m is not None and \
                float(np.linalg.norm(tr.pos[:2] - self.spawn[:2])) >= self.gnss_off_m:
            self.gnss_off()
        if self.t >= self._next_tt:
            self._next_tt = self.t + 1.0
            self.truth_track.append((self.t, float(tr.pos[0]), float(tr.pos[1]), float(tr.pos[2])))

    @property
    def done(self) -> bool:
        return self.recon.phase == "REPORT"

    def run(self, max_t: float | None = None, progress_s: float = 0.0) -> dict[str, Any]:
        max_t = max_t if max_t is not None else self.cfg.recon.battery_s + self.cfg.mission.preflight_s + 180.0
        next_p = 0.0
        while self.t < max_t and not self.done:
            self.step()
            if progress_s and self.t >= next_p:
                next_p = self.t + progress_s
                est = self.fs.autopilot.pose(report=False)
                err = float(np.linalg.norm(est.pos[:2] - self.fs.truth.state().pos[:2]))
                print(f"[recon s{self.seed}] t={self.t:6.0f} {self.recon.phase:8s} est err {err:6.1f} m  2s "
                      f"{est.sigma2_horizontal():6.1f} m  tracks {len(self.recon.tracks)}  wall {time.time() - self.t_wall0:5.0f} s",
                      flush=True)
        for _ in range(int(1.0 / self.cfg.sim.timestep)):  # settle on the pad
            if not self.done:
                break
            self.fs.step()
        return self.finish()

    def finish(self) -> dict[str, Any]:
        """Write the onboard report, then score it (harness): score.json + map_truth.png beside it."""
        m = self.recon
        if m.report is None:  # timeout: the report as it stands
            m.report = m.build_report(self.t)
            m.report["outcome"] += "_NOT_LANDED"
        write_report(m, self.out_dir)
        self.score = s = self._score()
        (self.out_dir / "score.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
        tt = np.array([r[1:3] for r in self.truth_track]) if self.truth_track else None
        import cv2

        cv2.imwrite(str(self.out_dir / "map_truth.png"), draw_map(
            m, truth_track=tt, truth_targets=[(t["xy"], t["kind"]) for t in true_targets(self.fs.mission_dir, self.scene)],
            title=f"SCORED (truth): {self.mission_id} seed {self.seed}"))
        return s

    def _score(self) -> dict[str, Any]:
        fs, m = self.fs, self.recon
        truth = fs.truth.state()
        pad_true = np.asarray(fs.world.a_enu, float)
        tg = true_targets(fs.mission_dir, self.scene)
        targets = [t for t in tg if t["kind"] == "target"]
        finds = (m.report or {}).get("finds", [])
        per_find = []
        for f in finds:
            xy = np.asarray(f["est_enu"], float)
            near = min(tg, key=lambda t: float(np.linalg.norm(t["xy"] - xy)), default=None)
            d = float(np.linalg.norm(near["xy"] - xy)) if near is not None else math.inf
            per_find.append({"id": f["id"], "label": f["label"], "err_m": round(d, 1), "nearest": near["name"] if near else None,
                             "nearest_kind": near["kind"] if near else None})
        true_finds = [p for p in per_find if p["nearest_kind"] == "target" and p["err_m"] <= 50.0]
        false_finds = [p for p in per_find if p not in true_finds]
        if targets:
            c = np.mean([t["xy"] for t in targets], axis=0) - pad_true
            true_az, true_dist = az_of(c), float(np.linalg.norm(c))
        else:
            true_az = true_dist = None
        landed = m.phase == "REPORT" and truth.pos[2] - fs.true_ground_z(truth.pos[0], truth.pos[1]) < 0.5
        est = np.array([r[1:3] for r in m.track_log]) if m.track_log else np.zeros((0, 2))
        if self.truth_track and m.track_log:  # truth at the estimate's timestamps
            T = np.array(self.truth_track)
            te = np.array([r[0] for r in m.track_log])
            tt = np.stack([np.interp(te, T[:, 0], T[:, 1]), np.interp(te, T[:, 0], T[:, 2])], -1)
        else:
            tt = np.zeros((0, 2))
        n = min(len(tt), len(est))
        # coverage: was a true target inside the searched (estimated) footprints? (truth position -> TRUE coverage
        # would need the true footprints; here: the true track's footprint ~ the estimated one shifted by the error)
        a = fs.autopilot
        s = {
            "mission": self.mission_id, "seed": self.seed, "nav": self.nav, "run": self.run_id,
            "bearing_in": {k: getattr(self.bearing, k, None) for k in ("az_deg", "sigma_az_deg", "label")},
            "true_cluster_az_deg": None if true_az is None else round(true_az, 1),
            "true_cluster_dist_m": None if true_dist is None else round(true_dist, 0),
            "bearing_err_deg": None if true_az is None else round((float(self.bearing.az_deg) - true_az + 180) % 360 - 180, 1),
            "outcome": (m.report or {}).get("outcome"), "summary": (m.report or {}).get("summary"),
            "found": bool(true_finds), "n_true_finds": len(true_finds), "n_false_finds": len(false_finds),
            "find_err_m": [p["err_m"] for p in true_finds], "finds_scored": per_find,
            "first_find_err_m": true_finds[0]["err_m"] if true_finds else None,
            "landed": bool(landed), "landing_err_m": round(float(np.linalg.norm(truth.pos[:2] - pad_true)), 2),
            "mission_time_s": round(m.elapsed(self.t), 1), "t_sim": round(self.t, 1),
            "within_battery": m.elapsed(m.t_landed or self.t) <= self.cfg.recon.battery_s,
            "return_reason": m.return_reason, "searched_km2": round(m.searched_area_km2(), 3),
            "max_dist_m": round(m.max_r), "gnss_off_t": self.gnss_off_t,
            "est_err_max_m": round(float(np.max(np.linalg.norm(tt[:n] - est[:n], axis=1))), 1) if n else None,
            "est_err_at_far_m": None, "detector_ms": round(m.det_ms / max(m.frames, 1), 1), "frames": m.frames,
            "sources_used": sorted(a.sources_used), "wall_s": round(time.time() - self.t_wall0, 1),
            "out_dir": str(self.out_dir),
        }
        if n:
            k = int(np.argmax(np.linalg.norm(est[:n] - m.pad, axis=1)))
            s["est_err_at_far_m"] = round(float(np.linalg.norm(tt[k] - est[k])), 1)
        return s

    def close(self) -> None:
        self.fs.close()
