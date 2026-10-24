"""Base-light localizer (recon-nav): rear NIR camera -> identified base lights -> position AND heading.

1. Blobs: bright compact spots on the dark NIR frame (threshold over the background median, intensity-weighted
   centroid around the brightest pixel).
2. Tracks: each blob becomes a world-frame direction (AHRS attitude, camera mount) and is associated frame to
   frame by azimuth / elevation (the image itself jumps by tens of pixels per degree of roll; the world direction
   to a light 2 km away hardly moves). A track records on / off per frame.
3. Ids: every light blinks its own cyclic code (onboard `base_lights.json`). The drone's clock is not the base's,
   so each track's on / off history is matched against every code at every phase; a track gets an id when one code
   explains >= `id_min_score` of the samples and clearly beats every other code. Steady lights (fires, a village)
   match no code.
4. Resection: Gauss-Newton on the pixel positions of the identified lights that are on in this frame. Unknowns:
   x, y, z and a small world-frame attitude correction (roll, pitch, yaw) of the AHRS; priors: z from the baro
   EKF, roll / pitch within `tilt_prior_deg`, yaw within `yaw_prior_deg` (loose: >= 2 lights fix the heading
   independently of the compass). >= 3 lights = resection; 2 lights = the same solver, its covariance is the
   long thin ellipse of a bearing-line fix (along the line it rests on the elevation angles + baro). The
   covariance is the solver's posterior; with >= 3 lights the residual RMS is checked (wrong ids -> no fix).

Onboard: sees only CameraFrame images, the AHRS attitude, the EKF prior and the surveyed light list.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import CameraConfig, LightsConfig


@dataclass
class BaseLight:
    id: str
    pos: np.ndarray  # (3,) ENU, z in the mission datum (ground under A = 0)
    code: str  # cyclic on/off pattern, "1" = on
    bit_hz: float
    ring: str = "fence"
    facing_deg: float | None = None  # ENU yaw of the beam axis (outward), None = omnidirectional
    beam_half_deg: float = 180.0


def load_base_lights(path: Path | str) -> list[BaseLight]:
    """Onboard survey `onboard/base_lights.json` (recon-world, sfhack-base-lights/1): {"bit_s", "lights": [{"id",
    "enu" [x, y, z], "code" "00010111", "ring", "aim_az_deg" (compass), "beam_h_deg" (half width)}]}. Also accepts
    per-light "bit_hz", "facing_deg" (ENU yaw), "beam_half_deg", "pos" / "xyz" for the position."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = d["lights"] if isinstance(d, dict) else d
    default_hz = 10.0
    if isinstance(d, dict):
        default_hz = 1.0 / float(d["bit_s"]) if "bit_s" in d else float(d.get("bit_hz", 10.0))
    out = []
    for r in rows:
        p = r.get("enu", r.get("pos", r.get("xyz")))
        p = [float(v) for v in p]
        if len(p) == 2:
            p.append(float(r.get("z", r.get("height", 0.0))))
        code = r["code"]
        if not isinstance(code, str):
            code = "".join("1" if int(b) else "0" for b in code)
        facing = r.get("facing_deg")
        if facing is None and "aim_az_deg" in r:
            facing = 90.0 - float(r["aim_az_deg"])
        out.append(BaseLight(str(r["id"]), np.array(p, float), code, float(r.get("bit_hz", default_hz)),
                             str(r.get("ring", "fence")), facing, float(r.get("beam_half_deg", r.get("beam_h_deg", 180.0)))))
    return out


def rear_mount(pitch_down_deg: float) -> np.ndarray:
    """Camera (OpenCV: x right, y down, z optical axis) -> body (FLU) rotation of the rear camera."""
    p = math.radians(pitch_down_deg)
    z = np.array([-math.cos(p), 0.0, -math.sin(p)])  # looks backwards, tilted down
    x = np.array([0.0, 1.0, 0.0])  # image right = body left when looking backwards
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def rear_intrinsics(cam: CameraConfig) -> dict[str, float]:
    f = cam.rear_width / 2.0 / math.tan(math.radians(cam.rear_hfov_deg) / 2.0)
    return {"fx": f, "fy": f, "cx": cam.rear_width / 2.0, "cy": cam.rear_height / 2.0, "w": float(cam.rear_width),
            "h": float(cam.rear_height), "hfov_deg": cam.rear_hfov_deg}


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _expm(phi: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(phi))
    K = _skew(phi)
    if th < 1e-9:
        return np.eye(3) + K
    return np.eye(3) + math.sin(th) / th * K + (1 - math.cos(th)) / th**2 * (K @ K)


@dataclass
class Blob:
    u: float
    v: float
    peak: float
    area: int


@dataclass
class LightTrack:
    tid: int
    az: float  # rad, world ENU azimuth of the last observation
    el: float
    t_last_on: float
    hist: list[tuple[float, bool]] = field(default_factory=list)
    light_id: str | None = None
    id_score: float = 0.0
    blob: Blob | None = None  # this frame's blob (None = off / not seen)


@dataclass
class LightFix:
    t: float
    xy: np.ndarray  # (2,)
    z: float
    yaw_corr: float  # rad, true heading - AHRS heading (positive = CCW)
    cov: np.ndarray  # (2, 2) m^2
    yaw_std: float  # rad
    n_lights: int
    ids: list[str]
    rms_norm: float
    kind: str  # "resection" (>= 3 lights) | "line" (2)
    range_m: float  # to the nearest used light
    tilt_corr: np.ndarray = field(default_factory=lambda: np.zeros(2))  # rad, world roll / pitch correction


class LightLocalizer:
    def __init__(self, lights: list[BaseLight], cfg: LightsConfig | None = None, cam: CameraConfig | None = None) -> None:
        """`cam`: the default mount / intrinsics (CameraConfig.rear_*); a NirFrame's own intrinsics and R_body_cam win."""
        self.lights = {lt.id: lt for lt in lights}
        self.cfg = cfg or LightsConfig()
        cam = cam or CameraConfig()
        self.R_bc = rear_mount(cam.rear_pitch_down_deg)
        k = rear_intrinsics(cam)
        self.f, self.cx, self.cy = k["fx"], k["cx"], k["cy"]
        self.tracks: list[LightTrack] = []
        self._next_tid = 0
        self.frames = 0
        self.n_fixes = 0
        self.last_fix: LightFix | None = None
        self.blobs: list[Blob] = []
        self.last_blobs: list[tuple[float, float, str | None]] = []  # display: (u, v, light id or None) this frame
        self.last_ids: dict[str, tuple[float, float]] = {}  # id -> (u, v) this frame (display)
        self.n_id_conflicts = 0
        fence = [lt.pos for lt in lights if lt.ring == "fence"] or [lt.pos for lt in lights]
        self.aim_point = np.mean(fence, axis=0)  # where the rear gimbal points (base centre, lamp height)
        # code bits as arrays (cyclic)
        self._codes = {k: np.array([c == "1" for c in lt.code], bool) for k, lt in self.lights.items()}

    # ------------------------------------------------------------------ detection

    def detect(self, img: np.ndarray) -> list[Blob]:
        cfg = self.cfg
        g = img if img.ndim == 2 else img[..., 0]
        g = g.astype(np.float32)
        bg = float(np.median(g[::8, ::8]))
        mask = (g > bg + cfg.min_peak).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        blobs: list[Blob] = []
        h, w = g.shape
        for i in range(1, n):
            x0, y0, bw, bh, area = stats[i]
            if area > cfg.max_blob_area:
                continue
            patch = g[y0:y0 + bh, x0:x0 + bw]
            py, px = np.unravel_index(int(np.argmax(patch)), patch.shape)
            cx, cy = x0 + px, y0 + py
            r = 3 if area < 30 else int(math.ceil(math.sqrt(area)))
            ya, yb, xa, xb = max(cy - r, 0), min(cy + r + 1, h), max(cx - r, 0), min(cx + r + 1, w)
            win = np.clip(g[ya:yb, xa:xb] - bg, 0.0, None)
            s = float(win.sum())
            if s <= 0:
                continue
            yy, xx = np.mgrid[ya:yb, xa:xb]
            blobs.append(Blob(float((win * xx).sum() / s) + 0.5, float((win * yy).sum() / s) + 0.5,
                              float(patch.max() - bg), int(area)))
        blobs.sort(key=lambda b: -b.peak)
        return blobs[:cfg.max_blobs]

    def _world_dir(self, u: float, v: float, R_wb: np.ndarray) -> np.ndarray:
        d = np.array([(u - self.cx) / self.f, (v - self.cy) / self.f, 1.0])
        d = R_wb @ self.R_bc @ d
        return d / np.linalg.norm(d)

    # ------------------------------------------------------------------ tracking + ids

    def _associate(self, t: float, blobs: list[Blob], R_wb: np.ndarray) -> None:
        cfg = self.cfg
        gate = math.radians(cfg.track_gate_deg)
        obs = []
        for b in blobs:
            d = self._world_dir(b.u, b.v, R_wb)
            obs.append((math.atan2(d[1], d[0]), math.asin(max(-1.0, min(1.0, d[2])))))
        pairs = []
        for i, (az, el) in enumerate(obs):
            for j, tr in enumerate(self.tracks):
                daz = math.atan2(math.sin(az - tr.az), math.cos(az - tr.az)) * math.cos(el)
                dist = math.hypot(daz, el - tr.el)
                if dist < gate:
                    pairs.append((dist, i, j))
        pairs.sort()
        used_b, used_t = set(), set()
        for tr in self.tracks:
            tr.blob = None
        for _, i, j in pairs:
            if i in used_b or j in used_t:
                continue
            used_b.add(i)
            used_t.add(j)
            tr = self.tracks[j]
            tr.az, tr.el = obs[i]
            tr.t_last_on = t
            tr.blob = blobs[i]
        for i, b in enumerate(blobs):
            if i not in used_b:
                tr = LightTrack(self._next_tid, obs[i][0], obs[i][1], t)
                tr.blob = b
                self._next_tid += 1
                self.tracks.append(tr)
        keep = []
        for tr in self.tracks:
            if t - tr.t_last_on > cfg.track_max_off_s:
                continue
            tr.hist.append((t, tr.blob is not None))
            del tr.hist[:-120]
            keep.append(tr)
        self.tracks = keep

    def _identify(self, tr: LightTrack) -> None:
        cfg = self.cfg
        if not tr.hist:
            return
        best: dict[str, float] = {}
        for lid, bits in self._codes.items():
            lt = self.lights[lid]
            period = len(bits) / lt.bit_hz
            ts = np.array([h[0] for h in tr.hist if h[0] >= tr.hist[-1][0] - cfg.id_window_periods * period])
            if len(ts) < 4 or ts[-1] - ts[0] < cfg.id_min_periods * period:
                continue
            on = np.array([h[1] for h in tr.hist[-len(ts):]], bool)
            phases = np.arange(cfg.id_phase_steps) / cfg.id_phase_steps * len(bits)
            idx = np.floor(ts[None, :] * lt.bit_hz + phases[:, None]).astype(int) % len(bits)
            score = (bits[idx] == on[None, :]).mean(axis=1)
            best[lid] = float(score.max())
        if not best:
            return
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        lid, s = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        if s >= cfg.id_min_score and s - second >= cfg.id_min_margin:
            tr.light_id, tr.id_score = lid, s
        elif tr.light_id is not None and best.get(tr.light_id, 0.0) < cfg.id_min_score - 0.1:
            tr.light_id = None  # it stopped matching (track jumped to another light)

    # ------------------------------------------------------------------ resection

    def _predict_px(self, theta: np.ndarray, R_wb0: np.ndarray, P: np.ndarray) -> np.ndarray:
        R_wc = _expm(theta[3:6]) @ R_wb0 @ self.R_bc
        c = (P - theta[:3]) @ R_wc  # rows: R_wc^T (L - p)
        z = np.maximum(c[:, 2], 1e-6)
        return np.stack([self.cx + self.f * c[:, 0] / z, self.cy + self.f * c[:, 1] / z], -1)

    def solve(self, uv: np.ndarray, P: np.ndarray, R_wb0: np.ndarray, p0: np.ndarray, z_std: float
              ) -> tuple[np.ndarray, np.ndarray, float] | None:
        """Gauss-Newton: returns (theta [x, y, z, rx, ry, rz], posterior covariance 6x6, normalized RMS)."""
        cfg = self.cfg
        rng = np.linalg.norm(P - p0, axis=1)
        sig = np.sqrt(cfg.sigma_px**2 + (self.f * cfg.survey_std_m / np.maximum(rng, 1.0)) ** 2)
        w = np.repeat(1.0 / sig, 2)
        prior_mu = np.array([p0[0], p0[1], p0[2], 0.0, 0.0, 0.0])
        prior_sd = np.array([5000.0, 5000.0, max(z_std, cfg.z_prior_min_std), math.radians(cfg.tilt_prior_deg),
                             math.radians(cfg.tilt_prior_deg), math.radians(cfg.yaw_prior_deg)])
        theta = prior_mu.copy()
        steps = np.array([0.05, 0.05, 0.05, 1e-5, 1e-5, 1e-5])
        for _ in range(12):
            r0 = ((self._predict_px(theta, R_wb0, P) - uv).ravel()) * w
            J = np.zeros((len(r0), 6))
            for k in range(6):
                th = theta.copy()
                th[k] += steps[k]
                J[:, k] = (((self._predict_px(th, R_wb0, P) - uv).ravel()) * w - r0) / steps[k]
            Jp = np.diag(1.0 / prior_sd)
            rp = (theta - prior_mu) / prior_sd
            A = J.T @ J + Jp.T @ Jp
            g = J.T @ r0 + Jp.T @ rp
            try:
                dx = -np.linalg.solve(A, g)
            except np.linalg.LinAlgError:
                return None
            theta = theta + dx
            if np.all(np.abs(dx[:3]) < 0.01) and np.all(np.abs(dx[3:]) < 1e-6):
                break
        r = ((self._predict_px(theta, R_wb0, P) - uv).ravel()) * w
        cov = np.linalg.inv(A)
        return theta, cov, float(np.sqrt(np.mean(r**2)))

    # ------------------------------------------------------------------ frame

    def update(self, t: float, image: np.ndarray, R_wb: np.ndarray, prior_pos: np.ndarray, z_std: float = 1.0,
               intrinsics: dict[str, float] | None = None, R_body_cam: np.ndarray | None = None) -> LightFix | None:
        cfg = self.cfg
        self.frames += 1
        if intrinsics is not None:
            self.f, self.cx, self.cy = float(intrinsics["fx"]), float(intrinsics["cx"]), float(intrinsics["cy"])
        if R_body_cam is not None:
            self.R_bc = np.asarray(R_body_cam, float)
        blobs = self.detect(image)
        self.blobs = blobs
        self._associate(t, blobs, R_wb)
        for tr in self.tracks:
            self._identify(tr)
        # one track per id (two claiming the same light: keep the better match)
        by_id: dict[str, LightTrack] = {}
        for tr in self.tracks:
            if tr.light_id is None or tr.blob is None:
                continue
            o = by_id.get(tr.light_id)
            if o is not None:
                self.n_id_conflicts += 1
                if o.id_score >= tr.id_score:
                    continue
            by_id[tr.light_id] = tr
        self.last_ids = {k: (tr.blob.u, tr.blob.v) for k, tr in by_id.items() if tr.blob is not None}
        self.last_blobs = [(tr.blob.u, tr.blob.v, tr.light_id if by_id.get(tr.light_id) is tr else None)
                           for tr in self.tracks if tr.blob is not None]
        if len(by_id) < cfg.min_lights:
            return None
        ids = sorted(by_id)
        uv = np.array([[by_id[k].blob.u, by_id[k].blob.v] for k in ids])
        P = np.array([self.lights[k].pos for k in ids])
        sol = self.solve(uv, P, R_wb, np.asarray(prior_pos, float), z_std)
        if sol is None:
            return None
        theta, cov, rms = sol
        if len(ids) >= 3 and rms > cfg.max_rms_norm:
            return None
        cxy = cov[:2, :2]
        if math.sqrt(max(np.linalg.eigvalsh(cxy).max(), 0.0)) > cfg.max_fix_std:
            return None
        rng = float(np.min(np.linalg.norm(P[:, :2] - theta[:2], axis=1)))
        self.n_fixes += 1
        self.last_fix = LightFix(t, theta[:2].copy(), float(theta[2]), float(theta[5]), cxy.copy(),
                                 float(math.sqrt(cov[5, 5])), len(ids), ids, rms,
                                 "resection" if len(ids) >= 3 else "line", rng, theta[3:5].copy())
        return self.last_fix

    def status(self) -> dict[str, Any]:
        return {"tracks": len(self.tracks), "identified": sum(tr.light_id is not None for tr in self.tracks),
                "ids_on": sorted(self.last_ids), "fixes": self.n_fixes, "frames": self.frames}
