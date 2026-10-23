"""Lever physics: how `Conditions` turn true values into degraded sensor data and forces.

Pure numpy, no MuJoCo, no truth access: the simulator passes the true values in and publishes the result.
Lever semantics follow the TypeScript reference (gnss-spoof-guard `main`, sensors/*.ts); the numbers are
tuned for the X2 and the vision localizer here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .conditions import Conditions, GnssConditions, WeatherConditions, WindConditions, compass_unit

# ---------------------------------------------------------------------------------------- GNSS

# A reported consumer fix is a filtered position, not a raw pseudorange.
# Open sky is the tight end of about 1-5 m CEP (u-blox NEO-M8 spec is 2.0 m; phones and
# weather / multipath sit toward 5 m). CEP ≈ 1.18 σ. This model is that open-sky end:
# σ ≈ 0.8 m, CEP ≈ 1.0 m. Most of it is multipath / atmosphere (Gauss-Markov, 30 s).
# Epoch jitter is a few decimetres. The old 1.5 m of white noise every 5 Hz fix made the
# pin jump metres while the filter averaged it away. The reported covariance is the
# variance actually added (jitter² + slow²).
GNSS_JITTER_M = 0.4  # epoch-to-epoch white noise at the receiver rate
GNSS_SLOW_SIGMA_M = 0.7  # Gauss-Markov, stationary σ
GNSS_SLOW_TAU_S = 30.0  # correlation time of the slow error
# Vertical is about 2× horizontal and is not fused (the reported cov is 2×2). Same split,
# so the pin does not hop by metres every fix.
GNSS_VERT_SIGMA_M = 1.5
GNSS_VERT_JITTER_M = 0.3
GNSS_SIGMA_M = GNSS_JITTER_M  # name kept for callers that pass a white-noise sigma


def gnss_horizontal_variance(jitter_m: float = GNSS_JITTER_M) -> float:
    """Variance the receiver reports, and the variance actually added to the fix (steady state)."""
    return jitter_m**2 + GNSS_SLOW_SIGMA_M**2


def gnss_sigma2_m(jitter_m: float = GNSS_JITTER_M) -> float:
    """2-sigma horizontal radius. The drawn GNSS ring and the reported covariance share this."""
    return 2.0 * math.sqrt(gnss_horizontal_variance(jitter_m))


@dataclass
class GnssFix:
    pos: np.ndarray  # (3,) reported ENU position
    cov: np.ndarray  # (2, 2) horizontal covariance the receiver reports (it does not know it is spoofed)


class GnssModel:
    """Receiver + attacker. `sample` is called at every GNSS epoch with the true position."""

    def __init__(self, rng: np.random.Generator, sigma: float = GNSS_SIGMA_M) -> None:
        self.rng = rng
        self.sigma = sigma
        self.offset = np.zeros(2)  # false offset applied by the attacker (drag / jump)
        self.slow = np.zeros(2)
        self.slow_z = 0.0
        self._last_t: float | None = None

    def sample(self, t: float, true_pos: np.ndarray, lever: GnssConditions) -> GnssFix | None:
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        a = math.exp(-dt / GNSS_SLOW_TAU_S) if dt > 0 else 1.0
        innov = math.sqrt(max(1.0 - a * a, 0.0))
        self.slow = a * self.slow + self.rng.normal(0.0, GNSS_SLOW_SIGMA_M * innov, 2)
        self.slow_z = a * self.slow_z + self.rng.normal(0.0, GNSS_VERT_SIGMA_M * innov)
        if lever.mode == "drag":
            ux, uy = compass_unit(lever.drag_heading_deg)
            self.offset += lever.drag_rate * dt * np.array([ux, uy])
        elif lever.mode == "jump":
            ux, uy = compass_unit(lever.jump_heading_deg)
            self.offset = lever.jump_m * np.array([ux, uy])
        else:
            self.offset[:] = 0.0
        if lever.mode in ("off", "jam"):
            return None
        pos = np.asarray(true_pos, float).copy()
        # Jitter is the only white term. The slow state and the attacker offset carry the metres.
        pos[:2] += self.offset + self.slow + self.rng.normal(0.0, self.sigma, 2)
        pos[2] += self.slow_z + self.rng.normal(0.0, GNSS_VERT_JITTER_M)
        return GnssFix(pos, np.eye(2) * gnss_horizontal_variance(self.sigma))


# ---------------------------------------------------------------------------------------- weather

PIXEL_SIGMA = 25.0  # gray levels of image noise at intensity 1 (x kind factor)
FRAME_DROP = 0.3  # chance a down-camera frame is lost at intensity 1 (x kind factor)
RANGE_SIGMA_M = 1.5  # range noise at intensity 1 (TS reference: 1.5 m)
RANGE_DROPOUT = 0.75  # chance a range return is lost at intensity 1 (TS reference: 0.75)
CLEAR_VISIBILITY_M = math.inf  # clear air: no haze
VISUALS_MAX_VISIBILITY_M = 10000.0  # cap for the viewer values

# kind: (pixel noise, frame drop, range noise, range dropout) factors, visibility at intensity 0 and 1 [m]
# (log-interpolated), fog color (RGB 0..1)
_KINDS: dict[str, tuple[tuple[float, float, float, float], tuple[float, float], tuple[float, float, float]]] = {
    "rain": ((1.0, 0.6, 1.0, 0.6), (5000.0, 150.0), (0.55, 0.58, 0.62)),
    "fog": ((0.3, 0.3, 0.5, 1.0), (2000.0, 12.0), (0.78, 0.80, 0.82)),
    "dust": ((0.7, 0.5, 1.0, 0.8), (3000.0, 30.0), (0.72, 0.60, 0.44)),
}


@dataclass
class WeatherEffect:
    pixel_sigma: float  # gray levels
    frame_drop: float  # probability
    range_sigma: float  # m
    range_dropout: float  # probability
    visibility_m: float  # meteorological visibility (contrast 5 %)
    fog_rgb: tuple[float, float, float]
    streaks: int  # rain streaks per frame


def weather_effect(w: WeatherConditions) -> WeatherEffect:
    i = min(1.0, max(0.0, w.intensity))
    if w.kind == "clear" or i == 0.0:
        return WeatherEffect(0.0, 0.0, 0.0, 0.0, CLEAR_VISIBILITY_M, (0.8, 0.8, 0.8), 0)
    (kp, kf, kr, kd), (v0, v1), rgb = _KINDS[w.kind]
    vis = math.exp(math.log(v0) + i * (math.log(v1) - math.log(v0)))
    return WeatherEffect(i * PIXEL_SIGMA * kp, i * FRAME_DROP * kf, i * RANGE_SIGMA_M * kr, i * RANGE_DROPOUT * kd,
                         vis, rgb, int(round(i * 120)) if w.kind == "rain" else 0)


def transmission(distance_m: float, visibility_m: float) -> float:
    """Koschmieder: fraction of scene contrast left after `distance_m` of air (5 % at the visibility)."""
    return math.exp(-3.0 * max(distance_m, 0.0) / max(visibility_m, 1e-3))


@dataclass
class WeatherVisuals:
    """What a viewer needs to draw the weather. fog_start_m / fog_end_m: linear fog ramp (MuJoCo
    `vis.map.fogstart` / `fogend` are these divided by `stat.extent`)."""

    kind: str
    intensity: float
    visibility_m: float
    fog_rgb: tuple[float, float, float]
    fog_start_m: float
    fog_end_m: float
    haze: float  # 0..1, overall haze strength for 2D panels (1 - transmission over 40 m)


def weather_visuals(w: WeatherConditions) -> WeatherVisuals:
    e = weather_effect(w)
    vis = min(e.visibility_m, VISUALS_MAX_VISIBILITY_M)
    return WeatherVisuals(w.kind, float(w.intensity), vis, e.fog_rgb, fog_start_m=0.05 * vis, fog_end_m=vis,
                          haze=1.0 - transmission(40.0, e.visibility_m))


def degrade_image(img: np.ndarray, e: WeatherEffect, distance_m: float, rng: np.random.Generator) -> np.ndarray:
    """Weather on one RGB frame seen through `distance_m` of air: haze toward the fog color, rain streaks,
    pixel noise. Returns a new uint8 image (the input is not modified)."""
    tr = transmission(distance_m, e.visibility_m)
    if tr > 0.999 and e.pixel_sigma <= 0 and e.streaks == 0:
        return img
    out = img.astype(np.float32)
    if tr < 0.999:
        fog = np.array(e.fog_rgb, np.float32) * 255.0
        out = out * tr + fog * (1.0 - tr)
    if e.streaks:
        h, w = img.shape[:2]
        layer = np.zeros((h, w), np.float32)
        xs, ys = rng.integers(0, w, e.streaks), rng.integers(0, h, e.streaks)
        lens = rng.integers(h // 20 + 1, h // 8 + 2, e.streaks)
        for x, y, ln in zip(xs, ys, lens):
            cv2.line(layer, (int(x), int(y)), (int(x) + int(ln) // 6, int(y) + int(ln)), 1.0, 1)
        out = out * (1.0 - 0.5 * layer[..., None]) + 220.0 * 0.5 * layer[..., None]
    if e.pixel_sigma > 0:
        out += rng.normal(0.0, e.pixel_sigma, out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def degrade_range(r: float, e: WeatherEffect, dropout: float, lost: bool, rng: np.random.Generator) -> float:
    """One range return: lever loss / dropout, then weather noise and dropout. nan = no return.
    The random draws always run, so the RNG stream does not depend on the levers' values."""
    u_lever, u_weather, noise = rng.random(), rng.random(), rng.normal()
    if lost or not math.isfinite(r) or u_lever < dropout or u_weather < e.range_dropout:
        return math.nan
    return max(0.0, r + noise * e.range_sigma)


# ---------------------------------------------------------------------------------------- wind

GUST_TAU_S = 2.0  # gust correlation time


class WindModel:
    """Air velocity = steady wind (speed, compass direction, slow rotation) + Ornstein-Uhlenbeck gusts.
    The force on the airframe is linear drag on the relative air speed (`drag` N per m/s)."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.gust = np.zeros(3)
        self.heading_deg: float | None = None  # current steady-wind direction (integrates the rotation)

    def air_velocity(self, w: WindConditions, dt: float) -> np.ndarray:
        if self.heading_deg is None or w.rotation_deg_s == 0.0:
            self.heading_deg = w.direction_deg
        else:
            self.heading_deg = (self.heading_deg + w.rotation_deg_s * dt) % 360.0
        if w.gust > 0 and dt > 0:
            a = math.exp(-dt / GUST_TAU_S)
            self.gust[:2] = a * self.gust[:2] + self.rng.normal(0.0, w.gust * math.sqrt(1 - a * a), 2)
            self.gust[2] = a * self.gust[2] + self.rng.normal(0.0, 0.3 * w.gust * math.sqrt(1 - a * a))
        elif w.gust <= 0:
            self.gust[:] = 0.0
        ux, uy = compass_unit(self.heading_deg)
        return np.array([w.speed * ux, w.speed * uy, 0.0]) + self.gust

    @staticmethod
    def force(w: WindConditions, air_velocity: np.ndarray, body_vel: np.ndarray, drag: float) -> np.ndarray:
        """Drag force [N] on the airframe. Calm air (no wind, no gusts) applies nothing (no airframe drag
        model in still air, as before the levers existed)."""
        if w.speed <= 0 and w.gust <= 0:
            return np.zeros(3)
        return drag * (np.asarray(air_velocity, float) - np.asarray(body_vel, float))


def wind_from_vector(vx: float, vy: float) -> tuple[float, float]:
    """ENU wind velocity -> (speed, compass direction toward which it blows, degrees)."""
    return math.hypot(vx, vy), math.degrees(math.atan2(vx, vy)) % 360.0


# ---------------------------------------------------------------------------------------- lidar

# Nadir cross-track fan, radians from straight down toward body-right (TS reference: 8 beams, +-0.14 rad).
LIDAR_BEAMS = (-0.14, -0.10, -0.06, -0.02, 0.02, 0.06, 0.10, 0.14)


def lidar_beam_dirs_body() -> np.ndarray:
    """(n, 3) unit beam directions in the body frame (FLU: x forward, y left, z up)."""
    a = np.asarray(LIDAR_BEAMS)
    return np.column_stack([np.zeros_like(a), -np.sin(a), -np.cos(a)])


def camera_frame_kept(c: Conditions, e: WeatherEffect, rng: np.random.Generator) -> bool:
    """False = the frame is never delivered (lever drop rate, then weather dropout)."""
    u1, u2 = rng.random(), rng.random()
    return not (u1 < c.camera.drop_rate or u2 < e.frame_drop)
