"""Simulated base UMA-16 (SIMULATOR SIDE: it reads the TRUE targets): target truth -> BearingEstimate.

    est = simulate_bearing(mission_dir, seed=0, levers=ArrayLevers(wind_m_s=4))   # None if nothing is heard
    python -m sfhack.acoustic.sim_array --mission ukraine-1 [--seed 0] [--mode model|audio] [--wind 4] [--out F]

The drone gets only the result (bearing.json, same type and file as the live path, sfhack/acoustic/bearing.py).

Model (mode "model", default; a heuristic budget, not a validated propagation code):
- Sources: running vehicles (world/target_scene.json, kind "vehicle") with a band level (50-2000 Hz) at 1 m:
  car 92, truck 105, tracked APC 110 dB SPL (under load). People are inaudible at km range.
- Propagation: spherical spreading + 3 dB/km (air absorption at a few hundred Hz plus ground excess attenuation
  at grazing incidence). A truck at 1.7 km arrives at ~35 dB, an APC at 3 km at ~31 dB.
- Noise at the array: quiet rural night 35 dB in the band, plus wind noise on the bare MEMS (25 + 20 log10(w) dB).
  Array gain 12 dB (16 mics). Detected when SNR >= 6 dB (recon-acoustic's detect_db).
- Azimuth error = bias + noise. Noise (3 s average): 2 deg at 20 dB SNR, x 10^(-(SNR - 20) / 20), clipped to 1..15
  deg, plus 0.4 deg per m/s of wind (turbulence) in quadrature. Bias: drawn once per scenario, N(0, 3 deg x
  (1 + r / 3 km)) (terrain reflections, refraction, north alignment of the array), plus wind refraction 0.15 deg per
  m/s of crosswind. Reported sigma_az = hypot(noise sigma, 5 deg floor) as the live path does (the array cannot
  know its bias), i.e. ~5-8 deg; with levers the true error can exceed it (the search cone should be generous).
- The group is heard as ONE source: the intensity-weighted mean direction of the vehicles.
- Elevation: true + N(0, 6 deg), reported sigma_el = hypot(6, 10) deg (poor near the horizon).
Mode "audio" (stretch): synthesizes 16-ch audio per vehicle with the real UMA-16 geometry (sfhack/acoustic/synth.py,
received levels from the budget above) and runs the live path's own pipeline (SRP-PHAT + detection + tracker,
sfhack/acoustic/detect.py) on it. The scenario bias is added on top (plane waves have no terrain reflections).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .bearing import BearingEstimate

SOURCE_DB = {"car": 92.0, "truck": 105.0, "apc": 110.0}  # band level at 1 m (dB SPL, 50-2000 Hz), under load
ARRAY_GAIN_DB = 12.0
DETECT_SNR_DB = 6.0
SIGMA_FLOOR_DEG = 5.0
SIGMA_EL_FLOOR_DEG = 10.0
# scenario clock: 23:00 local summer time in Ukraine (UTC+3), the thermal truth's hour / season
SCENARIO_T = datetime(2026, 7, 15, 23, 0, tzinfo=timezone(timedelta(hours=3))).timestamp()


@dataclass
class ArrayLevers:
    sigma_deg: float | None = None  # override the azimuth NOISE sigma (deg, 3 s average)
    bias_deg: float | None = None  # fixed azimuth bias (deg); None = drawn per scenario
    bias_sigma_deg: float = 3.0
    wind_m_s: float = 0.0  # wind at the array
    wind_from_deg: float = 270.0  # meteorological: direction the wind comes FROM, cw from north
    ambient_db: float = 35.0  # band noise floor without wind
    absorption_db_km: float = 3.0
    mode: str = "model"  # model | audio


def compass_az(dx: float, dy: float) -> float:
    return math.degrees(math.atan2(dx, dy)) % 360.0


def load_scene(mission_dir: Path, seed: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """(objects, base truth) for a mission; `seed` picks world/recon/seed_<k>/target_scene.json."""
    w = mission_dir / "world"
    ts = w / "recon" / f"seed_{seed}" / "target_scene.json" if seed is not None else w / "target_scene.json"
    if not ts.exists():
        ts = w / "target_scene.json"
    objs = json.loads(ts.read_text(encoding="utf-8")).get("objects", [])
    base = json.loads((w / "base.json").read_text(encoding="utf-8"))
    return objs, base


def received(objs: list[dict[str, Any]], array_enu: np.ndarray, lv: ArrayLevers) -> list[dict[str, Any]]:
    out = []
    for o in objs:
        if o.get("kind") != "vehicle":
            continue
        d = np.asarray(o["pos"][:3], float) - array_enu
        r = float(np.linalg.norm(d))
        level = SOURCE_DB.get(o.get("class", ""), 100.0) - 20 * math.log10(max(r, 1.0)) - lv.absorption_db_km * r / 1000
        out.append({"class": o.get("class"), "az": compass_az(d[0], d[1]), "el": math.degrees(math.atan2(d[2], math.hypot(d[0], d[1]))),
                    "r": r, "level_db": level})
    return out


def noise_db(lv: ArrayLevers) -> float:
    wind = 25.0 + 20 * math.log10(lv.wind_m_s) if lv.wind_m_s > 0.1 else -99.0
    return 10 * math.log10(10 ** (lv.ambient_db / 10) + 10 ** (wind / 10))


def simulate_bearing(mission_dir: Path | str, seed: int = 0, levers: ArrayLevers | None = None,
                     t_unix: float = SCENARIO_T, scene_seed: int | None = None) -> tuple[BearingEstimate | None, dict]:
    """(estimate or None, truth/debug dict). `scene_seed`: which recon scene (default: `seed`, falling back to the
    default world/target_scene.json)."""
    lv = levers or ArrayLevers()
    mdir = Path(mission_dir)
    objs, base = load_scene(mdir, seed if scene_seed is None else scene_seed)
    array_enu = np.asarray(base["uma16_enu"], float)
    rng = np.random.default_rng(seed + 9700)
    src = received(objs, array_enu, lv)
    info: dict[str, Any] = {"sources": src, "noise_db": noise_db(lv), "mode": lv.mode}
    if not src:
        return None, {**info, "reason": "no running vehicles"}
    p = np.array([10 ** (s["level_db"] / 10) for s in src])
    total_db = 10 * math.log10(p.sum())
    snr = total_db - info["noise_db"] + ARRAY_GAIN_DB
    z = (p * np.exp(1j * np.radians([s["az"] for s in src]))).sum()
    true_az = math.degrees(math.atan2(z.imag, z.real)) % 360.0
    true_el = float(np.average([s["el"] for s in src], weights=p))
    r_mean = float(np.average([s["r"] for s in src], weights=p))
    info.update({"snr_db": snr, "true_az_deg": true_az, "true_el_deg": true_el, "range_m": r_mean})
    if snr < DETECT_SNR_DB:
        return None, {**info, "reason": f"not heard (SNR {snr:.1f} dB < {DETECT_SNR_DB} dB)"}
    if lv.bias_deg is not None:
        bias = lv.bias_deg
    else:
        bias = float(rng.normal(0.0, lv.bias_sigma_deg * (1 + r_mean / 3000.0)))
    cross = math.sin(math.radians(lv.wind_from_deg - true_az))
    bias += 0.15 * lv.wind_m_s * cross
    noise_sigma = float(np.clip(2.0 * 10 ** (-(snr - 20.0) / 20.0), 1.0, 15.0))
    noise_sigma = math.hypot(noise_sigma, 0.4 * lv.wind_m_s)
    if lv.sigma_deg is not None:
        noise_sigma = lv.sigma_deg
    el_noise = 6.0
    conf = float(np.clip(0.5 + 0.5 * np.clip((snr - DETECT_SNR_DB) / 10.0, 0.0, 1.0), 0.0, 1.0))
    if lv.mode == "audio":
        est = _audio_estimate(src, info["noise_db"], base, rng, t_unix, seed)
        if est is None:
            return None, {**info, "reason": "audio pipeline: no stable detection"}
        est.az_deg = (est.az_deg + bias) % 360.0
        est.array_enu = tuple(float(v) for v in array_enu)
    else:
        az = (true_az + bias + rng.normal(0.0, noise_sigma)) % 360.0
        el = true_el + float(rng.normal(0.0, el_noise))
        est = BearingEstimate(t=t_unix, az_deg=az, el_deg=el, sigma_az_deg=math.hypot(noise_sigma, SIGMA_FLOOR_DEG),
                              sigma_el_deg=math.hypot(el_noise, SIGMA_EL_FLOOR_DEG), label="engine",
                              confidence=conf, array_enu=tuple(float(v) for v in array_enu))
    err = (est.az_deg - true_az + 180.0) % 360.0 - 180.0
    info.update({"bias_deg": bias, "noise_sigma_deg": noise_sigma, "az_error_deg": err})
    return est, info


def _audio_estimate(src: list[dict[str, Any]], noise_level_db: float, base: dict[str, Any], rng: np.random.Generator,
                    t_unix: float, seed: int) -> BearingEstimate | None:
    """16-ch synthesis + the live path's pipeline (recon-acoustic). Noise first (learned floor), then the group."""
    from ..config import AcousticConfig
    from .array import Mounting
    from .detect import AcousticPipeline
    from .synth import engine_signal, plane_wave

    cfg = AcousticConfig(x_axis_bearing_deg=float(base.get("uma16_x_axis_az_deg", 0.0)), mics_up=True)
    mount = Mounting(cfg.x_axis_bearing_deg, cfg.mics_up)
    fs, learn, dur = cfg.fs, cfg.learn_s, cfg.learn_s + cfg.stable_s + 3.0
    n, k0 = int(dur * fs), int(learn * fs)
    x = rng.standard_normal((n, 16))  # sensor + ambient noise, unit power per channel = noise_level_db
    for i, s in enumerate(src):
        mono = engine_signal(dur - learn, fs, rpm=float(rng.uniform(1400, 2400)), seed=seed * 10 + i)
        mono = mono / max(float(np.std(mono)), 1e-12) * 10 ** ((s["level_db"] - noise_level_db) / 20)
        x[k0:] += plane_wave(mono, mount.from_compass(s["az"]), s["el"], fs)[: n - k0]
    pipe = AcousticPipeline(cfg)
    w = int(cfg.window_s * fs)
    est = None
    for j in range(0, n - w + 1, w):
        _, b = pipe.process(x[j:j + w], t=t_unix - dur + j / fs)
        est = b or est
    return est


def main() -> None:
    from ..missionprep.build import mission_dir

    ap = argparse.ArgumentParser(prog="python -m sfhack.acoustic.sim_array", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mission", default="ukraine-1")
    ap.add_argument("--seed", type=int, default=0, help="noise seed and recon scene (world/recon/seed_<k>)")
    ap.add_argument("--mode", choices=("model", "audio"), default="model")
    ap.add_argument("--wind", type=float, default=0.0, help="wind at the array, m/s")
    ap.add_argument("--wind-from", type=float, default=270.0)
    ap.add_argument("--sigma", type=float, default=None, help="override the azimuth noise sigma (deg)")
    ap.add_argument("--bias", type=float, default=None, help="fixed azimuth bias (deg)")
    ap.add_argument("--out", default=None, help="bearing.json path (default: world/recon/seed_<k>/bearing.json "
                    "and $SFHACK_DATA/acoustic/bearing_sim.json)")
    a = ap.parse_args()
    mdir = mission_dir(a.mission)
    lv = ArrayLevers(sigma_deg=a.sigma, bias_deg=a.bias, wind_m_s=a.wind, wind_from_deg=a.wind_from, mode=a.mode)
    est, info = simulate_bearing(mdir, a.seed, lv)
    print(f"{len(info['sources'])} running vehicles, noise {info['noise_db']:.1f} dB, "
          f"SNR {info.get('snr_db', float('nan')):.1f} dB, true az {info.get('true_az_deg', float('nan')):.1f} deg "
          f"at {info.get('range_m', float('nan')):.0f} m")
    if est is None:
        print("no bearing:", info.get("reason"))
        return
    print(f"bearing {est.az_deg:.1f} deg +- {est.sigma_az_deg:.1f} (error {info['az_error_deg']:+.1f} deg: bias "
          f"{info['bias_deg']:+.1f}, noise sigma {info['noise_sigma_deg']:.1f}), el {est.el_deg:.1f} +- "
          f"{est.sigma_el_deg:.1f}, confidence {est.confidence:.2f}, label {est.label}")
    outs = [Path(a.out)] if a.out else [mdir / "world" / "recon" / f"seed_{a.seed}" / "bearing.json",
                                        mdir.parent.parent / "acoustic" / "bearing_sim.json"]
    for p in outs:
        est.save(p)
        print("wrote", p)


if __name__ == "__main__":
    main()
