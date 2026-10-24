"""Source detection + bearing tracking for the base UMA-16 (live or file, same code).

Per window (`AcousticConfig.window_s`):
1. levels per channel (dBFS) for the channel check;
2. detection: energy in the detection band (50-2000 Hz, engine-like) over a noise floor learned during the first
   `learn_s` seconds (keep the source quiet then); an engine score from the cepstrum (harmonic series of a
   firing frequency 20-200 Hz) sets the label;
3. DOA (SRP-PHAT, `doa.py`) with the learned noise floor as bin weights;
4. tracking: when the source has been detected for `stable_s` with a steady azimuth (circular std <
   `stable_std_deg`), a BearingEstimate is emitted (compass bearing through the mounting config).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d

from sfhack.config import AcousticConfig

from .array import Mounting
from .bearing import BearingEstimate
from .doa import DoaConfig, DoaResult, SrpPhat


@dataclass
class WindowResult:
    t: float
    levels_dbfs: np.ndarray       # (16,) per USB channel
    band_db: float                # detection-band energy over the floor (dB); 0 while learning
    detected: bool
    engine_score: float           # harmonic comb (dB), see engine_score()
    learning: bool
    doa: DoaResult | None


def circ_mean_std_deg(a_deg: np.ndarray) -> tuple[float, float]:
    z = np.exp(1j * np.radians(a_deg)).mean()
    r = min(abs(z), 1.0)
    return float(np.degrees(np.angle(z)) % 360.0), float(np.degrees(np.sqrt(-2.0 * np.log(max(r, 1e-12)))))


class AcousticPipeline:
    def __init__(self, cfg: AcousticConfig | None = None, array_enu=None, full_scale: float = 1.0):
        self.cfg = cfg or AcousticConfig()
        c = self.cfg
        self.srp = SrpPhat(DoaConfig(fs=c.fs, fmin=c.doa_fmin, fmax=c.doa_fmax))
        self.mount = Mounting(c.x_axis_bearing_deg, c.mics_up)
        self.array_enu = array_enu
        self.full_scale = full_scale
        self.t0: float | None = None
        self._learn: list[np.ndarray] = []
        self.floor_energy: float | None = None
        self.history: deque[tuple[float, float, float, float, float]] = deque()  # t, az, el, band_db, engine
        self.last_bearing: BearingEstimate | None = None
        self._f0_grid = np.arange(20.0, 200.0, 0.5)

    @property
    def learning(self) -> bool:
        return self.floor_energy is None

    def band_energy(self, x: np.ndarray) -> float:
        X = np.fft.rfft(x * np.hanning(len(x))[:, None], axis=0)
        f = np.fft.rfftfreq(len(x), 1.0 / self.cfg.fs)
        m = (f >= self.cfg.detect_fmin) & (f <= self.cfg.detect_fmax)
        return float((np.abs(X[m]) ** 2).mean())

    def engine_score(self, x: np.ndarray) -> float:
        """Harmonic comb (dB): max over f0 in 20-200 Hz of the median over harmonics (up to detect_fmax) of the
        level at h*f0 minus the level at (h+0.5)*f0. A single tone does not raise a median. Heuristic: white
        noise ~4 dB, synthetic engine 9-12 dB, the real tone recordings 6-10 dB."""
        mono = x.mean(axis=1)
        L = 10 * np.log10(np.abs(np.fft.rfft(mono * np.hanning(len(mono)))) ** 2 + 1e-30)
        L = maximum_filter1d(L, 3)
        df = self.cfg.fs / len(mono)
        best = 0.0
        for f0 in self._f0_grid:
            h = np.arange(1, int(self.cfg.detect_fmax // f0) + 1)
            if len(h) < 5:
                continue
            on = L[np.rint(h * f0 / df).astype(int)]
            off = L[np.rint((h + 0.5) * f0 / df).astype(int)]
            best = max(best, float(np.median(on - off)))
        return best

    def process(self, x: np.ndarray, t: float | None = None) -> tuple[WindowResult, BearingEstimate | None]:
        """x: (n, 16) float samples of one window (any scale; `full_scale` sets dBFS)."""
        c = self.cfg
        t = time.time() if t is None else t
        if self.t0 is None:
            self.t0 = t
        levels = 20 * np.log10(np.sqrt((x ** 2).mean(axis=0)) / self.full_scale + 1e-12)
        e = self.band_energy(x)
        if self.learning:
            self._learn.append(x)
            if t - self.t0 + c.window_s >= c.learn_s:
                noise = np.concatenate(self._learn)
                self.srp.learn_noise(noise)
                n = int(c.window_s * c.fs)
                es = [self.band_energy(noise[k:k + n]) for k in range(0, len(noise) - n + 1, n)]
                self.floor_energy = float(np.median(es))
                self._learn = []
            return WindowResult(t, levels, 0.0, False, 0.0, True, None), None
        band_db = 10 * np.log10(e / max(self.floor_energy, 1e-30))
        detected = band_db >= c.detect_db
        doa = self.srp.estimate(x)
        eng = self.engine_score(x)
        res = WindowResult(t, levels, band_db, detected, eng, False, doa)
        return res, self._track(res)

    def _track(self, r: WindowResult) -> BearingEstimate | None:
        c = self.cfg
        if not r.detected:
            self.history.clear()
            return None
        self.history.append((r.t, r.doa.array_az_deg, r.doa.array_el_deg, r.band_db, r.engine_score))
        while self.history and r.t - self.history[0][0] > c.stable_s:
            self.history.popleft()
        h = np.array(self.history)
        if r.t - h[0, 0] + c.window_s < c.stable_s:
            return None
        az_mean, az_std = circ_mean_std_deg(h[:, 1])
        if az_std > c.stable_std_deg:
            return None
        el_mean, el_std = float(h[:, 2].mean()), float(h[:, 2].std())
        # Window estimates are correlated (same source, same reflections): no 1/sqrt(n) shrink.
        sigma_az = float(np.hypot(az_std, c.sigma_az_floor_deg))
        sigma_el = float(np.hypot(el_std, c.sigma_el_floor_deg))
        snr_term = np.clip((h[:, 3].mean() - c.detect_db) / 10.0, 0.0, 1.0)
        conf = float(np.clip(0.5 + 0.5 * snr_term, 0, 1) * (1.0 - 0.5 * az_std / c.stable_std_deg))
        label = "engine" if np.median(h[:, 4]) >= self.cfg.engine_score_db else "unknown"
        self.last_bearing = BearingEstimate(
            t=float(r.t), az_deg=self.mount.to_compass(az_mean), el_deg=el_mean,
            sigma_az_deg=sigma_az, sigma_el_deg=sigma_el, label=label, confidence=conf,
            array_enu=self.array_enu)
        return self.last_bearing
