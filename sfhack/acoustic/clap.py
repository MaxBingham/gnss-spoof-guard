"""Clap -> BearingEstimate: the stage stand-in for a vehicle (the engine tracker in `detect.py` needs seconds of
steady harmonics, a clap is a 10 ms broadband burst).

A clap is an onset: the 1-3.5 kHz band RMS of a block jumps `threshold_ratio` above a slow noise tracker. The
window around the loudest sample goes through the same SRP-PHAT as the engine path, with a clap band (above
~4.1 kHz the 42 mm pitch aliases) and a learned noise floor so broadband bins keep full weight. Claps within
`memory_s` are averaged (circular mean) into one bearing; its sigma shrinks with the spread and count.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt

from .array import Mounting
from .bearing import BearingEstimate
from .detect import circ_mean_std_deg
from .doa import DoaConfig, DoaResult, SrpPhat


@dataclass
class ClapConfig:
    fs: int = 48000
    fmin: float = 1000.0
    fmax: float = 3500.0
    threshold_ratio: float = 8.0       # onset: band RMS over the noise tracker
    min_rms: float = 1.5e-3            # absolute floor (full scale 1.0) so a quiet room does not trigger on ticks
    refractory_s: float = 0.45         # one clap, one fix (room echoes ring for a few hundred ms)
    warmup_s: float = 1.0              # learn the noise floor first
    window_s: float = 0.05             # DOA window centred on the clap peak
    memory_s: float = 20.0             # claps older than this no longer count toward the bearing
    min_peak: float = 0.15             # SRP peak below this = no coherent wavefront (handling noise, overload)
    sigma_floor_deg: float = 5.0       # array + calibration error that averaging does not remove


@dataclass
class Clap:
    t: float                           # stream time of the clap peak (s)
    compass_az_deg: float
    el_deg: float
    doa: DoaResult
    rms: float


class ClapBearing:
    def __init__(self, cfg: ClapConfig, mount: Mounting, array_enu=None):
        self.cfg, self.mount, self.array_enu = cfg, mount, array_enu
        c = cfg
        self.srp = SrpPhat(DoaConfig(fs=c.fs, nfft=1024, hop=256, fmin=c.fmin, fmax=c.fmax, el_step=5.0))
        self.sos = butter(4, [c.fmin, c.fmax], btype="bandpass", fs=c.fs, output="sos")
        self.zi = None
        self.noise: float | None = None
        self.n_seen = 0
        self.last_clap_t = -1e9
        self.prev = np.zeros((0, 16))
        self.quiet: list[np.ndarray] = []
        self.claps: list[Clap] = []

    def process(self, block: np.ndarray, t_end: float) -> Clap | None:
        """block (n, 16) of the stream ending at stream time `t_end`; returns a Clap when one is accepted."""
        c = self.cfg
        if self.zi is None:
            self.zi = np.zeros((self.sos.shape[0], 2, block.shape[1]))
        band, self.zi = sosfilt(self.sos, block, axis=0, zi=self.zi)
        rms = float(np.sqrt(np.mean(band ** 2)))
        self.n_seen += len(block)
        buf = np.concatenate([self.prev, block])[-(len(block) + int(c.window_s * c.fs)):]
        self.prev = buf

        warm = self.n_seen < c.warmup_s * c.fs
        if self.noise is None:
            self.noise = rms
        onset = (not warm and rms > max(c.threshold_ratio * self.noise, c.min_rms)
                 and t_end - self.last_clap_t > c.refractory_s)
        if not onset:
            self.noise = 0.95 * self.noise + 0.05 * rms if rms < 3 * self.noise else self.noise
            self.quiet = (self.quiet + [block])[-20:]
            if warm or self.srp.noise_psd is None:
                self.srp.learn_noise(np.concatenate(self.quiet))
            return None

        k = int(np.argmax(np.abs(buf).sum(axis=1)))
        n = int(c.window_s * c.fs)
        s = int(np.clip(k - n // 2, 0, max(len(buf) - n, 0)))
        doa = self.srp.estimate(buf[s:s + n])
        self.last_clap_t = t_end
        if doa.peak < c.min_peak:
            return None
        t_peak = t_end - (len(buf) - k) / c.fs
        clap = Clap(t_peak, self.mount.to_compass(doa.array_az_deg), doa.array_el_deg, doa, rms)
        self.claps.append(clap)
        return clap

    def bearing(self, t_now: float) -> BearingEstimate | None:
        c = self.cfg
        recent = [k for k in self.claps if t_now - k.t <= c.memory_s]
        if not recent:
            return None
        az, std = circ_mean_std_deg(np.array([k.compass_az_deg for k in recent]))
        n = len(recent)
        # one clap: no spread to measure, assume 10 deg; more: standard error of the mean, never below the floor
        sigma = float(np.hypot(c.sigma_floor_deg, (std if n > 1 else 10.0) / np.sqrt(n)))
        el = float(np.mean([k.el_deg for k in recent]))
        conf = float(np.clip(np.mean([k.doa.peak for k in recent]) * min(n, 3) / 3 / 0.5, 0.0, 1.0))
        return BearingEstimate(t=time.time(), az_deg=az, el_deg=el, sigma_az_deg=sigma, sigma_el_deg=15.0,
                               label="clap", confidence=conf, array_enu=self.array_enu)
