"""Clap -> BearingEstimate: the stage stand-in for a vehicle (the engine tracker in `detect.py` needs seconds of
steady harmonics, a clap is a 10 ms broadband burst).

A clap is an onset: the 1-3.5 kHz band RMS of a block jumps `threshold_ratio` above a slow noise tracker. The
first `direct_s` after the onset (found to the millisecond, before echoes off the table, laptop and walls come in)
goes through the same SRP-PHAT as the engine path, with a clap band (above
~4.1 kHz the 42 mm pitch aliases) and a learned noise floor so broadband bins keep full weight. Claps within
`memory_s` are combined (circular mean, circular median from 3 on) into one bearing; its sigma shrinks with
the spread and count.
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
    refractory_s: float = 0.8          # one clap, one fix (a wall echo 0.5 s later came in as a second clap)
    warmup_s: float = 1.0              # learn the noise floor first
    window_s: float = 0.05             # DOA window centred on the clap peak (direct_s = 0)
    direct_s: float = 0.004            # > 0: DOA from only this much after the onset, before the echoes arrive
                                       # (demos/45_clap_bench.py: 3-5 ms all within 5 deg, 6 ms+ catches echoes)
    lead_s: float = 0.001              # direct window starts this much before the detected onset
    memory_s: float = 6.0              # claps older than this no longer count (a new spot takes over quickly)
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
        self.prev_band = np.zeros((0, 16))
        self.pending: float | None = None  # onset seen, direct window runs past the block: finish on the next one
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
        keep = len(block) + int(max(c.window_s, 2 * c.direct_s + 0.02) * c.fs)
        buf = np.concatenate([self.prev, block])[-keep:]
        bbuf = np.concatenate([self.prev_band, band])[-keep:]
        self.prev, self.prev_band = buf, bbuf

        if self.pending is not None:
            onset_rms, self.pending = self.pending, None
            return self._locate(buf, bbuf, t_end, onset_rms, len(block))
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
        self.last_clap_t = t_end
        return self._locate(buf, bbuf, t_end, rms, None)

    def _locate(self, buf, bbuf, t_end, rms, last_block) -> Clap | None:
        c = self.cfg
        if c.direct_s > 0:
            s, n = self._direct_window(bbuf, last_block)
            if s + n > len(buf):
                self.pending = rms
                return None
            seg = buf[s:s + n]
            nfft = self.srp.cfg.nfft
            pad = (nfft - n) // 2
            doa = self.srp.estimate(np.pad(seg, ((pad, nfft - n - pad), (0, 0))), noise_scale=n / nfft)
            k = s + int(c.lead_s * c.fs)
        else:
            k = int(np.argmax(np.abs(buf).sum(axis=1)))
            n = int(c.window_s * c.fs)
            s = int(np.clip(k - n // 2, 0, max(len(buf) - n, 0)))
            doa = self.srp.estimate(buf[s:s + n])
        if doa.peak < c.min_peak:
            return None
        clap = Clap(t_end - (len(buf) - k) / c.fs, self.mount.to_compass(doa.array_az_deg), doa.array_el_deg,
                    doa, rms)
        self.claps.append(clap)
        return clap

    def _direct_window(self, bbuf, last_block) -> tuple[int, int]:
        """Start and length of the direct-sound window: the first 1 ms frame of the band energy that rises
        clearly over the room, minus `lead_s`. On a deferred onset only the part before the new block counts."""
        c = self.cfg
        fr = max(int(c.fs / 1000), 1)
        e = (bbuf[: len(bbuf) // fr * fr] ** 2).mean(axis=1).reshape(-1, fr).mean(axis=1)
        search = e if last_block is None else e[: max((len(bbuf) - last_block) // fr, 1)]
        thr = max(0.1 * float(search.max()), 10.0 * (self.noise or 0.0) ** 2)
        i = int(np.argmax(search > thr))
        n = int(c.direct_s * c.fs)
        return max(i * fr - int(c.lead_s * c.fs), 0), n

    def bearing(self, t_now: float) -> BearingEstimate | None:
        c = self.cfg
        recent = [k for k in self.claps if t_now - k.t <= c.memory_s]
        if not recent:
            return None
        a = np.array([k.compass_az_deg for k in recent])
        n = len(recent)
        az, std = circ_mean_std_deg(a)
        if n >= 3:
            # circular median (the clap closest to all others): an echo or a stray sound cannot drag the bearing
            dev = np.abs((a[:, None] - a[None, :] + 180.0) % 360.0 - 180.0)
            az = float(a[np.argmin(dev.sum(axis=1))])
            std = 1.4826 * float(np.median(np.abs((a - az + 180.0) % 360.0 - 180.0)))
        # one clap: no spread to measure, assume 10 deg; more: standard error of the mean, never below the floor
        sigma = float(np.hypot(c.sigma_floor_deg, (std if n > 1 else 10.0) / np.sqrt(n)))
        el = float(np.mean([k.el_deg for k in recent]))
        conf = float(np.clip(np.mean([k.doa.peak for k in recent]) * min(n, 3) / 3 / 0.5, 0.0, 1.0))
        return BearingEstimate(t=time.time(), az_deg=az, el_deg=el, sigma_az_deg=sigma, sigma_el_deg=15.0,
                               label="clap", confidence=conf, array_enu=self.array_enu)
