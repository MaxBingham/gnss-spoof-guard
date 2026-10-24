"""Continuous bearing track of a MOVING source (a speaker playing a drone sound walked around the array).

`detect.AcousticPipeline` only reports a stationary source (3 s steady); this module follows one at ~5 Hz:

- `BandDetector`: per analysis window, 1/3-octave band levels (`band_fmin`-`band_fmax`) over a per-band floor
  learned during the first `learn_s`; level = the `detect_bands`-th largest excess (one tone is not enough).
  Hysteresis: `start_n` consecutive hits to switch on, `stop_n` consecutive misses to switch off.
- `BearingTracker`: circular alpha-beta filter on the compass azimuth (deg) and its rate (deg/s). Innovations
  beyond max(`gate_sigma` * sigma, `gate_min_deg`) are rejected unless `reinit_n` consecutive outliers agree
  (circular std < `reinit_agree_deg`), which re-initializes the track there. sigma_az = sqrt(EWMA of accepted
  innovations^2), floored at `sigma_floor_deg`.
- `TrackPipeline`: blocks in -> one analysis every `hop_s` on the last `window_s` (SRP-PHAT, `doa.py`) -> the UDP
  message (`message()`): BearingEstimate.to_dict() + az_rate_dps, level_db, detected, seq; one
  {"detected": false} when the source is lost; nothing before the track has `min_accepted` accepted windows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .array import Mounting
from .bearing import BearingEstimate
from .detect import circ_mean_std_deg
from .doa import DoaConfig, DoaResult, SrpPhat, angle_diff_deg


@dataclass
class TrackConfig:
    fs: int = 44100
    window_s: float = 0.4             # analysis window (last window_s of audio)
    hop_s: float = 0.2                # one analysis per hop (5 Hz)
    learn_s: float = 5.0              # noise floor (keep quiet)
    # detection
    band_fmin: float = 500.0
    band_fmax: float = 5000.0
    detect_db: float = 6.0
    detect_bands: int = 3
    start_n: int = 2
    stop_n: int = 3
    # DOA
    doa_fmin: float = 800.0
    doa_fmax: float = 4000.0          # above ~4.1 kHz the 42 mm pitch aliases
    max_el_deg: float | None = None   # windows from higher up are ignored (typing at the laptop: 40-65 deg)
    # tracker
    alpha: float = 0.35
    beta: float = 0.075               # ~Kalata-optimal for alpha 0.35
    max_rate_dps: float = 20.0
    gate_sigma: float = 3.0
    gate_min_deg: float = 25.0
    # 0.4 s windows every 0.2 s: a 0.2 s burst can touch 3 consecutive windows, so 4 must agree to re-init.
    reinit_n: int = 4
    reinit_agree_deg: float = 15.0
    sigma_init_deg: float = 10.0
    sigma_floor_deg: float = 3.0
    sigma_ewma: float = 0.2
    sigma_el_floor_deg: float = 10.0
    min_accepted: int = 2             # accepted windows before the first estimate goes out
    peak_good: float = 0.4            # SRP peak that counts as full confidence


class BandDetector:
    def __init__(self, cfg: TrackConfig):
        self.cfg = cfg
        edges = cfg.band_fmin * 2.0 ** (np.arange(0, 64) / 3.0)
        self.edges = edges[edges <= cfg.band_fmax * 1.0001]
        self.floor: np.ndarray | None = None
        self.on = False
        self._hits = 0
        self._misses = 0

    def band_energy(self, x: np.ndarray) -> np.ndarray:
        """Energy per 1/3-octave band (mean over channels)."""
        X = np.fft.rfft(x * np.hanning(len(x))[:, None], axis=0)
        p = (np.abs(X) ** 2).mean(axis=1)
        f = np.fft.rfftfreq(len(x), 1.0 / self.cfg.fs)
        return np.array([p[(f >= lo) & (f < hi)].mean() for lo, hi in zip(self.edges[:-1], self.edges[1:])])

    def learn(self, noise: np.ndarray) -> None:
        n = int(self.cfg.window_s * self.cfg.fs)
        h = int(self.cfg.hop_s * self.cfg.fs)
        es = [self.band_energy(noise[k:k + n]) for k in range(0, len(noise) - n + 1, h)]
        self.floor = np.median(es, axis=0)

    def level_db(self, x: np.ndarray) -> float:
        ex = np.sort(10 * np.log10(self.band_energy(x) / np.maximum(self.floor, 1e-30)))[::-1]
        return float(ex[min(self.cfg.detect_bands, len(ex)) - 1])

    def update(self, level_db: float) -> tuple[bool, bool]:
        """(this window above threshold, hysteresis state)."""
        hit = level_db >= self.cfg.detect_db
        if hit:
            self._hits += 1
            self._misses = 0
            if not self.on and self._hits >= self.cfg.start_n:
                self.on = True
        else:
            self._hits = 0
            self._misses += 1
            if self.on and self._misses >= self.cfg.stop_n:
                self.on = False
        return hit, self.on


@dataclass
class TrackState:
    t: float
    az_deg: float
    rate_dps: float
    sigma_az_deg: float
    el_deg: float
    sigma_el_deg: float
    n_accepted: int
    accepted: bool | None             # this update: True accepted, False rejected (outlier), None coasted


class BearingTracker:
    def __init__(self, cfg: TrackConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.az: float | None = None
        self.rate = 0.0
        self.var = self.cfg.sigma_init_deg ** 2
        self.el = 0.0
        self.var_el = self.cfg.sigma_el_floor_deg ** 2
        self.t: float | None = None
        self.n_accepted = 0
        self.outliers: list[tuple[float, float, float]] = []   # t, az, el

    @property
    def sigma(self) -> float:
        return max(float(np.sqrt(self.var)), self.cfg.sigma_floor_deg)

    def _init(self, t: float, az: float, el: float, rate: float = 0.0) -> None:
        self.az, self.rate, self.el, self.t = az % 360.0, rate, el, t
        self.var = self.cfg.sigma_init_deg ** 2
        self.n_accepted = 1
        self.outliers = []

    def predict(self, t: float) -> float:
        return (self.az + self.rate * (t - self.t)) % 360.0

    def state(self, t: float, accepted: bool | None) -> TrackState:
        return TrackState(t, self.predict(t), self.rate, self.sigma,
                          self.el, max(float(np.sqrt(self.var_el)), self.cfg.sigma_el_floor_deg),
                          self.n_accepted, accepted)

    def coast(self, t: float) -> TrackState | None:
        """No measurement this window: predicted state with the uncertainty grown by the rate uncertainty."""
        if self.az is None:
            return None
        s = self.state(t, None)
        s.sigma_az_deg = float(np.hypot(s.sigma_az_deg, 0.5 * (t - self.t)))
        return s

    def update(self, t: float, az: float, el: float) -> TrackState:
        c = self.cfg
        if self.az is None:
            self._init(t, az, el)
            return self.state(t, True)
        az_p = self.predict(t)
        innov = angle_diff_deg(az, az_p)
        if abs(innov) > max(c.gate_sigma * self.sigma, c.gate_min_deg):
            self.outliers.append((t, az, el))
            self.outliers = self.outliers[-c.reinit_n:]
            if len(self.outliers) >= c.reinit_n:
                o = np.array(self.outliers)
                mean, std = circ_mean_std_deg(o[:, 1])
                if std < c.reinit_agree_deg:
                    self._init(t, mean, float(o[:, 2].mean()))
                    return self.state(t, True)
            return self.state(t, False)
        self.outliers = []
        dt = max(t - self.t, 1e-3)
        self.az = (az_p + c.alpha * innov) % 360.0
        self.rate = float(np.clip(self.rate + c.beta / dt * innov, -c.max_rate_dps, c.max_rate_dps))
        self.t = t
        self.var = (1 - c.sigma_ewma) * self.var + c.sigma_ewma * innov ** 2
        de = el - self.el
        self.el += c.alpha * de
        self.var_el = (1 - c.sigma_ewma) * self.var_el + c.sigma_ewma * de ** 2
        self.n_accepted += 1
        return self.state(t, True)


@dataclass
class TrackStep:
    """One analysis window."""
    t: float
    learning: bool
    level_db: float = 0.0
    hit: bool = False                 # this window above threshold
    detected: bool = False            # hysteresis state
    doa: DoaResult | None = None
    compass_az: float | None = None   # raw window bearing
    state: TrackState | None = None
    message: dict | None = None       # UDP message (estimate or lost), None = send nothing
    note: str = ""


@dataclass
class TrackPipeline:
    cfg: TrackConfig = field(default_factory=TrackConfig)
    mount: Mounting = field(default_factory=Mounting)
    array_enu: tuple[float, float, float] | None = None

    def __post_init__(self):
        c = self.cfg
        self.srp = SrpPhat(DoaConfig(fs=c.fs, fmin=c.doa_fmin, fmax=c.doa_fmax))
        self.det = BandDetector(c)
        self.trk = BearingTracker(c)
        self.n_win = int(round(c.window_s * c.fs))
        self.n_hop = int(round(c.hop_s * c.fs))
        self.buf = np.zeros((0, 16))
        self.since = 0
        self.n_in = 0
        self._learn: list[np.ndarray] = []
        self.seq = 0
        self.emitting = False

    @property
    def learning(self) -> bool:
        return self.det.floor is None

    def process(self, block: np.ndarray, t: float, t_unix: float | None = None) -> TrackStep | None:
        """block: (n, 16); t: audio time at the END of the block; t_unix: message time (default t).
        Returns a TrackStep whenever an analysis ran (every hop_s once the window is full)."""
        c = self.cfg
        self.n_in += len(block)
        if self.learning:
            self._learn.append(block)
            if self.n_in >= c.learn_s * c.fs:
                noise = np.concatenate(self._learn)
                self.det.learn(noise)
                self.srp.learn_noise(noise)
                self._learn = []
                return TrackStep(t, True, note="noise floor learned")
            return None
        self.buf = np.concatenate([self.buf, block])[-self.n_win:]
        self.since += len(block)
        if len(self.buf) < self.n_win or self.since < self.n_hop:
            return None
        self.since = 0
        x = self.buf
        level = self.det.level_db(x)
        hit, on = self.det.update(level)
        step = TrackStep(t, False, level, hit, on)
        tm = t if t_unix is None else t_unix
        if not on:
            if self.emitting:
                self.seq += 1
                step.message = {"detected": False, "seq": self.seq, "t": float(tm)}
                step.note = "LOST"
            self.emitting = False
            self.trk.reset()
            return step
        if hit:
            doa = self.srp.estimate(x)
            step.doa = doa
            step.compass_az = self.mount.to_compass(doa.array_az_deg)
            if c.max_el_deg is not None and doa.array_el_deg > c.max_el_deg:
                step.note = f"el {doa.array_el_deg:.0f} > max"
                step.state = self.trk.coast(t)
            else:
                step.state = self.trk.update(t, step.compass_az, doa.array_el_deg)
        else:
            step.state = self.trk.coast(t)
        s = step.state
        if s is None or self.trk.n_accepted < c.min_accepted:
            return step
        self.emitting = True
        self.seq += 1
        margin = float(np.clip((level - c.detect_db) / 10.0, 0.0, 1.0))
        peak = float(np.clip(step.doa.peak / c.peak_good, 0.0, 1.0)) if step.doa is not None else 0.0
        conf = (0.5 * margin + 0.5 * peak) * (1.0 if s.accepted else 0.5)
        b = BearingEstimate(t=float(tm), az_deg=s.az_deg, el_deg=s.el_deg, sigma_az_deg=s.sigma_az_deg,
                            sigma_el_deg=s.sigma_el_deg, label="drone", confidence=float(np.clip(conf, 0, 1)),
                            array_enu=self.array_enu)
        step.message = {**b.to_dict(), "az_rate_dps": float(s.rate_dps), "level_db": float(level),
                        "detected": True, "seq": self.seq}
        return step
