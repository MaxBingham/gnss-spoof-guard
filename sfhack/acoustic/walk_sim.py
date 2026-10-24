"""Synthetic UMA-16 recordings of a speaker walked around the array (SIMULATOR / TEST SIDE: it knows the truth).

- `drone_signal()` / `band_noise()`: the mono test sounds (small quadcopter, 800-4000 Hz pink-ish fallback), built
  to loop seamlessly: every tone has an integer number of cycles over the file and the noise is shaped circularly.
- `room_noise()`: diffuse room noise (plane waves from many directions, + a little sensor noise) with the spectrum
  of the quiet clap sessions (`/home/max/sf-hack-data/acoustic/clap_*`): ~-90 dBFS per channel in 800-4000 Hz,
  coherent between neighbouring mics below ~1.6 kHz like the real room.
- `render_moving()`: near-field spherical propagation (1/r, exact fractional delays per mic) from a moving source,
  windowed overlap-add in short blocks so the motion is smooth; image sources for early reflections.
- `simulate_walk()`: the whole recording + truth; `doa_check()`: SRP-PHAT bearings vs truth.

Frames: world = ENU at the array (z up, array plane z = 0). Array frame and mounting as in `array.py`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .array import C_SOUND, Mounting, mic_positions, unit_vector

# Quiet-room PSD of the real UMA-16 (dB re FS^2/Hz, mean over channels, quietest 20 % of 0.5 s blocks,
# clap_20261024_211923). Only the shape is used; the level is set by `noise_dbfs`.
ROOM_PSD = ((35, -91), (75, -92), (150, -101), (300, -109), (600, -115), (1130, -124), (2260, -129),
            (4500, -134), (9000, -137), (17000, -138))
BAND = (800.0, 4000.0)


def _smooth_mask(f: np.ndarray, lo: float, hi: float, oct_edge: float = 0.25) -> np.ndarray:
    """1 inside [lo, hi], raised-cosine skirts `oct_edge` octaves wide outside."""
    lf = np.log2(np.maximum(f, 1e-3))
    m = np.ones_like(f)
    for edge, side in ((math.log2(lo), -1), (math.log2(hi), 1)):
        d = (lf - edge) * side / oct_edge
        m = np.where(d > 0, np.where(d < 1, 0.5 + 0.5 * np.cos(np.pi * d), 0.0), m)
    return m


def _shaped_noise(n: int, fs: int, mag, rng: np.random.Generator) -> np.ndarray:
    """Periodic noise of length n with |spectrum| = mag(f) (circular, so it loops seamlessly)."""
    f = np.fft.rfftfreq(n, 1.0 / fs)
    X = (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) * mag(f)
    X[0] = 0.0
    return np.fft.irfft(X, n=n)


def band_fraction(x: np.ndarray, fs: int, lo: float = 1000.0, hi: float = 4000.0) -> float:
    P = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    return float(P[(f >= lo) & (f <= hi)].sum() / max(P[1:].sum(), 1e-30))


def drone_signal(duration_s: float, fs: int = 44100, seed: int = 0, hf_frac: float = 0.55,
                 n_rotors: int = 4, blades: int = 2, pole_pairs: int = 7) -> np.ndarray:
    """Small quadcopter, mono, peak 1, exactly periodic over `duration_s` (seamless loop).

    Per rotor: blade-pass frequency 160-240 Hz drifting independently +-3 % (sum of slow sines, periods 4-60 s),
    blade-pass harmonics (1/h^0.75), half-order shaft-rate lines, motor whine at pole_pairs x shaft rate (+2
    harmonics). Broadband turbulence: noise peaking at ~2.2 kHz, amplitude-modulated by the blade passes. The
    broadband gain is solved so `hf_frac` of the energy lies in 1-4 kHz (where the array's DOA works).
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    T = n / fs
    t = np.arange(n) / fs
    tonal = np.zeros(n)
    pulses = np.zeros(n)
    for _ in range(n_rotors):
        f0 = 2.0 * round(rng.uniform(160.0, 240.0) * T / 2.0) / T    # f0 T even: shaft rate f0/2 also periodic
        ks = rng.integers(max(1, int(T / 60)), max(2, int(T / 4)) + 1, size=4)
        amps = rng.uniform(0.3, 1.0, 4)
        amps *= 0.03 / amps.sum()
        phis = rng.uniform(0, 2 * np.pi, 4)
        # phase = 2 pi int f0 (1 + sum a sin(2 pi k t / T + phi)) dt, closed form (periodic over T)
        ph = 2 * np.pi * f0 * t
        for a, k, p in zip(amps, ks, phis):
            ph -= f0 * a * T / k * (np.cos(2 * np.pi * k * t / T + p) - np.cos(p))
        drift_max = f0 * 1.03
        am = 1.0 + 0.15 * np.sin(2 * np.pi * rng.integers(2, 12) * t / T + rng.uniform(0, 6.28))
        h = 1
        while h * drift_max < 7000.0:
            tonal += am * h ** -0.75 * np.sin(h * ph + rng.uniform(0, 6.28))
            h += 1
        for h in (0.5, 1.5, 2.5):
            tonal += 0.12 * np.sin(h * ph + rng.uniform(0, 6.28))
        e = pole_pairs / blades                                       # electrical / blade-pass frequency
        for hh, a in ((1, 0.30), (2, 0.15), (3, 0.08)):
            tonal += a * am * np.sin(hh * e * ph + rng.uniform(0, 6.28))
        pulses += ((1.0 + np.cos(ph)) / 2.0) ** 6
    pulses /= np.mean(pulses) + 1e-12

    def turb_mag(f):
        lf = np.log2(np.maximum(f, 1.0) / 2200.0)
        return np.exp(-0.5 * (lf / 1.1) ** 2) * _smooth_mask(f, 250.0, 9000.0, 0.5)

    turb = _shaped_noise(n, fs, turb_mag, rng) * (0.4 + 0.6 * pulses)
    # gentle high-pass of the tonal part (a Bluetooth speaker hardly reproduces < 120 Hz anyway)
    Tf = np.fft.rfft(tonal)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    Tf *= _smooth_mask(f, 120.0, 12000.0, 0.5)
    tonal = np.fft.irfft(Tf, n=n)

    def energies(x):
        P = np.abs(np.fft.rfft(x)) ** 2
        return P[1:].sum(), P[(f >= 1000) & (f <= 4000)].sum()

    et, etb = energies(tonal)
    en, enb = energies(turb)
    g2 = (hf_frac * et - etb) / max(enb - hf_frac * en, 1e-30)
    x = tonal + math.sqrt(max(g2, 0.0)) * turb
    return x / np.max(np.abs(x))


def band_noise(duration_s: float, fs: int = 44100, seed: int = 0, lo: float = 800.0, hi: float = 4000.0) -> np.ndarray:
    """Pink-ish noise in [lo, hi] (the easiest DOA signal), peak 1, periodic."""
    rng = np.random.default_rng(seed)
    x = _shaped_noise(int(round(duration_s * fs)), fs,
                      lambda f: _smooth_mask(f, lo, hi, 0.1) / np.sqrt(np.maximum(f, 1.0)), rng)
    return x / np.max(np.abs(x))


def _band_power(x: np.ndarray, fs: int, lo: float = BAND[0], hi: float = BAND[1]) -> float:
    """Mean power per channel in [lo, hi] (x: (n,) or (n, ch))."""
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    sel = (f >= lo) & (f <= hi)
    p = (np.abs(X[sel]) ** 2).sum(axis=0) * 2.0 / len(x) ** 2
    return float(np.mean(p))


def room_noise(n: int, fs: int = 44100, noise_dbfs: float = -90.0, seed: int = 0, n_dirs: int = 40,
               sensor_db: float = -10.0, f_coherent: float = 6000.0,
               positions: np.ndarray | None = None) -> np.ndarray:
    """(n, 16) diffuse room noise, `noise_dbfs` = rms per channel in 800-4000 Hz (FS = 1.0)."""
    rng = np.random.default_rng(seed)
    pos = mic_positions() if positions is None else positions
    k = np.arange(n_dirs) + 0.5                                        # Fibonacci sphere
    el = np.degrees(np.arcsin(1 - 2 * k / n_dirs))
    az = np.degrees(np.pi * (1 + 5 ** 0.5) * k) % 360.0
    adv = unit_vector(az, el) @ pos.T / C_SOUND                       # (dirs, mics)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    lf, ld = np.log10([p[0] for p in ROOM_PSD]), np.array([p[1] for p in ROOM_PSD], float)
    amp = 10 ** (np.interp(np.log10(np.maximum(f, 1.0)), lf, ld) / 20)
    amp[0] = 0.0
    nb, nm = len(f), len(pos)
    X = (rng.standard_normal((nb, nm), np.float32) + 1j * rng.standard_normal((nb, nm), np.float32))
    X *= np.float32(10 ** (sensor_db / 20))
    k_coh = int(np.searchsorted(f, f_coherent))   # above: mics 42 mm apart are practically incoherent anyway
    ch = 8192
    adv32 = (2 * np.pi * adv).astype(np.float32)
    for s in range(0, k_coh, ch):
        e = min(s + ch, k_coh)
        N = (rng.standard_normal((e - s, n_dirs), np.float32) + 1j * rng.standard_normal((e - s, n_dirs), np.float32))
        E = np.exp(1j * (f[s:e, None, None].astype(np.float32) * adv32[None]))   # (bins, dirs, mics)
        X[s:e] += np.einsum("bd,bdm->bm", N, E) / np.float32(math.sqrt(n_dirs))
    X[k_coh:] += rng.standard_normal((nb - k_coh, nm), np.float32) + 1j * rng.standard_normal((nb - k_coh, nm), np.float32)
    X *= amp[:, None].astype(np.float32)
    x = np.fft.irfft(X, n=n, axis=0)
    return x * 10 ** (noise_dbfs / 20) / math.sqrt(_band_power(x, fs))


def render_moving(mono: np.ndarray, fs: int, paths, positions: np.ndarray | None = None,
                  block: int = 2048) -> np.ndarray:
    """(len(mono), 16): mono emitted from moving point sources. `paths(t)` -> [(pos_array_frame (3,), gain), ...]
    at time t (s); each path arrives with gain / r and delay r / c per mic (fractional, FFT domain). Hann-windowed
    overlap-add (hop block / 2); paths are evaluated at each block's centre. Path lengths must stay < ~15 m."""
    pos = mic_positions() if positions is None else positions
    hop = block // 2
    win = np.hanning(block + 1)[:-1]                                  # periodic Hann: sums to 1 at hop block/2
    nfft = 2 * block
    while nfft - block < 15.0 / C_SOUND * fs:
        nfft *= 2
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    x = np.concatenate([np.zeros(hop), mono, np.zeros(block)])
    out = np.zeros((len(x) + nfft, len(pos)))
    for s in range(0, len(x) - hop, hop):
        seg = x[s:s + block]
        if not np.any(seg):
            continue
        S = np.fft.rfft(seg * win[:len(seg)], nfft)
        tc = (s - hop + block / 2) / fs
        H = np.zeros((len(f), len(pos)), np.complex128)
        for p, g in paths(tc):
            r = np.linalg.norm(np.asarray(p)[None, :] - pos, axis=1)
            H += (g / r)[None, :] * np.exp(-2j * np.pi * f[:, None] * (r / C_SOUND)[None, :])
        out[s:s + nfft] += np.fft.irfft(S[:, None] * H, nfft, axis=0)
    return out[hop:hop + len(mono)]


@dataclass
class WalkConfig:
    az_from: float = 0.0          # compass deg
    az_to: float = 90.0
    rate: float = 1.0             # deg/s, the short way round
    pre_s: float = 5.0            # room noise only (the tracker learns its floor)
    hold_s: float = 10.0          # speaker on, standing at az_from
    hold_end_s: float = 20.0      # standing at az_to
    range_m: float = 2.5          # horizontal distance speaker - array centre
    height_m: float = 0.6         # speaker above the array plane
    snr_db: float = 20.0          # per channel, 800-4000 Hz, source (all paths) vs room noise
    noise_dbfs: float = -90.0     # room noise rms per channel in 800-4000 Hz (quiet clap sessions: -92..-88)
    reverb: bool = False
    north: float = 0.0            # Mounting.x_axis_bearing_deg
    mics_up: bool = True
    fs: int = 44100
    seed: int = 0
    # early reflections (with --reverb): amplitude relative to the direct path at the array
    table_gap_m: float = 0.015    # mics above the table top
    table_gain: float = 0.5       # -6 dB
    wall_compass: float = 270.0   # wall normal direction (compass) and distance from the array
    wall_dist_m: float = 3.0
    wall_gain: float = 0.35       # -9 dB
    ceiling_m: float = 1.8        # above the array plane (2.6 m room, table 0.8 m)
    ceiling_gain: float = 0.25    # -12 dB

    @property
    def span(self) -> float:
        return (self.az_to - self.az_from + 180.0) % 360.0 - 180.0

    @property
    def walk_s(self) -> float:
        return abs(self.span) / max(self.rate, 1e-9)

    @property
    def duration_s(self) -> float:
        return self.pre_s + self.hold_s + self.walk_s + self.hold_end_s

    def compass(self, t):
        k = np.clip((np.asarray(t, float) - self.pre_s - self.hold_s) / max(self.walk_s, 1e-9), 0.0, 1.0)
        return (self.az_from + k * self.span) % 360.0

    @property
    def el_deg(self) -> float:
        return math.degrees(math.atan2(self.height_m, self.range_m))

    def mounting(self) -> Mounting:
        return Mounting(self.north, self.mics_up)


def speaker_enu(cfg: WalkConfig, t: float) -> np.ndarray:
    a = math.radians(float(cfg.compass(t)))
    return np.array([cfg.range_m * math.sin(a), cfg.range_m * math.cos(a), cfg.height_m])


def enu_to_array(p: np.ndarray, m: Mounting) -> np.ndarray:
    """ENU point (array at the origin) -> array frame, via the compass <-> array-azimuth mapping of `Mounting`."""
    rh = math.hypot(p[0], p[1])
    a = math.radians(m.from_compass(math.degrees(math.atan2(p[0], p[1]))))
    return np.array([rh * math.cos(a), rh * math.sin(a), p[2] if m.mics_up else -p[2]])


def image_sources(cfg: WalkConfig, s: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """[(ENU position, reflection gain)]: direct path + (with reverb) table, wall and ceiling images."""
    out = [(s, 1.0)]
    if cfg.reverb:
        out.append((np.array([s[0], s[1], -2 * cfg.table_gap_m - s[2]]), cfg.table_gain))
        a = math.radians(cfg.wall_compass)
        nrm = np.array([math.sin(a), math.cos(a), 0.0])
        out.append((s - 2 * (s @ nrm - cfg.wall_dist_m) * nrm, cfg.wall_gain))
        out.append((np.array([s[0], s[1], 2 * cfg.ceiling_m - s[2]]), cfg.ceiling_gain))
    return out


def simulate_walk(cfg: WalkConfig, mono: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """(x (n, 16) float, FS = 1.0; truth rows every 0.1 s). `mono` is looped / cut to the source-on time."""
    fs = cfg.fs
    n = int(round(cfg.duration_s * fs))
    k_on = int(round(cfg.pre_s * fs))
    src = np.resize(np.asarray(mono, float), n - k_on)
    ramp = min(int(0.2 * fs), len(src))
    src[:ramp] *= np.linspace(0.0, 1.0, ramp)
    m = cfg.mounting()

    def paths(t):
        imgs = image_sources(cfg, speaker_enu(cfg, cfg.pre_s + t))
        r0 = np.linalg.norm(imgs[0][0])
        # reflection gains are relative to the direct path at the array (image geometry sets delay / direction)
        return [(enu_to_array(p, m), g * np.linalg.norm(p) / r0) for p, g in imgs]

    y = render_moving(src, fs, paths)
    noise = room_noise(n, fs, cfg.noise_dbfs, seed=cfg.seed + 101)
    p_noise = 10 ** (cfg.noise_dbfs / 10)
    y *= math.sqrt(p_noise * 10 ** (cfg.snr_db / 10) / _band_power(y, fs))
    x = noise
    x[k_on:] += y
    truth = [{"t": round(t, 3), "compass_az": round(float(cfg.compass(t)), 3), "el": round(cfg.el_deg, 3),
              "range_m": round(math.hypot(cfg.range_m, cfg.height_m), 3), "source_on": bool(t >= cfg.pre_s)}
             for t in np.arange(0.0, cfg.duration_s + 1e-9, 0.1)]
    return x, truth


def to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x * 32768.0), -32768, 32767).astype(np.int16)


def doa_check(x: np.ndarray, cfg: WalkConfig, win_s: float = 0.4, srp=None) -> np.ndarray:
    """SRP-PHAT (fs, nfft 1024, hop 256, 800-4000 Hz; floor learned on the noise-only lead-in) on win_s windows
    of the source-on part. Rows: (t_centre, true compass, estimated compass, error deg, est el)."""
    from .doa import DoaConfig, SrpPhat, angle_diff_deg

    s = srp or SrpPhat(DoaConfig(fs=cfg.fs, nfft=1024, hop=256, fmin=800, fmax=4000))
    k_on = int(round(cfg.pre_s * cfg.fs))
    s.learn_noise(x[:k_on])
    m = cfg.mounting()
    w = int(win_s * cfg.fs)
    rows = []
    for j in range(k_on + int(0.3 * cfg.fs), len(x) - w + 1, w):
        r = s.estimate(x[j:j + w])
        tc = (j + w / 2) / cfg.fs
        est = m.to_compass(r.array_az_deg)
        tru = float(cfg.compass(tc))
        rows.append((tc, tru, est, angle_diff_deg(est, tru), r.array_el_deg))
    return np.array(rows)
