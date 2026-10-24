"""Synthesize UMA-16 recordings: a far-field source at a known direction, with the real channel geometry.

Used for tests with known truth, for the `--file` fallback demo (a 16-ch "vehicle" WAV), and available to
recon-world's simulated array (stretch goal: run the same DOA code as the live path).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal

from .array import C_SOUND, FS, mic_positions, unit_vector


def engine_signal(duration_s: float, fs: int = FS, rpm: float = 1800.0, cylinders: int = 4,
                  seed: int = 0) -> np.ndarray:
    """Engine-like mono signal: firing-frequency harmonics with slow rpm wander + band-limited rumble/hiss."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    f0 = rpm / 60.0 * cylinders / 2.0                     # firing frequency (60 Hz at 1800 rpm, 4 cyl)
    wander = 1.0 + 0.03 * np.sin(2 * np.pi * 0.3 * t + rng.uniform(0, 6.28))
    phase = 2 * np.pi * np.cumsum(f0 * wander) / fs
    x = np.zeros(n)
    for h in range(1, 40):
        if h * f0 > 6000:
            break
        x += (1.0 / h ** 0.7) * np.sin(h * phase + rng.uniform(0, 6.28))
    noise = rng.standard_normal(n)
    b, a = signal.butter(2, [80, 5000], btype="band", fs=fs)
    x += 0.8 * signal.lfilter(b, a, noise) * np.std(x) / max(np.std(signal.lfilter(b, a, noise)), 1e-12)
    return x / np.max(np.abs(x))


def plane_wave(mono: np.ndarray, array_az_deg: float, array_el_deg: float, fs: int = FS,
               positions: np.ndarray | None = None) -> np.ndarray:
    """(n, 16): mono arriving from (az, el) in the array frame, exact fractional delays (FFT domain)."""
    pos = mic_positions() if positions is None else positions
    u = unit_vector(array_az_deg, array_el_deg)
    advance = pos @ u / C_SOUND                            # mic closer to the source hears it earlier
    pad = 256
    n = len(mono) + 2 * pad
    M = np.fft.rfft(np.pad(mono, pad), n=n)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    Y = M[:, None] * np.exp(2j * np.pi * f[:, None] * advance[None, :])
    return np.fft.irfft(Y, n=n, axis=0)[pad:pad + len(mono)]


def array_recording(array_az_deg: float, array_el_deg: float, duration_s: float = 10.0, snr_db: float = 10.0,
                    onset_s: float = 0.0, fs: int = FS, seed: int = 0, mono: np.ndarray | None = None,
                    noise: np.ndarray | None = None) -> np.ndarray:
    """Source from `onset_s` on (silence before, to learn the noise floor) + independent sensor noise per channel
    (or a given (n, 16) background, e.g. a real UMA-16 recording) at `snr_db` (source band power / noise)."""
    rng = np.random.default_rng(seed + 1)
    n = int(duration_s * fs)
    if mono is None:
        mono = engine_signal(duration_s - onset_s, fs, seed=seed)
    src = np.zeros((n, 16))
    k = int(onset_s * fs)
    src[k:k + len(mono)] = plane_wave(mono, array_az_deg, array_el_deg, fs)[: n - k]
    if noise is None:
        noise = rng.standard_normal((n, 16))
    else:
        noise = np.resize(noise, (n, 16)).astype(float)
    p_src = np.mean(src[k:] ** 2)
    p_noise = np.mean(noise ** 2)
    noise *= np.sqrt(p_src / p_noise / 10 ** (snr_db / 10))
    return src + noise


def write_wav(path: str | Path, x: np.ndarray, fs: int = FS) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, fs, (x / np.max(np.abs(x)) * 0.5 * 32767).astype(np.int16))
    return path
