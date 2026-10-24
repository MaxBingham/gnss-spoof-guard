"""Direction of arrival for the UMA-16: SRP-PHAT over a hemisphere grid (own numpy, no acoular needed).

For each of the 120 mic pairs the GCC-PHAT cross-correlation is averaged over the STFT frames of a window and
upsampled; the steered response of a direction is the sum over pairs of the GCC at that direction's TDOA.
Bins are weighted by how far they stick out of the learned noise floor (`noise_psd`), so a tone or engine
harmonics in broadband noise still dominate the map.

A planar array cannot tell above the board from below (TDOAs depend only on the in-plane components of the
direction), so the grid is the upper hemisphere: array az 0-360 (CCW from +x), el 0-90 above the board.
Near the horizon, elevation is poorly resolved (the in-plane component changes with cos(el)).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from .array import C_SOUND, FS, mic_positions, unit_vector


@dataclass
class DoaConfig:
    fs: int = FS
    nfft: int = 2048              # STFT frame (43 ms at 48 kHz)
    hop: int = 1024
    fmin: float = 300.0           # DOA band (Hz). Below ~300 Hz the 126 mm aperture hardly resolves anything;
    fmax: float = 4000.0          # above ~4.1 kHz the 42 mm pitch aliases.
    upsample: int = 8             # GCC interpolation
    az_step: float = 1.0
    el_step: float = 2.0
    el_max: float = 90.0


@dataclass
class DoaResult:
    array_az_deg: float           # CCW from array +x
    array_el_deg: float           # above the board
    peak: float                   # normalized SRP peak (0..1, 1 = all pairs agree perfectly)
    peak_ratio: float             # peak / median of the map (sharpness)
    second_dB: float              # second local max relative to the peak (dB, <= 0; -inf if none)
    srp: np.ndarray               # (n_el, n_az) map
    az_grid: np.ndarray
    el_grid: np.ndarray


class SrpPhat:
    def __init__(self, cfg: DoaConfig | None = None, positions: np.ndarray | None = None):
        self.cfg = cfg or DoaConfig()
        c = self.cfg
        self.pos = mic_positions() if positions is None else positions
        n = len(self.pos)
        self.pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)])
        self.freqs = np.fft.rfftfreq(c.nfft, 1.0 / c.fs)
        self.band = (self.freqs >= c.fmin) & (self.freqs <= c.fmax)
        self.window = np.hanning(c.nfft)
        self.az_grid = np.arange(0.0, 360.0, c.az_step)
        self.el_grid = np.arange(0.0, c.el_max + 1e-9, c.el_step)
        el, az = np.meshgrid(self.el_grid, self.az_grid, indexing="ij")
        u = unit_vector(az, el)                                      # (n_el, n_az, 3)
        dp = self.pos[self.pairs[:, 0]] - self.pos[self.pairs[:, 1]]  # (n_pairs, 3)
        # pair (i, j): x_i lags x_j by tau = t_i - t_j = -(p_i - p_j) . u / c
        tau = -np.einsum("pk,eak->eap", dp, u) / C_SOUND             # (n_el, n_az, n_pairs)
        self.n_up = c.nfft * c.upsample
        self.max_lag = int(np.ceil(np.abs(dp).sum(1).max() / C_SOUND * c.fs * c.upsample)) + 2
        self.lag_idx = np.rint(tau * c.fs * c.upsample).astype(int) + self.max_lag
        self.noise_psd: np.ndarray | None = None                    # (n_bins,) mean over channels

    # -- spectra --------------------------------------------------------------------------------------
    def stft(self, x: np.ndarray) -> np.ndarray:
        """x (n_samples, 16) -> (n_frames, n_bins, 16)."""
        c = self.cfg
        if len(x) < c.nfft:
            x = np.pad(x, ((0, c.nfft - len(x)), (0, 0)))
        starts = range(0, len(x) - c.nfft + 1, c.hop)
        frames = np.stack([x[s:s + c.nfft] for s in starts]) * self.window[None, :, None]
        return np.fft.rfft(frames, axis=1)

    def learn_noise(self, x: np.ndarray) -> None:
        """Noise floor per bin from a stretch without the source (median over frames, robust to bursts)."""
        X = self.stft(x)
        self.noise_psd = np.median((np.abs(X) ** 2).mean(axis=2), axis=0)

    def bin_weights(self, X: np.ndarray, noise_scale: float = 1.0) -> np.ndarray:
        p = (np.abs(X) ** 2).mean(axis=(0, 2))
        w = self.band.astype(float)
        # Without a learned floor: prominence over a running median across frequency (keeps tones / harmonics).
        floor = self.noise_psd * noise_scale if self.noise_psd is not None else median_filter(p, 63)
        w *= np.clip(1.0 - 2.0 * floor / np.maximum(p, 1e-30), 0.0, 1.0)
        if w.sum() < 1e-6:          # nothing above the floor: plain PHAT over the band
            w = self.band.astype(float)
        return w

    # -- DOA ------------------------------------------------------------------------------------------
    def gcc(self, X: np.ndarray, noise_scale: float = 1.0) -> np.ndarray:
        """Averaged, upsampled GCC-PHAT per pair: (n_pairs, 2 * max_lag + 1)."""
        w = self.bin_weights(X, noise_scale)
        i, j = self.pairs[:, 0], self.pairs[:, 1]
        cross = X[:, :, i] * np.conj(X[:, :, j])                     # (frames, bins, pairs)
        cross = (cross / np.maximum(np.abs(cross), 1e-30)).mean(axis=0) * w[:, None]
        r = np.fft.irfft(cross, n=self.n_up, axis=0) * self.cfg.upsample
        r = np.concatenate([r[-self.max_lag:], r[:self.max_lag + 1]], axis=0)  # lags -max..+max
        return r.T / max(w.sum(), 1e-12) * (self.cfg.nfft / 2)

    def estimate(self, x: np.ndarray, noise_scale: float = 1.0) -> DoaResult:
        """`noise_scale` < 1 when x is a short burst zero-padded into a frame (less noise energy per bin)."""
        X = self.stft(x)
        g = self.gcc(X, noise_scale)
        n_pairs = len(self.pairs)
        srp = g[np.arange(n_pairs)[None, None, :], self.lag_idx].sum(axis=2) / n_pairs  # (n_el, n_az)
        k = np.unravel_index(np.argmax(srp), srp.shape)
        az, el = self.az_grid[k[1]], self.el_grid[k[0]]
        az, el = self._refine(srp, k)
        peak = float(srp[k])
        med = float(np.median(srp))
        return DoaResult(az, el, peak, peak / max(abs(med), 1e-9), _second_peak_db(srp, k),
                         srp, self.az_grid, self.el_grid)

    def _refine(self, srp: np.ndarray, k) -> tuple[float, float]:
        """Parabolic sub-grid interpolation (az wraps; el clipped)."""
        ie, ia = k
        na = srp.shape[1]
        l, m, r = srp[ie, (ia - 1) % na], srp[ie, ia], srp[ie, (ia + 1) % na]
        d = 0.5 * (l - r) / (l - 2 * m + r) if (l - 2 * m + r) < 0 else 0.0
        az = (self.az_grid[ia] + float(np.clip(d, -0.5, 0.5)) * self.cfg.az_step) % 360.0
        el = self.el_grid[ie]
        if 0 < ie < srp.shape[0] - 1:
            l, r = srp[ie - 1, ia], srp[ie + 1, ia]
            d = 0.5 * (l - r) / (l - 2 * m + r) if (l - 2 * m + r) < 0 else 0.0
            el = el + float(np.clip(d, -0.5, 0.5)) * self.cfg.el_step
        return float(az), float(el)


def _second_peak_db(srp: np.ndarray, k) -> float:
    """Highest other local max in azimuth (along the el row of the peak, wrapping) at least 30 deg away."""
    row = srp[k[0]]
    na = len(row)
    is_max = (row > np.roll(row, 1)) & (row >= np.roll(row, -1))
    da = np.abs(((np.arange(na) - k[1]) * 360.0 / na + 180.0) % 360.0 - 180.0)
    cand = row[is_max & (da > 30.0)]
    if len(cand) == 0 or row[k[1]] <= 0:
        return float("-inf")
    return float(10 * np.log10(max(cand.max(), 1e-12) / row[k[1]]))


def angle_diff_deg(a: float, b: float) -> float:
    """Signed smallest difference a - b in degrees, in (-180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0
