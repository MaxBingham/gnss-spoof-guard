"""Drone sound for the Bluetooth speaker (counter-drone demo): stereo 44.1 kHz 16-bit WAV, seamless loop.

    python scripts/make_drone_sound.py                    # -> $SFHACK_DATA/acoustic/drone_sound.wav, 180 s
    python scripts/make_drone_sound.py --tone-test        # -> drone_tone_test.wav: pink-ish noise 800-4000 Hz only

Small quadcopter: 4 rotors, blade-pass 160-240 Hz drifting +-3 % independently, harmonics, half-order shaft lines,
motor whine, blade-modulated turbulence peaking at ~2 kHz, with --hf-frac of the energy in 1-4 kHz (the band the
UMA-16 DOA uses). Tones have whole cycles over the file and the noise is circular, so the loop is sample-exact;
the --fade-ms ramps at both ends only avoid a click on players that put a gap between repeats (0 = gapless).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile

from sfhack.acoustic.walk_sim import band_fraction, band_noise, drone_signal
from sfhack.config import data_dir

BANDS = ((20, 150), (150, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 22050))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--fs", type=int, default=44100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hf-frac", type=float, default=0.55, help="energy fraction in 1-4 kHz")
    ap.add_argument("--tone-test", action="store_true", help="pink-ish noise 800-4000 Hz instead of the drone")
    ap.add_argument("--peak-db", type=float, default=-3.0, help="peak level, dBFS")
    ap.add_argument("--fade-ms", type=float, default=20.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.tone_test:
        x = band_noise(a.seconds, a.fs, a.seed)
        name = "drone_tone_test.wav"
    else:
        x = drone_signal(a.seconds, a.fs, a.seed, hf_frac=a.hf_frac)
        name = "drone_sound.wav"
    seam = abs(x[0] - x[-1]) / np.mean(np.abs(np.diff(x)))
    nf = int(a.fade_ms * 1e-3 * a.fs)
    if nf > 0:
        ramp = np.sin(np.linspace(0, np.pi / 2, nf)) ** 2
        x[:nf] *= ramp
        x[-nf:] *= ramp[::-1]
    x *= 10 ** (a.peak_db / 20) / np.max(np.abs(x))
    pcm = np.clip(np.rint(x * 32767), -32767, 32767).astype(np.int16)
    out = Path(a.out) if a.out else data_dir() / "acoustic" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(out, a.fs, np.stack([pcm, pcm], axis=1))

    P = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / a.fs)
    tot = P[1:].sum()
    print(f"wrote {out}  ({a.seconds:.0f} s, stereo, {a.fs} Hz, int16, peak {20 * np.log10(np.abs(pcm).max() / 32768):.1f} "
          f"dBFS, rms {20 * np.log10(np.sqrt(np.mean(x ** 2))):.1f} dBFS)")
    print(f"energy in 1-4 kHz: {100 * band_fraction(x, a.fs):.0f} %, in 800-4000 Hz (DOA band): "
          f"{100 * band_fraction(x, a.fs, 800, 4000):.0f} %")
    print("band energy: " + ", ".join(f"{lo}-{hi} Hz {100 * P[(f >= lo) & (f < hi)].sum() / tot:.0f} %" for lo, hi in BANDS))
    print(f"loop seam jump (before fades): {seam:.2f} x the mean sample step")


if __name__ == "__main__":
    main()
