"""Single-clap accuracy bench for `ClapBearing`: every DOA variant on the same audio, no clapping needed.

    python demos/45_clap_bench.py                      # synthetic room + every session.wav under $SFHACK_DATA/acoustic
    python demos/45_clap_bench.py --synthetic-only
    python demos/45_clap_bench.py --sessions clap_20261024_211923

Synthetic: claps from 8 directions, each with a direct path and 4 echoes 1.5-20 ms later from random directions
(laptop, walls, the cardboard), at the mics-down level (direct ~4x the room). Truth is known, so it reports the
single-clap error.

Real: each session.wav runs through the live settings (mics down thresholds, --max-el 38). A `truth.json` in the
session folder (`{"segments": [[t0, t1, array_az_deg], ...]}`) gives the error; without one, all claps of a session
are assumed to come from one spot and the spread around their median is reported.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import scipy.io.wavfile as wavfile  # noqa: E402

from sfhack.acoustic.array import Mounting  # noqa: E402
from sfhack.acoustic.clap import ClapBearing, ClapConfig  # noqa: E402
from sfhack.acoustic.doa import angle_diff_deg  # noqa: E402
from sfhack.acoustic.synth import clap_scene  # noqa: E402
from sfhack.config import data_dir  # noqa: E402

FS = 44100
BLOCK_S = 0.1
MAX_EL = 38.0
VARIANTS = {
    "peak50": dict(direct_s=0.0),
    "direct4": dict(direct_s=0.004),
    "direct8": dict(direct_s=0.008),
}


def make_cfg(fs: int, **kw) -> ClapConfig:
    c = ClapConfig(fs=fs, threshold_ratio=3.0, min_rms=1.5e-4, min_peak=0.2)   # mics-down live settings
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def run(x: np.ndarray, fs: int, cfg: ClapConfig) -> list:
    det = ClapBearing(cfg, Mounting(0.0, True))
    n = int(BLOCK_S * fs)
    out = []
    for i in range(0, len(x) - n + 1, n):
        k = det.process(x[i:i + n], (i + n) / fs)
        if k is not None and k.el_deg <= MAX_EL:
            out.append(k)
    return out


def score_truth(claps, truth, tol_s=0.3):
    errs, missed, extra = [], 0, 0
    used = set()
    for t0, az in truth:
        m = [k for k in claps if abs(k.t - t0) < tol_s]
        if not m:
            missed += 1
            continue
        used.add(id(m[0]))
        errs.append(angle_diff_deg(m[0].doa.array_az_deg, az))
    extra = len([k for k in claps if id(k) not in used])
    return np.abs(errs), missed, extra


def fmt(errs: np.ndarray) -> str:
    if len(errs) == 0:
        return "no claps"
    return (f"median {np.median(errs):5.1f}  p90 {np.percentile(errs, 90):5.1f}  max {errs.max():5.1f}  "
            f"<5deg {np.mean(errs < 5) * 100:3.0f}%")


def load(path: Path) -> tuple[np.ndarray, int]:
    fs, x = wavfile.read(path)
    scale = {np.dtype("int16"): 32768.0, np.dtype("int32"): 2.0 ** 31}.get(x.dtype, 1.0)
    return x[:, :16].astype(np.float64) / scale, int(fs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic-only", action="store_true")
    ap.add_argument("--real-only", action="store_true")
    ap.add_argument("--sessions", nargs="*", help="session folder names (default: all with a session.wav)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    args = ap.parse_args()

    if not args.real_only:
        for level in (0.005, 0.03):   # band RMS over a 0.1 s block ~5x / ~30x the room
            print(f"\n== synthetic, 4 echoes per clap, burst level {level} ({args.seeds} x 24 claps)")
            scenes = [clap_scene(s, level, FS) for s in range(args.seeds)]
            for name in args.variants:
                errs, missed, extra = [], 0, 0
                for x, truth in scenes:
                    e, m, xtra = score_truth(run(x, FS, make_cfg(FS, **VARIANTS[name])), truth)
                    errs.append(e)
                    missed, extra = missed + m, extra + xtra
                print(f"  {name:9s} {fmt(np.concatenate(errs))}   missed {missed}  extra {extra}")

    if args.synthetic_only:
        return
    root = data_dir() / "acoustic"
    dirs = [root / s for s in args.sessions] if args.sessions else sorted(root.glob("clap_*"))
    for d in dirs:
        wav = d / "session.wav"
        if not wav.exists() or wav.stat().st_size < 1000:
            continue
        x, fs = load(wav)
        tf = d / "truth.json"
        truth = json.loads(tf.read_text())["segments"] if tf.exists() else None
        print(f"\n== {d.name}  {len(x) / fs:.0f} s @ {fs} Hz" + ("  (truth.json)" if truth else ""))
        for name in args.variants:
            claps = run(x, fs, make_cfg(fs, **VARIANTS[name]))
            if truth:
                lab = [(k, az) for k in claps for t0, t1, az in truth if t0 <= k.t <= t1]
                errs = np.abs([angle_diff_deg(k.doa.array_az_deg, az) for k, az in lab])
                print(f"  {name:9s} {len(claps):2d} claps  {fmt(errs)}  unlabelled {len(claps) - len(lab)}")
            elif claps:
                a = np.array([k.doa.array_az_deg for k in claps])
                dev = np.abs((a[:, None] - a[None, :] + 180.0) % 360.0 - 180.0)
                med = a[np.argmin(dev.sum(axis=1))]
                d_med = np.abs((a - med + 180.0) % 360.0 - 180.0)
                print(f"  {name:9s} {len(claps):2d} claps  median az {med:5.1f}  spread: "
                      f"median {np.median(d_med):4.1f} max {d_med.max():4.1f}   "
                      f"az {' '.join(f'{v:.0f}' for v in a)}  el {' '.join(f'{k.el_deg:.0f}' for k in claps)}")
            else:
                print(f"  {name:9s}  0 claps")


if __name__ == "__main__":
    main()
