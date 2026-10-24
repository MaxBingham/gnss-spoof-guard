"""Synthesize what the UMA-16 would record of the drone sound from a speaker walked around it (16-ch WAV + truth).

    python scripts/sim_walk_audio.py                           # 0 -> 90 deg at 1 deg/s, 2.5 m, speaker 0.6 m up
    python scripts/sim_walk_audio.py --reverb --snr-db 5 --check
    python scripts/sim_walk_audio.py --north 92.7 --mics-down  # the real mounting (acoustic/mount.json)

Timeline: --pre s of room noise only (the tracker learns its floor), then the speaker at --from for --hold s,
walking to --to at --rate deg/s, then --hold-end s. Output (default $SFHACK_DATA/acoustic/sim_walk/<name>.wav):
16 ch in USB channel order (`mic_positions()`), 44.1 kHz int16 at the real room's absolute level, plus
<name>.truth.jsonl (t, compass_az, el, range_m, source_on every 0.1 s). --check runs SRP-PHAT on 0.4 s windows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile

from sfhack.acoustic.walk_sim import WalkConfig, doa_check, drone_signal, band_noise, simulate_walk, to_int16
from sfhack.config import data_dir


def main() -> None:
    d = WalkConfig()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="az_from", type=float, default=d.az_from, help="compass deg")
    ap.add_argument("--to", dest="az_to", type=float, default=d.az_to)
    ap.add_argument("--rate", type=float, default=d.rate, help="deg/s")
    ap.add_argument("--pre", type=float, default=d.pre_s, help="noise-only lead-in (s)")
    ap.add_argument("--hold", type=float, default=d.hold_s)
    ap.add_argument("--hold-end", type=float, default=d.hold_end_s)
    ap.add_argument("--range", dest="range_m", type=float, default=d.range_m, help="horizontal range (m)")
    ap.add_argument("--height", type=float, default=d.height_m, help="speaker above the array plane (m)")
    ap.add_argument("--snr-db", type=float, default=d.snr_db, help="per channel, 800-4000 Hz")
    ap.add_argument("--noise-dbfs", type=float, default=d.noise_dbfs, help="room noise per channel, 800-4000 Hz")
    ap.add_argument("--reverb", action="store_true", help="table, wall (3 m, compass 270) and ceiling reflections")
    ap.add_argument("--north", type=float, default=d.north, help="compass bearing of the array +x axis")
    ap.add_argument("--mics-down", action="store_true")
    ap.add_argument("--sound", default=None, help="mono/stereo WAV (default: $SFHACK_DATA/acoustic/drone_sound.wav, "
                    "synthesized if missing)")
    ap.add_argument("--tone-test", action="store_true", help="use 800-4000 Hz pink-ish noise as the source")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true", help="SRP-PHAT bearings vs truth")
    a = ap.parse_args()

    cfg = WalkConfig(az_from=a.az_from, az_to=a.az_to, rate=a.rate, pre_s=a.pre, hold_s=a.hold, hold_end_s=a.hold_end,
                     range_m=a.range_m, height_m=a.height, snr_db=a.snr_db, noise_dbfs=a.noise_dbfs, reverb=a.reverb,
                     north=a.north, mics_up=not a.mics_down, seed=a.seed)
    src_s = cfg.duration_s - cfg.pre_s
    if a.tone_test:
        mono = band_noise(src_s, cfg.fs, a.seed)
    else:
        p = Path(a.sound) if a.sound else data_dir() / "acoustic" / "drone_sound.wav"
        if p.exists():
            fs, s = wavfile.read(p)
            assert fs == cfg.fs, f"{p}: {fs} Hz, need {cfg.fs}"
            mono = s.astype(float).mean(axis=1) if s.ndim == 2 else s.astype(float)
        else:
            print(f"{p} missing: synthesizing the drone sound")
            mono = drone_signal(max(src_s, 60.0), cfg.fs, a.seed)
    x, truth = simulate_walk(cfg, mono)

    if a.out:
        out = Path(a.out)
    else:
        tag = (f"walk_{a.az_from:g}_{a.az_to:g}_r{a.range_m:g}_snr{a.snr_db:g}" + ("_reverb" if a.reverb else "")
               + ("_tone" if a.tone_test else "") + ("" if cfg.mics_up and cfg.north == 0 else f"_n{a.north:g}"
                                                       + ("down" if a.mics_down else "up")))
        out = data_dir() / "acoustic" / "sim_walk" / f"{tag}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    pcm = to_int16(x)
    wavfile.write(out, cfg.fs, pcm)
    tp = out.with_suffix(".truth.jsonl")
    tp.write_text("".join(json.dumps(r) + "\n" for r in truth), encoding="utf-8")
    print(f"wrote {out} ({cfg.duration_s:.0f} s, 16 ch, {cfg.fs} Hz, int16, peak "
          f"{20 * np.log10(max(np.abs(pcm).max(), 1) / 32768):.1f} dBFS) and {tp.name}")
    print(f"walk {cfg.az_from:g} -> {cfg.az_to:g} deg compass at {cfg.rate:g} deg/s, range {cfg.range_m:g} m, "
          f"el {cfg.el_deg:.1f} deg, source on at {cfg.pre_s:g} s, SNR {cfg.snr_db:g} dB, reverb {cfg.reverb}, "
          f"mounting north {cfg.north:g} {'mics up' if cfg.mics_up else 'mics down'}")
    if a.check:
        r = doa_check(pcm.astype(float) / 32768.0, cfg)
        e = np.abs(r[:, 3])
        print(f"SRP-PHAT on {len(r)} x 0.4 s windows: error rms {np.sqrt(np.mean(e ** 2)):.2f} deg, median "
              f"{np.median(e):.2f}, p95 {np.percentile(e, 95):.2f}, max {e.max():.2f}; mean signed "
              f"{np.mean(r[:, 3]):+.2f}; est el median {np.median(r[:, 4]):.1f} (true {cfg.el_deg:.1f})")


if __name__ == "__main__":
    main()
