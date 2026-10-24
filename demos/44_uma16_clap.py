"""Base UMA-16, stage version: clap -> bearing.json for the recon mission.

    python demos/44_uma16_clap.py --north 0                         # LIVE: array flat, mics up, +x axis -> north
    python demos/44_uma16_clap.py --north 0 --truth-az 135          # stage calibration: clap from a known bearing
    python demos/44_uma16_clap.py --file rec.wav --fast             # same on a 16-ch WAV (e.g. a session.wav)

Keep quiet for the first second (noise floor), then clap 2-3 times from point B. Every accepted clap prints its
compass bearing and rewrites `bearing.json` (BearingEstimate, label "clap") with the circular mean of the claps
of the last `--memory` seconds; the mission reads it before takeoff (`demo_recon.py --bearing ...`).

Stage calibration: in the sim, point B sits at some compass bearing from the base. Clap from the stage corner
that stands for B with `--truth-az <that bearing>`; the summary prints `suggested_north`, the --north that maps
this corner onto B. Re-run with that --north for the demo.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
live = importlib.import_module("40_uma16_live")  # sources + device lookup (sets SD_ENABLE_ASIO first)

import numpy as np  # noqa: E402

from sfhack.acoustic.array import Mounting  # noqa: E402
from sfhack.acoustic.clap import ClapBearing, ClapConfig  # noqa: E402
from sfhack.acoustic.detect import circ_mean_std_deg  # noqa: E402
from sfhack.acoustic.doa import angle_diff_deg  # noqa: E402
from sfhack.config import AcousticConfig, data_dir  # noqa: E402

BLOCK_S = 0.1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list input devices and exit")
    ap.add_argument("--file", type=Path, help="16-ch WAV instead of the live array")
    ap.add_argument("--fast", action="store_true", help="file: as fast as possible (default: real time)")
    ap.add_argument("--device", type=int, help="sounddevice index (default: auto, name contains 'UMA16')")
    ap.add_argument("--north", type=float, default=None, help="compass bearing of the array +x axis (deg)")
    ap.add_argument("--mics-down", action="store_true", help="board lies mics down (mirrored frame)")
    ap.add_argument("--array-enu", type=str, default=None, help="x,y,z of the array in the mission ENU frame")
    ap.add_argument("--out", type=Path, default=None, help="bearing.json path (default $SFHACK_DATA/acoustic/)")
    ap.add_argument("--truth-az", type=float, default=None, help="true compass bearing of the clapper")
    ap.add_argument("--memory", type=float, default=None, help="claps older than this (s) are forgotten")
    ap.add_argument("--threshold", type=float, default=None, help="onset ratio over the noise floor (default 8)")
    ap.add_argument("--max-s", type=float, default=None, help="stop after this many seconds of audio")
    ap.add_argument("--no-record", action="store_true", help="live: do not save the raw 16-ch audio")
    args = ap.parse_args()

    if args.list:
        live.list_devices()
        return
    acfg = AcousticConfig()
    mount = Mounting(acfg.x_axis_bearing_deg if args.north is None else args.north,
                     acfg.mics_up and not args.mics_down)
    array_enu = tuple(float(v) for v in args.array_enu.split(",")) if args.array_enu else None
    session = data_dir() / "acoustic" / datetime.now().strftime("clap_%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=True)
    out = args.out or data_dir() / "acoustic" / "bearing.json"

    if args.file:
        src = live.FileSource(args.file, BLOCK_S, realtime=not args.fast, loop=False)
    else:
        src = live.LiveSource(args.device, acfg.device_match, BLOCK_S,
                              None if args.no_record else session / "session.wav", acfg.fs)
    ccfg = ClapConfig(fs=src.fs)
    if args.memory is not None:
        ccfg.memory_s = args.memory
    if args.threshold is not None:
        ccfg.threshold_ratio = args.threshold
    det = ClapBearing(ccfg, mount, array_enu)
    print(f"session: {session}\nmounting: +x -> {mount.x_axis_bearing_deg:.1f} deg, mics {'up' if mount.mics_up else 'down'}"
          f"\nquiet for {ccfg.warmup_s:.0f} s, then clap from point B")

    errs, b = [], None
    log = open(session / "claps.jsonl", "w", encoding="utf-8")
    try:
        while True:
            x = src.read()
            if x is None:
                break
            t = src.t()
            clap = det.process(x, t)
            if clap is not None:
                b = det.bearing(t)
                b.save(out)
                b.save(session / "bearing.json")
                rec = {"t": round(clap.t, 3), "compass_az": round(clap.compass_az_deg, 1),
                       "array_az": round(clap.doa.array_az_deg, 1), "el": round(clap.el_deg, 1),
                       "peak": round(clap.doa.peak, 3), "rms": round(clap.rms, 5)}
                extra = ""
                if args.truth_az is not None:
                    e = angle_diff_deg(clap.compass_az_deg, args.truth_az)
                    errs.append(e)
                    rec["err_deg"] = round(e, 1)
                    extra = f"  err {e:+.1f}"
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"CLAP t={clap.t:6.2f}s  compass {clap.compass_az_deg:5.1f} deg  el {clap.el_deg:4.1f}"
                      f"  peak {clap.doa.peak:.2f}{extra}   -> bearing {b.az_deg:5.1f} +- {b.sigma_az_deg:.1f}"
                      f" ({len([k for k in det.claps if t - k.t <= ccfg.memory_s])} claps)")
            if args.max_s is not None and t >= args.max_s:
                break
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        log.close()
    summary = {"source": src.name, "fs": src.fs, "seconds": round(src.t(), 1), "claps": len(det.claps),
               "bearing": b.to_dict() if b else None, "bearing_json": str(out) if b else None,
               "north": mount.x_axis_bearing_deg}
    if errs:
        mean_err = angle_diff_deg(circ_mean_std_deg(np.array(errs) % 360.0)[0], 0.0)
        summary.update(truth_az=args.truth_az, mean_err_deg=round(mean_err, 1),
                       rms_err_deg=round(float(np.sqrt(np.mean(np.square(errs)))), 1))
        # compass = north -/+ array az: shifting --north shifts every bearing by the same amount
        summary["suggested_north"] = round((mount.x_axis_bearing_deg - mean_err) % 360.0, 1)
    (session / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
