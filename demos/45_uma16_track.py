"""Base UMA-16, moving-source version: continuous compass bearing of a drone sound (speaker walked around the
array) at ~5 Hz -> console, track.jsonl, UDP to the counter-drone sim (demos/51_counter_drone.py, port 5005).

    python demos/45_uma16_track.py --udp 127.0.0.1                    # LIVE, mounting from mount.json if present
    python demos/45_uma16_track.py --north 90 --udp 192.168.1.20:5005 # ignore mount.json: +x axis points east
    python demos/45_uma16_track.py --calibrate --truth-az 0           # hold the speaker at compass 0 -> mount.json
    python demos/46_uma16_calibrate.py                                # several marks: north + handedness
    python demos/45_uma16_track.py --file session.wav --fast          # replay a session

Keep quiet for the first 5 s (noise floor). Then play the drone sound and walk slowly around the array.
Analysis every 0.2 s on the last 0.4 s (`sfhack/acoustic/track.py`): detection = 3rd-largest 1/3-octave band
excess (500-5000 Hz) over the floor >= --detect-db, SRP-PHAT in 800-4000 Hz, circular alpha-beta track.
UDP per estimate: BearingEstimate.to_dict() (label "drone") + az_rate_dps, level_db, detected=true, seq;
ONE {"detected": false, "seq", "t"} when the source is lost.
"""
from __future__ import annotations

import argparse
import importlib
import json
import signal
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
live = importlib.import_module("40_uma16_live")  # sources + device lookup (sets SD_ENABLE_ASIO first)

import numpy as np  # noqa: E402

from sfhack.acoustic.array import Mounting  # noqa: E402
from sfhack.acoustic.detect import circ_mean_std_deg  # noqa: E402
from sfhack.acoustic.doa import angle_diff_deg  # noqa: E402
from sfhack.acoustic.track import TrackConfig, TrackPipeline  # noqa: E402
from sfhack.config import AcousticConfig, data_dir  # noqa: E402

BLOCK_S = 0.1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list input devices and exit")
    ap.add_argument("--file", type=Path, help="16-ch WAV instead of the live array")
    ap.add_argument("--fast", action="store_true", help="file: as fast as possible (default: real time)")
    ap.add_argument("--device", type=int, help="sounddevice index (default: auto, name contains 'UMA16')")
    ap.add_argument("--north", type=float, default=None, help="compass bearing of the array +x axis (deg, default 0)")
    ap.add_argument("--mount", type=Path, nargs="?", default=None, const=data_dir() / "acoustic" / "mount.json",
                    help="load the mounting from mount.json (no path: $SFHACK_DATA/acoustic/mount.json, "
                         "written by --calibrate)")
    ap.add_argument("--mics-down", action="store_true", help="board lies mics down (mirrored frame)")
    ap.add_argument("--array-enu", type=str, default=None, help="x,y,z of the array in the mission ENU frame")
    ap.add_argument("--detect-db", type=float, default=None, help="detection threshold over the floor (dB, default 6)")
    ap.add_argument("--learn", type=float, default=None, help="noise-floor learning time (s, default 5)")
    ap.add_argument("--max-el", type=float, default=None,
                    help="ignore windows from higher than this (deg); typing at the laptop arrives at 40-65")
    ap.add_argument("--doa-band", type=str, default=None, help="fmin,fmax of the DOA band (Hz, default 800,4000)")
    ap.add_argument("--udp", type=str, default=None, help="host[:port] of the sim (demos/51_counter_drone.py, 5005)")
    ap.add_argument("--truth-az", type=float, default=None, help="true compass bearing of a static speaker")
    ap.add_argument("--calibrate", action="store_true",
                    help="speaker held at --truth-az (default 0): save the mounting to mount.json")
    ap.add_argument("--max-s", type=float, default=None, help="stop after this many seconds of audio")
    ap.add_argument("--no-record", action="store_true", help="live: do not save the raw 16-ch audio")
    ap.add_argument("--quiet", action="store_true", help="print only estimates / lost, not every window")
    args = ap.parse_args()

    if args.list:
        live.list_devices()
        return
    acfg = AcousticConfig()
    default_mount = data_dir() / "acoustic" / "mount.json"
    if args.mount is None and args.north is None and default_mount.exists():
        args.mount = default_mount
    if args.calibrate:
        args.truth_az = 0.0 if args.truth_az is None else args.truth_az
        # keep the handedness from mount.json (demos/46 fits it); only north is re-measured here
        up = (Mounting(**json.loads(default_mount.read_text(encoding="utf-8"))).mics_up
              if default_mount.exists() and not args.mics_down else not args.mics_down)
        mount = Mounting(0.0 if args.north is None else args.north, up)
    elif args.mount is not None:
        if not args.mount.exists():
            raise SystemExit(f"{args.mount} not found (run --calibrate or demos/46_uma16_calibrate.py first)")
        mount = Mounting(**json.loads(args.mount.read_text(encoding="utf-8")))
        print(f"mounting from {args.mount}")
    else:
        mount = Mounting(0.0 if args.north is None else args.north, not args.mics_down)
    array_enu = tuple(float(v) for v in args.array_enu.split(",")) if args.array_enu else None
    session = data_dir() / "acoustic" / datetime.now().strftime("track_%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=True)

    if args.file:
        src = live.FileSource(args.file, BLOCK_S, realtime=not args.fast, loop=False)
    else:
        src = live.LiveSource(args.device, acfg.device_match, BLOCK_S,
                              None if args.no_record else session / "session.wav", acfg.fs)
    tcfg = TrackConfig(fs=src.fs, max_el_deg=args.max_el)
    if args.detect_db is not None:
        tcfg.detect_db = args.detect_db
    if args.learn is not None:
        tcfg.learn_s = args.learn
    if args.doa_band:
        tcfg.doa_fmin, tcfg.doa_fmax = (float(v) for v in args.doa_band.split(","))
    pipe = TrackPipeline(tcfg, mount, array_enu)
    udp = None
    if args.udp:
        host, _, port = args.udp.partition(":")
        udp = (socket.socket(socket.AF_INET, socket.SOCK_DGRAM), (host, int(port or 5005)))
        print(f"bearings -> UDP {udp[1][0]}:{udp[1][1]}")
    (session / "config.json").write_text(json.dumps({**tcfg.__dict__, "source": src.name, "north":
                                                     mount.x_axis_bearing_deg, "mics_up": mount.mics_up}, indent=2))
    print(f"session: {session}\nmounting: +x -> {mount.x_axis_bearing_deg:.1f} deg, mics "
          f"{'up' if mount.mics_up else 'down'}, fs {src.fs} Hz\n"
          f">>> KEEP QUIET for {tcfg.learn_s:.0f} s: learning the noise floor <<<")

    signal.signal(signal.SIGTERM, signal.default_int_handler)   # systemctl stop: same clean exit as Ctrl-C
    log = open(session / "track.jsonl", "w", encoding="utf-8")
    errs, levels, n_est, n_lost, n_hit, n_win, last = [], [], 0, 0, 0, 0, None
    try:
        while True:
            x = src.read()
            if x is None:
                break
            t = src.t()
            s = pipe.process(x, t, t_unix=None if args.file else time.time())
            if s is None:
                continue
            if s.learning:
                print(f"t={t:6.1f}s  noise floor learned - play the drone sound now")
                continue
            n_win += 1
            n_hit += s.hit
            levels.append(s.level_db)
            m = s.message
            rec = {"t_audio": round(t, 2), "level_db": round(s.level_db, 1), "hit": s.hit, "on": s.detected}
            if s.doa is not None:
                rec.update(raw_az=round(s.compass_az, 1), raw_el=round(s.doa.array_el_deg, 1),
                           peak=round(s.doa.peak, 3))
            if s.state is not None:
                rec["accepted"] = s.state.accepted
            if m is not None:
                rec["msg"] = m
                if udp is not None:
                    udp[0].sendto(json.dumps(m).encode("utf-8"), udp[1])
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if m is not None and m["detected"]:
                n_est += 1
                last = m
                extra = ""
                if args.truth_az is not None:
                    e = angle_diff_deg(m["az_deg"], args.truth_az)
                    errs.append(e)
                    extra = f"  err {e:+5.1f}"
                tag = {True: "", False: " OUTLIER", None: " coast"}[s.state.accepted] + (f" {s.note}" if s.note else "")
                raw = f" raw {s.compass_az:5.1f}/{s.doa.array_el_deg:2.0f}" if s.doa is not None else ""
                print(f"t={t:6.1f}s #{m['seq']:<5d} az {m['az_deg']:5.1f} +-{m['sigma_az_deg']:4.1f}"
                      f"  {m['az_rate_dps']:+5.1f} dps  el {m['el_deg']:3.0f}  lvl {s.level_db:+5.1f}"
                      f"  conf {m['confidence']:.2f}{raw}{extra}{tag}")
            elif m is not None:
                n_lost += 1
                print(f"t={t:6.1f}s #{m['seq']:<5d} LOST (detected: false)")
            elif not args.quiet:
                raw = f"  raw {s.compass_az:5.1f} el {s.doa.array_el_deg:2.0f}" if s.doa is not None else ""
                print(f"t={t:6.1f}s  {'ACQUIRING' if s.detected else 'listening'}  lvl {s.level_db:+5.1f}{raw}"
                      + (f" {s.note}" if s.note else ""))
            if args.max_s is not None and t >= args.max_s:
                break
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        log.close()
    lv = np.array(levels) if levels else np.zeros(1)
    summary = {"source": src.name, "fs": src.fs, "seconds": round(src.t(), 1), "windows": n_win,
               "windows_over_threshold": n_hit, "estimates": n_est, "lost_msgs": n_lost,
               "level_db_median": round(float(np.median(lv)), 1), "level_db_p95": round(float(np.percentile(lv, 95)), 1),
               "last": last, "north": mount.x_axis_bearing_deg, "mics_up": mount.mics_up}
    if isinstance(src, live.LiveSource):
        summary.update(dropped_windows=src.dropped, stream_status_msgs=src.status_msgs)
    if errs:
        mean_err = angle_diff_deg(circ_mean_std_deg(np.array(errs) % 360.0)[0], 0.0)
        summary.update(truth_az=args.truth_az, mean_err_deg=round(mean_err, 1),
                       rms_err_deg=round(float(np.sqrt(np.mean(np.square(errs)))), 1))
        # compass = north -/+ array az: shifting --north shifts every bearing by the same amount
        summary["suggested_north"] = round((mount.x_axis_bearing_deg - mean_err) % 360.0, 1)
        if args.calibrate:
            default_mount.parent.mkdir(parents=True, exist_ok=True)
            default_mount.write_text(json.dumps({"x_axis_bearing_deg": summary["suggested_north"],
                                                 "mics_up": mount.mics_up}, indent=2), encoding="utf-8")
            print(f"saved {default_mount}: north {summary['suggested_north']}, mics {'up' if mount.mics_up else 'down'}")
    elif args.calibrate:
        print("calibrate: no estimates, mount.json NOT written")
    (session / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"session: {session}")


if __name__ == "__main__":
    main()
