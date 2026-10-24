"""Stage calibration with a continuous sound (the drone sound on a speaker): north and mics up / down.

    python demos/46_uma16_calibrate.py                       # 0, 90, 135, 180, 225, 270, 0; 15 s each
    python demos/46_uma16_calibrate.py --marks 0,90,180,270 --play-s 12
    python demos/46_uma16_calibrate.py --file <session.wav>  # re-fit a recorded session (same schedule)

A full-screen cue window says where to stand and when to play: first `--quiet-s` of silence (noise floor), then
per mark MOVE (`--move-s`, not used) and PLAY (`--play-s`). Each PLAY segment gives the circular median of the
SRP-PHAT array azimuth over 0.4 s windows where the 800-4000 Hz band is 6 dB over the floor. Both mountings are
fitted (mics up: compass = north - array az; mics down: north + array az); the one with the smaller residual wins
and is written to `$SFHACK_DATA/acoustic/mount.json` (the old file is kept as mount.prev.json).
"""
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
live = importlib.import_module("40_uma16_live")

import numpy as np  # noqa: E402
import scipy.io.wavfile as wavfile  # noqa: E402

from sfhack.acoustic.detect import circ_mean_std_deg  # noqa: E402
from sfhack.acoustic.doa import DoaConfig, SrpPhat, angle_diff_deg  # noqa: E402
from sfhack.config import AcousticConfig, data_dir  # noqa: E402

FMIN, FMAX = 800.0, 4000.0


def schedule(marks, quiet_s, move_s, play_s):
    """[(t0, t1, kind, mark)] in stream seconds."""
    out, t = [(0.0, quiet_s, "QUIET", None)], quiet_s
    for mk in marks:
        out += [(t, t + move_s, "MOVE", mk), (t + move_s, t + move_s + play_s, "PLAY", mk)]
        t += move_s + play_s
    return out + [(t, t + 3.0, "DONE", None)]


def circ_medoid(a: np.ndarray) -> float:
    dev = np.abs((a[:, None] - a[None, :] + 180.0) % 360.0 - 180.0)
    return float(a[np.argmin(dev.sum(axis=1))])


def analyse(x: np.ndarray, fs: int, sched, verbose=True) -> list[dict]:
    srp = SrpPhat(DoaConfig(fs=fs, nfft=1024, hop=256, fmin=FMIN, fmax=FMAX, el_step=5.0))
    from scipy.signal import butter, sosfilt  # noqa: PLC0415
    band = sosfilt(butter(4, [FMIN, FMAX], btype="bandpass", fs=fs, output="sos"), x, axis=0)
    q0, q1 = sched[0][0] + 1.0, sched[0][1] - 0.5
    srp.learn_noise(x[int(q0 * fs):int(q1 * fs)])
    floor = float(np.mean(band[int(q0 * fs):int(q1 * fs)] ** 2))
    win, hop = int(0.4 * fs), int(0.2 * fs)
    res = []
    for t0, t1, kind, mk in sched:
        if kind != "PLAY":
            continue
        az, el, lv = [], [], []
        for s in range(int((t0 + 2.0) * fs), int((t1 - 1.0) * fs) - win, hop):
            p = float(np.mean(band[s:s + win] ** 2))
            if p < 4.0 * floor:
                continue
            d = srp.estimate(x[s:s + win])
            az.append(d.array_az_deg)
            el.append(d.array_el_deg)
            lv.append(10 * np.log10(p / floor))
        r = {"mark": mk, "n": len(az)}
        if az:
            a = np.array(az)
            m = circ_medoid(a)
            r.update(array_az=round(m, 1), spread=round(1.4826 * float(np.median(np.abs(angle_diff_deg(a, m)))), 1),
                     el=round(float(np.median(el)), 1), level_db=round(float(np.median(lv)), 1))
        res.append(r)
        if verbose:
            print(f"mark {mk:5.0f}: {r}")
    return res


def fit(res: list[dict]) -> dict:
    ok = [r for r in res if r["n"] >= 5]
    best = None
    for up in (True, False):
        n = np.array([(r["mark"] + (r["array_az"] if up else -r["array_az"])) % 360.0 for r in ok])
        north = circ_mean_std_deg(n)[0]
        resid = [angle_diff_deg((north - r["array_az"] if up else north + r["array_az"]) % 360.0, r["mark"])
                 for r in ok]
        f = {"mics_up": up, "north": round(north, 1), "rms_deg": round(float(np.sqrt(np.mean(np.square(resid)))), 1),
             "resid": {f"{r['mark']:.0f}": round(e, 1) for r, e in zip(ok, resid)}}
        print(f"mics {'up  ' if up else 'down'}: north {f['north']:6.1f}  rms {f['rms_deg']:5.1f} deg  {f['resid']}")
        if best is None or f["rms_deg"] < best["rms_deg"]:
            best = f
    return best


class Cue:
    """Full-screen cue: what to do now, which mark, seconds left."""

    def __init__(self, sched, clock):
        import tkinter as tk  # noqa: PLC0415
        self.sched, self.clock = sched, clock
        self.root = tk.Tk()
        self.root.title("UMA-16 calibration")
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg="black")
        self.big = tk.Label(self.root, font=("Helvetica", 120, "bold"), fg="white", bg="black")
        self.small = tk.Label(self.root, font=("Helvetica", 48), fg="#bbbbbb", bg="black")
        self.big.pack(expand=True)
        self.small.pack(expand=True)
        self.root.bind("<Escape>", lambda e: self.root.destroy())

    def tick(self):
        t = self.clock()
        cur = next((s for s in self.sched if s[0] <= t < s[1]), self.sched[-1])
        t0, t1, kind, mk = cur
        nxt = self.sched[self.sched.index(cur) + 1] if cur != self.sched[-1] else None
        colour = {"QUIET": "#4aa3ff", "MOVE": "#ffcc00", "PLAY": "#33ff66", "DONE": "white"}[kind]
        text = {"MOVE": "GO TO", "PLAY": "PLAY "}.get(kind, kind) + (f" {mk:.0f}°" if mk is not None else "")
        sub = f"{t1 - t:4.0f} s" + (f"   next: {nxt[2]} {'' if nxt[3] is None else f'{nxt[3]:.0f}°'}" if nxt else "")
        if kind == "QUIET":
            sub = f"no sound please  {t1 - t:4.0f} s   (learning the room)"
        self.big.configure(text=text, fg=colour)
        self.small.configure(text=sub)
        if t >= self.sched[-1][1]:
            self.root.destroy()
            return
        self.root.after(100, self.tick)

    def run(self):
        self.tick()
        self.root.mainloop()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--marks", default="0,90,135,180,225,270,0", help="compass marks in order (deg)")
    ap.add_argument("--quiet-s", type=float, default=8.0)
    ap.add_argument("--move-s", type=float, default=6.0)
    ap.add_argument("--play-s", type=float, default=15.0)
    ap.add_argument("--file", type=Path, help="re-fit a recorded session.wav with the same schedule")
    ap.add_argument("--device", type=int)
    ap.add_argument("--no-save", action="store_true", help="do not write mount.json")
    ap.add_argument("--no-cue", action="store_true", help="no cue window, print the schedule instead")
    args = ap.parse_args()
    marks = [float(v) for v in args.marks.split(",")]
    sched = schedule(marks, args.quiet_s, args.move_s, args.play_s)

    if args.file:
        fs, x = wavfile.read(args.file)
        x = x[:, :16].astype(np.float64) / (32768.0 if x.dtype == np.int16 else 1.0)
        session = args.file.parent
    else:
        session = data_dir() / "acoustic" / datetime.now().strftime("calib_%Y%m%d_%H%M%S")
        session.mkdir(parents=True, exist_ok=True)
        src = live.LiveSource(args.device, AcousticConfig().device_match, 0.1, session / "session.wav", 48000)
        fs = src.fs
        blocks, stop = [], threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: stop.set())

        def reader():
            while not stop.is_set() and src.t() < sched[-1][1]:
                blocks.append(src.read())
            stop.set()

        th = threading.Thread(target=reader, daemon=True)
        th.start()
        print("schedule:", ", ".join(f"{k} {'' if m is None else f'{m:.0f}'} {a:.0f}-{b:.0f}s" for a, b, k, m in sched))
        if args.no_cue:
            last = None
            while not stop.is_set():
                t = src.t()
                cur = next((s for s in sched if s[0] <= t < s[1]), sched[-1])
                if cur != last:
                    last = cur
                    print(f"t={t:5.1f}s  {cur[2]} {'' if cur[3] is None else f'{cur[3]:.0f} deg'}", flush=True)
                time.sleep(0.1)
        else:
            try:
                Cue(sched, src.t).run()
            except Exception as e:  # noqa: BLE001
                print(f"no cue window ({e}); follow the printed schedule")
            stop.wait()
        th.join(timeout=2.0)
        src.close()
        x = np.concatenate(blocks)
    print(f"{len(x) / fs:.0f} s at {fs} Hz")
    res = analyse(x, fs, sched)
    best = fit(res)
    summary = {"marks": marks, "segments": res, "fit": best}
    (session / "calibration.json").write_text(json.dumps(summary, indent=2))
    mount = data_dir() / "acoustic" / "mount.json"
    if not args.no_save and best is not None:
        if mount.exists():
            shutil.copy(mount, mount.with_name("mount.prev.json"))
        mount.write_text(json.dumps({"x_axis_bearing_deg": best["north"], "mics_up": best["mics_up"]}, indent=2))
        print(f"saved {mount}: north {best['north']}, mics {'up' if best['mics_up'] else 'down'} "
              f"(rms {best['rms_deg']} deg)")


if __name__ == "__main__":
    main()
