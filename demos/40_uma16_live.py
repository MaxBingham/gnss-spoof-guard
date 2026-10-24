"""Base UMA-16 live: channel check, live direction map, engine-like source detection, bearing.json.

    python demos/40_uma16_live.py --list                         # input devices (is the UMA-16 / ASIO there?)
    python demos/40_uma16_live.py --north 0                      # LIVE: array flat on the table, +x axis -> north
    python demos/40_uma16_live.py --file rec.wav                 # same pipeline on a 16-ch WAV (fallback)
    python demos/40_uma16_live.py --make-demo-wav --demo-az 66   # synthetic 16-ch vehicle WAV (source at compass 66)
    python demos/40_uma16_live.py --file $SFHACK_DATA/acoustic/demo_vehicle_066.wav --fast --headless

Keep the source quiet for the first `--learn` seconds (noise floor). Then the panel shows the SRP-PHAT map
(compass bearing x elevation), the bearing rose, the 16 channel levels laid out as on the board (through the
geometry file; the USB channel order is not row by row), and the detection level over the floor. When a source
is detected with a steady bearing for ~3 s, `bearing.json` (BearingEstimate) is written to `--out`
(default $SFHACK_DATA/acoustic/bearing.json) and to the session folder; the drone mission reads it
(`demo_recon.py --bearing ...`).

Mounting: flat on the table, mics up; `--north DEG` = compass bearing the array's +x axis points to (measure it
with a phone compass along the board edge), `--mics-down` if the board lies mics down. Live accuracy: put the
speaker at a measured compass bearing and pass `--truth-az DEG`; errors go to the console and summary.json.

Live input: Windows needs the miniDSP ASIO driver; SD_ENABLE_ASIO=1 is set before sounddevice is imported.
Each live session's raw 16-ch audio is saved (session.wav) so it can be replayed with --file.
Keys: q / ESC quit, r relearn the noise floor, SPACE pause (file), w write the last stable bearing again.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("SD_ENABLE_ASIO", "1")      # BEFORE sounddevice is imported (miniDSP ASIO driver)

import numpy as np
import scipy.io.wavfile as wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sfhack.acoustic.array import FS, N_MICS, Mounting, mic_positions  # noqa: E402
from sfhack.acoustic.bearing import BearingEstimate  # noqa: E402
from sfhack.acoustic.detect import AcousticPipeline, WindowResult  # noqa: E402
from sfhack.acoustic.doa import angle_diff_deg  # noqa: E402
from sfhack.config import AcousticConfig, data_dir  # noqa: E402


# --------------------------------------------------------------------------------------------- sources
class FileSource:
    def __init__(self, path: Path, window: int, realtime: bool, loop: bool):
        fs, x = wavfile.read(path)
        if x.ndim != 2 or x.shape[1] < N_MICS:
            raise SystemExit(f"{path}: need >= {N_MICS} channels, got shape {x.shape}")
        if fs != FS:
            raise SystemExit(f"{path}: need {FS} Hz, got {fs}")
        scale = {np.dtype("int16"): 32768.0, np.dtype("int32"): 2.0 ** 31}.get(x.dtype, 1.0)
        self.x = x[:, :N_MICS].astype(np.float64) / scale
        self.window, self.realtime, self.loop = window, realtime, loop
        self.k = 0
        self.t_start = time.time()
        self.paused = False
        self.name = str(path)

    def read(self) -> np.ndarray | None:
        if self.k + self.window > len(self.x):
            if not self.loop:
                return None
            self.k = 0
        if self.realtime:
            delay = self.t_start + (self.k + self.window) / FS - time.time()
            if delay > 0:
                time.sleep(delay)
        out = self.x[self.k:self.k + self.window]
        self.k += self.window
        return out

    def t(self) -> float:
        return self.k / FS

    def close(self):
        pass


def list_devices():
    import sounddevice as sd
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            api = sd.query_hostapis(d["hostapi"])["name"]
            print(f"[{i:2d}] {d['name']:60s} {d['max_input_channels']:2d} ch  {d['default_samplerate']:.0f} Hz  {api}")


def find_device(match: str) -> int | None:
    import sounddevice as sd
    cands = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] < N_MICS:
            continue
        api = sd.query_hostapis(d["hostapi"])["name"]
        name = d["name"].lower()
        if match.lower() in name or ("asio" in api.lower() and "minidsp" in name):
            cands.append((0 if "asio" in api.lower() else 1, i))
    return min(cands)[1] if cands else None


class LiveSource:
    def __init__(self, device: int | None, match: str, window: int, record_path: Path | None):
        import sounddevice as sd
        self.sd = sd
        idx = device if device is not None else find_device(match)
        if idx is None:
            list_devices()
            raise SystemExit(f"No input device with >= {N_MICS} channels matching '{match}' (or a miniDSP ASIO "
                             "device). Is the UMA-16 plugged in and the miniDSP ASIO driver installed? "
                             "Fallback: --file <16ch.wav>")
        d = sd.query_devices(idx)
        self.name = f"[{idx}] {d['name']} ({sd.query_hostapis(d['hostapi'])['name']})"
        print("input:", self.name)
        self.window = window
        self.q: queue.Queue[np.ndarray] = queue.Queue()
        self.buf = np.zeros((0, N_MICS))
        self.n_read = 0
        self.dropped = 0
        self.status_msgs = 0
        self.rec = [] if record_path else None
        self.record_path = record_path
        self.paused = False

        def cb(indata, frames, time_info, status):
            if status:
                self.status_msgs += 1
            self.q.put(indata[:, :N_MICS].copy())

        self.stream = sd.InputStream(device=idx, channels=N_MICS, samplerate=FS, dtype="float32",
                                     blocksize=1024, callback=cb)
        self.stream.start()

    def read(self) -> np.ndarray | None:
        while len(self.buf) < self.window:
            blk = self.q.get(timeout=5.0)
            if self.rec is not None:
                self.rec.append(blk)
            self.buf = np.concatenate([self.buf, blk])
        # Processing fell behind: drop whole windows, keep the newest (live view stays live).
        while len(self.buf) >= 3 * self.window:
            self.buf = self.buf[self.window:]
            self.dropped += 1
        out, self.buf = self.buf[:self.window].astype(np.float64), self.buf[self.window:]
        self.n_read += self.window
        return out

    def t(self) -> float:
        return self.n_read / FS

    def close(self):
        self.stream.stop()
        self.stream.close()
        if self.rec:
            x = np.concatenate(self.rec)
            wavfile.write(self.record_path, FS, np.clip(x * 32767, -32768, 32767).astype(np.int16))
            print(f"raw audio: {self.record_path} ({len(x) / FS:.0f} s)")


# --------------------------------------------------------------------------------------------- view
class View:
    W, H = 1280, 720

    def __init__(self, cfg: AcousticConfig, mount: Mounting, source_name: str, camera: int | None, hfov: float):
        import cv2
        self.cv2 = cv2
        self.cfg, self.mount = cfg, mount
        self.source_name = source_name
        self.trace: list[tuple[float, float, float | None, bool]] = []  # t, band_db, compass az, detected
        self.pos = mic_positions()
        self.cam = cv2.VideoCapture(camera) if camera is not None else None
        self.hfov = hfov
        cv2.namedWindow("UMA-16 base array", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("UMA-16 base array", self.W, self.H)

    def srp_compass(self, r: WindowResult) -> np.ndarray:
        """SRP map resampled to compass bearing columns 0..359 (rows: el 0 at the bottom)."""
        d = r.doa
        cols = np.array([self.mount.from_compass(b) for b in range(360)])
        idx = np.rint(cols / (d.az_grid[1] - d.az_grid[0])).astype(int) % len(d.az_grid)
        return d.srp[::-1, idx]

    def draw(self, r: WindowResult, stable: BearingEstimate | None, last_written: str, status: str,
             truth_az: float | None) -> int:
        cv2 = self.cv2
        img = np.full((self.H, self.W, 3), 24, np.uint8)
        put = lambda s, xy, sc=0.5, col=(230, 230, 230), th=1: cv2.putText(  # noqa: E731
            img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)
        put(f"UMA-16 base array  |  {self.source_name}", (16, 26), 0.6)
        put(status, (16, 54), 0.7, (80, 220, 255) if "DETECT" in status or "STABLE" in status else (200, 200, 200), 2)

        # SRP map (compass x elevation)
        x0, y0, w, h = 16, 80, 760, 260
        if r.doa is not None:
            m = self.srp_compass(r)
            m = (m - m.min()) / max(np.ptp(m), 1e-9)
            heat = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)
            img[y0:y0 + h, x0:x0 + w] = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)
            caz = self.mount.to_compass(r.doa.array_az_deg)
            px = x0 + int(caz / 360 * w)
            py = y0 + h - int(r.doa.array_el_deg / self.cfg_el_max(r) * h)
            cv2.drawMarker(img, (px, py), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (120, 120, 120), 1)
        for b in range(0, 361, 45):
            xx = x0 + int(b / 360 * w)
            cv2.line(img, (xx, y0 + h), (xx, y0 + h + 5), (180, 180, 180), 1)
            put(f"{b:03d}", (xx - 14, y0 + h + 20), 0.45)
        put("compass bearing (deg) x elevation 0-90 (SRP-PHAT)", (x0, y0 - 6), 0.45, (180, 180, 180))
        if truth_az is not None:
            xx = x0 + int(truth_az % 360 / 360 * w)
            cv2.line(img, (xx, y0), (xx, y0 + h), (0, 255, 0), 1)

        # bearing rose
        cx, cy, R = 1010, 210, 120
        cv2.circle(img, (cx, cy), R, (120, 120, 120), 1)
        for b, lab in ((0, "N"), (90, "E"), (180, "S"), (270, "W")):
            a = np.radians(b)
            put(lab, (cx + int((R + 14) * np.sin(a)) - 6, cy - int((R + 14) * np.cos(a)) + 6), 0.55)
        if stable is not None:
            for s, col in ((stable.sigma_az_deg * 2, (60, 60, 140)), (stable.sigma_az_deg, (60, 90, 200))):
                cv2.ellipse(img, (cx, cy), (R, R), -90, stable.az_deg - s, stable.az_deg + s, col, -1)
            a = np.radians(stable.az_deg)
            cv2.line(img, (cx, cy), (cx + int(R * np.sin(a)), cy - int(R * np.cos(a))), (80, 220, 255), 3)
        if r.doa is not None:
            a = np.radians(self.mount.to_compass(r.doa.array_az_deg))
            col = (255, 255, 255) if r.detected else (110, 110, 110)
            cv2.circle(img, (cx + int(0.8 * R * np.sin(a)), cy - int(0.8 * R * np.cos(a))), 6, col, -1)
        if truth_az is not None:
            a = np.radians(truth_az)
            cv2.line(img, (cx, cy), (cx + int(R * np.sin(a)), cy - int(R * np.cos(a))), (0, 255, 0), 1)
        # array +x axis on the rose
        a = np.radians(self.mount.x_axis_bearing_deg)
        cv2.arrowedLine(img, (cx, cy), (cx + int(0.35 * R * np.sin(a)), cy - int(0.35 * R * np.cos(a))),
                        (160, 160, 160), 1, tipLength=0.3)
        put("array +x", (cx + int(0.4 * R * np.sin(a)) - 20, cy - int(0.4 * R * np.cos(a))), 0.4, (160, 160, 160))
        if stable is not None:
            put(f"bearing {stable.az_deg:05.1f} +- {stable.sigma_az_deg:.1f} deg  el {stable.el_deg:.0f}"
                f" +- {stable.sigma_el_deg:.0f}", (860, 365), 0.55, (80, 220, 255))
            put(f"{stable.label}  conf {stable.confidence:.2f}", (860, 390), 0.55, (80, 220, 255))
        put(last_written, (860, 415), 0.45, (150, 220, 150))

        # channel levels laid out as on the board (x right, y up)
        gx, gy, cell = 16, 400, 46
        put("channel levels (dBFS), board layout", (gx, gy - 8), 0.45, (180, 180, 180))
        for ch in range(N_MICS):
            ix = int(round((self.pos[ch, 0] + 0.063) / 0.042))
            iy = 3 - int(round((self.pos[ch, 1] + 0.063) / 0.042))
            lv = r.levels_dbfs[ch]
            v = np.clip((lv + 90) / 70, 0, 1)
            col = tuple(int(c) for c in cv2.applyColorMap(np.uint8([[v * 255]]), cv2.COLORMAP_VIRIDIS)[0, 0])
            p0 = (gx + ix * cell, gy + iy * cell)
            cv2.rectangle(img, p0, (p0[0] + cell - 4, p0[1] + cell - 4), col, -1)
            put(f"{ch}", (p0[0] + 3, p0[1] + 14), 0.4, (0, 0, 0))
            put(f"{lv:.0f}", (p0[0] + 3, p0[1] + 36), 0.4, (0, 0, 0))
        spread = np.ptp(r.levels_dbfs)
        put(f"spread {spread:.1f} dB" + ("  CHECK CHANNELS" if spread > 10 else ""), (gx, gy + 4 * cell + 16),
            0.45, (80, 80, 255) if spread > 10 else (180, 180, 180))

        # detection level + bearing trace (last 60 s)
        tx, ty, tw, th = 230, 400, 1030, 300
        cv2.rectangle(img, (tx, ty), (tx + tw, ty + th), (90, 90, 90), 1)
        put(f"detection band {self.cfg.detect_fmin:.0f}-{self.cfg.detect_fmax:.0f} Hz over floor (dB, top) / "
            f"window bearing (bottom)", (tx, ty - 8), 0.45, (180, 180, 180))
        self.trace.append((r.t, r.band_db, self.mount.to_compass(r.doa.array_az_deg) if r.doa else None,
                           r.detected))
        self.trace = [p for p in self.trace if r.t - p[0] <= 60.0]
        half = th // 2
        yth = ty + half - int(self.cfg.detect_db / 30 * half)
        cv2.line(img, (tx, yth), (tx + tw, yth), (60, 60, 160), 1)
        for t, db, az, det in self.trace:
            xx = tx + tw - int((r.t - t) / 60.0 * tw)
            yy = ty + half - int(np.clip(db, 0, 30) / 30 * half)
            cv2.circle(img, (xx, yy), 2, (80, 220, 255) if det else (150, 150, 150), -1)
            if az is not None:
                cv2.circle(img, (xx, ty + th - int(az / 360 * half)), 2,
                           (255, 255, 255) if det else (90, 90, 90), -1)
        put(f"level {r.band_db:+.1f} dB  (threshold {self.cfg.detect_db:.0f})  engine score "
            f"{r.engine_score:.1f} dB", (tx + 8, ty + 18), 0.45)

        self.last_img = img
        cv2.imshow("UMA-16 base array", img)
        if self.cam is not None and r.doa is not None:
            self.draw_camera(r)
        return cv2.waitKey(1) & 0xFF

    def cfg_el_max(self, r: WindowResult) -> float:
        return float(r.doa.el_grid[-1]) if r.doa is not None else 90.0

    def draw_camera(self, r: WindowResult):
        """Acoustic-camera overlay: ONLY meaningful when the array stands upright with the camera looking along
        the board normal (+z). Not the table mounting. Camera x = array x, image down = array -y (untested
        without the hardware; flip with the --cam-flip-x if the overlay is mirrored)."""
        cv2 = self.cv2
        ok, frame = self.cam.read()
        if not ok:
            return
        hh, ww = frame.shape[:2]
        f = (ww / 2) / np.tan(np.radians(self.hfov / 2))
        u, v = np.meshgrid(np.arange(0, ww, 8), np.arange(0, hh, 8))
        rx, ry, rz = (u - ww / 2) / f * self.cam_sx, -(v - hh / 2) / f, np.ones_like(u, float)
        n = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
        az = np.degrees(np.arctan2(ry, rx)) % 360
        el = np.degrees(np.arcsin(rz / n))
        d = r.doa
        ia = np.rint(az / (d.az_grid[1] - d.az_grid[0])).astype(int) % len(d.az_grid)
        ie = np.clip(np.rint(el / (d.el_grid[1] - d.el_grid[0])).astype(int), 0, len(d.el_grid) - 1)
        m = d.srp[ie, ia]
        m = np.clip((m - np.percentile(m, 50)) / max(np.ptp(m), 1e-9) * 2, 0, 1)
        heat = cv2.resize(cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET), (ww, hh))
        a = cv2.resize(m.astype(np.float32), (ww, hh))[..., None]
        cv2.imshow("UMA-16 camera overlay", (frame * (1 - 0.6 * a) + heat * 0.6 * a).astype(np.uint8))

    cam_sx = 1.0


# --------------------------------------------------------------------------------------------- main
def make_demo_wav(args, cfg: AcousticConfig) -> Path:
    from sfhack.acoustic.synth import array_recording, engine_signal, write_wav
    mount = Mounting(cfg.x_axis_bearing_deg, cfg.mics_up)
    az_arr = mount.from_compass(args.demo_az)
    mono = engine_signal(args.demo_s - args.learn - 2, rpm=args.demo_rpm, seed=args.seed)
    x = array_recording(az_arr, args.demo_el, duration_s=args.demo_s, snr_db=args.demo_snr,
                        onset_s=args.learn + 2, seed=args.seed, mono=mono)
    p = write_wav(data_dir() / "acoustic" / f"demo_vehicle_{int(round(args.demo_az)) % 360:03d}.wav", x)
    print(f"wrote {p}: engine ({args.demo_rpm:.0f} rpm) from compass {args.demo_az:.1f} deg, el {args.demo_el} deg, "
          f"SNR {args.demo_snr} dB, onset {args.learn + 2:.0f} s, mounting x-axis -> {cfg.x_axis_bearing_deg} deg")
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list input devices and exit")
    ap.add_argument("--file", type=Path, help="16-ch 48 kHz WAV instead of the live array")
    ap.add_argument("--fast", action="store_true", help="file: as fast as possible (default: real time)")
    ap.add_argument("--loop", action="store_true", help="file: loop forever")
    ap.add_argument("--device", type=int, help="sounddevice index (default: auto, name contains 'UMA16')")
    ap.add_argument("--north", type=float, default=None, help="compass bearing of the array +x axis (deg)")
    ap.add_argument("--mics-down", action="store_true", help="board lies mics down (mirrored frame)")
    ap.add_argument("--learn", type=float, default=None, help="noise-floor learning time (s), default 10")
    ap.add_argument("--detect-db", type=float, default=None, help="detection threshold over the floor (dB)")
    ap.add_argument("--array-enu", type=str, default=None, help="x,y,z of the array in the mission ENU frame")
    ap.add_argument("--out", type=Path, default=None, help="bearing.json path (default $SFHACK_DATA/acoustic/)")
    ap.add_argument("--truth-az", type=float, default=None, help="true compass bearing of the test speaker")
    ap.add_argument("--headless", action="store_true", help="no window, console only")
    ap.add_argument("--no-record", action="store_true", help="live: do not save the raw 16-ch audio")
    ap.add_argument("--camera", type=int, default=None, help="OpenCV camera index for the overlay (upright array)")
    ap.add_argument("--hfov", type=float, default=90.0, help="camera horizontal FOV (deg)")
    ap.add_argument("--cam-flip-x", action="store_true")
    ap.add_argument("--max-s", type=float, default=None, help="stop after this many seconds of audio")
    ap.add_argument("--make-demo-wav", action="store_true", help="write a synthetic 16-ch vehicle WAV and exit")
    ap.add_argument("--demo-az", type=float, default=66.0, help="demo WAV: compass bearing of the source")
    ap.add_argument("--demo-el", type=float, default=3.0)
    ap.add_argument("--demo-snr", type=float, default=3.0)
    ap.add_argument("--demo-rpm", type=float, default=1800.0)
    ap.add_argument("--demo-s", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.list:
        list_devices()
        return
    cfg = AcousticConfig()
    if args.north is not None:
        cfg.x_axis_bearing_deg = args.north
    if args.mics_down:
        cfg.mics_up = False
    if args.learn is not None:
        cfg.learn_s = args.learn
    if args.detect_db is not None:
        cfg.detect_db = args.detect_db
    if args.learn is None:
        args.learn = cfg.learn_s
    if args.make_demo_wav:
        make_demo_wav(args, cfg)
        return

    array_enu = tuple(float(v) for v in args.array_enu.split(",")) if args.array_enu else None
    session = data_dir() / "acoustic" / datetime.now().strftime("%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=True)
    out = args.out or data_dir() / "acoustic" / "bearing.json"
    window = int(cfg.window_s * cfg.fs)

    if args.file:
        src = FileSource(args.file, window, realtime=not args.fast, loop=args.loop)
        full_scale = 1.0
    else:
        src = LiveSource(args.device, cfg.device_match, window,
                         None if args.no_record else session / "session.wav")
        full_scale = 1.0
    pipe = AcousticPipeline(cfg, array_enu=array_enu, full_scale=full_scale)
    mount = pipe.mount
    view = None if args.headless else View(cfg, mount, src.name, args.camera, args.hfov)
    if view is not None and args.cam_flip_x:
        view.cam_sx = -1.0

    (session / "config.json").write_text(json.dumps({**cfg.__dict__, "source": src.name,
                                                      "truth_az": args.truth_az}, indent=2))
    log = open(session / "windows.jsonl", "w", encoding="utf-8")
    bearings = open(session / "bearings.jsonl", "w", encoding="utf-8")
    print(f"session: {session}\nlearning the noise floor for {cfg.learn_s:.0f} s - keep the source quiet")
    last_written, last_write_t = "no bearing yet", -1e9
    win_err, stable_err = [], []
    stable = None
    try:
        while True:
            if view is not None and getattr(src, "paused", False):
                k = view.cv2.waitKey(50) & 0xFF
                if k == ord(" "):
                    src.paused = False
                continue
            x = src.read()
            if x is None:
                break
            t = src.t()
            r, b = pipe.process(x, t=t)
            if b is not None:
                b.t = time.time() if not args.file else b.t
                stable = b
            caz = mount.to_compass(r.doa.array_az_deg) if r.doa is not None else None
            if r.learning:
                status = f"LEARNING NOISE FLOOR  {max(cfg.learn_s - t, 0):.0f} s"
            elif b is not None:
                status = f"STABLE SOURCE  {b.az_deg:05.1f} deg +- {b.sigma_az_deg:.1f}  ({b.label})"
            elif r.detected:
                status = f"DETECTED  window bearing {caz:05.1f} deg"
            else:
                status = "LISTENING"
            rec = {"t": round(t, 3), "learning": r.learning, "band_db": round(r.band_db, 2),
                   "detected": bool(r.detected), "engine_score": round(r.engine_score, 2),
                   "levels_dbfs": [round(v, 1) for v in r.levels_dbfs]}
            if r.doa is not None:
                rec.update(array_az=round(r.doa.array_az_deg, 2), array_el=round(r.doa.array_el_deg, 2),
                           compass_az=round(caz, 2), peak=round(r.doa.peak, 3), second_db=round(r.doa.second_dB, 2))
                if args.truth_az is not None and r.detected:
                    e = angle_diff_deg(caz, args.truth_az)
                    rec["err_deg"] = round(e, 2)
                    win_err.append(e)
            log.write(json.dumps(rec) + "\n")
            if b is not None:
                bearings.write(json.dumps(b.to_dict()) + "\n")
                if args.truth_az is not None:
                    stable_err.append(angle_diff_deg(b.az_deg, args.truth_az))
                if t - last_write_t >= 1.0:          # bearing.json at most once per second while stable
                    b.save(out)
                    b.save(session / "bearing.json")
                    last_write_t = t
                    last_written = f"bearing.json written {datetime.now():%H:%M:%S}"
            if args.headless:
                extra = f" err {rec['err_deg']:+.1f}" if "err_deg" in rec else ""
                print(f"t={t:6.1f}s {status:58s} lvl {r.band_db:+5.1f} dB  eng {r.engine_score:4.1f}"
                      + (f"  win az {caz:5.1f} el {r.doa.array_el_deg:4.1f}" if caz is not None else "") + extra)
            else:
                k = view.draw(r, stable, last_written, status, args.truth_az)
                if k in (ord("q"), 27):
                    break
                if k == ord("r"):
                    pipe = AcousticPipeline(cfg, array_enu=array_enu, full_scale=full_scale)
                    pipe.t0 = None
                    stable = None
                if k == ord(" ") and args.file:
                    src.paused = True
                if k == ord("w") and stable is not None:
                    stable.save(out)
                    last_written = f"bearing.json written {datetime.now():%H:%M:%S}"
            if args.max_s is not None and t >= args.max_s:
                break
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        log.close()
        bearings.close()
        if view is not None and getattr(view, "last_img", None) is not None:
            view.cv2.imwrite(str(session / "panel.png"), view.last_img)
    summary = {"source": src.name, "seconds": round(src.t(), 1), "bearing": stable.to_dict() if stable else None,
               "bearing_json": str(out) if stable else None}
    if isinstance(src, LiveSource):
        summary.update(dropped_windows=src.dropped, stream_status_msgs=src.status_msgs)
    if args.truth_az is not None:
        we, se = np.abs(win_err), np.abs(stable_err)
        summary["truth_az"] = args.truth_az
        summary["window_err_deg"] = ({"n": len(we), "mean_signed": float(np.mean(win_err)),
                                      "rms": float(np.sqrt(np.mean(we ** 2))), "p90": float(np.percentile(we, 90))}
                                     if len(we) else None)
        summary["stable_err_deg"] = ({"n": len(se), "last": float(stable_err[-1]),
                                      "rms": float(np.sqrt(np.mean(se ** 2)))} if len(se) else None)
    (session / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"session: {session}")


if __name__ == "__main__":
    main()
