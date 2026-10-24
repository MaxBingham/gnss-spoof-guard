"""UMA-16 DOA accuracy: real recordings + synthetic known-truth sources in real UMA-16 background audio.

    python demos/42_doa_eval.py                      # all, 6 workers
    python demos/42_doa_eval.py --quick              # fewer directions / SNRs

Recordings: Mic-Challenge `data/uma16_real/jduanen_acoustic_camera/` (MIT), found under --rec-dir
(default $SFHACK_DATA/ext/Mic-Challenge/...; clone https://github.com/JLrobot/Mic-Challenge there).

1. Real recordings (array UPRIGHT as an acoustic camera there, board normal = towards the scene):
   - channel sync: plane-wave fit of the per-channel phase of the 1 kHz calibration tone (cal_ref.wav);
   - cal_ref.wav: tone placed by hand "at boresight, 0.5-1 m": angle between our DOA and the board normal
     (the only recording with a documented direction, and only nominally);
   - capture_nb16.wav: direction NOT documented ("make a sound now"): spread only, no accuracy claim.
2. Synthetic, known truth: an engine at array az 0..330 (el 3 deg, flat-table mounting) mixed into the real
   UMA-16 background (both recordings, incl. their tones as interferers) or white noise, at several SNRs
   (source power / background power over the full band). "near" = broadband engine (speaker demo);
   "far" = the same through a 2nd-order 300 Hz low-pass (2-3 km: air absorption and ground effect remove the
   highs; most energy below the band where a 126 mm aperture resolves well); "far_mp" = far + one reflection
   (-3 dB, 15 ms, from 70-110 deg to the side; `--signals far_mp`). Runs the full AcousticPipeline
   (noise floor, detection, tracker) and scores the window DOAs and the stable BearingEstimate.

Output: $SFHACK_DATA/acoustic/eval/<stamp>/{results.json, table.md, err_vs_snr.png}
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sfhack.acoustic.array import C_SOUND, FS, mic_positions, unit_vector  # noqa: E402
from sfhack.acoustic.detect import AcousticPipeline, circ_mean_std_deg  # noqa: E402
from sfhack.acoustic.doa import DoaConfig, SrpPhat, angle_diff_deg  # noqa: E402
from sfhack.acoustic.synth import array_recording, engine_signal, plane_wave  # noqa: E402
from sfhack.config import AcousticConfig, data_dir  # noqa: E402

REC_REL = Path("ext/Mic-Challenge/data/uma16_real/jduanen_acoustic_camera")


def load_wav(p: Path) -> np.ndarray:
    fs, x = wavfile.read(p)
    assert fs == FS and x.shape[1] == 16, (fs, x.shape)
    return x.astype(np.float64) / 32768.0


# ----------------------------------------------------------------------------------------- real recordings
def channel_sync(x: np.ndarray, f0: float = 1000.0) -> dict:
    """Plane-wave fit of the tone phase per channel; residual = channel phase mismatch + noise."""
    pos = mic_positions()[:, :2]
    N = 19200
    res, dirs = [], []
    for s in range(0, len(x) - N + 1, N // 2):
        F = np.fft.rfft(x[s:s + N] * np.hanning(N)[:, None], axis=0)
        f = np.fft.rfftfreq(N, 1 / FS)
        band = np.where(np.abs(f - f0) < 30)[0]
        kb = band[np.argmax(np.abs(F[band]).sum(1))]
        X = F[kb] / np.abs(F[kb])
        k = 2 * np.pi * f[kb] / C_SOUND
        g = np.linspace(-1, 1, 401)
        UX, UY = np.meshgrid(g, g)
        st = np.exp(1j * k * (pos[:, 0, None, None] * UX + pos[:, 1, None, None] * UY))
        P = np.abs((X[:, None, None] * np.conj(st)).sum(0))
        P[UX ** 2 + UY ** 2 > 1] = 0
        i = np.unravel_index(np.argmax(P), P.shape)
        r = np.angle(X * np.exp(-1j * k * (pos @ [UX[i], UY[i]])))
        r = np.angle(np.exp(1j * (r - np.angle(np.exp(1j * r).mean()))))
        res.append(np.degrees(np.sqrt(np.mean(r ** 2))))
        dirs.append(np.degrees(np.arcsin(min(np.hypot(UX[i], UY[i]), 1.0))))
    return {"tone_hz": f0, "phase_resid_rms_deg": float(np.median(res)),
            "phase_resid_as_delay_us": float(np.median(res) / 360 / f0 * 1e6),
            "off_boresight_deg_mean": float(np.mean(dirs)), "off_boresight_deg_std": float(np.std(dirs))}


def real_windows(x: np.ndarray, fmin: float, fmax: float) -> dict:
    s = SrpPhat(DoaConfig(fmin=fmin, fmax=fmax))
    n = FS // 2
    off, az = [], []
    for k in range(0, len(x) - n + 1, n):
        r = s.estimate(x[k:k + n])
        off.append(90.0 - r.array_el_deg)        # angle from the board normal
        az.append(r.array_az_deg)
    u = unit_vector(np.array(az), 90.0 - np.array(off))
    m = u.mean(0)
    m /= np.linalg.norm(m)
    spread = np.degrees(np.arccos(np.clip(u @ m, -1, 1)))
    return {"band_hz": [fmin, fmax], "n_windows": len(off),
            "off_boresight_deg_mean": float(np.mean(off)), "off_boresight_deg_std": float(np.std(off)),
            "max_off_boresight_deg": float(np.max(off)),
            "spread_about_mean_dir_deg_rms": float(np.sqrt(np.mean(spread ** 2))),
            "mean_dir_off_boresight_deg": float(np.degrees(np.arccos(np.clip(m[2], -1, 1))))}


# ----------------------------------------------------------------------------------------- synthetic
def run_case(case: dict) -> dict:
    cfg = AcousticConfig(learn_s=case["learn_s"], doa_fmin=case["doa_fmin"])
    rng_seed = case["seed"]
    dur = case["learn_s"] + 2.0 + case["src_s"]
    mono = engine_signal(dur, rpm=case["rpm"], seed=rng_seed)
    if case["signal"].startswith("far"):
        b, a = signal.butter(2, 300.0, btype="low", fs=FS)
        mono = signal.lfilter(b, a, mono)
    mono = mono[: int((case["src_s"]) * FS)]
    noise = None
    if case["bg"] != "white":
        noise = np.load(case["bg_path"])
    x = array_recording(case["az"], case["el"], duration_s=dur, snr_db=case["snr"], onset_s=case["learn_s"] + 2.0,
                        seed=rng_seed, mono=mono, noise=noise)
    if case["signal"].endswith("_mp"):
        # One wall / building reflection: -3 dB, 15 ms later (~5 m extra path), from 70-110 deg to either side.
        rng = np.random.default_rng(rng_seed)
        az_r = case["az"] + rng.choice([-1, 1]) * rng.uniform(70, 110)
        d = int(0.015 * FS)
        refl = plane_wave(np.concatenate([np.zeros(d), mono])[: len(mono)] * 10 ** (-3 / 20), az_r, case["el"])
        k = int((case["learn_s"] + 2.0) * FS)
        x[k:k + len(refl)] += refl[: len(x) - k]     # same scale as the direct path (array_recording scales noise)
    p = AcousticPipeline(cfg)
    n = int(cfg.window_s * FS)
    onset = case["learn_s"] + 2.0
    win_err, det, first, last = [], 0, None, None
    n_src = 0
    for k in range(0, len(x) - n + 1, n):
        t = k / FS
        r, bear = p.process(x[k:k + n], t=t)
        if r.learning or t < onset:
            continue
        n_src += 1
        if r.detected:
            det += 1
            win_err.append(angle_diff_deg(r.doa.array_az_deg, case["az"]))
        if bear is not None:
            last = bear
            if first is None:
                first = (t - onset, bear)
    out = {k: v for k, v in case.items() if k != "bg_path"}
    out.update(det_rate=det / max(n_src, 1), n_windows=n_src,
               win_err=[round(e, 2) for e in win_err],
               stable=last is not None,
               t_first_s=None if first is None else round(first[0], 2))
    if last is not None:
        # mics up, x axis = north: compass = -array az
        e = angle_diff_deg(last.az_deg, (-case["az"]) % 360.0)
        out.update(stable_err=round(e, 2), sigma_az=round(last.sigma_az_deg, 2), label=last.label,
                   within_sigma=bool(abs(e) <= last.sigma_az_deg), el_est=round(last.el_deg, 1))
    return out


def summarize(rows: list[dict]) -> list[dict]:
    keys = sorted({(r["signal"], r["bg"], r["snr"]) for r in rows}, key=lambda k: (k[0], k[1], -k[2]))
    table = []
    for sig, bg, snr in keys:
        rr = [r for r in rows if (r["signal"], r["bg"], r["snr"]) == (sig, bg, snr)]
        we = np.abs(np.concatenate([r["win_err"] for r in rr])) if any(r["win_err"] for r in rr) else np.array([])
        st = [r for r in rr if r["stable"]]
        se = np.abs([r["stable_err"] for r in st])
        table.append({
            "signal": sig, "bg": bg, "snr_db": snr, "runs": len(rr),
            "det_rate": float(np.mean([r["det_rate"] for r in rr])),
            "win_err_med": float(np.median(we)) if len(we) else None,
            "win_err_p90": float(np.percentile(we, 90)) if len(we) else None,
            "stable_rate": len(st) / len(rr),
            "stable_err_rms": float(np.sqrt(np.mean(se ** 2))) if len(se) else None,
            "stable_err_max": float(np.max(se)) if len(se) else None,
            "within_sigma": float(np.mean([r["within_sigma"] for r in st])) if st else None,
            "sigma_mean": float(np.mean([r["sigma_az"] for r in st])) if st else None,
            "engine_label": float(np.mean([r["label"] == "engine" for r in st])) if st else None,
            "t_first_med": float(np.median([r["t_first_s"] for r in st])) if st else None,
        })
    return table


def fmt(v, f="{:.1f}"):
    return "-" if v is None else f.format(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec-dir", type=Path, default=data_dir() / REC_REL)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--doa-fmin", type=float, default=AcousticConfig.doa_fmin)
    ap.add_argument("--signals", default="near,far")
    args = ap.parse_args()
    out = data_dir() / "acoustic" / "eval" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    cal = load_wav(args.rec_dir / "cal_ref.wav")
    cap = load_wav(args.rec_dir / "capture_nb16.wav")
    results["real"] = {
        "channel_sync_cal_ref": channel_sync(cal, 1000.0),
        "cal_ref_broadband": real_windows(cal, 300, 4000),
        "cal_ref_tone_band": real_windows(cal, 900, 1100),
        "capture_nb16_broadband": real_windows(cap, 300, 4000),
        "capture_nb16_tone_band": real_windows(cap, 1000, 1400),
    }
    print(json.dumps(results["real"], indent=1))

    bg_paths = {}
    for name, x in (("real_cal", cal), ("real_cap", cap)):
        p = out / f"bg_{name}.npy"
        np.save(p, x)
        bg_paths[name] = str(p)
    azs = [0, 90, 200, 300] if args.quick else list(range(0, 360, 30))
    snrs = [10, 0, -10] if args.quick else [10, 5, 0, -5, -10]
    cases = []
    for sig in args.signals.split(","):
        for bg in ("white", "real_cal", "real_cap"):
            for snr in snrs:
                for i, az in enumerate(azs):
                    cases.append(dict(signal=sig, bg=bg, bg_path=bg_paths.get(bg), snr=snr, az=float(az), el=3.0,
                                      rpm=1500.0 + 100 * i, seed=1000 * i + snr + 17, learn_s=5.0, src_s=10.0,
                                      doa_fmin=args.doa_fmin))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(run_case, cases, chunksize=2))
    for p in bg_paths.values():
        Path(p).unlink()
    results["synthetic"] = rows
    table = summarize(rows)
    results["table"] = table
    (out / "results.json").write_text(json.dumps(results, indent=1))

    lines = ["| signal | background | SNR dB | detect | window err med / p90 (deg) | stable | stable err rms / max "
             "| within sigma | mean sigma | 'engine' | t to bearing (s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        lines.append(f"| {r['signal']} | {r['bg']} | {r['snr_db']:+d} | {r['det_rate']:.0%} | "
                     f"{fmt(r['win_err_med'])} / {fmt(r['win_err_p90'])} | {r['stable_rate']:.0%} | "
                     f"{fmt(r['stable_err_rms'])} / {fmt(r['stable_err_max'])} | "
                     f"{fmt(r['within_sigma'], '{:.0%}')} | {fmt(r['sigma_mean'])} | "
                     f"{fmt(r['engine_label'], '{:.0%}')} | {fmt(r['t_first_med'])} |")
    md = "\n".join(lines)
    (out / "table.md").write_text(md + "\n")
    print(md)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for sig, ls in zip(dict.fromkeys(r["signal"] for r in table), ("-", "--", ":")):
        for bg, col in (("white", "C0"), ("real_cal", "C1"), ("real_cap", "C2")):
            rr = [r for r in table if r["signal"] == sig and r["bg"] == bg]
            s = [r["snr_db"] for r in rr]
            ax[0].plot(s, [np.nan if r["stable_err_rms"] is None else r["stable_err_rms"] for r in rr], ls,
                       color=col, marker="o", label=f"{sig} / {bg}")
            ax[1].plot(s, [r["stable_rate"] for r in rr], ls, color=col, marker="o")
    ax[0].set(xlabel="SNR (dB)", ylabel="stable bearing error rms (deg)", title="Bearing error (known truth)")
    ax[1].set(xlabel="SNR (dB)", ylabel="fraction with a stable bearing", title="Bearing produced")
    ax[0].legend(fontsize=7)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "err_vs_snr.png", dpi=120)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
