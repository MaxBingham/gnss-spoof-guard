# STATUS recon-acoustic (2026-10-24 11:55)

## Works
- `sfhack/acoustic/bearing.py`: `BearingEstimate` + `bearing.json` per the contract (first commit, c5cfdda;
  recon-world / recon-mission were told to fetch it rather than write their own).
- `sfhack/acoustic/geometry/`: UMA-16 geometry from Mic-Challenge (Acoular, BSD-3, license kept);
  `array.py` maps USB channels through it; `Mounting` (`--north`, `--mics-down`) -> compass bearing.
- `doa.py`: SRP-PHAT (own numpy), 120 pairs, hemisphere grid, noise-floor bin weights, ~0.15-0.2 s per 0.5 s window.
- `detect.py`: per-1/3-octave detection over a learned floor, tracker (3 s steady) -> BearingEstimate with sigma,
  harmonic "engine" label (heuristic).
- `synth.py`: 16-ch plane-wave audio with the real geometry (known truth; usable by recon-world's sim_array).
- `demos/40_uma16_live.py`: live (ASIO) / `--file` / `--make-demo-wav`, OpenCV viewer, session logs,
  `--truth-az` accuracy + `suggested_north`. Demo WAV: `$SFHACK_DATA/acoustic/demo_vehicle_066.wav` -> 66.0 deg.
- `demos/42_doa_eval.py`: real recordings + synthetic known truth in real UMA-16 background.
- `tests/test_acoustic.py` (9 tests) green; truth separation untouched (no onboard code changed).

## Numbers (so far)
- Real recordings: channels synchronous (plane-wave phase residual 5 deg rms at 1 kHz = 14 us): no calibration;
  `cal.npy` is NOT applied (its offsets are tone-ambiguity artifacts). Calibration tone placed "at boresight":
  we find it 6.5-9.6 deg off the board normal, 1.6-2.0 deg rms window-to-window. `capture_nb16.wav` has no
  documented direction (spread 26-30 deg rms; source seems to move): no accuracy claim.
- Synthetic, known truth (quick run): near engine 0 dB in real background -> stable bearing error <= 0.2 deg;
  far (300 Hz low-pass) engine 0 dB white -> 0.3 deg rms; worst: far engine + moving tonal interferer at 0 dB ->
  no bearing emitted (not a wrong one). Full table in progress -> REPORT_recon-acoustic.md.

## Blockers / open
- No UMA-16 on this machine (and no ASIO driver): live accuracy with a speaker at measured bearings is NOT done.
  Procedure is in docs/UMA16_LIVE.md (`--truth-az`).
- Engine label is a heuristic (synthetic engine vs noise separates, real tone recordings overlap).

## Dependencies added
- `sounddevice>=0.5.6` (pyproject + uv.lock).
