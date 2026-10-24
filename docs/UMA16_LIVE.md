# Base UMA-16: live direction finding -> bearing.json

The base's miniDSP UMA-16 v2 hears a source, estimates its compass bearing with an uncertainty, and writes
`bearing.json` (`sfhack.acoustic.bearing.BearingEstimate`, the RECON_PLAN contract). That file is the only
thing the drone gets before takeoff (`demo_recon.py --bearing <file>`).

Code: `sfhack/acoustic/` (`array.py` geometry + mounting, `doa.py` SRP-PHAT, `detect.py` detection + tracker,
`synth.py` synthetic array audio, `bearing.py` contract type), parameters in `AcousticConfig` (`sfhack/config.py`).
Geometry `sfhack/acoustic/geometry/` from https://github.com/JLrobot/Mic-Challenge (Acoular 26.8, BSD-3).
**The USB channel order is not row by row**: everything maps channels through `uma16_positions.csv`.

## Setup (Windows)

1. Install the miniDSP UMA-16 ASIO driver (miniDSP website, UMA-16 v2 downloads). Plug the array in (USB).
2. `python demos/40_uma16_live.py --list`: the UMA-16 must show up with 16 input channels, preferably under
   the ASIO host API. The script sets `SD_ENABLE_ASIO=1` before importing sounddevice (its wheel ships the
   ASIO PortAudio DLL). If auto-detect picks the wrong entry: `--device <index>`.
3. Without the driver Windows usually exposes only 2 channels: then use the fallback below.

## Mounting and north

Flat on the table, mics up (default). In the geometry file the array's +x axis points from channels
8/10/12/14 (x = -63 mm) towards channels 1/3/5/7 (x = +63 mm); `--north <deg>` is the compass bearing of +x.
Which physical edge that is is easiest to get with a speaker instead of the silkscreen:

1. Speaker >= 3 m away at a bearing B measured with a phone compass from the array centre.
   `python demos/40_uma16_live.py --north 0 --truth-az B`, play for ~10 s after the 10 s quiet start, quit.
   `summary.json` / the console print `suggested_north`: use it as `--north` from now on.
2. Check with a second bearing ~90 deg away (`--north <suggested> --truth-az B2`). If the error is now about
   180 deg (the source moves the wrong way round), the board is mirrored: add `--mics-down` and redo 1.

The channel-level panel lays the mics out as in the geometry file (+x right, +y up) with channel numbers:
tap a mic and see which square lights up to confirm the orientation.

## Run

```
python demos/40_uma16_live.py --north 0                    # live, viewer; quiet for the first 10 s (noise floor)
python demos/40_uma16_live.py --north 0 --truth-az 66      # accuracy test: speaker at measured bearing 66
python demos/40_uma16_live.py --file <16ch_48k.wav>        # fallback: any 16-ch recording, same pipeline
python demos/40_uma16_live.py --make-demo-wav --demo-az 66 # synthetic vehicle WAV (source at compass 66)
python demos/40_uma16_live.py --file $SFHACK_DATA/acoustic/demo_vehicle_066.wav     # always works
python demos/42_doa_eval.py                                # offline accuracy (recordings + synthetic truth)
```

The viewer shows the SRP-PHAT map (compass bearing x elevation), the bearing rose (stable bearing with
1-sigma / 2-sigma wedge, current window dot, green truth line with `--truth-az`), the 16 channel levels laid
out as on the board (a dead or noisy mic stands out; "CHECK CHANNELS" if the spread is > 10 dB) and the
detection level over the learned floor. Keys: q quit, r relearn the floor, SPACE pause (file), w write again.

Every session goes to `$SFHACK_DATA/acoustic/<stamp>/`: `session.wav` (raw 16-ch, live only; replay with
`--file`), `windows.jsonl` (per 0.5 s: levels, detection level, DOA, error vs truth), `bearings.jsonl`,
`bearing.json`, `summary.json`, `panel.png` (last viewer frame). `bearing.json` also goes to `--out`
(default `$SFHACK_DATA/acoustic/bearing.json`), at most once per second while the source is stable.

## How it works

- **Detection** (`detect.py`): energy per 1/3-octave band, 50-2000 Hz, over a per-band floor learned in the
  first 10 s. Level = the 3rd-largest band excess, so one tone already there (a hum, a bird) is not enough, a
  broadband / harmonic engine raises several bands. Detected at >= 6 dB.
- **DOA** (`doa.py`): SRP-PHAT over the upper hemisphere (1 deg az x 2 deg el), 120 mic pairs, GCC-PHAT
  upsampled x8, 100-4000 Hz, bins weighted by their excess over the learned noise floor (without a floor:
  over a running median across frequency). ~0.15-0.2 s per 0.5 s window on a laptop. A flat array cannot
  tell above from below the board, and elevation near the horizon is poor (sigma_el floor 10 deg).
- **Tracker**: detected for 3 s with an azimuth circular std < 8 deg -> BearingEstimate: circular mean,
  `sigma_az = hypot(std, 5 deg)` (floor: north alignment with a phone compass, array bias, reflections, far
  low-frequency engines), `sigma_el = hypot(std, 10 deg)`, label "engine" if the harmonic-comb score
  (firing frequency 20-200 Hz) has a median >= 10 dB (heuristic, see below), confidence from level and std.
- **No phase calibration**: on the real recording the 16 channels fit a plane wave to 5 deg phase rms at 1 kHz
  (14 us): they are synchronous. The `cal.npy` in the recording folder is NOT applied: its 3-9 sample "offsets"
  come from integer-lag cross-correlation of a pure 1 kHz tone (ambiguous) and a source ~7 deg off boresight.

## Accuracy

See `REPORT_recon-acoustic.md` for the numbers (real recordings, synthetic known truth in real UMA-16
background, and the live procedure). Live accuracy needs the hardware: speaker at measured bearings
(phone compass from the array centre, >= 3 m away, several bearings around the circle), `--truth-az`, read
`summary.json` (`window_err_deg`, `stable_err_deg`).
