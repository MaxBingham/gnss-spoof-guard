# REPORT night-thermal (branch `night-thermal`, C:\dev\night-thermal)

Agent night-thermal, night of 2026-10-23 -> 24. Task: thermal world model (truth + onboard prior) and the
simulated downward LWIR camera, per the interface contract in NIGHT_PLAN.md. Details: `docs/THERMAL_MODEL.md`.

## Team result that uses this work (from night-demo's REPORT_night-demo.md, branch night-demo 52621e3)

Final 20-seed eval: night 23:00, 60 m AGL, GNSS completely off from 20 % of the route. It ran on a frozen copy
of my final thermal build (01:17; moving traffic on, truth contrast 0.7).

| Mission | Thermal + IMU + TRN (final build): within 5 m of B | IMU + TRN only (no thermal) |
| --- | --- | --- |
| berkeley | 20/20 (Wilson 95 % CI 84-100 %), mean 0.4 m | 19/20 |
| m-c804bb84 | 20/20 (84-100 %), mean 0.9 m | 11/20 |
| ukraine-1 | 20/20 (84-100 %), mean 1.9 m | 0/20 |

Mild night: their "low_contrast" row is 0.4 x my 0.7, i.e. 0.28 effective. Final build: berkeley 18/20,
ukraine-1 19/20. The berkeley losses were overconfident fixes; night-demo's next step is a contrast-aware fix
covariance. Numbers and caveats are night-demo's; see their report.

## What works

- `python -m sfhack.thermal build --mission <id> [--hour 23] [--season summer] [--seed 0] [--contrast 0.7]`
  writes `world/thermal_truth.npz` (simulator only) and `onboard/thermal_prior.npz` (+ a `drone/` copy). Built
  for **berkeley, m-c804bb84 and ukraine-1** (11 / 14 / 33 s). The final in-place build ran at ~01:17 and is
  frozen for night-demo's eval (agreed by message).
- **Material classes** per 0.5 m pixel from ortho + NIR (NDVI) + OSM (highway type and `surface`, landuse,
  parking, water, underground buildings) + LiDAR canopy height (US; NAIP alone cannot tell crowns from lawns),
  or an Otsu split + OSM wood / meadow (no LiDAR).
- **Physics**: 1-D surface energy balance per material x sun exposure x sky view factor (both from the DSM),
  Parton-Logan night air, at the local solar time of `hour`. Calibrated to night-research's literature at 23 h:
  asphalt +6.0 K over air, concrete +3.1, grass -4.0, canopy -1.9, water +3.8, dirt road -2.1 (ASTER: unpaved
  roads slightly cool), insulated roofs -3.8 / -6.6, heavy roofs +3.8, asphalt - grass 9.9 K, and asphalt
  cooling 2.6 K/h then 1.5 K/h. Measured asphalt - grass on clear calm nights is ~5 K, so the **truth is scaled
  by 0.7** around the air temperature (road - vegetation medians 4.5-5.2 K in the built missions). The prior
  keeps the full model.
- **Truth vs prior**: the truth adds roof construction per building, ~200-480 parked cars (cold, 15 % with a
  hot hood), wet and shaded patches, HVAC units, per-road / per-building / per-class offsets, cold-air pooling
  (1.4 K / 100 m), smooth fields and texture. The prior comes from onboard data only (degraded ortho, perturbed
  landmarks, onboard DSM, 5 % of minor OSM ways missing) with a per-pixel sigma.
  Prior vs truth: corr 0.72 / 0.74 / 0.67; roads RMSE 2.5 / 2.7 / 2.5 K; heavy-tailed residuals
  (75-80 % within 1 sigma, 90-97 % within 2 sigma).
- **Ukraine** (Sentinel-2 10 m, GLO-30, no LiDAR): coarse imagery does not hide OSM roads; untagged village
  lanes are uncertain (the truth makes 40 % of them paved; the prior marks them dirt with higher sigma, roads = 0).
- **Camera** `ThermalCamera`: 640 x 512, 50 deg HFOV, a `thermal` camera on the X2 (same mount as `down`).
  - One GL pass gives depth + segmentation; each pixel is back-projected to world xyz and samples the truth.
  - Walls, trunks, objects, the drone and the sky have their own temperatures; moving traffic (6 cars per flight
    on the paved roads, hot hoods).
  - Sensor effects: blur, vignetting, FPN, absolute calibration error (2 K / 3 %), drift with flat-field
    correction, NETD 50 mK.
  - `to_8bit` AGC, white-hot.
  - Speed: 64-89 ms per frame in flight on berkeley, 108 ms on ukraine-1, with `renderer=sim.renderer` (one GL
    context). GPU-bound while the detector training shares the GPU.
- **Is there signal?** `demos/22_thermal_matchability.py` correlates camera frames with the prior's predicted
  view over +-15 m at 60 m AGL. Zero-mean NCC peak within 2 m: 87 % (berkeley), 83 % (hill), 69 % (ukraine-1)
  of frames. With gradient magnitudes: 60 / 67 / 80 %. The numbers are unchanged at contrast 0.4. Failures are
  over big roofs and forest.
- **Integration QA (01:25)**: a scratch merge of night-thermal daf5917 + night-nav 4904433 passes all 71 tests;
  one berkeley night flight through `demos/21_night_eval.py --renderer full` (GNSS off at 20 %, 60 m AGL, seed 0):
  - method grad: thermal 48 ms per frame, 340 / 346 fixes accepted, error at B 0.3 m (estimate error mean 0.9 m,
    max 3.1 m);
  - IMU + TRN only: 0.7 m at B (max 6.9 m).
  One seed only; night-demo's 20-seed eval is the real number.
- Onboard helpers: `load_prior`, `ThermalPrior.predict_view` (expected image from a pose; agrees with the
  camera, corr > 0.95 at 3 yaws). `load_truth(contrast=)` for mild-night rows without rebuilding.
- NIR: berkeley and m-c804bb84 got `world/nir.png` + `onboard/nir.png` (NAIP band 4), added once. New missions
  get it from night-demo's missionprep (same name and format, agreed).
- Demos: `demos/20_thermal_view.py` (fly: camera | ortho + footprint | prior; `--record`, `--snap`),
  `demos/21_thermal_figures.py`, `demos/22_thermal_matchability.py`. Backup videos:
  `$SFHACK_DATA\night\thermal\thermal_view_berkeley.mp4` and `thermal_view_ukraine-1.mp4` (truth pose; they show
  the camera, not a nav result; recorded at contrast 1.0 before the 0.7 default).
- Tests: `tests/test_thermal.py` (24 tests): physics, night cooling, classification (roads, lawns, canopy,
  untagged roads abroad, underground buildings), camera geometry and orientation, noise / 8-bit, traffic,
  contrast, predict_view vs camera, and mission raster alignment. Truth separation is extended: the thermal truth
  and camera are simulator side; onboard code may only use `thermal/prior.py`. **Full suite green (59 tests).**

## What failed / limits / surprises

- Light insulated roofs come out at -6.6 K (literature typical -4..-6): slightly cold.
- First renders were 203 ms per frame in flight: a second GL context costs ~50 ms per switch on Windows. Fixed
  by drawing into the simulator's context. ukraine-1 is still ~108 ms (GPU shared with the training).
- Machine-wide (01:35): the Windows commit charge was nearly exhausted (1.7 GB free of 80.8 GB limit, 21.7 GB
  RAM free). night-nav's eval workers died twice with BrokenProcessPool (~01:15, ~01:32), at the times of my
  thermal rebuilds and a 2-worker QA flight. No process was killed; allocations most likely failed.
  - Each sim worker is ~1.3 GB working set.
  - Run fewer parallel workers, or enlarge the pagefile (a system setting, the user's call).
  - I ran nothing heavy after that until the wrap-up.
- Without LiDAR, canopy vs lawn is weak (ukraine-1 relies on OSM wood / meadow).
- OSM quirks show up in the thermal map, consistently in truth and prior because the sim world has the same
  buildings: an OSM "Wheeler Hall" outline (GNIS import) sits where the ortho shows a plaza, and some campus
  buildings are missing from OSM (their roofs become warm pavement). Underground buildings are handled.
- The truth is 2.5-D (one temperature per wall, nothing under roofs or crowns). Parked cars and wet patches are
  fixed per build; traffic is per flight.
- Only summer is calibrated; other seasons are rough presets. No weather (clouds, wind, rain) tonight.

## Next steps

- Validate the default contrast against a real hot-night thermal scene (night-research's ASTER work).
- Wind and cloud as parameters (contrast drops 30-80 %), then a robustness sweep in the nav eval.
- Per-flight randomization of parked cars / wet patches; crown-scale texture in forests.

## Disclosure (pre-hackathon work, all built tonight by an AI agent, Claude Opus 5.5)

Everything in this branch beyond `origin/v5-mission-planner` was written tonight, 2026-10-23 23:30 to
2026-10-24 07:00:
- `sfhack/thermal/*`
- `demos/20_thermal_view.py`, `demos/21_thermal_figures.py`, `demos/22_thermal_matchability.py`
- `docs/THERMAL_MODEL.md`, `docs/figures/*`
- `tests/test_thermal.py`, and the thermal checks in `tests/test_truth_separation.py`
- the `thermal` camera in `sfhack/sim/mujoco_sim.py` and the `CameraConfig.thermal_*` fields
- this report and the night-thermal section of `RESUME.md`

Data written into `$SFHACK_DATA` (new files only):
- `missions/*/world/thermal_truth.npz`, `world/thermal_build.json`, `onboard/thermal_prior.npz`,
  `drone/thermal_prior.npz`
- `world/nir.png` + `onboard/nir.png` (+ `drone/`) for berkeley and m-c804bb84
- `night/thermal/*` (videos, matchability json)

Material constants come from the literature; calibration targets and real-data checks came from
night-research's notes (by message and branch `night-research`).
