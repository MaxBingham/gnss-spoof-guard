# CLAUDE.md - gnss-free-nav-V2

GNSS-denied quadcopter navigation in MuJoCo (Skydio X2): fly A -> B on the onboard estimate only,
localizing the down camera against an onboard map (OSM landmarks + USGS orthophoto).
V1 (browser prototype) is the separate repo `gnss-free-nav/` next to this one (gitignored here).

## Run

```
. scripts/env.ps1                                  # Windows (bash: . scripts/env.sh); venv + data live outside OneDrive
uv run python -m pytest tests -q                   # all tests (~2 min)
uv run python demos/demo_live.py                   # LIVE DEMO (viewer); --headless / --record; see DEMO.md
uv run python demos/02_mission_ekf.py              # default Berkeley mission, EKF + mock localizer
uv run python demos/06_ekf_diagnostics.py          # dead reckoning only, plot -> $SFHACK_DATA/diagnostics
uv run python demos/07_eval_seeds.py --seeds 10    # success rate over seeds (parallel)
```

## HARD RULE: truth separation

Onboard code (`autopilot.py`, `estimation/`, `control/`, `planning/`, `perception/`, `world/onboard_map.py`)
never imports the simulator or `sim/truth.py`. Truth leaves the simulator only via `sfhack/sim/truth.py`:
- `TruthPoseSource`: the "truth" pose-source switch (controller tuning, demo 01, demo 06),
- `perception/mock_localizer.py`: marked `# SIMULATION OF PERCEPTION - uses ground truth`,
- harness code that only scores / displays (`runner.py`, demos, tests).
`tests/test_truth_separation.py` enforces it. Keep it green; never weaken it.
The degradation levers (`sfhack/degradation/`, `Conditions`: GNSS spoof / jam, weather, wind, camera and
lidar loss) are simulated reality too: only the simulator and harness read them; onboard code never imports
`degradation` or the `conditions` topic (same test). The onboard GNSS fusion is the NAIVE baseline on purpose
(every fix fused, no gate); see README "Degradation levers" and `demos/09_degradation_scenarios.py`.

## Estimation (decisions and why)

- **AHRS** (`estimation/ahrs.py`): complementary filter on onboard sensors only. Gyro integration (bias
  calibrated on the ground during `preflight_s`), accelerometer roll/pitch correction, compass yaw.
- **Root cause of the km-level drift (fixed):** `tilt_gain` was 0.02/s. The steady-state tilt error is
  residual gyro bias / tilt_gain, ~0.8 deg, and every degree of roll/pitch error leaks 0.17 m/s^2 of
  gravity into the horizontal acceleration the EKF integrates (142 m after 60 s, km after landing
  without zero-velocity updates). Now tilt_gain 0.3, applied only in steady flight (| |a| - g | < 0.05
  and |omega| < 0.1 rad/s, since a multirotor's accelerometer sees gravity only when not accelerating),
  preflight calibration 4 s. Dead reckoning (demo 06, seed 0): 14 m after 30 s, 70 m after 60 s
  (`tests/test_dead_reckoning.py` asserts < 20 m / < 80 m).
- Remaining dead-reckoning error: compass bias (~1 deg std) rotates every velocity change, plus no
  accel-bias states in the EKF (bias is calibrated on the ground only). Upgrade path: error-state EKF.
- **Closed-loop drift (fixed 2026-10-22):** flying on the EKF, the AHRS tilt error is unobservable for
  the accelerometer (the controller holds the drone tilted by exactly the error, so the specific force
  looks vertical in the estimated frame); the raw-sample tilt gate also passed only 1-8 % of the time
  (66 % on truth). Tilt error grew to ~2 deg, the leak drove 100+ m errors. Now the EKF has horizontal
  accel-bias states (bx, by) learned from the fixes, and the autopilot hands them to the AHRS every IMU
  step (`Ekf.take_tilt_correction` -> `Ahrs.correct_tilt_world`, error-state reset). 10 seeds: mean
  final error 325 m -> 32 m. Blind stretches still drift (residual gyro bias ~0.5 deg / 30 s ~ 35 m),
  so landing precision needs fixes all the way down.
- **EKF** (`estimation/ekf.py`): [x, y, z, vx, vy, vz, bx, by]. Visual fixes are gated by NIS with chi-square
  9.21 (2 dof, 99 %), logged in `ekf.nis_log`. After `nofix_inflate_after_s` without an accepted fix,
  a covariance floor of (0.5 m/s * seconds blind) keeps correct fixes from being gated out. After 5
  consecutive rejections the next fix is accepted with its noise inflated to NIS = gate (re-localization),
  logged as a `RELOCALIZE` decision.
- **Vision localizer** (`perception/map_localizer.py`, `--localizer vision`, the demo default): ortho-
  rectify the down camera with AHRS attitude + EKF height, NCC (`cv2.matchTemplate`) against the onboard
  ortho within 3 sigma of the prior. No fix when: score < 0.5, a second LOCAL max is within 0.08 (the
  shoulders of a broad true peak are not), the template is uniform, or the prior is too wide for the
  search window at that altitude (a capped window gave confident 40 m-wrong fixes). Works ~4-40 m.
  Benchmark: `demos/08_bench_localizer.py`. 10 seeds closed loop: 10/10 within 3 m, mean 0.8 m.
- **Roofs** (`world/scene.py`): each building gets a roof plane textured with the ortho under its
  footprint. Gray box tops rendered white and left the camera blind over the whole campus core
  (health HOLD -> LAND); a real roof looks like its aerial photo.
- **Mock localizer** is blind near B (no OSM landmarks there): the final descent is dead reckoning and
  lands 5-50 m off. Closed-loop dead reckoning has a neutral mode (drag-limited drift ~2-3 m/s), so
  landing precision needs fixes down to ~4 m. Keep `mock_min_landmarks = 2`: at 40 m cruise only ~15 % of frames on the Berkeley route
  hold 3+ mapped landmarks, which left the EKF blind most of the flight.

## Mission end

- Touchdown (any descending phase): downward rangefinder (sim sensor, own RNG stream, at-rest reading
  calibrated in preflight) < `touchdown_range` -> GROUND_CONTACT, motors to uniform idle; still low and
  IMU stationary for `touchdown_s` -> motors off, LANDED, zero-velocity updates. The idle stage matters:
  with position control on, the drone hopped and skidded on the ground chasing its estimate and the IMU
  never looked still. Fallback without rangefinder: estimated height < `landed_alt`.
- Flare at 4 m, land at 1 m/s (vision stops at ~4 m); DESCEND/LAND pause while the estimate is > 1.5 m
  from B horizontally (`RECENTER`).
- Above B: descend only if the 2-sigma radius <= `precision_sigma2`, else hover up to `precision_wait_s`
  for a fix (`WAIT_FOR_FIX` / `DESCEND` decisions).
- Decisions go to `autopilot.decisions` and the `decision` bus topic. The runner adds the scored outcome
  `ARRIVED` / `MISSED` / `MISSION_TIMEOUT` with the final error (harness, truth-scored).

## Conventions

World = local ENU (x east, y north, z up), origin = map center, yaw 0 = east CCW. All parameters in
`sfhack/config.py`. Generated data goes to `$SFHACK_DATA`, never into the repo. English everywhere.
