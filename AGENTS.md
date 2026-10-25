# AGENTS.md - gnss-free-nav-V2

GNSS-denied quadcopter navigation in MuJoCo (Skydio X2): fly A -> B on the onboard estimate only,
localizing the down camera against an onboard map (OSM landmarks + USGS orthophoto).
V1 (browser prototype) is the separate repo `gnss-free-nav/` next to this one (gitignored here).

## Run

```
. scripts/env.ps1                                  # Windows (bash: . scripts/env.sh); venv + data live outside OneDrive
uv run python -m pytest tests -q                   # all tests (~2 min)
uv run python -m sfhack.missionprep serve          # mission planner page (A / B on a map); see docs/MAP_PACKAGE.md
uv run python demos/demo_live.py --mission berkeley  # fly a prepared mission
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
- **EKF** (`estimation/ekf.py`): [x, y, z, vx, vy, vz]. Visual fixes are gated by NIS with chi-square
  9.21 (2 dof, 99 %), logged in `ekf.nis_log`. After `nofix_inflate_after_s` without an accepted fix,
  a covariance floor of (0.5 m/s * seconds blind) keeps correct fixes from being gated out. After 5
  consecutive rejections the next fix is accepted with its noise inflated to NIS = gate (re-localization),
  logged as a `RELOCALIZE` decision.
- **Mock localizer** `mock_min_landmarks = 2`: at 40 m cruise only ~15 % of frames on the Berkeley route
  hold 3+ mapped landmarks, which left the EKF blind most of the flight.

## Mission end

- Touchdown (any descending phase): estimated height < `landed_alt` and |vz| < `touchdown_vz` for
  `touchdown_s` -> motors off, LANDED, zero-velocity updates from then on.
- Above B: descend only if the 2-sigma radius <= `precision_sigma2`, else hover up to `precision_wait_s`
  for a fix (`WAIT_FOR_FIX` / `DESCEND` decisions).
- Decisions go to `autopilot.decisions` and the `decision` bus topic. The runner adds the scored outcome
  `ARRIVED` / `MISSED` / `MISSION_TIMEOUT` with the final error (harness, truth-scored).

## Mission packages (decisions and why)

`sfhack/missionprep` (tooling / harness: it makes the truth) builds `$SFHACK_DATA/missions/<id>/{world,onboard,drone}`;
format and frames in `docs/MAP_PACKAGE.md` (the contract with V3.2 and the object placer: change both together).
`FlightSystem(mission=id)` / `demo_live.py --mission id`. Onboard code reads only the onboard package via `OnboardMap`.
- **z datum = ground under A.** Spawn, baro-relative EKF z and the old flat sample all stay at z = 0 at A; B and the
  route carry absolute z. Scoring compares truth z with the TRUE ground (`FlightSystem.true_ground_z`).
- **Launch pad:** the X2 calibrates its IMU on the ground assuming level. On A's 2.5 deg slope that put 0.4 m/s^2 of
  gravity into the accel bias and it flew off downhill. A is flattened in the truth DTM, and invisible contact pads in
  the terrain plane sit at A and B: MuJoCo hfield contacts left the X2 resting 9 deg tilted, and rocking after
  touchdown on B's 6.5 deg slope (the IMU never looked still).
- **Vision over terrain:** `map_localizer` projects each template cell at the onboard DSM height (roofs, slopes) and
  sizes the template by the distance to the surface below. A single ground plane put 20 m roofs at 2-3x scale at
  30 m AGL and fixes stopped mid-route. Without a DSM it is the old flat z = 0 (sample unchanged).
- **Route:** planned on the onboard DSM (Dijkstra, cost grows with the altitude a cell forces): around the Campanile
  instead of 132 m over it (vision needs < ~60 m). Onboard buildings removed from the map are only ones lower than
  the 30 m minimum cruise AGL.
- Results (vision, seeds 0-4): berkeley 5/5 ARRIVED, mean 0.79 m from B; campus-glade 5/5, mean 1.21 m (worst 2.14 m).

## Conventions

World = local ENU (x east, y north, z up), origin = map center, yaw 0 = east CCW. All parameters in
`sfhack/config.py`. Generated data goes to `$SFHACK_DATA`, never into the repo. English everywhere.

## Cursor Cloud specific instructions

- Source `scripts/env.sh` before `uv run`. The venv is `~/.venvs/sf-hack-v5` and generated data is `~/sf-hack-data`, both outside the repo. `bash scripts/setup.sh` is idempotent (uv Python 3.12 + `uv sync`).
- There is no GPU. On Linux with neither `/dev/dri` nor `/dev/nvidia0`, `scripts/env.sh` sets `MUJOCO_GL=osmesa` and `PYOPENGL_PLATFORM=osmesa`. Leave those set: GLFW/EGL return black frames here, and the vision localizer needs real down-camera images.
- `uv run python -m pytest tests -q` is the flight check (the Berkeley ortho ships in `sfhack/world/sample_berkeley`, no download). Headless score: `uv run python demos/demo_live.py --headless`.
- Mission planner: `uv run python -m sfhack.missionprep serve` then `http://127.0.0.1:8765/`. Do not `uv sync --extra detector` unless you need YOLO; that extra pulls CUDA PyTorch.
