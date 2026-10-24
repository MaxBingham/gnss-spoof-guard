# RESUME - branch night-nav (2026-10-24, night before the hackathon)

Previous resume (V5 mission planner): `docs/RESUME_v5-mission-planner.md`; night-thermal's: `docs/RESUME_night-thermal.md`.
What happened tonight, numbers, what failed: `REPORT_night-nav.md`. Design + tables + plots: `docs/NIGHT_NAV.md`.

## State

All committed on `night-nav` (not pushed). Includes the V3.2 merge and night-thermal (up to 0220f38).
Night defaults: thermal method `grad`, gyro-bias learning (`ahrs.tilt_bias_ki` 0.05 when `night=True`), no HOLD
mid-route. Result (frozen data, GNSS off at 20 %, 60 m AGL, 20 seeds): berkeley 20/20, m-c804bb84 20/20,
ukraine-1 18/20 within 5 m of B (58/60, Wilson 95 % 89-99 %).

## Setup (own venv, data outside the repo)

```powershell
. scripts\env.ps1                                                 # sets the V5 venv; override it:
$env:UV_PROJECT_ENVIRONMENT = "$HOME\.venvs\sf-hack-night-nav"
uv sync
uv run python -m pytest tests -q                                  # 68 tests (truth separation included)
```

## Night flight (GNSS off at 20 % of the route, thermal + TRN + IMU)

```powershell
# one seed, per-second trace (phase, health, |est-truth|, claimed 2-sigma, yaw error, fixes, AGL)
uv run python demos/21_night_eval.py --mission berkeley --seeds 1 --methods grad --trace
# final table rows (frozen mission copy: night-thermal rebuilds thermal files in place); --resume skips done flights
$snap = "$env:SFHACK_DATA\night\nav_snap"
uv run python demos/21_night_eval.py --mission ukraine-1,m-c804bb84,berkeley --seeds 20 --methods grad --tag N --mission-root $snap --workers 9 --resume
uv run python demos/21_night_eval.py --mission berkeley,m-c804bb84,ukraine-1 --seeds 20 --methods off --tag N --mission-root $snap --workers 9 --resume
uv run python demos/21_night_eval.py --mission berkeley,m-c804bb84 --seeds 20 --methods grad --contrast 0.4 --tag N_c04 --mission-root $snap --workers 9
# variants: any config field, e.g. --set ahrs.tilt_bias_ki=0 (stage-1 behaviour), --set thermal.method=ncc
# open-loop localizer benchmark (per method, perturbed priors)
uv run python demos/22_bench_thermal.py --mission berkeley,m-c804bb84 --seeds 2 --methods ncc,grad,road,junction --sigmas 3,10
# plots -> docs/night_nav/
uv run python demos/23_night_plots.py --mission berkeley,m-c804bb84 --tags F_off,N_off,F_ncc,F_grad,N_grad --out docs/night_nav
```

Keep <= 9 workers on this laptop: each sim worker commits 2-3 GB and the Windows commit charge ran out tonight.

In code: `FlightSystem(cfg, mission=id, night=True, cruise_agl=60.0, conditions=Conditions().update({"gnss.mode": "ok"}))`,
then `fs.update_conditions({"gnss.mode": "off"})`. Method: `cfg.thermal.method` (`off` = IMU + TRN only).
HUD helpers: `fs.autopilot.night_nav_source(t)`, `fs.autopilot.pose().sigma2_horizontal()`, `fs.thermal_sensor.last`,
`fs.autopilot.last_thermal_fix`, `fs.autopilot.thermal.last_tpl` (rectified template).

## Data and housekeeping

- Per-flight results: `$SFHACK_DATA/night/nav_eval/<mission>/<tag>/seed*.json` (per-second trace, every thermal fix,
  decisions). Tags: `F_*` stage 1 (before the gyro-bias fix), `N_*` final, `F_ki05_grad` / `F_rw04_grad` /
  `F_cov4_grad` / `G_reloc3_grad` variants, `FINALDATA_grad` the check on night-thermal's final data.
- Frozen mission copy: `$SFHACK_DATA/night/nav_snap/{berkeley,m-c804bb84,ukraine-1}` (world + onboard).
- Eval worktree `C:\dev\night-nav-evalsnap` (detached HEAD, used so edits did not leak into running evals). Remove
  when done: `git worktree remove C:\dev\night-nav-evalsnap`; it left one `git stash` entry (a copy of the eval
  harness, safe to drop: `git stash drop`).

## Next

See "Next steps" in `REPORT_night-nav.md` (blind-stretch recovery, lidar floor for terrain following, ROADS route
for the ukraine demo, rotor drag, yaw feedback).
