# RESUME - branch night-nav (2026-10-24, night before the hackathon)

Previous resume (V5 mission planner): `docs/RESUME_v5-mission-planner.md`; night-thermal's: `docs/RESUME_night-thermal.md`.
What happened tonight, numbers, what failed: `REPORT_night-nav.md`. Design + tables: `docs/NIGHT_NAV.md`.

## Setup (own venv, data outside the repo)

```powershell
$env:UV_PROJECT_ENVIRONMENT = "$HOME\.venvs\sf-hack-night-nav"   # env.ps1 keeps the V5 venv line; override it
. scripts\env.ps1
$env:UV_PROJECT_ENVIRONMENT = "$HOME\.venvs\sf-hack-night-nav"
uv sync
uv run python -m pytest tests -q                                  # 63 tests
```

## Night flight (GNSS off at 20 % of the route, thermal + TRN + IMU)

```powershell
# one seed, per-second trace
uv run python demos/21_night_eval.py --mission berkeley --seeds 1 --methods grad --trace
# the evaluation table (frozen mission copy, because night-thermal rebuilds thermal files in place)
uv run python demos/21_night_eval.py --mission berkeley,m-c804bb84 --seeds 20 --methods grad,off,ncc,ncc+grad `
    --mission-root "$env:SFHACK_DATA\night\nav_snap" --tag F --workers 11
uv run python demos/21_night_eval.py --mission ukraine-1 --seeds 20 --methods grad,off --mission-root "$env:SFHACK_DATA\night\nav_snap" --tag F
uv run python demos/21_night_eval.py --mission berkeley,m-c804bb84 --seeds 20 --methods grad --contrast 0.4 --tag F_c04 --mission-root "$env:SFHACK_DATA\night\nav_snap"
# open-loop localizer benchmark (per method, perturbed priors)
uv run python demos/22_bench_thermal.py --mission berkeley,m-c804bb84 --seeds 2 --methods ncc,grad,road,junction --sigmas 3,10
# plots for docs/night_nav/
uv run python demos/23_night_plots.py --mission berkeley,m-c804bb84,ukraine-1 --tags F_off,F_grad --out docs/night_nav
```

In code: `FlightSystem(cfg, mission=id, night=True, cruise_agl=60.0, conditions=Conditions().update({"gnss.mode": "ok"}))`,
then `fs.update_conditions({"gnss.mode": "off"})`. Method: `cfg.thermal.method` (default `grad`; `off` = IMU + TRN).
HUD helpers: `fs.autopilot.night_nav_source(t)`, `fs.autopilot.pose().sigma2_horizontal()`, `fs.thermal_sensor.last`.

## Results so far

See `REPORT_night-nav.md` (final table) and `$SFHACK_DATA/night/nav_eval/<mission>/<tag>/seed*.json` (per-flight
JSON incl. per-second trace and every thermal fix).

## Next

See "Next steps" in `REPORT_night-nav.md`.
