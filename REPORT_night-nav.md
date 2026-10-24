# REPORT - branch night-nav (agent night-nav, night of 2026-10-23/24)

Goal (NIGHT_PLAN): the drone reaches B at night with GNSS completely off, on IMU + baro + lidar TRN + a downward
thermal camera matched against a thermal prior predicted from daytime imagery. Success = within 5 m of B, 20 seeds,
95 % Wilson interval. Landing not tonight.

STATUS: DRAFT - final numbers are filled in when the evaluation queue finishes.

## What works

- **V3.2 merged** (`origin/v3.2-gnss-free` -> night-nav): GNSS-free track + handover, capped bias step, re-localization,
  TRN. Kept all mission-planner features; env.ps1 / env.sh keep the V5 venv lines. TRN now uses the mission's
  `onboard/dsm.npz` when the package has one (`world/onboard_dsm.build_dsm`).
- **No HOLD mid-route** (`health.keep_route_when_blind`, default on): blind = DEGRADED "BLIND, keep route", full speed,
  the 2-sigma keeps growing in the HUD. V3.2 rule available with the flag off.
- **Night mode**: `FlightSystem(night=True, cruise_agl=60)` - RGB vision off, thermal camera (night-thermal's renderer,
  in the simulator's GL context, per-seed camera errors) on the `camera/thermal` topic, `ThermalLocalizer` onboard,
  fixes fused like vision (NIS gate) into the main filter and the GNSS-free track. 60 m AGL route re-planned over the
  onboard DTM (`planning.mission.cruise_at_agl`).
- **ThermalLocalizer** (`sfhack/perception/thermal_localizer.py`): ncc, grad (default), road, junction and
  combinations; peak-region covariance (ridge = 1-D fix), second-peak test, yaw search, big-jump confirmation.
- **Harness**: `demos/21_night_eval.py` (closed loop, GNSS off at 20 % of the route, Wilson CI, per-flight JSON with
  per-second trace, resumable, survives a broken process pool), `demos/22_bench_thermal.py` (open-loop localizer
  bench), `demos/23_night_plots.py`.
- Tests: 63 green before the last commits (incl. truth separation; `tests/test_thermal_nav.py` new: 7 tests).

## Numbers

(final table below when complete; see docs/NIGHT_NAV.md for the method details and the open-loop bench)

## What failed / caveats

- **IMU + TRN alone is not enough in this simulator.** Calm air has no airframe drag, so the accelerometer sees neither
  velocity nor the AHRS tilt error in closed loop; after GNSS loss the tilt drifts ~1 deg per 10-20 s, the gravity
  leak grows the error quadratically and the controller flies the real drone after the wrong estimate (km-level
  runaways, ukraine-1 always). TRN helps over the berkeley buildings, not on GLO-30 terrain (all 35 TRN fixes on
  ukraine-1 seed 0 were 100-900 m off and NIS-rejected).
- **ncc alone / ncc+grad**: occasional wrong fixes -> runaways (berkeley 4/20 resp. 2/20).
- **grad**: one runaway on m-c804bb84 (seed 0): EKF overconfident (claimed 2-sigma 2.5 m) while the AHRS tilt drifted,
  velocity error 0.4 -> 3 m/s in 10 s with thermal fixes accepted at small gain, then TRN fixes 8-14 m off passed
  the gate and correct thermal fixes were rejected. Candidate fix: `thermal.ekf_acc_bias_rw = 0.04` (being evaluated).
- **road / junction** alone are weak in these worlds (see the bench): campus and village roads are short, partly
  under trees, plazas / parking are as warm as roads.
- Data versions: night-thermal rebuilt the thermal files in place several times tonight. The final table runs on a
  frozen copy (`$SFHACK_DATA/night/nav_snap`, taken 00:55 after night-thermal 3e692ed: cold-air pooling, per-camera
  offset / gain, BEFORE the moving-traffic commit 40f5393 and the final 0.7 contrast rebuild at 01:17).
  night-demo's final eval uses the later data (traffic, contrast 0.7) with this branch at 4904433.
- Machine: the Windows commit charge ran out (~80 GB with ~27 sim workers across agents); worker processes died
  (BrokenProcessPool). The harness now resubmits; flights are resumable.

## Next steps

(filled in at the end)

## Disclosure (pre-hackathon work)

Everything on this branch since `28962ad` was built tonight (2026-10-23 23:30 -> 2026-10-24 07:00) by an AI agent
(Claude Code, agent "night-nav") before the hackathon: the V3.2 merge, night mode, thermal sensor wiring, the thermal
localizer, the eval / bench / plot scripts, the tests and the docs. The thermal world model and camera
(`sfhack/thermal/`) are night-thermal's work (merged), research notes are night-research's.
