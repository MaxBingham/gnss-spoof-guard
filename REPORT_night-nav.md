# REPORT - branch night-nav (agent night-nav, night of 2026-10-23/24)

Goal (NIGHT_PLAN): the drone reaches B at night with GNSS completely off, on IMU + baro + lidar TRN + a downward
thermal camera matched against a thermal prior predicted from daytime imagery. Success = within 5 m of B, 20 seeds,
95 % Wilson interval. Landing not tonight. Details, tables and plots: `docs/NIGHT_NAV.md`.

## Headline numbers

Night 23:00, 60 m AGL, GNSS completely off from 20 % of the route, 20 seeds each, frozen mission data (see
"Data versions"), default night configuration (gradient-orientation matching + gyro-bias learning):

| mission | within 5 m | 95 % Wilson CI | median / max error at B (m) | same without gyro-bias learning | IMU + TRN only |
|---|---|---|---|---|---|
| berkeley (569 m) | **20/20** | 84-100 % | 0.4 / 2.2 | 20/20 | 10/20 |
| m-c804bb84, hill (923 m) | **20/20** | 84-100 % | 0.7 / 1.6 | 19/20 | 0/20 |
| ukraine-1 (1870 m) | **18/20** | 70-97 % | 1.2 / 4143 (2 runaways) | 12/20 | 0/20 |
| all | **58/60** | 89-99 % | | 51/60 | 10/60 |

IMU + TRN with the same gyro-bias learning: 11/20, 3/20, 0/20 (no km runaways on berkeley / hill any more, median
3 km on ukraine-1).

Sanity check on night-thermal's final data (traffic, contrast 0.7), 2 seeds each: berkeley 2/2, hill 2/2,
ukraine-1 1/2. night-demo's independent 20-seed run of this fix on the final data (their numbers): berkeley 20/20,
hill 20/20, ukraine-1 with their ROADS route 20/20 (15/20 without the fix).

Mild night (true scene contrast 0.4x, prior unchanged): berkeley 20/20, m-c804bb84 20/20 (median 0.6 / 0.8 m).

Default = `grad` (gradient-orientation NCC of the rectified thermal frame vs the predicted thermal map) +
`ahrs.tilt_bias_ki` 0.05 in night mode (commit 651b398). Plots: `docs/night_nav/night_error_<mission>.png`.

## What works

- **V3.2 merged** (`origin/v3.2-gnss-free`): GNSS-free track + handover, capped bias step, re-localization, TRN.
  All mission-planner features kept; env.ps1 / env.sh keep the V5 venv lines. TRN uses the mission's
  `onboard/dsm.npz` when the package has one (`world/onboard_dsm.build_dsm`).
- **No HOLD mid-route** (`health.keep_route_when_blind`, default on): blind = DEGRADED "BLIND, keep route", full
  speed, the 2-sigma keeps growing in the HUD. The V3.2 rule is the flag off.
- **Night mode** `FlightSystem(night=True, cruise_agl=60)`: RGB vision off, thermal camera (night-thermal's
  renderer in the simulator's GL context, per-seed camera errors) on `camera/thermal`, `ThermalLocalizer` onboard,
  fixes fused like vision (NIS gate, big-jump confirmation) into the main filter and the GNSS-free track; 60 m AGL
  route re-planned over the onboard DTM; gyro-bias learning on (`ahrs.tilt_bias_ki` 0.05).
- **ThermalLocalizer** (`sfhack/perception/thermal_localizer.py`): ncc, grad (default), road, junction and
  combinations; template on the optical axis, peak-region covariance (ridge = 1-D fix), second-peak test, yaw search.
- **Harness**: `demos/21_night_eval.py` (closed loop, Wilson CI, per-flight JSON incl. per-second trace and every
  thermal fix, `--mission-root` for frozen data, `--resume`, survives a broken process pool),
  `demos/22_bench_thermal.py` (open-loop localizer bench), `demos/23_night_plots.py`.
- **HUD helpers for night-demo**: `Autopilot.night_nav_source(t)` -> (GNSS | THERMAL | TRN | INS, fix ages).
- **Tests**: 71/71 green at the final commit (incl. truth separation); new `tests/test_thermal_nav.py` (7 tests: localizer on a synthetic
  field for ncc / grad / road / ncc+road, uniform-image rejection, 60 m AGL route, keep-route health).

## Method comparison (closed loop, before the gyro-bias fix)

grad 20/20 (berkeley), 19/20 (hill), 12/20 (ukraine-1); ncc 16/20, 15/20; ncc+grad 18/20, 19/20;
IMU + TRN 10/20, 0/20, 0/20; road alone 2/10, 1/10 and junction alone 6/10, 0/10 (berkeley, hill; with the
gyro-bias fix). Open-loop bench: grad 66-79 % fix rate, median 0.5-0.65 m; road 35-47 % with median
5-9 m; junction 15-23 %. Full tables in `docs/NIGHT_NAV.md`.

## What failed / caveats

- **IMU + TRN alone is not enough in this simulator.** No airframe drag in calm air: the accelerometer sees neither
  velocity nor the AHRS tilt error in closed loop; after GNSS loss the tilt drifts ~1 deg per 10-20 s and the error
  grows quadratically (km-level runaways; ukraine-1 always). TRN helps over berkeley buildings only.
- **Blind stretches are the real risk**: the ukraine-1 straight route crosses a ~250 m forest with a uniform prior
  (~30 s without information). Before the gyro-bias fix 8/20 ukraine flights ran away there; after it 2/20 still do
  (seeds 5 and 9; seed 9 had arrived without the fix, so the gain is not uniform). Seed 5: a wrong forced
  re-localization before the forest, then a wrong match after it; it also descends to 26 m AGL because terrain
  following uses the wrong position - a real safety issue for terrain-following at night.
- **Tried and rejected**: re-localization guard (`thermal.reloc_min_span_s` 3 s; ukraine seeds 5 and 9 still run
  away), EKF accel-bias random walk x4 (wobble, flights stalled at waypoints), thermal fix
  covariance x4 (neutral), `yaw_feedback` (implemented, not evaluated; the compass bias is constant ~0.8 deg, there
  is no yaw drift).
- **road / junction alone** are weak in these worlds: short campus / village roads, trees over roads, plazas and
  parking lots as warm as roads. Kept as options.
- **Data versions**: night-thermal rebuilt the thermal files in place several times tonight. All my tables run on a
  frozen copy `$SFHACK_DATA/night/nav_snap` (taken ~00:55 after night-thermal 3e692ed: cold-air pooling, per-camera
  offset / gain; BEFORE moving traffic 40f5393 and the final 0.7 contrast rebuild at 01:17). night-demo's final eval
  uses the later data with this branch at 4904433 (without the gyro-bias fix; they got the fix commit 651b398).
  Note: in the first runs (before 3e692ed was merged) the thermal camera seed was not passed, so all seeds shared
  the seed-0 camera noise; the final tables do not have that problem.
- **Frozen code**: evaluations ran from a separate worktree `C:\dev\night-nav-evalsnap` (detached at the evaluated
  commit, so edits during the night did not leak into running evaluations): stage 1 at 883bd03 (+ the resilient
  harness, behaviour-identical), final rows at 651b398. Eight ukraine-1 seeds of the final grad row (0-7) were
  flown in the selection run with `--set ahrs.tilt_bias_ki=0.05`, which is the identical configuration, and reused.
- **Machine**: the Windows commit charge ran out (~80 GB with ~27 sim workers across agents, ~2-3 GB commit each);
  worker processes died (BrokenProcessPool). Agreed with night-demo: night-nav <= 9 workers, night-demo 6.

## Next steps

1. For the demo: fly ukraine-1 on night-demo's ROADS route (20/20 in their run with this fix) - the straight
   route's forest is the remaining failure case.
2. Blind-stretch recovery (ukraine-1 seeds 5, 9): search window from the distance flown blind x the velocity
   uncertainty, and a slower handover from the INS after a blind stretch (accept only fixes that agree over several
   seconds). The simple re-localization guard was not enough.
3. Terrain following on the estimate is unsafe when the estimate runs away (seed 5 went down to 26 m AGL): use the
   lidar height above ground as a floor for the altitude command.
4. Rotor-drag model + drag-velocity fusion (death-v3) would let IMU + TRN degrade gracefully; needs a team
   decision (changes baseline physics).
5. `yaw_feedback` closed-loop check (the ~0.8 deg compass bias is constant; thermal yaw corrections could remove it).
6. Full 20-seed rows of this branch on night-thermal's final data are in night-demo's eval; mine has 2 seeds each.

## Disclosure (pre-hackathon work)

Everything on this branch since `28962ad` was built tonight (2026-10-23 23:30 -> 2026-10-24 07:00) by an AI agent
(Claude Code, agent "night-nav"), before the hackathon: the V3.2 merge, night mode, thermal sensor wiring, the
thermal localizer, the gyro-bias learning, the eval / bench / plot scripts, the tests and the docs. The thermal
world model and camera (`sfhack/thermal/`) are night-thermal's work (merged in), the research notes
night-research's. V3.2 itself (`origin/v3.2-gnss-free`) predates tonight.
