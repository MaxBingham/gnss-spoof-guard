# Night navigation without GNSS (branch night-nav)

Pre-hackathon work, night of 2026-10-23/24 (see the disclosure in `REPORT_night-nav.md`).

## Scenario

Summer night 23:00, clear, no wind, 60 m AGL. GNSS on at takeoff; at 20 % of the route length (measured on the
true position, harness) the receiver goes completely off (`gnss.mode = off`: no message at all). From then on the
drone has IMU, barometer, nadir lidar (TRN on the onboard DSM) and a downward thermal camera (640 x 512, 50 deg
HFOV, NETD 50 mK, per-camera offset / gain error, fixed-pattern noise, drift with flat-field correction:
night-thermal's renderer). The RGB camera is dark and off. The drone never holds mid-route
(`health.keep_route_when_blind`). Scored when it commits to B (start of the descent, after up to 10 s of waiting
for a fix above B): horizontal distance truth <-> B. Landing is not part of tonight's scenario.

## How it works

```
thermal frame (K, 640x512)
  -> ortho-rectify: every 0.5 m cell of a north-up square around the optical axis' ground point is projected
     into the image at its onboard-DSM height (AHRS attitude, height above the surface from the nadir lidar),
     after area-averaging the image to ~the cell size                       -> template (62 x 62 cells = 31 m at 60 m)
onboard/thermal_prior.npz (23:00 temperatures predicted from daytime imagery + OSM, night-thermal)
  -> region around the EKF estimate, radius 3 sigma + 6 m (8..60 m)
match (method below, small yaw search +-3 deg with parabolic refinement)
  -> peak region (cells within 0.05 of the best score): its weighted spread is the covariance
     (a straight road or edge gives a ridge = 1-D fix, a corner / crossing a compact 2-D peak);
     rejected when the ridge leaves the window, the score is low, or a separate local maximum is within 0.06
  -> big-jump confirmation: a fix that moves the estimate > 5 m needs a second fix within 2 s with the same
     innovation (+-3 m)
  -> EKF NIS gate (chi2 9.21, 2 dof), then the same fix into the GNSS-free track
```

Code: `sfhack/perception/thermal_localizer.py` (onboard, no truth), `sfhack/sim/thermal_sensor.py` (simulator side,
renders with `sfhack.thermal.camera` in the simulator's GL context), `Autopilot.on_thermal`,
`FlightSystem(night=True, cruise_agl=60)`, parameters `Config.thermal` (`ThermalConfig`).
GNSS on: the GNSS-free track gets its own thermal search around its own prior (as V3.2 does for vision), so the
handover at GNSS loss lands on a thermal-only estimate. GNSS off: one search, both filters fuse it.

## Methods compared

| method | what it matches | fix type |
|---|---|---|
| `ncc` | zero-mean NCC of the rectified temperatures vs the predicted temperatures | 2-D or 1-D (peak shape) |
| `grad` (default) | NCC on 4 gradient-orientation channels mod 180 deg (CFOG-like), averaged: edges match even where the prior got the sign or level wrong (roofs: mostly cold, 1/3 warm; per-camera offset / gain) | 2-D or 1-D |
| `road` | warm elongated structures (local contrast, elongated components) thinned (Zhang-Suen) and matched by a symmetric truncated chamfer distance to the thinned road mask of the prior | 1-D on a straight road, 2-D at curves / crossings |
| `junction` | branch points (>= 3) of the observed centerlines vs those of the prior's road skeleton, unique nearest neighbour inside the gate | 2-D, sparse |
| `a+b` | all candidates computed; the largest mutually consistent set (pairwise chi2 < 9.21) is fused information-weighted | |
| `off` | no thermal: IMU + baro + TRN only (baseline) | |

## Open-loop benchmark (`demos/22_bench_thermal.py`)

The drone flies the route at 60 m AGL on GNSS; every 3rd thermal frame in cruise each method gets a prior shifted
by N(0, sigma) with covariance sigma^2. Scored against the true camera position. 2 seeds per mission, mission
data of night-thermal f65f560 (one calibration before the final eval data). "NIS ok" = share of fixes whose
error is consistent with their own covariance (Mahalanobis^2 < 9.21; 99 % expected).

| mission | method | prior sigma | fix rate | error median / p90 (m) | > 5 m off | NIS ok |
|---|---|---|---|---|---|---|
| berkeley | ncc | 3 m | 50 % | 0.68 / 8.1 | 16 | 88 % |
| berkeley | grad | 3 m | 74 % | 0.59 / 3.4 | 11 | 91 % |
| berkeley | grad | 10 m | 66 % | 0.65 / 7.2 | 18 | 87 % |
| berkeley | ncc+grad | 3 m | 79 % | 0.62 / 4.8 | 18 | 88 % |
| berkeley | road | 3 m | 47 % | 5.4 / 18.3 | 57 | 73 % |
| berkeley | junction | 3 m | 23 % | 2.4 / 8.7 | 10 | 78 % |
| m-c804bb84 | ncc | 3 m | 73 % | 0.58 / 2.7 | 12 | 96 % |
| m-c804bb84 | grad | 3 m | 77 % | 0.51 / 2.8 | 18 | 93 % |
| m-c804bb84 | grad | 10 m | 79 % | 0.55 / 4.2 | 29 | 90 % |
| m-c804bb84 | ncc+grad | 3 m | 84 % | 0.57 / 3.6 | 25 | 90 % |
| m-c804bb84 | road | 3 m | 35 % | 8.5 / 18.2 | 85 | 76 % |
| m-c804bb84 | junction | 3 m | 15 % | 6.0 / 10.2 | 35 | 45 % |

(238 / 388 frames per row. Road numbers are after the symmetric-chamfer fix; one-sided chamfer was 11-13 m.)

What the benchmark changed:
1. **Template on the optical axis.** A template centred under the camera left the 56 x 45 m footprint at ~5 deg
   pitch: almost no fixes in cruise, and the first closed-loop flight ran away (7 km).
2. **Covariance from the peak region.** With a round covariance from the curvature on the two grid axes, ridges
   (one straight road or roof edge) were reported as sharp 2-D fixes: 54 of 142 berkeley NCC fixes were > 5 m off
   at sigma 3 m. With the peak-region covariance: 16 of 120.
3. **Gradient orientation beats raw temperature** where the prior's temperature level is wrong (berkeley roofs,
   prior-vs-truth correlation 0.64): fix rate 74 vs 50 %, p90 3.4 vs 8.1 m. On the hill mission (correlation 0.77)
   both are close.
4. **Road / junction alone are weak here.** Campus and village roads are short, partly under trees, and plazas /
   parking lots are as warm as roads; most footprints show no clean road network. They stay as options.
5. **Yaw.** The yaw search sign was inverted at first (yaw correction error 2.3 deg vs 0.95 deg compass error);
   fixed with a synthetic test. Grad's yaw correction (0.6-0.9 deg median error) is about as good as the compass
   (0.9-1.0 deg bias, constant per flight; the compass anchors yaw, so there is no yaw DRIFT). Feeding it back
   (`thermal.yaw_feedback`) is implemented, off by default. Closed loop (final config + yaw_feedback, 10 seeds):
   berkeley 10/10 (median 0.4 m), m-c804bb84 10/10 (1.0 m); mean |yaw error| 0.78 -> 0.32 deg and 0.76 -> 0.54
   deg. ukraine-1: 6/9 (same seeds without it: 8/9), so it stays off

## Closed-loop results

`demos/21_night_eval.py`, 60 m AGL, GNSS off at 20 % of the route, 20 seeds, mission data frozen at
`$SFHACK_DATA/night/nav_snap` (night-thermal 3e692ed: cold-air pooling, per-camera offset / gain errors; no moving
traffic, contrast 1.0). "Within 5 m" = horizontal truth distance to B when the drone commits to the descent.
Runaway = > 50 m. Route lengths: berkeley 569 m, m-c804bb84 923 m (hill), ukraine-1 1870 m (straight line; it
crosses a ~250 m forest, x = -150..+100 m, where the prior is uniform: ~30 s without any thermal information).

### Final configuration (night defaults: `grad` + gyro-bias learning)

| mission | route | within 5 m | 95 % Wilson CI | error at B mean / median / max (m) | runaways (> 50 m) | max est. error over the flight, worst seed (m) | thermal fixes accepted / flight |
|---|---|---|---|---|---|---|---|
| berkeley | 569 m | **20/20** | 84-100 % | 0.5 / 0.4 / 2.2 | 0 | 23.7 | 318 |
| m-c804bb84 (hill) | 923 m | **20/20** | 84-100 % | 0.9 / 0.7 / 1.6 | 0 | 21.4 | 513 |
| ukraine-1 | 1870 m | **18/20** | 70-97 % | 236 / 1.2 / 4143 | 2 (seeds 5, 9) | 8446 | 912 |
| all three | | **58/60** | 89-99 % | | 2 | | |

The compass yaw error stays at its per-flight bias (mean |yaw error| 0.78 deg, no drift). Thermal processing 30-70 ms
per frame on berkeley / hill, 130-180 ms on ukraine-1 (larger search windows), at 5 Hz.

Mild night (`--contrast 0.4`: the TRUE scene's contrast scaled to 0.4x around air temperature, ~2-4 K road-grass
as in night-research's ASTER notes, while the prior keeps the full model contrast):

| mission | within 5 m | 95 % Wilson CI | error at B mean / median / max (m) | thermal fixes accepted / flight |
|---|---|---|---|---|
| berkeley | **20/20** | 84-100 % | 0.6 / 0.6 / 1.1 | 296 |
| m-c804bb84 | **20/20** | 84-100 % | 0.8 / 0.8 / 1.6 | 515 |

Normalized gradient matching does not care about the amplitude (NETD 50 mK is far below 2 K); what matters is
that the structure is where the prior says. (ukraine-1 at 0.4x not run: time.)

Baseline with the same gyro-bias learning (IMU + baro + TRN, no thermal), 20 seeds:

| mission | within 5 m | 95 % Wilson CI | error at B median / max (m) | without gyro-bias learning |
|---|---|---|---|---|
| berkeley | 11/20 | 34-74 % | 4.2 / 351 | 10/20, max 5944 m |
| m-c804bb84 | 3/20 | 5-36 % | 16.4 / 105 (3 timeouts) | 0/20, median 29.9 m, max 6107 m |
| ukraine-1 | 0/20 | 0-16 % | 3058 / 4941 | 0/20, median 6431 m |

The gyro-bias learning makes dead reckoning degrade more gracefully (no km runaways on berkeley / hill), but
without absolute fixes it is no substitute for the thermal matching.

Sanity check on night-thermal's FINAL data (moving traffic, truth contrast 0.7, rebuilt 01:17; the live mission
folders, latest camera code, this branch at 4e2f977), only 2 seeds each: berkeley 2/2 (0.1 / 0.4 m), m-c804bb84
2/2 (0.8 / 1.4 m), ukraine-1 1/2 (0.8 m; seed 1 ran away). Consistent with the frozen-data table, too few seeds
for more.

Independent run by night-demo (their numbers, reported to me at ~04:50; night-demo b5409f4 = this branch's 651b398
+ night-thermal 146a89f + their night route choice; frozen copy of the FINAL data with traffic and contrast 0.7;
error at the ABOVE_B decision), 20 seeds each: berkeley 20/20 (mean 0.4 m, max 0.9), m-c804bb84 20/20 (0.9 / 2.0),
ukraine-1 on the ROADS route 20/20 (1.9 / 2.5). Without the gyro-bias fix (4904433): 20/20, 20/20, 15/20.
ukraine-1 on the straight route with the fix: 5/6 at that time (one runaway at the forest).

### Stage 1: method comparison (before the gyro-bias fix)

| mission | row | within 5 m | 95 % Wilson CI | error at B median / max (m) | runaways | thermal fixes accepted / flight |
|---|---|---|---|---|---|---|
| berkeley | **grad** | **20/20** | 84-100 % | 0.8 / 1.5 | 0 | 313 |
| berkeley | ncc+grad | 18/20 | 70-97 % | 0.9 / 6921 | 2 | 309 |
| berkeley | ncc | 16/20 | 58-92 % | 1.2 / 7141 | 4 | 222 |
| berkeley | IMU + TRN only | 10/20 | 30-70 % | 5.1 / 5944 | 2 | - |
| m-c804bb84 | **grad** | **19/20** | 76-99 % | 1.4 / 76 | 1 | 495 |
| m-c804bb84 | ncc+grad | 19/20 | 76-99 % | 1.2 / 73 | 1 | 538 |
| m-c804bb84 | ncc | 15/20 | 53-89 % | 1.8 / 6388 | 5 | 431 |
| m-c804bb84 | IMU + TRN only | 0/20 | 0-16 % | 29.9 / 6107 | 9 | - |
| ukraine-1 | **grad** | **12/20** | 39-78 % | 2.1 / 7693 | 8 | 790 |
| ukraine-1 | IMU + TRN only | 0/20 | 0-16 % | 6431 / 7721 | 20 | - |

Road / junction alone, closed loop (final configuration incl. gyro-bias learning, 10 seeds):

| mission | row | within 5 m | 95 % Wilson CI | error at B median / max (m) | thermal fixes accepted / flight |
|---|---|---|---|---|---|
| berkeley | junction | 6/10 | 31-83 % | 3.6 / 8069 | 68 |
| berkeley | road | 2/10 | 6-51 % | 31.6 / 7925 | 92 |
| m-c804bb84 | junction | 0/10 | 0-28 % | 245 / 7666 | 56 |
| m-c804bb84 | road | 1/10 | 2-40 % | 77.2 / 6876 | 36 |

Plots: `docs/night_nav/night_error_<mission>.png` (error over time since GNSS off, per-seed error at B).

Why the alternatives lose:
- **ncc** accepts more wrong peaks where the prior's temperature LEVEL is wrong (roofs: mostly cold, 1/3 warm);
  a wrong fix pulls the INS and the bias states, the next correct fix is then outside the search window -> runaway.
  `grad` only needs the edges to be in the right place.
- **ncc+grad** fuses consistent candidates; it gains fix rate but inherits ncc's wrong peaks when both happen to
  agree with a drifting estimate.
- **IMU + TRN**: see "Why it runs away" below. TRN helps over the berkeley buildings (half the seeds within 5 m),
  not on the hill or the GLO-30 terrain of ukraine-1 (TRN fixes there were 100-900 m off and NIS-rejected).

### Why it runs away, and the fix

Instrumented flights (truth vs AHRS) showed the same pattern in every runaway:
1. With GNSS the AHRS tilt error is ~0.6 deg; the GNSS absorbs the 0.1 m/s^2 gravity leak.
2. After GNSS loss the tilt error grows ~1 deg per 10-20 s (residual roll / pitch gyro bias). In calm air the
   simulated X2 has no airframe drag, so the accelerometer only sees thrust along body z: in closed loop the tilt
   error is unobservable without position fixes (CLAUDE.md "Closed-loop drift").
3. Thermal fixes at 5 Hz keep the EKF's bias states (handed to the AHRS as tilt corrections) on track - until a
   blind stretch: the ukraine-1 forest (~30 s), or a run of ambiguous frames. The error then grows quadratically
   (6 -> 58 m in 16 s), the true position leaves the 3-sigma search window, and later "fixes" are wrong matches
   inside the window.

Candidates tried (ukraine-1 seeds 0-8, same frozen data):

| variant | within 5 m (seeds 0-8) | note |
|---|---|---|
| grad (stage 1) | 5/9 | |
| EKF accel-bias random walk 0.01 -> 0.04 | 5/9 | noisy tilt corrections: two flights wobbled and never "reached" a waypoint (stuck 1185 m before B with a 2 m estimate) |
| thermal fix covariance x4 (5 Hz fixes have correlated errors) | 6/9 | about neutral |
| **gyro-bias learning, `ahrs.tilt_bias_ki` 0.05** | **8/9** | the tilt corrections the EKF hands to the AHRS are integrated into the roll / pitch gyro bias (PI loop); learned already during the GNSS phase, so the drift stops in blind stretches. Seeds 2, 3, 4: km runaways -> 0.6 / 0.3 / 1.2 m |

`FlightSystem(night=True)` now sets `ahrs.tilt_bias_ki = cfg.thermal.night_tilt_bias_ki` (0.05) unless it is set
explicitly; daytime flights are unchanged (0).

### Remaining failure (ukraine-1 seed 5)

Before the forest a forced re-localization (5 consecutive NIS rejections = only 1 s at 5 Hz) accepted a match
11 m off; in the forest the error then grew 6 -> 58 m in 16 s and the first fix after the forest was a wrong
match inside a too-small search window (claimed 2-sigma 2.9 m at 65 m true error). The drone also descended to
26 m AGL: terrain following uses the (wrong) estimated position. Tried: forced re-localization from thermal only
after NIS rejections spanning >= 3 s with consistent innovations (`thermal.reloc_min_span_s`, opt-in): seeds 5 and
9 still run away (seed 9 unchanged), so it stays off. Further ideas: forced re-localization only after
rejections spread over several seconds and mutually consistent; search window at least the distance flown blind
x the velocity uncertainty; route choice around featureless stretches (night-demo's ROADS route avoids this forest).
