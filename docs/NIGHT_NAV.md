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
   (`thermal.yaw_feedback`) is implemented, off by default (not evaluated in closed loop yet).

## Closed-loop results

(filled in when the 20-seed runs finish)
