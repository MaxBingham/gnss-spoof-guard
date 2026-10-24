# Night navigation without GNSS (branch night-nav)

Status: WORK IN PROGRESS (numbers below are filled in as the evaluations finish).

## Scenario

Summer night 23:00, clear, no wind, 60 m AGL. GNSS on at takeoff; at 20 % of the route length (measured on the
true position, harness) the receiver goes completely off (`gnss.mode = off`: no message at all). From then on the
drone has IMU, barometer, nadir lidar (TRN on the onboard DSM) and a downward thermal camera (640 x 512, 50 deg
HFOV, NETD 50 mK, night-thermal's renderer). The RGB camera is dark and off. The drone never holds mid-route.
Scored when it commits to B (start of the descent, after up to 10 s waiting for a fix above B): horizontal
distance truth <-> B. Landing is not part of tonight's scenario.

## How it works

```
thermal frame (K) --> ortho-rectify at the onboard DSM (AHRS attitude, lidar height) --> north-up template
                                                                                         (0.5 m cells, centred
                                                                                          on the optical axis)
onboard/thermal_prior.npz (predicted 23:00 temperatures from daytime imagery + OSM) --> search window around
the EKF estimate (3 sigma + 6 m) --> match --> fix (x, y, 2x2 covariance, yaw correction, quality)
--> big-jump confirmation --> EKF NIS gate (chi2 9.21) --> EKF (+ GNSS-free track, + TRN)
```

Code: `sfhack/perception/thermal_localizer.py` (onboard, no truth), `sfhack/sim/thermal_sensor.py` (simulator side),
`Autopilot.on_thermal`, `FlightSystem(night=True, cruise_agl=60)`, config `Config.thermal` (`ThermalConfig`).

## Methods compared

| method | what it matches | fix type |
|---|---|---|
| `ncc` | zero-mean normalized cross-correlation of rectified temperatures vs predicted temperatures | 2-D or 1-D (peak shape) |
| `grad` | NCC on 4 gradient-orientation channels mod 180 deg (CFOG-like): edges match even where the prior got the sign wrong (roofs) | 2-D or 1-D |
| `road` | warm elongated structures thinned to centerlines, chamfer-matched to the prior's road centerlines | 1-D on a straight road, 2-D at curves / crossings |
| `junction` | junctions of the observed centerlines vs junctions of the mapped road skeleton | 2-D, sparse |
| `a+b` | candidates of each method, the largest mutually consistent set fused (information-weighted) | |
| `off` | no thermal: IMU + baro + TRN only (baseline) | |

## Results

(filled in below)
