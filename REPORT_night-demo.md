# REPORT - night-demo (Fri 2026-10-23 23:30 -> Sat 07:00)

Agent night-demo: Ukraine mission, route choice, night viewer, final integration + eval. Branch `night-demo` in
`C:\dev\night-demo` (based on origin/v5-mission-planner; merged night-thermal 146a89f and night-nav 4904433).
Nothing pushed. Commands to continue: `RESUME.md`. Saturday demo script: `DEMO_NIGHT.md`.

## Final eval (night 23:00, 60 m AGL, GNSS completely off from 20 % of the route, 20 seeds)

Harness `demos/30_night_eval.py` (truth-scored): GNSS ok at first, `gnss.mode = off` (no signal at all) when the
TRUE along-route progress passes 20 %; ARRIVAL error = |truth - B| (horizontal) at the moment the drone decides
it is above B (first ABOVE_B / WAIT_FOR_FIX decision); success = arrival error <= 5 m; 95 % Wilson interval.
Flights whose estimate is > 500 m off before arrival are stopped and counted as failures (DIVERGED). Data: a
frozen copy of the three missions taken after night-thermal's final rebuild (01:17; traffic on, truth contrast
0.7). Code: night-demo d2405bc = night-nav 4904433 + night-thermal 146a89f + night route choice (rows thermal,
imu_trn, straight, low_contrast); thermal_rw = same + `ekf.acc_bias_rw 0.04` (night-nav's first candidate,
rejected by them: flights wobble at waypoints); +tiltfix = night-demo b5409f4 with night-nav 651b398 (AHRS learns
the roll / pitch gyro bias from the EKF tilt corrections; the night default now).

| mission | row | n | within 5 m | rate (95 % Wilson) | arrival error mean / median / max (m) | est. error at arrival mean (m) | 2-sigma at arrival mean (m) | thermal fixes accepted / flight |
|---|---|---:|---:|---|---|---:|---:|---:|
| berkeley | thermal | 20 | 20 | 100% [84%, 100%] | 0.8 / 0.7 / 1.8 | 0.5 | 2.5 | 350 |
| berkeley | imu_trn | 20 | 19 | 95% [76%, 99%] | 2.1 / 1.8 / 7.5 | 1.9 | 3.0 | 0 |
| berkeley | low_contrast | 20 | 18 | 90% [70%, 97%] | 39.6 / 0.7 / 566.1 | 11.8 | 7.7 | 309 |
| m-c804bb84 | thermal | 20 | 20 | 100% [84%, 100%] | 1.3 / 1.3 / 2.3 | 0.8 | 2.5 | 561 |
| m-c804bb84 | imu_trn | 20 | 11 | 55% [34%, 74%] | 68.5 / 4.8 / 565.1 | 18.0 | 9.5 | 0 |
| ukraine-1 | thermal | 20 | 15 | 75% [53%, 89%] | 248.0 / 3.7 / 1662.3 | 4.6 | 2.6 | 962 |
| ukraine-1 | imu_trn | 20 | 0 | 0% [0%, 16%] | 1550.2 / 1586.1 / 1657.8 | nan | nan | 0 |
| ukraine-1 | straight | 20 | 11 | 55% [34%, 74%] | 499.1 / 2.9 / 1534.3 | 1.8 | 2.5 | 739 |
| ukraine-1 | low_contrast | 20 | 11 | 55% [34%, 74%] | 267.5 / 4.6 / 1641.3 | 10.0 | 3.4 | 916 |
| ukraine-1 | thermal_rw | 20 | 18 | 90% [70%, 97%] | 4.0 / 2.8 / 22.3 | 3.7 | 2.6 | 1125 |
| ukraine-1 | thermal+tiltfix | 4 | 4 | 100% [51%, 100%] | 2.1 / 2.1 / 2.5 | 1.6 | 2.5 | 1086 |

Rows: thermal = thermal localizer (`grad`) + TRN + IMU on the night route (ukraine-1: along the road);
imu_trn = no thermal (baseline); straight = thermal, but the planner's straight route (ukraine-1: 1129 m blind
over the forest); low_contrast = thermal with the TRUE scene's contrast x 0.4 (0.4 x 0.7 = ~2.8 K road - grass,
a mild night) while the prior still predicts the hot-day contrast.

![error over time](docs/night_eval/error_over_time.png)

Reading it:
- **berkeley / m-c804bb84: 20/20 within 5 m** with thermal (mean 0.8 / 1.3 m). IMU + TRN alone: 19/20 on the
  short Berkeley leg (450 m blind), 11/20 on the hill.
- **ukraine-1: 15/20** with thermal (before the tilt fix): 3 runaways right after GNSS off, two at 17 / 25 m.
  IMU + TRN alone: 0/20 (every flight diverges in calm air, see "What failed").
- **Route choice matters on ukraine-1**: the straight route over the forest diverges in 9/20 flights (11/20 within
  5 m) vs 3/20 along the roads (15/20). n = 20 per row: suggestive (divergences 9 vs 3, Fisher p = 0.08;
  within 5 m 11 vs 15, p = 0.32), not
  conclusive. When the straight flight survives it is slightly more precise (shorter blind time at B).
- **Mild night (0.4x contrast)**: berkeley 18/20 (two runaways), ukraine-1 11/20. Matching is scale-invariant, so
  most flights hold; the losses are runaways, the same failure mode as above.


## What works

1. **Missions outside the US** (`sfhack/missionprep/sentinel.py`, `build.py`): `--region auto` picks NAIP / 3DEP /
   3DEP LiDAR inside the conterminous US, else Sentinel-2 L2A (Earth Search STAC, no account; scene chosen by
   cloud + shadow INSIDE the bbox from the SCL band; only the needed COG tiles read by cached HTTP range requests
   via tifffile, no GDAL; own UTM projection, tested against quadrature) + Copernicus GLO-30 + OSM, synthetic
   point cloud, DSM = DEM + OSM building heights. New layers for every build: `world/osm.json` (raw OSM, for
   the thermal model), `nir.png` + `ndvi.png` (NAIP band 4 in the US, S2 B08/B04 abroad) in world/ and onboard/.
   `--agl` sets the route's minimum cruise height. The US path is unchanged (all old tests green; existing
   missions still count as prepared).
2. **Mission `ukraine-1`**: Skhidnytsia, Carpathian foothills south of Boryslav (V3's S2 area), A 49.2244 N
   23.3220 E -> B 49.2305 N 23.3456 E (25 m from the asphalt main road), 1.8 km, relief 137 m, S2 scene
   2025-08-28 (0 % cloud in the bbox), 41-54 s cold build. Paved / unpaved from OSM `surface`: 3.4 km paved
   (tertiary asphalt), 4.8 km unpaved (tracks), 7.9 km untagged (village lanes, mostly dirt) = 21 / 30 / 49 %.
   Table + figure in docs/MAP_PACKAGE.md ("Outside the US"), `docs/ukraine-1_roads.png`. Day flight on it with
   the RGB vision localizer: ARRIVED 1.06 m.
3. **Route choice** (`sfhack/planning/night_route.py`, onboard): straight while the thermal footprint (56 m at 60 m
   AGL) is expected to hold >= 150 m^2 of paved road, else Dijkstra with a blind-cell cost along the roads, if
   it shortens the longest blind stretch (> 250 m) and costs <= 60 % extra length. ukraine-1: ROADS (straight
   1870 m with a 1129 m blind stretch over the forested hill -> 2034 m along the road, longest blind 64 m);
   berkeley and m-c804bb84: STRAIGHT (longest blind 68 / 101 m). On by default in `FlightSystem(night=True)`.
   Parameters and reasons: docs/NIGHT_DEMO.md.
4. **Night viewer** (`demo_live.py --night`, `demos/liveview/night.py`): dark scene; large white-hot thermal panel
   showing the onboard sensor feed (night-nav's ThermalFrame) with a SEEN vs MAP PRIOR inset of the last match;
   map panel on the predicted night temperature with roads, true / estimated trails, the 2-sigma ellipse and the
   matched roads lit up; HUD with GNSS ON / OFF, nav source (THERMAL + TRN + IMU / INS ...), 2-sigma and true
   error with a 3-minute sparkline. Key N = GNSS completely off / on (no automatic switch). Missing thermal files
   are built on the fly (`ensure_thermal`, ~50 s). Planner page: "Night" checkbox -> `/api/fly {night: true}`.
5. **Eval harness** (`demos/30_night_eval.py`): parallel seeds, GNSS off when the TRUE along-route progress passes
   20 % (simulated operator), arrival error = |truth - B| when the drone decides it is above B, Wilson 95 % CI,
   per-second traces, rows thermal / imu_trn / straight / low_contrast, frozen mission copy, divergence abort.
6. Tests: 3 new test files / cases (global build path offline, UTM, night route straight vs detour, planner
   Night pass-through); all tests green after both merges.

## What failed / limits (honest)

- **IMU + TRN alone diverges on ukraine-1** (estimate 10 km off after 6 min): closed-loop tilt unobservability
  in calm air (no rotor-drag model in the sim), TRN on the 30 m GLO-30 surface model never gets a fix accepted
  after the error is large. night-nav analysed it; it is a physics / model decision, not fixed tonight. The
  thermal row keeps the estimate bounded only because it gets fixes several times per second along roads.
- The viewer ran at 2-3 fps tonight because CPU (four agents' evals) and GPU (detector training) were saturated;
  sim + thermal rendering + localizer is ~290 ms per displayed frame under that load. Expect better on Saturday;
  if not, use speed 1 or the backup videos.
- Backup videos are mp4v (no ffmpeg on this machine): play in VLC / browsers; convert if needed.
- Sentinel-2 at 10 m is blurry as the world texture (0.5 m grid); GLO-30 is a DSM, so the "terrain" of
  ukraine-1 contains the forest canopy shape; 346 of 348 buildings use the default 10 m height.
- The preliminary run1 (01:00) was stopped because night-thermal rebuilt the thermal files mid-run; the final
  eval runs on a frozen copy (`$SFHACK_DATA/night/demo_eval_snap`).

## Next steps

- Saturday: landing at B (thermal fixes stop near the ground; precision descent logic exists from V2).
- Rotor-drag model + drag-velocity fusion (as in death-v3) so blind stretches do not run away.
- Route choice: weight by thermal localizability (prior gradient energy) instead of road area only; add
  junction density; show the night route on the planner page.
- Weather: clouds / wind reduce contrast; tie `thermal_contrast` to a weather lever.

## Disclosure (all of this is PRE-HACKATHON work, built Fri 23:30 - Sat 07:00 by Claude Code agents)

Built by night-demo tonight: Sentinel-2 / GLO-30 global mission path (sentinel.py, region switch, NIR / NDVI,
osm.json, --agl), mission ukraine-1, scripts/plot_mission_roads.py, night route choice (night_route.py + the
extra-cost hook in plan_route), the night viewer (liveview/night.py, demo_live --night, key N, ensure_thermal),
planner Night checkbox, eval harness demos/30_night_eval.py, docs (NIGHT_DEMO.md, MAP_PACKAGE.md section,
DEMO_NIGHT.md, README section), the merges and the final eval. Merged from the other night agents (also
pre-hackathon): night-thermal (thermal world model, thermal camera, prior), night-nav (V3.2 GNSS-free track,
TRN, thermal localizer, thermal sensor), night-research (facts used in DEMO_NIGHT.md).
