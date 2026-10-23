# DEMO V3 - GNSS-denied navigation over real terrain

## Demo day cheat sheet

```
powershell -ExecutionPolicy Bypass -File scripts\demo_v3.ps1 s1      # one command, offline, opens the viewer
```

- **Seed 0** (the default). In S1, INS only misses B by 59 m with seed 0, and all sources land 0.1 m from B.
- **Running order:** `s1` live -> at ~60 s (cruise) press `T` and `I` (health DEGRADED, the ring grows) -> ~40 s later
  press `T` and `I` again (fixes pull it back, NOMINAL) -> landing at B. Press `R` to restart, then `1`/`2`/`4` to set the speed.
- **Pitch slide:** INS only misses B by **30 m (S1, 1.5 km)** and **690 m (S2, 8.4 km)** (median over 10 seeds).
  INS + TRN: **1.1 m** and **3.9 m**, and 20/20 flights succeed (chart: `$SFHACK_DATA/v3/eval/quick10/pitch.png`).
- **Backup:** `$SFHACK_DATA/v3/demo/*.mp4` (see Backup videos).

A Skydio X2 in MuJoCo flies A -> B with **GNSS disabled**. It navigates on an inertial backbone (AHRS + 10-state
EKF with rotor-drag fusion) corrected by two map-referenced sources it carries onboard:

- **TRN** (terrain-referenced navigation): a particle filter matches the altimeter's terrain profile against the
  onboard surface model (DSM). Works over hills and buildings, day or night, no camera.
- **IBN** (image-based navigation): the down camera, ortho-rectified with the estimated attitude, is matched (NCC)
  against the onboard orthophoto.

An integrity monitor gates every fix (NIS), excludes a source that disagrees with the INS and the other source,
and re-localizes after a long outage. A route planner places update zones where TRN / IBN have information.
The onboard map is never the simulated world: heights, missing buildings, blur, datum offset differ on purpose.

| | S1 urban hills | S2 natural terrain |
|---|---|---|
| where | San Francisco: Aquatic Park -> Russian Hill -> Washington Square | Carpathian foothills, western Ukraine (south of Boryslav) |
| route | ~1.5 km, 35 m above the surface, 10 m/s | ~8.4 km, 120 m above the surface, 18 m/s |
| altimeter | laser, 60 m | radar, 400 m |
| imagery | USGS orthophoto, everywhere | synthetic, only a 240 m landing patch around B |
| success | landed within 3 m of B | arrived within 10 m of B before the precision landing |

## Run it

```
. scripts/env.ps1                                         # Windows (bash: . scripts/env.sh)
uv run python -m sfhack.world.build_v3 s1 s2              # only on a new machine, online (cached in $SFHACK_DATA)
powershell -ExecutionPolicy Bypass -File scripts\demo_v3.ps1 s1          # live viewer (bash scripts/demo_v3.sh s1)
```

Scenes of `scripts/demo_v3.ps1` / `scripts/demo_v3.sh` (extra arguments go to `demos/v3_live.py`):

| scene | what | sim speed |
|---|---|---|
| `s1` (default) | S1, INS + TRN + IBN | 2x |
| `s1-ins` | S1, dead reckoning only: the estimate drifts, the drone misses B | 2x |
| `s1-outage` | S1, all sources, TRN + IBN off from 60 s to 100 s (keys `T`, `I` scripted), then back on | 2x |
| `s2` | S2, INS + TRN + landing-patch IBN | 8x (the sim runs ~4-5x real time on a laptop, so 8x is "as fast as possible") |
| `videos` | records the three backup videos below (offscreen, no window) | |

Direct: `uv run python demos/v3_live.py --world s1 --config ins_trn_ibn --seed 0 [--events "60:T,90:T"] [--headless] [--record]`.

### What the jury sees

- **3D**: the drone is the truth (simulator). Cyan sphere = where the drone *thinks* it is, cyan ring = its 2-sigma
  (also projected on the ground). White trail = true path, cyan trail = estimated path, yellow = planned route.
  A green, B red.
- **Top left**: phase, health (NOMINAL / DEGRADED / HOLD), active sources (an excluded source shows `EXCLUDED`),
  TRN and IBN status with accepted/total fixes, 2-sigma radius, position error (eval only, from the truth), GNSS DISABLED.
- **Bottom left, onboard map**: what the drone carries (orthophoto, or a hillshade of the DSM in S2), the TRN particle
  cloud (orange), the last fix (orange cross TRN, yellow IBN, red = rejected by the integrity monitor).
- **Down camera (IBN)**: the live frame while IBN is active; frame border yellow on an accepted match.
  The camera is switched on only in IBN update zones and for the landing (saves power, shows the planner at work).
- **Terrain profile (TRN)**: surface height under the drone measured (barometer - altimeter, white) against the
  onboard map at the estimate (orange). When they line up, TRN has locked.
- **Bottom right**: the onboard decisions (plan, update zones, health changes, exclusions, re-localization, landing).

### Keys (viewer)

| key | action |
|---|---|
| `T` / `I` | TRN / IBN on/off (operator switch; the health monitor reacts) |
| `W` | wind gust, 8 m/s crosswind for 5 s |
| `C` | camera blackout, 10 s |
| `A` | altimeter failure (TRN loses its sensor) |
| `R` | restart (new window) |
| `1 2 4 8` | sim speed |
| `SPACE` | pause |

The MuJoCo viewer's own bindings of these keys (transparency, inertia, wireframe, ...) are undone every frame.

## Talk track (~3 min)

1. **S1 live, all sources** (`s1`): "GNSS is off. The drone only knows its IMU, barometer, compass, a downward
   altimeter and a camera - and a map it loaded before take-off." Point at the cyan ring: it stays small, the
   TRN cloud collapses on the map, the camera frame turns yellow in the update zones. Landing within ~1 m of B.
2. **Outage** (`s1-outage`, or press `T` and `I` live mid-cruise): health goes DEGRADED ("no aiding source"),
   the ring grows - the drone *knows* it is getting worse. Switch back on: the first fixes pull the estimate back,
   health NOMINAL, landing still within ~1 m.
3. **Why it matters** (`s1-ins`, or the eval chart): INS only, the same drone misses B by tens of meters in S1 and
   by hundreds of meters in S2.
4. **S2** (`s2` or the video): 8.4 km over forested hills, no imagery en route - TRN alone holds the estimate to a few
   meters (watch the terrain profile), the planner detours over ridges for information, IBN takes over on the
   landing patch.
5. **Numbers** (table below, `final_error.png`).

## Results (calm air, seeds 0-9 per row)

| scenario | configuration | success | final error at B median / p90 [m] | arrival error median [m] | max 2-sigma median [m] | time DEGRADED+HOLD mean [s] |
|---|---|---|---|---|---|---|
| S1 | INS only | 0/10 | 29.9 / 57.2 | 24.9 | 85.2 | 259 |
| S1 | INS+TRN | 10/10 | 1.1 / 1.7 | 1.2 | 9.8 | 0 |
| S1 | INS+TRN+IBN | 10/10 | 1.1 / 1.5 | 0.7 | 5.5 | 0 |
| S2 | INS only | 0/10 | 692 / 886 | 608 | 329 | 779 |
| S2 | INS+TRN | 10/10 | 3.9 / 5.1 | 3.6 | 44.1 | 35 |
| S2 | INS+TRN+IBN | 10/10 | 0.4 / 0.7 | 1.1 | 44.1 | 35 |

No crashes, emergency landings or source exclusions in any of the 60 flights. In S1, IBN does not improve the
median final error (both land at ~1 m), but it halves the 2-sigma and the arrival
error. In S2, IBN only works over the landing patch, and it takes the final error from 3.9 m to 0.4 m.

Files in `$SFHACK_DATA/v3/eval/quick10/`: `table.md`, `pitch.png` (slide), `final_error.png`, `health.png`,
`runs.jsonl` (every flight). The S2 rows reuse seeds 0-9 of the 20-seed `calm20` run (same code); S1 was flown fresh.
Reproduce: `uv run python demos/v3_eval.py --configs ins,ins_trn,ins_trn_ibn --seeds 10 --tag quick10 --headless`
(~5 min for S1 on 18 workers, S2 ~10 min more), then `uv run python demos/v3_pitch_chart.py --tag quick10`.

## Backup videos

If the viewer fails on the demo machine, play these (1280x720, panels + text burned in):

| file (in `$SFHACK_DATA/v3/demo/`) | content |
|---|---|
| `v3_s1_ins_trn_ibn_seed0.mp4` | S1, all sources, seed 0, take-off to landing at B |
| `v3_s1_outage_seed0.mp4` | S1, TRN + IBN off 60-100 s, DEGRADED, recovery, landing |
| (not recorded yet) | S2, 8x: `demo_v3.ps1 videos` records it as the third file |

Re-record: `scripts/demo_v3.ps1 videos` (`bash scripts/demo_v3.sh videos`).

## Honest limits (say these if asked)

- **Wind is not solved yet.** All numbers above are calm air. A wind model (steady wind per seed + turbulence, wind
  states in the EKF) is work in progress on branch `v3-wind-wip`; the gust key `W` is the only wind in this demo.
- **TRN needs terrain.** Over flat ground or water the altimeter profile carries no position information, TRN stays
  uncertain and the drone falls back to INS (+ IBN where the ground has texture; water has none). The planner routes
  through update zones for this reason, but a long flight over the bay or a flat field would drift like INS only.
- S2 has no imagery en route by design (synthetic texture, landing patch only): IBN there is the landing aid.
- The simulated sensors are idealized in places (no lens distortion, no motion blur, rangefinder on a flat pad).
- Truth is used only to display and score (`tests/test_truth_separation.py`); the "Position error" row is labeled
  eval only.

## What existed before the hackathon (Oct 24)

Everything in this repo up to `death-v3` was built before the hackathon day:
- **V1** (separate repo `gnss-free-nav`): browser prototype of map-based localization.
- **V2** (`gnss-free-nav-V2`, Oct 20-22): MuJoCo Skydio X2, noisy sensors, AHRS + EKF, NIS gating, camera
  localizer against OSM + USGS orthophoto, precision landing on the Berkeley route.
- **V3** (this repo, Oct 22-23): real-terrain worlds S1/S2, TRN particle filter, IBN, integrity monitor, update-zone
  planner, outage recovery, eval harness, this live viewer.
Hackathon-day work goes on top of `death-v3`; see `git log` for what was added on the day.

## Troubleshooting

- Viewer window too tall: it is maximized automatically on Windows; otherwise maximize it by hand (panels at the bottom).
- macOS: the viewer needs `mjpython` (the scripts pick it).
- "world package not found": run `uv run python -m sfhack.world.build_v3 s1 s2` once (needs network).
- Slow: the sim runs ~3x (S1) / ~5x (S2) real time headless on one core; the viewer drops sim time rather than
  stuttering when it cannot keep up with the selected speed.
