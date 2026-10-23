# DEMO V3 - GNSS-denied navigation over real terrain

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
uv run python -m sfhack.world.build_v3 s1 s2              # once, online: builds the world packages
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

## Results (calm air, 20 seeds per row)

RESULTS_TABLE

Charts: `$SFHACK_DATA/v3/eval/calm20/final_error.png`, `health.png`; every flight in `runs.jsonl`
(`uv run python demos/v3_eval.py --seeds 20 --tag calm20`, ~EVAL_MIN min on 18 workers).

## Backup videos

If the viewer fails on the demo machine, play these (1280x720, panels + text burned in):

| file (in `$SFHACK_DATA/v3/demo/`) | content |
|---|---|
VIDEO_ROWS

Re-record: `scripts/demo_v3.ps1 videos` (`bash scripts/demo_v3.sh videos`).

## Honest limits

- **Calm air.** All numbers above are without steady wind. A wind model (steady wind per seed + turbulence, wind
  states in the EKF) is work in progress on branch `v3-wind-wip`; the gust key `W` is the only wind in this demo.
- S2 has no imagery en route by design (synthetic texture, landing patch only): IBN there is the landing aid.
- The simulated sensors are idealized in places (no lens distortion, no motion blur, rangefinder on a flat pad).
- Truth is used only to display and score (`tests/test_truth_separation.py`); the "Position error" row is labeled
  eval only.

## Troubleshooting

- Viewer window too tall: it is maximized automatically on Windows; otherwise maximize it by hand (panels at the bottom).
- macOS: the viewer needs `mjpython` (the scripts pick it).
- "world package not found": run `uv run python -m sfhack.world.build_v3 s1 s2` once (needs network).
- Slow: the sim runs ~3x (S1) / ~5x (S2) real time headless on one core; the viewer drops sim time rather than
  stuttering when it cannot keep up with the selected speed.
