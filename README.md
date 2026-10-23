# gnss-free-nav-V2

GNSS-free quadcopter navigation in MuJoCo (hackathon prototype). Full setup instructions follow in milestone M6.

## Disclosure

Prior work, built before the hackathon: the MuJoCo Skydio X2 simulation, the AHRS and EKF state estimation, the vision map-localizer, the Berkeley world, the autopilot and landing logic, the degradation levers, and the live viewer. Built on Saturday: detection, trust ranking, and the decision policy.

The earlier TypeScript rig that previously lived on `main` is archived at tag [`ts-rig-archive`](https://github.com/MaxBingham/gnss-spoof-guard/tree/ts-rig-archive).

## Pre-hackathon status (branch `v4-target-selection`, as of 2026-10-23)

Also built BEFORE Saturday, on this branch (disclosed for judging): V4 target selection + marker landing.
A randomized scene (white car, brown car, brown truck with an ArUco landing pad, tent, hut, tree; `easy` /
`hard` = car parked next to the truck, tree crown over its cab), an onboard detector (change detection vs
the onboard ortho, lidar heights, tracks, class probabilities), the decision logic (decide only with a clear
margin, otherwise fly over and look closer), and a marker-confirmed landing (marker fused into the EKF as an
anchor; no marker -> reject the candidate, try the next; never touch down unconfirmed).

```
uv run python demos/10_target_landing.py --standalone --seed 3 --difficulty hard     # viewer
uv run python demos/10_target_landing.py --standalone --seed 3 --headless            # print + truth score
uv run python demos/11_bench_target_detector.py --seeds 10                            # detector only
uv run python demos/12_eval_target_landing.py --seeds 20 --difficulty easy hard       # closed loop
```

Numbers (2026-10-23): detector at 30 m, 10 easy seeds: brown truck ranked first 10/10, all six objects found
8/10 (the tree missed twice). Closed loop, `--nav gnss` (honest GNSS; the subject is target selection),
20 easy + 20 hard seeds: correct target 40/40, landed on the 2 m pad 40/40 (pad error <= 0.40 m), wrong
landings 0. GNSS-free `--nav vision` (same 40 seeds): on the pad 29/40, wrong landings 0; the other 11 never land:
the vision localizer loses lock over the procedural grass and the EKF diverges (km), the drone times out
in the air. That is the biggest open problem.

## Degradation levers

All levers live in one `Conditions` dataclass (`sfhack/degradation/`). They are simulated reality: the
simulator applies them, onboard code never reads them (`tests/test_truth_separation.py`). Directions are
compass degrees (0 = north, 90 = east), the direction the wind / false position moves toward.

| Lever | Fields | Effect |
| --- | --- | --- |
| GNSS | `gnss.mode` = `off` (default, no receiver) / `ok` / `jam` / `drag` / `jump`; `drag_rate`, `drag_heading_deg`, `jump_m`, `jump_heading_deg` | 5 Hz fixes, 1.5 m white + 0.7 m slow noise. `jam`: no fix. `drag`: false offset grows at `drag_rate` m/s. `jump`: false offset of `jump_m` at once. |
| Weather | `weather.kind` = `clear` / `rain` / `fog` / `dust`, `weather.intensity` 0-1 | Down camera: haze toward the fog color by visibility, pixel noise, rain streaks, frame dropouts. Rangefinder + lidar: range noise and return dropouts. `sim.weather_visuals()` gives visibility / fog ramp / color for the viewer. |
| Wind | `wind.speed`, `wind.direction_deg`, `wind.rotation_deg_s`, `wind.gust` | Drag force on the X2 in MuJoCo from steady wind (optionally turning) plus Ornstein-Uhlenbeck gusts. The live demo's `W` key is a 5 s case of this lever. |
| Camera | `camera.lost`, `camera.drop_rate` | Lost: black down-camera frames (the `C` key's 10 s blackout still works). Drop rate: frames never delivered. |
| Lidar | `lidar.lost`, `lidar.dropout` | Applies to the downward rangefinder and the new 8-beam nadir lidar (`lidar` topic). Touchdown falls back to the estimated-height rule without the rangefinder. |

The onboard baseline is naive on purpose: whenever a GNSS fix exists it is fused into the EKF without a
gate, so a drag spoof pulls the real drone off course while the estimate stays confident. No spoof
detection here.

Ways to set the levers:

```
# config: initial levers
cfg = default_config(); cfg.conditions.update("gnss=ok,weather=fog:0.5")

# CLI (demo_live.py; the same flags work in any demo that calls sfhack.degradation.cli.add_cli_args)
uv run python demos/demo_live.py --gnss ok --cond-at "30:gnss=drag:1.5@90"
uv run python demos/demo_live.py --weather rain:0.7 --wind 6@90 --cond wind.gust=2 --camera-drop 0.3 --lidar-lost

# runtime (viewer / harness): a setter, or a ConditionsMsg on the `conditions` bus topic
fs.update_conditions("gnss=jump:40@90")          # or fs.update_conditions(gnss__mode="jam")
fs.set_conditions(Conditions(...)); fs.conditions  # replace all / read a copy
FlightSystem(..., conditions=..., schedule=parse_schedule("30:gnss=drag;60:camera=lost"))
```

Assignments are dotted paths (`gnss.drag_rate=2`) or group shorthands: `gnss=drag:RATE@HEADING`,
`gnss=jump:METERS@HEADING`, `weather=KIND:INTENSITY`, `wind=SPEED@DIRECTION`, `camera=lost|ok|DROP_RATE`,
`lidar=lost|ok|DROPOUT`. One headless flight per lever, scored on truth vs estimate:
`uv run python demos/09_degradation_scenarios.py [--verbose] [scenario ...]`.
