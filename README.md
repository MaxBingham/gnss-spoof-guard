# gnss-free-nav-V2

GNSS-free quadcopter navigation in MuJoCo (hackathon prototype). Full setup instructions follow in milestone M6.

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
