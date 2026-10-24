# gnss-free-nav-V2

GNSS-free quadcopter navigation in MuJoCo (hackathon prototype). Full setup instructions follow in milestone M6.

## Current idea: acoustic cue -> autonomous night recon quadcopter (no GNSS, no data link)

**Problem.** A base in Ukraine at night. Movement outside the perimeter (e.g. vehicles or troops 2-3 km
away) cannot be seen in the dark, but it can be heard. GNSS is jammed, and there is **no data link** between
the base and the drone at any time: whatever the drone does after takeoff, it does on its own.

**Setup.**
- **Base:** one miniDSP UMA-16 v2 microphone array (16 MEMS mics, 4 x 4, 42 mm pitch) in the middle of the
  base listens and gives the direction (azimuth, rough elevation, with uncertainty) a sound comes from. A fence
  and a landing pad.
- **Drone: a quadcopter (reconnaissance only).** IMU, compass, barometer, GNSS (only for the first 50 m in the
  GNSS-denied version), cameras. No lidar. No microphone array on the drone (maybe later).

**MVP (what we build first).** Someone claps next to the real UMA-16 (on a Linux PC). Its bearing goes over UDP
to the simulation laptop. The drone in the sim takes off from the base, flies out along that bearing, looks,
comes back and lands. Daytime, GNSS on.

3D view of the MVP: `uv run python demos/50_base_3d.py` (bearing from the arrow keys or UDP 5005; `--headless
--bearing 60` flies once without a window). Default world `--world farm`: a fictional, procedurally generated 4 x 4 km
rural area (no real place or map data, `demos/base3d_farm.py`): storage hall with the UMA-16 mast in the middle of its
forecourt, HESCO wall, rolling fields with windbreaks, farmsteads, a village on a paved road; the drone holds 30 m
above the terrain, default range 1200 m. `--world simple` is the old flat compound. Renders: `docs/base3d/`
(`--shots docs/base3d` regenerates them and prints the render frame rate).

**Next steps after the MVP.**
1. GNSS off after 50 m: show the drift and keep it small (preflight alignment, optical flow, compass).
2. A search pattern that widens with distance (bearing uncertainty + own drift).
3. Night: thermal camera to find vehicles / people along the bearing.
4. Return and report: no data link, so the drone flies back and hands over what it found on the pad.

UMA-16 geometry, real recordings and open drone-audio datasets: https://github.com/JLrobot/Mic-Challenge

## Disclosure

Prior work, built before the hackathon: the MuJoCo Skydio X2 simulation, the AHRS and EKF state estimation, the vision map-localizer, the Berkeley world, the autopilot and landing logic, the degradation levers, and the live viewer. Built on Saturday: detection, trust ranking, and the decision policy.

The earlier TypeScript rig that previously lived on `main` is archived at tag [`ts-rig-archive`](https://github.com/MaxBingham/gnss-spoof-guard/tree/ts-rig-archive).

## Setup

macOS / Linux (Windows: the `.ps1` twins in `scripts/`). Needs [uv](https://docs.astral.sh/uv/); Python 3.12
is installed by uv. The venv lives in `~/.venvs/sf-hack-v5` (this branch), generated data in `~/sf-hack-data`.

```
bash scripts/setup.sh                          # once
. scripts/env.sh                               # every shell
uv run python -m pytest tests -q               # all tests (~2 min)
```

## Live demo

```
bash scripts/demo.sh                           # viewer window: Berkeley A -> B, vision localizer, GNSS ok
bash scripts/demo.sh --fullscreen              # same, fullscreen (Tab toggles)
bash scripts/demo.sh --scenario-at 25          # plays the spoof scenario at t = 25 s (or press P any time)
uv run python demos/demo_live.py --record out.mp4 --scenario-at 25   # same view, offscreen, 1920x1080 MP4
uv run python demos/demo_live.py --headless --scenario-at 25         # no window: fly, print, score
```

The viewer is an OpenCV window over an offscreen MuJoCo render; run it with plain `python` (not `mjpython`).
`--record` writes H.264 through `ffmpeg` when it is installed, else OpenCV's `mp4v`. `--size WxH` sets the
frame size (viewer 1280x720, video 1920x1080), `--snap-at 30,60` saves PNGs at those sim times, `Z` saves
one from the window. All lever flags below (`--gnss`, `--weather`, `--cond-at`, ...) work here too.

What is on screen:
- **Solid white X2** = the true drone (simulator), drawn 9x life size or more so it reads across a room.
  **Cyan ghost** = where the drone thinks it is (EKF) with its 2σ bubble and ground ring. White / cyan
  trails = true / estimated path, red dashes = the error between them. A green, B red.
- **Sensors:** yellow frustum + live image = down camera, red beams = rangefinder + 8-beam lidar,
  orange pin and ring = the GNSS fix. A lost sensor goes dark (gray frustum, `CAMERA LOST` image, no beams).
- **Weather:** depth-based fog / haze from `sim.weather_visuals()` (visibility, color per kind), rain
  streaks, dust, and a wind arrow plus drifting streaks that follow the wind lever.
- **HUD:** mission phase and health, true error vs claimed 2σ (red `OVERCONFIDENT` when the error is
  outside it), GNSS state from `fs.conditions.gnss.mode`, active degradations, sensor strip, lever panel,
  wind compass. The violet **Detection / Trust ranking / Decision log** panel is deliberately empty and
  badged *built at hackathon*.

Truth is used only to draw and score (the true drone, true error, camera frustum pose); the autopilot
never sees it.

| Key | Lever / action | Key | Lever / action |
| --- | --- | --- | --- |
| `G` | GNSS: ok, drag, jump, jam, off | `C` | camera lost on / off |
| `F` | weather intensity 0, 0.35, 0.65, 1 | `L` | lidar + rangefinder lost on / off |
| `T` | weather kind: fog, rain, dust | `V` | localizer on / off (dead reckoning) |
| `W` | wind 0, 4, 8, 12 m/s | `P` | play the spoof scenario |
| `D` | wind direction +45 deg | `X` | reset all levers |
| `U` | gusts on / off | `R` | restart the flight |
| `SPACE` | pause | `1` `2` `4` | sim speed |
| `O` | reset the camera | `Q` / `Esc` | quit |

Mouse: drag to orbit, wheel to zoom, click a lever row to press its key.

Spoof scenario (`P`, times after the press): GNSS drag 1.5 m/s east; +14 s fog 0.65 and a 7 m/s
crosswind with gusts; +24 s camera lost; +34 s camera back and weather clears while the spoof keeps
dragging. From t = 25 s (seed 0) the naive baseline lands about 120 m from B with 2σ near 1.2 m and health
NOMINAL throughout.

## Night flight (GNSS off, thermal camera)

Built the night before the hackathon (see the disclosure in REPORT_night-demo.md). 23:00, clear summer night:
the X2 flies A -> B at 60 m AGL, key N removes GNSS completely, and it navigates on IMU, baro, rangefinder,
TRN and a downward thermal camera matched against a thermal map predicted from daytime data.

```
uv run python demos/demo_live.py --mission ukraine-1 --night                 # N = GNSS off / on
uv run python demos/demo_live.py --mission berkeley --night --cruise-agl 60
uv run python demos/30_night_eval.py --seeds 20                              # % within 5 m of B, Wilson 95 % CI
```

Details: [docs/NIGHT_DEMO.md](docs/NIGHT_DEMO.md) (viewer, route choice), [DEMO_NIGHT.md](DEMO_NIGHT.md)
(what to click and say), docs/THERMAL_MODEL.md (night-thermal), docs/NIGHT_NAV.md (night-nav). Missions outside
the US (Sentinel-2 + Copernicus DEM): docs/MAP_PACKAGE.md, "Outside the US".

## Mission planner (any A / B)

Pick A and B on a map, download open data, build the simulated world and the onboard map package,
"upload" it to the drone and fly it in the live demo. Package format (the contract for the LiDAR / TRN
stack and the object placer): [docs/MAP_PACKAGE.md](docs/MAP_PACKAGE.md).

```
uv run python -m sfhack.missionprep serve                  # http://127.0.0.1:8765/: search, click A and B,
                                                           # radius slider, Prepare mission: with "Fly automatically"
                                                           # (default on) it uploads and opens the live viewer itself
uv run python -m sfhack.missionprep build --id berkeley    # same without the browser (default A / B)
uv run python -m sfhack.missionprep upload berkeley        # simulated 2 MB/s link, SHA-256 per layer
uv run python demos/demo_live.py --mission berkeley        # fly it (also --headless / --record)
```

Data: OpenStreetMap (Overpass), USDA NAIP ortho, USGS 3DEP 1 m DEM and 3DEP LiDAR (read straight from
the public Entwine octrees on AWS, no PDAL), with fallbacks (OSM API, USGS tiles, Copernicus GLO-30,
a synthetic cloud), all cached in `$SFHACK_DATA/cache/missionprep`. Default Berkeley: 37 s on the first
run, 1.6 s once prepared. The world has real terrain (height field) and extruded footprints with
ortho-textured roofs; the onboard package is a deliberately imperfect copy (heights +-20 %, 10 % of the
buildings and trees missing, noisy DTM, re-processed ortho, real LiDAR cloud), and the route is planned
on that onboard map around tall obstacles.

One click: "Prepare mission" runs prepare -> upload (2 MB/s) -> launch, on one progress bar ("preparing", "uploading
to drone", "launching sim"). Same A / B / radius already prepared and uploaded: straight to the launch. The server
starts `demos/demo_live.py --mission <id>` with its own interpreter (one viewer at a time, output in
`<mission>/viewer.log`, a crash within 10 s shows the log on the page); every prepared mission has a Fly button,
and the terminal command stays on the page as a fallback. API: `POST /api/fly {mission_id}`, `GET /api/fly/status`,
`POST /api/fly/stop`.

## Degradation levers

All levers live in one `Conditions` dataclass (`sfhack/degradation/`). They are simulated reality: the
simulator applies them, onboard code never reads them (`tests/test_truth_separation.py`). Directions are
compass degrees (0 = north, 90 = east), the direction the wind / false position moves toward.

| Lever | Fields | Effect |
| --- | --- | --- |
| GNSS | `gnss.mode` = `off` (default, no receiver) / `ok` / `jam` / `drag` / `jump`; `drag_rate`, `drag_heading_deg`, `jump_m`, `jump_heading_deg` | 5 Hz fixes, 1.5 m white + 0.7 m slow noise. `jam`: no fix. `drag`: false offset grows at `drag_rate` m/s. `jump`: false offset of `jump_m` at once. |
| Weather | `weather.kind` = `clear` / `rain` / `fog` / `dust`, `weather.intensity` 0-1 | Down camera: haze toward the fog color by visibility, pixel noise, rain streaks, frame dropouts. Rangefinder + lidar: range noise and return dropouts. `sim.weather_visuals()` gives visibility / fog ramp / color for the viewer. |
| Wind | `wind.speed`, `wind.direction_deg`, `wind.rotation_deg_s`, `wind.gust` | Drag force on the X2 in MuJoCo from steady wind (optionally turning) plus Ornstein-Uhlenbeck gusts. In the live demo, `W` steps the speed, `D` turns the direction, `U` toggles gusts. |
| Camera | `camera.lost`, `camera.drop_rate` | Lost: black down-camera frames (`C` in the live demo). Drop rate: frames never delivered. |
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
