# RECON PLAN - acoustic cue -> autonomous night recon quadcopter (today)

Read this whole file and the "Current idea" section of README.md first. This is the shared plan and interface
contract for 4 agents working in parallel. You are ONE of them; your prompt says which. Read the other sections
so you know what to fetch from whom. Everything here is built TODAY (all earlier branches are prior work; list
what you reuse in your report under "Disclosure").

## How you work
- Do not wait for me; pick the most reasonable option, note it, keep going. If a command is denied, do not retry
  it; find another way or note it.
- Windows, PowerShell, uv. Own venv: `$env:UV_PROJECT_ENVIRONMENT = "$HOME\.venvs\sf-hack-<your branch>"`, then
  `scripts\setup.ps1`, then `. scripts\env.ps1`. Add dependencies with `uv add` (they land in pyproject/uv.lock;
  say so in your report so the integrator can merge them).
- Base branch: `origin/main` (acoustic-recon = night-demo + this plan was merged into main; if your branch was made from origin/acoustic-recon that is the same code). Reuse what exists: Ukraine mission `ukraine-1`
  (`$SFHACK_DATA\missions\ukraine-1`), thermal model + thermal camera (`sfhack/thermal/`), EKF / AHRS / autopilot,
  night viewer (`demos/liveview/night.py`), eval harness style (`demos/30_night_eval.py`).
- Remote branches are only `main`, `jean` (my branch) and Max's own branch. Agent branches (recon-*) stay LOCAL,
  never push them. Commit at every working step. Do not touch main or Max's branch. recon-mission integrates all
  four and I push that result as `jean` (`git push origin recon-mission:jean`).
- All agent folders live in `C:\Users\jean-\OneDrive\Dokumente\5 Claude Projects\14 Mic Challenge` (paths have spaces: always quote them). Fetch others:
  `git fetch "C:\Users\jean-\OneDrive\Dokumente\5 Claude Projects\14 Mic Challenge\<branch>" <branch>`, merge FETCH_HEAD. Never edit another agent's folder. OneDrive: `git config gc.auto 0` is set, never run git gc.
- Truth separation stays (CLAUDE.md HARD RULE): onboard code never imports sim / truth / degradation. Tests green.
- Max 6 parallel sim workers per agent. Evaluate on Ukraine (`ukraine-1`) only.
- Every ~90 min update `STATUS_<branch>.md` (what works, numbers, blockers) and commit it.
- Deadlines: first working version (even crude) by 13:00; integration starts 14:30; I announce the freeze.

## Scenario (fixed)
A base in Ukraine at night (23:00, summer, clear, no wind unless a lever sets it). GNSS jammed; NO data link
between base and drone, ever. The base has one UMA-16 array, navigation lights on its fence, a landing pad.
1. The base UMA-16 hears a source (vehicles / troops, 1.5-3 km out) -> bearing + uncertainty.
2. The bearing is loaded into the quadcopter (Skydio X2 model) before takeoff. Takeoff from the pad.
3. GNSS works for the first 50 m of flight, then it is gone (key N; the eval harness switches it at 50 m).
4. Outbound along the bearing on IMU + compass + baro + optical flow (downward thermal camera) + base-light fixes
   (rear camera). The drone does not know the distance to the source or what it is.
5. Search pattern widening with distance (cone from bearing sigma + own position sigma) at ~60-100 m AGL; the
   downward thermal camera looks for warm objects (vehicles, people).
6. On a find: orbit it for ~30 s relative to the target (no absolute position needed), store thermal snapshots
   + estimated position (bearing + distance from the base).
7. Return: home on the base lights, precision landing on the pad, write the report (no link: the report is
   handed over on the pad).
Battery budget: whole mission <= 20 min. If time runs out before a find: return anyway and report "nothing found
in the searched area" with the area covered.

## Interface contract
- **Bearing** (`sfhack/acoustic/bearing.py`): `BearingEstimate(t, az_deg, el_deg, sigma_az_deg, sigma_el_deg,
  label, confidence, array_enu)`. `az_deg` = degrees clockwise from north (compass convention), `el_deg` above
  horizon. JSON file form: `bearing.json` with the same keys. Live (real UMA-16) and simulated bearings use the
  same type. The drone gets ONLY this (plus its own sensors and the onboard base-light survey).
- **Base** (`sfhack/world/base.py`, recon-world): base centre = mission point A of `ukraine-1` (flattened pad
  exists there). Fence ~150 m square or round, 12 IR lights on the fence (~every 30 deg), optional outer ring of
  6-8 posts at ~400 m. Each light: id, ENU position, height, blink code (unique on/off pattern, e.g. 8-bit at
  10 Hz). Truth world: lights as emissive geoms (VISIBLY glowing in the 3D viewer). Onboard package:
  `onboard/base_lights.json` (surveyed positions + codes, small survey error), `onboard/base.json` (pad
  position + orientation, UMA-16 position).
- **Targets** (recon-world): `world/target_scene.json` gets 2-3 vehicles (+ optional people) at 1.5-3 km from the
  base along a chosen true bearing, with thermal temperatures (engine warm). Plus distractors (warm rocks /
  a barn roof, a cooling car). Seeded placement.
- **Simulated array** (`sfhack/acoustic/sim_array.py`, recon-world): from target truth -> `BearingEstimate` with a
  realistic error (bias + noise; sigma 5-10 deg for low-frequency engine noise; worse with distance / wind
  lever). Stretch: synthesize 16-ch audio with the UMA-16 geometry and run the same DOA code as the live path.
- **Rear camera** (recon-world renders, sim side): narrow FOV (~20 deg H), NIR look: dark scene, lights bright,
  blinking per code. `sfhack/sim/rear_camera.py`.
- **Light localizer** (`sfhack/perception/light_localizer.py`, recon-nav): detect blobs, decode ids over a few
  frames, resection with >= 3 lights -> `Fix(xy, yaw, cov, n_lights)`; with 2 lights -> bearing-line fix;
  elevation angle + baro -> range. Fused in the EKF with NIS gating.
- **Optical flow** (`sfhack/perception/optical_flow.py`, recon-nav): downward thermal frames + baro AGL + AHRS
  attitude -> ground velocity + cov, fused in the EKF.
- **Nav configs** (recon-nav): `imu` (IMU + compass + baro), `flow` (+ optical flow), `lights` (+ flow + lights).
- **Thermal detection** (`sfhack/perception/thermal_detect.py`, recon-mission): warm blobs -> `Detection(xy_rel,
  size_m, temp_K, conf)`; reject small / cold / static-hot distractors where possible.
- **Recon mission** (`sfhack/mission/recon.py`, recon-mission): state machine TAKEOFF -> OUTBOUND -> SEARCH ->
  OBSERVE -> RETURN -> HOME (lights) -> LAND -> REPORT; decisions logged (`decision` topic).
- **Report** (recon-mission): `$SFHACK_DATA\reports\<mission>\<run>\report.json` + PNG snapshots + track;
  report screen in the viewer.
- **Viewer** (recon-mission): `demos/demo_recon.py --mission ukraine-1 [--bearing bearing.json] [--nav lights]`:
  3D chase view with the base, fence and glowing lights; panels: rear camera (lights with ids), thermal camera
  (detections), map (bearing line, search cone, true / estimated tracks for the three nav configs or the active
  one, finds), HUD (GNSS ON/OFF, nav sources, 2-sigma, state, battery). Keys: N GNSS off, L lights on/off,
  SPACE pause, 1/2/4 speed, R restart.

## Agents
| Folder | Branch | Owner of |
|---|---|---|
| C:\Users\jean-\OneDrive\Dokumente\5 Claude Projects\14 Mic Challenge\recon-acoustic | recon-acoustic | real UMA-16 live path: capture, DOA, detection, bearing.json |
| C:\Users\jean-\OneDrive\Dokumente\5 Claude Projects\14 Mic Challenge\recon-world | recon-world | base + fence + lights, targets, simulated array bearing, rear NIR camera |
| C:\Users\jean-\OneDrive\Dokumente\5 Claude Projects\14 Mic Challenge\recon-nav | recon-nav | optical flow, light localizer, EKF fusion, 3-config drift eval |
| C:\Users\jean-\OneDrive\Dokumente\5 Claude Projects\14 Mic Challenge\recon-mission | recon-mission | recon state machine, thermal detection, return + landing, report, viewer, integration |

### recon-acoustic (real hardware; may be handed to Max)
1. Geometry: copy `hardware/uma16/geometry/` from https://github.com/JLrobot/Mic-Challenge (BSD-3, keep the
   license) into `sfhack/acoustic/geometry/`. Channel order is NOT row by row: always map through the file.
2. Offline first: develop DOA (SRP-PHAT and/or MUSIC; acoular or pyroomacoustics, or own numpy) on the real
   UMA-16 recordings in that repo (`data/uma16_real/jduanen_acoustic_camera/`: speaker at a known direction +
   calibration). Report the direction error.
3. Live: Windows needs the miniDSP ASIO driver; in Python set `os.environ["SD_ENABLE_ASIO"]="1"` BEFORE
   `import sounddevice`, find the device (name contains "UMA16"), 16 ch, 48 kHz. Script
   `demos/40_uma16_live.py`: per-channel level check, live azimuth/elevation heatmap (camera overlay if the
   UMA camera is plugged in), detection of an engine-like source (band energy 50-2000 Hz over a noise floor
   learned in the first 10 s; optional: harmonic / fingerprint features), writes `bearing.json`
   (BearingEstimate) when a source is stable for ~3 s. Array mounting: flat on the table (az from the array's
   x axis; a config offset maps it to north for the demo).
4. Fallback that always works: `--file <wav>` plays a 16-ch recording through the same pipeline.
5. Report accuracy (deg) on the recordings and live (speaker at measured angles), and the bearing sigma you put
   into BearingEstimate.

### recon-world
1. Base on `ukraine-1` at A: fence geometry, 12 fence lights (+ optional outer ring) as emissive geoms with blink
   codes, pad, UMA-16 mast. Write `onboard/base_lights.json`, `onboard/base.json` (with small survey error).
2. Targets per contract, seeded, 1.5-3 km out, with thermal temperatures that the existing thermal camera renders
   (hook into `sfhack/thermal`).
3. `sim_array.py`: truth -> BearingEstimate with realistic error; levers: sigma, bias, wind.
4. Rear NIR camera render (narrow FOV, lights bright + blinking, rest dark, noise) at >= 10 Hz.
5. `demos/41_base_view.py`: fly around the base at night, show the glowing lights in 3D and in the rear camera.
6. First version (base + lights visible + rear camera + one target + simulated bearing) by 13:00.

### recon-nav
1. Until recon-world delivers, use stubs (lights as points, simple rear-camera projection).
2. Optical flow from the downward thermal camera (low texture at night: report where it fails) + baro AGL +
   attitude -> velocity; fuse.
3. Light localizer per contract (blob detection, blink-code ids, resection, 2-light line fix, elevation range).
4. Configs `imu`, `flow`, `lights`. GNSS off at 50 m of flight.
5. Eval (ukraine-1, 20 seeds, out along a bearing to 2.5 km and back): position error vs distance from the base,
   heading error, light-fix rate vs distance, Wilson 95 % CI for "within 50 m at 2 km". Plot for the pitch.

### recon-mission
1. State machine + search cone + thermal detection + orbit + return + homing on lights + pad landing (reuse the
   existing landing logic; flare with rangefinder if present, else baro + lights) + report.
2. Viewer per contract (start from the night viewer).
3. From 14:30: integration. Fetch recon-world, recon-nav, recon-acoustic; merge into recon-mission; one command
   runs the full demo: (optional) live UMA-16 -> bearing.json -> sim mission. Write `DEMO_RECON.md` (commands,
   what to click, what to say). Eval: 20 seeds, full mission: find rate, report position error, landing error,
   mission time.
