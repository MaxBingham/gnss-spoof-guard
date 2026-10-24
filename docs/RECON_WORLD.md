# Recon world (recon-world): base, lights, targets, simulated array, rear NIR camera

Generate once per mission (simulator side, writes truth + the onboard survey):

```
uv run python -m sfhack.sim.recon_world --mission ukraine-1 [--seed 0] [--seeds 20] [--bearing 70 --dist 1800] [--no-outer]
uv run python -m sfhack.acoustic.sim_array --mission ukraine-1 --seed 0 [--mode audio] [--wind 4] [--sigma 6] [--bias 3]
uv run python demos/41_base_view.py [--mission ukraine-1] [--headless] [--camera overview]
```

## Files

| file | side | content |
|---|---|---|
| `onboard/base_lights.json` | onboard | `BaseLights`: `bit_s` 0.1, `code_bits` 8, `survey_sigma_m` 0.05, `lights[]`: `id`, `ring` (fence / outer), `enu` [x, y, z], `code` (canonical), `aim_az_deg`, beam, `phase` |
| `onboard/base.json` | onboard | `BaseInfo`: `center_enu`, `pad_enu`, `pad_yaw_deg`, `pad_size_m`, `uma16_enu`, `uma16_x_axis_az_deg`, `fence_half_m` |
| `world/base_lights.json`, `world/base.json` | truth | the same without survey error (+ `fence` geometry) |
| `world/stray_lights.json` | truth | lamps (steady) and a fire (flicker): visible, never a valid code |
| `world/target_scene.json` | truth | objects: `kind` (vehicle / person / distractor), `role` (target / distractor), `class`, `name`, `pos` [x, y, ground_z], `yaw_deg`, `temp_K` (max), `parts[]` (`type` box / cylinder / sphere, `size` half extents, `pos` object frame, `temp_K`, `rgba`) |
| `world/recon_truth.json` | truth (scoring) | `bearing_deg`, `dist_m`, `group_enu` of the vehicle group from the base centre, counts |
| `world/recon/seed_<k>/` | truth / input | `target_scene.json`, `recon_truth.json` per seed, `bearing.json` from sim_array |

Onboard code reads only `onboard/` via `sfhack/world/base.py` (`load_base_lights`, `load_base`, `decode_code`).
Seeds: `PackageWorld(world_dir, target_scene=<seed file>)` and `thermal.camera.load_truth(..., target_scene=<same>)`.

## Lights and codes
- Fence: 150 m square centred on A, F00 north, clockwise every 30 deg, 4 m poles, beam aimed outward (Gaussian,
  70 deg horizontal / 25 deg vertical half width at 1/e^2, centre 3 deg up). Outer posts O00..: 400 m out (only where
  the map has terrain), 5 m poles, twice as bright, 60 deg beam.
- 22 aperiodic 8-bit necklaces of weight 3-5 (`lyndon_codes()`): no rotation of one equals a rotation of another.
  The drone samples at an unknown phase (`decode_code` canonicalizes). Each light starts its code `phase` bits late
  (otherwise every light would be dark at once: canonical codes start with 0s). Bit rate 10 Hz; camera 20 Hz.
- Base clock offset to the sim clock: seeded, unknown to the drone.

## Rear NIR camera (`sfhack/sim/rear_camera.py`)
`RearCameraSensor(sim, bus, world_dir, seed)`, `.step()` after every sim step -> `NirFrame` on `camera/rear`:
`image` (480, 640) uint8, `intrinsics` (fx = fy = 1815 px, 20 deg HFOV), `R_body_cam` (columns = OpenCV camera axes
in the body frame x fwd / y left / z up), `tilt_deg`, `exposure_s`. Gimbal: `CameraTiltCmd(t, tilt_deg)` on
`camera/rear/tilt` (deg below body -x, -10..70, 60 deg/s). Truth for display / scoring: `sensor.last_truth`.
Radiometry: a fence lamp on its axis peaks ~150 DN at 2 km (PSF sigma 0.9 px), ~56 DN at 3 km; saturated core grows
closer in; background 3-10 DN; occlusion by terrain / buildings / trees (rays, cached per 1 m of camera motion).
Cost ~8 ms per frame.

## Simulated array (`sfhack/acoustic/sim_array.py`)
`simulate_bearing(mission_dir, seed, ArrayLevers(...)) -> (BearingEstimate | None, info)`; `info` has the true
azimuth, SNR, bias, noise sigma and the error. See the module docstring for the budget (heuristic, not validated).
