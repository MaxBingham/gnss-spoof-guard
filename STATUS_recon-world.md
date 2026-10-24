# STATUS recon-world

Updated 2026-10-24 (first version).

## Works
- **Data:** `uv run python -m sfhack.sim.recon_world --mission ukraine-1 [--seeds 20]` writes the base, lights, stray
  lights and targets into the mission package (already run on the shared `$SFHACK_DATA`, seeds 0-19).
- **Base at A** (`sfhack/world/base.py` onboard-safe types + readers; `sfhack/sim/recon_world.py` generator;
  `world/scene.py` `_add_base`): 150 m square fence, 12 fence lights (F00 north, clockwise, every 30 deg, 4 m poles),
  6 outer posts at 400 m (O00-O05, only az -15..105 deg: A is 150 m from the map's W and S edges), visible pad with an
  "H", UMA-16 mast. Lights are emissive geoms in group 2, blinked by `sim/base_lights.py BlinkAnimator`.
  Onboard: `onboard/base_lights.json` (5 cm survey error), `onboard/base.json` (pad, pad yaw, UMA-16, fence).
- **Blink codes:** 22 aperiodic 8-bit necklaces (weight 3-5), 10 Hz bit rate, every light with its own phase;
  `decode_code(bits_any_phase, codes)`.
- **Targets** (`world/target_scene.json`, per seed `world/recon/seed_<k>/`): 2-3 running vehicles (truck / car /
  APC) in a column on the nearest road, 0-4 people, 3-5 distractors (cooling car, barn, warm rocks, cow, campfire).
  Each object is made of primitive parts with their own temp_K (thermal/camera.py patched to read them). Truth for
  scoring: `recon_truth.json` (bearing, distance, group position).
- **Simulated array** (`sfhack/acoustic/sim_array.py`): a level budget (source level, spreading + 3 dB/km, night
  ambient + wind noise, 12 dB array gain) -> SNR -> bias + noise -> `BearingEstimate` (sigma 5-9 deg). Levers:
  sigma, bias, wind. Audio mode runs recon-acoustic's real SRP-PHAT pipeline on synthesized 16-ch audio.
  Seed 0: true 77.7 deg, measured 80.1 +- 5.6 (calm); 8 m/s wind: 83.8 +- 8.9.
- **Rear NIR camera** (`sfhack/sim/rear_camera.py`, `NirFrame` on `camera/rear`, 640 x 480, 20 deg, 20 Hz): lights
  projected with the true pose (PSF + halo + bloom, 1/r^2, beam lobe, haze), terrain/building occlusion by ray, a
  dark MuJoCo background, noise and hot pixels. 1-axis tilt gimbal (`CameraTiltCmd` on `camera/rear/tilt`).
- **Demo:** `uv run python demos/41_base_view.py [--headless] [--camera overview]`.
- Tests: `tests/test_recon_world.py` (codes, onboard-safe reader, array statistics + levers, rear camera spots at their
  projection). Truth separation stays green.

## Numbers (ukraine-1, 80 m AGL along 77 deg, gimbal aimed at the base)
Decodable lights (in view, not occluded, peak >= 25 DN when on): 250 m orbit 0-2, 0.36 km 2, 0.6 km 7, 0.84 km 4,
1.1-1.8 km 1-3 (outer posts only). **All fence lights are behind terrain beyond ~0.9 km** toward 60-90 deg (a ~100 m
hill 400 m east of A; Copernicus GLO-30 is a surface model).

## ukraine-recon (optional mission, built today)
Built with missionprep (A 49.2922 N 23.4312 E on the plain ~7 km NE of ukraine-1, B 3 km NE, 600 m margin) + the
thermal build; 3.3 x 3.3 km, 1281 buildings. Base, lights and 20 target seeds generated (groups 1.6-2.6 km out at
10-105 deg). Line of sight from the fence lights at 60-80 m AGL: clear to 3 km over 30-60 deg, to 2.7 km over 0-90 deg.
Fly it with `--mission ukraine-recon` wherever a mission id is taken.

## Simulated bearings (seeds 0-19, calm)
Every group is heard (each has >= 1 truck or APC). ukraine-1: |error| median ~4.5 deg, max 12.8 deg, reported sigma
5.2-7.1 deg. ukraine-recon (farther groups): median ~4.8 deg, max 19.7 deg, sigma 5.3-10.6 deg.

## Blockers / decisions
- ukraine-1 is 2.0 x 1.0 km with A in the SW corner: targets fit only at 1.5-1.9 km toward 60-90 deg, and the fence
  lights are hidden there. ukraine-recon fixes both; ukraine-1 stays the default until the integrator decides.
- Codes: recon-nav notes some 8-bit codes are 1 bit apart under rotation; 10-bit weight 5-7 codes with cyclic distance
  >= 2 would be more robust (not changed: their decoder resolves it with a margin).
