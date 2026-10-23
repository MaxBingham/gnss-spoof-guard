# MERGE_NOTES - branch `v4-target-selection`

V4 adds target recognition and a marker-confirmed precision landing on the brown truck. Almost all of it
lives in new files. With the feature off (no `extension=` passed to `FlightSystem`, `camera.down_blur_px`
at its default 0), every existing demo and test behaves exactly as before.

## New files

| File | Side | What |
| --- | --- | --- |
| `sfhack/world/target_scene.py` | simulator (truth) | seeded layout of the six objects, difficulty levels, ArUco pad texture, plain-field map package |
| `sfhack/sim/target_world.py` | simulator + harness | MuJoCo geometry of the scene; `TargetExtension` (FlightSystem hook, truth scoring) |
| `tests/test_target_scene.py` | test | layouts, difficulty constraints, marker resolvable only low, rangefinder sees the roof |

## Shared-file changes (all additive)

| File | Change | Why |
| --- | --- | --- |
| `scripts/env.ps1`, `scripts/env.sh`, `scripts/setup.ps1`, `scripts/setup.sh` | default venv `~/.venvs/sf-hack-v4` | separate venv for this branch (task setup). **Revert on merge** if main keeps `sf-hack`. |
| `sfhack/config.py` | new `TargetConfig` dataclass + `Config.target` field | all V4 parameters in the existing config style; ignored unless the target extension is used |
| `sfhack/config.py` | new `CameraConfig.down_blur_px = 0.0` | camera optics. Sharp MuJoCo renders let a 1.6 m marker decode from 25 m; 1.2 px blur (target extension only) makes it resolvable below ~14 m only. Default 0 = unchanged images |
| `sfhack/sim/mujoco_sim.py` | `import cv2`; blur the down frame when `down_blur_px > 0` | applies the optics above (no-op at the default) |
| `sfhack/messages.py` | new `TargetCandidate`, `TargetCandidates`, `MarkerDetection` | bus messages of the target pipeline (overlays, logging) |
| `sfhack/bus.py` | new topics `target/candidates`, `target/marker` | same |
