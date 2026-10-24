# REPORT night-thermal (branch `night-thermal`, C:\dev\night-thermal)

Agent night-thermal, night of 2026-10-23 -> 24. Task: thermal world model (truth + onboard prior) and the
simulated downward LWIR camera, per the interface contract in NIGHT_PLAN.md. Details: `docs/THERMAL_MODEL.md`.

## What works

- `python -m sfhack.thermal build --mission <id> [--hour 23] [--season summer] [--seed 0] [--contrast 1.0]`
  writes `world/thermal_truth.npz` (simulator only) and `onboard/thermal_prior.npz` (+ `drone/` copy). Built for
  **berkeley, m-c804bb84 and ukraine-1** (8 / 15 / 27 s).
- Physics: material classes per 0.5 m pixel from ortho + NIR (NDVI) + OSM (highway type and `surface`, landuse,
  parking, water) + LiDAR canopy height (US) / Otsu + OSM wood (elsewhere); 1-D surface energy balance per
  material x sun exposure x sky view factor (from the DSM), at the local solar time of `hour`.
  Calibrated to night-research's literature ranges at 23 h: asphalt +6.0 K over air, grass -4.0, canopy -1.9,
  water +3.8, dirt roads -2.1 (ASTER: unpaved roads are slightly cool), insulated roofs -3.8 / -6.6, heavy roofs
  +3.8, asphalt - grass 9.9 K, and asphalt cooling 2.6 K/h
  then 1.5 K/h.
- Truth vs prior: the truth adds roof construction per building, parked cars (cold; 15 % with a hot hood),
  wet and shaded patches, HVAC, per-road / per-building / per-class offsets, cold-air pooling (1.4 K / 100 m),
  smooth fields, and texture. The prior is built from onboard data only (degraded ortho, perturbed landmarks,
  onboard DSM, 5 % of minor OSM ways missing) and has a per-pixel sigma.
  Prior vs truth: corr 0.72 / 0.74 / 0.71; roads RMSE 3.0 / 2.6 / 2.5 K; the residuals are heavy-tailed.
- Ukraine (Sentinel-2 10 m, GLO-30, no LiDAR): coarse imagery does not hide OSM roads; untagged village lanes
  are uncertain (the truth makes 40 % of them paved; the prior marks them dirt with higher sigma, `roads = 0`).
- NIR: older NAIP missions get `world/nir.png` + `onboard/nir.png` (NAIP band 4) added once by the thermal
  build. New missions get it from night-demo's missionprep (same file name and format, agreed by message).
- Camera `ThermalCamera`: 640 x 512, 50 deg HFOV, a `thermal` camera on the X2 (same mount as `down`). One GL pass
  gives depth + segmentation; each pixel goes back to world xyz and samples the truth raster; walls, trunks,
  objects, drone and sky get their own temperatures. Then blur, vignetting, FPN, absolute calibration error
  (2 K / 3 %), drift with flat-field correction, NETD 50 mK, and `to_8bit` AGC white-hot.
  **70 ms per frame (14 Hz)** in flight when it shares the simulator's GL context (`renderer=sim.renderer`);
  129 ms with a second context. The requirement was >= 10 Hz.
- Onboard helpers: `load_prior`, `ThermalPrior.predict_view` (expected image from a pose; tested against the
  camera, corr > 0.95 at 3 yaws).
- `load_truth(contrast=0.4)`: a mild-night robustness row without rebuilding (night-research's ASTER check:
  1-3 K road contrast on mild nights).
- Demos: `demos/20_thermal_view.py` (fly the route: camera | ortho | prior, `--record`, `--snap`),
  `demos/21_thermal_figures.py` (docs figures). Docs: `docs/THERMAL_MODEL.md` + `docs/figures/`.
- Tests: `tests/test_thermal.py` covers physics, classification, camera geometry and orientation, noise,
  contrast, predict_view vs camera, and mission raster alignment. Truth separation is extended (the thermal
  truth and camera are simulator side; onboard code may only use `thermal/prior.py`). Full suite green.

## Numbers

See the tables in `docs/THERMAL_MODEL.md` (calibration vs literature; per-mission stats).

## What failed / limits

- Light insulated roofs come out at -6.6 K (literature typical -4..-6): slightly cold.
- NAIP alone cannot tell crowns from lawns (same NDVI and brightness). Without LiDAR the spectral fallback is
  weak; ukraine-1 leans on OSM wood / meadow polygons.
- The truth is 2.5-D (one temperature per wall, nothing under roofs or crowns). Cars and wet patches are static
  per build.
- Hour / season: summer is calibrated; other seasons are rough presets. Weather (clouds, wind, rain) is not
  modelled tonight; the hooks are in `Climate` / `seb.py`.
- With a second GL context the camera falls to ~8 Hz in flight; always pass `renderer=sim.renderer`.

## Next steps

- Validate against a real night thermal scene (night-research's ASTER work) and tune the contrast per site.
- Per-flight randomization of cars / wet patches (seeded) instead of per build.
- Wind and cloud as parameters (contrast drops 30-80 %), then a robustness sweep in the nav eval.

## Disclosure (pre-hackathon work, all built tonight by an AI agent, Claude Opus 5.5)

Everything in this branch beyond `origin/v5-mission-planner` was written tonight, 2026-10-23 23:30 to
2026-10-24 07:00: `sfhack/thermal/*`, `demos/20_thermal_view.py`, `demos/21_thermal_figures.py`,
`docs/THERMAL_MODEL.md`, `docs/figures/*`, `tests/test_thermal.py`, the `thermal` camera in
`sfhack/sim/mujoco_sim.py`, the `CameraConfig.thermal_*` fields, and the thermal checks in
`tests/test_truth_separation.py`. Data written into `$SFHACK_DATA/missions/*` (new files only):
`world/thermal_truth.npz`, `world/thermal_build.json`, `onboard/thermal_prior.npz`, `drone/thermal_prior.npz`,
and `world/nir.png` + `onboard/nir.png` (+ `drone/`) for berkeley and m-c804bb84.
Material constants come from the literature; calibration targets came from night-research's notes.
