# V3.2 merge notes: robust position without GNSS

Branch `v3.2-gnss-free`, based on `cursor/viewer-polish` @ `ce137a4`. Not merged, no PR.
Venv: `~/.venvs/sf-hack-v32` (set unconditionally by `scripts/env.ps1` / `env.sh` on this branch).
Tests: 34/34 green, including `tests/test_truth_separation.py` (unchanged, not weakened).

## What was wrong (step 1)

In the live demo, `G` cycles GNSS `ok -> drag -> jump -> jam -> off`. A live "jam" therefore always comes right
after a 50 m GNSS jump, and the naive EKF fuses every GNSS fix. The plain jam (`--cond-at "40:gnss=jam"`) was
already 10/10 headless. The jump was the trigger:

1. The jump moved the EKF accel-bias states (bx, by) by ~1 m/s^2 in one update. `take_tilt_correction` handed
   that to the AHRS as a **5.5 deg tilt error** within ~1 s.
2. After the jam the drone ran off at ~20 m/s while the EKF claimed 2-sigma 1-5 m. Vision searched a 6-10 m
   window around a prior that was 50-200 m wrong. Health went HOLD -> RETURN -> LAND, and the drone landed
   0.9-3 km from B (0/10).

The checks you suggested, answered:
- Vision already ran while GNSS was ok.
- Correct vision fixes were not rejected by NIS. They were never produced, because the search window around
  the wrong prior was too small. Worse, during the jump vision "confirmed" the spoofed position (wrong matches
  in the small window, NIS 2.2).
- Health went to LAND (2-sigma > 45 m) about 30 s after the jam.
- A large covariance inflation (I tried 20 m with a 60 m window) made things worse. A 60 m NCC window at 40 m
  altitude is ambiguous almost everywhere: no fixes for 27 s, then a confident fix 65 m wrong. The plain jam
  dropped to 4/10, so that change was reverted.

Fix: cap the bias step per update, run a GNSS-free track in parallel and hand over to it on GNSS loss, fall back
to a modest inflation plus a strong-fix re-localization, and HOLD instead of LAND while blind.

## Evaluation (step 4)

`uv run python demos/11_eval_table.py --seeds 10` (120 flights, ~6 min on 18 workers). Live-demo mission: vision
localizer, GNSS ok from takeoff, jammed at t = 40 s. Weather and wind are set from t = 0. "Camera lost" starts
with the jam. "max true est. error" is |estimate - truth| during flight (scored on truth, harness only).

| Scenario | TRN | within 3 m | within 10 m | final error median / mean / max (m) | max 2-sigma (m) | max true est. error (m) | sources fused |
|---|---|---|---|---|---|---|---|
| no jam (GNSS ok) | on | 10/10 | 10/10 | 0.9 / 1.1 / 2.6 | 0.7 | 2.4 | gnss, vision |
| no jam (GNSS ok) | off | 10/10 | 10/10 | 0.9 / 1.1 / 2.6 | 0.7 | 2.4 | gnss, vision |
| jam at 40 s | on | 10/10 | 10/10 | 0.6 / 0.7 / 1.4 | 4.0 | 5.5 | gnss, trn, vision |
| jam at 40 s | off | 10/10 | 10/10 | 0.7 / 0.8 / 1.5 | 5.0 | 7.9 | gnss, vision |
| jam + fog 0.65 | on | 10/10 | 10/10 | 0.7 / 0.8 / 1.7 | 6.9 | 6.7 | gnss, trn, vision |
| jam + fog 0.65 | off | 10/10 | 10/10 | 0.8 / 0.9 / 2.0 | 19.1 | 23.1 | gnss, vision |
| jam + camera lost | on | 0/10 | 1/10 * | 35.2 / 30.1 / 39.2 | 8772.3 | 2204.9 | gnss, trn, vision |
| jam + camera lost | off | 0/10 | 0/10 | 649.5 / 798.3 / 2057.2 | 12531.6 | 2780.7 | gnss, vision |
| jam + wind 6 m/s | on | 10/10 | 10/10 | 0.9 / 0.8 / 1.1 | 5.3 | 9.0 | gnss, trn, vision |
| jam + wind 6 m/s | off | 10/10 | 10/10 | 0.9 / 0.9 / 1.2 | 5.4 | 9.4 | gnss, vision |
| G-key cycle: drag, jump, jam | on | 10/10 | 10/10 | 0.6 / 0.7 / 1.4 | 5.5 | 55.4 ** | gnss, trn, vision |
| G-key cycle: drag, jump, jam | off | 10/10 | 10/10 | 0.8 / 0.8 / 1.4 | 11.2 | 55.4 ** | gnss, vision |

\* The one seed within 10 m (5.3 m) ended as MISSION_TIMEOUT, not a landing.
\*\* During the 4 s of naive spoof fusion before the jam. The handover then moves the estimate ~76 m back.

Before V3.2 (base `ce137a4`, same harness): G-key cycle 0/10, final error 0.9-3 km. Plain jam 10/10 (0.77 m).
No jam: identical numbers. GNSS off (V2 mode): base 10/10 with mean 0.80 m, the step-1 code bit-identical to
that, and 10/10 with mean 0.72 m once TRN is on. The spoof scenario `--scenario-at 25`, seeds 0-4, is
**bit-identical** to the base (117.5-119.5 m from B, health NOMINAL), both after step 1 and with TRN on (default).

### TRN acceptance (step 2): not met, and why

Target: GNSS jammed and camera lost, B reached within ~10 m in >= 7/10 seeds. Result: 0/10 landed within 10 m.
The final error is ~30 m, down from ~650 m without TRN.
- Over buildings the TRN filter works. The cloud converges to 1-3 m and its own estimate stays within 0-3 m of
  truth (measured per second, `demos/10_eval_levers.py --trace`, plus a filter test).
- Flat campus: the route has a 120 m lawn and plaza stretch (y = -25..95 m) and the **last 38 m before B are
  flat**. Without terrain relief there is no information, and closed-loop dead reckoning drifts **2-3 m/s** in
  this sim. Calm air applies no airframe drag, so the accelerometer cannot see velocity. Most seeds end ~38 m
  short of B: health holds at the last building edge, and hovering gives TRN nothing new.
- Paths forward, not done here: a rotor-drag model in the sim plus drag-velocity fusion (death-v3 has both;
  it changes baseline physics, so it needs a team decision), fusing TRN's per-particle velocity-error estimate
  into the EKF velocity, or a planner that routes the final approach over structure.

Step 3 (LiDAR odometry / KISS-ICP) was skipped on request.

## Changed shared files and why

| File | Change |
|---|---|
| `sfhack/estimation/ekf.py` | Per-update cap on the accel-bias step (`acc_bias_step_max`). GNSS-loss detection (`gnss_lost_now`). Fallback `inflate_for_gnss_loss` with strong-fix-only re-localization (`relocalizing`, `reloc_min_quality`). `adopt()` for the handover. `bias_handed` (so a parallel filter on the same AHRS stays consistent). `vel_var_growth` (TRN). `update_fix(..., allow_reloc=False)` so TRN never forces itself in. |
| `sfhack/autopilot.py` | GNSS-free track `ekf_free`: a deepcopy at the first GNSS fix, then IMU + baro + vision only (its own localizer call around its own prior, before the main call so the panels still show the main one). `_on_gnss_lost`: handover or fallback, logged as `HANDOVER` / `GNSS_LOST` decisions. TRN wiring (`on_lidar`; prior = GNSS-free track; fixes go into the GNSS-free track always and into the main filter only while no GNSS arrives). `trn_tracking`, `receiving_gnss`, `sources_used`. New ctor arg `seed`. |
| `sfhack/planning/health.py` | `hold_when_blind`: a huge 2-sigma means HOLD, and LAND only after `land_no_fix_s` without any fix. `update(..., trn_tracking)` uses wider HOLD limits while TRN tracks (hovering starves TRN). |
| `sfhack/config.py` | New `EkfConfig`, `HealthConfig` and `TrnConfig` fields (below), `Config.trn`. `vision_search_max_m` unchanged (40). |
| `sfhack/messages.py` | `VisualFix.quality` (0..1, default 1.0; vision = NCC peak), used to gate re-localization. |
| `sfhack/perception/map_localizer.py` | Sets `quality=best` on its fixes (one argument). |
| `sfhack/runner.py` | Passes `seed` to the autopilot (TRN RNG). |
| `sfhack/world/scene.py` | `footprint_obb` / `building_boxes` moved to `world/buildings.py` and re-imported (same names, same geometry), because the onboard DSM needs them and scene.py imports MuJoCo. |
| `demos/demo_live.py` | `--trn on\|off`. `LiveDemo(..., cfg=)`. HUD "Fixes from" row (`sources_line`). |
| `demos/liveview/hud.py` | `HudState.sources_line` (default ""); the mission card grows 17 px when it is set. |
| `scripts/env.ps1`, `scripts/env.sh` | Venv `~/.venvs/sf-hack-v32`, always set (do **not** merge these two lines to main). |

New files: `sfhack/perception/trn.py` (port of death-v3's TRN, 8-beam lidar), `sfhack/world/onboard_dsm.py`,
`sfhack/world/buildings.py`, `sfhack/world/surface.py` (verbatim from death-v3), `demos/10_eval_levers.py`
(multi-seed eval with lever schedule + per-second trace), `demos/11_eval_table.py`, `tests/test_gnss_loss.py`.

Truth separation: the new onboard modules (`perception/trn.py`, `world/onboard_dsm.py`, `world/buildings.py`,
`world/surface.py`) import neither the simulator nor `degradation`. The DSM is built from the onboard map package
(`landmarks.json`), with its own fixed map error (0.5 m building height std, 1 m raster).

## New config flags and how to turn each feature off

| Feature | Flag (default) | Off |
|---|---|---|
| Bias-step cap | `ekf.acc_bias_step_max = 0.03` m/s^2 | `ekf.acc_bias_step_max=1e9` |
| GNSS-free track + handover | `ekf.gnss_free_track = True`, `handover_max_fix_age_s = 5`, `handover_max_sigma2 = 10` | `ekf.gnss_free_track=False` |
| GNSS-loss detection | `ekf.gnss_loss_after_s = 1.0` | `ekf.gnss_loss_after_s=1e9` (disables handover and fallback) |
| Fallback inflation | `ekf.gnss_loss_pos_std = 5`, `gnss_loss_vel_std = 1`, `reloc_min_quality = 0.6` | `ekf.gnss_loss_pos_std=0`, `ekf.gnss_loss_vel_std=0`, `ekf.reloc_min_quality=0` |
| HOLD instead of LAND while blind | `health.hold_when_blind = True` | `health.hold_when_blind=False` |
| TRN | `trn.enabled = True` | `--trn off` (demo_live, 10_eval_levers) or `cfg.trn.enabled = False` |
| TRN: keep flying while tracking | `health.hold_sigma2_trn = 40`, `hold_no_fix_s_trn = 30` | set them to `hold_sigma2` / `hold_no_fix_s` (20 / 15) |

`demos/10_eval_levers.py --set section.field=value` overrides any of these per run.

## Reproduce

```
. scripts/env.ps1      # or: . scripts/env.sh
uv run python -m pytest tests -q
uv run python demos/demo_live.py --headless --events "40:G,42:G,44:G"          # the live G-key path, lands at B
uv run python demos/10_eval_levers.py --seeds 10 --cond-at "40:gnss=jam" --trace
uv run python demos/10_eval_levers.py --seeds 10 --cond-at "40:gnss=jam,camera=lost" --trn on
uv run python demos/11_eval_table.py --seeds 10
```
