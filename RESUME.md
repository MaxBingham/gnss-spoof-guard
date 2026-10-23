# RESUME - V4 truck detector (branch `v4-detector`), paused 2026-10-23 15:51

## State

- Dataset done: `$SFHACK_DATA/v4/detector/dataset` (train 6000 / val 1000 / test 1000 / test_nopad 300, data.yaml,
  DATASET_CARD.md, examples/). No need to regenerate.
- Training (yolo26n, 640 px, batch 8, AMP, RTX 500) stopped cleanly after **epoch 5 of 20** (Ultralytics sized the
  run to 20 epochs from the 1 h time cap). Val after epoch 5: P 0.95-ish, R 0.947, mAP50 0.985, mAP50-95 0.876.
  - `$SFHACK_DATA/v4/detector/train/weights/last.pt` = end of epoch 5, with optimizer state (verified loadable).
  - `last_epoch5_backup.pt` = copy of it. `best.pt` = same epoch.
- Not run yet: evaluation (15), end-to-end selection (16), OWLv2 baseline (17), report (18). The scripts passed a
  smoke test on a small dataset (`dataset_quick`, `eval/smoke`, `e2e/yolo_test_smoke` are throwaway outputs).
- Commits are local only: `git push` was blocked by the permission classifier. Push by hand:
  `git push -u origin v4-detector`.
- GPU check done: torch 2.11.0+cu128, `torch.cuda.is_available()` True, RTX 500 Ada 4 GB. Batch 8 = 1.3 GB.

## Commands (PowerShell, repo root)

```
$env:UV_PROJECT_ENVIRONMENT = "$HOME\.venvs\sf-hack-det"; . scripts/env.ps1
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # must be True

# 1. resume training from last.pt (same args: batch 8, cache ram, time cap, early stopping)
uv run yolo train resume model="$HOME\sf-hack-data\v4\detector\train\weights\last.pt"
#    (time cap: Ultralytics restores time=1.0 h, so it trains up to ~1 h more; to stop earlier,
#     Ctrl+C after any epoch - last.pt / best.pt are saved every epoch)
#    train_summary.json is written only by demos/14; after a resume it is optional (the report skips it).

# 2. evaluate on the held-out test (writes train/calibration.json = val-chosen threshold + temperature)
uv run python demos/15_det_eval.py

# 3. end-to-end selection: sanity on val episodes, then the 200 held-out episodes
uv run python demos/16_det_select.py --split val --episodes 100
uv run python demos/16_det_select.py

# 4. OWLv2 baseline (weights already in the HF cache), then its e2e (optional)
uv run python demos/17_det_owlv2.py
uv run python demos/17_det_owlv2.py --negatives
uv run python demos/16_det_select.py --detector owlv2          # optional, ~20 min

# 5. report -> $SFHACK_DATA/v4/detector/RESULTS.md (+ runtime ms/frame GPU and CPU), then tests
uv run python demos/18_det_report.py
uv run python -m pytest tests -q
```

## Then still to do

- Fill the numbers into README "V4 learned truck detector" and print CHECKPOINT 3-5 (training time, test metrics with
  Wilson CIs, e2e k/200, OWLv2 comparison, failure modes).
- `tests/test_truck_detector.py::test_detector_on_rendered_frame_and_speed` runs once the weights exist.
- The selector parameters (`DetectorConfig.select_*`) were set a priori; check them on the val episodes only, never
  tune on the test episodes.
