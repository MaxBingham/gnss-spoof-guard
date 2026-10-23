"""V4 detector, step 3: train Ultralytics YOLO nano on the synthetic dataset (demos/13_det_dataset.py).

    uv run python demos/14_det_train.py                         # yolo26n, 640 px, GPU, <= 60 min, early stopping
    uv run python demos/14_det_train.py --device cpu --epochs 5   # CPU fallback (slow)

Ultralytics is AGPL-3.0 (fine for a private hackathon prototype; see README). Weights land in
$SFHACK_DATA/v4/detector/train/weights/best.pt (= DetectorConfig.weights); train_summary.json records the
model, epochs, batch, device and wall time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sfhack.config import data_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo26n.pt", help="pretrained nano checkpoint (COCO)")
    ap.add_argument("--data", type=Path, default=data_dir() / "v4" / "detector" / "dataset" / "data.yaml")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hours", type=float, default=1.0, help="wall-time cap (Ultralytics 'time', overrides epochs)")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8, help="RTX 500 (4 GB), AMP: 8 = 1.3 GB, 17 img/s; 16 = 2.6 GB, 13 img/s")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--name", default="train")
    a = ap.parse_args()

    import torch
    from ultralytics import YOLO

    root = data_dir() / "v4" / "detector"
    pre = root / "pretrained"
    pre.mkdir(parents=True, exist_ok=True)
    os.chdir(pre)  # Ultralytics downloads the checkpoint (and its AMP-check model) into the working directory
    device = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    if device == "cpu" and a.device != "cpu":
        print("CUDA not available: training on CPU (fewer epochs advised)")
    model = YOLO(a.model)
    t0 = time.time()
    batch = a.batch
    while True:
        try:
            model.train(data=str(a.data), imgsz=640, epochs=a.epochs, time=a.hours, patience=a.patience, batch=batch,
                        device=device, workers=a.workers, cache="ram", project=str(root), name=a.name, exist_ok=True,
                        seed=0, deterministic=False, close_mosaic=5, flipud=0.5, fliplr=0.5, degrees=0.0,
                        hsv_h=0.01, plots=True, verbose=True)
            break
        except torch.OutOfMemoryError:
            if batch <= 4:
                raise
            batch //= 2
            print(f"CUDA out of memory: retrying with batch {batch}")
            torch.cuda.empty_cache()
            model = YOLO(a.model)
    wall = time.time() - t0
    out = root / a.name
    rows = (out / "results.csv").read_text(encoding="utf-8").strip().splitlines()
    summary = {"model": a.model, "device": device, "gpu": torch.cuda.get_device_name(0) if device != "cpu" else None,
               "batch": batch, "imgsz": 640, "epochs_run": len(rows) - 1, "epochs_max": a.epochs, "time_cap_h": a.hours,
               "patience": a.patience, "wall_s": round(wall, 1), "weights": str(out / "weights" / "best.pt"),
               "torch": torch.__version__, "ultralytics": __import__("ultralytics").__version__}
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
