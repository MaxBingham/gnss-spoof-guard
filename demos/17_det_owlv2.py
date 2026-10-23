"""V4 detector, baseline: zero-shot open-vocabulary detection (OWLv2, Hugging Face transformers) on the same
held-out test set, same truck metrics (demos/15_det_eval.py), threshold and temperature chosen on val.

    uv run python demos/17_det_owlv2.py                    # prompt "a brown box truck" only
    uv run python demos/17_det_owlv2.py --negatives        # + competing prompts (car, pickup, SUV, tent, hut, tree)

Model: google/owlv2-base-patch16-ensemble (~620 MB, Apache-2.0), fp16 on the GPU. Val is subsampled
(--val-n) to keep the time box; test is the full 1000 images + test_nopad.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import detlib as dl  # noqa: E402

from sfhack.config import data_dir  # noqa: E402

ev = __import__("importlib").import_module("15_det_eval")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=data_dir() / "v4" / "detector" / "dataset")
    ap.add_argument("--negatives", action="store_true")
    ap.add_argument("--val-n", type=int, default=400)
    a = ap.parse_args()
    name = "owlv2_neg" if a.negatives else "owlv2"
    out = data_dir() / "v4" / "detector" / "eval" / name
    out.mkdir(parents=True, exist_ok=True)
    det = dl.OwlDetector(negatives=a.negatives, conf_min=0.01)
    ms = []
    for split, n in (("val", a.val_n), ("test", None), ("test_nopad", None)):
        path = out / f"preds_{split}.npz"
        if path.exists():
            continue
        gts = dl.load_gt(a.root, split)[:n]
        preds = []
        t0 = time.time()
        for g in gts:
            rgb = cv2.cvtColor(cv2.imread(str(a.root / "images" / split / f"{g.image}.jpg")), cv2.COLOR_BGR2RGB)
            preds.append(det.predict_raw(rgb))
            ms.append(det.last_ms)
        dl.save_preds(path, [g.image for g in gts], preds)
        print(f"  {split}: {len(gts)} images in {time.time() - t0:.0f} s", flush=True)
    res = ev.evaluate(name, out, a.root, out, classes=(dl.TRUCK,))
    cal = {"temperature": res["calibration"]["T"], "truck_threshold": res["truck_threshold"], "model": dl.OWL_MODEL,
           "prompts": det.prompts, "ms_per_image_median": float(np.median(ms)) if ms else None}
    (out / "calibration.json").write_text(json.dumps(cal, indent=2), encoding="utf-8")
    t = res["test"]["truck"]
    print(f"[{name}] T = {cal['temperature']:.3f}, threshold {cal['truck_threshold']:.3f} (val), "
          f"test truck AP50 {res['test']['per_class']['truck']['ap50']:.3f}")
    print(f"  test truck recall {dl.fmt_rate(t['recall_k'], t['recall_n'])}; FP images {dl.fmt_rate(t['fp_images'], t['n_images'])}")
    print(f"  truck ECE test: {res['calibration']['test']['truck_ece_before']:.3f} -> {res['calibration']['test']['truck_ece_after']:.3f}")
    print("  called truck:", {k: f"{v[0]}/{v[1]}" for k, v in t["called_truck"].items()})


if __name__ == "__main__":
    main()
