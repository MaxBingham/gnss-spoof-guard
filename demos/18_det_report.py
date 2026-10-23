"""V4 detector: assemble $SFHACK_DATA/v4/detector/RESULTS.md from the dataset card, training summary, evaluation
(demos/15, 17), end-to-end selection (demos/16) and a runtime benchmark of the onboard detector (GPU, CPU).

    uv run python demos/18_det_report.py
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import detlib as dl  # noqa: E402

from sfhack.config import data_dir  # noqa: E402

D = data_dir() / "v4" / "detector"


def load(p: Path) -> dict | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pct(k: int, n: int) -> str:
    return dl.fmt_rate(k, n)


def bench() -> dict:
    import torch

    from sfhack.perception.truck_detector import TruckDetector
    files = sorted((D / "dataset" / "images" / "test").glob("*.jpg"))[:40]
    imgs = [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in files]
    R = np.eye(3)
    out = {}
    for dev in (["0"] if torch.cuda.is_available() else []) + ["cpu"]:
        det = TruckDetector(device=dev)
        for im in imgs[:5]:
            det.detect(im, 30.0, R)
        ms = []
        for im in imgs:
            t0 = time.perf_counter()
            det.detect(im, 30.0, R)
            ms.append(1000 * (time.perf_counter() - t0))
        out["GPU (" + torch.cuda.get_device_name(0) + ")" if dev != "cpu" else "CPU"] = (float(np.median(ms)), float(np.percentile(ms, 95)))
    return out


def main() -> None:
    card = (D / "dataset" / "DATASET_CARD.md").read_text(encoding="utf-8")
    train = load(D / "train" / "train_summary.json")
    y = load(D / "eval" / "yolo26n" / "metrics.json")
    owl = {k: load(D / "eval" / k / "metrics.json") for k in ("owlv2", "owlv2_neg")}
    owl_cal = {k: load(D / "eval" / k / "calibration.json") for k in owl}
    e2e = {k: load(D / "e2e" / f"{k}_test" / "e2e.json") for k in ("yolo", "owlv2", "owlv2_neg")}
    e2e_val = load(D / "e2e" / "yolo_val" / "e2e.json")
    speed = bench()
    figs = D / "figures"
    figs.mkdir(exist_ok=True)
    for src, dst in ((D / "eval" / "yolo26n" / "pr_truck.png", "pr_truck.png"),
                     (D / "eval" / "yolo26n" / "confusion_test.png", "confusion_test.png"),
                     (D / "eval" / "yolo26n" / "reliability.png", "reliability.png"),
                     (D / "eval" / "yolo26n" / "breakdown_test.png", "breakdown_test.png"),
                     (D / "dataset" / "dataset_conditions.png", "dataset_conditions.png"),
                     (D / "train" / "results.png", "training_curves.png")):
        if src.exists():
            shutil.copy(src, figs / dst)

    t = y["test"]["truck"]
    md = []
    E = e2e["yolo"]
    if E:
        lo, hi = E["ci95"]
        pitch = (f"Correct target selection: {E['correct']}/{E['episodes']} = {100 * E['correct'] / E['episodes']:.1f} % "
                 f"(95 % CI {100 * lo:.1f}-{100 * hi:.1f} %), sim, held-out models")
        md += ["# V4 truck detector - results", "", f"**{pitch}**", ""]
    md += ["Everything below is simulation (MuJoCo renders of library models). Test = held-out truck model "
           "(`box_truck_b`) and car model (`hatch_b`), held-out ground textures, new seeds. Operating point and "
           "calibration were chosen on val only. Intervals: 95 % Wilson.", ""]

    md += ["## Model and training", ""]
    if train:
        md += [f"- {train['model']} (COCO-pretrained, Ultralytics {train['ultralytics']}, AGPL-3.0), 640 px, batch {train['batch']}, "
               f"AMP, on {train['gpu'] or 'CPU'}; {train['epochs_run']} epochs in {train['wall_s'] / 60:.0f} min "
               f"(wall-time cap {train['time_cap_h']} h, early-stopping patience {train['patience']}).",
               f"- Data: 6000 train / 1000 val / 1000 test / 300 test_nopad images, 640 x 480 (see dataset card below).",
               "- Calibration (val): temperature T = " + f"{y['calibration']['T']:.3f}; truck threshold "
               f"{y['truck_threshold']:.3f} (best truck F1 on val after calibration).", "",
               "![training](figures/training_curves.png)", ""]

    md += ["## Detection metrics (IoU 0.5)", "",
           "Precision / recall at the operating point (truck: val-chosen threshold, other classes 0.25), AP50 over all "
           "thresholds.", "", "| class | val AP50 | test AP50 | test precision | test recall | test labels |", "|---|---|---|---|---|---|"]
    for c in dl.CLASSES:
        v, s = y["val"]["per_class"][c], y["test"]["per_class"][c]
        md.append(f"| {c} | {v['ap50']:.3f} | {s['ap50']:.3f} | {s['precision']:.3f} | {s['recall']:.3f} | {s['n_gt']} |")
    md.append(f"| **mAP50** | {y['val']['map50']:.3f} | {y['test']['map50']:.3f} | | | |")
    md += ["", "![PR](figures/pr_truck.png)", "",
           "### Truck at the operating point (test)", "",
           f"- Truck recall (per labeled truck): **{pct(t['recall_k'], t['recall_n'])}**",
           f"- Images with a false truck box: **{pct(t['fp_images'], t['n_images'])}**",
           f"- Truck precision: {y['test']['per_class']['truck']['precision']:.3f}", "",
           "Which non-truck objects get a truck box (IoU >= 0.5, conf >= threshold), test:", "",
           "| object | called truck |", "|---|---|"]
    for k, (a, n) in sorted(t["called_truck"].items(), key=lambda kv: -kv[1][0] / max(kv[1][1], 1)):
        md.append(f"| {k} | {pct(a, n)} |")
    md += ["", "![confusion](figures/confusion_test.png)", ""]

    c = y["calibration"]
    md += ["## Calibration", "",
           f"Temperature scaling fitted on val truck predictions (conf >= 0.1, correct = IoU >= 0.5 with a truck): T = {c['T']:.3f}.", "",
           "| split | truck ECE before | truck ECE after | all-class ECE before | all-class ECE after | truck predictions |",
           "|---|---|---|---|---|---|"]
    for s in ("val", "test"):
        r = c[s]
        md.append(f"| {s} | {r['truck_ece_before']:.3f} | {r['truck_ece_after']:.3f} | {r['all_ece_before']:.3f} | "
                  f"{r['all_ece_after']:.3f} | {r['n_truck_preds']} |")
    md += ["", "![reliability](figures/reliability.png)", ""]

    md += ["## Breakdown (test, truck recall per labeled truck, images with a false truck box)", "",
           "| dimension | bin | recall | false-truck images |", "|---|---|---|---|"]
    for dim in ("altitude", "tilt", "weather", "occlusion", "ground"):
        for b, r in sorted(y["test"]["breakdown"][dim].items()):
            md.append(f"| {dim} | {b} | {pct(r['k'], r['n'])} | {pct(r['fp'], r['imgs']) if dim != 'occlusion' else '-'} |")
    md += ["", "![breakdown](figures/breakdown_test.png)", ""]
    if "pad_ablation" in y:
        p = y["pad_ablation"]
        md += ["### Does the model rely on the landing pad?", "",
               "The same 300 test scenes rendered with the pad + marker hidden (train hid it in ~20 % of images):", "",
               "| | truck AP50 | truck recall | false-truck images |", "|---|---|---|---|",
               f"| with pad | {p['ap50_with']:.3f} | {pct(p['with_pad'][0], p['with_pad'][1])} | {pct(p['with_pad'][2], p['with_pad'][3])} |",
               f"| pad hidden | {p['ap50_without']:.3f} | {pct(p['without_pad'][0], p['without_pad'][1])} | {pct(p['without_pad'][2], p['without_pad'][3])} |", ""]

    if E:
        md += ["## End-to-end target selection", "",
               f"{E['episodes']} randomized episodes (new seeds, held-out variants and grounds, weather as in the dataset). "
               f"The drone hovers above B at {E['alt']:.0f} m (arrival offset <= 3 m, hover jitter ~0.4 m / ~2 deg), "
               f"{E['frames_per_episode']} frames (2 s at 5 Hz). Detector inputs: RGB + altitude estimate (bias ~0.3 m) + "
               "attitude estimate (~0.5 deg error); ground positions in the EKF frame (nav bias ~1 m). Onboard "
               f"`TruckSelector`: clusters within {E['selector']['cluster_m']} m, truck seen in >= {E['selector']['min_frames']} "
               f"frames, margin {E['selector']['margin']} over the runner-up, else abstain. Correct = the chosen point "
               "lies on the truck footprint + 1.5 m (truth).", "",
               "| detector | correct | abstain | wrong (chosen object) | truck never in view |", "|---|---|---|---|---|"]
        for k, r in e2e.items():
            if r:
                md.append(f"| {k} | {pct(r['correct'], r['episodes'])} | {r['abstain']} | {r['wrong']} {r['wrong_objects'] or ''} | {r['truck_never_in_view']} |")
        if e2e_val:
            md += ["", f"Sanity check on val episodes (seen variants): {pct(e2e_val['correct'], e2e_val['episodes'])}."]
        fails = [x for x in E["episodes_detail"] if x["outcome"] != "correct"]
        md += ["", "Failure cases (images: `e2e/yolo_test/failures/`):", "",
               "| episode | outcome | chosen | reason | weather | ground | truck frames in view | occluding trees near truck |",
               "|---|---|---|---|---|---|---|---|"]
        for x in fails:
            md.append(f"| {x['episode']} | {x['outcome']} | {x['chosen']} | {x['reason']} | {x['weather']} {x['weather_i'] or ''} | "
                      f"{x['ground']} | {x['truck_in_view_frames']} | {x['truck_occluders']} |")
        md.append("")

    md += ["## Baseline: zero-shot OWLv2", ""]
    if any(owl.values()):
        md += [f"`{dl.OWL_MODEL}`, prompt \"{dl.OWL_TRUCK_PROMPT}\" (and a variant where competing prompts "
               f"{', '.join(dl.OWL_NEGATIVES)} must lose). Same test images and truck metrics; threshold and temperature "
               "chosen on 400 val images.", "",
               "| detector | test truck AP50 | recall @ val thr | false-truck images | brown car called truck | truck ECE (after T) | ms / image (GPU) |",
               "|---|---|---|---|---|---|---|"]
        rows = [("yolo26n (ours)", y, None)] + [(k, owl[k], owl_cal[k]) for k in owl if owl[k]]
        for nm, r, cal in rows:
            tt = r["test"]["truck"]
            bc = tt["called_truck"].get("brown_car", [0, 0])
            ms = f"{cal['ms_per_image_median']:.0f}" if cal and cal.get("ms_per_image_median") else \
                (f"{next(iter(speed.values()))[0]:.0f}" if nm.startswith("yolo") else "-")
            md.append(f"| {nm} | {r['test']['per_class']['truck']['ap50']:.3f} | {pct(tt['recall_k'], tt['recall_n'])} | "
                      f"{pct(tt['fp_images'], tt['n_images'])} | {pct(bc[0], bc[1])} | {r['calibration']['test']['truck_ece_after']:.3f} | {ms} |")
    else:
        md += ["Not run."]
    md.append("")

    md += ["## Runtime (onboard `perception/truck_detector.py`, 640 x 480 frame, median / p95 over 40 frames)", "",
           "| device | ms per frame | p95 |", "|---|---|---|"]
    for k, (m, p95) in speed.items():
        md.append(f"| {k} | {m:.1f} | {p95:.1f} |")
    md += ["", "## Dataset", "", card.split("\n", 1)[1].replace("![conditions](dataset_conditions.png)", "![conditions](figures/dataset_conditions.png)"), ""]
    (D / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:6]))
    print(f"wrote {D / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
