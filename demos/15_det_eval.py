"""V4 detector, step 4: honest evaluation on the held-out test set (unseen truck / car variants, unseen grounds).

    uv run python demos/15_det_eval.py                 # YOLO weights from DetectorConfig
    uv run python demos/15_det_eval.py --preds-only    # just cache predictions

Per class precision / recall / AP50, confusion matrix, truck recall and false-positive rates with 95 % Wilson
intervals, calibration (reliability diagram, ECE, temperature scaling fitted on val), breakdowns by altitude,
tilt, weather, ground, occlusion, and the pad ablation (same test scenes with the landing pad hidden).
The operating point (truck threshold) and the temperature are chosen on VAL only and written to
<train dir>/calibration.json, which the onboard TruckDetector loads next to its weights.
Outputs: $SFHACK_DATA/v4/detector/eval/<name>/ (metrics.json, PNG charts).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import detlib as dl  # noqa: E402

from sfhack.config import DetectorConfig, data_dir  # noqa: E402

SPLITS = ("val", "test", "test_nopad")


def predict_yolo(weights: Path, root: Path, split: str, out: Path, device: str) -> Path:
    path = out / f"preds_{split}.npz"
    if path.exists():
        return path
    from ultralytics import YOLO

    model = YOLO(str(weights))
    gts = dl.load_gt(root, split)
    files = [str(root / "images" / split / f"{g.image}.jpg") for g in gts]
    preds = []
    for k in range(0, len(files), 32):
        for r in model.predict(files[k:k + 32], imgsz=640, conf=0.001, max_det=300, device=device, verbose=False):
            b = r.boxes
            preds.append(dl.ImagePred(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)))
    dl.save_preds(path, [g.image for g in gts], preds)
    return path


def evaluate(name: str, pred_dir: Path, root: Path, out: Path, classes: tuple[int, ...] = tuple(range(7)),
             fixed: dict | None = None) -> dict:
    """All metrics from cached predictions. `fixed` = {"T":, "thr":} to reuse an operating point."""
    gts = {s: dl.load_gt(root, s) for s in SPLITS if (pred_dir / f"preds_{s}.npz").exists()}
    raw = {s: dl.load_preds(pred_dir / f"preds_{s}.npz") for s in gts}
    gts = {s: [g for g in gts[s] if g.image in raw[s]] for s in gts}  # a baseline may cover a subset
    res: dict = {"name": name, "classes": [dl.CLASSES[c] for c in classes]}

    # ---- calibration: temperature on val truck predictions
    cv, yv = dl.calib_samples(gts["val"], raw["val"], dl.TRUCK)
    T = fixed["T"] if fixed else dl.fit_temperature(cv, yv)
    cal = {s: dl.scale_conf(raw[s], lambda c: dl.apply_temperature(c, T)) for s in raw}
    calib = {"T": T}
    for s in ("val", "test"):
        # the same predictions (raw conf >= 0.1) before and after scaling
        c0, y0 = dl.calib_samples(gts[s], raw[s], dl.TRUCK)
        c1 = dl.apply_temperature(c0, T)
        e0, rows0 = dl.ece(c0, y0)
        e1, rows1 = dl.ece(c1, y0)
        a0, ay = dl.calib_samples(gts[s], raw[s], None)
        a1 = dl.apply_temperature(a0, T)
        calib[s] = {"truck_ece_before": e0, "truck_ece_after": e1, "n_truck_preds": int(len(c0)),
                    "all_ece_before": dl.ece(a0, ay)[0], "all_ece_after": dl.ece(a1, ay)[0],
                    "bins_before": rows0, "bins_after": rows1}
    res["calibration"] = calib

    # ---- operating point on VAL (calibrated): best truck F1
    thr = fixed["thr"] if fixed else dl.best_f1_threshold(gts["val"], cal["val"], dl.TRUCK)[0]
    res["truck_threshold"] = thr
    thr_map = {c: (thr if c == dl.TRUCK else 0.25) for c in range(len(dl.CLASSES))}

    for s in gts:
        g, p = gts[s], cal[s]
        per_class = {}
        for c in classes:
            ap, rec, prec, cf = dl.ap50(g, p, c)
            per_class[dl.CLASSES[c]] = {"ap50": ap, **dl.pr_at(g, p, c, thr_map[c])}
            if c == dl.TRUCK:
                per_class[dl.CLASSES[c]]["curve"] = {"recall": rec[::max(1, len(rec) // 300)].tolist(),
                                                    "precision": prec[::max(1, len(rec) // 300)].tolist()}
        ts = dl.truck_stats(g, p, thr)
        res[s] = {"n_images": len(g), "per_class": per_class,
                  "map50": float(np.mean([per_class[dl.CLASSES[c]]["ap50"] for c in classes])),
                  "truck": {k: v for k, v in ts.items() if k != "per_image"},
                  "confusion": dl.confusion(g, p, thr_map).tolist() if len(classes) > 1 else None,
                  "breakdown": dl.breakdown(g, p, thr)}
        res[s]["truck"]["recall_ci"] = dl.wilson(ts["recall_k"], ts["recall_n"])
        res[s]["truck"]["fp_image_ci"] = dl.wilson(ts["fp_images"], ts["n_images"])
        res[s]["per_image"] = ts["per_image"]
    # Pad ablation: the same scenes with / without the pad (first N test seeds)
    if "test_nopad" in gts:
        names = {g.meta["seed"] for g in gts["test_nopad"]}
        sub = [g for g in gts["test"] if g.meta["seed"] in names]
        ts = dl.truck_stats(sub, cal["test"], thr)
        res["pad_ablation"] = {"with_pad": [ts["recall_k"], ts["recall_n"], ts["fp_images"], ts["n_images"]],
                               "without_pad": [res["test_nopad"]["truck"][k] for k in ("recall_k", "recall_n", "fp_images", "n_images")],
                               "ap50_with": dl.ap50(sub, cal["test"], dl.TRUCK)[0],
                               "ap50_without": res["test_nopad"]["per_class"]["truck"]["ap50"]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    plots(res, out)
    return res


def plots(res: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    name = res["name"]
    # Reliability (test, truck)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
    for a, key, title in ((ax[0], "bins_before", "before"), (ax[1], "bins_after", f"after (T = {res['calibration']['T']:.2f})")):
        rows = [r for r in res["calibration"]["test"][key] if r["n"] > 0]
        mids = [(r["lo"] + r["hi"]) / 2 for r in rows]
        a.bar(mids, [r["acc"] for r in rows], width=0.09, color="#4C78A8", label="fraction correct")
        a.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        a.plot([r["conf"] for r in rows], [r["acc"] for r in rows], "o", color="#E45756", ms=4, label="mean confidence")
        for m, r in zip(mids, rows):
            a.text(m, 0.02, str(r["n"]), ha="center", fontsize=7, color="white")
        e = res["calibration"]["test"]["truck_ece_" + ("before" if key == "bins_before" else "after")]
        a.set_title(f"{name} truck, test: {title}, ECE {e:.3f}", fontsize=10)
        a.set_xlabel("confidence"); a.set_ylabel("fraction correct (IoU >= 0.5)")
        a.set_xlim(0, 1); a.set_ylim(0, 1); a.grid(alpha=0.3)
    ax[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(out / "reliability.png", dpi=110); plt.close(fig)

    # PR curves (truck)
    fig, a = plt.subplots(figsize=(5.2, 4.4))
    for s, col in (("val", "#4C78A8"), ("test", "#E45756"), ("test_nopad", "#999999")):
        if s in res:
            cu = res[s]["per_class"]["truck"]["curve"]
            a.plot(cu["recall"], cu["precision"], color=col, label=f"{s}: AP50 {res[s]['per_class']['truck']['ap50']:.3f}")
    a.set_xlabel("recall"); a.set_ylabel("precision"); a.set_title(f"{name}: truck precision-recall", fontsize=10)
    a.set_xlim(0, 1); a.set_ylim(0, 1.02); a.grid(alpha=0.3); a.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(out / "pr_truck.png", dpi=110); plt.close(fig)

    # Confusion matrix (test)
    if res["test"]["confusion"] is not None:
        cm = np.array(res["test"]["confusion"])
        labels = list(dl.CLASSES) + ["bg / missed"]
        fig, a = plt.subplots(figsize=(7, 6))
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        a.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j]:
                    a.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8, color="white" if norm[i, j] > 0.5 else "black")
        a.set_xticks(range(len(labels)), labels, rotation=45, ha="right"); a.set_yticks(range(len(labels)), labels)
        a.set_xlabel("predicted"); a.set_ylabel("true")
        a.set_title(f"{name}, test: truck thr {res['truck_threshold']:.2f}, others 0.25", fontsize=10)
        fig.tight_layout(); fig.savefig(out / "confusion_test.png", dpi=110); plt.close(fig)

    # Breakdown (test, truck recall with CI)
    bd = res["test"]["breakdown"]
    dims = ["altitude", "tilt", "weather", "occlusion", "ground"]
    fig, axs = plt.subplots(1, len(dims), figsize=(17, 3.8), sharey=True)
    for a, d in zip(axs, dims):
        items = sorted(bd[d].items())
        ks = [v["k"] for _, v in items]; ns = [v["n"] for _, v in items]
        r = [k / max(n, 1) for k, n in zip(ks, ns)]
        ci = [dl.wilson(k, n) for k, n in zip(ks, ns)]
        err = np.clip(np.nan_to_num(np.array([[ri - lo for ri, (lo, hi) in zip(r, ci)], [hi - ri for ri, (lo, hi) in zip(r, ci)]])), 0, None)
        a.bar(range(len(items)), r, yerr=err, capsize=3, color="#4C78A8")
        a.set_xticks(range(len(items)), [f"{b}\nn={n}" for (b, _), n in zip(items, ns)], fontsize=7, rotation=30, ha="right")
        a.set_title(d, fontsize=10); a.set_ylim(0, 1.05); a.grid(alpha=0.3, axis="y")
    axs[0].set_ylabel("truck recall (95 % CI)")
    fig.suptitle(f"{name}, test, truck threshold {res['truck_threshold']:.2f}", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "breakdown_test.png", dpi=110); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, default=data_dir() / DetectorConfig().weights)
    ap.add_argument("--root", type=Path, default=data_dir() / "v4" / "detector" / "dataset")
    ap.add_argument("--name", default="yolo26n")
    ap.add_argument("--device", default="0")
    ap.add_argument("--preds-only", action="store_true")
    a = ap.parse_args()
    out = data_dir() / "v4" / "detector" / "eval" / a.name
    out.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        if (a.root / "meta" / f"{s}.jsonl").exists():
            predict_yolo(a.weights, a.root, s, out, a.device)
    if a.preds_only:
        return
    res = evaluate(a.name, out, a.root, out)
    cal = {"temperature": res["calibration"]["T"], "truck_threshold": res["truck_threshold"],
           "chosen_on": "val (temperature: truck NLL; threshold: best truck F1 after calibration)", "weights": str(a.weights)}
    (a.weights.parent.parent / "calibration.json").write_text(json.dumps(cal, indent=2), encoding="utf-8")
    t = res["test"]["truck"]
    print(f"[{a.name}] T = {cal['temperature']:.3f}, truck threshold {cal['truck_threshold']:.3f} (val)")
    for s in ("val", "test", "test_nopad"):
        if s in res:
            pc = res[s]["per_class"]
            print(f"  {s:10s} mAP50 {res[s]['map50']:.3f} | " + " ".join(f"{k} {v['ap50']:.3f}" for k, v in pc.items()))
    print(f"  test truck recall {dl.fmt_rate(t['recall_k'], t['recall_n'])}; FP images {dl.fmt_rate(t['fp_images'], t['n_images'])}")
    print(f"  truck ECE test: {res['calibration']['test']['truck_ece_before']:.3f} -> {res['calibration']['test']['truck_ece_after']:.3f}")
    print("  called truck:", {k: f"{v[0]}/{v[1]}" for k, v in t["called_truck"].items()})


if __name__ == "__main__":
    main()
