"""V4 detector, step 2: auto-labeled synthetic dataset (YOLO format) from the randomized MuJoCo scenes.

HARNESS (offline data generation): uses the simulator's truth (segmentation -> boxes). Nothing here runs onboard.

    uv run python demos/13_det_dataset.py                      # train 6000, val 1000, test 1000, test_nopad 300
    uv run python demos/13_det_dataset.py --train 200 --val 50 --test 50 --nopad 20 --out <dir>   # quick check

Output ($SFHACK_DATA/v4/detector/dataset): images/<split>/*.jpg, labels/<split>/*.txt (YOLO: cls cx cy w h,
normalized), meta/<split>.jsonl (view, layout, every visible object incl. dropped ones), data.yaml,
DATASET_CARD.md (+ PNG histograms), examples/ (20 images with boxes drawn).
Splits: train / val = seen variants + seen grounds; test = held-out truck + car variants and held-out grounds;
test_nopad = the first N test scenes again with the landing pad hidden (how much does the model lean on it?).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sfhack.config import data_dir  # noqa: E402
from sfhack.sim import detector_scene as ds  # noqa: E402

SPLIT_SEED0 = {"train": 0, "val": 100_000, "test": 200_000, "test_nopad": 200_000}

_scene: ds.DetectorScene | None = None
_lib: dict | None = None
_grounds: dict | None = None


def _init(det_root: str) -> None:
    global _scene, _lib, _grounds
    _lib = ds.load_library()
    _grounds = ds.ensure_ground_textures(Path(det_root))
    _scene = ds.DetectorScene(_lib, _grounds)


def render_one(split: str, seed: int) -> tuple[ds.RenderOut, ds.Layout, ds.View, str]:
    """Deterministic sample (split, seed). test_nopad = the test scene of the same seed, pad hidden."""
    assert _scene is not None and _lib is not None and _grounds is not None
    base = "test" if split == "test_nopad" else split
    lay = ds.sample_layout(_lib, seed, base)
    if split == "test_nopad":
        lay.pad = False
    rng = np.random.default_rng([seed, ds.SPLITS.index(base), 23])
    view = ds.sample_view(rng, len(_grounds[ds.ground_group(base)]), layout=lay, lib=_lib)
    out = _scene.render(lay, view, rng)
    kind = _grounds[ds.ground_group(base)][view.ground]["kind"]
    return out, lay, view, kind


def _work(args: tuple[str, int, str]) -> dict:
    split, seed, root = args
    out, lay, view, kind = render_one(split, seed)
    name = f"{split}_{seed:06d}"
    r = Path(root)
    cv2.imwrite(str(r / "images" / split / f"{name}.jpg"), cv2.cvtColor(out.rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 93])
    rows = []
    for lab in out.labels:
        x0, y0, x1, y1 = lab.box
        rows.append(f"{lab.cls} {(x0 + x1) / 2 / ds.W:.6f} {(y0 + y1) / 2 / ds.H:.6f} {(x1 - x0) / ds.W:.6f} {(y1 - y0) / ds.H:.6f}")
    (r / "labels" / split / f"{name}.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    def lab_json(lab: ds.Label, kept: bool) -> dict:
        return {"cls": lab.cls, "key": lab.key, "role": lab.role, "color": lab.color, "variant": lab.variant,
                "box": [round(v, 1) for v in lab.box], "visible": round(lab.visible_frac, 3), "pixels": lab.pixels,
                "xy": [round(float(lab.xy[0]), 3), round(float(lab.xy[1]), 3)], "kept": kept}

    return {"image": name, "split": split, "seed": seed, "ground_kind": kind, "view": view.to_json(),
            "layout": lay.to_json(),
            "objects": [lab_json(lab, True) for lab in out.labels] + [lab_json(lab, False) for lab in out.dropped]}


def generate(root: Path, counts: dict[str, int], workers: int) -> dict[str, list[dict]]:
    for split in counts:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    ds.ensure_ground_textures(root.parent)  # once, before the workers start
    jobs = [(split, SPLIT_SEED0[split] + i, str(root)) for split, n in counts.items() for i in range(n)]
    metas: dict[str, list[dict]] = {s: [] for s in counts}
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(str(root.parent),)) as pool:
        for k, m in enumerate(pool.imap_unordered(_work, jobs, chunksize=8)):
            metas[m["split"]].append(m)
            if (k + 1) % 500 == 0:
                el = time.time() - t0
                print(f"  {k + 1}/{len(jobs)} images, {el:.0f} s, eta {el / (k + 1) * (len(jobs) - k - 1):.0f} s", flush=True)
    for split, ms in metas.items():
        ms.sort(key=lambda m: m["seed"])
        with open(root / "meta" / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for m in ms:
                f.write(json.dumps(m) + "\n")
    print(f"rendered {len(jobs)} images in {time.time() - t0:.0f} s ({workers} workers)")
    return metas


def load_meta(root: Path, split: str) -> list[dict]:
    return [json.loads(line) for line in open(root / "meta" / f"{split}.jsonl", encoding="utf-8")]


def write_yaml(root: Path, splits: list[str]) -> None:
    lines = [f"path: {root.as_posix()}", "train: images/train", "val: images/val"]
    lines += [f"{s}: images/{s}" for s in splits if s not in ("train", "val")]
    lines += ["names:"] + [f"  {i}: {n}" for i, n in enumerate(ds.CLASSES)]
    (root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataset_card(root: Path, metas: dict[str, list[dict]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = list(metas)
    md = ["# V4 truck detector - synthetic dataset card", "",
          "Auto-labeled MuJoCo renders (640 x 480, fovy 70 deg, down camera) of randomized scenes around B built from",
          "the object library (real-scale STL). Labels = boxes of the VISIBLE pixels from the segmentation buffer;",
          "boxes < 8 px or > 80 % hidden by trees are dropped. Classes: " + ", ".join(ds.CLASSES) + " (car color is",
          "not a class). Generator: `demos/13_det_dataset.py`, scene: `sfhack/sim/detector_scene.py`.", "",
          "Held-out (never in train / val): " + ", ".join(f"{k}: {', '.join(v)}" for k, v in ds.HELDOUT.items()) + ".",
          "Test uses only the held-out truck and car variants and only held-out grounds (north part of the",
          "Berkeley ortho, dry grass, concrete, new grass / asphalt textures). `test_nopad` = the first test scenes",
          "again with the landing pad hidden. Train hides the pad in ~20 % of the images.", "",
          "## Images and boxes per class", "",
          "| split | images | " + " | ".join(ds.CLASSES) + " | dropped (small / occluded) |",
          "|---|---|" + "---|" * len(ds.CLASSES) + "---|"]
    for s in splits:
        c = Counter(o["key"] for m in metas[s] for o in m["objects"] if o["kept"])
        dropped = sum(1 for m in metas[s] for o in m["objects"] if not o["kept"])
        md.append(f"| {s} | {len(metas[s])} | " + " | ".join(str(c[k]) for k in ds.CLASSES) + f" | {dropped} |")
    md += ["", "## Truck appearance and conditions", "",
           "| split | truck labeled | truck variants | brown cars | pad visible | fog | rain | ground kinds |",
           "|---|---|---|---|---|---|---|---|"]
    for s in splits:
        ms = metas[s]
        tv = Counter(o["variant"] for m in ms for o in m["objects"] if o["kept"] and o["key"] == "truck")
        bc = sum(1 for m in ms for o in m["objects"] if o["kept"] and o["role"] == "brown_car")
        w = Counter(m["view"]["weather"] for m in ms)
        g = Counter(m["ground_kind"] for m in ms)
        pad = sum(m["layout"]["pad"] for m in ms)
        md.append(f"| {s} | {sum(tv.values())} | " + ", ".join(f"{k} {v}" for k, v in sorted(tv.items())) +
                  f" | {bc} | {pad} | {w['fog']} | {w['rain']} | " + ", ".join(f"{k} {v}" for k, v in sorted(g.items())) + " |")
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    for s in splits:
        ms = metas[s]
        kw = dict(histtype="step", density=True, label=s, linewidth=1.5)
        ax[0, 0].hist([m["view"]["pos"][2] for m in ms], bins=20, **kw)
        ax[0, 1].hist([m["view"]["tilt_deg"] for m in ms], bins=20, **kw)
        ax[0, 2].hist([m["view"]["sun_el_deg"] for m in ms], bins=20, **kw)
        ax[1, 0].hist([o["visible"] for m in ms for o in m["objects"] if o["key"] != "tree"], bins=20, range=(0, 1), **kw)
        ax[1, 1].hist([o["box"][2] - o["box"][0] for m in ms for o in m["objects"]
                       if o["kept"] and o["key"] == "truck"], bins=20, **kw)
        ax[1, 2].hist([m["view"]["weather_i"] for m in ms if m["view"]["weather"] != "clear"], bins=10, **kw)
    for a, t in zip(ax.flat, ["camera altitude [m]", "camera tilt [deg]", "sun elevation [deg]",
                              "visible fraction (non-tree objects)", "truck box width [px]", "fog / rain intensity"]):
        a.set_title(t, fontsize=10)
        a.grid(alpha=0.3)
    ax[0, 0].legend()
    fig.tight_layout()
    fig.savefig(root / "dataset_conditions.png", dpi=110)
    plt.close(fig)
    md += ["", "![conditions](dataset_conditions.png)", "", "Examples with boxes: `examples/`.", ""]
    (root / "DATASET_CARD.md").write_text("\n".join(md), encoding="utf-8")


def examples(root: Path, metas: dict[str, list[dict]], n: int = 20) -> None:
    ex = root / "examples"
    ex.mkdir(exist_ok=True)
    picks = [m for m in metas["train"][:8]] + [m for m in metas.get("val", [])[:4]] + [m for m in metas.get("test", [])[:8]]
    for m in picks[:n]:
        img = cv2.cvtColor(cv2.imread(str(root / "images" / m["split"] / f"{m['image']}.jpg")), cv2.COLOR_BGR2RGB)
        labs = [ds.Label(o["cls"], o["key"], o["role"], o["color"], o["variant"], tuple(o["box"]), o["visible"],
                         o["pixels"], tuple(o["xy"])) for o in m["objects"] if o["kept"]]
        out = ds.draw_labels(img, labs)
        v = m["view"]
        txt = f"{m['split']} alt {v['pos'][2]:.0f} m tilt {v['tilt_deg']:.0f} {v['weather']} {m['ground_kind']}"
        cv2.putText(out, txt, (6, ds.H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(ex / f"{m['image']}.jpg"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, default=6000)
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--nopad", type=int, default=300, help="test scenes re-rendered with the pad hidden")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (mp.cpu_count() or 2) // 2)))
    ap.add_argument("--out", type=Path, default=data_dir() / "v4" / "detector" / "dataset")
    a = ap.parse_args()
    counts = {"train": a.train, "val": a.val, "test": a.test, "test_nopad": a.nopad}
    counts = {k: v for k, v in counts.items() if v > 0}
    metas = generate(a.out, counts, a.workers)
    write_yaml(a.out, list(counts))
    dataset_card(a.out, metas)
    examples(a.out, metas)
    print((a.out / "DATASET_CARD.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
