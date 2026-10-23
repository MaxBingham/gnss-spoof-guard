"""Shared harness code for the V4 detector evaluation (demos/15-18): ground truth from the dataset meta, box
matching, AP50, Wilson intervals, calibration (ECE, temperature scaling), breakdowns. Harness only (truth)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CLASSES = ("truck", "car", "pickup", "suv", "tent", "hut", "tree")
TRUCK = 0
IOU = 0.5


# ---------------------------------------------------------------------------------------- data

@dataclass
class ImageGT:
    image: str
    boxes: np.ndarray  # (N, 4) kept labels
    cls: np.ndarray  # (N,)
    roles: list[str]
    colors: list[str]
    visible: np.ndarray  # (N,)
    ign_boxes: np.ndarray  # (M, 4) visible but dropped (tiny / > 80 % occluded): neither TP nor FP
    ign_cls: np.ndarray
    meta: dict


@dataclass
class ImagePred:
    boxes: np.ndarray  # (K, 4) xyxy px
    conf: np.ndarray  # (K,)
    cls: np.ndarray  # (K,)


def load_gt(root: Path, split: str) -> list[ImageGT]:
    out = []
    for line in open(Path(root) / "meta" / f"{split}.jsonl", encoding="utf-8"):
        m = json.loads(line)
        kept = [o for o in m["objects"] if o["kept"]]
        ign = [o for o in m["objects"] if not o["kept"]]
        out.append(ImageGT(m["image"], np.array([o["box"] for o in kept], float).reshape(-1, 4),
                           np.array([o["cls"] for o in kept], int), [o["role"] for o in kept],
                           [o["color"] for o in kept], np.array([o["visible"] for o in kept], float),
                           np.array([o["box"] for o in ign], float).reshape(-1, 4),
                           np.array([o["cls"] for o in ign], int), m))
    return out


def save_preds(path: Path, names: list[str], preds: list[ImagePred]) -> None:
    idx = np.concatenate([np.full(len(p.conf), i) for i, p in enumerate(preds)]) if preds else np.zeros(0)
    np.savez_compressed(path, names=np.array(names), idx=idx.astype(np.int32),
                        boxes=np.concatenate([p.boxes for p in preds]).astype(np.float32),
                        conf=np.concatenate([p.conf for p in preds]).astype(np.float32),
                        cls=np.concatenate([p.cls for p in preds]).astype(np.int16))


def load_preds(path: Path) -> dict[str, ImagePred]:
    z = np.load(path)
    out = {}
    for i, n in enumerate(z["names"]):
        s = z["idx"] == i
        out[str(n)] = ImagePred(z["boxes"][s].astype(float), z["conf"][s].astype(float), z["cls"][s].astype(int))
    return out


def scale_conf(preds: dict[str, ImagePred], fn) -> dict[str, ImagePred]:
    return {k: ImagePred(p.boxes, np.asarray(fn(p.conf), float), p.cls) for k, p in preds.items()}


# ---------------------------------------------------------------------------------------- matching

def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match_class(gt: ImageGT, p: ImagePred, c: int, thr: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Class-aware greedy matching for class c. Returns (conf, is_tp, matched_gt_index or -1, n_gt) for the
    predictions of class c with conf >= thr that are not ignored (overlap a dropped label of class c)."""
    sel = (p.cls == c) & (p.conf >= thr)
    pb, pc = p.boxes[sel], p.conf[sel]
    order = np.argsort(-pc)
    pb, pc = pb[order], pc[order]
    gi = np.nonzero(gt.cls == c)[0]
    M = iou(pb, gt.boxes[gi])
    ign = iou(pb, gt.ign_boxes[gt.ign_cls == c])
    used = np.zeros(len(gi), bool)
    keep, tp, gidx = [], [], []
    for k in range(len(pc)):
        j = -1
        if len(gi):
            cand = np.where(~used, M[k], -1.0)
            jj = int(np.argmax(cand))
            if cand[jj] >= IOU:
                j = jj
        if j >= 0:
            used[j] = True
            keep.append(k); tp.append(True); gidx.append(int(gi[j]))
        elif ign.shape[1] and ign[k].max() >= IOU:
            continue
        else:
            keep.append(k); tp.append(False); gidx.append(-1)
    return pc[keep] if keep else np.zeros(0), np.array(tp, bool), np.array(gidx, int), len(gi)


def ap50(gts: list[ImageGT], preds: dict[str, ImagePred], c: int) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """All-point interpolated AP at IoU 0.5 for class c. Returns (AP, recall, precision, conf) curves."""
    confs, tps, n = [], [], 0
    for g in gts:
        cf, tp, _, ng = match_class(g, preds[g.image], c)
        confs.append(cf); tps.append(tp); n += ng
    cf = np.concatenate(confs); tp = np.concatenate(tps)
    o = np.argsort(-cf)
    cf, tp = cf[o], tp[o]
    ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
    rec = ctp / max(n, 1)
    prec = ctp / np.maximum(ctp + cfp, 1)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[1], prec, [0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    i = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])), rec, prec, cf


def pr_at(gts: list[ImageGT], preds: dict[str, ImagePred], c: int, thr: float) -> dict:
    tp = fp = n = 0
    for g in gts:
        _, t, _, ng = match_class(g, preds[g.image], c, thr)
        tp += int(t.sum()); fp += int((~t).sum()); n += ng
    return {"tp": tp, "fp": fp, "n_gt": n, "precision": tp / max(tp + fp, 1), "recall": tp / max(n, 1),
            "f1": 2 * tp / max(2 * tp + fp + (n - tp), 1)}


def best_f1_threshold(gts: list[ImageGT], preds: dict[str, ImagePred], c: int) -> tuple[float, float]:
    _, rec, prec, cf = ap50(gts, preds, c)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    k = int(np.argmax(f1))
    return float(cf[k]), float(f1[k])


# ---------------------------------------------------------------------------------------- statistics

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, center - half), min(1.0, center + half))


def fmt_rate(k: int, n: int) -> str:
    if n == 0:
        return "n/a (n = 0)"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.1f} % ({100 * lo:.1f}-{100 * hi:.1f} %)"


# ---------------------------------------------------------------------------------------- truck metrics

def truck_stats(gts: list[ImageGT], preds: dict[str, ImagePred], thr: float) -> dict:
    """Truck recall (per labeled truck), false-positive images, and which objects get called truck."""
    rec_k = rec_n = fp_img = 0
    called: dict[str, list[int]] = {}
    per_image = []
    for g in gts:
        p = preds[g.image]
        _, tp, gidx, ng = match_class(g, p, TRUCK, thr)
        found = int(tp.sum())
        rec_k += found; rec_n += ng
        nfp = int((~tp).sum())
        fp_img += nfp > 0
        # Every truck box >= thr that is not THE truck: which labeled object (if any) is under it?
        tb = p.boxes[(p.cls == TRUCK) & (p.conf >= thr)]
        M = iou(tb, g.boxes)
        for j, (c, role, col) in enumerate(zip(g.cls, g.roles, g.colors)):
            if c == TRUCK:
                continue
            key = role if role in ("brown_car", "white_car") else (f"{CLASSES[c]} ({col})" if col in ("brown", "white") else CLASSES[c])
            hit = int(M.shape[0] > 0 and M[:, j].max() >= IOU)
            called.setdefault(key, [0, 0])
            called[key][0] += hit; called[key][1] += 1
        per_image.append({"image": g.image, "trucks": ng, "found": found, "fp": nfp})
    return {"thr": thr, "recall_k": rec_k, "recall_n": rec_n, "fp_images": int(fp_img), "n_images": len(gts),
            "called_truck": called, "per_image": per_image}


def confusion(gts: list[ImageGT], preds: dict[str, ImagePred], thr: dict[int, float]) -> np.ndarray:
    """(C+1, C+1) rows = true class (+ background), cols = predicted class (+ missed). Class-agnostic greedy
    matching at IoU 0.5 of the predictions above their class threshold."""
    C = len(CLASSES)
    cm = np.zeros((C + 1, C + 1), int)
    for g in gts:
        p = preds[g.image]
        keep = np.array([p.conf[i] >= thr.get(int(p.cls[i]), 0.25) for i in range(len(p.conf))], bool)
        pb, pc, pk = p.boxes[keep], p.conf[keep], p.cls[keep]
        o = np.argsort(-pc)
        pb, pk = pb[o], pk[o]
        M = iou(pb, g.boxes)
        ign = iou(pb, g.ign_boxes)
        used = np.zeros(len(g.cls), bool)
        for k in range(len(pk)):
            cand = np.where(~used, M[k], -1.0) if len(g.cls) else np.zeros(0)
            if len(cand) and cand.max() >= IOU:
                j = int(np.argmax(cand))
                used[j] = True
                cm[g.cls[j], pk[k]] += 1
            elif ign.shape[1] and ign[k].max() >= IOU:
                continue
            else:
                cm[C, pk[k]] += 1
        for j in np.nonzero(~used)[0]:
            cm[g.cls[j], C] += 1
    return cm


# ---------------------------------------------------------------------------------------- calibration

def calib_samples(gts: list[ImageGT], preds: dict[str, ImagePred], c: int | None = TRUCK, min_conf: float = 0.1
                  ) -> tuple[np.ndarray, np.ndarray]:
    """(confidence, correct) of every prediction >= min_conf (class c, or all classes if None)."""
    cs, ys = [], []
    for g in gts:
        for k in ([c] if c is not None else range(len(CLASSES))):
            cf, tp, _, _ = match_class(g, preds[g.image], k, min_conf)
            cs.append(cf); ys.append(tp)
    return np.concatenate(cs), np.concatenate(ys).astype(float)


def ece(conf: np.ndarray, y: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    edges = np.linspace(0, 1, bins + 1)
    rows, e = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf >= lo) & ((conf < hi) | (hi == 1.0))
        if s.sum() == 0:
            rows.append({"lo": lo, "hi": hi, "n": 0, "conf": float("nan"), "acc": float("nan")})
            continue
        mc, acc = float(conf[s].mean()), float(y[s].mean())
        e += s.sum() / len(conf) * abs(mc - acc)
        rows.append({"lo": lo, "hi": hi, "n": int(s.sum()), "conf": mc, "acc": acc})
    return float(e), rows


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def apply_temperature(p: np.ndarray, T: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logit(np.asarray(p, float)) / T))


def fit_temperature(conf: np.ndarray, y: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar

    z = logit(conf)

    def nll(logT: float) -> float:
        q = np.clip(1.0 / (1.0 + np.exp(-z / math.exp(logT))), 1e-9, 1 - 1e-9)
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

    r = minimize_scalar(nll, bounds=(math.log(0.05), math.log(20.0)), method="bounded")
    return float(math.exp(r.x))


# ---------------------------------------------------------------------------------------- breakdowns

def truck_conditions(g: ImageGT) -> dict[str, str]:
    v = g.meta["view"]
    alt, tilt = v["pos"][2], v["tilt_deg"]
    w = v["weather"]
    wb = "clear" if w == "clear" else f"{w} {'light' if v['weather_i'] < 0.45 else 'heavy'}"
    return {"altitude": next(b for lo, hi, b in ((0, 20, "12-20 m"), (20, 28, "20-28 m"), (28, 36, "28-36 m"),
                                                    (36, 99, "36-45 m")) if lo <= alt < hi),
            "tilt": next(b for lo, hi, b in ((0, 8, "0-8 deg"), (8, 16, "8-16 deg"), (16, 99, "16-25 deg")) if lo <= tilt < hi),
            "weather": wb, "ground": g.meta["ground_kind"]}


def occlusion_bin(vis: float) -> str:
    return next(b for lo, hi, b in ((0, 0.5, "50-80 % hidden"), (0.5, 0.8, "20-50 % hidden"),
                                    (0.8, 0.97, "3-20 % hidden"), (0.97, 2, "unoccluded")) if lo <= vis < hi)


def breakdown(gts: list[ImageGT], preds: dict[str, ImagePred], thr: float) -> dict[str, dict[str, dict]]:
    """Truck recall per labeled truck and FP-image rate, by altitude / tilt / weather / ground / occlusion."""
    out: dict[str, dict[str, dict]] = {}
    for g in gts:
        p = preds[g.image]
        _, tp, gidx, _ = match_class(g, p, TRUCK, thr)
        found = set(int(j) for j in gidx[tp])
        conds = truck_conditions(g)
        fp = int((~tp).sum()) > 0
        for dim, b in conds.items():
            r = out.setdefault(dim, {}).setdefault(b, {"k": 0, "n": 0, "fp": 0, "imgs": 0})
            r["imgs"] += 1; r["fp"] += fp
        for j in np.nonzero(g.cls == TRUCK)[0]:
            ob = occlusion_bin(float(g.visible[j]))
            for dim, b in list(conds.items()) + [("occlusion", ob)]:
                r = out.setdefault(dim, {}).setdefault(b, {"k": 0, "n": 0, "fp": 0, "imgs": 0})
                r["n"] += 1; r["k"] += int(j in found)
    return out


# ---------------------------------------------------------------------------------------- OWLv2 baseline

OWL_MODEL = "google/owlv2-base-patch16-ensemble"
OWL_TRUCK_PROMPT = "a brown box truck"
OWL_NEGATIVES = ("a car", "a pickup truck", "an SUV", "a tent", "a small hut", "a tree")


class OwlDetector:
    """Zero-shot open-vocabulary baseline (OWLv2, Hugging Face transformers). Same `detect` interface as the
    onboard TruckDetector. negatives=False: only the truck prompt (every box is a 'truck' with its score);
    negatives=True: the other class prompts compete and a box is a truck only where the truck prompt wins."""

    def __init__(self, negatives: bool = False, device: str | None = None, conf_min: float = 0.01,
                 temperature: float = 1.0) -> None:
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.proc = Owlv2Processor.from_pretrained(OWL_MODEL)
        self.model = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL, torch_dtype=self.dtype).to(self.device).eval()
        self.prompts = [OWL_TRUCK_PROMPT] + (list(OWL_NEGATIVES) if negatives else [])
        self.negatives = negatives
        self.conf_min = conf_min
        self.temperature = temperature
        self.last_ms = 0.0

    def predict_raw(self, rgb: np.ndarray) -> ImagePred:
        import time
        torch = self.torch
        t0 = time.perf_counter()
        h, w = rgb.shape[:2]
        inp = self.proc(text=[self.prompts], images=rgb, return_tensors="pt").to(self.device)
        inp["pixel_values"] = inp["pixel_values"].to(self.dtype)
        with torch.no_grad():
            out = self.model(**inp)
        logits = out.logits[0].float()  # (boxes, prompts)
        probs = torch.sigmoid(logits)
        score, which = probs.max(-1)
        boxes = out.pred_boxes[0].float()  # cx, cy, w, h normalized to the padded square image
        s = max(h, w)
        xyxy = torch.stack([(boxes[:, 0] - boxes[:, 2] / 2) * s, (boxes[:, 1] - boxes[:, 3] / 2) * s,
                            (boxes[:, 0] + boxes[:, 2] / 2) * s, (boxes[:, 1] + boxes[:, 3] / 2) * s], -1)
        truck = probs[:, 0]
        keep = truck >= self.conf_min
        if self.negatives:  # truck only where the truck prompt beats every other prompt
            keep &= which == 0
        keep_idx = torch.nonzero(keep).flatten()
        b, c = xyxy[keep_idx].cpu().numpy(), truck[keep_idx].cpu().numpy()
        # NMS (OWL emits one box per patch)
        import torchvision
        k = torchvision.ops.nms(torch.as_tensor(b), torch.as_tensor(c), 0.3).numpy()[:300] if len(c) else np.zeros(0, int)
        self.last_ms = 1000 * (time.perf_counter() - t0)
        b = np.clip(b[k], 0, [w, h, w, h]) if len(k) else np.zeros((0, 4))
        return ImagePred(b, c[k] if len(k) else np.zeros(0), np.zeros(len(k), int))

    def detect(self, rgb: np.ndarray, altitude: float, attitude: np.ndarray) -> list:
        from sfhack.config import DetectorConfig
        from sfhack.perception.truck_detector import Detection, calibrate, ground_offsets
        p = self.predict_raw(rgb)
        if len(p.conf) == 0:
            return []
        cfg = DetectorConfig()
        h, w = rgb.shape[:2]
        centers = np.column_stack([(p.boxes[:, 0] + p.boxes[:, 2]) / 2, (p.boxes[:, 1] + p.boxes[:, 3]) / 2])
        g = ground_offsets(centers, np.full(len(centers), cfg.class_heights[0] / 2), w, h, cfg.fovy_deg, altitude, attitude)
        cal = calibrate(p.conf, self.temperature)
        return [Detection("truck", float(q), tuple(float(v) for v in bb), gg, float(r))
                for bb, q, r, gg in zip(p.boxes, cal, p.conf, g)]
