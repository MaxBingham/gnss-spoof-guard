"""Enemy-drone detector, step 1: auto-labeled synthetic dataset (YOLO format) rendered in the farm world.

HARNESS (offline data generation): uses the simulator's truth (MuJoCo segmentation -> boxes). Nothing here runs
onboard. Adapted from the V4 truck-detector pipeline (tag archive/v4-detector, demos/13_det_dataset.py).

    uv run python demos/53_drone_dataset.py                                  # train 7000, val 1000, test 1000
    uv run python demos/53_drone_dataset.py --train 600 --val 100 --test 100 --out <dir>   # quick first dataset

Classes: 0 drone, 1 bird; everything else is background (trees, buildings, our base, vehicles, fields, sky).
Each image: the gimbal camera of our drone (640 x 480, 60 deg horizontal FOV, level horizon +- a little roll)
somewhere over the 4 x 4 km farm map, 10-80 m above the terrain, looking at a drone 10-300 m away against the sky or
the ground (88 % of the images; the rest have no drone), 0-3 birds (crow, pigeon, buzzard, stork) 10-200 m away,
~12 % of the drones partly behind a tree. Six drone looks (quad 0.6-2.1 m, hexa, octo, with / without payload, the
demo enemy among them) in random colours; random sun direction and strength, ambient, distance fog on / off; motion
blur, sensor noise, JPEG. Boxes = the visible pixels in the segmentation buffer; boxes smaller than 4 px are dropped
(the object stays in the image, unlabeled: the detector cannot be asked to find 2 px dots).
Splits: train / val / test from disjoint seeds; test also uses held-out drone colours (sand / white) and the
north-east quarter of the map only for 30 % of its images (`meta` records everything).

Output ($SFHACK_DATA/tracker/detector/dataset): images/<split>/*.jpg, labels/<split>/*.txt, meta/<split>.jsonl,
data.yaml, DATASET_CARD.md (+ conditions PNG), examples/ (images with boxes drawn).
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import enemy_world as ew  # noqa: E402
from sfhack.config import data_dir  # noqa: E402
from sfhack.sim import enemy as en  # noqa: E402

SPLIT_SEED0 = {"train": 0, "val": 100_000, "test": 200_000}
DRONE_COLORS = {"charcoal": (0.10, 0.10, 0.11), "black": (0.04, 0.04, 0.05), "gray": (0.42, 0.43, 0.45),
                "olive": (0.27, 0.30, 0.17), "darkgreen": (0.13, 0.20, 0.12), "lightgray": (0.70, 0.70, 0.72),
                "sand": (0.70, 0.62, 0.45), "white": (0.92, 0.92, 0.90)}
HELDOUT_COLORS = ("sand", "white")  # test only
TREES_R = 600.0  # trees closer than this to the base are 3D for sure (base3d_farm keeps the nearest ~2400)

_scene: "DatasetScene | None" = None


class DatasetScene:
    def __init__(self) -> None:
        self.w = ew.build("farm", dataset=True)
        m = self.w.model
        self.model, self.data = m, mujoco.MjData(m)
        mujoco.mj_forward(m, self.data)
        self.sr = ew.SegRenderer(m)
        self.ground = self.w.ground
        self.sun = m.light("sun").id
        self.drone_mocap = [int(m.body_mocapid[m.body(n).id]) for n in self.w.drones]
        self.drone_geoms = [[g for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name == n] for n in self.w.drones]
        self.bird_mocap = [tuple(int(m.body_mocapid[m.body(x).id]) for x in (n, f"{n}_wingl", f"{n}_wingr"))
                           for n, _k in self.w.birds]
        self.bird_geoms = [[g for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name.startswith(n + "_")
                            or m.body(int(m.geom_bodyid[g])).name == n] for n, _k in self.w.birds]
        self.bird_rgba = {g: m.geom_rgba[g].copy() for gs in self.bird_geoms for g in gs}
        fw = self.w.info.world
        trees = np.array([(x, y, r, rz, tr) for x, y, r, rz, tr, _c, _wb in fw.trees])
        self.trees = trees[np.hypot(trees[:, 0], trees[:, 1]) < TREES_R]
        self.hfov, self.vfov = ew.TRK.hfov_deg, ew.VFOV

    # ------------------------------------------------------------------ sampling

    def _hide_all(self) -> None:
        d = self.data
        for mid in self.drone_mocap:
            d.mocap_pos[mid] = en.HIDDEN
        for ids in self.bird_mocap:
            for mid in ids:
                d.mocap_pos[mid] = en.HIDDEN

    def _camera(self, rng, split: str) -> np.ndarray:
        u = rng.random()
        if u < 0.22:  # around the base (hall, wall, vehicles, mast)
            xy = rng.normal(0, 80, 2)
        elif split == "test" and u < 0.5:  # north-east quarter (also seen in train, just weighted)
            xy = rng.uniform(300, 1500, 2)
        else:
            a, r = rng.uniform(0, 2 * math.pi), 1500 * math.sqrt(rng.random())
            xy = r * np.array([math.cos(a), math.sin(a)])
        agl = rng.uniform(10, 80)
        return np.array([xy[0], xy[1], float(self.ground(*xy)) + agl])

    def _drone_color(self, rng, split: str, gs: list[int]) -> str:
        names = [k for k in DRONE_COLORS if (split == "test") or k not in HELDOUT_COLORS]
        p = np.array([3.0 if k in HELDOUT_COLORS else 1.0 for k in names]) if split == "test" else None
        name = str(rng.choice(names, p=None if p is None else p / p.sum()))
        base = np.array(DRONE_COLORS[name])
        arm = np.clip(base * rng.uniform(0.7, 1.3) + rng.normal(0, 0.04, 3), 0, 1)
        prop = np.clip(rng.uniform(0.15, 0.45) * np.ones(3), 0, 1)
        m = self.model
        for g in gs:
            gname = m.geom(g).name
            c = prop if "_prop" in gname else arm if ("_arm" in gname or "_payload" in gname) else base
            c = np.clip(np.asarray(c) * rng.uniform(0.92, 1.08), 0, 1)
            m.geom_rgba[g] = [*c, 1.0]
        return name

    def _place_drone(self, k: int, pos, yaw: float, rng) -> None:
        d = self.data
        sp = rng.uniform(0, 12)
        pose = en.EnemyPose(np.asarray(pos, float), np.array([sp * math.cos(yaw), sp * math.sin(yaw), 0.0]), yaw, True)
        d.mocap_pos[self.drone_mocap[k]] = pose.pos
        q = en.enemy_attitude(pose)
        roll = math.radians(rng.normal(0, 6))
        qr = np.array([math.cos(roll / 2), math.sin(roll / 2), 0, 0])
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, q, qr)
        d.mocap_quat[self.drone_mocap[k]] = out

    def _in_view(self, rng, cam, fwd, roll, d_rng: tuple[float, float], log_d: bool = True) -> np.ndarray:
        """A point at a random pixel of the view, at a random distance."""
        u, v = rng.uniform(0.05, 0.95) * ew.W, rng.uniform(0.05, 0.95) * ew.H
        fy = (ew.H / 2) / math.tan(math.radians(self.vfov) / 2)
        f = fwd / np.linalg.norm(fwd)
        right = np.cross(f, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, f)
        c, s = math.cos(roll), math.sin(roll)
        right, up = c * right + s * up, -s * right + c * up
        ray = f + (u - ew.W / 2) / fy * right - (v - ew.H / 2) / fy * up
        ray /= np.linalg.norm(ray)
        lo, hi = d_rng
        dist = math.exp(rng.uniform(math.log(lo), math.log(hi))) if log_d else rng.uniform(lo, hi)
        return cam + ray * dist

    def sample(self, split: str, seed: int) -> dict:
        rng = np.random.default_rng([seed, list(SPLIT_SEED0).index(split), 52])
        m, d = self.model, self.data
        self._hide_all()
        # light: sun direction / strength, ambient, fog
        az, el = rng.uniform(0, 2 * math.pi), math.radians(rng.uniform(12, 75))
        m.light_dir[self.sun] = [-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), -math.sin(el)]
        s = rng.uniform(0.35, 0.8)
        m.light_diffuse[self.sun] = [s, s * 0.97, s * rng.uniform(0.85, 0.95)]
        amb = rng.uniform(0.18, 0.42)
        m.vis.headlight.ambient[:] = [amb, amb, amb * 1.03]
        self.sr.fog = bool(rng.random() < 0.6)
        meta: dict = {"seed": seed, "split": split, "sun_el_deg": round(math.degrees(el), 1), "sun": round(s, 2),
                      "ambient": round(amb, 2), "fog": self.sr.fog, "objects": []}
        kind = rng.random()
        occlude = False
        # camera + the main drone
        for _attempt in range(30):
            cam = self._camera(rng, split)
            yaw = rng.uniform(0, 2 * math.pi)
            roll = math.radians(rng.normal(0, 3))
            if kind < 0.12:  # no drone (negatives: background, maybe birds)
                tilt = math.radians(rng.uniform(-50, 20))
                fwd = ew.gimbal_forward(yaw, 0.0, tilt)
                target = None
                break
            if kind < 0.24 and len(self.trees):  # partly behind a tree
                t = self.trees[rng.integers(len(self.trees))]
                tz = float(self.ground(t[0], t[1])) + t[4] + t[3]  # canopy centre
                a = rng.uniform(0, 2 * math.pi)
                dc = rng.uniform(12, 45)
                cam = np.array([t[0] - dc * math.cos(a), t[1] - dc * math.sin(a), 0.0])
                cam[2] = max(float(self.ground(*cam[:2])) + 3.0, tz + rng.uniform(-4, 4))
                ray = np.array([t[0], t[1], tz]) - cam
                ray /= np.linalg.norm(ray)
                lat = np.cross(ray, [0, 0, 1.0])
                lat /= max(np.linalg.norm(lat), 1e-6)
                behind = rng.uniform(8, 70)
                # near the edge of the canopy's silhouette at the drone's distance (inside it the drone is hidden)
                edge = (dc + behind) / dc
                ang = rng.uniform(0, 2 * math.pi)
                k = rng.uniform(0.75, 1.25)
                target = (np.array([t[0], t[1], tz]) + ray * behind + lat * math.cos(ang) * t[2] * edge * k
                          + np.array([0, 0, 1.0]) * math.sin(ang) * t[3] * edge * k)
                occlude = True
            else:
                if rng.random() < 0.7:
                    dist = math.exp(rng.uniform(math.log(10), math.log(150)))
                else:
                    dist = rng.uniform(150, 300)
                sky = rng.random() < 0.55
                elv = math.radians(rng.uniform(1, 30) if sky else rng.uniform(-40, 1))
                a = rng.uniform(0, 2 * math.pi)
                target = cam + dist * np.array([math.cos(a) * math.cos(elv), math.sin(a) * math.cos(elv), math.sin(elv)])
            if target[2] < float(self.ground(target[0], target[1])) + 3.0:
                continue
            ray = target - cam
            # aim so that the drone lands at a random pixel: offset the axis by a random angle within the FOV
            yaw = math.atan2(ray[1], ray[0]) + math.radians(rng.uniform(-0.45, 0.45) * self.hfov)
            tilt = math.atan2(ray[2], math.hypot(ray[0], ray[1])) + math.radians(rng.uniform(-0.45, 0.45) * self.vfov)
            tilt = float(np.clip(tilt, math.radians(-70), math.radians(35)))
            fwd = ew.gimbal_forward(yaw, 0.0, tilt)
            uv = ew.project(target[None], cam, fwd, roll)[0]
            if uv[2] > 1.0 and 4 <= uv[0] <= ew.W - 4 and 4 <= uv[1] <= ew.H - 4:
                break
        meta["camera"] = {"pos": [round(float(v), 1) for v in cam],
                          "agl": round(float(cam[2] - self.ground(*cam[:2])), 1),
                          "tilt_deg": round(math.degrees(math.asin(fwd[2] / np.linalg.norm(fwd))), 1),
                          "roll_deg": round(math.degrees(roll), 1)}
        placed = {}
        if target is not None:
            k = int(rng.integers(len(self.drone_mocap)))
            col = self._drone_color(rng, split, self.drone_geoms[k])
            self._place_drone(k, target, rng.uniform(0, 2 * math.pi), rng)
            placed[self.w.drones[k]] = {"cls": 0, "variant": k, "color": col, "occluder": occlude,
                                        "dist": float(np.linalg.norm(target - cam))}
            if rng.random() < 0.08:  # a second drone
                k2 = int((k + 1 + rng.integers(len(self.drone_mocap) - 1)) % len(self.drone_mocap))
                p2 = self._in_view(rng, cam, fwd, roll, (15, 250))
                if p2[2] > float(self.ground(p2[0], p2[1])) + 3.0:
                    col2 = self._drone_color(rng, split, self.drone_geoms[k2])
                    self._place_drone(k2, p2, rng.uniform(0, 2 * math.pi), rng)
                    placed[self.w.drones[k2]] = {"cls": 0, "variant": k2, "color": col2, "occluder": False,
                                                 "dist": float(np.linalg.norm(p2 - cam))}
        nb = int(rng.choice(4, p=[0.45, 0.30, 0.15, 0.10]))
        if kind < 0.12:
            nb = int(rng.choice(4, p=[0.35, 0.30, 0.20, 0.15]))
        for j in rng.choice(len(self.bird_mocap), size=nb, replace=False):
            p = self._in_view(rng, cam, fwd, roll, (10, 200))
            if p[2] < float(self.ground(p[0], p[1])) + 2.0:
                continue
            name, bkind = self.w.birds[j]
            st = en.BIRDS[bkind]
            flap = math.radians(rng.uniform(-st.flap_deg, st.flap_deg)) if rng.random() > st.glide else math.radians(
                rng.uniform(0, 12))
            for g in self.bird_geoms[j]:
                m.geom_rgba[g][:3] = np.clip(self.bird_rgba[g][:3] * rng.uniform(0.85, 1.15), 0, 1)
            en.set_bird(d, self.bird_mocap[j], p, rng.uniform(0, 2 * math.pi), math.radians(rng.normal(0, 15)), flap)
            placed[name] = {"cls": 1, "kind": bkind, "dist": float(np.linalg.norm(p - cam))}
        mujoco.mj_kinematics(m, d)
        self.sr.place(d, cam, fwd, roll)
        rgb = self.sr.rgb(d)
        seg = self.sr.seg(d)
        kept, dropped = ew.boxes_from_seg(seg, self.sr)
        rgb = camera_effects(rgb, rng, meta)
        labels = []
        for row, keep in [(r, True) for r in kept] + [(r, False) for r in dropped]:
            cls, inst, box, npx = row
            name = self.sr.inst_names[inst]
            o = dict(placed.get(name, {"cls": cls}))
            o.update({"name": name, "box": [round(v, 1) for v in box], "pixels": npx, "kept": keep})
            meta["objects"].append(o)
            if keep:
                labels.append((cls, box))
        for name, o in placed.items():  # placed but not visible at all (hidden by a tree / out of view)
            if not any(x["name"] == name for x in meta["objects"]):
                meta["objects"].append(dict(o, name=name, box=None, pixels=0, kept=False))
        return {"rgb": rgb, "labels": labels, "meta": meta}


def camera_effects(rgb: np.ndarray, rng, meta: dict) -> np.ndarray:
    img = rgb.astype(np.float32)
    blur = 0.0
    if rng.random() < 0.4:  # motion blur: the drone moves / the gimbal slews
        L = int(rng.integers(2, 10))
        k = np.zeros((L * 2 + 1, L * 2 + 1), np.float32)
        a = rng.uniform(0, math.pi)
        c, s = math.cos(a), math.sin(a)
        cv2.line(k, (int(L - L * c), int(L - L * s)), (int(L + L * c), int(L + L * s)), 1.0, 1, cv2.LINE_AA)
        img = cv2.filter2D(img, -1, k / k.sum())
        blur = L
    elif rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.0))
    img = img * rng.uniform(0.85, 1.15) + rng.uniform(-12, 12)  # exposure
    noise = rng.uniform(0, 5)
    img += rng.normal(0, noise, img.shape)
    meta["blur_px"], meta["noise"] = blur, round(noise, 1)
    return np.clip(img, 0, 255).astype(np.uint8)


def _init() -> None:
    global _scene
    _scene = DatasetScene()


def _work(args: tuple[str, int, str]) -> dict:
    split, seed, root = args
    assert _scene is not None
    out = _scene.sample(split, seed)
    name = f"{split}_{seed:06d}"
    r = Path(root)
    q = int(np.random.default_rng(seed).integers(82, 96))
    cv2.imwrite(str(r / "images" / split / f"{name}.jpg"), cv2.cvtColor(out["rgb"], cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, q])
    rows = [f"{c} {(b[0] + b[2]) / 2 / ew.W:.6f} {(b[1] + b[3]) / 2 / ew.H:.6f} {(b[2] - b[0]) / ew.W:.6f} "
            f"{(b[3] - b[1]) / ew.H:.6f}" for c, b in out["labels"]]
    (r / "labels" / split / f"{name}.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return dict(out["meta"], image=name)


def generate(root: Path, counts: dict[str, int], workers: int) -> dict[str, list[dict]]:
    for split in counts:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    jobs = [(split, SPLIT_SEED0[split] + i, str(root)) for split, n in counts.items() for i in range(n)]
    metas: dict[str, list[dict]] = {s: [] for s in counts}
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers, initializer=_init) as pool:
        for k, m in enumerate(pool.imap_unordered(_work, jobs, chunksize=8)):
            metas[m["split"]].append(m)
            if (k + 1) % 250 == 0:
                el = time.time() - t0
                print(f"  {k + 1}/{len(jobs)} images, {el:.0f} s, eta {el / (k + 1) * (len(jobs) - k - 1):.0f} s",
                      flush=True)
    for split, ms in metas.items():
        ms.sort(key=lambda m: m["seed"])
        with open(root / "meta" / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for m in ms:
                f.write(json.dumps(m) + "\n")
    print(f"rendered {len(jobs)} images in {time.time() - t0:.0f} s ({workers} workers)")
    return metas


def write_yaml(root: Path, splits: list[str]) -> None:
    lines = [f"path: {root.as_posix()}", "train: images/train", "val: images/val"]
    lines += [f"{s}: images/{s}" for s in splits if s not in ("train", "val")]
    lines += ["names:"] + [f"  {i}: {n}" for i, n in enumerate(ew.CLASSES)]
    (root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataset_card(root: Path, metas: dict[str, list[dict]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    md = ["# Enemy-drone detector - synthetic dataset card", "",
          "Auto-labeled MuJoCo renders of the farm world (demos/base3d_farm.py) from the gimbal camera of our drone",
          f"({ew.W} x {ew.H}, {ew.TRK.hfov_deg:.0f} deg horizontal FOV). Labels = boxes of the VISIBLE pixels from the",
          "segmentation buffer; boxes < 4 px are dropped (object left unlabeled). Classes: " + ", ".join(ew.CLASSES) + ".",
          "Generator: `demos/53_drone_dataset.py`, world: `demos/enemy_world.py`, bodies: `sfhack/sim/enemy.py`.", "",
          f"Test holds out the drone colours {', '.join(HELDOUT_COLORS)} (never in train / val).", "",
          "| split | images | drone boxes | bird boxes | no-drone images | drones behind trees | dropped (< 4 px) |",
          "|---|---|---|---|---|---|---|"]
    for s, ms in metas.items():
        c = Counter(o["cls"] for m in ms for o in m["objects"] if o["kept"])
        nod = sum(1 for m in ms if not any(o["cls"] == 0 for o in m["objects"]))
        occ = sum(1 for m in ms for o in m["objects"] if o.get("occluder") and o["kept"])
        drop = sum(1 for m in ms for o in m["objects"] if not o["kept"] and o["pixels"] > 0)
        md.append(f"| {s} | {len(ms)} | {c[0]} | {c[1]} | {nod} | {occ} | {drop} |")
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    for s, ms in metas.items():
        kw = dict(histtype="step", density=True, label=s, linewidth=1.5)
        ax[0, 0].hist([o["dist"] for m in ms for o in m["objects"] if o["cls"] == 0 and "dist" in o], bins=30, **kw)
        ax[0, 1].hist([max(o["box"][2] - o["box"][0], o["box"][3] - o["box"][1]) for m in ms for o in m["objects"]
                       if o["cls"] == 0 and o["kept"]], bins=40, range=(0, 120), **kw)
        ax[0, 2].hist([m["camera"]["agl"] for m in ms], bins=20, **kw)
        ax[1, 0].hist([m["camera"]["tilt_deg"] for m in ms], bins=20, **kw)
        ax[1, 1].hist([m["sun_el_deg"] for m in ms], bins=20, **kw)
        ax[1, 2].hist([o["dist"] for m in ms for o in m["objects"] if o["cls"] == 1 and "dist" in o], bins=30, **kw)
    for a, t in zip(ax.flat, ["drone distance [m]", "drone box long side [px] (kept)", "camera AGL [m]",
                              "camera tilt [deg]", "sun elevation [deg]", "bird distance [m]"]):
        a.set_title(t, fontsize=10)
        a.grid(alpha=0.3)
    ax[0, 0].legend()
    fig.tight_layout()
    fig.savefig(root / "dataset_conditions.png", dpi=110)
    plt.close(fig)
    md += ["", "![conditions](dataset_conditions.png)", "", "Examples with boxes: `examples/`.", ""]
    (root / "DATASET_CARD.md").write_text("\n".join(md), encoding="utf-8")


def draw_boxes(rgb: np.ndarray, objs: list[dict]) -> np.ndarray:
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for o in objs:
        if o["box"] is None:
            continue
        c = (60, 60, 255) if o["cls"] == 0 else (255, 200, 0)
        if not o["kept"]:
            c = (128, 128, 128)
        x0, y0, x1, y1 = (int(round(v)) for v in o["box"])
        cv2.rectangle(out, (x0 - 2, y0 - 2), (x1 + 1, y1 + 1), c, 1)
        txt = ew.CLASSES[o["cls"]] + (f" {o['dist']:.0f}m" if "dist" in o else "")
        cv2.putText(out, txt, (x0, max(y0 - 5, 9)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, c, 1, cv2.LINE_AA)
    return out


def examples(root: Path, metas: dict[str, list[dict]], n: int = 24) -> None:
    ex = root / "examples"
    ex.mkdir(exist_ok=True)
    picks = [m for s in metas for m in metas[s][: max(4, n // len(metas))]][:n]
    for m in picks:
        img = cv2.cvtColor(cv2.imread(str(root / "images" / m["split"] / f"{m['image']}.jpg")), cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(ex / f"{m['image']}.jpg"), draw_boxes(img, m["objects"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, default=7000)
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (mp.cpu_count() or 2) // 3)))
    ap.add_argument("--out", type=Path, default=data_dir() / "tracker" / "detector" / "dataset")
    a = ap.parse_args()
    counts = {k: v for k, v in {"train": a.train, "val": a.val, "test": a.test}.items() if v > 0}
    metas = generate(a.out, counts, a.workers)
    write_yaml(a.out, list(counts))
    dataset_card(a.out, metas)
    examples(a.out, metas)
    print((a.out / "DATASET_CARD.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
