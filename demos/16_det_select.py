"""V4 detector, step 4b: end-to-end target selection over randomized held-out episodes.

    uv run python demos/16_det_select.py                       # 200 test episodes, YOLO detector
    uv run python demos/16_det_select.py --detector owlv2      # zero-shot baseline, same episodes
    uv run python demos/16_det_select.py --split val --episodes 100   # sanity check on seen variants

Each episode: a new seeded scene (test split = held-out truck / car variants, held-out grounds, random sun,
weather, distractors, occluding trees). The drone hovers above B at 30 m (arrival offset up to 3 m, position
and attitude jitter of a hovering multirotor) for 2 s of down-camera frames (5 Hz -> 10 frames). Every frame
goes through the onboard detector with the altitude / attitude ESTIMATES (noisy), detections are placed on the
ground in the EKF frame (with a per-episode navigation bias) and accumulated by the onboard TruckSelector,
which picks the truck with its margin rule or abstains. HARNESS: the choice is scored on the truth layout
(correct = the chosen point lies on the truck's footprint + 1.5 m); failures are saved as images.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import detlib as dl  # noqa: E402

from sfhack.config import data_dir, default_config  # noqa: E402
from sfhack.perception import truck_detector as td  # noqa: E402
from sfhack.sim import detector_scene as ds  # noqa: E402

SEED0 = {"test": 500_000, "val": 600_000}


def small_rotation(rng: np.random.Generator, sigma_deg: float) -> np.ndarray:
    v = rng.normal(0, math.radians(sigma_deg), 3)
    a = float(np.linalg.norm(v))
    return np.eye(3) if a < 1e-12 else ds.rot_axis(v, a)


def on_object(lay: ds.Layout, lib: dict, xy: np.ndarray, margin: float = 1.5) -> str | None:
    """Truth: which object's footprint (+ margin) contains xy (vehicles / huts / tents first, then trees)."""
    best = None
    for o in sorted(lay.objects, key=lambda o: o.key == "tree"):
        L, W = lib[o.key][o.variant]["size_m"][:2]
        c, s = math.cos(o.yaw), math.sin(o.yaw)
        d = np.asarray(xy) - np.asarray(o.xy)
        lx, ly = c * d[0] + s * d[1], -s * d[0] + c * d[1]
        if abs(lx) <= L / 2 + margin and abs(ly) <= W / 2 + margin:
            best = o.role if o.role in ("target", "brown_car", "white_car") else f"{o.key} ({o.color})" if o.color != "default" else o.key
            break
    return best


def run(a: argparse.Namespace) -> dict:
    lib = ds.load_library()
    grounds = ds.ensure_ground_textures(data_dir() / "v4" / "detector")
    scene = ds.DetectorScene(lib, grounds)
    cfg = default_config()
    n_frames = int(round(a.seconds * cfg.sim.down_cam_hz))
    if a.detector == "yolo":
        det = td.TruckDetector(weights=a.weights, device=a.device)
        dcfg = det.cfg
    else:
        cal = json.loads((data_dir() / "v4" / "detector" / "eval" / a.detector / "calibration.json").read_text())
        det = dl.OwlDetector(negatives=a.detector == "owlv2_neg", temperature=cal["temperature"])
        from dataclasses import replace
        dcfg = replace(cfg.detector, truck_threshold=cal["truck_threshold"])
    out = data_dir() / "v4" / "detector" / "e2e" / f"{a.detector}_{a.split}{a.tag}"
    (out / "failures").mkdir(parents=True, exist_ok=True)
    episodes = []
    t_start = time.time()
    ms = []
    for e in range(a.episodes):
        seed = SEED0[a.split] + e
        lay = ds.sample_layout(lib, seed, a.split)
        rng = np.random.default_rng([seed, 31])
        base = ds.sample_view(rng, len(grounds[ds.ground_group(a.split)]), alt=(a.alt, a.alt), tilt_max_deg=0.0,
                              offset_max=3.0, layout=lay, lib=lib)
        heading = float(rng.uniform(-math.pi, math.pi))
        nav_bias = rng.normal(0, 1.0, 2)  # EKF horizontal error over the burst (m)
        alt_bias = float(rng.normal(0, 0.3))
        R_bias = small_rotation(rng, 0.5)  # AHRS attitude error
        pos = base.pos.copy()
        tilt = np.zeros(2)
        sel = td.TruckSelector(dcfg)
        frames = []
        scene.apply_layout(lay)
        truck_in_view = 0
        for k in range(n_frames):
            # hover jitter: OU position (0.4 m) and roll / pitch (2 deg), yaw 1 deg
            pos[:2] += -0.3 * (pos[:2] - base.pos[:2]) + rng.normal(0, 0.15, 2)
            pos[2] = a.alt + rng.normal(0, 0.1)
            tilt += -0.4 * tilt + rng.normal(0, math.radians(1.2), 2)
            R = ds.rot_axis(np.array([1.0, 0, 0]), tilt[0]) @ ds.rot_axis(np.array([0, 1.0, 0]), tilt[1]) @ \
                ds.rot_z(heading + rng.normal(0, math.radians(1.0)))
            v = ds.View(pos.copy(), R, math.degrees(float(np.linalg.norm(tilt))), base.sun_az, base.sun_el, base.sun,
                        base.ambient, base.ground, base.ground_xy, base.ground_yaw, base.weather, base.weather_i,
                        base.blur_px, base.noise, base.pad_gray)
            r = scene.render(lay, v, rng, layout_applied=True)
            truck_in_view += any(lab.key == "truck" for lab in r.labels)
            alt_est = pos[2] + alt_bias + rng.normal(0, 0.1)
            R_est = R_bias @ R
            xy_est = pos[:2] + nav_bias + rng.normal(0, 0.05, 2)
            dets = det.detect(r.rgb, alt_est, R_est)
            ms.append(det.last_ms)
            dets = [d for d in dets if d.conf >= dcfg.conf_min]
            sel.add(dets, xy_est)
            frames.append((r.rgb, dets))
        s = sel.decide()
        chosen_true = None if s.xy is None else s.xy - nav_bias  # back into the truth frame (harness)
        kind = None if chosen_true is None else on_object(lay, lib, chosen_true)
        outcome = "correct" if kind == "target" else ("abstain" if s.xy is None else "wrong")
        t = lay.truck
        ep = {"episode": e, "seed": seed, "outcome": outcome, "chosen": kind, "reason": s.reason,
              "score": round(s.score, 3), "runner_up": round(s.runner_up, 3), "truck_in_view_frames": truck_in_view,
              "weather": base.weather, "weather_i": round(base.weather_i, 2), "ground": grounds[ds.ground_group(a.split)][base.ground]["kind"],
              "n_distractors": sum(o.role == "distractor" for o in lay.objects),
              "truck_occluders": sum(o.role == "occluder" and math.hypot(o.xy[0] - t.xy[0], o.xy[1] - t.xy[1]) < 6 for o in lay.objects),
              "err_m": None if chosen_true is None else round(float(np.linalg.norm(chosen_true - np.asarray(t.xy))), 2)}
        episodes.append(ep)
        if outcome != "correct":
            save_failure(out / "failures" / f"ep{e:03d}_{outcome}.jpg", frames, ep, dcfg)
        if (e + 1) % 20 == 0:
            k = sum(x["outcome"] == "correct" for x in episodes)
            print(f"  {e + 1}/{a.episodes}: correct {k}, {time.time() - t_start:.0f} s", flush=True)
    scene.close()
    k = sum(x["outcome"] == "correct" for x in episodes)
    n = len(episodes)
    lo, hi = dl.wilson(k, n)
    res = {"detector": a.detector, "split": a.split, "episodes": n, "correct": k, "ci95": [lo, hi],
           "abstain": sum(x["outcome"] == "abstain" for x in episodes), "wrong": sum(x["outcome"] == "wrong" for x in episodes),
           "wrong_objects": {}, "frames_per_episode": n_frames, "alt": a.alt, "ms_per_frame_median": float(np.median(ms)),
           "selector": {"cluster_m": dcfg.select_cluster_m, "min_frames": dcfg.select_min_frames, "margin": dcfg.select_margin,
                        "truck_threshold": dcfg.truck_threshold, "temperature": dcfg.temperature},
           "truck_never_in_view": sum(x["truck_in_view_frames"] == 0 for x in episodes), "episodes_detail": episodes}
    for x in episodes:
        if x["outcome"] == "wrong":
            res["wrong_objects"][x["chosen"] or "nothing"] = res["wrong_objects"].get(x["chosen"] or "nothing", 0) + 1
    (out / "e2e.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[{a.detector}/{a.split}] correct target selection: {k}/{n} = {100 * k / n:.1f} % (95 % CI {100 * lo:.1f}-{100 * hi:.1f} %)"
          f", abstain {res['abstain']}, wrong {res['wrong']} {res['wrong_objects']}, truck never in view {res['truck_never_in_view']}")
    return res


def save_failure(path: Path, frames: list, ep: dict, dcfg) -> None:
    rgb, dets = frames[len(frames) // 2]
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for d in dets:
        if d.conf < 0.2:
            continue
        x0, y0, x1, y1 = (int(round(v)) for v in d.box)
        col = (60, 60, 255) if d.cls == "truck" else (255, 200, 60)
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 1)
        cv2.putText(img, f"{d.cls} {d.conf:.2f}", (x0, max(y0 - 3, 9)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
    txt = f"ep {ep['episode']} {ep['outcome']} chosen={ep['chosen']} ({ep['reason']}) {ep['weather']} {ep['ground']}"
    cv2.putText(img, txt, (6, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--split", default="test", choices=list(SEED0))
    ap.add_argument("--alt", type=float, default=30.0)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--detector", default="yolo", choices=["yolo", "owlv2", "owlv2_neg"])
    ap.add_argument("--weights", default=None, help="YOLO weights (default: DetectorConfig.weights)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default="", help="suffix of the output directory")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
