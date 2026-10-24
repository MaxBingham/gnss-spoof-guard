"""Write the recon report the drone hands over on the pad (no data link: this is the only output of the flight).

    <out>/report.json        what / where (bearing + distance from the base) / when, searched area, decisions
    <out>/find<id>_<k>.png   thermal snapshots (crop around the find | full frame with the box)
    <out>/track.csv          estimated track, 1 Hz: t, x, y, z, 2-sigma, phase
    <out>/map.png            top view: searched area, search pattern, estimated track, finds, bearing cone

`draw_map` takes optional truth overlays: the harness draws a second, scored map with them (map_truth.png); the
onboard report never contains truth.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .recon import ReconMission, az_to_unit, cone_halfwidth


def write_report(m: ReconMission, out_dir: Path | str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rep = m.report if m.report is not None else m.build_report(m.track_log[-1][0] if m.track_log else 0.0)
    for tr in m._finds():
        for k, s in enumerate(tr.snapshots):
            crop = cv2.cvtColor(cv2.resize(s["crop"], (256, 256), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
            frame = s["frame"]
            h = max(crop.shape[0], frame.shape[0])
            pad = lambda im: cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0, cv2.BORDER_CONSTANT)  # noqa: E731
            img = np.hstack([pad(crop), np.zeros((h, 6, 3), np.uint8), pad(frame)])
            cv2.putText(img, f"#{tr.id} {s['label']} {s['clock']} conf {s['conf']:.2f} AGL {s['agl_m']:.0f} m",
                        (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(out / f"find{tr.id}_{k}.png"), img)
    with open(out / "track.csv", "w", encoding="utf-8") as f:
        f.write("t,x,y,z,sigma2,phase\n")
        for t, x, y, z, s2, ph in m.track_log:
            f.write(f"{t:.1f},{x:.2f},{y:.2f},{z:.2f},{s2:.2f},{ph}\n")
    cv2.imwrite(str(out / "map.png"), draw_map(m))
    (out / "report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return out


def draw_map(m: ReconMission, size: int = 900, truth_track: np.ndarray | None = None,
             truth_targets: list[tuple[np.ndarray, str]] | None = None, title: str = "") -> np.ndarray:
    """Top view centred between the pad and the far end of the search, north up."""
    c = m.c
    u = az_to_unit(m.az)
    far = max(m.max_r, m.plan.r_max if m.plan is not None else 1000.0)
    centre = m.pad + u * far / 2
    span = far * 1.25 + 400.0
    s = size / span

    def px(xy: np.ndarray) -> np.ndarray:
        xy = np.atleast_2d(np.asarray(xy, float))
        return np.stack([(xy[:, 0] - centre[0]) * s + size / 2, size / 2 - (xy[:, 1] - centre[1]) * s], -1).round().astype(np.int32)

    img = np.full((size, size, 3), 24, np.uint8)
    # searched area (thermal footprints)
    cov = m.coverage
    ys, xs = np.nonzero(cov)
    if len(xs):
        cells = np.stack([m.cov_origin[0] + (xs + 0.5) * m.cov_cell, m.cov_origin[1] + (cov.shape[0] - ys - 0.5) * m.cov_cell], -1)
        p = px(cells)
        r = max(1, int(math.ceil(m.cov_cell * s / 2)))
        for q in p:
            cv2.rectangle(img, tuple(q - r), tuple(q + r), (70, 60, 30), -1)
    # bearing cone (cone_k sigma) and line
    n = np.array([u[1], -u[0]])
    rs = np.linspace(0, far * 1.1, 30)
    left = [m.pad + u * r - n * cone_halfwidth(r, m.sigma_az, c) for r in rs]
    right = [m.pad + u * r + n * cone_halfwidth(r, m.sigma_az, c) for r in rs]
    cv2.polylines(img, [px(np.array(left))], False, (0, 140, 255), 1, cv2.LINE_AA)
    cv2.polylines(img, [px(np.array(right))], False, (0, 140, 255), 1, cv2.LINE_AA)
    cv2.line(img, tuple(px(m.pad)[0]), tuple(px(m.pad + u * far * 1.1)[0]), (0, 200, 255), 1, cv2.LINE_AA)
    if m.plan is not None:
        cv2.polylines(img, [px(m.plan.path)], False, (110, 110, 110), 1, cv2.LINE_AA)
    if truth_track is not None and len(truth_track):
        cv2.polylines(img, [px(truth_track[:, :2])], False, (80, 220, 80), 2, cv2.LINE_AA)
    if m.track_log:
        est = np.array([[r[1], r[2]] for r in m.track_log])
        cv2.polylines(img, [px(est)], False, (255, 200, 60), 2, cv2.LINE_AA)
    for tr in m.tracks:
        q = tuple(px(tr.xy)[0])
        if tr.confirmed_t is not None:
            cv2.circle(img, q, 7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f"#{tr.id} {tr.label}", (q[0] + 9, q[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        elif tr.moving:
            cv2.drawMarker(img, q, (200, 0, 200), cv2.MARKER_TILTED_CROSS, 8, 1)
    for xy, kind in truth_targets or []:
        col = (80, 255, 80) if kind == "target" else (150, 150, 150)
        cv2.drawMarker(img, tuple(px(xy)[0]), col, cv2.MARKER_DIAMOND, 9, 1, cv2.LINE_AA)
    cv2.drawMarker(img, tuple(px(m.pad)[0]), (255, 255, 255), cv2.MARKER_SQUARE, 12, 2)
    rep = m.report or {}
    lines = [title or "RECON REPORT", rep.get("summary", ""), f"searched {m.searched_area_km2():.2f} km^2, "
             f"bearing {m.az:03.0f} +- {m.sigma_az:.0f} deg (cone {c.cone_k:.1f} sigma)"]
    legend = "blue: est. track" + ("  green: true track / targets" if truth_track is not None else "") + \
             "  red: finds  orange: cone  gray: pattern  dark: searched"
    for k, line in enumerate(lines + [legend]):
        cv2.putText(img, line, (10, 22 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    # scale bar 500 m
    L = int(500 * s)
    cv2.line(img, (20, size - 20), (20 + L, size - 20), (230, 230, 230), 2)
    cv2.putText(img, "500 m   N up", (20, size - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    return img
