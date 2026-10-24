"""Demo 23 (night-nav): plots from the demos/21_night_eval.py results.

    python demos/23_night_plots.py --mission berkeley --tags off,ncc,grad,ncc_grad --out docs/night_nav
Per mission: true |estimate - truth| over time since GNSS off (median and 25-75 % band over the seeds) with the
claimed 2-sigma (dashed), and the error at B per seed (strip plot with the 5 m line).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

COLORS = ["#4269d0", "#efb118", "#ff725c", "#6cc5b0", "#3ca951", "#ff8ab7", "#a463f2", "#97bbf5"]


def load(root: Path, mission: str, tag: str) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / mission / tag).glob("seed*.json"))]


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sfhack.config import data_dir

    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default="berkeley")
    ap.add_argument("--tags", default="off,ncc")
    ap.add_argument("--labels", default="", help="comma-separated display names (default: the tags)")
    ap.add_argument("--root", default=str(data_dir() / "night" / "nav_eval"))
    ap.add_argument("--out", default="docs/night_nav")
    ap.add_argument("--tmax", type=float, default=150.0)
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tags = args.tags.split(",")
    labels = args.labels.split(",") if args.labels else ["IMU + TRN only" if t == "off" else t for t in tags]
    for mission in args.mission.split(","):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
        for k, (tag, lab) in enumerate(zip(tags, labels)):
            runs = load(root, mission, tag)
            if not runs:
                continue
            col = COLORS[k % len(COLORS)]
            grid = np.arange(0.0, args.tmax, 1.0)
            E, S = [], []
            for r in runs:
                t0 = r["gnss_off_t"] or 0.0
                tr = np.array([[row[0], row[3], row[4]] for row in r["trace"]], float)
                rel = tr[:, 0] - t0
                e = np.interp(grid, rel, tr[:, 1], right=np.nan)
                s = np.interp(grid, rel, tr[:, 2], right=np.nan)
                E.append(e)
                S.append(s)
            E, S = np.array(E), np.array(S)
            with np.errstate(all="ignore"):
                med = np.nanmedian(E, 0)
                lo, hi = np.nanpercentile(E, 25, 0), np.nanpercentile(E, 75, 0)
                smed = np.nanmedian(S, 0)
            ax.plot(grid, med, color=col, lw=2, label=f"{lab} (n={len(runs)})")
            ax.fill_between(grid, lo, hi, color=col, alpha=0.18, lw=0)
            ax.plot(grid, smed, color=col, lw=1, ls="--")
            fe = np.array([r["final_err"] for r in runs])
            bx.scatter(np.full(len(fe), k) + np.random.default_rng(k).uniform(-0.15, 0.15, len(fe)), fe, s=16, color=col)
            bx.plot([k - 0.3, k + 0.3], [np.median(fe)] * 2, color=col, lw=2)
        ax.set_yscale("symlog", linthresh=5)
        ax.set_xlabel("seconds since GNSS off")
        ax.set_ylabel("|estimate - truth| (m); dashed: claimed 2-sigma")
        ax.set_title(f"{mission}: position error after GNSS loss (median, 25-75 %)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
        bx.axhline(5.0, color="#888", lw=1, ls=":")
        bx.set_yscale("symlog", linthresh=5)
        bx.set_xticks(range(len(tags)), labels, rotation=30, ha="right", fontsize=8)
        bx.set_ylabel("error at B (m)")
        bx.set_title("per seed; dotted: 5 m")
        bx.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        p = out / f"night_error_{mission}.png"
        fig.savefig(p, dpi=110)
        print(p)


if __name__ == "__main__":
    main()
