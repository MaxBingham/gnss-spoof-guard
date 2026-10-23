"""Pitch-slide chart from an eval run: final error at B per configuration, one panel per scenario.

    uv run python demos/v3_pitch_chart.py --tag quick10    # -> $SFHACK_DATA/v3/eval/quick10/pitch.png

Harness only (reads the scored runs.jsonl of demos/v3_eval.py).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

CONFIGS = [("ins", "INS only", "#8a8985"), ("ins_trn", "INS + TRN", "#2a78d6"),
           ("ins_trn_ibn", "INS + TRN + IBN", "#eb6834")]
WORLDS = {"s1": ("S1  San Francisco, 1.5 km urban hills", 3.0, "landed within 3 m"),
          "s2": ("S2  Carpathian foothills, 8.4 km", 10.0, "arrived within 10 m")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="quick10")
    args = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sfhack.world.build_v3 import v3_dir

    out = v3_dir() / "eval" / args.tag
    runs = [json.loads(line) for line in (out / "runs.jsonl").read_text(encoding="utf-8").splitlines() if line]
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 12, "axes.edgecolor": grid, "xtick.color": ink2, "ytick.color": ink})
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), facecolor="#fcfcfb")
    rng = np.random.default_rng(0)
    for ax, (w, (title, ok_m, ok_txt)) in zip(axes, WORLDS.items()):
        ax.set_facecolor("#fcfcfb")
        labels = []
        for row, (c, label, color) in enumerate(reversed(CONFIGS)):
            fe = np.array([r["final_error"] for r in runs if r["world"] == w and r["config"] == c])
            succ = sum(r["success"] for r in runs if r["world"] == w and r["config"] == c)
            if fe.size == 0:
                labels.append(label)
                continue
            fe = np.clip(fe, 0.1, 5000)
            ax.scatter(fe, row + rng.uniform(-0.12, 0.12, fe.size), s=64, color=color, alpha=0.55,
                       edgecolors="#fcfcfb", linewidths=2, zorder=3)
            med = float(np.median(fe))
            ax.plot([med, med], [row - 0.3, row + 0.3], color=ink, lw=2.5, solid_capstyle="round", zorder=4)
            ax.text(med, row + 0.36, f"median {med:.1f} m" if med < 100 else f"median {med:.0f} m",
                    ha="center", va="bottom", color=ink, fontsize=11, fontweight="bold")
            labels.append(f"{label}\n{succ}/{fe.size} {ok_txt.split()[0]}")
        ax.axvline(ok_m, color=ink2, lw=1, ls=(0, (4, 3)), zorder=2)
        ax.text(ok_m * 1.1, -0.62, f"success: {ok_txt}", color=ink2, fontsize=10, va="bottom")
        ax.set_xscale("log")
        ax.set_xlim(0.07, 3000)
        ax.set_ylim(-0.7, len(CONFIGS) - 0.25)
        ax.set_yticks(range(len(CONFIGS)), labels)
        ax.set_xlabel("final position error at B [m], log scale (one dot per seed)", color=ink2)
        ax.set_title(title, loc="left", color=ink, fontsize=13, fontweight="bold")
        ax.grid(True, axis="x", color=grid, lw=0.8, zorder=0)
        ax.tick_params(length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    n = len({r["seed"] for r in runs})
    fig.suptitle(f"GNSS disabled, calm air, {n} seeds per row: terrain matching turns a miss into a landing",
                 x=0.01, ha="left", color=ink, fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "pitch.png", dpi=150, facecolor=fig.get_facecolor())
    print(out / "pitch.png")


if __name__ == "__main__":
    main()
