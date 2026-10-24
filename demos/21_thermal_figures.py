"""Figures for docs/THERMAL_MODEL.md (writes nothing into the missions).

    uv run python demos/21_thermal_figures.py [--missions berkeley ukraine-1] [--out docs/figures]

  thermal_diurnal.png            apparent temperature over 24 h per material (energy balance) + air
  thermal_maps_<mission>.png     ortho | truth at 21 / 23 / 03 h (white-hot, one scale) | prior | prior sigma
  thermal_hist_<mission>.png     23 h truth: paved roads vs vegetation vs roofs (+ air)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sfhack.missionprep.build import mission_dir  # noqa: E402
from sfhack.thermal.materials import CLS, SEASONS  # noqa: E402
from sfhack.thermal.model import build_thermal, solar_hour  # noqa: E402
from sfhack.thermal.seb import simulate  # noqa: E402

# reference categorical palette (dataviz skill, light surface), fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def diurnal(out: Path, lat: float = 37.87, lon: float = -122.26) -> None:
    c = simulate(lat, SEASONS["summer"])
    show = [("asphalt", "asphalt road"), ("concrete", "concrete"), ("grass", "irrigated grass"),
            ("tree", "tree canopy"), ("water", "water"), ("roof_dark", "roof, insulated"),
            ("roof_gravel", "roof, heavy"), ("dirt_road", "dirt road")]
    tz_shift = -solar_hour(0.0, lon, "summer")[0] % 24  # clock = solar + shift
    clock = (c.hours + tz_shift) % 24
    order = np.argsort(clock)
    fig, ax = plt.subplots(figsize=(10, 5.8), facecolor=SURFACE)
    style(ax)
    full = (len(c.exposures) - 1) * len(c.svfs) + len(c.svfs) - 1  # exposure 1, svf 1 variant
    for k, (name, lab) in enumerate(show):
        row = c.t_app[CLS[name] * len(c.exposures) * len(c.svfs) + full] - 273.15
        ax.plot(clock[order], row[order], color=SERIES[k], lw=2, label=lab)
    ax.plot(clock[order], c.t_air[order] - 273.15, color=INK2, lw=1.5, ls="--", label="air (2 m)")
    for h in (21, 23, 3):
        ax.axvline(h, color=INK2, lw=0.8, ls=":")
        ax.text(h + 0.1, ax.get_ylim()[1] - 1, f"{h:02d} h", color=INK2, fontsize=8, va="top")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("local time (PDT, Berkeley, mid-July, clear and calm); full sun, open sky", color=INK2)
    ax.set_ylabel("apparent LWIR temperature (deg C)", color=INK2)
    ax.set_title("Energy-balance model: what stays warm after a hot day", color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.13), labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out / "thermal_diurnal.png", dpi=110)
    plt.close(fig)
    print(out / "thermal_diurnal.png")


def maps(mid: str, out: Path, crop: tuple[slice, slice] | None) -> None:
    mdir = mission_dir(mid)
    res = {h: build_thermal(mdir, hour=h, write=False, log=lambda s: None)["arrays"] for h in (21.0, 23.0, 3.0)}
    rgb = cv2.cvtColor(cv2.imread(str(mdir / "world" / "ortho.jpg")), cv2.COLOR_BGR2RGB)
    sl = crop or (slice(None), slice(None))
    t23 = res[23.0]["truth"][sl]
    lo, hi = np.percentile(np.concatenate([r["truth"][sl].ravel()[::7] for r in res.values()]), [1, 99])
    panels = [("daytime ortho", rgb[sl], None)]
    panels += [(f"truth {int(h):02d} h  (air {r['air_K'] - 273.15:.1f} C)", r["truth"][sl] - 273.15, "gray")
               for h, r in res.items()]
    panels += [("onboard prior 23 h", res[23.0]["prior"][sl] - 273.15, "gray"),
               ("prior sigma (K)", res[23.0]["sigma"][sl], "Blues")]
    h, w = t23.shape
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.0 * h / w + 0.6), facecolor=SURFACE)
    for ax, (title, img, cmap) in zip(axes, panels):
        if cmap is None:
            ax.imshow(img)
        elif cmap == "gray":
            ax.imshow(img, cmap="gray", vmin=lo - 273.15, vmax=hi - 273.15)
        else:
            im = ax.imshow(img, cmap=cmap, vmin=0, vmax=6)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=7, colors=INK2)
        ax.set_title(title, fontsize=8, color=INK)
        ax.axis("off")
    fig.suptitle(f"{mid}: apparent temperature, white-hot, one scale {lo - 273.15:.0f}..{hi - 273.15:.0f} C",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / f"thermal_maps_{mid}.png", dpi=110)
    plt.close(fig)
    print(out / f"thermal_maps_{mid}.png")

    r = res[23.0]
    t = r["truth"] - 273.15
    groups = [("paved roads (prior road mask)", r["roads"] > 0),
              ("vegetation (grass, canopy)", np.isin(r["cls_truth"], [CLS["grass"], CLS["dry_grass"], CLS["tree"]])),
              ("roofs", np.isin(r["cls_truth"], [CLS[n] for n in ("roof_dark", "roof_light", "roof_gravel", "roof_metal")]))]
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    style(ax)
    bins = np.linspace(np.percentile(t, 0.5), np.percentile(t, 99.8), 90)
    for k, (lab, m) in enumerate(groups):
        if not m.any():
            continue
        hcount, edges = np.histogram(t[m], bins=bins, density=True)
        ax.step(edges[:-1], hcount, where="post", color=SERIES[k], lw=2,
                label=f"{lab}: median {np.median(t[m]):.1f} C")
    ax.axvline(r["air_K"] - 273.15, color=INK2, ls="--", lw=1)
    ax.text(r["air_K"] - 273.15 + 0.1, ax.get_ylim()[1] * 0.95, "air", color=INK2, fontsize=8)
    road, veg = groups[0][1], groups[1][1]
    if road.any() and veg.any():
        ax.set_title(f"{mid}, 23 h truth: road - vegetation median contrast "
                     f"{np.median(t[road]) - np.median(t[veg]):.1f} K", color=INK, loc="left", fontsize=11)
    ax.set_xlabel("apparent temperature (deg C)", color=INK2)
    ax.set_ylabel("share of pixels (density)", color=INK2)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out / f"thermal_hist_{mid}.png", dpi=110)
    plt.close(fig)
    print(out / f"thermal_hist_{mid}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--missions", nargs="+", default=["berkeley", "ukraine-1"])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "docs" / "figures"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    diurnal(out)
    crops = {"berkeley": (slice(300, 1000), slice(0, 800)), "ukraine-1": (slice(100, 800), slice(2600, 4000)),
             "m-c804bb84": (slice(100, 800), slice(600, 1800))}
    for m in a.missions:
        maps(m, out, crops.get(m))


if __name__ == "__main__":
    main()
