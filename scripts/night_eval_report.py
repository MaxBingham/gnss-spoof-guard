"""Combine night eval runs (demos/30_night_eval.py results.json) into one table + error-over-time plot.

    uv run python scripts/night_eval_report.py RUN_DIR[::SUFFIX] [RUN_DIR[::SUFFIX] ...] --out docs/night_eval

SUFFIX is appended to the row names of that run (e.g. `final_fix_b5409f4::+tiltfix` -> row `thermal+tiltfix`).
Plot: |estimate - truth| over time since GNSS off; a DIVERGED (aborted) flight keeps its last error (it is a
failure, not gone), an arrived flight drops out; the median / IQR is drawn only while at least half of the
flights of a row are still in the air. Log y axis.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("night_eval", HERE.parent / "demos" / "30_night_eval.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)  # type: ignore[union-attr]


def load(specs: list[str]) -> list[dict]:
    out = []
    for s in specs:
        path, _, suffix = s.partition("::")
        p = Path(path)
        p = p / "results.json" if p.is_dir() else p
        for r in json.loads(p.read_text(encoding="utf-8")):
            r = dict(r)
            r["row"] = r["row"] + suffix
            out.append(r)
    return out


def plot(results: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    missions = sorted({r["mission"] for r in results})
    rows = sorted({r["row"] for r in results})
    colors = dict(zip(rows, plt.rcParams["axes.prop_cycle"].by_key()["color"] * 3))
    fig, axes = plt.subplots(1, len(missions), figsize=(6.2 * len(missions), 4.6), squeeze=False)
    for ax, m in zip(axes[0], missions):
        for row in rows:
            rs = [r for r in results if r["mission"] == m and r["row"] == row]
            if not rs:
                continue
            tmax = max(np.array(r["trace"])[:, 0].max() - (r["gnss_off_t"] or 0.0) for r in rs)
            T = np.arange(0.0, tmax + 1.0, 1.0)
            E = np.full((len(rs), len(T)), np.nan)
            for i, r in enumerate(rs):
                tr = np.array(r["trace"])
                t = tr[:, 0] - (r["gnss_off_t"] or 0.0)
                e = np.interp(T, t, tr[:, 1], left=np.nan, right=np.nan)
                if r.get("outcome") == "DIVERGED":  # a failure stays in the statistics at its last error
                    e[T > t.max()] = tr[-1, 1]
                E[i] = e
            alive = np.sum(~np.isnan(E), axis=0) >= max(1, len(rs) / 2)
            with np.errstate(all="ignore"):
                med = np.nanmedian(E, axis=0)
                q1, q3 = np.nanpercentile(E, 25, axis=0), np.nanpercentile(E, 75, axis=0)
            k = int((np.array([r["arrival_err"] for r in rs]) <= ev.SUCCESS_M).sum())
            ax.plot(T[alive], med[alive], color=colors[row], label=f"{row}: {k}/{len(rs)} within 5 m")
            ax.fill_between(T[alive], q1[alive], q3[alive], color=colors[row], alpha=0.18)
        ax.set_yscale("log")
        ax.set_ylim(0.1, 3000)
        ax.axhline(ev.SUCCESS_M, color="k", lw=0.8, ls="--")
        ax.set_title(m)
        ax.set_xlabel("time since GNSS off (s)")
        ax.set_ylabel("|estimate - truth| (m), median + IQR")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="docs/night_eval")
    args = ap.parse_args()
    res = load(args.runs)
    ev.ROWS.update({r["row"]: {} for r in res if r["row"] not in ev.ROWS})  # table() orders rows by ROWS
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = ev.table(res)
    (out / "table.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    plot(res, out / "error_over_time.png")
    print(f"\n{out / 'table.md'}\n{out / 'error_over_time.png'}")


if __name__ == "__main__":
    main()
