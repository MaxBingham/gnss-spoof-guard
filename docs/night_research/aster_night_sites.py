"""ASTER night check at more sites: hot-summer US towns and the Ukraine mission area (NIGHT_RESEARCH.md s.6).

Same idea as aster_night_check.py + aster_night_analyze.py (Berkeley), generalized:
- night (ascending) ASTER L1T scenes from Planetary Computer, band 13 brightness temperature, 90 m;
- ESA WorldCover 2021 + Copernicus GLO-30 on the same grid;
- roads from Overture Maps (OSM-derived), with class AND surface (paved / unpaved) -> lets us test whether
  unpaved roads are warm;
- each scene registered to the paved-road share on its own (night geolocation errors differ per scene);
- statistics inside the valid, registered area.

    uv run --no-project --with numpy --with scipy --with matplotlib --with rasterio --with shapely --with pyproj \
        --with pystac-client --with planetary-computer --with requests --with duckdb \
        python docs/night_research/aster_night_sites.py --site davis
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aster_night_analyze as ana  # noqa: E402
import aster_night_check as chk  # noqa: E402

OUT = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("SFHACK_DATA", Path.home() / "sf-hack-data")) / "night" / "research" / "aster"

SITES = {
    # hot Central Valley summer: small grid town in farmland (analog for a Ukrainian town)
    "davis": dict(bbox=(-121.81, 38.515, -121.68, 38.575), utc_offset=-7, months=(5, 6, 7, 8, 9)),
    "fresno": dict(bbox=(-119.86, 36.70, -119.72, 36.80), utc_offset=-7, months=(6, 7, 8)),
    # Ukraine: Vinnytsia city + fields; Drohobych / Boryslav / Skhidnytsia (mission ukraine-1 area)
    "vinnytsia": dict(bbox=(28.36, 49.19, 28.56, 49.27), utc_offset=3, months=(5, 6, 7, 8, 9)),
    "drohobych": dict(bbox=(23.28, 49.20, 23.56, 49.38), utc_offset=3, months=(5, 6, 7, 8, 9)),
}
PAVED = {"paved", "asphalt", "concrete", "paving_stones", "sett", "cobblestone", "metal"}
UNPAVED = {"unpaved", "gravel", "dirt", "ground", "compacted", "fine_gravel", "earth", "sand", "grass", "mud", "rock",
           "pebblestone", "woodchips"}
MAJOR = {"motorway", "trunk", "primary", "secondary"}


def overture_roads(bbox, out: Path):
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))["elements"]
    import duckdb
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial; SET s3_region='us-west-2';")
    q = f"""
    SELECT id, class, to_json(road_surface) AS surf, ST_AsGeoJSON(geometry) AS g
    FROM read_parquet('s3://overturemaps-us-west-2/release/2026-10-21.0/theme=transportation/type=segment/*',
                      hive_partitioning=1)
    WHERE subtype = 'road' AND bbox.xmin < {bbox[2]} AND bbox.xmax > {bbox[0]}
      AND bbox.ymin < {bbox[3]} AND bbox.ymax > {bbox[1]}
    """
    els = []
    for id_, cls, surf, g in con.execute(q).fetchall():
        gj = json.loads(g)
        if gj["type"] != "LineString":
            continue
        s = None
        try:
            sl = json.loads(surf) if surf else None
            if sl:
                s = sl[0].get("value")
        except (ValueError, TypeError, AttributeError):
            s = None
        els.append({"type": "way", "id": id_, "tags": {"highway": cls, "surface": s or ""},
                    "geometry": [{"lon": x, "lat": y} for x, y in gj["coordinates"]]})
    out.write_text(json.dumps({"elements": els, "source": "Overture Maps 2026-10-21.0 (OSM-derived, ODbL)"}),
                   encoding="utf-8")
    return els


def paved_class(e):
    """'paved', 'unpaved' or 'unknown' from the surface tag, falling back to the road class (NIGHT_RESEARCH s.4.2)."""
    s, h = e["tags"].get("surface") or "", e["tags"]["highway"]
    if s in PAVED:
        return "paved"
    if s in UNPAVED or h == "track":
        return "unpaved"
    if h in MAJOR or h in ("motorway_link", "trunk_link", "primary_link", "secondary_link"):
        return "paved"
    return "unknown"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import planetary_computer
    import pystac_client
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    from scipy.ndimage import shift as nd_shift

    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, choices=list(SITES))
    ap.add_argument("--clear", default="", help="comma list of dates judged clear from the contact sheet")
    args = ap.parse_args()
    site = SITES[args.site]
    data = ROOT / args.site
    data.mkdir(parents=True, exist_ok=True)
    chk.BBOX = site["bbox"]
    chk.DATA = data
    width_m = dict(chk.ROAD_WIDTH_M, track=4)

    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                    modifier=planetary_computer.sign_inplace)
    b = site["bbox"]
    geom = {"type": "Polygon", "coordinates": [[[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]], [b[0], b[1]]]]}
    items = [i for i in cat.search(collections=["aster-l1t"], intersects=geom, limit=500).items()
             if i.properties.get("sat:orbit_state") == "ascending" and i.datetime.month in site["months"]]
    items.sort(key=lambda i: i.datetime)
    print(f"{args.site}: {len(items)} night scenes in months {site['months']}")
    if not items:
        return
    ref_bt, transform, crs = chk.read_tir(items[0])
    shape = ref_bt[13].shape
    scenes = {}
    for it in items:
        bt, tf2, crs2 = chk.read_tir(it)
        t = bt[13]
        if t.shape != shape or tf2 != transform or crs2 != crs:
            o = np.full(shape, np.nan)
            reproject(t, o, src_transform=tf2, src_crs=crs2, dst_transform=transform, dst_crs=crs,
                      resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
            t = o
        local = (it.datetime.hour + it.datetime.minute / 60 + site["utc_offset"]) % 24
        key = f"{it.datetime.date().isoformat()} {int(local):02d}:{int(local % 1 * 60):02d}"
        scenes[key] = t - 273.15
    wc = chk.reproject_to(chk.stac_href("esa-worldcover", "map"), shape, transform, crs, chk.WC_CLASSES)
    dem = chk.reproject_to(chk.stac_href("cop-dem-glo-30", "data"), shape, transform, crs)
    els = overture_roads(b, data / "overture_roads.json")
    roads = [e for e in els if e["tags"]["highway"] in width_m and e["tags"]["highway"] != "service"]
    groups = {"paved": [e for e in roads if paved_class(e) == "paved"],
              "unpaved": [e for e in roads if paved_class(e) == "unpaved"],
              "unknown": [e for e in roads if paved_class(e) == "unknown"]}
    chk.ROAD_WIDTH_M = width_m
    pad = 45
    hi = {k: ana.fine_raster(chk.road_fraction(shape, transform, crs, {"elements": v})[1], transform, shape, pad)
          if v else np.zeros((shape[0] * 9 + 2 * pad, shape[1] * 9 + 2 * pad), np.float32) for k, v in groups.items()}
    f = {k: ana.share_at(v, shape, pad) for k, v in hi.items()}
    np.savez_compressed(data / "scenes.npz", **{k.replace(" ", "_").replace(":", ""): v for k, v in scenes.items()})

    # contact sheet
    n = len(scenes)
    ncol = min(4, n)
    fig, axes = plt.subplots(int(np.ceil(n / ncol)), ncol, figsize=(4 * ncol, 3.4 * np.ceil(n / ncol)), squeeze=False)
    for ax, (k, t) in zip(axes.ravel(), scenes.items()):
        if np.isfinite(t).any():
            lo, hi_ = np.nanpercentile(t, [2, 98])
            ax.imshow(t, cmap="inferno", vmin=lo, vmax=hi_)
            ax.set_title(f"{k} local  {lo:.0f}..{hi_:.0f} C", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / f"aster_night_{args.site}_contact.png", dpi=80)
    plt.close(fig)

    clear = [c.strip() for c in args.clear.split(",") if c.strip()]
    if not clear:
        print("scenes:", list(scenes), "-> look at the contact sheet, then rerun with --clear DATE,DATE")
        return

    land = wc["water"] < 0.05
    veg = wc["grass"] + wc["shrub"] + wc["crop"]
    L = [f"# ASTER night check: {args.site}", "",
         f"bbox {b}; band 13 brightness temperature, 90 m; roads: Overture 2026-10-21.0 (OSM-derived), paved /"
         " unpaved from the surface tag, else from the class (track = unpaved, motorway..secondary = paved);"
         f" road km in box: " + ", ".join(f"{k} {sum(len(e['geometry']) for e in v) and _km(v):.0f} km"
                                          for k, v in groups.items()) + ".", ""]
    rows = []
    for key, t0 in scenes.items():
        if key.split()[0] not in clear:
            continue
        valid0 = np.isfinite(t0)
        tf = np.where(valid0, t0, np.nanmedian(t0))
        mask = land & valid0
        for _ in range(3):
            mask &= np.roll(mask, 1, 0) & np.roll(mask, -1, 0) & np.roll(mask, 1, 1) & np.roll(mask, -1, 1)
        mask[:6] = mask[-6:] = False
        mask[:, :6] = mask[:, -6:] = False
        reg = ana.register(hi["paved"], shape, pad, [tf], mask)
        sh = (reg["oy"] / 9.0, reg["ox"] / 9.0)
        t = nd_shift(tf, sh, order=1, mode="nearest")
        valid = nd_shift(valid0.astype(float), sh, order=0, mode="constant", cval=0) > 0.5
        for _ in range(3):
            valid &= np.roll(valid, 1, 0) & np.roll(valid, -1, 0) & np.roll(valid, 1, 1) & np.roll(valid, -1, 1)
        valid[:6] = valid[-6:] = False
        valid[:, :6] = valid[:, -6:] = False
        base = land & valid
        east, north = -reg["ox"] * ana_d(transform), reg["oy"] * ana_d(transform)
        L += [f"## {key} local time", "",
              f"Registration to paved-road share: r {reg['corr0']:+.3f} as delivered -> **{reg['best']:+.3f}**"
              f" registered; the delivered image sits {east:+.0f} m east, {north:+.0f} m north of the map.", ""]
        edges = [0, 0.03, 0.10, 0.20, 0.35, 1.01]
        for k in ("paved", "unpaved"):
            cells, meds = [], []
            for a, b2 in zip(edges[:-1], edges[1:]):
                mm = base & (f[k] >= a) & (f[k] < b2)
                if k == "unpaved":
                    mm &= f["paved"] < 0.03  # isolate unpaved roads from paved ones
                meds.append(np.median(t[mm]) if mm.sum() >= 5 else np.nan)
                cells.append(f"{meds[-1]:.2f} (n={int(mm.sum())})" if mm.sum() >= 5 else f"- (n={int(mm.sum())})")
            L.append(f"- {k} road share bins {[f'{int(a*100)}-{int(min(b2,1)*100)} %' for a, b2 in zip(edges[:-1], edges[1:])]}:"
                     f" {' | '.join(cells)}")
        m = base
        x = np.column_stack([np.ones(m.sum()), f["paved"][m], f["unpaved"][m], f["unknown"][m],
                             np.clip(wc["built"][m] - f["paved"][m] - f["unknown"][m], 0, 1), wc["tree"][m], veg[m],
                             wc["bare"][m], dem[m] / 100.0])
        coef, *_ = np.linalg.lstsq(x, t[m], rcond=None)
        r2 = 1 - np.sum((t[m] - x @ coef) ** 2) / np.sum((t[m] - t[m].mean()) ** 2)
        names = ["paved road", "unpaved road", "unknown-surface road", "other built-up", "tree", "grass/shrub/crop",
                 "bare", "elevation per 100 m"]
        L.append(f"- regression (per 100 % share vs. the remainder; n = {int(m.sum())}, R^2 = {r2:.2f}): " +
                 ", ".join(f"{nm} {c:+.1f} K" for nm, c in zip(names, coef[1:])))
        cls = {"built >= 70 %": wc["built"] >= 0.7, "tree >= 50 %": wc["tree"] >= 0.5,
               "grass/shrub/crop >= 50 %": veg >= 0.5, "water >= 90 %": (wc["water"] >= 0.9) & valid}
        L.append("- class medians: " + ", ".join(
            f"{c} {np.median(t[(mk & base) if 'water' not in c else mk]):.1f} C (n={int(((mk & base) if 'water' not in c else mk).sum())})"
            for c, mk in cls.items() if ((mk & base) if "water" not in c else mk).sum() >= 5))
        L.append("")
        rows.append(dict(scene=key, shift_east_m=east, shift_north_m=north, r0=reg["corr0"], r=reg["best"],
                         coef=dict(zip(names, map(float, coef[1:]))), r2=float(r2)))
        # figure per clear scene
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
        lo, hi_ = np.nanpercentile(t[base], [2, 99])
        im = axes[0].imshow(np.where(valid, t, np.nan), cmap="inferno", vmin=lo, vmax=hi_)
        fig.colorbar(im, ax=axes[0], shrink=0.75, label="deg C")
        axes[0].set_title(f"ASTER night {args.site} {key}, registered")
        hp = ana.highpass(t)
        axes[1].imshow(np.where(valid, np.clip(hp, -1.5, 1.5), np.nan), cmap="inferno")
        for k, col in (("paved", "cyan"), ("unpaved", "lime")):
            yy, xx = np.nonzero((f[k] > 0.12) & valid)
            axes[1].scatter(xx, yy, s=2, c=col, alpha=0.45, label=f"{k} roads")
        axes[1].legend(loc="lower left", fontsize=8)
        axes[1].set_title("high-passed temperature (+-1.5 K) with roads")
        for ax in axes[:2]:
            ax.set_xticks([])
            ax.set_yticks([])
        d = ana_d(transform)
        ext = [reg["offs"][0] * d, reg["offs"][-1] * d, reg["offs"][-1] * d, reg["offs"][0] * d]
        im3 = axes[2].imshow(reg["corr"], extent=ext, cmap="viridis")
        axes[2].plot(reg["ox"] * d, reg["oy"] * d, "r+", ms=14, mew=2)
        axes[2].plot(0, 0, "wo", ms=6, mfc="none")
        axes[2].set_title(f"correlation with paved-road share vs shift\nr {reg['corr0']:+.2f} -> {reg['best']:+.2f}")
        axes[2].set_xlabel("x shift (m)")
        axes[2].set_ylabel("y shift (m, down = south)")
        fig.colorbar(im3, ax=axes[2], shrink=0.75)
        fig.tight_layout()
        fig.savefig(OUT / f"aster_night_{args.site}_{key.split()[0]}.png", dpi=100)
        plt.close(fig)
    (OUT / f"aster_night_{args.site}_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / f"aster_night_{args.site}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\n".join(L))


def ana_d(transform):
    return transform.a / 9


def _km(elements):
    from math import cos, radians
    tot = 0.0
    for e in elements:
        g = e["geometry"]
        for p, q in zip(g[:-1], g[1:]):
            dx = (q["lon"] - p["lon"]) * 111320 * cos(radians(p["lat"]))
            dy = (q["lat"] - p["lat"]) * 110540
            tot += (dx * dx + dy * dy) ** 0.5
    return tot / 1000


if __name__ == "__main__":
    main()
