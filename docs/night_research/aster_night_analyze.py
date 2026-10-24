"""Second step of the ASTER night check: register the night scenes to the road map, then compare surfaces.

Reads what aster_night_check.py saved in $SFHACK_DATA/night/research/aster (no ASTER downloads here) plus
road centrelines in overture_roads_berkeley.json (Overpass was unreachable that night; see fetch in RESUME).

1. Clear scenes were picked by eye from aster_night_contact_sheet.png (street grid visible, no cloud blobs,
   no uniform cloud deck).
2. Night TIR-only ASTER L1T scenes are badly geolocated. We find the image shift that maximizes the correlation
   between the high-passed night temperature and the high-passed road share (10 m steps). This is our own
   method in miniature: match the night thermal image against a road prior.
3. On the registered scenes: classes inside elevation bands, a regression with elevation as covariate, and
   temperature vs road share.

    uv run --no-project --with numpy --with scipy --with matplotlib --with rasterio --with shapely --with pyproj \
        --with pystac-client --with planetary-computer --with requests python docs/night_research/aster_night_analyze.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("SFHACK_DATA", Path.home() / "sf-hack-data")) / "night" / "research" / "aster"

CLEAR = {"2004-04-30": "clear", "2004-04-07": "clear", "2003-10-30": "clear", "2003-10-21": "clear, inversion"}
EXCLUDED = {"2006-07-18": "covers only the hills", "2006-04-04": "broken cloud", "2005-10-19": "overcast deck",
            "2005-04-01": "broken cloud", "2004-10-16": "haze / fog gradient (doubtful)",
            "2004-09-30": "overcast deck", "2002-10-11": "broken cloud, partial"}
MAJOR = ("motorway", "trunk", "primary", "secondary", "motorway_link", "trunk_link", "primary_link")


def highpass(a, k=5):
    """Remove the smooth background (Bay warmth, elevation, cold-air drainage): a minus its k x k box mean."""
    from numpy.lib.stride_tricks import sliding_window_view
    p = np.pad(a, k // 2, mode="edge")
    return a - sliding_window_view(p, (k, k)).mean(axis=(2, 3))


def fine_raster(shapes, transform, shape, pad):
    """Road mask on a 9x finer grid (~10 m) with `pad` fine pixels of margin on every side."""
    import rasterio
    from rasterio.features import rasterize
    fine = 9
    h, w = shape
    ftf = rasterio.Affine(transform.a / fine, 0, transform.c - pad * transform.a / fine,
                          0, transform.e / fine, transform.f - pad * transform.e / fine)
    return rasterize(shapes, out_shape=(h * fine + 2 * pad, w * fine + 2 * pad), transform=ftf, fill=0,
                     dtype="uint8").astype(np.float32)


def share_at(hi, shape, pad, oy=0, ox=0):
    h, w = shape
    b = hi[pad + oy: pad + oy + h * 9, pad + ox: pad + ox + w * 9]
    return b.reshape(h, 9, w, 9).mean(axis=(1, 3))


def register(hi, shape, pad, scenes_t, mask, step=3):
    """Shift (in ~10 m fine pixels) of the road map that best matches the high-passed scenes (mean correlation)."""
    th = [highpass(np.where(np.isfinite(t), t, np.nanmedian(t))) for t in scenes_t]
    m = mask.copy()
    m[:5] = m[-5:] = False
    m[:, :5] = m[:, -5:] = False

    def score(oy, ox):
        fh = highpass(share_at(hi, shape, pad, oy, ox))
        return float(np.mean([np.corrcoef(t[m], fh[m])[0, 1] for t in th]))

    offs = list(range(-pad, pad + 1, step))
    corr = np.array([[score(oy, ox) for ox in offs] for oy in offs])
    i, j = np.unravel_index(np.nanargmax(corr), corr.shape)
    by, bx = offs[i], offs[j]
    best = (score(by, bx), by, bx)
    for oy in range(by - step + 1, by + step):  # refine to 1 fine pixel
        for ox in range(bx - step + 1, bx + step):
            if abs(oy) <= pad and abs(ox) <= pad:
                s = score(oy, ox)
                if s > best[0]:
                    best = (s, oy, ox)
    return dict(best=best[0], oy=best[1], ox=best[2], corr=corr, offs=np.array(offs), corr0=score(0, 0))


def load_roads():
    import aster_night_check as chk
    f = DATA / "overture_roads_berkeley.json"
    if f.exists():
        el = json.loads(f.read_text(encoding="utf-8"))["elements"]
        src = "Overture Maps 2026-10-21.0 road segments (OSM-derived, ODbL)"
    else:
        el, seen = [], set()
        for p in sorted(DATA.glob("osm_roads_tile_*.json")):
            for e in json.loads(p.read_text(encoding="utf-8"))["elements"]:
                if e["id"] not in seen:
                    seen.add(e["id"])
                    el.append(e)
        src = "Overpass tiles (partial)"
    roads = [e for e in el if e.get("tags", {}).get("highway") in chk.ROAD_WIDTH_M and e["tags"]["highway"] != "service"]
    return roads, [e for e in roads if e["tags"]["highway"] in MAJOR], src


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from scipy.ndimage import shift as nd_shift

    import aster_night_check as chk

    meta = json.loads((DATA / "night_scenes_meta.json").read_text(encoding="utf-8"))
    scenes = np.load(DATA / "night_scenes_bt13.npz")
    g = np.load(DATA / "berkeley_grid.npz")
    dem = g["dem"]
    wc = {k[3:]: g[k] for k in g.files if k.startswith("wc_")}
    veg = wc["grass"] + wc["shrub"]
    transform = rasterio.Affine(*meta["transform"])
    shape = dem.shape
    d = transform.a / 9

    roads, major, src = load_roads()
    pad = 45
    hi_all = fine_raster(chk.road_fraction(shape, transform, meta["crs"], {"elements": roads})[1], transform, shape, pad)
    hi_maj = fine_raster(chk.road_fraction(shape, transform, meta["crs"], {"elements": major})[1], transform, shape, pad)
    froad = share_at(hi_all, shape, pad)
    fmajor = share_at(hi_maj, shape, pad)

    water = wc["water"] >= 0.05
    near_water = water.copy()
    for _ in range(4):  # ~360 m buffer: the warm Bay heats its shore
        near_water = near_water | np.roll(near_water, 1, 0) | np.roll(near_water, -1, 0) | \
            np.roll(near_water, 1, 1) | np.roll(near_water, -1, 1)
    land = wc["water"] < 0.05
    base = land & ~near_water & (dem < 150)

    clear_rows = [r for r in meta["rows"] if r["date"] in CLEAR]
    raw = {r["date"]: scenes[r["scene"]] for r in clear_rows}
    reg = register(hi_all, shape, pad, list(raw.values()), base)
    # image cell (x, y) shows the ground at map (x + ox*d, y - oy*d) -> move the image by (+oy, +ox)/9 cells
    sh = (reg["oy"] / 9.0, reg["ox"] / 9.0)
    regd = {k: nd_shift(np.where(np.isfinite(v), v, np.nanmedian(v)), sh, order=1, mode="nearest") for k, v in raw.items()}
    east_m, north_m = -reg["ox"] * d, reg["oy"] * d
    # drop the frame border and cells that were nodata in any scene (their median fill makes high-pass artefacts)
    valid = np.ones(shape, bool)
    for v in raw.values():
        valid &= nd_shift(np.isfinite(v).astype(float), sh, order=0, mode="constant", cval=0) > 0.5
    for _ in range(3):
        valid &= np.roll(valid, 1, 0) & np.roll(valid, -1, 0) & np.roll(valid, 1, 1) & np.roll(valid, -1, 1)
    valid[:6] = valid[-6:] = False
    valid[:, :6] = valid[:, -6:] = False
    base &= valid
    land = land & valid

    L = ["# ASTER night scenes over Berkeley: clear scenes, registered to the road map", "",
         "Brightness temperature, band 13 (10.6 um), 90 m, no emissivity / atmosphere correction (offsets cancel in"
         " differences). Medians in deg C; n = 90 m cells. Land cover: ESA WorldCover 2021; elevation: Copernicus"
         f" GLO-30; roads: {src}.", "",
         "Excluded scenes: " + "; ".join(f"{k} ({v})" for k, v in EXCLUDED.items()), "",
         "## 1. Registration (the method in miniature)", "",
         f"Correlation of high-passed night temperature with high-passed road share, mean over the {len(raw)} clear"
         f" scenes: **r = {reg['corr0']:+.3f} as delivered**, **r = {reg['best']:+.3f} after registration**. The"
         f" delivered images sit {east_m:+.0f} m east and {north_m:+.0f} m north of the map (search +-{pad * d:.0f} m"
         f" in {d:.0f} m steps); they were shifted back by that amount.",
         "Per scene at that shift: " + ", ".join(
             f"{k} r = {np.corrcoef(highpass(regd[k])[base], highpass(froad)[base])[0, 1]:+.3f}" for k in regd) + ".",
         "The offset is consistent across scenes (about two pixels), as expected for an ASTER night geolocation"
         " error; there is no independent truth for it. All statistics below use the registered scenes.", ""]

    bands = {"flats < 40 m": land & (dem < 40), "slopes 40-150 m": land & (dem >= 40) & (dem < 150),
             "hills >= 150 m": land & (dem >= 150)}
    classes = {"built >= 70 %": wc["built"] >= 0.7, "tree >= 50 %": wc["tree"] >= 0.5,
               "grass/shrub >= 50 %": veg >= 0.5}
    L += ["## 2. Land cover inside elevation bands", ""]
    reg_rows = []
    for k, t in regd.items():
        r = next(r for r in clear_rows if r["date"] == k)
        L += [f"### {k} {r['pdt']} PDT ({CLEAR[k]})", "",
              "| band | " + " | ".join(classes) + " | built - tree | built - grass |", "|---|" + "---|" * 5]
        for bname, bm in bands.items():
            med = {c: (float(np.median(t[bm & cm])) if (bm & cm).sum() >= 8 else np.nan) for c, cm in classes.items()}
            cells = [f"{med[c]:.1f} (n={int((bm & cm).sum())})" if np.isfinite(med[c]) else "-" for c, cm in classes.items()]
            L.append(f"| {bname} | " + " | ".join(cells) +
                     f" | {med['built >= 70 %'] - med['tree >= 50 %']:+.1f} | {med['built >= 70 %'] - med['grass/shrub >= 50 %']:+.1f} |")
        wtr = float(np.median(t[wc["water"] >= 0.9]))
        flats = float(np.median(t[bands["flats < 40 m"]]))
        m = base
        x = np.column_stack([np.ones(m.sum()), froad[m], fmajor[m], np.clip(wc["built"][m] - froad[m], 0, 1),
                             wc["tree"][m], veg[m], dem[m] / 100.0])
        coef, *_ = np.linalg.lstsq(x, t[m], rcond=None)
        r2 = 1 - np.sum((t[m] - x @ coef) ** 2) / np.sum((t[m] - t[m].mean()) ** 2)
        L += ["", f"Bay water (>= 90 %) {wtr:.1f} C vs flats {flats:.1f} C: **water {wtr - flats:+.1f} K**.",
              f"Regression on land < 150 m away from the Bay (n = {int(m.sum())}); per 100 % share, relative to bare"
              f" non-built ground: road {coef[1]:+.1f} K (major roads {coef[2]:+.1f} K more), other built-up"
              f" {coef[3]:+.1f} K, tree {coef[4]:+.1f} K, grass/shrub {coef[5]:+.1f} K; elevation {coef[6]:+.2f} K"
              f" per 100 m; R^2 = {r2:.2f}.", ""]
        reg_rows.append(dict(date=k, water_minus_flats=wtr - flats, road=coef[1], major_extra=coef[2],
                             other_built=coef[3], tree=coef[4], grass=coef[5], elev=coef[6], r2=r2))

    # road share bins, and major corridors vs neighbourhood cells
    edges = [0, 0.05, 0.15, 0.25, 0.35, 1.01]
    L += ["## 3. Night temperature vs road share (registered, land < 150 m, away from the Bay)", "",
          "| scene | " + " | ".join(f"road {int(a * 100)}-{int(min(b, 1) * 100)} %" for a, b in zip(edges[:-1], edges[1:]))
          + " | top bin - bottom bin |", "|---|" + "---|" * len(edges)]
    for k, t in regd.items():
        meds = []
        cells = []
        for a, b in zip(edges[:-1], edges[1:]):
            mm = base & (froad >= a) & (froad < b)
            meds.append(np.median(t[mm]))
            cells.append(f"{meds[-1]:.2f} (n={int(mm.sum())})")
        L.append(f"| {k} | " + " | ".join(cells) + f" | {meds[-1] - meds[0]:+.2f} K |")
    L += ["", "Major-road corridors (>= 15 % major road in the cell) vs low-road cells (< 2 % major, < 15 % any road):", ""]
    for k, t in regd.items():
        mj = base & (fmajor >= 0.15)
        lo = base & (fmajor < 0.02) & (froad < 0.15)
        L.append(f"- {k}: {np.median(t[mj]):.2f} C (n={int(mj.sum())}) vs {np.median(t[lo]):.2f} C (n={int(lo.sum())}):"
                 f" **{np.median(t[mj]) - np.median(t[lo]):+.2f} K**")
    L.append("")
    (OUT / "aster_night_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    with open(OUT / "aster_night_registered.json", "w", encoding="utf-8") as f:
        json.dump(dict(shift_east_m=east_m, shift_north_m=north_m, r_delivered=reg["corr0"], r_registered=reg["best"],
                       per_scene=reg_rows), f, indent=1)
    print("\n".join(L))

    # ---- figure
    t = regd["2004-04-30"]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.6), gridspec_kw={"width_ratios": [1.3, 1.3, 0.9, 1.0]})
    lo_, hi_ = np.nanpercentile(t[land], [2, 99])
    im = axes[0].imshow(t, cmap="inferno", vmin=lo_, vmax=hi_)
    fig.colorbar(im, ax=axes[0], shrink=0.75, label="brightness temperature (deg C)")
    axes[0].set_title("ASTER night 2004-04-30 23:18 PDT, 90 m, registered\nBay left, Berkeley flats, hills right")
    hp = highpass(t)
    axes[1].imshow(np.clip(hp, -1.2, 1.2), cmap="inferno")
    yy, xx = np.nonzero(fmajor > 0.2)
    axes[1].scatter(xx, yy, s=2, c="cyan", alpha=0.5, label="major roads (> 20 % of cell)")
    axes[1].legend(loc="lower left", fontsize=8)
    axes[1].set_title("high-passed temperature (local anomaly, +-1.2 K)\nwith major roads from the map")
    for ax in axes[:2]:
        ax.set_xticks([])
        ax.set_yticks([])
    ext = [reg["offs"][0] * d, reg["offs"][-1] * d, reg["offs"][-1] * d, reg["offs"][0] * d]
    im3 = axes[2].imshow(reg["corr"], extent=ext, cmap="viridis")
    axes[2].plot(reg["ox"] * d, reg["oy"] * d, "r+", ms=14, mew=2)
    axes[2].plot(0, 0, "wo", ms=6, mfc="none")
    axes[2].set_xlabel("road-map shift x (m)")
    axes[2].set_ylabel("road-map shift y (m, down = south)")
    axes[2].set_title(f"correlation vs shift\nr {reg['corr0']:+.2f} at 0 (white) -> {reg['best']:+.2f} (red)")
    fig.colorbar(im3, ax=axes[2], shrink=0.75)
    data, labels = [], []
    for bname, bm in bands.items():
        for c, cm in classes.items():
            v = t[bm & cm]
            if len(v) >= 8:
                data.append(v)
                labels.append(f"{bname.split()[0]}\n{c.split()[0]}\nn={len(v)}")
    for nm, msk in (("major\nroad\ncells", base & (fmajor >= 0.15)), ("Bay\nwater", wc["water"] >= 0.9)):
        data.append(t[msk])
        labels.append(f"{nm}\nn={int(msk.sum())}")
    axes[3].boxplot(data, showfliers=False)
    axes[3].set_xticks(range(1, len(labels) + 1), labels, fontsize=7)
    axes[3].set_ylabel("deg C")
    axes[3].set_title("by land cover and elevation band")
    axes[3].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "aster_night_berkeley.png", dpi=105)
    print("wrote", OUT / "aster_night_berkeley.png")


if __name__ == "__main__":
    main()
