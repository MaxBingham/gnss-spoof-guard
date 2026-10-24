"""Validation: is asphalt / built-up ground warmer than vegetation at night over Berkeley? (NIGHT_RESEARCH.md s.6)

Open data only, no account:
- ASTER L1T night thermal scenes (Terra, ~22:15 local solar = ~23:15 PDT, 90 m, bands 13/14) from the
  Microsoft Planetary Computer STAC (collection `aster-l1t`, 2000-2006).
- ESA WorldCover 2021 10 m land cover (`esa-worldcover`), Copernicus GLO-30 DEM (`cop-dem-glo-30`), same STAC.
- OpenStreetMap roads (Overpass API), rasterized with a width per highway class.

Output (figures, CSV, table) goes next to this file; downloaded rasters to $SFHACK_DATA/night/research/aster.

    uv run --no-project --with pystac-client --with planetary-computer --with rasterio --with numpy \
        --with matplotlib --with shapely --with pyproj --with requests python docs/night_research/aster_night_check.py
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import planetary_computer
import pystac_client
import rasterio
import requests
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds
from shapely.geometry import LineString, mapping

OUT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("SFHACK_DATA", Path.home() / "sf-hack-data")) / "night" / "research" / "aster"
DATA.mkdir(parents=True, exist_ok=True)

# Berkeley: bay shore (west) .. flats .. campus .. hills / Tilden (east)
BBOX = (-122.325, 37.840, -122.215, 37.905)  # lon_min, lat_min, lon_max, lat_max
DST_CRS = "EPSG:32610"

# ASTER TIR: radiance = (DN - 1) * UCC [W m-2 sr-1 um-1]; brightness temperature = K2 / ln(K1 / L + 1)
UCC = {13: 0.005693, 14: 0.005225}
K1 = {13: 865.65, 14: 649.60}
K2 = {13: 1349.82, 14: 1274.49}

WC_CLASSES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare", 80: "water", 90: "wetland"}
ROAD_WIDTH_M = {"motorway": 30, "trunk": 22, "primary": 16, "secondary": 13, "tertiary": 11, "residential": 9,
                "unclassified": 8, "motorway_link": 8, "trunk_link": 8, "primary_link": 8, "secondary_link": 8,
                "living_street": 6, "service": 5}

catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                    modifier=planetary_computer.sign_inplace)


def night_items():
    geom = {"type": "Polygon", "coordinates": [[[BBOX[0], BBOX[1]], [BBOX[2], BBOX[1]], [BBOX[2], BBOX[3]],
                                                [BBOX[0], BBOX[3]], [BBOX[0], BBOX[1]]]]}
    items = catalog.search(collections=["aster-l1t"], intersects=geom, limit=500).items()
    return [it for it in items if it.properties.get("sat:orbit_state") == "ascending"]


def read_tir(item):
    with rasterio.open(item.assets["TIR"].href) as src:
        b = transform_bounds("EPSG:4326", src.crs, *BBOX)
        win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
        dn = src.read(window=win).astype(np.float64)  # 5 bands: 10..14
        transform = src.window_transform(win)
        crs = src.crs
    bt = {}
    for band in (13, 14):
        d = dn[band - 10]
        rad = (d - 1.0) * UCC[band]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = K2[band] / np.log(K1[band] / rad + 1.0)
        t[d <= 1] = np.nan
        bt[band] = t
    return bt, transform, crs


def reproject_to(src_href, shape, transform, crs, onehot_classes=None, resampling=Resampling.average):
    with rasterio.open(src_href) as src:
        b = transform_bounds(crs, src.crs, *rasterio.transform.array_bounds(shape[0], shape[1], transform))
        b = (b[0] - 0.01, b[1] - 0.01, b[2] + 0.01, b[3] + 0.01)
        win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
        arr = src.read(1, window=win)
        stf = src.window_transform(win)
        scrs = src.crs
    if onehot_classes is None:
        out = np.full(shape, np.nan, dtype=np.float64)
        reproject(arr.astype(np.float64), out, src_transform=stf, src_crs=scrs, dst_transform=transform,
                  dst_crs=crs, resampling=resampling)
        return out
    fr = {}
    for code, name in onehot_classes.items():
        out = np.zeros(shape, dtype=np.float64)
        reproject((arr == code).astype(np.float64), out, src_transform=stf, src_crs=scrs, dst_transform=transform,
                  dst_crs=crs, resampling=Resampling.average)
        fr[name] = out
    return fr


def stac_href(collection, asset):
    geom = {"type": "Point", "coordinates": [(BBOX[0] + BBOX[2]) / 2, (BBOX[1] + BBOX[3]) / 2]}
    it = next(iter(catalog.search(collections=[collection], intersects=geom, max_items=1).items()))
    return it.assets[asset].href


def osm_roads():
    cache = DATA / "osm_roads_berkeley.json"
    if not cache.exists():
        # 3 x 3 tiles: the public servers time out on the whole box
        classes = "|".join(k for k in ROAD_WIDTH_M if k != "service")
        lon_edges = np.linspace(BBOX[0], BBOX[2], 4)
        lat_edges = np.linspace(BBOX[1], BBOX[3], 4)
        elements, seen = [], set()
        for i in range(3):
            for j in range(3):
                q = (f"[out:json][timeout:90];way[highway~\"^({classes})$\"]"
                     f"({lat_edges[j]},{lon_edges[i]},{lat_edges[j + 1]},{lon_edges[i + 1]});out tags geom;")
                for attempt in range(6):
                    url = ("https://maps.mail.ru/osm/tools/overpass/api/interpreter", "https://overpass-api.de/api/interpreter",
                           "https://overpass.private.coffee/api/interpreter")[attempt % 3]
                    try:
                        r = requests.post(url, data={"data": q}, timeout=150,
                                          headers={"User-Agent": "sfhack-night-research/0.1"})
                        r.raise_for_status()
                        for el in r.json()["elements"]:
                            if el["id"] not in seen:
                                seen.add(el["id"])
                                elements.append(el)
                        break
                    except Exception as e:  # noqa: BLE001 - retry / next mirror
                        print("overpass failed:", url, type(e).__name__)
                else:
                    raise RuntimeError(f"Overpass tile {i},{j} failed")
        cache.write_text(json.dumps({"elements": elements}), encoding="utf-8")
    return json.loads(cache.read_text(encoding="utf-8"))


def road_fraction(shape, transform, crs, osm):
    """Road area fraction per 90 m cell: rasterize buffered centrelines on a 10 m grid, then average."""
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    shapes = []
    for el in osm["elements"]:
        if el.get("type") != "way" or "geometry" not in el:
            continue
        w = ROAD_WIDTH_M.get(el["tags"].get("highway"), 6)
        if el["tags"].get("tunnel") in ("yes", "building_passage") or el["tags"].get("covered") == "yes":
            continue
        xy = [tf.transform(p["lon"], p["lat"]) for p in el["geometry"]]
        if len(xy) >= 2:
            shapes.append((mapping(LineString(xy).buffer(w / 2, cap_style=2)), 1))
    fine = 9
    fine_tf = rasterio.Affine(transform.a / fine, 0, transform.c, 0, transform.e / fine, transform.f)
    hi = rasterize(shapes, out_shape=(shape[0] * fine, shape[1] * fine), transform=fine_tf, fill=0, dtype="uint8")
    return hi.reshape(shape[0], fine, shape[1], fine).mean(axis=(1, 3)), shapes


def fit(y, cols: dict):
    names = list(cols)
    x = np.column_stack([np.ones_like(y)] + [cols[n] for n in names])
    ok = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    coef, *_ = np.linalg.lstsq(x[ok], y[ok], rcond=None)
    pred = x[ok] @ coef
    r2 = 1 - np.sum((y[ok] - pred) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2)
    return dict(zip(["const"] + names, coef)), r2, int(ok.sum())


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = night_items()
    print(f"{len(items)} night (ascending) ASTER scenes over the box")
    first = items[0]
    bt0, transform, crs = read_tir(first)
    shape = bt0[13].shape
    wc = reproject_to(stac_href("esa-worldcover", "map"), shape, transform, crs, WC_CLASSES)
    dem = reproject_to(stac_href("cop-dem-glo-30", "data"), shape, transform, crs)
    osm = osm_roads()
    froad, road_shapes = road_fraction(shape, transform, crs, osm)
    np.savez_compressed(DATA / "berkeley_grid.npz", dem=dem, froad=froad, **{f"wc_{k}": v for k, v in wc.items()})

    dominant = {name: wc[name] >= 0.7 for name in ("built", "tree", "grass", "water")}
    rows = []
    per_scene = {}
    for it in items:
        bt, tf2, _ = read_tir(it)
        t = bt[13]
        if t.shape != shape or tf2 != transform:
            out = np.full(shape, np.nan)
            reproject(t, out, src_transform=tf2, src_crs=crs, dst_transform=transform, dst_crs=crs,
                      resampling=Resampling.bilinear)
            t = out
        tc = t - 273.15
        land = wc["water"] < 0.05
        # cloud screen: clouds at night are much colder than any surface; flag scenes with cold outliers
        p5, p50 = np.nanpercentile(tc[land], 5), np.nanpercentile(tc[land], 50)
        cold_share = float(np.nanmean(tc[land] < p50 - 6))
        means = {k: float(np.nanmean(tc[m])) if m.any() else np.nan for k, m in dominant.items()}
        coef, r2, n = fit(tc, {"f_road": froad, "f_built_nonroad": np.clip(wc["built"] - froad, 0, 1),
                               "f_tree": wc["tree"], "f_grass": wc["grass"] + wc["shrub"],
                               "f_water": wc["water"], "elev_100m": dem / 100.0})
        local = it.datetime.hour + it.datetime.minute / 60 - 7  # PDT (all scenes Apr-Oct)
        row = dict(scene=it.id, date=it.datetime.date().isoformat(), utc=it.datetime.strftime("%H:%M"),
                   pdt=f"{int(local % 24):02d}:{int((local % 1) * 60):02d}", cold_share=round(cold_share, 3),
                   median_land_C=round(float(p50), 2), p5_land_C=round(float(p5), 2),
                   **{f"mean_{k}_C": round(v, 2) for k, v in means.items()},
                   built_minus_tree=round(means["built"] - means["tree"], 2),
                   built_minus_grass=round(means["built"] - means["grass"], 2),
                   **{f"coef_{k}": round(float(v), 2) for k, v in coef.items()}, r2=round(float(r2), 3), n=n)
        rows.append(row)
        per_scene[it.id] = tc
        print(json.dumps(row))

    with open(OUT / "aster_night_berkeley.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # figure for the clearest scene (fewest cold outliers)
    best = min(rows, key=lambda r: r["cold_share"])
    tc = per_scene[best["scene"]]
    ext = rasterio.transform.array_bounds(shape[0], shape[1], transform)
    extent = (ext[0], ext[2], ext[1], ext[3])
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]})
    land = wc["water"] < 0.05
    vmin, vmax = np.nanpercentile(tc[land], [2, 99.5])
    im = axes[0].imshow(tc, extent=extent, cmap="inferno", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=axes[0], shrink=0.8, label="brightness temperature, band 13 (deg C)")
    axes[0].set_title(f"ASTER night TIR {best['date']} {best['pdt']} PDT (90 m)")
    im2 = axes[1].imshow(froad, extent=extent, cmap="gray", vmin=0, vmax=0.6)
    fig.colorbar(im2, ax=axes[1], shrink=0.8, label="OSM road area fraction per 90 m cell")
    axes[1].contour(np.flipud(wc["tree"] >= 0.5), levels=[0.5], colors="lime", linewidths=0.6,
                    extent=extent, origin="lower")
    axes[1].set_title("OSM roads (white), tree cover >= 50 % (green)")
    for ax in axes[:2]:
        ax.set_xticks([])
        ax.set_yticks([])
    groups = [("built >=70 %", dominant["built"]), ("road frac >=35 %", froad >= 0.35),
              ("grass/shrub >=70 %", (wc["grass"] + wc["shrub"]) >= 0.7), ("tree >=70 %", dominant["tree"]),
              ("water >=70 %", dominant["water"])]
    data = [tc[m & np.isfinite(tc)] for _, m in groups]
    axes[2].boxplot(data, showfliers=False)
    axes[2].set_xticks(range(1, len(groups) + 1), [f"{g}\n(n={len(d)})" for (g, _), d in zip(groups, data)],
                       fontsize=7)
    axes[2].set_ylabel("deg C")
    axes[2].set_title("night brightness temperature by cover")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "aster_night_berkeley.png", dpi=110)
    print("figure scene:", best["scene"])


if __name__ == "__main__":
    main()
