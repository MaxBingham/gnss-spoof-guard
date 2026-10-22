"""V3 world builder (offline tool, SIMULATOR side): scenario -> world package with a TRUTH part (what
MuJoCo simulates) and an ONBOARD part (the deliberately imperfect map the drone carries).

    uv run python -m sfhack.world.build_v3 s1            # or s2, s2us; --a LAT,LON --b LAT,LON --name X

Package ($SFHACK_DATA/v3/worlds/<name>/):
  manifest.json            origin, extent, A/B, scenario parameters, sources (read by both sides)
  truth/terrain.npz        terrain heightfield grid (Copernicus GLO-30 + small-scale relief) -> MuJoCo hfield
  truth/buildings.npz      building meshes (extruded OSM footprints, true heights), chunked
  truth/dsm.npz            terrain + buildings raster (harness only: scoring, plots, crash check)
  truth/texture.jpg        world ground texture (USGS orthophoto or synthetic land cover)
  truth/patch.jpg          natural scenes: high-res texture of the flat landing site around B
  onboard/dsm.npz          onboard surface model: coarsened DEM + datum offset + perturbed / missing buildings
  onboard/ortho.jpg        onboard imagery for IBN (perturbed copy; natural scenes: landing patch only)
  onboard_heavy/...        the 'map mismatch heavy' onboard variant

Data: Copernicus DEM GLO-30 (AWS open data, COG, no key; ESA, (c) DLR/Airbus, free licence), OSM buildings via
Overpass (ODbL), USGS National Map imagery (public domain). All downloads are cached in $SFHACK_DATA/v3/cache.
The optional USGS 3DEP lidar DSM was skipped (time box): Copernicus + OSM heights is the S1 surface.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
from scipy import ndimage

from ..config import data_dir
from ..geo import GeoOrigin, LatLon, bbox_around
from .scenarios import MapMismatch, Scenario, get_scenario, heavy_mismatch
from .surface import Grid

UA = {"User-Agent": "sfhack-v3/0.1 (GNSS-free navigation research prototype)"}
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
USGS_EXPORT = "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/export"
COP_URL = "https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif"
MESH_CHUNK_M = 150.0  # building meshes are grouped in tiles of this size (cheap ray casts and culling)


def v3_dir() -> Path:
    p = data_dir() / "v3"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir(sub: str) -> Path:
    p = v3_dir() / "cache" / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


def worlds_dir() -> Path:
    p = v3_dir() / "worlds"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------------------------- DEM


def _cop_tile_name(lat0: int, lon0: int) -> str:
    ns = f"N{lat0:02d}" if lat0 >= 0 else f"S{-lat0:02d}"
    ew = f"E{lon0:03d}" if lon0 >= 0 else f"W{-lon0:03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"


def fetch_copernicus(south: float, west: float, north: float, east: float) -> list[Path]:
    paths = []
    for lat0 in range(math.floor(south), math.floor(north) + 1):
        for lon0 in range(math.floor(west), math.floor(east) + 1):
            name = _cop_tile_name(lat0, lon0)
            p = cache_dir("dem") / f"{name}.tif"
            if not p.exists():
                print(f"  downloading {name} ...")
                with httpx.stream("GET", COP_URL.format(n=name), timeout=120, follow_redirects=True) as r:
                    r.raise_for_status()
                    p.write_bytes(r.read())
            paths.append(p)
    return paths


def _read_dem(path: Path) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """(z, (lon0, dlon, lat0, dlat)) of a GeoTIFF; rasterio, fallback tifffile (Copernicus tags)."""
    try:
        import rasterio

        with rasterio.open(path) as ds:
            t = ds.transform
            return ds.read(1).astype(np.float32), (t.c, t.a, t.f, t.e)
    except ImportError:
        import tifffile

        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            z = page.asarray().astype(np.float32)
            sx, sy, _ = page.tags["ModelPixelScaleTag"].value
            tie = page.tags["ModelTiepointTag"].value
            return z, (tie[3], sx, tie[4], -sy)


def sample_dem(lat: np.ndarray, lon: np.ndarray, order: int = 3) -> np.ndarray:
    """DEM height at lat/lon arrays (cubic by default), from the cached Copernicus tiles."""
    out = np.full(lat.shape, np.nan, np.float32)
    for p in fetch_copernicus(float(lat.min()), float(lon.min()), float(lat.max()), float(lon.max())):
        z, (lon0, dlon, lat0, dlat) = _read_dem(p)
        col = (lon - lon0) / dlon - 0.5
        row = (lat - lat0) / dlat - 0.5
        inside = (col >= -0.5) & (col <= z.shape[1] - 0.5) & (row >= -0.5) & (row <= z.shape[0] - 0.5)
        if not inside.any():
            continue
        out[inside] = ndimage.map_coordinates(z, [row[inside], col[inside]], order=order, mode="nearest")
    if np.isnan(out).any():
        raise RuntimeError("DEM does not cover the requested area")
    return out


def grid_latlon(origin: GeoOrigin, g: Grid) -> tuple[np.ndarray, np.ndarray]:
    X, Y = g.cell_centers()
    lat = origin.lat + Y / origin.m_per_deg_lat
    lon = origin.lon + X / origin.m_per_deg_lon
    return lat, lon


def band_noise(shape: tuple[int, int], mpp: float, scales_m: tuple[float, ...], rms: float,
               rng: np.random.Generator) -> np.ndarray:
    """Band-limited random relief: sum of Gaussian-smoothed white noise at the given scales, scaled to rms."""
    if rms <= 0:
        return np.zeros(shape, np.float32)
    acc = np.zeros(shape)
    for s in scales_m:
        n = ndimage.gaussian_filter(rng.normal(0, 1, shape), max(s / mpp / 2.0, 0.5))
        acc += n / (n.std() + 1e-9)
    return (acc / (acc.std() + 1e-9) * rms).astype(np.float32)


def coarsen(g: Grid, cell_m: float) -> Grid:
    """Average to cell_m and interpolate back to the original grid (loses relief below cell_m)."""
    if cell_m <= g.mpp:
        return Grid(g.z.copy(), g.min_x, g.max_y, g.mpp)
    c = g.resampled(cell_m)
    X, Y = g.cell_centers()
    return Grid(c.sample(np.stack([X, Y], -1)).astype(np.float32), g.min_x, g.max_y, g.mpp)


# ---------------------------------------------------------------------------------------------- OSM


def _parse_height(tags: dict[str, str]) -> float | None:
    for key in ("height", "building:height"):
        v = tags.get(key)
        if v:
            try:
                s = v.lower().replace("m", "").replace(",", ".").strip()
                if "'" in s or "ft" in s:
                    return float(s.split("'")[0].replace("ft", "")) * 0.3048
                return float(s.split(";")[0])
            except ValueError:
                pass
    lv = tags.get("building:levels")
    if lv:
        try:
            return float(lv.split(";")[0]) * 3.2 + 1.0
        except ValueError:
            pass
    return None


DATASF = "https://data.sfgov.org/resource/ynuv-fyni.json"  # SF building footprints with lidar heights


def fetch_datasf_buildings(south: float, west: float, north: float, east: float) -> list[dict[str, Any]]:
    """Fallback for San Francisco when Overpass is down: DataSF footprints (heights from 2010 lidar)."""
    key = f"{south:.5f}_{west:.5f}_{north:.5f}_{east:.5f}"
    p = cache_dir("osm") / f"datasf_{key}.json"
    if not p.exists():
        params = {"$select": "globalid,hgt_median_m,shape", "$limit": 50000,
                  "$where": f"within_box(shape, {north}, {west}, {south}, {east})"}
        r = httpx.get(DATASF, params=params, headers=UA, timeout=180, follow_redirects=True)
        r.raise_for_status()
        p.write_text(r.text, encoding="utf-8")
    out = []
    for row in json.loads(p.read_text(encoding="utf-8")):
        shape = row.get("shape") or {}
        polys = shape.get("coordinates", []) if shape.get("type") == "MultiPolygon" else [shape.get("coordinates", [])]
        h = float(row["hgt_median_m"]) if row.get("hgt_median_m") else None
        for poly in polys:
            if poly and len(poly[0]) >= 4:
                out.append({"osm_id": f"datasf/{row.get('globalid')}", "ring": [(la, lo) for lo, la in poly[0]],
                            "height": h, "kind": "yes"})
    return out


def fetch_osm_buildings(south: float, west: float, north: float, east: float,
                        attempts: int = 6) -> list[dict[str, Any]]:
    key = f"{south:.5f}_{west:.5f}_{north:.5f}_{east:.5f}"
    p = cache_dir("osm") / f"buildings_{key}.json"
    if not p.exists():
        q = (f"[out:json][timeout:180];(way[\"building\"]({south},{west},{north},{east});"
             f"relation[\"building\"][\"type\"=\"multipolygon\"]({south},{west},{north},{east}););out tags geom;")
        last = None
        for attempt in range(attempts):
            url = OVERPASS[attempt % len(OVERPASS)]
            try:
                r = httpx.post(url, data={"data": q}, headers=UA, timeout=200)
                if r.status_code == 200 and '"elements"' in r.text[:2000]:
                    p.write_text(r.text, encoding="utf-8")
                    break
                last = f"HTTP {r.status_code}"
            except httpx.HTTPError as e:
                last = repr(e)
            wait = min(10 * 2**attempt, 60)
            print(f"  overpass {url}: {last}, retry in {wait} s")
            time.sleep(wait)
        else:
            raise RuntimeError(f"Overpass failed: {last}")
    data = json.loads(p.read_text(encoding="utf-8"))
    out = []
    for el in data["elements"]:
        tags = el.get("tags", {})
        rings: list[list[tuple[float, float]]] = []
        if el["type"] == "way" and "geometry" in el:
            rings = [[(g["lat"], g["lon"]) for g in el["geometry"]]]
        elif el["type"] == "relation":
            rings = [[(g["lat"], g["lon"]) for g in m["geometry"]] for m in el.get("members", [])
                     if m.get("role") == "outer" and "geometry" in m]
        for ring in rings:
            if len(ring) >= 4:
                out.append({"osm_id": f"{el['type']}/{el['id']}", "ring": ring, "height": _parse_height(tags),
                            "kind": tags.get("building", "yes")})
    return out


DEFAULT_HEIGHT = {"house": 8.0, "residential": 10.0, "apartments": 14.0, "garage": 3.5, "shed": 3.0,
                  "commercial": 12.0, "retail": 7.0, "church": 15.0, "school": 10.0, "roof": 4.0}


# ---------------------------------------------------------------------------------------------- imagery


def fetch_usgs_ortho(origin: GeoOrigin, extent: tuple[float, float, float, float], mpp: float) -> np.ndarray:
    """USGS National Map orthophoto covering the ENU extent, north-up RGB at mpp (tiled export requests)."""
    min_x, min_y, max_x, max_y = extent
    W, H = int(round((max_x - min_x) / mpp)), int(round((max_y - min_y) / mpp))
    key = f"{origin.lat:.5f}_{origin.lon:.5f}_{min_x:.0f}_{min_y:.0f}_{max_x:.0f}_{max_y:.0f}_{mpp}"
    p = cache_dir("usgs") / f"ortho_{key}.jpg"
    if p.exists():
        return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    img = np.zeros((H, W, 3), np.uint8)
    T = 2000
    for r0 in range(0, H, T):
        for c0 in range(0, W, T):
            h, w = min(T, H - r0), min(T, W - c0)
            x0, x1 = min_x + c0 * mpp, min_x + (c0 + w) * mpp
            y1, y0 = max_y - r0 * mpp, max_y - (r0 + h) * mpp
            s, wl = origin.to_latlon(x0, y0)
            n, e = origin.to_latlon(x1, y1)
            params = {"bbox": f"{wl},{s},{e},{n}", "bboxSR": 4326, "imageSR": 4326, "size": f"{w},{h}",
                      "format": "jpg", "f": "image"}
            for attempt in range(5):
                try:
                    r = httpx.get(USGS_EXPORT, params=params, headers=UA, timeout=120)
                    r.raise_for_status()
                    tile = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
                    if tile is None:
                        raise RuntimeError(r.text[:200])
                    break
                except Exception as ex:  # noqa: BLE001 - network: retry with backoff
                    print(f"  USGS tile retry ({ex})")
                    time.sleep(5 * (attempt + 1))
            else:
                raise RuntimeError("USGS imagery export failed")
            tile = cv2.resize(tile, (w, h)) if tile.shape[:2] != (h, w) else tile
            img[r0:r0 + h, c0:c0 + w] = tile[:, :, ::-1]
    cv2.imwrite(str(p), img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    return img


def hillshade(z: np.ndarray, mpp: float, az_deg: float = 315.0, alt_deg: float = 45.0) -> np.ndarray:
    gy, gx = np.gradient(z.astype(np.float64), mpp)
    gy = -gy  # row index grows southwards
    az, alt = math.radians(az_deg), math.radians(alt_deg)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1)


def synthetic_landcover(terrain: Grid, out_mpp: float, rng: np.random.Generator) -> np.ndarray:
    """Plausible aerial texture for natural terrain: forest on slopes / ridges, fields in valleys, hillshade."""
    g = terrain.resampled(out_mpp)
    z = g.z
    H, W = z.shape
    slope = np.hypot(*np.gradient(z, out_mpp))
    forest_noise = ndimage.gaussian_filter(rng.normal(0, 1, (H, W)), 60 / out_mpp)
    forest = (slope + 0.08 * forest_noise / (forest_noise.std() + 1e-9)) > 0.12
    fields = rng.integers(0, 4, (H // 40 + 2, W // 40 + 2))
    fields = cv2.resize(fields.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    palette_fields = np.array([[150, 160, 90], [120, 150, 70], [175, 165, 110], [100, 135, 60]], np.float64)
    img = palette_fields[fields]
    tex = ndimage.gaussian_filter(rng.normal(0, 1, (H, W)), 1.5)
    forest_col = np.stack([45 + 10 * tex, 75 + 15 * tex, 40 + 8 * tex], -1)
    img = np.where(forest[..., None], forest_col, img)
    hs = hillshade(z, out_mpp)[..., None]
    img = img * (0.55 + 0.6 * hs)
    return np.clip(img, 0, 255).astype(np.uint8)


def landing_patch_texture(size_px: int, rng: np.random.Generator) -> np.ndarray:
    """High-res 'aerial photo' of the landing meadow: grass noise, field strips, paths, a few objects."""
    n = size_px
    y, x = np.mgrid[0:n, 0:n] / n
    noise = sum(np.sin((x * fx + y * fy) * 2 * np.pi + ph)
                for fx, fy, ph in rng.uniform([-25, -25, 0], [25, 25, 6.28], (16, 3))) / 16
    img = np.stack([0.32 + 0.08 * noise, 0.46 + 0.10 * noise, 0.22 + 0.05 * noise], -1)
    for _ in range(60):
        cx, cy, rx, ry = rng.uniform([0, 0, 0.01, 0.01], [1, 1, 0.06, 0.06])
        col = rng.choice(np.array([[0.55, 0.52, 0.45], [0.62, 0.60, 0.40], [0.25, 0.35, 0.15], [0.45, 0.42, 0.40],
                                   [0.70, 0.66, 0.55]]))
        img[(abs(x - cx) < rx) & (abs(y - cy) < ry)] = col
    for _ in range(18):
        c = rng.uniform(0, 1)
        w = rng.uniform(0.002, 0.006)
        if rng.random() < 0.5:
            img[abs(y - c) < w] = [0.75, 0.73, 0.70]
        else:
            img[abs(x - c) < w] = [0.75, 0.73, 0.70]
    # landing pad marker at the center (a big 'H' on a light square)
    c0, c1 = int(0.47 * n), int(0.53 * n)
    img[c0:c1, c0:c1] = [0.85, 0.85, 0.82]
    t = max(n // 300, 2)
    m0, m1 = int(0.485 * n), int(0.515 * n)
    img[m0:m1, m0:m0 + t] = img[m0:m1, m1 - t:m1] = [0.8, 0.15, 0.1]
    img[(m0 + m1) // 2 - t // 2:(m0 + m1) // 2 + t // 2 + 1, m0:m1] = [0.8, 0.15, 0.1]
    img += rng.normal(0, 0.02, img.shape)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def perturb_imagery(img: np.ndarray, mm: MapMismatch, rng: np.random.Generator) -> np.ndarray:
    """Onboard copy of the imagery: different exposure / sharpness, optionally regions that changed."""
    f = img.astype(np.float32) / 255.0
    f = np.clip(0.5 + mm.ortho_contrast * (f - 0.5), 0, 1) ** mm.ortho_gamma
    if mm.ortho_blur_px > 0:
        f = cv2.GaussianBlur(f, (0, 0), mm.ortho_blur_px)
    if mm.ortho_change_frac > 0:
        H, W = f.shape[:2]
        area = 0.0
        while area < mm.ortho_change_frac * H * W:
            h, w = rng.integers(H // 30, H // 8), rng.integers(W // 30, W // 8)
            r, c = rng.integers(0, H - h), rng.integers(0, W - w)
            f[r:r + h, c:c + w] = rng.uniform(0.2, 0.8, 3)
            area += h * w
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------------------------- buildings


def _polygons_enu(buildings: list[dict[str, Any]], origin: GeoOrigin, extent: tuple[float, float, float, float]):
    from shapely.geometry import Polygon, box
    from shapely.geometry.polygon import orient

    clip = box(*extent).buffer(-5.0)
    out = []
    for b in buildings:
        pts = [origin.to_enu(lat, lon) for lat, lon in b["ring"]]
        try:
            poly = Polygon(pts).buffer(0)
        except Exception:  # noqa: BLE001 - malformed OSM geometry
            continue
        if poly.is_empty or poly.area < 6.0:
            continue
        poly = poly.intersection(clip)
        if poly.is_empty:
            continue
        geoms = list(poly.geoms) if poly.geom_type == "MultiPolygon" else [poly]
        for g in geoms:
            if g.geom_type != "Polygon" or g.area < 6.0:
                continue
            g = orient(g.simplify(0.3, preserve_topology=True), 1.0)
            h = b["height"] if b["height"] is not None else DEFAULT_HEIGHT.get(b["kind"], 9.0)
            out.append({"poly": g, "height": float(np.clip(h, 2.5, 250.0)), "osm_id": b["osm_id"],
                        "tagged_height": b["height"] is not None})
    return out


def rasterize_buildings(dsm: np.ndarray, g: Grid, polys: list[Any], tops: np.ndarray) -> None:
    """In place: dsm = max(dsm, top) inside each footprint (pixel centers inside the polygon)."""
    for poly, top in zip(polys, tops):
        x0, y0, x1, y1 = poly.bounds
        c0, c1 = int((x0 - g.min_x) / g.mpp) - 1, int((x1 - g.min_x) / g.mpp) + 2
        r0, r1 = int((g.max_y - y1) / g.mpp) - 1, int((g.max_y - y0) / g.mpp) + 2
        c0, r0 = max(c0, 0), max(r0, 0)
        c1, r1 = min(c1, dsm.shape[1]), min(r1, dsm.shape[0])
        if c1 <= c0 or r1 <= r0:
            continue
        mask = np.zeros((r1 - r0, c1 - c0), np.uint8)
        for k, ring in enumerate([poly.exterior, *poly.interiors]):  # exterior filled, courtyards cleared
            xy = np.asarray(ring.coords)
            px = np.stack([(xy[:, 0] - g.min_x) / g.mpp - 0.5 - c0, (g.max_y - xy[:, 1]) / g.mpp - 0.5 - r0], -1)
            cv2.fillPoly(mask, [np.round(px * 16).astype(np.int32)], 1 if k == 0 else 0, shift=4)
        sub = dsm[r0:r1, c0:c1]
        np.maximum(sub, np.where(mask > 0, top, -1e9), out=sub)


def building_meshes(polys: list[Any], bases: np.ndarray, tops: np.ndarray, extent: tuple[float, float, float, float]):
    """Extruded footprints (roof + walls + floor, flat-shaded) grouped in MESH_CHUNK_M tiles.

    Returns a dict of arrays for np.savez: chunk_i_vert (N,3), chunk_i_face (M,3), chunk_i_uv (N,2) with uv in
    the world texture (u = east fraction, v = south fraction of the extent)."""
    import shapely

    min_x, min_y, max_x, max_y = extent
    chunks: dict[tuple[int, int], tuple[list[np.ndarray], list[np.ndarray]]] = {}
    for poly, zb, zt in zip(polys, bases, tops):
        cx, cy = poly.centroid.x, poly.centroid.y
        key = (int((cx - min_x) // MESH_CHUNK_M), int((cy - min_y) // MESH_CHUNK_M))
        verts, faces = chunks.setdefault(key, ([], []))
        n0 = sum(len(v) for v in verts)
        tri = shapely.constrained_delaunay_triangles(poly)
        vlist: list[np.ndarray] = []
        flist: list[list[int]] = []
        k = 0
        for t in tri.geoms:
            c = np.asarray(t.exterior.coords)[:3]
            # CCW seen from above -> normal up (roof); reversed for the floor
            if (c[1, 0] - c[0, 0]) * (c[2, 1] - c[0, 1]) - (c[1, 1] - c[0, 1]) * (c[2, 0] - c[0, 0]) < 0:
                c = c[::-1]
            vlist.append(np.column_stack([c, np.full(3, zt)]))
            flist.append([k, k + 1, k + 2])
            vlist.append(np.column_stack([c[::-1], np.full(3, zb)]))
            flist.append([k + 3, k + 4, k + 5])
            k += 6
        for ring in [poly.exterior, *poly.interiors]:
            xy = np.asarray(ring.coords)
            for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
                q = np.array([[x0, y0, zb], [x1, y1, zb], [x1, y1, zt], [x0, y0, zt]])
                vlist.append(q)
                flist += [[k, k + 1, k + 2], [k, k + 2, k + 3]]
                k += 4
        verts.append(np.concatenate(vlist))
        faces.append(np.asarray(flist, np.int64) + n0)
    out: dict[str, np.ndarray] = {}
    for i, (key, (verts, faces)) in enumerate(sorted(chunks.items())):
        v = np.concatenate(verts).astype(np.float32)
        out[f"chunk{i}_vert"] = v
        out[f"chunk{i}_face"] = np.concatenate(faces).astype(np.int32)
        out[f"chunk{i}_uv"] = np.column_stack([(v[:, 0] - min_x) / (max_x - min_x),
                                               (v[:, 1] - min_y) / (max_y - min_y)]).astype(np.float32)
    out["n_chunks"] = np.array(len(chunks))
    return out


# ---------------------------------------------------------------------------------------------- build


def _flatten_landing_site(g: Grid, center: np.ndarray, radius: float, blend: float) -> float:
    """Natural scenes: level the terrain around B (a meadow), in place. Returns the site height."""
    X, Y = g.cell_centers()
    d = np.hypot(X - center[0], Y - center[1])
    h = float(np.median(g.z[d <= radius]))
    w = np.clip((d - radius) / blend, 0.0, 1.0)
    w = w * w * (3 - 2 * w)
    g.z[:] = (h * (1 - w) + g.z * w).astype(np.float32)
    return h


def _pad_size(sc: Scenario) -> tuple[float, float]:
    """(radius, blend) of the level launch pad at A: the AHRS calibrates on the ground assuming level."""
    return max(6.0, 1.5 * sc.hfield_mpp), max(10.0, 2.0 * sc.hfield_mpp)


def _onboard_map(sc: Scenario, mm: MapMismatch, origin: GeoOrigin, dsm_grid: Grid, polys: list[Any],
                 heights: np.ndarray, a_enu: np.ndarray, b_enu: np.ndarray, site_h: float | None, out: Path,
                 imagery: np.ndarray | None, img_extent: tuple[float, float, float, float] | None,
                 rng: np.random.Generator) -> dict[str, Any]:
    """Onboard DSM + imagery: what the drone carries. Never the truth array (coarsened, offset, perturbed)."""
    out.mkdir(parents=True, exist_ok=True)
    lat, lon = grid_latlon(origin, dsm_grid)
    dem = Grid(sample_dem(lat, lon, order=1), dsm_grid.min_x, dsm_grid.max_y, dsm_grid.mpp)
    _flatten_landing_site(dem, a_enu, *_pad_size(sc))  # the launch pad at A is known and level
    if site_h is not None:  # the landing site is a known, surveyed flat meadow
        _flatten_landing_site(dem, b_enu, sc.landing_patch_m / 2, 150.0)
    onboard = coarsen(dem, mm.dem_coarsen_m)
    onboard.z += np.float32(mm.vertical_offset)
    n_kept = 0
    if polys:
        keep = rng.random(len(polys)) >= mm.building_drop_frac
        h_pert = heights * (1.0 + rng.uniform(-mm.building_height_frac, mm.building_height_frac, len(polys)))
        kept = [p for p, k in zip(polys, keep) if k]
        ground = np.array([float(np.median(onboard.sample(np.asarray(p.exterior.coords)))) for p in kept])
        rasterize_buildings(onboard.z, onboard, kept, ground + h_pert[keep])
        n_kept = int(keep.sum())
    onboard.save(out / "dsm.npz")
    info: dict[str, Any] = {"dsm": "dsm.npz", "dem_coarsen_m": mm.dem_coarsen_m, "vertical_offset": mm.vertical_offset,
                            "buildings_kept": n_kept, "buildings_total": len(polys),
                            "building_height_frac": mm.building_height_frac}
    if imagery is not None and img_extent is not None:
        ob = perturb_imagery(imagery, mm, rng)
        cv2.imwrite(str(out / "ortho.jpg"), ob[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
        info["ortho"] = {"file": "ortho.jpg", "extent": list(img_extent),
                         "mpp": (img_extent[2] - img_extent[0]) / ob.shape[1]}
    return info


def build(sc: Scenario, seed: int = 0, out_name: str | None = None) -> Path:
    t_start = time.time()
    rng = np.random.default_rng(seed)
    out = worlds_dir() / (out_name or sc.name)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    a, b = LatLon(*sc.a), LatLon(*sc.b)
    bb = bbox_around(a, b, margin_m=sc.margin_m)
    origin = bb.center
    sx, sy = origin.to_enu(bb.south, bb.west)
    ex, ey = origin.to_enu(bb.north, bb.east)
    a_enu, b_enu = np.array(origin.to_enu(*sc.a)), np.array(origin.to_enu(*sc.b))
    print(f"[{sc.name}] {sc.title}\n  bbox S{bb.south:.5f} W{bb.west:.5f} N{bb.north:.5f} E{bb.east:.5f}"
          f"  ({ex - sx:.0f} x {ey - sy:.0f} m), A-B {np.linalg.norm(b_enu - a_enu):.0f} m")

    # ---- truth terrain (heightfield): DEM (cubic) + small-scale relief the onboard map does not have
    def make_grid(mpp: float) -> Grid:
        w, h = int((ex - sx) // mpp), int((ey - sy) // mpp)
        return Grid(np.zeros((h, w), np.float32), -w * mpp / 2, h * mpp / 2, mpp)

    tg = make_grid(sc.hfield_mpp)
    lat, lon = grid_latlon(origin, tg)
    tg.z = sample_dem(lat, lon, order=3)
    tg.z += band_noise(tg.shape, tg.mpp, sc.mismatch.small_relief_scales_m, sc.mismatch.small_relief_rms, rng)
    _flatten_landing_site(tg, a_enu, *_pad_size(sc))
    site_h = None
    if sc.landing_patch_m > 0:
        site_h = _flatten_landing_site(tg, b_enu, sc.landing_patch_m / 2, 150.0)
    np.maximum(tg.z, -5.0, out=tg.z)  # sea: Copernicus water ~0, keep the hfield sane
    tg.save(out / "truth" / "terrain.npz")
    extent = tg.extent
    print(f"  terrain {tg.shape} @ {tg.mpp} m, z {tg.z.min():.0f}..{tg.z.max():.0f} m")

    # ---- truth DSM raster (harness) on the dsm grid, same extent
    dg = Grid(np.zeros((int(round(tg.shape[0] * tg.mpp / sc.dsm_mpp)), int(round(tg.shape[1] * tg.mpp / sc.dsm_mpp))),
                       np.float32), tg.min_x, tg.max_y, sc.dsm_mpp)
    X, Y = dg.cell_centers()
    dg.z = tg.sample(np.stack([X, Y], -1)).astype(np.float32)

    polys: list[Any] = []
    heights = np.zeros(0)
    sources = [{"name": "Copernicus DEM GLO-30 (AWS open data)", "license": "Copernicus DEM licence (free)",
                "attribution": "(c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018, provided under "
                               "COPERNICUS by the European Union and ESA"}]
    if sc.kind == "urban":
        try:
            raw = fetch_osm_buildings(bb.south, bb.west, bb.north, bb.east)
            src = {"name": "OpenStreetMap buildings (Overpass API)", "license": "ODbL-1.0",
                   "attribution": "(c) OpenStreetMap contributors"}
        except RuntimeError as e:
            if not (37.70 < origin.lat < 37.83 and -122.53 < origin.lon < -122.35):
                raise
            print(f"  {e} -> falling back to DataSF building footprints (San Francisco)")
            raw = fetch_datasf_buildings(bb.south, bb.west, bb.north, bb.east)
            src = {"name": "DataSF Building Footprints (lidar heights)", "license": "PDDL (DataSF open data)",
                   "attribution": "City and County of San Francisco, DataSF"}
        bl = _polygons_enu(raw, origin, extent)
        polys = [x["poly"] for x in bl]
        heights = np.array([x["height"] for x in bl])
        bases = np.array([float(tg.sample(np.asarray(p.exterior.coords)).min()) - 0.5 for p in polys])
        tops = np.array([float(np.median(tg.sample(np.asarray(p.exterior.coords)))) for p in polys]) + heights
        rasterize_buildings(dg.z, dg, polys, tops)
        meshes = building_meshes(polys, bases, tops, extent)
        np.savez_compressed(out / "truth" / "buildings.npz", **meshes)
        tagged = sum(x["tagged_height"] for x in bl)
        print(f"  buildings: {len(polys)} footprints ({tagged} with a height tag), {int(meshes['n_chunks'])} mesh chunks, "
              f"height median {np.median(heights):.1f} m, max {heights.max():.0f} m")
        sources.append(src)
    dg.save(out / "truth" / "dsm.npz")

    # ---- imagery: world texture + onboard copy
    patch_info = None
    if sc.imagery == "usgs":
        tex = fetch_usgs_ortho(origin, extent, sc.ortho_mpp)
        sources.append({"name": "USGS The National Map - USGSImageryOnly", "license": "Public Domain",
                        "attribution": "Imagery: USGS National Map"})
        onboard_img, onboard_ext = tex, extent
    else:
        tex = synthetic_landcover(tg, sc.ortho_mpp, rng)
        onboard_img, onboard_ext = None, None
    cv2.imwrite(str(out / "truth" / "texture.jpg"), tex[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
    if sc.landing_patch_m > 0:
        n = int(round(sc.landing_patch_m / sc.landing_patch_mpp))
        patch = landing_patch_texture(n, rng)
        cv2.imwrite(str(out / "truth" / "patch.jpg"), patch[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        half = sc.landing_patch_m / 2
        onboard_img = patch
        onboard_ext = (b_enu[0] - half, b_enu[1] - half, b_enu[0] + half, b_enu[1] + half)
        patch_info = {"file": "patch.jpg", "extent": list(onboard_ext), "z": site_h}

    # ---- onboard maps (nominal + heavy mismatch); separate RNG streams so they differ from each other
    onboard = _onboard_map(sc, sc.mismatch, origin, dg, polys, heights, a_enu, b_enu, site_h, out / "onboard",
                           onboard_img, onboard_ext, np.random.default_rng(seed + 1))
    heavy = _onboard_map(sc, heavy_mismatch(sc), origin, dg, polys, heights, a_enu, b_enu, site_h, out / "onboard_heavy",
                         onboard_img, onboard_ext, np.random.default_rng(seed + 2))

    ob = Grid.load(out / "onboard" / "dsm.npz")
    diff = ob.z - dg.z
    print(f"  onboard DSM - truth DSM: mean {diff.mean():+.2f} m, rms {np.sqrt((diff ** 2).mean()):.2f} m, "
          f"identical cells {100 * np.mean(np.abs(diff) < 1e-3):.1f} %")
    manifest = {
        "version": 3, "name": out.name, "scenario": sc.to_json(), "seed": seed,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "bbox": {"south": bb.south, "west": bb.west, "north": bb.north, "east": bb.east},
        "origin": {"lat": origin.lat, "lon": origin.lon},
        "coordinate_frame": "local ENU (x=east, y=north, z=up = height above the DEM datum) in meters, "
                            "equirectangular around origin",
        "extent_enu": list(extent),
        "a": {"lat": sc.a[0], "lon": sc.a[1], "enu": a_enu.round(2).tolist()},
        "b": {"lat": sc.b[0], "lon": sc.b[1], "enu": b_enu.round(2).tolist()},
        "no_fly_enu": [[list(np.round(origin.to_enu(la, lo), 2)) for la, lo in poly] for poly in sc.no_fly],
        "truth": {"terrain": "truth/terrain.npz", "dsm": "truth/dsm.npz", "texture": "truth/texture.jpg",
                  "buildings": "truth/buildings.npz" if polys else None, "patch": patch_info,
                  "small_relief_rms": sc.mismatch.small_relief_rms},
        "onboard": onboard, "onboard_heavy": heavy,
        "sources": sources,
        "notes": "USGS 3DEP lidar DSM skipped (time box); S1 surface = Copernicus GLO-30 + OSM building heights.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  wrote {out} in {time.time() - t_start:.0f} s")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", nargs="+", help="s1 | s2 | s2us (several allowed)")
    ap.add_argument("--a", help="override A as LAT,LON")
    ap.add_argument("--b", help="override B as LAT,LON")
    ap.add_argument("--name", help="output world name (default: scenario name)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from dataclasses import replace

    for name in args.scenario:
        sc = get_scenario(name)
        if args.a:
            sc = replace(sc, a=tuple(float(v) for v in args.a.split(",")))
        if args.b:
            sc = replace(sc, b=tuple(float(v) for v in args.b.split(",")))
        build(sc, seed=args.seed, out_name=args.name)


if __name__ == "__main__":
    main()
