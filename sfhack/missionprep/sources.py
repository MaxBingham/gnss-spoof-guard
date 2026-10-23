"""Open-data downloads for the mission package, each with an on-disk cache, retries and a fallback.

| Layer     | Primary                                        | Fallback                                    |
| --------- | ---------------------------------------------- | ------------------------------------------- |
| OSM       | Overpass API (overpass-api.de), one query      | Overpass mirror, then the OSM API `map` call |
| Ortho     | USGS NAIP ImageServer exportImage (~0.6 m)     | USGS National Map imagery tiles, z16        |
| Terrain   | USGS 3DEP ImageServer (1 m DEM where it exists) | Copernicus GLO-30 DSM (AWS open data)       |
| LiDAR     | USGS 3DEP Entwine point tiles (AWS usgs-lidar-public), read with laspy | (caller: synthetic cloud) |

Every response is cached under $SFHACK_DATA/cache/missionprep keyed by URL + parameters, so a second
prepare of the same area needs no network. All requests send a descriptive User-Agent.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx
import numpy as np
from PIL import Image

from ..config import data_dir
from ..geo import BBox, GeoOrigin
from .geometry import Grid

USER_AGENT = "sfhack-mission-planner/0.5 (GNSS-free nav research prototype; contact via github.com/MaxBingham/gnss-spoof-guard)"
OVERPASS_ENDPOINTS = [
    ("https://overpass-api.de/api/interpreter", 90.0),
    ("https://overpass.private.coffee/api/interpreter", 45.0),
]
OSM_API_MAP = "https://api.openstreetmap.org/api/0.6/map"
NAIP_EXPORT = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage"
USGS_TILE = "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
DEM_EXPORT = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
TNM_PRODUCTS = "https://tnmaccess.nationalmap.gov/api/v1/products"
COPERNICUS = "https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif"
NOMINATIM = "https://nominatim.openstreetmap.org/search"
MAX_EXPORT_PX = 4000  # ArcGIS ImageServer size limit

SRC_OSM = {"name": "OpenStreetMap", "license": "ODbL-1.0", "attribution": "© OpenStreetMap contributors",
           "url": "https://www.openstreetmap.org/copyright"}
SRC_NAIP = {"name": "USDA NAIP via USGS The National Map (USGSNAIPImagery)", "license": "Public Domain",
            "attribution": "Imagery: USDA NAIP / USGS", "url": "https://imagery.nationalmap.gov/"}
SRC_USGS_TILES = {"name": "USGS The National Map USGSImageryOnly tiles", "license": "Public Domain",
                  "attribution": "Imagery: USGS National Map", "url": "https://basemap.nationalmap.gov/"}
SRC_3DEP = {"name": "USGS 3DEP elevation (3DEPElevation ImageServer)", "license": "Public Domain",
            "attribution": "Elevation: USGS 3DEP", "url": "https://elevation.nationalmap.gov/"}
SRC_COP = {"name": "Copernicus DEM GLO-30 (AWS open data)", "license": "Copernicus DEM licence (free, attribution)",
           "attribution": "© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, provided under "
                          "COPERNICUS by the European Union and ESA", "url": "https://registry.opendata.aws/copernicus-dem/"}
SRC_LIDAR = {"name": "USGS 3DEP LiDAR point cloud (Entwine, AWS usgs-lidar-public)", "license": "Public Domain",
             "attribution": "LiDAR: USGS 3DEP", "url": "https://registry.opendata.aws/usgs-lidar/"}


class FetchError(RuntimeError):
    pass


def cache_dir() -> Path:
    p = data_dir() / "cache" / "missionprep"
    p.mkdir(parents=True, exist_ok=True)
    return p


class Fetcher:
    """httpx client + content-addressed file cache + retries. Thread-safe (httpx.Client is)."""

    def __init__(self, offline: bool = False, log: Callable[[str], None] | None = None) -> None:
        self.client = httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True,
                                   limits=httpx.Limits(max_connections=16))
        self.offline = offline
        self.log = log or (lambda s: None)
        self.lock = threading.Lock()
        self.bytes_net = 0
        self.bytes_cache = 0

    def close(self) -> None:
        self.client.close()

    def _path(self, key: str) -> Path:
        h = hashlib.sha1(key.encode()).hexdigest()
        return cache_dir() / h[:2] / h

    def get(self, url: str, params: dict[str, Any] | None = None, *, data: dict[str, str] | None = None,
            timeout: float = 60.0, retries: int = 2, check: Callable[[httpx.Response], None] | None = None,
            cache: bool = True) -> bytes:
        """GET (or POST form `data`) with cache. `check` raises on a bad 200 (HTML error page etc.)."""
        key = json.dumps([url, params or {}, data or {}], sort_keys=True)
        path = self._path(key)
        if cache and path.exists():
            b = path.read_bytes()
            with self.lock:
                self.bytes_cache += len(b)
            return b
        if self.offline:
            raise FetchError(f"offline and not cached: {url}")
        err: Exception | None = None
        for attempt in range(retries + 1):
            try:
                if data is not None:
                    r = self.client.post(url, data=data, timeout=timeout)
                else:
                    r = self.client.get(url, params=params, timeout=timeout)
                if r.status_code in (429, 502, 503, 504):
                    raise FetchError(f"HTTP {r.status_code}")
                r.raise_for_status()
                if check:
                    check(r)
                b = r.content
                with self.lock:
                    self.bytes_net += len(b)
                if cache:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(f".tmp{threading.get_ident()}")
                    tmp.write_bytes(b)
                    tmp.replace(path)
                return b
            except (httpx.HTTPError, FetchError, ValueError) as e:
                err = e
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
        raise FetchError(f"{url}: {err!r}")


def _is_image(r: httpx.Response) -> None:
    if not r.headers.get("content-type", "").startswith("image"):
        raise FetchError(f"not an image: {r.headers.get('content-type')} {r.text[:200]!r}")


# ------------------------------------------------------------------------------------------------ OSM


def overpass_query(bbox: BBox) -> str:
    b = f"({bbox.south:.6f},{bbox.west:.6f},{bbox.north:.6f},{bbox.east:.6f})"
    return f"""[out:json][timeout:80];
(
  way["building"]{b};
  relation["building"]{b};
  node["natural"="tree"]{b};
  way["natural"="tree_row"]{b};
  node["man_made"="tower"]{b};
  way["man_made"="tower"]{b};
  node["amenity"="fountain"]{b};
  way["amenity"="fountain"]{b};
  way["leisure"~"^(park|garden|pitch|playground)$"]{b};
  way["landuse"]{b};
  way["natural"~"^(water|wood|scrub|grassland)$"]{b};
  way["highway"]{b};
  way["amenity"="parking"]{b};
);
out geom;"""


def _check_overpass(r: httpx.Response) -> None:
    d = r.json()
    if "elements" not in d:
        raise FetchError("Overpass response without elements")
    if d.get("remark", "").lower().startswith(("runtime error", "error")):
        raise FetchError(f"Overpass: {d['remark'][:200]}")


def osm_api_to_elements(xml_bytes: bytes) -> dict[str, Any]:
    """OSM API 0.6 XML -> the Overpass `out geom` JSON shape (ways/relations carry their geometry)."""
    root = ET.fromstring(xml_bytes)
    nodes: dict[int, tuple[float, float]] = {}
    ways: dict[int, list[dict[str, float]]] = {}
    out: list[dict[str, Any]] = []

    def tags(el: ET.Element) -> dict[str, str]:
        return {t.get("k", ""): t.get("v", "") for t in el.findall("tag")}

    for n in root.findall("node"):
        nid = int(n.get("id", 0))
        nodes[nid] = (float(n.get("lat", 0)), float(n.get("lon", 0)))
        t = tags(n)
        if t:
            out.append({"type": "node", "id": nid, "lat": nodes[nid][0], "lon": nodes[nid][1], "tags": t})
    for w in root.findall("way"):
        wid = int(w.get("id", 0))
        geom = [{"lat": nodes[r][0], "lon": nodes[r][1]} for r in (int(nd.get("ref", 0)) for nd in w.findall("nd"))
                if r in nodes]
        ways[wid] = geom
        t = tags(w)
        if t:
            out.append({"type": "way", "id": wid, "geometry": geom, "tags": t})
    for rel in root.findall("relation"):
        t = tags(rel)
        if "building" not in t:
            continue
        members = [{"type": "way", "role": m.get("role", ""), "geometry": ways[int(m.get("ref", 0))]}
                   for m in rel.findall("member") if m.get("type") == "way" and int(m.get("ref", 0)) in ways]
        out.append({"type": "relation", "id": int(rel.get("id", 0)), "members": members, "tags": t})
    return {"elements": out}


def fetch_osm(f: Fetcher, bbox: BBox) -> tuple[dict[str, Any], dict[str, Any]]:
    """One Overpass query (cached); on failure the mirror, then the OSM API map call."""
    errors = []
    q = overpass_query(bbox)
    for url, timeout in OVERPASS_ENDPOINTS:
        try:
            t0 = time.time()
            b = f.get(url, data={"data": q}, timeout=timeout, retries=1, check=_check_overpass)
            return json.loads(b), {**SRC_OSM, "endpoint": url, "seconds": round(time.time() - t0, 1)}
        except FetchError as e:
            errors.append(str(e))
            f.log(f"  OSM: {url} failed ({str(e)[:120]}), trying the next endpoint")
    try:
        params = {"bbox": f"{bbox.west:.6f},{bbox.south:.6f},{bbox.east:.6f},{bbox.north:.6f}"}
        b = f.get(OSM_API_MAP, params, timeout=90.0, retries=1)
        return osm_api_to_elements(b), {**SRC_OSM, "endpoint": OSM_API_MAP}
    except FetchError as e:
        errors.append(str(e))
    raise FetchError("OSM download failed: " + " | ".join(errors))


# ------------------------------------------------------------------------------------------------ ortho


def _real_imagery(img: Image.Image) -> bool:
    return float(np.asarray(img.convert("L"), np.float32).std()) > 4.0


def fetch_ortho(f: Fetcher, bbox: BBox, grid: Grid) -> tuple[np.ndarray, dict[str, Any]]:
    """RGB ortho (rows, cols, 3) exactly on `grid` (north-up, equirectangular = linear in lat/lon)."""
    w, h = grid.cols, grid.rows
    s = min(1.0, MAX_EXPORT_PX / max(w, h))
    rw, rh = max(1, round(w * s)), max(1, round(h * s))
    try:
        params = {"bbox": f"{bbox.west},{bbox.south},{bbox.east},{bbox.north}", "bboxSR": 4326, "imageSR": 4326,
                  "size": f"{rw},{rh}", "format": "jpg", "interpolation": "RSP_BilinearInterpolation", "f": "image"}
        img = Image.open(io.BytesIO(f.get(NAIP_EXPORT, params, timeout=90.0, check=_is_image))).convert("RGB")
        if img.size != (rw, rh) or not _real_imagery(img):
            raise FetchError(f"NAIP export returned {img.size}, expected {(rw, rh)} / blank")
        src = {**SRC_NAIP, "request_px": [rw, rh]}
    except (FetchError, OSError) as e:
        f.log(f"  ortho: NAIP failed ({str(e)[:120]}), falling back to USGS tiles z16")
        img = _usgs_tiles(f, bbox, 16)
        src = {**SRC_USGS_TILES, "zoom": 16}
    img = img.resize((w, h), Image.Resampling.LANCZOS) if img.size != (w, h) else img
    return np.asarray(img), src


def _lonlat_px(lat: float, lon: float, z: int) -> tuple[float, float]:
    n = 256 * 2 ** z
    lr = math.radians(lat)
    return (lon + 180.0) / 360.0 * n, (1.0 - math.log(math.tan(lr) + 1.0 / math.cos(lr)) / math.pi) / 2.0 * n


def _usgs_tiles(f: Fetcher, bbox: BBox, z: int) -> Image.Image:
    x0, y0 = _lonlat_px(bbox.north, bbox.west, z)
    x1, y1 = _lonlat_px(bbox.south, bbox.east, z)
    tx0, ty0, tx1, ty1 = int(x0 // 256), int(y0 // 256), int(x1 // 256), int(y1 // 256)
    coords = [(tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]

    def tile(c: tuple[int, int]) -> Image.Image:
        return Image.open(io.BytesIO(f.get(USGS_TILE.format(z=z, x=c[0], y=c[1]), timeout=30.0, check=_is_image))).convert("RGB")

    with ThreadPoolExecutor(4) as ex:
        tiles = list(ex.map(tile, coords))
    mosaic = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256))
    for (tx, ty), t in zip(coords, tiles):
        mosaic.paste(t, ((tx - tx0) * 256, (ty - ty0) * 256))
    # Web Mercator rows are not linear in latitude, but over < 3 km the difference is < 0.1 px.
    return mosaic.crop((round(x0 - tx0 * 256), round(y0 - ty0 * 256), round(x1 - tx0 * 256), round(y1 - ty0 * 256)))


# ------------------------------------------------------------------------------------------------ terrain


def _tiff(b: bytes) -> np.ndarray:
    import tifffile

    return np.asarray(tifffile.imread(io.BytesIO(b)), np.float32)


def has_3dep_1m(f: Fetcher, bbox: BBox) -> bool:
    try:
        params = {"datasets": "Digital Elevation Model (DEM) 1 meter", "max": 1, "outputFormat": "JSON",
                  "bbox": f"{bbox.west},{bbox.south},{bbox.east},{bbox.north}"}
        return int(json.loads(f.get(TNM_PRODUCTS, params, timeout=30.0)).get("total", 0)) > 0
    except (FetchError, ValueError):
        return False


def fetch_dem(f: Fetcher, bbox: BBox, grid: Grid) -> tuple[np.ndarray, dict[str, Any]]:
    """Elevation (m above the source's vertical datum) on `grid`. 3DEP where it covers the bbox, else GLO-30."""
    one_m = has_3dep_1m(f, bbox)
    try:
        s = min(1.0, MAX_EXPORT_PX / max(grid.cols, grid.rows))
        rw, rh = max(2, round(grid.cols * s)), max(2, round(grid.rows * s))
        params = {"bbox": f"{bbox.west},{bbox.south},{bbox.east},{bbox.north}", "bboxSR": 4326, "imageSR": 4326,
                  "size": f"{rw},{rh}", "format": "tiff", "pixelType": "F32", "noData": -9999,
                  "interpolation": "RSP_BilinearInterpolation", "f": "image"}
        dem = _tiff(f.get(DEM_EXPORT, params, timeout=90.0))
        if dem.shape != (rh, rw) or not np.isfinite(dem).all() or (dem < -500).any() or float(dem.std()) == 0.0:
            raise FetchError(f"3DEP export shape {dem.shape} / nodata")
        src = {**SRC_3DEP, "product": "3DEP 1 m DEM" if one_m else "3DEP best available (1/3 arc-second or better)",
               "native_res_m": 1.0 if one_m else 10.0, "vertical_datum": "NAVD88 (m)"}
        dem_grid = Grid(grid.min_x, grid.max_y, grid.res * grid.cols / rw, rh, rw)
    except (FetchError, OSError, ValueError) as e:
        f.log(f"  terrain: 3DEP failed ({str(e)[:120]}), falling back to Copernicus GLO-30")
        dem, dem_grid = _copernicus(f, bbox, grid)
        src = {**SRC_COP, "product": "Copernicus GLO-30 DSM", "native_res_m": 30.0, "vertical_datum": "EGM2008 (m)"}
    if dem_grid != grid:
        from .geometry import resample

        dem = resample(dem, dem_grid, grid)
    return dem.astype(np.float32), src


def _copernicus(f: Fetcher, bbox: BBox, grid: Grid) -> tuple[np.ndarray, Grid]:
    lat0, lon0 = math.floor(bbox.south), math.floor(bbox.west)
    if math.floor(bbox.north) != lat0 or math.floor(bbox.east) != lon0:
        raise FetchError("bbox crosses a 1-degree Copernicus tile edge (not supported)")
    ns, ew = ("N" if lat0 >= 0 else "S"), ("E" if lon0 >= 0 else "W")
    name = f"Copernicus_DSM_COG_10_{ns}{abs(lat0):02d}_00_{ew}{abs(lon0):03d}_00_DEM"
    tile = _tiff(f.get(COPERNICUS.format(n=name), timeout=180.0))  # 1x1 deg, row 0 = north edge
    th, tw = tile.shape
    origin = GeoOrigin((bbox.south + bbox.north) / 2, (bbox.west + bbox.east) / 2)
    # tile pixel of the grid corners -> crop with a margin, as its own grid in ENU
    r0 = int(math.floor((lat0 + 1 - bbox.north) * th)) - 2
    r1 = int(math.ceil((lat0 + 1 - bbox.south) * th)) + 2
    c0 = int(math.floor((bbox.west - lon0) * tw)) - 2
    c1 = int(math.ceil((bbox.east - lon0) * tw)) + 2
    r0, c0, r1, c1 = max(r0, 0), max(c0, 0), min(r1, th), min(c1, tw)
    crop = tile[r0:r1, c0:c1]
    lat_top, lon_left = lat0 + 1 - r0 / th, lon0 + c0 / tw
    x_left, y_top = origin.to_enu(lat_top, lon_left)
    res_x = origin.m_per_deg_lon / tw
    res_y = origin.m_per_deg_lat / th
    # anisotropic cells: resample onto a square-cell grid first
    g_src = Grid(x_left, y_top, res_y, crop.shape[0], int(round(crop.shape[1] * res_x / res_y)))
    import cv2

    crop = cv2.resize(crop, (g_src.cols, g_src.rows), interpolation=cv2.INTER_LINEAR)
    return crop, g_src


# ------------------------------------------------------------------------------------------------ search


class Nominatim:
    """Polite geocoder proxy: descriptive User-Agent, at most 1 request/s, cached."""

    def __init__(self, f: Fetcher) -> None:
        self.f = f
        self.lock = threading.Lock()
        self.last = 0.0

    def search(self, q: str, limit: int = 5) -> list[dict[str, Any]]:
        params = {"q": q, "format": "jsonv2", "limit": limit}
        key_path = self.f._path(json.dumps([NOMINATIM, params, {}], sort_keys=True))
        if not key_path.exists():
            with self.lock:
                wait = 1.05 - (time.time() - self.last)
                if wait > 0:
                    time.sleep(wait)
                self.last = time.time()
                b = self.f.get(NOMINATIM, params, timeout=20.0, retries=0)
        else:
            b = self.f.get(NOMINATIM, params)
        return [{"name": r.get("display_name", ""), "lat": float(r["lat"]), "lon": float(r["lon"]),
                 "bbox": [float(v) for v in r.get("boundingbox", [])]} for r in json.loads(b)]


@dataclass
class Timed:
    """Result of one layer download, for the build report."""

    value: Any = None
    source: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    error: str = ""
