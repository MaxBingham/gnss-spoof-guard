"""Sentinel-2 L2A imagery for missions outside the US (no NAIP there): RGB + NIR + NDVI at 10 m.

Scenes come from the open Earth Search STAC (Element84, AWS open data, no account). The bands are
Cloud-Optimized GeoTIFFs in UTM; only the 1024 x 1024 tiles that cover the bbox are read, by HTTP range
requests through the cached Fetcher (tifffile parses the header, so no GDAL / rasterio is needed).
The scene is picked by the cloud / shadow fraction INSIDE the bbox (scene classification layer), then by the
scene cloud cover, newest first, over the summers in SUMMERS.

Output grids are the mission's north-up ENU grids (0.5 m ortho grid: a 10 m pixel covers 20 x 20 cells,
resampled bicubic for RGB and bilinear for NIR / NDVI).
"""

from __future__ import annotations

import io
import json
import math
from typing import Any

import cv2
import numpy as np

from ..geo import BBox, GeoOrigin
from .geometry import Grid
from .sources import FetchError, Fetcher

STAC_SEARCH = "https://earth-search.aws.element84.com/v1/search"
COLLECTION = "sentinel-2-l2a"
SUMMERS = (2025, 2024, 2023)  # June - August of each, newest first
MAX_SCENE_CLOUD = 10.0  # % (whole tile) prefilter
MAX_LOCAL_CLOUD = 0.01  # cloud + shadow + nodata fraction inside the bbox
SCL_BAD = (0, 1, 3, 8, 9, 10)  # nodata, saturated, cloud shadow, cloud medium / high, cirrus
BOA_OFFSET = 1000  # processing baseline >= 04.00: reflectance = (DN - 1000) / 10000 (if STAC gives no scale)
NIR_PNG_SCALE = 425.0  # nir.png value = NIR reflectance * 425 (255 = 0.6)

SRC_S2 = {"name": "Copernicus Sentinel-2 L2A (Earth Search STAC, AWS open data)",
          "license": "Copernicus Sentinel data terms (free, full and open)",
          "attribution": "Contains modified Copernicus Sentinel data, processed by ESA; COGs by Element84 / AWS",
          "url": "https://registry.opendata.aws/sentinel-2-l2a-cogs/"}


# ------------------------------------------------------------------------------------------------ UTM


def utm_zone(lon: float) -> int:
    return int((lon + 180.0) // 6.0) + 1


def latlon_to_utm(lat: np.ndarray, lon: np.ndarray, zone: int, north: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """WGS84 -> UTM easting / northing (m), Krueger series to n^3 (mm accuracy within the zone)."""
    a, f = 6378137.0, 1 / 298.257223563
    n = f / (2 - f)
    A = a / (1 + n) * (1 + n * n / 4 + n ** 4 / 64)
    al = (n / 2 - 2 * n ** 2 / 3 + 5 * n ** 3 / 16, 13 * n ** 2 / 48 - 3 * n ** 3 / 5, 61 * n ** 3 / 240)
    k0, e = 0.9996, math.sqrt(f * (2 - f))
    phi, lam = np.radians(lat), np.radians(lon) - math.radians(zone * 6 - 183)
    t = np.sinh(np.arctanh(np.sin(phi)) - e * np.arctanh(e * np.sin(phi)))
    xi, eta = np.arctan2(t, np.cos(lam)), np.arctanh(np.sin(lam) / np.sqrt(1 + t * t))
    E, N = eta.copy(), xi.copy()
    for j, aj in enumerate(al, start=1):
        E += aj * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
        N += aj * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
    return 500000.0 + k0 * A * E, (0.0 if north else 10000000.0) + k0 * A * N


# ------------------------------------------------------------------------------------------------ COG reader


def _remote_size(f: Fetcher, url: str) -> int:
    """Content length by HEAD, cached next to the range blocks (offline rebuilds need it too)."""
    path = f._path(json.dumps([url, {"head": "content-length"}, {}], sort_keys=True))
    if path.exists():
        return int(path.read_text())
    if f.offline:
        raise FetchError(f"offline and not cached: HEAD {url}")
    try:
        r = f.client.head(url, timeout=30.0)
        r.raise_for_status()
        n = int(r.headers["content-length"])
    except Exception as e:  # noqa: BLE001 - httpx errors, missing header
        raise FetchError(f"HEAD {url}: {e!r}") from e
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(n))
    return n


class _RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests, in cached 64 KiB blocks (for the TIFF header)."""

    BLOCK = 1 << 16

    def __init__(self, f: Fetcher, url: str) -> None:
        self.f, self.url, self.pos = f, url, 0
        self.blocks: dict[int, bytes] = {}
        self.size = _remote_size(f, url)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, b: Any) -> int:
        n, out = max(0, min(len(b), self.size - self.pos)), bytearray()
        while len(out) < n:
            k, o = divmod(self.pos + len(out), self.BLOCK)
            if k not in self.blocks:
                self.blocks[k] = self.f.get_range(self.url, k * self.BLOCK, (k + 1) * self.BLOCK - 1)
            chunk = self.blocks[k][o:o + n - len(out)]
            if not chunk:
                break
            out += chunk
        b[:len(out)] = out
        self.pos += len(out)
        return len(out)


class Cog:
    """One single-resolution COG band: geotransform + windowed tile reads."""

    def __init__(self, f: Fetcher, url: str) -> None:
        import tifffile

        self.f, self.url = f, url
        self.tif = tifffile.TiffFile(io.BufferedReader(_RangeFile(f, url), 1 << 16))
        p = self.page = self.tif.pages[0]
        sx, sy = p.tags["ModelPixelScaleTag"].value[:2]
        tp = p.tags["ModelTiepointTag"].value
        self.x0, self.y0, self.sx, self.sy = tp[3] - tp[0] * sx, tp[4] + tp[1] * sy, sx, sy
        self.h, self.w = p.shape[:2]
        self.tw, self.th = p.tilewidth, p.tilelength

    def read(self, c0: int, r0: int, c1: int, r1: int) -> np.ndarray:
        """Pixels [r0:r1, c0:c1] (clipped to the image; outside = 0)."""
        p = self.page
        spp = p.samplesperpixel
        out = np.zeros((r1 - r0, c1 - c0, spp), p.dtype)
        across = math.ceil(self.w / self.tw)
        for tr in range(max(r0, 0) // self.th, (min(r1, self.h) - 1) // self.th + 1):
            for tc in range(max(c0, 0) // self.tw, (min(c1, self.w) - 1) // self.tw + 1):
                i = tr * across + tc
                off, cnt = int(p.dataoffsets[i]), int(p.databytecounts[i])
                seg, _, _ = p.decode(self.f.get_range(self.url, off, off + cnt - 1), i)
                seg = np.asarray(seg).reshape(self.th, self.tw, spp)
                ty, tx = tr * self.th, tc * self.tw
                ya, yb = max(r0, ty), min(r1, ty + self.th)
                xa, xb = max(c0, tx), min(c1, tx + self.tw)
                out[ya - r0:yb - r0, xa - c0:xb - c0] = seg[ya - ty:yb - ty, xa - tx:xb - tx]
        return out


def _band_on_grid(cog: Cog, grid: Grid, origin: GeoOrigin, zone: int, interp: int) -> np.ndarray:
    """Resample a UTM band onto a north-up ENU grid (float32, one array per sample)."""
    X, Y = grid.centers()
    lat, lon = origin.lat + Y / origin.m_per_deg_lat, origin.lon + X / origin.m_per_deg_lon
    E, N = latlon_to_utm(lat, lon, zone)
    u = (E - cog.x0) / cog.sx - 0.5  # continuous pixel coords, centre = integer
    v = (cog.y0 - N) / cog.sy - 0.5
    c0, r0 = int(math.floor(u.min())) - 3, int(math.floor(v.min())) - 3
    c1, r1 = int(math.ceil(u.max())) + 4, int(math.ceil(v.max())) + 4
    win = cog.read(c0, r0, c1, r1).astype(np.float32)
    mu, mv = (u - c0).astype(np.float32), (v - r0).astype(np.float32)
    return np.dstack([cv2.remap(win[..., k], mu, mv, interp, borderMode=cv2.BORDER_REPLICATE)
                      for k in range(win.shape[2])])


# ------------------------------------------------------------------------------------------------ scenes


def search_scenes(f: Fetcher, bbox: BBox) -> list[dict[str, Any]]:
    """Summer L2A items covering the bbox center, scene cloud cover <= MAX_SCENE_CLOUD (cached GETs)."""
    items = []
    for year in SUMMERS:
        params = {"collections": COLLECTION, "bbox": f"{bbox.west},{bbox.south},{bbox.east},{bbox.north}",
                  "datetime": f"{year}-06-01T00:00:00Z/{year}-08-31T23:59:59Z", "limit": 100}
        try:
            d = json.loads(f.get(STAC_SEARCH, params, timeout=60.0))
        except (FetchError, ValueError) as e:
            f.log(f"  sentinel-2: STAC search {year} failed ({str(e)[:120]})")
            continue
        for it in d.get("features", []):
            p = it.get("properties", {})
            if p.get("eo:cloud_cover", 100.0) <= MAX_SCENE_CLOUD:
                items.append(it)
    # cloud cover ascending (rounded to 1 %), newest first among equal cover (stable sorts)
    items.sort(key=lambda it: it["properties"]["datetime"], reverse=True)
    items.sort(key=lambda it: round(it["properties"]["eo:cloud_cover"]))
    return items


def _scale_offset(item: dict[str, Any], asset: str) -> tuple[float, float]:
    """DN -> BOA reflectance: the asset's STAC raster:bands scale / offset (Earth Search v1 says 1e-4 / -0.1
    for baseline >= 04.00), else the ESA convention by processing baseline."""
    rb = (item["assets"][asset].get("raster:bands") or [{}])[0]
    if "scale" in rb:
        return float(rb["scale"]), float(rb.get("offset", 0.0))
    try:
        new = float(item["properties"].get("s2:processing_baseline", "0")) >= 4.0
    except ValueError:
        new = False
    return 1e-4, (-BOA_OFFSET * 1e-4 if new else 0.0)


def pick_scene(f: Fetcher, bbox: BBox, origin: GeoOrigin, max_tries: int = 8) -> tuple[dict[str, Any], float]:
    """First candidate whose scene classification is clear inside the bbox (checked on a 20 m grid)."""
    items = search_scenes(f, bbox)
    if not items:
        raise FetchError("no Sentinel-2 L2A summer scene covers this area")
    hw = (bbox.east - bbox.west) / 2 * origin.m_per_deg_lon
    hh = (bbox.north - bbox.south) / 2 * origin.m_per_deg_lat
    g20 = Grid(-hw, hh, 20.0, max(2, int(2 * hh / 20)), max(2, int(2 * hw / 20)))
    best: tuple[float, dict[str, Any]] | None = None
    for it in items[:max_tries]:
        try:
            scl = _band_on_grid(Cog(f, it["assets"]["scl"]["href"]), g20, origin, _zone(it), cv2.INTER_NEAREST)[..., 0]
        except (FetchError, KeyError, OSError, ValueError) as e:
            f.log(f"  sentinel-2: {it['id']} SCL unreadable ({str(e)[:100]})")
            continue
        bad = float(np.isin(np.round(scl), SCL_BAD).mean())
        if best is None or bad < best[0]:
            best = (bad, it)
        if bad <= MAX_LOCAL_CLOUD:
            break
    if best is None:
        raise FetchError("no readable Sentinel-2 scene")
    return best[1], best[0]


def _zone(item: dict[str, Any]) -> int:
    p = item["properties"]
    code = p.get("proj:epsg") or int(str(p.get("proj:code", "EPSG:0")).split(":")[-1])
    code = int(code)
    if not (32601 <= code <= 32660):
        raise FetchError(f"{item['id']}: not a northern UTM tile (EPSG {code})")
    return code - 32600


def fetch_s2(f: Fetcher, bbox: BBox, grid: Grid) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """RGB (true colour, the ESA TCI product), NIR png (reflectance x 425) and NDVI png ((ndvi + 1) x 127.5)
    on `grid`, all uint8, plus the source block for the manifest."""
    origin = GeoOrigin((bbox.south + bbox.north) / 2, (bbox.west + bbox.east) / 2)
    item, bad = pick_scene(f, bbox, origin)
    zone = _zone(item)
    a = item["assets"]
    f.log(f"  sentinel-2: scene {item['id']} ({item['properties']['datetime'][:10]}, scene cloud "
          f"{item['properties']['eo:cloud_cover']:.2f} %, {bad * 100:.2f} % cloud/shadow in the bbox)")
    rgb = np.clip(_band_on_grid(Cog(f, a["visual"]["href"]), grid, origin, zone, cv2.INTER_CUBIC), 0, 255)

    def refl(asset: str) -> np.ndarray:
        s, o = _scale_offset(item, asset)
        dn = _band_on_grid(Cog(f, a[asset]["href"]), grid, origin, zone, cv2.INTER_LINEAR)[..., 0]
        return np.maximum(dn * s + o, 0.0)

    red, nir = refl("red"), refl("nir")
    off = _scale_offset(item, "nir")[1]
    ndvi = (nir - red) / np.maximum(nir + red, 1e-4)
    src = {**SRC_S2, "scene": item["id"], "date": item["properties"]["datetime"][:10],
           "scene_cloud_pct": round(float(item["properties"]["eo:cloud_cover"]), 3),
           "bbox_cloud_shadow_frac": round(bad, 4), "utm_zone": zone, "native_res_m": 10.0,
           "reflectance_offset": off, "nir_png": f"NIR (B08) reflectance x {NIR_PNG_SCALE:g}",
           "ndvi_png": "(NDVI + 1) x 127.5 from B04 / B08 reflectance"}
    return (rgb.astype(np.uint8), np.clip(nir * NIR_PNG_SCALE + 0.5, 0, 255).astype(np.uint8),
            np.clip((ndvi + 1) * 127.5 + 0.5, 0, 255).astype(np.uint8), src)
