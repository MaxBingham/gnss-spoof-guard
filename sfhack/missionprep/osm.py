"""OpenStreetMap elements -> landmarks (buildings, towers, fountains, trees) in local ENU.

Ported from the V1 prototype (gnss-free-nav backend/app/map_builder.py) that produced
sfhack/world/sample_berkeley, so mission packages carry the same landmark schema:
  {id, osm_id, type, name?, centroid_enu, footprint_enu? (CCW), height, radius, area_m2?}
plus `height_source` ("tag" | "levels" | "lidar" | "default") and `base_z` (filled in by the builder).
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ..geo import BBox, GeoOrigin

DEFAULT_BUILDING_HEIGHT = 10.0
LEVEL_HEIGHT = 3.2
TREE_HEIGHT = 12.0
TREE_RADIUS = 4.0
TREE_ROW_SPACING = 8.0
TOWER_DEFAULT_HEIGHT = 30.0
FOUNTAIN_HEIGHT = 2.0
FOUNTAIN_RADIUS = 3.0


def parse_length(value: str | None) -> float | None:
    if not value:
        return None
    m = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)", value.replace(",", "."))
    if not m:
        return None
    num = float(m.group(1))
    if "'" in value or "ft" in value:
        num *= 0.3048
    return num


def building_height(tags: dict[str, str]) -> tuple[float, str]:
    h = parse_length(tags.get("height"))
    if h and h > 0:
        return h, "tag"
    levels = parse_length(tags.get("building:levels"))
    if levels and levels > 0:
        return levels * LEVEL_HEIGHT, "levels"
    return DEFAULT_BUILDING_HEIGHT, "default"


def polygon_area_centroid(pts: list[tuple[float, float]]) -> tuple[float, tuple[float, float]]:
    a = cx = cy = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        a += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    a *= 0.5
    if abs(a) < 1e-9:
        return 0.0, (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)
    return a, (cx / (6 * a), cy / (6 * a))


def assemble_rings(segments: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    """Join open way segments of a multipolygon relation into closed rings."""
    rings: list[list[tuple[float, float]]] = []
    pending = [s[:] for s in segments if len(s) >= 2]
    while pending:
        ring = pending.pop(0)
        changed = True
        while ring[0] != ring[-1] and changed:
            changed = False
            for i, seg in enumerate(pending):
                if seg[0] == ring[-1]:
                    ring.extend(seg[1:])
                elif seg[-1] == ring[-1]:
                    ring.extend(reversed(seg[:-1]))
                elif seg[-1] == ring[0]:
                    ring = seg[:-1] + ring
                elif seg[0] == ring[0]:
                    ring = list(reversed(seg[1:])) + ring
                else:
                    continue
                pending.pop(i)
                changed = True
                break
        if len(ring) >= 4:
            rings.append(ring)
    return rings


def _geom(geom: list[dict[str, float]]) -> list[tuple[float, float]]:
    return [(g["lat"], g["lon"]) for g in geom if g is not None]


def element_rings(el: dict[str, Any]) -> list[list[tuple[float, float]]]:
    """Outer rings as lat/lon lists."""
    if el["type"] == "way" and "geometry" in el:
        return [_geom(el["geometry"])]
    if el["type"] == "relation":
        outers = [_geom(m["geometry"]) for m in el.get("members", [])
                  if m.get("type") == "way" and m.get("role", "outer") in ("outer", "") and m.get("geometry")]
        return assemble_rings(outers)
    return []


def r2(v: float) -> float:
    return round(float(v), 2)


def extract_landmarks(osm: dict[str, Any], origin: GeoOrigin, bbox: BBox) -> list[dict[str, Any]]:
    landmarks: list[dict[str, Any]] = []
    seen_trees: set[tuple[int, int]] = set()

    def inside(lat: float, lon: float) -> bool:
        return bbox.south <= lat <= bbox.north and bbox.west <= lon <= bbox.east

    def add_tree(lat: float, lon: float, osm_id: str, tags: dict[str, str]) -> None:
        if not inside(lat, lon):
            return
        x, y = origin.to_enu(lat, lon)
        key = (round(x), round(y))
        if key in seen_trees:
            return
        seen_trees.add(key)
        h = parse_length(tags.get("height")) or TREE_HEIGHT
        crown = parse_length(tags.get("diameter_crown"))
        landmarks.append({"osm_id": osm_id, "type": "tree", "centroid_enu": [r2(x), r2(y)],
                          "height": r2(min(max(h, 3.0), 40.0)), "radius": r2(crown / 2 if crown else TREE_RADIUS),
                          "height_source": "tag" if tags.get("height") else "default"})

    for el in osm["elements"]:
        tags: dict[str, str] = el.get("tags", {})
        osm_id = f"{el['type']}/{el['id']}"
        if el["type"] == "node":
            lat, lon = el["lat"], el["lon"]
            if tags.get("natural") == "tree":
                add_tree(lat, lon, osm_id, tags)
            elif tags.get("man_made") == "tower" or tags.get("amenity") == "fountain":
                if not inside(lat, lon):
                    continue
                x, y = origin.to_enu(lat, lon)
                is_tower = tags.get("man_made") == "tower"
                h = parse_length(tags.get("height"))
                landmarks.append({"osm_id": osm_id, "type": "tower" if is_tower else "fountain", "name": tags.get("name"),
                                  "centroid_enu": [r2(x), r2(y)],
                                  "height": r2(h or (TOWER_DEFAULT_HEIGHT if is_tower else FOUNTAIN_HEIGHT)),
                                  "radius": 4.0 if is_tower else FOUNTAIN_RADIUS,
                                  "height_source": "tag" if h else "default"})
            continue
        if tags.get("natural") == "tree_row" and el["type"] == "way":
            enu = [origin.to_enu(lat, lon) for lat, lon in _geom(el.get("geometry", []))]
            for (x0, y0), (x1, y1) in zip(enu, enu[1:]):
                n = max(1, int(math.hypot(x1 - x0, y1 - y0) // TREE_ROW_SPACING))
                for k in range(n):
                    lat, lon = origin.to_latlon(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n)
                    add_tree(lat, lon, osm_id, tags)
            continue
        is_building = "building" in tags and tags.get("building") != "no"
        is_tower = tags.get("man_made") == "tower"
        is_fountain = tags.get("amenity") == "fountain"
        if not (is_building or is_tower or is_fountain):
            continue
        for ring in element_rings(el):
            if len(ring) < 4:
                continue
            clat = sum(p[0] for p in ring) / len(ring)
            clon = sum(p[1] for p in ring) / len(ring)
            if not inside(clat, clon):
                continue
            enu = [origin.to_enu(lat, lon) for lat, lon in ring]
            if enu[0] == enu[-1]:
                enu = enu[:-1]
            area, (cx, cy) = polygon_area_centroid(enu)
            if area < 0:  # store counter-clockwise footprints
                enu.reverse()
                area = -area
            if is_tower:
                kind = "tower"
                h = parse_length(tags.get("height"))
                height, hsrc = (h, "tag") if h else (TOWER_DEFAULT_HEIGHT, "default")
            elif is_fountain and not is_building:
                kind, height, hsrc = "fountain", FOUNTAIN_HEIGHT, "default"
            else:
                kind = "building"
                height, hsrc = building_height(tags)
            radius = max(math.hypot(x - cx, y - cy) for x, y in enu)
            landmarks.append({"osm_id": osm_id, "type": kind, "name": tags.get("name"), "centroid_enu": [r2(cx), r2(cy)],
                              "footprint_enu": [[r2(x), r2(y)] for x, y in enu], "height": r2(height),
                              "radius": r2(radius), "area_m2": r2(area), "height_source": hsrc})

    order = {"tower": 0, "fountain": 1, "building": 2, "tree": 3}
    landmarks.sort(key=lambda lm: (order[lm["type"]], lm["osm_id"]))
    for i, lm in enumerate(landmarks):
        lm["id"] = i
        if lm.get("name") is None:
            lm.pop("name", None)
    return landmarks


GROUND_STYLE: list[tuple[str, tuple[int, int, int]]] = [
    ("grass", (96, 130, 78)), ("wood", (62, 96, 58)), ("water", (70, 110, 150)),
    ("paved", (128, 128, 124)), ("road", (105, 105, 102)), ("building", (70, 70, 74)),
]


def render_osm_ground(osm: dict[str, Any], origin: GeoOrigin, extent: dict[str, float], size: tuple[int, int]) -> np.ndarray:
    """Stylized ground texture from OSM polygons (last-resort fallback when no imagery can be downloaded)."""
    w, h = size
    mx = (extent["max_x"] - extent["min_x"]) / w
    my = (extent["max_y"] - extent["min_y"]) / h
    img = Image.new("RGB", (w, h), (150, 145, 125))
    draw = ImageDraw.Draw(img)

    def px(lat: float, lon: float) -> tuple[float, float]:
        x, y = origin.to_enu(lat, lon)
        return (x - extent["min_x"]) / mx, (extent["max_y"] - y) / my

    layers: dict[str, list[list[tuple[float, float]]]] = {k: [] for k, _ in GROUND_STYLE}
    roads: list[tuple[list[tuple[float, float]], float]] = []
    for el in osm["elements"]:
        tags: dict[str, str] = el.get("tags", {})
        if el["type"] == "node":
            continue
        if "highway" in tags and el["type"] == "way":
            hw = tags["highway"]
            width = 12.0 if hw in ("primary", "secondary", "tertiary") else 8.0 if hw in ("residential", "service") else 3.0
            roads.append(([px(a, b) for a, b in _geom(el.get("geometry", []))], width / mx))
            continue
        layer = ("building" if "building" in tags else "water" if tags.get("natural") == "water"
                 else "wood" if tags.get("natural") in ("wood", "scrub") or tags.get("landuse") == "forest"
                 else "paved" if tags.get("amenity") == "parking"
                 else "grass" if tags.get("leisure") or tags.get("landuse") in ("grass", "meadow", "recreation_ground")
                 else None)
        if layer:
            for ring in element_rings(el):
                layers[layer].append([px(a, b) for a, b in ring])
    for name, color in GROUND_STYLE:
        if name == "road":
            for pts, width in roads:
                if len(pts) >= 2:
                    draw.line(pts, fill=color, width=max(1, int(width)))
            continue
        for poly in layers[name]:
            if len(poly) >= 3:
                draw.polygon(poly, fill=color)
    return np.asarray(img)
