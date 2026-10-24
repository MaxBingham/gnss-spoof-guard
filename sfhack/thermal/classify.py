"""Material class per 0.5 m pixel from the orthophoto (+ NIR) and OpenStreetMap.

Order (later wins):
 1. spectral base: NDVI (or RGB greenness without NIR) -> vegetation; dark vegetation = tree canopy, bright =
    grass; yellowish non-vegetation = dry grass / soil; other non-vegetation = paved, dark -> asphalt, bright ->
    concrete.
 2. OSM areas: water, parking (paved pixels), pitches.
 3. OSM highways buffered to a width per highway type, material from `surface` (default per type); only where
    the image does not show vegetation (a road under a tree crown is hidden by the crown, a verge is grass).
 4. OSM tree landmarks as crown discs.
 5. building footprints: roof (dark / light by the roof's brightness in the ortho), fountains = water.
Works at any ortho resolution (NAIP 0.6 m or Sentinel-2 10 m resampled to the 0.5 m grid): the spectral
layer just gets blurrier; OSM geometry keeps the roads sharp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..geo import GeoOrigin
from ..missionprep.geometry import Grid
from ..missionprep.osm import _geom, element_rings
from .materials import CLS

VEHICLE_WIDTH = {"motorway": 14.0, "trunk": 12.0, "primary": 11.0, "secondary": 10.0, "tertiary": 8.0,
                 "residential": 7.0, "unclassified": 6.5, "service": 4.5, "living_street": 5.0, "road": 6.0,
                 "busway": 6.0, "motorway_link": 6.0, "trunk_link": 6.0, "primary_link": 6.0,
                 "secondary_link": 6.0, "tertiary_link": 6.0}
PED_WIDTH = {"pedestrian": 6.0, "footway": 2.5, "cycleway": 2.5, "path": 1.8, "steps": 2.5, "track": 3.0,
             "bridleway": 2.0, "platform": 3.0}
MINOR = {"service", "living_street", "footway", "path", "cycleway", "steps", "track", "bridleway"}
SURFACE = {
    "asphalt": "asphalt", "chipseal": "asphalt", "paved": None,
    "concrete": "concrete", "concrete:plates": "concrete", "concrete:lanes": "concrete", "wood": "concrete",
    "metal": "concrete",
    "paving_stones": "pavers", "sett": "pavers", "brick": "pavers", "bricks": "pavers", "cobblestone": "pavers",
    "unhewn_cobblestone": "pavers", "stepping_stones": "pavers",
    "unpaved": "dirt_road", "dirt": "dirt_road", "ground": "dirt_road", "earth": "dirt_road", "gravel": "dirt_road",
    "fine_gravel": "dirt_road", "compacted": "dirt_road", "pebblestone": "dirt_road", "sand": "dirt_road",
    "mud": "dirt_road", "rock": "dirt_road",
    "grass": "grass", "grass_paver": "grass", "artificial_turf": "turf",
}
PAVED = {"asphalt", "asphalt_minor", "concrete", "pavers", "parking"}
ROAD_KIND = {"none": 0, "vehicle": 1, "pedestrian": 2, "parking": 3}


def road_material(tags: dict[str, str]) -> tuple[str, str, float] | None:
    """(material, kind, width_m) of a highway way, None if it is not visible from above."""
    hw = tags.get("highway", "")
    if tags.get("tunnel") in ("yes", "building_passage") or tags.get("covered") == "yes" or hw == "corridor":
        return None
    try:
        if int(tags.get("layer", "0")) < 0:
            return None
    except ValueError:
        pass
    if hw in VEHICLE_WIDTH:
        kind, width = "vehicle", VEHICLE_WIDTH[hw]
    elif hw in PED_WIDTH:
        kind, width = "pedestrian", PED_WIDTH[hw]
    else:
        return None
    try:
        width = float(tags["width"].split()[0]) if "width" in tags else width
    except ValueError:
        pass
    lanes = tags.get("lanes")
    if "width" not in tags and lanes and lanes.isdigit() and kind == "vehicle":
        width = max(width, 3.3 * int(lanes) + 1.0)
    surf = SURFACE.get(tags.get("surface", ""), "")
    if surf == "":  # no or unknown surface tag: default per type
        surf = None
    if surf is None:
        if hw == "track":
            surf = "dirt_road"
        elif hw == "path":
            surf = "dirt_road"
        elif kind == "vehicle":
            surf = "asphalt_minor" if hw in ("service", "living_street") else "asphalt"
        else:
            surf = "concrete"
    if surf == "asphalt" and hw in ("service", "living_street", "footway", "cycleway", "path"):
        surf = "asphalt_minor"
    return surf, kind, float(np.clip(width, 1.0, 30.0))


@dataclass
class ClassMap:
    cls: np.ndarray  # uint8 (rows, cols)
    roads: np.ndarray  # uint8 0/1: paved highway pixel visible from above
    road_kind: np.ndarray  # uint8 ROAD_KIND
    seg_id: np.ndarray  # int32 index into roads_json, -1 elsewhere
    bldg_id: np.ndarray  # int32 landmark id of the roof, -1 elsewhere
    uncertain: np.ndarray  # bool: deep shadow in the ortho (spectral class unreliable)
    roads_json: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def spectral_base(rgb: np.ndarray, nir: np.ndarray | None,
                  chm: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """`chm`: canopy height above ground (m) on the same grid, from LiDAR; without it trees are the dark part
    of the vegetation (weak: NAIP crowns and lawns have the same NDVI and brightness)."""
    f = rgb.astype(np.float32)
    R, G, B = f[..., 0], f[..., 1], f[..., 2]
    V = f.mean(axis=2)
    if nir is not None:
        n = nir.astype(np.float32)
        vi = (n - R) / (n + R + 1e-3)
        veg = vi > 0.12
        idx = "ndvi"
    else:
        vi = (2 * G - R - B) / (R + G + B + 1.0)
        veg = vi > 0.05
        idx = "exg"
    veg = cv2.morphologyEx(veg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    if chm is not None:
        tree = veg & (chm > 2.5)
        tree = cv2.morphologyEx(tree.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    else:
        # crowns are darker than lawns / meadows: Otsu split of the brightness of the vegetated pixels
        # (imagery-relative: Sentinel-2 is much darker than NAIP)
        Vb = cv2.GaussianBlur(V, (0, 0), 1.0)
        thr = 85.0
        if veg.sum() > 1000:
            v8 = np.clip(Vb[veg], 0, 255).astype(np.uint8).reshape(-1, 1)
            thr = float(np.clip(cv2.threshold(v8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0], 30.0, 120.0))
        tree = veg & (Vb < thr)
    tree = cv2.morphologyEx(tree.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    warm_tone = (R - B) > 18.0
    dry = ~veg & warm_tone & (V > 95.0) & ((vi > -0.02) if nir is not None else True)
    cls = np.full(rgb.shape[:2], CLS["concrete"], np.uint8)
    cls[V < 105.0] = CLS["asphalt_minor"]
    cls[dry] = CLS["dry_grass"]
    cls[veg] = CLS["grass"]
    cls[tree] = CLS["tree"]
    uncertain = (V < 40.0) & ~veg
    stats = {"index": idx, "canopy": "lidar" if chm is not None else "spectral (+ OSM wood / meadow)", "veg_frac": round(float(veg.mean()), 3), "tree_frac": round(float(tree.mean()), 3),
             "dry_frac": round(float(dry.mean()), 3), "shadow_frac": round(float(uncertain.mean()), 3)}
    return cls, uncertain, {**stats, "veg_mask": veg}


def classify(rgb: np.ndarray, nir: np.ndarray | None, grid: Grid, osm: dict[str, Any] | None, origin: GeoOrigin,
             landmarks: list[dict[str, Any]], drop_minor_frac: float = 0.0,
             rng: np.random.Generator | None = None, chm: np.ndarray | None = None,
             untagged_paved_p: float | None = None, truth: bool = False, coarse: bool = False) -> ClassMap:
    """`untagged_paved_p`: outside the US, minor roads without a `surface` tag are often dirt (Ukrainian village
    lanes): the truth decides per way (paved with this probability), the prior marks them dirt + uncertain and
    keeps them out of the `roads` mask. `coarse`: imagery coarser than ~2 m (Sentinel-2): mixed pixels look
    vegetated next to every narrow road, so the image does not get to hide OSM roads (only tree landmarks do)."""
    rng = rng or np.random.default_rng(0)
    cls, uncertain, st = spectral_base(rgb, nir, chm)
    veg = st.pop("veg_mask")
    if coarse:
        veg_for_roads = np.zeros_like(veg)
    else:
        veg_for_roads = veg
    shape = grid.shape
    roads = np.zeros(shape, np.uint8)
    road_kind = np.zeros(shape, np.uint8)
    seg_id = np.full(shape, -1, np.int32)
    bldg_id = np.full(shape, -1, np.int32)
    roads_json: list[dict[str, Any]] = []

    def window(pts: np.ndarray, pad_m: float) -> tuple[tuple[slice, slice], np.ndarray] | None:
        """Sub-window of the grid holding pts (+ pad) and the points in its 1/16 px coordinates (drawing
        each feature on a full 8 Mpx raster made the Sentinel-2 mission slow)."""
        uv = grid.to_px(pts)
        pad = pad_m / grid.res + 2
        c0, r0 = np.floor(uv.min(axis=0) - pad).astype(int)
        c1, r1 = np.ceil(uv.max(axis=0) + pad).astype(int) + 1
        c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1, grid.cols), min(r1, grid.rows)
        if c1 <= c0 or r1 <= r0:
            return None
        return (slice(r0, r1), slice(c0, c1)), np.round((uv - [c0, r0] + 0.5) * 16).astype(np.int32)

    def poly_mask(pts: np.ndarray) -> tuple[tuple[slice, slice], np.ndarray] | None:
        w = window(pts, 0.0)
        if w is None:
            return None
        sl, p16 = w
        m = np.zeros((sl[0].stop - sl[0].start, sl[1].stop - sl[1].start), np.uint8)
        cv2.fillPoly(m, [p16], 1, lineType=cv2.LINE_8, shift=4)
        return sl, m > 0

    elements = (osm or {}).get("elements", [])
    # 2. areas
    for el in elements:
        tags = el.get("tags", {})
        if el["type"] == "node" or "highway" in tags or "building" in tags:
            continue
        target = None
        if tags.get("natural") in ("wood",) or tags.get("landuse") == "forest":
            target = "wood"
        elif chm is None and (tags.get("landuse") in ("meadow", "grass", "farmland", "village_green", "recreation_ground")
                              or tags.get("natural") in ("grassland", "heath") or tags.get("leisure") in ("park", "pitch")):
            target = "meadow"
        elif tags.get("natural") == "water" or tags.get("water") or tags.get("landuse") in ("basin", "reservoir"):
            target = "water"
        elif tags.get("amenity") == "parking" and tags.get("parking", "surface") in ("surface", ""):
            target = "parking"
        elif tags.get("leisure") == "pitch" and tags.get("surface") == "artificial_turf":
            target = "turf"
        if target is None:
            continue
        for ring in element_rings(el):
            pts = np.array([origin.to_enu(a, b) for a, b in ring])
            if len(pts) < 3:
                continue
            pm = poly_mask(pts)
            if pm is None:
                continue
            sl, m = pm
            c, v = cls[sl], veg[sl]
            if target == "wood":
                c[m & v] = CLS["tree"]
            elif target == "meadow":
                c[m & v & (c == CLS["tree"])] = CLS["grass"]
            elif target == "parking":
                sel = m & ~v
                c[sel] = CLS["parking"]
                road_kind[sl][sel] = ROAD_KIND["parking"]
            else:
                c[m] = CLS[target]
    # 3. highways, wide first so narrow paths on top keep their own id
    ways = []
    for el in elements:
        tags = el.get("tags", {})
        if el["type"] != "way" or "highway" not in tags:
            continue
        rm = road_material(tags)
        if rm is None:
            continue
        pts = np.array([origin.to_enu(a, b) for a, b in _geom(el.get("geometry", []))])
        if len(pts) < 2:
            continue
        hw = tags["highway"]
        if drop_minor_frac > 0 and hw in MINOR and rng.uniform() < drop_minor_frac:
            continue  # OSM is incomplete: the onboard copy misses some minor ways
        ways.append((rm[2], el, tags, pts, rm))
    ways.sort(key=lambda w: -w[0])
    n_unc = 0
    for width, el, tags, pts, (mat, kind, _) in ways:
        sid = len(roads_json)
        unsure = (untagged_paved_p is not None and "surface" not in tags
                  and tags["highway"] in ("residential", "unclassified", "service", "living_street", "road"))
        if unsure:
            n_unc += 1
            if truth:  # per way, the same answer for every build
                paved = np.random.default_rng(int(el["id"]) % 2**32).uniform() < untagged_paved_p
                mat = "asphalt_minor" if paved else "dirt_road"
            else:
                mat = "dirt_road"
        roads_json.append({"id": sid, "osm_id": f"way/{el['id']}", "highway": tags["highway"],
                           "surface": tags.get("surface"), "material": mat, "kind": kind, "width_m": width,
                           "paved": None if unsure and not truth else mat in PAVED,
                           "pts_enu": np.round(pts, 2).tolist()})
        w = window(pts, width)
        if w is None:
            continue
        sl, p16 = w
        m = np.zeros((sl[0].stop - sl[0].start, sl[1].stop - sl[1].start), np.uint8)
        area = tags.get("area") == "yes" and len(pts) >= 4 and np.allclose(pts[0], pts[-1])
        if area:
            cv2.fillPoly(m, [p16], 1, lineType=cv2.LINE_8, shift=4)
        else:
            th = max(1, int(round(width / grid.res)))
            cv2.polylines(m, [p16], False, 1, thickness=th, lineType=cv2.LINE_8, shift=4)
        sel = (m > 0) & ~veg_for_roads[sl]
        if unsure and not truth:
            uncertain[sl][sel] = True
        cls[sl][sel] = CLS[mat]
        seg_id[sl][sel] = sid
        road_kind[sl][sel] = ROAD_KIND[kind]
        roads[sl][sel] = 1 if mat in PAVED else 0
    # 4. trees from the landmark list (crown discs)
    for lm in landmarks:
        if lm["type"] != "tree":
            continue
        rad = float(lm.get("radius", 3.0))
        w = window(np.asarray([lm["centroid_enu"]], float), rad)
        if w is None:
            continue
        sl, p16 = w
        tm = np.zeros((sl[0].stop - sl[0].start, sl[1].stop - sl[1].start), np.uint8)
        cv2.circle(tm, (int(p16[0, 0]), int(p16[0, 1])), int(round(rad / grid.res * 16)), 1, -1,
                   lineType=cv2.LINE_8, shift=4)
        cls[sl][tm > 0] = CLS["tree"]
    # 5. roofs
    V = rgb.astype(np.float32).mean(axis=2)
    for lm in landmarks:
        if lm["type"] not in ("building", "tower", "fountain") or not lm.get("footprint_enu"):
            continue
        pm = poly_mask(np.asarray(lm["footprint_enu"], float))
        if pm is None or not pm[1].any():
            continue
        sl, m = pm
        if lm["type"] == "fountain":
            cls[sl][m] = CLS["water"]
            continue
        bright = float(np.median(V[sl][m]))
        cls[sl][m] = CLS["roof_light"] if bright > 150.0 else CLS["roof_dark"]
        bldg_id[sl][m] = lm["id"]
        roads[sl][m] = 0
        road_kind[sl][m] = 0
        seg_id[sl][m] = -1
    hidden = cls == CLS["tree"]
    roads[hidden] = 0
    seg_id[hidden] = -1
    road_kind[hidden] = 0
    st.update({"road_frac": round(float(roads.mean()), 4), "n_ways": len(roads_json), "n_ways_surface_unknown": n_unc,
               "class_frac": {k: round(float((cls == v).mean()), 4) for k, v in CLS.items()}})
    return ClassMap(cls, roads, road_kind, seg_id, bldg_id, uncertain, roads_json, st)
