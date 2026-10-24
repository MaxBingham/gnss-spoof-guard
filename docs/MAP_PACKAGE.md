# Mission package format (`sfhack-mission/1`)

The contract between the mission planner (`sfhack/missionprep`), the simulator, the onboard stack
(V2 vision localizer, V3.2 LiDAR / KISS-ICP / TRN) and the object placer. Everything here can be read
without the planner's code: JSON, JPEG, NumPy `.npz` and binary PLY only.

```
$SFHACK_DATA/missions/<id>/          <id>: [A-Za-z0-9][A-Za-z0-9_-]{0,63}
  build_report.json                  inputs, per-layer sources / seconds / notes, file sizes
  viewer.log                         output of the last live viewer the planner page launched
  world/                             TRUTH: what the simulator builds MuJoCo from (onboard code never reads it)
    manifest.json
    landmarks.json
    ortho.jpg
    terrain.npz
    target_scene.json                objects in the search area around B (object placer writes here)
    osm.json                         raw OSM download (Overpass `out geom` shape): roads + surface, landuse, ... (thermal model)
    nir.png, ndvi.png                near infrared + NDVI on the ortho grid (NAIP band 4 / Sentinel-2 B08, B04), if available
    thermal_truth.npz                night thermal truth (sfhack.thermal build, night-thermal), if built
  onboard/                           the package "uploaded to the drone": a deliberately imperfect copy
    manifest.json                    index: origin, frame, layers with bytes + SHA-256, sources, licenses
    mission.json                     A, B, search radius, route, target_scene (search area only)
    ortho.jpg                        vision localizer
    landmarks.json                   vision / mock localizer, display
    dsm.npz                          DTM + DSM rasters (TRN, terrain-aware vision, route)
    map_cloud.ply                    3D point cloud (LiDAR scan-to-map)
    nir.png, ndvi.png                copies of the world bands (0.5 px blur), if available
    thermal_prior.npz                predicted night temperature + road mask (sfhack.thermal build), if built
  drone/                             onboard/ after the simulated upload, every file SHA-256 verified
    ...same files...  upload_receipt.json
```

Make one:

```
. scripts/env.ps1                                                    # (bash: . scripts/env.sh)
uv run python -m sfhack.missionprep serve                            # browser: http://127.0.0.1:8765/ (Prepare = build,
                                                                     # upload and launch the viewer in one click)
uv run python -m sfhack.missionprep build --id berkeley              # default A / B, no browser
uv run python -m sfhack.missionprep build --a 37.8716 -122.2638 --b 37.87215 -122.2566 --radius 40 --id campus-glade
uv run python -m sfhack.missionprep build --a 49.2244 23.3220 --b 49.2305 23.3456 --id ukraine-1 --agl 60  # outside the US
uv run python -m sfhack.missionprep upload berkeley --mbps 2         # simulated link + checksum check
uv run python demos/demo_live.py --mission berkeley                  # fly it (viewer; --headless to score)
```

In code: `FlightSystem(mission="berkeley")` builds the world from `world/` and the onboard map plus route
from `drone/` (if a verified upload exists) else `onboard/`. `OnboardMap(<mission>/drone)` loads the
package as the drone sees it (`.dtm`, `.dsm`, `.dem_grid`, `.mission`, `ground_z(xy)`, `surface_z(xy)`).

## Frames, units, datums

| What | Convention |
| --- | --- |
| Horizontal | Local ENU in metres: x = east, y = north. Origin = midpoint of A and B (`manifest.origin` lat / lon, WGS84). Equirectangular: `x = (lon - lon0) * m_per_deg_lon`, `y = (lat - lat0) * m_per_deg_lat` with `m_per_deg_lat = pi/180 * 6378137`, `m_per_deg_lon = m_per_deg_lat * cos(lat0)` (`sfhack/geo.py`, error < 0.1 m over 2 km). |
| Vertical | z up, metres. **z = 0 is the ground under A** (`manifest.z0_elevation_m` above the DEM's vertical datum, `manifest.z_datum`). Every z in every file uses this datum: terrain, roofs, route, point cloud, B's `ground_z`. |
| Yaw | 0 = +x (east), counter-clockwise positive, radians in code (`yaw_deg` in degrees where named). |
| Extent | `extent_enu` = A/B bounding box + 150 m margin (at least 400 m a side), symmetric about the origin, half sizes rounded up to 2 m so every raster tiles it exactly. `bbox` is the same rectangle in lat / lon (what was downloaded). |
| Rasters | North-up, `array[row, col]`, **row 0 = north edge**, cell (row, col) centred at `x = min_x + (col + 0.5) * res`, `y = max_y - (row + 0.5) * res`. Same for `ortho.jpg`, `terrain.npz`, `dsm.npz`. |
| Polygons | `footprint_enu`: list of `[x, y]`, counter-clockwise, not closed (first point not repeated). |

Default Berkeley mission: extent 400 x 868 m, origin 37.87285 N 122.25925 W, z0 = 83.14 m NAVD88, B at
z = +26.2 m. A is a level launch pad (terrain flattened within 4 m, blended back by 10 m): the IMU is
calibrated on the ground assuming it stands level (see "Known differences" below).

## Onboard files

### `onboard/manifest.json`

| Key | Content |
| --- | --- |
| `format`, `version`, `kind`, `package_id`, `created_at` | `sfhack-mission/1`, 2, `onboard map package`, mission id, UTC ISO time |
| `bbox`, `origin`, `coordinate_frame`, `z_datum`, `z0_elevation_m` | see Frames |
| `a`, `b` | `{lat, lon, enu: [x, y], ground_z}`; `b.search_radius_m` |
| `ortho` | `{file, source, width_px, height_px, extent_enu: {min_x, min_y, max_x, max_y}, m_per_px: [x, y], processing}` (the V2 `OnboardMap` reads this block) |
| `dsm` | `{file, min_x, max_y, max_x, min_y, res_m, rows, cols}` |
| `map_cloud` | `{file, kind: "real" or "synthetic", voxel_m, points, lidar: {collection, depth, spacing_m, points_raw, vertical_offset_to_dem_m, ...}}` (`lidar` only when real) |
| `layers` | per layer `{file, description, bytes, sha256, ...}`: `ortho`, `landmarks`, `dsm`, `map_cloud`, `mission` (the manifest itself is the index and has no entry) |
| `differences_from_truth` | what was perturbed, including the ids of removed landmarks (see below) |
| `sources`, `attribution` | name, license, attribution, URL for every source used |
| `landmark_counts`, `files` | counts per type, file list |

### `onboard/mission.json`

```jsonc
{
  "mission_id": "berkeley", "format": "sfhack-mission/1",
  "a": {"lat": 37.8703, "lon": -122.2595, "enu": [-21.97, -283.86], "ground_z": 0.0},
  "b": {"lat": 37.8754, "lon": -122.259, "enu": [21.97, 283.86], "ground_z": 26.15, "search_radius_m": 30.0},
  "cruise": {"speed_m_s": 8.0, "clearance_m": 15.0, "min_agl_m": 30.0, "obstacle_margin_m": 15.0,
             "max_leg_m": 120.0, "max_z": 59.0, "legs": [{"leg": 1, "required_z": 36.6, "z": 38.0}, ...]},
  "route": [   // ENU, z absolute (datum above); flown in order by sfhack.planning.mission.route_plan
    {"phase": "TAKEOFF", "enu": [x, y, z], "speed_m_s": 3.0},   // straight up at A
    {"phase": "CRUISE",  "enu": [x, y, z], "speed_m_s": 8.0},   // 1..n legs, <= 120 m each
    {"phase": "DESCEND", "enu": [bx, by, b.ground_z + 4.0], "speed_m_s": 2.5},
    {"phase": "LAND",    "enu": [bx, by, b.ground_z - 0.3], "speed_m_s": 1.0}
  ],
  "target_scene": {"status": "pending", "center_enu": [bx, by], "radius_m": 30.0, "ground_z": 26.15, "objects": []},
  "warnings": []   // e.g. B on a mapped roof
}
```

The route is planned **on the onboard map only** (what the drone carries): required altitude per 4 m cell
= max(onboard DSM within 15 m + 15 m clearance, onboard DTM within 15 m + 30 m). Dijkstra on that raster,
each metre costing `1 + (altitude it forces above what A and B need) / 5 m`, so the route goes around a
tower rather than 100 m over it (vision fixes need roughly < 60 m above the surface). Each leg flies at the
highest requirement along it and the next leg. `target_scene` here is the search area only: the drone
does not know where the objects are.

### `onboard/ortho.jpg`

RGB JPEG (quality 80), north-up, `m_per_px` 0.5 m, covering `ortho.extent_enu` exactly (pixel
(u, v) centre at `x = min_x + (u + 0.5) * 0.5`, `y = max_y - (v + 0.5) * 0.5`). Source NAIP (USDA,
~0.6 m native, the newest mosaic the service serves), fallback USGS National Map tiles z16 (~1.9 m), last resort a stylized
OSM rendering (`ortho.source` says which).

### `onboard/landmarks.json`

List of objects, same schema as `sfhack/world/sample_berkeley`:

| Key | Content |
| --- | --- |
| `id` | int, the **same id as in `world/landmarks.json`** (gaps = landmarks removed from the onboard copy) |
| `type` | `building`, `tower`, `fountain`, `tree` |
| `osm_id`, `name` | OSM `way/123`, `relation/..`, `node/..`; name when tagged |
| `centroid_enu` | `[x, y]` m |
| `footprint_enu` | CCW polygon `[[x, y], ...]` (buildings, ways), absent for node landmarks |
| `height` | m above `base_z` (onboard copy: perturbed) |
| `base_z` | m, lowest onboard-DTM height under the footprint (the building's foot) |
| `radius`, `area_m2` | m (max centroid distance to the footprint), m^2 |
| `height_source` | `tag` (OSM `height`), `levels` (`building:levels` x 3.2 m), `lidar` (90th percentile of the 3DEP returns inside the footprint minus `base_z`, only for buildings without tags), `default` (10 m; trees 12 m, towers 30 m) |

### `onboard/dsm.npz`

`np.load` gives float32 arrays `dtm` (bare earth) and `dsm` (surface: DTM raised to `base_z + height`
inside every onboard footprint, tree crowns as discs of crown radius at crown top), both (rows, cols),
**1.0 m** cells, z in the mission datum; and float64 scalars `min_x`, `max_y`, `max_x`, `min_y`, `res_m`,
`rows`, `cols` (raster convention above). Default Berkeley: 868 x 400, 2.1 MB.
Lookup: `sfhack.missionprep.geometry.sample_bilinear(arr, Grid(min_x, max_y, res_m, rows, cols), xy)`.

### `onboard/map_cloud.ply`

Binary little-endian PLY, one `vertex` element: `float x, float y, float z` (ENU m, mission datum) and
`uchar classification` (ASPRS: 1 unclassified, 2 ground, 5 high vegetation, 6 building, 17 bridge deck,
...; noise classes 7 and 18 removed). Voxel-filtered: one point per occupied **1.0 m** voxel at the mean
of its points. Header comments say `real` or `synthetic`. Reader: `sfhack.missionprep.lidar.read_ply`.

- `real`: USGS 3DEP airborne LiDAR read from the public Entwine octree on AWS (`usgs-lidar-public`,
  newest collection covering the bbox; Berkeley: `CA_AlamedaCo_3_2021`), all octree depths down to
  ~0.5 m spacing, cropped to the bbox, vertically checked against the DEM (`vertical_offset_to_dem_m`,
  shifted if > 0.5 m). The 2021 Alameda collection classifies only ground (2) and unclassified (1).
  Default Berkeley: 8.7 M raw points -> 1.29 M points, 16.8 MB.
- `synthetic` (no collection, download failure, 90 s time box, or `--no-lidar`): sampled from the
  onboard footprints extruded to `base_z + height` (roof lattice + walls), the onboard DTM (ground
  outside footprints) and tree crowns (upper hemispheres), 5 cm Gaussian noise, then the same voxel filter.

## World files (truth, simulator only)

| File | Content |
| --- | --- |
| `world/manifest.json` | as onboard (`version` 2, `kind` `world (simulation truth)`), plus `terrain: {file, source, min_x, max_y, max_x, min_y, res_m, rows, cols}` and `"buildings": "extruded"` (tells `PackageWorld` to build terrain + footprint meshes; the flat V2 sample has neither and keeps its boxes) |
| `world/landmarks.json` | the true landmarks (schema above, unperturbed heights, all of them, `base_z` on the true DTM) |
| `world/ortho.jpg` | the original imagery (JPEG q92), same grid as the onboard ortho |
| `world/terrain.npz` | `z` float32 (rows, cols) at **2.0 m** (block mean of the 1 m truth DTM), same scalar keys as `dsm.npz` |
| `world/target_scene.json` | `{status, center_enu, radius_m, ground_z, objects: [], object_format}` |

How the simulator builds it (`sfhack/world/scene.py`): MuJoCo height field from `terrain.npz` with the
ortho draped on it; every footprint extruded from `base_z - 1` to `base_z + height` (visual mesh) with a
5 cm roof slab texture-mapped into the ortho (a real roof looks like its aerial photo), collision as convex
triangle prisms (group 3, hidden), trees as trunk + crown spheres (visual, seen by the camera and the
lidar rays). Invisible contact pads in the local terrain plane at A (2.5 m) and B (4 m).

### `world/target_scene.json` (object placer)

Write objects into `objects`; the simulator loads them as static mesh geoms (visible to camera, lidar,
rangefinder; they collide):

```json
{"mesh": "C:/.../7 SF Hack Objects/objects/car/sedan_a/sedan_a.stl", "pos": [27.9, 285.9], "yaw_deg": 30,
 "scale": 1.0, "rgba": [0.8, 0.1, 0.1, 1]}
```

`mesh`: STL, absolute or relative to `world/`, in the catalog convention (metres, +z up, origin =
bottom centre). `pos`: `[x, y]` or `[x, y, z]` ENU; without z the object stands on the true ground there
(`PackageWorld.ground_z`, bilinear on `terrain.npz`). Keep objects within `radius_m` of `center_enu`.
Do not write them into `onboard/`: the drone must find them. Re-running `build --force` recreates the
mission directory (and drops placed objects and the drone copy).

## Onboard vs truth (deliberate differences)

| Layer | Truth (`world/`) | Onboard (`onboard/`) |
| --- | --- | --- |
| Buildings, towers | OSM footprints, heights tag / levels / LiDAR / 10 m | heights x U(0.8, 1.2) per object |
| Buildings removed | none | 10 % of the `building` objects lower than 30 m (the minimum cruise height above ground, so a missing building is never in the flight path); ids in `differences_from_truth.buildings_removed` |
| Trees | OSM | 10 % removed, positions jittered 0.5 m (std), heights x U(0.8, 1.2) |
| Terrain | 3DEP DEM, 2 m height field | 1 m DTM + smooth noise (0.15 m std, ~20 m correlation) |
| Ortho | NAIP, JPEG q92 | same pixels re-processed: gamma U(0.9, 1.1), contrast x0.88-0.95, saturation x0.88, 0.5 px blur, JPEG q80 (geometry identical) |
| Point cloud | (the simulator ray-casts the world meshes) | real 3DEP LiDAR: a different source than the OSM-extruded world (real trees, cars, roof shapes, 2021 state); synthetic: from the perturbed onboard meshes + noise |

All random draws are seeded from the mission id, so a rebuild reproduces the same package.

## Outside the US (`region: global`)

`--region auto` (default) uses the US sources when the bbox lies inside the conterminous US
(24.4-49.4 N, 125.0-66.9 W), else the global ones (`--region us|global` forces):

| Layer | US | Global |
| --- | --- | --- |
| Ortho | NAIP ~0.6 m RGB (+ band 4 NIR) | Sentinel-2 L2A 10 m: ESA true-colour TCI as RGB, B08 NIR, NDVI from B04 / B08 |
| Terrain | 3DEP 1 m DEM (GLO-30 fallback) | Copernicus GLO-30 (a DSM: forest canopy and dense roofs are in the "terrain") |
| Point cloud | 3DEP LiDAR (synthetic fallback) | synthetic (no open LiDAR); DSM = terrain + OSM building heights |

Sentinel-2 (`sfhack/missionprep/sentinel.py`): Earth Search STAC (`sentinel-2-l2a`, AWS open data, no account),
June-August of 2025 / 2024 / 2023; the scene with the least cloud + shadow + nodata INSIDE the bbox (scene
classification layer on a 20 m grid; first one <= 1 % wins). Only the COG tiles covering the bbox are read, by
HTTP range requests through the cached Fetcher (tifffile, no GDAL). UTM -> local ENU with our own transverse
Mercator (Krueger series, checked against quadrature). Reflectance = DN x 1e-4: Earth Search items with
`earthsearch:boa_offset_applied` already had the -1000 removed (their raster:bands still say offset -0.1;
applying it zeroed the red band). `nir.png` = B08 reflectance x 425 (255 = 0.6); `ndvi.png` = (NDVI + 1) x 127.5.
The 10 m pixels are resampled onto the 0.5 m grid (bicubic RGB, bilinear NIR / NDVI): blurry by design.

Mission `ukraine-1` (built 2026-10-24): Skhidnytsia, Carpathian foothills south of Boryslav (V3's S2 area),
A 49.2244 N 23.3220 E on the asphalt road west of the village -> B 49.2305 N 23.3456 E beside the main road,
1.8 km, extent 2016 x 936 m, relief -6 .. +131 m, scene S2C_34UFV_20250828 (0 % cloud in the bbox), 348
buildings (346 at the default 10 m), route planned at 60 m AGL (`--agl 60`). Build: 41-54 s cold. Roads in
the extent by OSM `surface` (`scripts/plot_mission_roads.py ukraine-1`, figure `docs/ukraine-1_roads.png`):

| highway | paved km | unpaved km | untagged km |
|---|---:|---:|---:|
| secondary | 0.14 | 0.00 | 0.00 |
| tertiary | 2.31 | 0.00 | 0.00 |
| unclassified | 0.35 | 0.00 | 0.00 |
| residential | 0.30 | 0.39 | 1.71 |
| service | 0.26 | 0.86 | 6.21 |
| track | 0.00 | 3.52 | 0.00 |
| **all** | **3.36** | **4.77** | **7.92** |

21 % paved, 30 % unpaved, 49 % untagged (village lanes and driveways, mostly dirt per night-research). Only the
tagged-paved roads count as warm at night (the thermal prior's `roads` mask).

## Sources and licenses

| Layer | Primary | Fallback | License |
| --- | --- | --- | --- |
| Buildings, trees, roads, landuse | OpenStreetMap via Overpass (overpass-api.de, one query) | overpass.private.coffee, then the OSM API `map` call | ODbL 1.0, (c) OpenStreetMap contributors |
| Ortho | USGS NAIP ImageServer `exportImage` (EPSG:4326 = linear in our frame) | USGS National Map imagery tiles z16 | Public domain |
| Terrain | USGS 3DEP 3DEPElevation ImageServer (1 m DEM where TNM lists one, else best available) | Copernicus GLO-30 DSM (AWS open data) | Public domain / Copernicus DEM licence |
| Point cloud | USGS 3DEP LiDAR, Entwine EPT on AWS `usgs-lidar-public` | synthetic | Public domain |
| Imagery outside the US | Copernicus Sentinel-2 L2A COGs (Earth Search STAC, Element84 / AWS) | - | Copernicus Sentinel data terms (free, full and open) |
| Search box | Nominatim (User-Agent, <= 1 request/s, cached, proxied by the planner) | - | ODbL |

Downloads are cached under `$SFHACK_DATA/cache/missionprep` (keyed by URL + parameters): a second
prepare of the same area needs no network.

## Build times and sizes (default Berkeley A / B, this machine, 2026-10-23)

| Layer | First run (network) | Size in the package |
| --- | --- | --- |
| OSM (Overpass) | 34 s (queue-bound; 52 s for campus-glade) | landmarks.json 0.06 MB |
| Ortho (NAIP) | 4.3 s | world 0.35 MB, onboard 0.23 MB |
| Terrain (3DEP 1 m) | 3.8 s | terrain.npz 0.31 MB, dsm.npz 2.1 MB |
| LiDAR (3DEP EPT, 225 nodes, 90 MB) | 4.7 s download + ~5 s decode | map_cloud.ply 16.8 MB |
| World + onboard build | ~3 s | |
| **Total** | **37 s** (downloads run in parallel) | onboard upload 19.3 MB = ~10 s at 2 MB/s |
| Cold cache (empty `SFHACK_DATA`), later run | 99.6 s: both Overpass servers failed, OSM API answered (timeouts since cut to 60 s + 25 s) | 94 MB downloaded |
| Rebuild from cache | 9-13 s (LAZ decode dominates) | |
| Prepared mission reused | 1.6 s (CLI incl. Python start) | |

## Known differences from reality

- Terrain-aware vision: the localizer rectifies the camera onto the onboard DSM (roofs and slopes);
  without a DSM it keeps the flat-ground assumption at z = 0 (sample package unchanged).
- Launch pad: the X2 calibrates its IMU on the ground assuming level; on the raw 2.5 deg slope at A it
  then flew off downhill. A is flattened (4 m) and both A and B have invisible contact pads, because
  MuJoCo height-field contacts left the X2 resting 9 deg tilted.
- The world's buildings are prisms (flat roofs); the real LiDAR cloud has real roof shapes, trees and
  cars: expect scan-to-map residuals of a few metres around vegetation.
