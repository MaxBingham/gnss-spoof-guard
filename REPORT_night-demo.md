# REPORT - night-demo (Fri 2026-10-23 23:30 -> Sat 07:00)

Agent night-demo: Ukraine mission, route choice, night viewer, final integration + eval. Branch `night-demo` in
`C:\dev\night-demo` (based on origin/v5-mission-planner; merged night-thermal 146a89f and night-nav 4904433).
Nothing pushed. Commands to continue: `RESUME.md`. Saturday demo script: `DEMO_NIGHT.md`.

## Final eval (night 23:00, 60 m AGL, GNSS completely off from 20 % of the route, 20 seeds)

EVAL_TABLE_PLACEHOLDER

## What works

1. **Missions outside the US** (`sfhack/missionprep/sentinel.py`, `build.py`): `--region auto` picks NAIP / 3DEP /
   3DEP LiDAR inside the conterminous US, else Sentinel-2 L2A (Earth Search STAC, no account; scene chosen by
   cloud + shadow INSIDE the bbox from the SCL band; only the needed COG tiles read by cached HTTP range requests
   via tifffile, no GDAL; own UTM projection, tested against quadrature) + Copernicus GLO-30 + OSM, synthetic
   point cloud, DSM = DEM + OSM building heights. New layers for every build: `world/osm.json` (raw OSM, for
   the thermal model), `nir.png` + `ndvi.png` (NAIP band 4 in the US, S2 B08/B04 abroad) in world/ and onboard/.
   `--agl` sets the route's minimum cruise height. The US path is unchanged (all old tests green; existing
   missions still count as prepared).
2. **Mission `ukraine-1`**: Skhidnytsia, Carpathian foothills south of Boryslav (V3's S2 area), A 49.2244 N
   23.3220 E -> B 49.2305 N 23.3456 E (25 m from the asphalt main road), 1.8 km, relief 137 m, S2 scene
   2025-08-28 (0 % cloud in the bbox), 41-54 s cold build. Paved / unpaved from OSM `surface`: 3.4 km paved
   (tertiary asphalt), 4.8 km unpaved (tracks), 7.9 km untagged (village lanes, mostly dirt) = 21 / 30 / 49 %.
   Table + figure in docs/MAP_PACKAGE.md ("Outside the US"), `docs/ukraine-1_roads.png`. Day flight on it with
   the RGB vision localizer: ARRIVED 1.06 m.
3. **Route choice** (`sfhack/planning/night_route.py`, onboard): straight while the thermal footprint (56 m at 60 m
   AGL) is expected to hold >= 150 m^2 of paved road, else Dijkstra with a blind-cell cost along the roads, if
   it shortens the longest blind stretch (> 250 m) and costs <= 60 % extra length. ukraine-1: ROADS (straight
   1870 m with a 1129 m blind stretch over the forested hill -> 2034 m along the road, longest blind 64 m);
   berkeley and m-c804bb84: STRAIGHT (longest blind 68 / 101 m). On by default in `FlightSystem(night=True)`.
   Parameters and reasons: docs/NIGHT_DEMO.md.
4. **Night viewer** (`demo_live.py --night`, `demos/liveview/night.py`): dark scene; large white-hot thermal panel
   showing the onboard sensor feed (night-nav's ThermalFrame) with a SEEN vs MAP PRIOR inset of the last match;
   map panel on the predicted night temperature with roads, true / estimated trails, the 2-sigma ellipse and the
   matched roads lit up; HUD with GNSS ON / OFF, nav source (THERMAL + TRN + IMU / INS ...), 2-sigma and true
   error with a 3-minute sparkline. Key N = GNSS completely off / on (no automatic switch). Missing thermal files
   are built on the fly (`ensure_thermal`, ~50 s). Planner page: "Night" checkbox -> `/api/fly {night: true}`.
5. **Eval harness** (`demos/30_night_eval.py`): parallel seeds, GNSS off when the TRUE along-route progress passes
   20 % (simulated operator), arrival error = |truth - B| when the drone decides it is above B, Wilson 95 % CI,
   per-second traces, rows thermal / imu_trn / straight / low_contrast, frozen mission copy, divergence abort.
6. Tests: 3 new test files / cases (global build path offline, UTM, night route straight vs detour, planner
   Night pass-through); all tests green after both merges.

## What failed / limits (honest)

- **IMU + TRN alone diverges on ukraine-1** (estimate 10 km off after 6 min): closed-loop tilt unobservability
  in calm air (no rotor-drag model in the sim), TRN on the 30 m GLO-30 surface model never gets a fix accepted
  after the error is large. night-nav analysed it; it is a physics / model decision, not fixed tonight. The
  thermal row keeps the estimate bounded only because it gets fixes several times per second along roads.
- The viewer ran at 2-3 fps tonight because CPU (four agents' evals) and GPU (detector training) were saturated;
  sim + thermal rendering + localizer is ~290 ms per displayed frame under that load. Expect better on Saturday;
  if not, use speed 1 or the backup videos.
- Backup videos are mp4v (no ffmpeg on this machine): play in VLC / browsers; convert if needed.
- Sentinel-2 at 10 m is blurry as the world texture (0.5 m grid); GLO-30 is a DSM, so the "terrain" of
  ukraine-1 contains the forest canopy shape; 346 of 348 buildings use the default 10 m height.
- The preliminary run1 (01:00) was stopped because night-thermal rebuilt the thermal files mid-run; the final
  eval runs on a frozen copy (`$SFHACK_DATA/night/demo_eval_snap`).

## Next steps

- Saturday: landing at B (thermal fixes stop near the ground; precision descent logic exists from V2).
- Rotor-drag model + drag-velocity fusion (as in death-v3) so blind stretches do not run away.
- Route choice: weight by thermal localizability (prior gradient energy) instead of road area only; add
  junction density; show the night route on the planner page.
- Weather: clouds / wind reduce contrast; tie `thermal_contrast` to a weather lever.

## Disclosure (all of this is PRE-HACKATHON work, built Fri 23:30 - Sat 07:00 by Claude Code agents)

Built by night-demo tonight: Sentinel-2 / GLO-30 global mission path (sentinel.py, region switch, NIR / NDVI,
osm.json, --agl), mission ukraine-1, scripts/plot_mission_roads.py, night route choice (night_route.py + the
extra-cost hook in plan_route), the night viewer (liveview/night.py, demo_live --night, key N, ensure_thermal),
planner Night checkbox, eval harness demos/30_night_eval.py, docs (NIGHT_DEMO.md, MAP_PACKAGE.md section,
DEMO_NIGHT.md, README section), the merges and the final eval. Merged from the other night agents (also
pre-hackathon): night-thermal (thermal world model, thermal camera, prior), night-nav (V3.2 GNSS-free track,
TRN, thermal localizer, thermal sensor), night-research (facts used in DEMO_NIGHT.md).
