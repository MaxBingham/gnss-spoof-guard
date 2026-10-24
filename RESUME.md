# RESUME - branch night-research (2026-10-24, pre-hackathon night)

Docs-only branch. Read in this order: `REPORT_night-research.md` -> `docs/NIGHT_NOTES_FOR_TEAM.md` (short numbers) ->
`docs/NIGHT_RESEARCH.md` (long report, sources, pitch). No code in `sfhack/` changed; tests unaffected.

## Re-run the research scripts (no project venv needed; ad-hoc uv envs; data goes to $SFHACK_DATA/night/research)
```
cd C:\dev\night-research
$env:UV_LINK_MODE = "copy"; $env:SFHACK_DATA = "$HOME\sf-hack-data"
# 1-D surface energy balance (~4 min): docs/night_research/seb_*.png, seb_output.txt, seb_results.csv
uv run --no-project --with numpy --with matplotlib python docs/night_research/surface_energy_balance.py
# Berkeley ASTER night scenes: download + contact sheet (~2 min; NIGHT_SKIP_OSM=1 skips the Overpass road fetch)
$env:NIGHT_SKIP_OSM = "1"
uv run --no-project --with pystac-client --with planetary-computer --with rasterio --with numpy --with matplotlib --with shapely --with pyproj --with requests python docs/night_research/aster_night_check.py
# Berkeley registration + statistics (needs $SFHACK_DATA/night/research/aster/overture_roads_berkeley.json, see below)
cd docs/night_research
uv run --no-project --with numpy --with scipy --with matplotlib --with rasterio --with shapely --with pyproj --with pystac-client --with planetary-computer --with requests python aster_night_analyze.py
# Other sites (fetches Overture roads itself via DuckDB): first without --clear -> contact sheet, then with the clear dates
uv run --no-project --with numpy --with scipy --with matplotlib --with rasterio --with shapely --with pyproj --with pystac-client --with planetary-computer --with requests --with duckdb python aster_night_sites.py --site fresno --clear 2001-06-11,2002-06-14,2002-07-16,2003-07-28,2005-07-08,2005-07-24
#   davis --clear 2006-07-18 ; vinnytsia --clear 2005-09-13,2005-09-20 ; drohobych (cloudy, no clear date)
```
Berkeley roads file: produced once with DuckDB from Overture (release 2026-10-21.0, `theme=transportation/type=segment`,
bbox -122.325 37.840 -122.215 37.905); `aster_night_sites.py::overture_roads()` does the same for any bbox. To refetch for
Berkeley (from the repo root):
```
uv run --no-project --with numpy --with scipy --with matplotlib --with rasterio --with shapely --with pyproj --with pystac-client --with planetary-computer --with requests --with duckdb python -c "import sys, pathlib; sys.path.insert(0, 'docs/night_research'); import aster_night_sites as s; s.overture_roads((-122.325, 37.840, -122.215, 37.905), pathlib.Path.home() / 'sf-hack-data/night/research/aster/overture_roads_berkeley.json')"
```

## Open / next
- ECOSTRESS summer night scene over Berkeley (70 m): needs a free Earthdata Login, steps in REPORT_night-research.md.
- Night-time measurements of wet asphalt, and of insulated light roofs in summer: none found (estimates in the notes).
- Check the final night-nav / night-demo eval for the `low_contrast` (0.4x) row against section 6.1 of NIGHT_RESEARCH.md.

---

# RESUME - branch v5-mission-planner (2026-10-23)

## Done (all committed + pushed, 38/38 tests green incl. truth separation)
- `sfhack/missionprep`: pick A/B -> parallel cached downloads (Overpass -> mirror -> OSM API; NAIP -> USGS z16;
  3DEP 1 m DEM -> Copernicus GLO-30; real 3DEP LiDAR from the AWS EPT octree via laspy, synthetic fallback)
  -> truth world (terrain hfield, extruded footprints, ortho roofs) + imperfect onboard package + route on the
  onboard DSM (Dijkstra, avoids the Campanile) -> simulated upload with SHA-256 -> live viewer.
- Contract: `docs/MAP_PACKAGE.md`. Object placer writes `world/target_scene.json` (STL loader in `PackageWorld`).
- Flight: `FlightSystem(mission=id)`, `demo_live.py --mission id`. Vision localizer rectifies on the onboard DSM.
  Launch pad at A + contact pads at A/B (hfield contact issues). 10/10 seeds arrive (berkeley mean 0.79 m,
  campus-glade 1.21 m).
- Timing: default Berkeley cold 37 s (99.6 s once when Overpass failed -> OSM API; timeouts now 60 + 25 s),
  prepared mission 0.5-1.6 s, rebuild from cache 9-13 s. Real LiDAR worked (CA_AlamedaCo_3_2021).
- One-click (last request): Prepare -> upload -> fly on one progress bar; checkbox "Fly automatically" (default on);
  skip to fly when already prepared + uploaded; Fly button per mission; `/api/fly`, `/api/fly/status`,
  `/api/fly/stop`; viewer output in `<mission>/viewer.log`, crash tail shown on the page; user Stop is not a crash.
  Verified: API end-to-end (viewer alive after 10 s, stopped), page one-click, Fly button, Stop; unit test with
  Popen mocked.

## Start it
```
. scripts/env.ps1                                   # venv ~/.venvs/sf-hack-v5
uv run python -m sfhack.missionprep serve           # http://127.0.0.1:8765/ -> Prepare mission
uv run python demos/demo_live.py --mission berkeley # direct
```

## Next / open
- Chrome screenshots of the page timed out in this session: page verified via DOM/JS only; take a visual look.
- Planner server once exited silently after ~40 min (no traceback): watch for it.
- Cached rebuild is LAZ-decode bound (~5 s): cache the decoded cloud if < 10 s matters for rebuilds.
- Object placer: fill `world/target_scene.json` (format in MAP_PACKAGE.md).
