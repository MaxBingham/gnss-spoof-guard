# RESUME - branch night-thermal (2026-10-24, night agent night-thermal)

Thermal world model + simulated LWIR camera. Report: `REPORT_night-thermal.md`. Model and formats:
`docs/THERMAL_MODEL.md`. Folder C:\dev\night-thermal (not pushed; the owner pushes in the morning).

## Continue

```powershell
cd C:\dev\night-thermal
. .\scripts\env.ps1
$env:UV_PROJECT_ENVIRONMENT = "$HOME\.venvs\sf-hack-night-thermal"     # env.ps1 sets the v5 venv; override after
uv run python -m pytest tests -q                                        # full suite (thermal: tests/test_thermal.py)
uv run python -m sfhack.thermal build --mission berkeley m-c804bb84 ukraine-1   # 11 / 14 / 33 s; REWRITES the npz files (frozen for the final eval until 06:30)
uv run python -m sfhack.thermal build --mission berkeley --hour 21      # other hour (local clock); --contrast 0.4 = mild night (default 0.7)
uv run python demos/20_thermal_view.py --mission berkeley               # window: camera | ortho | prior (Esc, Space)
uv run python demos/20_thermal_view.py --mission ukraine-1 --record     # mp4 -> $SFHACK_DATA\night\thermal\
uv run python demos/21_thermal_figures.py                               # docs/figures/* (writes nothing into missions)
uv run python demos/22_thermal_matchability.py                          # is there signal? NCC vs predicted view, 60 m AGL
```

In code:

```python
from sfhack.thermal.camera import ThermalCamera, load_truth   # simulator side
sim.render("down"); cam = ThermalCamera(sim.model, renderer=sim.renderer)
img_K = cam.render(sim.data, load_truth(mission_dir))          # (512, 640) float32 K; cam.to_8bit(img_K)
from sfhack.thermal.prior import load_prior                    # onboard side
prior = load_prior(onboard_dir); pred_K, sig_K = prior.predict_view(xy, agl, cam.intrinsics, yaw=yaw)
```

## Open

- Light insulated roofs -6.6 K (literature typical -4..-6). Seasons other than summer are rough presets. No weather.
- Cars and wet patches are fixed per build (`--seed`), not per flight.
- night-nav (thermal localizer, eval) and night-demo (viewer, Ukraine, final eval) consume these files; see their
  branches for the flight results.

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
