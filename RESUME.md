# RESUME - branch night-demo (night of 2026-10-23/24)

Folder `C:\dev\night-demo`, branch `night-demo` (based on origin/v5-mission-planner, merged: night-thermal,
night-nav). Nothing pushed. Report: `REPORT_night-demo.md`. Demo script for Saturday: `DEMO_NIGHT.md`.
The previous RESUME (v5 mission planner) is in git: `git show 28962ad:RESUME.md`.

## Environment

```
cd C:\dev\night-demo
. scripts\env.ps1                                                  # sets SFHACK_DATA, UV_LINK_MODE, ...
$env:UV_PROJECT_ENVIRONMENT = "$HOME\.venvs\sf-hack-night-demo"   # tonight's venv (env.ps1 sets sf-hack-v5)
uv sync                                                            # once, if the venv is missing
uv run python -m pytest tests -q                                   # all tests
```

## Run

```
# missions (outside the US: Sentinel-2 + GLO-30 + OSM; --agl = cruise height above ground for the route)
uv run python -m sfhack.missionprep build --a 49.2244 23.3220 --b 49.2305 23.3456 --radius 30 --id ukraine-1 --agl 60
uv run python -m sfhack.thermal build --mission ukraine-1          # night-thermal: thermal truth + prior
uv run python scripts/plot_mission_roads.py ukraine-1 --out docs/ukraine-1_roads.png   # paved / unpaved table

# night viewer (key N = GNSS off / on); missions built at 30 m AGL: add --cruise-agl 60
uv run python demos/demo_live.py --mission ukraine-1 --night
uv run python demos/demo_live.py --mission berkeley --night --cruise-agl 60
uv run python demos/demo_live.py --mission ukraine-1 --night --record out.mp4 --events "45:N"   # video

# planner page with the Night checkbox
uv run python -m sfhack.missionprep serve                          # http://127.0.0.1:8765/

# route choice alone (prints the decision, writes a figure)
uv run python -m sfhack.planning.night_route ukraine-1 --png docs/night_route_ukraine-1.png

# eval: 20 seeds, GNSS off at 20 % of the route, arrival error within 5 m, Wilson 95 % CI
uv run python demos/30_night_eval.py --missions berkeley,m-c804bb84,ukraine-1 --seeds 20 --workers 8 --rows thermal,imu_trn
uv run python demos/30_night_eval.py --only ukraine-1:straight,ukraine-1:low_contrast --seeds 10 --workers 8
```

Long evals: run them from a frozen worktree so later edits do not leak in:
`git worktree add C:\dev\night-demo-eval HEAD --detach`, then in that folder set
`$env:PYTHONPATH = "C:\dev\night-demo-eval"` and `uv run --no-sync python demos/30_night_eval.py ...`.

## Where things are

| What | Where |
|---|---|
| Sentinel-2 / global data path | `sfhack/missionprep/sentinel.py`, `build.py` (region, NIR / NDVI, osm.json) |
| Night viewer | `demos/liveview/night.py`, hooks in `demos/demo_live.py` (`--night`, key N, `ensure_thermal`) |
| Route choice | `sfhack/planning/night_route.py`, `FlightSystem(night=True, night_route=True)` in `sfhack/runner.py` |
| Eval | `demos/30_night_eval.py` -> `$SFHACK_DATA/night/eval/<run>/` (results.json, table.md, error_over_time.png) |
| Backup videos | `$SFHACK_DATA/night/demo/` |
| Docs | `docs/NIGHT_DEMO.md`, `docs/MAP_PACKAGE.md` ("Outside the US"), `DEMO_NIGHT.md` |

## Next

See REPORT_night-demo.md, "Next steps".
