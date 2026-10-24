# Night demo: viewer, route choice, Ukraine mission (night-demo branch)

Scenario: summer night, 23:00, clear, no wind. The X2 flies A -> B at 60 m AGL with GNSS at first; key N
removes GNSS completely (no signal). From then on it navigates on IMU, barometer, laser rangefinder, TRN on the
onboard DSM and a downward thermal camera (640 x 512, 50 deg HFOV) matched against a thermal prior predicted
from daytime data (night-thermal's model, night-nav's localizer). If thermal gives nothing it keeps flying the
route on IMU + TRN and the HUD shows the growing uncertainty.

## Viewer: `demos/demo_live.py --night`

```
uv run python demos/demo_live.py --mission ukraine-1 --night                 # window; N = GNSS off / on
uv run python demos/demo_live.py --mission berkeley --night --cruise-agl 60  # missions built at 30 m AGL
uv run python demos/demo_live.py --mission ukraine-1 --night --record out.mp4 --events "45:N"   # video
```

Code: `demos/liveview/night.py` (NightComposer, NightState, ThermalFeed); `demo_live.py` only adds the flags,
key N and the composer choice. Layout:

- **Thermal panel** (right, large): the frames the simulator publishes on `camera/thermal` (what the onboard
  localizer sees), 8-bit AGC white-hot. Fallbacks: night-thermal's ThermalCamera rendered by the viewer, else a
  labelled STUB (nadir warp of the truth raster or of warm = low NDVI). Caption: source, last fix (method,
  quality, accepted / rejected). Inset bottom right: what the localizer matched, SEEN (the rectified frame,
  north-up, `autopilot.thermal.last_tpl`) next to the MAP PRIOR crop at the fix.
- **Map panel** (bottom left): the onboard thermal prior (predicted temperature, inferno colormap) with the
  prior road mask in orange; route; true (white) and estimated (cyan) trails; the 2-sigma ELLIPSE of the
  filter covariance; the last fix (yellow cross accepted, red rejected); the road pixels inside the camera
  footprint around an accepted thermal fix light up yellow ("matched roads").
- **HUD** (top left): `GNSS: ON / OFF` chip (time of the switch), active nav source (night-nav's
  `autopilot.night_nav_source`: GNSS / THERMAL / TRN / INS with fix ages), "BLIND: keep route" when the health
  monitor reports blind flight, 2-sigma uncertainty and the true error (eval only) in metres, and a 3-minute
  sparkline of both with the GNSS-off moment marked.
- **3D**: model lights x 0.18, headlight dimmed, night sky; drone, ghost, trails stay readable.

Key N toggles `gnss.mode` between `off` (no receiver output at all, not a jam) and `ok`. Nothing switches
automatically. The planner page has a "Night" checkbox; `/api/fly {mission_id, night: true}` starts the viewer
with `--night`.

## Route choice (`sfhack/planning/night_route.py`)

At night the thermal localizer lives on warm asphalt. Over forest or fields it sees nothing, and the estimate
drifts on IMU + TRN. The route choice keeps the straight (obstacle-avoiding) route while the camera can expect
road, and detours along roads otherwise. Onboard code: reads only the onboard DSM and `thermal_prior.npz`.

1. Per 4 m cell: paved-road area (prior `roads` mask) inside the camera footprint at the cruise height. The
   footprint is 2 x 60 m x tan(25 deg) = 56 m wide at 60 m AGL; a disc of radius 0.9 x 28 m is used (inscribed,
   any heading). Visible if >= `min_road_area_m2`.
2. The planner's route at the cruise AGL is sampled every 2 m. Longest blind stretch <= `max_blind_gap_m`:
   **STRAIGHT**.
3. Else: re-plan with the same Dijkstra (`missionprep.build.plan_route`, new `extra_cost`) where every metre over
   a blind cell costs `blind_cost` more; string pulling may only shortcut when it adds no blind cost. Kept if it
   shortens the longest blind stretch and is at most `max_detour_frac` longer: **ROADS**.

| Parameter | Default | Why |
|---|---:|---|
| `cruise_agl_m` | 60 | scenario |
| `min_road_area_m2` | 150 | ~ a 5 m wide road crossing 30 m of the 56 m footprint: enough edge for a fix |
| `max_blind_gap_m` | 250 | ~30 s at 8 m/s: what IMU + TRN hold to a few metres |
| `blind_cost` | 3.0 | a blind metre costs 4x a seen one: hugs the road, still cuts hairpins |
| `max_detour_frac` | 0.6 | never more than 60 % longer than the straight route |

`FlightSystem(night=True)` applies it by default (`night_route=False` turns it off; result in
`fs.night_route`). CLI with figure: `uv run python -m sfhack.planning.night_route <mission> --png out.png`.

| Mission | Choice | Straight: length, road visible, longest blind | Roads: length, visible, longest blind |
|---|---|---|---|
| ukraine-1 | ROADS | 1870 m, 10 %, 1129 m | 2034 m (+9 %), 96 %, 64 m |
| berkeley | STRAIGHT | 569 m, 70 %, 68 m | - |
| m-c804bb84 | STRAIGHT | 923 m, 75 %, 101 m | - |

Figures: `docs/night_route_ukraine-1.png`, `docs/night_route_berkeley.png`, `docs/night_route_m-c804bb84.png`.

## Ukraine mission

See docs/MAP_PACKAGE.md, "Outside the US": Sentinel-2 L2A + Copernicus GLO-30 + OSM, mission `ukraine-1`
(Skhidnytsia), paved / unpaved road table, figure `docs/ukraine-1_roads.png`.
