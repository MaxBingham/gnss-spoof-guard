# DEMO NIGHT - what to run, click and say (Saturday)

All built Friday night before the hackathon (see REPORT_night-demo.md, disclosure).

## Before the jury arrives (once)

```
cd C:\dev\night-demo                       # or wherever night-demo is checked out
. scripts\env.ps1
$env:UV_PROJECT_ENVIRONMENT = "$HOME\.venvs\sf-hack-night-demo"   # tonight's venv (or run scripts\setup.ps1 for yours)
uv run python -m sfhack.missionprep list   # berkeley, m-c804bb84, ukraine-1 must be there
```

Missions need `world/thermal_truth.npz` + `onboard/thermal_prior.npz`; if missing:
`uv run python -m sfhack.thermal build --mission ukraine-1`.

Backup videos (if the live sim misbehaves), in `$env:SFHACK_DATA\night\demo\`: `berkeley_night_v2.mp4` (77 s,
2x, N at 40 s) and `ukraine-1_night_v2.mp4` (80 s, 4x, N at 76 s, ends "ABOVE B 3.8 m from B"); 1920x1080, mp4v
codec (no ffmpeg on this machine: play with VLC or a browser). The files without `_v2` are the same flights but end
on the landing-outcome banner (landing is not part of tonight's scenario): prefer v2. Both were recorded before
night-nav's tilt-bias fix (651b398), on code night-demo 9141527.

## Live, option A: straight from the terminal

```
uv run python demos/demo_live.py --mission ukraine-1 --night --fullscreen
uv run python demos/demo_live.py --mission berkeley --night --cruise-agl 60 --fullscreen
```

## Live, option B: the planner page

```
uv run python -m sfhack.missionprep serve      # http://127.0.0.1:8765/
```
Tick **Night**, then press **Fly** next to `ukraine-1` (or prepare a new A / B; the thermal files are built with
`python -m sfhack.thermal build --mission <id>` before a night flight).

## In the viewer

| Key | What |
|---|---|
| **N** | GNSS completely off (no signal at all) / back on. Press it after ~20 s of cruise. |
| SPACE | pause (to explain a frame) |
| 1 / 2 / 4 | sim speed (2x default; 4x for the long Ukraine leg) |
| R | restart the flight |
| Tab / Q | fullscreen / quit |

What to point at:
1. Top left: **GNSS: ON** -> press N -> **GNSS: OFF**, the nav source changes from GNSS to THERMAL + TRN + IMU.
2. Right: the live thermal camera, white-hot. The warm asphalt road is the bright band; forest and fields are
   dark. Bottom right of that panel: SEEN (what the camera saw, turned north-up) vs MAP PRIOR (what we predicted
   from daytime satellite data and OpenStreetMap). They match -> yellow "THERMAL FIX accepted".
3. Bottom left: the map. White = where the drone really is, cyan = where it thinks it is, cyan ellipse = how
   sure it is. Orange = roads we expect to be warm.
4. Ukraine: the route does not take the straight line over the forested hill; it follows the road, because
   at night the road is what the camera can see (route choice).
5. Over a stretch without road the uncertainty grows (sparkline, ellipse), the drone keeps flying the route,
   and shrinks back at the next road.

## What to say (short)

- "At night a drone without GNSS is blind to normal cameras. But asphalt stores the day's heat and stays
  several degrees warmer than grass all night, while grass and trees cool down."
- "From open daytime data, satellite imagery and OpenStreetMap, we predict what the ground looks like in
  thermal at 23:00. The drone matches its thermal camera against that prediction to know where it is."
- "We plan the route where the camera will see roads. If it sees nothing, it keeps flying on inertial and
  terrain matching and tells you honestly how uncertain it is."
- Result (fill from the final table): "<X>/20 flights end within 5 m of B in Berkeley, <Y>/20 in the
  Ukraine mission, GNSS off from 20 % of the route."

Real-data backing (night-research NIGHT_NOTES v4, branch night-research, docs/night_research/): ASTER satellite
night thermal (90 m, ~23:00) shows the paved road grid on hot summer nights (Fresno, 6 nights Jun-Jul; Davis July
23:13: town +3.2 K over the fields). No evidence that unpaved roads are warm (Davis, Vinnytsia: 0-0.9 K cooler
than road-free cells), consistent with our paved-only road mask. Registering each night image to the road map
finds a consistent 150-340 m offset, consistent with the satellite's own night geolocation error (no independent
truth for it). Slide figure: `docs/night_research/aster_night_fresno_2003-07-28.png`. Asphalt vs grass: about
5 K measured (up to ~10 K in our model) on clear calm nights after a hot day. Fielded GNSS-free systems that
publish numbers quote ~10-30 m (vendor claims); one public night-thermal flight: < 15 m RMS.

Honest limits (if asked): simulated thermal world and camera; hot day = the favourable case (our scenario);
mild nights have less contrast (ASTER over Berkeley in spring: road cells only ~0.1-0.7 K warmer at 90 m) -> the
eval's `low_contrast` row (0.4x) stands for a mild spring / autumn night; clear skies on only ~1/3 of Ukrainian
July nights; without thermal fixes for long stretches the IMU + TRN estimate diverges in this sim (no rotor-drag
model); landing at B is tomorrow's work.
