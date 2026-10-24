# ASTER night check: davis

bbox (-121.81, 38.515, -121.68, 38.575); band 13 brightness temperature, 90 m; roads: Overture 2026-10-21.0 (OSM-derived), paved / unpaved from the surface tag, else from the class (track = unpaved, motorway..secondary = paved); road km in box: paved 285 km, unpaved 155 km, unknown 164 km.

## 2006-07-18 23:13 local time

Registration to paved-road share: r -0.072 as delivered -> **+0.274** after shifting the image +260 m east, -100 m north.

- paved road share bins ['0-3 %', '3-10 %', '10-20 %', '20-35 %', '35-100 %']: 22.52 (n=3819) | 24.63 (n=316) | 24.59 (n=715) | 24.86 (n=269) | 24.46 (n=185)
- unpaved road share bins ['0-3 %', '3-10 %', '10-20 %', '20-35 %', '35-100 %']: 22.60 (n=3489) | 22.14 (n=183) | 21.96 (n=145) | - (n=2) | - (n=0)
- regression (per 100 % share vs. the remainder; n = 5304, R^2 = 0.62): paved road +3.2 K, unpaved road +1.2 K, unknown-surface road +3.0 K, other built-up +1.8 K, tree +0.4 K, grass/shrub/crop -1.8 K, bare -0.6 K, elevation per 100 m +5.7 K
- class medians: built >= 70 % 24.9 C (n=947), tree >= 50 % 24.2 C (n=955), grass/shrub/crop >= 50 % 21.7 C (n=2500)

