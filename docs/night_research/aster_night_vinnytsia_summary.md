# ASTER night check: vinnytsia

bbox (28.36, 49.19, 28.56, 49.27); band 13 brightness temperature, 90 m; roads: Overture 2026-10-21.0 (OSM-derived), paved / unpaved from the surface tag, else from the class (track = unpaved, motorway..secondary = paved); road km in box: paved 271 km, unpaved 157 km, unknown 494 km.

## 2005-09-13 22:51 local time

Registration to paved-road share: r +0.017 as delivered -> **+0.099** after shifting the image +280 m east, -160 m north.

- paved road share bins ['0-3 %', '3-10 %', '10-20 %', '20-35 %', '35-100 %']: 11.90 (n=10388) | 12.45 (n=913) | 12.63 (n=1211) | 12.80 (n=242) | 12.40 (n=57)
- unpaved road share bins ['0-3 %', '3-10 %', '10-20 %', '20-35 %', '35-100 %']: 11.90 (n=9591) | 11.80 (n=582) | 12.09 (n=213) | - (n=2) | - (n=0)
- regression (per 100 % share vs. the remainder; n = 12811, R^2 = 0.23): paved road +1.2 K, unpaved road -0.3 K, unknown-surface road -1.4 K, other built-up +0.0 K, tree -0.6 K, grass/shrub/crop -1.1 K, bare -1.2 K, elevation per 100 m -1.2 K
- class medians: built >= 70 % 12.6 C (n=3058), tree >= 50 % 11.7 C (n=5489), grass/shrub/crop >= 50 % 11.4 C (n=1763), water >= 90 % 14.5 C (n=146)

## 2005-09-20 22:56 local time

Registration to paved-road share: r +0.014 as delivered -> **+0.112** after shifting the image +310 m east, -140 m north.

- paved road share bins ['0-3 %', '3-10 %', '10-20 %', '20-35 %', '35-100 %']: 5.13 (n=5735) | 5.08 (n=590) | 5.47 (n=802) | 5.38 (n=189) | 4.80 (n=53)
- unpaved road share bins ['0-3 %', '3-10 %', '10-20 %', '20-35 %', '35-100 %']: 5.19 (n=5338) | 4.36 (n=304) | 4.31 (n=92) | - (n=1) | - (n=0)
- regression (per 100 % share vs. the remainder; n = 7369, R^2 = 0.28): paved road +0.3 K, unpaved road -3.4 K, unknown-surface road -2.4 K, other built-up +0.8 K, tree +0.4 K, grass/shrub/crop -1.3 K, bare -1.8 K, elevation per 100 m -0.0 K
- class medians: built >= 70 % 5.6 C (n=2083), tree >= 50 % 5.2 C (n=3120), grass/shrub/crop >= 50 % 3.9 C (n=843), water >= 90 % 8.6 C (n=95)

