# ASTER night scenes over Berkeley: clear scenes, registered to the road map

Brightness temperature, band 13 (10.6 um), 90 m, no emissivity / atmosphere correction (offsets cancel in differences). Medians in deg C; n = 90 m cells. Land cover: ESA WorldCover 2021; elevation: Copernicus GLO-30; roads: Overture Maps 2026-10-21.0 road segments (OSM-derived, ODbL).

Excluded scenes: 2006-07-18 (covers only the hills); 2006-04-04 (broken cloud); 2005-10-19 (overcast deck); 2005-04-01 (broken cloud); 2004-10-16 (haze / fog gradient (doubtful)); 2004-09-30 (overcast deck); 2002-10-11 (broken cloud, partial)

## 1. Registration (the method in miniature)

Correlation of high-passed night temperature with high-passed road share, mean over the 4 clear scenes: **r = -0.133 as delivered**, **r = +0.258 after registration**. The delivered images sit +200 m east and -110 m north of the map (search +-450 m in 10 m steps); they were shifted back by that amount.
Per scene at that shift: 2004-04-30 r = +0.288, 2004-04-07 r = +0.242, 2003-10-30 r = +0.103, 2003-10-21 r = +0.249.
The offset is consistent across scenes (about two pixels), as expected for an ASTER night geolocation error; there is no independent truth for it. All statistics below use the registered scenes.

## 2. Land cover inside elevation bands

### 2004-04-30 23:18 PDT (clear)

| band | built >= 70 % | tree >= 50 % | grass/shrub >= 50 % | built - tree | built - grass |
|---|---|---|---|---|---|
| flats < 40 m | 12.2 (n=1620) | 12.0 (n=43) | 12.2 (n=113) | +0.2 | +0.0 |
| slopes 40-150 m | 12.2 (n=1043) | 11.6 (n=404) | 12.4 (n=8) | +0.6 | -0.2 |
| hills >= 150 m | 11.5 (n=108) | 11.3 (n=1607) | 11.2 (n=198) | +0.1 | +0.3 |

Bay water (>= 90 %) 16.6 C vs flats 12.2 C: **water +4.4 K**.
Regression on land < 150 m away from the Bay (n = 3262); per 100 % share, relative to bare non-built ground: road -0.7 K (major roads +1.6 K more), other built-up -0.4 K, tree -1.3 K, grass/shrub -0.3 K; elevation -0.04 K per 100 m; R^2 = 0.22.

### 2004-04-07 23:13 PDT (clear)

| band | built >= 70 % | tree >= 50 % | grass/shrub >= 50 % | built - tree | built - grass |
|---|---|---|---|---|---|
| flats < 40 m | 9.1 (n=1620) | 9.1 (n=43) | 8.8 (n=113) | +0.1 | +0.3 |
| slopes 40-150 m | 9.1 (n=1043) | 8.8 (n=404) | 9.3 (n=8) | +0.3 | -0.2 |
| hills >= 150 m | 8.0 (n=108) | 7.3 (n=1607) | 6.8 (n=198) | +0.8 | +1.3 |

Bay water (>= 90 %) 12.6 C vs flats 9.1 C: **water +3.5 K**.
Regression on land < 150 m away from the Bay (n = 3262); per 100 % share, relative to bare non-built ground: road -0.7 K (major roads +0.8 K more), other built-up -0.7 K, tree -1.0 K, grass/shrub -0.5 K; elevation -0.10 K per 100 m; R^2 = 0.13.

### 2003-10-30 23:13 PDT (clear)

| band | built >= 70 % | tree >= 50 % | grass/shrub >= 50 % | built - tree | built - grass |
|---|---|---|---|---|---|
| flats < 40 m | 9.5 (n=1620) | 9.8 (n=43) | 10.1 (n=113) | -0.3 | -0.6 |
| slopes 40-150 m | 9.2 (n=1043) | 9.5 (n=404) | 9.3 (n=8) | -0.3 | -0.1 |
| hills >= 150 m | 8.3 (n=108) | 8.2 (n=1607) | 7.6 (n=198) | +0.1 | +0.7 |

Bay water (>= 90 %) 14.7 C vs flats 9.5 C: **water +5.2 K**.
Regression on land < 150 m away from the Bay (n = 3262); per 100 % share, relative to bare non-built ground: road -1.1 K (major roads +0.7 K more), other built-up -1.3 K, tree -0.6 K, grass/shrub -1.4 K; elevation -0.25 K per 100 m; R^2 = 0.09.

### 2003-10-21 23:18 PDT (clear, inversion)

| band | built >= 70 % | tree >= 50 % | grass/shrub >= 50 % | built - tree | built - grass |
|---|---|---|---|---|---|
| flats < 40 m | 13.4 (n=1620) | 13.4 (n=43) | 12.5 (n=113) | +0.0 | +0.9 |
| slopes 40-150 m | 13.7 (n=1043) | 14.6 (n=404) | 13.2 (n=8) | -0.8 | +0.6 |
| hills >= 150 m | 16.8 (n=108) | 17.9 (n=1607) | 16.2 (n=198) | -1.0 | +0.6 |

Bay water (>= 90 %) 17.0 C vs flats 13.4 C: **water +3.6 K**.
Regression on land < 150 m away from the Bay (n = 3262); per 100 % share, relative to bare non-built ground: road -1.0 K (major roads +1.4 K more), other built-up -0.6 K, tree -0.4 K, grass/shrub -0.8 K; elevation +1.38 K per 100 m; R^2 = 0.60.

## 3. Night temperature vs road share (registered, land < 150 m, away from the Bay)

| scene | road 0-5 % | road 5-15 % | road 15-25 % | road 25-35 % | road 35-100 % | top bin - bottom bin |
|---|---|---|---|---|---|---|
| 2004-04-30 | 12.11 (n=354) | 11.98 (n=1506) | 12.07 (n=1058) | 12.31 (n=237) | 12.81 (n=107) | +0.70 K |
| 2004-04-07 | 9.11 (n=354) | 8.98 (n=1506) | 9.05 (n=1058) | 9.19 (n=237) | 9.50 (n=107) | +0.39 K |
| 2003-10-30 | 9.45 (n=354) | 9.34 (n=1506) | 9.33 (n=1058) | 9.51 (n=237) | 9.59 (n=107) | +0.14 K |
| 2003-10-21 | 13.71 (n=354) | 13.50 (n=1506) | 13.52 (n=1058) | 13.67 (n=237) | 14.15 (n=107) | +0.44 K |

Major-road corridors (>= 15 % major road in the cell) vs low-road cells (< 2 % major, < 15 % any road):

- 2004-04-30: 12.51 C (n=309) vs 11.99 C (n=1778): **+0.53 K**
- 2004-04-07: 9.30 C (n=309) vs 8.99 C (n=1778): **+0.31 K**
- 2003-10-30: 9.51 C (n=309) vs 9.35 C (n=1778): **+0.17 K**
- 2003-10-21: 13.89 C (n=309) vs 13.51 C (n=1778): **+0.38 K**

