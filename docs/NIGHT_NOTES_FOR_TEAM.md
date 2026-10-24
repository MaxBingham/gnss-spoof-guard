# Night notes for the team (night-research, v4, 2026-10-24 01:15)

Short and concrete, for night-thermal / night-nav / night-demo. The full report with sources is
`docs/NIGHT_RESEARCH.md`.
- Numbers marked **model** come from my 1-D surface energy balance (`docs/night_research/surface_energy_balance.py`,
  table in `docs/night_research/seb_output.txt`).
- Numbers marked **meas.** are field measurements from the literature.
- Where the two disagree I give a range.
- v4 applies an independent fact-check (see the changelog).

## 1. Apparent LWIR temperature at 23:00, clear, calm, after a hot summer day

"Apparent" = what an 8-14 um camera reads with emissivity 1 (emitted plus reflected sky). Air = 2 m air temperature.

| Class | Berkeley 23 h (model) | Ukraine 23 h (model) | Offset vs air, use this | Prior sigma | Notes |
|---|---|---|---|---|---|
| air (2 m) | 18.7 C | 21.1 C | 0 | - | Berkeley normal July night ~15-16 C, hot day 28 C |
| **asphalt road / parking** | 25.6 | 28.0 | **+4 .. +8 K** | 2 K | warmest common surface all night |
| concrete pavement / sidewalk | 24.0 | 26.5 | +3 .. +6 K | 2 K | 1-2 K below asphalt (higher albedo) |
| gravel / dirt road, compacted | 20.2 | 23.0 | -2 .. +2 K | 3 K | **unreliable**; satellite data (section 6): no evidence they are warm, mostly slightly cooler than surroundings |
| bare soil, dry, loose | 16.5 | 19.6 | -4 .. +1 K | 3 K | |
| grass / lawn / dry grass | 15.8 | 18.8 | **-5 .. -2 K** | 2 K | coldest natural surface on clear calm nights |
| trees (canopy top) | 16.9 | 19.5 | -2 .. 0 K | 1.5 K | **warmer than lawn at night**, hides the road below |
| water (pond, river, bay) | 16.5 (SF Bay ~16-17 C) | 23.5 | ~ its daily-mean temperature | **3 K** | **not a cool class**: usually warmer than grass; a cold bay can sit below the air |
| roof, light-weight + insulated (shingle, membrane) | 11.3 | 14.6 | **-6 .. -2 K** (model -7; -11 K extreme) | 4 K | **COLD**: radiates to the sky, no heat storage; meas. 1-6 K below air (Khedari 2000) |
| wet asphalt (sprinkler, fresh) | - | - | dry asphalt -3..-6 K (estimate, no night measurement) | 2 K | recovers over 1-3 h |
| roof, heavy (concrete slab, gravel) | 23.3 | 25.8 | +2 .. +5 K | 3 K | campus / commercial buildings |
| roof, bare metal (low emissivity) | 5.9 | 9.7 | -15 .. -8 K | 5 K | reflects the cold sky; looks coldest of all |
| parked car (since afternoon) | - | - | -5 .. 0 K | - | thin metal + glass reflect the sky: **cold blob on warm road** |

Contrast at 23 h:
- **asphalt minus grass: ~5 K measured, up to ~10 K in the model.**
  - Meas.: Erzurum, clear calm August, 5.4 K at 22:00 (smallest of the day; 23 K at 13:00). A midnight UAV survey
    reports roads ~4.7 K above grass.
  - Model: 9.8 K Berkeley, 9.2 K Ukraine, about twice the measurements.
- asphalt minus trees: 6-9 K (model). Asphalt minus dirt road: ~5 K (model; in reality often less).
- Through the night (model, Berkeley): asphalt - lawn 11 K at 21 h, 10 K at 23 h, 7 K at 03 h.
  - **No crossover** between asphalt and grass during the night; contrast is smallest ~1 h after sunrise.
  - Crossovers that do happen: trees vs lawn and soil vs grass, around sunset and sunrise.
  - 23:00 (~2.5 h after sunset in Berkeley, ~2 h in Ukraine) is in the stable night regime.
- Light breeze (~3 m/s): asphalt - lawn drops 9.8 -> 6.4 K (model).
- Overcast all day and night: est. 1-3 K. Rain: ~0-2 K. Not in tonight's scenario.

## 2. Camera numbers (640 x 512, 50 deg HFOV, 60 m AGL)

- Real part: FLIR Boson 640, 12 um pixels, 8-14 um, 50 deg HFOV = 8.7 mm lens (9.2 mm compact / radiometric).
  VFOV = 40.9 deg.
- Footprint at 60 m AGL: **56.0 m x 44.8 m**, **8.7 cm/px** (IFOV 1.46 mrad for an exact 50 deg pinhole). At 30 m
  AGL: 28 x 22 m.
- NETD: Boson <= 40 mK (industrial), <= 50 mK (professional), <= 60 mK (consumer). **50 mK is right.**
- Scene contrast 2-10 K = 40-200 x NETD. **Noise is not the problem**; mismatch between prior and reality is.
- Uncooled cameras: absolute accuracy is poor (typ. +-5 K or 5 %) and the offset drifts. **Match normalized
  images (zero-mean NCC, gradients, ranks), never absolute Kelvin.**
- Frame rate: 60 Hz baseline, 30 Hz selectable, 9 Hz variant (fine for us at 5 Hz).
- Motion blur: typical bolometer time constant ~10 ms; 8 m/s cruise -> ~8 cm = 1 px. Negligible.
- Path absorption over 60 m: well under 0.5 K. Ignore.
- A residential street (9-12 m) is 100-140 px wide. A 56 m frame over a park or a big roof can contain **no road at
  all**.

## 3. Pitfalls (please model these as truth deviations or as large prior sigma)

1. **Roofs are not "warm".** Insulated residential roofs are among the coldest surfaces at night (meas. 1-6 K below
   air, up to 11 K on a clear winter night). Only heavy concrete roofs stay warm. Bare metal looks very cold.
   Prior: give roofs large sigma or use only their edges, not their sign.
2. **Water is not cool** (usually warmer than grass at night), but not reliably the warmest either. Large sigma.
3. **Trees are warmer than lawns** at night and hide the road. Pavement that was **shaded in the afternoon** (under
   trees, east side of tall buildings) is 2-5 K cooler at night than sunlit asphalt (estimate).
4. **Parked cars** along the curb = cold blobs (they reflect the sky). Moving cars: hot engine / exhaust spots.
5. **Sprinklers**: Berkeley lawns are irrigated at night; wet sidewalk and road segments cool by evaporation.
6. **Dirt / gravel roads** (Ukraine villages, field tracks) give weak and unpredictable contrast. Only roads with a
   paved `surface` tag (or untagged motorway..secondary) should be "warm" in the prior.
7. Emissivity:
   - asphalt 0.95, concrete 0.92-0.97, grass 0.98, water 0.985;
   - metal: new galvanized ~0.2, polished < 0.1, weathered / painted 0.6-0.9.
   - The clear night sky in the 8-14 um window is far colder than the air, so low-emissivity surfaces look cold
     whatever their real temperature.
8. Berkeley reality check: summer nights there are often under the marine layer (fog, low cloud; climatology, not
   shown in our data). Hot, clear nights are more typical in September / October. Fine for the scenario; mention
   it if the jury asks.

## 4. Suggestions per agent

- **night-thermal**: per-class offset vs air from the table plus a damped-sinusoid / exponential night cooling.
  - Cooling (model): asphalt ~2 K/h from sunset to 23 h, then ~1.4 K/h until dawn; lawn ~1.5 K/h, then ~0.8 K/h.
  - Truth deviations worth adding:
    - parked cars (cold);
    - shaded-in-afternoon road segments (-2..-5 K);
    - wet patches;
    - per-road offset +-1.5 K;
    - roof class confusion.
- **night-nav**:
  - Match gradients / zero-mean NCC, not Kelvin.
  - Expect frames with no road: return None, do not force a fix.
  - A single straight road fixes only the cross-track position: send a 1-D fix or an inflated along-track
    covariance. Junctions give the along-track position.
- **night-demo (Ukraine)**: count paved vs unpaved by OSM `surface` (missing tag on `track` = unpaved; missing on
  `residential` in villages = often unpaved). Route over paved roads.

## 5. Real systems in one line

- Fielded camera-to-map navigation that publishes numbers quotes ~10-30 m (vendor claims: Maxar Raptor < 10 m RMSE,
  UAV Navigation ~30 m, Twist OSCAR 20 m). Honeywell VAN publishes no figure ("GPS-like").
- The only night thermal flight I found with a number: Inertial Labs + Maxar Raptor, FLIR Boson+, night, < 15 m RMS
  over ~58 km.
- Research thermal-to-satellite geo-localization: 7-15 m at 1 m/px.
- Our 5 m target at 8.7 cm/px over mapped roads is ambitious but not unreasonable. Sources in NIGHT_RESEARCH.md.

## 6. Real-data check: ASTER night thermal (90 m, ~23 h local, 2001-2006), NIGHT_RESEARCH.md section 6

Each night image was registered to a road-share raster built from OSM / Overture roads.

- **Hot summer nights (Fresno 6 nights, Davis 1): the paved road grid is plainly visible even at 90 m.**
  - Paved-road coefficient (per 100 % share, holding built-up, vegetation and elevation fixed): Fresno
    +0.4..+3.9 K, Davis +3.2 K.
  - Road-dense cells sit +1.0..+1.9 K above road-free cells, but that comparison is confounded with town vs fields.
  - Davis town is +3.2 K above the fields.
  - These 90 m numbers are lower bounds (pixel mixing, blur), and they are below the model's 5-10 K.
- **Mild nights (Berkeley, 4 clear nights in Apr / Oct):** road-dense cells only +0.1..+0.7 K, major corridors
  +0.2..+0.5 K. The 0.4x contrast eval row stands for this case.
- **Ukraine (Vinnytsia, September):** one of two nights shows +0.5..+0.9 K for road-dense cells; the colder night
  (5 C) none. The river is +1.9 K over the town.
- **No evidence that unpaved roads are warm:** binned cells with only unpaved roads are -0.9..+0.2 K vs road-free
  cells, mostly cooler (Davis, Vinnytsia). Keep the paved-only rule.
- **Water:** the Bay is +3.5..+5.2 K over Berkeley's land on every clear night. In Fresno small canals / ponds were
  about equal to the town on half the nights.
- Built vs tree vs grass at 90 m in Berkeley: mostly < 1 K apart, up to 1.3 K (warm roads and cold roofs cancel).
  Do not make "built-up" a warm class.
- **Terrain:** on an inversion night the hills were warmer (+1.4 K / 100 m). This is an unmodelled low-frequency
  signal, so **high-pass or zero-mean the frame before matching.**
- **Registration:**
  - On every site, correlating the night image with the paved-road raster converges on a consistent offset: the
    delivered images sit 80-310 m east and 40-160 m south of the map. That fits an ASTER night geolocation error;
    there is no independent truth.
  - Correlation r ~ -0.1..0 -> +0.26..+0.33 (Vinnytsia only +0.10).
  - The correlation surface shows a **north-south ridge** (Berkeley arterials) and **grid side lobes** (Fresno).
    Along-road and periodic ambiguity are real, so keep the 1-D fix and the second-peak test.
- No usable summer night scene exists for the ukraine-1 area (the only one, 2006-06-01, is cloudy).

## Changelog
- v4 (01:15): independent fact-check applied.
  - The Honeywell "10 m CEP50" was unsupported and is removed.
  - asphalt-grass is now "~5 K measured, up to 10 K model".
  - "Pure road contrast" unmixing dropped (confounded); regression coefficients quoted instead.
  - Marine layer marked as climatology, not shown by our data.
  - Water: "not cool" with large sigma.
  - Unpaved: "-0.9..+0.2 K".
  - Ukraine: one of two nights.
  - Footprint 44.8 m, IFOV 1.46 mrad, emissivities unified.
- v3 (00:50): hot-summer and Ukraine sites.
- v2 (00:35): ASTER Berkeley check; light insulated roofs typical -4..-6 K, -11 K extreme; wet asphalt estimate.
- v1 (00:05): first version.
