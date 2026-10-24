# Night notes for the team (night-research, v2, 2026-10-24 00:35)

Short and concrete, for night-thermal / night-nav / night-demo. The full report with sources follows in
`docs/NIGHT_RESEARCH.md`. Numbers marked **model** come from my 1-D surface energy balance
(`docs/night_research/surface_energy_balance.py`, table in `docs/night_research/seb_output.txt`). Numbers marked
**meas.** are field measurements from the literature. Where the two disagree I give a range.

## 1. Apparent LWIR temperature at 23:00, clear, calm, after a hot summer day

"Apparent" = what an 8-14 um camera reads with emissivity 1 (emitted plus reflected sky). Air = 2 m air temperature.

| Class | Berkeley 23 h (model) | Ukraine 23 h (model) | Offset vs air, use this | Prior sigma | Notes |
|---|---|---|---|---|---|
| air (2 m) | 18.7 C | 21.1 C | 0 | - | Berkeley normal July night ~15-16 C, hot day 28 C |
| **asphalt road / parking** | 25.6 | 28.0 | **+4 .. +8 K** | 2 K | warmest common surface all night |
| concrete pavement / sidewalk | 24.0 | 26.5 | +3 .. +6 K | 2 K | 1-2 K below asphalt (higher albedo) |
| gravel / dirt road, compacted | 20.2 | 23.0 | -2 .. +2 K | 3 K | **unreliable**; meas.: most dirt roads invisible at night, a few cool |
| bare soil, dry, loose | 16.5 | 19.6 | -4 .. +1 K | 3 K | |
| grass / lawn / dry grass | 15.8 | 18.8 | **-5 .. -2 K** | 2 K | coldest natural surface on clear calm nights |
| trees (canopy top) | 16.9 | 19.5 | -2 .. 0 K | 1.5 K | **warmer than lawn at night**, hides the road below |
| water (pond, river) | 16.5 (SF Bay ~16-17 C) | 23.5 | ~ daily mean air, Ukraine +2 K | 2 K | **warm, not cool**: at night water is warmer than vegetation |
| roof, light-weight + insulated (shingle, membrane) | 11.3 | 14.6 | **-6 .. -2 K** (typ. -4..-5; -11 K extreme) | 4 K | **COLD**: radiates to the sky, no heat storage; meas. 1-6 K below air (Khedari 2000) |
| wet asphalt (sprinkler, fresh) | - | - | dry asphalt -3..-6 K (estimate, no night measurement) | 2 K | recovers over 1-3 h |
| roof, heavy (concrete slab, gravel) | 23.3 | 25.8 | +2 .. +5 K | 3 K | campus / commercial buildings |
| roof, bare metal (low emissivity) | 5.9 | 9.7 | -15 .. -8 K | 5 K | reflects the cold sky; looks coldest of all |
| parked car (since afternoon) | - | - | -5 .. 0 K | - | thin metal + glass reflect the sky: **cold blob on warm road** |

Contrast at 23 h:
- **asphalt minus grass: 5-10 K.** Meas. Erzurum, clear calm August: 5.4 K at 22:00 (smallest of the day; 23 K at 13:00).
  Model: 9.8 K Berkeley, 9.2 K Ukraine. A drone thermal survey reports asphalt clearly warmer than grass at 23:00.
- asphalt minus trees: 6-9 K (model). Asphalt minus dirt road: ~5 K (model; in reality often less).
- Through the night (model, Berkeley): asphalt - lawn 11 K at 21 h, 10 K at 23 h, 7 K at 03 h. **No crossover** between
  asphalt and grass during the night; contrast is smallest ~1 h after sunrise. Crossovers that do happen: trees vs
  lawn and soil vs grass around sunset and sunrise. 23:00 (2.5 h after sunset) is in the stable night regime.
- Light breeze (~3 m/s): asphalt - lawn drops 9.8 -> 6.4 K. Overcast all day + night or rain: contrast mostly gone
  (< 1-2 K). Not in tonight's scenario.

## 2. Camera numbers (640 x 512, 50 deg HFOV, 60 m AGL)

- Real part: FLIR Boson 640, 12 um pixels, 8-14 um, 50 deg HFOV = 8.7 mm lens (9.2 mm compact). VFOV = 40.9 deg.
- Footprint at 60 m AGL: **56.0 m x 44.7 m**, **8.7 cm/px** (IFOV 1.36 mrad). At 30 m AGL: 28 x 22 m.
- NETD: Boson <= 40 mK (industrial), <= 50 mK (professional), <= 60 mK (consumer). **50 mK is right.**
- Scene contrast 5-10 K = 100-200 x NETD. **Noise is not the problem**; mismatch between prior and reality is.
- Uncooled cameras: absolute accuracy is poor (typ. +-5 K or 5 %) and the offset drifts. **Match normalized
  images (zero-mean NCC, gradients, ranks), never absolute Kelvin.**
- Frame rate: 60 Hz baseline, 30 Hz selectable; export variants are 9 Hz (fine for us at 5 Hz).
- Motion blur: bolometer time constant ~10 ms; 8 m/s cruise -> ~8 cm = 1 px. Negligible.
- Path absorption over 60 m: well under 0.5 K. Ignore.
- A residential street (9-12 m) is 100-140 px wide. A 56 m frame over a park or a big roof can contain **no road at
  all**.

## 3. Pitfalls (please model these as truth deviations or as large prior sigma)

1. **Roofs are not "warm".** Insulated residential roofs are among the coldest surfaces at night (meas.: membranes
   over insulation run up to 11 K below air). Only heavy concrete roofs stay warm. Bare metal looks very cold.
   Prior: give roofs large sigma or use only their edges, not their sign.
2. **Water is warm at night** (warmer than grass and trees). Do not put it in the "cool" group.
3. **Trees are warmer than lawns** at night and hide the road. Pavement that was **shaded in the afternoon** (under
   trees, east side of tall buildings) is 2-5 K cooler at night than sunlit asphalt.
4. **Parked cars** along the curb = cold blobs (they reflect the sky). Moving cars: hot engine / exhaust spots.
5. **Sprinklers**: Berkeley lawns are irrigated at night; wet sidewalk and road segments cool by evaporation.
6. **Dirt / gravel roads** (Ukraine villages, field tracks) give weak and unpredictable contrast. Only roads with a
   paved `surface` tag (asphalt, concrete, paved) should be "warm" in the prior.
7. Emissivity: asphalt 0.95, concrete 0.92-0.94, grass 0.98, water 0.985, bare metal 0.05-0.3. A clear night sky is
   ~20-30 K colder than the air, so low-emissivity surfaces look cold whatever their real temperature.
8. Berkeley reality check: summer nights there are often under the marine layer (fog, low cloud). Hot, clear nights
   are more typical in September / October. Fine for the scenario; mention it if the jury asks.

## 4. Suggestions per agent

- **night-thermal**: per-class offset vs air from the table plus a damped-sinusoid / exponential night cooling.
  Asphalt cools ~2 K/h from sunset to 23 h, then ~1.4 K/h until dawn (model); lawn ~1.5 K/h then ~0.8 K/h. Truth deviations worth adding: parked cars
  (cold), shaded-in-afternoon road segments (-2..-5 K), wet patches, per-road offset +-1.5 K, roof class confusion.
- **night-nav**: match gradients / zero-mean NCC, not Kelvin. Expect frames with no road: return None, do not force
  a fix. A single straight road fixes only the cross-track position (send a 1-D fix or an inflated along-track
  covariance). Junctions give the along-track position.
- **night-demo (Ukraine)**: count paved vs unpaved by OSM `surface` (missing tag on `track` = unpaved; missing on
  `residential` in villages = often unpaved). Route over paved roads.

## 5. Real systems in one line

Fielded camera-to-map navigation is reported at ~10-30 m (vendor claims: Maxar Raptor < 10 m RMSE, Honeywell 10 m
CEP50). The only night thermal flight I found with a number: Inertial Labs + Maxar Raptor, FLIR Boson+, night,
< 15 m RMS over ~58 km. Research thermal-to-satellite geo-localization: 7-15 m at 1 m/px. Our 5 m target at 8.7 cm/px
over mapped roads is ambitious but not unreasonable. Sources in NIGHT_RESEARCH.md.

## 6. Real-data check (v2): ASTER night scenes over Berkeley, 23:13-23:18 PDT, 90 m

Four clear nights (Apr / Oct; no clear summer night exists, the marine layer covers them). Details: NIGHT_RESEARCH.md
section 6.
- **Roads warm, confirmed, but small at 90 m:** road-dense cells are +0.1..+0.7 K warmer and major-road corridors
  +0.2..+0.5 K. This implies only ~1-3 K pure road contrast on those mild nights, below the 5-10 K after a hot day.
  **Please add one eval row with the road contrast scaled down to ~0.3-0.5x.**
- **Water is warm:** +3.5..+5.2 K above the land, every night.
- Built vs tree vs grass at 90 m: < 1 K apart (warm roads and cold roofs cancel). Do not make "built-up" a warm class.
- **Terrain:** inversion nights make hills warmer (+1.4 K / 100 m). This is an unmodelled low-frequency signal, so
  **high-pass or zero-mean the frame before matching.**
- **Registration works:** correlating the night image with the road map recovers a 230 m error in the satellite
  scene's geolocation (r -0.13 -> +0.26). The correlation peak is a north-south ridge along the arterials:
  expect along-road ambiguity in flight too.

## Changelog
- v2 (00:35): ASTER real-data check; light insulated roofs typical -4..-6 K, -11 K extreme (Khedari 2000: 1-6 K
  below ambient; NIST: up to 11 K); wet asphalt estimate.
- v1 (00:05): first version.
