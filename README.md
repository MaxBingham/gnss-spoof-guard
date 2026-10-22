# gnss-spoof-guard

GNSS-free drone navigation that knows when GNSS is lying. Built for the SF Hackathon (Oct 24, 2026) on top of the [gnss-free-nav prototype](https://gnss-free-nav.pages.dev) (password-protected).

## Pitch

Most GNSS-free navigation demos show a single dot. This one shows a cloud of hypotheses that collapses onto the truth, reports honestly how sure it is, and catches a GNSS spoofer that would otherwise fly the drone into a building.

A particle filter localizes the drone by matching unlabeled camera detections (buildings, trees, towers) against an onboard OpenStreetMap landmark map. It never sees GNSS, so it acts as an independent witness: a spoof detector compares GNSS against it, rejects lying fixes, and falls back to vision.

## Planned features

- **Particle filter localizer** that replaces the prototype's mock localizer (truth plus noise) with real inference over ID-free landmark detections, including an `AMBIGUOUS` state when the belief is multimodal.
- **Simulated detector** that projects map landmarks, strips their IDs, and adds pixel noise, misses and clutter.
- **GNSS sensor and attacker** with slow drag-off spoofing, jump spoofing and jamming.
- **SpoofGuard**, which combines a chi-square innovation gate with a CUSUM on the GNSS-minus-particle-filter residual, driving a `TRUSTED → SUSPECT → REJECTED` trust state machine.
- **Dual-drone comparison**: a naive GNSS-trusting drone and a protected drone flown side by side under the same attack.
- **Integrity metrics**: 2σ coverage, effective sample size, spoof detection latency and false alarms.
- **Disturbance controls**: spoof slider, jam, camera blackout, kidnap and wind.

## Stack

TypeScript in the browser (Vite), three.js and Leaflet. No backend.

## Contract

`sensors/`, `estimation/`, `pipeline/`, `sim/`, and `ui/` are the shared contract. `sim/` is headless. `Conditions` carries one weather intensity, a wind force (speed, compass direction, and optional rotation), camera and lidar occlusion switches, a GNSS jam switch, and a GNSS spoof drift rate where zero is off. The pass-through pipeline forwards a GNSS fix into the 6-state Kalman filter.

```bash
npm test
npm run scenario -- --seed 42
```

## Disclosure

The simulator, drone dynamics, sensor models, Kalman filter, planner and Berkeley map package come from the earlier gnss-free-nav prototype. Everything listed under "Planned features" is built during the hackathon.
