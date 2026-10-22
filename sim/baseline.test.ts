import { describe, expect, it } from "vitest";
import { CLEAR } from "../sensors/types.ts";
import type { Conditions } from "../sensors/types.ts";
import { runScenario } from "./scenario.ts";
import type { ScenarioSample } from "./scenario.ts";

function horizontalError(sample: ScenarioSample): number {
  return Math.hypot(sample.trace.estimate.x - sample.truth.x, sample.trace.estimate.y - sample.truth.y);
}

function withConditions(over: {
  weather?: Conditions["weather"];
  wind?: Conditions["wind"];
  camera?: Conditions["camera"];
  lidar?: Conditions["lidar"];
  gnss?: Partial<Conditions["gnss"]>;
}): Conditions {
  return {
    ...CLEAR,
    ...over,
    gnss: { ...CLEAR.gnss, ...over.gnss },
  };
}

describe("naive baseline", () => {
  it("trusts a GNSS fix when one exists and coasts with no position fix otherwise", () => {
    const result = runScenario({ seed: 42, maxTimeS: 2 });
    const fixes = result.samples.filter((sample) => sample.frame.gnss);
    const coasts = result.samples.filter((sample) => !sample.frame.gnss);
    expect(fixes.length).toBeGreaterThan(1);
    expect(coasts.length).toBeGreaterThan(fixes.length);
    for (const sample of fixes) {
      expect(sample.trace.fixes).toEqual([
        {
          source: "gnss",
          x: sample.frame.gnss!.x,
          y: sample.frame.gnss!.y,
          cov: [...sample.frame.gnss!.cov],
        },
      ]);
    }
    expect(coasts.every((sample) => sample.trace.fixes.length === 0)).toBe(true);
    expect(result.samples.some((sample) => sample.frame.camera && sample.frame.lidar)).toBe(true);
    expect(result.samples.every((sample) => sample.trace.fixes.every((fix) => fix.source === "gnss"))).toBe(true);
  });

  it("pulls the estimate off the truth as the spoof drift rises", () => {
    const clear = runScenario({ seed: 42, maxTimeS: 8 });
    const spoof = runScenario({
      seed: 42,
      maxTimeS: 8,
      conditions: withConditions({ gnss: { spoofRateMS: 3, spoofDirectionRad: Math.PI / 2 } }),
    });
    const clearLast = clear.samples.at(-1)!;
    const spoofLast = spoof.samples.at(-1)!;
    expect(horizontalError(clearLast)).toBeLessThan(2);
    expect(spoofLast.trace.estimate.x - spoofLast.truth.x).toBeGreaterThan(18);
    expect(Math.abs(spoofLast.trace.estimate.y - spoofLast.truth.y)).toBeLessThan(3);
    expect(horizontalError(spoofLast)).toBeGreaterThan(horizontalError(clearLast) + 15);

    const trusted = spoof.samples.filter((sample) => sample.frame.gnss).at(-1)!;
    const gapToFix = Math.hypot(
      trusted.trace.estimate.x - trusted.frame.gnss!.x,
      trusted.trace.estimate.y - trusted.frame.gnss!.y,
    );
    expect(gapToFix).toBeLessThan(horizontalError(trusted));
  });

  it("removes GNSS when jammed and coasts on the filter", () => {
    const clear = runScenario({ seed: 42, maxTimeS: 15 });
    const jammed = runScenario({
      seed: 42,
      maxTimeS: 15,
      conditions: withConditions({ gnss: { jammed: true, spoofRateMS: 4 } }),
    });
    expect(jammed.samples.every((sample) => sample.frame.gnss === undefined && sample.trace.fixes.length === 0)).toBe(
      true,
    );
    expect(jammed.samples.some((sample) => sample.frame.camera && sample.frame.lidar)).toBe(true);
    const jammedCov = jammed.samples.at(-1)!.trace.estimate.cov[0];
    const clearCov = clear.samples.at(-1)!.trace.estimate.cov[0];
    expect(jammedCov).toBeGreaterThan(clearCov * 10);
    expect(jammedCov).toBeGreaterThan(100);
    expect(jammed.samples.at(-1)!.trace.estimate.x).toBeGreaterThan(10);
  });

  it("worsens camera and lidar observations as weather noise and dropout rise", () => {
    const clear = runScenario({ seed: 42, maxTimeS: 25 });
    const storm = runScenario({
      seed: 42,
      maxTimeS: 25,
      conditions: withConditions({ weather: { intensity: 1 } }),
    });
    expect(storm.samples.at(-1)!.truth.x).toBe(clear.samples.at(-1)!.truth.x);
    expect(storm.samples.at(-1)!.truth.y).toBe(clear.samples.at(-1)!.truth.y);

    let clearBlobs = 0;
    let stormBlobs = 0;
    let shifted = 0;
    let lidarDropouts = 0;
    let lidarNoisy = 0;
    for (let i = 0; i < clear.samples.length; i++) {
      const calmCamera = clear.samples[i]!.frame.camera!.landmarks;
      const stormCamera = storm.samples[i]!.frame.camera!.landmarks;
      clearBlobs += calmCamera.length;
      stormBlobs += stormCamera.length;
      for (const blob of stormCamera) {
        const exact = calmCamera.some((point) => point.type === blob.type && point.u === blob.u && point.v === blob.v);
        if (!exact) shifted += 1;
      }
      const calmBeams = clear.samples[i]!.frame.lidar!.beams;
      const stormBeams = storm.samples[i]!.frame.lidar!.beams;
      for (let beam = 0; beam < calmBeams.length; beam++) {
        const degraded = stormBeams[beam]!.rangeM;
        const clean = calmBeams[beam]!.rangeM;
        if (degraded === null) lidarDropouts += 1;
        else if (clean !== null && Math.abs(degraded - clean) > 1e-3) lidarNoisy += 1;
      }
    }
    expect(clearBlobs).toBeGreaterThan(100);
    expect(stormBlobs).toBeGreaterThan(0);
    expect(stormBlobs).toBeLessThan(clearBlobs / 2);
    expect(shifted).toBe(stormBlobs);
    expect(lidarDropouts).toBeGreaterThan(1000);
    expect(lidarNoisy).toBeGreaterThan(1000);
  });

  it("pushes the true path along the wind", () => {
    const calm = runScenario({ seed: 42, maxTimeS: 8 });
    const wind = runScenario({
      seed: 42,
      maxTimeS: 8,
      conditions: withConditions({ wind: { speedMS: 10, directionRad: Math.PI / 2, rotationRadS: 0 } }),
    });
    const calmTruth = calm.samples.at(-1)!.truth;
    const windTruth = wind.samples.at(-1)!.truth;
    expect(windTruth.x).toBeGreaterThan(calmTruth.x + 2);
    expect(Math.abs(windTruth.y - calmTruth.y)).toBeLessThan(1);
  });

  it("deletes camera observations when the camera-loss switch is on", () => {
    const result = runScenario({
      seed: 42,
      maxTimeS: 1,
      conditions: withConditions({ camera: { occluded: true } }),
    });
    expect(result.samples.every((sample) => sample.frame.camera === undefined)).toBe(true);
    expect(result.samples.every((sample) => sample.frame.lidar !== undefined)).toBe(true);
  });

  it("deletes lidar observations when the lidar-loss switch is on", () => {
    const result = runScenario({
      seed: 42,
      maxTimeS: 1,
      conditions: withConditions({ lidar: { occluded: true } }),
    });
    expect(result.samples.every((sample) => sample.frame.lidar === undefined)).toBe(true);
    expect(result.samples.every((sample) => sample.frame.camera !== undefined)).toBe(true);
  });
});
