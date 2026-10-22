import { describe, expect, it } from "vitest";
import { positionErrorM, uncertaintyRadiusM } from "./geometry.ts";
import { initialLevers } from "./levers.ts";
import { scenarioFlight } from "./scenario-flight.ts";
import { errorSeries, uncertaintySeries } from "./series.ts";

describe("scenarioFlight", () => {
  it("plays the seed-42 baseline with truth, estimate, uncertainty, and lever values", () => {
    const flight = scenarioFlight(initialLevers());
    expect(flight.conditions.gnss.spoofRateMS).toBe(0);
    expect(flight.conditions.weather.intensity).toBe(0);
    expect(flight.samples.length).toBeGreaterThan(20);
    const last = flight.samples.at(-1)!;
    expect(last.truth.x).toBeGreaterThan(100);
    expect(positionErrorM(last.truth, last.trace.estimate)).toBeLessThan(5);
    const error = errorSeries(flight.samples);
    const sigma = uncertaintySeries(flight.samples);
    expect(error.every((point) => Number.isFinite(point.value))).toBe(true);
    expect(sigma.some((point) => point.value > 0)).toBe(true);
    expect(error.at(-1)!.t).toBe(last.trace.t);
    for (let i = 1; i < flight.samples.length; i++) {
      expect(flight.samples[i]!.trace.t).toBeGreaterThan(flight.samples[i - 1]!.trace.t);
    }
  });

  it("shows the spoofed estimate leaving the truth, which the hand-written mock does not compute", () => {
    const flight = scenarioFlight(
      { ...initialLevers(), spoofRateMS: 3, spoofDirectionDeg: 90 },
      { maxTimeS: 8 },
    );
    const last = flight.samples.at(-1)!;
    expect(flight.conditions.gnss.spoofRateMS).toBe(3);
    expect(flight.conditions.gnss.spoofDirectionRad).toBeCloseTo(Math.PI / 2);
    expect(last.trace.estimate.x - last.truth.x).toBeGreaterThan(18);
    expect(uncertaintyRadiusM(last.trace.estimate.cov)).not.toBeCloseTo(3);
    expect(flight.samples.some((sample) => sample.trace.fixes.some((fix) => fix.source === "gnss"))).toBe(true);
  });

  it("reports camera and lidar loss on the trace the dashboard draws", () => {
    const dark = scenarioFlight({ ...initialLevers(), cameraLost: true }, { maxTimeS: 1 });
    const blind = scenarioFlight({ ...initialLevers(), lidarLost: true }, { maxTimeS: 1 });
    expect(dark.conditions.camera.occluded).toBe(true);
    expect(dark.samples.every((sample) => sample.cameraBlobs === null && sample.lidarBeams !== null)).toBe(true);
    expect(blind.conditions.lidar.occluded).toBe(true);
    expect(blind.samples.every((sample) => sample.lidarReturns === null && sample.cameraBlobs !== null)).toBe(true);
  });
});
