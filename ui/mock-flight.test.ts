import { describe, expect, it } from "vitest";
import { summarizeTrace } from "./index.ts";
import { uncertaintyRadiusM } from "./geometry.ts";
import { positionErrorM } from "./geometry.ts";
import { initialLevers } from "./levers.ts";
import { mockFlight } from "./mock-flight.ts";

describe("mockFlight", () => {
  it("tracks truth while GNSS is clear", () => {
    const flight = mockFlight(initialLevers());
    const last = flight.samples.at(-1)!;
    expect(last.trace.fixes.some((fix) => fix.source === "gnss")).toBe(true);
    expect(positionErrorM(last.truth, last.trace.estimate)).toBeCloseTo(0);
    expect(uncertaintyRadiusM(last.trace.estimate.cov)).toBeCloseTo(3);
    expect(summarizeTrace(last.trace)).toContain("gnss");
  });

  it("drops the GNSS fix when the jam switch is on", () => {
    const flight = mockFlight({ ...initialLevers(), gnssJammed: true, cameraLost: true, lidarLost: true });
    expect(flight.samples.every((sample) => sample.trace.fixes.length === 0)).toBe(true);
    const start = uncertaintyRadiusM(flight.samples[0].trace.estimate.cov);
    const end = uncertaintyRadiusM(flight.samples.at(-1)!.trace.estimate.cov);
    expect(end).toBeGreaterThan(start);
  });

  it("drifts the estimate off truth at the spoof rate", () => {
    const flight = mockFlight({ ...initialLevers(), spoofRateMS: 2, spoofDirectionDeg: 90 });
    const at10 = flight.samples.find((sample) => sample.trace.t === 10)!;
    expect(at10.trace.estimate.x - at10.truth.x).toBeCloseTo(20);
    expect(at10.trace.estimate.y - at10.truth.y).toBeCloseTo(0);
    expect(positionErrorM(at10.truth, at10.trace.estimate)).toBeCloseTo(20);
    expect(uncertaintyRadiusM(at10.trace.estimate.cov)).toBeCloseTo(3);
  });

  it("leaves spoof off when the drift lever is zero", () => {
    const flight = mockFlight({ ...initialLevers(), spoofRateMS: 0, spoofDirectionDeg: 90 });
    const last = flight.samples.at(-1)!;
    expect(last.trace.estimate.x).toBeCloseTo(last.truth.x);
  });

  it("moves truth with the wind while a jammed estimate stays put", () => {
    const calm = mockFlight({ ...initialLevers(), gnssJammed: true, cameraLost: true, lidarLost: true });
    const windy = mockFlight({
      ...initialLevers(),
      gnssJammed: true,
      cameraLost: true,
      lidarLost: true,
      windSpeedMS: 10,
      windDirectionDeg: 90,
    });
    const calmTruth = calm.samples.at(-1)!.truth;
    const windyTruth = windy.samples.at(-1)!.truth;
    expect(windyTruth.x).toBeGreaterThan(calmTruth.x);
    expect(windy.samples.at(-1)!.trace.estimate.x).toBeCloseTo(0);
  });

  it("widens the camera fix as weather noise rises, and drops it when the camera is lost", () => {
    const jammed = { ...initialLevers(), gnssJammed: true, lidarLost: true };
    const calm = mockFlight({ ...jammed, weather: 0, cameraLost: false });
    const storm = mockFlight({ ...jammed, weather: 1, cameraLost: false });
    const dark = mockFlight({ ...jammed, weather: 1, cameraLost: true });
    const calmCam = calm.samples[10].trace.fixes.find((fix) => fix.source === "camera")!;
    const stormCam = storm.samples[10].trace.fixes.find((fix) => fix.source === "camera")!;
    expect(stormCam.cov[0]).toBeGreaterThan(calmCam.cov[0]);
    expect(uncertaintyRadiusM(storm.samples[10].trace.estimate.cov)).toBeGreaterThan(
      uncertaintyRadiusM(calm.samples[10].trace.estimate.cov),
    );
    expect(dark.samples.every((sample) => sample.trace.fixes.every((fix) => fix.source !== "camera"))).toBe(true);
  });

  it("omits lidar when that sensor is lost", () => {
    const flight = mockFlight({ ...initialLevers(), lidarLost: true });
    expect(flight.samples.every((sample) => sample.trace.fixes.every((fix) => fix.source !== "lidar"))).toBe(true);
  });
});
