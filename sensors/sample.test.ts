import { describe, expect, it } from "vitest";
import { CLEAR } from "./types.ts";
import type { BodyState, Conditions } from "./types.ts";
import {
  LIDAR_BEAMS,
  LANDMARKS,
  cleanLidarRange,
  createSensorState,
  projectLandmark,
  sampleFrame,
} from "./sample.ts";

const HOVER: BodyState = {
  x: 0,
  y: 0,
  z: 50,
  ax: 0,
  ay: 0,
  az: 0,
  yaw: 0,
  yawRateRadS: 0,
};

function withGnss(gnss: Conditions["gnss"]): Conditions {
  return { ...CLEAR, gnss };
}

describe("sensor levers", () => {
  it("keeps a GNSS fix on the true position when the spoof rate is zero", () => {
    const state = createSensorState(7);
    let frame = sampleFrame(HOVER, CLEAR, state, 0.1, 0);
    for (let i = 0; i < 100; i++) frame = sampleFrame(HOVER, CLEAR, state, 0.1, i * 0.1);
    expect(frame.gnss).toBeDefined();
    expect(Math.abs(frame.gnss!.x - HOVER.x)).toBeLessThan(8);
    expect(Math.abs(frame.gnss!.y - HOVER.y)).toBeLessThan(8);
  });

  it("drifts a false GNSS position at the spoof rate", () => {
    const state = createSensorState(7);
    const conditions = withGnss({ jammed: false, spoofRateMS: 2, spoofDirectionRad: Math.PI / 2 });
    let frame = sampleFrame(HOVER, conditions, state, 0.1, 0);
    for (let i = 0; i < 100; i++) frame = sampleFrame(HOVER, conditions, state, 0.1, (i + 1) * 0.1);
    expect(frame.gnss!.x - HOVER.x).toBeGreaterThan(15);
    expect(Math.abs(frame.gnss!.y - HOVER.y)).toBeLessThan(8);
  });

  it("deletes the GNSS fix when jammed, including while a spoof rate is set", () => {
    const state = createSensorState(3);
    const conditions = withGnss({ jammed: true, spoofRateMS: 4, spoofDirectionRad: 0 });
    const frame = sampleFrame(HOVER, conditions, state, 0.1, 1);
    expect(frame.gnss).toBeUndefined();
  });

  it("deletes camera data on occlusion and leaves lidar in place", () => {
    const frame = sampleFrame(HOVER, { ...CLEAR, camera: { occluded: true } }, createSensorState(1), 0.02, 0);
    expect(frame.camera).toBeUndefined();
    expect(frame.lidar?.beams.length).toBe(LIDAR_BEAMS.length);
  });

  it("deletes lidar data on occlusion and leaves the camera in place", () => {
    const frame = sampleFrame(HOVER, { ...CLEAR, lidar: { occluded: true } }, createSensorState(1), 0.02, 0);
    expect(frame.lidar).toBeUndefined();
    expect(frame.camera).toBeDefined();
  });

  it("reports clean camera and lidar readings at weather intensity zero", () => {
    const frame = sampleFrame(HOVER, CLEAR, createSensorState(4), 0.02, 0);
    const expected = LANDMARKS.map((landmark) => projectLandmark(HOVER, landmark)).filter(
      (point): point is { u: number; v: number } => point !== null,
    );
    expect(expected.length).toBeGreaterThan(0);
    expect(frame.camera?.landmarks).toEqual(expected);
    expect(frame.lidar?.beams.map((beam) => beam.rangeM)).toEqual(
      LIDAR_BEAMS.map((angle) => cleanLidarRange(HOVER.z, angle)),
    );
  });

  it("adds dropout and noise to camera and lidar as weather intensity rises", () => {
    const storm: Conditions = { ...CLEAR, weather: { intensity: 1 } };
    let dropped = 0;
    let noisy = false;
    const state = createSensorState(9);
    for (let i = 0; i < 10; i++) {
      const frame = sampleFrame(HOVER, storm, state, 0.02, i * 0.02);
      for (const beam of frame.lidar!.beams) {
        if (beam.rangeM === null) dropped += 1;
        else if (Math.abs(beam.rangeM - cleanLidarRange(HOVER.z, beam.angleRad)) > 1e-6) noisy = true;
      }
    }
    expect(dropped).toBeGreaterThan(0);
    expect(noisy).toBe(true);
  });
});
