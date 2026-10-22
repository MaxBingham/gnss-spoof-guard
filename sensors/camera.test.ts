import { describe, expect, it } from "vitest";
import { projectLandmark, sampleCamera } from "./camera.ts";
import { LANDMARKS } from "./map.ts";
import { mulberry32 } from "./rng.ts";
import { createSensorState, sampleFrame } from "./sample.ts";
import type { BodyState, CameraLandmark } from "./types.ts";
import { CLEAR } from "./types.ts";

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

const CLEAR_WEATHER = { intensity: 0, pixelSigma: 0, dropoutP: 0 };

function projected(pose: Pick<BodyState, "x" | "y" | "z">): { type: CameraLandmark["type"]; u: number; v: number }[] {
  return LANDMARKS.flatMap((landmark) => {
    const point = projectLandmark(pose, landmark);
    if (!point) return [];
    return [{ type: landmark.type, u: point.u, v: point.v }];
  });
}

describe("camera", () => {
  it("emits {type, u, v} blobs and drops the landmark id", () => {
    const reading = sampleCamera(HOVER, CLEAR_WEATHER, mulberry32(1));
    const expected = projected(HOVER);
    expect(expected.length).toBeGreaterThan(0);
    expect(LANDMARKS.some((landmark) => landmark.id.length > 0)).toBe(true);
    expect(reading.landmarks).toEqual(expected);
    for (const blob of reading.landmarks) {
      expect(Object.keys(blob).sort()).toEqual(["type", "u", "v"]);
      expect(blob).not.toHaveProperty("id");
    }
  });

  it("adds noise, misses, and a few false blobs as the weather lever rises", () => {
    const pose = { x: 70, y: -10, z: 200 };
    const truth = projected(pose);
    expect(truth.length).toBe(LANDMARKS.length);
    const uniform = mulberry32(4);
    let misses = 0;
    let falseBlobs = 0;
    let noisy = 0;
    for (let i = 0; i < 24; i++) {
      const reading = sampleCamera(pose, { intensity: 1, pixelSigma: 6, dropoutP: 0.75 }, uniform);
      expect(reading.landmarks.length).toBeLessThanOrEqual(truth.length + 3);
      const seen = new Set<number>();
      for (const blob of reading.landmarks) {
        const match = truth.findIndex(
          (landmark, index) =>
            !seen.has(index) &&
            landmark.type === blob.type &&
            Math.hypot(blob.u - landmark.u, blob.v - landmark.v) < 40,
        );
        if (match < 0) {
          falseBlobs += 1;
          continue;
        }
        seen.add(match);
        const landmark = truth[match]!;
        if (Math.hypot(blob.u - landmark.u, blob.v - landmark.v) > 0.25) noisy += 1;
      }
      misses += truth.length - seen.size;
    }
    expect(misses).toBeGreaterThan(0);
    expect(falseBlobs).toBeGreaterThan(0);
    expect(noisy).toBeGreaterThan(0);
  });

  it("draws no clutter when the weather lever is off", () => {
    const reading = sampleCamera(HOVER, CLEAR_WEATHER, mulberry32(99));
    expect(reading.landmarks).toEqual(projected(HOVER));
  });

  it("deletes the camera reading when the switch is on", () => {
    const frame = sampleFrame(HOVER, { ...CLEAR, camera: { occluded: true } }, createSensorState(1), 0.02, 0);
    expect(frame.camera).toBeUndefined();
    expect(frame.lidar).toBeDefined();
  });
});
