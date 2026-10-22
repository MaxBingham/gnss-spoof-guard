import { describe, expect, it } from "vitest";
import { sampleGnss } from "./gnss.ts";
import type { GnssState } from "./gnss.ts";
import { mulberry32 } from "./rng.ts";
import { createSensorState, sampleFrame } from "./sample.ts";
import type { BodyState, Conditions } from "./types.ts";
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

function gnss(over: Partial<Conditions["gnss"]> = {}): Conditions {
  return { ...CLEAR, gnss: { ...CLEAR.gnss, ...over } };
}

function clock(): GnssState {
  return { spoofXM: 0, spoofYM: 0, gnssWaitS: 0 };
}

describe("GNSS sensor", () => {
  it("publishes a fix at 5 Hz with 1.5 m covariance", () => {
    const state = createSensorState(3);
    const hits: number[] = [];
    for (let i = 0; i < 50; i++) {
      const frame = sampleFrame(HOVER, CLEAR, state, 0.02, i * 0.02);
      if (frame.gnss) {
        hits.push(i);
        expect(frame.gnss.cov).toEqual([2.25, 0, 2.25]);
        expect(frame.gnss.quality).toBe(1);
      }
    }
    expect(hits).toEqual([0, 10, 20, 30, 40]);
  });

  it("draws about 1.5 m of noise on each axis", () => {
    const state = clock();
    const uniform = mulberry32(11);
    const errors: number[] = [];
    for (let i = 0; i < 400; i++) {
      const fix = sampleGnss({ x: 0, y: 0 }, CLEAR.gnss, state, uniform, 0.2);
      expect(fix).toBeDefined();
      errors.push(fix!.x, fix!.y);
    }
    const mean = errors.reduce((sum, value) => sum + value, 0) / errors.length;
    const variance = errors.reduce((sum, value) => sum + (value - mean) ** 2, 0) / errors.length;
    expect(Math.abs(mean)).toBeLessThan(0.3);
    expect(variance).toBeGreaterThan(1.6);
    expect(variance).toBeLessThan(3);
  });

  it("deletes the fix while jammed, including on later epochs and while spoofing", () => {
    const state = createSensorState(3);
    const conditions = gnss({ jammed: true, spoofRateMS: 4, spoofDirectionRad: 0 });
    for (let i = 0; i < 50; i++) {
      const frame = sampleFrame(HOVER, conditions, state, 0.02, i * 0.02);
      expect(frame.gnss).toBeUndefined();
    }
    expect(state.spoofYM).toBeCloseTo(4, 6);
  });

  it("drifts a false position at a non-zero spoof rate and leaves zero as off", () => {
    const state = clock();
    const uniform = mulberry32(5);
    const east = { jammed: false, spoofRateMS: 4, spoofDirectionRad: Math.PI / 2 };
    let fix = sampleGnss({ x: 20, y: 5 }, east, state, uniform, 0.2);
    for (let i = 1; i < 10; i++) fix = sampleGnss({ x: 20, y: 5 }, east, state, uniform, 0.2);
    expect(state.spoofXM).toBeCloseTo(8, 6);
    expect(state.spoofYM).toBeCloseTo(0, 6);
    expect(fix!.x).toBeGreaterThan(20 + 8 - 6);
    expect(Math.abs(fix!.y - 5)).toBeLessThan(6);

    const cleared = sampleGnss({ x: 20, y: 5 }, CLEAR.gnss, state, uniform, 0.2);
    expect(state.spoofXM).toBe(0);
    expect(state.spoofYM).toBe(0);
    expect(Math.abs(cleared!.x - 20)).toBeLessThan(8);
    expect(Math.abs(cleared!.y - 5)).toBeLessThan(8);
  });
});
