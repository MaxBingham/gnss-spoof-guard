import { describe, expect, it } from "vitest";
import { createSensorState, sampleFrame } from "./sample.ts";
import { CLEAR } from "./types.ts";
import type { BodyState, Conditions } from "./types.ts";
import {
  WEATHER_DROPOUT,
  WEATHER_PIXEL_SIGMA,
  WEATHER_RANGE_SIGMA_M,
  degradeLandmark,
  degradeRange,
  weatherEffect,
} from "./weather.ts";

const HOVER: BodyState = {
  x: 70,
  y: -15,
  z: 150,
  ax: 0,
  ay: 0,
  az: 0,
  yaw: 0,
  yawRateRadS: 0,
};

function at(intensity: number): Conditions {
  return { ...CLEAR, weather: { intensity } };
}

function scripted(draws: number[]): () => number {
  let index = 0;
  return () => {
    const next = draws[index];
    if (next === undefined) throw new Error("rng exhausted");
    index += 1;
    return next;
  };
}

describe("weather lever", () => {
  it("maps one intensity onto pixel noise, range noise, and dropout", () => {
    expect(weatherEffect(0)).toEqual({ pixelSigma: 0, rangeSigma: 0, dropoutP: 0 });
    expect(weatherEffect(1)).toEqual({
      pixelSigma: WEATHER_PIXEL_SIGMA,
      rangeSigma: WEATHER_RANGE_SIGMA_M,
      dropoutP: WEATHER_DROPOUT,
    });
    expect(weatherEffect(0.5).pixelSigma).toBeCloseTo(WEATHER_PIXEL_SIGMA / 2);
    expect(weatherEffect(0.5).rangeSigma).toBeCloseTo(WEATHER_RANGE_SIGMA_M / 2);
    expect(weatherEffect(0.5).dropoutP).toBeCloseTo(WEATHER_DROPOUT / 2);
    expect(weatherEffect(-2)).toEqual(weatherEffect(0));
    expect(weatherEffect(4)).toEqual(weatherEffect(1));
  });

  it("adds Gaussian pixel noise and drops the landmark more often as intensity rises", () => {
    const gaussian = Math.sqrt(-2 * Math.log(0.3));
    const kept = degradeLandmark({ u: 10, v: 20 }, weatherEffect(1), scripted([0.9, 0.3, 0, 0.3, 0.25]));
    expect(kept).not.toBeNull();
    expect(kept!.u).toBeCloseTo(10 + gaussian * WEATHER_PIXEL_SIGMA);
    expect(kept!.v).toBeCloseTo(20);

    const half = degradeLandmark({ u: 10, v: 20 }, weatherEffect(0.5), scripted([0.9, 0.3, 0, 0.3, 0.25]));
    expect(half!.u - 10).toBeCloseTo((kept!.u - 10) / 2);
    expect(half!.v).toBeCloseTo(20);

    const draw = [0.5, 0.3, 0, 0.3, 0];
    expect(degradeLandmark({ u: 1, v: 1 }, weatherEffect(0.5), scripted(draw))).not.toBeNull();
    expect(degradeLandmark({ u: 1, v: 1 }, weatherEffect(1), scripted(draw))).toBeNull();

    let index = 0;
    const draws = [0.1, 0.3, 0, 0.3, 0, 0.4];
    const uniform = () => {
      const next = draws[index];
      if (next === undefined) throw new Error("rng exhausted");
      index += 1;
      return next;
    };
    expect(degradeLandmark({ u: 0, v: 0 }, weatherEffect(1), uniform)).toBeNull();
    expect(index).toBe(5);
    expect(uniform()).toBe(0.4);
  });

  it("adds Gaussian range noise, drops returns, and refuses a negative range", () => {
    const noisy = degradeRange(50, weatherEffect(1), scripted([0.99, 0.3, 0]));
    const gaussian = Math.sqrt(-2 * Math.log(0.3));
    expect(noisy).toBeCloseTo(50 + gaussian * WEATHER_RANGE_SIGMA_M);

    expect(degradeRange(50, weatherEffect(1), scripted([0.2, 0.3, 0]))).toBeNull();
    expect(degradeRange(1, weatherEffect(1), scripted([0.99, 1e-12, 0.5]))).toBe(0);
    expect(degradeRange(40, weatherEffect(0), scripted([0, 0.2, 0.7]))).toBe(40);
  });

  it("leaves GNSS and the inertial sensors alone while it degrades camera and lidar", () => {
    const clear = sampleFrame(HOVER, CLEAR, createSensorState(11), 0.02, 0);
    const storm = sampleFrame(HOVER, at(1), createSensorState(11), 0.02, 0);
    expect(clear.camera!.landmarks.length).toBeGreaterThan(1);
    expect(storm.imu).toEqual(clear.imu);
    expect(storm.baro).toEqual(clear.baro);
    expect(storm.compass).toEqual(clear.compass);
    expect(storm.gnss).toEqual(clear.gnss);
    expect(storm.camera!.landmarks.length).toBeLessThanOrEqual(clear.camera!.landmarks.length);
    expect(storm.lidar!.beams.some((beam) => beam.rangeM === null)).toBe(true);

    const mild = sampleFrame(HOVER, at(0.25), createSensorState(11), 0.02, 0);
    expect(storm.camera!.landmarks.length).toBeLessThanOrEqual(mild.camera!.landmarks.length);
    for (let i = 0; i < mild.lidar!.beams.length; i++) {
      const low = mild.lidar!.beams[i]!;
      const high = storm.lidar!.beams[i]!;
      if (low.rangeM === null) expect(high.rangeM).toBeNull();
      if (low.rangeM !== null && high.rangeM !== null) {
        const clean = clear.lidar!.beams[i]!.rangeM!;
        expect(high.rangeM - clean).toBeCloseTo((low.rangeM - clean) * 4);
      }
    }
  });
});
