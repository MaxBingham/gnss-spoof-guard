import { gaussian } from "./rng.ts";
import type { SensorFrame } from "./types.ts";

/** Pixel sigma on camera landmarks at weather intensity 1. */
export const WEATHER_PIXEL_SIGMA = 6;
/** Range sigma on lidar returns at weather intensity 1, metres. */
export const WEATHER_RANGE_SIGMA_M = 1.5;
/**
 * Chance a camera landmark or lidar return is dropped at intensity 1.
 * Occlusion is a separate switch and deletes the whole reading.
 */
export const WEATHER_DROPOUT = 0.75;

export interface WeatherEffect {
  pixelSigma: number;
  rangeSigma: number;
  dropoutP: number;
}

/** Map the single weather lever in [0, 1] onto noise and dropout. */
export function weatherEffect(intensity: number): WeatherEffect {
  const scaled = Math.min(1, Math.max(0, intensity));
  return {
    pixelSigma: scaled * WEATHER_PIXEL_SIGMA,
    rangeSigma: scaled * WEATHER_RANGE_SIGMA_M,
    dropoutP: scaled * WEATHER_DROPOUT,
  };
}

/**
 * Gaussian noise on one landmark, then maybe drop it.
 * The noise draws always run, so a dropped reading still consumes the same RNG.
 */
export function degradeLandmark(
  landmark: { u: number; v: number },
  effect: WeatherEffect,
  uniform: () => number,
): { u: number; v: number } | null {
  const dropped = uniform() < effect.dropoutP;
  const u = landmark.u + gaussian(uniform) * effect.pixelSigma;
  const v = landmark.v + gaussian(uniform) * effect.pixelSigma;
  if (dropped) return null;
  return { u, v };
}

/** Gaussian noise on one lidar range, then maybe drop it. Ranges stay non-negative. */
export function degradeRange(rangeM: number, effect: WeatherEffect, uniform: () => number): number | null {
  const dropped = uniform() < effect.dropoutP;
  const range = Math.max(0, rangeM + gaussian(uniform) * effect.rangeSigma);
  if (dropped) return null;
  return range;
}

/**
 * Apply the weather lever to raw camera landmarks and lidar ranges.
 * GNSS, IMU, barometer, and compass are left alone. Missing sensors stay missing.
 */
export function applyWeather(frame: SensorFrame, intensity: number, uniform: () => number): void {
  const effect = weatherEffect(intensity);
  if (frame.camera) {
    const landmarks = [];
    for (const landmark of frame.camera.landmarks) {
      const degraded = degradeLandmark(landmark, effect, uniform);
      if (degraded) landmarks.push(degraded);
    }
    frame.camera = { landmarks, intrinsics: frame.camera.intrinsics };
  }
  if (frame.lidar) {
    frame.lidar = {
      beams: frame.lidar.beams.map((beam) => ({
        angleRad: beam.angleRad,
        rangeM: beam.rangeM === null ? null : degradeRange(beam.rangeM, effect, uniform),
      })),
    };
  }
}
