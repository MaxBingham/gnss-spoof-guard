import { sampleCamera } from "./camera.ts";
import { compassUnit } from "./compass.ts";
import { sampleLidar } from "./lidar.ts";
import { gaussian, mulberry32 } from "./rng.ts";
import type { BodyState, Conditions, SensorFrame } from "./types.ts";

export { CAMERA, projectLandmark, sampleCamera } from "./camera.ts";
export { BUILDINGS, KNOWN_BUILDING, LANDMARKS } from "./map.ts";
export type { Building, MapLandmark } from "./map.ts";
export { LIDAR_BEAMS, beamPoint, castDownBeam, cleanLidarRange, sampleLidar } from "./lidar.ts";

const GNSS_SIGMA_M = 1.5;
const BARO_SIGMA_M = 0.5;
const COMPASS_SIGMA_RAD = (2 * Math.PI) / 180;
const IMU_ACCEL_SIGMA = 0.02;
const IMU_YAW_RATE_SIGMA = 0.001;
const WEATHER_PIXEL_SIGMA = 6;
const WEATHER_RANGE_SIGMA_M = 1.5;
const WEATHER_DROPOUT = 0.75;

export interface SensorState {
  uniform: () => number;
  spoofXM: number;
  spoofYM: number;
  biasAx: number;
  biasAy: number;
}

export function createSensorState(seed: number): SensorState {
  const uniform = mulberry32(seed);
  return {
    uniform,
    spoofXM: 0,
    spoofYM: 0,
    biasAx: gaussian(uniform) * 0.03,
    biasAy: gaussian(uniform) * 0.03,
  };
}

function weatherScale(intensity: number): { pixelSigma: number; rangeSigma: number; dropoutP: number } {
  const scaled = Math.min(1, Math.max(0, intensity));
  return {
    pixelSigma: scaled * WEATHER_PIXEL_SIGMA,
    rangeSigma: scaled * WEATHER_RANGE_SIGMA_M,
    dropoutP: scaled * WEATHER_DROPOUT,
  };
}

/**
 * Build one frame from the body state and the levers.
 * Jam omits GNSS. Occlusion omits that sensor. Spoof rate 0 leaves the fix unbiased.
 * Weather scales camera pixel noise, misses, and a few false blobs, plus lidar range noise and dropout.
 */
export function sampleFrame(
  pose: BodyState,
  conditions: Conditions,
  state: SensorState,
  dt: number,
  t: number,
): SensorFrame {
  const { pixelSigma, rangeSigma, dropoutP } = weatherScale(conditions.weather.intensity);
  const frame: SensorFrame = {
    t,
    imu: {
      ax: pose.ax + state.biasAx + gaussian(state.uniform) * IMU_ACCEL_SIGMA,
      ay: pose.ay + state.biasAy + gaussian(state.uniform) * IMU_ACCEL_SIGMA,
      az: pose.az + gaussian(state.uniform) * IMU_ACCEL_SIGMA,
      yawRateRadS: pose.yawRateRadS + gaussian(state.uniform) * IMU_YAW_RATE_SIGMA,
    },
    baro: { altM: pose.z + gaussian(state.uniform) * BARO_SIGMA_M },
    compass: { yawRad: wrap(pose.yaw + gaussian(state.uniform) * COMPASS_SIGMA_RAD) },
  };

  if (conditions.gnss.spoofRateMS === 0) {
    state.spoofXM = 0;
    state.spoofYM = 0;
  } else {
    const drift = compassUnit(conditions.gnss.spoofDirectionRad);
    state.spoofXM += conditions.gnss.spoofRateMS * dt * drift.x;
    state.spoofYM += conditions.gnss.spoofRateMS * dt * drift.y;
  }
  if (!conditions.gnss.jammed) {
    frame.gnss = {
      x: pose.x + state.spoofXM + gaussian(state.uniform) * GNSS_SIGMA_M,
      y: pose.y + state.spoofYM + gaussian(state.uniform) * GNSS_SIGMA_M,
      cov: [GNSS_SIGMA_M * GNSS_SIGMA_M, 0, GNSS_SIGMA_M * GNSS_SIGMA_M],
      quality: 1,
    };
  }

  if (!conditions.camera.occluded) {
    frame.camera = sampleCamera(
      pose,
      { intensity: conditions.weather.intensity, pixelSigma, dropoutP },
      state.uniform,
    );
  }

  if (!conditions.lidar.occluded) {
    frame.lidar = sampleLidar(pose, { rangeSigma, dropoutP }, state.uniform);
  }

  return frame;
}

function wrap(rad: number): number {
  return Math.atan2(Math.sin(rad), Math.cos(rad));
}
