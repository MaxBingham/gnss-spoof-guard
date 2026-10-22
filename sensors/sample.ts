import { sampleGnss } from "./gnss.ts";
import type { GnssState } from "./gnss.ts";
import { gaussian, mulberry32 } from "./rng.ts";
import type { BodyState, Conditions, SensorFrame } from "./types.ts";

export const CAMERA = {
  fx: 500,
  fy: 500,
  cx: 320,
  cy: 240,
  width: 640,
  height: 480,
} as const;

/** Map points the camera can see. Ids stay off the reading. */
export const LANDMARKS = [
  { x: 15, y: 8 },
  { x: 40, y: -20 },
  { x: 70, y: -10 },
  { x: 100, y: -40 },
  { x: 130, y: -25 },
] as const;

/** Nadir cross-track beam angles, radians from straight down. */
export const LIDAR_BEAMS = [-0.14, -0.1, -0.06, -0.02, 0.02, 0.06, 0.1, 0.14] as const;

const BARO_SIGMA_M = 0.5;
const COMPASS_SIGMA_RAD = (2 * Math.PI) / 180;
const IMU_ACCEL_SIGMA = 0.02;
const IMU_YAW_RATE_SIGMA = 0.001;
const WEATHER_PIXEL_SIGMA = 6;
const WEATHER_RANGE_SIGMA_M = 1.5;
const WEATHER_DROPOUT = 0.75;

export interface SensorState extends GnssState {
  uniform: () => number;
  biasAx: number;
  biasAy: number;
}

export function createSensorState(seed: number): SensorState {
  const uniform = mulberry32(seed);
  return {
    uniform,
    spoofXM: 0,
    spoofYM: 0,
    gnssWaitS: 0,
    biasAx: gaussian(uniform) * 0.03,
    biasAy: gaussian(uniform) * 0.03,
  };
}

export function projectLandmark(
  pose: Pick<BodyState, "x" | "y" | "z">,
  landmark: { x: number; y: number },
): { u: number; v: number } | null {
  if (pose.z <= 1) return null;
  const u = CAMERA.cx + (CAMERA.fx * (landmark.x - pose.x)) / pose.z;
  const v = CAMERA.cy + (CAMERA.fy * (landmark.y - pose.y)) / pose.z;
  if (u < 0 || v < 0 || u > CAMERA.width || v > CAMERA.height) return null;
  return { u, v };
}

export function cleanLidarRange(altitudeM: number, angleRad: number): number {
  return altitudeM / Math.cos(angleRad);
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
 * GNSS is 5 Hz: jam omits the fix, and a zero spoof rate leaves it unbiased.
 * Occlusion omits that sensor.
 * Weather noise and dropout apply only to camera landmark positions and lidar ranges.
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

  const gnss = sampleGnss(pose, conditions.gnss, state, state.uniform, dt);
  if (gnss) frame.gnss = gnss;

  if (!conditions.camera.occluded) {
    const landmarks = [];
    for (const landmark of LANDMARKS) {
      const projected = projectLandmark(pose, landmark);
      if (!projected) continue;
      const dropped = state.uniform() < dropoutP;
      const u = projected.u + gaussian(state.uniform) * pixelSigma;
      const v = projected.v + gaussian(state.uniform) * pixelSigma;
      if (!dropped) landmarks.push({ u, v });
    }
    frame.camera = { landmarks, intrinsics: CAMERA };
  }

  if (!conditions.lidar.occluded) {
    frame.lidar = {
      beams: LIDAR_BEAMS.map((angleRad) => {
        const dropped = state.uniform() < dropoutP;
        const range = Math.max(0, cleanLidarRange(pose.z, angleRad) + gaussian(state.uniform) * rangeSigma);
        return { angleRad, rangeM: dropped ? null : range };
      }),
    };
  }

  return frame;
}

function wrap(rad: number): number {
  return Math.atan2(Math.sin(rad), Math.cos(rad));
}
