import { CAMERA, projectLandmark } from "./camera.ts";
import { sampleGnss } from "./gnss.ts";
import type { GnssState } from "./gnss.ts";
import { LIDAR_BEAMS, castDownBeam } from "./lidar.ts";
import { LANDMARKS } from "./map.ts";
import { gaussian, mulberry32 } from "./rng.ts";
import type { BodyState, CameraLandmark, Conditions, SensorFrame } from "./types.ts";
import { applyWeather } from "./weather.ts";

export { CAMERA, projectLandmark, sampleCamera } from "./camera.ts";
export { BUILDINGS, KNOWN_BUILDING, LANDMARKS } from "./map.ts";
export type { Building, MapLandmark } from "./map.ts";
export { LIDAR_BEAMS, beamPoint, castDownBeam, cleanLidarRange, sampleLidar } from "./lidar.ts";

const BARO_SIGMA_M = 0.5;
const COMPASS_SIGMA_RAD = (2 * Math.PI) / 180;
const IMU_ACCEL_SIGMA = 0.02;
const IMU_YAW_RATE_SIGMA = 0.001;

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

/**
 * Build one frame from the body state and the levers.
 * GNSS is 5 Hz: jam omits the fix, and a zero spoof rate leaves it unbiased.
 * Occlusion omits that sensor.
 * Camera blobs keep `{type, u, v}` and lidar ranges hit building prisms.
 * Weather noise and dropout apply only to those camera positions and lidar ranges.
 */
export function sampleFrame(
  pose: BodyState,
  conditions: Conditions,
  state: SensorState,
  dt: number,
  t: number,
): SensorFrame {
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
    const landmarks: CameraLandmark[] = [];
    for (const landmark of LANDMARKS) {
      const projected = projectLandmark(pose, landmark);
      if (projected) landmarks.push({ type: landmark.type, u: projected.u, v: projected.v });
    }
    frame.camera = { landmarks, intrinsics: CAMERA };
  }

  if (!conditions.lidar.occluded) {
    frame.lidar = {
      beams: LIDAR_BEAMS.map((angleRad) => ({
        angleRad,
        rangeM: castDownBeam(pose, angleRad),
      })),
    };
  }

  applyWeather(frame, conditions.weather.intensity, state.uniform);
  return frame;
}

function wrap(rad: number): number {
  return Math.atan2(Math.sin(rad), Math.cos(rad));
}
