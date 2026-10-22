import type { Conditions } from "../sensors/types.ts";

/** Panel controls. Degrees are compass degrees; the contract stores radians. */
export interface LeverState {
  weather: number;
  windSpeedMS: number;
  windDirectionDeg: number;
  cameraLost: boolean;
  lidarLost: boolean;
  gnssJammed: boolean;
  spoofRateMS: number;
  spoofDirectionDeg: number;
}

export const LEVER_LIMITS = {
  weather: { min: 0, max: 1, step: 0.01 },
  windSpeedMS: { min: 0, max: 15, step: 0.1 },
  windDirectionDeg: { min: 0, max: 360, step: 1 },
  spoofRateMS: { min: 0, max: 5, step: 0.1 },
  spoofDirectionDeg: { min: 0, max: 360, step: 1 },
} as const;

const CARDINALS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"] as const;

export function initialLevers(): LeverState {
  return {
    weather: 0,
    windSpeedMS: 0,
    windDirectionDeg: 0,
    cameraLost: false,
    lidarLost: false,
    gnssJammed: false,
    spoofRateMS: 0,
    spoofDirectionDeg: 0,
  };
}

export function leversToConditions(state: LeverState): Conditions {
  return {
    weather: { intensity: clamp(state.weather, 0, 1) },
    wind: {
      speedMS: clamp(state.windSpeedMS, 0, LEVER_LIMITS.windSpeedMS.max),
      directionRad: degToRad(state.windDirectionDeg),
      rotationRadS: 0,
    },
    camera: { occluded: state.cameraLost },
    lidar: { occluded: state.lidarLost },
    gnss: {
      jammed: state.gnssJammed,
      spoofRateMS: clamp(state.spoofRateMS, 0, LEVER_LIMITS.spoofRateMS.max),
      spoofDirectionRad: degToRad(state.spoofDirectionDeg),
    },
  };
}

export function conditionsToLevers(conditions: Conditions): LeverState {
  return {
    weather: clamp(conditions.weather.intensity, 0, 1),
    windSpeedMS: Math.max(0, conditions.wind.speedMS),
    windDirectionDeg: radToDeg(conditions.wind.directionRad),
    cameraLost: conditions.camera.occluded,
    lidarLost: conditions.lidar.occluded,
    gnssJammed: conditions.gnss.jammed,
    spoofRateMS: Math.max(0, conditions.gnss.spoofRateMS),
    spoofDirectionDeg: radToDeg(conditions.gnss.spoofDirectionRad),
  };
}

export function cardinal(deg: number): string {
  const turns = ((deg % 360) + 360) % 360;
  return CARDINALS[Math.round(turns / 45) % 8];
}

export function degToRad(deg: number): number {
  if (!Number.isFinite(deg)) return 0;
  return (deg * Math.PI) / 180;
}

export function radToDeg(rad: number): number {
  if (!Number.isFinite(rad)) return 0;
  return (rad * 180) / Math.PI;
}

function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}
