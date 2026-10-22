import { compassUnit } from "./compass.ts";
import type { Conditions } from "./types.ts";

/**
 * Acceleration per 1 m/s of the wind speed lever.
 * Mass is normalized, so this is also the sideways force scale.
 */
export const WIND_ACCEL_PER_MPS = 0.25;

/**
 * Sideways acceleration from the wind lever at time t.
 * Direction is compass radians (0 north, π/2 east), held fixed when
 * `rotationRadS` is 0 and turned slowly otherwise. Speed 0 is calm.
 */
export function windAcceleration(wind: Conditions["wind"], t: number): { ax: number; ay: number } {
  if (wind.speedMS === 0) return { ax: 0, ay: 0 };
  const theta = wind.directionRad + wind.rotationRadS * t;
  const accel = wind.speedMS * WIND_ACCEL_PER_MPS;
  const unit = compassUnit(theta);
  return { ax: accel * unit.x, ay: accel * unit.y };
}
