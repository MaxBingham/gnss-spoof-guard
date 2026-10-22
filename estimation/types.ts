import type { Cov2 } from "../sensors/types.ts";

/** Horizontal position, velocity, altitude, and heading after a filter step. */
export interface Estimate {
  x: number;
  y: number;
  vx: number;
  vy: number;
  z: number;
  yaw: number;
  /** Horizontal position covariance, `[xx, xy, yy]`. */
  cov: Cov2;
}
