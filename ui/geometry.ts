import type { Cov2 } from "../sensors/types.ts";

/** Horizontal distance between truth and the filter estimate, meters. */
export function positionErrorM(truth: { x: number; y: number }, estimate: { x: number; y: number }): number {
  return Math.hypot(estimate.x - truth.x, estimate.y - truth.y);
}

/**
 * Radius of the 2σ uncertainty circle, meters.
 * The circle uses the major axis of the horizontal covariance so it covers the 2σ ellipse.
 */
export function uncertaintyRadiusM(cov: Cov2): number {
  const xx = cov[0];
  const xy = cov[1];
  const yy = cov[2];
  const mean = (xx + yy) / 2;
  const split = Math.sqrt(Math.max(0, ((xx - yy) / 2) ** 2 + xy * xy));
  return 2 * Math.sqrt(Math.max(0, mean + split));
}
