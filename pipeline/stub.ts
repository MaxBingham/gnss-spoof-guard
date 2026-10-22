import type { SensorFrame } from "../sensors/types.ts";
import type { PositionFix } from "./types.ts";

/**
 * Naive baseline. Trust a GNSS fix whenever the frame has one.
 * With no fix, return nothing and let the Kalman filter coast.
 * Camera and lidar stay on the frame. This step does not associate landmarks.
 */
export function baselineFixes(frame: SensorFrame): PositionFix[] {
  if (!frame.gnss) return [];
  const { x, y, cov } = frame.gnss;
  return [{ source: "gnss", x, y, cov: [cov[0], cov[1], cov[2]] }];
}

/** Earlier name for {@link baselineFixes}. */
export function passThrough(frame: SensorFrame): PositionFix[] {
  return baselineFixes(frame);
}
