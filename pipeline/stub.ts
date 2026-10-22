import type { SensorFrame } from "../sensors/types.ts";
import type { PositionFix } from "./types.ts";

/**
 * Pass-through pipeline. Forwards the GNSS fix when the frame has one.
 * Camera and lidar readings stay on the frame for a later module.
 */
export function passThrough(frame: SensorFrame): PositionFix[] {
  if (!frame.gnss) return [];
  const { x, y, cov } = frame.gnss;
  return [{ source: "gnss", x, y, cov: [cov[0], cov[1], cov[2]] }];
}
