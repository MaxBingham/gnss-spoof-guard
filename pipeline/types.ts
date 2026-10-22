import type { Estimate } from "../estimation/types.ts";
import type { Cov2 } from "../sensors/types.ts";

/** A horizontal position fix the filter is allowed to ingest. */
export interface PositionFix {
  source: "gnss" | "camera" | "lidar";
  x: number;
  y: number;
  cov: Cov2;
}

/** One control cycle after the stub has forwarded fixes. */
export interface PipelineTrace {
  t: number;
  fixes: PositionFix[];
  estimate: Estimate;
}
