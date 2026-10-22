import type { PipelineTrace } from "../pipeline/types.ts";

/** Receives traces from the flight. Rendering stays outside this contract. */
export type TraceHandler = (trace: PipelineTrace) => void;

/** One line of text for a trace. This module does not touch the DOM. */
export function summarizeTrace(trace: PipelineTrace): string {
  const fix = trace.fixes[0];
  const fixText = fix ? `${fix.source} ${fix.x.toFixed(1)},${fix.y.toFixed(1)}` : "none";
  const { x, y, z } = trace.estimate;
  return `t=${trace.t.toFixed(2)}s fix=${fixText} est=${x.toFixed(1)},${y.toFixed(1)},${z.toFixed(1)}`;
}
