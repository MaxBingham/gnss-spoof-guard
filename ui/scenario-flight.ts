import type { Conditions } from "../sensors/types.ts";
import type { ScenarioSample } from "../sim/scenario.ts";
import { runScenario } from "../sim/scenario.ts";
import type { FlightSample } from "./mock-flight.ts";
import { leversToConditions, type LeverState } from "./levers.ts";

/** Same seed as `npm run scenario -- --seed 42`. */
export const SCENARIO_SEED = 42;

/**
 * Display step. The sim runs at 50 Hz; the dashboard keeps every GNSS epoch
 * and always keeps the final step.
 */
const DISPLAY_STRIDE = 10;

export interface ScenarioViewSample extends FlightSample {
  /** Null when the camera-loss switch deleted the reading. */
  cameraBlobs: number | null;
  /** Null when the lidar-loss switch deleted the reading. */
  lidarReturns: number | null;
  lidarBeams: number | null;
}

export interface ScenarioFlight {
  conditions: Conditions;
  samples: ScenarioViewSample[];
}

/** Play one headless baseline flight for the current lever values. */
export function scenarioFlight(levers: LeverState, options?: { maxTimeS?: number }): ScenarioFlight {
  const conditions = leversToConditions(levers);
  const result = runScenario({
    seed: SCENARIO_SEED,
    conditions,
    maxTimeS: options?.maxTimeS,
  });
  const samples: ScenarioViewSample[] = [];
  for (let i = 0; i < result.samples.length; i += DISPLAY_STRIDE) {
    samples.push(viewSample(result.samples[i]!));
  }
  const last = result.samples.at(-1);
  if (last && samples.at(-1)?.trace !== last.trace) samples.push(viewSample(last));
  return { conditions, samples };
}

function viewSample(sample: ScenarioSample): ScenarioViewSample {
  const camera = sample.frame.camera;
  const lidar = sample.frame.lidar;
  return {
    trace: sample.trace,
    truth: { x: sample.truth.x, y: sample.truth.y },
    cameraBlobs: camera ? camera.landmarks.length : null,
    lidarReturns: lidar ? lidar.beams.filter((beam) => beam.rangeM !== null).length : null,
    lidarBeams: lidar ? lidar.beams.length : null,
  };
}
