import type { Estimate } from "../estimation/types.ts";
import type { PipelineTrace, PositionFix } from "../pipeline/types.ts";
import type { Cov2 } from "../sensors/types.ts";
import type { Conditions } from "../sensors/types.ts";
import { leversToConditions, type LeverState } from "./levers.ts";

/** Mock mission, meters. Matches the headless mission's start and goal. */
export const MOCK_GOAL = { x: 120, y: -30 } as const;

const DT_S = 0.5;
const DURATION_S = 30;
const CRUISE_S = 18;
const GNSS_VAR = 2.25;
const COAST_VAR0 = 25;
const COAST_VAR_PER_S = 4;
/** Meters of truth drift per 1 m/s of wind, per second. */
const WIND_DRIFT = 0.4;

export interface FlightSample {
  trace: PipelineTrace;
  truth: { x: number; y: number };
}

export interface MockFlight {
  conditions: Conditions;
  samples: FlightSample[];
}

/**
 * Deterministic naive flight for the dashboard.
 * The estimate trusts a GNSS fix whenever one exists, so a spoof carries it off
 * the truth. Jam drops that fix. Camera and lidar fixes remain on the trace and
 * are used only while GNSS is absent. Weather noise hits those two sensors.
 */
export function mockFlight(levers: LeverState): MockFlight {
  const conditions = leversToConditions(levers);
  const samples: FlightSample[] = [];
  const steps = Math.round(DURATION_S / DT_S);

  for (let i = 0; i <= steps; i++) {
    const t = i * DT_S;
    const progress = Math.min(1, t / CRUISE_S);
    const wind = compass(conditions.wind.directionRad);
    const drift = conditions.wind.speedMS * WIND_DRIFT * t;
    const truth = {
      x: MOCK_GOAL.x * progress + wind.x * drift,
      y: MOCK_GOAL.y * progress + wind.y * drift,
    };
    const spoof = compass(conditions.gnss.spoofDirectionRad);
    const spoofM = conditions.gnss.spoofRateMS * t;
    const fixes = fixesAt(conditions, truth, t, spoof, spoofM);
    const estimate = estimateAt(fixes, t);
    samples.push({ trace: { t, fixes, estimate }, truth });
  }

  return { conditions, samples };
}

function fixesAt(
  conditions: Conditions,
  truth: { x: number; y: number },
  t: number,
  spoof: { x: number; y: number },
  spoofM: number,
): PositionFix[] {
  const fixes: PositionFix[] = [];
  if (!conditions.gnss.jammed) {
    fixes.push({
      source: "gnss",
      x: truth.x + spoof.x * spoofM,
      y: truth.y + spoof.y * spoofM,
      cov: [GNSS_VAR, 0, GNSS_VAR],
    });
  }

  const weather = Math.min(1, Math.max(0, conditions.weather.intensity));
  const wobble = weather * 8 * Math.sin(t * 0.7);
  const sigma = 3 + weather * 12;
  const sensorVar = sigma * sigma;
  if (!conditions.camera.occluded) {
    fixes.push({
      source: "camera",
      x: truth.x + wobble,
      y: truth.y + wobble * 0.25,
      cov: [sensorVar, 0, sensorVar],
    });
  }
  if (!conditions.lidar.occluded) {
    fixes.push({
      source: "lidar",
      x: truth.x - wobble * 0.5,
      y: truth.y + wobble * 0.5,
      cov: [sensorVar, 0, sensorVar],
    });
  }
  return fixes;
}

function estimateAt(fixes: PositionFix[], t: number): Estimate {
  const gnss = fixes.find((fix) => fix.source === "gnss");
  const fused = gnss ?? fuse(fixes.filter((fix) => fix.source !== "gnss"));
  const x = fused?.x ?? 0;
  const y = fused?.y ?? 0;
  const cov: Cov2 = fused?.cov ?? [COAST_VAR0 + COAST_VAR_PER_S * t, 0, COAST_VAR0 + COAST_VAR_PER_S * t];
  return { x, y, vx: 0, vy: 0, z: 30 * Math.min(1, t / 4), yaw: 0, cov };
}

function fuse(fixes: PositionFix[]): { x: number; y: number; cov: Cov2 } | undefined {
  if (fixes.length === 0) return undefined;
  let weight = 0;
  let x = 0;
  let y = 0;
  for (const fix of fixes) {
    const inverse = 1 / Math.max(fix.cov[0], 1e-6);
    weight += inverse;
    x += inverse * fix.x;
    y += inverse * fix.y;
  }
  const variance = 1 / weight;
  return { x: x / weight, y: y / weight, cov: [variance, 0, variance] };
}

function compass(directionRad: number): { x: number; y: number } {
  return { x: Math.sin(directionRad), y: Math.cos(directionRad) };
}
