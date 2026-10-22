import { KalmanFilter } from "../estimation/kalman.ts";
import { baselineFixes } from "../pipeline/stub.ts";
import type { PipelineTrace } from "../pipeline/types.ts";
import { CLEAR, createSensorState, sampleFrame } from "../sensors/index.ts";
import type { BodyState, Conditions, SensorFrame } from "../sensors/types.ts";
import { commandFor, nextPhase, type Phase } from "./controller.ts";
import { stepDynamics, windAcceleration, type TruthPose } from "./dynamics.ts";
import { MISSION } from "./mission.ts";

export interface ScenarioOptions {
  seed: number;
  conditions?: Conditions;
  maxTimeS?: number;
}

/**
 * One step of a flight.
 * `truth` is for the log, the metrics, and the dashboard.
 * `trace` is what the pipeline produced. It does not carry truth.
 * `frame` is the sensor sample, including camera and lidar when those switches leave them on.
 */
export interface ScenarioSample {
  truth: { x: number; y: number; z: number };
  trace: PipelineTrace;
  frame: SensorFrame;
}

export interface FlightResult {
  seed: number;
  landed: boolean;
  steps: number;
  durationS: number;
  finalErrorM: number;
  finalAltM: number;
  /** Pipeline traces only. Same objects as `samples[i].trace`. */
  traces: PipelineTrace[];
  /** Truth beside the estimate, plus the sensor frame for that step. */
  samples: ScenarioSample[];
}

/**
 * Fly one mission.
 * The baseline ingests a GNSS fix whenever one exists and otherwise coasts on the filter.
 */
export function runScenario(options: ScenarioOptions): FlightResult {
  const conditions = options.conditions ?? CLEAR;
  const dt = MISSION.dt;
  let pose: TruthPose = {
    x: MISSION.startX,
    y: MISSION.startY,
    z: 0,
    vx: 0,
    vy: 0,
    vz: 0,
    yaw: 0,
  };
  const filter = new KalmanFilter();
  const sensors = createSensorState(options.seed);
  const samples: ScenarioSample[] = [];
  let phase: Phase = "takeoff";
  const maxSteps = Math.ceil((options.maxTimeS ?? MISSION.maxTimeS) / dt);
  let steps = 0;
  let landed = false;

  while (steps < maxSteps) {
    const t = steps * dt;
    phase = nextPhase(phase, filter.estimate());
    const command = commandFor(phase, filter.estimate());
    const stepped = stepDynamics(pose, command, windAcceleration(conditions.wind, t), dt);
    const body: BodyState = {
      x: stepped.pose.x,
      y: stepped.pose.y,
      z: stepped.pose.z,
      ax: stepped.ax,
      ay: stepped.ay,
      az: stepped.az,
      yaw: stepped.pose.yaw,
      yawRateRadS: stepped.yawRateRadS,
    };
    const frame = sampleFrame(body, conditions, sensors, dt, t + dt);
    const fixes = baselineFixes(frame);
    filter.predict(frame.imu, dt);
    filter.updateBaro(frame.baro.altM);
    filter.updateCompass(frame.compass.yawRad);
    for (const fix of fixes) filter.updatePosition(fix.x, fix.y, fix.cov);
    pose = stepped.pose;
    samples.push({
      truth: { x: pose.x, y: pose.y, z: pose.z },
      trace: { t: frame.t, fixes, estimate: filter.estimate() },
      frame,
    });
    steps += 1;
    const error = Math.hypot(pose.x - MISSION.goalX, pose.y - MISSION.goalY);
    if (phase === "land" && pose.z === 0 && error <= MISSION.successRadiusM) {
      landed = true;
      break;
    }
  }

  return {
    seed: options.seed,
    landed,
    steps,
    durationS: steps * dt,
    finalErrorM: Math.hypot(pose.x - MISSION.goalX, pose.y - MISSION.goalY),
    finalAltM: pose.z,
    traces: samples.map((sample) => sample.trace),
    samples,
  };
}
