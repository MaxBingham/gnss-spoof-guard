import { KalmanFilter } from "../estimation/kalman.ts";
import type { PipelineTrace } from "../pipeline/types.ts";
import { passThrough } from "../pipeline/stub.ts";
import { CLEAR, createSensorState, sampleFrame } from "../sensors/index.ts";
import type { BodyState, Conditions } from "../sensors/types.ts";
import { commandFor, nextPhase, type Phase } from "./controller.ts";
import { stepDynamics, windAcceleration, type TruthPose } from "./dynamics.ts";
import { MISSION } from "./mission.ts";

export interface ScenarioOptions {
  seed: number;
  conditions?: Conditions;
  maxTimeS?: number;
}

export interface FlightResult {
  seed: number;
  landed: boolean;
  steps: number;
  durationS: number;
  finalErrorM: number;
  finalAltM: number;
  traces: PipelineTrace[];
}

/** Fly one mission. The stub forwards GNSS; the Kalman filter localizes. */
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
  const traces: PipelineTrace[] = [];
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
    const fixes = passThrough(frame);
    filter.predict(frame.imu, dt);
    filter.updateBaro(frame.baro.altM);
    filter.updateCompass(frame.compass.yawRad);
    for (const fix of fixes) filter.updatePosition(fix.x, fix.y, fix.cov);
    traces.push({ t: frame.t, fixes, estimate: filter.estimate() });
    pose = stepped.pose;
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
    traces,
  };
}
