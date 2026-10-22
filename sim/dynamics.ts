import { clamp, wrapAngle } from "../estimation/angles.ts";
import { MISSION } from "./mission.ts";

export { windAcceleration, WIND_ACCEL_PER_MPS } from "../sensors/wind.ts";

export interface TruthPose {
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  yaw: number;
}

export interface VelocityCommand {
  vx: number;
  vy: number;
  vz: number;
}

export interface DynamicsStep {
  pose: TruthPose;
  ax: number;
  ay: number;
  az: number;
  yawRateRadS: number;
}

/** Velocity-commanded point mass with a first-order lag, plus the wind force. */
export function stepDynamics(
  pose: TruthPose,
  command: VelocityCommand,
  wind: { ax: number; ay: number },
  dt: number,
): DynamicsStep {
  const tau = MISSION.velocityLagS;
  let ax = (command.vx - pose.vx) / tau + wind.ax;
  let ay = (command.vy - pose.vy) / tau + wind.ay;
  let az = (command.vz - pose.vz) / tau;
  let vx = pose.vx + ax * dt;
  let vy = pose.vy + ay * dt;
  let vz = pose.vz + az * dt;
  const x = pose.x + pose.vx * dt + 0.5 * ax * dt * dt;
  const y = pose.y + pose.vy * dt + 0.5 * ay * dt * dt;
  let z = pose.z + pose.vz * dt + 0.5 * az * dt * dt;
  if (z <= 0) {
    z = 0;
    if (vz < 0) {
      az = dt > 0 ? -pose.vz / dt : 0;
      vz = 0;
    }
  }
  const yawRateRadS = yawRateToward(pose.yaw, command);
  const yaw = wrapAngle(pose.yaw + yawRateRadS * dt);
  return { pose: { x, y, z, vx, vy, vz, yaw }, ax, ay, az, yawRateRadS };
}

function yawRateToward(yaw: number, command: VelocityCommand): number {
  const speed = Math.hypot(command.vx, command.vy);
  if (speed < 0.3) return 0;
  const desired = Math.atan2(command.vx, command.vy);
  const error = wrapAngle(desired - yaw);
  const maxRate = (60 * Math.PI) / 180;
  return clamp(error / 0.4, -maxRate, maxRate);
}
