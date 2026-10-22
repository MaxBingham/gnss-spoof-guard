import { clamp } from "../estimation/angles.ts";
import type { Estimate } from "../estimation/types.ts";
import type { VelocityCommand } from "./dynamics.ts";
import { MISSION } from "./mission.ts";

export type Phase = "takeoff" | "cruise" | "land";

const HORIZONTAL_GAIN = 0.45;

/** Advance the flight phase from the filter estimate. Phases only move forward. */
export function nextPhase(phase: Phase, estimate: Estimate): Phase {
  if (phase === "takeoff" && estimate.z >= MISSION.cruiseAltM - 0.5) return "cruise";
  if (phase === "cruise") {
    const distance = Math.hypot(MISSION.goalX - estimate.x, MISSION.goalY - estimate.y);
    if (distance < 15) return "land";
  }
  return phase;
}

/** Velocity command that flies the planned route from the filter estimate. */
export function commandFor(phase: Phase, estimate: Estimate): VelocityCommand {
  const targetX = phase === "takeoff" ? MISSION.startX : MISSION.goalX;
  const targetY = phase === "takeoff" ? MISSION.startY : MISSION.goalY;
  const dx = targetX - estimate.x;
  const dy = targetY - estimate.y;
  const distance = Math.hypot(dx, dy);
  const speed = Math.min(MISSION.maxSpeedMS, HORIZONTAL_GAIN * distance);
  const vx = distance > 1e-6 ? (dx / distance) * speed : 0;
  const vy = distance > 1e-6 ? (dy / distance) * speed : 0;
  const zTarget = phase === "land" ? 0 : MISSION.cruiseAltM;
  const zGain = phase === "land" ? 0.5 : 0.6;
  const climb = phase === "land" ? 0 : MISSION.climbMS;
  const vz = clamp(zGain * (zTarget - estimate.z), -MISSION.sinkMS, climb);
  return { vx, vy, vz };
}
