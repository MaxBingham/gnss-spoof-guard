import { describe, expect, it } from "vitest";
import { stepDynamics, windAcceleration, WIND_ACCEL_PER_MPS, type TruthPose } from "./dynamics.ts";
import { MISSION } from "./mission.ts";

const REST: TruthPose = { x: 0, y: 0, z: 10, vx: 0, vy: 0, vz: 0, yaw: 0 };
const HOLD = { vx: 0, vy: 0, vz: 0 };

describe("wind", () => {
  it("points the force along the compass direction and rotates it when asked", () => {
    const north = windAcceleration({ speedMS: 4, directionRad: 0, rotationRadS: 0 }, 0);
    expect(north.ax).toBeCloseTo(0);
    expect(north.ay).toBeCloseTo(4 * WIND_ACCEL_PER_MPS);

    const east = windAcceleration({ speedMS: 4, directionRad: Math.PI / 2, rotationRadS: 0 }, 0);
    expect(east.ax).toBeCloseTo(4 * WIND_ACCEL_PER_MPS);
    expect(east.ay).toBeCloseTo(0);

    const turned = windAcceleration({ speedMS: 4, directionRad: 0, rotationRadS: Math.PI / 2 }, 1);
    expect(turned.ax).toBeCloseTo(4 * WIND_ACCEL_PER_MPS);
    expect(turned.ay).toBeCloseTo(0);

    const calm = windAcceleration({ speedMS: 0, directionRad: 1, rotationRadS: 2 }, 5);
    expect(calm).toEqual({ ax: 0, ay: 0 });
  });

  it("accelerates the airframe sideways while the speed lever is up", () => {
    let pose = REST;
    const wind = windAcceleration({ speedMS: 4, directionRad: 0, rotationRadS: 0 }, 0);
    for (let i = 0; i < 50; i++) pose = stepDynamics(pose, HOLD, wind, 0.02).pose;
    expect(pose.y).toBeGreaterThan(0.15);
    expect(pose.vy).toBeGreaterThan(0.3);
    expect(Math.abs(pose.x)).toBeLessThan(1e-6);

    for (let i = 0; i < 450; i++) pose = stepDynamics(pose, HOLD, wind, 0.02).pose;
    expect(pose.vy).toBeCloseTo(wind.ay * MISSION.velocityLagS, 2);
    expect(pose.y).toBeGreaterThan(2);
  });
});
