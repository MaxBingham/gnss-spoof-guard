import { describe, expect, it } from "vitest";
import { CLEAR } from "../sensors/types.ts";
import { windAcceleration as sensorWind } from "../sensors/wind.ts";
import { stepDynamics, windAcceleration, type TruthPose } from "./dynamics.ts";
import { runScenario } from "./scenario.ts";

const REST: TruthPose = { x: 0, y: 0, z: 10, vx: 0, vy: 0, vz: 0, yaw: 0 };
const HOLD = { vx: 0, vy: 0, vz: 0 };

describe("wind on the airframe", () => {
  it("uses the sensor wind model", () => {
    const lever = { speedMS: 3, directionRad: 0.4, rotationRadS: 0.1 };
    expect(windAcceleration(lever, 2.5)).toEqual(sensorWind(lever, 2.5));
  });

  it("accelerates sideways in the compass direction and ignores heading", () => {
    const wind = windAcceleration({ speedMS: 4, directionRad: 0, rotationRadS: 0 }, 0);
    const noseNorth = stepDynamics(REST, HOLD, wind, 0.02);
    const noseEast = stepDynamics({ ...REST, yaw: Math.PI / 2 }, HOLD, wind, 0.02);
    expect(noseNorth.ax).toBeCloseTo(0);
    expect(noseNorth.ay).toBeCloseTo(wind.ay);
    expect(noseNorth.az).toBeCloseTo(0);
    expect(noseEast.ax).toBe(noseNorth.ax);
    expect(noseEast.ay).toBe(noseNorth.ay);
    expect(noseNorth.pose.y).toBeGreaterThan(0);
    expect(noseNorth.pose.x).toBeCloseTo(0);
  });

  it("drifts farther as the speed lever rises, and turns as the direction rotates", () => {
    const coast = (speedMS: number, rotationRadS: number) => {
      let pose = REST;
      const dt = 0.02;
      const wind = { speedMS, directionRad: 0, rotationRadS };
      for (let step = 0; step < 200; step++) {
        pose = stepDynamics(pose, HOLD, windAcceleration(wind, step * dt), dt).pose;
      }
      return pose;
    };

    const light = coast(2, 0);
    const strong = coast(6, 0);
    expect(light.x).toBeCloseTo(0);
    expect(strong.y).toBeGreaterThan(light.y * 2);
    expect(strong.x).toBeCloseTo(0);

    const turning = coast(4, Math.PI / 8);
    const fixed = coast(4, 0);
    expect(Math.abs(turning.x)).toBeGreaterThan(0.2);
    expect(turning.y).toBeLessThan(fixed.y);
  });

  it("moves the flown path when the scenario wind lever is up", () => {
    const calm = runScenario({ seed: 5, maxTimeS: 4 });
    const crosswind = runScenario({
      seed: 5,
      maxTimeS: 4,
      conditions: { ...CLEAR, wind: { speedMS: 8, directionRad: Math.PI / 2, rotationRadS: 0 } },
    });
    const rotating = runScenario({
      seed: 5,
      maxTimeS: 4,
      conditions: { ...CLEAR, wind: { speedMS: 8, directionRad: 0, rotationRadS: 0.2 } },
    });
    const calmX = calm.traces.at(-1)!.estimate.x;
    expect(crosswind.traces.at(-1)!.estimate.x).not.toBe(calmX);
    expect(rotating.traces.at(-1)!.estimate.x).not.toBe(crosswind.traces.at(-1)!.estimate.x);
  });
});
