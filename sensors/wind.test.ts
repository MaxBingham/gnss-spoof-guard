import { describe, expect, it } from "vitest";
import { WIND_ACCEL_PER_MPS, windAcceleration } from "./wind.ts";

const NORTH = { speedMS: 4, directionRad: 0, rotationRadS: 0 };

describe("wind lever", () => {
  it("is calm at speed 0, whatever the direction and rotation are", () => {
    expect(windAcceleration({ speedMS: 0, directionRad: 1.2, rotationRadS: 3 }, 8)).toEqual({ ax: 0, ay: 0 });
  });

  it("pushes along a fixed compass direction and scales with speed", () => {
    const north = windAcceleration(NORTH, 0);
    expect(north.ax).toBeCloseTo(0);
    expect(north.ay).toBeCloseTo(4 * WIND_ACCEL_PER_MPS);

    const east = windAcceleration({ speedMS: 4, directionRad: Math.PI / 2, rotationRadS: 0 }, 0);
    expect(east.ax).toBeCloseTo(4 * WIND_ACCEL_PER_MPS);
    expect(east.ay).toBeCloseTo(0);

    const south = windAcceleration({ speedMS: 4, directionRad: Math.PI, rotationRadS: 0 }, 0);
    expect(south.ax).toBeCloseTo(0);
    expect(south.ay).toBeCloseTo(-4 * WIND_ACCEL_PER_MPS);

    const west = windAcceleration({ speedMS: 4, directionRad: -Math.PI / 2, rotationRadS: 0 }, 0);
    expect(west.ax).toBeCloseTo(-4 * WIND_ACCEL_PER_MPS);
    expect(west.ay).toBeCloseTo(0);

    const harder = windAcceleration({ ...NORTH, speedMS: 8 }, 10);
    expect(harder.ay).toBeCloseTo(north.ay * 2);
    expect(harder.ax).toBeCloseTo(0);
  });

  it("holds the direction fixed when rotation is zero", () => {
    const early = windAcceleration(NORTH, 0);
    const later = windAcceleration(NORTH, 30);
    expect(later).toEqual(early);
  });

  it("rotates that compass direction slowly when asked", () => {
    const quarterTurn = windAcceleration({ speedMS: 4, directionRad: 0, rotationRadS: Math.PI / 2 }, 1);
    expect(quarterTurn.ax).toBeCloseTo(4 * WIND_ACCEL_PER_MPS);
    expect(quarterTurn.ay).toBeCloseTo(0);

    const startedEast = windAcceleration(
      { speedMS: 2, directionRad: Math.PI / 2, rotationRadS: Math.PI / 2 },
      1,
    );
    expect(startedEast.ax).toBeCloseTo(0);
    expect(startedEast.ay).toBeCloseTo(-2 * WIND_ACCEL_PER_MPS);

    const barely = windAcceleration({ speedMS: 4, directionRad: 0, rotationRadS: 0.05 }, 1);
    expect(barely.ax).toBeGreaterThan(0);
    expect(barely.ay).toBeGreaterThan(4 * WIND_ACCEL_PER_MPS * 0.9);
  });
});
