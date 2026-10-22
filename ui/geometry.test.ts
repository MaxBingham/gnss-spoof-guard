import { describe, expect, it } from "vitest";
import { positionErrorM, uncertaintyRadiusM } from "./geometry.ts";

describe("positionErrorM", () => {
  it("is the horizontal gap between truth and the estimate", () => {
    expect(positionErrorM({ x: 0, y: 0 }, { x: 3, y: 4 })).toBe(5);
    expect(positionErrorM({ x: 10, y: -2 }, { x: 10, y: -2 })).toBe(0);
  });
});

describe("uncertaintyRadiusM", () => {
  it("is twice the square root of the largest covariance eigenvalue", () => {
    expect(uncertaintyRadiusM([4, 0, 1])).toBeCloseTo(4);
    expect(uncertaintyRadiusM([1, 0, 1])).toBeCloseTo(2);
    expect(uncertaintyRadiusM([2, 1, 2])).toBeCloseTo(2 * Math.sqrt(3));
  });

  it("is zero for a singular covariance", () => {
    expect(uncertaintyRadiusM([0, 0, 0])).toBe(0);
  });
});
