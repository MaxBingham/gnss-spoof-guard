import { describe, expect, it } from "vitest";
import { CLEAR } from "../sensors/types.ts";
import {
  cardinal,
  conditionsToLevers,
  initialLevers,
  leversToConditions,
} from "./levers.ts";

describe("levers", () => {
  it("starts on the clear Conditions contract", () => {
    expect(leversToConditions(initialLevers())).toEqual(CLEAR);
  });

  it("binds every dashboard lever onto Conditions", () => {
    const conditions = leversToConditions({
      weather: 0.4,
      windSpeedMS: 6,
      windDirectionDeg: 90,
      cameraLost: true,
      lidarLost: false,
      gnssJammed: true,
      spoofRateMS: 1.5,
      spoofDirectionDeg: 180,
    });
    expect(conditions.weather.intensity).toBeCloseTo(0.4);
    expect(conditions.wind.speedMS).toBeCloseTo(6);
    expect(conditions.wind.directionRad).toBeCloseTo(Math.PI / 2);
    expect(conditions.wind.rotationRadS).toBe(0);
    expect(conditions.camera.occluded).toBe(true);
    expect(conditions.lidar.occluded).toBe(false);
    expect(conditions.gnss.jammed).toBe(true);
    expect(conditions.gnss.spoofRateMS).toBeCloseTo(1.5);
    expect(conditions.gnss.spoofDirectionRad).toBeCloseTo(Math.PI);
  });

  it("round-trips a Conditions value back to lever state", () => {
    const levers = initialLevers();
    levers.weather = 0.25;
    levers.windDirectionDeg = 45;
    levers.spoofRateMS = 0.5;
    expect(conditionsToLevers(leversToConditions(levers))).toEqual(levers);
  });

  it("clamps weather and treats a zero spoof rate as off", () => {
    const conditions = leversToConditions({
      ...initialLevers(),
      weather: 4,
      spoofRateMS: 0,
    });
    expect(conditions.weather.intensity).toBe(1);
    expect(conditions.gnss.spoofRateMS).toBe(0);
  });

  it("names compass directions", () => {
    expect(cardinal(0)).toBe("N");
    expect(cardinal(90)).toBe("E");
    expect(cardinal(180)).toBe("S");
    expect(cardinal(270)).toBe("W");
  });
});
