import { describe, expect, it } from "vitest";
import { KalmanFilter } from "./kalman.ts";

const STILL = { ax: 0, ay: 0, az: 0, yawRateRadS: 0 };

describe("KalmanFilter", () => {
  it("dead-reckons a constant acceleration", () => {
    const filter = new KalmanFilter();
    const dt = 0.02;
    for (let i = 0; i < 50; i++) filter.predict({ ax: 1, ay: -0.5, az: 0, yawRateRadS: 0 }, dt);
    const estimate = filter.estimate();
    expect(estimate.vx).toBeCloseTo(1, 4);
    expect(estimate.vy).toBeCloseTo(-0.5, 4);
    expect(estimate.x).toBeCloseTo(0.5, 4);
    expect(estimate.y).toBeCloseTo(-0.25, 4);
  });

  it("corrects horizontal position from a fix", () => {
    const filter = new KalmanFilter();
    filter.updatePosition(10, -4, [2.25, 0, 2.25]);
    const estimate = filter.estimate();
    const gain = 25 / (25 + 2.25);
    expect(estimate.x).toBeCloseTo(gain * 10, 4);
    expect(estimate.y).toBeCloseTo(gain * -4, 4);
  });

  it("tracks altitude from the barometer", () => {
    const filter = new KalmanFilter();
    for (let i = 0; i < 100; i++) {
      filter.predict(STILL, 0.05);
      filter.updateBaro(20);
    }
    expect(filter.estimate().z).toBeCloseTo(20, 1);
  });

  it("corrects heading from the compass", () => {
    const filter = new KalmanFilter();
    for (let i = 0; i < 80; i++) {
      filter.predict(STILL, 0.05);
      filter.updateCompass(Math.PI / 2);
    }
    expect(filter.estimate().yaw).toBeCloseTo(Math.PI / 2, 2);
  });

  it("attributes a constant accelerometer offset to bias while position stays put", () => {
    const filter = new KalmanFilter();
    const dt = 0.05;
    for (let i = 0; i < 400; i++) {
      filter.predict({ ax: 0.2, ay: 0, az: 0, yawRateRadS: 0 }, dt);
      filter.updatePosition(0, 0, [0.25, 0, 0.25]);
    }
    const estimate = filter.estimate();
    expect(Math.abs(estimate.x)).toBeLessThan(0.5);
    expect(Math.abs(estimate.vx)).toBeLessThan(0.2);
  });
});
