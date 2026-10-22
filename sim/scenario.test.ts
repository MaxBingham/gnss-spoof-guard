import { describe, expect, it } from "vitest";
import { GNSS_RATE_HZ } from "../sensors/gnss.ts";
import { CLEAR } from "../sensors/types.ts";
import { parseSeed } from "./args.ts";
import { runScenario } from "./scenario.ts";

describe("scenario", () => {
  it("reads the seed flag and defaults to 42", () => {
    expect(parseSeed(["--seed", "42"])).toBe(42);
    expect(parseSeed([])).toBe(42);
    expect(() => parseSeed(["--seed", "nope"])).toThrow(/--seed/);
  });

  it("finishes one flight from seed 42 and repeats it", () => {
    const first = runScenario({ seed: 42 });
    const second = runScenario({ seed: 42 });
    expect(first.landed).toBe(true);
    expect(first.finalErrorM).toBeLessThanOrEqual(5);
    expect(first.finalAltM).toBe(0);
    expect(first.steps).toBe(second.steps);
    expect(first.finalErrorM).toBe(second.finalErrorM);
    expect(first.traces.at(-1)?.estimate.x).toBe(second.traces.at(-1)?.estimate.x);
    const gnss = first.traces.filter((trace) => trace.fixes.length > 0);
    expect(gnss.length).toBeGreaterThan(1);
    expect(gnss.length).toBeLessThan(first.traces.length);
    expect(gnss.every((trace) => trace.fixes.length === 1 && trace.fixes[0]?.source === "gnss")).toBe(true);
    for (let i = 1; i < gnss.length; i++) {
      expect(gnss[i]!.t - gnss[i - 1]!.t).toBeCloseTo(1 / GNSS_RATE_HZ, 6);
    }
  });

  it("still finishes when a steady crosswind is blowing", () => {
    const result = runScenario({
      seed: 42,
      conditions: {
        ...CLEAR,
        wind: { speedMS: 4, directionRad: Math.PI / 2, rotationRadS: 0 },
      },
    });
    expect(result.landed).toBe(true);
    expect(result.finalErrorM).toBeLessThanOrEqual(5);
  });
});
