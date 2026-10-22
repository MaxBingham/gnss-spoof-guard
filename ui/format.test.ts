import { describe, expect, it } from "vitest";
import { summarizeTrace } from "./index.ts";

describe("summarizeTrace", () => {
  it("prints the forwarded fix and the estimate", () => {
    const line = summarizeTrace({
      t: 1,
      fixes: [{ source: "gnss", x: 4, y: -2, cov: [1, 0, 1] }],
      estimate: { x: 3.5, y: -2.2, vx: 0, vy: 0, z: 10, yaw: 0, cov: [1, 0, 1] },
    });
    expect(line).toContain("gnss 4.0,-2.0");
    expect(line).toContain("est=3.5,-2.2,10.0");
  });
});
