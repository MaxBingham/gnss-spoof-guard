import { describe, expect, it } from "vitest";
import type { SensorFrame } from "../sensors/types.ts";
import { baselineFixes, passThrough } from "./stub.ts";

const FRAME: SensorFrame = {
  t: 1.5,
  imu: { ax: 0, ay: 0, az: 0, yawRateRadS: 0 },
  baro: { altM: 12 },
  compass: { yawRad: 0.2 },
  gnss: { x: 4, y: -3, cov: [2.25, 0, 2.25], quality: 1 },
  camera: { landmarks: [{ type: "tree", u: 10, v: 20 }], intrinsics: { fx: 1, fy: 1, cx: 0, cy: 0, width: 10, height: 10 } },
  lidar: { beams: [{ angleRad: 0, rangeM: 12 }] },
};

describe("baselineFixes", () => {
  it("trusts the GNSS fix and leaves camera and lidar unread", () => {
    const fixes = baselineFixes(FRAME);
    expect(fixes).toEqual([{ source: "gnss", x: 4, y: -3, cov: [2.25, 0, 2.25] }]);
    expect(passThrough(FRAME)).toEqual(fixes);
  });

  it("returns nothing when the frame has no GNSS fix, so the filter can coast", () => {
    const { gnss: _gnss, ...jammed } = FRAME;
    expect(baselineFixes(jammed)).toEqual([]);
    expect(jammed.camera).toBeDefined();
    expect(jammed.lidar).toBeDefined();
  });
});
