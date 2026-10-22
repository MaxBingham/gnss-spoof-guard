import { describe, expect, it } from "vitest";
import { beamPoint, castDownBeam, cleanLidarRange, LIDAR_BEAMS } from "./lidar.ts";
import type { Building } from "./map.ts";
import { KNOWN_BUILDING } from "./map.ts";
import { createSensorState, sampleFrame } from "./sample.ts";
import type { BodyState } from "./types.ts";
import { CLEAR } from "./types.ts";

const HALL_POSE: BodyState = {
  x: 40,
  y: -20,
  z: 50,
  ax: 0,
  ay: 0,
  az: 0,
  yaw: 0,
  yawRateRadS: 0,
};

describe("lidar", () => {
  it("returns the roof point cloud of a known building", () => {
    const frame = sampleFrame(HALL_POSE, CLEAR, createSensorState(2), 0.02, 0);
    const beams = frame.lidar?.beams;
    expect(beams?.map((beam) => beam.angleRad)).toEqual([...LIDAR_BEAMS]);
    for (const beam of beams ?? []) {
      const expected = (HALL_POSE.z - KNOWN_BUILDING.heightM) / Math.cos(beam.angleRad);
      expect(beam.rangeM).toBeCloseTo(expected, 6);
      expect(expected).toBeLessThan(cleanLidarRange(HALL_POSE.z, beam.angleRad) - 1);
      const point = beamPoint(HALL_POSE, beam.angleRad, beam.rangeM!);
      expect(point.z).toBeCloseTo(KNOWN_BUILDING.heightM, 5);
      expect(point.x).toBeGreaterThan(30);
      expect(point.x).toBeLessThan(50);
      expect(point.y).toBeGreaterThan(-30);
      expect(point.y).toBeLessThan(-10);
    }
  });

  it("hits the known hall's roof from straight down and the ground beside it", () => {
    const nadir = castDownBeam(HALL_POSE, 0);
    expect(nadir).toBeCloseTo(HALL_POSE.z - KNOWN_BUILDING.heightM, 6);
    const beside = { ...HALL_POSE, x: 28 };
    for (const angle of LIDAR_BEAMS) {
      const range = castDownBeam(beside, angle);
      if (angle <= 0.02) {
        expect(range).toBeCloseTo(cleanLidarRange(beside.z, angle), 6);
      } else {
        const point = beamPoint(beside, angle, range);
        expect(range).toBeLessThan(cleanLidarRange(beside.z, angle) - 1);
        expect(point.z).toBeGreaterThan(0);
        expect(point.z).toBeLessThanOrEqual(KNOWN_BUILDING.heightM + 1e-6);
        expect(point.x).toBeGreaterThanOrEqual(30 - 1e-6);
      }
    }
  });

  it("casts a wall, clears a short block, and steers the fan with yaw", () => {
    const wall: Building = {
      id: "wall",
      heightM: 40,
      footprint: [
        { x: 4, y: -10 },
        { x: 12, y: -10 },
        { x: 12, y: 10 },
        { x: 4, y: 10 },
      ],
    };
    const angle = 0.3;
    const pose = { x: 0, y: 0, z: 20, yaw: 0 };
    const range = castDownBeam(pose, angle, [wall]);
    expect(range).toBeCloseTo(4 / Math.sin(angle), 6);
    const point = beamPoint(pose, angle, range);
    expect(point.x).toBeCloseTo(4, 4);
    expect(point.z).toBeGreaterThan(1);
    expect(point.z).toBeLessThan(wall.heightM);

    const low: Building = {
      id: "low",
      heightM: 2,
      footprint: [
        { x: 4, y: -3 },
        { x: 5, y: -3 },
        { x: 5, y: 3 },
        { x: 4, y: 3 },
      ],
    };
    const over = { x: 0, y: 0, z: 30, yaw: 0 };
    expect(castDownBeam(over, 0.2, [low])).toBeCloseTo(cleanLidarRange(over.z, 0.2), 6);

    const north: Building = {
      id: "north",
      heightM: 10,
      footprint: [
        { x: -8, y: 4 },
        { x: 8, y: 4 },
        { x: 8, y: 20 },
        { x: -8, y: 20 },
      ],
    };
    const level = { x: 0, y: 0, z: 40, yaw: 0 };
    const facing = { ...level, yaw: -Math.PI / 2 };
    expect(castDownBeam(level, 0.2, [north])).toBeCloseTo(cleanLidarRange(level.z, 0.2), 6);
    expect(castDownBeam(facing, 0.2, [north])).toBeCloseTo((level.z - north.heightM) / Math.cos(0.2), 6);
  });

  it("leaves open ground unchanged and ignores a tree with no footprint", () => {
    const open = { x: 0, y: 0, z: 50, yaw: 0.4 };
    for (const angle of LIDAR_BEAMS) {
      expect(castDownBeam(open, angle)).toBe(cleanLidarRange(open.z, angle));
    }
    const overTree = { x: 15, y: 8, z: 40, yaw: 0 };
    expect(castDownBeam(overTree, 0)).toBe(40);
  });

  it("deletes the lidar reading when the switch is on", () => {
    const frame = sampleFrame(HALL_POSE, { ...CLEAR, lidar: { occluded: true } }, createSensorState(1), 0.02, 1);
    expect(frame.lidar).toBeUndefined();
    expect(frame.camera).toBeDefined();
  });
});
