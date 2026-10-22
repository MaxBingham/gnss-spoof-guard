import type { Building, FootprintPoint } from "./map.ts";
import { BUILDINGS } from "./map.ts";
import { gaussian } from "./rng.ts";
import type { BodyState, LidarReading } from "./types.ts";

/** Nadir cross-track beam angles, radians from straight down toward body-right. */
export const LIDAR_BEAMS = [-0.14, -0.1, -0.06, -0.02, 0.02, 0.06, 0.1, 0.14] as const;

export interface LidarWeather {
  rangeSigma: number;
  dropoutP: number;
}

/** Flat-ground range. A beam that misses every building returns this. */
export function cleanLidarRange(altitudeM: number, angleRad: number): number {
  return altitudeM / Math.cos(angleRad);
}

/**
 * First range from a downward ray to a building prism or the ground.
 * Positive angles look along body-right. Yaw 0 points that fan east.
 */
export function castDownBeam(
  pose: Pick<BodyState, "x" | "y" | "z" | "yaw">,
  angleRad: number,
  buildings: readonly Building[] = BUILDINGS,
): number {
  const cosA = Math.cos(angleRad);
  const sinA = Math.sin(angleRad);
  if (pose.z <= 0 || !(cosA > 1e-8)) return 0;
  const dx = sinA * Math.cos(pose.yaw);
  const dy = -sinA * Math.sin(pose.yaw);
  let best = cleanLidarRange(pose.z, angleRad);
  for (const building of buildings) {
    const hit = hitPrism(pose.x, pose.y, pose.z, dx, dy, cosA, building);
    if (hit !== null && hit < best) best = hit;
  }
  return best;
}

/** Point in the world frame for one returned beam. `rangeM` is null on dropout. */
export function beamPoint(
  pose: Pick<BodyState, "x" | "y" | "z" | "yaw">,
  angleRad: number,
  rangeM: number,
): { x: number; y: number; z: number } {
  const sinA = Math.sin(angleRad);
  return {
    x: pose.x + rangeM * sinA * Math.cos(pose.yaw),
    y: pose.y - rangeM * sinA * Math.sin(pose.yaw),
    z: pose.z - rangeM * Math.cos(angleRad),
  };
}

/**
 * Downward multi-beam cloud. Each beam is a polar point: angle from nadir, range to
 * the first roof, wall, or the ground. Weather noise and dropout follow the lever.
 * The occlusion switch lives in `sampleFrame`, which omits this reading entirely.
 */
export function sampleLidar(
  pose: Pick<BodyState, "x" | "y" | "z" | "yaw">,
  weather: LidarWeather,
  uniform: () => number,
): LidarReading {
  return {
    beams: LIDAR_BEAMS.map((angleRad) => {
      const dropped = uniform() < weather.dropoutP;
      const range = Math.max(0, castDownBeam(pose, angleRad) + gaussian(uniform) * weather.rangeSigma);
      return { angleRad, rangeM: dropped ? null : range };
    }),
  };
}

function hitPrism(
  ox: number,
  oy: number,
  oz: number,
  dx: number,
  dy: number,
  cosA: number,
  building: Building,
): number | null {
  const { heightM, footprint } = building;
  if (heightM <= 0 || footprint.length < 3) return null;
  if (oz >= 0 && oz <= heightM && pointInPolygon(ox, oy, footprint)) return 0;

  let best: number | null = null;
  const consider = (s: number) => {
    if (s >= 0 && (best === null || s < best)) best = s;
  };

  if (oz > heightM) {
    const sRoof = (oz - heightM) / cosA;
    if (pointInPolygon(ox + sRoof * dx, oy + sRoof * dy, footprint)) consider(sRoof);
  }

  for (let i = 0; i < footprint.length; i++) {
    const a = footprint[i]!;
    const b = footprint[(i + 1) % footprint.length]!;
    const s = raySegment(ox, oy, dx, dy, a.x, a.y, b.x, b.y);
    if (s === null) continue;
    const zHit = oz - s * cosA;
    if (zHit >= -1e-6 && zHit <= heightM + 1e-6) consider(s);
  }
  return best;
}

function raySegment(
  ox: number,
  oy: number,
  dx: number,
  dy: number,
  ax: number,
  ay: number,
  bx: number,
  by: number,
): number | null {
  const ex = bx - ax;
  const ey = by - ay;
  const det = dx * ey - dy * ex;
  if (Math.abs(det) < 1e-12) return null;
  const qpx = ax - ox;
  const qpy = ay - oy;
  const s = (qpx * ey - qpy * ex) / det;
  const u = (qpx * dy - qpy * dx) / det;
  if (u < -1e-8 || u > 1 + 1e-8 || s < -1e-6) return null;
  return s < 0 ? 0 : s;
}

function pointInPolygon(x: number, y: number, ring: readonly FootprintPoint[]): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const pi = ring[i]!;
    const pj = ring[j]!;
    const crosses = pi.y > y !== pj.y > y;
    if (crosses && x < ((pj.x - pi.x) * (y - pi.y)) / (pj.y - pi.y) + pi.x) inside = !inside;
  }
  return inside;
}
