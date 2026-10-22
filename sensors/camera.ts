import { LANDMARKS } from "./map.ts";
import { gaussian } from "./rng.ts";
import type { BodyState, CameraIntrinsics, CameraLandmark, CameraReading, LandmarkType } from "./types.ts";

export const CAMERA = {
  fx: 500,
  fy: 500,
  cx: 320,
  cy: 240,
  width: 640,
  height: 480,
} as const satisfies CameraIntrinsics;

const FALSE_BLOB_TRIES = 3;
const FALSE_BLOB_P = 0.8;
const BLOB_TYPES: readonly LandmarkType[] = ["building", "tree", "tower"];

export interface CameraWeather {
  /** Weather lever, clamped to [0, 1]. Drives false blobs. */
  intensity: number;
  pixelSigma: number;
  dropoutP: number;
}

/**
 * Nadir pinhole shared with the contract. Image x is east and image y is north,
 * so the pixels do not rotate with yaw.
 * Returns null when the landmark is behind the camera or outside the frame.
 */
export function projectLandmark(
  pose: Pick<BodyState, "x" | "y" | "z">,
  landmark: { x: number; y: number },
): { u: number; v: number } | null {
  if (pose.z <= 1) return null;
  const u = CAMERA.cx + (CAMERA.fx * (landmark.x - pose.x)) / pose.z;
  const v = CAMERA.cy + (CAMERA.fy * (landmark.y - pose.y)) / pose.z;
  if (u < 0 || v < 0 || u > CAMERA.width || v > CAMERA.height) return null;
  return { u, v };
}

/**
 * ID-free blobs `{type, u, v}`. Weather adds pixel noise, misses, and up to a few false blobs.
 * Intensity 0 keeps the projected landmarks exactly and draws no clutter.
 */
export function sampleCamera(
  pose: Pick<BodyState, "x" | "y" | "z">,
  weather: CameraWeather,
  uniform: () => number,
): CameraReading {
  const landmarks: CameraLandmark[] = [];
  for (const landmark of LANDMARKS) {
    const projected = projectLandmark(pose, landmark);
    if (!projected) continue;
    const dropped = uniform() < weather.dropoutP;
    const u = projected.u + gaussian(uniform) * weather.pixelSigma;
    const v = projected.v + gaussian(uniform) * weather.pixelSigma;
    if (!dropped) landmarks.push({ type: landmark.type, u, v });
  }

  const intensity = clamp01(weather.intensity);
  if (intensity > 0) {
    for (let i = 0; i < FALSE_BLOB_TRIES; i++) {
      if (uniform() >= intensity * FALSE_BLOB_P) continue;
      const type = BLOB_TYPES[Math.min(BLOB_TYPES.length - 1, Math.floor(uniform() * BLOB_TYPES.length))]!;
      landmarks.push({
        type,
        u: uniform() * CAMERA.width,
        v: uniform() * CAMERA.height,
      });
    }
  }

  return { landmarks, intrinsics: CAMERA };
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}
