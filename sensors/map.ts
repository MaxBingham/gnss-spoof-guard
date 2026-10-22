import type { LandmarkType } from "./types.ts";

/**
 * Stand-in map. Main has no Berkeley building footprints yet.
 * Landmarks carry an id for the map only; the camera drops it.
 * Lidar casts against the footprints and roof heights. Trees have no footprint.
 */
export interface MapLandmark {
  id: string;
  type: LandmarkType;
  x: number;
  y: number;
}

export interface FootprintPoint {
  x: number;
  y: number;
}

export interface Building {
  id: string;
  heightM: number;
  /** Vertical prism from the ground up to `heightM`. Vertices in order, ring not repeated. */
  footprint: readonly FootprintPoint[];
}

/** Same positions the contract projected, now with a class. */
export const LANDMARKS: readonly MapLandmark[] = [
  { id: "oak-a", type: "tree", x: 15, y: 8 },
  { id: "known-hall", type: "building", x: 40, y: -20 },
  { id: "campanile", type: "tower", x: 70, y: -10 },
  { id: "east-block", type: "building", x: 100, y: -40 },
  { id: "oak-e", type: "tree", x: 130, y: -25 },
];

/** Wide hall used by the lidar test. The camera landmark sits on its roof center. */
export const KNOWN_BUILDING: Building = {
  id: "known-hall",
  heightM: 20,
  footprint: [
    { x: 30, y: -30 },
    { x: 50, y: -30 },
    { x: 50, y: -10 },
    { x: 30, y: -10 },
  ],
};

export const BUILDINGS: readonly Building[] = [
  KNOWN_BUILDING,
  {
    id: "campanile",
    heightM: 45,
    footprint: [
      { x: 68, y: -12 },
      { x: 72, y: -12 },
      { x: 72, y: -8 },
      { x: 68, y: -8 },
    ],
  },
  {
    id: "east-block",
    heightM: 12,
    footprint: [
      { x: 92, y: -48 },
      { x: 108, y: -48 },
      { x: 108, y: -32 },
      { x: 92, y: -32 },
    ],
  },
];
