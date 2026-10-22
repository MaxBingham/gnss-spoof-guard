/** Horizontal covariance, row-major `[xx, xy, yy]`. */
export type Cov2 = readonly [xx: number, xy: number, yy: number];

export interface ImuSample {
  /** Specific force, m/s². */
  ax: number;
  ay: number;
  az: number;
  /** Yaw rate, rad/s. */
  yawRateRadS: number;
}

export interface GnssFix {
  x: number;
  y: number;
  cov: Cov2;
  /** 0..1 self-reported quality. The fix is omitted entirely when jammed. */
  quality: number;
}

/** Raw camera landmark position in pixels. No landmark id. */
export interface CameraLandmark {
  u: number;
  v: number;
}

export interface CameraIntrinsics {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  width: number;
  height: number;
}

export interface CameraReading {
  landmarks: CameraLandmark[];
  intrinsics: CameraIntrinsics;
}

export interface LidarBeam {
  angleRad: number;
  /** Null when weather dropout deletes this return. */
  rangeM: number | null;
}

export interface LidarReading {
  beams: LidarBeam[];
}

/**
 * One sample of every onboard sensor.
 * GNSS, camera, and lidar are omitted when their switch deletes that data.
 */
export interface SensorFrame {
  t: number;
  imu: ImuSample;
  baro: { altM: number };
  compass: { yawRad: number };
  gnss?: GnssFix;
  camera?: CameraReading;
  lidar?: LidarReading;
}

/**
 * Levers for one moment of a flight.
 * Weather is one intensity. Wind is a sideways force. Occlusion, jam, and
 * spoof are separate from each other.
 */
export interface Conditions {
  /**
   * One lever, in [0, 1]. Scales Gaussian noise and dropout on raw camera
   * landmark positions and lidar ranges.
   */
  weather: { intensity: number };
  /**
   * Sideways force in a compass direction.
   * `directionRad` is radians clockwise from north (0 north, π/2 east).
   * `rotationRadS` is the optional slow rotation of that direction; 0 keeps it fixed.
   */
  wind: { speedMS: number; directionRad: number; rotationRadS: number };
  /** Occlusion switch. True deletes the camera reading. */
  camera: { occluded: boolean };
  /** Occlusion switch. True deletes the lidar reading. */
  lidar: { occluded: boolean };
  /**
   * `jammed` deletes the GNSS fix.
   * `spoofRateMS` drifts a false position at this rate. Zero means spoof is off.
   * `spoofDirectionRad` is the compass direction of that drift.
   */
  gnss: { jammed: boolean; spoofRateMS: number; spoofDirectionRad: number };
}

/** Kinematic state the sensor models read. The pipeline never receives this. */
export interface BodyState {
  x: number;
  y: number;
  z: number;
  ax: number;
  ay: number;
  az: number;
  yaw: number;
  yawRateRadS: number;
}

export const CLEAR: Conditions = {
  weather: { intensity: 0 },
  wind: { speedMS: 0, directionRad: 0, rotationRadS: 0 },
  camera: { occluded: false },
  lidar: { occluded: false },
  gnss: { jammed: false, spoofRateMS: 0, spoofDirectionRad: 0 },
};
