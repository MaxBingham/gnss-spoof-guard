import { compassUnit } from "./compass.ts";
import { gaussian } from "./rng.ts";
import type { Conditions, Cov2, GnssFix } from "./types.ts";

/** Fix rate. The sim steps faster; frames between epochs omit GNSS. */
export const GNSS_RATE_HZ = 5;

/** Horizontal noise, meters, on each published fix. */
export const GNSS_SIGMA_M = 1.5;

const GNSS_PERIOD_S = 1 / GNSS_RATE_HZ;

const GNSS_COV: Cov2 = [GNSS_SIGMA_M * GNSS_SIGMA_M, 0, GNSS_SIGMA_M * GNSS_SIGMA_M];

/** Drift and schedule carried on the sensor state. */
export interface GnssState {
  spoofXM: number;
  spoofYM: number;
  /**
   * Seconds until the next fix.
   * `0` publishes on this sample, then arms a full period.
   */
  gnssWaitS: number;
}

/**
 * One GNSS fix, or nothing when this step is between epochs or the jam switch is on.
 * `spoofRateMS` of zero clears the false offset. Any other rate drifts it, m/s,
 * along `spoofDirectionRad` (compass: 0 north, π/2 east).
 */
export function sampleGnss(
  pose: { x: number; y: number },
  lever: Conditions["gnss"],
  state: GnssState,
  uniform: () => number,
  dt: number,
): GnssFix | undefined {
  if (lever.spoofRateMS === 0) {
    state.spoofXM = 0;
    state.spoofYM = 0;
  } else {
    const drift = compassUnit(lever.spoofDirectionRad);
    state.spoofXM += lever.spoofRateMS * dt * drift.x;
    state.spoofYM += lever.spoofRateMS * dt * drift.y;
  }

  if (!takeEpoch(state, dt) || lever.jammed) return undefined;

  return {
    x: pose.x + state.spoofXM + gaussian(uniform) * GNSS_SIGMA_M,
    y: pose.y + state.spoofYM + gaussian(uniform) * GNSS_SIGMA_M,
    cov: GNSS_COV,
    quality: 1,
  };
}

function takeEpoch(state: GnssState, dt: number): boolean {
  if (state.gnssWaitS <= 1e-9) {
    state.gnssWaitS = GNSS_PERIOD_S;
    return true;
  }
  state.gnssWaitS -= dt;
  if (state.gnssWaitS > 1e-9) return false;
  state.gnssWaitS = GNSS_PERIOD_S;
  return true;
}
