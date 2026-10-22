import type { Conditions } from "../sensors/types.ts";
import { positionErrorM, uncertaintyRadiusM } from "./geometry.ts";
import type { FlightSample } from "./mock-flight.ts";

export interface ChartPoint {
  t: number;
  value: number;
}

export function errorSeries(samples: readonly FlightSample[]): ChartPoint[] {
  return samples.map((sample) => ({
    t: sample.trace.t,
    value: positionErrorM(sample.truth, sample.trace.estimate),
  }));
}

export function uncertaintySeries(samples: readonly FlightSample[]): ChartPoint[] {
  return samples.map((sample) => ({
    t: sample.trace.t,
    value: uncertaintyRadiusM(sample.trace.estimate.cov),
  }));
}

/** Flat view of the Conditions object the lever panel is bound to. */
export function conditionRows(conditions: Conditions): { field: string; value: string }[] {
  return [
    { field: "weather.intensity", value: conditions.weather.intensity.toFixed(2) },
    { field: "wind.speedMS", value: conditions.wind.speedMS.toFixed(1) },
    { field: "wind.directionRad", value: conditions.wind.directionRad.toFixed(2) },
    { field: "camera.occluded", value: String(conditions.camera.occluded) },
    { field: "lidar.occluded", value: String(conditions.lidar.occluded) },
    { field: "gnss.jammed", value: String(conditions.gnss.jammed) },
    { field: "gnss.spoofRateMS", value: conditions.gnss.spoofRateMS.toFixed(1) },
    { field: "gnss.spoofDirectionRad", value: conditions.gnss.spoofDirectionRad.toFixed(2) },
  ];
}
