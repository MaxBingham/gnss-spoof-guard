import { describe, expect, it } from "vitest";
import { initialLevers, leversToConditions } from "./levers.ts";
import { mockFlight } from "./mock-flight.ts";
import { conditionRows, errorSeries, uncertaintySeries } from "./series.ts";

describe("chart series", () => {
  it("grows position error under an eastward spoof and keeps uncertainty flat", () => {
    const flight = mockFlight({ ...initialLevers(), spoofRateMS: 1, spoofDirectionDeg: 90 });
    const error = errorSeries(flight.samples);
    const sigma = uncertaintySeries(flight.samples);
    expect(error[0].value).toBeCloseTo(0);
    expect(error.at(-1)!.value).toBeGreaterThan(error[0].value);
    expect(sigma[0].t).toBe(0);
    expect(sigma.at(-1)!.value).toBeCloseTo(sigma[0].value);
  });

  it("lists the Conditions fields the levers write", () => {
    const rows = conditionRows(leversToConditions({ ...initialLevers(), gnssJammed: true, weather: 0.5 }));
    expect(rows.map((row) => row.field)).toEqual([
      "weather.intensity",
      "wind.speedMS",
      "wind.directionRad",
      "camera.occluded",
      "lidar.occluded",
      "gnss.jammed",
      "gnss.spoofRateMS",
      "gnss.spoofDirectionRad",
    ]);
    expect(rows.find((row) => row.field === "gnss.jammed")?.value).toBe("true");
    expect(rows.find((row) => row.field === "weather.intensity")?.value).toBe("0.50");
  });
});
