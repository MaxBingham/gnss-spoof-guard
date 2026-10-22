/// <reference lib="dom" />
import * as L from "leaflet";
import { uncertaintyRadiusM } from "./geometry.ts";
import { MOCK_GOAL, type FlightSample } from "./mock-flight.ts";
import { sliceToCursor } from "./playback.ts";

const TRUTH = "#1e3a5f";
const ESTIMATE = "#c2410c";

export class MapView {
  private readonly map: L.Map;
  private readonly grid = L.layerGroup();
  private readonly truthLine: L.Polyline;
  private readonly estimateLine: L.Polyline;
  private readonly sigma: L.Polygon;
  private readonly truthMark: L.CircleMarker;
  private readonly estimateMark: L.CircleMarker;
  private fittedKey = "";

  constructor(container: HTMLElement) {
    this.map = L.map(container, {
      crs: L.CRS.Simple,
      zoomControl: false,
      attributionControl: false,
      minZoom: -4,
      zoomSnap: 0.25,
    });
    L.control.zoom({ position: "topright" }).addTo(this.map);
    this.grid.addTo(this.map);
    this.sigma = L.polygon([], {
      color: ESTIMATE,
      weight: 1.5,
      fillColor: ESTIMATE,
      fillOpacity: 0.15,
    }).addTo(this.map);
    this.truthLine = L.polyline([], { color: TRUTH, weight: 2 }).addTo(this.map);
    this.estimateLine = L.polyline([], { color: ESTIMATE, weight: 2, dashArray: "6 4" }).addTo(this.map);
    this.truthMark = L.circleMarker([0, 0], {
      radius: 6,
      color: TRUTH,
      fillColor: TRUTH,
      fillOpacity: 1,
      weight: 2,
    }).addTo(this.map);
    this.estimateMark = L.circleMarker([0, 0], {
      radius: 6,
      color: ESTIMATE,
      fillColor: "#fff7ed",
      fillOpacity: 1,
      weight: 2,
    }).addTo(this.map);
    L.circleMarker([MOCK_GOAL.y, MOCK_GOAL.x], {
      radius: 5,
      color: "#44403c",
      fillColor: "#fafaf9",
      fillOpacity: 1,
      weight: 1.5,
    })
      .addTo(this.map)
      .bindTooltip("Goal");
    L.circleMarker([0, 0], {
      radius: 4,
      color: "#44403c",
      fillColor: "#fafaf9",
      fillOpacity: 1,
      weight: 1.5,
    })
      .addTo(this.map)
      .bindTooltip("Start");
    this.map.setView([0, 40], 1);
  }

  setFlight(samples: readonly FlightSample[], cursor: number, refit: boolean): void {
    const visible = sliceToCursor(samples, cursor);
    const current = visible.at(-1);
    if (!current) return;
    this.truthLine.setLatLngs(visible.map((sample) => latLng(sample.truth.x, sample.truth.y)));
    this.estimateLine.setLatLngs(
      visible.map((sample) => latLng(sample.trace.estimate.x, sample.trace.estimate.y)),
    );
    const radius = uncertaintyRadiusM(current.trace.estimate.cov);
    this.sigma.setLatLngs(circle(current.trace.estimate.x, current.trace.estimate.y, radius));
    this.truthMark.setLatLng(latLng(current.truth.x, current.truth.y));
    this.estimateMark.setLatLng(latLng(current.trace.estimate.x, current.trace.estimate.y));
    const key = boundsKey(samples);
    if (refit || this.fittedKey !== key) {
      this.drawGrid(samples);
      const bounds = L.latLngBounds(
        samples.flatMap((sample) => [
          latLng(sample.truth.x, sample.truth.y),
          latLng(sample.trace.estimate.x, sample.trace.estimate.y),
        ]),
      );
      bounds.extend(latLng(0, 0));
      bounds.extend(latLng(MOCK_GOAL.x, MOCK_GOAL.y));
      this.map.fitBounds(bounds.pad(0.2));
      this.fittedKey = key;
    }
  }

  invalidate(): void {
    this.map.invalidateSize();
  }

  private drawGrid(samples: readonly FlightSample[]): void {
    this.grid.clearLayers();
    let minX = 0;
    let maxX: number = MOCK_GOAL.x;
    let minY: number = MOCK_GOAL.y;
    let maxY = 0;
    for (const sample of samples) {
      minX = Math.min(minX, sample.truth.x, sample.trace.estimate.x);
      maxX = Math.max(maxX, sample.truth.x, sample.trace.estimate.x);
      minY = Math.min(minY, sample.truth.y, sample.trace.estimate.y);
      maxY = Math.max(maxY, sample.truth.y, sample.trace.estimate.y);
    }
    const step = 20;
    const x0 = Math.floor((minX - 20) / step) * step;
    const x1 = Math.ceil((maxX + 20) / step) * step;
    const y0 = Math.floor((minY - 20) / step) * step;
    const y1 = Math.ceil((maxY + 20) / step) * step;
    const style = { color: "#d6d3d1", weight: 1 };
    for (let x = x0; x <= x1; x += step) {
      L.polyline([latLng(x, y0), latLng(x, y1)], style).addTo(this.grid);
    }
    for (let y = y0; y <= y1; y += step) {
      L.polyline([latLng(x0, y), latLng(x1, y)], style).addTo(this.grid);
    }
  }
}

function latLng(x: number, y: number): L.LatLng {
  return L.latLng(y, x);
}

function circle(x: number, y: number, radius: number, steps = 64): L.LatLng[] {
  const points: L.LatLng[] = [];
  const safe = Math.max(radius, 0.05);
  for (let i = 0; i < steps; i++) {
    const angle = (i / steps) * Math.PI * 2;
    points.push(latLng(x + safe * Math.cos(angle), y + safe * Math.sin(angle)));
  }
  return points;
}

function boundsKey(samples: readonly FlightSample[]): string {
  const last = samples.at(-1);
  if (!last) return "";
  return [
    samples.length,
    last.truth.x.toFixed(1),
    last.truth.y.toFixed(1),
    last.trace.estimate.x.toFixed(1),
    last.trace.estimate.y.toFixed(1),
  ].join(":");
}
