/// <reference lib="dom" />
import "leaflet/dist/leaflet.css";
import "./styles.css";
import { ChartPanel } from "./chart-view.ts";
import { positionErrorM, uncertaintyRadiusM } from "./geometry.ts";
import { mountLevers } from "./lever-panel.ts";
import { initialLevers, type LeverState } from "./levers.ts";
import { MapView } from "./map-view.ts";
import { mockFlight, type MockFlight } from "./mock-flight.ts";
import { clampCursor, sliceToCursor, stepCursor } from "./playback.ts";
import { conditionRows, errorSeries, uncertaintySeries } from "./series.ts";
import { PIPELINE_SLOTS } from "./slots.ts";

const root = document.querySelector("#app");
if (!root) throw new Error("Missing #app");

const shell = document.createElement("div");
shell.className = "app";
shell.innerHTML = `
  <header class="header">
    <div>
      <h1>GNSS Spoof Guard</h1>
      <p>Degradation test rig. Mock baseline over a local ENU map, in meters.</p>
    </div>
    <div class="badge">Pre-built test rig</div>
  </header>
  <section class="map-panel" aria-label="Map">
    <div id="map"></div>
    <div class="map-toolbar">
      <button type="button" id="play">Play</button>
      <input id="scrub" type="range" min="0" max="0" value="0" aria-label="Time" />
    </div>
    <div class="legend">
      <span><i class="swatch truth"></i>Truth</span>
      <span><i class="swatch estimate"></i>Estimate</span>
      <span><i class="swatch sigma"></i>Uncertainty</span>
    </div>
  </section>
  <aside class="side">
    <h2>Conditions</h2>
    <div id="levers"></div>
    <p class="readout" id="readout"></p>
    <p class="sources" id="sources"></p>
    <dl class="conditions" id="conditions"></dl>
  </aside>
  <div class="charts">
    <section class="chart-card" aria-label="Position error">
      <h2>Position error</h2>
      <canvas id="error-chart"></canvas>
    </section>
    <section class="chart-card" aria-label="Uncertainty">
      <h2>Uncertainty</h2>
      <canvas id="sigma-chart"></canvas>
    </section>
  </div>
  <div class="pipeline" id="slots"></div>
`;
root.append(shell);

const slots = document.querySelector("#slots");
if (!slots) throw new Error("Missing slots");
for (const slot of PIPELINE_SLOTS) {
  const section = document.createElement("section");
  section.className = "slot";
  section.dataset.slot = slot.id;
  section.setAttribute("aria-label", slot.label);
  const title = document.createElement("h2");
  title.textContent = slot.label;
  const body = document.createElement("div");
  body.className = "slot-body";
  section.append(title, body);
  slots.append(section);
}

const leversEl = required<HTMLElement>("#levers");
const conditionsEl = required<HTMLElement>("#conditions");
const readoutEl = required<HTMLElement>("#readout");
const sourcesEl = required<HTMLElement>("#sources");
const mapEl = required<HTMLElement>("#map");
const playButton = required<HTMLButtonElement>("#play");
const scrub = required<HTMLInputElement>("#scrub");
const errorCanvas = required<HTMLCanvasElement>("#error-chart");
const sigmaCanvas = required<HTMLCanvasElement>("#sigma-chart");

const map = new MapView(mapEl);
const charts = new ChartPanel(errorCanvas, sigmaCanvas);

let levers: LeverState = initialLevers();
let flight: MockFlight = mockFlight(levers);
let cursor = flight.samples.length - 1;
let playing = false;
let timer: ReturnType<typeof setInterval> | undefined;

mountLevers(leversEl, levers, (next) => {
  levers = next;
  flight = mockFlight(levers);
  cursor = flight.samples.length - 1;
  playing = false;
  sync(true);
});

playButton.addEventListener("click", () => {
  if (!playing && cursor >= flight.samples.length - 1) cursor = 0;
  playing = !playing;
  sync(false);
});

scrub.addEventListener("input", () => {
  playing = false;
  cursor = clampCursor(flight.samples.length, Number(scrub.value));
  sync(false);
});

window.addEventListener("resize", () => map.invalidate());
sync(true);
requestAnimationFrame(() => map.invalidate());

function required<T extends Element>(selector: string): T {
  const node = document.querySelector<T>(selector);
  if (!node) throw new Error(`Missing ${selector}`);
  return node;
}

function sync(refit: boolean): void {
  cursor = clampCursor(flight.samples.length, cursor);
  if (playing) {
    if (timer === undefined) timer = setInterval(tick, 80);
  } else if (timer !== undefined) {
    clearInterval(timer);
    timer = undefined;
  }
  playButton.textContent = playing ? "Pause" : "Play";
  scrub.max = String(Math.max(0, flight.samples.length - 1));
  scrub.value = String(cursor);
  const visible = sliceToCursor(flight.samples, cursor);
  map.setFlight(flight.samples, cursor, refit);
  charts.setSeries(errorSeries(visible), uncertaintySeries(visible));
  paintConditions();
  paintReadout(visible.at(-1));
}

function tick(): void {
  const stepped = stepCursor(flight.samples.length, cursor);
  cursor = stepped.cursor;
  playing = stepped.playing;
  sync(false);
}

function paintConditions(): void {
  conditionsEl.replaceChildren();
  for (const row of conditionRows(flight.conditions)) {
    const field = document.createElement("dt");
    field.textContent = row.field;
    const value = document.createElement("dd");
    value.textContent = row.value;
    conditionsEl.append(field, value);
  }
}

function paintReadout(sample: MockFlight["samples"][number] | undefined): void {
  if (!sample) {
    readoutEl.textContent = "";
    sourcesEl.textContent = "";
    return;
  }
  const error = positionErrorM(sample.truth, sample.trace.estimate);
  const sigma = uncertaintyRadiusM(sample.trace.estimate.cov);
  readoutEl.textContent = `t ${sample.trace.t.toFixed(1)} s · error ${error.toFixed(1)} m · 2σ ${sigma.toFixed(1)} m · truth ${sample.truth.x.toFixed(1)}, ${sample.truth.y.toFixed(1)} · est ${sample.trace.estimate.x.toFixed(1)}, ${sample.trace.estimate.y.toFixed(1)}`;
  const sources = sample.trace.fixes.map((fix) => fix.source);
  sourcesEl.textContent = sources.length > 0 ? `Fixes: ${sources.join(", ")}` : "Fixes: none";
}
