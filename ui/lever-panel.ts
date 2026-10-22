/// <reference lib="dom" />
import { LEVER_LIMITS, cardinal, type LeverState } from "./levers.ts";

type LeverKey = keyof LeverState;

export function mountLevers(container: HTMLElement, initial: LeverState, onChange: (next: LeverState) => void): void {
  const state: LeverState = { ...initial };
  container.replaceChildren();

  const sliders: [LeverKey, string, { min: number; max: number; step: number }, (value: number) => string][] = [
    ["weather", "Weather noise", LEVER_LIMITS.weather, (value) => value.toFixed(2)],
    ["windSpeedMS", "Wind speed", LEVER_LIMITS.windSpeedMS, (value) => `${value.toFixed(1)} m/s`],
    ["windDirectionDeg", "Wind direction", LEVER_LIMITS.windDirectionDeg, (value) => `${Math.round(value)}° ${cardinal(value)}`],
    ["spoofRateMS", "GNSS spoof drift", LEVER_LIMITS.spoofRateMS, (value) => (value === 0 ? "off" : `${value.toFixed(1)} m/s`)],
    ["spoofDirectionDeg", "Spoof direction", LEVER_LIMITS.spoofDirectionDeg, (value) => `${Math.round(value)}° ${cardinal(value)}`],
  ];

  const form = document.createElement("div");
  form.className = "levers";
  for (const [key, label, limits, format] of sliders) {
    form.append(slider(state, key, label, limits, format, () => onChange({ ...state })));
  }
  form.append(switches(state, () => onChange({ ...state })));
  container.append(form);
}

function slider(
  state: LeverState,
  key: LeverKey,
  label: string,
  limits: { min: number; max: number; step: number },
  format: (value: number) => string,
  onInput: () => void,
): HTMLElement {
  const row = document.createElement("label");
  row.className = "lever";
  const name = document.createElement("span");
  name.textContent = label;
  const value = document.createElement("span");
  const input = document.createElement("input");
  input.type = "range";
  input.min = String(limits.min);
  input.max = String(limits.max);
  input.step = String(limits.step);
  input.value = String(state[key]);
  input.name = key;
  input.setAttribute("aria-label", label);
  const paint = () => {
    value.textContent = format(Number(state[key]));
  };
  input.addEventListener("input", () => {
    state[key] = Number(input.value) as never;
    paint();
    onInput();
  });
  paint();
  row.append(name, value, input);
  return row;
}

function switches(state: LeverState, onInput: () => void): HTMLElement {
  const row = document.createElement("div");
  row.className = "switches";
  const items: [LeverKey, string][] = [
    ["cameraLost", "Camera loss"],
    ["lidarLost", "Lidar loss"],
    ["gnssJammed", "GNSS jam"],
  ];
  for (const [key, label] of items) {
    const wrap = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.name = key;
    input.checked = Boolean(state[key]);
    input.setAttribute("aria-label", label);
    input.addEventListener("change", () => {
      state[key] = input.checked as never;
      onInput();
    });
    wrap.append(input, document.createTextNode(label));
    row.append(wrap);
  }
  return row;
}
