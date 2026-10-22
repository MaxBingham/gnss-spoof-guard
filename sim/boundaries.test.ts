import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");

function sources(folder: string): string {
  return readdirSync(join(root, folder))
    .filter((name) => name.endsWith(".ts") && !name.endsWith(".test.ts"))
    .map((name) => readFileSync(join(root, folder, name), "utf8"))
    .join("\n");
}

function imports(folder: string, layer: string): boolean {
  return new RegExp(`from ["'](?:\\.\\./)+${layer}/`).test(sources(folder));
}

describe("module boundaries", () => {
  it("keeps sim free of DOM and rendering libraries", () => {
    const src = sources("sim");
    expect(src).not.toMatch(/\b(document|window|localStorage|HTMLElement)\b/);
    expect(src).not.toMatch(/from ["'](?:three|leaflet)/);
  });

  it("keeps sensors, estimation, and the pipeline off the sim", () => {
    expect(imports("sensors", "sim")).toBe(false);
    expect(imports("estimation", "sim")).toBe(false);
    expect(imports("pipeline", "sim")).toBe(false);
    expect(imports("ui", "sim")).toBe(false);
    expect(imports("sensors", "estimation")).toBe(false);
    expect(imports("sensors", "pipeline")).toBe(false);
    expect(imports("estimation", "pipeline")).toBe(false);
  });
});
