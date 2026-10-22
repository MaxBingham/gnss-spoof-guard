import { describe, expect, it } from "vitest";
import { PIPELINE_SLOTS } from "./slots.ts";

describe("pipeline slots", () => {
  it("leaves Detection, Ranking, and Decision labeled and with no body", () => {
    expect(PIPELINE_SLOTS.map((slot) => slot.label)).toEqual(["Detection", "Ranking", "Decision"]);
    expect(PIPELINE_SLOTS.every((slot) => !("body" in slot))).toBe(true);
  });
});
