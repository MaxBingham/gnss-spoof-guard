import { describe, expect, it } from "vitest";
import { clampCursor, sliceToCursor, stepCursor } from "./playback.ts";

describe("playback", () => {
  it("clamps the playhead into the flight", () => {
    expect(clampCursor(5, -3)).toBe(0);
    expect(clampCursor(5, 9)).toBe(4);
    expect(clampCursor(0, 2)).toBe(0);
  });

  it("keeps samples up through the playhead", () => {
    expect(sliceToCursor(["a", "b", "c"], 1)).toEqual(["a", "b"]);
    expect(sliceToCursor(["a", "b", "c"], 20)).toEqual(["a", "b", "c"]);
  });

  it("stops when the playhead reaches the last sample", () => {
    expect(stepCursor(4, 2)).toEqual({ cursor: 3, playing: true });
    expect(stepCursor(4, 3)).toEqual({ cursor: 3, playing: false });
  });
});
