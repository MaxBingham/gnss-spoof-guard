export function clampCursor(length: number, cursor: number): number {
  if (length <= 0) return 0;
  const index = Number.isFinite(cursor) ? Math.round(cursor) : 0;
  return Math.min(length - 1, Math.max(0, index));
}

export function sliceToCursor<T>(items: readonly T[], cursor: number): T[] {
  if (items.length === 0) return [];
  return items.slice(0, clampCursor(items.length, cursor) + 1);
}

/** Advance one sample. Reaching the end stops playback on the last sample. */
export function stepCursor(length: number, cursor: number): { cursor: number; playing: boolean } {
  const current = clampCursor(length, cursor);
  if (length <= 1 || current >= length - 1) return { cursor: current, playing: false };
  return { cursor: current + 1, playing: true };
}
