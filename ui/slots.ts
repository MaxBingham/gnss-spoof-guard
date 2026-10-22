/** Empty panel slots. Saturday fills these; prep leaves them labeled and blank. */
export const PIPELINE_SLOTS = [
  { id: "detection", label: "Detection" },
  { id: "ranking", label: "Ranking" },
  { id: "decision", label: "Decision" },
] as const;

export type PipelineSlotId = (typeof PIPELINE_SLOTS)[number]["id"];
