/** Read `--seed <number>` from argv. Missing flag defaults to 42. */
export function parseSeed(argv: string[]): number {
  const index = argv.indexOf("--seed");
  if (index === -1) return 42;
  const raw = argv[index + 1];
  const value = raw === undefined ? Number.NaN : Number(raw);
  if (!Number.isFinite(value)) {
    throw new Error(`Expected a number after --seed, got ${raw ?? "nothing"}`);
  }
  return value;
}
