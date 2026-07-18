/**
 * Detects degenerate ASR transcripts — the "!!!!"-repeated (or empty, or
 * single-character-dominated) garbage a broken accelerated backend can emit
 * even after its `pipeline()` call succeeds (the WebGPU garbage-output
 * incident this module exists to catch; see worker.ts's post-build output
 * probe and CLAUDE.md rule 9). Pure and framework-free (CLAUDE.md rule 2/3):
 * no browser globals, exhaustively unit-tested in the sibling `*.test.ts`.
 *
 * A transcript is degenerate when, after trimming and ignoring internal
 * whitespace, it is:
 *   - empty,
 *   - made of a single repeated character ("!!!!", ".....", "! ! ! !"),
 *   - fewer than 2 alphabetic characters (numbers/punctuation alone are not
 *     a plausible transcript of real speech), or
 *   - dominated by one character that makes up more than 90% of the
 *     non-space output (mostly-noise text with a stray real character).
 * Healthy speech ("and so my fellow Americans", "OK.", "1 2 3 go") satisfies
 * none of these.
 */
export function isDegenerateOutput(text: string): boolean {
  const trimmed = text.trim();
  if (trimmed.length === 0) return true;

  const nonSpace = trimmed.replace(/\s+/g, '');
  if (nonSpace.length === 0) return true;

  const firstChar = nonSpace[0];
  if ([...nonSpace].every((ch) => ch === firstChar)) return true;

  const alphabeticCount = (nonSpace.match(/[a-zA-Z]/g) ?? []).length;
  if (alphabeticCount < 2) return true;

  const counts = new Map<string, number>();
  for (const ch of nonSpace) counts.set(ch, (counts.get(ch) ?? 0) + 1);
  const maxCount = Math.max(...counts.values());
  if (maxCount / nonSpace.length > 0.9) return true;

  return false;
}
