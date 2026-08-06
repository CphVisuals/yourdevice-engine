/**
 * Degeneracy detection for autoregressive text *generation* — the LLM sibling
 * of {@link ./degenerateOutput}'s ASR-transcript check. A broken accelerated
 * backend (e.g. the onnxruntime-web WebGPU fp16 path overflowing to garbage,
 * onnxruntime#26732) can build its pipeline and still emit a runaway
 * repetition loop ("I'm sorry I'm not able to get." ×N) that a `pipeline()`
 * call cannot see. Where ASR garbage tends to be single-character noise
 * ("!!!!"), generation garbage is typically a repeated *phrase*, so the signal
 * here is a low distinct-n-gram ratio rather than single-char domination.
 *
 * Pure and framework-free (CLAUDE.md rule 2/3): no model, no browser globals —
 * callers feed it only text. Consumed by the summarizer's coherence probe.
 */

/** Below this many words a text is too short to judge for a repetition loop. */
const MIN_WORDS_FOR_REPETITION = 12;

/**
 * Fraction of word n-grams that are *distinct*. Non-repeating prose sits near
 * 1; a repetition loop (the same phrase over and over) drives it toward 0.
 * Returns 1 when the text is too short to contain a meaningful n-gram.
 */
export function distinctNgramRatio(text: string, n = 3): number {
  const words = text.toLowerCase().split(/\s+/).filter(Boolean);
  if (words.length < n + 1) return 1;
  const grams: string[] = [];
  for (let i = 0; i + n <= words.length; i++) grams.push(words.slice(i, i + n).join(' '));
  return new Set(grams).size / grams.length;
}

/**
 * Whether generated text is degenerate: empty, essentially letter-free, or a
 * runaway repetition loop. Healthy prose (distinct-n-gram ratio near 1) passes;
 * the observed garbage loops (distinct ratio ≈ 0.05 on the broken q8 export)
 * fail. The 0.5 threshold sits well clear of both; the word-count floor keeps
 * legitimately short answers ("The capital of France is Paris.") from tripping
 * the repetition test.
 */
export function isDegenerateGeneration(text: string, minDistinctRatio = 0.5): boolean {
  const trimmed = text.trim();
  if (trimmed.length === 0) return true;
  if ((trimmed.match(/[a-zA-Z]/g) ?? []).length < 2) return true;

  const words = trimmed.split(/\s+/).filter(Boolean);
  if (
    words.length >= MIN_WORDS_FOR_REPETITION &&
    distinctNgramRatio(trimmed, 3) < minDistinctRatio
  ) {
    return true;
  }
  return false;
}
