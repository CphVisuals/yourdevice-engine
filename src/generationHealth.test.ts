import { describe, expect, it } from 'vitest';
import { distinctNgramRatio, isDegenerateGeneration } from './generationHealth.js';

// The exact garbage a broken q8 export produced on the coherence probe,
// captured off-device — the failure mode this guard exists to catch.
const LOOP_GARBAGE = "ry you can't get.\n" + "I'm sorry I'm not able to get.\n".repeat(16);
const HEALTHY_SUMMARY =
  'Ana and Ben propose launching the new pricing page on Aug 12. They agree to keep the free tier. Marketing will draft the announcement email, due Monday.';

describe('distinctNgramRatio', () => {
  it('is near 1 for non-repeating prose', () => {
    expect(distinctNgramRatio(HEALTHY_SUMMARY)).toBeGreaterThan(0.9);
  });
  it('collapses toward 0 for a repetition loop', () => {
    expect(distinctNgramRatio(LOOP_GARBAGE)).toBeLessThan(0.2);
  });
  it('returns 1 for text too short to judge', () => {
    expect(distinctNgramRatio('capital of France')).toBe(1);
  });
});

describe('isDegenerateGeneration', () => {
  it('flags empty and whitespace-only text', () => {
    expect(isDegenerateGeneration('')).toBe(true);
    expect(isDegenerateGeneration('   \n  ')).toBe(true);
  });
  it('flags text with essentially no letters', () => {
    expect(isDegenerateGeneration('12 34 .. !! —— 99 00 11 22 33 44 55 66')).toBe(true);
  });
  it('flags a runaway repetition loop', () => {
    expect(isDegenerateGeneration(LOOP_GARBAGE)).toBe(true);
  });
  it('passes healthy prose', () => {
    expect(isDegenerateGeneration(HEALTHY_SUMMARY)).toBe(false);
  });
  it('does not flag a short but valid answer', () => {
    expect(isDegenerateGeneration('The capital of France is Paris.')).toBe(false);
  });
});
