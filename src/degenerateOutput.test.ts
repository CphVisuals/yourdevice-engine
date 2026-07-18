import { describe, expect, it } from 'vitest';
import { isDegenerateOutput } from './degenerateOutput.js';

describe('isDegenerateOutput', () => {
  it.each([
    ['empty string', ''],
    ['whitespace only', '   '],
    ['whitespace only (tabs/newlines)', ' \t\n '],
    ['repeated exclamation marks (the production symptom)', '!!!!'],
    ['repeated single letter', 'aaaa'],
    ['repeated dots', '.....'],
    ['single-token repeat, space-separated', '! ! ! ! ! !'],
    ['a single character', '!'],
    ['a single letter', 'a'],
    ['digits only, no letters', '1234 5678'],
    ['punctuation only', '... --- ...'],
    ['one real letter buried in noise', 'a!!!!!!!!!!!'],
  ])('flags %s as degenerate', (_label, text) => {
    expect(isDegenerateOutput(text)).toBe(true);
  });

  it.each([
    ['a real sentence', 'and so my fellow Americans'],
    ['a short acknowledgement', 'OK.'],
    ['digits mixed with real words', '1 2 3 go'],
    ['a single short real word', 'Hello'],
    ['a sentence with punctuation', 'ask not what your country can do for you.'],
  ])('accepts %s as healthy', (_label, text) => {
    expect(isDegenerateOutput(text)).toBe(false);
  });

  it('treats exactly 2 alphabetic characters as the healthy boundary', () => {
    expect(isDegenerateOutput('OK')).toBe(false);
    expect(isDegenerateOutput('O.')).toBe(true); // only 1 alphabetic char
  });

  it('treats exactly 90% single-character dominance as still healthy', () => {
    // 9 of 10 non-space chars identical, 1 different: 90% exactly, not > 90%.
    expect(isDegenerateOutput('aaaaaaaaab')).toBe(false);
  });

  it('flags just over 90% single-character dominance', () => {
    // 10 of 11 non-space chars identical: > 90%.
    expect(isDegenerateOutput('aaaaaaaaaab')).toBe(true);
  });
});
