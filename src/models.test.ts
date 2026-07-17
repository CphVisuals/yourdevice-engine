import { describe, expect, it } from 'vitest';
import { isModelId, MODELS } from './models.js';

describe('MODELS', () => {
  it('keys every spec by its own id', () => {
    for (const [key, spec] of Object.entries(MODELS)) {
      expect(spec.id).toBe(key);
    }
  });

  it('declares a plausible download size for every model', () => {
    for (const spec of Object.values(MODELS)) {
      expect(spec.approxSizeMb).toBeGreaterThan(0);
    }
  });
});

describe('isModelId', () => {
  it('accepts every registered model id', () => {
    for (const key of Object.keys(MODELS)) {
      expect(isModelId(key)).toBe(true);
    }
  });

  it('rejects unknown ids and non-strings', () => {
    expect(isModelId('gpt-5')).toBe(false);
    expect(isModelId('')).toBe(false);
    expect(isModelId(42)).toBe(false);
    expect(isModelId(null)).toBe(false);
    expect(isModelId(undefined)).toBe(false);
    expect(isModelId({})).toBe(false);
  });

  it('rejects keys inherited from Object.prototype', () => {
    expect(isModelId('toString')).toBe(false);
    expect(isModelId('constructor')).toBe(false);
    expect(isModelId('hasOwnProperty')).toBe(false);
    expect(isModelId('__proto__')).toBe(false);
  });
});
