import { describe, expect, it } from 'vitest';
import { getValidationClip, VALIDATION_CLIP_SAMPLE_RATE } from './validationClip.js';

describe('getValidationClip', () => {
  it('decodes to 16 kHz mono PCM in [-1, 1]', () => {
    const clip = getValidationClip();
    expect(clip).toBeInstanceOf(Float32Array);
    expect(VALIDATION_CLIP_SAMPLE_RATE).toBe(16_000);
    // ~2 s at 16 kHz.
    expect(clip.length).toBeGreaterThan(16_000 * 1.5);
    expect(clip.length).toBeLessThanOrEqual(16_000 * 2.5);
    for (const sample of clip) {
      expect(sample).toBeGreaterThanOrEqual(-1);
      expect(sample).toBeLessThanOrEqual(1);
    }
  });

  it('is real audio, not silence (has meaningful amplitude variance)', () => {
    const clip = getValidationClip();
    let sumSquares = 0;
    for (const sample of clip) sumSquares += sample * sample;
    const rms = Math.sqrt(sumSquares / clip.length);
    // Silence/DC would be ~0; real speech has a healthy RMS.
    expect(rms).toBeGreaterThan(0.01);
  });

  it('memoizes: repeated calls return the same array without re-decoding', () => {
    expect(getValidationClip()).toBe(getValidationClip());
  });
});
