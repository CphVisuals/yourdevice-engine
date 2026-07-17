import { describe, expect, it } from 'vitest';
import { resampleTo16kMono } from './audio.js';

describe('resampleTo16kMono', () => {
  it('is a no-op for already-mono 16 kHz audio', () => {
    const channel = new Float32Array([0, 0.25, 0.5, 0.75, 1]);
    const result = resampleTo16kMono({ channelData: [channel], sampleRate: 16_000 });
    expect(result).toEqual(channel);
  });

  it('averages channels down to mono before resampling', () => {
    const left = new Float32Array([1, 1, 1, 1]);
    const right = new Float32Array([-1, -1, -1, -1]);
    const result = resampleTo16kMono({ channelData: [left, right], sampleRate: 16_000 });
    expect(result).toEqual(new Float32Array([0, 0, 0, 0]));
  });

  it('returns silence for zero channels', () => {
    const result = resampleTo16kMono({ channelData: [], sampleRate: 16_000 });
    expect(result).toEqual(new Float32Array(0));
  });

  it('downsamples 48 kHz to 16 kHz (1/3 the length)', () => {
    const input = new Float32Array(48_000);
    for (let i = 0; i < input.length; i++) input[i] = i / input.length;
    const result = resampleTo16kMono({ channelData: [input], sampleRate: 48_000 });
    expect(result.length).toBe(16_000);
    // Linear-interpolated ramp should stay monotonically non-decreasing.
    for (let i = 1; i < result.length; i++) {
      expect(result[i]).toBeGreaterThanOrEqual(result[i - 1] ?? 0);
    }
    expect(result[0]).toBeCloseTo(input[0] ?? 0, 5);
  });

  it('upsamples 8 kHz to 16 kHz (double the length)', () => {
    const input = new Float32Array([0, 1, 0, -1]);
    const result = resampleTo16kMono({ channelData: [input], sampleRate: 8_000 });
    expect(result.length).toBe(8);
    // Original samples should reappear at even indices.
    expect(result[0]).toBeCloseTo(0, 5);
    expect(result[2]).toBeCloseTo(1, 5);
    expect(result[4]).toBeCloseTo(0, 5);
    expect(result[6]).toBeCloseTo(-1, 5);
  });

  it('interpolates linearly between samples on a 2x upsample', () => {
    // fromRate/toRate ratio 0.5 (mirrors 8 kHz -> 16 kHz): each output sample
    // sits either exactly on a source sample or exactly halfway between two.
    const input = new Float32Array([0, 10]);
    const result = resampleTo16kMono({ channelData: [input], sampleRate: 8_000 });
    expect(Array.from(result)).toEqual([0, 5, 10, 10]);
  });

  it('handles empty input', () => {
    const result = resampleTo16kMono({ channelData: [new Float32Array(0)], sampleRate: 44_100 });
    expect(result).toEqual(new Float32Array(0));
  });
});
