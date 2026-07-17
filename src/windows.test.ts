import { describe, expect, it } from 'vitest';
import {
  dedupeWindowSegments,
  planWindows,
  STRIDE_SECONDS,
  WINDOW_SECONDS,
  WINDOW_STEP_SECONDS,
  type AudioWindow,
} from './windows.js';

const RATE = 16_000;
const seconds = (s: number): number => s * RATE;

describe('planWindows', () => {
  it('returns no windows for empty audio', () => {
    expect(planWindows(0, RATE)).toEqual([]);
    expect(planWindows(-5, RATE)).toEqual([]);
    expect(planWindows(seconds(10), 0)).toEqual([]);
  });

  it('returns a single window for audio that fits in one', () => {
    const windows = planWindows(seconds(10), RATE);
    expect(windows).toHaveLength(1);
    expect(windows[0]).toEqual({
      startSample: 0,
      endSample: seconds(10),
      startSeconds: 0,
      endSeconds: 10,
      isFirst: true,
      isLast: true,
    });
  });

  it('returns a single window for audio exactly one window long', () => {
    const windows = planWindows(seconds(WINDOW_SECONDS), RATE);
    expect(windows).toHaveLength(1);
    expect(windows[0]?.isLast).toBe(true);
  });

  it('steps by 25 s with a 5 s overlap between consecutive windows', () => {
    const windows = planWindows(seconds(80), RATE);
    expect(windows.map((w) => [w.startSeconds, w.endSeconds])).toEqual([
      [0, 30],
      [25, 55],
      [50, 80],
    ]);
    // Consecutive windows overlap by exactly the stride.
    for (let i = 1; i < windows.length; i++) {
      const prev = windows[i - 1];
      const curr = windows[i];
      expect((prev?.endSeconds ?? 0) - (curr?.startSeconds ?? 0)).toBe(STRIDE_SECONDS);
      expect((curr?.startSeconds ?? 0) - (prev?.startSeconds ?? 0)).toBe(WINDOW_STEP_SECONDS);
    }
  });

  it('marks only the first window first and only the last window last', () => {
    const windows = planWindows(seconds(80), RATE);
    expect(windows.map((w) => w.isFirst)).toEqual([true, false, false]);
    expect(windows.map((w) => w.isLast)).toEqual([false, false, true]);
  });

  it('covers every sample exactly up to the end (no gap, short final window ok)', () => {
    const total = seconds(33); // 30 s window + a 8 s tail window [25, 33]
    const windows = planWindows(total, RATE);
    expect(windows).toHaveLength(2);
    expect(windows[1]).toMatchObject({ startSeconds: 25, endSeconds: 33, isLast: true });
    expect(windows[windows.length - 1]?.endSample).toBe(total);
  });

  it('does not emit an empty trailing window when the audio ends on a step boundary', () => {
    // 55 s: windows [0,30] and [25,55] — a third window starting at 50 would
    // be needed only if audio extended past 55.
    const windows = planWindows(seconds(55), RATE);
    expect(windows).toHaveLength(2);
    expect(windows[1]?.endSeconds).toBe(55);
  });
});

describe('dedupeWindowSegments', () => {
  const middle: AudioWindow = {
    startSample: seconds(25),
    endSample: seconds(55),
    startSeconds: 25,
    endSeconds: 55,
    isFirst: false,
    isLast: false,
  };

  it('drops segments whose midpoint falls in the leading stride half', () => {
    const kept = dedupeWindowSegments(
      [
        { start: 25, end: 27, text: 'leading (mid 26)' }, // mid < 27.5 → dropped
        { start: 26, end: 29, text: 'kept (mid 27.5)' }, // mid = 27.5 → kept (>= cut)
        { start: 30, end: 32, text: 'kept (mid 31)' },
      ],
      middle,
    );
    expect(kept.map((s) => s.text)).toEqual(['kept (mid 27.5)', 'kept (mid 31)']);
  });

  it('drops segments whose midpoint falls in the trailing stride half', () => {
    const kept = dedupeWindowSegments(
      [
        { start: 50, end: 52, text: 'kept (mid 51)' },
        { start: 51, end: 53.5, text: 'kept (mid 52.25, just below cut)' },
        { start: 52.5, end: 54, text: 'trailing (mid 53.25)' }, // mid >= 52.5 → dropped
      ],
      middle,
    );
    expect(kept.map((s) => s.text)).toEqual(['kept (mid 51)', 'kept (mid 52.25, just below cut)']);
  });

  it('keeps leading segments on the first window and trailing on the last', () => {
    const first: AudioWindow = { ...middle, startSample: 0, startSeconds: 0, isFirst: true };
    expect(
      dedupeWindowSegments([{ start: 0, end: 1, text: 'very first words' }], first),
    ).toHaveLength(1);

    const last: AudioWindow = { ...middle, isLast: true };
    expect(
      dedupeWindowSegments([{ start: 54, end: 55, text: 'very last words' }], last),
    ).toHaveLength(1);
  });

  it('emits every segment exactly once across two overlapping windows', () => {
    // Two windows [0,30] and [25,55]; cut at 27.5. Segments spread across the
    // overlap must land in exactly one window's output.
    const a: AudioWindow = {
      startSample: 0,
      endSample: seconds(30),
      startSeconds: 0,
      endSeconds: 30,
      isFirst: true,
      isLast: false,
    };
    const b: AudioWindow = { ...middle, isLast: true };
    // The same speech as seen by both windows (identical absolute times).
    const overlapSegments = [
      { start: 25.5, end: 27, text: 'before cut (mid 26.25)' },
      { start: 27, end: 29, text: 'after cut (mid 28)' },
    ];
    const fromA = dedupeWindowSegments(overlapSegments, a).map((s) => s.text);
    const fromB = dedupeWindowSegments(overlapSegments, b).map((s) => s.text);
    expect(fromA).toEqual(['before cut (mid 26.25)']);
    expect(fromB).toEqual(['after cut (mid 28)']);
    expect([...fromA, ...fromB].sort()).toEqual(
      overlapSegments.map((s) => s.text).sort(), // exactly once each
    );
  });

  it('returns an empty list unchanged', () => {
    expect(dedupeWindowSegments([], middle)).toEqual([]);
  });
});
