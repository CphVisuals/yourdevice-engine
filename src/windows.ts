/**
 * Pure windowing logic for long-audio transcription, kept out of worker.ts so
 * it is unit-testable (CLAUDE.md rule 3: pure logic separated from browser
 * globals).
 *
 * Long audio is processed in 30 s windows advancing 25 s at a time, so each
 * pair of consecutive windows shares a 5 s overlap (the "stride"). Whisper
 * sees every stretch of speech in at least one window with real context on
 * both sides; the overlap is then deduplicated by cutting at its midline —
 * each window keeps only the segments whose midpoint falls on its side of
 * the cut (leading-stride-half drop on the later window, mirrored
 * trailing-stride-half drop on the earlier one, so nothing is emitted twice
 * and nothing is lost).
 */

import type { TranscriptSegment } from './protocol.js';

export const WINDOW_SECONDS = 30;
export const STRIDE_SECONDS = 5;
/** How far consecutive windows advance: 30 s window − 5 s overlap. */
export const WINDOW_STEP_SECONDS = WINDOW_SECONDS - STRIDE_SECONDS;

export interface AudioWindow {
  /** Sample offsets into the full 16 kHz mono buffer; `endSample` exclusive. */
  startSample: number;
  endSample: number;
  /** The same bounds in seconds, for timestamp offsetting and progress. */
  startSeconds: number;
  endSeconds: number;
  isFirst: boolean;
  isLast: boolean;
}

/**
 * Plans the windows covering `totalSamples` of audio. Audio that fits in a
 * single window yields exactly one window (callers can skip the loop
 * machinery); empty audio yields none.
 */
export function planWindows(totalSamples: number, sampleRate: number): AudioWindow[] {
  if (totalSamples <= 0 || sampleRate <= 0) return [];

  const windowSamples = WINDOW_SECONDS * sampleRate;
  const stepSamples = WINDOW_STEP_SECONDS * sampleRate;
  const windows: AudioWindow[] = [];

  for (let start = 0; ; start += stepSamples) {
    const end = Math.min(start + windowSamples, totalSamples);
    windows.push({
      startSample: start,
      endSample: end,
      startSeconds: start / sampleRate,
      endSeconds: end / sampleRate,
      isFirst: start === 0,
      isLast: false,
    });
    if (end >= totalSamples) break;
  }

  const last = windows[windows.length - 1];
  if (last) last.isLast = true;
  return windows;
}

/**
 * Overlap dedupe for one window's segments (already offset to absolute time).
 * The 5 s overlap between consecutive windows is cut at its midline
 * (2.5 s in): a non-first window drops segments whose midpoint falls in the
 * leading stride half, and a non-last window drops segments whose midpoint
 * falls in the trailing stride half — the neighbouring window keeps exactly
 * those, so every segment is emitted once.
 */
export function dedupeWindowSegments(
  segments: readonly TranscriptSegment[],
  window: AudioWindow,
): TranscriptSegment[] {
  const leadingCut = window.startSeconds + STRIDE_SECONDS / 2;
  const trailingCut = window.endSeconds - STRIDE_SECONDS / 2;
  return segments.filter((segment) => {
    const midpoint = (segment.start + segment.end) / 2;
    if (!window.isFirst && midpoint < leadingCut) return false;
    if (!window.isLast && midpoint >= trailingCut) return false;
    return true;
  });
}
