/**
 * Pure transcript exporters. No DOM, no I/O — segments in, string out; the
 * site decides filenames/download mechanics.
 *
 * SRT and VTT are block formats where a blank line terminates a cue, so cue
 * text is normalized: CRLF/CR become LF and blank lines inside a segment are
 * dropped. Segments whose text is empty after normalization are skipped in
 * SRT/VTT/TXT (an empty cue is malformed SRT); JSON preserves every segment
 * verbatim so no information is ever lost in the machine-readable export.
 */

import type { TranscriptSegment } from './protocol.js';

function pad(value: number, width: number): string {
  return String(value).padStart(width, '0');
}

/** `HH:MM:SS<sep>mmm` — comma for SRT, dot for VTT. Hours grow past 99 unpadded. */
function formatTimestamp(seconds: number, decimalSeparator: ',' | '.'): string {
  const totalMs = Math.round(Math.max(0, seconds) * 1000);
  const ms = totalMs % 1000;
  const totalS = (totalMs - ms) / 1000;
  const s = totalS % 60;
  const totalM = (totalS - s) / 60;
  const m = totalM % 60;
  const h = (totalM - m) / 60;
  return `${pad(h, 2)}:${pad(m, 2)}:${pad(s, 2)}${decimalSeparator}${pad(ms, 3)}`;
}

/** Normalizes line endings and strips blank lines (they would terminate the cue). */
function cueText(text: string): string {
  return text
    .replace(/\r\n?/g, '\n')
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .join('\n');
}

export function toSrt(segments: readonly TranscriptSegment[]): string {
  const blocks: string[] = [];
  for (const segment of segments) {
    const text = cueText(segment.text);
    if (!text) continue;
    const index = blocks.length + 1; // 1-based, counts emitted cues only
    const start = formatTimestamp(segment.start, ',');
    const end = formatTimestamp(segment.end, ',');
    blocks.push(`${index}\n${start} --> ${end}\n${text}`);
  }
  return blocks.length === 0 ? '' : `${blocks.join('\n\n')}\n`;
}

export function toVtt(segments: readonly TranscriptSegment[]): string {
  const cues: string[] = [];
  for (const segment of segments) {
    const text = cueText(segment.text);
    if (!text) continue;
    const start = formatTimestamp(segment.start, '.');
    const end = formatTimestamp(segment.end, '.');
    cues.push(`${start} --> ${end}\n${text}`);
  }
  return cues.length === 0 ? 'WEBVTT\n' : `WEBVTT\n\n${cues.join('\n\n')}\n`;
}

export function toTxt(segments: readonly TranscriptSegment[]): string {
  const lines = segments.map((segment) => cueText(segment.text)).filter((line) => line.length > 0);
  return lines.length === 0 ? '' : `${lines.join('\n')}\n`;
}

export function toJson(segments: readonly TranscriptSegment[]): string {
  // Copy the known fields explicitly so the output shape stays stable even if
  // callers pass segments carrying extra properties.
  const clean = segments.map(({ start, end, text }) => ({ start, end, text }));
  return JSON.stringify(clean, null, 2);
}
