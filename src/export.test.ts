import { describe, expect, it } from 'vitest';
import { toJson, toSrt, toTxt, toVtt } from './export.js';
import type { TranscriptSegment } from './protocol.js';

const simple: TranscriptSegment[] = [
  { start: 0, end: 1.5, text: 'Hello world.' },
  { start: 1.5, end: 4.25, text: 'Second segment.' },
];

describe('toSrt', () => {
  it('renders 1-based index blocks with comma-millisecond timestamps', () => {
    expect(toSrt(simple)).toBe(
      '1\n00:00:00,000 --> 00:00:01,500\nHello world.\n' +
        '\n2\n00:00:01,500 --> 00:00:04,250\nSecond segment.\n',
    );
  });

  it('returns an empty string for zero segments', () => {
    expect(toSrt([])).toBe('');
  });

  it('rounds sub-millisecond timestamps to the nearest millisecond', () => {
    const out = toSrt([{ start: 0.0004, end: 1.23456, text: 'x' }]);
    expect(out).toContain('00:00:00,000 --> 00:00:01,235');
    // .9995 rounds up into the next whole second.
    expect(toSrt([{ start: 0, end: 59.9995, text: 'x' }])).toContain('--> 00:01:00,000');
  });

  it('formats timestamps beyond one hour', () => {
    const out = toSrt([{ start: 3599.5, end: 3600 + 61.25, text: 'late' }]);
    expect(out).toContain('00:59:59,500 --> 01:01:01,250');
  });

  it('formats timestamps beyond 24 hours without wrapping', () => {
    const out = toSrt([{ start: 0, end: 100 * 3600, text: 'marathon' }]);
    expect(out).toContain('--> 100:00:00,000');
  });

  it('normalizes newlines and drops blank lines inside cue text', () => {
    const out = toSrt([{ start: 0, end: 1, text: 'line one\r\n\r\n  line two  \rline three' }]);
    expect(out).toBe('1\n00:00:00,000 --> 00:00:01,000\nline one\nline two\nline three\n');
  });

  it('skips segments whose text is empty and keeps indices contiguous', () => {
    const out = toSrt([
      { start: 0, end: 1, text: '   ' },
      { start: 1, end: 2, text: 'kept' },
    ]);
    expect(out).toBe('1\n00:00:01,000 --> 00:00:02,000\nkept\n');
  });

  it('clamps negative timestamps to zero', () => {
    expect(toSrt([{ start: -1, end: 1, text: 'x' }])).toContain('00:00:00,000 -->');
  });
});

describe('toVtt', () => {
  it('renders a WEBVTT header and dot-decimal timestamps', () => {
    expect(toVtt(simple)).toBe(
      'WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nHello world.\n' +
        '\n00:00:01.500 --> 00:00:04.250\nSecond segment.\n',
    );
  });

  it('returns only the header for zero segments', () => {
    expect(toVtt([])).toBe('WEBVTT\n');
  });

  it('formats timestamps beyond one hour', () => {
    expect(toVtt([{ start: 7261.75, end: 7322, text: 'x' }])).toContain(
      '02:01:01.750 --> 02:02:02.000',
    );
  });

  it('never emits a comma decimal separator', () => {
    expect(toVtt(simple)).not.toContain(',');
  });

  it('drops blank lines inside cue text (a blank line would end the cue)', () => {
    const out = toVtt([{ start: 0, end: 1, text: 'a\n\nb' }]);
    expect(out).toBe('WEBVTT\n\n00:00:00.000 --> 00:00:01.000\na\nb\n');
  });
});

describe('toTxt', () => {
  it('renders one segment per line with a trailing newline', () => {
    expect(toTxt(simple)).toBe('Hello world.\nSecond segment.\n');
  });

  it('returns an empty string for zero segments', () => {
    expect(toTxt([])).toBe('');
  });

  it('skips empty segments and normalizes internal newlines', () => {
    expect(
      toTxt([
        { start: 0, end: 1, text: ' ' },
        { start: 1, end: 2, text: 'a\r\nb' },
      ]),
    ).toBe('a\nb\n');
  });
});

describe('toJson', () => {
  it('serializes segments with stable fields and 2-space indentation', () => {
    expect(JSON.parse(toJson(simple))).toEqual(simple);
    expect(toJson(simple)).toContain('\n  {');
  });

  it('serializes zero segments as an empty array', () => {
    expect(toJson([])).toBe('[]');
  });

  it('preserves empty and newline-bearing text verbatim (lossless export)', () => {
    const segments = [{ start: 0, end: 1, text: 'a\n\nb ' }];
    expect(JSON.parse(toJson(segments))).toEqual(segments);
  });

  it('drops unknown extra properties from the output shape', () => {
    // Extra properties are legal on a structurally-typed segment; the export
    // must not leak them into the stable output shape.
    const dirty: TranscriptSegment[] = [
      Object.assign({ start: 0, end: 1, text: 'x' }, { confidence: 0.9 }),
    ];
    expect(JSON.parse(toJson(dirty))).toEqual([{ start: 0, end: 1, text: 'x' }]);
  });

  it('includes speaker only when present', () => {
    const withSpeaker: TranscriptSegment[] = [{ start: 0, end: 1, text: 'x', speaker: 'Alice' }];
    expect(JSON.parse(toJson(withSpeaker))).toEqual([
      { start: 0, end: 1, text: 'x', speaker: 'Alice' },
    ]);
    // Non-diarized JSON is unchanged (no speaker key).
    expect(toJson([{ start: 0, end: 1, text: 'x' }])).not.toContain('speaker');
  });
});

describe('speaker labels in text exports', () => {
  const diarized: TranscriptSegment[] = [
    { start: 0, end: 2, text: 'Hi there.', speaker: 'Alice' },
    { start: 2, end: 4, text: 'Hello.', speaker: 'Bob' },
  ];

  it('prefixes SRT cues with "Speaker: "', () => {
    const srt = toSrt(diarized);
    expect(srt).toContain('Alice: Hi there.');
    expect(srt).toContain('Bob: Hello.');
  });

  it('prefixes VTT cues', () => {
    expect(toVtt(diarized)).toContain('Alice: Hi there.');
  });

  it('prefixes TXT lines', () => {
    expect(toTxt(diarized)).toBe('Alice: Hi there.\nBob: Hello.\n');
  });

  it('leaves non-diarized exports byte-identical (no prefix)', () => {
    const plain: TranscriptSegment[] = [{ start: 0, end: 2, text: 'Hi there.' }];
    expect(toTxt(plain)).toBe('Hi there.\n');
    // The cue text line is the bare text — no "Name: " prefix.
    expect(toSrt(plain)).toBe('1\n00:00:00,000 --> 00:00:02,000\nHi there.\n');
  });
});
