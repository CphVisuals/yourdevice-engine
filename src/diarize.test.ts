import { describe, expect, it } from 'vitest';
import {
  assembleTurns,
  assignSpeakers,
  clusterEmbeddings,
  decodeSpeechFrames,
  diarize,
  framesToSpeechRegions,
  gatherSpeechSamples,
  mergeRegions,
  planEmbedWindows,
  shouldCollapseToSingle,
  smoothLabels,
  speakerCount,
  speechOverlapSeconds,
  type DiarizeConfig,
  type EmbeddingRunner,
  type SegmentationRunner,
} from './diarize.js';
import type { TranscriptSegment } from './protocol.js';

const SR = 16_000;
// embedStepSec is 1.5 here (the shipped default is 1.0) purely to keep the
// planEmbedWindows expectation below small and readable; the synthetic
// count tests pass with either stride.
const CONFIG: DiarizeConfig = {
  sampleRate: SR,
  vadWindowSec: 10,
  embedWindowSec: 3,
  embedStepSec: 1.5,
  minSpeechFraction: 0.5,
  minSpeechSec: 1.0,
  clusterThreshold: 0.6,
  mergeGapSec: 1.5,
  maxClusterSamples: 400,
  smoothingRadius: 1,
  maxWindows: 20_000,
};

// ---------------------------------------------------------------------------
// Synthetic audio + models. Each 0.1 s "frame" of audio is a constant value:
// 0 = silence, 1/2/3 = a speaker's identity. The fake segmentation reports
// speech wherever the value is non-zero (VAD only — matching how the real
// pipeline uses it). The fake embedding one-hot-encodes the *majority* value
// of the window's gathered samples, so distinct voices sit at cosine distance
// 1 and the clustering must separate them. This exercises the whole
// calibration path without ONNX.
// ---------------------------------------------------------------------------

const FRAME_SAMPLES = 0.1 * SR;

function buildAudio(script: [number, number][]): Float32Array {
  const total = script.reduce((n, [, dur]) => n + Math.round(dur * SR), 0);
  const audio = new Float32Array(total);
  let offset = 0;
  for (const [value, dur] of script) {
    const len = Math.round(dur * SR);
    audio.fill(value, offset, offset + len);
    offset += len;
  }
  return audio;
}

function oneHot(cls: number): number[] {
  const row = new Array<number>(7).fill(0);
  row[cls] = 10;
  return row;
}

// argmax class 0 = silence, else speech; the exact non-zero class is arbitrary
// (the pipeline only reads speech vs non-speech from segmentation).
const fakeSegmentation: SegmentationRunner = async (windowAudio) => {
  const numFrames = Math.max(1, Math.floor(windowAudio.length / FRAME_SAMPLES));
  const logits: number[][] = [];
  for (let f = 0; f < numFrames; f++) {
    const centre = Math.min(windowAudio.length - 1, Math.floor((f + 0.5) * FRAME_SAMPLES));
    const value = Math.round(windowAudio[centre] ?? 0);
    logits.push(oneHot(value === 0 ? 0 : 2)); // class 2 = a single speaker
  }
  return logits;
};

const fakeEmbedding: EmbeddingRunner = async (audio) => {
  // Majority identity value among the gathered speech samples.
  const counts = new Map<number, number>();
  for (const s of audio) {
    const v = Math.round(s);
    counts.set(v, (counts.get(v) ?? 0) + 1);
  }
  let best = 0;
  let bestN = -1;
  for (const [v, n] of counts) if (n > bestN) [best, bestN] = [v, n];
  const vec = new Float32Array(4);
  if (best >= 0 && best < vec.length) vec[best] = 1;
  return vec;
};

describe('decodeSpeechFrames', () => {
  it('marks NO_SPEAKER as non-speech and every other class as speech', () => {
    expect(decodeSpeechFrames([oneHot(0), oneHot(1), oneHot(2), oneHot(6)])).toEqual([
      false,
      true,
      true,
      true,
    ]);
  });
});

describe('framesToSpeechRegions', () => {
  it('turns contiguous speech runs into absolute-time regions', () => {
    // 10 frames over 1 s window starting at t=5 → 0.1 s per frame.
    const frames = [false, true, true, false, false, true, false, false, false, false];
    const regions = framesToSpeechRegions(frames, 5, SR, SR);
    expect(regions).toHaveLength(2);
    expect(regions[0]!.start).toBeCloseTo(5.1, 5);
    expect(regions[0]!.end).toBeCloseTo(5.3, 5);
    expect(regions[1]!.start).toBeCloseTo(5.5, 5);
  });
});

describe('mergeRegions', () => {
  it('merges regions within the gap and sorts', () => {
    const merged = mergeRegions(
      [
        { start: 2, end: 3 },
        { start: 0, end: 1 },
        { start: 1.05, end: 2 },
      ],
      0.1,
    );
    expect(merged).toEqual([{ start: 0, end: 3 }]);
  });

  it('keeps regions separated by more than the gap', () => {
    expect(
      mergeRegions(
        [
          { start: 0, end: 1 },
          { start: 2, end: 3 },
        ],
        0.1,
      ),
    ).toHaveLength(2);
  });
});

describe('speechOverlapSeconds', () => {
  it('sums overlap between a window and the regions', () => {
    const regions = [
      { start: 0, end: 2 },
      { start: 4, end: 6 },
    ];
    expect(speechOverlapSeconds(regions, 1, 5)).toBeCloseTo(2, 5); // [1,2] + [4,5]
  });
});

describe('gatherSpeechSamples', () => {
  it('concatenates only the speech samples inside the window', () => {
    const audio = buildAudio([
      [1, 1],
      [0, 1],
      [2, 1],
    ]); // spk, silence, spk
    const regions = [
      { start: 0, end: 1 },
      { start: 2, end: 3 },
    ];
    const gathered = gatherSpeechSamples(audio, regions, 0, 3 * SR, SR);
    expect(gathered.length).toBeCloseTo(2 * SR, -2); // ~2 s of speech, no silence
    expect([...gathered].every((v) => v !== 0)).toBe(true);
  });
});

describe('planEmbedWindows', () => {
  it('slides overlapping windows and clamps the tail', () => {
    const w = planEmbedWindows(5 * SR, CONFIG);
    // Stops once a window reaches the end of the audio.
    expect(w.map((x) => [x.startSec, x.endSec])).toEqual([
      [0, 3],
      [1.5, 4.5],
      [3, 5],
    ]);
  });
});

describe('shouldCollapseToSingle', () => {
  it('collapses when every pair is within threshold', () => {
    expect(
      shouldCollapseToSingle(
        [
          [1, 0],
          [0.99, 0.01],
        ],
        0.6,
      ),
    ).toBe(true);
  });
  it('does not collapse when a pair exceeds threshold', () => {
    expect(
      shouldCollapseToSingle(
        [
          [1, 0],
          [0, 1],
        ],
        0.6,
      ),
    ).toBe(false);
  });
  it('never collapses fewer than two embeddings', () => {
    expect(shouldCollapseToSingle([[1, 0]], 0.6)).toBe(false);
  });
});

describe('smoothLabels', () => {
  it('corrects an isolated misclustered window from its neighbours', () => {
    expect(smoothLabels([0, 1, 0], 1)).toEqual([0, 0, 0]);
    expect(smoothLabels([0, 0, 1, 0, 0], 1)).toEqual([0, 0, 0, 0, 0]);
  });

  it('preserves genuine speaker boundaries (runs longer than the radius)', () => {
    expect(smoothLabels([0, 0, 1, 1], 1)).toEqual([0, 0, 1, 1]);
    expect(smoothLabels([0, 0, 0, 1, 1, 1], 1)).toEqual([0, 0, 0, 1, 1, 1]);
  });

  it('is a no-op at radius 0', () => {
    expect(smoothLabels([0, 1, 0], 0)).toEqual([0, 1, 0]);
  });
});

describe('clusterEmbeddings', () => {
  const near = (base: readonly number[], jitter: number): Float32Array =>
    Float32Array.from(base.map((v) => v + (Math.random() - 0.5) * jitter));

  it('matches agglomerative for small N (two separated groups)', () => {
    const embs = [
      Float32Array.from([1, 0]),
      Float32Array.from([0.98, 0.02]),
      Float32Array.from([0, 1]),
      Float32Array.from([0.02, 0.98]),
    ];
    expect(new Set(clusterEmbeddings(embs, 0.6, 400)).size).toBe(2);
  });

  it('collapses a single cluster to one label', () => {
    const embs = [Float32Array.from([1, 0]), Float32Array.from([0.99, 0.01])];
    expect(new Set(clusterEmbeddings(embs, 0.6, 400)).size).toBe(1);
  });

  it('bounds large N via subsample + nearest-centroid while keeping the count', () => {
    // 600 embeddings in two well-separated groups; cap 400 forces the
    // subsample/centroid path. It must still resolve exactly two speakers,
    // and every embedding must be labelled.
    const embs: Float32Array[] = [];
    for (let i = 0; i < 300; i++) embs.push(near([1, 0, 0], 0.05));
    for (let i = 0; i < 300; i++) embs.push(near([0, 1, 0], 0.05));
    const labels = clusterEmbeddings(embs, 0.6, 400);
    expect(labels).toHaveLength(600);
    expect(new Set(labels).size).toBe(2);
    // The two groups get different labels.
    expect(labels[0]).not.toBe(labels[599]);
  });
});

describe('assembleTurns', () => {
  it('merges consecutive same-speaker windows and relabels by appearance', () => {
    const turns = assembleTurns(
      [
        { startSec: 0, endSec: 3, cluster: 5 },
        { startSec: 1.5, endSec: 4.5, cluster: 5 },
        { startSec: 6, endSec: 9, cluster: 2 },
      ],
      1.5,
    );
    expect(turns).toEqual([
      { start: 0, end: 4.5, speaker: 'Speaker 1' },
      { start: 6, end: 9, speaker: 'Speaker 2' },
    ]);
  });

  it('clips overlapping different-speaker turns so turns never overlap', () => {
    const turns = assembleTurns(
      [
        { startSec: 0, endSec: 3, cluster: 0 },
        { startSec: 2, endSec: 5, cluster: 1 },
      ],
      1.5,
    );
    expect(turns[0]!.end).toBeLessThanOrEqual(turns[1]!.start);
  });
});

describe('diarize (speaker-count accuracy on synthetic fixtures)', () => {
  it('returns 1 speaker for single-speaker audio spanning two VAD windows', async () => {
    const turns = await diarize(buildAudio([[1, 15]]), fakeSegmentation, fakeEmbedding, CONFIG);
    expect(speakerCount(turns)).toBe(1);
    expect(turns[0]!.speaker).toBe('Speaker 1');
  });

  it('returns 2 speakers for an alternating two-speaker conversation', async () => {
    const audio = buildAudio([
      [1, 5],
      [2, 5],
      [1, 5],
      [2, 5],
    ]);
    const turns = await diarize(audio, fakeSegmentation, fakeEmbedding, CONFIG);
    expect(speakerCount(turns)).toBe(2);
  });

  it('returns 3 speakers for a three-speaker conversation', async () => {
    const audio = buildAudio([
      [1, 4],
      [2, 4],
      [3, 4],
      [1, 4],
      [2, 4],
      [3, 4],
    ]);
    const turns = await diarize(audio, fakeSegmentation, fakeEmbedding, CONFIG);
    expect(speakerCount(turns)).toBe(3);
  });

  it('returns no turns for pure silence', async () => {
    expect(await diarize(buildAudio([[0, 5]]), fakeSegmentation, fakeEmbedding, CONFIG)).toEqual(
      [],
    );
  });

  it('labels trailing speech that lands only in a short clamped final window', async () => {
    // 3.8 s silence then 1.2 s of one speaker (total 5.0 s). The only window
    // covering that speech is the clamped 2.0 s final window [3–5]; its 1.2 s
    // of speech clears a fraction gate against its ACTUAL length but not
    // against the nominal 3 s (which would demand 1.5 s), so the fixed gate
    // dropped it and left the tail unlabeled.
    const turns = await diarize(
      buildAudio([
        [0, 3.8],
        [1, 1.2],
      ]),
      fakeSegmentation,
      fakeEmbedding,
      CONFIG,
    );
    expect(speakerCount(turns)).toBe(1);
    expect(turns[turns.length - 1]!.end).toBeGreaterThan(4.5); // reaches the tail
  });

  it('reports embedding progress and skips audio past the window ceiling', async () => {
    const audio = buildAudio([
      [1, 5],
      [2, 5],
    ]);
    const progress: [number, number][] = [];
    await diarize(audio, fakeSegmentation, fakeEmbedding, CONFIG, (done, total) =>
      progress.push([done, total]),
    );
    expect(progress.length).toBeGreaterThan(0);
    expect(progress[progress.length - 1]![1]).toBeGreaterThan(0); // total set

    // A tiny maxWindows ceiling skips diarization (transcript still succeeds).
    const capped = await diarize(audio, fakeSegmentation, fakeEmbedding, {
      ...CONFIG,
      maxWindows: 1,
    });
    expect(capped).toEqual([]);
  });

  it('produces non-overlapping, ordered turns', async () => {
    const audio = buildAudio([
      [1, 5],
      [2, 5],
      [1, 5],
    ]);
    const turns = await diarize(audio, fakeSegmentation, fakeEmbedding, CONFIG);
    for (let i = 1; i < turns.length; i++) {
      expect(turns[i]!.start).toBeGreaterThanOrEqual(turns[i - 1]!.end - 1e-9);
    }
    expect(turns.every((t) => t.end > t.start)).toBe(true);
    expect(turns[0]!.speaker).toBe('Speaker 1');
  });

  it('stops the embedding loop promptly on abort (does not drain every window)', async () => {
    const audio = buildAudio([
      [1, 5],
      [2, 5],
      [1, 5],
      [2, 5],
    ]);
    // Baseline: how many embeddings a full, un-aborted run performs.
    let full = 0;
    await diarize(
      audio,
      fakeSegmentation,
      async (a) => {
        full++;
        return fakeEmbedding(a);
      },
      CONFIG,
    );
    expect(full).toBeGreaterThan(2);

    // Aborted run: cancel as soon as the first embedding has been computed.
    let calls = 0;
    const turns = await diarize(
      audio,
      fakeSegmentation,
      async (a) => {
        calls++;
        return fakeEmbedding(a);
      },
      CONFIG,
      undefined,
      () => calls >= 1,
    );
    expect(turns).toEqual([]); // aborted → no turns (caller discards)
    expect(calls).toBeLessThan(full); // stopped early, didn't embed every window
  });
});

describe('assignSpeakers', () => {
  const turns = [
    { start: 0, end: 5, speaker: 'Speaker 1' },
    { start: 5, end: 10, speaker: 'Speaker 2' },
  ];

  it('labels each segment by the turn it overlaps most', () => {
    const segments: TranscriptSegment[] = [
      { start: 0.2, end: 4.5, text: 'hello' },
      { start: 5.1, end: 9.0, text: 'world' },
    ];
    const out = assignSpeakers(segments, turns);
    expect(out[0]!.speaker).toBe('Speaker 1');
    expect(out[1]!.speaker).toBe('Speaker 2');
  });

  it('picks the dominant speaker for a segment straddling a turn boundary', () => {
    const out = assignSpeakers([{ start: 4, end: 9, text: 'x' }], turns);
    expect(out[0]!.speaker).toBe('Speaker 2'); // 1 s in S1, 4 s in S2
  });

  it('leaves speaker undefined when there is no overlap', () => {
    const out = assignSpeakers([{ start: 20, end: 25, text: 'x' }], turns);
    expect(out[0]!.speaker).toBeUndefined();
  });

  it('does not mutate the input segments', () => {
    const segments: TranscriptSegment[] = [{ start: 0, end: 5, text: 'x' }];
    assignSpeakers(segments, turns);
    expect(segments[0]!.speaker).toBeUndefined();
  });
});
