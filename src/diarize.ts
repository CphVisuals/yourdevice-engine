/**
 * Speaker diarization ("who spoke when") — framework-free pipeline that turns
 * 16 kHz mono audio into speaker turns, ready to merge onto a transcript.
 *
 * Two ONNX models do the heavy lifting, but this module never imports
 * Transformers.js: the model calls are injected as `runSegmentation` /
 * `runEmbedding` (the same globals-injection pattern as `DetectionScope`), so
 * every calibration step below is unit-testable on synthetic model outputs
 * with no browser or network. The worker supplies the real runners.
 *
 * Architecture (validated against real fixtures in the diarization spike):
 *   - The pyannote **segmentation** model is used ONLY as voice activity
 *     detection. Its per-window local speaker *slots* proved unreliable for
 *     separation — it lumps back-to-back distinct speakers into one slot — so
 *     we take only its speech/non-speech decision.
 *   - Speaker separation comes from the **wespeaker embedding** model, which
 *     discriminates voices cleanly (spike: same-speaker cosine ~0.40,
 *     different ~0.80). We slide short fixed windows over the speech, embed
 *     each, and cluster the embeddings globally.
 *
 * Calibration knobs (the accuracy-critical part), all unit-tested:
 *   - NO_SPEAKER frames dropped so silence never becomes a "speaker"
 *   - 3 s embedding windows (short windows are too noisy; the spike found 3 s
 *     the stable point) over speech only, silence samples excluded
 *   - a minimum speech fraction / duration before a window is embedded
 *   - a collapse-to-one guard for single-speaker audio
 *   - a tuned cosine distance threshold (0.6, centre of the 0.55–0.65 band
 *     where 1/2/3-speaker fixtures all resolve correctly)
 */

import { agglomerative, cosineDistance, maxPairwiseDistance, type Vector } from './clustering.js';
import type { TranscriptSegment } from './protocol.js';

/** A contiguous span attributed to one speaker. `speaker` is "Speaker 1".."Speaker N". */
export interface SpeakerTurn {
  start: number;
  end: number;
  speaker: string;
}

export interface TimeRegion {
  start: number;
  end: number;
}

/**
 * Per-frame powerset logits for one window: `logits[frame]` has 7 scores for
 * the pyannote-segmentation-3.0 powerset classes (index 0 = NO_SPEAKER, 1–3 a
 * single speaker, 4–6 an overlapping pair). Injected so tests don't need the
 * model.
 */
export type SegmentationRunner = (windowAudio: Float32Array) => Promise<number[][]>;
/** Returns one speaker-embedding vector for the given audio. */
export type EmbeddingRunner = (audio: Float32Array) => Promise<Float32Array>;

export interface DiarizeConfig {
  sampleRate: number;
  /** Segmentation (VAD) is run on windows of this length (pyannote trains on 10 s). */
  vadWindowSec: number;
  /** Sliding embedding window length; 3 s is the stable point from the spike. */
  embedWindowSec: number;
  /** Embedding window advance (< window ⇒ overlap, so boundaries are covered). */
  embedStepSec: number;
  /** A window must be at least this fraction speech to be embedded. */
  minSpeechFraction: number;
  /** …and contain at least this many seconds of actual speech. */
  minSpeechSec: number;
  /** Cosine-distance threshold for clustering embeddings into speakers. */
  clusterThreshold: number;
  /** Merge same-speaker turns separated by a gap no larger than this. */
  mergeGapSec: number;
  /**
   * Above this many embeddings, cluster a uniform subsample of this size and
   * assign the rest to the nearest cluster centroid — bounds the O(n³)
   * clustering to O(cap³) + O(n·cap) so long meetings don't hang the worker.
   */
  maxClusterSamples: number;
  /**
   * Neighbourhood radius for majority-vote smoothing of the per-window label
   * sequence (0 disables). Corrects an isolated misclustered window before
   * turns are assembled, so one bad window can't permanently mislabel a span.
   */
  smoothingRadius: number;
  /**
   * Practical ceiling on embedding windows. Beyond this the audio is too long
   * to diarize in a reasonable time; the pipeline returns no turns (the
   * transcript still succeeds, just without speaker labels).
   */
  maxWindows: number;
}

/**
 * Defaults tuned against the diarization fixtures (jfk 1-speaker, and
 * JFK+MLK / JFK+MLK+TED 2- and 3-speaker): with 3 s windows, threshold 0.6
 * sits in the middle of the 0.55–0.65 band where all three resolve correctly.
 */
export const DEFAULT_DIARIZE_CONFIG: DiarizeConfig = {
  sampleRate: 16_000,
  vadWindowSec: 10,
  embedWindowSec: 3,
  // 1 s step (heavy overlap): the finer stride yields enough windows that
  // isolate each voice for clustering to recover all speakers. With this
  // stride 1/2/3-speaker fixtures all resolve correctly across the whole
  // 0.5–0.7 threshold band (spike sweep); 1.5 s lost the third speaker.
  embedStepSec: 1.0,
  minSpeechFraction: 0.5,
  minSpeechSec: 1.0,
  clusterThreshold: 0.6,
  mergeGapSec: 1.5,
  maxClusterSamples: 400,
  smoothingRadius: 1,
  // ~20k windows ≈ 5.5 h of audio; beyond this the linear embedding pass is
  // impractical and diarization is skipped (transcript still returned).
  maxWindows: 20_000,
};

/**
 * pyannote-segmentation-3.0 powerset → whether that class is speech. Class 0
 * (NO_SPEAKER) is silence; every other class has at least one active speaker.
 */
const NUM_POWERSET_CLASSES = 7;

function argmax(row: readonly number[]): number {
  let best = 0;
  for (let i = 1; i < row.length; i++) if ((row[i] ?? 0) > (row[best] ?? 0)) best = i;
  return best;
}

/** Per-frame speech decision (argmax over the 7 classes ≠ NO_SPEAKER). */
export function decodeSpeechFrames(logits: readonly (readonly number[])[]): boolean[] {
  return logits.map((row) => {
    const cls = argmax(row);
    return cls > 0 && cls < NUM_POWERSET_CLASSES;
  });
}

/**
 * Contiguous speech runs of one window's frames, in absolute seconds. Frame k
 * covers `windowStartSec + [k, k+1) * ratio`, ratio derived from the actual
 * frame count (so the model's offset/step is never hard-coded).
 */
export function framesToSpeechRegions(
  speechFrames: readonly boolean[],
  windowStartSec: number,
  windowSamples: number,
  sampleRate: number,
): TimeRegion[] {
  const numFrames = speechFrames.length;
  if (numFrames === 0 || windowSamples === 0) return [];
  const ratio = windowSamples / numFrames / sampleRate;
  const regions: TimeRegion[] = [];
  let runStart = -1;
  for (let f = 0; f <= numFrames; f++) {
    const active = f < numFrames && speechFrames[f] === true;
    if (active) {
      if (runStart < 0) runStart = f;
    } else if (runStart >= 0) {
      regions.push({
        start: windowStartSec + runStart * ratio,
        end: windowStartSec + f * ratio,
      });
      runStart = -1;
    }
  }
  return regions;
}

/** Sorts and merges regions that touch or sit within `gapSec`. */
export function mergeRegions(regions: readonly TimeRegion[], gapSec: number): TimeRegion[] {
  const sorted = [...regions].sort((a, b) => a.start - b.start);
  const merged: TimeRegion[] = [];
  for (const r of sorted) {
    const last = merged[merged.length - 1];
    if (last && r.start - last.end <= gapSec) {
      last.end = Math.max(last.end, r.end);
    } else {
      merged.push({ ...r });
    }
  }
  return merged;
}

/** Total overlap, in seconds, between a window and a set of (merged) regions. */
export function speechOverlapSeconds(
  regions: readonly TimeRegion[],
  winStartSec: number,
  winEndSec: number,
): number {
  let total = 0;
  for (const r of regions) {
    const o = Math.min(winEndSec, r.end) - Math.max(winStartSec, r.start);
    if (o > 0) total += o;
  }
  return total;
}

/** Concatenates the speech samples of a window (silence excluded). */
export function gatherSpeechSamples(
  audio: Float32Array,
  regions: readonly TimeRegion[],
  winStartSample: number,
  winEndSample: number,
  sampleRate: number,
): Float32Array {
  const out: number[] = [];
  for (const r of regions) {
    const a = Math.max(winStartSample, Math.floor(r.start * sampleRate));
    const b = Math.min(winEndSample, Math.floor(r.end * sampleRate));
    for (let s = a; s < b; s++) out.push(audio[s] ?? 0);
  }
  return Float32Array.from(out);
}

/** Plans overlapping embedding windows over `totalSamples`. */
export function planEmbedWindows(
  totalSamples: number,
  config: DiarizeConfig,
): { startSample: number; endSample: number; startSec: number; endSec: number }[] {
  const win = Math.round(config.embedWindowSec * config.sampleRate);
  const step = Math.round(config.embedStepSec * config.sampleRate);
  const windows: { startSample: number; endSample: number; startSec: number; endSec: number }[] =
    [];
  if (totalSamples <= 0) return windows;
  for (let start = 0; start < totalSamples; start += step) {
    const end = Math.min(start + win, totalSamples);
    windows.push({
      startSample: start,
      endSample: end,
      startSec: start / config.sampleRate,
      endSec: end / config.sampleRate,
    });
    if (end >= totalSamples) break;
  }
  return windows;
}

/**
 * Whether every embedding is within `threshold` of every other — i.e. the
 * audio is a single speaker. Explicit guard so single-speaker input reliably
 * returns one speaker even if a stray pair sits just under the clustering
 * chain.
 */
export function shouldCollapseToSingle(embeddings: readonly Vector[], threshold: number): boolean {
  return embeddings.length >= 2 && maxPairwiseDistance(embeddings) < threshold;
}

interface LabeledWindow {
  startSec: number;
  endSec: number;
  cluster: number;
}

/**
 * Turn labeled (overlapping) embedding windows into clean, non-overlapping
 * speaker turns: merge consecutive same-speaker windows (gap ≤ mergeGapSec),
 * and clip a turn against a following different-speaker turn it overlaps.
 */
export function assembleTurns(
  windows: readonly LabeledWindow[],
  mergeGapSec: number,
): SpeakerTurn[] {
  const sorted = [...windows].sort((a, b) => a.startSec - b.startSec);
  // Relabel clusters "Speaker N" by first appearance.
  const clusterToSpeaker = new Map<number, string>();
  for (const w of sorted) {
    if (!clusterToSpeaker.has(w.cluster)) {
      clusterToSpeaker.set(w.cluster, `Speaker ${clusterToSpeaker.size + 1}`);
    }
  }

  const turns: SpeakerTurn[] = [];
  for (const w of sorted) {
    const speaker = clusterToSpeaker.get(w.cluster) as string;
    const last = turns[turns.length - 1];
    if (last && last.speaker === speaker && w.startSec - last.end <= mergeGapSec) {
      last.end = Math.max(last.end, w.endSec);
    } else {
      // Clip the previous (different-speaker) turn so turns never overlap.
      if (last && last.end > w.startSec) last.end = w.startSec;
      turns.push({ start: w.startSec, end: w.endSec, speaker });
    }
  }
  return turns.filter((t) => t.end > t.start);
}

/** Mean vector of the given embeddings (assumes ≥1, equal length). */
function centroidOf(members: readonly Float32Array[]): Float32Array {
  const dim = members[0]!.length;
  const sum = new Float32Array(dim);
  for (const v of members) for (let i = 0; i < dim; i++) sum[i]! += v[i]!;
  for (let i = 0; i < dim; i++) sum[i]! /= members.length;
  return sum;
}

/**
 * Clusters embeddings into speaker labels. Below `maxSamples` this is exactly
 * `agglomerative` (with the collapse-to-one guard), so the calibrated small-N
 * behaviour is unchanged. Above it, a uniform subsample of `maxSamples` is
 * clustered at the same threshold, per-cluster centroids are computed, and
 * every embedding is assigned to its nearest centroid — bounding the cost for
 * long meetings while preserving the tuned threshold semantics.
 */
export function clusterEmbeddings(
  embeddings: readonly Float32Array[],
  threshold: number,
  maxSamples: number,
): number[] {
  const n = embeddings.length;
  if (n === 0) return [];

  // Uniform subsample (or all of it, when small enough).
  const sampleIdx: number[] =
    n <= maxSamples
      ? embeddings.map((_, i) => i)
      : Array.from({ length: maxSamples }, (_, k) => Math.floor((k * n) / maxSamples));
  const sample = sampleIdx.map((i) => embeddings[i]!);

  if (shouldCollapseToSingle(sample, threshold)) return new Array<number>(n).fill(0);
  const sampleLabels = agglomerative(sample, threshold);
  if (n <= maxSamples) return sampleLabels;

  // Centroid per cluster (from the sampled points), then nearest-centroid
  // assignment for every embedding.
  const membersByCluster = new Map<number, Float32Array[]>();
  sampleLabels.forEach((label, k) => {
    const list = membersByCluster.get(label) ?? [];
    list.push(sample[k]!);
    membersByCluster.set(label, list);
  });
  const clusterIds = [...membersByCluster.keys()].sort((a, b) => a - b);
  const centroids = clusterIds.map((id) => centroidOf(membersByCluster.get(id)!));

  return embeddings.map((e) => {
    let best = 0;
    let bestDist = Infinity;
    for (let c = 0; c < centroids.length; c++) {
      const d = cosineDistance(e, centroids[c]!);
      if (d < bestDist) {
        bestDist = d;
        best = clusterIds[c]!;
      }
    }
    return best;
  });
}

/**
 * Majority-vote smoothing over a label sequence: each position takes the most
 * common label in `[i-radius, i+radius]`, ties keeping the position's own
 * label. Corrects an isolated misclustered window without disturbing genuine
 * speaker boundaries (a real run of ≥ radius+1 windows survives).
 */
export function smoothLabels(labels: readonly number[], radius: number): number[] {
  if (radius <= 0 || labels.length === 0) return [...labels];
  return labels.map((own, i) => {
    const counts = new Map<number, number>();
    for (let j = Math.max(0, i - radius); j <= Math.min(labels.length - 1, i + radius); j++) {
      counts.set(labels[j]!, (counts.get(labels[j]!) ?? 0) + 1);
    }
    let best = own;
    let bestN = counts.get(own) ?? 0;
    for (const [label, n] of counts) {
      if (n > bestN) {
        bestN = n;
        best = label;
      }
    }
    return best;
  });
}

/** Progress during the (sequential, potentially long) embedding loop. */
export type DiarizationProgress = (processedWindows: number, totalWindows: number) => void;

/**
 * Full pipeline. Returns speaker turns sorted by start, labelled
 * "Speaker 1".."Speaker N" by order of first appearance. Empty when no speech
 * is detected (or the audio is longer than `config.maxWindows` allows).
 */
export async function diarize(
  audio: Float32Array,
  runSegmentation: SegmentationRunner,
  runEmbedding: EmbeddingRunner,
  config: DiarizeConfig = DEFAULT_DIARIZE_CONFIG,
  onProgress?: DiarizationProgress,
  /**
   * Polled before each (sequential, model-bound) window so a host cancel can
   * stop the loop promptly instead of draining the whole file. On abort the
   * pipeline returns no turns — the caller, which already knows it aborted,
   * discards the result rather than emitting a speaker-less transcript.
   */
  shouldAbort?: () => boolean,
): Promise<SpeakerTurn[]> {
  const { sampleRate } = config;

  // --- 1. Voice activity: run segmentation per VAD window, keep only its
  // speech/non-speech decision, and collect global speech regions. ---
  const vadWinSamples = Math.round(config.vadWindowSec * sampleRate);
  const rawRegions: TimeRegion[] = [];
  for (let start = 0; start < audio.length; start += vadWinSamples) {
    if (shouldAbort?.()) return [];
    const end = Math.min(start + vadWinSamples, audio.length);
    const windowAudio = audio.subarray(start, end);
    const logits = await runSegmentation(windowAudio);
    const speech = decodeSpeechFrames(logits);
    rawRegions.push(
      ...framesToSpeechRegions(speech, start / sampleRate, windowAudio.length, sampleRate),
    );
  }
  const speechRegions = mergeRegions(rawRegions, 0.1);
  if (speechRegions.length === 0) return [];

  // --- 2. Slide embedding windows over speech; embed the ones with enough. ---
  // The window is embedded as one contiguous slice (not gathered speech
  // samples): concatenating non-contiguous speech across silences introduces
  // spectral discontinuities that blur voices together — validated in the
  // spike, where gathering merged a third speaker that contiguous windows
  // kept distinct. The VAD gate below only *selects* which windows to embed.
  const plan = planEmbedWindows(audio.length, config);
  // Practical guard for pathologically long audio: skip diarization rather
  // than run a multi-hour embedding loop; the transcript is still returned.
  if (plan.length > config.maxWindows) return [];

  const embeddings: Float32Array[] = [];
  const embWindows: { startSec: number; endSec: number }[] = [];
  for (const win of plan) {
    if (shouldAbort?.()) return [];
    // Gate against the window's ACTUAL length, not the nominal one: the final
    // window is clamped shorter, and a fixed 3 s threshold would drop its
    // qualifying trailing speech (leaving the last segments unlabeled).
    const winLen = win.endSec - win.startSec;
    const need = Math.max(config.minSpeechFraction * winLen, Math.min(config.minSpeechSec, winLen));
    if (speechOverlapSeconds(speechRegions, win.startSec, win.endSec) >= need) {
      embeddings.push(await runEmbedding(audio.subarray(win.startSample, win.endSample)));
      embWindows.push({ startSec: win.startSec, endSec: win.endSec });
    }
    onProgress?.(embeddings.length, plan.length);
  }
  if (embeddings.length === 0) return [];

  // --- 3. Cluster (bounded for large N), then smooth isolated bad windows. ---
  const rawLabels = clusterEmbeddings(
    embeddings,
    config.clusterThreshold,
    config.maxClusterSamples,
  );
  const labels = smoothLabels(rawLabels, config.smoothingRadius);

  // --- 4. Assemble turns. ---
  const labeled: LabeledWindow[] = embWindows.map((w, i) => ({
    startSec: w.startSec,
    endSec: w.endSec,
    cluster: labels[i] as number,
  }));
  return assembleTurns(labeled, config.mergeGapSec);
}

/** Number of distinct speakers in a turn list (a small helper for tests/UI). */
export function speakerCount(turns: readonly SpeakerTurn[]): number {
  return new Set(turns.map((t) => t.speaker)).size;
}

/**
 * Label each transcript segment by the speaker turn it overlaps most (pure —
 * directive item 2). A segment with no overlap keeps `speaker` undefined. Does
 * not mutate the input.
 */
export function assignSpeakers(
  segments: readonly TranscriptSegment[],
  turns: readonly SpeakerTurn[],
): TranscriptSegment[] {
  return segments.map((segment) => {
    let bestSpeaker: string | undefined;
    let bestOverlap = 0;
    for (const turn of turns) {
      const overlap = Math.min(segment.end, turn.end) - Math.max(segment.start, turn.start);
      if (overlap > bestOverlap) {
        bestOverlap = overlap;
        bestSpeaker = turn.speaker;
      }
    }
    return bestSpeaker === undefined ? { ...segment } : { ...segment, speaker: bestSpeaker };
  });
}
