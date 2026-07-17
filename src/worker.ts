/**
 * Engine worker entry point. Runs Whisper via Transformers.js inside a
 * dedicated Web Worker so the main thread never blocks. Consumed by hosts as
 * `@yourdevice/engine/worker` (see package.json `exports`); never imported
 * directly — the host constructs a `Worker` pointed at this module's URL and
 * talks to it through `EngineClient` (client.ts) using the typed protocol in
 * protocol.ts. Nothing here trusts a `postMessage` payload without running it
 * through `isHostMessage` first, and nothing leaves this file without passing
 * `isWorkerMessage` (CLAUDE.md: "Worker boundary messages must be validated").
 */

import type {
  AutomaticSpeechRecognitionOutput,
  AutomaticSpeechRecognitionPipeline,
  ProgressInfo,
} from '@huggingface/transformers';
import { env, pipeline } from '@huggingface/transformers';
import { TARGET_SAMPLE_RATE } from './audio.js';
import {
  buildCapabilityReport,
  planBackendOrder,
  type BackendId,
  type CapabilityReport,
  type DetectionScope,
} from './backends.js';
import { MODELS, type ModelId } from './models.js';
import {
  isHostMessage,
  isWorkerMessage,
  type HostMessage,
  type TranscriptSegment,
  type WorkerMessage,
} from './protocol.js';
import { dedupeWindowSegments, planWindows, WINDOW_SECONDS } from './windows.js';

// No COOP/COEP at launch (CLAUDE.md rule 7: no ad-iframe-breaking cross-origin
// isolation headers), so there is no SharedArrayBuffer — the WASM floor must
// stay single-threaded. `wasm` is typed optional (transformers.js types
// `backends.onnx` as `Partial<Env>`) even though it's always populated at
// runtime, so guard rather than assert non-null.
if (env.backends.onnx.wasm) {
  env.backends.onnx.wasm.numThreads = 1;
}

const DEVICE_BY_BACKEND: Record<BackendId, 'webnn' | 'webgpu' | 'wasm'> = {
  webnn: 'webnn',
  webgpu: 'webgpu',
  wasm: 'wasm',
};

/** The one ASR pipeline currently loaded, if `init` has completed successfully. */
let activePipeline: AutomaticSpeechRecognitionPipeline | null = null;
let activeModelId: ModelId | null = null;

/** requestId of the operation currently in flight, so `abort` can target it. */
let inFlightInitRequestId: string | null = null;
let inFlightTranscribeRequestId: string | null = null;

/** requestIds that were aborted while their operation was still running. */
const abortedRequestIds = new Set<string>();

/** Per-request, per-file download progress, aggregated into one byte count. */
const downloadProgressByRequest = new Map<string, Map<string, { loaded: number; total: number }>>();

function send(message: WorkerMessage): void {
  // Debug assertion: every outbound message must satisfy the same guard the
  // host uses to validate inbound messages, so a protocol drift trips here
  // (in the worker, loudly) instead of silently confusing the host.
  if (!isWorkerMessage(message)) {
    throw new Error(`engine worker built an invalid outbound message: ${JSON.stringify(message)}`);
  }
  postMessage(message);
}

function describeError(err: unknown): string {
  if (err instanceof Error) return err.message;
  if (typeof err === 'string') return err;
  return 'unknown error';
}

function buildDetectionScope(): DetectionScope {
  return {
    // WorkerNavigator's TS type doesn't declare `gpu`/`ml`/`deviceMemory`
    // (they postdate the shipped lib.webworker.d.ts); DetectionScope only
    // needs their shape, so assert through `unknown` rather than `any`.
    navigator: navigator as unknown as DetectionScope['navigator'],
    WebAssembly,
  };
}

/**
 * A real runtime handshake for a backend, independent of Transformers.js —
 * see the long comment on `handleInit` for why this has to happen *before*
 * we ever call `pipeline()`. `wasm` has no separate handshake: its
 * "detected" bit (the `WebAssembly` global existing) already is the
 * strongest signal available short of actually building the graph.
 */
async function probeBackend(backend: BackendId): Promise<boolean> {
  switch (backend) {
    case 'webgpu': {
      const gpu = (navigator as unknown as { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
      if (!gpu) return false;
      try {
        return (await gpu.requestAdapter()) != null;
      } catch {
        return false;
      }
    }
    case 'webnn': {
      const ml = (navigator as unknown as { ml?: { createContext(): Promise<unknown> } }).ml;
      if (!ml) return false;
      try {
        await ml.createContext();
        return true;
      } catch {
        return false;
      }
    }
    case 'wasm':
      return true;
  }
}

function handleDownloadProgress(requestId: string, modelId: ModelId, info: ProgressInfo): void {
  if (abortedRequestIds.has(requestId) || info.status !== 'progress') return;
  let files = downloadProgressByRequest.get(requestId);
  if (!files) {
    files = new Map();
    downloadProgressByRequest.set(requestId, files);
  }
  files.set(info.file, { loaded: info.loaded, total: info.total });

  let loadedBytes = 0;
  let totalBytes = 0;
  for (const file of files.values()) {
    loadedBytes += file.loaded;
    totalBytes += file.total;
  }
  send({ type: 'download-progress', requestId, modelId, loadedBytes, totalBytes });
}

async function handleInit(message: Extract<HostMessage, { type: 'init' }>): Promise<void> {
  const { requestId, modelId, backendPreference } = message;
  inFlightInitRequestId = requestId;

  try {
    const scope = buildDetectionScope();
    const report = buildCapabilityReport(scope);
    const order = planBackendOrder(report.detected, backendPreference);

    let active: BackendId | null = null;
    let lastError: unknown = null;
    let failedBackend: BackendId | null = null;

    // Try-init-with-fallback (CLAUDE.md rule 6): detection only gates what we
    // *try*; the first backend that successfully builds the pipeline wins.
    //
    // This runs in two stages instead of one straight `pipeline()` loop
    // because of an onnxruntime-web limitation surfaced through
    // Transformers.js 3.8.1: `createInferenceSession` caches its *first*
    // session-creation promise in a module-level `wasmInitPromise` and every
    // later call — for any device — awaits that same promise before doing
    // its own work. If the first backend we try fails (e.g. WebGPU with no
    // real adapter, exactly the case CI's headless Chromium hits), that
    // failure is cached, and it deterministically breaks every later
    // `pipeline()` call in this worker, even for `wasm`. So we do the real
    // browser-native handshake for each candidate first (`probeBackend` —
    // this *is* a runtime attempt, not feature-sniffing: `requestAdapter()`
    // can and does fail even when `navigator.gpu` exists), and only ever
    // call `pipeline()` once, for the first candidate whose handshake
    // actually succeeds.
    //
    // If that one `pipeline()` call still fails (WebNN's documented failure
    // mode: a context that builds but a graph that doesn't), this worker is
    // spent — with the bug above, a second `pipeline()` call could not be
    // trusted. So we report `model-load-failed` *with the failed backend
    // named* and the host (`EngineClient.init`) resumes the ladder on a
    // fresh Worker (fresh module state → unpoisoned `wasmInitPromise`),
    // re-initing with that backend excluded from `backendPreference`.
    for (const backend of order) {
      if (abortedRequestIds.has(requestId)) break;
      const handshakeOk = await probeBackend(backend);
      if (!handshakeOk) {
        lastError = new Error(`${backend}: real runtime handshake failed`);
        continue;
      }
      try {
        const asr = await pipeline('automatic-speech-recognition', MODELS[modelId].hfRepo, {
          device: DEVICE_BY_BACKEND[backend],
          dtype: 'q8',
          progress_callback: (info) => handleDownloadProgress(requestId, modelId, info),
        });
        // Aborted while the pipeline was building: discard it *before*
        // installing, or this stale continuation would overwrite whatever a
        // newer init has since loaded (wrong model for every later
        // transcribe, with no error surfaced).
        if (abortedRequestIds.has(requestId)) break;
        activePipeline = asr;
        activeModelId = modelId;
        active = backend;
      } catch (err) {
        lastError = err;
        failedBackend = backend;
      }
      break;
    }

    if (abortedRequestIds.has(requestId)) {
      // Already acknowledged synchronously in handleAbort; nothing more to send.
      return;
    }

    if (active === null) {
      if (failedBackend !== null) {
        // A backend passed its handshake but failed to build the pipeline.
        // Name it so the host can retry on a fresh worker without it.
        send({
          type: 'error',
          requestId,
          code: 'model-load-failed',
          backend: failedBackend,
          message: describeError(lastError),
        });
        return;
      }
      const detail =
        lastError === null ? 'no backend detected in this browser' : describeError(lastError);
      send({ type: 'error', requestId, code: 'no-backend', message: detail });
      return;
    }

    const capabilities: CapabilityReport = { ...report, active };
    send({ type: 'ready', requestId, capabilities, modelId });
  } catch (err) {
    if (!abortedRequestIds.has(requestId)) {
      send({ type: 'error', requestId, code: 'model-load-failed', message: describeError(err) });
    }
  } finally {
    if (inFlightInitRequestId === requestId) inFlightInitRequestId = null;
    abortedRequestIds.delete(requestId);
    downloadProgressByRequest.delete(requestId);
  }
}

function toSegments(
  output: AutomaticSpeechRecognitionOutput | AutomaticSpeechRecognitionOutput[],
  totalSeconds: number,
): TranscriptSegment[] {
  const result = Array.isArray(output) ? output[0] : output;
  if (!result) return [];
  if (!result.chunks || result.chunks.length === 0) {
    return result.text.trim() ? [{ start: 0, end: totalSeconds, text: result.text.trim() }] : [];
  }
  return result.chunks.map((chunk, index, chunks) => {
    const [start, rawEnd] = chunk.timestamp;
    // Whisper's timestamp decoder legitimately leaves the final chunk's end
    // timestamp `null` (the closing timestamp token is not always emitted)
    // even though the type says `[number, number]`. Fall back to the audio's
    // total duration for the last chunk — collapsing it to `start` would
    // render real speech as a zero-width range.
    const end =
      typeof rawEnd === 'number' ? rawEnd : index === chunks.length - 1 ? totalSeconds : start;
    return { start, end, text: chunk.text.trim() };
  });
}

async function handleTranscribe(
  message: Extract<HostMessage, { type: 'transcribe' }>,
): Promise<void> {
  const { requestId, audio, options } = message;

  if (activePipeline === null || activeModelId === null) {
    send({
      type: 'error',
      requestId,
      code: 'transcribe-failed',
      message: 'engine is not initialized; call init before transcribe',
    });
    return;
  }

  // One transcription at a time: the pipeline/session is not reentrant, and
  // allowing overlap would clobber the in-flight bookkeeping (an abort for
  // the first request would silently no-op against the second's id).
  if (inFlightTranscribeRequestId !== null) {
    send({
      type: 'error',
      requestId,
      code: 'transcribe-failed',
      message: 'a transcription is already in flight; await or abort it first',
    });
    return;
  }

  inFlightTranscribeRequestId = requestId;
  try {
    const spec = MODELS[activeModelId];
    // English-only models throw if language/task are passed at all, so omit
    // both entirely for them; multilingual models take task always and
    // language only when the caller supplied one (otherwise auto-detect).
    const asrOptions = spec.multilingual
      ? {
          return_timestamps: true as const,
          chunk_length_s: WINDOW_SECONDS,
          task: options.task,
          ...(options.language ? { language: options.language } : {}),
        }
      : { return_timestamps: true as const, chunk_length_s: WINDOW_SECONDS };

    const totalSeconds = audio.length / TARGET_SAMPLE_RATE;
    const windows = planWindows(audio.length, TARGET_SAMPLE_RATE);

    // Audio that fits in one window skips the loop machinery entirely.
    if (windows.length <= 1) {
      const output = await activePipeline(audio, asrOptions);
      if (abortedRequestIds.has(requestId)) return;
      send({ type: 'complete', requestId, segments: toSegments(output, totalSeconds) });
      return;
    }

    // Long audio: process window by window (30 s each, advancing 25 s so
    // consecutive windows share a 5 s overlap), streaming 'partial' after
    // each. `subarray` views into the one buffer — no per-window copies, so
    // memory stays bounded by the input plus one window of model state. The
    // abort check at the top of each iteration is what makes abort actually
    // responsive on long files: between windows we are back in JS and can
    // stop before the next pipeline call.
    const merged: TranscriptSegment[] = [];
    for (const window of windows) {
      if (abortedRequestIds.has(requestId)) return; // ack already sent by handleAbort
      const slice = audio.subarray(window.startSample, window.endSample);
      const output = await activePipeline(slice, asrOptions);
      if (abortedRequestIds.has(requestId)) return;

      const windowSeconds = window.endSeconds - window.startSeconds;
      const absolute = toSegments(output, windowSeconds).map((segment) => ({
        start: segment.start + window.startSeconds,
        end: segment.end + window.startSeconds,
        text: segment.text,
      }));
      merged.push(...dedupeWindowSegments(absolute, window));

      if (!window.isLast) {
        send({
          type: 'partial',
          requestId,
          segments: [...merged],
          processedSeconds: Math.min(window.endSeconds, totalSeconds),
          totalSeconds,
        });
      }
    }

    send({ type: 'complete', requestId, segments: merged });
  } catch (err) {
    if (!abortedRequestIds.has(requestId)) {
      send({ type: 'error', requestId, code: 'transcribe-failed', message: describeError(err) });
    }
  } finally {
    if (inFlightTranscribeRequestId === requestId) inFlightTranscribeRequestId = null;
    abortedRequestIds.delete(requestId);
  }
}

function handleAbort(message: Extract<HostMessage, { type: 'abort' }>): void {
  const { requestId } = message;
  const matchesInit = requestId === inFlightInitRequestId;
  const matchesTranscribe = requestId === inFlightTranscribeRequestId;
  if (!matchesInit && !matchesTranscribe) {
    // Nothing in flight under this requestId (already finished, or never
    // started) — nothing to cancel.
    return;
  }

  abortedRequestIds.add(requestId);
  if (matchesInit) inFlightInitRequestId = null;
  if (matchesTranscribe) inFlightTranscribeRequestId = null;

  // Best-effort: the underlying Transformers.js call keeps running (it has
  // no cancellation hook), but we stop waiting on it and reply immediately so
  // the host is never left hanging.
  send({ type: 'error', requestId, code: 'aborted', message: 'aborted by host request' });
}

addEventListener('message', (event: MessageEvent<unknown>) => {
  const data = event.data;
  if (!isHostMessage(data)) {
    console.error('engine worker received a message that failed protocol validation', data);
    return;
  }
  switch (data.type) {
    case 'init':
      void handleInit(data);
      break;
    case 'transcribe':
      void handleTranscribe(data);
      break;
    case 'abort':
      handleAbort(data);
      break;
  }
});
