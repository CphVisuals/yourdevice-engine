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
  AutomaticSpeechRecognitionPipelineCallback,
  ProgressInfo,
} from '@huggingface/transformers';
import {
  AutoModel,
  AutoModelForAudioFrameClassification,
  AutoProcessor,
  env,
  pipeline,
} from '@huggingface/transformers';
import { TARGET_SAMPLE_RATE } from './audio.js';
import {
  buildCapabilityReport,
  planBackendOrder,
  type BackendId,
  type CapabilityReport,
  type DetectionScope,
} from './backends.js';
import { attemptBackend } from './backendInit.js';
import {
  assignSpeakers,
  diarize,
  type EmbeddingRunner,
  type SegmentationRunner,
} from './diarize.js';
import { MODELS, type ModelId } from './models.js';
import {
  isHostMessage,
  isWorkerMessage,
  type HostMessage,
  type TranscriptSegment,
  type WorkerMessage,
} from './protocol.js';
import { dedupeWindowSegments, planWindows, WINDOW_SECONDS } from './windows.js';

/** Diarization ONNX models (loaded lazily on the first diarize request). */
const DIARIZATION_SEG_REPO = 'onnx-community/pyannote-segmentation-3.0';
const DIARIZATION_EMB_REPO = 'onnx-community/wespeaker-voxceleb-resnet34-LM';

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
let activePipeline: AutomaticSpeechRecognitionPipelineCallback | null = null;
let activeModelId: ModelId | null = null;
/**
 * The device the ASR pipeline settled on after try-init-with-fallback
 * (CLAUDE.md rule 6) — including the output-validation ladder in
 * `attemptBackend`, which falls a garbage-producing accelerated backend off
 * to WASM. Diarization has no output-validation net of its own, so its
 * models must be built on this SAME device rather than probing/picking one
 * independently: if the ASR ended up on WASM because of a bad GPU,
 * diarization loads on WASM too.
 */
let activeDevice: 'webnn' | 'webgpu' | 'wasm' | null = null;

/**
 * Lazily-built diarization runners (segmentation + embedding), created on the
 * first `transcribe` with `diarize: true`. Off the default fast path: when
 * diarization is never requested these models are never downloaded.
 */
let diarizer: { runSegmentation: SegmentationRunner; runEmbedding: EmbeddingRunner } | null = null;

/**
 * Prefer a same-origin `/models/` mirror when the host serves one (see the
 * site worker): Hugging Face's edge 503s browser requests from free-hosted
 * origins, and third-party model CDNs appear on ad-blocker filter lists —
 * same-origin traffic dodges both. One probe per worker; a miss (local dev,
 * preview, no mirror deployed) leaves the default Hugging Face host in place.
 */
let modelHostConfigured = false;

async function configureModelHost(modelId: ModelId): Promise<void> {
  if (modelHostConfigured) return;
  modelHostConfigured = true;
  try {
    const probeUrl = new URL(
      `/models/${MODELS[modelId].hfRepo}/resolve/main/config.json`,
      self.location.href,
    );
    const probe = await fetch(probeUrl, { method: 'HEAD' });
    if (probe.ok) {
      env.remoteHost = new URL('/models/', self.location.href).href;
      // Same layout as the Hugging Face default, so R2 keys mirror HF paths.
      env.remotePathTemplate = '{model}/resolve/{revision}/';
    }
  } catch {
    // Mirror unreachable — keep the Hugging Face default.
  }
}

/** requestId of the operation currently in flight, so `abort` can target it. */
let inFlightInitRequestId: string | null = null;
let inFlightTranscribeRequestId: string | null = null;

/** requestIds that were aborted while their operation was still running or queued. */
const abortedRequestIds = new Set<string>();

/**
 * Transcribe requests are serialized, not rejected on overlap: the pipeline
 * isn't reentrant, and after a host-side cancel the current window keeps
 * draining (there is no cancellation hook inside a pipeline call) while the
 * host is already free to submit the next file. Each new request chains
 * behind the previous one's completion; a failure in one job must never
 * poison the chain for its successors.
 */
let transcribeChain: Promise<void> = Promise.resolve();
/** requestIds accepted but not yet started, so `abort` can cancel them too. */
const queuedTranscribeIds = new Set<string>();

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

export async function handleInit(message: Extract<HostMessage, { type: 'init' }>): Promise<void> {
  const { requestId, modelId, backendPreference } = message;
  inFlightInitRequestId = requestId;

  try {
    await configureModelHost(modelId);

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
    // mode: a context that builds but a graph that doesn't) — or, for an
    // accelerated backend, builds fine but then fails an output validation
    // probe against a tiny bundled clip (some real GPUs produce degenerate
    // "!!!!"-repeated garbage despite a clean graph build; see
    // backendInit.ts) — this worker is spent: with the bug above, a second
    // `pipeline()` call could not be trusted. So we report
    // `model-load-failed` *with the failed backend named* and the host
    // (`EngineClient.init`) resumes the ladder on a fresh Worker (fresh
    // module state → unpoisoned `wasmInitPromise`), re-initing with that
    // backend excluded from `backendPreference`.
    for (const backend of order) {
      if (abortedRequestIds.has(requestId)) break;
      const handshakeOk = await probeBackend(backend);
      if (!handshakeOk) {
        lastError = new Error(`${backend}: real runtime handshake failed`);
        continue;
      }
      // q8 is correct and compact on WASM, but int8 kernels on the WebGPU
      // execution provider produce garbage tokens at a crawl (verified on
      // real hardware) — accelerated backends get the per-model float
      // encoder + q4 decoder config instead (see AcceleratedDtype).
      // (Inline literal rather than the spec object: TS only assigns fresh
      // object literals, not interface-typed aliases, to Record dtypes.)
      const accelerated = MODELS[modelId].acceleratedDtype;
      const dtype =
        backend === 'wasm'
          ? ('q8' as const)
          : {
              encoder_model: accelerated.encoder_model,
              decoder_model_merged: accelerated.decoder_model_merged,
            };
      const result = await attemptBackend(backend, (b) =>
        pipeline('automatic-speech-recognition', MODELS[modelId].hfRepo, {
          device: DEVICE_BY_BACKEND[b],
          dtype,
          progress_callback: (info) => handleDownloadProgress(requestId, modelId, info),
        }),
      );
      if (result.ok) {
        // Aborted while the pipeline was building/validating: discard it
        // *before* installing, or this stale continuation would overwrite
        // whatever a newer init has since loaded (wrong model for every
        // later transcribe, with no error surfaced).
        if (abortedRequestIds.has(requestId)) break;
        activePipeline = result.asr;
        activeModelId = modelId;
        activeDevice = DEVICE_BY_BACKEND[backend];
        // A model change invalidates nothing about diarization (separate
        // models), but a backend/device change does — rebuild on next use.
        diarizer = null;
        active = backend;
      } else {
        lastError = result.error;
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

/**
 * Builds the diarization runners on first use (segmentation for voice
 * activity + wespeaker for speaker embeddings), reusing the ASR's SETTLED
 * device (`activeDevice`, set only once `attemptBackend`'s output-validation
 * has passed — see the comment on that variable) rather than probing/picking
 * one independently. q8 on WASM; float on accelerated backends to sidestep
 * the q8-on-WebGPU garbage class of bug (CLAUDE.md rule 9) — real-hardware
 * WebGPU verification is still required before merge.
 */
/**
 * Retry a transient async op with linear backoff. The diarization model files
 * are fetched live on first use; under network contention (or several tabs at
 * once) a fetch can hiccup, and without a retry the whole opt-in feature fails
 * silently for that run. Successful `from_pretrained` calls are cached, so a
 * retry re-resolves the already-loaded pieces cheaply.
 */
async function withRetry<T>(fn: () => Promise<T>, attempts = 3, delayMs = 500): Promise<T> {
  let lastError: unknown;
  for (let attempt = 0; attempt < attempts; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      if (attempt < attempts - 1) {
        await new Promise((resolve) => setTimeout(resolve, delayMs * (attempt + 1)));
      }
    }
  }
  throw lastError;
}

async function ensureDiarizer(
  requestId: string,
): Promise<{ runSegmentation: SegmentationRunner; runEmbedding: EmbeddingRunner }> {
  if (diarizer) return diarizer;
  const device = activeDevice ?? 'wasm';
  const dtype = device === 'wasm' ? ('q8' as const) : ('fp32' as const);
  const labelModelId = activeModelId; // progress labelling only
  const progress_callback = (info: ProgressInfo): void => {
    if (labelModelId) handleDownloadProgress(requestId, labelModelId, info);
  };

  const [segModel, segProcessor, embModel, embProcessor] = await withRetry(() =>
    Promise.all([
      AutoModelForAudioFrameClassification.from_pretrained(DIARIZATION_SEG_REPO, {
        device,
        dtype,
        progress_callback,
      }),
      AutoProcessor.from_pretrained(DIARIZATION_SEG_REPO),
      AutoModel.from_pretrained(DIARIZATION_EMB_REPO, { device, dtype, progress_callback }),
      AutoProcessor.from_pretrained(DIARIZATION_EMB_REPO),
    ]),
  );

  const runSegmentation: SegmentationRunner = async (windowAudio) => {
    const inputs = await segProcessor(windowAudio);
    const output = await segModel(inputs);
    return output.logits.tolist()[0] as number[][];
  };
  const runEmbedding: EmbeddingRunner = async (audio) => {
    const inputs = await embProcessor(audio);
    const output = await embModel(inputs);
    // WeSpeakerResNetModel exposes the vector as `embeddings`; fall back
    // defensively to keep working if the output key ever changes.
    const tensor = output.embeddings ?? output.logits ?? Object.values(output)[0];
    return Float32Array.from(tensor.tolist()[0] as number[]);
  };

  diarizer = { runSegmentation, runEmbedding };
  return diarizer;
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
      await completeTranscribe(requestId, toSegments(output, totalSeconds), audio, options);
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

    await completeTranscribe(requestId, merged, audio, options);
  } catch (err) {
    if (!abortedRequestIds.has(requestId)) {
      send({ type: 'error', requestId, code: 'transcribe-failed', message: describeError(err) });
    }
  } finally {
    if (inFlightTranscribeRequestId === requestId) inFlightTranscribeRequestId = null;
    abortedRequestIds.delete(requestId);
  }
}

/**
 * Sends the final transcript, optionally running diarization first. Off the
 * fast path: without `options.diarize`, this is a plain `complete` send.
 * Diarization is best-effort — if it fails, the transcript is returned
 * without speaker labels rather than failing the whole request.
 */
async function completeTranscribe(
  requestId: string,
  segments: TranscriptSegment[],
  audio: Float32Array,
  options: { diarize?: boolean },
): Promise<void> {
  let finalSegments = segments;
  if (options.diarize) {
    try {
      const { runSegmentation, runEmbedding } = await ensureDiarizer(requestId);
      if (abortedRequestIds.has(requestId)) return;
      const turns = await diarize(
        audio,
        runSegmentation,
        runEmbedding,
        undefined,
        (processed, total) => {
          // Surface embedding-loop progress so long jobs don't look hung.
          if (!abortedRequestIds.has(requestId)) {
            send({ type: 'diarization-progress', requestId, processed, total });
          }
        },
        // Stop the (minutes-long on WASM) embedding loop promptly on cancel.
        () => abortedRequestIds.has(requestId),
      );
      if (abortedRequestIds.has(requestId)) return;
      finalSegments = assignSpeakers(segments, turns);
    } catch (err) {
      console.error('diarization failed; returning transcript without speakers', err);
      finalSegments = segments;
    }
  }
  send({ type: 'complete', requestId, segments: finalSegments });
}

function handleAbort(message: Extract<HostMessage, { type: 'abort' }>): void {
  const { requestId } = message;
  const matchesInit = requestId === inFlightInitRequestId;
  const matchesTranscribe = requestId === inFlightTranscribeRequestId;
  const matchesQueued = queuedTranscribeIds.has(requestId);
  if (!matchesInit && !matchesTranscribe && !matchesQueued) {
    // Nothing running or queued under this requestId (already finished, or
    // never started) — nothing to cancel.
    return;
  }

  // A queued job is marked aborted here and skipped when its turn comes (the
  // serialization chain checks abortedRequestIds before starting it).
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
      // Serialize: run after the previous transcribe fully drains (see the
      // comment on `transcribeChain`).
      queuedTranscribeIds.add(data.requestId);
      transcribeChain = transcribeChain
        .then(() => {
          queuedTranscribeIds.delete(data.requestId);
          if (abortedRequestIds.has(data.requestId)) {
            // Aborted while queued: skip it — handleAbort already replied.
            abortedRequestIds.delete(data.requestId);
            return;
          }
          return handleTranscribe(data);
        })
        // handleTranscribe reports its own failures over the port; this catch
        // only guards the chain itself so one job can never block successors.
        .catch(() => {});
      break;
    case 'abort':
      handleAbort(data);
      break;
  }
});
