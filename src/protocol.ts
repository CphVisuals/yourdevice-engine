import { BACKEND_LADDER, type BackendId, type CapabilityReport } from './backends.js';
import { isModelId, type ModelId } from './models.js';

/**
 * Message protocol between the host page and the engine worker. The worker
 * boundary is untyped at runtime, so both sides validate with the guards
 * below instead of trusting `postMessage` payloads.
 *
 * Every long-running operation — including `init`, which downloads model
 * weights — carries a `requestId`, so `abort` can target any of them.
 */

export interface TranscribeOptions {
  /** Whisper language code (e.g. 'en', 'sv'); omit to auto-detect. */
  language?: string;
  task: 'transcribe' | 'translate';
  /**
   * Opt in to speaker diarization ("who spoke"). Off by default: when absent
   * or false the fast path is byte-identical and the diarization models are
   * never loaded. When true, the worker labels each segment with a speaker
   * (see `TranscriptSegment.speaker`).
   */
  diarize?: boolean;
}

export interface TranscriptSegment {
  /** Seconds from the start of the input. */
  start: number;
  end: number;
  text: string;
  /**
   * Anonymous speaker label ("Speaker 1".."Speaker N"), present only when
   * diarization ran and this segment overlapped a speaker turn. The host may
   * rename these for display/export; the engine only ever emits the anonymous
   * form.
   */
  speaker?: string;
}

/**
 * `worker-failed` is raised by the host client when the Worker itself dies at
 * the DOM level (script 404/CSP block/crash — an `error` event, not a
 * protocol message); the worker never sends it, but it shares this vocabulary
 * so hosts handle one error type.
 */
export type EngineErrorCode =
  | 'no-backend'
  | 'model-load-failed'
  | 'decode-failed'
  | 'transcribe-failed'
  | 'aborted'
  | 'worker-failed';

/** Host page -> worker. */
export type HostMessage =
  | { type: 'init'; requestId: string; modelId: ModelId; backendPreference?: BackendId[] }
  | { type: 'transcribe'; requestId: string; audio: Float32Array; options: TranscribeOptions }
  | { type: 'abort'; requestId: string };

/** Worker -> host page. */
export type WorkerMessage =
  | { type: 'ready'; requestId: string; capabilities: CapabilityReport; modelId: ModelId }
  | {
      type: 'download-progress';
      requestId: string;
      modelId: ModelId;
      loadedBytes: number;
      totalBytes: number;
    }
  | {
      type: 'partial';
      requestId: string;
      segments: TranscriptSegment[];
      processedSeconds: number;
      totalSeconds: number;
    }
  | {
      /** Progress of the (opt-in, post-transcription) diarization pass. */
      type: 'diarization-progress';
      requestId: string;
      processed: number;
      total: number;
    }
  | { type: 'complete'; requestId: string; segments: TranscriptSegment[] }
  | {
      type: 'error';
      requestId?: string;
      code: EngineErrorCode;
      message: string;
      /**
       * The backend whose runtime init failed, when the failure is
       * attributable to one (init-time `model-load-failed`). Lets the host
       * retry on a fresh worker with that backend excluded — necessary
       * because onnxruntime-web caches its first failed session-creation
       * promise at module scope, so in-worker fallback after a failed
       * `pipeline()` call is impossible (see worker.ts `handleInit`).
       */
      backend?: BackendId;
    };

const BACKEND_IDS = new Set<string>(BACKEND_LADDER);
const ERROR_CODES = new Set<string>([
  'no-backend',
  'model-load-failed',
  'decode-failed',
  'transcribe-failed',
  'aborted',
  'worker-failed',
]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function isBackendIdArray(value: unknown): value is BackendId[] {
  return Array.isArray(value) && value.every((id) => typeof id === 'string' && BACKEND_IDS.has(id));
}

function isTranscribeOptions(value: unknown): value is TranscribeOptions {
  if (!isRecord(value)) return false;
  if (value.task !== 'transcribe' && value.task !== 'translate') return false;
  if (value.language !== undefined && typeof value.language !== 'string') return false;
  return value.diarize === undefined || typeof value.diarize === 'boolean';
}

function isTranscriptSegment(value: unknown): value is TranscriptSegment {
  return (
    isRecord(value) &&
    isFiniteNumber(value.start) &&
    isFiniteNumber(value.end) &&
    typeof value.text === 'string' &&
    (value.speaker === undefined || typeof value.speaker === 'string')
  );
}

function isSegmentArray(value: unknown): value is TranscriptSegment[] {
  return Array.isArray(value) && value.every(isTranscriptSegment);
}

function isCapabilityReport(value: unknown): value is CapabilityReport {
  if (!isRecord(value) || !isRecord(value.detected)) return false;
  for (const id of BACKEND_LADDER) {
    if (typeof value.detected[id] !== 'boolean') return false;
  }
  const activeOk =
    value.active === null || (typeof value.active === 'string' && BACKEND_IDS.has(value.active));
  const memoryOk = value.deviceMemoryGb === null || isFiniteNumber(value.deviceMemoryGb);
  return activeOk && memoryOk && typeof value.isMobile === 'boolean';
}

export function isHostMessage(value: unknown): value is HostMessage {
  if (!isRecord(value)) return false;
  switch (value.type) {
    case 'init':
      return (
        typeof value.requestId === 'string' &&
        isModelId(value.modelId) &&
        (value.backendPreference === undefined || isBackendIdArray(value.backendPreference))
      );
    case 'transcribe':
      return (
        typeof value.requestId === 'string' &&
        value.audio instanceof Float32Array &&
        isTranscribeOptions(value.options)
      );
    case 'abort':
      return typeof value.requestId === 'string';
    default:
      return false;
  }
}

export function isWorkerMessage(value: unknown): value is WorkerMessage {
  if (!isRecord(value)) return false;
  switch (value.type) {
    case 'ready':
      return (
        typeof value.requestId === 'string' &&
        isCapabilityReport(value.capabilities) &&
        isModelId(value.modelId)
      );
    case 'download-progress':
      return (
        typeof value.requestId === 'string' &&
        isModelId(value.modelId) &&
        isFiniteNumber(value.loadedBytes) &&
        isFiniteNumber(value.totalBytes)
      );
    case 'partial':
      return (
        typeof value.requestId === 'string' &&
        isSegmentArray(value.segments) &&
        isFiniteNumber(value.processedSeconds) &&
        isFiniteNumber(value.totalSeconds)
      );
    case 'diarization-progress':
      return (
        typeof value.requestId === 'string' &&
        isFiniteNumber(value.processed) &&
        isFiniteNumber(value.total)
      );
    case 'complete':
      return typeof value.requestId === 'string' && isSegmentArray(value.segments);
    case 'error':
      return (
        (value.requestId === undefined || typeof value.requestId === 'string') &&
        typeof value.code === 'string' &&
        ERROR_CODES.has(value.code) &&
        typeof value.message === 'string' &&
        (value.backend === undefined ||
          (typeof value.backend === 'string' && BACKEND_IDS.has(value.backend)))
      );
    default:
      return false;
  }
}
