import type { BackendId, CapabilityReport } from './backends.js';
import { isModelId, type ModelId } from './models.js';

/**
 * Message protocol between the host page and the engine worker. The worker
 * boundary is untyped at runtime, so both sides validate with the guards
 * below instead of trusting `postMessage` payloads.
 */

export interface TranscribeOptions {
  /** Whisper language code (e.g. 'en', 'sv'); omit to auto-detect. */
  language?: string;
  task: 'transcribe' | 'translate';
}

export interface TranscriptSegment {
  /** Seconds from the start of the input. */
  start: number;
  end: number;
  text: string;
}

export type EngineErrorCode =
  'no-backend' | 'model-load-failed' | 'decode-failed' | 'transcribe-failed' | 'aborted';

/** Host page -> worker. */
export type HostMessage =
  | { type: 'init'; modelId: ModelId; backendPreference?: BackendId[] }
  | { type: 'transcribe'; requestId: string; audio: Float32Array; options: TranscribeOptions }
  | { type: 'abort'; requestId: string };

/** Worker -> host page. */
export type WorkerMessage =
  | { type: 'ready'; capabilities: CapabilityReport; modelId: ModelId }
  | { type: 'download-progress'; modelId: ModelId; loadedBytes: number; totalBytes: number }
  | {
      type: 'partial';
      requestId: string;
      segments: TranscriptSegment[];
      processedSeconds: number;
      totalSeconds: number;
    }
  | { type: 'complete'; requestId: string; segments: TranscriptSegment[] }
  | { type: 'error'; requestId?: string; code: EngineErrorCode; message: string };

const HOST_MESSAGE_TYPES = new Set(['init', 'transcribe', 'abort']);
const WORKER_MESSAGE_TYPES = new Set([
  'ready',
  'download-progress',
  'partial',
  'complete',
  'error',
]);

function hasType(value: unknown): value is { type: string } {
  return (
    typeof value === 'object' &&
    value !== null &&
    typeof (value as { type?: unknown }).type === 'string'
  );
}

export function isHostMessage(value: unknown): value is HostMessage {
  if (!hasType(value) || !HOST_MESSAGE_TYPES.has(value.type)) return false;
  const msg = value as Partial<HostMessage> & { type: string };
  switch (msg.type) {
    case 'init':
      return isModelId((msg as { modelId?: unknown }).modelId);
    case 'transcribe': {
      const t = msg as { requestId?: unknown; audio?: unknown; options?: unknown };
      return (
        typeof t.requestId === 'string' &&
        t.audio instanceof Float32Array &&
        typeof t.options === 'object' &&
        t.options !== null
      );
    }
    case 'abort':
      return typeof (msg as { requestId?: unknown }).requestId === 'string';
    default:
      return false;
  }
}

export function isWorkerMessage(value: unknown): value is WorkerMessage {
  return hasType(value) && WORKER_MESSAGE_TYPES.has(value.type);
}
