/**
 * `EngineClient` — the framework-free host-side wrapper around the engine
 * worker. Dependency-injected with a `Worker` instance (constructed by the
 * host, e.g. `new Worker(new URL('@yourdevice/engine/worker', import.meta.url),
 * { type: 'module' })`) so this class stays testable with a fake
 * message-port and bundler-agnostic — it never constructs a `Worker` itself.
 *
 * Never trusts a message off the worker port: every inbound message is
 * validated with `isWorkerMessage` before it can resolve/reject a pending
 * call or fire a callback (CLAUDE.md: "Worker boundary messages must be
 * validated with the guards in protocol.ts — never trust `postMessage`
 * payloads").
 */

import type { BackendId, CapabilityReport } from './backends.js';
import type { ModelId } from './models.js';
import {
  isWorkerMessage,
  type EngineErrorCode,
  type HostMessage,
  type TranscribeOptions,
  type TranscriptSegment,
} from './protocol.js';

/**
 * A `Worker`-shaped dependency: exactly what `EngineClient` needs, nothing
 * more. A real `Worker` satisfies this structurally; tests can pass a plain
 * message-port stub instead (see client.test.ts) without implementing the
 * full DOM `Worker` interface (its overloaded `addEventListener` accepts
 * every `WorkerEventMap` key, which a minimal fake has no reason to model).
 */
export interface EngineWorkerLike {
  postMessage(message: HostMessage): void;
  addEventListener(type: 'message', listener: (event: MessageEvent<unknown>) => void): void;
  removeEventListener(type: 'message', listener: (event: MessageEvent<unknown>) => void): void;
}

export class EngineError extends Error {
  readonly code: EngineErrorCode;

  constructor(code: EngineErrorCode, message: string) {
    super(message);
    this.name = 'EngineError';
    this.code = code;
  }
}

export interface InitOptions {
  backendPreference?: readonly BackendId[];
}

export interface TranscribeCallbacks {
  onPartial?: (
    segments: TranscriptSegment[],
    processedSeconds: number,
    totalSeconds: number,
  ) => void;
}

export type DownloadProgressCallback = (info: {
  modelId: ModelId;
  loadedBytes: number;
  totalBytes: number;
}) => void;

interface PendingInit {
  kind: 'init';
  resolve: (report: CapabilityReport) => void;
  reject: (err: EngineError) => void;
}

interface PendingTranscribe {
  kind: 'transcribe';
  resolve: (segments: TranscriptSegment[]) => void;
  reject: (err: EngineError) => void;
  onPartial: TranscribeCallbacks['onPartial'];
}

type Pending = PendingInit | PendingTranscribe;

export class EngineClient {
  private readonly worker: EngineWorkerLike;
  private readonly pending = new Map<string, Pending>();
  private requestCounter = 0;
  private downloadProgressCallback: DownloadProgressCallback | null = null;

  constructor(worker: EngineWorkerLike) {
    this.worker = worker;
    this.worker.addEventListener('message', this.handleMessage);
  }

  /** Fires on every `download-progress` message, regardless of which `init` requested it. */
  set onDownloadProgress(callback: DownloadProgressCallback | null) {
    this.downloadProgressCallback = callback;
  }

  init(modelId: ModelId, options: InitOptions = {}): Promise<CapabilityReport> {
    const requestId = this.nextRequestId('init');
    return new Promise<CapabilityReport>((resolve, reject) => {
      this.pending.set(requestId, { kind: 'init', resolve, reject });
      const message: HostMessage = {
        type: 'init',
        requestId,
        modelId,
        backendPreference: options.backendPreference ? [...options.backendPreference] : undefined,
      };
      this.worker.postMessage(message);
    });
  }

  transcribe(
    audio: Float32Array,
    options: TranscribeOptions,
    callbacks: TranscribeCallbacks = {},
  ): Promise<TranscriptSegment[]> {
    const requestId = this.nextRequestId('transcribe');
    return new Promise<TranscriptSegment[]>((resolve, reject) => {
      this.pending.set(requestId, {
        kind: 'transcribe',
        resolve,
        reject,
        onPartial: callbacks.onPartial,
      });
      const message: HostMessage = { type: 'transcribe', requestId, audio, options };
      this.worker.postMessage(message);
    });
  }

  /** Best-effort: asks the worker to cancel `requestId` if it is still in flight. */
  abort(requestId: string): void {
    const message: HostMessage = { type: 'abort', requestId };
    this.worker.postMessage(message);
  }

  /** Detaches the message listener. The worker itself is owned by the caller. */
  dispose(): void {
    this.worker.removeEventListener('message', this.handleMessage);
    this.pending.clear();
  }

  private nextRequestId(prefix: string): string {
    this.requestCounter += 1;
    return `${prefix}-${this.requestCounter}`;
  }

  private readonly handleMessage = (event: MessageEvent<unknown>): void => {
    const data = event.data;
    // Guards against a malicious/malformed payload on the port: anything
    // that doesn't satisfy the protocol is dropped rather than trusted.
    if (!isWorkerMessage(data)) return;

    switch (data.type) {
      case 'ready': {
        const pending = this.takePending(data.requestId, 'init');
        pending?.resolve(data.capabilities);
        break;
      }
      case 'download-progress': {
        this.downloadProgressCallback?.({
          modelId: data.modelId,
          loadedBytes: data.loadedBytes,
          totalBytes: data.totalBytes,
        });
        break;
      }
      case 'partial': {
        const pending = this.pending.get(data.requestId);
        if (pending?.kind === 'transcribe') {
          pending.onPartial?.(data.segments, data.processedSeconds, data.totalSeconds);
        }
        break;
      }
      case 'complete': {
        const pending = this.takePending(data.requestId, 'transcribe');
        pending?.resolve(data.segments);
        break;
      }
      case 'error': {
        const err = new EngineError(data.code, data.message);
        if (data.requestId !== undefined) {
          const pending = this.pending.get(data.requestId);
          if (pending) {
            this.pending.delete(data.requestId);
            pending.reject(err);
          }
          break;
        }
        // No requestId: a worker-level failure not tied to one call. Fail
        // every call still waiting rather than let them hang forever.
        for (const [id, waiting] of this.pending) {
          this.pending.delete(id);
          waiting.reject(err);
        }
        break;
      }
    }
  };

  private takePending<K extends Pending['kind']>(
    requestId: string,
    kind: K,
  ): Extract<Pending, { kind: K }> | undefined {
    const pending = this.pending.get(requestId);
    if (!pending || pending.kind !== kind) return undefined;
    this.pending.delete(requestId);
    return pending as Extract<Pending, { kind: K }>;
  }
}
