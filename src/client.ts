/**
 * `EngineClient` — the framework-free host-side wrapper around the engine
 * worker. Dependency-injected with a *worker factory* (the host closes over
 * its own `new Worker(new URL('@yourdevice/engine/worker', import.meta.url),
 * { type: 'module' })` expression) so this class stays testable with a fake
 * message-port and bundler-agnostic — it never names the worker URL itself.
 *
 * A factory rather than an instance because init-time backend fallback needs
 * fresh workers: when a backend passes its runtime handshake but fails to
 * build the model graph (WebNN's documented failure mode), onnxruntime-web's
 * module-level session-promise cache leaves that worker unable to try any
 * other backend (see worker.ts `handleInit`). `init` recovers here instead —
 * it disposes the dead worker, spins up a fresh one from the factory, and
 * re-inits with the failed backend excluded, until the ladder is exhausted.
 *
 * Never trusts a message off the worker port: every inbound message is
 * validated with `isWorkerMessage` before it can resolve/reject a pending
 * call or fire a callback (CLAUDE.md: "Worker boundary messages must be
 * validated with the guards in protocol.ts — never trust `postMessage`
 * payloads").
 */

import { BACKEND_LADDER, type BackendId, type CapabilityReport } from './backends.js';
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
  addEventListener(type: 'error' | 'messageerror', listener: (event: Event) => void): void;
  removeEventListener(type: 'message', listener: (event: MessageEvent<unknown>) => void): void;
  removeEventListener(type: 'error' | 'messageerror', listener: (event: Event) => void): void;
  terminate(): void;
}

export class EngineError extends Error {
  readonly code: EngineErrorCode;
  /** The backend whose init failed, when the worker could attribute the failure. */
  readonly backend?: BackendId;

  constructor(code: EngineErrorCode, message: string, backend?: BackendId) {
    super(message);
    this.name = 'EngineError';
    this.code = code;
    this.backend = backend;
  }
}

export interface InitOptions {
  backendPreference?: readonly BackendId[];
  /**
   * Fires with the requestId of every init attempt this call makes (backend
   * fallback retries each get a fresh id), so the host can `abort()` the one
   * currently in flight — e.g. from a Cancel button during model download.
   */
  onRequestStart?: (requestId: string) => void;
}

export interface TranscribeCallbacks {
  onPartial?: (
    segments: TranscriptSegment[],
    processedSeconds: number,
    totalSeconds: number,
  ) => void;
  /** Fires during the opt-in diarization pass so long jobs can show progress. */
  onDiarizationProgress?: (processed: number, total: number) => void;
  /** Fires with this call's requestId before it is posted, so the host can `abort()` it. */
  onRequestStart?: (requestId: string) => void;
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
  onDiarizationProgress: TranscribeCallbacks['onDiarizationProgress'];
}

type Pending = PendingInit | PendingTranscribe;

export class EngineClient {
  private readonly workerFactory: () => EngineWorkerLike;
  private worker: EngineWorkerLike;
  private readonly pending = new Map<string, Pending>();
  private requestCounter = 0;
  private downloadProgressCallback: DownloadProgressCallback | null = null;
  private disposed = false;

  constructor(workerFactory: () => EngineWorkerLike) {
    this.workerFactory = workerFactory;
    this.worker = workerFactory();
    this.attach(this.worker);
  }

  /** Wires up all listeners on a (fresh or initial) worker. */
  private attach(worker: EngineWorkerLike): void {
    worker.addEventListener('message', this.handleMessage);
    worker.addEventListener('error', this.handleWorkerFailure);
    worker.addEventListener('messageerror', this.handleWorkerFailure);
  }

  private detach(worker: EngineWorkerLike): void {
    worker.removeEventListener('message', this.handleMessage);
    worker.removeEventListener('error', this.handleWorkerFailure);
    worker.removeEventListener('messageerror', this.handleWorkerFailure);
  }

  /**
   * Fires on every `download-progress` message, regardless of which `init`
   * requested it — including re-downloads on replacement workers during
   * backend fallback (those are served from the browser Cache API, so they
   * flash by fast, but they are still reported).
   */
  set onDownloadProgress(callback: DownloadProgressCallback | null) {
    this.downloadProgressCallback = callback;
  }

  /**
   * Initializes the engine, resuming the backend ladder across workers when
   * needed: if a worker reports `model-load-failed` for a *named* backend
   * (handshake passed, graph build failed — that worker is unrecoverable,
   * see worker.ts), the dead worker is replaced with a fresh one from the
   * factory and init retries with every already-failed backend excluded.
   * Rejects with the last error once the ladder is exhausted.
   */
  async init(modelId: ModelId, options: InitOptions = {}): Promise<CapabilityReport> {
    // The full sequence of backends this init may ever try: the caller's
    // preference first (deduplicated), then the rest of the ladder — the
    // same merge planBackendOrder performs in the worker.
    const fullOrder = [...new Set([...(options.backendPreference ?? []), ...BACKEND_LADDER])];
    const failed = new Set<BackendId>();
    let preference = options.backendPreference ? [...options.backendPreference] : undefined;

    for (;;) {
      try {
        return await this.initOnce(modelId, preference, options.onRequestStart);
      } catch (err) {
        if (
          !(err instanceof EngineError) ||
          err.code !== 'model-load-failed' ||
          err.backend === undefined ||
          failed.has(err.backend) || // repeated failure of the same backend: give up
          this.disposed
        ) {
          throw err;
        }
        failed.add(err.backend);
        const remaining = fullOrder.filter((backend) => !failed.has(backend));
        if (remaining.length === 0) throw err;
        this.replaceWorker();
        preference = remaining;
      }
    }
  }

  transcribe(
    audio: Float32Array,
    options: TranscribeOptions,
    callbacks: TranscribeCallbacks = {},
  ): Promise<TranscriptSegment[]> {
    const requestId = this.nextRequestId('transcribe');
    callbacks.onRequestStart?.(requestId);
    return new Promise<TranscriptSegment[]>((resolve, reject) => {
      this.pending.set(requestId, {
        kind: 'transcribe',
        resolve,
        reject,
        onPartial: callbacks.onPartial,
        onDiarizationProgress: callbacks.onDiarizationProgress,
      });
      const message: HostMessage = { type: 'transcribe', requestId, audio, options };
      this.worker.postMessage(message);
    });
  }

  /**
   * Cancels `requestId`: rejects the matching in-flight call immediately with
   * code 'aborted' and notifies the worker best-effort.
   *
   * The local rejection is deliberate, not just an optimization: the worker
   * may be deep inside a synchronous WASM inference call and unable to
   * service the abort message until the current window finishes — and if
   * that window was the last one, its 'complete' can race ahead of the abort
   * entirely. The host must not depend on winning that race; it settles now
   * and any late 'complete'/'error' for this id is dropped (its pending
   * entry is gone). Only reaches the *current* worker — a request that
   * belonged to a worker since replaced by backend fallback was already
   * rejected when the worker was swapped.
   */
  abort(requestId: string): void {
    const message: HostMessage = { type: 'abort', requestId };
    this.worker.postMessage(message);
    const pending = this.pending.get(requestId);
    if (pending) {
      this.pending.delete(requestId);
      pending.reject(new EngineError('aborted', 'aborted by host request'));
      // Aborting an init needs more than a local reject: the worker would
      // keep downloading model weights in the background (terminating it is
      // the only real cancellation), and a later init would call pipeline()
      // concurrently with the aborted-but-still-running one in the same
      // worker — re-exposing the poisoned wasmInitPromise bug the ladder
      // design exists to avoid. A fresh worker gives clean module state.
      if (pending.kind === 'init' && !this.disposed) {
        this.replaceWorker();
      }
    }
  }

  /**
   * Detaches listeners, terminates the current worker, and rejects every
   * in-flight call with `aborted` — a promise that never settles would retain
   * its suspended continuation (and any captured audio buffer) for the life
   * of the page, and would hang the init retry loop before its own
   * disposed-check could run.
   */
  dispose(): void {
    this.disposed = true;
    this.detach(this.worker);
    this.worker.terminate();
    this.rejectAllPending(new EngineError('aborted', 'engine client disposed'));
  }

  /** One init attempt against the current worker. */
  private initOnce(
    modelId: ModelId,
    backendPreference: readonly BackendId[] | undefined,
    onRequestStart?: (requestId: string) => void,
  ): Promise<CapabilityReport> {
    const requestId = this.nextRequestId('init');
    onRequestStart?.(requestId);
    return new Promise<CapabilityReport>((resolve, reject) => {
      this.pending.set(requestId, { kind: 'init', resolve, reject });
      const message: HostMessage = {
        type: 'init',
        requestId,
        modelId,
        backendPreference: backendPreference ? [...backendPreference] : undefined,
      };
      this.worker.postMessage(message);
    });
  }

  /** Swaps the dead worker for a fresh one (used by init-time backend fallback). */
  private replaceWorker(): void {
    this.detach(this.worker);
    this.worker.terminate();
    // Anything else still pending was waiting on the dead worker; fail it
    // now rather than let it hang forever. (The failed init that triggered
    // the replacement has already been taken off the map.)
    this.rejectAllPending(
      new EngineError('aborted', 'worker was replaced during backend fallback'),
    );
    this.worker = this.workerFactory();
    this.attach(this.worker);
  }

  private rejectAllPending(err: EngineError): void {
    for (const [id, waiting] of this.pending) {
      this.pending.delete(id);
      waiting.reject(err);
    }
  }

  /**
   * The Worker died at the DOM level (script failed to load, CSP block,
   * crash) — no protocol message will ever arrive, so without this every
   * in-flight call would hang forever.
   */
  private readonly handleWorkerFailure = (event: Event): void => {
    const detail =
      'message' in event && typeof (event as ErrorEvent).message === 'string'
        ? (event as ErrorEvent).message
        : 'engine worker failed to load or crashed';
    this.rejectAllPending(new EngineError('worker-failed', detail));
  };

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
      case 'diarization-progress': {
        const pending = this.pending.get(data.requestId);
        if (pending?.kind === 'transcribe') {
          pending.onDiarizationProgress?.(data.processed, data.total);
        }
        break;
      }
      case 'complete': {
        const pending = this.takePending(data.requestId, 'transcribe');
        pending?.resolve(data.segments);
        break;
      }
      case 'error': {
        const err = new EngineError(data.code, data.message, data.backend);
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
