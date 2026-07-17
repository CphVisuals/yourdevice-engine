import { describe, expect, it } from 'vitest';
import { EngineClient, EngineError, type EngineWorkerLike } from './client.js';
import type { HostMessage, WorkerMessage } from './protocol.js';

/**
 * A minimal in-memory stand-in for the message-port half of a `Worker`: it
 * records everything posted to it and lets the test drive replies by
 * dispatching `MessageEvent`s straight to the registered listeners, exactly
 * like a real worker's `message` event would.
 *
 * An optional `script` auto-replies to posted messages on a microtask (real
 * workers are always asynchronous), which is what the retry tests use to
 * play out multi-worker scenarios under a plain `await client.init(...)`.
 */
class FakeWorker implements EngineWorkerLike {
  readonly posted: HostMessage[] = [];
  terminated = false;
  private listeners = new Set<(event: MessageEvent<unknown>) => void>();

  constructor(private readonly script?: (message: HostMessage, worker: FakeWorker) => void) {}

  postMessage(message: unknown): void {
    const hostMessage = message as HostMessage;
    this.posted.push(hostMessage);
    if (this.script) {
      queueMicrotask(() => this.script?.(hostMessage, this));
    }
  }

  addEventListener(_type: 'message', listener: (event: MessageEvent<unknown>) => void): void {
    this.listeners.add(listener);
  }

  removeEventListener(_type: 'message', listener: (event: MessageEvent<unknown>) => void): void {
    this.listeners.delete(listener);
  }

  terminate(): void {
    this.terminated = true;
  }

  /** Simulates the worker replying with `message`. */
  emit(message: WorkerMessage): void {
    const event = { data: message } as MessageEvent<unknown>;
    for (const listener of this.listeners) listener(event);
  }

  /** Simulates arbitrary noise on the port that does not satisfy the protocol. */
  emitRaw(data: unknown): void {
    const event = { data } as MessageEvent<unknown>;
    for (const listener of this.listeners) listener(event);
  }
}

const capabilities = {
  detected: { webnn: false, webgpu: true, wasm: true },
  active: 'webgpu' as const,
  deviceMemoryGb: 8,
};

describe('EngineClient.init', () => {
  it('resolves with the capability report on a ready reply (happy path)', async () => {
    const worker = new FakeWorker();
    const client = new EngineClient(() => worker);

    const initPromise = client.init('whisper-base');
    expect(worker.posted).toHaveLength(1);
    const sent = worker.posted[0];
    expect(sent?.type).toBe('init');
    expect(sent && sent.type === 'init' ? sent.modelId : undefined).toBe('whisper-base');

    const requestId = sent && sent.type === 'init' ? sent.requestId : '';
    worker.emit({ type: 'ready', requestId, capabilities, modelId: 'whisper-base' });

    await expect(initPromise).resolves.toEqual(capabilities);
  });

  it('reports download progress via the callback while init is pending', () => {
    const worker = new FakeWorker();
    const client = new EngineClient(() => worker);
    const progressUpdates: number[] = [];
    client.onDownloadProgress = (info) => progressUpdates.push(info.loadedBytes);

    void client.init('whisper-base');
    const requestId = worker.posted[0]?.requestId ?? '';
    worker.emit({
      type: 'download-progress',
      requestId,
      modelId: 'whisper-base',
      loadedBytes: 1024,
      totalBytes: 4096,
    });
    worker.emit({
      type: 'download-progress',
      requestId,
      modelId: 'whisper-base',
      loadedBytes: 4096,
      totalBytes: 4096,
    });

    expect(progressUpdates).toEqual([1024, 4096]);
  });

  it('rejects with an EngineError on an error reply (error path)', async () => {
    const worker = new FakeWorker();
    const client = new EngineClient(() => worker);

    const initPromise = client.init('whisper-base');
    const requestId = worker.posted[0]?.requestId ?? '';
    worker.emit({ type: 'error', requestId, code: 'no-backend', message: 'nothing available' });

    await expect(initPromise).rejects.toThrow(EngineError);
    await expect(initPromise).rejects.toMatchObject({
      code: 'no-backend',
      message: 'nothing available',
    });
  });

  it('generates distinct requestIds for concurrent init calls', () => {
    const worker = new FakeWorker();
    const client = new EngineClient(() => worker);
    void client.init('whisper-base');
    void client.init('whisper-small');
    const ids = worker.posted.map((m) => m.requestId);
    expect(new Set(ids).size).toBe(2);
  });
});

describe('EngineClient.init backend fallback across workers', () => {
  it('replaces the worker and resumes the ladder when a named backend fails', async () => {
    const workers: FakeWorker[] = [];
    const factory = (): FakeWorker => {
      const index = workers.length;
      const worker = new FakeWorker((message, w) => {
        if (message.type !== 'init') return;
        if (index === 0) {
          // First worker: webgpu passes its handshake but the graph build
          // fails — that worker is unrecoverable (poisoned session cache).
          w.emit({
            type: 'error',
            requestId: message.requestId,
            code: 'model-load-failed',
            backend: 'webgpu',
            message: 'graph build failed',
          });
        } else {
          // Fresh worker: cached weights re-report progress, then wasm succeeds.
          w.emit({
            type: 'download-progress',
            requestId: message.requestId,
            modelId: message.modelId,
            loadedBytes: 4096,
            totalBytes: 4096,
          });
          w.emit({
            type: 'ready',
            requestId: message.requestId,
            capabilities: { ...capabilities, active: 'wasm' },
            modelId: message.modelId,
          });
        }
      });
      workers.push(worker);
      return worker;
    };

    const client = new EngineClient(factory);
    const progressUpdates: number[] = [];
    client.onDownloadProgress = (info) => progressUpdates.push(info.loadedBytes);

    const report = await client.init('whisper-base');

    expect(workers).toHaveLength(2);
    expect(report.active).toBe('wasm');
    expect(workers[0]?.terminated).toBe(true);
    expect(workers[1]?.terminated).toBe(false);

    // The retry excludes the failed backend from the preference it sends.
    const retryInit = workers[1]?.posted[0];
    expect(retryInit?.type).toBe('init');
    expect(retryInit && retryInit.type === 'init' ? retryInit.backendPreference : []).toEqual([
      'webnn',
      'wasm',
    ]);

    // Download progress from the replacement worker still reaches the callback.
    expect(progressUpdates).toEqual([4096]);
  });

  it('rejects with the last error once the ladder is exhausted', async () => {
    const ladder = ['webnn', 'webgpu', 'wasm'] as const;
    const workers: FakeWorker[] = [];
    const factory = (): FakeWorker => {
      const index = workers.length;
      const worker = new FakeWorker((message, w) => {
        if (message.type !== 'init') return;
        const backend = ladder[index];
        if (!backend) throw new Error('factory called more times than the ladder has backends');
        w.emit({
          type: 'error',
          requestId: message.requestId,
          code: 'model-load-failed',
          backend,
          message: `${backend} graph build failed`,
        });
      });
      workers.push(worker);
      return worker;
    };

    const client = new EngineClient(factory);
    const initPromise = client.init('whisper-base');

    await expect(initPromise).rejects.toMatchObject({
      code: 'model-load-failed',
      backend: 'wasm',
      message: 'wasm graph build failed',
    });
    // One worker per ladder rung — no fourth worker after exhaustion.
    expect(workers).toHaveLength(3);
    expect(workers.slice(0, 2).every((w) => w.terminated)).toBe(true);
  });

  it('does not retry when the error names no backend', async () => {
    const workers: FakeWorker[] = [];
    const factory = (): FakeWorker => {
      const worker = new FakeWorker((message, w) => {
        if (message.type !== 'init') return;
        w.emit({
          type: 'error',
          requestId: message.requestId,
          code: 'model-load-failed',
          message: 'network dropped mid-download',
        });
      });
      workers.push(worker);
      return worker;
    };

    const client = new EngineClient(factory);
    await expect(client.init('whisper-base')).rejects.toMatchObject({
      code: 'model-load-failed',
    });
    expect(workers).toHaveLength(1);
  });

  it('gives up if the same backend is reported failed twice', async () => {
    const workers: FakeWorker[] = [];
    const factory = (): FakeWorker => {
      const worker = new FakeWorker((message, w) => {
        if (message.type !== 'init') return;
        // Misbehaving worker: always blames wasm, even when told to skip it.
        w.emit({
          type: 'error',
          requestId: message.requestId,
          code: 'model-load-failed',
          backend: 'wasm',
          message: 'wasm keeps failing',
        });
      });
      workers.push(worker);
      return worker;
    };

    const client = new EngineClient(factory);
    await expect(client.init('whisper-base')).rejects.toMatchObject({ backend: 'wasm' });
    // First failure consumes wasm; a second report of wasm aborts the loop.
    expect(workers).toHaveLength(2);
  });

  it('respects the caller backendPreference when narrowing', async () => {
    const workers: FakeWorker[] = [];
    const factory = (): FakeWorker => {
      const index = workers.length;
      const worker = new FakeWorker((message, w) => {
        if (message.type !== 'init') return;
        if (index === 0) {
          w.emit({
            type: 'error',
            requestId: message.requestId,
            code: 'model-load-failed',
            backend: 'wasm',
            message: 'preferred backend failed',
          });
        } else {
          w.emit({
            type: 'ready',
            requestId: message.requestId,
            capabilities,
            modelId: message.modelId,
          });
        }
      });
      workers.push(worker);
      return worker;
    };

    const client = new EngineClient(factory);
    await client.init('whisper-base', { backendPreference: ['wasm'] });

    // Preferred-but-failed wasm is excluded; the rest of the ladder remains.
    const retryInit = workers[1]?.posted[0];
    expect(retryInit && retryInit.type === 'init' ? retryInit.backendPreference : []).toEqual([
      'webnn',
      'webgpu',
    ]);
  });
});

describe('EngineClient.transcribe', () => {
  it('resolves with segments on a complete reply (happy path)', async () => {
    const worker = new FakeWorker();
    const client = new EngineClient(() => worker);

    const transcribePromise = client.transcribe(new Float32Array(16_000), { task: 'transcribe' });
    const requestId = worker.posted[0]?.requestId ?? '';
    const segments = [{ start: 0, end: 1.2, text: 'hello world' }];
    worker.emit({ type: 'complete', requestId, segments });

    await expect(transcribePromise).resolves.toEqual(segments);
  });

  it('invokes onPartial for partial replies without resolving', async () => {
    const worker = new FakeWorker();
    const client = new EngineClient(() => worker);
    const partials: number[] = [];

    const transcribePromise = client.transcribe(
      new Float32Array(16_000),
      { task: 'transcribe' },
      { onPartial: (_segments, processed) => partials.push(processed) },
    );
    const requestId = worker.posted[0]?.requestId ?? '';
    worker.emit({
      type: 'partial',
      requestId,
      segments: [],
      processedSeconds: 5,
      totalSeconds: 30,
    });
    worker.emit({
      type: 'partial',
      requestId,
      segments: [],
      processedSeconds: 10,
      totalSeconds: 30,
    });
    worker.emit({ type: 'complete', requestId, segments: [] });

    await transcribePromise;
    expect(partials).toEqual([5, 10]);
  });

  it('rejects with an EngineError on an error reply (error path)', async () => {
    const worker = new FakeWorker();
    const client = new EngineClient(() => worker);

    const transcribePromise = client.transcribe(new Float32Array(16_000), { task: 'transcribe' });
    const requestId = worker.posted[0]?.requestId ?? '';
    worker.emit({ type: 'error', requestId, code: 'transcribe-failed', message: 'boom' });

    await expect(transcribePromise).rejects.toMatchObject({ code: 'transcribe-failed' });
  });

  it('rejects every pending call on a requestId-less error (worker-level failure)', async () => {
    const worker = new FakeWorker();
    const client = new EngineClient(() => worker);

    const a = client.transcribe(new Float32Array(1), { task: 'transcribe' });
    const b = client.transcribe(new Float32Array(1), { task: 'transcribe' });
    worker.emit({ type: 'error', code: 'transcribe-failed', message: 'worker crashed' });

    await expect(a).rejects.toMatchObject({ code: 'transcribe-failed' });
    await expect(b).rejects.toMatchObject({ code: 'transcribe-failed' });
  });
});

describe('EngineClient malformed-message handling', () => {
  it('ignores payloads that fail isWorkerMessage instead of resolving/rejecting', async () => {
    const worker = new FakeWorker();
    const client = new EngineClient(() => worker);

    const initPromise = client.init('whisper-base');
    const requestId = worker.posted[0]?.requestId ?? '';

    // Noise that doesn't satisfy the protocol at all.
    worker.emitRaw({ type: 'ready' }); // missing capabilities/modelId
    worker.emitRaw('just a string');
    worker.emitRaw(null);
    worker.emitRaw({
      type: 'ready',
      requestId,
      capabilities: { bogus: true },
      modelId: 'whisper-base',
    });
    // An error with a bogus backend field must also be dropped, not trusted.
    worker.emitRaw({
      type: 'error',
      requestId,
      code: 'model-load-failed',
      message: 'x',
      backend: 'cuda',
    });

    // The pending call must still be alive and resolve normally afterwards.
    worker.emit({ type: 'ready', requestId, capabilities, modelId: 'whisper-base' });
    await expect(initPromise).resolves.toEqual(capabilities);
  });

  it('does not throw when the fake worker emits raw junk with no listeners registered yet', () => {
    const worker = new FakeWorker();
    expect(() => worker.emitRaw(undefined)).not.toThrow();
    // Instantiate after, just to exercise the constructor's listener wiring.
    const client = new EngineClient(() => worker);
    expect(client).toBeInstanceOf(EngineClient);
  });
});

describe('EngineClient.abort', () => {
  it('posts an abort message with the given requestId', () => {
    const worker = new FakeWorker();
    const client = new EngineClient(() => worker);
    client.abort('transcribe-1');
    expect(worker.posted).toEqual([{ type: 'abort', requestId: 'transcribe-1' }]);
  });
});

describe('EngineClient.dispose', () => {
  it('terminates the worker and stops reacting to its messages', async () => {
    const worker = new FakeWorker();
    const client = new EngineClient(() => worker);
    const initPromise = client.init('whisper-base');
    const requestId = worker.posted[0]?.requestId ?? '';

    client.dispose();
    expect(worker.terminated).toBe(true);
    worker.emit({ type: 'ready', requestId, capabilities, modelId: 'whisper-base' });

    // Give any stray microtask a chance to run, then assert nothing settled it.
    const race = await Promise.race([
      initPromise.then(() => 'resolved' as const),
      new Promise((resolve) => setTimeout(() => resolve('pending' as const), 10)),
    ]);
    expect(race).toBe('pending');
  });
});
