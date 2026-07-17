import { describe, expect, it } from 'vitest';
import { EngineClient, EngineError, type EngineWorkerLike } from './client.js';
import type { HostMessage, WorkerMessage } from './protocol.js';

/**
 * A minimal in-memory stand-in for the message-port half of a `Worker`: it
 * records everything posted to it and lets the test drive replies by
 * dispatching `MessageEvent`s straight to the registered listeners, exactly
 * like a real worker's `message` event would.
 */
class FakeWorker implements EngineWorkerLike {
  readonly posted: HostMessage[] = [];
  private listeners = new Set<(event: MessageEvent<unknown>) => void>();

  postMessage(message: unknown): void {
    this.posted.push(message as HostMessage);
  }

  addEventListener(_type: 'message', listener: (event: MessageEvent<unknown>) => void): void {
    this.listeners.add(listener);
  }

  removeEventListener(_type: 'message', listener: (event: MessageEvent<unknown>) => void): void {
    this.listeners.delete(listener);
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
    const client = new EngineClient(worker);

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
    const client = new EngineClient(worker);
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
    const client = new EngineClient(worker);

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
    const client = new EngineClient(worker);
    void client.init('whisper-base');
    void client.init('whisper-small');
    const ids = worker.posted.map((m) => m.requestId);
    expect(new Set(ids).size).toBe(2);
  });
});

describe('EngineClient.transcribe', () => {
  it('resolves with segments on a complete reply (happy path)', async () => {
    const worker = new FakeWorker();
    const client = new EngineClient(worker);

    const transcribePromise = client.transcribe(new Float32Array(16_000), { task: 'transcribe' });
    const requestId = worker.posted[0]?.requestId ?? '';
    const segments = [{ start: 0, end: 1.2, text: 'hello world' }];
    worker.emit({ type: 'complete', requestId, segments });

    await expect(transcribePromise).resolves.toEqual(segments);
  });

  it('invokes onPartial for partial replies without resolving', async () => {
    const worker = new FakeWorker();
    const client = new EngineClient(worker);
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
    const client = new EngineClient(worker);

    const transcribePromise = client.transcribe(new Float32Array(16_000), { task: 'transcribe' });
    const requestId = worker.posted[0]?.requestId ?? '';
    worker.emit({ type: 'error', requestId, code: 'transcribe-failed', message: 'boom' });

    await expect(transcribePromise).rejects.toMatchObject({ code: 'transcribe-failed' });
  });

  it('rejects every pending call on a requestId-less error (worker-level failure)', async () => {
    const worker = new FakeWorker();
    const client = new EngineClient(worker);

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
    const client = new EngineClient(worker);

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

    // The pending call must still be alive and resolve normally afterwards.
    worker.emit({ type: 'ready', requestId, capabilities, modelId: 'whisper-base' });
    await expect(initPromise).resolves.toEqual(capabilities);
  });

  it('does not throw when the fake worker emits raw junk with no listeners registered yet', () => {
    const worker = new FakeWorker();
    expect(() => worker.emitRaw(undefined)).not.toThrow();
    // Instantiate after, just to exercise the constructor's listener wiring.
    const client = new EngineClient(worker);
    expect(client).toBeInstanceOf(EngineClient);
  });
});

describe('EngineClient.abort', () => {
  it('posts an abort message with the given requestId', () => {
    const worker = new FakeWorker();
    const client = new EngineClient(worker);
    client.abort('transcribe-1');
    expect(worker.posted).toEqual([{ type: 'abort', requestId: 'transcribe-1' }]);
  });
});

describe('EngineClient.dispose', () => {
  it('stops reacting to worker messages after dispose', async () => {
    const worker = new FakeWorker();
    const client = new EngineClient(worker);
    const initPromise = client.init('whisper-base');
    const requestId = worker.posted[0]?.requestId ?? '';

    client.dispose();
    worker.emit({ type: 'ready', requestId, capabilities, modelId: 'whisper-base' });

    // Give any stray microtask a chance to run, then assert nothing settled it.
    const race = await Promise.race([
      initPromise.then(() => 'resolved' as const),
      new Promise((resolve) => setTimeout(() => resolve('pending' as const), 10)),
    ]);
    expect(race).toBe('pending');
  });
});
