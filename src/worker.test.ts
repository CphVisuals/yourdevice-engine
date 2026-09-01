import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type * as WorkerModuleTypes from './worker.js';

/**
 * worker.ts is a side-effecting entry point: it calls `addEventListener`
 * and `postMessage` as soon as it is imported, and it imports
 * `@huggingface/transformers` for real. To exercise `handleInit`'s reported
 * behavior without touching real ONNX or a real Worker global scope, this
 * file:
 *   - mocks `@huggingface/transformers` entirely (an injected fake ASR
 *     runner stands in for the pipeline — no real model, no real inference),
 *   - stubs the worker globals worker.ts touches (`postMessage`,
 *     `addEventListener`, `self`, `fetch`, `navigator`),
 *   - dynamically imports worker.ts *after* stubbing, once per test (via
 *     `vi.resetModules`), so its top-level `addEventListener` call and
 *     module-level state never leak between tests.
 *
 * The degenerate-output-detection and validation-clip logic themselves are
 * unit-tested in isolation in degenerateOutput.test.ts, validationClip.test.ts,
 * and backendInit.test.ts; this file only checks that `handleInit` wires
 * that logic into the same `model-load-failed`-with-`backend` protocol
 * message the pre-existing graph-build-failure path uses (so
 * `EngineClient`'s cross-worker fallback in client.ts picks it up).
 */

interface MockedTransformersModule {
  pipeline: ReturnType<typeof vi.fn>;
  env: { backends: { onnx: { wasm: { numThreads: number } | undefined } } };
  AutoModel: { from_pretrained: ReturnType<typeof vi.fn> };
  AutoModelForAudioFrameClassification: { from_pretrained: ReturnType<typeof vi.fn> };
  AutoProcessor: { from_pretrained: ReturnType<typeof vi.fn> };
}

vi.mock('@huggingface/transformers', () => ({
  env: { backends: { onnx: { wasm: { numThreads: 0 } } } },
  pipeline: vi.fn(),
  // Diarization's two graphs plus their processors. Only the lifecycle
  // (`from_pretrained` → `dispose`) matters here; what they compute is
  // diarize.test.ts's subject, and `./diarize.js` is mocked out below.
  AutoModel: { from_pretrained: vi.fn() },
  AutoModelForAudioFrameClassification: { from_pretrained: vi.fn() },
  AutoProcessor: { from_pretrained: vi.fn() },
}));

// The diarization algorithm is unit-tested in diarize.test.ts. These tests are
// about *when the models are resident*, so the pass itself is a no-op.
vi.mock('./diarize.js', () => ({
  diarize: vi.fn().mockResolvedValue([]),
  assignSpeakers: vi.fn((segments: unknown) => segments),
}));

async function loadMockedTransformers(): Promise<MockedTransformersModule> {
  return (await import('@huggingface/transformers')) as unknown as MockedTransformersModule;
}

// Type-only, so it is erased at compile time: importing worker.ts for real at
// module scope would run its `addEventListener`/`postMessage` side effects
// before the globals are stubbed (see the docblock).
type WorkerModule = typeof WorkerModuleTypes;

let postMessageSpy: ReturnType<typeof vi.fn>;

beforeEach(() => {
  vi.resetModules();
  postMessageSpy = vi.fn();
  vi.stubGlobal('postMessage', postMessageSpy);
  vi.stubGlobal('addEventListener', vi.fn());
  vi.stubGlobal('self', { location: { href: 'https://example.test/' } });
  // No mirror in tests: configureModelHost's HEAD probe must fail closed.
  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('no network in tests')));
  // WebGPU "detected and handshakes ok"; WebNN absent, so only webgpu is a
  // candidate when it's the caller's preference.
  vi.stubGlobal('navigator', {
    gpu: { requestAdapter: vi.fn().mockResolvedValue({}) },
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe('handleInit — accelerated output validation', () => {
  it('reports model-load-failed with the backend when webgpu builds but its probe transcript is degenerate', async () => {
    const { pipeline } = await loadMockedTransformers();
    const fakeAsr = vi.fn().mockResolvedValue({ text: '!!!!' });
    pipeline.mockResolvedValue(fakeAsr);

    const { handleInit } = await import('./worker.js');
    await handleInit({
      type: 'init',
      requestId: 'req-1',
      modelId: 'whisper-tiny',
      backendPreference: ['webgpu'],
    });

    expect(fakeAsr).toHaveBeenCalledTimes(1); // the probe ran exactly once
    expect(postMessageSpy).toHaveBeenCalledTimes(1);
    const sent = postMessageSpy.mock.calls[0]?.[0] as {
      type: string;
      requestId: string;
      code: string;
      backend: string;
      message: string;
    };
    expect(sent).toMatchObject({
      type: 'error',
      requestId: 'req-1',
      code: 'model-load-failed',
      backend: 'webgpu',
    });
    expect(sent.message).toContain('degenerate');
  });

  it('reports ready and stays on webgpu when its probe transcript is healthy', async () => {
    const { pipeline } = await loadMockedTransformers();
    const fakeAsr = vi.fn().mockResolvedValue({ text: 'and so my fellow Americans' });
    pipeline.mockResolvedValue(fakeAsr);

    const { handleInit } = await import('./worker.js');
    await handleInit({
      type: 'init',
      requestId: 'req-2',
      modelId: 'whisper-tiny',
      backendPreference: ['webgpu'],
    });

    expect(postMessageSpy).toHaveBeenCalledTimes(1);
    const sent = postMessageSpy.mock.calls[0]?.[0] as {
      type: string;
      requestId: string;
      capabilities: { active: string | null };
    };
    expect(sent.type).toBe('ready');
    expect(sent.capabilities.active).toBe('webgpu');
  });

  it('never runs the probe on wasm, even when the fake ASR would return degenerate output', async () => {
    // No WebGPU/WebNN in this test's navigator — only wasm is detected.
    vi.stubGlobal('navigator', {});
    const { pipeline } = await loadMockedTransformers();
    const fakeAsr = vi.fn().mockResolvedValue({ text: '!!!!' });
    pipeline.mockResolvedValue(fakeAsr);

    const { handleInit } = await import('./worker.js');
    await handleInit({
      type: 'init',
      requestId: 'req-3',
      modelId: 'whisper-tiny',
      backendPreference: ['wasm'],
    });

    expect(fakeAsr).not.toHaveBeenCalled(); // wasm is trusted; no probe call
    expect(postMessageSpy).toHaveBeenCalledTimes(1);
    const sent = postMessageSpy.mock.calls[0]?.[0] as { type: string; capabilities?: unknown };
    expect(sent.type).toBe('ready');
  });
});

/**
 * The model lifecycle — what a diarized run holds in memory, and when.
 *
 * The failure this covers has no stack trace: an iPhone 13 recording with
 * speaker ID on reloaded the tab mid-run. iOS Safari enforces its per-tab
 * memory ceiling by reloading, silently, so the only evidence is what the
 * worker was holding. It was holding three graphs — Whisper, pyannote
 * segmentation, wespeaker — for two phases that never run at the same time,
 * and nothing in the worker could free any of them.
 */
describe('model lifecycle — one phase, one set of models', () => {
  /** A callable pipeline stub with a disposal spy, as Transformers.js returns. */
  function fakeAsr(order: string[], label = 'asr'): ReturnType<typeof vi.fn> {
    const asr = vi.fn().mockResolvedValue({ text: 'hello world', chunks: [] });
    return Object.assign(asr, {
      dispose: vi.fn(() => {
        order.push(`${label}-dispose`);
        return Promise.resolve();
      }),
    }) as unknown as ReturnType<typeof vi.fn>;
  }

  function fakeModel(order: string[], label: string): { dispose: ReturnType<typeof vi.fn> } {
    return {
      dispose: vi.fn(() => {
        order.push(`${label}-dispose`);
        return Promise.resolve([]);
      }),
    };
  }

  async function initOnWasm(order: string[]): Promise<{
    handleTranscribe: WorkerModule['handleTranscribe'];
    transformers: MockedTransformersModule;
  }> {
    // No accelerated backend in this navigator: wasm only, so no probe call
    // and no validation clip in the way of the lifecycle assertions.
    vi.stubGlobal('navigator', {});
    const transformers = await loadMockedTransformers();
    transformers.pipeline.mockImplementation(() => {
      order.push('asr-build');
      return Promise.resolve(fakeAsr(order));
    });
    transformers.AutoModelForAudioFrameClassification.from_pretrained.mockImplementation(() => {
      order.push('segmentation-build');
      return Promise.resolve(fakeModel(order, 'segmentation'));
    });
    transformers.AutoModel.from_pretrained.mockImplementation(() => {
      order.push('embedding-build');
      return Promise.resolve(fakeModel(order, 'embedding'));
    });
    transformers.AutoProcessor.from_pretrained.mockResolvedValue(vi.fn());

    const worker = await import('./worker.js');
    await worker.handleInit({
      type: 'init',
      requestId: 'init-1',
      modelId: 'whisper-tiny',
      backendPreference: ['wasm'],
    });
    return { handleTranscribe: worker.handleTranscribe, transformers };
  }

  it('disposes the ASR pipeline before it builds the diarization models', async () => {
    const order: string[] = [];
    const { handleTranscribe } = await initOnWasm(order);

    await handleTranscribe({
      type: 'transcribe',
      requestId: 'run-1',
      audio: new Float32Array(16_000),
      options: { task: 'transcribe', diarize: true },
    });

    // The whole point: the transcription graph is gone from memory before the
    // diarization graphs arrive, so peak is max(...) and not sum(...).
    expect(order).toEqual([
      'asr-build',
      'asr-dispose',
      'segmentation-build',
      'embedding-build',
      'segmentation-dispose',
      'embedding-dispose',
    ]);
  });

  it('still returns the transcript, with the diarization pass applied', async () => {
    const order: string[] = [];
    const { handleTranscribe } = await initOnWasm(order);
    postMessageSpy.mockClear();

    await handleTranscribe({
      type: 'transcribe',
      requestId: 'run-1',
      audio: new Float32Array(16_000),
      options: { task: 'transcribe', diarize: true },
    });

    const complete = postMessageSpy.mock.calls
      .map((call) => call[0] as { type: string; segments?: { text: string }[] })
      .find((message) => message.type === 'complete');
    expect(complete?.segments?.[0]?.text).toBe('hello world');
  });

  it('rebuilds the released pipeline for the next file instead of failing', async () => {
    const order: string[] = [];
    const { handleTranscribe, transformers } = await initOnWasm(order);

    await handleTranscribe({
      type: 'transcribe',
      requestId: 'run-1',
      audio: new Float32Array(16_000),
      options: { task: 'transcribe', diarize: true },
    });
    postMessageSpy.mockClear();

    // A released pipeline is an *initialized* engine with its weights
    // unloaded. The host does not re-init (it tracks the model itself), so
    // the worker has to rebuild silently or this second file errors out.
    await handleTranscribe({
      type: 'transcribe',
      requestId: 'run-2',
      audio: new Float32Array(16_000),
      options: { task: 'transcribe' },
    });

    expect(transformers.pipeline).toHaveBeenCalledTimes(2);
    const types = postMessageSpy.mock.calls.map((call) => (call[0] as { type: string }).type);
    expect(types).toContain('complete');
    expect(types).not.toContain('error');
  });

  it('keeps the pipeline loaded across runs that do not diarize', async () => {
    const order: string[] = [];
    const { handleTranscribe, transformers } = await initOnWasm(order);

    for (const requestId of ['run-1', 'run-2']) {
      await handleTranscribe({
        type: 'transcribe',
        requestId,
        audio: new Float32Array(16_000),
        options: { task: 'transcribe' },
      });
    }

    // No diarization, no release: the fast path is untouched by this change.
    expect(transformers.pipeline).toHaveBeenCalledTimes(1);
    expect(order).toEqual(['asr-build']);
  });

  it('reports the engine as uninitialized when transcribe arrives before init', async () => {
    vi.stubGlobal('navigator', {});
    const { handleTranscribe } = await import('./worker.js');
    await handleTranscribe({
      type: 'transcribe',
      requestId: 'run-1',
      audio: new Float32Array(16_000),
      options: { task: 'transcribe' },
    });

    const sent = postMessageSpy.mock.calls[0]?.[0] as { code: string; message: string };
    expect(sent.code).toBe('transcribe-failed');
    expect(sent.message).toContain('not initialized');
  });

  it('releasing twice is a no-op, and releasing nothing never throws', async () => {
    const order: string[] = [];
    const worker = await import('./worker.js');
    // Nothing loaded at all.
    await expect(worker.releaseAsrPipeline()).resolves.toBeUndefined();
    await expect(worker.releaseDiarizer()).resolves.toBeUndefined();

    const { handleTranscribe } = await initOnWasm(order);
    await handleTranscribe({
      type: 'transcribe',
      requestId: 'run-1',
      audio: new Float32Array(16_000),
      options: { task: 'transcribe', diarize: true },
    });
    const disposals = order.filter((step) => step.endsWith('-dispose')).length;
    await worker.releaseAsrPipeline();
    await worker.releaseDiarizer();
    expect(order.filter((step) => step.endsWith('-dispose')).length).toBe(disposals);
  });
});
