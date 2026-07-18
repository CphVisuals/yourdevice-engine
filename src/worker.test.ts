import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

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
}

vi.mock('@huggingface/transformers', () => ({
  env: { backends: { onnx: { wasm: { numThreads: 0 } } } },
  pipeline: vi.fn(),
}));

async function loadMockedTransformers(): Promise<MockedTransformersModule> {
  return (await import('@huggingface/transformers')) as unknown as MockedTransformersModule;
}

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
