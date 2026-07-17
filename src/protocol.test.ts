import { describe, expect, it } from 'vitest';
import { isHostMessage, isWorkerMessage } from './protocol.js';

const capabilities = {
  detected: { webnn: false, webgpu: true, wasm: true },
  active: null,
  deviceMemoryGb: 8,
};

describe('isHostMessage', () => {
  it('accepts a valid init message', () => {
    expect(isHostMessage({ type: 'init', requestId: 'i1', modelId: 'whisper-base' })).toBe(true);
    expect(
      isHostMessage({
        type: 'init',
        requestId: 'i1',
        modelId: 'whisper-base',
        backendPreference: ['webgpu', 'wasm'],
      }),
    ).toBe(true);
  });

  it('rejects init with an unknown model, missing requestId, or bad preference', () => {
    expect(isHostMessage({ type: 'init', requestId: 'i1', modelId: 'gpt-5' })).toBe(false);
    expect(isHostMessage({ type: 'init', modelId: 'whisper-base' })).toBe(false);
    // Model ids inherited from Object.prototype must not validate.
    expect(isHostMessage({ type: 'init', requestId: 'i1', modelId: 'toString' })).toBe(false);
    // backendPreference must be an array of known backend ids, not a string.
    expect(
      isHostMessage({
        type: 'init',
        requestId: 'i1',
        modelId: 'whisper-base',
        backendPreference: 'webgpu',
      }),
    ).toBe(false);
    expect(
      isHostMessage({
        type: 'init',
        requestId: 'i1',
        modelId: 'whisper-base',
        backendPreference: ['cuda'],
      }),
    ).toBe(false);
  });

  it('accepts a valid transcribe message', () => {
    expect(
      isHostMessage({
        type: 'transcribe',
        requestId: 'r1',
        audio: new Float32Array(16000),
        options: { task: 'transcribe' },
      }),
    ).toBe(true);
    expect(
      isHostMessage({
        type: 'transcribe',
        requestId: 'r1',
        audio: new Float32Array(0),
        options: { task: 'translate', language: 'sv' },
      }),
    ).toBe(true);
  });

  it('rejects transcribe with a bad payload or incomplete options', () => {
    expect(
      isHostMessage({
        type: 'transcribe',
        requestId: 'r1',
        audio: [0, 0.5],
        options: { task: 'transcribe' },
      }),
    ).toBe(false);
    // options.task is required and must be a known task.
    expect(
      isHostMessage({
        type: 'transcribe',
        requestId: 'r1',
        audio: new Float32Array(1),
        options: {},
      }),
    ).toBe(false);
    expect(
      isHostMessage({
        type: 'transcribe',
        requestId: 'r1',
        audio: new Float32Array(1),
        options: { task: 'summarize' },
      }),
    ).toBe(false);
  });

  it('accepts abort and rejects junk', () => {
    expect(isHostMessage({ type: 'abort', requestId: 'r1' })).toBe(true);
    expect(isHostMessage({ type: 'abort' })).toBe(false);
    expect(isHostMessage(null)).toBe(false);
    expect(isHostMessage('init')).toBe(false);
    expect(isHostMessage({ type: 'ready' })).toBe(false);
  });
});

describe('isWorkerMessage', () => {
  it('accepts valid worker messages', () => {
    expect(
      isWorkerMessage({ type: 'ready', requestId: 'i1', capabilities, modelId: 'whisper-base' }),
    ).toBe(true);
    expect(
      isWorkerMessage({
        type: 'download-progress',
        requestId: 'i1',
        modelId: 'whisper-base',
        loadedBytes: 1024,
        totalBytes: 4096,
      }),
    ).toBe(true);
    expect(
      isWorkerMessage({
        type: 'partial',
        requestId: 'r1',
        segments: [{ start: 0, end: 1.5, text: 'hello' }],
        processedSeconds: 1.5,
        totalSeconds: 30,
      }),
    ).toBe(true);
    expect(isWorkerMessage({ type: 'complete', requestId: 'r1', segments: [] })).toBe(true);
    expect(isWorkerMessage({ type: 'error', code: 'no-backend', message: 'x' })).toBe(true);
    expect(isWorkerMessage({ type: 'error', requestId: 'r1', code: 'aborted', message: '' })).toBe(
      true,
    );
    // The optional backend field (init-time fallback attribution) must accept
    // every ladder backend and stay optional.
    expect(
      isWorkerMessage({
        type: 'error',
        requestId: 'i1',
        code: 'model-load-failed',
        message: 'graph build failed',
        backend: 'webnn',
      }),
    ).toBe(true);
    expect(
      isWorkerMessage({
        type: 'error',
        requestId: 'i1',
        code: 'model-load-failed',
        message: 'x',
        backend: 'wasm',
      }),
    ).toBe(true);
  });

  it('rejects messages whose payload does not match their type', () => {
    // The guard must validate payloads, not just the type discriminant.
    expect(isWorkerMessage({ type: 'ready' })).toBe(false);
    expect(
      isWorkerMessage({
        type: 'ready',
        requestId: 'i1',
        capabilities: {},
        modelId: 'whisper-base',
      }),
    ).toBe(false);
    expect(isWorkerMessage({ type: 'complete', requestId: 'r1' })).toBe(false);
    expect(isWorkerMessage({ type: 'complete', requestId: 'r1', segments: [{ start: 0 }] })).toBe(
      false,
    );
    expect(
      isWorkerMessage({
        type: 'partial',
        requestId: 'r1',
        segments: [],
        processedSeconds: 'half',
        totalSeconds: 30,
      }),
    ).toBe(false);
    expect(isWorkerMessage({ type: 'error', code: 'out-of-cheese', message: 'x' })).toBe(false);
    // backend, when present, must be a known ladder backend.
    expect(
      isWorkerMessage({ type: 'error', code: 'model-load-failed', message: 'x', backend: 'cuda' }),
    ).toBe(false);
    expect(
      isWorkerMessage({ type: 'error', code: 'model-load-failed', message: 'x', backend: 42 }),
    ).toBe(false);
    expect(
      isWorkerMessage({ type: 'error', code: 'model-load-failed', message: 'x', backend: null }),
    ).toBe(false);
  });

  it('rejects host message types and junk', () => {
    expect(isWorkerMessage({ type: 'init' })).toBe(false);
    expect(isWorkerMessage(undefined)).toBe(false);
    expect(isWorkerMessage({})).toBe(false);
  });
});
