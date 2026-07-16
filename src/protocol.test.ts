import { describe, expect, it } from 'vitest';
import { isHostMessage, isWorkerMessage } from './protocol.js';

describe('isHostMessage', () => {
  it('accepts a valid init message', () => {
    expect(isHostMessage({ type: 'init', modelId: 'whisper-base' })).toBe(true);
  });

  it('rejects init with an unknown model', () => {
    expect(isHostMessage({ type: 'init', modelId: 'gpt-5' })).toBe(false);
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
  });

  it('rejects transcribe without a typed-array payload', () => {
    expect(
      isHostMessage({
        type: 'transcribe',
        requestId: 'r1',
        audio: [0, 0.5],
        options: { task: 'transcribe' },
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
  it('accepts known worker message types', () => {
    expect(isWorkerMessage({ type: 'ready', capabilities: {}, modelId: 'whisper-base' })).toBe(
      true,
    );
    expect(isWorkerMessage({ type: 'error', code: 'no-backend', message: 'x' })).toBe(true);
  });

  it('rejects host message types and junk', () => {
    expect(isWorkerMessage({ type: 'init' })).toBe(false);
    expect(isWorkerMessage(undefined)).toBe(false);
    expect(isWorkerMessage({})).toBe(false);
  });
});
