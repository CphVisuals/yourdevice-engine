import type { AutomaticSpeechRecognitionPipeline } from '@huggingface/transformers';
import { describe, expect, it, vi } from 'vitest';
import { attemptBackend, type PipelineBuilder } from './backendInit.js';

/**
 * A fake pipeline builder that resolves to an injected fake ASR runner.
 * `asr` is a bare `vi.fn()` callable — the engine only ever *calls* the
 * pipeline, so we cast the mock to the library's (now class-shaped in v4)
 * pipeline type at this single injection boundary rather than stubbing the
 * class's dozen private members.
 */
function fakeBuilder(asr: unknown): PipelineBuilder {
  return vi.fn().mockResolvedValue(asr as AutomaticSpeechRecognitionPipeline);
}

const CLIP = new Float32Array([0.1, -0.1, 0.2, -0.2]);

describe('attemptBackend', () => {
  it('fails with the build error when the pipeline factory rejects', async () => {
    const buildError = new Error('webnn: graph build failed');
    const builder: PipelineBuilder = vi.fn().mockRejectedValue(buildError);

    const result = await attemptBackend('webnn', builder, CLIP);

    expect(result).toEqual({ ok: false, error: buildError });
  });

  it('accepts an accelerated backend whose probe transcript is healthy', async () => {
    const asr = vi.fn().mockResolvedValue({ text: 'and so my fellow Americans' });
    const builder = fakeBuilder(asr);

    const result = await attemptBackend('webgpu', builder, CLIP);

    expect(result.ok).toBe(true);
    if (result.ok) expect(result.asr).toBe(asr);
    expect(asr).toHaveBeenCalledWith(CLIP, expect.objectContaining({ return_timestamps: false }));
  });

  it('rejects an accelerated backend whose probe transcript is degenerate', async () => {
    const asr = vi.fn().mockResolvedValue({ text: '!!!!' });
    const builder = fakeBuilder(asr);

    const result = await attemptBackend('webgpu', builder, CLIP);

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.error).toBeInstanceOf(Error);
      expect((result.error as Error).message).toContain('webgpu');
      expect((result.error as Error).message).toContain('degenerate');
    }
  });

  it('rejects webnn the same way when its probe transcript is degenerate', async () => {
    const asr = vi.fn().mockResolvedValue({ text: '' });
    const builder = fakeBuilder(asr);

    const result = await attemptBackend('webnn', builder, CLIP);

    expect(result.ok).toBe(false);
    if (!result.ok) expect((result.error as Error).message).toContain('webnn');
  });

  it('handles array-shaped pipeline output the same way as single-object output', async () => {
    const asr = vi.fn().mockResolvedValue([{ text: '!!!!' }]);
    const builder = fakeBuilder(asr);

    const result = await attemptBackend('webgpu', builder, CLIP);

    expect(result.ok).toBe(false);
  });

  it('never probes wasm, even if the pipeline would produce degenerate output', async () => {
    const asr = vi.fn().mockResolvedValue({ text: '!!!!' });
    const builder = fakeBuilder(asr);

    const result = await attemptBackend('wasm', builder, CLIP);

    expect(result).toEqual({ ok: true, asr });
    expect(asr).not.toHaveBeenCalled();
  });

  it('fails the attempt if the accelerated probe call itself throws', async () => {
    const probeError = new Error('inference crashed');
    const asr = vi.fn().mockRejectedValue(probeError);
    const builder = fakeBuilder(asr);

    const result = await attemptBackend('webgpu', builder, CLIP);

    expect(result).toEqual({ ok: false, error: probeError });
  });

  it('defaults to the bundled validation clip when none is injected', async () => {
    const asr = vi.fn().mockResolvedValue({ text: 'and so my fellow Americans' });
    const builder = fakeBuilder(asr);

    await attemptBackend('webgpu', builder);

    const [audioArg] = asr.mock.calls[0] as [Float32Array, unknown];
    expect(audioArg).toBeInstanceOf(Float32Array);
    expect(audioArg.length).toBeGreaterThan(0);
  });
});
