import { describe, expect, it } from 'vitest';
import {
  buildCapabilityReport,
  detectBackends,
  pickDefaultModel,
  planBackendOrder,
} from './backends.js';

const ALL_DETECTED = { webnn: true, webgpu: true, wasm: true } as const;

describe('detectBackends', () => {
  it('reports everything absent on an empty scope', () => {
    expect(detectBackends({})).toEqual({ webnn: false, webgpu: false, wasm: false });
  });

  it('detects each backend from its global', () => {
    expect(detectBackends({ navigator: { ml: {}, gpu: {} }, WebAssembly: {} })).toEqual(
      ALL_DETECTED,
    );
  });

  it('treats WebGPU without WebNN correctly (the common Chrome case)', () => {
    expect(detectBackends({ navigator: { gpu: {} }, WebAssembly: {} })).toEqual({
      webnn: false,
      webgpu: true,
      wasm: true,
    });
  });
});

describe('planBackendOrder', () => {
  it('follows the ladder when there is no preference', () => {
    expect(planBackendOrder(ALL_DETECTED)).toEqual(['webnn', 'webgpu', 'wasm']);
  });

  it('skips undetected backends', () => {
    expect(planBackendOrder({ webnn: false, webgpu: true, wasm: true })).toEqual([
      'webgpu',
      'wasm',
    ]);
  });

  it('honors an explicit preference but keeps the floor as fallback', () => {
    expect(planBackendOrder(ALL_DETECTED, ['wasm'])).toEqual(['wasm', 'webnn', 'webgpu']);
    expect(planBackendOrder(ALL_DETECTED, ['webgpu'])).toEqual(['webgpu', 'webnn', 'wasm']);
  });

  it('ignores preferred backends that were not detected', () => {
    expect(planBackendOrder({ webnn: false, webgpu: false, wasm: true }, ['webnn'])).toEqual([
      'wasm',
    ]);
  });

  it('returns empty when nothing is detected', () => {
    expect(planBackendOrder({ webnn: false, webgpu: false, wasm: false })).toEqual([]);
  });
});

describe('buildCapabilityReport', () => {
  it('captures device memory when reported', () => {
    const report = buildCapabilityReport({
      navigator: { gpu: {}, deviceMemory: 8 },
      WebAssembly: {},
    });
    expect(report.deviceMemoryGb).toBe(8);
    expect(report.active).toBeNull();
  });

  it('leaves device memory null when unavailable (Safari/Firefox)', () => {
    expect(buildCapabilityReport({ WebAssembly: {} }).deviceMemoryGb).toBeNull();
  });
});

describe('pickDefaultModel', () => {
  it('defaults to whisper-base', () => {
    const report = buildCapabilityReport({ WebAssembly: {} });
    expect(pickDefaultModel(report)).toBe('whisper-base');
  });

  it('upgrades to whisper-small only with acceleration and >= 8 GB memory', () => {
    const capable = buildCapabilityReport({
      navigator: { gpu: {}, deviceMemory: 8 },
      WebAssembly: {},
    });
    expect(pickDefaultModel(capable)).toBe('whisper-small');

    const lowMemory = buildCapabilityReport({
      navigator: { gpu: {}, deviceMemory: 4 },
      WebAssembly: {},
    });
    expect(pickDefaultModel(lowMemory)).toBe('whisper-base');

    const noGpu = buildCapabilityReport({
      navigator: { deviceMemory: 16 },
      WebAssembly: {},
    });
    expect(pickDefaultModel(noGpu)).toBe('whisper-base');
  });

  it('stays conservative when device memory is unknown', () => {
    const unknownMemory = buildCapabilityReport({ navigator: { gpu: {} }, WebAssembly: {} });
    expect(pickDefaultModel(unknownMemory)).toBe('whisper-base');
  });
});
